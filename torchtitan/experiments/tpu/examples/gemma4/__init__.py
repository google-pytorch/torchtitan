from collections.abc import Callable
import copy
from functools import partial

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.tpu.examples.gemma4.infra.parallelize import (
    parallelize_gemma4,
)
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.models.common.attention import (
    FusedQKVLinear,
    QKVLinear,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_token_dispatcher_config,
)
from torchtitan.models.common.linear import Linear, RouterGateLinear
from torchtitan.models.common.moe import MoELoadBalanceAuxLoss, RoutedExperts
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.param_init import skip_param_init
from torchtitan.models.muse_glimmer.model import SoftCappedLinear
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import (
    Gemma4Attention,
    Gemma4Embedding,
    Gemma4FeedForward,
    Gemma4MoE,
    Gemma4Model,
    Gemma4RoPE,
    Gemma4Router,
    Gemma4TransformerBlock,
    QKVLinearKeqV,
)
from .sharding import set_gemma4_sharding_config
from .state_dict_adapter import Gemma4StateDictAdapter

__all__ = [
    "Gemma4Attention",
    "Gemma4Embedding",
    "Gemma4FeedForward",
    "Gemma4MoE",
    "Gemma4Model",
    "Gemma4RoPE",
    "Gemma4Router",
    "Gemma4StateDictAdapter",
    "Gemma4TransformerBlock",
    "QKVLinearKeqV",
    "parallelize_gemma4",
    "gemma4_configs",
    "model_registry",
]

_NORM_INIT: dict[str, Callable] = {"weight": nn.init.ones_}
_EMBEDDING_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.normal_, mean=0.0, std=0.02)
}
_EMBEDDING_SKIP_INIT: dict[str, Callable] = {"weight": skip_param_init}
_OUTPUT_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.normal_, mean=0.0, std=0.02)
}
_LOGIT_SOFT_CAP = 30.0


def _gemma4_lm_head(dim: int, vocab_size: int) -> SoftCappedLinear.Config:
    return SoftCappedLinear.Config(
        in_features=dim,
        out_features=vocab_size,
        param_init=_OUTPUT_INIT,
        output_soft_cap_temp=_LOGIT_SOFT_CAP,
    )


def _gemma4_norm(dim: int, eps: float = 1e-6) -> RMSNorm.Config:
    return RMSNorm.Config(
        normalized_shape=dim, eps=eps, param_init=_NORM_INIT
    )


def _make_param_init(depth: int) -> dict[str, dict[str, Callable]]:
    """Create standard parameter initializers scaled by depth."""
    init_std = 0.02 / (2 * (depth + 1)) ** 0.5
    return {
        "linear_in": {"weight": partial(nn.init.trunc_normal_, std=0.02)},
        "linear_out": {"weight": partial(nn.init.trunc_normal_, std=init_std)},
        "experts": {
            "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
            "w2_EDF": partial(nn.init.trunc_normal_, std=init_std),
            "w3_EFD": partial(nn.init.trunc_normal_, std=init_std),
        },
        "router": {
            "scale": nn.init.ones_,
            "per_expert_scale": nn.init.ones_,
        },
    }


def _make_moe_config(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    norm_eps: float,
    param_init: dict[str, dict[str, Callable]],
    comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    load_balance_coeff: float | None = None,
    aux_loss_coeff: float = 0.0,
) -> Gemma4MoE.Config:
    return Gemma4MoE.Config(
        num_experts=num_experts,
        load_balance_coeff=load_balance_coeff,
        pre_feedforward_layernorm_2=_gemma4_norm(dim, eps=norm_eps),
        router=Gemma4Router.Config(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            score_func="softmax",
            route_norm=True,
            aux_loss=MoELoadBalanceAuxLoss.Config(
                type="sequence_wise",
                coeff=aux_loss_coeff,
                top_k=top_k,
            )
            if aux_loss_coeff > 0
            else None,
            norm=RMSNorm.Config(
                normalized_shape=dim,
                eps=norm_eps,
                elementwise_affine=False,
            ),
            gate=RouterGateLinear.Config(
                in_features=dim,
                out_features=num_experts,
                param_init=param_init["linear_out"],
            ),
            param_init=param_init["router"],
        ),
        routed_experts=RoutedExperts.Config(
            inner_experts=TpuGroupedExperts.Config(
                activation="gelu_tanh",
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                param_init=param_init["experts"],
            ),
            token_dispatcher=make_token_dispatcher_config(
                num_experts=num_experts,
                top_k=top_k,
                comm_backend=comm_backend,
                non_blocking_capacity_factor=non_blocking_capacity_factor,
                hidden_dim=dim,
            ),
        ),
    )


def _build_gemma4_layers(
    n_layers: int,
    dim: int,
    hidden_dim: int,
    n_heads: int,
    n_kv_heads: int,
    n_global_kv_heads: int,
    head_dim: int,
    global_head_dim: int,
    norm_eps: float,
    local_rope_theta: float = 10_000.0,
    global_rope_theta: float = 1_000_000.0,
    global_partial_rotary_factor: float = 0.25,
    max_seq_len: int = 131072,
    sliding_window_size: int = 1024,
    global_attn_every_n_layers: int = 6,
    use_k_eq_v_on_global: bool = True,
    moe_enabled: bool = False,
    num_experts: int = 128,
    top_k: int = 8,
    moe_inter_dim: int = 704,
    load_balance_coeff: float | None = None,
    aux_loss_coeff: float = 0.0,
    attn_backend: str = "sdpa",
    use_fused_qkv: bool = False,
    comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
) -> list[Gemma4TransformerBlock.Config]:
    layers = []
    for layer_id in range(n_layers):
        p = _make_param_init(layer_id)
        is_global = (layer_id + 1) % global_attn_every_n_layers == 0
        is_sliding = not is_global

        layer_head_dim = global_head_dim if is_global else head_dim
        layer_n_kv_heads = n_global_kv_heads if is_global else n_kv_heads
        layer_use_k_eq_v = use_k_eq_v_on_global if is_global else False
        layer_rope_theta = global_rope_theta if is_global else local_rope_theta
        layer_partial_rotary = (
            global_partial_rotary_factor if is_global else 1.0
        )

        if layer_use_k_eq_v:
            qkv_linear_cfg = QKVLinearKeqV.Config(
                head_dim=layer_head_dim,
                wq=Linear.Config(
                    in_features=dim,
                    out_features=n_heads * layer_head_dim,
                    param_init=p["linear_in"],
                ),
                wkv=Linear.Config(
                    in_features=dim,
                    out_features=layer_n_kv_heads * layer_head_dim,
                    param_init=p["linear_in"],
                ),
            )
        elif use_fused_qkv:
            r_dim = n_heads // layer_n_kv_heads + 2
            qkv_linear_cfg = FusedQKVLinear.Config(
                n_heads=n_heads,
                n_kv_heads=layer_n_kv_heads,
                head_dim=layer_head_dim,
                wqkv=Linear.Config(
                    in_features=dim,
                    out_features=layer_n_kv_heads * r_dim * layer_head_dim,
                    param_init=p["linear_in"],
                ),
            )
        else:
            qkv_linear_cfg = QKVLinear.Config(
                head_dim=layer_head_dim,
                wq=Linear.Config(
                    in_features=dim,
                    out_features=n_heads * layer_head_dim,
                    param_init=p["linear_in"],
                ),
                wkv=Linear.Config(
                    in_features=dim,
                    out_features=layer_n_kv_heads * layer_head_dim,
                    param_init=p["linear_in"],
                ),
            )

        norm_cfg = _gemma4_norm(dim, eps=norm_eps)
        layers.append(
            Gemma4TransformerBlock.Config(
                attention_norm=norm_cfg,
                post_attention_layernorm=norm_cfg,
                ffn_norm=norm_cfg,
                post_feedforward_layernorm=norm_cfg,
                attention=Gemma4Attention.Config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=layer_n_kv_heads,
                    head_dim=layer_head_dim,
                    is_sliding=is_sliding,
                    sliding_window_size=(
                        sliding_window_size if is_sliding else None
                    ),
                    qk_norm=_gemma4_norm(layer_head_dim, eps=norm_eps),
                    v_norm=RMSNorm.Config(
                        normalized_shape=layer_head_dim,
                        eps=norm_eps,
                        elementwise_affine=False,
                    ),
                    wo=Linear.Config(
                        in_features=n_heads * layer_head_dim,
                        out_features=dim,
                        param_init=p["linear_out"],
                    ),
                    qkv_linear=qkv_linear_cfg,
                    inner_attention=get_attention_config(attn_backend),
                    rope=Gemma4RoPE.Config(
                        dim=layer_head_dim,
                        max_context_length=max_seq_len,
                        theta=layer_rope_theta,
                        partial_rotary_factor=layer_partial_rotary,
                    ),
                ),
                feed_forward=Gemma4FeedForward.Config(
                    w1=Linear.Config(
                        in_features=dim,
                        out_features=hidden_dim,
                        param_init=p["linear_in"],
                    ),
                    w2=Linear.Config(
                        in_features=hidden_dim,
                        out_features=dim,
                        param_init=p["linear_out"],
                    ),
                    w3=Linear.Config(
                        in_features=dim,
                        out_features=hidden_dim,
                        param_init=p["linear_out"],
                    ),
                ),
                moe=_make_moe_config(
                    dim=dim,
                    hidden_dim=moe_inter_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    norm_eps=norm_eps,
                    param_init=p,
                    comm_backend=comm_backend,
                    non_blocking_capacity_factor=non_blocking_capacity_factor,
                    load_balance_coeff=load_balance_coeff,
                    aux_loss_coeff=aux_loss_coeff,
                )
                if moe_enabled
                else None,
                post_feedforward_layernorm_1=norm_cfg if moe_enabled else None,
                post_feedforward_layernorm_2=norm_cfg if moe_enabled else None,
            )
        )
    return layers


def _testmodel(attn_backend: str = "sdpa") -> Gemma4Model.Config:
    dim = 128
    head_dim = 16
    global_head_dim = 32
    n_layers = 2
    vocab_size = 2048
    sliding_window_size = 64
    config = Gemma4Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        sliding_window_size=sliding_window_size,
        norm=_gemma4_norm(dim),
        tok_embeddings=Gemma4Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
        ),
        lm_head=_gemma4_lm_head(dim, vocab_size),
        # pytype: disable=wrong-keyword-args
        layers=_build_gemma4_layers(
            n_layers=n_layers,
            dim=dim,
            hidden_dim=256,
            n_heads=8,
            n_kv_heads=8,
            n_global_kv_heads=8,
            head_dim=head_dim,
            global_head_dim=global_head_dim,
            norm_eps=1e-6,
            max_seq_len=128,
            sliding_window_size=sliding_window_size,
            global_attn_every_n_layers=2,
            attn_backend=attn_backend,
        ),
        # pytype: enable=wrong-keyword-args
    )
    set_gemma4_sharding_config(config, enable_sp=True, enable_ep=False)
    return config


def _testmodel_moe(
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
) -> Gemma4Model.Config:
    dim = 128
    head_dim = 16
    global_head_dim = 32
    n_layers = 2
    vocab_size = 2048
    sliding_window_size = 64
    config = Gemma4Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        sliding_window_size=sliding_window_size,
        norm=_gemma4_norm(dim),
        tok_embeddings=Gemma4Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
        ),
        lm_head=_gemma4_lm_head(dim, vocab_size),
        # pytype: disable=wrong-keyword-args
        layers=_build_gemma4_layers(
            n_layers=n_layers,
            dim=dim,
            hidden_dim=256,
            n_heads=8,
            n_kv_heads=8,
            n_global_kv_heads=8,
            head_dim=head_dim,
            global_head_dim=global_head_dim,
            norm_eps=1e-6,
            max_seq_len=128,
            sliding_window_size=sliding_window_size,
            global_attn_every_n_layers=2,
            moe_enabled=True,
            num_experts=8,
            top_k=2,
            moe_inter_dim=128,
            attn_backend=attn_backend,
            comm_backend=moe_comm_backend,
        ),
        # pytype: enable=wrong-keyword-args
    )
    set_gemma4_sharding_config(config, enable_sp=True, enable_ep=False)
    return config


_gemma4_factories = {
    "debugmodel": partial(
        Gemma4Model.Config,
        vocab_size=2048,
        dim=256,
        sliding_window_size=128,
        tok_embeddings=Gemma4Embedding.Config(
            num_embeddings=2048,
            embedding_dim=256,
        ),
        norm=_gemma4_norm(256, eps=1e-6),
        lm_head=_gemma4_lm_head(256, 2048),
        layers=partial(
            _build_gemma4_layers,
            n_layers=6,
            dim=256,
            hidden_dim=512,
            n_heads=8,
            n_kv_heads=8,
            n_global_kv_heads=8,
            head_dim=32,
            global_head_dim=64,
            norm_eps=1e-6,
            max_seq_len=4096,
            sliding_window_size=128,
            global_attn_every_n_layers=6,
        ),
    ),
    "debugmodel_moe": partial(
        Gemma4Model.Config,
        vocab_size=2048,
        dim=256,
        sliding_window_size=128,
        tok_embeddings=Gemma4Embedding.Config(
            num_embeddings=2048,
            embedding_dim=256,
        ),
        norm=_gemma4_norm(256, eps=1e-6),
        lm_head=_gemma4_lm_head(256, 2048),
        layers=partial(
            _build_gemma4_layers,
            n_layers=6,
            dim=256,
            hidden_dim=512,
            n_heads=8,
            n_kv_heads=8,
            n_global_kv_heads=8,
            head_dim=32,
            global_head_dim=64,
            norm_eps=1e-6,
            max_seq_len=4096,
            sliding_window_size=128,
            global_attn_every_n_layers=6,
            moe_enabled=True,
            num_experts=8,
            top_k=2,
            moe_inter_dim=128,
        ),
    ),
    "31B": partial(
        Gemma4Model.Config,
        vocab_size=262144,
        dim=5376,
        sliding_window_size=1024,
        tok_embeddings=Gemma4Embedding.Config(
            num_embeddings=262144,
            embedding_dim=5376,
        ),
        norm=_gemma4_norm(5376, eps=1e-6),
        lm_head=_gemma4_lm_head(5376, 262144),
        layers=partial(
            _build_gemma4_layers,
            n_layers=60,
            dim=5376,
            hidden_dim=21504,
            n_heads=32,
            n_kv_heads=16,
            n_global_kv_heads=4,
            head_dim=256,
            global_head_dim=512,
            norm_eps=1e-6,
            max_seq_len=131072,
            sliding_window_size=1024,
            global_attn_every_n_layers=6,
        ),
    ),
    "26B-A4B": partial(
        Gemma4Model.Config,
        vocab_size=262144,
        dim=2816,
        sliding_window_size=1024,
        tok_embeddings=Gemma4Embedding.Config(
            num_embeddings=262144,
            embedding_dim=2816,
        ),
        norm=_gemma4_norm(2816, eps=1e-6),
        lm_head=_gemma4_lm_head(2816, 262144),
        layers=partial(
            _build_gemma4_layers,
            n_layers=30,
            dim=2816,
            hidden_dim=2112,
            n_heads=16,
            n_kv_heads=8,
            n_global_kv_heads=2,
            head_dim=256,
            global_head_dim=512,
            norm_eps=1e-6,
            max_seq_len=131072,
            sliding_window_size=1024,
            global_attn_every_n_layers=6,
            moe_enabled=True,
            num_experts=128,
            top_k=8,
            moe_inter_dim=704,
        ),
    ),
}


def _set_weight_tying(config: Gemma4Model.Config, enable: bool) -> None:
    """Tie ``lm_head`` to the token embedding (HF default) or untie it.

    When tied, the shared weight is initialized through ``lm_head``, so the
    embedding's own init is skipped.
    """
    config.enable_weight_tying = enable
    config.tok_embeddings.param_init = (
        _EMBEDDING_SKIP_INIT if enable else _EMBEDDING_INIT
    )


def _build_flavor_config(
    flavor: str,
    attn_backend: str = "sdpa",
    use_fused_qkv: bool = False,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    enable_sp: bool = True,
    enable_ep: bool = False,
    converters: list[ModelConfigConverter.Config] | None = None,
    enable_weight_tying: bool = True,
) -> Gemma4Model.Config:
    if flavor == "testmodel":
        config = _testmodel(attn_backend=attn_backend)
    elif flavor == "testmodel_moe":
        config = _testmodel_moe(
            attn_backend=attn_backend, moe_comm_backend=moe_comm_backend
        )
    else:
        factory = _gemma4_factories[flavor]
        layers_partial = factory.keywords["layers"]
        extra_kw: dict = {
            "attn_backend": attn_backend,
            "use_fused_qkv": use_fused_qkv,
        }
        if layers_partial.keywords.get("moe_enabled", False):
            extra_kw["comm_backend"] = moe_comm_backend
            extra_kw["non_blocking_capacity_factor"] = (
                non_blocking_capacity_factor
            )
        config = copy.deepcopy(factory(layers=layers_partial(**extra_kw)))

    _set_weight_tying(config, enable_weight_tying)
    set_gemma4_sharding_config(
        config, enable_sp=enable_sp, enable_ep=enable_ep
    )

    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            config = c.build().convert(config)
    return config


gemma4_configs = {
    "testmodel": _build_flavor_config("testmodel", "sdpa"),
    "testmodel_moe": _build_flavor_config("testmodel_moe", "sdpa"),
    "debugmodel": _build_flavor_config("debugmodel", "sdpa"),
    "debugmodel_moe": _build_flavor_config("debugmodel_moe", "sdpa"),
    "31B": _build_flavor_config("31B", "sdpa"),
    "26B-A4B": _build_flavor_config("26B-A4B", "sdpa"),
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    use_fused_qkv: bool = False,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    enable_sp: bool = True,
    enable_ep: bool = False,
    converters: list[ModelConfigConverter.Config] | None = None,
    enable_weight_tying: bool = True,
) -> ModelSpec:
    config = _build_flavor_config(
        flavor,
        attn_backend=attn_backend,
        use_fused_qkv=use_fused_qkv,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        enable_sp=enable_sp,
        enable_ep=enable_ep,
        converters=converters,
        enable_weight_tying=enable_weight_tying,
    )
    assert config.layers[0].attention.rope is not None
    return ModelSpec(
        name="gemma4",
        flavor=flavor,
        model=config,
        max_context_length=config.layers[0].attention.rope.max_context_length,
        parallelize_fn=parallelize_gemma4,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Gemma4StateDictAdapter,
    )
