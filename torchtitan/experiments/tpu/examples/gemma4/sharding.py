from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.experiments.tpu.examples.gemma4.model import (
    Gemma4Attention,
    Gemma4Router,
)
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    colwise_sharding_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    feed_forward_sharding_config,
    gqa_attention_sharding_config,
    inner_attention_local_spmd_sharding_config,
    lm_head_sharding_config,
    norm_sharding_config as norm_config,
    replicated_activation_placement,
    residual_activation_placement,
    rowwise_sharding_config,
    stacked_colwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.moe.sharding import (
    moe_sharding_config,
    routed_expert_linear_sharding_config,
    routed_experts_sharding_config,
    router_gate_sharding_config,
    router_sharding_config,
)
from torchtitan.models.base.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.experiments.tpu.examples.gemma4.model import (
        Gemma4Model,
        Gemma4TransformerBlock,
    )


def set_decoder_sharding_config(config: "Gemma4Model.Config", *, enable_sp: bool) -> None:
    if config.tok_embeddings is not None:
        config.tok_embeddings.sharding_config = token_embedding_sharding_config(enable_sp=enable_sp)
    if config.norm is not None:
        config.norm.sharding_config = norm_config(enable_sp=enable_sp)
    if config.lm_head is not None:
        config.lm_head.sharding_config = lm_head_sharding_config()


def set_gqa_attention_sharding(attention: Gemma4Attention.Config, *, enable_sp: bool) -> None:
    attention.sharding_config = gqa_attention_sharding_config(enable_sp=enable_sp)
    in_layout = residual_activation_placement(enable_sp=enable_sp)
    attention.qkv_linear.wqkv.sharding_config = colwise_sharding_config(input_layout=in_layout)
    attention.wo.sharding_config = rowwise_sharding_config(
        output_layout=in_layout
    )


def set_gqa_inner_attention_local_spmd(inner_attention) -> None:
    inner_attention.sharding_config = inner_attention_local_spmd_sharding_config()


def set_dense_ffn_sharding(ffn, *, attn_x_layout, enable_sp: bool) -> None:
    ffn.sharding_config = feed_forward_sharding_config(enable_sp=enable_sp)
    ffn.w13.sharding_config = stacked_colwise_sharding_config(input_layout=attn_x_layout)
    ffn.w2.sharding_config = rowwise_sharding_config(
        output_layout=attn_x_layout
    )


def set_moe_sharding_config(moe, *, enable_ep: bool, enable_sp: bool) -> None:
    moe.sharding_config = moe_sharding_config(enable_sp=enable_sp)
    moe.router.sharding_config = router_sharding_config()
    moe.router.gate.sharding_config = router_gate_sharding_config()
    moe.routed_experts.sharding_config = routed_experts_sharding_config()
    if getattr(moe.routed_experts, "inner_experts", None) is not None:
        moe.routed_experts.inner_experts.sharding_config = (
            routed_expert_linear_sharding_config(enable_ep=enable_ep)
        )


def set_gemma4_sharding_config(
    config: "Gemma4Model.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all Gemma4 sub-configs."""
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_gemma4_layer_sharding(
            layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep
        )


def _set_gemma4_layer_sharding(
    layer_cfg: "Gemma4TransformerBlock.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Set sharding on one Gemma4 transformer layer."""
    attention = layer_cfg.attention
    assert isinstance(attention, Gemma4Attention.Config)

    layer_cfg.sharding_config = ShardingConfig(
        state_shardings={
            "layer_scalar": dense_param_placement(
                tp=spmd.R if enable_sp else spmd.I
            )
        }
    )

    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.post_attention_layernorm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    layer_cfg.post_feedforward_layernorm.sharding_config = norm
    if layer_cfg.post_feedforward_layernorm_1 is not None:
        layer_cfg.post_feedforward_layernorm_1.sharding_config = norm
    if layer_cfg.post_feedforward_layernorm_2 is not None:
        layer_cfg.post_feedforward_layernorm_2.sharding_config = norm

    set_gqa_attention_sharding(attention, enable_sp=enable_sp)
    set_gqa_inner_attention_local_spmd(attention.inner_attention)

    # Q, K, and V norms operate on (T, local_heads, head_dim) with TP sharding dim 1.
    head_layout = attention_activation_placement()
    if attention.qk_norm is not None:
        attention.qk_norm.sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.R)},
            input_shardings={"input": head_layout},
            output_shardings=head_layout,
        )
    attention.v_norm.sharding_config = ShardingConfig(
        input_shardings={"input": head_layout},
        output_shardings=head_layout,
    )

    # Dense FFN (present on both dense and MoE layers in Gemma4)
    if layer_cfg.feed_forward is not None:
        attn_x_layout = (
            dense_sequence_parallel_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
        )
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_layout=attn_x_layout,
            enable_sp=enable_sp,
        )

    # MoE FFN (MoE-enabled layers only)
    if layer_cfg.moe is not None:
        set_moe_sharding_config(
            layer_cfg.moe,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
        )
        moe_in_layout = (
            (
                dense_sequence_parallel_placement()
                if enable_sp
                else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
            )
            if enable_ep
            else dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        )
        moe_norm_weight_tp = (
            (spmd.R if enable_sp else spmd.I) if enable_ep else spmd.R
        )
        if layer_cfg.moe.pre_feedforward_layernorm_2 is not None:
            layer_cfg.moe.pre_feedforward_layernorm_2.sharding_config = (
                ShardingConfig(
                    state_shardings={
                        "weight": dense_param_placement(tp=moe_norm_weight_tp)
                    },
                    input_shardings={"input": moe_in_layout},
                    output_shardings=moe_in_layout,
                )
            )
        router = layer_cfg.moe.router
        assert isinstance(router, Gemma4Router.Config)
        router.norm.sharding_config = ShardingConfig(
            input_shardings={"input": moe_in_layout},
            output_shardings=moe_in_layout,
        )
        # Extend the router config set_moe_sharding_config filled in (it
        # carries the tokens_per_expert_E buffer and padding-mask layouts) with
        # Gemma4Router's own parameters.
        router_config = router.sharding_config or ShardingConfig()
        router_config.state_shardings = {
            **(router_config.state_shardings or {}),
            "scale": dense_param_placement(tp=moe_norm_weight_tp),
            "per_expert_scale": dense_param_placement(tp=spmd.R),
        }
        router.sharding_config = router_config
