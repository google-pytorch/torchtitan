from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.experiments.tpu.examples.gemma4.model import (
    Gemma4Attention,
    Gemma4Router,
)
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.experiments.tpu.examples.gemma4.model import (
        Gemma4Model,
        Gemma4TransformerBlock,
    )


_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
}


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
    set_gqa_inner_attention_local_map(attention.inner_attention)

    # Q, K, and V norms operate on (T, local_heads, head_dim) with TP sharding dim 1.
    head_layout = attention_activation_placement()
    if attention.qk_norm is not None:
        attention.qk_norm.sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.R)},
            in_src_shardings={"input": head_layout},
            in_dst_shardings={"input": head_layout},
            out_src_shardings=head_layout,
            out_dst_shardings=head_layout,
        )
    attention.v_norm.sharding_config = ShardingConfig(
        in_src_shardings={"input": head_layout},
        in_dst_shardings={"input": head_layout},
        out_src_shardings=head_layout,
        out_dst_shardings=head_layout,
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
            expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
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
                    in_src_shardings={"input": moe_in_layout},
                    out_src_shardings=moe_in_layout,
                )
            )
        router = layer_cfg.moe.router
        assert isinstance(router, Gemma4Router.Config)
        router.norm.sharding_config = ShardingConfig(
            in_src_shardings={"input": moe_in_layout},
            out_src_shardings=moe_in_layout,
        )
        router.sharding_config = ShardingConfig(
            state_shardings={
                "scale": dense_param_placement(tp=moe_norm_weight_tp),
                "per_expert_scale": dense_param_placement(tp=spmd.R),
            },
        )
