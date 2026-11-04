"""Sharding configs for Flint: Qwen3 MoE sharding plus the per-head sinks."""

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.experiments.tpu.examples.flint.model import FlintAttention
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
  from torchtitan.experiments.tpu.examples.flint.model import FlintModel


_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
}


def set_flint_sharding_config(
    config: "FlintModel.Config", *, enable_sp: bool, enable_ep: bool
) -> None:
  """Fill ``sharding_config`` on all Flint sub-configs."""
  set_decoder_sharding_config(config, enable_sp=enable_sp)
  for layer_cfg in config.layers:
    attention = layer_cfg.attention
    assert isinstance(attention, FlintAttention.Config)

    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm

    set_gqa_attention_sharding(attention, enable_sp=enable_sp)
    # Sinks are replicated: Flint rejects TP (see FlintModel.Config).
    attention.sharding_config.state_shardings = {
        "sinks": dense_param_placement(tp=spmd.R)
    }
    set_gqa_inner_attention_local_map(attention.inner_attention)

    head_layout = attention_activation_placement()
    attention.qk_norm.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"input": head_layout},
        in_dst_shardings={"input": head_layout},
        out_src_shardings=head_layout,
        out_dst_shardings=head_layout,
    )

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

    if layer_cfg.moe is not None:
      set_moe_sharding_config(
          layer_cfg.moe,
          enable_ep=enable_ep,
          enable_sp=enable_sp,
          expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
      )
