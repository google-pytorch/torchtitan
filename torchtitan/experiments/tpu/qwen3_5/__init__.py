"""Qwen3.5 (text-only Dense and MoE) on TPU.

Wraps upstream ``torchtitan.models.qwen3_5`` for the TPU lane
(``use_tpu_modules``):

* Drops the vision encoder (text-only training).
* Swaps every GatedDeltaNet's inner module for ``TpuInnerGatedDeltaNet`` and
  its output norm for ``TpuRMSNormGated``; upstream's inner module runs
  CUDA/Triton ``attn_gym`` kernels.
* Swaps every layer for ``TpuQwen35TransformerBlock`` and every GatedDeltaNet
  for ``TpuGatedDeltaNet``, which pass ``positions`` down to
  ``TpuInnerGatedDeltaNet``, so packed documents restart the causal
  convolution and the delta-rule state (upstream: ``cu_seqlens``, built only
  for the flex and varlen backends), unless ``document_resets.enabled=False``
  (the pretraining benchmark recipes).
* Swaps every full-attention layer for ``TpuQwen35Attention`` with a
  ``SlidingWindowScaledDotProductInnerAttention`` without a window, so packed
  documents are masked on SDPA and the Splash patch sees ``positions``.
* Uses ``TpuQwen35Model``, whose context-parallel input sharding goes through
  the CP inner attention's ``cp_shard`` (as core ``Decoder`` does). CP recipes
  also run the GatedDeltaNet head-parallel (``cp_gated_delta``; see
  ``config_registry._qwen35_cp``).
* Turns off the expert-bias balancer (``load_balance_coeff=None``): HF
  checkpoints, and so ``Qwen35StateDictAdapter``, have no ``expert_bias``.
* Gives the GatedDeltaNet's depthwise convolutions a ``param_init`` (the
  ``nn.Conv1d`` default): their ``reset_parameters()`` would initialize
  SimpleFSDP's all-gathered copy, not the sharded weight.
* Parallelizes with ``infra.parallelize.parallelize_qwen3_5``.
"""

import dataclasses
from functools import partial
import math

from torch import nn
from torchtitan.config import derive
from torchtitan.experiments.tpu.muse_glimmer.attention import SlidingWindowScaledDotProductInnerAttention

# Before `torchtitan.models.qwen3_5`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu.qwen3_5 import attn_gym_stub  # pylint: disable=unused-import

# Registers the GatedDeltaNet CP validator and reverter
# (validate_tpu_context_parallel / revert_tpu_context_parallel).
from torchtitan.experiments.tpu.qwen3_5 import cp_gated_delta  # pylint: disable=unused-import
from torchtitan.experiments.tpu.qwen3_5.gated_delta import (
    TpuGatedDeltaKernel,
    TpuGatedDeltaNet,
    TpuInnerGatedDeltaNet,
    TpuRMSNormGated,
)
from torchtitan.experiments.tpu.qwen3_5.infra.parallelize import parallelize_qwen3_5
from torchtitan.experiments.tpu.qwen3_5.model import TpuQwen35Attention, TpuQwen35Model, TpuQwen35TransformerBlock
import torchtitan.models.qwen3_5 as upstream_qwen3_5
from torchtitan.models.qwen3_5 import Qwen35Model, qwen3_5_configs
from torchtitan.config.transform import ModelConfigConverter

__all__ = [
    "model_registry",
    "qwen3_5_configs",
    "use_tpu_modules",
]

# `nn.Conv1d.reset_parameters`'s initializer (the convolutions have no bias).
_CONV1D_INIT = {"weight": partial(nn.init.kaiming_uniform_, a=math.sqrt(5))}


def use_tpu_modules(config: Qwen35Model.Config) -> TpuQwen35Model.Config:
  """Returns ``config`` as a text-only TPU config (see the module docstring).

  Args:
    config: An upstream Qwen3.5 model config; its layer configs are modified in
      place.

  Returns:
    A ``TpuQwen35Model.Config`` without a vision encoder.
  """
  for i, layer in enumerate(config.layers):
    if layer.attention is not None:
      layer.attention = derive(
          layer.attention,
          TpuQwen35Attention.Config,
          inner_attention=SlidingWindowScaledDotProductInnerAttention.Config(),
      )
    if layer.delta_net is not None:
      layer.delta_net = derive(
          layer.delta_net,
          TpuGatedDeltaNet.Config,
          inner_gated_delta_net=TpuInnerGatedDeltaNet.Config(
              kernel=TpuGatedDeltaKernel.Config()
          ),
          norm=derive(layer.delta_net.norm, TpuRMSNormGated.Config),
      )
      # TPU: SimpleFSDP-safe init; reset_parameters() inits a gathered copy.
      for conv in (
          layer.delta_net.conv_q,
          layer.delta_net.conv_k,
          layer.delta_net.conv_v,
      ):
        conv.param_init = _CONV1D_INIT
    if layer.moe is not None:
      layer.moe.load_balance_coeff = None
    config.layers[i] = derive(layer, TpuQwen35TransformerBlock.Config)
  return derive(config, TpuQwen35Model.Config, vision_encoder=None)


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "sdpa",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> Qwen35Model.Config:
  """Returns a TPU ``Qwen35Model.Config`` for a text-only Qwen3.5 flavor."""
  if attn_backend != "sdpa":
    raise ValueError(
        "Qwen3.5 on TPU supports only attn_backend='sdpa', got "
        f"{attn_backend!r}"
    )
  config = upstream_qwen3_5.model_registry(
      flavor,
      seq_len=seq_len,
      attn_backend="sdpa",
      converters=converters,
  )
  return use_tpu_modules(config)


from .config_registry import *  # pylint: disable=wildcard-import,g-import-not-at-top,g-bad-import-order
