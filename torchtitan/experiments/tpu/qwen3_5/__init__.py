"""Qwen3.5 (text-only Dense and MoE) on TPU.

Wraps upstream ``torchtitan.models.qwen3_5`` for the TPU lane
(``use_tpu_modules``):

* Drops the vision encoder (text-only training).
* Swaps every GatedDeltaNet's inner module for ``TpuInnerGatedDeltaNet``;
  upstream's runs CUDA/Triton ``attn_gym`` kernels.
* Swaps every layer for ``TpuQwen35TransformerBlock`` and every GatedDeltaNet
  for ``TpuGatedDeltaNet``, which pass ``positions`` down to
  ``TpuInnerGatedDeltaNet``, so packed documents restart the causal
  convolution and the delta-rule state (upstream: ``cu_seqlens``, built only
  for the flex and varlen backends).
* Swaps every full-attention layer for ``TpuQwen35Attention`` with a
  ``SlidingWindowScaledDotProductInnerAttention`` without a window, so packed
  documents are masked on SDPA and the Splash patch sees ``positions``.
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
from torchtitan.experiments.tpu.qwen3_5.gated_delta import TpuGatedDeltaKernel, TpuGatedDeltaNet, TpuInnerGatedDeltaNet
from torchtitan.experiments.tpu.qwen3_5.infra.parallelize import parallelize_qwen3_5
from torchtitan.experiments.tpu.qwen3_5.model import TpuQwen35Attention, TpuQwen35Model, TpuQwen35TransformerBlock
from torchtitan.models.common.moe import register_moe_load_balancing_hook
import torchtitan.models.qwen3_5 as upstream_qwen3_5
from torchtitan.models.qwen3_5 import Qwen35Model, qwen3_5_configs
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

__all__ = [
    "model_registry",
    "qwen3_5_configs",
    "use_tpu_modules",
]

# `nn.Conv1d.reset_parameters`'s initializer (the convolutions have no bias).
_CONV1D_INIT = {"weight": partial(nn.init.kaiming_uniform_, a=math.sqrt(5))}


def use_tpu_modules(config: Qwen35Model.Config) -> Qwen35Model.Config:
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

def _split_optimizer_groups_by_mesh(optimizers_container) -> None:
  """Splits every optimizer param group into one group per DeviceMesh.

  With Expert Parallelism the routed experts are DTensors on the
  `(efsdp, ep)` mesh and all other parameters on the `fsdp` mesh; torch_tpu's
  fused and foreach AdamW take one mesh per param group.

  Args:
    optimizers_container: The `OptimizersContainer` to update in place.
  """

  def mesh_key(param: torch.Tensor) -> int | None:
    spec = getattr(param, "_spec", None)  # A DTensor's `DTensorSpec`.
    return None if spec is None else id(spec.mesh)

  for opt in optimizers_container.optimizers:
    new_groups = []
    for group in opt.param_groups:
      names = group.get("param_names")
      options = {
          k: v for k, v in group.items() if k not in ("params", "param_names")
      }
      buckets: dict[int | None, dict[str, list]] = {}
      for i, param in enumerate(group["params"]):
        bucket = buckets.setdefault(
            mesh_key(param), {"params": [], "param_names": []}
        )
        bucket["params"].append(param)
        if names is not None:
          bucket["param_names"].append(names[i])
      for bucket in buckets.values():
        if names is None:
          del bucket["param_names"]
        new_groups.append({**bucket, **options})
    if len(new_groups) > len(opt.param_groups):
      opt.param_groups = []
      for group in new_groups:
        opt.add_param_group(group)


def post_optimizer_build_qwen3_5(
    optimizers,
    model_parts: list[nn.Module],
    parallel_dims,
) -> None:
  """Splits multi-mesh optimizer param groups, then registers the MoE hook."""
  _split_optimizer_groups_by_mesh(optimizers)
  register_moe_load_balancing_hook(optimizers, model_parts, parallel_dims)



def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "sdpa",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
  """Returns a TPU ``ModelSpec`` for a text-only Qwen3.5 flavor.

  Args:
    flavor: An upstream Qwen3.5 flavor (e.g. ``debugmodel``, ``debugmodel_moe``,
      ``35B-A3B``).
    seq_len: Context length; defaults to the flavor's maximum.
    attn_backend: Only ``"sdpa"``, the document-masked SDPA that the Splash and
      Tokamax attention patches replace.
    converters: Optional model-config converters.

  Returns:
    The model spec, using ``parallelize_qwen3_5``.
  """
  if attn_backend != "sdpa":
    raise ValueError(
        "Qwen3.5 on TPU supports only attn_backend='sdpa', got "
        f"{attn_backend!r}"
    )
  model_spec = upstream_qwen3_5.model_registry(
      flavor,
      seq_len=seq_len,
      attn_backend="sdpa",
      converters=converters,
  )
  return dataclasses.replace(
      model_spec,
      model=use_tpu_modules(model_spec.model),
      parallelize_fn=parallelize_qwen3_5,
      pipelining_fn=None,
  )


from .config_registry import *  # pylint: disable=wildcard-import,g-import-not-at-top,g-bad-import-order
