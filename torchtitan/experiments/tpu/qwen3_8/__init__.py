"""Qwen3.8 (text-only) on TPU.

Qwen3.8 checkpoints use the Qwen3.5 architecture: upstream
``torchtitan.models.qwen3_8`` is a version-specific registry over
``torchtitan.models.qwen3_5``. Likewise, this package is a registry over the
upstream ``qwen3_8`` flavors that applies the Qwen3.5 TPU lane
(``experiments/tpu/qwen3_5``, ``to_tpu_model_spec``): TPU GatedDeltaNet and
document-masked SDPA modules, no vision encoder, SimpleFSDP-aware
parallelization. It imports the Qwen3.5 TPU package because it is the same
model, not a different one.

``scaled_down`` builds the real-per-rank-shape proxies of the 2.4T-A95B
flavor: full width (dim, heads, expert hidden size), fewer layers and fewer
experts.
"""

import dataclasses

# Before `torchtitan.models.qwen3_5`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu.qwen3_5 import attn_gym_stub  # pylint: disable=unused-import,g-bad-import-order
from torchtitan.experiments.tpu.qwen3_5 import to_tpu_model_spec
from torchtitan.models.common.moe import MoE
from torchtitan.models.qwen3_5 import Qwen35Model
import torchtitan.models.qwen3_8 as upstream_qwen3_8
from torchtitan.models.qwen3_8 import qwen3_8_configs
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

__all__ = [
    "model_registry",
    "qwen3_8_configs",
    "scaled_down",
]


def _set_num_experts(config: object, num_experts: int) -> int:
  """Sets every ``num_experts`` field reachable from ``config``.

  Returns:
    How many fields were set.
  """
  count = 0
  if not dataclasses.is_dataclass(config):
    return count
  for field in dataclasses.fields(config):
    value = getattr(config, field.name)
    if field.name == "num_experts" and isinstance(value, int):
      setattr(config, field.name, num_experts)
      count += 1
    elif dataclasses.is_dataclass(value):
      count += _set_num_experts(value, num_experts)
  return count


def scaled_down(
    config: Qwen35Model.Config,
    *,
    num_layers: int | None = None,
    num_experts: int | None = None,
) -> Qwen35Model.Config:
  """Returns ``config`` with its first ``num_layers`` layers and ``num_experts``.

  Every other dimension stays as in ``config``, so per-rank shapes match the
  full model when ``num_experts / expert_parallel_degree`` matches it.

  Args:
    config: A Qwen3.5-architecture model config; modified in place.
    num_layers: Layers to keep, from the first one. ``None`` keeps all.
    num_experts: Routed experts per MoE layer. ``None`` keeps the count.

  Returns:
    ``config``.
  """
  if num_layers is not None:
    if not 0 < num_layers <= len(config.layers):
      raise ValueError(
          f"num_layers={num_layers} must be in [1, {len(config.layers)}]"
      )
    config.layers = config.layers[:num_layers]
  if num_experts is not None:
    for _, moe, _, _ in config.traverse(MoE.Config):
      if num_experts < moe.router.top_k:
        raise ValueError(
            f"num_experts={num_experts} is below top_k={moe.router.top_k}"
        )
      # The router gate's output width is the expert count too; without it
      # the router picks expert ids beyond `num_experts`, which the token
      # counts drop, leaving grouped-matmul rows in no group.
      if moe.router.gate.out_features != moe.num_experts:
        raise ValueError(
            f"router gate out_features={moe.router.gate.out_features} does "
            f"not match num_experts={moe.num_experts}"
        )
      moe.router.gate.out_features = num_experts
      # MoE, router, inner experts and token dispatcher each carry the count.
      if _set_num_experts(moe, num_experts) < 4:
        raise ValueError("Expected num_experts on MoE, router, experts and "
                         "token dispatcher")
  return config


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "sdpa",
    converters: list[ModelConfigConverter.Config] | None = None,
    num_layers: int | None = None,
    num_experts: int | None = None,
) -> ModelSpec:
  """Returns a TPU ``ModelSpec`` for a text-only Qwen3.8 flavor.

  Args:
    flavor: An upstream Qwen3.8 flavor (``debugmodel``, ``debugmodel_moe``,
      ``27B``, ``2.4T-A95B``).
    seq_len: Context length; defaults to the flavor's maximum.
    attn_backend: Only ``"sdpa"`` (see ``experiments/tpu/qwen3_5``).
    converters: Optional model-config converters.
    num_layers: Keep only the first ``num_layers`` layers (proxies).
    num_experts: Routed experts per MoE layer (proxies).

  Returns:
    The model spec, named ``qwen3_8``.
  """
  if attn_backend != "sdpa":
    raise ValueError(
        "Qwen3.8 on TPU supports only attn_backend='sdpa', got "
        f"{attn_backend!r}"
    )
  model_spec = upstream_qwen3_8.model_registry(
      flavor, seq_len=seq_len, attn_backend="sdpa", converters=converters
  )
  # `to_tpu_model_spec` turns off the expert-bias balancer because HF
  # checkpoints carry no `expert_bias`. Keep upstream's: its Qwen3.8 recipes
  # train with it.
  balance_coeffs = [
      moe.load_balance_coeff
      for _, moe, _, _ in model_spec.model.traverse(MoE.Config)
  ]
  model_spec = to_tpu_model_spec(model_spec)
  for (_, moe, _, _), coeff in zip(
      model_spec.model.traverse(MoE.Config), balance_coeffs, strict=True
  ):
    moe.load_balance_coeff = coeff
  if num_layers is not None or num_experts is not None:
    scaled_down(model_spec.model, num_layers=num_layers, num_experts=num_experts)
  return model_spec


from .config_registry import *  # pylint: disable=wildcard-import,g-import-not-at-top,g-bad-import-order
