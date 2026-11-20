"""TPU configuration registry for Qwen3.8 (text-only) models.

Recipe helpers come from the Qwen3.5 TPU registry: Qwen3.8 is the same
architecture (see the package docstring).
"""

from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRoutedExpertsConverter,
)
from torchtitan.experiments.tpu.qwen3_5 import config_registry as qwen35_recipes
from torchtitan.experiments.tpu.qwen3_8 import model_registry
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

# The Qwen3.5 TPU MoE converters plus `TpuRoutedExperts`, which binds the
# EP rank that the ragged EP transport needs (a no-op otherwise, with
# `scale_before_down_proj` off). As in the Qwen3 EP recipes.
# pylint: disable-next=protected-access
_TPU_MOE_CONVERTERS = qwen35_recipes._TPU_MOE_CONVERTERS + [
    TpuRoutedExpertsConverter.Config()
]


def _use_ragged_ep_transport(cfg: TPUTrainerConfig) -> None:
  """Selects the dropless `ragged_all_to_all` EP transport, bounded.

  torch_tpu's `all_to_all` rejects asymmetric output buffers, which
  upstream's variable-size EP transport produces. The ragged transport's
  receive buffer would otherwise be sized for the worst case (`ep_size * T *
  top_k` rows per rank: about 60 of the 91 GiB peak per rank of the TPU7x
  proxy, and 64x the routed rows at EP=64). `ep_capacity_factor=2.0` with
  `ep_capacity_overflow="grow"` sizes it at twice this rank's routed rows,
  doubling as needed so that no route is dropped (the default "drop" would
  drop the routes past the cap).
  """
  assert cfg.model_spec is not None
  dispatchers = list(
      cfg.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config)
  )
  if not dispatchers:
    raise ValueError("Expected TpuAllToAllTokenDispatcher (_TPU_MOE_CONVERTERS)")
  for _, dispatcher, _, _ in dispatchers:
    dispatcher.ep_transport = "ragged_all_to_all"
    dispatcher.ep_capacity_factor = 2.0
    dispatcher.ep_capacity_overflow = "grow"


def qwen38_debugmodel() -> TPUTrainerConfig:
  """Debug config for hybrid Qwen3.8 (dense) on TPU (= `qwen35_debugmodel`)."""
  cfg = qwen35_recipes.qwen35_debugmodel()
  cfg.model_spec = model_registry("debugmodel", seq_len=256)
  return cfg


def qwen38_moe_debugmodel() -> TPUTrainerConfig:
  """Debug config for 4-layer hybrid Qwen3.8 MoE on TPU.

  TPU token dispatcher and grouped experts, with upstream's expert-bias load
  balancer (`load_balance_coeff`), as upstream's `qwen38_debugmodel_moe`.
  """
  cfg = qwen38_debugmodel()
  cfg.model_spec = model_registry(
      "debugmodel_moe", seq_len=256, converters=_TPU_MOE_CONVERTERS
  )
  return cfg


def qwen38_moe_debugmodel_ep8() -> TPUTrainerConfig:
  """`qwen38_moe_debugmodel` with EP=8 on 8 ranks (FSDP 8)."""
  cfg = qwen38_moe_debugmodel()
  cfg.parallelism.expert_parallel_degree = 8
  _use_ragged_ep_transport(cfg)
  return cfg
