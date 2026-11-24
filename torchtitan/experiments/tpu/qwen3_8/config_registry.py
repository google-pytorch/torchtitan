"""TPU configuration registry for Qwen3.8 (text-only) models.

Recipe helpers come from the Qwen3.5 TPU registry: Qwen3.8 is the same
architecture (see the package docstring).
"""

from torchtitan.experiments.tpu.qwen3_5 import config_registry as qwen35_recipes
from torchtitan.experiments.tpu.qwen3_8 import model_registry
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

# pylint: disable-next=protected-access
_TPU_MOE_CONVERTERS = qwen35_recipes._TPU_MOE_CONVERTERS


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
