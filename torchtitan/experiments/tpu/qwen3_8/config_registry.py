"""TPU configuration registry for Qwen3.8 (text-only) models.

Recipe helpers come from the Qwen3.5 TPU registry: Qwen3.8 is the same
architecture (see the package docstring).
"""

from torchtitan.experiments.tpu.qwen3_5 import config_registry as qwen35_recipes
from torchtitan.experiments.tpu.qwen3_8 import model_registry
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig


def qwen38_debugmodel() -> TPUTrainerConfig:
  """Debug config for hybrid Qwen3.8 (dense) on TPU (= `qwen35_debugmodel`)."""
  cfg = qwen35_recipes.qwen35_debugmodel()
  cfg.model_spec = model_registry("debugmodel", seq_len=256)
  return cfg
