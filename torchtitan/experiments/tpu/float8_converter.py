"""Torchao float8 linear converter for TPU.

`Float8LinearConverter` targets CUDA/ROCm, and two of its choices do not fit
TPU:

  1. Its constructor raises unless the device is NVIDIA SM89+ or AMD gfx942+.
     TPU v7x has native float8 matmuls (TorchTPU lowers `torch._scaled_mm`).
  2. torchao's recipes request `use_fast_accum=True` for the forward GEMM,
     which TorchTPU's `_scaled_mm` rejects; see `TpuFloat8LinearConverter`.

`TpuFloat8LinearConverter` builds the torchao config itself instead of patching
the CUDA converter. The model swap (`convert`) is inherited, so the resulting
`Float8Linear` layers are identical to the upstream ones.
"""

import dataclasses
from functools import partial

from torchao.float8.config import Float8LinearConfig
from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.components.quantization.utils import module_filter_fn
from torchtitan.tools.logging import logger


def tpu_float8_linear_config(
    recipe_name: str = "rowwise", *, emulate: bool = False
) -> Float8LinearConfig:
  """Returns a torchao `Float8LinearConfig` with `use_fast_accum=False` for TPU."""
  if emulate:
    torchao_config = Float8LinearConfig(emulate=True)
  else:
    torchao_config = Float8LinearConfig.from_recipe_name(recipe_name)
  return dataclasses.replace(
      torchao_config,
      gemm_config_output=dataclasses.replace(
          torchao_config.gemm_config_output, use_fast_accum=False
      ),
  )


class TpuFloat8LinearConverter(Float8LinearConverter):
  """Replaces matching `Linear.Config`s with torchao `Float8Linear` on TPU."""

  @dataclasses.dataclass(kw_only=True, slots=True)
  class Config(Float8LinearConverter.Config):
    pass

  # Intentionally does not call `Float8LinearConverter.__init__`: its only
  # TPU-relevant work is building the torchao config, done here without the
  # CUDA/ROCm capability check.
  def __init__(self, config: Config):  # pylint: disable=super-init-not-called
    self.config = config
    self.torchao_config = tpu_float8_linear_config(
        config.recipe_name, emulate=config.emulate
    )
    if "auto_filter_small_kn" in config.filter_fqns:
      raise ValueError(
          "auto_filter_small_kn uses CUDA-tuned thresholds; list the FQNs to"
          " skip explicitly on TPU."
      )
    self.filter_fn = partial(module_filter_fn, filter_fqns=config.filter_fqns)
    self.enabled = True
    logger.info(f"TPU float8 training active with recipe {config.recipe_name}")


__all__ = ["TpuFloat8LinearConverter", "tpu_float8_linear_config"]
