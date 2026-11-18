"""Torchao float8 linear converter for TPU.

`Float8LinearConverter` targets CUDA/ROCm: its constructor raises unless the
device is NVIDIA SM89+ or AMD gfx942+, whereas TPU v7x has native float8
matmuls (TorchTPU lowers `torch._scaled_mm`).

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
    if config.emulate:
      torchao_config = Float8LinearConfig(emulate=True)
    else:
      torchao_config = Float8LinearConfig.from_recipe_name(config.recipe_name)
    # `use_fast_accum` is a cuBLAS option. Hopper/Ada FP8 tensor cores
    # accumulate in reduced precision; by default cuBLAS periodically promotes
    # the partial sums to FP32, and fast accumulation skips that promotion,
    # trading accuracy for speed. torchao enables it for the forward GEMM
    # (`gemm_config_output`) and keeps the two gradient GEMMs exact. TPU has no
    # such mode: its matmuls accumulate in FP32, and TorchTPU's `_scaled_mm`
    # treats `use_fast_accum` as a no-op, giving the forward GEMM the same
    # accumulation as the gradient GEMMs.
    self.torchao_config = torchao_config
    if "auto_filter_small_kn" in config.filter_fqns:
      raise ValueError(
          "auto_filter_small_kn uses CUDA-tuned thresholds; list the FQNs to"
          " skip explicitly on TPU."
      )
    self.filter_fn = partial(module_filter_fn, filter_fqns=config.filter_fqns)
    self.enabled = True
    logger.info(f"TPU float8 training active with recipe {config.recipe_name}")


__all__ = ["TpuFloat8LinearConverter"]
