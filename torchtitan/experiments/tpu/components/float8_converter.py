"""Torchao float8 linear converter for TPU.

Upstream removed Float8 training in #4961; the TPU lane retains
`TpuFloat8LinearConverter` and the `Float8Linear` module under
`experiments/tpu/components/float8/` because TPU v7x has native float8 matmuls
(TorchTPU lowers `torch._scaled_mm`).
"""

import dataclasses
import logging
from functools import partial
from typing import Literal

from torchao.float8.config import Float8LinearConfig
from torchtitan.config.transform.converter import ModelConfigConverter
from torchtitan.config.transform.quantization import _validate_quantizable_linear
from torchtitan.experiments.tpu.components.float8 import Float8Linear
from torchtitan.models.common.linear import Linear
from torchtitan.quantization.utils import get_quantized_linear


def module_filter_fn(
    mod: Linear.Config,
    fqn: str,
    filter_fqns: list[str],
) -> bool:
  """Return whether ``mod`` at ``fqn`` should be quantized."""
  del mod
  return not any(f in fqn for f in filter_fqns)

logger = logging.getLogger(__name__)


class TpuFloat8LinearConverter(ModelConfigConverter):
  """Replaces matching `Linear.Config`s with `Float8Linear` on TPU."""

  @dataclasses.dataclass(kw_only=True, slots=True)
  class Config(ModelConfigConverter.Config):
    recipe_name: Literal["rowwise", "rowwise_with_gw_hp"] = "rowwise"
    """Float8 recipe name."""

    filter_fqns: list[str] = dataclasses.field(default_factory=list)
    """Fully qualified names of modules to skip applying float8 training to."""

    emulate: bool = False
    """If True, emulation is used instead of hardware accelerated gemm."""

  def __init__(self, config: Config):
    self.config = config
    if config.emulate:
      torchao_config = Float8LinearConfig(emulate=True)
    else:
      torchao_config = Float8LinearConfig.from_recipe_name(config.recipe_name)
    self.torchao_config = torchao_config
    if "auto_filter_small_kn" in config.filter_fqns:
      raise ValueError(
          "auto_filter_small_kn uses CUDA-tuned thresholds; list the FQNs to"
          " skip explicitly on TPU."
      )
    self.filter_fn = partial(module_filter_fn, filter_fqns=config.filter_fqns)
    self.enabled = True
    logger.info(f"TPU float8 training active with recipe {config.recipe_name}")

  def convert(self, model_config):
    if not self.enabled:
      return model_config

    for fqn, linear_config, parent, attr in model_config.traverse(Linear.Config):
      if self.filter_fn(linear_config, fqn):
        _validate_quantizable_linear(linear_config, fqn)
        owner = linear_config._owner
        assert owner is not None and issubclass(owner, Linear)
        config_cls = get_quantized_linear(Float8Linear, owner).Config
        new_config = config_cls(
            in_features=linear_config.in_features,
            out_features=linear_config.out_features,
            num_linears=linear_config.num_linears,
            bias=linear_config.bias,
            param_init=linear_config.param_init,
            sharding_config=linear_config.sharding_config,
            recipe_name=self.config.recipe_name,
            emulate=self.config.emulate,
        )
        if parent is None:
          model_config = new_config
        elif isinstance(parent, list):
          parent[attr] = new_config
        else:
          setattr(parent, attr, new_config)

    logger.info("Swapped to Float8Linear layers")
    return model_config


__all__ = ["TpuFloat8LinearConverter"]
