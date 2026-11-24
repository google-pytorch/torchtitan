"""Activation-checkpoint (remat) policies for the TPU lane."""

from dataclasses import dataclass, field
from typing import Any

from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)


@dataclass(kw_only=True, slots=True)
class TPUSelectiveACConfig(DefaultActivationRematConfig):
  """``DefaultActivationRematConfig`` for TPU context parallelism."""


class TPUSelectiveAC:
  """Compatibility wrapper around ``TPUSelectiveACConfig``."""

  Config = TPUSelectiveACConfig


def tpu_selective_ac_config(ac_config: Any) -> Any:
  """Return ``ac_config`` as a ``TPUSelectiveACConfig`` if it is a default policy."""
  if type(ac_config) is not DefaultActivationRematConfig:
    return ac_config
  return TPUSelectiveACConfig()


@dataclass(kw_only=True, slots=True)
class AttentionSavingSelectiveACConfig(ActivationRematConfig):
  """``ActivationRematConfig`` that saves the inner-attention output on TPU."""

  save_regions: list[str] = field(default_factory=lambda: ["*inner_attention*"])
  recompute_regions: list[str] = field(default_factory=list)


class AttentionSavingSelectiveAC:
  """Compatibility wrapper around ``AttentionSavingSelectiveACConfig``."""

  Config = AttentionSavingSelectiveACConfig


__all__ = [
    "ActivationRematConfig",
    "AttentionSavingSelectiveAC",
    "AttentionSavingSelectiveACConfig",
    "DefaultActivationRematConfig",
    "FullActivationRematConfig",
    "TPUSelectiveAC",
    "TPUSelectiveACConfig",
    "tpu_selective_ac_config",
]
