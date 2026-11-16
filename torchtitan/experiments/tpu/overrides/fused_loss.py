# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Output-projection override for fused linear cross entropy on TPU."""

from dataclasses import dataclass

import torch

from torchtitan.config import derive, override
from torchtitan.models.common import Linear
from torchtitan.models.muse_glimmer.model import SoftCappedLinear


class FusedLossLinearWrapper(Linear):
  """LM-head replacement that supplies operands to a fused loss.

  This module owns the same ``lm_head.weight`` as an ordinary ``Linear``, but
  its forward does not execute the projection or the loss. Instead, it returns
  ``(hidden_states, weight.T)``. ``PallasCrossEntropyLoss`` consumes that pair
  and performs the deferred linear projection, softmax, and cross entropy in a
  single fused kernel without materializing the full logits tensor.

  Constructing the wrapper from ``Linear.Config`` preserves the normal weight
  initialization, sharding, optimizer ownership, and checkpoint key.

  When the replaced head was a ``SoftCappedLinear`` (Muse Glimmer), its
  ``soft_cap`` / ``output_multiplier`` are recorded on the wrapper so the head
  it stands in for is auditable; the transform itself is applied by
  ``PallasCrossEntropyLoss`` (configured with the same two values), since the
  wrapper never sees the logits.
  """

  # Tells workarounds.use_output_projection_patch that forward already returns
  # the fused-loss operands, so the overridden forward is safe to bypass.
  returns_loss_operands = True

  @dataclass(kw_only=True, slots=True)
  class Config(Linear.Config):
    soft_cap: float | None = None
    output_multiplier: float = 1.0

  def __init__(self, config: Config):
    if config.bias:
      raise ValueError("Fused linear loss does not support a biased LM head")
    super().__init__(config)
    self.soft_cap = config.soft_cap
    self.output_multiplier = config.output_multiplier

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x, self.weight.t()


@override(
    # Upstream d771abb0 made `override()` keyword-only and removed the separate
    # name argument: an override is now identified by the decorated factory's
    # import path, which is what `override.imports` lists.
    target=Linear.Config,
    fqns=["model.lm_head"],
    exact=True,
    description="Defer the Qwen LM-head projection to TPU fused cross entropy.",
)
def fused_loss_lm_head(cfg: Linear.Config) -> FusedLossLinearWrapper.Config:
  return derive(cfg, FusedLossLinearWrapper.Config)


@override(
    target=SoftCappedLinear.Config,
    fqns=["model.lm_head"],
    exact=True,
    description=(
        "Defer the Muse Glimmer soft-capped LM-head projection to TPU fused"
        " cross entropy (the cap is re-applied inside the loss kernel)."
    ),
)
def fused_loss_soft_capped_lm_head(
    cfg: SoftCappedLinear.Config,
) -> FusedLossLinearWrapper.Config:
  return derive(
      cfg,
      FusedLossLinearWrapper.Config,
      soft_cap=cfg.output_soft_cap_temp,
      output_multiplier=cfg.output_multiplier,
  )


__all__ = [
    "FusedLossLinearWrapper",
    "fused_loss_lm_head",
    "fused_loss_soft_capped_lm_head",
]
