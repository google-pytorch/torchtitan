"""Sends the router's `tokens_per_expert_E` accumulation to a throwaway tensor.

The counter only ever feeds the expert-bias update, so with no
`load_balance_coeff` nothing reads it -- but `TokenChoiceTopKRouter.forward`
writes it anyway, and a scan body cannot write to a tensor it closed over.
"""

from dataclasses import dataclass, replace

import torch

from torchtitan.config import derive, override
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter


class _DiscardedAuxLossMixin:
  """Avoid mutating aux_loss accumulator buffers inside scan."""

  def _accumulate_expert_counts(self, counts_E: torch.Tensor) -> torch.Tensor:
    return self._reduce_token_partials(counts_E, include_dp_axis=True)

  def __getattr__(self, name: str):
    value = super().__getattr__(name)
    if name != "instance_acc":
      return value
    return torch.zeros_like(value)


_DISCARDED_AUX_LOSS_SUBCLASSES: dict[type, type] = {}


class TpuDiscardedTokenCounterRouter(TokenChoiceTopKRouter):
  """`TokenChoiceTopKRouter` that hands out a throwaway in place of its counter.

  Only the `tokens_per_expert_E` attribute differs. Every parameter, buffer, and
  checkpoint key is inherited unchanged, and the output is bit-identical to the
  base class.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TokenChoiceTopKRouter.Config):
    pass

  def __init__(self, config: Config):
    super().__init__(config)
    if self.aux_loss is not None:
      cls = type(self.aux_loss)
      if not issubclass(cls, _DiscardedAuxLossMixin):
        sub_cls = _DISCARDED_AUX_LOSS_SUBCLASSES.get(cls)
        if sub_cls is None:
          sub_cls = type(
              f"TpuDiscarded{cls.__name__}",
              (_DiscardedAuxLossMixin, cls),
              {},
          )
          _DISCARDED_AUX_LOSS_SUBCLASSES[cls] = sub_cls
        self.aux_loss.__class__ = sub_cls

  def __getattr__(self, name: str):
    value = super().__getattr__(name)
    if name != "tokens_per_expert_E":
      return value
    # The base class's `add_` has to land on a tensor allocated *inside* the
    # scan body. Swapping `_buffers` around the call instead would not do: that
    # is a mutation of state the body closed over, which is the very thing a
    # higher-order op rejects. A plain property cannot do this either --
    # `register_buffer` probes `hasattr` and would refuse the name.
    return torch.zeros_like(value)


class TpuDiscardedTokenCounterMoE(MoE):
  """`MoE` whose router discards the token counter when nothing reads it.

  Kept as the override's target type so recipes and tests can detect it; the
  behavior lives in `TpuDiscardedTokenCounterRouter`.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(MoE.Config):
    pass


@override(
    # Upstream d771abb0 made `override()` keyword-only and removed the separate
    # name argument: an override is now identified by the decorated factory's
    # import path (`...overrides.discarded_token_counter.tpu_discarded_token_counter`),
    # which is what `override.imports` lists.
    target=MoE.Config,
    exact=True,
    description=(
        "Send the router's dead token-counter accumulation to a throwaway so a"
        " decoder layer can be traced inside scan."
    ),
)
def tpu_discarded_token_counter(
    config: MoE.Config,
) -> TpuDiscardedTokenCounterMoE.Config:
  config = derive(config, TpuDiscardedTokenCounterMoE.Config)
  # With a coefficient the expert-bias hook reads the counter, so it must
  # accumulate; the counter moved from MoE to the router upstream (d34a13fd).
  if config.load_balance_coeff is None and (
      type(config.router) is TokenChoiceTopKRouter.Config
  ):
    config = replace(
        config,
        router=derive(config.router, TpuDiscardedTokenCounterRouter.Config),
    )
  return config


__all__ = [
    "TpuDiscardedTokenCounterMoE",
    "TpuDiscardedTokenCounterRouter",
    "tpu_discarded_token_counter",
]
