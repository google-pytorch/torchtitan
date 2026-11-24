# Copyright 2026 Google LLC

"""Two-projection SwiGLU feed-forward for the non-scan TPU trainer."""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.config import derive
from torchtitan.config.transform import ModelConfigConverter
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear


class TpuTwoProjFeedForward(FeedForward):
  """Run the gate and up projections as two GEMMs over the stacked ``w13``.

  Upstream ``FeedForward`` (pytorch/torchtitan#4808) runs one
  ``[T, D] x [2F, D]`` GEMM and splits the ``[T, 2, F]`` activation with
  ``unbind(-2)``, reading ``w13.weight`` twice per call (once for the GEMM,
  once for its shape). This class reads the weight once, so SimpleFSDP gathers
  it once, and splits it instead: ``w13.weight`` is ``[2, F, D]`` with
  contiguous gate and up halves, so ``unbind(0)`` yields the two projection
  matrices. The parameter stays upstream's: same name, shape, sharding and
  state dict.

  Used by the non-scan ``TPUTrainer`` only. Measured on v6e-8, 15 steps,
  ``llama3_8b``: upstream's form 3.239 s/step and 30.53 GiB peak HBM, this
  form 2.955 s/step and 29.84 GiB (main, on the former ``TpuPackedFeedForward``:
  2.946 s, 29.70 GiB). The graph trainer and scan layers keep upstream's single
  projection: the two-projection form measured slower under scan (185d6b57),
  and the graph-trainer rows did not regress on upstream's form.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(FeedForward.Config):
    pass

  def _w13_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = self.w13.weight
    bias = self.w13.bias
    if isinstance(weight, DTensor):
      # Not on the SPMD-typed TPU paths, whose parameters reach the forward as
      # local tensors; keep upstream's single projection for DTensor weights.
      gate_TF, up_TF = self.w13(x).unbind(-2)
      return gate_TF, up_TF
    gate_weight, up_weight = weight.unbind(0)
    if bias is None:
      gate_bias = up_bias = None
    else:
      gate_bias, up_bias = bias.unbind(0)
    return F.linear(x, gate_weight, gate_bias), F.linear(x, up_weight, up_bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate_TF, up_TF = remat.region(
        self._w13_forward,
        self.remat_region_name("w13"),
        recompute=self.remat_should_recompute("w13"),
    )(x)
    remat.recompute_needs_tensor(gate_TF, up_TF)
    out_TD = remat.region(
        self.w2,
        self.remat_region_name("w2"),
        recompute=self.remat_should_recompute("w2"),
    )(self.activation_fn(gate_TF, up_TF))
    remat.recompute_needs_tensor(out_TD)
    return out_TD


class TpuTwoProjFeedForwardConverter(ModelConfigConverter):
  """Select two-projection FeedForwards before model construction."""

  @dataclass(kw_only=True, slots=True)
  class Config(ModelConfigConverter.Config):
    pass

  def __init__(self, config: Config):
    pass

  def convert(self, model_config):
    matches = list(model_config.traverse(FeedForward.Config))
    for _, config, parent, field_name in matches:
      # Exact types only: a subclass may add projections or override forward
      # (SigmoidGatedFeedForward). Gemma4FeedForward only changes the
      # activation, which `derive` carries over.
      if type(config) is not FeedForward.Config and (
          type(config).__qualname__ != "Gemma4FeedForward.Config"
      ):
        continue
      # E.g. dist_gemm or quantized projections own their GEMM.
      if type(config.w13) is not Linear.Config:
        continue
      replacement = derive(config, TpuTwoProjFeedForward.Config)
      if parent is None:
        model_config = replacement
      elif isinstance(parent, list):
        assert isinstance(field_name, int)
        parent[field_name] = replacement
      else:
        assert isinstance(field_name, str)
        setattr(parent, field_name, replacement)
    return model_config


__all__ = [
    "TpuTwoProjFeedForward",
    "TpuTwoProjFeedForwardConverter",
]
