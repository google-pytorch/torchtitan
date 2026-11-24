# Copyright 2026 Google LLC

"""Two-projection SwiGLU feed-forward for the non-scan TPU trainer."""

from collections.abc import Callable
from dataclasses import dataclass, replace
import logging

import torch
from torch.distributed.tensor import DTensor, Replicate
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.config import derive
from torchtitan.config.transform import ModelConfigConverter
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.decoder_sharding import colwise_config
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear

logger = logging.getLogger(__name__)


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

  Measured against upstream's form on the regular trainer (v6e-8, 15 steps,
  ``llama3_8b``): 3.239 -> 2.955 s/step and 30.53 -> 29.84 GiB peak HBM (main:
  2.946 s, 29.70 GiB). Zero-2 graph recipes and scan layers keep upstream's
  single projection: two projections cost ``qwen3_8b`` +3.3% step time and
  +4.3 GiB, and measured slower under scan (185d6b57).

  This class keeps upstream's stacked parameter and is what tensor-parallel
  runs of the regular trainer use; with TP degree 1 the converter selects
  ``TpuPackedFeedForward``, which also restores a dim-0 FSDP shard.
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


def _replicated(tensor: torch.Tensor) -> torch.Tensor:
  if isinstance(tensor, DTensor):
    return tensor.redistribute(
        tensor.device_mesh, [Replicate()] * tensor.device_mesh.ndim
    )
  return tensor


class _PackedW13Init:
  """Run upstream's stacked ``[2, ...]`` initializer for a packed parameter.

  A top-level class rather than a closure so model configs stay picklable.
  """

  def __init__(self, stacked_init: Callable) -> None:
    self.stacked_init = stacked_init

  def __call__(self, parameter: torch.Tensor) -> None:
    # A new shape, so for a sharded DTensor parameter this is a replicated
    # DTensor: the values are drawn replicated and `copy_` scatters them.
    stacked = parameter.new_empty(
        (2, parameter.shape[0] // 2, *parameter.shape[1:])
    )
    self.stacked_init(stacked)
    with torch.no_grad():
      parameter.copy_(_replicated(stacked).flatten(0, 1))


class TpuPackedFeedForward(TpuTwoProjFeedForward):
  """Two-projection FFN whose ``w13`` is stored as ``[2F, D]``.

  Upstream stores ``w13.weight`` stacked as ``[2, F, D]``. SimpleFSDP has to
  shard that on ``F`` (dim 1), because dim 0 has two entries, so each rank's
  shard lands in two separate places of the gathered weight. On recipes whose
  collectives are on the critical path that costs step time: measured same-pod
  against the ``[2F, D]`` layout of PR #193, ``qwen3_1_7b_ulysses_cp4`` +3.5%
  and ``muse_glimmer_30b_full_logits`` +3.8%, with the extra time on the
  SparseCore lanes and identical TensorCore ops.

  This class keeps the parameter physically as ``[2F, D]``, which is the
  stacked weight with its first two dims flattened: rows ``[gate; up]``. FSDP
  shards dim 0 and every rank holds one contiguous block. State dicts still
  carry upstream's stacked ``[2, F, D]`` ``w13.weight`` (and ``[2, F]`` bias),
  so checkpoints and the Hugging Face adapters are unaffected. On a sharded
  model the save hook gathers ``w13`` to reshape it, as main's packed FFN did.

  Tensor parallelism is not supported: TP shards ``F`` within each projection,
  which this row order cannot express. ``TpuTwoProjFeedForwardConverter`` only
  selects this class for TP degree 1.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TpuTwoProjFeedForward.Config):
    pass

  def __init__(self, config: Config):
    # Skip FeedForward.__init__, which builds the stacked ``w13``.
    super(FeedForward, self).__init__()
    stacked = config.w13
    if stacked.num_linears != 2:
      raise ValueError(
          "FeedForward w13 requires num_linears=2 for gate/up projections"
      )
    self.w13 = replace(
        stacked,
        out_features=2 * stacked.out_features,
        num_linears=1,
        param_init=(
            None
            if stacked.param_init is None
            else {
                name: _PackedW13Init(init)
                for name, init in stacked.param_init.items()
            }
        ),
        # The stacked sharding declares a 3-D weight; ``[2F, D]`` is an
        # ordinary column-parallel weight (TP degree 1 only, see above).
        sharding_config=(
            None if stacked.sharding_config is None else colwise_config()
        ),
    ).build()
    self.w2 = config.w2.build()
    self.activation_fn = config.activation_fn.build()
    self.register_state_dict_post_hook(self._stack_w13_on_save)
    self.register_load_state_dict_pre_hook(self._flatten_w13_on_load)

  def parallelize(self, parallel_dims: ParallelDims) -> None:
    if parallel_dims.tp > 1:
      raise ValueError(
          "TpuPackedFeedForward does not support tensor parallelism "
          f"(parallel_dims.tp={parallel_dims.tp})"
      )
    super().parallelize(parallel_dims)

  @staticmethod
  def _stack_w13_on_save(module, state_dict, prefix, _local_metadata) -> None:
    """Expose ``[2F, ...]`` parameters in upstream's stacked ``[2, F, ...]``."""
    for param_name in ("weight", "bias"):
      key = f"{prefix}w13.{param_name}"
      if key in state_dict:
        state_dict[key] = _replicated(state_dict[key]).unflatten(0, (2, -1))

  @staticmethod
  def _flatten_w13_on_load(module, state_dict, prefix, *_args) -> None:
    """Accept upstream's stacked ``[2, F, ...]`` ``w13`` entries."""
    for param_name in ("weight", "bias"):
      key = f"{prefix}w13.{param_name}"
      tensor = state_dict.get(key)
      if tensor is None:
        continue
      # `_parameters`, not attribute access: SimpleFSDP gathers on the latter.
      if tensor.ndim == module.w13._parameters[param_name].ndim + 1:
        state_dict[key] = _replicated(tensor).flatten(0, 1)

  def _w13_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = self.w13.weight
    bias = self.w13.bias
    if isinstance(weight, DTensor):
      # Not on the SPMD-typed TPU paths, whose parameters reach the forward as
      # local tensors: one projection, split along the output features.
      gate_TF, up_TF = torch.chunk(self.w13(x), 2, dim=-1)
      return gate_TF, up_TF
    # Split once along dim 0 so autograd concatenates the two gradients back
    # into the packed parameter for a single FSDP reduce-scatter.
    gate_weight, up_weight = torch.chunk(weight, 2, dim=0)
    if bias is None:
      gate_bias = up_bias = None
    else:
      gate_bias, up_bias = torch.chunk(bias, 2, dim=0)
    return F.linear(x, gate_weight, gate_bias), F.linear(x, up_weight, up_bias)


class TpuTwoProjFeedForwardConverter(ModelConfigConverter):
  """Select two-projection FeedForwards before model construction.

  ``packed`` selects ``TpuPackedFeedForward`` (TP degree 1 only); otherwise the
  stacked parameter is kept (``TpuTwoProjFeedForward``).
  """

  @dataclass(kw_only=True, slots=True)
  class Config(ModelConfigConverter.Config):
    packed: bool = False

  def __init__(self, config: Config):
    self.packed = config.packed

  def convert(self, model_config):
    target_cls = (
        TpuPackedFeedForward.Config
        if self.packed
        else TpuTwoProjFeedForward.Config
    )
    num_converted = 0
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
      replacement = derive(config, target_cls)
      num_converted += 1
      if parent is None:
        model_config = replacement
      elif isinstance(parent, list):
        assert isinstance(field_name, int)
        parent[field_name] = replacement
      else:
        assert isinstance(field_name, str)
        setattr(parent, field_name, replacement)
    # The only startup evidence of which FFN runs: the config dump keeps
    # upstream's ``w13`` block for all three classes.
    logger.info(
        "Using %s for %d of %d FeedForward module(s)",
        target_cls.__qualname__.removesuffix(".Config"),
        num_converted,
        len(matches),
    )
    return model_config


__all__ = [
    "TpuPackedFeedForward",
    "TpuTwoProjFeedForward",
    "TpuTwoProjFeedForwardConverter",
]
