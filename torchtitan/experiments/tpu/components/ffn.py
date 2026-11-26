# Copyright 2026 Google LLC

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.config import derive
from torchtitan.config.transform import ModelConfigConverter
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear


class _PackedW13Init:
  """Un-interleave upstream's fused gate-up initializer into [w1; w3]."""

  def __init__(self, interleaved_init: Callable) -> None:
    self.interleaved_init = interleaved_init

  def __call__(self, parameter: torch.Tensor) -> None:
    interleaved = parameter.new_empty(parameter.shape)
    self.interleaved_init(interleaved)
    if isinstance(interleaved, DTensor):
      interleaved = interleaved.redistribute(
          interleaved.device_mesh,
          [Replicate()] * interleaved.device_mesh.ndim,
      )
    tail = interleaved.shape[1:]
    # Rows (gate_0, up_0, gate_1, up_1, ...) -> (gate_0, ..., up_0, ...).
    packed = interleaved.view(-1, 2, *tail).transpose(0, 1).reshape(-1, *tail)
    with torch.no_grad():
      parameter.copy_(packed)


def _packed_w13_param_init(
    param_init: dict[str, Callable],
) -> dict[str, Callable]:
  return {
      name: _PackedW13Init(initializer)
      for name, initializer in param_init.items()
  }


def _logical_placements(tensor: DTensor) -> list[Shard | Replicate]:
  """Placements for the logical w1/w3 checkpoint entries.

  w1 and w3 are split along dim 0 of the packed parameter, so a matrix stays
  sharded on dim 1 and each device keeps holding 1/N of it: the checkpointer's
  ``ModelWrapper`` caches these entries for the whole run, and replicating
  them would keep the full unsharded FFN on every device. A bias is 1-D and
  small, so it is replicated.
  """
  placement = Shard(1) if tensor.ndim > 1 else Replicate()
  return [placement] * tensor.device_mesh.ndim


class TpuPackedFeedForward(FeedForward):
  """Run contiguous w1/w3 projections from one FSDP parameter on TPU.

  Upstream ``FeedForward`` stores ``w13`` as interleaved ``(gate, up)`` rows
  and splits activations via ``gate_up.unflatten(-1, (-1, 2)).unbind(-1)``.
  On TPU, stride-2 slicing along the minor-most (128-lane) dimension forces
  cross-lane relayouts in forward, remat, and backward.

  ``TpuPackedFeedForward`` keeps a single Linear parameter (one FSDP
  all-gather and one reduce-scatter per layer), stores its rows contiguously
  as ``[all w1 rows; all w3 rows]``, and splits the weight along ``dim=0``
  before running two ``F.linear``s. Checkpoints continue to expose canonical
  ``w1``/``w3`` keys.

  The Linear is registered as ``w13_packed`` rather than ``w13`` (the config
  field is still ``w13``). The row order differs from upstream's ``w13``, and
  per-parameter optimizer state is checkpointed under the physical parameter
  name and is not repacked on load, so the two layouts must not share a name:
  loading one layout's optimizer state into the other then fails on the
  missing key instead of applying moments to the wrong rows.

  TP=1 only. ``TpuPackedFeedForwardConverter`` leaves upstream's
  ``FeedForward`` in place under tensor parallelism.

  TODO(#195): remove TpuPackedFeedForward and its converter calls. Upstream
  pytorch/torchtitan#4808 stores w13 stacked as [2, F, D] with contiguous
  gate/up halves, which lands with PR #195; re-check dense step time there
  before deleting.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(FeedForward.Config):
    pass

  def __init__(self, config: Config):
    super(FeedForward, self).__init__()
    if config.w13.out_features % 2 != 0:
      raise ValueError(
          f"w13.out_features ({config.w13.out_features}) must be divisible by 2"
      )
    w13_cfg = config.w13
    if w13_cfg.param_init is not None:
      w13_cfg = replace(
          w13_cfg,
          param_init=_packed_w13_param_init(w13_cfg.param_init),
      )
    self.w13_packed = w13_cfg.build()
    self.w2 = config.w2.build()
    self.activation_fn = config.activation_fn.build()
    self.register_state_dict_post_hook(self._split_w13_on_save)
    self.register_load_state_dict_pre_hook(self._merge_w13_on_load)

  def parallelize(self, parallel_dims: ParallelDims) -> None:
    if parallel_dims.tp > 1:
      raise ValueError("TpuPackedFeedForward supports TP=1 only")
    super().parallelize(parallel_dims)

  @staticmethod
  def _split_w13_on_save(module, state_dict, prefix, _local_metadata) -> None:
    """Expose the packed [w1; w3] parameter under logical w1/w3 keys."""
    for param_name in ("weight", "bias"):
      fused_key = f"{prefix}w13_packed.{param_name}"
      if fused_key not in state_dict:
        continue
      tensor = state_dict.pop(fused_key)
      if isinstance(tensor, DTensor):
        tensor = tensor.redistribute(
            tensor.device_mesh,
            _logical_placements(tensor),
        )
      unpacked = tensor.view(2, -1, *tensor.shape[1:])
      state_dict[f"{prefix}w1.{param_name}"] = unpacked[0].contiguous()
      state_dict[f"{prefix}w3.{param_name}"] = unpacked[1].contiguous()

  @staticmethod
  def _merge_w13_on_load(module, state_dict, prefix, *_args) -> None:
    """Pack logical w1/w3 checkpoint entries into [w1; w3]."""
    for param_name in ("weight", "bias"):
      gate_key = f"{prefix}w1.{param_name}"
      up_key = f"{prefix}w3.{param_name}"
      if gate_key not in state_dict or up_key not in state_dict:
        continue
      gate = state_dict.pop(gate_key)
      up = state_dict.pop(up_key)
      if isinstance(gate, DTensor):
        placements = _logical_placements(gate)
        gate = gate.redistribute(gate.device_mesh, placements)
        up = up.redistribute(up.device_mesh, placements)
      state_dict[f"{prefix}w13_packed.{param_name}"] = torch.stack(
          (gate, up), dim=0
      ).reshape(-1, *gate.shape[1:])

  def _w13_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = self.w13_packed.weight
    bias = self.w13_packed.bias
    # Split once along dim=0 so autograd concatenates w1/w3 gradients back
    # into the packed parameter for a single FSDP reduce-scatter.
    w1_weight, w3_weight = torch.chunk(weight, 2, dim=0)
    if bias is None:
      w1_bias = w3_bias = None
    else:
      w1_bias, w3_bias = torch.chunk(bias, 2, dim=0)
    return F.linear(x, w1_weight, w1_bias), F.linear(x, w3_weight, w3_bias)

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


class TpuSingleProjPackedFeedForward(TpuPackedFeedForward):
  """Single-projection variant of ``TpuPackedFeedForward`` for scan layers.

  Runs one ``w13_packed`` projection over the contiguous ``[w1; w3]`` rows and
  splits the output contiguously along ``dim=-1`` (no weight split, no two
  ``F.linear``s, no ``cat`` of weight grads).
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TpuPackedFeedForward.Config):
    pass

  def _w13_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.chunk(self.w13_packed(x), 2, dim=-1)


class TpuPackedFeedForwardConverter(ModelConfigConverter):
  """Select packed FeedForward projections before model construction."""

  @dataclass(kw_only=True, slots=True)
  class Config(ModelConfigConverter.Config):
    tp_degree: int = 1
    single_projection: bool = False

  def __init__(self, config: Config):
    self.tp_degree = config.tp_degree
    self.single_projection = config.single_projection

  def convert(self, model_config):
    # The packed row order would depend on the TP degree, and optimizer moments
    # are not repacked on load, so keep upstream's layout under TP.
    if self.tp_degree > 1:
      return model_config
    target_cls = (
        TpuSingleProjPackedFeedForward.Config
        if self.single_projection
        else TpuPackedFeedForward.Config
    )
    matches = list(model_config.traverse(FeedForward.Config))
    for _, config, parent, field_name in matches:
      if isinstance(config, TpuPackedFeedForward.Config):
        continue
      if type(config) is not FeedForward.Config and (
          type(config).__qualname__ != "Gemma4FeedForward.Config"
      ):
        continue
      if type(config.w13) is not Linear.Config:
        continue
      replacement = derive(config, target_cls)
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
    "TpuPackedFeedForward",
    "TpuPackedFeedForwardConverter",
    "TpuSingleProjPackedFeedForward",
]
