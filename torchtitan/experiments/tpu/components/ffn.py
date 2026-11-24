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
  """Un-interleave upstream's fused gate-up initializer into per-TP [w1; w3]."""

  def __init__(self, interleaved_init: Callable, *, tp_degree: int = 1) -> None:
    self.interleaved_init = interleaved_init
    self.tp_degree = tp_degree

  def __call__(self, parameter: torch.Tensor) -> None:
    interleaved = parameter.new_empty(parameter.shape)
    self.interleaved_init(interleaved)
    if isinstance(interleaved, DTensor):
      interleaved = interleaved.redistribute(
          interleaved.device_mesh,
          [Replicate()] * interleaved.device_mesh.ndim,
      )
    tail = interleaved.shape[1:]
    # Group rows into `tp_degree` shards so Shard(0) gives each TP rank
    # `[w1_shard_r; w3_shard_r]` contiguously.
    packed = (
        interleaved.view(self.tp_degree, -1, 2, *tail)
        .transpose(1, 2)
        .reshape(-1, *tail)
    )
    with torch.no_grad():
      parameter.copy_(packed)


def _packed_w13_param_init(
    param_init: dict[str, Callable], *, tp_degree: int = 1
) -> dict[str, Callable]:
  return {
      name: _PackedW13Init(initializer, tp_degree=tp_degree)
      for name, initializer in param_init.items()
  }


def _tp_w13_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
  if isinstance(weight, DTensor):
    mesh = weight.device_mesh
    x_local = x.to_local() if isinstance(x, DTensor) else x
    w_local = weight.to_local()
    b_local = bias.to_local() if isinstance(bias, DTensor) else bias
    w1_weight, w3_weight = torch.chunk(w_local, 2, dim=0)
    if b_local is None:
      w1_bias = w3_bias = None
    else:
      w1_bias, w3_bias = torch.chunk(b_local, 2, dim=0)
    out1_local = F.linear(x_local, w1_weight, w1_bias)
    out3_local = F.linear(x_local, w3_weight, w3_bias)
    out_placements = tuple(
        Shard(out1_local.ndim - 1) if name == "tp" else p
        for name, p in zip(mesh.mesh_dim_names or (), weight.placements)
    )
    return (
        DTensor.from_local(out1_local, mesh, out_placements, run_check=False),
        DTensor.from_local(out3_local, mesh, out_placements, run_check=False),
    )

  w1_weight, w3_weight = torch.chunk(weight, 2, dim=0)
  if bias is None:
    w1_bias = w3_bias = None
  else:
    w1_bias, w3_bias = torch.chunk(bias, 2, dim=0)
  return F.linear(x, w1_weight, w1_bias), F.linear(x, w3_weight, w3_bias)


class TpuPackedFeedForward(FeedForward):
  """Run contiguous w1/w3 projections from one FSDP parameter on TPU.

  Upstream ``FeedForward`` stores ``w13`` as interleaved ``(gate, up)`` rows
  and splits activations via ``gate_up.unflatten(-1, (-1, 2)).unbind(-1)``.
  On TPU, stride-2 slicing along the minor-most (128-lane) dimension forces
  cross-lane relayouts in forward, remat, and backward.

  ``TpuPackedFeedForward`` keeps the single ``self.w13`` Linear parameter (one
  FSDP all-gather and one reduce-scatter per layer, preserving ``w13`` FQN
  patterns), stores rows contiguously per TP shard as ``[w1_shard_r;
  w3_shard_r]`` (which is ``[all w1 rows; all w3 rows]`` when ``tp_degree=1``),
  and splits ``w13.weight`` along ``dim=0`` before running two ``F.linear``s.
  Checkpoints continue to expose canonical ``w1``/``w3`` keys.

  TODO(#195): remove TpuPackedFeedForward and its converter calls. Upstream
  pytorch/torchtitan#4808 stores w13 stacked as [2, F, D] with contiguous
  gate/up halves, which lands with PR #195; re-check dense step time there
  before deleting.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(FeedForward.Config):
    tp_degree: int = 1

  def __init__(self, config: Config):
    super(FeedForward, self).__init__()
    if config.tp_degree < 1:
      raise ValueError(f"tp_degree ({config.tp_degree}) must be >= 1")
    if config.w13.out_features % (2 * config.tp_degree) != 0:
      raise ValueError(
          f"w13.out_features ({config.w13.out_features}) must be divisible by "
          f"2 * tp_degree ({2 * config.tp_degree})"
      )
    self._tp_degree = config.tp_degree
    w13_cfg = config.w13
    if w13_cfg.param_init is not None:
      w13_cfg = replace(
          w13_cfg,
          param_init=_packed_w13_param_init(
              w13_cfg.param_init, tp_degree=config.tp_degree
          ),
      )
    self.w13 = w13_cfg.build()
    self.w2 = config.w2.build()
    self.activation_fn = config.activation_fn.build()
    self.register_state_dict_post_hook(self._split_w13_on_save)
    self.register_load_state_dict_pre_hook(self._merge_w13_on_load)

  def parallelize(self, parallel_dims: ParallelDims) -> None:
    if parallel_dims.tp > 1 and parallel_dims.tp != self._tp_degree:
      raise ValueError(
          f"TpuPackedFeedForward was configured with tp_degree={self._tp_degree}, "
          f"but parallel_dims.tp={parallel_dims.tp}"
      )
    super().parallelize(parallel_dims)

  @staticmethod
  def _split_w13_on_save(module, state_dict, prefix, _local_metadata) -> None:
    """Expose per-TP-shard [w1; w3] parameters under logical w1/w3 keys."""
    tp_degree = module._tp_degree
    for param_name in ("weight", "bias"):
      fused_key = f"{prefix}w13.{param_name}"
      if fused_key not in state_dict:
        continue
      tensor = state_dict.pop(fused_key)
      if isinstance(tensor, DTensor):
        tensor = tensor.redistribute(
            tensor.device_mesh,
            [Replicate()] * tensor.device_mesh.ndim,
        )
      tail = tensor.shape[1:]
      unpacked = tensor.view(tp_degree, 2, -1, *tail)
      state_dict[f"{prefix}w1.{param_name}"] = (
          unpacked[:, 0].reshape(-1, *tail).contiguous()
      )
      state_dict[f"{prefix}w3.{param_name}"] = (
          unpacked[:, 1].reshape(-1, *tail).contiguous()
      )

  @staticmethod
  def _merge_w13_on_load(module, state_dict, prefix, *_args) -> None:
    """Pack logical w1/w3 checkpoint entries into per-TP-shard [w1; w3]."""
    tp_degree = module._tp_degree
    for param_name in ("weight", "bias"):
      gate_key = f"{prefix}w1.{param_name}"
      up_key = f"{prefix}w3.{param_name}"
      if gate_key not in state_dict or up_key not in state_dict:
        continue
      gate = state_dict.pop(gate_key)
      up = state_dict.pop(up_key)
      if isinstance(gate, DTensor):
        placements = [Replicate()] * gate.device_mesh.ndim
        gate = gate.redistribute(gate.device_mesh, placements)
        up = up.redistribute(up.device_mesh, placements)
      tail = gate.shape[1:]
      state_dict[f"{prefix}w13.{param_name}"] = torch.stack(
          (
              gate.view(tp_degree, -1, *tail),
              up.view(tp_degree, -1, *tail),
          ),
          dim=1,
      ).reshape(-1, *tail)

  def _w13_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = self.w13.weight
    bias = self.w13.bias
    if self._tp_degree > 1:
      return _tp_w13_linear(x, weight, bias)
    # Split once along dim=0 so autograd concatenates w1/w3 gradients back
    # into the packed w13 parameter for a single FSDP reduce-scatter.
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


def _tp_w13_single_linear(
    x: torch.Tensor,
    weight: DTensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
  mesh = weight.device_mesh
  x_local = x.to_local() if isinstance(x, DTensor) else x
  w_local = weight.to_local()
  b_local = bias.to_local() if isinstance(bias, DTensor) else bias
  gate_up_local = F.linear(x_local, w_local, b_local)
  out1_local, out3_local = torch.chunk(gate_up_local, 2, dim=-1)
  out_placements = tuple(
      Shard(out1_local.ndim - 1) if name == "tp" else p
      for name, p in zip(mesh.mesh_dim_names or (), weight.placements)
  )
  return (
      DTensor.from_local(out1_local, mesh, out_placements, run_check=False),
      DTensor.from_local(out3_local, mesh, out_placements, run_check=False),
  )


class TpuSingleProjPackedFeedForward(TpuPackedFeedForward):
  """Single-projection variant of ``TpuPackedFeedForward`` for scan layers.

  Runs one ``w13`` projection over the contiguous ``[w1; w3]`` rows and splits
  the output contiguously along ``dim=-1`` (no weight split, no two
  ``F.linear``s, no ``cat`` of weight grads).
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TpuPackedFeedForward.Config):
    pass

  def _w13_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if self._tp_degree > 1 and isinstance(self.w13.weight, DTensor):
      return _tp_w13_single_linear(x, self.w13.weight, self.w13.bias)
    return torch.chunk(self.w13(x), 2, dim=-1)


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
      replacement = derive(config, target_cls, tp_degree=self.tp_degree)
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
