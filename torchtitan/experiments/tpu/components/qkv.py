# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, replace

import spmd_types as spmd
import torch
from torch.distributed.tensor import DTensor, Replicate
import torch.nn.functional as F
from torchtitan.config import derive
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.attention import BaseQKVLinear, FusedQKVLinear
from torchtitan.protocols.model import ModelConfigConverter


class _PackedQKVInit:

  def __init__(
      self,
      interleaved_init: Callable,
      *,
      n_heads: int,
      n_kv_heads: int,
      head_dim: int,
  ) -> None:
    self.interleaved_init = interleaved_init
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = head_dim

  def __call__(self, parameter: torch.Tensor) -> None:
    interleaved = parameter.new_empty(parameter.shape)
    self.interleaved_init(interleaved)
    if isinstance(interleaved, DTensor):
      interleaved = interleaved.redistribute(
          interleaved.device_mesh,
          [Replicate()] * interleaved.device_mesh.ndim,
      )
    heads_per_kv = self.n_heads // self.n_kv_heads
    r_dim = heads_per_kv + 2
    tail = interleaved.shape[1:]
    interleaved = interleaved.reshape(
        self.n_kv_heads, r_dim, self.head_dim, *tail
    )
    packed = torch.cat(
        (
            interleaved[:, :heads_per_kv].reshape(-1, *tail),
            interleaved[:, heads_per_kv].reshape(-1, *tail),
            interleaved[:, heads_per_kv + 1].reshape(-1, *tail),
        ),
        dim=0,
    )
    with torch.no_grad():
      parameter.copy_(packed)


def _packed_param_init(config: FusedQKVLinear.Config) -> dict[str, Callable]:
  param_init = config.wqkv.param_init
  if param_init is None:
    raise ValueError("Packed QKV requires the fused QKV parameter initializer")
  return {
      name: _PackedQKVInit(
          initializer,
          n_heads=config.n_heads,
          n_kv_heads=config.n_kv_heads,
          head_dim=config.head_dim,
      )
      for name, initializer in param_init.items()
  }


class TpuPackedQKVLinear(BaseQKVLinear):
  """Run three contiguous Q/K/V projections from one FSDP parameter.

  The physical parameter stores ``[all Q rows; all K rows; all V rows]``.
  This removes the fused projection's interleaved output materialization while
  preserving a single FSDP all-gather and gradient reduce-scatter. Model
  checkpoints continue to expose canonical ``wq``/``wk``/``wv`` keys.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(FusedQKVLinear.Config):
    pass

  def __init__(self, config: Config):
    super().__init__(config)
    if config.n_heads % config.n_kv_heads != 0:
      raise ValueError(
          f"n_heads ({config.n_heads}) must be divisible by "
          f"n_kv_heads ({config.n_kv_heads})"
      )
    self.n_heads = config.n_heads
    self.n_kv_heads = config.n_kv_heads
    self.heads_per_kv = config.n_heads // config.n_kv_heads
    self.r_dim = self.heads_per_kv + 2
    self.q_rows = config.n_heads * config.head_dim
    self.kv_rows = config.n_kv_heads * config.head_dim
    expected_rows = self.q_rows + 2 * self.kv_rows
    if config.wqkv.out_features != expected_rows:
      raise ValueError(
          f"wqkv.out_features ({config.wqkv.out_features}) must equal "
          f"(n_heads + 2 * n_kv_heads) * head_dim ({expected_rows})"
      )
    self.wqkv_packed = replace(
        config.wqkv,
        param_init=_packed_param_init(config),
    ).build()
    self.register_state_dict_post_hook(self._split_qkv_on_save)
    self.register_load_state_dict_pre_hook(self._merge_qkv_on_load)

  def parallelize(self, parallel_dims: ParallelDims) -> None:
    if parallel_dims.tp > 1:
      raise ValueError(
          "TpuPackedQKVLinear supports TP=1 only; a global contiguous "
          "[Q; K; V] layout is incompatible with the existing Shard(0) "
          "tensor-parallel plan"
      )
    super().parallelize(parallel_dims)

  @spmd.local_map(
      out_types=(
          (
              {"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
              spmd.PartitionSpec(("dp", "cp"), "tp", None),
          ),
      )
      * 3
  )
  def forward(  # pyrefly: ignore[bad-override]
      self, x_TD: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = x_TD.shape[0]
    weight = self.wqkv_packed.weight
    bias = self.wqkv_packed.bias
    split_sizes = (self.q_rows, self.kv_rows, self.kv_rows)
    # Split once so autograd concatenates Q/K/V gradients into the packed
    # parameter. Three independent slices create three full-size
    # slice-backward buffers and delay the single FSDP reduce-scatter.
    q_weight, k_weight, v_weight = torch.split(weight, split_sizes, dim=0)
    if bias is None:
      q_bias = k_bias = v_bias = None
    else:
      q_bias, k_bias, v_bias = torch.split(bias, split_sizes, dim=0)
    q_THK = F.linear(
        x_TD,
        q_weight,
        q_bias,
    ).view(num_tokens, -1, self.head_dim)
    k_THK = F.linear(
        x_TD,
        k_weight,
        k_bias,
    ).view(num_tokens, -1, self.head_dim)
    v_THV = F.linear(
        x_TD,
        v_weight,
        v_bias,
    ).view(num_tokens, -1, self.head_dim)
    return q_THK, k_THK, v_THV

  @staticmethod
  def _split_qkv_on_save(module, state_dict, prefix, _local_metadata) -> None:
    q_end = module.q_rows
    k_end = q_end + module.kv_rows
    for param in ("weight", "bias"):
      key = f"{prefix}wqkv_packed.{param}"
      if key not in state_dict:
        continue
      tensor = state_dict.pop(key)
      if isinstance(tensor, DTensor):
        tensor = tensor.redistribute(
            tensor.device_mesh,
            [Replicate()] * tensor.device_mesh.ndim,
        )
      state_dict[f"{prefix}wq.{param}"] = tensor[:q_end].contiguous()
      state_dict[f"{prefix}wk.{param}"] = tensor[q_end:k_end].contiguous()
      state_dict[f"{prefix}wv.{param}"] = tensor[k_end:].contiguous()

  @staticmethod
  def _merge_qkv_on_load(module, state_dict, prefix, *_args) -> None:
    for param in ("weight", "bias"):
      keys = [f"{prefix}{name}.{param}" for name in ("wq", "wk", "wv")]
      if not all(key in state_dict for key in keys):
        continue
      q, k, v = (state_dict.pop(key) for key in keys)
      if isinstance(q, DTensor):
        placements = [Replicate()] * q.device_mesh.ndim
        q = q.redistribute(q.device_mesh, placements)
        k = k.redistribute(k.device_mesh, placements)
        v = v.redistribute(v.device_mesh, placements)
      state_dict[f"{prefix}wqkv_packed.{param}"] = torch.cat((q, k, v), dim=0)


class TpuPackedQKVConverter(ModelConfigConverter):
  """Select packed QKV projections before model construction."""

  @dataclass(kw_only=True, slots=True)
  class Config(ModelConfigConverter.Config):
    pass

  def __init__(self, config: Config):
    pass

  def convert(self, model_config):
    matches = list(model_config.traverse(FusedQKVLinear.Config))
    for _, config, parent, field_name in matches:
      if isinstance(config, TpuPackedQKVLinear.Config):
        continue
      if type(config) is not FusedQKVLinear.Config:
        continue
      if config.wqkv.param_init is None:
        raise ValueError(
            "TpuPackedQKVConverter requires an explicit fused QKV "
            "parameter initializer"
        )
      replacement = derive(config, TpuPackedQKVLinear.Config)
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
    "TpuPackedQKVConverter",
    "TpuPackedQKVLinear",
]
