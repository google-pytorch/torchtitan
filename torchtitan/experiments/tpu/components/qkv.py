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
from torchtitan.models.common.attention import QKVLinear
from torchtitan.config.transform import ModelConfigConverter
from torchtitan.protocols.module import Module


class _PackedQKVInit:

  def __init__(
      self,
      interleaved_init: Callable,
      *,
      n_heads: int,
      n_kv_heads: int,
      head_dim: int,
      k_eq_v: bool = False,
  ) -> None:
    self.interleaved_init = interleaved_init
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = head_dim
    self.k_eq_v = k_eq_v

  def __call__(self, parameter: torch.Tensor) -> None:
    interleaved = parameter.new_empty(parameter.shape)
    self.interleaved_init(interleaved)
    if isinstance(interleaved, DTensor):
      interleaved = interleaved.redistribute(
          interleaved.device_mesh,
          [Replicate()] * interleaved.device_mesh.ndim,
      )
    heads_per_kv = self.n_heads // self.n_kv_heads
    r_dim = heads_per_kv + (1 if self.k_eq_v else 2)
    tail = interleaved.shape[1:]
    interleaved = interleaved.reshape(
        self.n_kv_heads, r_dim, self.head_dim, *tail
    )
    slices = [
        interleaved[:, :heads_per_kv].reshape(-1, *tail),
        interleaved[:, heads_per_kv].reshape(-1, *tail),
    ]
    if not self.k_eq_v:
      slices.append(interleaved[:, heads_per_kv + 1].reshape(-1, *tail))
    packed = torch.cat(tuple(slices), dim=0)
    with torch.no_grad():
      parameter.copy_(packed)


def _packed_param_init(
    config: QKVLinear.Config, *, k_eq_v: bool = False
) -> dict[str, Callable]:
  param_init = config.wqkv.param_init
  if param_init is None:
    raise ValueError("Packed QKV requires the fused QKV parameter initializer")
  return {
      name: _PackedQKVInit(
          initializer,
          n_heads=config.n_heads,
          n_kv_heads=config.n_kv_heads,
          head_dim=config.head_dim,
          k_eq_v=k_eq_v,
      )
      for name, initializer in param_init.items()
  }


class TpuPackedQKVLinear(Module):
  """Run contiguous Q/K/V (or Q/K for K=V) projections from one FSDP parameter.

  The physical parameter stores ``[all Q rows; all K rows; all V rows]`` (or
  ``[all Q rows; all K rows]`` when ``k_eq_v=True``).
  This removes the fused projection's interleaved output materialization while
  preserving a single FSDP all-gather and gradient reduce-scatter. Model
  checkpoints expose ``QKVLinear``'s native interleaved ``wqkv`` key (per KV
  group ``[q rows; k row; v row]``, or ``[q rows; k row]`` when ``k_eq_v``).
  """

  @dataclass(kw_only=True, slots=True)
  class Config(QKVLinear.Config):
    k_eq_v: bool = False

  def __init__(self, config: Config):
    super().__init__()
    self.head_dim = config.head_dim
    if config.n_heads % config.n_kv_heads != 0:
      raise ValueError(
          f"n_heads ({config.n_heads}) must be divisible by "
          f"n_kv_heads ({config.n_kv_heads})"
      )
    self.n_heads = config.n_heads
    self.n_kv_heads = config.n_kv_heads
    self.heads_per_kv = config.n_heads // config.n_kv_heads
    self.k_eq_v = config.k_eq_v
    self.r_dim = self.heads_per_kv + (1 if self.k_eq_v else 2)
    self.q_rows = config.n_heads * config.head_dim
    self.kv_rows = config.n_kv_heads * config.head_dim
    expected_rows = self.q_rows + (1 if self.k_eq_v else 2) * self.kv_rows
    if config.wqkv.out_features != expected_rows:
      expected_expr = (
          "(n_heads + n_kv_heads) * head_dim"
          if self.k_eq_v
          else "(n_heads + 2 * n_kv_heads) * head_dim"
      )
      raise ValueError(
          f"wqkv.out_features ({config.wqkv.out_features}) must equal "
          f"{expected_expr} ({expected_rows})"
      )
    self.wqkv_packed = replace(
        config.wqkv,
        param_init=_packed_param_init(config, k_eq_v=self.k_eq_v),
    ).build()
    self.register_state_dict_post_hook(self._split_qkv_on_save)
    self.register_load_state_dict_pre_hook(self._merge_qkv_on_load)

  def _parallelize(self, parallel_dims: ParallelDims) -> None:
    if parallel_dims.tp > 1:
      raise ValueError(
          "TpuPackedQKVLinear supports TP=1 only; a global contiguous "
          "[Q; K; V] layout is incompatible with the existing Shard(0) "
          "tensor-parallel plan"
      )
    super()._parallelize(parallel_dims)

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
    if self.k_eq_v:
      split_sizes = (self.q_rows, self.kv_rows)
      q_weight, k_weight = torch.split(weight, split_sizes, dim=0)
      if bias is None:
        q_bias = k_bias = None
      else:
        q_bias, k_bias = torch.split(bias, split_sizes, dim=0)
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
      return q_THK, k_THK, k_THK

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
    """Expose the packed parameter in ``QKVLinear``'s native ``wqkv`` layout.

    Upstream keeps the physical, per-KV-group interleaved ``wqkv`` in native
    state dicts and converts to logical Q/K/V only in the HF adapters
    (1e4b1f68). Re-interleaving here keeps packed and fused checkpoints
    interchangeable and lets the adapters' ``wqkv`` split apply unchanged.
    """
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
      tail = tensor.shape[1:]
      parts = [
          tensor[:q_end].reshape(
              module.n_kv_heads, module.heads_per_kv, module.head_dim, *tail
          ),
          tensor[q_end:k_end].reshape(
              module.n_kv_heads, 1, module.head_dim, *tail
          ),
      ]
      if not module.k_eq_v:
        parts.append(
            tensor[k_end:].reshape(module.n_kv_heads, 1, module.head_dim, *tail)
        )
      state_dict[f"{prefix}wqkv.{param}"] = (
          torch.cat(parts, dim=1).reshape(-1, *tail).contiguous()
      )

  @staticmethod
  def _merge_qkv_on_load(module, state_dict, prefix, *_args) -> None:
    """Accept ``wqkv`` (interleaved) or logical ``wq``/``wk``(/``wv``) keys."""
    names = ("wq", "wk") if module.k_eq_v else ("wq", "wk", "wv")
    for param in ("weight", "bias"):
      fused_key = f"{prefix}wqkv.{param}"
      logical_keys = [f"{prefix}{name}.{param}" for name in names]
      if fused_key in state_dict:
        tensor = state_dict.pop(fused_key)
        if isinstance(tensor, DTensor):
          tensor = tensor.redistribute(
              tensor.device_mesh, [Replicate()] * tensor.device_mesh.ndim
          )
        tail = tensor.shape[1:]
        grouped = tensor.reshape(
            module.n_kv_heads, module.r_dim, module.head_dim, *tail
        )
        tensors = [
            grouped[:, : module.heads_per_kv].reshape(-1, *tail),
            grouped[:, module.heads_per_kv].reshape(-1, *tail),
        ]
        if not module.k_eq_v:
          tensors.append(grouped[:, module.heads_per_kv + 1].reshape(-1, *tail))
      elif all(key in state_dict for key in logical_keys):
        tensors = [state_dict.pop(key) for key in logical_keys]
        if isinstance(tensors[0], DTensor):
          placements = [Replicate()] * tensors[0].device_mesh.ndim
          tensors = [
              t.redistribute(t.device_mesh, placements) for t in tensors
          ]
      else:
        continue
      state_dict[f"{prefix}wqkv_packed.{param}"] = torch.cat(
          tuple(tensors), dim=0
      )


class TpuPackedQKVConverter(ModelConfigConverter):
  """Select packed QKV projections before model construction."""

  @dataclass(kw_only=True, slots=True)
  class Config(ModelConfigConverter.Config):
    pass

  def __init__(self, config: Config):
    pass

  def convert(self, model_config):
    matches = list(model_config.traverse(QKVLinear.Config))
    for _, config, parent, field_name in matches:
      if isinstance(config, TpuPackedQKVLinear.Config):
        continue
      k_eq_v = bool(getattr(config, "k_eq_v", False))
      if type(config) is not QKVLinear.Config and not k_eq_v:
        continue
      if config.wqkv.param_init is None:
        raise ValueError(
            "TpuPackedQKVConverter requires an explicit fused QKV "
            "parameter initializer"
        )
      replacement = derive(config, TpuPackedQKVLinear.Config, k_eq_v=k_eq_v)
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
