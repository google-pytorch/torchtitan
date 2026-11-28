# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""No-drop capacity chunking for capacity-bounded Ring-of-Experts EP.

With ``ep_capacity_factor`` the Ring-of-Experts dispatch keeps only the first
``cap`` expert-sorted local routes in its static ``[cap, D]`` expert buffer
and drops the rest. ``ep_overflow_mode="chunked"`` keeps that buffer and its
fast kernels as chunk 0, and adds an overflow tail that runs the routes past
``cap`` in further windows of ``cap`` sorted slots, so no route is dropped
(the JAX recipe's ``moe_chunk_capacity``, ``qwen35_jax/moe_kernels.py``).

The number of overflow windows is data dependent. torch_tpu lowers no
``while_loop``/``cond`` higher-order op (only ``scan``, with a static trip
count), so on TPU the tail is one ``jax_op`` per direction with a
``lax.while_loop`` inside (``kernels/roe_overflow_tc.py``): it runs zero
iterations when nothing overflows. The backward recomputes each overflow
window, so the tail saves no per-window residuals.

Off TPU (CPU tests, reference) the tail runs in PyTorch: eagerly it loops over
exactly the overflow windows (one host sync for the count); traced it loops
over every possible window with masks, which is exact but only meant for
small test shapes.

Layouts: ``w1``/``w3`` are ``[e, D, F]`` and ``w2`` is ``[e, F, D]`` (input dim
before output dim, as the Pallas grouped matmuls take them). The routing
weight multiplies each hidden row before W2 (``scale_before_down_proj``
math). ``partial_G*`` and ``x_all_G*`` may be ``[G, D]`` or ``[G, D / 128,
128]``; only overflow rows are reshaped.
"""

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F

EpOverflowMode = Literal["drop", "chunked"]


@dataclass(frozen=True, kw_only=True)
class EpOverflowRouting:
  """Routing of every local route of one Ring-of-Experts dispatch.

  ``slot_order_M[s]`` is the all-gathered (token * top_k + k) slot of sorted
  slot ``s``; this rank's ``count_total`` local routes come first, in local
  expert order (stable within an expert, as the Pallas routing sorts them).
  ``group_sizes_e`` holds the unclamped per-local-expert route counts. Sorted
  slots ``[0, cap)`` are chunk 0 (the capped expert buffer); the tail runs
  ``[cap, count_total)``.
  """

  x_all_G: torch.Tensor
  scores_flat_M: torch.Tensor
  slot_order_M: torch.Tensor
  group_sizes_e: torch.Tensor
  count_total: torch.Tensor
  cap: int
  top_k: int


def local_routing(
    ids_flat_M: torch.Tensor,
    rank_idx: torch.Tensor,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """``(slot_order_M, group_sizes_e, count_total)`` of this rank's local routes.

  ``ids_flat_M`` holds the global expert id of every all-gathered slot and
  ``rank_idx`` this rank's EP index (shape ``[1]`` or ``[]``).
  """
  with torch.no_grad():
    e = num_local_experts
    rel_M = (
        ids_flat_M.to(torch.int32) - rank_idx.to(torch.int32).reshape(()) * e
    )
    is_local_M = (rel_M >= 0) & (rel_M < e)
    key_M = torch.where(is_local_M, rel_M, torch.full_like(rel_M, e))
    slot_order_M = torch.argsort(key_M, stable=True).to(torch.int32)
    counts_ep1 = torch.zeros(
        (e + 1,), dtype=torch.int32, device=ids_flat_M.device
    ).scatter_add_(0, key_M.long(), torch.ones_like(key_M))
    group_sizes_e = counts_ep1[:e]
    count_total = group_sizes_e.sum(dtype=torch.int32)
  return slot_order_M, group_sizes_e, count_total


def _activation(x: torch.Tensor, activation: str) -> torch.Tensor:
  if activation == "gelu_tanh":
    return F.gelu(x, approximate="tanh")
  if activation != "silu":
    raise ValueError(f"unsupported overflow activation {activation!r}")
  return F.silu(x)


def _window_reference(
    acc_G: torch.Tensor,
    routing: EpOverflowRouting,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    start: int,
    activation: str,
) -> torch.Tensor:
  """Adds sorted slots ``[start, start + cap)`` of the tail to ``acc_G``."""
  w1, w3, w2 = weights
  cap, top_k = routing.cap, routing.top_k
  m = routing.slot_order_M.shape[0]
  e = routing.group_sizes_e.shape[0]
  device = acc_G.device
  idx = start + torch.arange(cap, device=device, dtype=torch.int32)
  valid = idx < routing.count_total
  slots = torch.index_select(
      routing.slot_order_M, 0, torch.clamp(idx, max=max(m - 1, 0)).long()
  )
  tok = torch.where(valid, slots // top_k, torch.zeros_like(slots)).long()
  x_flat = routing.x_all_G.reshape(routing.x_all_G.shape[0], -1)
  rows = torch.index_select(x_flat, 0, tok)
  w_rows = torch.where(
      valid,
      torch.index_select(routing.scores_flat_M, 0, slots.long()).float(),
      0.0,
  )
  # Local expert of each sorted slot.
  ends = routing.group_sizes_e.cumsum(0, dtype=torch.int32)
  eid = torch.clamp(torch.bucketize(idx, ends, right=True), max=max(e - 1, 0))
  y = rows.new_zeros((cap, w2.shape[-1]), dtype=torch.float32)
  rows_f = rows.float()
  for j in range(e):
    sel = (valid & (eid == j)).unsqueeze(-1)
    gate = rows_f @ w1[j].float()
    up = rows_f @ w3[j].float()
    h = (_activation(gate, activation) * up * w_rows.unsqueeze(-1)).to(
        rows.dtype
    )
    y = y + torch.where(sel, h.float() @ w2[j].float(), 0.0)
  y = torch.where(valid.unsqueeze(-1), y, 0.0).to(acc_G.dtype)
  return acc_G.index_add(0, tok, y.reshape((cap,) + tuple(acc_G.shape[1:])))


def _is_tracing(tensor: torch.Tensor) -> bool:
  # pylint: disable=g-import-not-at-top
  from torch.fx.experimental.proxy_tensor import get_proxy_mode

  return (
      torch.compiler.is_compiling()
      or isinstance(tensor, torch._subclasses.FakeTensor)
      or get_proxy_mode() is not None
  )


def overflow_tail_reference(
    partial_G: torch.Tensor,
    routing: EpOverflowRouting,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    activation: str = "silu",
) -> torch.Tensor:
  """PyTorch overflow tail: ``partial_G`` plus every route past ``cap``."""
  cap = routing.cap
  m = routing.slot_order_M.shape[0]
  if cap <= 0 or m <= cap:
    return partial_G
  if _is_tracing(partial_G):
    num_windows = math.ceil((m - cap) / cap)
  else:
    overflow = max(int(routing.count_total.item()) - cap, 0)
    num_windows = math.ceil(overflow / cap)
  out = partial_G
  for k in range(num_windows):
    out = _window_reference(out, routing, weights, cap * (k + 1), activation)
  return out


def overflow_tail(
    partial_G: torch.Tensor,
    routing: EpOverflowRouting,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    activation: str = "silu",
) -> torch.Tensor:
  """Adds the Ring-of-Experts routes past ``cap`` to the partial combine.

  On TPU this is the ``lax.while_loop`` jax_op pair (zero iterations without
  overflow); elsewhere the PyTorch reference.
  """
  if partial_G.device.type == "tpu":
    # pylint: disable=g-import-not-at-top
    from torchtitan.experiments.tpu.kernels.roe_overflow_tc import (
        roe_overflow_tail,
    )

    return roe_overflow_tail(partial_G, routing, weights, activation)
  return overflow_tail_reference(partial_G, routing, weights, activation)


def overflow_tail_weights(
    inner_experts: torch.nn.Module,
    pre_unsharded_weights: tuple[torch.Tensor, ...] | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], str]:
  """``((w1 [e, D, F], w3 [e, D, F], w2 [e, F, D]), activation)`` of experts.

  ``pre_unsharded_weights``, if given, is the experts' ``unshard_weights()``
  result (possibly wrapped per chunk), so the tail's weight gradients reach
  the same tensors as the chunk-0 experts'.
  """
  activation = getattr(inner_experts, "activation", "silu")
  custom = getattr(inner_experts, "overflow_tail_weights", None)
  if callable(custom):
    return custom(pre_unsharded_weights), activation
  if hasattr(inner_experts, "w1_EFD"):
    w1, w2, w3 = (
        pre_unsharded_weights
        if pre_unsharded_weights is not None
        else (inner_experts.w1_EFD, inner_experts.w2_EDF, inner_experts.w3_EFD)
    )
    return (
        w1.transpose(-2, -1),
        w3.transpose(-2, -1),
        w2.transpose(-2, -1),
    ), activation
  raise TypeError(
      "ep_overflow_mode='chunked' needs routed experts with"
      " `overflow_tail_weights()` or canonical w1_EFD/w2_EDF/w3_EFD weights,"
      f" got {type(inner_experts).__name__}"
  )


__all__ = [
    "EpOverflowMode",
    "EpOverflowRouting",
    "local_routing",
    "overflow_tail",
    "overflow_tail_reference",
    "overflow_tail_weights",
]
