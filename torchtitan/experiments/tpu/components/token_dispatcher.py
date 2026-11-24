# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU VJPs for the standard MoE dispatcher's local permutation work.

The route reorder precedes expert transport, so its custom VJP applies to both
EP=1 and EP>1. The fixed-K combine is deliberately scoped to EP=1 and the
ragged EP transport. Standard
dropless EP also returns exactly ``top_k`` rows per source token, so the same
reduction should be applicable after the return all-to-all. Supporting it also
requires preserving the EP unpermute and collective, then placing an SP-local
reduction into global sequence coordinates. That broader path was not needed
for the FSDP/EP1 target and has not been benchmarked here.

This scope does not make the reduction depend on bounded ragged all-to-all.
Ragged all-to-all is a separate transport choice that can call the same local
reduction once its capacity and overflow semantics have been validated. The
earlier bounded-ragged prototype did exactly that around ``combine_route``.
``ep_transport="ragged_all_to_all"`` defaults to the dropless form of that
transport (receive buffer sized for the worst case), and ``ep_capacity_factor``
optionally bounds each rank's expert buffer on both ``ragged_all_to_all`` and
``ring_of_experts`` (see ``TpuAllToAllTokenDispatcher.Config`` for what it
bounds on each).
"""

from dataclasses import dataclass, replace
import math
from typing import Literal, get_args
import weakref

import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F
from torch.distributed._functional_collectives import (
    AsyncCollectiveTensor,
    all_gather_single,
    reduce_scatter_single,
)
from torch.distributed._composable.fsdp import FSDPModule
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.traceback import annotate, annotate_fn

from torchtitan.config import derive
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.experiments.tpu.kernels.route_score import route_score_grad
from torchtitan.experiments.tpu.components.remat_regions import (
    annotate_region,
    MOE_EP_DISPATCH,
    MOE_ROUTING,
)
from torchtitan.models.common.moe import _tokens_per_expert, RoutedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllDispatchMetadata,
    AllToAllTokenDispatcher,
    LocalDispatchMetadata,
)
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.tools.logging import logger

EpTransport = Literal[
    "all_to_all", "ragged_all_to_all", "ring_of_experts", "auto"
]


def _wait(tensor: torch.Tensor) -> torch.Tensor:
  """Waits on a functional collective's result."""
  if isinstance(tensor, AsyncCollectiveTensor):
    return tensor.wait()
  return tensor


def _ep_all_gather_impl(
    operand: torch.Tensor,
    ep_mesh,
    pg,
) -> torch.Tensor:
  """All-gathers ``operand`` along dim 0 over ``ep_mesh``."""
  src = operand.detach().contiguous()
  if ep_mesh is None:
    return src
  ep_size = ep_mesh.size()
  if ep_size <= 1:
    return src
  if src.device.type == "tpu":
    if pg is None or not hasattr(pg, "group_name"):
      raise RuntimeError(
          "ring_of_experts on TPU needs an EP process group with a"
          f" group_name; got pg={pg!r}"
      )
    return torch.tpu.ops.all_gather_dim(src, 0, group=pg)
  if (
      pg is not None
      and get_proxy_mode() is None
      and not isinstance(src, torch._subclasses.FakeTensor)
  ):
    return _wait(all_gather_single(src, gather_dim=0, group=ep_mesh))
  raise NotImplementedError(
      "Ring-of-Experts EP all_gather needs a TPU process group with"
      " `group_name` when traced; off TPU it only runs eagerly with an EP mesh"
      f" (device={src.device.type}, pg={pg!r}, ep_mesh={ep_mesh!r})."
  )


def _ep_reduce_scatter_sum_impl(
    operand: torch.Tensor,
    ep_mesh,
    pg,
) -> torch.Tensor:
  """Reduce-scatters ``operand`` with ``sum`` along dim 0 over ``ep_mesh``."""
  src = operand.detach().contiguous()
  if ep_mesh is None:
    return src
  ep_size = ep_mesh.size()
  if ep_size <= 1:
    return src
  if src.device.type == "tpu":
    if pg is None or not hasattr(pg, "group_name"):
      raise RuntimeError(
          "ring_of_experts on TPU needs an EP process group with a"
          f" group_name; got pg={pg!r}"
      )
    return torch.tpu.ops.reduce_scatter_dim(src, 0, group=pg, reduce_op="sum")
  if (
      pg is not None
      and get_proxy_mode() is None
      and not isinstance(src, torch._subclasses.FakeTensor)
  ):
    return _wait(
        reduce_scatter_single(src, reduceOp="sum", scatter_dim=0, group=ep_mesh)
    )
  raise NotImplementedError(
      "Ring-of-Experts EP reduce_scatter needs a TPU process group with"
      " `group_name` when traced; off TPU it only runs eagerly with an EP mesh"
      f" (device={src.device.type}, pg={pg!r}, ep_mesh={ep_mesh!r})."
  )


class _EpAllGather(torch.autograd.Function):
  """Differentiable EP ``all_gather(dim=0)`` with ``reduce_scatter(sum, dim=0)`` VJP."""

  @staticmethod
  def forward(ctx, operand: torch.Tensor, ep_mesh, pg) -> torch.Tensor:
    ctx.set_materialize_grads(False)
    ctx.ep_mesh = ep_mesh
    ctx.pg = pg
    return _ep_all_gather_impl(operand, ep_mesh, pg)

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_out: torch.Tensor | None):
    if grad_out is None or not ctx.needs_input_grad[0]:
      return None, None, None
    grad_in = _ep_reduce_scatter_sum_impl(grad_out, ctx.ep_mesh, ctx.pg)
    return grad_in, None, None


class _EpReduceScatter(torch.autograd.Function):
  """Differentiable EP ``reduce_scatter(sum, dim=0)`` with ``all_gather(dim=0)`` VJP."""

  @staticmethod
  def forward(ctx, operand: torch.Tensor, ep_mesh, pg) -> torch.Tensor:
    ctx.set_materialize_grads(False)
    ctx.ep_mesh = ep_mesh
    ctx.pg = pg
    return _ep_reduce_scatter_sum_impl(operand, ep_mesh, pg)

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_out: torch.Tensor | None):
    if grad_out is None or not ctx.needs_input_grad[0]:
      return None, None, None
    grad_in = _ep_all_gather_impl(grad_out, ctx.ep_mesh, ctx.pg)
    return grad_in, None, None


def _ragged_all_to_all_impl(
    operand: torch.Tensor,
    output_rows: int,
    input_offsets: torch.Tensor,
    send_sizes: torch.Tensor,
    output_offsets: torch.Tensor,
    recv_sizes: torch.Tensor,
    ep_mesh,
    pg,
) -> torch.Tensor:
  """Executes a functional ragged all-to-all without built-in autograd syncs.

  ``torch.tpu.ops.ragged_all_to_all``'s C++ autograd kernel checks
  ``operand.requires_grad()`` and issues eager ``pg->alltoall_base`` calls on
  the offset vectors, which cannot be traced by ``TpuGraphTrainer``'s
  ``make_fx`` + ``FakeTensorMode`` path. Passing ``operand.detach()`` inside our
  custom ``_RaggedAllToAll`` ``autograd.Function`` bypasses that branch while
  preserving exact precomputed backward offsets.
  """
  src = operand.detach().contiguous()
  output = src.new_zeros((output_rows, *src.shape[1:]))
  if src.shape[0] == 0 or output_rows == 0:
    return output

  if src.device.type == "tpu":
    if pg is None or not hasattr(pg, "group_name"):
      raise RuntimeError(
          "ragged_all_to_all on TPU needs an EP process group with a"
          f" group_name; got pg={pg!r}"
      )
    return torch.tpu.ops.ragged_all_to_all(
        src,
        output,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        pg.group_name,
    )

  # Off TPU: a slow eager reference built from all-gathers, for CPU/gloo tests.
  if (
      ep_mesh is not None
      and get_proxy_mode() is None
      and not isinstance(src, torch._subclasses.FakeTensor)
  ):
    ep_size = ep_mesh.size()
    if ep_size <= 1:
      num_updates = input_offsets.numel()
      for k in range(num_updates):
        sz = int(send_sizes[k].item())
        if sz > 0:
          in_off = int(input_offsets[k].item())
          out_off = int(output_offsets[k].item())
          output[out_off : out_off + sz] = src[in_off : in_off + sz]
      return output
    if pg is not None:
      in_rows_t = torch.tensor(
          [src.shape[0]], dtype=torch.int64, device=src.device
      )
      all_in_rows = _wait(
          all_gather_single(in_rows_t, gather_dim=0, group=ep_mesh)
      )
      max_in_rows = int(all_in_rows.max().item())
      if src.shape[0] < max_in_rows:
        pad_src = src.new_zeros((max_in_rows, *src.shape[1:]))
        pad_src[: src.shape[0]] = src
      else:
        pad_src = src
      gathered_src = _wait(
          all_gather_single(pad_src, gather_dim=0, group=ep_mesh)
      ).view(ep_size, max_in_rows, *src.shape[1:])

      meta_local = torch.stack(
          [
              input_offsets.to(torch.int64),
              send_sizes.to(torch.int64),
              output_offsets.to(torch.int64),
          ],
          dim=0,
      ).contiguous()
      gathered_meta = _wait(
          all_gather_single(meta_local, gather_dim=0, group=ep_mesh)
      ).view(ep_size, 3, -1)

      num_blocks = input_offsets.numel()
      updates_per_peer = num_blocks // ep_size
      local_rank = ep_mesh.get_local_rank()
      for src_rank in range(ep_size):
        for k in range(updates_per_peer):
          b = local_rank * updates_per_peer + k
          sz = int(gathered_meta[src_rank, 1, b].item())
          if sz > 0:
            in_off = int(gathered_meta[src_rank, 0, b].item())
            out_off = int(gathered_meta[src_rank, 2, b].item())
            output[out_off : out_off + sz] = gathered_src[
                src_rank, in_off : in_off + sz
            ]
      return output

  # Tracing (make_fx / Dynamo) off TPU, or no process group: there is no
  # traceable exchange here, and returning the zero-initialized buffer would
  # silently zero the MoE output and gradients.
  raise NotImplementedError(
      "Dropless EP ragged_all_to_all needs a TPU process group with"
      " `group_name` when traced; off TPU it only runs eagerly with an EP mesh"
      f" (device={src.device.type}, pg={pg!r}, ep_mesh={ep_mesh!r})."
  )


class _RaggedAllToAll(torch.autograd.Function):
  """Differentiable ragged all-to-all with precomputed adjoint offsets."""

  @staticmethod
  def forward(
      ctx,
      operand: torch.Tensor,
      output_rows: int,
      fwd_input_offsets: torch.Tensor,
      fwd_send_sizes: torch.Tensor,
      fwd_output_offsets: torch.Tensor,
      fwd_recv_sizes: torch.Tensor,
      bwd_input_offsets: torch.Tensor,
      bwd_send_sizes: torch.Tensor,
      bwd_output_offsets: torch.Tensor,
      bwd_recv_sizes: torch.Tensor,
      ep_mesh,
      pg,
  ) -> torch.Tensor:
    ctx.set_materialize_grads(False)
    ctx.operand_rows = operand.shape[0]
    ctx.ep_mesh = ep_mesh
    ctx.pg = pg
    ctx.save_for_backward(
        bwd_input_offsets,
        bwd_send_sizes,
        bwd_output_offsets,
        bwd_recv_sizes,
    )
    return _ragged_all_to_all_impl(
        operand,
        output_rows,
        fwd_input_offsets,
        fwd_send_sizes,
        fwd_output_offsets,
        fwd_recv_sizes,
        ep_mesh,
        pg,
    )

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_out: torch.Tensor | None):
    if grad_out is None or not ctx.needs_input_grad[0]:
      return (None,) * 12
    (
        bwd_input_offsets,
        bwd_send_sizes,
        bwd_output_offsets,
        bwd_recv_sizes,
    ) = ctx.saved_tensors
    grad_operand = _ragged_all_to_all_impl(
        grad_out,
        ctx.operand_rows,
        bwd_input_offsets,
        bwd_send_sizes,
        bwd_output_offsets,
        bwd_recv_sizes,
        ctx.ep_mesh,
        ctx.pg,
    )
    return (grad_operand,) + (None,) * 11


@dataclass(frozen=True, kw_only=True)
class TpuEpDispatchMetadata:
  """Dispatch metadata shared by the static-shape TPU EP transports.

  ``valid_count`` is the number of routed rows this rank kept in its expert
  buffer and ``dropped_count`` the number ``ep_capacity_factor`` dropped (zero
  when dropless). ``row_scale_R`` carries the routing weight of each buffer
  row for ``scale_before_down_proj``; ``row_scale_applied`` tells combine that
  the experts already applied it.
  """

  token_indices_experts_sorted_N: torch.Tensor
  valid_count: torch.Tensor
  dropped_count: torch.Tensor
  row_scale_R: torch.Tensor | None = None
  row_scale_applied: bool = False


@dataclass(frozen=True, kw_only=True)
class TpuRaggedDispatchMetadata(TpuEpDispatchMetadata):
  """Static-shape EP dispatch metadata for ``ragged_all_to_all``."""

  topk_scores_experts_sorted_N: torch.Tensor
  routed_rows: int
  disp_input_offsets: torch.Tensor
  disp_send_sizes: torch.Tensor
  disp_output_offsets: torch.Tensor
  disp_recv_sizes: torch.Tensor
  comb_input_offsets: torch.Tensor
  comb_send_sizes: torch.Tensor
  comb_output_offsets: torch.Tensor
  comb_recv_sizes: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class TpuRoEDispatchMetadata(TpuEpDispatchMetadata):
  """Ring-of-Experts metadata for the Pallas TensorCore gather/reduce kernels.

  ``token_indices_experts_sorted_N`` holds the global token of each local
  route in expert order, ``scores_flat_M`` the all-gathered routing scores,
  and the remaining tensors the routing tables from ``roe_compute_routing``.
  """

  num_global_tokens: int
  scores_flat_M: torch.Tensor
  window_slot_order_cap: torch.Tensor
  start_count_2: torch.Tensor
  src_rows: torch.Tensor
  tokens: torch.Tensor
  blocks: torch.Tensor
  gather_block_size: int
  reduce_block_size: int
  zero_pad_tail: bool


def _ring_of_experts_unavailable_message(reason: str) -> str:
  return (
      f"ep_transport='ring_of_experts' cannot run: {reason}. Ring-of-Experts"
      " has only the Pallas TensorCore implementation; use"
      " ep_transport='ragged_all_to_all', or 'auto', which picks ragged"
      " all-to-all when Ring-of-Experts cannot run."
  )


class _ScaleRouteWeights(torch.autograd.Function):
  """Scale expert outputs and fuse the FP32 route-score gradient reduction."""

  @staticmethod
  def forward(ctx, routed_output_ND, scores_N):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(routed_output_ND, scores_N)
    return (
        routed_output_ND.float() * scores_N.reshape(-1, 1)
    ).to(routed_output_ND.dtype)

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_output_ND):
    if grad_output_ND is None:
      return None, None

    routed_output_ND, scores_N = ctx.saved_tensors
    grad_routed_output_ND = None
    if ctx.needs_input_grad[0]:
      grad_routed_output_ND = (
          grad_output_ND.float() * scores_N.reshape(-1, 1)
      ).to(routed_output_ND.dtype)

    grad_scores_N = None
    if ctx.needs_input_grad[1]:
      grad_scores_N = route_score_grad(grad_output_ND, routed_output_ND)
    return grad_routed_output_ND, grad_scores_N


def scale_route_weights(
    routed_output_ND: torch.Tensor,
    scores_N: torch.Tensor,
) -> torch.Tensor:
  """Weight expert outputs while optimizing the supported TPU backward."""
  if routed_output_ND.ndim != 2 or scores_N.ndim != 1:
    raise ValueError("route weighting expects routed[N, D] and scores[N]")
  if routed_output_ND.shape[0] != scores_N.shape[0]:
    raise ValueError(
        "route weighting row count must match score count: "
        f"{routed_output_ND.shape[0]} != {scores_N.shape[0]}"
    )

  # For Qwen's BF16 expert outputs and FP32 normalized scores, save the expert
  # output in BF16 and fuse its conversion, multiplication, and hidden-dimension
  # reduction in dScore. This avoids native autograd's saved FP32 [N, D] tensor
  # and matches the gradient implied by MaxText's FP32 route-weighted combine:
  # https://github.com/AI-Hypercomputer/maxtext/blob/263b8c18e32ed1da4dbb4eb1476c5e3d87b4d9eb/src/maxtext/layers/moe.py#L1044-L1063
  # XLA may reassociate the FP32 reduction, so numerical equivalence is expected,
  # but bitwise identity is not guaranteed. Other dtypes retain native autograd.
  if (
      routed_output_ND.dtype != torch.bfloat16
      or scores_N.dtype != torch.float32
      or routed_output_ND.device.type not in ("cpu", "tpu")
  ):
    return (
        routed_output_ND.float() * scores_N.reshape(-1, 1)
    ).to(routed_output_ND.dtype)
  return _ScaleRouteWeights.apply(routed_output_ND, scores_N)


class _StableGroupedReduce(torch.autograd.Function):
  """Combine a dropless EP=1 route set without collision scatter autograd.

  Local dispatch produces exactly ``top_k`` rows per input token, but those
  rows are in expert-major order. A stable sort by the original token row makes
  them contiguous, after which the forward is a reshape to ``(T, K, D)`` and
  an FP32 sum over ``K``. Stability keeps each token's routes in their original
  expert-major order before the reduction, avoiding another gratuitous change
  in deterministic accumulation order. The derivative is simply
  ``grad_output[index_N]``, so backward saves the one-dimensional row index.

  The upstream deterministic scatter is more general, but its generated VJP
  expanded the row index across the hidden dimension and materialized a
  768-MiB index per layer at the production shape. The native forward had
  already learned a row-scatter lowering; that did not fix this backward
  gather. Saving only ``index_N`` makes the VJP a row gather with no collision.

  We also tried reusing dispatch's inverse permutation, fusing score scaling
  into this function, replacing the gather/reduce with an opaque Pallas op,
  and replacing stable sorts with exact composite keys. Some won isolated
  microbenchmarks, but none produced a material whole-model improvement after
  accounting for metadata, scheduling, or changed accumulation order. This is
  therefore the best measured whole-graph implementation, not a proof that no
  faster formulation exists.
  """

  @staticmethod
  def forward(ctx, src_ND, index_N, top_k):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(index_N)
    permutation_N = torch.argsort(index_N, stable=True)
    grouped_TKD = torch.index_select(src_ND, 0, permutation_N).reshape(
        -1, top_k, src_ND.shape[1]
    )
    return grouped_TKD.float().sum(dim=1).to(src_ND.dtype)

  @staticmethod
  def backward(ctx, grad_output_TD):
    (index_N,) = ctx.saved_tensors
    grad_src_ND = None
    if grad_output_TD is not None:
      grad_src_ND = torch.index_select(grad_output_TD, 0, index_N)
    return grad_src_ND, None, None


def stable_grouped_reduce(
    src_ND: torch.Tensor,
    index_N: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
  """Reduce fixed-K routes by token with FP32 accumulation."""
  return _StableGroupedReduce.apply(src_ND, index_N, top_k)


class _LocalReorder(torch.autograd.Function):
  """Invert the expert permutation instead of scattering repeated rows.

  Forward sorts the flattened ``T * K`` routes by expert and gathers tokens
  and scores into expert-major order. Native autograd sees repeated token rows
  and emits a collision scatter for the token gradient. At the route level,
  however, the expert sort is a permutation: invert it once, gather gradients
  back to token-major order, reshape to ``(T, K, D)``, and sum the fixed ``K``
  copies in FP32. This restores original top-k-slot order rather than the
  expert-major collision order used by the generated backward. Both are
  deterministic, but their different FP32 association explains the observed
  short-prefix gradient drift. Score gradients use the same inverse gather
  without a sum.

  The generated backward and native ``index_add`` both express a collision
  reduction over repeated token rows. We tested those paths in BF16 and FP32,
  inverse permutation by generic scatter, inverse argsort plus row gathers,
  unstable expert sorting, and exact composite-key stable sorting. FP32
  ``index_add`` removed the visible row gather but regressed two full-model
  brackets; generic inverse scatter was substantially worse on TPU. Composite
  keys improved isolated sorts but not the full step. Inverse argsort plus row
  gathers is therefore retained. The forward sort remains stable because
  changing equal-expert route order changes BF16 expert dWeight accumulation
  and produced an unacceptable training-trajectory shift.
  """

  @staticmethod
  def forward(ctx, x_TD, scores_TK, permutation_N, top_k):
    ctx.set_materialize_grads(False)
    inverse_N = torch.argsort(permutation_N)
    token_indices_N = permutation_N // top_k
    ctx.save_for_backward(inverse_N)
    ctx.input_shape = x_TD.shape
    ctx.score_shape = scores_TK.shape
    ctx.top_k = top_k
    ctx.mark_non_differentiable(token_indices_N)
    return (
        torch.index_select(x_TD, 0, token_indices_N),
        token_indices_N,
        torch.index_select(scores_TK.reshape(-1), 0, permutation_N),
    )

  @staticmethod
  def backward(ctx, grad_routed_ND, _grad_token_indices_N, grad_scores_N):
    (inverse_N,) = ctx.saved_tensors
    grad_x_TD = None
    if grad_routed_ND is not None:
      grad_x_TD = torch.index_select(grad_routed_ND, 0, inverse_N)
      grad_x_TD = (
          grad_x_TD.float()
          .reshape(ctx.input_shape[0], ctx.top_k, ctx.input_shape[1])
          .sum(dim=1)
          .to(grad_routed_ND.dtype)
      )
    grad_scores_TK = None
    if grad_scores_N is not None:
      grad_scores_TK = torch.index_select(
          grad_scores_N, 0, inverse_N
      ).reshape(ctx.score_shape)
    return grad_x_TD, grad_scores_TK, None, None


def local_reorder(
    x_TD: torch.Tensor,
    scores_TK: torch.Tensor,
    expert_ids_TK: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Sort routes by expert while keeping all gathers row-indexed."""
  # Marked so that a selective memory policy can save the sort order.
  with annotate_region(MOE_ROUTING):
    permutation_N = torch.argsort(expert_ids_TK.reshape(-1), stable=True)
  return _LocalReorder.apply(x_TD, scores_TK, permutation_N, top_k)


class TpuAllToAllTokenDispatcher(AllToAllTokenDispatcher):
  """Standard dispatcher with TPU-local permutation VJPs and static-shape EP transports.

  ``_local_reorder`` is shared by EP=1 and EP>1 and is optimized in both
  cases. Under EP > 1, ``ep_transport`` selects between upstream's host-syncing
  ``"all_to_all"``, the static-shape ``"ragged_all_to_all"``,
  ``"ring_of_experts"`` (all-gather + Pallas TensorCore ragged gather/reduce +
  reduce-scatter, TPU only), or ``"auto"`` (Ring-of-Experts when
  ``ep_size <= top_k`` and it can run, otherwise ragged all-to-all).
  ``ep_capacity_factor`` defaults to ``None`` (dropless worst-case buffer);
  see ``Config`` for what it bounds on each transport.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(AllToAllTokenDispatcher.Config):
    fuse_route_score_backward: bool = False
    # EP > 1 transport. EP = 1 ignores it.
    # - None (unset): `use_expert_parallel` sets "auto"; a dispatcher never
    #   passed through it runs upstream's "all_to_all". An explicit value,
    #   including "all_to_all", is always kept.
    # - "all_to_all": upstream's variable-size all-to-all (host sync).
    # - "ragged_all_to_all": static-shape ragged_all_to_all.
    # - "ring_of_experts": all-gather + Pallas TensorCore ragged
    #   gather/reduce + reduce-scatter. Needs a TPU and hidden_dim % 128 == 0,
    #   and raises otherwise.
    # - "auto": "ring_of_experts" when ep_size <= top_k and it can run, else
    #   "ragged_all_to_all" (see `_resolve_ep_transport`).
    ep_transport: EpTransport | None = None
    # Optional per-rank capacity factor, in multiples of this rank's balanced
    # routed rows `T * top_k` (MaxText's `ragged_buffer_factor`). None keeps
    # the dropless worst case `ep_size * T * top_k` rows. Routes past the cap
    # are dropped and counted in the metadata's `dropped_count`. It bounds:
    # - "ragged_all_to_all": the dispatch receive buffer, which is also the
    #   expert (grouped matmul) row count: `int(T * top_k * factor)` rows.
    # - "ring_of_experts": only the expert rows gathered locally from the
    #   all-gathered tokens. The all-gather and reduce-scatter always move all
    #   `ep_size * T` tokens. The row count is rounded up to a multiple of
    #   1024 (the Pallas combine backward's block) and never exceeds the worst
    #   case.
    ep_capacity_factor: float | None = None
    roe_gather_block_size: int = 1024
    roe_reduce_block_size: int = 896
    roe_zero_pad_tail: bool = False

  def __init__(self, config: Config):
    super().__init__(config)
    self.fuse_route_score_backward = config.fuse_route_score_backward
    if (
        config.ep_transport is not None
        and config.ep_transport not in get_args(EpTransport)
    ):
      raise ValueError(
          f"ep_transport must be one of {get_args(EpTransport)} or None, got"
          f" {config.ep_transport!r}"
      )
    # Unset means upstream's transport at runtime; `use_expert_parallel`
    # replaces an unset value with "auto" before the dispatcher is built.
    self.ep_transport: str = (
        config.ep_transport if config.ep_transport is not None else "all_to_all"
    )
    self.ep_capacity_factor = config.ep_capacity_factor
    if self.ep_capacity_factor is not None:
      if (
          isinstance(self.ep_capacity_factor, bool)
          or not math.isfinite(self.ep_capacity_factor)
          or self.ep_capacity_factor <= 0.0
      ):
        raise ValueError(
            "ep_capacity_factor must be a finite positive float or None"
            f" (dropless), got {self.ep_capacity_factor!r}"
        )
    self.roe_gather_block_size = config.roe_gather_block_size
    self.roe_reduce_block_size = config.roe_reduce_block_size
    self.roe_zero_pad_tail = config.roe_zero_pad_tail
    self._scale_before_down_proj = False
    self._ep_rank_holder_ref: weakref.ReferenceType | None = None
    self._ep_local_rank_buf: torch.Tensor | None = None
    self.last_dropped_count: torch.Tensor | None = None

  def init_buffer(self) -> None:
    super().init_buffer()
    self._ep_local_rank_buf = None
    self.last_dropped_count = None

  def _ep_buffer_capacity(
      self, num_routed: int, ep_size: int, align_rows: int = 1
  ) -> int:
    """Computes the static expert-buffer rows ``cap`` for one EP rank.

    ``num_routed`` is this rank's ``T * top_k`` routes. Without
    ``ep_capacity_factor`` this is the dropless worst case
    ``ep_size * num_routed``. With it, ``int(num_routed * factor)`` rounded up
    to a multiple of ``align_rows`` and never above the worst case.
    """
    worst_case = ep_size * max(num_routed, 1)
    if self.ep_capacity_factor is None:
      return worst_case
    if num_routed <= 0:
      return 1
    cap = max(int(num_routed * self.ep_capacity_factor), 1)
    cap = -(-cap // align_rows) * align_rows
    return min(cap, worst_case)

  def _record_dropped_count(
      self, x: torch.Tensor, dropped_count: torch.Tensor
  ) -> None:
    # Eager runs always keep the latest count. A traced count is kept only when
    # the owning TpuRoutedExperts returns drop stats as a step output
    # (``configure_ep_drop_stats``); otherwise it would outlive its trace.
    holder = (
        self._ep_rank_holder_ref()
        if self._ep_rank_holder_ref is not None
        else None
    )
    if getattr(holder, "collect_ep_drop_stats", False) or (
        get_proxy_mode() is None
        and not isinstance(x, torch._subclasses.FakeTensor)
    ):
      self.last_dropped_count = dropped_count

  def enable_row_scale_dispatch(self) -> None:
    """Requests per-row routing weights in EP dispatch metadata for ``scale_before_down_proj``."""
    self._scale_before_down_proj = True

  def bind_ep_rank_holder(self, holder: torch.nn.Module) -> None:
    """Binds the root module holding the non-persistent ``_ep_local_rank`` buffer."""
    self._ep_rank_holder_ref = weakref.ref(holder)

  def ring_of_experts_unsupported_reason(
      self, device_type: str, hidden_dim: int | None = None
  ) -> str | None:
    """Returns why ``ring_of_experts`` cannot run, or ``None`` if it can.

    Ring-of-Experts runs only on its Pallas TensorCore kernels, which need a
    TPU and a hidden size that is a multiple of the 128-wide lane dimension.
    """
    if device_type != "tpu":
      return f"it needs a TPU, got device type {device_type!r}"
    if hidden_dim is not None and hidden_dim % 128 != 0:
      return f"its Pallas kernels need hidden_dim % 128 == 0, got {hidden_dim}"
    return None

  def _resolve_ep_transport(self, ep_size: int, x_TD: torch.Tensor) -> str:
    """Returns the transport for this dispatch; raises if it cannot run."""
    reason = self.ring_of_experts_unsupported_reason(
        x_TD.device.type, x_TD.shape[-1]
    )
    if self.ep_transport == "auto":
      # Per rank, with T tokens of width D: Ring-of-Experts all-gathers the
      # tokens and reduce-scatters the partial outputs, (EP - 1) * T * D
      # elements in each direction whatever the routing. Ragged all-to-all
      # sends each of the T * top_k routes to its expert's rank and back; with
      # uniform routing (EP - 1) / EP of them leave the rank, about
      # T * top_k * (EP - 1) / EP * D elements each way, and more on hot ranks
      # under skew. Ring-of-Experts thus moves no more data iff
      # EP - 1 <= top_k * (EP - 1) / EP, i.e. EP <= top_k. Both also exchange
      # small routing data: ids and scores for Ring-of-Experts, per-expert
      # counts for ragged all-to-all.
      if ep_size <= self.top_k and reason is None:
        return "ring_of_experts"
      return "ragged_all_to_all"
    if self.ep_transport == "ring_of_experts" and reason is not None:
      raise ValueError(_ring_of_experts_unavailable_message(reason))
    return self.ep_transport

  def _get_ep_local_rank_tensor(self, x: torch.Tensor) -> torch.Tensor:
    assert self.ep_mesh is not None
    holder = (
        self._ep_rank_holder_ref()
        if self._ep_rank_holder_ref is not None
        else None
    )
    rank_buf = None
    if holder is not None:
      rank_buf = getattr(holder, "_ep_local_rank", None)
      if (
          isinstance(rank_buf, torch.Tensor)
          and rank_buf.device == x.device
          and isinstance(rank_buf, torch._subclasses.FakeTensor)
          == isinstance(x, torch._subclasses.FakeTensor)
      ):
        return rank_buf

    if get_proxy_mode() is not None or isinstance(
        x, torch._subclasses.FakeTensor
    ):
      # A traced graph must read the rank from the owner's buffer, a graph
      # input, so that every rank traces the same program. A constant would
      # bake this rank's index into the graph, so fail instead under EP > 1.
      ep_size = self.ep_mesh.size()
      if ep_size > 1:
        found = (
            f"{type(rank_buf).__name__} on {rank_buf.device}"
            if isinstance(rank_buf, torch.Tensor)
            else "missing"
        )
        raise RuntimeError(
            f"Tracing EP dispatch at ep_size={ep_size} needs the bound"
            " TpuRoutedExperts' `_ep_local_rank` buffer as a graph input on"
            f" the input's device (buffer: {found}; input:"
            f" {type(x).__name__} on {x.device}). A constant would bake this"
            " rank's index into the graph. TpuRoutedExperts.parallelize"
            " registers the buffer; trace with module buffers as graph"
            " inputs, as TpuGraphTrainer does."
        )
      # A single EP rank has index 0 everywhere.
      return torch.zeros((1,), dtype=torch.long, device=x.device)

    local_rank = int(self.ep_mesh.get_local_rank())
    buf = self._ep_local_rank_buf
    if buf is None or buf.device != x.device:
      buf = torch.tensor([local_rank], dtype=torch.long).to(device=x.device)
      self._ep_local_rank_buf = buf
    return buf

  def _gather_expert_counts(
      self,
      local_counts_E: torch.Tensor,
      pg,
      ep_size: int,
  ) -> torch.Tensor:
    if (
        local_counts_E.device.type == "tpu"
        and pg is not None
        and hasattr(pg, "group_name")
    ):
      return torch.tpu.ops.all_gather_dim(local_counts_E, 0, group=pg)
    if (
        self.ep_mesh is not None
        and pg is not None
        and get_proxy_mode() is None
        and not isinstance(local_counts_E, torch._subclasses.FakeTensor)
    ):
      return _wait(
          all_gather_single(local_counts_E, gather_dim=0, group=self.ep_mesh)
      )
    if ep_size == 1:
      return local_counts_E
    raise NotImplementedError(
        f"Gathering expert counts over {ep_size} EP ranks needs a TPU process"
        " group when traced or a live EP mesh eagerly."
    )

  def _ragged_all_to_all(
      self,
      operand: torch.Tensor,
      output_rows: int,
      fwd_input_offsets: torch.Tensor,
      fwd_send_sizes: torch.Tensor,
      fwd_output_offsets: torch.Tensor,
      fwd_recv_sizes: torch.Tensor,
      bwd_input_offsets: torch.Tensor,
      bwd_send_sizes: torch.Tensor,
      bwd_output_offsets: torch.Tensor,
      bwd_recv_sizes: torch.Tensor,
      pg,
  ) -> torch.Tensor:
    return _RaggedAllToAll.apply(
        operand,
        output_rows,
        fwd_input_offsets,
        fwd_send_sizes,
        fwd_output_offsets,
        fwd_recv_sizes,
        bwd_input_offsets,
        bwd_send_sizes,
        bwd_output_offsets,
        bwd_recv_sizes,
        self.ep_mesh,
        pg,
    )

  def _local_reorder(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      topk_expert_ids_TK: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return local_reorder(
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        self.top_k,
    )

  def _empty_ring_of_experts_dispatch(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      cap: int,
      num_local_experts: int,
  ) -> tuple[torch.Tensor, torch.Tensor, TpuRoEDispatchMetadata]:
    """Zero-token Ring-of-Experts dispatch: no collectives and no kernels.

    Every EP rank has the same static token count, so all ranks take this
    path together; combine returns zeros for it.
    """
    device = x_TD.device
    empty_i32 = torch.zeros((0,), dtype=torch.int32, device=device)
    zero_i32 = torch.zeros((), dtype=torch.int32, device=device)
    self._record_dropped_count(x_TD, zero_i32)
    metadata = TpuRoEDispatchMetadata(
        token_indices_experts_sorted_N=empty_i32,
        valid_count=zero_i32,
        dropped_count=zero_i32,
        row_scale_R=(
            topk_scores_TK.new_zeros((cap,))
            if self._scale_before_down_proj
            else None
        ),
        num_global_tokens=0,
        scores_flat_M=topk_scores_TK.reshape(-1),
        window_slot_order_cap=torch.zeros(
            (cap,), dtype=torch.int32, device=device
        ),
        start_count_2=torch.zeros((2,), dtype=torch.int32, device=device),
        src_rows=empty_i32,
        tokens=empty_i32,
        blocks=empty_i32,
        gather_block_size=self.roe_gather_block_size,
        reduce_block_size=self.roe_reduce_block_size,
        zero_pad_tail=self.roe_zero_pad_tail,
    )
    return (
        x_TD.new_zeros((cap, x_TD.shape[1])),
        torch.zeros((num_local_experts,), dtype=torch.int32, device=device),
        metadata,
    )

  def _dispatch_ring_of_experts(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      topk_expert_ids_TK: torch.Tensor,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      TpuRoEDispatchMetadata,
  ]:
    """Ring-of-Experts dispatch on the Pallas TensorCore kernels.

    All-gathers every EP rank's tokens, scores, and expert ids, then gathers
    the routes to this rank's local experts into a static ``[cap, D]``
    buffer. ``_resolve_ep_transport`` has already checked that these kernels
    can run (TPU, ``hidden_dim % 128 == 0``); there is no other production
    implementation. ``torchtitan/experiments/tpu/testing/roe_reference.py``
    holds a pure-PyTorch reference for tests.
    """
    assert self.ep_mesh is not None
    # Lazy import so CPU-only environments never import JAX/Pallas.
    # pylint: disable=g-import-not-at-top
    from torchtitan.experiments.tpu.kernels.ragged_gather_tc import (
        roe_compute_routing,
        roe_gather_row_scales,
        roe_sort_gather,
        safe_gather_block_size,
        safe_reduce_block_size,
        SCALE_DSCORE_BLOCK_ROWS,
    )
    # pylint: enable=g-import-not-at-top

    ep_size = self.ep_mesh.size()
    e = self.num_experts // ep_size
    num_tokens, hidden_dim = x_TD.shape
    g_tokens = ep_size * num_tokens
    num_routed = num_tokens * self.top_k
    worst_case_cap = ep_size * max(num_routed, 1)
    # The combine backward (`_ragged_scale_and_dscore_tc`) runs its Pallas
    # kernel only for a multiple of SCALE_DSCORE_BLOCK_ROWS rows and an XLA
    # fallback otherwise, so a capped buffer is rounded up to that multiple:
    # capping never moves the step off the kernel. A multiple of 1024 rows
    # also needs no row padding in the Pallas grouped matmuls.
    cap = self._ep_buffer_capacity(
        num_routed, ep_size, align_rows=SCALE_DSCORE_BLOCK_ROWS
    )
    is_capped = cap < worst_case_cap

    if num_tokens == 0:
      return self._empty_ring_of_experts_dispatch(x_TD, topk_scores_TK, cap, e)

    with maybe_set_sparse_mesh():
      pg = self.ep_mesh.get_group()
      reduce_block = safe_reduce_block_size(
          hidden_dim=hidden_dim,
          itemsize=x_TD.element_size(),
          top_k=self.top_k,
          requested_block_size=self.roe_reduce_block_size,
      )
      gather_block = min(
          safe_gather_block_size(
              hidden_dim=hidden_dim,
              itemsize=x_TD.element_size(),
              requested_block_size=self.roe_gather_block_size,
          ),
          cap,
      )

      with torch.no_grad():
        rank_idx_i32 = self._get_ep_local_rank_tensor(x_TD).to(torch.int32)
        ids_i32_TK = topk_expert_ids_TK.to(torch.int32).contiguous()
        with annotate_region(MOE_ROUTING):
          ids_all_GK = _ep_all_gather_impl(ids_i32_TK, self.ep_mesh, pg)
        (
            num_global_tokens_per_local_expert_e,
            token_ids_sorted_M,
            window_slot_order_cap,
            start_count_2,
            src_rows,
            tokens,
            blocks,
        ) = roe_compute_routing(
            ids_all_GK.reshape(-1),
            rank_idx_i32,
            num_experts=self.num_experts,
            ep_size=ep_size,
            top_k=self.top_k,
            num_out_tokens=g_tokens,
            cap=cap,
            reduce_block_size=reduce_block,
        )
        valid_count = start_count_2[1]
        if is_capped:
          rel_ids_GK = ids_all_GK - rank_idx_i32 * e
          unclamped_count = (
              (rel_ids_GK >= 0) & (rel_ids_GK < e)
          ).sum(dtype=torch.int32)
          dropped_count = unclamped_count - valid_count
        else:
          dropped_count = valid_count.new_zeros(())
        self._record_dropped_count(x_TD, dropped_count)

      with annotate_region(MOE_EP_DISPATCH):
        x_all_GD = _EpAllGather.apply(x_TD, self.ep_mesh, pg)
        scores_all_GK = _EpAllGather.apply(topk_scores_TK, self.ep_mesh, pg)

      routed_input_RD = roe_sort_gather(
          x_all_GD,
          token_ids_sorted_M,
          start_count_2,
          src_rows,
          tokens,
          blocks,
          cap=cap,
          top_k=self.top_k,
          gather_block_size=gather_block,
          reduce_block_size=reduce_block,
          zero_pad_tail=self.roe_zero_pad_tail,
      )
      row_scale_R = None
      if self._scale_before_down_proj:
        row_scale_R = roe_gather_row_scales(
            scores_all_GK.reshape(-1),
            window_slot_order_cap,
            start_count_2,
        )
      metadata = TpuRoEDispatchMetadata(
          token_indices_experts_sorted_N=token_ids_sorted_M,
          valid_count=valid_count,
          dropped_count=dropped_count,
          row_scale_R=row_scale_R,
          num_global_tokens=g_tokens,
          scores_flat_M=scores_all_GK.reshape(-1),
          window_slot_order_cap=window_slot_order_cap,
          start_count_2=start_count_2,
          src_rows=src_rows,
          tokens=tokens,
          blocks=blocks,
          gather_block_size=gather_block,
          reduce_block_size=reduce_block,
          zero_pad_tail=self.roe_zero_pad_tail,
      )
      return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

  def _combine_ring_of_experts(
      self,
      routed_output_RD: torch.Tensor,
      metadata: TpuEpDispatchMetadata,
      x_TD: torch.Tensor,
  ) -> torch.Tensor:
    """Weights, unsorts, and reduces expert outputs, then reduce-scatters them."""
    assert self.ep_mesh is not None
    if not isinstance(metadata, TpuRoEDispatchMetadata):
      raise TypeError(
          "Ring-of-Experts combine expects TpuRoEDispatchMetadata, got"
          f" {type(metadata).__name__}"
      )
    if metadata.num_global_tokens == 0:
      return x_TD.new_zeros(x_TD.shape)

    # pylint: disable=g-import-not-at-top
    from torchtitan.experiments.tpu.kernels.ragged_gather_tc import (
        roe_unsort_combine,
        roe_unsort_reduce,
    )
    # pylint: enable=g-import-not-at-top

    with maybe_set_sparse_mesh():
      pg = self.ep_mesh.get_group()
      if metadata.row_scale_applied:
        partial_GD = roe_unsort_reduce(
            routed_output_RD,
            metadata.token_indices_experts_sorted_N,
            metadata.start_count_2,
            metadata.src_rows,
            metadata.tokens,
            metadata.blocks,
            num_out_tokens=metadata.num_global_tokens,
            top_k=self.top_k,
            gather_block_size=metadata.gather_block_size,
            reduce_block_size=metadata.reduce_block_size,
        )
      else:
        partial_GD = roe_unsort_combine(
            routed_output_RD,
            metadata.scores_flat_M,
            metadata.token_indices_experts_sorted_N,
            metadata.window_slot_order_cap,
            metadata.start_count_2,
            metadata.src_rows,
            metadata.tokens,
            metadata.blocks,
            num_out_tokens=metadata.num_global_tokens,
            top_k=self.top_k,
            gather_block_size=metadata.gather_block_size,
            reduce_block_size=metadata.reduce_block_size,
            zero_pad_tail=metadata.zero_pad_tail,
        )
      return _EpReduceScatter.apply(partial_GD, self.ep_mesh, pg)

  @annotate_fn({"EP": "dispatch"})
  def dispatch(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      topk_expert_ids_TK: torch.Tensor,
      num_local_tokens_per_expert_E: torch.Tensor,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      (
          AllToAllDispatchMetadata
          | LocalDispatchMetadata
          | TpuEpDispatchMetadata
      ),
  ]:
    if self.ep_mesh is None or self.ep_transport == "all_to_all":
      return super().dispatch(
          x_TD,
          topk_scores_TK,
          topk_expert_ids_TK,
          num_local_tokens_per_expert_E,
      )

    if self._ep_rank_holder_ref is None or self._ep_rank_holder_ref() is None:
      raise RuntimeError(
          f"ep_transport={self.ep_transport!r} requires"
          " TpuAllToAllTokenDispatcher to be bound to a TpuRoutedExperts owner"
          " via bind_ep_rank_holder under EP > 1."
      )
    ep_size = self.ep_mesh.size()
    resolved_transport = self._resolve_ep_transport(ep_size, x_TD)

    if resolved_transport == "ring_of_experts":
      return self._dispatch_ring_of_experts(
          x_TD, topk_scores_TK, topk_expert_ids_TK
      )

    e = self.num_experts // ep_size
    (
        routed_input_ND,
        token_indices_experts_sorted_N,
        topk_scores_experts_sorted_N,
    ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

    num_routed = routed_input_ND.shape[0]
    worst_case_rows = ep_size * max(num_routed, 1)
    r_static = self._ep_buffer_capacity(num_routed, ep_size)
    is_capped = r_static < worst_case_rows

    with maybe_set_sparse_mesh():
      pg = self.ep_mesh.get_group()
      with torch.no_grad():
        with annotate_region(MOE_ROUTING):
          # Integer on purpose: TpuGraphTrainer's Splash and FSDP passes skip
          # integer gathers, so they never take this one for a weight gather.
          local_counts_E = num_local_tokens_per_expert_E.to(
              torch.int32
          ).reshape(-1)
          all_sizes_SE = (
              self._gather_expert_counts(local_counts_E, pg, ep_size)
              .to(torch.int32)
              .reshape(ep_size, self.num_experts)
          )
        counts_SDE = all_sizes_SE.view(ep_size, ep_size, e)
        all_send_offsets_SE = (
            all_sizes_SE.cumsum(dim=-1, dtype=torch.int32) - all_sizes_SE
        )

        recv_flat_D_eS = counts_SDE.permute(1, 2, 0).reshape(
            ep_size, e * ep_size
        )
        rank_idx = self._get_ep_local_rank_tensor(x_TD)
        if is_capped:
          untrunc_recv_sizes = (
              torch.index_select(counts_SDE, 1, rank_idx)
              .reshape(-1)
              .contiguous()
          )
          recv_cum_D_eS = recv_flat_D_eS.cumsum(dim=-1, dtype=torch.int32)
          clamped_recv_cum_D_eS = torch.clamp(recv_cum_D_eS, max=r_static)
          prev_clamped_D_eS = F.pad(clamped_recv_cum_D_eS[:, :-1], (1, 0))
          eff_recv_flat_D_eS = clamped_recv_cum_D_eS - prev_clamped_D_eS
          eff_counts_SDE = (
              eff_recv_flat_D_eS.view(ep_size, e, ep_size)
              .permute(2, 0, 1)
              .contiguous()
          )
          eff_sizes_SE = eff_counts_SDE.reshape(ep_size, self.num_experts)
        else:
          eff_recv_flat_D_eS = recv_flat_D_eS
          eff_counts_SDE = counts_SDE
          eff_sizes_SE = all_sizes_SE

        write_off_DES = (
            eff_recv_flat_D_eS.cumsum(dim=-1, dtype=torch.int32)
            - eff_recv_flat_D_eS
        ).view(ep_size, e, ep_size)

        disp_input_offsets = (
            torch.index_select(all_send_offsets_SE, 0, rank_idx)
            .reshape(-1)
            .contiguous()
        )
        disp_send_sizes = (
            torch.index_select(eff_sizes_SE, 0, rank_idx)
            .reshape(-1)
            .contiguous()
        )
        disp_output_offsets = (
            torch.index_select(write_off_DES, 2, rank_idx)
            .reshape(-1)
            .contiguous()
        )
        disp_recv_sizes = (
            torch.index_select(eff_counts_SDE, 1, rank_idx)
            .reshape(-1)
            .contiguous()
        )

        comb_input_offsets = (
            torch.index_select(write_off_DES, 0, rank_idx)
            .squeeze(0)
            .transpose(0, 1)
            .reshape(-1)
            .contiguous()
        )
        comb_send_sizes = disp_recv_sizes
        comb_output_offsets = (
            torch.index_select(
                all_send_offsets_SE.view(ep_size, ep_size, e), 1, rank_idx
            )
            .reshape(-1)
            .contiguous()
        )
        comb_recv_sizes = disp_send_sizes

        num_global_tokens_per_local_expert_e = disp_recv_sizes.view(
            ep_size, e
        ).sum(dim=0, dtype=torch.int32)
        valid_count = num_global_tokens_per_local_expert_e.sum(
            dtype=torch.int32
        )
        if is_capped:
          dropped_count = untrunc_recv_sizes.sum(dtype=torch.int32) - valid_count
        else:
          dropped_count = valid_count.new_zeros(())
        self._record_dropped_count(x_TD, dropped_count)

      routed_input_RD = self._ragged_all_to_all(
          routed_input_ND,
          r_static,
          disp_input_offsets,
          disp_send_sizes,
          disp_output_offsets,
          disp_recv_sizes,
          comb_input_offsets,
          comb_send_sizes,
          comb_output_offsets,
          comb_recv_sizes,
          pg,
      )
      row_scale_R = None
      if self._scale_before_down_proj:
        row_scale_R = self._ragged_all_to_all(
            topk_scores_experts_sorted_N.unsqueeze(-1),
            r_static,
            disp_input_offsets,
            disp_send_sizes,
            disp_output_offsets,
            disp_recv_sizes,
            comb_input_offsets,
            comb_send_sizes,
            comb_output_offsets,
            comb_recv_sizes,
            pg,
        ).squeeze(-1)

    metadata = TpuRaggedDispatchMetadata(
        token_indices_experts_sorted_N=token_indices_experts_sorted_N,
        topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
        routed_rows=num_routed,
        disp_input_offsets=disp_input_offsets,
        disp_send_sizes=disp_send_sizes,
        disp_output_offsets=disp_output_offsets,
        disp_recv_sizes=disp_recv_sizes,
        comb_input_offsets=comb_input_offsets,
        comb_send_sizes=comb_send_sizes,
        comb_output_offsets=comb_output_offsets,
        comb_recv_sizes=comb_recv_sizes,
        valid_count=valid_count,
        dropped_count=dropped_count,
        row_scale_R=row_scale_R,
    )
    return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

  @annotate_fn({"EP": "combine"})
  def combine(
      self,
      routed_output_RD: torch.Tensor,
      metadata: (
          AllToAllDispatchMetadata
          | LocalDispatchMetadata
          | TpuEpDispatchMetadata
      ),
      x_TD: torch.Tensor,
  ) -> torch.Tensor:
    if self.ep_mesh is not None and self.ep_transport == "all_to_all":
      return super().combine(routed_output_RD, metadata, x_TD)

    if self.ep_mesh is None:
      # EP=1: the rows are already this rank's own routes.
      assert isinstance(metadata, LocalDispatchMetadata)
      routed_output_ND = routed_output_RD
      row_scale_applied = False
    elif not isinstance(metadata, TpuRaggedDispatchMetadata):
      # Ring-of-Experts weights and reduces inside its own combine.
      assert isinstance(metadata, TpuEpDispatchMetadata), type(metadata)
      return self._combine_ring_of_experts(routed_output_RD, metadata, x_TD)
    else:
      # Ragged EP: return each route's row to its source rank first. The
      # returned rows are in the same expert-sorted order as local dispatch,
      # so the same fixed-K reduction applies.
      row_scale_applied = metadata.row_scale_applied
      with maybe_set_sparse_mesh():
        pg = self.ep_mesh.get_group()
        routed_output_ND = self._ragged_all_to_all(
            routed_output_RD,
            metadata.routed_rows,
            metadata.comb_input_offsets,
            metadata.comb_send_sizes,
            metadata.comb_output_offsets,
            metadata.comb_recv_sizes,
            metadata.disp_input_offsets,
            metadata.disp_send_sizes,
            metadata.disp_output_offsets,
            metadata.disp_recv_sizes,
            pg,
        )

    if not row_scale_applied:
      if self.fuse_route_score_backward:
        routed_output_ND = scale_route_weights(
            routed_output_ND,
            metadata.topk_scores_experts_sorted_N,
        )
      else:
        routed_output_ND = (
            routed_output_ND.float()
            * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
        ).to(routed_output_ND.dtype)
    return stable_grouped_reduce(
        routed_output_ND,
        metadata.token_indices_experts_sorted_N,
        self.top_k,
    )


class TpuRoutedExperts(RoutedExperts):
  """Routed experts that can apply the routing scores before W2.

  W2 is linear, so scaling its input rows gives the same result as scaling its
  output rows. Scaling the input makes the score gradient
  ``rowsum(hidden * dHidden)``, which does not read the W2 output, so a
  rematerializing backward no longer recomputes the W2 forward. Only rounding
  changes: each BF16 hidden row is scaled in FP32 and rounded before W2,
  instead of each W2 output row after it. Under EP > 1, the TPU transports
  (``ring_of_experts``, ``ragged_all_to_all``, ``auto``) carry per-row routing
  weights to the expert rank and skip the post-W2 weight multiply in combine.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(RoutedExperts.Config):
    scale_before_down_proj: bool = False
    num_token_chunks: int = 1
    max_inflight_chunks: int = 2
    accumulate_chunk_weight_grads_fp32: bool = True
    # With token chunking, add each chunk's expert dWeight onto a running sum
    # inside the TGMM kernels (MegaBlox `existing_out`) instead of storing one
    # dWeight per chunk and summing them in autograd through an FP32 copy of
    # the weights. Needs the BF16 Pallas grouped matmuls; other experts keep
    # the autograd sums.
    accumulate_chunk_weight_grads_in_kernel: bool = True

  def __init__(self, config: Config):
    super().__init__(config)
    # Capacity-bounded EP only, enabled by ``configure_ep_drop_stats``: each
    # forward's ``[dropped, total]`` route counts. A plain attribute, not a
    # buffer, because TpuGraphTrainer does not write buffer updates made inside
    # a traced step back to the module; the step returns these as an output.
    self.collect_ep_drop_stats = False
    self._last_ep_drop_stats_2: torch.Tensor | None = None
    if config.num_token_chunks < 1:
      raise ValueError(
          f"num_token_chunks must be >= 1, got {config.num_token_chunks}"
      )
    if config.max_inflight_chunks < 1:
      raise ValueError(
          f"max_inflight_chunks must be >= 1, got {config.max_inflight_chunks}"
      )
    self.num_token_chunks = config.num_token_chunks
    self.max_inflight_chunks = config.max_inflight_chunks
    self.accumulate_chunk_weight_grads_fp32 = (
        config.accumulate_chunk_weight_grads_fp32
    )
    self.accumulate_chunk_weight_grads_in_kernel = (
        config.accumulate_chunk_weight_grads_in_kernel
    )
    self._warned_chunk_snap: set[tuple[int, int]] = set()
    self.scale_before_down_proj = config.scale_before_down_proj
    if not self.scale_before_down_proj:
      return
    if not isinstance(self.token_dispatcher, TpuAllToAllTokenDispatcher):
      raise ValueError(
          "scale_before_down_proj requires TpuAllToAllTokenDispatcher"
      )
    if self.token_dispatcher.fuse_route_score_backward:
      raise ValueError(
          "fuse_route_score_backward fuses the score multiply after W2, which"
          " scale_before_down_proj removes; enable only one"
      )
    self.token_dispatcher.enable_row_scale_dispatch()

  def _init_self_buffers(
      self, *, buffer_device: torch.device | None = None
  ) -> None:
    super()._init_self_buffers(buffer_device=buffer_device)
    ep_mesh = getattr(self, "_ep_mesh", None)
    if ep_mesh is not None and "_ep_local_rank" in self._buffers:
      target_dev = (
          buffer_device
          if buffer_device is not None
          else self._ep_local_rank.device
      )
      self._ep_local_rank = torch.tensor(
          [int(ep_mesh.get_local_rank())], dtype=torch.long
      ).to(device=target_dev)

  def parallelize(self, parallel_dims) -> None:
    # Upstream's `all_to_all` transport keeps routing scores on the source rank
    # while W2 runs on the expert's rank after dispatch. The TPU EP transports
    # (`ring_of_experts`, `ragged_all_to_all`, `auto`) move per-row scores to
    # the expert rank, so only `all_to_all` must be refused here.
    dispatcher = getattr(self, "token_dispatcher", None)
    ep_transport = getattr(dispatcher, "ep_transport", "all_to_all")
    if (
        self.scale_before_down_proj
        and parallel_dims.get_optional_mesh("ep") is not None
        and ep_transport == "all_to_all"
    ):
      raise ValueError(
          "scale_before_down_proj with ep_transport='all_to_all' requires"
          " expert_parallel_degree=1; use ep_transport in ('ring_of_experts',"
          " 'ragged_all_to_all', 'auto') or disable it to run expert"
          " parallelism"
      )
    ep_mesh = parallel_dims.get_optional_mesh("ep")
    # An explicit Ring-of-Experts transport that cannot run fails here rather
    # than in the first forward when the mesh says so; dispatch checks again
    # with the real hidden size.
    mesh_device_type = getattr(ep_mesh, "device_type", None)
    if (
        isinstance(dispatcher, TpuAllToAllTokenDispatcher)
        and ep_transport == "ring_of_experts"
        and isinstance(mesh_device_type, str)
    ):
      reason = dispatcher.ring_of_experts_unsupported_reason(mesh_device_type)
      if reason is not None:
        raise ValueError(_ring_of_experts_unavailable_message(reason))
    super().parallelize(parallel_dims)
    if ep_mesh is not None and callable(getattr(ep_mesh, "get_local_rank", None)):
      if ep_transport in ("ragged_all_to_all", "ring_of_experts", "auto"):
        self._ep_mesh = ep_mesh
        param = next(self.parameters(), None)
        param_dev = param.device if param is not None else torch.device("cpu")
        rank_buf = (
            torch.empty((1,), dtype=torch.long, device="meta")
            if param_dev.type == "meta"
            else torch.tensor(
                [int(ep_mesh.get_local_rank())], dtype=torch.long
            ).to(device=param_dev)
        )
        self.register_buffer("_ep_local_rank", rank_buf, persistent=False)
        dispatcher.bind_ep_rank_holder(self)
        if isinstance(dispatcher, TpuAllToAllTokenDispatcher) and not getattr(
            self.inner_experts, "use_pallas_grouped_mm", False
        ):
          dispatcher.roe_zero_pad_tail = True

  @property
  def ep_capacity_bounded(self) -> bool:
    """Whether this layer runs a TPU EP transport with a capped expert buffer."""
    return (
        getattr(self, "_ep_mesh", None) is not None
        and getattr(self.token_dispatcher, "ep_capacity_factor", None)
        is not None
    )

  def _update_ep_drop_stats(
      self, dropped_count: torch.Tensor, num_tokens: int
  ) -> None:
    """Records this forward's ``[dropped, total]`` local routes as int32 ``[2]``."""
    with torch.no_grad():
      dropped = dropped_count.to(torch.int32).reshape(())
      total = torch.full_like(dropped, num_tokens * self.token_dispatcher.top_k)
      self._last_ep_drop_stats_2 = torch.stack([dropped, total])

  def _forward_ep_chunked(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      topk_expert_ids_TK: torch.Tensor,
      num_local_tokens_per_expert_E: torch.Tensor,
      *,
      num_chunks: int,
      ep_size: int,
  ) -> torch.Tensor:
    if isinstance(self.inner_experts, FSDPModule):
      raise NotImplementedError(
          "EP token chunking unshards the expert weights once per layer and"
          " needs SimpleFSDP (--parallelism.use_simple_fsdp); FSDP2 reshards"
          " them when the unshard call returns. Use num_token_chunks=1."
      )
    is_roe = (
        isinstance(self.token_dispatcher, TpuAllToAllTokenDispatcher)
        and self.token_dispatcher._resolve_ep_transport(ep_size, x_TD)
        == "ring_of_experts"
    )
    needs_chunk_counts = not is_roe

    unsharded_weights = None
    dweight_accumulators = None
    if hasattr(self.inner_experts, "unshard_weights"):
      # Unshard the SimpleFSDP expert parameters once, outside the chunk loop
      # (`TpuGroupedExperts.unshard_weights` explains how its all-gathers keep
      # the experts' `module_fqn` for SimpleFSDP bucketing).
      with maybe_set_sparse_mesh():
        unsharded_weights = self.inner_experts.unshard_weights()
        supports_in_kernel = getattr(
            self.inner_experts, "supports_chunk_dweight_accumulation", None
        )
        if (
            self.accumulate_chunk_weight_grads_in_kernel
            and supports_in_kernel is not None
            and supports_in_kernel()
        ):
          # Each chunk's TGMMs add their dWeights onto one running sum per
          # weight inside the kernel, in FP32 by default, and the weights get
          # the total once (`TpuGroupedExperts.chunk_dweight_accumulators`).
          # The chunks then need the weights only in BF16, cast once here.
          dweight_accumulators = self.inner_experts.chunk_dweight_accumulators(
              unsharded_weights,
              accumulate_fp32=self.accumulate_chunk_weight_grads_fp32,
          )
          unsharded_weights = tuple(
              w if w.dtype == torch.bfloat16 else w.bfloat16()
              for w in unsharded_weights
          )
        elif self.accumulate_chunk_weight_grads_fp32:
          # Widening the unsharded weights to FP32 here causes
          # ``TpuGroupedExperts.forward``'s per-chunk ``.bfloat16()`` casts to
          # widen each chunk's dWeight to FP32 before PyTorch autograd sums
          # across chunks; ``insert_ep_chunk_barriers`` keeps those widening
          # casts out of forward barriers so XLA folds ``bf16(fp32(w_bf16))``
          # to ``w_bf16``.
          # TODO(b/587943723): TpuGraphTrainer ZeRO-3 with chunk barriers
          # folds this cast away (see insert_ep_chunk_barriers), but the eager
          # trainer without activation checkpointing saves one BF16 weight
          # copy per chunk (each chunk's `.bfloat16()` re-cast), and
          # GraphTrainer ZeRO-2 saves this FP32 cast as the unshard output
          # (graph_trainer/fsdp_patterns.py, `_find_last_user_in_wait_chain`).
          # The in-kernel accumulation above avoids both.
          unsharded_weights = tuple(
              w.float() if w.dtype != torch.float32 else w
              for w in unsharded_weights
          )

    x_chunks = x_TD.chunk(num_chunks, dim=0)
    score_chunks = topk_scores_TK.chunk(num_chunks, dim=0)
    id_chunks = topk_expert_ids_TK.chunk(num_chunks, dim=0)
    num_experts = num_local_tokens_per_expert_E.numel()
    extra_expert_kwargs = (
        {"pre_unsharded_weights": unsharded_weights}
        if unsharded_weights is not None
        else {}
    )

    out_chunks: list[torch.Tensor] = []
    dropped_chunks: list[torch.Tensor] = []
    # Sum the chunks' dropped routes eagerly, and when tracing only if this
    # layer returns drop stats from the step (dropless graphs are unchanged).
    record_eager_drops = isinstance(
        self.token_dispatcher, TpuAllToAllTokenDispatcher
    ) and (
        self.collect_ep_drop_stats
        or (
            get_proxy_mode() is None
            and not isinstance(x_TD, torch._subclasses.FakeTensor)
        )
    )
    for chunk_idx, (x_c, scores_c, ids_c) in enumerate(
        zip(x_chunks, score_chunks, id_chunks)
    ):
      with annotate({
          "ep_chunk_idx": chunk_idx,
          "ep_max_inflight_chunks": self.max_inflight_chunks,
      }):
        counts_c = (
            _tokens_per_expert(ids_c, num_experts)
            if needs_chunk_counts
            else num_local_tokens_per_expert_E
        )
        routed_input_c, num_tokens_per_expert_e, metadata_c = (
            self.token_dispatcher.dispatch(x_c, scores_c, ids_c, counts_c)
        )
        if (
            record_eager_drops
            and isinstance(metadata_c, TpuEpDispatchMetadata)
        ):
          dropped_chunks.append(metadata_c.dropped_count)
        expert_kwargs = dict(extra_expert_kwargs)
        if dweight_accumulators is not None:
          expert_kwargs["dweight_accumulators"] = dweight_accumulators
          expert_kwargs["last_chunk"] = chunk_idx == len(x_chunks) - 1
        if self.scale_before_down_proj:
          if not isinstance(metadata_c, TpuEpDispatchMetadata):
            raise ValueError(
                "Chunked EP scale_before_down_proj requires TPU EP dispatch"
                f" metadata, got {type(metadata_c).__name__}"
            )
          if metadata_c.row_scale_R is None:
            raise ValueError(
                "scale_before_down_proj expected EP dispatch metadata with"
                " row_scale_R populated"
            )
          with maybe_set_sparse_mesh():
            routed_output_c = self.inner_experts(
                routed_input_c,
                num_tokens_per_expert_e,
                row_scale_R=metadata_c.row_scale_R,
                **expert_kwargs,
            )
          metadata_c = replace(metadata_c, row_scale_applied=True)
        else:
          with maybe_set_sparse_mesh():
            routed_output_c = self.inner_experts(
                routed_input_c,
                num_tokens_per_expert_e,
                **expert_kwargs,
            )
        if dweight_accumulators is not None:
          routed_output_c, dweight_accumulators = routed_output_c
        out_chunks.append(
            self.token_dispatcher.combine(routed_output_c, metadata_c, x_c)
        )
    if dropped_chunks and isinstance(
        self.token_dispatcher, TpuAllToAllTokenDispatcher
    ):
      self.token_dispatcher._record_dropped_count(
          x_TD, torch.stack(dropped_chunks).sum(dtype=torch.int32)
      )
    return torch.cat(out_chunks, dim=0)

  def forward(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      topk_expert_ids_TK: torch.Tensor,
      num_local_tokens_per_expert_E: torch.Tensor,
  ) -> torch.Tensor:
    if not self.collect_ep_drop_stats:
      return self._forward_impl(
          x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
      )
    # Every TPU EP dispatch path records its count in the dispatcher's
    # last_dropped_count. Clear both first so a path that doesn't record makes
    # collect_ep_drop_stat_tensors raise instead of reporting stale counts.
    self._last_ep_drop_stats_2 = None
    self.token_dispatcher.last_dropped_count = None
    out = self._forward_impl(
        x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
    )
    dropped_count = self.token_dispatcher.last_dropped_count
    if dropped_count is not None:
      self._update_ep_drop_stats(dropped_count, x_TD.shape[0])
    return out

  def _forward_impl(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      topk_expert_ids_TK: torch.Tensor,
      num_local_tokens_per_expert_E: torch.Tensor,
  ) -> torch.Tensor:
    ep_mesh = getattr(self.token_dispatcher, "ep_mesh", None)
    num_tokens = x_TD.shape[0]
    if (
        ep_mesh is not None
        and ep_mesh.size() > 1
        and self.num_token_chunks > 1
        and num_tokens > 0
        and getattr(self.token_dispatcher, "ep_transport", "all_to_all")
        in ("ring_of_experts", "ragged_all_to_all", "auto")
    ):
      num_chunks = self.num_token_chunks
      if num_tokens % num_chunks != 0:
        num_chunks = next(
            d
            for d in range(min(num_chunks, num_tokens), 0, -1)
            if num_tokens % d == 0
        )
        if (num_tokens, num_chunks) not in self._warned_chunk_snap:
          self._warned_chunk_snap.add((num_tokens, num_chunks))
          logger.warning(
              "%d tokens are not divisible by num_token_chunks=%d; using %d"
              " chunks (fewer chunks keep more EP buffers live).",
              num_tokens,
              self.num_token_chunks,
              num_chunks,
          )
      if num_chunks > 1:
        return self._forward_ep_chunked(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
            num_chunks=num_chunks,
            ep_size=ep_mesh.size(),
        )

    if not self.scale_before_down_proj:
      return super().forward(
          x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
      )
    routed_input_RD, num_tokens_per_expert_e, metadata = (
        self.token_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
    )
    if isinstance(metadata, TpuEpDispatchMetadata):
      if metadata.row_scale_R is None:
        raise ValueError(
            "scale_before_down_proj expected EP dispatch metadata with"
            " row_scale_R populated"
        )
      with maybe_set_sparse_mesh():
        routed_output_RD = self.inner_experts(
            routed_input_RD,
            num_tokens_per_expert_e,
            row_scale_R=metadata.row_scale_R,
        )
      return self.token_dispatcher.combine(
          routed_output_RD,
          replace(metadata, row_scale_applied=True),
          x_TD,
      )
    # AllToAllDispatchMetadata subclasses LocalDispatchMetadata, so check for
    # it explicitly: the local reduce below has no return all-to-all.
    if not isinstance(metadata, LocalDispatchMetadata) or isinstance(
        metadata, AllToAllDispatchMetadata
    ):
      raise ValueError(
          "scale_before_down_proj requires EP=1 local dispatch or TPU EP"
          " dispatch metadata"
      )
    with maybe_set_sparse_mesh():
      routed_output_RD = self.inner_experts(
          routed_input_RD,
          num_tokens_per_expert_e,
          row_scale_R=metadata.topk_scores_experts_sorted_N,
      )
    return stable_grouped_reduce(
        routed_output_RD,
        metadata.token_indices_experts_sorted_N,
        self.token_dispatcher.top_k,
    )


def _ep_capacity_bounded_modules(
    model_parts: list[torch.nn.Module],
) -> list[TpuRoutedExperts]:
  return [
      module
      for part in model_parts
      for module in part.modules()
      if isinstance(module, TpuRoutedExperts) and module.ep_capacity_bounded
  ]


def ep_capacity_bounded(model_parts: list[torch.nn.Module]) -> bool:
  """Whether any MoE layer runs EP with a capacity-capped expert buffer."""
  return bool(_ep_capacity_bounded_modules(model_parts))


def configure_ep_drop_stats(
    model_parts: list[torch.nn.Module], enabled: bool
) -> bool:
  """Turns per-layer dropped-route counts on or off for capacity-bounded EP.

  Off by default. A trainer that reads them turns them on before its first
  step, and a traced step must then return ``collect_ep_drop_stat_tensors``
  as an output. Returns whether any layer now records them.
  """
  modules = _ep_capacity_bounded_modules(model_parts)
  for module in modules:
    module.collect_ep_drop_stats = enabled
    module._last_ep_drop_stats_2 = None  # pylint: disable=protected-access
  return enabled and bool(modules)


def collect_ep_drop_stat_tensors(
    model_parts: list[torch.nn.Module],
) -> list[torch.Tensor]:
  """Returns ``[stats_L2]``, each recording layer's ``[dropped, total]`` routes.

  Int32 counts of this rank's routes from the latest forward, one row per
  layer in module order, or ``[]`` when no layer records them. The dropped
  count is taken on the rank that receives the routes and the total on the
  rank that sends them, so only sums over the EP group are meaningful. Inside
  a traced step, call it after the forward and return the result as a graph
  output: attributes set inside the trace are not written back, the same
  reason the MoE routing metrics are returned as outputs.
  """
  stats = []
  for module in _ep_capacity_bounded_modules(model_parts):
    if not module.collect_ep_drop_stats:
      continue
    layer_stats = module._last_ep_drop_stats_2  # pylint: disable=protected-access
    if layer_stats is None:
      raise RuntimeError(
          "EP drop stats are enabled, but a capacity-bounded MoE layer's"
          " latest forward did not record them"
      )
    stats.append(layer_stats)
  return [torch.stack(stats)] if stats else []


class TpuRoutedExpertsConverter(ModelConfigConverter):
  """Select ``TpuRoutedExperts`` for the standard routed-experts block."""

  @dataclass(kw_only=True, slots=True)
  class Config(ModelConfigConverter.Config):
    pass

  def __init__(self, config: Config):
    pass

  def convert(self, model_config):
    matches = list(model_config.traverse(RoutedExperts.Config))
    for _, config, parent, field_name in matches:
      if type(config) is not RoutedExperts.Config:
        continue
      replacement = derive(config, TpuRoutedExperts.Config)
      if parent is None:
        model_config = replacement
      elif isinstance(parent, list):
        assert isinstance(field_name, int)
        parent[field_name] = replacement
      else:
        assert isinstance(field_name, str)
        setattr(parent, field_name, replacement)
    return model_config


class TpuTokenDispatcherConverter(ModelConfigConverter):
  """Select TPU local reorder and combine VJPs for the standard dispatcher."""

  @dataclass(kw_only=True, slots=True)
  class Config(ModelConfigConverter.Config):
    pass

  def __init__(self, config: Config):
    pass

  def convert(self, model_config):
    matches = list(model_config.traverse(AllToAllTokenDispatcher.Config))
    for _, config, parent, field_name in matches:
      if type(config) is not AllToAllTokenDispatcher.Config:
        continue
      replacement = derive(config, TpuAllToAllTokenDispatcher.Config)
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
    "EpTransport",
    "TpuAllToAllTokenDispatcher",
    "TpuEpDispatchMetadata",
    "TpuRaggedDispatchMetadata",
    "TpuRoEDispatchMetadata",
    "TpuRoutedExperts",
    "TpuRoutedExpertsConverter",
    "TpuTokenDispatcherConverter",
    "collect_ep_drop_stat_tensors",
    "configure_ep_drop_stats",
    "ep_capacity_bounded",
    "local_reorder",
    "scale_route_weights",
    "stable_grouped_reduce",
]
