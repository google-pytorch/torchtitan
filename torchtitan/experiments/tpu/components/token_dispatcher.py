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
``ep_transport="ragged_all_to_all"`` adds the dropless form of that transport:
its receive buffer is sized for the worst case rather than a capacity, so it
has no overflow, and its combine calls the same local reduction after the
return exchange. Bounded capacity is still future work.
"""

from dataclasses import dataclass
import weakref

import torch
from torch.autograd.function import once_differentiable
from torch.distributed._functional_collectives import (
    AsyncCollectiveTensor,
    all_gather_single,
)
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.traceback import annotate_fn

from torchtitan.config import derive
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.experiments.tpu.kernels.route_score import route_score_grad
from torchtitan.experiments.tpu.components.remat_regions import (
    annotate_region,
    MOE_ROUTING,
)
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllDispatchMetadata,
    AllToAllTokenDispatcher,
    LocalDispatchMetadata,
)
from torchtitan.protocols.model import ModelConfigConverter


def _wait(tensor: torch.Tensor) -> torch.Tensor:
  """Waits on a functional collective's result."""
  if isinstance(tensor, AsyncCollectiveTensor):
    return tensor.wait()
  return tensor


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
class TpuRaggedDispatchMetadata:
  """Dropless static-shape EP dispatch metadata for ``ragged_all_to_all``."""

  token_indices_experts_sorted_N: torch.Tensor
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
  """Standard dispatcher with TPU-local permutation VJPs.

  ``_local_reorder`` is shared by EP=1 and EP>1 and is optimized in both
  cases. ``combine`` selects the fixed-K reduction only when ``ep_mesh`` is
  absent. The standard dropless EP path should also admit fixed-K reduction,
  but must retain its return all-to-all and separately handle SP-local rows in
  the global output layout. It intentionally delegates until that path has its
  own correctness and performance gate.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(AllToAllTokenDispatcher.Config):
    fuse_route_score_backward: bool = False
    # EP > 1 transport. "all_to_all": upstream's variable-size all-to-all,
    # which copies the splits to the host. "ragged_all_to_all": dropless and
    # static-shape (no host sync; traceable), with a worst-case receive buffer
    # of ep_size * T * top_k rows per rank. EP=1 ignores this.
    ep_transport: str = "all_to_all"

  def __init__(self, config: Config):
    super().__init__(config)
    self.fuse_route_score_backward = config.fuse_route_score_backward
    self.ep_transport = config.ep_transport
    if self.ep_transport not in ("all_to_all", "ragged_all_to_all"):
      raise ValueError(
          f"ep_transport must be 'all_to_all' or 'ragged_all_to_all', got"
          f" {self.ep_transport!r}"
      )
    self._ep_rank_holder_ref: weakref.ReferenceType | None = None
    self._ep_local_rank_buf: torch.Tensor | None = None

  def init_buffer(self) -> None:
    super().init_buffer()
    self._ep_local_rank_buf = None

  def bind_ep_rank_holder(self, holder: torch.nn.Module) -> None:
    """Binds the root module holding the non-persistent ``_ep_local_rank`` buffer."""
    self._ep_rank_holder_ref = weakref.ref(holder)

  def _get_ep_local_rank_tensor(self, x: torch.Tensor) -> torch.Tensor:
    assert self.ep_mesh is not None
    holder = (
        self._ep_rank_holder_ref()
        if self._ep_rank_holder_ref is not None
        else None
    )
    if holder is not None:
      rank_buf = getattr(holder, "_ep_local_rank", None)
      if (
          isinstance(rank_buf, torch.Tensor)
          and rank_buf.device == x.device
          and isinstance(rank_buf, torch._subclasses.FakeTensor)
          == isinstance(x, torch._subclasses.FakeTensor)
      ):
        return rank_buf

    local_rank = int(self.ep_mesh.get_local_rank())
    if get_proxy_mode() is not None or isinstance(
        x, torch._subclasses.FakeTensor
    ):
      return torch.full((1,), local_rank, dtype=torch.long, device=x.device)

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
    if self.ep_mesh is not None and pg is not None:
      return _wait(
          all_gather_single(local_counts_E, gather_dim=0, group=self.ep_mesh)
      )
    if ep_size == 1:
      return local_counts_E
    raise NotImplementedError(
        f"Gathering expert counts over {ep_size} EP ranks needs a process group."
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
      AllToAllDispatchMetadata | LocalDispatchMetadata | TpuRaggedDispatchMetadata,
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
          "ragged_all_to_all requires TpuAllToAllTokenDispatcher to be bound to a "
          "TpuRoutedExperts owner via bind_ep_rank_holder under EP > 1."
      )

    ep_size = self.ep_mesh.size()
    e = self.num_experts // ep_size
    (
        routed_input_ND,
        token_indices_experts_sorted_N,
        topk_scores_experts_sorted_N,
    ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

    num_routed = routed_input_ND.shape[0]
    r_static = ep_size * max(num_routed, 1)

    with maybe_set_sparse_mesh():
      pg = self.ep_mesh.get_group()
      with torch.no_grad():
        with annotate_region(MOE_ROUTING):
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
        write_off_DES = (
            recv_flat_D_eS.cumsum(dim=-1, dtype=torch.int32) - recv_flat_D_eS
        ).view(ep_size, e, ep_size)

        rank_idx = self._get_ep_local_rank_tensor(x_TD)
        disp_input_offsets = (
            torch.index_select(all_send_offsets_SE, 0, rank_idx)
            .reshape(-1)
            .contiguous()
        )
        disp_send_sizes = (
            torch.index_select(all_sizes_SE, 0, rank_idx)
            .reshape(-1)
            .contiguous()
        )
        disp_output_offsets = (
            torch.index_select(write_off_DES, 2, rank_idx)
            .reshape(-1)
            .contiguous()
        )
        disp_recv_sizes = (
            torch.index_select(counts_SDE, 1, rank_idx).reshape(-1).contiguous()
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
    )
    return routed_input_RD, num_global_tokens_per_local_expert_e, metadata


  @annotate_fn({"EP": "combine"})
  def combine(
      self,
      routed_output_RD: torch.Tensor,
      metadata: (
          AllToAllDispatchMetadata
          | LocalDispatchMetadata
          | TpuRaggedDispatchMetadata
      ),
      x_TD: torch.Tensor,
  ) -> torch.Tensor:
    if self.ep_mesh is not None and self.ep_transport == "all_to_all":
      return super().combine(routed_output_RD, metadata, x_TD)

    if self.ep_mesh is None:
      # EP=1: the rows are already this rank's own routes.
      assert isinstance(metadata, LocalDispatchMetadata)
      routed_output_ND = routed_output_RD
    else:
      # Ragged EP: return each route's row to its source rank first. The
      # returned rows are in the same expert-sorted order as local dispatch,
      # so the same fixed-K reduction applies.
      assert isinstance(metadata, TpuRaggedDispatchMetadata)
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
  instead of each W2 output row after it.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(RoutedExperts.Config):
    scale_before_down_proj: bool = False

  def __init__(self, config: Config):
    super().__init__(config)
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
    # Under EP > 1 the routing scores stay on the source rank while W2 runs on
    # the expert's rank after dispatch, so the rows can't be scaled before W2.
    # Refuse the combination when the model is parallelized, before any step.
    if (
        self.scale_before_down_proj
        and parallel_dims.get_optional_mesh("ep") is not None
    ):
      raise ValueError(
          "scale_before_down_proj requires expert_parallel_degree=1; disable"
          " it to run expert parallelism"
      )
    super().parallelize(parallel_dims)
    ep_mesh = parallel_dims.get_optional_mesh("ep")
    if ep_mesh is not None and callable(getattr(ep_mesh, "get_local_rank", None)):
      dispatcher = getattr(self, "token_dispatcher", None)
      if getattr(dispatcher, "ep_transport", None) == "ragged_all_to_all":
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

  def forward(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      topk_expert_ids_TK: torch.Tensor,
      num_local_tokens_per_expert_E: torch.Tensor,
  ) -> torch.Tensor:
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
    # AllToAllDispatchMetadata subclasses LocalDispatchMetadata, so check for
    # it explicitly: the local reduce below has no return all-to-all.
    if not isinstance(metadata, LocalDispatchMetadata) or isinstance(
        metadata, AllToAllDispatchMetadata
    ):
      raise ValueError("scale_before_down_proj requires EP=1 local dispatch")
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
    "TpuAllToAllTokenDispatcher",
    "TpuRaggedDispatchMetadata",
    "TpuRoutedExperts",
    "TpuRoutedExpertsConverter",
    "TpuTokenDispatcherConverter",
    "local_reorder",
    "scale_route_weights",
    "stable_grouped_reduce",
]
