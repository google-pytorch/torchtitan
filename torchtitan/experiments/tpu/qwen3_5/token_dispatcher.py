"""Dropless static-shape expert-parallel token dispatchers for Qwen3.5 MoE on TPU.

``TpuQwen35TokenDispatcher`` dispatches and combines tokens across EP ranks
without dropping tokens or syncing shapes to the host. It supports two dropless
transport modes:

1. ``"ring_of_experts"`` (default via ``"auto"`` when ``ep_size <= top_k``,
   adapted from MaxText's MLPerf Training v6.1 Ironwood TPU7x MoE submissions):
   all-gathers ``(x_TD, topk_scores_TK, topk_expert_ids_TK)`` across ``ep_mesh``
   via SparseCore ``torch.tpu.ops.all_gather_dim``, gathers only this rank's
   ``count`` local-expert tokens via Pallas TensorCore async DMA
   (``roe_sort_gather``), fuses route-weight scaling + FP32 token reduction in
   VMEM (``roe_unsort_combine``), and sums partial outputs across ``ep_mesh``
   via SparseCore ``torch.tpu.ops.reduce_scatter_dim``. When ``ep_size <= K``,
   this cuts activation ICI traffic by up to ``K / (ep_size - 1)`` (``2.67x``
   at ``EP=4, K=8``, ``1.14x`` at ``EP=8, K=8``) and eliminates
   ``ragged_all_to_all`` gather/scatter stalls.
2. ``"ragged_all_to_all"`` (default via ``"auto"`` when ``ep_size > top_k``):
   gathers ``int32[E]`` per-expert token counts via ``all_gather_dim`` and
   exchanges only the routed token slices via
   ``torch.tpu.ops.ragged_all_to_all``.

``TpuQwen35RoutedExperts`` optionally pipelines ``dispatch -> experts ->
combine``
across ``num_token_chunks`` sequence chunks while unsharding FSDP expert weights
once per layer, cutting peak static ``(cap, D)`` HBM buffers by
``num_token_chunks``
and overlapping chunk ``c`` MXU expert GEMMs with chunk ``c + 1`` SparseCore
collectives.
"""

import dataclasses
from dataclasses import dataclass
from typing import get_args, Literal
import weakref

import torch
from torch.autograd.function import once_differentiable
from torch.distributed._functional_collectives import (
    AsyncCollectiveTensor,
    all_gather_single,
    reduce_scatter_single,
)
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.traceback import annotate, annotate_fn
import torchtitan.config
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.experiments.tpu.remat_regions import (
    MOE_EP_DISPATCH,
    MOE_ROUTING,
    annotate_region,
)
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRoutedExperts,
    _LocalReorder,
    scale_route_weights,
    stable_grouped_reduce,
)
from torchtitan.models.common.moe import _tokens_per_expert, RoutedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllDispatchMetadata,
    AllToAllTokenDispatcher,
    LocalDispatchMetadata,
)
from torchtitan.tools.logging import logger

CommMode = Literal["auto", "ring_of_experts", "ragged_all_to_all"]


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
  """All-gathers ``operand`` along dim 0 over ``ep_mesh``.

  On TPU, uses SparseCore ``torch.tpu.ops.all_gather_dim``. GraphTrainer's
  FSDP passes classify collectives by role (parameter gather / gradient
  reduction), so these activation collectives are left in place; that is also
  why EP needs ``backward_partition_mode="by_layer"``.
  """
  src = operand.detach().contiguous()
  if ep_mesh is None:
    return src
  ep_size = ep_mesh.size()
  if ep_size <= 1:
    return src
  if src.device.type == "tpu" and pg is not None and hasattr(pg, "group_name"):
    return torch.tpu.ops.all_gather_dim(src, 0, group=pg)
  if pg is not None:
    return _wait(all_gather_single(src, gather_dim=0, group=ep_mesh))
  # No process group: single-process test stand-in in which every EP rank
  # holds the same shard.
  return src.repeat(ep_size, *([1] * (src.ndim - 1)))


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
  if src.device.type == "tpu" and pg is not None and hasattr(pg, "group_name"):
    return torch.tpu.ops.reduce_scatter_dim(src, 0, group=pg, reduce_op="sum")
  if pg is not None:
    return _wait(
        reduce_scatter_single(src, reduceOp="sum", scatter_dim=0, group=ep_mesh)
    )
  # No process group: single-process test stand-in (see `_ep_all_gather_impl`).
  chunk = src.shape[0] // ep_size
  return src.view(ep_size, chunk, *src.shape[1:]).sum(dim=0)


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

  if src.device.type == "tpu" and pg is not None and hasattr(pg, "group_name"):
    return torch.tpu.ops.ragged_all_to_all(
        src,
        output,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        pg.group_name,
    )

  if (
      ep_mesh is not None
      and pg is not None
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
class _EpDispatchMetadata:
  """Base class for EP > 1 dispatch metadata (distinct from ``LocalDispatchMetadata``)."""

  token_indices_experts_sorted_N: torch.Tensor
  topk_scores_experts_sorted_N: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class TpuQwen35DispatchMetadata(_EpDispatchMetadata):
  """Dropless ``ragged_all_to_all`` EP dispatch metadata."""

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
class TpuQwen35RoEDispatchMetadata(_EpDispatchMetadata):
  """Dropless Ring-of-Experts (``all_gather`` + ``reduce_scatter``) metadata."""

  use_pallas_tc: bool
  num_global_tokens: int
  local_mask_sorted_M: torch.Tensor | None = None
  scores_flat_M: torch.Tensor | None = None
  window_slot_order_cap: torch.Tensor | None = None
  start_count_2: torch.Tensor | None = None
  src_rows: torch.Tensor | None = None
  tokens: torch.Tensor | None = None
  blocks: torch.Tensor | None = None
  gather_block_size: int = 1024
  reduce_block_size: int = 896
  zero_pad_tail: bool = False
  # FP32 routing weight per ``[cap]`` buffer row (0 past the valid count).
  # Set when ``roe_weights_on_activation``: the routed experts multiply their
  # hidden activation by it before ``w2`` and set ``row_scale_applied`` so the
  # combine is an unweighted gather-reduce.
  expert_row_scale_R: torch.Tensor | None = None
  row_scale_applied: bool = False


class TpuQwen35TokenDispatcher(TpuAllToAllTokenDispatcher):
  """Dropless static-shape token dispatcher supporting Ring-of-Experts and RA2A."""

  @dataclass(kw_only=True, slots=True)
  class Config(TpuAllToAllTokenDispatcher.Config):
    comm_mode: CommMode = "auto"
    roe_use_pallas_tc: bool = True
    roe_gather_block_size: int = 1024
    roe_reduce_block_size: int = 896
    roe_zero_pad_tail: bool = False
    # Apply routing weights on the routed experts' hidden activation (before
    # ``w2``) instead of in the combine, as MaxText's
    # ``moe_tc_ragged_weights_on_activation``: the weight scale and its
    # gradient run over the expert hidden size (e.g. 1024 for 397B-A17B)
    # instead of the model dim (4096), and the combine is an unweighted
    # gather-reduce.
    roe_weights_on_activation: bool = True

  def __init__(self, config: Config):
    super().__init__(config)
    if config.comm_mode not in get_args(CommMode):
      raise ValueError(
          f"comm_mode must be one of {get_args(CommMode)}, got"
          f" {config.comm_mode!r}"
      )
    self.comm_mode: CommMode = config.comm_mode
    self.roe_use_pallas_tc = config.roe_use_pallas_tc
    self.roe_gather_block_size = config.roe_gather_block_size
    self.roe_reduce_block_size = config.roe_reduce_block_size
    self.roe_zero_pad_tail = config.roe_zero_pad_tail
    self.roe_weights_on_activation = config.roe_weights_on_activation
    self._ep_rank_holder_ref: weakref.ReferenceType | None = None
    self._ep_local_rank_buf: torch.Tensor | None = None

  def init_buffer(self) -> None:
    super().init_buffer()
    self._ep_local_rank_buf = None

  def bind_ep_rank_holder(self, holder: torch.nn.Module) -> None:
    """Binds the root module holding the non-persistent ``_ep_local_rank`` buffer."""
    self._ep_rank_holder_ref = weakref.ref(holder)

  def _resolve_comm_mode(self, ep_size: int) -> str:
    if self.comm_mode == "auto":
      return "ring_of_experts" if ep_size <= self.top_k else "ragged_all_to_all"
    return self.comm_mode

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

  def _dispatch_ring_of_experts(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      topk_expert_ids_TK: torch.Tensor,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      TpuQwen35RoEDispatchMetadata,
  ]:
    assert self.ep_mesh is not None
    ep_size = self.ep_mesh.size()
    e = self.num_experts // ep_size
    num_tokens, hidden_dim = x_TD.shape
    g_tokens = ep_size * num_tokens
    num_routed = num_tokens * self.top_k
    cap = ep_size * max(num_routed, 1)

    use_pallas_tc = (
        x_TD.device.type == "tpu"
        and self.roe_use_pallas_tc
        and num_tokens > 0
        and hidden_dim % 128 == 0
    )

    with maybe_set_sparse_mesh():
      pg = self.ep_mesh.get_group()
      if use_pallas_tc:
        # Lazy import so CPU-only environments never import JAX/Pallas at module load.
        # pylint: disable=g-import-not-at-top
        from torchtitan.experiments.tpu.kernels.ragged_gather_tc import (
            roe_compute_routing,
            roe_gather_row_scales,
            roe_sort_gather,
            safe_gather_block_size,
            safe_reduce_block_size,
        )
        # pylint: enable=g-import-not-at-top

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
          # TODO: `roe_compute_routing` (packed sort + gather/reduce metadata)
          # is outside `moe.routing`, so `tpu_selective` recomputes it in
          # backward for every chunk. Saving it trades that compute for HBM,
          # which is tight at 397B 24K tokens/rank; measure before moving it.
          routing_tuple = roe_compute_routing(
              ids_all_GK.reshape(-1),
              rank_idx_i32,
              num_experts=self.num_experts,
              ep_size=ep_size,
              top_k=self.top_k,
              num_out_tokens=g_tokens,
              cap=cap,
              reduce_block_size=reduce_block,
          )
          (
              num_global_tokens_per_local_expert_e,
              token_ids_sorted_M,
              window_slot_order_cap,
              start_count_2,
              src_rows,
              tokens,
              blocks,
          ) = routing_tuple

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
        empty_f32 = topk_scores_TK.new_empty((0,))
        expert_row_scale_R = None
        if self.roe_weights_on_activation:
          expert_row_scale_R = roe_gather_row_scales(
              scores_all_GK.reshape(-1),
              window_slot_order_cap,
              start_count_2,
          )
        metadata = TpuQwen35RoEDispatchMetadata(
            token_indices_experts_sorted_N=token_ids_sorted_M,
            topk_scores_experts_sorted_N=empty_f32,
            use_pallas_tc=True,
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
            expert_row_scale_R=expert_row_scale_R,
        )
        return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

      # Pure-PyTorch Ring-of-Experts path (CPU gloo tests & non-Pallas fallback).
      with torch.no_grad():
        rank_idx_i32 = self._get_ep_local_rank_tensor(x_TD).to(torch.int32)
        ids_i32_TK = topk_expert_ids_TK.to(torch.int32).contiguous()
        with annotate_region(MOE_ROUTING):
          ids_all_GK = _ep_all_gather_impl(ids_i32_TK, self.ep_mesh, pg)
          ids_flat_M = ids_all_GK.reshape(-1)
          rel_ids_M = ids_flat_M - rank_idx_i32 * e
          is_local_M = (rel_ids_M >= 0) & (rel_ids_M < e)
          sort_key_M = torch.where(
              is_local_M,
              rel_ids_M,
              torch.full_like(rel_ids_M, e),
          )
          perm_M = torch.argsort(sort_key_M, stable=True)
        local_mask_sorted_M = torch.index_select(is_local_M, 0, perm_M)
        all_counts_ep1 = torch.histc(
            sort_key_M.float(),
            bins=e + 1,
            min=0,
            max=e + 1,
        ).to(torch.int32)
        num_global_tokens_per_local_expert_e = all_counts_ep1[:e]

      with annotate_region(MOE_EP_DISPATCH):
        x_all_GD = _EpAllGather.apply(x_TD, self.ep_mesh, pg)
        scores_all_GK = _EpAllGather.apply(topk_scores_TK, self.ep_mesh, pg)

      if g_tokens == 0:
        routed_input_RD = x_TD.new_zeros((cap, hidden_dim))
        token_indices_M = perm_M
        scores_sorted_M = scores_all_GK.reshape(-1)
      else:
        routed_input_MD, token_indices_M, scores_sorted_M = _LocalReorder.apply(
            x_all_GD,
            scores_all_GK,
            perm_M,
            self.top_k,
        )
        zero_x = torch.zeros(
            (), dtype=routed_input_MD.dtype, device=routed_input_MD.device
        )
        zero_s = torch.zeros(
            (), dtype=scores_sorted_M.dtype, device=scores_sorted_M.device
        )
        routed_input_RD = torch.where(
            local_mask_sorted_M.unsqueeze(-1), routed_input_MD, zero_x
        )
        scores_sorted_M = torch.where(
            local_mask_sorted_M, scores_sorted_M, zero_s
        )

      metadata = TpuQwen35RoEDispatchMetadata(
          token_indices_experts_sorted_N=token_indices_M,
          topk_scores_experts_sorted_N=scores_sorted_M,
          use_pallas_tc=False,
          num_global_tokens=g_tokens,
          local_mask_sorted_M=local_mask_sorted_M,
      )
      return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

  def _combine_ring_of_experts(
      self,
      routed_output_RD: torch.Tensor,
      metadata: TpuQwen35RoEDispatchMetadata,
      x_TD: torch.Tensor,
  ) -> torch.Tensor:
    assert self.ep_mesh is not None
    if metadata.num_global_tokens == 0:
      return x_TD.new_zeros(x_TD.shape)

    with maybe_set_sparse_mesh():
      pg = self.ep_mesh.get_group()
      if metadata.use_pallas_tc:
        # pylint: disable=g-import-not-at-top
        from torchtitan.experiments.tpu.kernels.ragged_gather_tc import (
            roe_unsort_combine,
            roe_unsort_reduce,
        )
        # pylint: enable=g-import-not-at-top

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
          return _EpReduceScatter.apply(partial_GD, self.ep_mesh, pg)

        assert metadata.scores_flat_M is not None
        assert metadata.window_slot_order_cap is not None
        assert metadata.start_count_2 is not None
        assert metadata.src_rows is not None
        assert metadata.tokens is not None
        assert metadata.blocks is not None
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

      assert metadata.local_mask_sorted_M is not None
      zero_y = torch.zeros(
          (), dtype=routed_output_RD.dtype, device=routed_output_RD.device
      )
      masked_output_MD = torch.where(
          metadata.local_mask_sorted_M.unsqueeze(-1), routed_output_RD, zero_y
      )
      if self.fuse_route_score_backward:
        weighted_MD = scale_route_weights(
            masked_output_MD,
            metadata.topk_scores_experts_sorted_N,
        )
      else:
        weighted_MD = (
            masked_output_MD.float()
            * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
        ).to(masked_output_MD.dtype)
      partial_GD = stable_grouped_reduce(
          weighted_MD,
          metadata.token_indices_experts_sorted_N,
          self.top_k,
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
      AllToAllDispatchMetadata | LocalDispatchMetadata | _EpDispatchMetadata,
  ]:
    if self.ep_mesh is None:
      return super().dispatch(
          x_TD,
          topk_scores_TK,
          topk_expert_ids_TK,
          num_local_tokens_per_expert_E,
      )

    ep_size = self.ep_mesh.size()
    if self._resolve_comm_mode(ep_size) == "ring_of_experts":
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

        # Lay out each destination rank's buffer in expert-major order
        # `[local_e = 0 (src = 0..P-1), ..., local_e = e - 1 (src = 0..P-1)]`.
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

    metadata = TpuQwen35DispatchMetadata(
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
          AllToAllDispatchMetadata | LocalDispatchMetadata | _EpDispatchMetadata
      ),
      x_TD: torch.Tensor,
  ) -> torch.Tensor:
    if self.ep_mesh is None:
      return super().combine(routed_output_RD, metadata, x_TD)
    if isinstance(metadata, TpuQwen35RoEDispatchMetadata):
      return self._combine_ring_of_experts(routed_output_RD, metadata, x_TD)
    assert isinstance(metadata, TpuQwen35DispatchMetadata)

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


class TpuQwen35RoutedExperts(TpuRoutedExperts):
  """Routed experts supporting sequence-chunked EP overlap and single weight unshard."""

  @dataclass(kw_only=True, slots=True)
  class Config(TpuRoutedExperts.Config):
    num_token_chunks: int = 1
    auto_scale_token_chunks: bool = False
    max_inflight_chunks: int = 2

  def __init__(self, config: Config):
    super().__init__(config)
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
    self._warned_chunk_snap: set[tuple[int, int]] = set()

  def forward(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      topk_expert_ids_TK: torch.Tensor,
      num_local_tokens_per_expert_E: torch.Tensor,
  ) -> torch.Tensor:
    num_chunks = self.num_token_chunks
    num_tokens = x_TD.shape[0]
    if num_tokens > 0 and num_tokens % num_chunks != 0:
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
    ep_mesh = getattr(self.token_dispatcher, "ep_mesh", None)
    weights_on_activation = (
        ep_mesh is not None
        and isinstance(self.token_dispatcher, TpuQwen35TokenDispatcher)
        and self.token_dispatcher.roe_weights_on_activation
        and getattr(self.inner_experts, "use_pallas_grouped_mm", False)
    )
    if (
        (num_chunks <= 1 and not weights_on_activation)
        or ep_mesh is None
        or num_tokens == 0
    ):
      return super().forward(
          x_TD,
          topk_scores_TK,
          topk_expert_ids_TK,
          num_local_tokens_per_expert_E,
      )

    if self.scale_before_down_proj:
      raise ValueError(
          "scale_before_down_proj requires EP=1 local dispatch; under EP use"
          " roe_weights_on_activation instead"
      )
    needs_chunk_counts = True
    if isinstance(self.token_dispatcher, TpuQwen35TokenDispatcher):
      needs_chunk_counts = (
          self.token_dispatcher._resolve_comm_mode(ep_mesh.size())  # pylint: disable=protected-access
          != "ring_of_experts"
      )

    # Unshard FSDP expert parameters once outside the chunk loop via
    # ``inner_experts.__call__`` so FX preserves ``module_fqn`` metadata for
    # SimpleFSDP collective bucketing.
    with maybe_set_sparse_mesh():
      unsharded_weights = self.inner_experts(
          x_TD,
          num_local_tokens_per_expert_E,
          unshard_only=True,
      )
    if getattr(self.inner_experts, "use_pallas_grouped_mm", False):
      # The Pallas expert matmuls run in BF16: cast FP32 master weights once
      # here rather than once per chunk.
      unsharded_weights = tuple(w.bfloat16() for w in unsharded_weights)

    x_chunks = x_TD.chunk(num_chunks, dim=0)
    score_chunks = topk_scores_TK.chunk(num_chunks, dim=0)
    id_chunks = topk_expert_ids_TK.chunk(num_chunks, dim=0)
    num_experts = num_local_tokens_per_expert_E.numel()

    out_chunks: list[torch.Tensor] = []
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
        row_scale_c = (
            getattr(metadata_c, "expert_row_scale_R", None)
            if weights_on_activation
            else None
        )
        with maybe_set_sparse_mesh():
          routed_output_c = self.inner_experts(
              routed_input_c,
              num_tokens_per_expert_e,
              row_scale_R=row_scale_c,
              pre_unsharded_weights=unsharded_weights,
          )
        if row_scale_c is not None:
          metadata_c = dataclasses.replace(metadata_c, row_scale_applied=True)
        out_chunks.append(
            self.token_dispatcher.combine(routed_output_c, metadata_c, x_c)
        )
    return torch.cat(out_chunks, dim=0)


def default_num_token_chunks(ep_degree: int) -> int:
  """Default EP sequence chunks per MoE layer: ``ep_degree`` clamped to 4..16."""
  return max(4, min(ep_degree, 16))


def use_ep_modules(
    model_cfg,
) -> tuple[
    list[TpuQwen35RoutedExperts.Config], list[TpuQwen35TokenDispatcher.Config]
]:
  """Swaps in the dropless EP routed experts and token dispatchers, in place.

  Plain and TPU ``RoutedExperts`` / all-to-all dispatcher configs are replaced
  with ``TpuQwen35RoutedExperts`` / ``TpuQwen35TokenDispatcher``; other
  subclasses are left alone (they keep their own dispatch).

  Returns:
    Every ``TpuQwen35RoutedExperts.Config`` and ``TpuQwen35TokenDispatcher.Config``
    in ``model_cfg`` after the swap.
  """
  for cfg_type, new_type, swap_types in (
      (
          RoutedExperts.Config,
          TpuQwen35RoutedExperts.Config,
          (RoutedExperts.Config, TpuRoutedExperts.Config),
      ),
      (
          AllToAllTokenDispatcher.Config,
          TpuQwen35TokenDispatcher.Config,
          (AllToAllTokenDispatcher.Config, TpuAllToAllTokenDispatcher.Config),
      ),
  ):
    for _, sub_cfg, parent, field_name in list(model_cfg.traverse(cfg_type)):
      if type(sub_cfg) not in swap_types:  # pylint: disable=unidiomatic-typecheck
        continue
      new_cfg = torchtitan.config.derive(sub_cfg, new_type)
      if isinstance(parent, list):
        parent[field_name] = new_cfg
      else:
        setattr(parent, field_name, new_cfg)
  routed = [
      cfg
      for _, cfg, _, _ in model_cfg.traverse(TpuQwen35RoutedExperts.Config)
  ]
  dispatchers = [
      cfg
      for _, cfg, _, _ in model_cfg.traverse(TpuQwen35TokenDispatcher.Config)
  ]
  return routed, dispatchers
