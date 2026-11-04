"""Dropless static-shape expert-parallel token dispatcher for Qwen3.5 MoE on TPU.

``TpuQwen35TokenDispatcher`` dispatches and combines tokens across EP ranks
without dropping tokens or syncing shapes to the host: it gathers the per-expert
token counts with ``torch.tpu.ops.all_gather_dim`` and exchanges only the
routed token slices with ``torch.tpu.ops.ragged_all_to_all`` (offloaded to
SparseCore on Ironwood TPU7x) into a static ``(ep_size * max(N, 1), D)``
expert-major buffer.
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
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.experiments.tpu.remat_regions import (
    MOE_ROUTING,
    annotate_region,
)
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    scale_route_weights,
    stable_grouped_reduce,
)
from torchtitan.models.common.token_dispatcher import (
    AllToAllDispatchMetadata,
    LocalDispatchMetadata,
)


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

  return output


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
  """Dropless static-shape EP dispatch metadata for ``TpuQwen35TokenDispatcher``."""

  routed_rows: int
  disp_input_offsets: torch.Tensor
  disp_send_sizes: torch.Tensor
  disp_output_offsets: torch.Tensor
  disp_recv_sizes: torch.Tensor
  comb_input_offsets: torch.Tensor
  comb_send_sizes: torch.Tensor
  comb_output_offsets: torch.Tensor
  comb_recv_sizes: torch.Tensor


class TpuQwen35TokenDispatcher(TpuAllToAllTokenDispatcher):
  """Dropless token dispatcher with static-shape ``ragged_all_to_all`` EP.

  Upstream ``AllToAllTokenDispatcher`` copies per-rank token splits to the host
  (``.to("cpu").tolist()``) for a variable-shape ``all_to_all_single``, which
  ``TpuGraphTrainer``'s static-shape ``make_fx`` / ``StaticCompiler`` path
  cannot trace.

  With ``ep_mesh is None`` (EP = 1) this is ``TpuAllToAllTokenDispatcher``.
  With EP > 1:

  1. Each rank reorders its ``N = T * K`` routed tokens into global-expert
     order ``0 .. E - 1`` (where ``E = ep_size * e``) and all-gathers the
     ``int32[E]`` per-expert token counts across ``ep_mesh`` via
     ``torch.tpu.ops.all_gather_dim`` into ``all_sizes_SE`` (``int32[ep_size,
     E]``).
  2. From ``all_sizes_SE`` and a 1-element device rank buffer (registered as a
     non-persistent buffer on ``TpuQwen35Model`` so every rank compiles the
     exact same HLO), each rank derives ``E``-element ``int32`` metadata vectors
     (``num_updates_per_replica = e``) that scatter every ``(src, dst,
     local_e)`` slice directly into contiguous expert-major order
     ``[0 : total_valid]`` inside a static zero-initialized
     ``(ep_size * max(N, 1), D)`` buffer via
     ``torch.tpu.ops.ragged_all_to_all``.
  3. ``num_global_tokens_per_local_expert_e`` holds the exact unpadded local
     expert counts (``sum == total_valid <= ep_size * N``), so Pallas MegaBlox
     ``gmm`` / ``tgmm`` executes only active ``tile_m`` tiles (~``1x`` FLOPs)
     and SparseCore transfers only the ``total_valid`` routed tokens (~``1x``
     ICI bytes), with zero dropped tokens regardless of routing skew.
  4. ``combine`` executes the symmetric ``ragged_all_to_all`` directly back into
     ``routed_output_ND`` (``[0 : N]``), followed by ``scale_route_weights`` and
     ``stable_grouped_reduce``.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TpuAllToAllTokenDispatcher.Config):
    pass

  def __init__(self, config: Config):
    super().__init__(config)
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
    return local_counts_E.repeat(ep_size)

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
