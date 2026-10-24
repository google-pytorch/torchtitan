"""Static-shape expert-parallel token dispatcher for Qwen3.5 MoE on TPU.

``TpuQwen35TokenDispatcher`` dispatches and combines tokens across EP ranks with
uniform all-to-alls over static ``(ep_size, cap, D)`` buffers, so
``TpuGraphTrainer`` can trace EP without host syncs.
"""

from dataclasses import dataclass
import math

import spmd_types as spmd
import torch
from torch.autograd.function import once_differentiable
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.fx.traceback import annotate_fn
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    scale_route_weights,
    stable_grouped_reduce,
)
from torchtitan.models.common.token_dispatcher import (
    AllToAllDispatchMetadata,
    LocalDispatchMetadata,
)


class _MaskedRowGatherWithInverse(torch.autograd.Function):
  """Row gather ``x[fwd_idx] * fwd_mask`` whose backward is a gather too.

  ``bwd_idx`` must invert ``fwd_idx`` on the rows where the masks are set, so
  the input gradient ``(dy * fwd_mask)[bwd_idx] * bwd_mask`` needs no
  scatter-add.
  """

  @staticmethod
  def forward(
      ctx,
      x_MD: torch.Tensor,
      fwd_idx_K: torch.Tensor,
      fwd_mask_K1: torch.Tensor,
      bwd_idx_M: torch.Tensor,
      bwd_mask_M1: torch.Tensor,
  ) -> torch.Tensor:
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(fwd_mask_K1, bwd_idx_M, bwd_mask_M1)
    if x_MD.shape[0] == 0 or fwd_idx_K.shape[0] == 0:
      return x_MD.new_zeros((fwd_idx_K.shape[0], x_MD.shape[1]))
    gathered = torch.index_select(x_MD, 0, fwd_idx_K)
    return gathered * fwd_mask_K1.to(gathered.dtype)

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_out_KD: torch.Tensor | None):
    if grad_out_KD is None or not ctx.needs_input_grad[0]:
      return None, None, None, None, None
    fwd_mask_K1, bwd_idx_M, bwd_mask_M1 = ctx.saved_tensors
    if grad_out_KD.shape[0] == 0 or bwd_idx_M.shape[0] == 0:
      return (
          grad_out_KD.new_zeros((bwd_idx_M.shape[0], grad_out_KD.shape[1])),
          None,
          None,
          None,
          None,
      )
    masked_grad_KD = grad_out_KD * fwd_mask_K1.to(grad_out_KD.dtype)
    grad_x_MD = torch.index_select(masked_grad_KD, 0, bwd_idx_M)
    grad_x_MD = grad_x_MD * bwd_mask_M1.to(grad_x_MD.dtype)
    return grad_x_MD, None, None, None, None


def _wait(tensor: torch.Tensor) -> torch.Tensor:
  """Waits on a functional collective's result.

  GraphTrainer's graph splitter requires a collective's ``wait_tensor`` in the
  same executable as the collective.

  Args:
    tensor: A collective's result, possibly an ``AsyncCollectiveTensor``.

  Returns:
    The plain tensor.
  """
  if isinstance(tensor, AsyncCollectiveTensor):
    return tensor.wait()
  return tensor


@dataclass(frozen=True, kw_only=True)
class TpuQwen35DispatchMetadata(LocalDispatchMetadata):
  """Static-shape EP dispatch metadata for ``TpuQwen35TokenDispatcher``."""

  send_flat_idx: torch.Tensor
  send_valid_flat: torch.Tensor
  send_slot_N: torch.Tensor  # pylint: disable=invalid-name
  send_token_valid_N: torch.Tensor  # pylint: disable=invalid-name
  perm_idx_R: torch.Tensor  # pylint: disable=invalid-name
  expert_valid_flat: torch.Tensor
  inv_perm_idx_R: torch.Tensor  # pylint: disable=invalid-name
  recv_valid_flat: torch.Tensor


class TpuQwen35TokenDispatcher(TpuAllToAllTokenDispatcher):
  """Token dispatcher with static-shape EP > 1 dispatch and combine.

  Upstream ``AllToAllTokenDispatcher`` copies the per-rank token splits to the
  host (``.to("cpu").tolist()``) for a variable-size all-to-all, which
  ``TpuGraphTrainer``'s static-shape ``make_fx`` / ``StaticCompiler`` path
  cannot trace.

  With ``ep_mesh is None`` (EP = 1) this is ``TpuAllToAllTokenDispatcher``.
  With EP > 1, each rank packs its ``N = T * K`` expert-sorted routed tokens
  into a static ``(ep_size, cap, D)`` send buffer, ``cap`` tokens per
  destination rank, and exchanges token counts and buffers with uniform
  all-to-alls. The receiver permutes the received tokens into expert-major
  order in a static ``(ep_size * cap, D)`` buffer whose trailing zero rows
  are assigned to the last local expert. The permutations are computed with
  int32 compares and ``argsort`` (no host syncs, no ``searchsorted``), and
  their backward passes are gathers (``_MaskedRowGatherWithInverse``).

  ``capacity_factor=None`` is dropless (``cap = N``, an ``ep_size``-times larger
  buffer than the routed tokens). Otherwise ``cap = ceil(capacity_factor * N /
  ep_size)``: a rank's tokens beyond ``cap`` for one destination rank are
  dropped (zero expert output), unlike the dropless EP = 1 path.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TpuAllToAllTokenDispatcher.Config):
    capacity_factor: float | None = None

  def __init__(self, config: Config):
    super().__init__(config)
    self.capacity_factor = config.capacity_factor

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
      AllToAllDispatchMetadata | LocalDispatchMetadata,
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

    if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
      spmd.mutate_type(
          num_local_tokens_per_expert_E,
          src=spmd.P,
          dst={"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
      )

    with maybe_set_sparse_mesh():
      pg = (
          "ep"
          if get_spmd_backend() == "spmd_types"
          else self.ep_mesh.get_group()
      )
      if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
        num_local_tokens_per_expert_E = spmd.reinterpret_mesh(
            num_local_tokens_per_expert_E, spmd.current_mesh()
        )
        routed_input_ND = spmd.reinterpret_mesh(
            routed_input_ND, spmd.current_mesh()
        )

      num_routed = routed_input_ND.shape[0]
      if self.capacity_factor is None:
        cap = max(num_routed, 1)
      else:
        cap = max(
            1,
            min(
                max(num_routed, 1),
                math.ceil(self.capacity_factor * num_routed / ep_size),
            ),
        )
      r_static = ep_size * cap
      device = x_TD.device

      with torch.no_grad():
        local_counts_EP_e = num_local_tokens_per_expert_E.to(
            torch.int32
        ).reshape(ep_size, e)
        if self.capacity_factor is not None:
          local_cum_e = local_counts_EP_e.cumsum(dim=1).clamp(max=cap)
          local_counts_capped_EP_e = torch.cat(
              [local_cum_e[:, :1], local_cum_e[:, 1:] - local_cum_e[:, :-1]],
              dim=1,
          )
        else:
          local_counts_capped_EP_e = local_counts_EP_e

        send_counts_EP = local_counts_EP_e.sum(dim=1)
        send_counts_capped_EP = local_counts_capped_EP_e.sum(dim=1)
        send_ends_EP = send_counts_EP.cumsum(dim=0)
        send_starts_EP = send_ends_EP - send_counts_EP

        slot_cap = torch.arange(cap, device=device, dtype=torch.int32)
        send_valid_EP_cap = slot_cap.unsqueeze(
            0
        ) < send_counts_capped_EP.unsqueeze(1)
        send_src_idx_EP_cap = (
            send_starts_EP.unsqueeze(1) + slot_cap.unsqueeze(0)
        ).clamp(min=0, max=max(num_routed - 1, 0))
        send_flat_idx = send_src_idx_EP_cap.reshape(-1).to(torch.long)
        send_valid_flat = send_valid_EP_cap.reshape(-1, 1)

        idx_N = torch.arange(num_routed, device=device, dtype=torch.int32)
        after_rank_N_EP = idx_N.unsqueeze(1) >= send_ends_EP.unsqueeze(0)
        dst_rank_N = (
            after_rank_N_EP.to(torch.int32)
            .sum(dim=1)
            .clamp(max=max(ep_size - 1, 0))
        )
        send_start_N = (
            after_rank_N_EP.to(torch.int32) * send_counts_EP.unsqueeze(0)
        ).sum(dim=1)
        pos_in_rank_N = idx_N - send_start_N
        send_token_valid_N = (pos_in_rank_N < cap).unsqueeze(1)
        send_slot_N = (
            dst_rank_N * cap + pos_in_rank_N.clamp(min=0, max=max(cap - 1, 0))
        ).to(torch.long)

        recv_counts_EP_e = _wait(
            self._token_count_exchange(
                local_counts_capped_EP_e.reshape(-1),
                pg,
                ep_size,
            )
        )
        t_mat = recv_counts_EP_e.to(torch.int32).reshape(ep_size, e)

        rank_local_ends_EP_e = t_mat.cumsum(dim=1)
        recv_valid_EP_cap = slot_cap.unsqueeze(0) < rank_local_ends_EP_e[:, -1:]
        recv_valid_flat = recv_valid_EP_cap.reshape(-1, 1)

        slot_expert_R = (
            (slot_cap.view(1, cap, 1) >= rank_local_ends_EP_e.unsqueeze(1))
            .to(torch.int32)
            .sum(dim=-1)
            .reshape(-1)
        )
        perm_idx_R = torch.argsort(slot_expert_R, stable=True)
        inv_perm_idx_R = torch.argsort(perm_idx_R)

        local_expert_counts_e = t_mat.sum(dim=0)
        total_valid = local_expert_counts_e.sum()
        q_R = torch.arange(r_static, device=device, dtype=torch.int32)
        expert_valid_flat = (q_R < total_valid).unsqueeze(1)
        pad_rows = r_static - total_valid
        num_global_tokens_per_local_expert_e = torch.cat([
            local_expert_counts_e[:-1],
            local_expert_counts_e[-1:] + pad_rows,
        ])

      send_buf_flat = _MaskedRowGatherWithInverse.apply(
          routed_input_ND,
          send_flat_idx,
          send_valid_flat,
          send_slot_N,
          send_token_valid_N,
      )
      # Uniform splits: every rank sends `cap` rows to every rank.
      recv_buf_flat = _wait(
          self._dispatch_token_exchange(send_buf_flat, pg, None, None)
      )
      routed_input_RD = _MaskedRowGatherWithInverse.apply(
          recv_buf_flat,
          perm_idx_R,
          expert_valid_flat,
          inv_perm_idx_R,
          recv_valid_flat,
      )

    metadata = TpuQwen35DispatchMetadata(
        token_indices_experts_sorted_N=token_indices_experts_sorted_N,
        topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
        send_flat_idx=send_flat_idx,
        send_valid_flat=send_valid_flat,
        send_slot_N=send_slot_N,
        send_token_valid_N=send_token_valid_N,
        perm_idx_R=perm_idx_R,
        expert_valid_flat=expert_valid_flat,
        inv_perm_idx_R=inv_perm_idx_R,
        recv_valid_flat=recv_valid_flat,
    )
    return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

  @annotate_fn({"EP": "combine"})
  def combine(
      self,
      routed_output_RD: torch.Tensor,
      metadata: AllToAllDispatchMetadata | LocalDispatchMetadata,
      x_TD: torch.Tensor,
  ) -> torch.Tensor:
    if self.ep_mesh is None:
      return super().combine(routed_output_RD, metadata, x_TD)
    assert isinstance(metadata, TpuQwen35DispatchMetadata)

    with maybe_set_sparse_mesh():
      pg = (
          "ep"
          if get_spmd_backend() == "spmd_types"
          else self.ep_mesh.get_group()
      )
      recv_back_flat = _MaskedRowGatherWithInverse.apply(
          routed_output_RD,
          metadata.inv_perm_idx_R,
          metadata.recv_valid_flat,
          metadata.perm_idx_R,
          metadata.expert_valid_flat,
      )
      send_back_flat = _wait(
          self._combine_token_exchange(recv_back_flat, pg, None, None)
      )
      routed_output_ND = _MaskedRowGatherWithInverse.apply(
          send_back_flat,
          metadata.send_slot_N,
          metadata.send_token_valid_N,
          metadata.send_flat_idx,
          metadata.send_valid_flat,
      )

    if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
      routed_output_ND = spmd.reinterpret_mesh(
          routed_output_ND, spmd.current_mesh()
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


# Per-destination-rank token capacity of `TpuQwen35TokenDispatcher` under EP.
EP_CAPACITY_FACTOR = 1.25
