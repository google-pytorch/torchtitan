# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure-PyTorch Ring-of-Experts reference for tests.

Production Ring-of-Experts runs only on the Pallas TensorCore kernels in
``kernels/ragged_gather_tc.py``. ``ReferenceRoETokenDispatcher`` swaps in a
pure-PyTorch dispatch and combine with the same semantics, so CPU (gloo) tests
can exercise everything around them (transport selection, capacity and
dropped routes, routing weights before W2, chunking) and TPU tests can compare
the kernels against it. It is not a fallback: production code never builds it.
"""

from dataclasses import dataclass

import torch

from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.experiments.tpu.components import token_dispatcher as td
from torchtitan.experiments.tpu.components.remat_regions import (
    annotate_region,
    MOE_EP_DISPATCH,
    MOE_ROUTING,
)


@dataclass(frozen=True, kw_only=True)
class RoEReferenceDispatchMetadata(td.TpuEpDispatchMetadata):
  """Metadata of the reference Ring-of-Experts dispatch.

  ``token_indices_experts_sorted_N`` and ``topk_scores_experts_sorted_N`` hold
  every global route, local routes first in expert order; ``local_mask_sorted_M``
  marks the local routes kept within the capacity.
  """

  topk_scores_experts_sorted_N: torch.Tensor
  num_global_tokens: int
  local_mask_sorted_M: torch.Tensor


class ReferenceRoETokenDispatcher(td.TpuAllToAllTokenDispatcher):
  """``TpuAllToAllTokenDispatcher`` with a pure-PyTorch Ring-of-Experts.

  Runs on any device. Capacity uses ``_ep_buffer_capacity`` without the
  Pallas kernel's 1024-row alignment, so small test shapes can overflow.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(td.TpuAllToAllTokenDispatcher.Config):
    pass

  def ring_of_experts_unsupported_reason(
      self, device_type: str, hidden_dim: int | None = None
  ) -> str | None:
    del device_type, hidden_dim
    return None

  def _dispatch_ring_of_experts(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      topk_expert_ids_TK: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, RoEReferenceDispatchMetadata]:
    assert self.ep_mesh is not None
    ep_size = self.ep_mesh.size()
    e = self.num_experts // ep_size
    num_tokens, hidden_dim = x_TD.shape
    g_tokens = ep_size * num_tokens
    num_routed = num_tokens * self.top_k
    worst_case_cap = ep_size * max(num_routed, 1)
    cap = self._ep_buffer_capacity(num_routed, ep_size)
    is_capped = cap < worst_case_cap

    with maybe_set_sparse_mesh():
      pg = self.ep_mesh.get_group()
      with torch.no_grad():
        rank_idx_i32 = self._get_ep_local_rank_tensor(x_TD).to(torch.int32)
        ids_i32_TK = topk_expert_ids_TK.to(torch.int32).contiguous()
        with annotate_region(MOE_ROUTING):
          # Looked up on the module so tests can patch the collective.
          ids_all_GK = td._ep_all_gather_impl(ids_i32_TK, self.ep_mesh, pg)
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
        if cap < ids_flat_M.shape[0]:
          within_cap_M = (
              torch.arange(ids_flat_M.shape[0], device=perm_M.device) < cap
          )
          local_mask_sorted_M = local_mask_sorted_M & within_cap_M
        all_counts_ep1 = torch.histc(
            sort_key_M.float(),
            bins=e + 1,
            min=0,
            max=e + 1,
        ).to(torch.int32)
        unclamped_local_counts_e = all_counts_ep1[:e]
        unclamped_count = unclamped_local_counts_e.sum(dtype=torch.int32)
        if is_capped and e > 0:
          cum_e = unclamped_local_counts_e.cumsum(dim=0, dtype=torch.int32)
          clamped_cum_e = torch.clamp(cum_e, max=cap)
          prev_clamped_e = torch.nn.functional.pad(clamped_cum_e[:-1], (1, 0))
          num_global_tokens_per_local_expert_e = clamped_cum_e - prev_clamped_e
          valid_count = clamped_cum_e[-1]
          dropped_count = unclamped_count - valid_count
        else:
          num_global_tokens_per_local_expert_e = unclamped_local_counts_e
          valid_count = unclamped_count
          dropped_count = valid_count.new_zeros(())
        self._record_dropped_count(x_TD, dropped_count)

      with annotate_region(MOE_EP_DISPATCH):
        x_all_GD = td._EpAllGather.apply(x_TD, self.ep_mesh, pg)
        scores_all_GK = td._EpAllGather.apply(topk_scores_TK, self.ep_mesh, pg)

      if g_tokens == 0:
        routed_input_RD = x_TD.new_zeros((cap, hidden_dim))
        token_indices_M = perm_M
        scores_sorted_M = scores_all_GK.reshape(-1)
        row_scale_R = (
            x_TD.new_zeros((cap,), dtype=scores_sorted_M.dtype)
            if self._scale_before_down_proj
            else None
        )
      else:
        routed_input_MD, token_indices_M, scores_sorted_M = (
            td._LocalReorder.apply(
                x_all_GD,
                scores_all_GK,
                perm_M,
                self.top_k,
            )
        )
        zero_x = torch.zeros(
            (), dtype=routed_input_MD.dtype, device=routed_input_MD.device
        )
        zero_s = torch.zeros(
            (), dtype=scores_sorted_M.dtype, device=scores_sorted_M.device
        )
        scores_sorted_M = torch.where(
            local_mask_sorted_M, scores_sorted_M, zero_s
        )
        routed_input_RD = torch.where(
            local_mask_sorted_M[:cap].unsqueeze(-1),
            routed_input_MD[:cap],
            zero_x,
        )
        row_scale_R = (
            scores_sorted_M[:cap] if self._scale_before_down_proj else None
        )

      metadata = RoEReferenceDispatchMetadata(
          token_indices_experts_sorted_N=token_indices_M,
          valid_count=valid_count,
          dropped_count=dropped_count,
          row_scale_R=row_scale_R,
          topk_scores_experts_sorted_N=scores_sorted_M,
          num_global_tokens=g_tokens,
          local_mask_sorted_M=local_mask_sorted_M,
      )
      return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

  def _combine_ring_of_experts(
      self,
      routed_output_RD: torch.Tensor,
      metadata: td.TpuEpDispatchMetadata,
      x_TD: torch.Tensor,
  ) -> torch.Tensor:
    assert self.ep_mesh is not None
    assert isinstance(metadata, RoEReferenceDispatchMetadata), type(metadata)
    if metadata.num_global_tokens == 0:
      return x_TD.new_zeros(x_TD.shape)

    with maybe_set_sparse_mesh():
      pg = self.ep_mesh.get_group()
      total_slots = metadata.local_mask_sorted_M.shape[0]
      if routed_output_RD.shape[0] < total_slots:
        routed_output_MD = torch.nn.functional.pad(
            routed_output_RD, (0, 0, 0, total_slots - routed_output_RD.shape[0])
        )
      else:
        routed_output_MD = routed_output_RD[:total_slots]
      zero_y = torch.zeros(
          (), dtype=routed_output_MD.dtype, device=routed_output_MD.device
      )
      masked_output_MD = torch.where(
          metadata.local_mask_sorted_M.unsqueeze(-1), routed_output_MD, zero_y
      )
      if metadata.row_scale_applied:
        weighted_MD = masked_output_MD
      elif self.fuse_route_score_backward:
        weighted_MD = td.scale_route_weights(
            masked_output_MD,
            metadata.topk_scores_experts_sorted_N,
        )
      else:
        weighted_MD = (
            masked_output_MD.float()
            * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
        ).to(masked_output_MD.dtype)
      partial_GD = td.stable_grouped_reduce(
          weighted_MD,
          metadata.token_indices_experts_sorted_N,
          self.top_k,
      )
      return td._EpReduceScatter.apply(partial_GD, self.ep_mesh, pg)


__all__ = ["ReferenceRoETokenDispatcher", "RoEReferenceDispatchMetadata"]
