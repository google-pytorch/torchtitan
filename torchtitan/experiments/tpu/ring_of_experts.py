# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static-shape Ring-of-Experts expert-parallel (EP) transport for TPU.

Each EP rank all-gathers the *un-expanded* local tokens, routing scores, and
expert ids of its EP group, selects the routes owned by its local experts, and
gathers those rows into an expert-major buffer of static size ``max_R``. After
the expert computation, each rank weights and sums its partial outputs per
token and a reduce-scatter returns every token's sum to its owner. This
replaces the two ragged all-to-alls of the standard dispatcher with an
all-gather and a reduce-scatter, which are cheap on the TPU ICI and need no
dynamic shapes. This follows MaxText's ``ring_of_experts`` MoE strategy.

Routes beyond a rank's capacity ``max_R`` are dropped, in local-expert order
and within an expert in global route order. Rows past the valid count are zero
and belong to no expert: the group sizes sum to the valid count, so the
grouped matmuls skip those row tiles instead of computing the capacity
padding, and may leave their outputs unspecified. The combine and its backward
read only the valid rows and mask with selects, so those rows never reach a
result.

The forward and backward of each step are custom autograd functions so that
every token-sized gather and reduction runs in the TensorCore kernels of
``kernels/ragged_gather.py`` and ``kernels/ragged_gather_reduce.py``. Nothing
here depends on the dispatcher class.
"""

from dataclasses import dataclass
import math

import torch
from torch.autograd.function import once_differentiable
import torch.distributed._functional_collectives as funcol

from torchtitan.experiments.tpu.kernels.ragged_gather import (
    inverse_permute_scatter,
    ragged_gather,
    to_token_3d,
)
from torchtitan.experiments.tpu.kernels.ragged_gather_reduce import (
    ragged_gather_reduce,
    ragged_gather_reduce_metadata,
)
from torchtitan.experiments.tpu.kernels.route_score import route_score_grad
from torchtitan.experiments.tpu.remat_regions import (
    annotate_region,
    MOE_ROUTING,
)

# The per-rank row capacity is rounded up to a multiple of the grouped-matmul
# row tile.
CAPACITY_TILE = 256


def ring_of_experts_capacity(
    num_routed_tokens: int, ep_size: int, capacity_factor: float | None
) -> int:
  """Static number of expert rows `max_R` per EP rank.

  Args:
    num_routed_tokens: local routes, `T * top_k`.
    ep_size: EP degree.
    capacity_factor: rows per rank as a multiple of `num_routed_tokens`, or
      `None` for drop-free (`ep_size * num_routed_tokens`).

  Returns:
    `max_R`, a multiple of `CAPACITY_TILE`.
  """
  max_rows = ep_size * num_routed_tokens
  if capacity_factor is not None:
    max_rows = min(max_rows, math.ceil(capacity_factor * num_routed_tokens))
  return -(-max(max_rows, 1) // CAPACITY_TILE) * CAPACITY_TILE


@dataclass(frozen=True, slots=True)
class RingOfExpertsMetadata:
  """Dispatch state reused by `ring_of_experts_combine`."""

  token_ids_R: torch.Tensor
  valid_count_1: torch.Tensor
  scores_R: torch.Tensor
  reduce_metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
  num_out_tokens: int
  top_k: int
  ep_mesh: object


def _exclusive_cumsum_int32(x: torch.Tensor) -> torch.Tensor:
  """Exclusive int32 cumsum of a 1-D tensor without `torch.cumsum`.

  `torch.cumsum` lowers to an XLA `kScan`, which XLA mishandles when the
  enclosing while body (the scanned layer stack) is cloned. A log-depth
  Hillis-Steele scan lowers to slices, adds, and concatenates only.
  """
  x = x.to(torch.int32)
  out = x
  step = 1
  while step < out.shape[0]:
    out = torch.cat([out[:step], out[step:] + out[:-step]])
    step *= 2
  return out - x


def _packed_stable_argsort(keys: torch.Tensor, max_key: int) -> torch.Tensor:
  """Stable int32 argsort of non-negative `keys <= max_key`.

  Packs `(key, position)` into one int32 so an unstable single-operand sort is
  stable, which is much cheaper on TPU than a stable key-value sort.
  """
  n = int(keys.shape[0])
  if n <= 1:
    return torch.zeros_like(keys, dtype=torch.int32)
  pos_bits = (n - 1).bit_length()
  if pos_bits + max(1, int(max_key).bit_length()) > 31:
    return torch.argsort(keys, stable=True).to(torch.int32)
  positions = torch.arange(n, device=keys.device, dtype=torch.int32)
  packed = (keys.to(torch.int32) << pos_bits) | positions
  return torch.sort(packed).values & ((1 << pos_bits) - 1)


class _GatherTokens(torch.autograd.Function):
  """EP all-gather of local tokens, then gather this rank's expert rows.

  Backward is the transpose: an FP32 per-token sum of the row gradients, then
  a reduce-scatter back to the token owners.
  """

  @staticmethod
  def forward(ctx, x_TD, token_ids_R, valid_count_1, reduce_metadata, top_k,
              ep_mesh):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(valid_count_1, *reduce_metadata)
    ctx.top_k = top_k
    ctx.ep_mesh = ep_mesh
    ctx.local_shape = tuple(x_TD.shape)
    # Relayout the local shard, not the ep_size-times larger gathered tensor.
    x_global_T3 = funcol.all_gather_tensor(
        to_token_3d(x_TD), gather_dim=0, group=ep_mesh
    )
    ctx.num_out_tokens = int(x_global_T3.shape[0])
    return ragged_gather(x_global_T3, token_ids_R, valid_count_1)

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_RD):
    if grad_RD is None or not ctx.needs_input_grad[0]:
      return (None,) * 6
    valid_count_1, *reduce_metadata = ctx.saved_tensors
    grad_global_T3 = ragged_gather_reduce(
        grad_RD,
        None,
        valid_count_1,
        tuple(reduce_metadata),
        num_out_tokens=ctx.num_out_tokens,
        top_k=ctx.top_k,
    )
    grad_T3 = funcol.reduce_scatter_tensor(
        grad_global_T3, "sum", scatter_dim=0, group=ctx.ep_mesh
    )
    return (grad_T3.reshape(ctx.local_shape),) + (None,) * 5


class _GatherRoutes(torch.autograd.Function):
  """One EP all-gather of every rank's expert ids and routing scores.

  The ids and the FP32 score bits travel in a single int32 buffer, so each
  dispatch issues one small collective instead of separate ones for counts,
  ids, and scores (small collectives queue behind the large token ones).
  """

  @staticmethod
  def forward(ctx, scores_TK, expert_ids_TK, ep_mesh):
    ctx.set_materialize_grads(False)
    ctx.score_shape = tuple(scores_TK.shape)
    ctx.score_dtype = scores_TK.dtype
    ctx.ep_mesh = ep_mesh
    ep_size = ep_mesh.size()
    packed = torch.cat([
        expert_ids_TK.to(torch.int32).reshape(-1),
        scores_TK.to(torch.float32).reshape(-1).view(torch.int32),
    ])
    gathered = funcol.all_gather_tensor(
        packed, gather_dim=0, group=ep_mesh
    ).view(ep_size, 2, -1)
    expert_ids_N = gathered[:, 0].reshape(-1)
    scores_N = gathered[:, 1].contiguous().view(-1).view(torch.float32)
    ctx.mark_non_differentiable(expert_ids_N)
    return expert_ids_N, scores_N

  @staticmethod
  @once_differentiable
  def backward(ctx, _grad_ids_N, grad_scores_N):
    if grad_scores_N is None or not ctx.needs_input_grad[0]:
      return None, None, None
    grad_TK = funcol.reduce_scatter_tensor(
        grad_scores_N, "sum", scatter_dim=0, group=ctx.ep_mesh
    )
    return grad_TK.reshape(ctx.score_shape).to(ctx.score_dtype), None, None


class _SelectScores(torch.autograd.Function):
  """Select this rank's routes from the gathered scores, zero past `valid`."""

  @staticmethod
  def forward(ctx, scores_N, slots_R, perm_N, valid_count_1):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(perm_N, valid_count_1)
    selected_R = torch.index_select(scores_N, 0, slots_R)
    valid_R = (
        torch.arange(slots_R.shape[0], device=slots_R.device)
        < valid_count_1.reshape(1)
    )
    return torch.where(valid_R, selected_R, torch.zeros_like(selected_R))

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_R):
    if grad_R is None or not ctx.needs_input_grad[0]:
      return None, None, None, None
    perm_N, valid_count_1 = ctx.saved_tensors
    # `slots_R` is a prefix of the permutation `perm_N`, so the transpose is a
    # sort keyed on `perm_N` instead of a scalar scatter over all routes.
    return (
        inverse_permute_scatter(perm_N, grad_R, valid_count_1),
        None,
        None,
        None,
    )


class _CombineTokens(torch.autograd.Function):
  """Weight rows by score, sum per token in FP32, reduce-scatter over EP."""

  @staticmethod
  def forward(ctx, routed_RD, scores_R, token_ids_R, valid_count_1,
              reduce_metadata, num_out_tokens, top_k, ep_mesh,
              fuse_route_score_backward):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(routed_RD, scores_R, token_ids_R, valid_count_1)
    ctx.ep_mesh = ep_mesh
    ctx.fuse_route_score_backward = fuse_route_score_backward
    # Same values as `(routed.float() * scores).to(routed.dtype)` followed by
    # the per-token sum; the reduce-scatter runs in the kernels' token layout
    # so only the local shard is relayouted back to [T, D].
    partial_T3 = ragged_gather_reduce(
        routed_RD,
        scores_R,
        valid_count_1,
        reduce_metadata,
        num_out_tokens=num_out_tokens,
        top_k=top_k,
    )
    out_T3 = funcol.reduce_scatter_tensor(
        partial_T3, "sum", scatter_dim=0, group=ep_mesh
    )
    return out_T3.reshape(out_T3.shape[0], -1)

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_TD):
    if grad_TD is None:
      return (None,) * 9
    routed_RD, scores_R, token_ids_R, valid_count_1 = ctx.saved_tensors
    grad_global_T3 = funcol.all_gather_tensor(
        to_token_3d(grad_TD), gather_dim=0, group=ctx.ep_mesh
    )
    grad_weighted_RD = ragged_gather(grad_global_T3, token_ids_R, valid_count_1)
    grad_routed_RD = None
    if ctx.needs_input_grad[0]:
      grad_routed_RD = (
          grad_weighted_RD.float() * scores_R.reshape(-1, 1)
      ).to(routed_RD.dtype)
    grad_scores_R = None
    if ctx.needs_input_grad[1]:
      if (
          ctx.fuse_route_score_backward
          and routed_RD.dtype == torch.bfloat16
          and scores_R.dtype == torch.float32
          and routed_RD.device.type in ("cpu", "tpu")
      ):
        grad_scores_R = route_score_grad(grad_weighted_RD, routed_RD)
      else:
        grad_scores_R = (
            (grad_weighted_RD.float() * routed_RD.float())
            .sum(dim=1)
            .to(scores_R.dtype)
        )
    return (grad_routed_RD, grad_scores_R) + (None,) * 7


def ring_of_experts_dispatch(
    x_TD: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    *,
    num_experts: int,
    ep_mesh,
    capacity_factor: float | None,
) -> tuple[torch.Tensor, torch.Tensor, RingOfExpertsMetadata]:
  """Gather this EP rank's expert rows from all ranks of `ep_mesh`.

  Args:
    x_TD: local tokens; `D` must be a multiple of 128.
    topk_scores_TK: local routing scores.
    topk_expert_ids_TK: local global expert ids.
    num_experts: number of global experts, divisible by the EP degree.
    ep_mesh: 1-D EP device mesh.
    capacity_factor: see `ring_of_experts_capacity`.

  Returns:
    `(routed_RD, group_sizes_e, metadata)`: expert-major rows for the local
    experts (rows past the valid count are zero), the local-expert group sizes
    (summing to the valid count, at most `R`), and the state for
    `ring_of_experts_combine`. The expert computation may leave its output
    rows past the valid count unspecified.
  """
  ep_size = ep_mesh.size()
  if num_experts % ep_size:
    raise ValueError(
        f"num_experts ({num_experts}) must be divisible by the EP degree"
        f" ({ep_size})"
    )
  num_local_experts = num_experts // ep_size
  experts_start = ep_mesh.get_local_rank() * num_local_experts
  local_T, top_k = topk_expert_ids_TK.shape
  num_out_tokens = ep_size * local_T
  num_global_routes = num_out_tokens * top_k
  max_R = ring_of_experts_capacity(local_T * top_k, ep_size, capacity_factor)

  with annotate_region(MOE_ROUTING):
    expert_ids_N, scores_N = _GatherRoutes.apply(
        topk_scores_TK, topk_expert_ids_TK, ep_mesh
    )
    local_e_N = expert_ids_N - experts_start
    is_local_N = (local_e_N >= 0) & (local_e_N < num_local_experts)
    sort_key_N = torch.where(
        is_local_N, local_e_N, torch.full_like(local_e_N, num_local_experts)
    )
    # Per-local-expert route counts, from the gathered ids rather than another
    # collective.
    counts_e = (
        sort_key_N.unsqueeze(1)
        == torch.arange(
            num_local_experts, device=sort_key_N.device, dtype=torch.int32
        )
    ).sum(dim=0, dtype=torch.int32)
    # Truncate to `max_R` rows in local-expert order.
    starts_e = _exclusive_cumsum_int32(counts_e)
    valid_e = torch.minimum(counts_e, (max_R - starts_e).clamp(min=0))
    valid_count_1 = valid_e.sum(dtype=torch.int32).reshape(1)
    # Rows past the valid count belong to no expert, so the grouped matmuls
    # skip them rather than compute the capacity padding.
    group_sizes_e = valid_e

    perm_N = _packed_stable_argsort(sort_key_N, num_local_experts)
    if max_R <= num_global_routes:
      slots_R = perm_N[:max_R]
    else:
      slots_R = torch.nn.functional.pad(perm_N, (0, max_R - num_global_routes))
    token_ids_R = slots_R // top_k
    reduce_metadata = ragged_gather_reduce_metadata(
        token_ids_R, valid_count_1, top_k=top_k, num_out_tokens=num_out_tokens
    )

  routed_RD = _GatherTokens.apply(
      x_TD, token_ids_R, valid_count_1, reduce_metadata, top_k, ep_mesh
  )
  scores_R = _SelectScores.apply(scores_N, slots_R, perm_N, valid_count_1)
  metadata = RingOfExpertsMetadata(
      token_ids_R=token_ids_R,
      valid_count_1=valid_count_1,
      scores_R=scores_R,
      reduce_metadata=reduce_metadata,
      num_out_tokens=num_out_tokens,
      top_k=top_k,
      ep_mesh=ep_mesh,
  )
  return routed_RD, group_sizes_e, metadata


def ring_of_experts_combine(
    routed_output_RD: torch.Tensor,
    metadata: RingOfExpertsMetadata,
    *,
    fuse_route_score_backward: bool = False,
) -> torch.Tensor:
  """Score-weight, sum per token, and return each token to its EP owner.

  Args:
    routed_output_RD: expert outputs in the order of `ring_of_experts_dispatch`.
    metadata: state returned by `ring_of_experts_dispatch`.
    fuse_route_score_backward: compute the score gradient with the fused
      `route_score_grad` kernel (BF16 rows and FP32 scores only).

  Returns:
    `[T, D]` combined local tokens.
  """
  return _CombineTokens.apply(
      routed_output_RD,
      metadata.scores_R,
      metadata.token_ids_R,
      metadata.valid_count_1,
      metadata.reduce_metadata,
      metadata.num_out_tokens,
      metadata.top_k,
      metadata.ep_mesh,
      fuse_route_score_backward,
  )


__all__ = [
    "CAPACITY_TILE",
    "RingOfExpertsMetadata",
    "ring_of_experts_capacity",
    "ring_of_experts_combine",
    "ring_of_experts_dispatch",
]
