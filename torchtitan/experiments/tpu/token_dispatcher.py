# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU VJPs for the standard MoE dispatcher's local permutation work.

The route reorder precedes expert transport, so its custom VJP applies to both
EP=1 and EP>1. The fixed-K combine is deliberately scoped to EP=1. Standard
dropless EP also returns exactly ``top_k`` rows per source token, so the same
reduction should be applicable after the return all-to-all. Supporting it also
requires preserving the EP unpermute and collective, then placing an SP-local
reduction into global sequence coordinates. That broader path was not needed
for the FSDP/EP1 target and has not been benchmarked here.

This scope does not make the reduction depend on bounded ragged all-to-all.
Ragged all-to-all is a separate transport choice that can call the same local
reduction once its capacity and overflow semantics have been validated. The
earlier bounded-ragged prototype did exactly that around ``combine_route``;
the transport was omitted here while the reusable local VJPs were retained.
"""

from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable

from torchtitan.config import derive
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.experiments.tpu.kernels.route_score import route_score_grad
from torchtitan.experiments.tpu.remat_regions import (
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

  def __init__(self, config: Config):
    super().__init__(config)
    self.fuse_route_score_backward = config.fuse_route_score_backward

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

  def combine(
      self,
      routed_output_RD: torch.Tensor,
      metadata: AllToAllDispatchMetadata | LocalDispatchMetadata,
      x_TD: torch.Tensor,
  ) -> torch.Tensor:
    if self.ep_mesh is not None:
      return super().combine(routed_output_RD, metadata, x_TD)

    assert isinstance(metadata, LocalDispatchMetadata)
    if self.fuse_route_score_backward:
      routed_output_ND = scale_route_weights(
          routed_output_RD,
          metadata.topk_scores_experts_sorted_N,
      )
    else:
      routed_output_ND = (
          routed_output_RD.float()
          * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
      ).to(routed_output_RD.dtype)
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
    if not isinstance(metadata, LocalDispatchMetadata):
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
    "TpuRoutedExperts",
    "TpuRoutedExpertsConverter",
    "TpuTokenDispatcherConverter",
    "local_reorder",
    "scale_route_weights",
    "stable_grouped_reduce",
]
