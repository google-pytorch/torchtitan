# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expert parallelism helpers for TPU.

Provides `use_expert_parallel` to configure a model's MoE layers for TPU EP.
"""

import torchtitan.config
from torchtitan.experiments.tpu.components.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.components.token_dispatcher import TpuAllToAllTokenDispatcher, TpuRoutedExperts
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher


def default_num_token_chunks(ep_degree: int) -> int:
  """Recommended EP sequence chunks per MoE layer: ``ep_degree`` clamped to 4..16."""
  if ep_degree <= 1:
    return 1
  return max(4, min(ep_degree, 16))


def use_expert_parallel(
    model_config,
    *,
    ep_degree: int,
    ep_transport: str | None = None,
    scale_before_down_proj: bool | None = None,
    ep_capacity_factor: float | None = None,
    ep_overflow_mode: str | None = None,
    num_token_chunks: int | None = None,
    max_inflight_chunks: int | None = None,
    accumulate_chunk_weight_grads_fp32: bool | None = None,
    paired_grouped_mm: bool | None = None,
) -> None:
  """Configures a MoE model config for expert parallelism on TPU.

  A no-op when ``ep_degree`` is 1. Under EP > 1 it:

  - turns plain ``RoutedExperts`` configs into ``TpuRoutedExperts``, whose
    ``parallelize`` binds the EP rank buffer that the TPU transports need, and
    optionally sets ``scale_before_down_proj``, ``num_token_chunks``, and
    ``max_inflight_chunks``;
  - turns the all-to-all dispatchers into ``TpuAllToAllTokenDispatcher``,
    sets ``ep_transport`` (see below), and optionally sets
    ``ep_capacity_factor``;
  - keeps the paired (w1 + w3) Pallas grouped matmul where the recipe enabled
    it, unless ``paired_grouped_mm`` is False.

  Args:
    model_config: The model configuration to update in place.
    ep_degree: The expert-parallel degree.
    ep_transport: Explicit EP transport to set on every dispatcher, or
      ``None`` to keep each dispatcher's explicit ``ep_transport`` (including
      ``"all_to_all"``) and set ``"auto"`` only where it is unset (``None``).
      ``"auto"`` picks Ring-of-Experts when ``ep_degree <= top_k`` and it can
      run (TPU, ``hidden_dim % 128 == 0``), else ragged all-to-all. An
      explicit ``"ring_of_experts"`` that cannot run raises.
    scale_before_down_proj: Optional override for applying routing weights
      before ``w2`` on ``TpuRoutedExperts``. Requires the routed experts to
      use the Pallas grouped matmuls (the only ones that take per-row scales);
      raises ``ValueError`` otherwise.
    ep_capacity_factor: Optional per-rank capacity factor, in multiples of the
      balanced per-rank routed rows ``T * top_k`` (MaxText's
      ``ragged_buffer_factor``). On ``ragged_all_to_all`` it bounds the
      receive buffer, which is also the expert row count; on
      ``ring_of_experts`` it bounds only the locally gathered expert rows,
      rounded up to a multiple of 1024 (the all-gather and reduce-scatter
      stay full size). ``None`` preserves any factor set on the dispatcher
      (default ``None``, dropless worst-case buffer).
    ep_overflow_mode: Optional ``"drop"`` or ``"chunked"``: what happens
      to routes past ``ep_capacity_factor``'s capacity. ``"chunked"`` runs
      them in extra capacity-sized windows (Ring-of-Experts only), so no
      route is dropped. ``None`` keeps the dispatcher's mode (``"drop"``).
    num_token_chunks: Optional override for the number of sequence chunks per
      MoE layer on ``TpuRoutedExperts``.
    max_inflight_chunks: Optional override for the maximum simultaneously live
      EP chunks enforced by ``insert_ep_chunk_barriers``.
    accumulate_chunk_weight_grads_fp32: Optional override for summing the
      chunked expert weight gradients across token chunks in FP32 (the
      ``TpuRoutedExperts`` default) instead of the weight dtype.
    paired_grouped_mm: ``False`` switches paired Pallas grouped matmuls to the
      unpaired kernels under EP; ``None`` or ``True`` leaves the recipe's
      choice. The paired kernel skips rows past the routed count like the
      unpaired ones, so it is safe on EP's static-size receive buffers.
  """
  if ep_degree > 1:
    # TpuRoutedExperts binds the EP rank buffer in `parallelize`.
    for _, experts, parent, field_name in list(
        model_config.traverse(RoutedExperts.Config)
    ):
      if type(experts) is RoutedExperts.Config:  # pylint: disable=unidiomatic-typecheck
        ep_experts = torchtitan.config.derive(
            experts,
            TpuRoutedExperts.Config,
        )
        if isinstance(parent, list):
          parent[field_name] = ep_experts
        else:
          setattr(parent, field_name, ep_experts)
      elif type(experts) is TpuRoutedExperts.Config:  # pylint: disable=unidiomatic-typecheck
        ep_experts = experts
      else:
        continue

      if scale_before_down_proj is not None:
        ep_experts.scale_before_down_proj = scale_before_down_proj
      if num_token_chunks is not None:
        ep_experts.num_token_chunks = num_token_chunks
      if max_inflight_chunks is not None:
        ep_experts.max_inflight_chunks = max_inflight_chunks
      if accumulate_chunk_weight_grads_fp32 is not None:
        ep_experts.accumulate_chunk_weight_grads_fp32 = (
            accumulate_chunk_weight_grads_fp32
        )
      # Routing weights before W2 pass `row_scale_R`, which only the Pallas
      # grouped matmuls accept (stock `GroupedExperts` has no such argument).
      # Fail at config time, not in the first forward, and leave the kernel
      # choice to the config.
      inner = ep_experts.inner_experts
      if ep_experts.scale_before_down_proj and not (
          isinstance(inner, TpuGroupedExperts.Config)
          and inner.use_pallas_grouped_mm
      ):
        raise ValueError(
            "scale_before_down_proj needs the Pallas grouped matmuls; set"
            " use_pallas_grouped_mm on the routed experts (for example with"
            " _use_tpu_moe_kernels)"
        )

    if paired_grouped_mm is False:
      for _, group_experts, _, _ in model_config.traverse(
          TpuGroupedExperts.Config
      ):
        group_experts.use_pallas_paired_grouped_mm = False

    for _, dispatcher, parent, field_name in list(
        model_config.traverse(AllToAllTokenDispatcher.Config)
    ):
      if type(dispatcher) in (  # pylint: disable=unidiomatic-typecheck
          AllToAllTokenDispatcher.Config,
          TpuAllToAllTokenDispatcher.Config,
      ):
        # Upstream's config has no `ep_transport`; on the TPU config `None`
        # means unset. Only an unset transport becomes "auto".
        existing_transport = getattr(dispatcher, "ep_transport", None)
        if ep_transport is not None:
          chosen_transport = ep_transport
        elif existing_transport is not None:
          chosen_transport = existing_transport
        else:
          chosen_transport = "auto"
        derive_kwargs = {"ep_transport": chosen_transport}
        if ep_capacity_factor is not None:
          derive_kwargs["ep_capacity_factor"] = ep_capacity_factor
        if ep_overflow_mode is not None:
          derive_kwargs["ep_overflow_mode"] = ep_overflow_mode
        ep_dispatcher = torchtitan.config.derive(
            dispatcher,
            TpuAllToAllTokenDispatcher.Config,
            **derive_kwargs,
        )
        if isinstance(parent, TpuRoutedExperts.Config) and (
            parent.scale_before_down_proj
        ):
          ep_dispatcher.fuse_route_score_backward = False
        if isinstance(parent, list):
          parent[field_name] = ep_dispatcher
        else:
          setattr(parent, field_name, ep_dispatcher)
