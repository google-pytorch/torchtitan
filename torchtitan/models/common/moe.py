# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_remat as remat
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import implicit_replication

from torchtitan.distributed.spmd_types import (
    maybe_set_sparse_mesh,
    spmd_local_context,
    spmd_mesh_size,
    spmd_sparse_mesh,
)
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.aux_loss import _AuxLossInjection, AuxLoss
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import RouterGateLinear
from torchtitan.protocols.module import Module

from .token_dispatcher import LocalTokenDispatcher

# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   T = num tokens, D = model dimension,
#   F = hidden (FFN intermediate) dimension, E = num experts,
#   e = num local experts (E / EP, used in token dispatcher for
#       per-local-expert token counts after EP dispatch /_permute),
#   K = top-k, N = routed tokens (T*K),
#   R = routed tokens assigned to local experts,
#   O = expert output features, I = expert input features
#       (roles, not model dims: the _grouped_mm seam takes the expert
#        weight in its stored (E, O, I) orientation, which is (E, F, D)
#        for the up/gate projections and (E, D, F) for the down one)


# BEGIN PR3000 cherry-pick from db0c4a456b2fd3377f68a4f0cddef015e83ae10e.
# Keep the load-balance math in this block in sync with the upstream PR. The
# module plumbing is adapted to d771abb0, which moved MoE auxiliary losses into
# the router (``TokenChoiceTopKRouter.Config.aux_loss``).
def _sequence_wise_load_balance_loss(
    scores_BLE: torch.Tensor,
    routing_map_BLE: torch.Tensor,
    top_k: int,
    aux_loss_coeff: float,
    *,
    cp_mesh: DeviceMesh | None = None,
    tp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """Sequence-wise auxiliary load-balance loss (DeepSeek-V3 Eqs 17-20).

    For each sequence, compute f_i * p_i across that sequence's expert
    probability scores, then average the resulting load-balance losses over the
    batch.

    With sequence partitioning, counts_BE starts as per-batch,
    per-sequence-shard expert counts and is all-reduced across sequence-shard
    groups (CP, and TP when EP enables sequence parallelism) to get full-sequence
    expert counts.
    """
    B, L, num_experts = scores_BLE.shape
    denom = scores_BLE.sum(dim=-1, keepdim=True) + 1e-20
    probs_BLE = scores_BLE / denom
    prob_sums_BE = probs_BLE.sum(dim=1)
    counts_BE = routing_map_BLE.sum(dim=1).to(dtype=scores_BLE.dtype)

    counts_BE = _all_reduce_expert_counts(
        counts_BE,
        reduction="sequence",
        dp_mesh=None,
        cp_mesh=cp_mesh,
        tp_mesh=tp_mesh,
    )
    num_tokens_B = counts_BE.sum(dim=1) / top_k

    p_BE = prob_sums_BE / num_tokens_B.unsqueeze(1)
    f_BE = counts_BE * (num_experts / (top_k * num_tokens_B.unsqueeze(1)))
    return (f_BE * p_BE).sum(dim=1).mean() * aux_loss_coeff


def _batch_wise_load_balance_loss(
    scores_BLE: torch.Tensor,
    routing_map_BLE: torch.Tensor,
    top_k: int,
    aux_loss_coeff: float,
    *,
    dp_mesh: DeviceMesh | None = None,
    cp_mesh: DeviceMesh | None = None,
    tp_mesh: DeviceMesh | None = None,
    num_tokens_per_expert: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch-wise auxiliary load-balance loss.

    Computes f_i * p_i using expert frequency across the global batch. Counts
    start as an [E] per-expert count for the local batch and sequence shard, then
    are all-reduced across DP and sequence-shard groups (CP, and TP when EP
    enables sequence parallelism) to get global-batch expert counts.
    """
    B, L, num_experts = scores_BLE.shape
    if num_tokens_per_expert is None:
        num_tokens_per_expert = routing_map_BLE.sum(dim=(0, 1))
    probs_BLE = scores_BLE / (scores_BLE.sum(dim=-1, keepdim=True) + 1e-20)
    prob_sums_E = probs_BLE.sum(dim=(0, 1))

    num_tokens_per_expert = num_tokens_per_expert.to(scores_BLE.dtype)
    global_num_tokens_per_expert = _all_reduce_expert_counts(
        num_tokens_per_expert,
        reduction="batch",
        dp_mesh=dp_mesh,
        cp_mesh=cp_mesh,
        tp_mesh=tp_mesh,
    )
    global_tokens = global_num_tokens_per_expert.sum() / top_k

    # DP contributes through summed gradients and scaled_coeff, so p_i uses the
    # non-DP token count for this local batch contribution.
    dp_degree = dp_mesh.size() if dp_mesh is not None else 1
    local_non_dp_tokens = global_tokens / dp_degree
    p_i = prob_sums_E / local_non_dp_tokens
    f_i = global_num_tokens_per_expert * (
        num_experts / (top_k * global_tokens)
    )
    return (f_i * p_i).sum() * aux_loss_coeff


def _all_reduce_expert_counts(
    counts: torch.Tensor,
    *,
    reduction: Literal["batch", "sequence"],
    dp_mesh: DeviceMesh | None,
    cp_mesh: DeviceMesh | None,
    tp_mesh: DeviceMesh | None,
) -> torch.Tensor:
    """Sum non-differentiable expert counts over token-partition mesh axes.

    The tensor shape is preserved. For batch-wise loss this is ``[E]``; for
    sequence-wise loss this is ``[B, E]`` so the per-sequence objective never
    reduces across the batch dimension.
    """
    assert not counts.requires_grad, "MoE load-balance counts should not require grad"
    if reduction == "batch":
        assert counts.ndim == 1, (
            f"batch-wise expert counts must be 1D, got {counts.shape}"
        )
        reduce_meshes = (tp_mesh, dp_mesh, cp_mesh)
    else:
        assert reduction == "sequence"
        assert counts.ndim == 2, (
            f"sequence-wise expert counts must be 2D, got {counts.shape}"
        )
        assert dp_mesh is None, "sequence-wise expert counts must not reduce over DP"
        reduce_meshes = (tp_mesh, cp_mesh)

    with torch.no_grad():
        for reduce_mesh in reduce_meshes:
            if reduce_mesh is None:
                continue
            dist.all_reduce(
                counts, group=reduce_mesh.get_group(), op=dist.ReduceOp.SUM
            )
    return counts


class MoELoadBalanceAuxLoss(AuxLoss):
    """MoE auxiliary load-balance loss, configured on the router.

    Set as ``TokenChoiceTopKRouter.Config.aux_loss``: the router calls it in
    training with its scores and routing map, and it injects the loss gradient
    into the router's top-k scores without changing the model output (PP-safe),
    through the base class's ``_AuxLossInjection``.

    It does not go through ``AuxLoss.inject``, for two reasons:

    - The normalization is this loss's own. ``batch_wise`` is Qwen's
      global-batch loss: f_i and p_i are taken over the global batch (counts
      are all-reduced over the meshes set by
      ``configure_moe_aux_loss_reduction``) and each forward is weighted by
      ``B / global_batch_size`` (see ``update_moe_aux_loss_configs``), rather
      than by the step's global valid-token count.
    - ``inject`` accumulates its logged value into the ``instance_acc`` buffer
      during the forward, and a scanned layer stack cannot write to a buffer it
      closed over. The per-layer value is logged by the MoE routing metrics
      (``moe/global_aux_loss_normalized``) instead, so this loss is not
      registered with ``collect_aux_loss_metrics``. Because the value is
      overwritten on each forward rather than accumulated, it is also not
      double-counted when activation checkpointing replays the forward.

    Unregistering means adjusting the private ``AuxLoss._group_counts`` after
    ``super().__init__``; ``test_global_batch_aux_loss_is_not_a_logged_aux_loss_group``
    fails if upstream changes that bookkeeping.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        type: Literal["sequence_wise", "batch_wise"]
        """Loss variant. sequence_wise (DeepSeek-V3 Eqs 17-20) computes
        per-sequence then averages; batch_wise averages over all tokens.
        Set router.aux_loss to None to disable aux loss entirely."""
        top_k: int
        """Number of experts per token. Must match the router's top_k."""
        global_batch_size: int | None = None
        """Global batch size to normalize aux loss across all batches/microbatches.
        Torchtitan generally uses sum aggregation & global_bs division.
        Because batch_wise aux loss doesn't commute over microbatches, we ban
        batch_wise + PP usage in any case."""
        seq_len: int | None = None
        """Tokens per sequence (``training.max_context_length``). Converts the
        router's ``[T, E]`` token count to samples for the ``global_batch_size``
        weight, and regroups tokens into sequences for ``sequence_wise``. Set
        by ``update_moe_aux_loss_configs``."""

    def __init__(self, config: Config):
        if config.coeff <= 0:
            raise ValueError(
                f"aux_loss.coeff must be > 0, got {config.coeff}. "
                "Set router.aux_loss to None to disable aux loss."
            )
        if config.type not in ("sequence_wise", "batch_wise"):
            raise ValueError(
                f"Unknown aux loss type '{config.type}', "
                f"expected one of ('sequence_wise', 'batch_wise')"
            )
        super().__init__(config)
        # Not reported by ``collect_aux_loss_metrics``; see the class docstring.
        metric_group = (self.reduce_mesh, self.metric_name)
        AuxLoss._group_counts[metric_group] -= 1
        if AuxLoss._group_counts[metric_group] == 0:
            del AuxLoss._group_counts[metric_group]
        # Unused: this loss doesn't call inject(), so drop the base class's
        # metric buffer.
        del self.instance_acc
        self.type = config.type
        self.global_batch_size = config.global_batch_size
        self.seq_len = config.seq_len
        self.top_k = config.top_k
        self.dp_mesh: DeviceMesh | None = None
        self.cp_mesh: DeviceMesh | None = None
        self.tp_mesh: DeviceMesh | None = None
        self.collect_routing_metrics = False

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        # No buffers: the base class's instance_acc is dropped in __init__.
        pass

    def set_meshes(
        self,
        *,
        dp_mesh: DeviceMesh | None,
        cp_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        self.dp_mesh = dp_mesh
        self.cp_mesh = cp_mesh
        self.tp_mesh = tp_mesh

    def forward(
        self,
        scores_TE: torch.Tensor,
        routing_map_TE: torch.Tensor | None,
        *,
        carrier: torch.Tensor,
        expert_ids_TK: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the load-balance loss and inject its gradient.

        Args:
            scores_TE: Router scores ``(T, E)``.
            routing_map_TE: One-hot routing map ``(T, E)`` for the same tokens,
                as counted by the router; None for batch-wise with
                ``token_count_impl="histc"``.
            carrier: Tensor whose backward path carries the injected gradient
                (the router's top-k scores).
            expert_ids_TK: The router's top-k expert ids ``(T, K)``, counted
                with ``histc`` when ``routing_map_TE`` is None.

        Returns:
            ``carrier`` unchanged (identity forward).
        """
        # The expert counts are all-reduced explicitly over the meshes wired by
        # ``configure_moe_aux_loss_reduction``, so the loss works on this
        # rank's local tokens. ``to_local`` keeps the gradient placement of the
        # router output.
        scores_TE = (
            scores_TE.to_local() if isinstance(scores_TE, DTensor) else scores_TE
        )
        routing_map_TE = (
            routing_map_TE.to_local()
            if isinstance(routing_map_TE, DTensor)
            else routing_map_TE
        )
        num_experts = scores_TE.size(-1)
        if self.global_batch_size is None:
            # Falling back to B would drop the 1 / dp-degree factor, making the
            # aux gradient dp-times too large after the DP gradient sum.
            raise ValueError(
                "MoELoadBalanceAuxLoss.global_batch_size is not set. Call "
                "update_moe_aux_loss_configs from the model's "
                "update_from_config, which also validates PP and gradient "
                "accumulation."
            )
        # [B, L, E] inputs carry their own sequence length; the router's
        # [T, E] outputs need the configured one.
        seq_len = scores_TE.size(1) if scores_TE.ndim == 3 else self.seq_len
        if seq_len is None:
            raise ValueError(
                "MoELoadBalanceAuxLoss.seq_len is not set. Call "
                "update_moe_aux_loss_configs from the model's "
                "update_from_config."
            )
        local_counts_E = None
        if routing_map_TE is None:
            ids_TK = (
                expert_ids_TK.to_local()
                if isinstance(expert_ids_TK, DTensor)
                else expert_ids_TK
            )
            local_counts_E = _tokens_per_expert(ids_TK, num_experts)
        # This rank's share of the global batch, B / global_batch_size, counted
        # in tokens so the batch-wise loss never regroups tokens into sequences.
        num_local_tokens = scores_TE.numel() // num_experts
        scaled_coeff = (
            self.coeff * num_local_tokens / (self.global_batch_size * seq_len)
        )
        if self.type == "sequence_wise":
            # Per-sequence expert frequencies need the sequence boundaries.
            if num_local_tokens % seq_len != 0:
                raise ValueError(
                    f"The router's local token count ({num_local_tokens}) is "
                    f"not a multiple of seq_len ({seq_len}), so it cannot be "
                    "regrouped into whole sequences."
                )
            aux_loss = _sequence_wise_load_balance_loss(
                scores_TE.reshape(-1, seq_len, num_experts),
                routing_map_TE.reshape(-1, seq_len, num_experts),
                self.top_k,
                scaled_coeff,
                cp_mesh=self.cp_mesh,
                tp_mesh=self.tp_mesh,
            )
        else:
            # The batch-wise loss only sums over tokens, so a single
            # [1, T, E] "sequence" gives the same result as any [B, L, E]
            # grouping of the same tokens.
            scores_1TE = scores_TE.reshape(1, -1, num_experts)
            aux_loss = _batch_wise_load_balance_loss(
                scores_1TE,
                routing_map_TE.reshape(1, -1, num_experts)
                if routing_map_TE is not None
                else None,
                self.top_k,
                scaled_coeff,
                dp_mesh=self.dp_mesh,
                cp_mesh=self.cp_mesh,
                tp_mesh=self.tp_mesh,
                num_tokens_per_expert=local_counts_E,
            )
            if self.collect_routing_metrics:
                self._last_local_normalized_aux_loss = (
                    aux_loss / scaled_coeff
                ).detach().float()
        if self.collect_routing_metrics:
            # Local routed-token counts for the MoE routing metrics; read by
            # ``collect_moe_routing_metric_tensors`` right after the forward.
            self._last_local_tokens_per_expert_E = (
                local_counts_E
                if local_counts_E is not None
                else routing_map_TE.reshape(-1, num_experts).sum(dim=0)
            ).detach().float()
        return _AuxLossInjection.apply(carrier, aux_loss)


def configure_moe_aux_loss_reduction(model: nn.Module, parallel_dims) -> None:
    """Wire process meshes used to reduce MoE auxiliary-loss counts.

    The loss functions choose which meshes they need from these handles:
    batch-wise reduces ``[E]`` counts over DP, CP, and TP when applicable.
    Sequence-wise reduces ``[B, E]`` counts over TP (when applicable) and CP
    only, preserving per-sequence counts and avoiding DP.

    In the current MoE sharding, TP only splits router/count tokens when EP is
    enabled: the TP axis doubles as a sequence-parallel split for routing. When
    EP is disabled, router counts are replicated over TP and must not be
    all-reduced on the TP mesh.
    """
    # TODO: the meshes are wired by hand, assuming TP splits router tokens only
    # under EP. Upstream's global batch-wise loss (pytorch/torchtitan#4772)
    # infers this from spmd_types placements; switch when it lands.
    dp_mesh = parallel_dims.get_optional_mesh("batch")
    cp_mesh = parallel_dims.get_optional_mesh("cp")
    tp_mesh = (
        parallel_dims.get_optional_mesh("tp") if parallel_dims.ep_enabled else None
    )
    aux_losses = [
        m for m in model.modules() if isinstance(m, MoELoadBalanceAuxLoss)
    ]
    if aux_losses and (cp_mesh is not None or tp_mesh is not None):
        # The loss treats the router's rank-local tokens as whole sequences of
        # the global seq_len (its share of the batch, and sequence_wise's
        # grouping), which is wrong once CP, or TP with EP, splits them.
        raise NotImplementedError(
            "MoELoadBalanceAuxLoss does not support context parallelism or "
            "tensor parallelism with expert parallelism: it would group the "
            "rank-local tokens into the wrong batch size without error. Use "
            "MicrobatchWiseLoadBalanceLoss, or wait for "
            "pytorch/torchtitan#4772."
        )
    for module in aux_losses:
        module.set_meshes(dp_mesh=dp_mesh, cp_mesh=cp_mesh, tp_mesh=tp_mesh)


# END PR3000 cherry-pick.


def _tokens_per_expert(
    topk_expert_ids_BLK: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Count selected routes without materializing a ``(B, L, E)`` map."""
    return torch.histc(
        topk_expert_ids_BLK.reshape(-1).float(),
        bins=num_experts,
        min=0,
        max=num_experts,
    ).to(torch.int64)


class GroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.w1_EFD = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.w2_EDF = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
        self.w3_EFD = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine.

        Shape suffixes here describe logical grouped-mm inputs, not physical
        sharding. Under EP, E may be a local shard of experts; under TP,
        expert weights shard hidden dimensions instead; under SP, R may be a
        local token shard. Keep logical capital suffixes here to avoid encoding
        a specific parallel layout in these local tensor names.
        """
        if isinstance(self.w1_EFD, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if (
            get_spmd_backend() == "spmd_types"
            and spmd.is_type_checking()
            and spmd_mesh_size("ep") == 1
        ):
            for axis in ("dp", "cp"):
                # if no EP, convert to V for grouped_mm, which would otherwise see
                # x:R, w1:V, offsets:P in local SPMD typechecking.
                # spmd.P is not currently allowed to mix with spmd.V.
                # TODO(pianpwk): likely relax this in spmd_types.
                spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

        h_RF = F.silu(
            self._grouped_mm(A=x_RD.bfloat16(), weight_EOI=w1_EFD, offs=offsets_E)
        )
        h_RF = h_RF * self._grouped_mm(
            A=x_RD.bfloat16(), weight_EOI=w3_EFD, offs=offsets_E
        )
        return self._grouped_mm(A=h_RF, weight_EOI=w2_EDF, offs=offsets_E).type_as(x_RD)

    def _grouped_mm(
        self, *, A: torch.Tensor, weight_EOI: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        """Grouped matmul of ``A @ weight_EOI.transpose(-2, -1)``.

        ``weight_EOI`` is the grouped expert weight in its stored
        ``(experts, out_features, in_features)`` orientation; the transpose to
        the grouped-GEMM right operand happens here. Overridable seam for
        low-precision variants (e.g. the MXFP8 converter swaps this for a
        scaled grouped GEMM). Variants receive the weight rather than its
        transpose because a quantized representation may be owned by the
        weight's FSDP unshard lifetime and is keyed off the stored orientation.
        Keeping the op here -- rather than behind a tensor-subclass
        ``__torch_function__`` -- means it is captured by FX tracers such as
        graph_trainer's make_fx path.
        """
        return torch._grouped_mm(A, weight_EOI.bfloat16().transpose(-2, -1), offs=offs)


class RoutedExperts(Module):
    """Routed-expert ``local_map`` region: composes token_dispatcher + inner_experts
    as sibling nodes so each can be overridden independently."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        inner_experts: GroupedExperts.Config
        token_dispatcher: LocalTokenDispatcher.Config

    def __init__(self, config: Config):
        super().__init__()
        self.inner_experts = config.inner_experts.build()
        self.token_dispatcher = config.token_dispatcher.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add.

        When parallelized, ``local_map`` (from ``sharding_config``) handles
        DTensor→local conversion on entry and local→DTensor(Partial) wrapping
        on exit. The forward body operates on plain local tensors.
        """
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = self.token_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        with maybe_set_sparse_mesh():
            routed_output_RD = self.inner_experts(
                routed_input_RD, num_global_tokens_per_local_expert_e
            )
        out_TD = self.token_dispatcher.combine(
            routed_output_RD,
            metadata,
            x_TD,
        )
        return out_TD

    def parallelize(self, parallel_dims) -> None:
        """Parallelize the grouped experts, then wire the EP mesh on the
        dispatcher so dispatch/combine see the right mesh at runtime."""
        super().parallelize(parallel_dims)
        # TODO(@pianpwk): With spmd_types and set_current_spmd_mesh, replace wire_meshes
        # with current_spmd_mesh calls inside AllToAllTokenDispatcher and
        # DeepEPTokenDispatcher.
        self.token_dispatcher.wire_meshes(
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
        )


def _replicate_plain_tensors(t: torch.Tensor):
    """Lets ``t``'s ops mix in plain tensors (the iotas below) when it is a
    DTensor, as under TP; a no-op for plain tensors."""
    return implicit_replication() if isinstance(t, DTensor) else nullcontext()


@spmd.register_local_autograd_function
class _GatherTopKScores(torch.autograd.Function):
    """``scores_TE.gather(-1, ids_TK)`` with a dense backward.

    The stock backward scatter-adds ``grad_TK`` into a zero ``(T, E)``
    tensor, which XLA lowers on TPU to a slow element-serial scatter. This one
    builds it as a compare-select-sum over the K slots,
    ``grad_TE[t, e] = sum_k where(ids_TK[t, k] == e, grad_TK[t, k], 0)``.
    Top-k ids are distinct within a row, so each ``(t, e)`` gets at most one
    nonzero term and the result equals the scatter-add exactly.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, scores_TE: torch.Tensor, ids_TK: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(ids_TK)
        ctx.num_experts = scores_TE.size(-1)
        return scores_TE.gather(dim=-1, index=ids_TK)

    @staticmethod
    def backward(ctx, grad_TK: torch.Tensor):  # pyrefly: ignore[bad-override]
        (ids_TK,) = ctx.saved_tensors
        experts_E = torch.arange(
            ctx.num_experts, device=ids_TK.device, dtype=ids_TK.dtype
        )
        with _replicate_plain_tensors(ids_TK):
            hit_TKE = ids_TK.unsqueeze(-1) == experts_E
            grad_TE = torch.where(hit_TKE, grad_TK.unsqueeze(-1), 0).sum(dim=-2)
        return grad_TE, None


class TokenChoiceTopKRouter(Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Optionally supports node-limited (group-limited) routing where experts are divided into groups
    (e.g., by node), and only num_limited_groups groups are considered before selecting top_k experts.
    This reduces cross-node communication in distributed settings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        gate: RouterGateLinear.Config
        num_expert_groups: int | None = None  # must be a divisor of num_experts
        num_limited_groups: int | None = None
        top_k: int = 1
        score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sigmoid"
        route_norm: bool = False
        route_scale: float = 1.0
        aux_loss: AuxLoss.Config | None = None
        token_count_impl: Literal["histc", "routing_map"] = "histc"
        """How routed tokens are counted per expert. ``"routing_map"`` builds
        upstream's one-hot ``(T, E)`` map with ``zeros_like + scatter_``
        (pytorch/torchtitan#3450) and sums it; ``"histc"`` counts the top-k ids
        directly, skipping that scatter, which is slow on TPU. The map is still
        built when ``aux_loss`` consumes it: any aux loss except the batch-wise
        MoELoadBalanceAuxLoss, which counts the top-k ids itself."""
        _debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()
        self.num_experts = config.num_experts
        self.num_expert_groups = config.num_expert_groups
        self.num_limited_groups = config.num_limited_groups
        self.top_k = config.top_k
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None
        # MoELoadBalanceAuxLoss's batch-wise form counts the top-k ids itself;
        # other aux losses consume the map.
        aux_loss_needs_map = self.aux_loss is not None and not (
            isinstance(self.aux_loss, MoELoadBalanceAuxLoss)
            and self.aux_loss.type == "batch_wise"
        )
        self.build_routing_map = (
            config.token_count_impl == "routing_map" or aux_loss_needs_map
        )
        self._debug_force_load_balance = config._debug_force_load_balance

    def _debug_force_load_balance_routing(
        self, scores_TE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns expert IDs and scores with shape ``(T, K)``.
        """
        num_tokens = scores_TE.shape[0]
        # Round-robin indices with exact balance
        topk_expert_ids_TK = (
            torch.arange(
                num_tokens * self.top_k,
                device=scores_TE.device,
                dtype=torch.int64,
            ).reshape(num_tokens, self.top_k)
            % self.num_experts
        )
        topk_scores_TK = _GatherTopKScores.apply(scores_TE, topk_expert_ids_TK)
        return topk_expert_ids_TK, topk_scores_TK

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice_TE: torch.Tensor,
    ) -> torch.Tensor:
        """Select num_limited_groups groups based on group scores,
        and set expert scores in non-selected groups as -inf.

        Args:
            scores_for_choice_TE: Router scores with expert_bias, shape ``(T, E)``.

        Returns:
            Router scores with shape ``(T, E)``.
        """
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        assert self.num_expert_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")
        scores_grouped = scores_for_choice_TE.unflatten(
            -1, (self.num_expert_groups, experts_per_group)
        )
        # Sum of each group's top 2 scores without topk, which on TPU falls
        # back to a bitonic sort at 32 experts per group: the max, plus the max
        # with the first argmax masked out. Masking one position (not every
        # copy of the max) keeps a tied max as the second value, as topk does.
        max1_scores_in_group = scores_grouped.amax(dim=-1)
        max1_idx_in_group = scores_grouped.argmax(dim=-1, keepdim=True)
        experts_in_group = torch.arange(
            experts_per_group,
            device=max1_idx_in_group.device,
            dtype=max1_idx_in_group.dtype,
        )
        with _replicate_plain_tensors(scores_grouped):
            max2_scores_in_group = scores_grouped.masked_fill(
                experts_in_group == max1_idx_in_group, float("-inf")
            ).amax(dim=-1)
        group_scores = max1_scores_in_group + max2_scores_in_group
        _, group_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(-1, group_idx, False)  # False = selected groups (keep)
        # Mask out experts from non-selected groups
        scores_for_choice_TE = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).flatten(-2)

        return scores_for_choice_TE

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        scores_for_choice_TE = (
            scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
        )
        if self.num_expert_groups is not None:
            scores_for_choice_TE = self._get_node_limited_routing_scores(
                scores_for_choice_TE
            )
        return torch.topk(
            # TODO(b/510002414): Force sorted=True for TorchTPU Aten kernel compatibility.
            scores_for_choice_TE, k=self.top_k, dim=-1, sorted=True
        ).indices

    def forward(
        self,
        x_TD: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x_TD: Input ``(T, D)``.
            expert_bias_E: Optional load-balancing bias ``(E,)``.

        Returns:
            topk_scores_TK: Routing scores ``(T, K)``.
            topk_expert_ids_TK: Expert indices ``(T, K)``.
            routing_map_TE: One-hot boolean routing map ``(T, E)``, or None
                with ``token_count_impl="histc"`` (count ``topk_expert_ids_TK``).
        """
        scores_TE = self.gate(x_TD)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion.
        # RouterGateLinear returns scores_TE in FP32.
        if self.score_func == "sigmoid":
            scores_TE = torch.sigmoid(scores_TE)
        elif self.score_func == "softmax":
            scores_TE = F.softmax(scores_TE, dim=-1)
        elif self.score_func == "sqrtsoftplus":
            scores_TE = F.softplus(scores_TE).sqrt()
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self._debug_force_load_balance:
            topk_expert_ids_TK, topk_scores_TK = self._debug_force_load_balance_routing(
                scores_TE
            )
        else:
            # Routing choices must remain identical between forward and replay.
            topk_expert_ids_TK = remat.region(
                self._select_experts,
                "routing_decision",
                recompute=False,
            )(scores_TE, expert_bias_E, **router_kwargs)
            remat.recompute_needs_tensor(topk_expert_ids_TK)

            # The expert bias is only used for routing. The gating value is
            # still derived from the original scores.
            topk_scores_TK = _GatherTopKScores.apply(scores_TE, topk_expert_ids_TK)

        if self.route_norm:
            denominator = topk_scores_TK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_TK = topk_scores_TK / denominator
        topk_scores_TK = topk_scores_TK * self.route_scale

        # Build a one-hot boolean routing map (T, E) marking the experts each
        # token is routed to.  Under TP/SP the router outputs are DTensors
        # sharded on the token dim; scatter_ writes along the (replicated)
        # expert dim, so DTensor runs it as a local op with no redistribution.
        routing_map_TE = (
            torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
                -1,
                topk_expert_ids_TK,
                True,
            )
            if self.build_routing_map
            else None
        )

        # Auxiliary load-balance loss (DeepSeek-V3 Sec 2.1.2 Eqs 17-20).
        # The gradient is injected into topk_scores_TK on backward; the loss
        # itself keeps its forward-side metric accumulation from being re-run
        # by activation checkpointing (see ``AuxLoss.inject``).  The
        # routing map is passed in so the loss counts exactly the tokens this
        # router counted: once the router masks padding positions out of the
        # map, the loss and its token count follow without further changes.
        if self.training and self.aux_loss is not None:
            aux_kwargs = (
                {"expert_ids_TK": topk_expert_ids_TK}
                if isinstance(self.aux_loss, MoELoadBalanceAuxLoss)
                else {}
            )
            topk_scores_TK = self.aux_loss(
                scores_TE,
                routing_map_TE,
                carrier=topk_scores_TK,
                **aux_kwargs,
            )

        return (
            topk_scores_TK,
            topk_expert_ids_TK,
            routing_map_TE,
        )


class MicrobatchWiseLoadBalanceLoss(AuxLoss):
    """Per-forward MoE load-balance gradient (DeepSeek-V3 Sec 2.1.2 Eqs 17-20).

    The balancing unit is one forward's folded token stream (a DP-local
    microbatch).  Global (corpus-level) balance is left to the
    auxiliary-loss-free bias path (``expert_bias_E``); this loss only
    discourages extreme load imbalance within individual forwards (samples),
    per the DeepSeek-V3 design (Sec 2.1.2, "Complementary Sequence-Wise
    Auxiliary Loss").

    With ``E`` experts, top-``K`` selection and ``T`` tokens per forward:

    Eq. 18: ``f_i = (E / (K T)) * sum_t 1[token t routes to expert i]``
    Eq. 19: ``p_i = (1 / T) * sum_t s'_t,i``,
            where ``s'_t,i = s_t,i / sum_j s_t,j`` is the per-token
            normalized score.
    Eq. 17: ``L_bal = sum_i f_i * p_i``

    The returned value is ``T * L_bal`` (token-mode): Eqs 17-20 define a
    per-token-normalized value, while ``AuxLoss`` scales every auxiliary
    loss by ``1 / global_valid_tokens`` (the step's valid-token count), so the
    sum-type form keeps the injected weight at ``coeff * L_bal``.

    The counts (Eq. 18) and normalized-score sums (Eq. 19) are sums over the
    folded token dim, hence Partial over the mesh axes that shard it (CP
    always, plus TP under EP).  They are all-reduced to Invariant before
    the formula, so every rank computes the same per-forward loss.  The
    one-hot counts are non-differentiable: the gradient reaches the router
    only through the normalized-score sums and the top-k score carrier.
    ``T`` is the forward's token count: the code evaluates Eq. 18 in the
    T-free form ``f_i = E * counts_i / sum_j counts_j``, which equals
    ``(E / (K T)) * counts_i`` because each token contributes K entries, so
    ``sum_j counts_j = K T``.  That needs no shape or mesh-degree assumption
    and follows any masking the router applies to the routing map.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        """Same fields as ``AuxLoss.Config``; this loss adds no knobs.

        A distinct Config is required even without new fields: ``Config.build()``
        constructs the class that owns the config (``__init_subclass__`` sets
        ``_owner``), so a router configured with ``AuxLoss.Config`` would
        build a plain ``AuxLoss``, which has no ``forward``.
        """

    def _reduce_token_partials(
        self, partial_E: torch.Tensor, axes: tuple[str, ...]
    ) -> torch.Tensor:
        """Partial -> Invariant all-reduce over the token-partition axes.

        Axes are passed by name, so spmd_types resolves them against the
        ambient mesh and no DeviceMesh escapes into model code; an inactive
        axis is skipped rather than run as a no-op collective.  ``P -> I`` is
        an all-reduce in forward with an identity backward: the reduced sums,
        and hence the loss and its gradient, are identical on every rank of
        the reduction group.
        """
        for axis in axes:
            if spmd_mesh_size(axis) == 1:
                # No mesh context or a size-1 axis: nothing shards the tokens.
                continue
            partial_E = spmd.redistribute(
                partial_E,
                axis,
                src=spmd.Partial,
                dst=spmd.Invariant,
                backward_options={"op_dtype": partial_E.dtype},
            )
        return partial_E

    def forward(
        self,
        scores_TE: torch.Tensor,
        routing_map_TE: torch.Tensor,
        *,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the per-forward balance loss and inject its gradient.

        Args:
            scores_TE: Router scores ``(T, E)`` for the forward's tokens.
            routing_map_TE: One-hot routing map ``(T, E)`` for the same tokens,
                as counted by the router.
            carrier: Tensor whose backward path carries the injected
                gradient (the router's top-k scores).

        Returns:
            ``carrier`` unchanged (identity forward).
        """
        # Mark DP local for the counts arithmetic: each DP rank owns an
        # independent token stream, so DP must not be reduced; only the
        # global axes that shard the stream (CP, TP under EP) are.
        with spmd_local_context("dp"):
            E = scores_TE.size(-1)
            # Axes that shard the router output's token dim: CP in every
            # layout, TP only under EP, which distributes tokens over TP (the
            # gate computes and emits dense_sequence_parallel_placement
            # whenever EP is on, and tokens_per_expert_E is TP-Partial for the
            # same reason).
            axes = ("cp", "tp") if spmd_sparse_mesh() is not None else ("cp",)

            # Eq. 18: per-expert routing frequency counts_i over the forward's
            # tokens, then f_i = E * counts_i / sum_j counts_j (so
            # sum_i f_i = E).  The latter is the (E / (K T)) form with
            # T = sum_j counts_j / K, so it needs no token count, shape or mesh
            # degree and follows any masking the router applies to the map.
            # The map is cast to float before the reduction: casting a Partial
            # tensor is non-linear and rejected by spmd_types.
            counts_E = self._reduce_token_partials(
                routing_map_TE.to(scores_TE.dtype).sum(dim=0), axes
            )
            f_E = F.normalize(counts_E, p=1, dim=0) * E

            # Eq. 19: p_i = (1/T) sum_t s'_t,i, the per-token L1-normalized
            # scores.  F.normalize's eps clamp only guards an all-zero score
            # row: the scores are non-negative, so the norm is a plain sum.
            probs_TE = F.normalize(scores_TE, p=1, dim=-1)
            p_E = self._reduce_token_partials(probs_TE.sum(dim=0), axes)

            # Eq. 17: L_bal = sum_i f_i * p_i
            loss = (f_E * p_E).sum()
            return self.inject(loss, carrier=carrier)


class MoE(Module):
    """Mixture of Experts layer.

    The forward pass proceeds as:
    1. Router computes expert assignments (stays on DTensor)
    2. RoutedExperts.forward() converts DTensor to local, then handles:
       a. dispatch (TokenDispatcher) — reorder tokens by expert assignment.
          With EP, also performs all-to-all communication to send tokens
          to expert-owning ranks.
       b. expert computation (GroupedExperts, local tensors)
       c. combine (TokenDispatcher) — reverse the dispatch reordering.
          - LocalTokenDispatcher (no EP): scatter_add only.
          - AllToAll: all-to-all communication, then scatter_add.
          - DeepEP: combine_tokens followed by backend synchronization.
          - HybridEP: synchronous combine_tokens.
    3. Shared experts compute their output.
    4. Routed and shared expert outputs are summed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int = 8
        routed_experts: RoutedExperts.Config
        router: TokenChoiceTopKRouter.Config
        load_balance_coeff: float | None = 1e-3
        shared_experts: FeedForward.Config | None = None

    def __init__(self, config: Config):
        super().__init__()

        num_experts = config.num_experts
        self.routed_experts = config.routed_experts.build()
        self.router = config.router.build()
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert_E is accumulated in the model forward pass.
        #       expert_bias_E is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = config.load_balance_coeff
        if self.load_balance_coeff is not None and isinstance(
            self.router.aux_loss, MoELoadBalanceAuxLoss
        ):
            raise ValueError(
                "load_balance_coeff and router.aux_loss (MoELoadBalanceAuxLoss) "
                "are mutually exclusive"
            )
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias_E",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias_E = None
        # tokens_per_expert_E will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert_E",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x_TD: torch.Tensor, **router_kwargs) -> torch.Tensor:
        """
        Args:
            x_TD: Input ``(T, D)``.

        Returns:
            Output ``(T, D)``.

        Under TP, the MoE wrapper's ``sharding_config`` (set by
        ``set_moe_sharding_config``) handles input/output redistribution:
        input is redistributed from sp_layout to desired_input_layouts;
        output is redistributed to sp_layout. MoE.forward() operates on
        DTensors; the DTensor->local conversion happens at the GroupedExperts
        boundary. GroupedExperts operates on local tensors. When EP internally
        sequence-shards tokens across TP, the caller must provide a TP-divisible
        token count.
        """
        # topk scores and expert IDs have shape (T, K); the routing map (T, E)
        # marks the experts each token is routed to (built inside the router).
        (
            topk_scores_TK,
            topk_expert_ids_TK,
            routing_map_TE,
        ) = self.router(x_TD, self.expert_bias_E, **router_kwargs)
        num_local_tokens_per_expert_E = (
            routing_map_TE.sum(dim=0)
            if routing_map_TE is not None
            else _tokens_per_expert(
                topk_expert_ids_TK, self.tokens_per_expert_E.numel()
            )
        )

        # tokens_per_expert_E will be used to update the expert bias for load balancing,
        # and also to count the expert usage.
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert_E --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        if self.training:
            with torch.no_grad():
                self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

        out_TD = self.routed_experts(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )

        shared_out_TD = (
            self.shared_experts(x_TD) if self.shared_experts is not None else None
        )

        if shared_out_TD is not None:
            out_TD = out_TD + shared_out_TD
        return out_TD

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            # After ``to_empty()``, the existing buffer records the target device.
            # Reinitialize MoE counters there when no explicit buffer device is passed.
            buffer_device = self.tokens_per_expert_E.device

        with torch.device(buffer_device):
            self.tokens_per_expert_E = torch.zeros(
                self.routed_experts.inner_experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias_E = torch.zeros(
                    self.routed_experts.inner_experts.num_experts, dtype=torch.float32
                )
