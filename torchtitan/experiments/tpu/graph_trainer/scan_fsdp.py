"""All-gather one decoder layer's weights, inside the scan body.

FSDP gathers on attribute access. Reading a stacked parameter therefore pulls
the whole ``[n_layers, ...]`` tensor, before the loop that consumes it, so
every layer's weights are resident at once. That stops fitting past a few tens
of billions of parameters.

Here the body takes this rank's shard of one layer and gathers that instead --
``n_layers`` times smaller. A functional collective traces into a scan body, so
the gather lowers into the loop and lands next to the compute that needs it.

``tpu_fsdp.unshard`` is what the body calls on one row; its spec is the stack's
``ShardSpec.layer_slice()`` (``layer_stack.scan_plan``). This module keeps the
funcol patch the eager SimpleFSDP lanes need (collectives moved onto dim 0,
2-D shards folded to 8-row tiles; ``tpu_fsdp`` does the same inside its own
collectives) and the shard-axis planning for stacks and vocab-sized weights.

Recomputing this gather in the backward is safe, despite upstream's *Note
[Recomputing collectives in the partitioner]*: what that bans is a duplication
some ranks make and others don't, and every rank here compiles the same SPMD
graph.
"""

from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
from torch import nn

from torchtitan.experiments.tpu.graph_trainer.tpu_fsdp import _tile_fold
from torchtitan.tools.logging import logger


__all__ = ["configure_simple_fsdp_shard_dims"]


_orig_all_gather_single = funcol.all_gather_single
_orig_reduce_scatter_single = funcol.reduce_scatter_single


def _movedim_all_gather_tensor(
    self: torch.Tensor,
    gather_dim: int,
    group: Any,
    tag: str = "",
) -> torch.Tensor:
    """TPU-friendly drop-in for ``funcol.all_gather_single`` / ``all_gather_tensor``.

    Installed on import for the eager SimpleFSDP lanes, whose collectives come
    out of DTensor's ``redistribute`` and so cannot be rewritten at the call
    site. The graph trainer's ``tpu_fsdp`` does both fixes in its own
    ``_gather_on`` / ``_scatter_on`` and does not depend on this patch (it is
    idempotent on an already folded, dim-0 gather).

    1. Gathering on ``gather_dim > 0``: PyTorch's default splits the output
       into ``world_size`` chunks and concatenates them along ``gather_dim``,
       thousands of slice/concat ops at scale. Transpose the axis to dim 0,
       run one contiguous dim-0 all-gather, transpose back.
    2. Unaligned 2-D shards (``rows % 8 != 0``): fold columns into rows so
       the collective is 8-row aligned (``tpu_fsdp._tile_fold``).
    """
    if gather_dim < 0:
        gather_dim += self.ndim
    t = self.movedim(gather_dim, 0).contiguous() if gather_dim != 0 else self
    t, orig_cols = _tile_fold(t, t.shape[0])
    out = _orig_all_gather_single(t, gather_dim=0, group=group, tag=tag)
    if orig_cols is not None:
        out = out.reshape(-1, orig_cols)
    return out.movedim(0, gather_dim) if gather_dim != 0 else out


def _movedim_reduce_scatter_tensor(
    self: torch.Tensor,
    reduceOp: str,
    scatter_dim: int,
    group: Any,
    tag: str = "",
) -> torch.Tensor:
    """TPU-friendly drop-in for ``funcol.reduce_scatter_single`` / ``reduce_scatter_tensor``.

    The backward counterpart of ``_movedim_all_gather_tensor``: scatter on
    dim 0, and fold when the per-rank output would have ``rows % 8 != 0``.
    """
    if scatter_dim < 0:
        scatter_dim += self.ndim
    t = self.movedim(scatter_dim, 0).contiguous() if scatter_dim != 0 else self
    orig_cols = None
    if t.ndim == 2:
        try:
            group_size = funcol.c10d._get_group_size_by_name(
                funcol._resolve_group(group, tag)
            )
        except Exception:  # pylint: disable=broad-except
            group_size = 0
        if group_size > 0 and t.shape[0] % group_size == 0:
            t, orig_cols = _tile_fold(t, t.shape[0] // group_size)
    out = _orig_reduce_scatter_single(
        t, reduceOp=reduceOp, scatter_dim=0, group=group, tag=tag
    )
    if orig_cols is not None:
        out = out.reshape(-1, orig_cols)
    return out.movedim(0, scatter_dim) if scatter_dim != 0 else out


funcol.all_gather_single = _movedim_all_gather_tensor
funcol.all_gather_tensor = _movedim_all_gather_tensor
funcol.reduce_scatter_single = _movedim_reduce_scatter_tensor
funcol.reduce_scatter_tensor = _movedim_reduce_scatter_tensor


def configure_simple_fsdp_shard_dims(
    model: nn.Module,
    parallel_dims: Any,
) -> list[str]:
    """Picks FSDP shard axes that divide evenly, for any decoder model.

    Plans an axis per layer stack (calling ``flatten_for`` first if present) and
    moves ``tok_embeddings`` / ``lm_head`` to their hidden axis (dim 1) when the
    vocabulary does not divide the FSDP degree (e.g. Llama 3's 128,256 over 512+ ranks).

    Args:
        model: The model to configure, before ``apply_tpu_fsdp``.
        parallel_dims: Parallel dimensions; FSDP degree is ``dp_shard * cp``.

    Returns:
        FQNs of the modules moved to dim 1.
    """
    if parallel_dims is None:
        return []
    fsdp_degree = getattr(parallel_dims, "dp_shard", 1) * getattr(
        parallel_dims, "cp", 1
    )
    if fsdp_degree <= 1:
        return []

    # Layer stacking moves the layer weights into a holder, which plans an axis
    # per stack -- dim 1 of a stack is one axis further in than dim 1 of a layer.
    for holder in model.modules():
        if hasattr(holder, "plan_shard_dims"):
            if hasattr(holder, "flatten_for"):
                holder.flatten_for(fsdp_degree)
            holder._fsdp_shard_dim = holder.plan_shard_dims(fsdp_degree)

    hidden_axis_modules: list[str] = []
    for module_fqn in ("tok_embeddings", "lm_head"):
        module = getattr(model, module_fqn, None)
        if module is None or not hasattr(module, "weight"):
            continue
        weight = module.weight
        if weight.shape[0] % fsdp_degree == 0:
            continue
        if weight.ndim != 2 or (
            weight.shape[1] % fsdp_degree != 0 and weight.shape[1] >= fsdp_degree
        ):
            raise ValueError(
                f"Cannot FSDP-shard {module_fqn}.weight with shape "
                f"{tuple(weight.shape)} over {fsdp_degree} ranks"
            )
        # This is the per-submodule shard-axis hook `apply_tpu_fsdp` reads,
        # also used by LayerStackHolder for stacked layer weights. When
        # `weight.shape[1] < fsdp_degree` (e.g. Llama 3 8B at 8192 ranks),
        # `shard_module` downgrades `Shard(1)` to `Replicate()`.
        if weight.shape[1] < fsdp_degree:
            replicated_mib = weight.numel() * weight.element_size() / (1024**2)
            logger.warning(
                "%s.weight with shape %s has hidden dim %d < fsdp_degree %d; "
                "FSDP will replicate it (%.1f MiB parameter) instead of "
                "sharding.",
                module_fqn,
                tuple(weight.shape),
                weight.shape[1],
                fsdp_degree,
                replicated_mib,
            )
        setattr(module, "_fsdp_shard_dim", 1)
        hidden_axis_modules.append(module_fqn)
    return hidden_axis_modules
