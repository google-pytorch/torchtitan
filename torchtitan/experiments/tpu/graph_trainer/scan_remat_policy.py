"""Which values the scan body keeps for its backward, chosen by region.

``scan`` splits its body into forward and backward while it traces, before any
GraphTrainer memory policy runs. So ``scan_with_remat_policy`` runs the body
under selective activation checkpointing, and the policy decides from each
op's ``remat_region`` (see ``remat_regions.py``):

* ``moe.expert_counts``: save. Redoing the all-reduce is an exposed round trip.
* a region in ``save_regions``, e.g. ``attention.inner_attention``: save.
* everything else, the FSDP all-gather included: recompute. Saving the
  gather would keep every layer's whole weight.
"""

import functools
from collections.abc import Callable, Set

import torch
from torch import fx
from torch._higher_order_ops.scan import scan
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from torchtitan.experiments.tpu.components.remat_regions import (
    ATTENTION_INNER_ATTENTION,
    MOE_EXPERT_COUNTS,
    REMAT_REGION,
)

__all__ = ["SCAN_SAVE_REGIONS", "scan_body_policy", "scan_with_remat_policy"]

# The regions a scan body can be asked to save.
SCAN_SAVE_REGIONS = frozenset({ATTENTION_INNER_ATTENTION})


def scan_body_policy(
    save_regions: Set[str], saved: set[str] | None = None
) -> Callable:
    """The SAC policy: save the expert counts and ``save_regions``, recompute the rest.

    Adds each region it saves to ``saved``, if given.
    """
    save = {MOE_EXPERT_COUNTS, *save_regions}

    def policy(ctx, op, *args, **kwargs) -> CheckpointPolicy:
        custom = fx.traceback.get_current_meta().get("custom") or {}
        region = custom.get(REMAT_REGION)
        if region in save:
            if saved is not None:
                saved.add(region)
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.MUST_RECOMPUTE

    return policy


def _torch_has_scan_sac_fixes() -> bool:
    """Whether PyTorch has pytorch/pytorch#203334 and #203534.

    `_tag_recompute_for_partitioner` is #203534's switch. #203334's fix (re-running the
    body through an interpreter so node meta survives) adds no new name;
    `HopJointGraph._mark_rng_ops_as_must_save` was added in the same commit.
    """
    from torch._higher_order_ops import partitioner
    from torch.utils import checkpoint
    return hasattr(checkpoint, "_tag_recompute_for_partitioner") and hasattr(
        partitioner.HopJointGraph, "_mark_rng_ops_as_must_save"
    )


def scan_with_remat_policy(
    body: Callable, init, xs, save_regions: Set[str] = frozenset()
):
    """``scan(body, init, xs)``, with the body saving only what the policy picks.

    Raises if a requested save region matches no op in the body, e.g.
    ``attention.inner_attention`` with SDPA instead of Splash.
    """
    if not _torch_has_scan_sac_fixes():
        raise RuntimeError(
            "--compile.use-scan-layers needs PyTorch with pytorch/pytorch#203334 "
            "and #203534 (nightly >= 2.16.0.dev20261120); on older PyTorch the "
            "scan body would keep every layer's gathered weights."
        )
    unknown = set(save_regions) - SCAN_SAVE_REGIONS
    if unknown:
        raise ValueError(f"a scan body cannot save regions {sorted(unknown)}")
    if not torch.is_grad_enabled():
        # No backward (validation, inference): checkpoint would not run the policy
        # (it yields early without grad), so run plain scan and skip the region check.
        return scan(body, init, xs)
    # Mutation allowed: the MoE aux loss all-reduces its saved counts in place,
    # once per mesh axis. This turns off SAC's mutation check for the whole
    # body. Safe: scan traces and functionalizes the body to build its backward,
    # so backward reads each saved value as it was when saved.
    saved: set[str] = set()
    context_fn = functools.partial(
        create_selective_checkpoint_contexts,
        scan_body_policy(save_regions, saved),
        allow_cache_entry_mutation=True,
    )

    def checkpointed(*args):
        return checkpoint(body, *args, use_reentrant=False, context_fn=context_fn)

    # No budget pin and no preserve_node_meta: min-cut honours every tag at
    # torch's default budget (1.0), and scan's partitioner keeps node meta
    # while it re-traces the body.
    result = scan(checkpointed, init, xs)
    missing = set(save_regions) - saved
    if missing:
        raise ValueError(
            f"Save region(s) {sorted(missing)} matched no op in the scan body; "
            "the model or its tracing no longer exposes them (e.g. SDPA "
            "instead of Splash for attention.inner_attention)."
        )
    return result
