"""Which values the scan body keeps for its backward, chosen by region.

``scan`` splits its body into forward and backward while it traces, before any
GraphTrainer memory policy runs. So ``scan_with_remat_policy`` runs the body
under selective activation checkpointing, and the policy decides from each
op's ``remat_region`` (see ``remat_regions.py``):

* ``moe.expert_counts``: save. Redoing the all-reduce is an exposed round trip.
* a region in ``save_regions``, e.g. ``attention.inner_attention``: save.
* everything else, ``fsdp.gather`` included: recompute. Saving the gather
  would keep every layer's whole weight.
"""

import functools
from collections.abc import Callable, Set

import torch
import torch.utils.checkpoint as checkpoint_module
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


def scan_with_remat_policy(
    body: Callable, init, xs, save_regions: Set[str] = frozenset()
):
    """``scan(body, init, xs)``, with the body saving only what the policy picks.

    Raises if a requested save region matches no op in the body, e.g.
    ``attention.inner_attention`` with SDPA instead of Splash.
    """
    unknown = set(save_regions) - SCAN_SAVE_REGIONS
    if unknown:
        raise ValueError(f"a scan body cannot save regions {sorted(unknown)}")
    if not hasattr(checkpoint_module, "_tag_recompute_for_partitioner"):
        # Then, under the graph trainer's non-strict trace, checkpoint takes its
        # eager path, no node is tagged, and the body silently saves the
        # gathered weights of every layer.
        raise RuntimeError(
            "--compile.use-scan-layers needs PyTorch with pytorch/pytorch#203334 and "
            "#203534 (not in 2.15). Found torch "
            f"{torch.__version__}."
        )
    # Mutation allowed: the MoE aux loss all-reduces its saved counts in place,
    # once per mesh axis.
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
