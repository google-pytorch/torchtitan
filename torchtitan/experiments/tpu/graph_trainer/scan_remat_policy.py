"""Which values the scan body keeps for its backward, chosen by region.

The scan body is split into forward and backward inside ``scan``'s own
autograd, before any GraphTrainer memory policy runs. So while ``scan``
traces, ``scan_body_remat_window`` tags each forward node of the body from its
``remat_region`` (see ``remat_regions.py``):

* ``fsdp.gather``: recompute. Saving it would keep every layer's whole weight.
* ``moe.expert_counts``: save. Redoing the all-reduce is an exposed round trip.
* a region in ``save_regions``, e.g. ``attention.inner_attention``: save.
* everything else: recompute.
"""

import contextlib
from collections.abc import Callable, Iterator, Set

import torch
from torch import fx
from torch._functorch import config as functorch_config
from torch._functorch import partitioners
from torch._higher_order_ops import partitioner as hop_partitioner
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.tpu.remat_regions import (
    ATTENTION_INNER_ATTENTION,
    FSDP_GATHER,
    MOE_EXPERT_COUNTS,
    REMAT_REGION,
)

__all__ = [
    "SCAN_SAVE_REGIONS",
    "scan_body_remat_window",
    "tag_scan_body_joint_graph",
]

# The regions a scan body can be asked to save.
SCAN_SAVE_REGIONS = frozenset({ATTENTION_INNER_ATTENTION})


def _region(node: fx.Node) -> str | None:
    return (node.meta.get("custom") or {}).get(REMAT_REGION)


def _depends_on_tangents(graph: fx.Graph) -> set[fx.Node]:
    """Nodes only backward can compute: ``tangents_*`` and what reads them."""
    closure: set[fx.Node] = set()
    for node in graph.nodes:
        if node.op == "placeholder":
            if str(node.target).startswith("tangents"):
                closure.add(node)
        elif any(i in closure for i in node.all_input_nodes):
            closure.add(node)
    return closure


def tag_scan_body_joint_graph(graph: fx.Graph, save_regions: Set[str]) -> None:
    """Set ``meta["recompute"]`` on every forward node, from its region.

    Nodes that already have a tag keep it: ``scan`` partitions each body twice,
    and the HOP partitioner pre-tags complex SymInt expressions.

    Raises if the body has a collective or ``save_regions`` but no region
    meta: then ``scan`` dropped it while re-tracing the body, and the counts
    and saves would silently be recomputed.
    """
    if not any(_region(n) for n in graph.nodes):
        has_collective = any(
            isinstance(n.target, torch._ops.OpOverload)
            and n.target.namespace == "_c10d_functional"
            for n in graph.nodes
        )
        if save_regions or has_collective:
            raise RuntimeError(
                "the scan body's joint graph has no remat_region meta; is "
                "scan_body_remat_window's re-trace patch still in effect?"
            )
    backward_only = _depends_on_tangents(graph)
    for node in graph.nodes:
        if node.op != "call_function" or node in backward_only:
            continue
        if node.meta.get("recompute") is not None:
            continue
        region = _region(node)
        save = region != FSDP_GATHER and (
            region == MOE_EXPERT_COUNTS or region in save_regions
        )
        node.meta["recompute"] = (
            CheckpointPolicy.MUST_SAVE if save else CheckpointPolicy.MUST_RECOMPUTE
        )


def _keeping_meta(fn: Callable) -> Callable:
    """Run a traced graph so its re-trace copies each node's meta.

    TODO: drop once torch's ``create_hop_joint_graph`` does this itself (it has
    ``_maybe_run_with_interpreter`` for it).
    """
    if not isinstance(fn, fx.GraphModule):
        return fn

    def run(*args):
        with fx.traceback.preserve_node_meta():
            return fx.Interpreter(fn).run(*args)

    return run


@contextlib.contextmanager
def scan_body_remat_window(save_regions: Set[str] = frozenset()) -> Iterator[None]:
    """Wrap a ``scan`` call to partition its body by region.

    ``create_hop_joint_graph`` re-traces the body twice, and each re-trace drops
    node meta; both get patched to keep it. Then the partitioner tags the joint
    graph and runs at budget 1.0, where every tag is honoured. Everything is
    process-global, so it is a window and restored on exit.
    """
    unknown = set(save_regions) - SCAN_SAVE_REGIONS
    if unknown:
        raise ValueError(f"a scan body cannot save regions {sorted(unknown)}")
    min_cut = partitioners.min_cut_rematerialization_partition
    create_bw_fn = hop_partitioner.create_bw_fn
    functionalize = torch.func.functionalize

    def partition(joint_module: fx.GraphModule, joint_inputs, *args, **kwargs):
        tag_scan_body_joint_graph(joint_module.graph, save_regions)
        with functorch_config.patch(activation_memory_budget=1.0):
            return min_cut(joint_module, joint_inputs, *args, **kwargs)

    partitioners.min_cut_rematerialization_partition = partition
    hop_partitioner.create_bw_fn = lambda fn, *a, **k: create_bw_fn(
        _keeping_meta(fn), *a, **k
    )
    torch.func.functionalize = lambda fn, *a, **k: functionalize(
        _keeping_meta(fn), *a, **k
    )
    try:
        yield
    finally:
        partitioners.min_cut_rematerialization_partition = min_cut
        hop_partitioner.create_bw_fn = create_bw_fn
        torch.func.functionalize = functionalize
