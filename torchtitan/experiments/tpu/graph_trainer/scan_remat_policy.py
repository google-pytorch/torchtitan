"""Which values the scan body keeps across its forward/backward cut.

``LayerStack.run`` fully rematerializes the scan body: its backward recomputes
the whole forward from the body's inputs. That is what keeps the in-body FSDP
gather from being carried out of the loop, and it is done with an
``activation_memory_budget=0.0`` window around the ``scan`` call.

Budget 0.0 is absolute, though. The partitioner returns the inputs without
looking at any tag (``choose_saved_values_set``: ``if memory_budget == 0: return
node_info.inputs``), so *every* collective in the body is redone in backward.
For the MoE batch-wise aux loss that includes the global expert-count
all-reduce, which spans DP -- and so DCN on a multi-slice run. Its result is
consumed straight away by the router-score gradient, so in backward it is a
fully exposed, per-layer DCN round trip for 128 floats.

This module keeps the budget-0.0 behaviour for everything except those counts:

* All-reduces whose operand is an expert count (``histc``/``bincount``, or a
  sum over the router's boolean routing map, through dtype casts, views and
  earlier count all-reduces) are tagged
  ``MUST_SAVE``. Other collectives on counts -- expert parallelism's routing
  count ``all_to_all_single`` -- are not.
* Every other forward node is tagged ``MUST_RECOMPUTE``, which is what budget
  0.0 did to it anyway, the FSDP gather included.
* The partition then runs at budget 1.0, where the min-cut honours those tags.

The saved counts are the values backward would have recomputed: the same
collective over the same deterministic inputs. The math is unchanged; only
where the value comes from moves.

A body with no count all-reduce -- no batch-wise aux loss, with or without
expert parallelism -- is partitioned exactly as before, at 0.0 and untagged.

Tags are applied to the joint graph just before it is partitioned rather than
at trace time, because the body's re-trace drops ``node.meta["custom"]`` (see
``LayerStack.run``). The hook goes in by swapping
``partitioners.min_cut_rematerialization_partition`` for the window's duration:
``HopGraphMinCutPartitioner.create_partitioned_graph`` imports it at call time.
"""

import contextlib
from collections.abc import Iterator

import torch
from torch import fx
from torch._functorch import config as functorch_config
from torch._functorch import partitioners
from torch.utils.checkpoint import CheckpointPolicy

__all__ = [
    "count_collectives",
    "scan_body_remat_window",
    "tag_scan_body_joint_graph",
]

_aten = torch.ops.aten

# Where an expert count is born: from expert ids, or (see `_sums_a_routing_map`)
# by summing the router's one-hot routing map.
_COUNT_SOURCES = frozenset({_aten.histc.default, _aten.bincount.default})
_SUMS = frozenset({_aten.sum.dim_IntList, _aten.sum.default})

# Ops a count may pass through on its way to (and between) the collectives
# without becoming something else: casts, views and copies. `.to(dtype)` in
# `_batch_wise_load_balance_loss` is the one that matters.
_COUNT_PASSTHROUGH = frozenset(
    {
        _aten._to_copy.default,
        _aten.clone.default,
        _aten.alias.default,
        _aten.view.default,
        _aten.reshape.default,
        _aten._unsafe_view.default,
        _aten.detach.default,
        _aten.copy.default,
        torch.ops.prims.convert_element_type.default,
    }
)

# The collectives a count may flow through: the all-reduces that sum it over the
# token-partition axes, and their waits. Deliberately not `all_to_all_single`:
# `_tokens_per_expert` also starts at a `histc`, and with expert parallelism its
# routing counts go through one (`_token_count_exchange`). Saving that exchange
# is not what this is for, and picking it up would move every expert-parallel
# MoE body off the budget-0.0 path even with no aux loss.
_COUNT_COLLECTIVES = frozenset(
    {
        torch.ops._c10d_functional.all_reduce.default,
        torch.ops._c10d_functional.all_reduce_.default,
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        torch.ops._c10d_functional.wait_tensor.default,
    }
)


def _is_bool(node: fx.Node) -> bool:
    val = node.meta.get("val")
    return isinstance(val, torch.Tensor) and val.dtype == torch.bool


def _sums_a_routing_map(node: fx.Node) -> bool:
    """A sum over a boolean (one-hot routing map) tensor, through casts/views.

    The MoE aux losses count tokens per expert this way: MoELoadBalanceAuxLoss
    as ``routing_map.sum(...).to(dtype)``, upstream's
    MicrobatchWiseLoadBalanceLoss as ``routing_map.to(dtype).sum(...)``.
    """
    if node.op != "call_function" or node.target not in _SUMS:
        return False
    src = node.args[0]
    while isinstance(src, fx.Node):
        if _is_bool(src):
            return True
        if src.op != "call_function" or src.target not in _COUNT_PASSTHROUGH:
            return False
        src = src.args[0]
    return False


def _is_count_source(node: fx.Node) -> bool:
    return node.op == "call_function" and (
        node.target in _COUNT_SOURCES or _sums_a_routing_map(node)
    )


def _is_collective(node: fx.Node) -> bool:
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace == "_c10d_functional"
    )


def _reached_from_counts(graph: fx.Graph) -> set[fx.Node]:
    """Count sources and everything a count flows to unmixed.

    Flows forward from each count source (``histc``/``bincount``, or a sum over
    a boolean routing map) through casts, views, all-reduces and waits only.
    Anything that mixes the count with another value ends the flow, and so does
    any other collective.
    """
    reached: set[fx.Node] = set()
    stack = [n for n in graph.nodes if _is_count_source(n)]
    while stack:
        node = stack.pop()
        if node in reached:
            continue
        reached.add(node)
        for user in node.users:
            if user.op == "call_function" and (
                user.target in _COUNT_COLLECTIVES
                or user.target in _COUNT_PASSTHROUGH
            ):
                stack.append(user)
    return reached


def count_collectives(graph: fx.Graph) -> list[fx.Node]:
    """The all-reduces in ``graph`` that carry expert counts, in graph order.

    An all-reduce is picked up only if what it reduces *is* a count; see
    `_reached_from_counts`.
    """
    reached = _reached_from_counts(graph)
    return [n for n in graph.nodes if n in reached and _is_collective(n)]


def _count_values_to_save(graph: fx.Graph) -> list[fx.Node]:
    """The count collectives, and every count value downstream of one.

    Saving just the collectives is not enough. An in-place ``dist.all_reduce``
    -- what `_all_reduce_expert_counts` does -- functionalizes to
    ``aten.copy(dst=<counts before the reduce>, src=<reduced>)``, and a
    recomputed copy needs its ``dst``, which drags the count source back into
    backward. Everything after the first count all-reduce is an ``[E]``-sized
    count, so saving all of it costs nothing.
    """
    reached = _reached_from_counts(graph)
    after: list[fx.Node] = []
    after_set: set[fx.Node] = set()
    for node in graph.nodes:
        if node not in reached:
            continue
        if _is_collective(node) or any(
            i in after_set for i in node.all_input_nodes
        ):
            after.append(node)
            after_set.add(node)
    return after


def _depends_on_tangents(graph: fx.Graph) -> set[fx.Node]:
    """Nodes downstream of a tangent, i.e. the ones only backward can compute.

    ``HopJointGraph`` names the body's cotangent inputs ``tangents_*``.
    """
    closure: set[fx.Node] = set()
    for node in graph.nodes:
        if node.op == "placeholder":
            if str(node.target).startswith("tangents"):
                closure.add(node)
        elif any(i in closure for i in node.all_input_nodes):
            closure.add(node)
    return closure


def tag_scan_body_joint_graph(graph: fx.Graph) -> list[fx.Node]:
    """Tag a scan body's joint graph to save its expert counts, and only those.

    Returns the nodes tagged ``MUST_SAVE``: the count all-reduces and the count
    values after them. If there are none, the graph is left untouched: the
    caller should partition it as before.

    Tags already present are kept. The HOP partitioner sets ``MUST_RECOMPUTE``
    on complex SymInt expressions before calling the partition function, and
    that must stand.
    """
    to_save = _count_values_to_save(graph)
    if not to_save:
        return []
    save = set(to_save)
    backward_only = _depends_on_tangents(graph)
    for node in graph.nodes:
        if node.op != "call_function" or node in backward_only:
            continue
        if node.meta.get("recompute") is not None:
            continue
        node.meta["recompute"] = (
            CheckpointPolicy.MUST_SAVE
            if node in save
            else CheckpointPolicy.MUST_RECOMPUTE
        )
    return to_save


def _collective_group_name(node: fx.Node) -> str | None:
    """Return the process-group name of a non-wait `_c10d_functional` op."""
    if not _is_collective(node) or (
        node.target is torch.ops._c10d_functional.wait_tensor.default
    ):
        return None
    for arg in reversed(node.args):
        if isinstance(arg, str):
            return arg
    return None


def _hoist_initial_group_collectives(
    fw_module: fx.GraphModule,
    bw_module: fx.GraphModule,
) -> None:
    """Ensure `bw_module` starts with the same collective group as `fw_module`.

    TorchTPU resolves a scan body's XLA device assignment from the first
    collective in creation (FX node) order and requires forward and backward
    scans in the same program to resolve to the same device assignment. In
    `min_cut_rematerialization_partition`, `reordering_to_mimic_autograd_engine`
    walks from the first user of `tangents_*`, which for an EP MoE layer is the
    combine backward's EP `all_gather_into_tensor` (depending only on
    `tangents_*`), placing that EP collective before the recomputed FSDP
    parameter `all_gather_into_tensor`s (which depend only on `bw_module`
    placeholders). Hoisting the independent `fw_group` collectives (and their
    `wait_tensor`s and input chains) to right after `bw_module`'s placeholders
    restores the forward collective group ordering without disturbing the
    per-chunk rematerialization schedule.
    """
    fw_group = next(
        (
            group
            for node in fw_module.graph.nodes
            if (group := _collective_group_name(node)) is not None
        ),
        None,
    )
    if fw_group is None:
        return

    bw_first_group = next(
        (
            group
            for node in bw_module.graph.nodes
            if (group := _collective_group_name(node)) is not None
        ),
        None,
    )
    if bw_first_group is None or bw_first_group == fw_group:
        return

    blocked: set[fx.Node] = _depends_on_tangents(bw_module.graph)
    for node in bw_module.graph.nodes:
        group = _collective_group_name(node)
        if (group is not None and group != fw_group) or any(
            inp in blocked for inp in node.all_input_nodes
        ):
            blocked.add(node)

    targets: list[fx.Node] = []
    for node in bw_module.graph.nodes:
        if node in blocked or _collective_group_name(node) != fw_group:
            continue
        targets.append(node)
        for user in node.users:
            if (
                user not in blocked
                and user.op == "call_function"
                and user.target
                is torch.ops._c10d_functional.wait_tensor.default
            ):
                targets.append(user)
    if not targets:
        return

    closure: set[fx.Node] = set()
    stack = list(targets)
    while stack:
        node = stack.pop()
        if node in closure or node.op in ("placeholder", "get_attr"):
            continue
        closure.add(node)
        stack.extend(node.all_input_nodes)

    ordered = [node for node in bw_module.graph.nodes if node in closure]
    anchor = next(
        (
            node
            for node in reversed(bw_module.graph.nodes)
            if node.op in ("placeholder", "get_attr")
        ),
        None,
    )
    if anchor is None or not ordered:
        return
    for node in ordered:
        anchor.append(node)
        anchor = node
    bw_module.graph.lint()
    bw_module.recompile()


@contextlib.contextmanager
def scan_body_remat_window() -> Iterator[None]:
    """Fully rematerialize a scan body, except for its expert counts.

    Wrap the ``scan`` call in this. Both the config and the partitioner are
    process-global, so it has to be a window; both are restored on exit.
    """
    inner = partitioners.min_cut_rematerialization_partition

    def partition(joint_module: fx.GraphModule, joint_inputs, *args, **kwargs):
        if not tag_scan_body_joint_graph(joint_module.graph):
            # No counts to keep: exactly the old path, at the window's 0.0.
            fw_module, bw_module = inner(
                joint_module, joint_inputs, *args, **kwargs
            )
        else:
            # Every forward node now carries a tag, so the min-cut has nothing
            # left to choose; 1.0 just stops the 0.0 early return from ignoring
            # the tags.
            with functorch_config.patch(activation_memory_budget=1.0):
                fw_module, bw_module = inner(
                    joint_module, joint_inputs, *args, **kwargs
                )
        _hoist_initial_group_collectives(fw_module, bw_module)
        return fw_module, bw_module

    with functorch_config.patch(activation_memory_budget=0.0):
        partitioners.min_cut_rematerialization_partition = partition
        try:
            yield
        finally:
            partitioners.min_cut_rematerialization_partition = inner
