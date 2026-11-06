"""FX pass: bound in-flight chunks in sequence-chunked MoE EP layers.

When ``TpuQwen35RoutedExperts`` splits ``T`` tokens into ``C`` sequence chunks,
each chunk runs ``dispatch -> inner_experts -> combine`` independently. Without
cross-chunk scheduling edges, XLA can launch all ``C`` chunks' collectives and
gathers ahead of their consumers (especially in activation-rematerialized
backward), keeping ``C`` copies of the ``(ep_size * (T / C) * top_k, D)``
expert buffers live simultaneously (``O(ep_size * T * top_k)`` HBM instead of
``O(T * top_k)``).

Inserting ``_optimization_barrier`` between chunk ``k - W``'s exits (including
any completed intra-layer gradient accumulation ops) and chunk ``k``'s entry
inputs bounds simultaneously live chunks to ``W`` (default ``W = 2`` for
double-buffered SparseCore/TensorCore pipelining) in both forward and backward.
"""

from __future__ import annotations

from collections import defaultdict
import operator
from typing import Any

import torch

from torchtitan.experiments.graph_trainer.common_utils import (
    _MODULE_FQN,
    _get_layer_id,
    _get_module_fqn,
    _is_backward_node,
)
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    _optimization_barrier,
)
from torchtitan.tools.logging import logger


_CHUNK_IDX_KEY = "ep_chunk_idx"
_MAX_INFLIGHT_KEY = "ep_max_inflight_chunks"
_DEFAULT_MAX_INFLIGHT_CHUNKS = 2


def _chunk_idx(node: torch.fx.Node) -> int | None:
    custom = node.meta.get("custom")
    if not isinstance(custom, dict) or _CHUNK_IDX_KEY not in custom:
        return None
    return int(custom[_CHUNK_IDX_KEY])


def _max_inflight_chunks(node: torch.fx.Node) -> int:
    custom = node.meta.get("custom")
    if not isinstance(custom, dict):
        return _DEFAULT_MAX_INFLIGHT_CHUNKS
    return max(1, int(custom.get(_MAX_INFLIGHT_KEY, _DEFAULT_MAX_INFLIGHT_CHUNKS)))


def apply_ep_chunk_barrier_pass(
    gm: torch.fx.GraphModule,
    example_inputs: Any = None,
) -> torch.fx.GraphModule:
    """Insert ``_optimization_barrier`` to cap live MoE EP sequence chunks."""
    del example_inputs
    graph = gm.graph
    nodes = list(graph.nodes)
    order = {node: idx for idx, node in enumerate(nodes)}

    by_layer_phase: dict[tuple[int, bool], dict[int, list[torch.fx.Node]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    # Unchunked nodes of each (layer, phase) in graph order: the only
    # candidates for "completed after chunk k - W", so each chunk scans its own
    # layer's bucket instead of the whole joint graph. PyTorch autograd does
    # not attach ``seq_nr`` or ``autograd_backward`` to cross-chunk weight-grad
    # accumulation ``aten.add`` ops (``setup_stacktrace_preservation_hooks``
    # resets both in the node posthook before ``InputBuffer::add`` runs), so
    # unchunked nodes with ``layer_id < 0`` inherit their bucket when all of
    # their input nodes belong to the same ``(layer_id, is_bwd)`` bucket.
    unchunked: dict[tuple[int, bool], list[torch.fx.Node]] = defaultdict(list)
    node_key: dict[torch.fx.Node, tuple[int, bool]] = {}
    for node in nodes:
        if node.op != "call_function":
            continue
        cidx = _chunk_idx(node)
        layer_id = _get_layer_id(node)
        if cidx is not None:
            if layer_id < 0:
                continue
            key = (layer_id, _is_backward_node(node))
            by_layer_phase[key][cidx].append(node)
            node_key[node] = key
            continue
        if not node.all_input_nodes:
            continue
        if layer_id >= 0:
            key = (layer_id, _is_backward_node(node))
        else:
            inp_keys = {node_key.get(inp) for inp in node.all_input_nodes}
            if len(inp_keys) != 1 or None in inp_keys:
                continue
            key = next(iter(inp_keys))  # type: ignore[assignment]
        unchunked[key].append(node)
        node_key[node] = key

    if not by_layer_phase:
        return gm

    plans: list[
        tuple[
            bool,
            torch.fx.Node,
            dict[torch.fx.Node, list[torch.fx.Node]],
            list[torch.fx.Node],
        ]
    ] = []

    for (layer_id, is_bwd), chunk_map in by_layer_phase.items():
        ordered_chunks = sorted(
            chunk_map.values(),
            key=lambda grp: order[grp[0]],
        )
        max_inflight = _max_inflight_chunks(ordered_chunks[0][0])
        if len(ordered_chunks) <= max_inflight:
            continue

        completed_prev: set[torch.fx.Node] = set()
        for seq_pos in range(max_inflight, len(ordered_chunks)):
            curr_nodes = ordered_chunks[seq_pos]
            curr_set = set(curr_nodes)
            first_curr = curr_nodes[0]
            first_curr_pos = order[first_curr]

            prev_chunk_nodes = ordered_chunks[seq_pos - max_inflight]
            if order[prev_chunk_nodes[-1]] >= first_curr_pos:
                continue

            completed_curr: set[torch.fx.Node] = set().union(
                *ordered_chunks[: seq_pos - max_inflight + 1]
            )
            for candidate in unchunked[(layer_id, is_bwd)]:
                if order[candidate] >= first_curr_pos:
                    break
                if candidate not in completed_curr and all(
                    inp in completed_curr for inp in candidate.all_input_nodes
                ):
                    completed_curr.add(candidate)

            newly_completed = completed_curr - completed_prev
            completed_prev = completed_curr

            prev_exits = [
                exit_node
                for exit_node in sorted(newly_completed, key=order.__getitem__)
                if order[exit_node] < first_curr_pos
                and isinstance(exit_node.meta.get("val"), torch.Tensor)
                and any(user not in completed_curr for user in exit_node.users)
            ]
            if not prev_exits:
                continue

            entry_users: dict[torch.fx.Node, list[torch.fx.Node]] = {}
            for curr_node in curr_nodes:
                for inp in curr_node.all_input_nodes:
                    if (
                        inp in curr_set
                        or inp.op != "call_function"
                        or order[inp] >= first_curr_pos
                        or not isinstance(inp.meta.get("val"), torch.Tensor)
                    ):
                        continue
                    users = entry_users.setdefault(inp, [])
                    if curr_node not in users:
                        users.append(curr_node)

            prev_exits_unique = [
                exit_node
                for exit_node in prev_exits
                if exit_node not in entry_users
            ]
            if not entry_users or not prev_exits_unique:
                continue

            plans.append((is_bwd, first_curr, entry_users, prev_exits_unique))

    if not plans:
        return gm

    n_fwd_barriers = 0
    n_bwd_barriers = 0
    for is_bwd, first_curr, entry_users, prev_exits_unique in plans:
        entries = list(entry_users.keys())
        boundaries = (*entries, *prev_exits_unique)
        curr_fqn = _get_module_fqn(first_curr)
        curr_stack = first_curr.meta.get("nn_module_stack")

        with graph.inserting_before(first_curr):
            barrier = graph.call_function(
                _optimization_barrier,
                args=boundaries,
            )
            barrier.meta["val"] = tuple(b.meta["val"] for b in boundaries)
            barrier.meta["custom"] = {_MODULE_FQN: curr_fqn}
            if curr_stack is not None:
                barrier.meta["nn_module_stack"] = curr_stack.copy()
            if is_bwd:
                barrier.meta["autograd_backward"] = True

            for idx, entry in enumerate(entries):
                output = graph.call_function(
                    operator.getitem,
                    args=(barrier, idx),
                )
                output.meta = entry.meta.copy()
                output.meta["custom"] = dict(entry.meta.get("custom", {}))
                output.meta["custom"][_MODULE_FQN] = curr_fqn
                if curr_stack is not None:
                    output.meta["nn_module_stack"] = curr_stack.copy()
                if is_bwd:
                    output.meta["autograd_backward"] = True
                else:
                    output.meta.pop("autograd_backward", None)
                for user in entry_users[entry]:
                    user.replace_input_with(entry, output)

        if is_bwd:
            n_bwd_barriers += 1
        else:
            n_fwd_barriers += 1

    graph.lint()
    gm.recompile()
    logger.info(
        "[insert_ep_chunk_barriers] inserted %d forward + %d backward chunk barriers",
        n_fwd_barriers,
        n_bwd_barriers,
    )
    return gm


__all__ = ["apply_ep_chunk_barrier_pass"]
