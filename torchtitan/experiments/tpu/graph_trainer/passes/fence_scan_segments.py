# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FX pass: finish one scan segment before the next one starts.

With the layers run as several scans (``compile.save_last_n_layers``), XLA
starts the next loop's weight all-gathers while the previous loop's last
reduce-scatters are still in flight, and both sets of buffers are live: 37.4
vs 17.7 GiB on four Qwen3-235B layers. An optimization barrier between
consecutive scans in the same direction stops that. One scan per direction is
a no-op.
"""

import operator

import torch
import torch.fx
from torch.fx import Node

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node
from torchtitan.tools.logging import logger

_SCAN = torch.ops.higher_order.scan


def _tensors(args) -> list[Node]:
    out: list[Node] = []
    for a in args:
        if isinstance(a, (list, tuple)):
            out += _tensors(a)
        elif isinstance(a, Node) and isinstance(a.meta.get("val"), torch.Tensor):
            out.append(a)
    return out


def _from_scan(node: Node) -> bool:
    return node.target is operator.getitem and node.args[0].target is _SCAN


def _reads_as(reader: Node, scan: Node) -> bool:
    """Whether ``reader`` runs in ``scan``'s direction. The output reads both."""
    return reader.op == "output" or _is_backward_node(reader) == _is_backward_node(scan)


def _fence_values(a: Node, b: Node) -> list[Node]:
    """What scan ``b`` waits for: ``a``'s outputs, and ``b``'s own inputs.

    Stacks saved for the other direction stay out: they are written per layer
    already, and one may be on host. ``b``'s init carry is always in, so ``b``
    waits for ``a`` even when the carry is also saved.
    """
    outs = [
        u
        for u in _tensors(a.users)
        if u.users and all(_reads_as(r, a) for r in u.users)
    ]
    init = _tensors(b.args[1:2])
    inputs = [v for v in _tensors(b.args[2:]) if not _from_scan(v)]
    return list(dict.fromkeys(outs + init + inputs))


def fence_scan_segments(
    gm: torch.fx.GraphModule,
    example_inputs=None,
    *,
    barrier=None,
    expects_segments: bool = False,
) -> torch.fx.GraphModule:
    """Put an optimization barrier before each scan that follows another.

    ``barrier`` defaults to ``tpu::_optimization_barrier``; tests pass a
    stand-in. ``expects_segments`` is set when last-N splits the layers: a
    graph that has scans but no consecutive pair then raises, rather than
    running unfenced. Graphs with no scan (optimizer, grad norm) stay silent.
    """
    graph = gm.graph
    scans = list(graph.find_nodes(op="call_function", target=_SCAN))
    pairs = [
        (a, b)
        for a, b in zip(scans, scans[1:])
        if _is_backward_node(a) == _is_backward_node(b)
    ]
    if not pairs:
        if scans and expects_segments:
            raise RuntimeError(
                "fence_scan_segments: compile.save_last_n_layers splits the "
                f"layers, but no pair of scans in one direction was found among "
                f"{len(scans)} scans, so no scan boundary is fenced"
            )
        return gm
    barrier = barrier or torch.ops.tpu._optimization_barrier.default
    for a, b in pairs:
        order = {n: i for i, n in enumerate(graph.nodes)}
        values = _fence_values(a, b)
        meta = {"autograd_backward": _is_backward_node(b)}
        with graph.inserting_before(b):
            fence = graph.call_function(barrier, (values,))
            fence.meta.update(meta, val=[v.meta["val"] for v in values])
            fenced = []
            for i, v in enumerate(values):
                g = graph.call_function(operator.getitem, (fence, i))
                g.meta.update(meta, val=v.meta["val"])
                fenced.append(g)
        for v, g in zip(values, fenced):
            # Readers before ``b`` keep the unfenced value, so the graph stays
            # in order; so do the other direction's, which the fence is not for.
            for u in list(v.users):
                if u in order and order[u] >= order[b] and _reads_as(u, b):
                    u.replace_input_with(v, g)
    graph.lint()
    gm.recompile()
    logger.info("fence_scan_segments: fenced %d scan boundaries", len(pairs))
    return gm
