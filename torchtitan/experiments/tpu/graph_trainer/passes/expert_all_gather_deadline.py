# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Finish each layer's expert FSDP all-gathers under its Splash attention.

The expert weight all-gathers run on a SparseCore, serially with the
routed-token gathers that follow routing. Left to itself, the TPU layer
scheduler overlaps at most one of them with attention and queues the rest
behind the dispatch gather, where nothing hides them; meanwhile the SparseCore
idles through attention.

This pass joins the gathered expert weights of a layer with that layer's Splash
output in one optimization barrier. Nothing that reads the Splash output can
start until every expert all-gather is done, so the scheduler issues them
during QKV and Splash. The Splash output is a kernel output, materialized
either way, so the barrier changes no values.

Each layer copy that runs Splash gets its own barrier: the forward, and the
backward copy that recomputes it. A backward copy that reuses a saved Splash
output has no Splash to anchor on and is left alone.
"""

from __future__ import annotations

import operator
from collections import defaultdict

import torch
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor,
    is_wait_tensor,
)

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _get_module_fqn,
    _NOT_IN_LAYERS,
)
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    _is_recomputed,
    _optimization_barrier,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_memory_policy import (
    _is_splash_forward_op,
)
from torchtitan.tools.logging import logger


def _is_gathered_expert_weight(node: torch.fx.Node) -> bool:
    """A gathered expert weight: TorchTPU's dim-d all-gather, or a funcol wait."""
    if node.op != "call_function" or ".moe." not in _get_module_fqn(node) + ".":
        return False
    if is_wait_tensor(node):
        return isinstance(node.args[0], torch.fx.Node) and is_all_gather_into_tensor(
            node.args[0]
        )
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target.name() == "tpu::_all_gather_dim"
    )


def _is_splash_output(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is operator.getitem
        and node.args[1] == 0
        and isinstance(node.args[0], torch.fx.Node)
        and _is_splash_forward_op(node.args[0].target)
    )


def _layer_copy(node: torch.fx.Node) -> tuple[int, bool, bool]:
    """Which copy of which layer ``node`` belongs to."""
    return (
        _get_layer_id(node),
        _is_recomputed(node),
        bool(node.meta.get("autograd_backward")),
    )


def apply_expert_all_gather_deadline(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    gathered: dict[tuple, list[torch.fx.Node]] = defaultdict(list)
    splash: dict[tuple, torch.fx.Node] = {}
    for node in gm.graph.nodes:
        if _get_layer_id(node) == _NOT_IN_LAYERS:
            continue
        if _is_gathered_expert_weight(node):
            gathered[_layer_copy(node)].append(node)
        elif _is_splash_output(node):
            splash[_layer_copy(node)] = node

    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    barriers = 0
    for layer_copy, weights in gathered.items():
        anchor = splash.get(layer_copy)
        if anchor is None:
            continue
        values = [anchor, *weights]
        users = {value: list(value.users) for value in values}
        with gm.graph.inserting_after(anchor):
            barrier = gm.graph.call_function(_optimization_barrier, args=tuple(values))
        barrier.meta = {**anchor.meta, "val": tuple(v.meta["val"] for v in values)}

        # The weights' gather chains depend only on parameters; move the parts
        # traced after the anchor ahead of the barrier.
        hoisted: set[torch.fx.Node] = set()

        def hoist(node: torch.fx.Node) -> None:
            if order.get(node, -1) <= order[anchor] or node in hoisted:
                return
            hoisted.add(node)
            for input_node in node.all_input_nodes:
                hoist(input_node)
            barrier.prepend(node)

        for weight in weights:
            hoist(weight)

        insertion_point = barrier
        for index, value in enumerate(values):
            with gm.graph.inserting_after(insertion_point):
                output = gm.graph.call_function(operator.getitem, args=(barrier, index))
            output.meta = {**value.meta}
            insertion_point = output
            for user in users[value]:
                if (
                    _layer_copy(user) == layer_copy
                    and order.get(user, len(order)) > order[anchor]
                ):
                    user.replace_input_with(value, output)
        barriers += 1

    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Joined the expert all-gathers of %d layer copies with their Splash output",
        barriers,
    )
    return gm
