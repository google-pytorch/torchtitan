# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lower selected FSDP reduce-scatters via all-reduce plus local slicing.

XLA can combine compatible all-reduces into one tuple-result collective, while
TorchTPU currently lowers a coalesced reduce-scatter list as separate physical
collectives. Full-world all-reduce plus a rank-local slice is mathematically
equivalent to reduce-scatter, but sends roughly twice the ring traffic and
materializes the full result. Recipes must therefore select parameter
gradients explicitly and bound their validated payload size. A native
multi-operand reduce-scatter remains the preferred backend fix. The rewrite
preserves dtype but changes collective reduction order, so BF16 recipes must
validate the resulting numerical trajectory.
"""

from __future__ import annotations

import fnmatch
import logging

import torch
import torch.distributed as dist

from torchtitan.experiments.graph_trainer.common_utils import (
    PARAMETER_GRADIENT_FQNS_META,
)

logger = logging.getLogger(__name__)

_ALL_REDUCE = torch.ops._c10d_functional.all_reduce.default
_REDUCE_SCATTER = torch.ops._c10d_functional.reduce_scatter_tensor.default
_WAIT_TENSOR = torch.ops._c10d_functional.wait_tensor.default


def matching_parameter_patterns(
    parameter_fqn: str, parameter_patterns: tuple[str, ...]
) -> tuple[str, ...]:
    """Return the patterns that select ``parameter_fqn``.

    A pattern is a case-sensitive ``fnmatch`` glob over the whole parameter
    FQN, as ``named_parameters()`` reports it. Recipe tests use this to check
    their patterns against a model's parameters with the pass's own rule.
    """
    return tuple(
        pattern
        for pattern in parameter_patterns
        if fnmatch.fnmatchcase(parameter_fqn, pattern)
    )


def _find_reduce_scatter(
    gradient: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    """Find the reduce-scatter/wait in a tagged gradient's unary FSDP tail."""
    node = gradient
    seen: set[torch.fx.Node] = set()
    while node not in seen:
        seen.add(node)
        if node.op == "call_function" and node.target is _WAIT_TENSOR:
            collective = node.args[0] if node.args else None
            if (
                isinstance(collective, torch.fx.Node)
                and collective.op == "call_function"
                and collective.target is _REDUCE_SCATTER
            ):
                return collective, node
        inputs = node.all_input_nodes
        if len(inputs) != 1:
            return None
        node = inputs[0]
    return None


def apply_selected_reduce_scatter_allreduce_pass(
    gm: torch.fx.GraphModule,
    example_inputs,
    *,
    parameter_patterns: tuple[str, ...],
    max_input_bytes: int,
) -> torch.fx.GraphModule:
    """Rewrite all selected gradients, or raise before changing the graph."""
    del example_inputs
    if not parameter_patterns:
        return gm
    if max_input_bytes <= 0:
        raise ValueError("max_input_bytes must be positive when patterns are set")

    matched_patterns: set[str] = set()
    selected: dict[torch.fx.Node, tuple[torch.fx.Node, set[str]]] = {}
    for gradient in gm.graph.nodes:
        fqns = gradient.meta.get("custom", {}).get(
            PARAMETER_GRADIENT_FQNS_META, ()
        )
        if not isinstance(fqns, tuple) or not all(
            isinstance(fqn, str) for fqn in fqns
        ):
            raise ValueError(
                f"Invalid {PARAMETER_GRADIENT_FQNS_META} metadata on "
                f"{gradient.name}: {fqns!r}"
            )
        patterns_by_fqn = tuple(
            (fqn, matching_parameter_patterns(fqn, parameter_patterns))
            for fqn in fqns
        )
        matching_fqns = tuple(fqn for fqn, patterns in patterns_by_fqn if patterns)
        if not matching_fqns:
            continue
        matched_patterns.update(
            pattern for _, patterns in patterns_by_fqn for pattern in patterns
        )
        found = _find_reduce_scatter(gradient)
        if found is None:
            raise ValueError(
                f"{', '.join(matching_fqns)} has no unary FSDP "
                "reduce-scatter tail"
            )
        reduce_scatter, wait = found
        prior = selected.setdefault(reduce_scatter, (wait, set()))
        if prior[0] is not wait:
            raise ValueError(
                f"{', '.join(matching_fqns)} has an ambiguous reduce-scatter wait"
            )
        prior[1].update(matching_fqns)

    missing = [
        pattern for pattern in parameter_patterns if pattern not in matched_patterns
    ]
    if missing:
        raise ValueError(
            "FSDP all-reduce-plus-slice pattern(s) matched no parameter "
            f"gradient: {missing}"
        )

    world_size = dist.get_world_size()
    # Validate every selected reduction before mutating any of them.
    rewrites = []
    for reduce_scatter, (wait, fqns) in selected.items():
        label = ", ".join(sorted(fqns))
        if len(reduce_scatter.args) < 4:
            raise ValueError(f"{label}: reduce-scatter schema has fewer than four args")
        if len(reduce_scatter.users) != 1 or wait not in reduce_scatter.users:
            raise ValueError(f"{label}: reduce-scatter does not have one wait user")
        input_node, reduce_op, group_size, group_name = reduce_scatter.args[:4]
        if not isinstance(input_node, torch.fx.Node):
            raise ValueError(f"{label}: reduce-scatter input is not an FX node")
        if reduce_op != "sum":
            raise ValueError(f"{label}: reduction is {reduce_op!r}, not 'sum'")
        if group_size != world_size:
            raise ValueError(
                f"{label}: group size {group_size!r} is not world size {world_size}"
            )
        if not isinstance(group_name, str):
            raise ValueError(f"{label}: process-group name is not a string")

        input_value = input_node.meta.get("val")
        output_value = reduce_scatter.meta.get("val")
        if not isinstance(input_value, torch.Tensor) or not isinstance(
            output_value, torch.Tensor
        ):
            raise ValueError(f"{label}: reduce-scatter tensor metadata is missing")
        supported_dtypes = (torch.float32, torch.bfloat16)
        if (
            input_value.dtype not in supported_dtypes
            or output_value.dtype != input_value.dtype
        ):
            raise ValueError(
                f"{label}: expected matching FP32 or BF16 input/output, got "
                f"{input_value.dtype}/{output_value.dtype}"
            )
        input_bytes = int(input_value.numel()) * input_value.element_size()
        if input_bytes > max_input_bytes:
            raise ValueError(
                f"{label}: {input_bytes} input bytes exceed the configured "
                f"maximum {max_input_bytes}"
            )
        if input_value.shape[0] != world_size * output_value.shape[0]:
            raise ValueError(f"{label}: dimension zero is not world-sharded")
        if input_value.shape[1:] != output_value.shape[1:]:
            raise ValueError(f"{label}: non-sharded dimensions do not match")
        rewrites.append(
            (reduce_scatter, wait, input_node, reduce_op, group_name, output_value)
        )

    group_ranks: dict[str, int] = {}
    graph = gm.graph
    for reduce_scatter, wait, input_node, reduce_op, group_name, output_value in rewrites:
        if group_name not in group_ranks:
            group = dist.distributed_c10d._resolve_process_group(group_name)
            group_ranks[group_name] = dist.get_rank(group)
        group_rank = group_ranks[group_name]
        local_rows = output_value.shape[0]
        with graph.inserting_before(reduce_scatter):
            all_reduce = graph.call_function(
                _ALL_REDUCE, args=(input_node, reduce_op, group_name)
            )
            all_reduce.meta.update(input_node.meta)
            all_reduce_wait = graph.call_function(_WAIT_TENSOR, args=(all_reduce,))
            all_reduce_wait.meta.update(input_node.meta)
            # The shard is the parameter gradient that the compiled step
            # returns, so emit a copy rather than a slice view. TorchTPU keeps
            # a returned view's layout, so a view at row offset
            # group_rank * local_rows came back as a view into a buffer of
            # group_rank + 1 shards. Programs traced from those gradients (the
            # compiled gradient norm) then differed across ranks, and their
            # all-reduce returned different values on different ranks. XLA
            # emits the same slice either way.
            local_shard = graph.call_function(
                torch.ops.aten.slice_copy.Tensor,
                args=(
                    all_reduce_wait,
                    0,
                    group_rank * local_rows,
                    (group_rank + 1) * local_rows,
                ),
            )
            local_shard.meta.update(wait.meta)
            local_shard.meta["val"] = output_value
        wait.replace_all_uses_with(local_shard)
        graph.erase_node(wait)
        graph.erase_node(reduce_scatter)

    graph.lint()
    gm.recompile()
    logger.info(
        "[fsdp_allreduce_slice] rewrote %d reductions selected by %s",
        len(rewrites),
        parameter_patterns,
    )
    return gm


__all__ = [
    "apply_selected_reduce_scatter_allreduce_pass",
    "matching_parameter_patterns",
]
