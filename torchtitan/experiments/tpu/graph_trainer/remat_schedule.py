# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU scheduling constraints for GraphTrainer activation rematerialization."""

from __future__ import annotations

import operator
from collections import defaultdict

import torch
from torch._dynamo.graph_deduplication import _stable_topological_sort

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
    _NOT_IN_LAYERS,
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    find_fsdp_reduce_grad_input,
    is_all_gather_into_tensor,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    _collect_bucketed_fsdp_comms,
)
from torchtitan.tools.logging import logger


_TPU_REMAT_BARRIER_KIND_META = "tpu_remat_barrier_kind"
_TPU_REMAT_REGION_ID_META = "tpu_remat_region_id"
_TPU_REMAT_LAYER_ENTRY_ANCHOR_META = "tpu_remat_layer_entry_anchor"
_TPU_FSDP_RS_DEADLINE_META = "tpu_fsdp_rs_deadline_waits"
# Where the routed experts' parameters live in the module tree. Upstream
# d771abb0 interposed `RoutedExperts`, moving them down from `.moe.experts.`.
_EXPERT_FQN_INFIX = ".moe.routed_experts.inner_experts."


def _optimization_barrier(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if not values:
        return ()
    device_types = {value.device.type for value in values}
    if device_types == {"cpu"}:
        return values
    if device_types != {"tpu"}:
        raise ValueError(
            "TPU optimization barriers require all-CPU test inputs or all-TPU "
            f"runtime inputs, got device types {sorted(device_types)}"
        )
    return tuple(torch.tpu.ops.optimization_barrier(list(values)))


def _is_tensor_node(node: torch.fx.Node) -> bool:
    return isinstance(node.meta.get("val"), torch.Tensor)


def _is_recomputed(node: torch.fx.Node) -> bool:
    """Whether selective_activation_remat_pass added this node as a recompute."""
    return node.name.endswith("_recomputed")


def _find_local_grad_before_reduce_dtype_cast(
    reduce_grad_input: torch.fx.Node,
) -> torch.fx.Node | None:
    """Match the expert BF16 dWeight before its FP32 FSDP reduction cast."""
    node = reduce_grad_input
    while hasattr(node.target, "is_view") and node.target.is_view:
        if len(node.all_input_nodes) != 1:
            return None
        node = node.all_input_nodes[0]

    if (
        node.target is not torch.ops.aten._to_copy.default
        or len(node.all_input_nodes) != 1
    ):
        return None

    source = node.all_input_nodes[0]
    source_val = source.meta.get("val")
    cast_val = node.meta.get("val")
    if not isinstance(source_val, torch.Tensor) or not isinstance(
        cast_val, torch.Tensor
    ):
        return None
    if (
        source_val.dtype != torch.bfloat16
        or cast_val.dtype != torch.float32
        or source_val.shape != cast_val.shape
        or source_val.device != cast_val.device
    ):
        return None
    return source


def _extract_layer_index_from_fqn(fqn: str) -> int | None:
    """Extracts the transformer block index from a parameter FQN, or None.

    Native TorchTitan models name parameters `layers.<N>.*`, whereas wrapped
    HuggingFace models keep their adapter prefix on `named_parameters()` (e.g.
    `model.esmc.layers.<N>.*` or `model.encoder.layer.<N>.*`) even after forward
    node `module_fqn` metadata is canonicalized to `layers.<N>.*`.
    """
    parts = fqn.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part in ("layers", "layer"):
            try:
                return int(parts[idx + 1])
            except ValueError:
                continue
    return None


def _parameter_layer_id(parameter_fqns: tuple[str, ...]) -> int:
    layer_ids = {
        layer_id
        for fqn in parameter_fqns
        if (layer_id := _extract_layer_index_from_fqn(fqn)) is not None
    }
    if len(layer_ids) > 1:
        raise RuntimeError(
            "TPU remat scheduling does not support parameters shared across "
            f"remat regions: {parameter_fqns}"
        )
    return next(iter(layer_ids), _NOT_IN_LAYERS)


def _ordering_gradient(
    gradient: torch.fx.Node,
    parameter_fqns: tuple[str, ...],
) -> torch.fx.Node:
    reduce_grad_input = find_fsdp_reduce_grad_input(gradient) or gradient
    # Upstream d771abb0 interposed `RoutedExperts` between the MoE and its
    # grouped experts, moving these parameters to
    # `moe.routed_experts.inner_experts.*`.
    if not any(_EXPERT_FQN_INFIX in f".{fqn}." for fqn in parameter_fqns):
        return reduce_grad_input

    # Depending on the BF16 local dWeight lets its FP32 cast and reduce-scatter
    # overlap the preceding layer's backward computation. This was measured
    # only for the large expert gradients; if the same ordering point is
    # neutral or beneficial for other parameters, generalize it and remove the
    # parameter-name special case.
    return (
        _find_local_grad_before_reduce_dtype_cast(reduce_grad_input)
        or reduce_grad_input
    )


def apply_tpu_remat_schedule(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    num_static_inputs: int,
) -> torch.fx.GraphModule:
    """Keep each layer's parameter gradients adjacent to its recomputation.

    Selective rematerialization labels the recompute and backward nodes for
    each transformer layer. For adjacent layers 0 and 1, the relevant graph is
    conceptually::

        saved_x1 -> recompute_1 -> backward_1 -> (dw1, dx1)
        saved_x0 ------------------------------> recompute_0(dx1) -> backward_0

    XLA may otherwise schedule ``recompute_1`` early, follow ``dx1`` into layer
    0, and defer ``dw1``. That keeps layer 1's recomputed workspace live while
    layer 0 is running. This pass rewrites only the inputs to layer 0's
    recompute/backward entry::

        saved_x0_barrier, dx1_barrier, _ = optimization_barrier(
            saved_x0, dx1, dw1
        )
        recompute_0(saved_x0_barrier)
        backward_0(dx1_barrier)

    The tuple barrier does not reshape or numerically change any tensor. The
    inserted ``getitem`` nodes preserve each input's original shape and dtype;
    they only make ``dw1`` an XLA scheduling dependency of layer 0's entry.
    Thus layer 1 must finish its local parameter gradients before layer 0 can
    start, while the later FSDP cast and reduce-scatter may still overlap with
    layer 0.

    Every saved primal is barriered. The earliest external backward value
    entering the region anchors it at backward entry; later cotangents retain
    their natural dependencies because they may be produced after local
    backward has begun. Each tagged parameter gradient must belong to exactly
    one region. Parameters shared across regions are not supported. Trace-time
    gradient identity makes this work whether gradients are graph outputs or
    are consumed by an optimizer inside the graph.

    The barrier uses the next layer's local parameter gradients before FSDP
    synchronization. The currently validated expert path anchors before the
    BF16-to-FP32 reduce cast, so both the cast and reduce-scatter can overlap
    the preceding layer's backward computation. Other parameters retain the
    generic FSDP split at the collective input pending equivalent performance
    validation. The collective data path remains unchanged.
    """
    del example_inputs

    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    static_placeholders = set(placeholders[:num_static_inputs])
    nonstatic_cache: dict[torch.fx.Node, bool] = {}

    # Nonstatic values depend on runtime batch inputs rather than only model
    # state. Only those activation boundaries should anchor rematerialization.
    def depends_on_nonstatic_input(node: torch.fx.Node) -> bool:
        cached = nonstatic_cache.get(node)
        if cached is not None:
            return cached
        if node.op == "placeholder":
            result = node not in static_placeholders
        elif node.op == "get_attr":
            result = False
        else:
            result = any(
                depends_on_nonstatic_input(inp) for inp in node.all_input_nodes
            )
        nonstatic_cache[node] = result
        return result

    recompute_by_region: dict[int, set[torch.fx.Node]] = defaultdict(set)
    backward_by_region: dict[int, set[torch.fx.Node]] = defaultdict(set)
    for node in gm.graph.nodes:
        region_id = _get_layer_id(node)
        if region_id == _NOT_IN_LAYERS:
            continue
        if _is_recomputed(node):
            recompute_by_region[region_id].add(node)
        elif _is_backward_node(node):
            backward_by_region[region_id].add(node)

    if not recompute_by_region:
        return gm
    if recompute_by_region.keys() != backward_by_region.keys():
        raise RuntimeError(
            "TPU remat scheduling requires matching recompute and backward regions"
        )

    tagged_gradients_by_region: dict[
        int, list[tuple[torch.fx.Node, tuple[str, ...]]]
    ] = defaultdict(list)
    gradient_regions: dict[torch.fx.Node, set[int]] = defaultdict(set)
    gradient_fqns: dict[torch.fx.Node, list[str]] = defaultdict(list)
    for gradient in gm.graph.nodes:
        parameter_fqns = gradient.meta.get("custom", {}).get(
            PARAMETER_GRADIENT_FQNS_META
        )
        if parameter_fqns is None:
            continue
        if not isinstance(parameter_fqns, tuple) or not all(
            isinstance(parameter_fqn, str) for parameter_fqn in parameter_fqns
        ):
            raise RuntimeError(
                "Parameter-gradient metadata must be a tuple of FQNs, got "
                f"{parameter_fqns!r} on {gradient.name}"
            )
        region_id = _parameter_layer_id(parameter_fqns)
        if region_id in backward_by_region:
            tagged_gradients_by_region[region_id].append((gradient, parameter_fqns))
            gradient_regions[gradient].add(region_id)
            gradient_fqns[gradient].extend(parameter_fqns)

    shared_across_regions = {
        gradient.name: tuple(gradient_fqns[gradient])
        for gradient, regions in gradient_regions.items()
        if len(regions) > 1
    }
    if shared_across_regions:
        raise RuntimeError(
            "TPU remat scheduling does not support parameter-gradient values "
            f"shared across remat regions: {shared_across_regions}"
        )

    dweights_by_region = {
        region_id: list(
            dict.fromkeys(
                _ordering_gradient(gradient, parameter_fqns)
                for gradient, parameter_fqns in sorted(
                    gradients, key=lambda item: order[item[0]]
                )
            )
        )
        for region_id, gradients in tagged_gradients_by_region.items()
    }

    region_ids = sorted(recompute_by_region)
    barrier_count = 0
    gradient_count = 0
    for region_index, region_id in enumerate(region_ids):
        recompute_nodes = recompute_by_region[region_id]
        backward_nodes = backward_by_region[region_id]

        # Region metadata labels internal nodes, not their live-ins. Derive the
        # saved activations and incoming cotangent from edges crossing into the
        # recompute/backward region.
        primal_boundaries = sorted(
            {
                inp
                for node in recompute_nodes
                for inp in node.all_input_nodes
                if inp not in recompute_nodes
                and _is_tensor_node(inp)
                and depends_on_nonstatic_input(inp)
            },
            key=order.__getitem__,
        )
        # A recomputed forward value is tagged as backward but is not a
        # cotangent. A recomputed value shared across layers is recomputed
        # once, before its first reader, so it would otherwise come before the
        # real incoming gradient of every later region.
        cotangent_candidates = sorted(
            {
                inp
                for node in backward_nodes
                for inp in node.all_input_nodes
                if inp not in backward_nodes
                and not _is_recomputed(inp)
                and _is_backward_node(inp)
                and _is_tensor_node(inp)
            },
            key=order.__getitem__,
        )
        if not primal_boundaries:
            raise RuntimeError(f"remat region {region_id} has no saved activation")
        if not cotangent_candidates:
            raise RuntimeError(
                f"remat region {region_id} has no incoming backward value"
            )
        cotangent = cotangent_candidates[0]
        next_region_id = (
            region_ids[region_index + 1] if region_index + 1 < len(region_ids) else None
        )
        ordering_gradients = (
            dweights_by_region.get(next_region_id, [])
            if next_region_id is not None
            else []
        )
        if next_region_id is not None and not ordering_gradients:
            raise RuntimeError(
                f"remat region {next_region_id} has no parameter-gradient markers"
            )

        # The next layer's local gradients are dependency-only barrier inputs;
        # only the saved primals and cotangent have users redirected through
        # their corresponding tuple outputs.
        boundaries = list(
            dict.fromkeys(primal_boundaries + [cotangent] + ordering_gradients)
        )
        redirect_users: dict[torch.fx.Node, set[torch.fx.Node]] = {
            boundary: {user for user in boundary.users if user in recompute_nodes}
            for boundary in primal_boundaries
        }
        redirect_users[cotangent] = {
            user for user in cotangent.users if _is_backward_node(user)
        }
        redirect_users = {
            boundary: users for boundary, users in redirect_users.items() if users
        }
        first_user = min(
            (user for users in redirect_users.values() for user in users),
            key=order.__getitem__,
        )
        late_boundaries = [
            boundary.name
            for boundary in boundaries
            if order[boundary] >= order[first_user]
        ]
        if late_boundaries:
            raise RuntimeError(
                f"remat region {region_id} has boundary values unavailable before "
                f"{first_user.name}: {late_boundaries}"
            )

        with gm.graph.inserting_before(first_user):
            barrier = gm.graph.call_function(
                _optimization_barrier,
                args=tuple(boundaries),
            )
            barrier.meta["val"] = tuple(boundary.meta["val"] for boundary in boundaries)
            barrier.meta["autograd_backward"] = True
            barrier.meta[_TPU_REMAT_REGION_ID_META] = region_id
            barrier.meta[_TPU_REMAT_BARRIER_KIND_META] = "layer_entry"

            barrier_outputs: dict[torch.fx.Node, torch.fx.Node] = {}
            for index, boundary in enumerate(boundaries):
                output = gm.graph.call_function(operator.getitem, args=(barrier, index))
                output.meta = boundary.meta.copy()
                output.meta["autograd_backward"] = True
                output.meta[_TPU_REMAT_REGION_ID_META] = region_id
                barrier_outputs[boundary] = output

            barrier_outputs[cotangent].meta[
                _TPU_REMAT_LAYER_ENTRY_ANCHOR_META
            ] = region_id

        for boundary, output in barrier_outputs.items():
            for user in redirect_users.get(boundary, ()):
                user.replace_input_with(boundary, output)

        barrier_count += 1
        gradient_count += len(ordering_gradients)

    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Inserted %d TPU remat barriers ordering %d parameter gradients",
        barrier_count,
        gradient_count,
    )
    return gm


def apply_tpu_remat_fsdp_all_gather_schedule(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    num_static_inputs: int,
    prefetch_distance: int = 1,
) -> torch.fx.GraphModule:
    """Keep recomputed FSDP unshards distinct and eligible for prefetch.

    Graph SAC duplicates the forward FSDP unshard chain in backward. Without a
    data dependency that distinguishes the duplicate, XLA can CSE the two
    all-gathers and retain the forward full parameter until backward. The
    layer-entry barrier above intentionally excludes static parameter inputs,
    so this pass adds a separate barrier only on the recomputed unshard roots.

    The loss value releases the final ``prefetch_distance`` layers' recomputed
    all-gathers after forward completes. Each lower layer is released by the
    explicit backward entry anchor that many layers ahead. This leaves XLA's
    latency-hiding scheduler a bounded window in which to overlap unshards with
    backward compute.
    """
    del example_inputs

    if prefetch_distance < 1:
        raise ValueError(
            "TPU FSDP remat prefetch distance must be positive, got "
            f"{prefetch_distance}"
        )

    if any(
        node.meta.get(_TPU_REMAT_BARRIER_KIND_META) == "fsdp_all_gather"
        for node in gm.graph.nodes
    ):
        # Keep the pass idempotent for debugging and composed pass pipelines.
        return gm

    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    static_placeholders = set(placeholders[:num_static_inputs])
    state_input_cache: dict[torch.fx.Node, bool] = {}

    def depends_on_model_state(node: torch.fx.Node) -> bool:
        """Whether ``node`` derives from GraphTrainer parameter/buffer inputs."""
        cached = state_input_cache.get(node)
        if cached is not None:
            return cached
        if node.op == "placeholder":
            result = node in static_placeholders
        elif node.op == "get_attr":
            result = False
        else:
            result = any(
                depends_on_model_state(inp) for inp in node.all_input_nodes
            )
        state_input_cache[node] = result
        return result

    recompute_by_region: dict[int, set[torch.fx.Node]] = defaultdict(set)
    # TODO: Classify non-layer FSDP unshards. Input-side parameters should use
    # layer 0 as their anchor; loss-side parameters should use the loss anchor.
    for node in gm.graph.nodes:
        region_id = _get_layer_id(node)
        if region_id != _NOT_IN_LAYERS and _is_recomputed(node):
            recompute_by_region[region_id].add(node)

    region_ids = sorted(recompute_by_region)
    all_gathers_by_region = {
        region_id: sorted(
            (
                node
                for node in recompute_by_region[region_id]
                if is_all_gather_into_tensor(node)
            ),
            key=order.__getitem__,
        )
        for region_id in region_ids
    }

    layer_entry_anchor_by_region: dict[int, torch.fx.Node] = {}
    # apply_tpu_remat_schedule marks exactly one incoming-cotangent output at
    # each layer's backward entry; collect those explicit anchors by layer.
    for node in gm.graph.nodes:
        region_id = node.meta.get(_TPU_REMAT_LAYER_ENTRY_ANCHOR_META)
        if region_id is None:
            continue
        if not isinstance(region_id, int):
            raise RuntimeError(
                "TPU remat layer-entry anchor requires a transformer layer ID, "
                f"got {region_id!r} on {node.name}"
            )
        if region_id in layer_entry_anchor_by_region:
            raise RuntimeError(
                f"remat region {region_id} has multiple layer-entry anchors"
            )
        if not _is_tensor_node(node):
            raise RuntimeError(
                f"remat region {region_id} has a non-tensor layer-entry anchor"
            )
        layer_entry_anchor_by_region[region_id] = node

    loss_anchor: torch.fx.Node | None = None
    barrier_count = 0
    all_gather_count = 0
    for region_index, region_id in enumerate(region_ids):
        recompute_nodes = recompute_by_region[region_id]
        redirects: dict[torch.fx.Node, set[torch.fx.Node]] = defaultdict(set)
        protected_all_gather_count = 0
        region_all_gathers = all_gathers_by_region[region_id]
        for all_gather in region_all_gathers:
            launch_input = all_gather.args[0]
            if not isinstance(launch_input, torch.fx.Node):
                raise RuntimeError(
                    f"Recomputed FSDP all-gather {all_gather.name} has a non-node input"
                )

            # Walk through recomputed unary views/casts to the original sharded
            # parameter input. Only that root is redirected through the barrier.
            boundary = launch_input
            consumer = all_gather
            while boundary in recompute_nodes:
                tensor_inputs = [
                    inp for inp in boundary.all_input_nodes if _is_tensor_node(inp)
                ]
                if len(tensor_inputs) != 1:
                    raise RuntimeError(
                        "Recomputed FSDP all-gather input chain must be unary, got "
                        f"{len(tensor_inputs)} tensor inputs at {boundary.name}"
                    )
                consumer = boundary
                boundary = tensor_inputs[0]
            if not depends_on_model_state(boundary):
                continue
            redirects[boundary].add(consumer)
            protected_all_gather_count += 1

        if not redirects:
            continue

        first_recompute_user = min(
            (user for users in redirects.values() for user in users),
            key=order.__getitem__,
        )
        anchor_index = region_index + prefetch_distance
        anchor_region_id = (
            region_ids[anchor_index] if anchor_index < len(region_ids) else None
        )
        if anchor_region_id is not None:
            prefetch_anchor = layer_entry_anchor_by_region.get(anchor_region_id)
            if prefetch_anchor is None:
                raise RuntimeError(
                    "TPU FSDP remat scheduling requires an explicit layer-entry "
                    f"anchor for remat region {anchor_region_id}"
                )
        else:
            if loss_anchor is None:
                output_node = next(
                    node for node in gm.graph.nodes if node.op == "output"
                )
                graph_outputs = output_node.args[0]
                if not isinstance(graph_outputs, (tuple, list)) or not graph_outputs:
                    raise RuntimeError(
                        "TPU FSDP remat scheduling requires loss-first outputs"
                    )
                candidate = graph_outputs[0]
                if not isinstance(candidate, torch.fx.Node) or not _is_tensor_node(
                    candidate
                ):
                    raise RuntimeError(
                        "TPU FSDP remat scheduling requires a tensor loss as "
                        "output zero"
                    )
                loss_anchor = candidate
            prefetch_anchor = loss_anchor

        if order[prefetch_anchor] >= order[first_recompute_user]:
            raise RuntimeError(
                "TPU FSDP remat scheduling requires the prefetch anchor before "
                f"recomputed all-gathers in region {region_id}"
            )
        # The barrier passes parameter roots through to their recompute chains;
        # the anchor is dependency-only and releases those roots at L+N.
        boundaries = sorted(redirects, key=order.__getitem__)
        barrier_inputs = list(boundaries)
        if prefetch_anchor not in redirects:
            barrier_inputs.append(prefetch_anchor)
        late_boundaries = [
            boundary.name
            for boundary in boundaries
            if order[boundary] >= order[prefetch_anchor]
        ]
        if late_boundaries:
            raise RuntimeError(
                f"remat region {region_id} has FSDP inputs unavailable at "
                f"{prefetch_anchor.name}: {late_boundaries}"
            )

        with gm.graph.inserting_after(prefetch_anchor):
            barrier = gm.graph.call_function(
                _optimization_barrier,
                args=tuple(barrier_inputs),
            )
            barrier.meta["val"] = tuple(
                boundary.meta["val"] for boundary in barrier_inputs
            )
            barrier.meta["autograd_backward"] = True
            barrier.meta[_TPU_REMAT_BARRIER_KIND_META] = "fsdp_all_gather"
            barrier.meta[_TPU_REMAT_REGION_ID_META] = region_id

            barrier_outputs: dict[torch.fx.Node, torch.fx.Node] = {}
            insertion_point = barrier
            for index, boundary in enumerate(barrier_inputs):
                with gm.graph.inserting_after(insertion_point):
                    output = gm.graph.call_function(
                        operator.getitem, args=(barrier, index)
                    )
                output.meta = boundary.meta.copy()
                output.meta["autograd_backward"] = True
                output.meta[_TPU_REMAT_BARRIER_KIND_META] = "fsdp_all_gather"
                output.meta[_TPU_REMAT_REGION_ID_META] = region_id
                barrier_outputs[boundary] = output
                insertion_point = output

        for boundary, output in barrier_outputs.items():
            for user in redirects.get(boundary, ()):
                user.replace_input_with(boundary, output)

        barrier_count += 1
        all_gather_count += protected_all_gather_count

    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Inserted %d TPU FSDP remat barriers protecting %d recomputed all-gathers",
        barrier_count,
        all_gather_count,
    )
    return gm


def _collect_fsdp_all_gather_barriers(
    gm: torch.fx.GraphModule,
) -> dict[int, torch.fx.Node]:
    barriers: dict[int, torch.fx.Node] = {}
    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or node.target is not _optimization_barrier
            or node.meta.get(_TPU_REMAT_BARRIER_KIND_META) != "fsdp_all_gather"
        ):
            continue
        region_id = node.meta.get(_TPU_REMAT_REGION_ID_META)
        if not isinstance(region_id, int):
            raise RuntimeError(
                "TPU FSDP AG barrier requires a transformer layer ID, "
                f"got {region_id!r} on {node.name}"
            )
        if region_id in barriers:
            raise RuntimeError(
                f"remat region {region_id} has multiple TPU FSDP AG barriers"
            )
        barriers[region_id] = node
    return barriers


def _collect_rs_wait_deadlines(
    gm: torch.fx.GraphModule,
    barrier_by_region: dict[int, torch.fx.Node],
    *,
    n_layers: int,
) -> dict[torch.fx.Node, tuple[torch.fx.Node, ...]]:
    """Assign RS waits to the next transformer-layer AG bundle in FX order."""
    # The TPU pipeline does not currently combine parameters into larger
    # collective buckets. This shared helper also recognizes those unbucketed
    # collectives through its legacy metadata fallback.
    _, all_comms = _collect_bucketed_fsdp_comms(gm, n_layers=n_layers)
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    backward_comms = sorted(
        (comm for comm in all_comms if comm.direction == "bwd"),
        key=lambda comm: order[comm.launch],
    )

    pending_rs_waits: list[torch.fx.Node] = []
    seen_barriers: set[torch.fx.Node] = set()
    deadlines: dict[torch.fx.Node, tuple[torch.fx.Node, ...]] = {}
    for comm in backward_comms:
        if comm.kind == "rs":
            pending_rs_waits.append(comm.wait)
            continue
        if comm.kind != "ag":
            continue

        barrier = barrier_by_region.get(comm.logical_index)
        if barrier is None or barrier in seen_barriers:
            continue
        seen_barriers.add(barrier)
        if pending_rs_waits:
            deadlines[barrier] = tuple(pending_rs_waits)
            pending_rs_waits.clear()
    return deadlines


def _depends_on(node: torch.fx.Node, dependency: torch.fx.Node) -> bool:
    """Return whether adding dependency -> node would create a cycle."""
    pending = list(node.all_input_nodes)
    seen: set[torch.fx.Node] = set()
    while pending:
        candidate = pending.pop()
        if candidate is dependency:
            return True
        if candidate not in seen:
            seen.add(candidate)
            pending.extend(candidate.all_input_nodes)
    return False


def apply_tpu_fsdp_rs_deadline_schedule(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    n_layers: int,
    strict: bool = False,
) -> torch.fx.GraphModule:
    """Keep TPU LHS from moving later backward AGs ahead of scheduled RSs.

    ``schedule_fsdp_comms_to_dense_regions_pass`` places alternating backward
    AG/RS launches in FX order, but FX order is not an XLA dependency. TPU LHS
    can therefore regroup all AGs ahead of all RSs. This pass adds the RS waits
    since the previous layer's AG bundle to the existing AG barrier. No third
    barrier is created.

    An ordinary tensor dependency orders RS completion, not only RS launch. This
    is a deliberate prototype constraint, but may overconstrain hardware that
    can execute AG and RS independently.

    TODO: Replace this completion dependency with an RS-start to AG-start
    ordering primitive after TPU async collective materialization. If later
    profiles show that per-layer deadlines are too restrictive, revisit the
    experimentally explored blocked AG/RS schedule together with collective
    grouping; the standalone blocked schedule did not help the current 8B and
    30B graphs enough to justify a public tuning knob.
    """
    del example_inputs

    barrier_by_region = _collect_fsdp_all_gather_barriers(gm)
    if not barrier_by_region:
        return gm
    if all(
        _TPU_FSDP_RS_DEADLINE_META in barrier.meta
        for barrier in barrier_by_region.values()
    ):
        return gm

    deadlines = _collect_rs_wait_deadlines(
        gm,
        barrier_by_region,
        n_layers=n_layers,
    )
    added_waits = 0
    updated_barriers = 0
    skipped: list[str] = []
    for barrier, waits in deadlines.items():
        if _TPU_FSDP_RS_DEADLINE_META in barrier.meta:
            continue

        cyclic_waits = [wait.name for wait in waits if _depends_on(wait, barrier)]
        if cyclic_waits:
            skipped.append(
                f"{barrier.name}: RS waits depend on the AG barrier: {cyclic_waits}"
            )
            continue

        barrier.args = (*barrier.args, *waits)
        barrier.meta["val"] = (
            *barrier.meta["val"],
            *(wait.meta["val"] for wait in waits),
        )
        barrier.meta[_TPU_FSDP_RS_DEADLINE_META] = tuple(wait.name for wait in waits)
        added_waits += len(waits)
        updated_barriers += 1

    if skipped and strict:
        raise RuntimeError(
            "Could not add all TPU FSDP RS deadlines:\n"
            + "\n".join(f"- {reason}" for reason in skipped)
        )
    if added_waits:
        # Adding wait -> barrier data edges may invalidate the old textual FX
        # order. A stable topological sort restores it without manually moving
        # waits past existing optimizer consumers.
        _stable_topological_sort(gm.graph, {})
        gm.graph.lint()
        gm.recompile()
    logger.info(
        "Added %d RS completion dependencies to %d TPU FSDP AG barriers",
        added_waits,
        updated_barriers,
    )
    if skipped:
        logger.warning("Skipped TPU FSDP RS deadlines: %s", skipped[:4])
    return gm


__all__ = [
    "apply_tpu_fsdp_rs_deadline_schedule",
    "apply_tpu_remat_fsdp_all_gather_schedule",
    "apply_tpu_remat_schedule",
]
