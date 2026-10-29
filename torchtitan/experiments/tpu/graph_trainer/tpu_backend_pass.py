# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Register and construct the 'tpu' pass pipeline for torchtitan's GraphTrainer."""

from __future__ import annotations

from collections import Counter
import functools
import logging
import operator
import time
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch._dynamo.utils import detect_fake_mode
from torch.utils import _pytree

from torchtitan.experiments.graph_trainer.cpu_offload import (
    apply_cpu_offload_pass,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    schedule_fsdp_comms_to_dense_regions_pass,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    tag_with_memory_policy_pass,
)
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    remove_parameter_gradient_markers_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    apply_tpu_fsdp_rs_deadline_schedule,
    apply_tpu_remat_fsdp_all_gather_schedule,
    apply_tpu_remat_schedule,
)
# Imported for the side effect: they register TPU-only memory policies before
# ``tag_with_memory_policy_pass`` looks them up by name.
import torchtitan.experiments.tpu.graph_trainer.tpu_host_offload  # noqa: F401
import torchtitan.experiments.tpu.graph_trainer.tpu_memory_policy  # noqa: F401
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


def _import_static_compiler():
    """Lazy import so this module is safe to import across environments."""
    from torch_tpu._internal.compile.compiler import StaticCompiler  # pytype: disable=import-error # noqa: F401
    return StaticCompiler


def _optimization_barrier_fake(
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    outputs = []
    for value in inputs:
        output = torch.empty_strided(
            value.shape,
            value.stride(),
            dtype=value.dtype,
            device=value.device,
            requires_grad=value.requires_grad,
        )
        if value.storage_offset():
            output = output.as_strided(
                value.shape,
                value.stride(),
                value.storage_offset(),
            )
        outputs.append(output)
    return outputs


def _ensure_optimization_barrier_fake() -> None:
    # Temporary compatibility shim for TorchTPU 0.0.42.1. The upstream fix is
    # https://github.com/google-pytorch/torch_tpu/pull/2562.
    op_name = "tpu::_optimization_barrier"
    if torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "Meta"):
        return
    torch.library.register_fake(op_name, _optimization_barrier_fake)


def _tensor_storage_id(value: Any) -> int | None:
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        return None
    return value.untyped_storage()._cdata


def _submodule_outputs(
    submodule: torch.fx.GraphModule,
) -> tuple[torch.fx.Node, ...]:
    output = next(node for node in submodule.graph.nodes if node.op == "output")
    return tuple(
        value
        for value in _pytree.tree_leaves(output.args[0])
        if isinstance(value, torch.fx.Node)
    )


def _producer_output_storage(
    parent: torch.fx.GraphModule,
    source: torch.fx.Node,
) -> int | None:
    """Resolve a parent-graph value to the storage produced by a split child.

    ``split_module`` represents a child output as either its ``call_module``
    node (one output) or ``getitem(call_module, index)`` (multiple outputs).
    Follow that reference into the child GraphModule and read the fake tensor
    storage ID from the corresponding output. Values that are not child
    boundaries, or whose output cannot be identified exactly, return ``None``.
    """
    producer = source
    output_index = 0
    if (
        source.op == "call_function"
        and source.target is operator.getitem
        and len(source.args) == 2
        and isinstance(source.args[0], torch.fx.Node)
        and isinstance(source.args[1], int)
    ):
        producer = source.args[0]
        output_index = source.args[1]
    if producer.op != "call_module":
        return None

    outputs = _submodule_outputs(parent.get_submodule(str(producer.target)))
    if source is producer and len(outputs) != 1:
        return None
    if output_index < 0 or output_index >= len(outputs):
        return None
    return _tensor_storage_id(outputs[output_index].meta.get("val"))


def _donatable_input_positions(
    parent: torch.fx.GraphModule,
    call: torch.fx.Node,
    last_use_by_storage: dict[int, int],
    node_order: dict[torch.fx.Node, int],
    protected_storages: set[int],
) -> tuple[int, ...]:
    """Select child inputs whose storage is dead after this parent-graph call.

    Safety is computed over the whole parent before this function is called:
    ``last_use_by_storage`` records the final parent node that consumes each
    split-boundary storage. An input position is donatable only when this call
    is that final consumer, the storage appears once in the call arguments, it
    is not an original graph input or constant, and the child does not return
    an alias of it. XLA may then reuse the input buffer for this child because
    no later child or graph output can observe its previous contents.
    """
    child = parent.get_submodule(str(call.target))
    output_storages = {
        storage
        for output in _submodule_outputs(child)
        if (storage := _tensor_storage_id(output.meta.get("val"))) is not None
    }
    candidates = []
    for position, source in enumerate(call.args):
        if not isinstance(source, torch.fx.Node):
            continue
        storage = _producer_output_storage(parent, source)
        if (
            storage is None
            or last_use_by_storage.get(storage) != node_order[call]
        ):
            continue
        candidates.append((position, storage))

    storage_counts = Counter(storage for _, storage in candidates)
    return tuple(
        position
        for position, storage in candidates
        if storage_counts[storage] == 1
        and storage not in protected_storages
        and storage not in output_storages
    )


def _annotate_internal_donations(
    gm: torch.fx.GraphModule,
    *,
    donate_internal_buffers: bool = True,
) -> None:
    """Annotate forced donors and safe last-use boundaries between children.

    An internal boundary is a tensor produced by one ``call_module`` child and
    passed to another. The analysis first resolves boundary aliases by fake
    tensor storage ID, then scans the complete parent graph to find each
    storage's final consumer. At that consumer, a boundary is donated only if
    it is uniquely referenced, is not an original input or constant, and is
    not also returned by the consuming child. Missing alias metadata disables
    donation for the graph rather than guessing. Inputs explicitly marked
    ``tpu_force_donate`` are always included; ``donate_internal_buffers`` gates
    only inference of additional ordinary device-buffer donors.
    """
    calls = [node for node in gm.graph.nodes if node.op == "call_module"]
    submodules = [gm.get_submodule(str(call.target)) for call in calls]
    for submodule in submodules:
        submodule.meta.pop("tpu_donated_input_positions", None)

    node_order: dict[torch.fx.Node, int] = {}
    last_use_by_storage: dict[int, int] = {}
    protected_storages: set[int] = set()
    inference_disabled_due_to_metadata = False
    if donate_internal_buffers:
        protected_nodes = [
            node
            for node in gm.graph.nodes
            if node.op in ("placeholder", "get_attr")
        ]
        boundary_outputs = [
            output
            for submodule in submodules
            for output in _submodule_outputs(submodule)
        ]
        missing_values = [
            node.name
            for node in (*protected_nodes, *boundary_outputs)
            if "tensor_meta" in node.meta
            and not isinstance(node.meta.get("val"), torch.Tensor)
        ]
        missing_values.extend(
            node.name for node in boundary_outputs if "val" not in node.meta
        )
        if missing_values:
            logger.warning(
                "Disabling inferred TPU split boundary donation: tensor "
                "metadata lacks values for %s",
                missing_values,
            )
            donate_internal_buffers = False
            inference_disabled_due_to_metadata = True
        else:
            node_order = {
                node: index for index, node in enumerate(gm.graph.nodes)
            }
            for node in gm.graph.nodes:
                if node.op == "call_function" and node.target is operator.getitem:
                    continue
                for source in node.all_input_nodes:
                    storage = _producer_output_storage(gm, source)
                    if storage is not None:
                        last_use_by_storage[storage] = node_order[node]

            protected_storages = {
                storage
                for node in protected_nodes
                if (storage := _tensor_storage_id(node.meta.get("val")))
                is not None
            }

    donated_count = 0
    donated_bytes = 0
    for call, submodule in zip(calls, submodules, strict=True):
        inferred_positions = (
            _donatable_input_positions(
                gm,
                call,
                last_use_by_storage,
                node_order,
                protected_storages,
            )
            if donate_internal_buffers
            else ()
        )
        placeholders = [
            node for node in submodule.graph.nodes if node.op == "placeholder"
        ]
        forced_positions = tuple(
            position
            for position, placeholder in enumerate(placeholders)
            if placeholder.meta.get("tpu_force_donate", False)
        )
        if inference_disabled_due_to_metadata and not forced_positions:
            continue
        positions = tuple(dict.fromkeys((*inferred_positions, *forced_positions)))
        # _SubmodCompiler deep-copies children; GraphModule.meta is preserved.
        submodule.meta["tpu_donated_input_positions"] = positions
        for position in positions:
            value = placeholders[position].meta.get("val")
            if isinstance(value, torch.Tensor):
                donated_bytes += value.numel() * value.element_size()
        donated_count += len(positions)
    logger.info(
        "TPU split boundary donation: %d child inputs (%d aggregate bytes)",
        donated_count,
        donated_bytes,
    )


def tpu_backend_compile_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    is_fwd: bool = True,
    donated_input_positions: tuple[int, ...] = (),
) -> torch.fx.GraphModule:
    """Terminal pass in the TPU graph pipeline: lowers gm into a static XLA binary.

    This pass inspects the traced GraphModule (`gm`), logs op and collective statistics
    when debug logging is enabled, filters out non-Tensor slots if necessary, and
    compiles the graph using `StaticCompiler` (`lazy PJRT/XLA lowering`). The forward
    method of the module is then rebound to execute the compiled XLA executable directly,
    eliminating per-node host dispatch overhead during training steps.
    """
    StaticCompiler = _import_static_compiler()

    example_inputs = _restore_pinned_host_compile_inputs(gm, example_inputs)

    # Only iterate over nodes and count op/collective statistics when debug logging is enabled.
    if logger.isEnabledFor(logging.DEBUG):
        op_counter: Counter = Counter()
        for n in gm.graph.nodes:
            if n.op == "call_function":
                op_counter[str(n.target)] += 1
        ag = sum(c for t, c in op_counter.items() if "all_gather" in t)
        rs = sum(c for t, c in op_counter.items() if "reduce_scatter" in t)
        wait = sum(c for t, c in op_counter.items() if "wait_tensor" in t)
        ob = sum(c for t, c in op_counter.items() if "optimization_barrier" in t)
        n_nodes = sum(1 for _ in gm.graph.nodes)
        n_call = sum(1 for n in gm.graph.nodes if n.op == "call_function")
        logger.debug(
            f"[tpu_backend_pass] graph in: {n_nodes} nodes "
            f"({n_call} call_function); collectives: {ag} all_gather, "
            f"{rs} reduce_scatter, {wait} wait_tensor; "
            f"optimization_barrier nodes: {ob}"
        )

    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    original_outs = output_node.args[0]
    # StaticCompiler returns tensor values but does not write an input buffer
    # back merely because make_fx recorded add_ on it. MoE usage counters are
    # otherwise dead outputs, silently disabling the optimizer's balancer.
    # Lift direct additive buffer updates into explicit executable outputs,
    # then restore them at the Python boundary. Leave parameter/view mutation
    # handling unchanged; this contract is for direct, non-grad input buffers.
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    input_positions = {n: i for i, n in enumerate(placeholders)}
    buffer_updates = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.add_.Tensor:
            source = node.args[0]
            value = (
                source.meta.get("val") if isinstance(source, torch.fx.Node) else None
            )
            if (
                source in input_positions
                and isinstance(value, torch.Tensor)
                and not value.requires_grad
            ):
                buffer_updates[input_positions[source]] = node
    if buffer_updates:
        logger.info("Preserving %d additive input-buffer updates", len(buffer_updates))
    tensor_outs: list = []
    non_tensor_slots: list[tuple[int, object]] = []
    for i, out in enumerate(original_outs):
        if out is None:
            tensor_outs.append(out)
        elif isinstance(out, torch.fx.Node):
            val = out.meta.get("val")
            if val is None or isinstance(val, torch.Tensor):
                tensor_outs.append(out)
            else:
                non_tensor_slots.append((i, val))
        else:
            non_tensor_slots.append((i, out))
    num_tensor_outputs = len(tensor_outs)
    if non_tensor_slots:
        logger.info(
            f"[tpu_backend_pass] filtering {len(non_tensor_slots)} non-Tensor "
            f"output(s) before TpuBackend lowering: positions "
            f"{[p for p, _ in non_tensor_slots]}"
        )
    if non_tensor_slots or buffer_updates:
        output_node.args = (tuple(tensor_outs) + tuple(buffer_updates.values()),)
        gm.graph.lint()
        gm.recompile()

    t0 = time.time()
    logger.info("[tpu_backend_pass] StaticCompiler(gm, example_inputs) ...")
    donated_positions = set(donated_input_positions)
    donated_inputs = []
    tensor_index = 0
    for position, value in enumerate(example_inputs):
        if not isinstance(value, torch.Tensor):
            continue
        if position in donated_positions:
            donated_inputs.append(tensor_index)
        tensor_index += 1
    compiler_kwargs = {"is_fwd": is_fwd}
    if donated_inputs:
        # Added in TorchTPU PR #2800. Keep this keyword conditional so the
        # default no-donation path also works with older TorchTPU builds.
        compiler_kwargs["donated_inputs"] = donated_inputs
    executable = StaticCompiler()(gm, list(example_inputs), **compiler_kwargs)
    elapsed = time.time() - t0
    logger.info(
        f"[tpu_backend_pass] StaticCompiler ready in {elapsed:.1f}s "
        "(PJRT compile finishes lazily on first execute)"
    )

    gm._tpu_executable = executable

    if non_tensor_slots or buffer_updates:
        def _compiled_forward(*flat_inputs):
            outs = list(executable(*flat_inputs))
            with torch.no_grad():
                for position, value in zip(
                    buffer_updates, outs[num_tensor_outputs:], strict=True
                ):
                    flat_inputs[position].copy_(value)
            del outs[num_tensor_outputs:]
            for pos, val in non_tensor_slots:
                outs.insert(pos, val)
            return tuple(outs)
    else:
        def _compiled_forward(*flat_inputs):
            return executable(*flat_inputs)

    gm.forward = _compiled_forward  # type: ignore[method-assign]
    return gm


def _restore_pinned_host_compile_inputs(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
) -> tuple:
    """Represent AO host inputs as TPU tensors in PJRT pinned-host memory.

    Forward/backward separation and ``split_module`` propagate an AO host
    output as a CPU fake tensor. Its runtime value is instead a TPU tensor
    backed by PJRT pinned-host memory. StaticCompiler reads that placement from
    a TPU fake tensor's ``_is_pinned_host`` marker, so reconstruct the compiler
    input for both whole and partitioned graphs.
    """
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    pinned_positions = []
    for index, placeholder in enumerate(placeholders):
        metadata_value = placeholder.meta.get("val")
        reload_consumer = any(
            user.op == "call_function"
            and user.target is torch.ops.ao.reload.default
            and user.args
            and user.args[0] is placeholder
            for user in placeholder.users
        )
        if (
            isinstance(metadata_value, torch.Tensor)
            and metadata_value.device.type == "cpu"
            and (placeholder.meta.get("tpu_force_donate") or reload_consumer)
        ):
            pinned_positions.append(index)

    if not pinned_positions:
        return example_inputs
    if len(placeholders) != len(example_inputs):
        raise ValueError(
            "TPU graph input count does not match its placeholders: "
            f"{len(example_inputs)} != {len(placeholders)}"
        )

    fake_mode = detect_fake_mode(example_inputs)
    if fake_mode is None:
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
    compile_inputs = list(example_inputs)
    with fake_mode:
        for index in pinned_positions:
            value = compile_inputs[index]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    "TPU pinned-host graph input must have a Tensor example: "
                    f"position {index} is {type(value).__name__}"
                )
            pinned_value = torch.empty_strided(
                value.shape,
                value.stride(),
                dtype=value.dtype,
                device="tpu",
                requires_grad=value.requires_grad,
            )
            pinned_value._is_pinned_host = True
            compile_inputs[index] = pinned_value
    return tuple(compile_inputs)


def tpu_backend_compile_partitioned_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    is_fwd: bool = True,
    donate_internal_buffers: bool = False,
) -> torch.fx.GraphModule:
    """Discover child inputs and compile each split GraphModule independently."""
    # TorchTPU is an optional dependency, so importing its private split helper
    # at module load time would break host-only TorchTitan tests.
    from torch_tpu._internal.compile.split_compiler import _SubmodCompiler

    _ensure_optimization_barrier_fake()
    fake_mode = detect_fake_mode(example_inputs)
    if fake_mode is None:
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

    _annotate_internal_donations(
        gm,
        donate_internal_buffers=donate_internal_buffers,
    )

    def compile_submodule(
        submodule: torch.fx.GraphModule,
        submodule_inputs: list[torch.Tensor],
        *,
        module_name: str | None = None,
    ) -> torch.fx.GraphModule:
        # Newer TorchTPU passes this diagnostic name to the callback, while
        # the canonical build calls it with only the module and inputs.
        del module_name
        return tpu_backend_compile_pass(
            submodule,
            tuple(submodule_inputs),
            is_fwd=is_fwd,
            donated_input_positions=tuple(
                submodule.meta.get("tpu_donated_input_positions", ())
            ),
        )

    compiler = _SubmodCompiler(gm, compile_submodule, fake_mode)
    # ProcessGroups can cross split-module boundaries as opaque inputs. The
    # FakeTensor dispatch cache compares inputs while propagating each child's
    # output shapes, but torchbind ProcessGroups do not implement equality.
    # This setting is scoped to child-output propagation and does not alter
    # graph tracing or TorchTPU executable caches.
    cache_was_enabled = fake_mode.cache_enabled
    fake_mode.cache_enabled = False
    # Match TorchTPU's SplitCompiler: fake execution must not fire user saved-
    # tensor pack/unpack hooks that are intended to run once at runtime.
    with (
        torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing(),
    ):
        try:
            compiler.run(*example_inputs)
        finally:
            fake_mode.cache_enabled = cache_was_enabled
    gm.recompile()
    return gm


def construct_tpu_transform_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
) -> list[Callable]:
    """Build transforms that must run on the complete joint FX graph.

    Graph splitting consumes remat and FSDP metadata produced here, so these
    transforms run once before partitioning. Compilation is deliberately absent:
    split mode compiles each resulting child graph independently.
    """
    passes: list[Callable] = [remove_parameter_gradient_markers_pass]

    memory_policy = getattr(config.compile, "memory_policy", "none")
    fsdp_barrier_policy = config.compile.fsdp_barrier_policy
    if fsdp_barrier_policy != "none" and (
        memory_policy in (None, "none")
        or not config.compile.enable_remat_barriers
    ):
        raise ValueError(
            "compile.fsdp_barrier_policy requires GraphTrainer "
            "rematerialization and compile.enable_remat_barriers"
        )

    if memory_policy and memory_policy != "none":
        passes.append(
            functools.partial(tag_with_memory_policy_pass, config=config)
        )
        # Same order as the CUDA pipeline in ``graph_trainer/passes.py``: tag,
        # then inject, then remat. A no-op unless the policy tagged something
        # MUST_CPU_OFFLOAD, but without it the offloading policies set their
        # tags and this pipeline drops them -- the run is correct and uses
        # exactly the memory it would have used with offloading off.
        passes.append(
            functools.partial(
                apply_cpu_offload_pass,
                # Split mode places complete H2D pairs by executable after
                # partition ownership is known. Keep the generic pre-split
                # rewrite at the first consumer in that mode.
                prefetch_lookahead=(
                    0
                    if config.compile.graph_split.enabled
                    else config.compile.cpu_offload_prefetch_n_layers
                ),
                defer_n_layers=(
                    0
                    if config.compile.graph_split.enabled
                    else config.compile.cpu_offload_defer_n_layers
                ),
            )
        )
        passes.append(selective_activation_remat_pass)
        if config.compile.enable_remat_barriers:
            passes.append(
                functools.partial(
                    apply_tpu_remat_schedule,
                    num_static_inputs=traced_result.num_static_inputs,
                )
            )
            if fsdp_barrier_policy != "none":
                passes.append(
                    functools.partial(
                        apply_tpu_remat_fsdp_all_gather_schedule,
                        num_static_inputs=traced_result.num_static_inputs,
                        prefetch_distance=(
                            config.compile.fsdp_barrier_prefetch_distance
                        ),
                    )
                )
        logger.info(
            f"[tpu_backend_pass] activation checkpointing enabled "
            f"(memory_policy={memory_policy!r}); prepended "
            f"tag_with_memory_policy + selective_activation_remat"
        )

    from torchtitan.experiments.tpu.graph_trainer.passes.insert_combine_barriers import (
        apply_combine_barrier_pass,
    )
    passes.append(apply_combine_barrier_pass)

    if fsdp_barrier_policy != "none":
        model_config = config.model_spec.model
        n_layers = len(model_config.layers)
        # TransformerBlock.Config exposes ``moe`` independently of the concrete
        # expert implementation, including TPU-converted GroupedExperts.
        moe_layer_ids = frozenset(
            index
            for index, layer in enumerate(model_config.layers)
            if layer.moe is not None
        )
        passes.append(
            functools.partial(
                schedule_fsdp_comms_to_dense_regions_pass,
                moe_layer_ids=moe_layer_ids,
                n_layers=n_layers,
                strict=False,
            )
        )
        if fsdp_barrier_policy == "interleave":
            passes.append(
                functools.partial(
                    apply_tpu_fsdp_rs_deadline_schedule,
                    n_layers=n_layers,
                    strict=False,
                )
            )

    parameter_patterns = config.compile.fsdp_allreduce_slice_parameter_patterns
    if parameter_patterns:
        from torchtitan.experiments.tpu.graph_trainer.passes.reduce_scatter_allreduce import (
            apply_selected_reduce_scatter_allreduce_pass,
        )

        passes.append(
            functools.partial(
                apply_selected_reduce_scatter_allreduce_pass,
                parameter_patterns=parameter_patterns,
                max_input_bytes=(
                    config.compile.fsdp_allreduce_slice_max_input_bytes
                ),
            )
        )
    return passes


def construct_tpu_pass_pipeline(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
) -> list[Callable]:
    """Build the normal unsplit pipeline: joint transforms, then compilation."""
    return [
        *construct_tpu_transform_passes(traced_result, config),
        tpu_backend_compile_pass,
    ]


# Global registry dict so both construct_tpu_pass_pipeline and PASS_PIPELINE_REGISTRY["tpu"] work
PASS_PIPELINE_REGISTRY: dict[str, Callable] = {
    "tpu": construct_tpu_pass_pipeline
}


def register() -> None:
    """Ensure the 'tpu' pipeline is registered."""
    PASS_PIPELINE_REGISTRY["tpu"] = construct_tpu_pass_pipeline


__all__ = [
    "tpu_backend_compile_pass",
    "tpu_backend_compile_partitioned_pass",
    "construct_tpu_transform_passes",
    "construct_tpu_pass_pipeline",
    "register",
    "PASS_PIPELINE_REGISTRY",
]
