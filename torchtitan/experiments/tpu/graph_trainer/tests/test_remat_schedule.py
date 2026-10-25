# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from types import SimpleNamespace
import unittest

from absl.testing import absltest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch import nn
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
    annotate_module_fqns,
    compute_parameter_gradients,
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _full_memory_policy_pass,
    _make_default_memory_policy,
    _make_full_memory_policy,
    tag_sac_policy,
)
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    remove_parameter_gradient_markers_pass,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    _collect_bucketed_fsdp_comms,
    schedule_fsdp_comms_to_dense_regions_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_fsdp import shard_module
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    _TPU_FSDP_RS_DEADLINE_META,
    _TPU_REMAT_BARRIER_KIND_META,
    _TPU_REMAT_LAYER_ENTRY_ANCHOR_META,
    _collect_fsdp_all_gather_barriers,
    _optimization_barrier,
    apply_tpu_fsdp_rs_deadline_schedule,
    apply_tpu_remat_fsdp_all_gather_schedule,
    apply_tpu_remat_schedule,
)


def _set_tensor_meta(
    node: torch.fx.Node,
    *,
    layer_id: int | None = None,
    backward: bool = False,
    recomputed: bool = False,
) -> torch.fx.Node:
    node.meta["val"] = torch.empty(4, 4, dtype=torch.bfloat16)
    if layer_id is not None:
        node.meta["custom"] = {"module_fqn": f"layers.{layer_id}.linear"}
    if backward:
        node.meta["autograd_backward"] = True
    if recomputed:
        node.name = f"{node.name}_recomputed"
    return node


class _Projection(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = x @ self.weight.to(torch.bfloat16)
        return output if mask is None else output * mask


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        # Mirror upstream d771abb0's module tree: `RoutedExperts` now sits
        # between the MoE and its grouped experts, so the expert parameters are
        # at `moe.routed_experts.inner_experts.*`. `remat_schedule` keys its
        # expert-gradient ordering off that path.
        self.moe = nn.Module()
        self.moe.routed_experts = nn.Module()
        self.moe.routed_experts.inner_experts = _Projection(dim)
        self.attention = _Projection(dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        experts = self.moe.routed_experts.inner_experts
        return torch.relu(experts(x)) + self.attention(x, mask)


class _Epilogue(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.square()


class _ToyModel(nn.Module):
    def __init__(
        self, dim: int = 4, n_layers: int = 2, masked: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Block(dim) for _ in range(n_layers))
        self.norm = _Epilogue()
        self.masked = masked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Like Splash segment IDs, the mask is made once outside the layers
        # and read by every layer's attention.
        mask = (x > 0).to(x.dtype) if self.masked else None
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def _shard_every_module(model, mesh, *, param_dtype, reduce_dtype=None):
    """Shard every module's direct parameters, as ``apply_tpu_fsdp`` does."""
    for module in model.modules():
        shard_module(
            module,
            mesh,
            "fully_shard",
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        )
    return model


def _trace_simple_fsdp_remat_graph(n_layers: int = 3):
    """Trace a real FSDP graph and add its layer-entry barriers."""
    assert not dist.is_initialized()
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
    try:
        mesh = DeviceMesh("cpu", [0, 1])
        with torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True):
            model = _ToyModel(dim=8, n_layers=n_layers)
            annotate_module_fqns(model)
            model = _shard_every_module(model, mesh, param_dtype=torch.bfloat16)
            inputs = torch.randn(
                2, 8, dtype=torch.bfloat16, requires_grad=True
            )

            def step(inputs: torch.Tensor) -> list[torch.Tensor]:
                loss = model(inputs).float().sum()
                named_params = list(model.named_parameters())
                gradients = compute_parameter_gradients(loss, named_params)
                return [loss, *gradients]

            traced = minimal_fx_tracer(step, module=model)(inputs)
    finally:
        dist.destroy_process_group()

    _full_memory_policy_pass(
        traced.gm,
        config=SimpleNamespace(
            parallelism=SimpleNamespace(
                fsdp_reshard_after_forward="always",
                pipeline_parallel_degree=1,
            ),
            # Upstream d771abb0 added `--compile.full_recompute_save_ops`,
            # read here by `_full_memory_policy_pass`. Its default, the empty
            # string, means "save nothing" -- full recompute, which is what
            # this test wants.
            compile=SimpleNamespace(full_recompute_save_ops=""),
        ),
    )
    selective_activation_remat_pass(traced.gm)
    apply_tpu_remat_schedule(
        traced.gm,
        traced.example_inputs,
        num_static_inputs=traced.num_static_inputs,
    )
    return traced


class TestTpuFsdpRematSchedule(unittest.TestCase):
    def test_all_gather_barrier_uses_configured_layer_distance(self) -> None:
        for distance, expected_anchor_ids in (
            (1, (1, 2, None)),
            (2, (2, None, None)),
        ):
            with self.subTest(distance=distance):
                traced = _trace_simple_fsdp_remat_graph()
                gm = traced.gm
                layer_entry_anchors = {
                    node.meta[_TPU_REMAT_LAYER_ENTRY_ANCHOR_META]: node
                    for node in gm.graph.nodes
                    if _TPU_REMAT_LAYER_ENTRY_ANCHOR_META in node.meta
                }
                output = next(node for node in gm.graph.nodes if node.op == "output")
                loss = output.args[0][0]

                apply_tpu_remat_fsdp_all_gather_schedule(
                    gm,
                    traced.example_inputs,
                    num_static_inputs=traced.num_static_inputs,
                    prefetch_distance=distance,
                )

                barriers = _collect_fsdp_all_gather_barriers(gm)
                self.assertEqual(set(barriers), {0, 1, 2})
                for layer_id, barrier in barriers.items():
                    expected_anchor_id = expected_anchor_ids[layer_id]
                    expected_anchor = (
                        loss
                        if expected_anchor_id is None
                        else layer_entry_anchors[expected_anchor_id]
                    )
                    *parameters, anchor = barrier.args
                    self.assertTrue(parameters)
                    self.assertIs(anchor, expected_anchor)
                    for index, parameter in enumerate(parameters):
                        barrier_output = next(
                            user
                            for user in barrier.users
                            if user.target is operator.getitem
                            and user.args[1] == index
                        )
                        self.assertTrue(barrier_output.users)
                        self.assertTrue(
                            all(
                                user.name.endswith("_recomputed")
                                for user in barrier_output.users
                            )
                        )
                        self.assertIn(barrier, parameter.users)
                        self.assertTrue(
                            any(user is not barrier for user in parameter.users)
                        )

                graph_after_first_application = str(gm.graph)
                apply_tpu_remat_fsdp_all_gather_schedule(
                    gm,
                    traced.example_inputs,
                    num_static_inputs=traced.num_static_inputs,
                    prefetch_distance=distance,
                )
                self.assertEqual(str(gm.graph), graph_after_first_application)

    def test_all_gather_barrier_requires_positive_prefetch_distance(self) -> None:
        traced = _trace_simple_fsdp_remat_graph()

        with self.assertRaisesRegex(ValueError, "must be positive"):
            apply_tpu_remat_fsdp_all_gather_schedule(
                traced.gm,
                traced.example_inputs,
                num_static_inputs=traced.num_static_inputs,
                prefetch_distance=0,
            )

    def test_all_gather_barrier_ignores_activation_collectives(self) -> None:
        graph = torch.fx.Graph()
        _set_tensor_meta(graph.placeholder("weight"))
        activation = _set_tensor_meta(graph.placeholder("activation"))
        all_gather = _set_tensor_meta(
            graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(activation, 1, "tp"),
            ),
            layer_id=0,
            backward=True,
            recomputed=True,
        )
        graph.output(all_gather)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        apply_tpu_remat_fsdp_all_gather_schedule(gm, (), num_static_inputs=1)

        self.assertFalse(
            any(
                node.meta.get(_TPU_REMAT_BARRIER_KIND_META) == "fsdp_all_gather"
                for node in gm.graph.nodes
            )
        )
        self.assertIs(all_gather.args[0], activation)

    def test_rs_deadline_reuses_all_gather_barriers(self) -> None:
        traced = _trace_simple_fsdp_remat_graph()
        gm = traced.gm
        apply_tpu_remat_fsdp_all_gather_schedule(
            gm,
            traced.example_inputs,
            num_static_inputs=traced.num_static_inputs,
        )
        schedule_fsdp_comms_to_dense_regions_pass(
            gm,
            traced.example_inputs,
            moe_layer_ids=frozenset(),
            n_layers=3,
            strict=False,
        )

        barriers = _collect_fsdp_all_gather_barriers(gm)
        _, comms = _collect_bucketed_fsdp_comms(gm, n_layers=3)
        order = {node: index for index, node in enumerate(gm.graph.nodes)}
        # Each region's first backward AG consumes every RS wait launched since
        # the preceding region's first AG.
        pending_rs_waits: list[torch.fx.Node] = []
        seen_regions: set[int] = set()
        expected_deadlines: dict[int, tuple[torch.fx.Node, ...]] = {}
        for comm in sorted(
            (comm for comm in comms if comm.direction == "bwd"),
            key=lambda comm: order[comm.launch],
        ):
            if comm.kind == "rs":
                pending_rs_waits.append(comm.wait)
            elif (
                comm.kind == "ag"
                and comm.logical_index is not None
                and comm.logical_index not in seen_regions
            ):
                seen_regions.add(comm.logical_index)
                if pending_rs_waits:
                    expected_deadlines[comm.logical_index] = tuple(
                        pending_rs_waits
                    )
                    pending_rs_waits.clear()

        self.assertEqual(set(expected_deadlines), {0, 1})
        early_wait = expected_deadlines[1][0]
        with gm.graph.inserting_after(early_wait):
            early_user = gm.graph.call_function(
                torch.ops.aten.alias.default, args=(early_wait,)
            )
        early_user.meta = early_wait.meta.copy()
        gm.graph.lint()
        gm.recompile()

        apply_tpu_fsdp_rs_deadline_schedule(
            gm,
            traced.example_inputs,
            n_layers=3,
            strict=True,
        )

        self.assertEqual(_collect_fsdp_all_gather_barriers(gm), barriers)
        for region_id, waits in expected_deadlines.items():
            barrier = barriers[region_id]
            self.assertEqual(
                barrier.meta[_TPU_FSDP_RS_DEADLINE_META],
                tuple(wait.name for wait in waits),
            )
            for wait in waits:
                self.assertIn(wait, barrier.args)
        self.assertNotIn(_TPU_FSDP_RS_DEADLINE_META, barriers[2].meta)

        order = {node: index for index, node in enumerate(gm.graph.nodes)}
        for region_id, waits in expected_deadlines.items():
            for wait in waits:
                self.assertLess(order[wait], order[barriers[region_id]])
        self.assertLess(order[early_wait], order[early_user])
        self.assertLess(order[early_user], order[barriers[1]])


def _nearest_parameter_fqns(node: torch.fx.Node) -> tuple[str, ...]:
    frontier = {node}
    seen: set[torch.fx.Node] = set()
    while frontier:
        fqns = {
            fqn
            for candidate in frontier
            for fqn in candidate.meta.get("custom", {}).get(
                PARAMETER_GRADIENT_FQNS_META, ()
            )
        }
        if fqns:
            return tuple(sorted(fqns))
        seen.update(frontier)
        frontier = {
            user
            for candidate in frontier
            for user in candidate.users
            if user not in seen
        }
    return ()


def _describe_barriers(gm: torch.fx.GraphModule) -> dict[int, dict[str, tuple]]:
    description = {}
    for barrier in gm.graph.nodes:
        if barrier.op != "call_function" or barrier.target is not _optimization_barrier:
            continue

        redirected_edges = []
        ordering_edges = []
        redirected_regions = set()
        outputs = {
            user.args[1]: user
            for user in barrier.users
            if user.op == "call_function" and user.target is operator.getitem
        }
        for index, value in enumerate(barrier.args):
            destinations = {
                (
                    _get_layer_id(user),
                    (
                        "recompute"
                        if user.name.endswith("_recomputed")
                        else "backward"
                        if _is_backward_node(user)
                        else "forward"
                    ),
                )
                for user in outputs[index].users
            }
            dtype = str(value.meta["val"].dtype)
            if destinations:
                redirected_regions.update(region_id for region_id, _ in destinations)
                redirected_edges.append((tuple(sorted(destinations)), dtype))
            else:
                ordering_edges.append((_nearest_parameter_fqns(value), dtype))

        assert len(redirected_regions) == 1
        description[redirected_regions.pop()] = {
            "redirected": tuple(sorted(redirected_edges)),
            "ordering": tuple(sorted(ordering_edges)),
        }
    return description


class TestTpuRematSchedule(unittest.TestCase):
    def test_traced_two_layer_schedule(self) -> None:
        for policy_name, policy_fn, masked in (
            ("full", _make_full_memory_policy(), False),
            ("sac", _make_default_memory_policy(set()), False),
            # Saving each layer's masked attention output leaves the mask to be
            # recomputed once, before its first reader, for every layer's
            # backward. That shared recompute must not be taken as a cotangent.
            ("full, saved masked attention", _make_full_memory_policy(), True),
        ):
            with self.subTest(policy=policy_name):
                torch.manual_seed(0)
                model = _ToyModel(masked=masked)
                annotate_module_fqns(model)

                def step(x: torch.Tensor) -> list[torch.Tensor]:
                    loss = model(x).float().sum()
                    named_params = [
                        (name, parameter)
                        for name, parameter in model.named_parameters(
                            remove_duplicate=False
                        )
                        if parameter.requires_grad
                    ]
                    grads = compute_parameter_gradients(loss, named_params)
                    updated_params = [
                        parameter - 0.01 * grad
                        for (_, parameter), grad in zip(
                            named_params, grads, strict=True
                        )
                    ]
                    return [loss, *updated_params]

                x = torch.randn(2, 4, dtype=torch.bfloat16)
                traced = minimal_fx_tracer(step, module=model)(x)
                expected_outputs = [
                    output.detach().clone()
                    for output in run_traced(traced, module=model)(x)
                ]
                remove_parameter_gradient_markers_pass(
                    traced.gm, traced.example_inputs
                )
                saved_attention = {
                    node
                    for node in traced.gm.graph.nodes
                    if masked
                    and node.target is torch.ops.aten.mul.Tensor
                    and not _is_backward_node(node)
                    and node.meta.get("custom", {})
                    .get("module_fqn", "")
                    .endswith(".attention")
                }
                tag_sac_policy(
                    traced.gm, policy_fn=policy_fn, force_save_nodes=saved_attention
                )
                selective_activation_remat_pass(traced.gm)
                # The masked case only guards the fix while the mask is
                # recomputed once and read by both layers' backward.
                shared_recomputes = [
                    node
                    for node in traced.gm.graph.nodes
                    if node.name.endswith("_recomputed")
                    and {_get_layer_id(user) for user in node.users} == {0, 1}
                ]
                self.assertEqual(len(shared_recomputes), 1 if masked else 0)
                apply_tpu_remat_schedule(
                    traced.gm,
                    traced.example_inputs,
                    num_static_inputs=traced.num_static_inputs,
                )

                # This single topology comparison proves that every layer entry
                # redirects its saved activation and incoming cotangent through
                # one barrier; layer 0 additionally waits for both layer 1 local
                # dWeights even though an optimizer-like update consumes the
                # tagged gradients instead of returning them. The non-expert
                # path preserves the generic FSDP ordering point at the FP32
                # collective input. The validated expert specialization instead
                # orders on BF16 local dWeight, leaving its cast and collective
                # outside the barrier and available to overlap. Generalizing
                # that ordering point to all eligible gradients is a follow-up.
                self.assertEqual(
                    _describe_barriers(traced.gm),
                    {
                        0: {
                            "redirected": (
                                (((0, "backward"),), "torch.bfloat16"),
                                (((0, "recompute"),), "torch.bfloat16"),
                            ),
                            "ordering": (
                                (("layers.1.attention.weight",), "torch.float32"),
                                (("layers.1.moe.routed_experts.inner_experts.weight",), "torch.bfloat16"),
                            ),
                        },
                        1: {
                            "redirected": (
                                (((1, "backward"),), "torch.bfloat16"),
                                (((1, "recompute"),), "torch.bfloat16"),
                            ),
                            "ordering": (),
                        },
                    },
                )

                actual_outputs = run_traced(traced, module=model)(x)
                for actual, expected in zip(
                    actual_outputs, expected_outputs, strict=True
                ):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestScanRecomputePolicy(unittest.TestCase):

    def test_a_scan_over_the_layers_is_never_recomputed(self) -> None:
        """Recomputing one scan node reruns the whole decoder loop in backward.

        It is the one recompute decision here with no visible symptom: the loss
        stays exact and the saved set really does shrink, so the memory number
        improves while the step does twice the forward work.
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        xs = graph.placeholder("xs")
        node = graph.call_function(
            torch.ops.higher_order.scan, args=(None, [x], [xs], ())
        )
        graph.output(node)
        gm = torch.fx.GraphModule(nn.Module(), graph)

        tag_sac_policy(gm, policy_fn=_make_full_memory_policy())

        self.assertEqual(node.meta["recompute"], CheckpointPolicy.MUST_SAVE)


if __name__ == "__main__":
    absltest.main()
