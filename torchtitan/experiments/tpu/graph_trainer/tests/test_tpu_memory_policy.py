# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the TPU selective memory policies on traced toy graphs."""

import operator
from types import SimpleNamespace
import unittest

from absl.testing import absltest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
    _NOT_IN_LAYERS,
    annotate_module_fqns,
    compute_parameter_gradients,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _find_fsdp_unshard_save_nodes,
)
from torchtitan.experiments.graph_trainer.registry import MEMORY_POLICY_REGISTRY
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    remove_parameter_gradient_markers_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_fsdp import shard_module
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    _optimization_barrier,
    apply_tpu_remat_schedule,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_memory_policy import (
    full_remat_with_saves,
    is_splash_forward_op_name,
    resolve_save_regions,
    ROUTING_REGION,
    SaveRegion,
    select_route_permutations,
    select_routing,
    select_routing_decisions,
    select_saves,
    select_splash_attention,
    SPLASH_ATTENTION_REGION,
)
from torchtitan.experiments.tpu.remat_regions import MOE_ROUTING, REMAT_REGION
from torchtitan.experiments.tpu.token_dispatcher import (
    local_reorder,
    stable_grouped_reduce,
)
from torchtitan.models.common.linear import RouterGateLinear
from torchtitan.models.common.moe import TokenChoiceTopKRouter

_TOKENS = 8
_DIM = 8
_EXPERTS = 4
_TOP_K = 2
_HEADS = 2


def _parallelism(reshard_after_forward: str) -> SimpleNamespace:
    return SimpleNamespace(
        fsdp_reshard_after_forward=reshard_after_forward,
        pipeline_parallel_degree=1,
    )


def _register_splash_like_op(name: str, model_layout: bool):
    """Register ``pallas::<name>``: (q, k, v) -> (output, LSE), like Splash.

    Like the real kernels, backward is a separate op that reads the saved
    output and LSE. The LSE is (batch, heads, seq); the output keeps the
    input layout, which is (batch, heads, seq, dim) for stock Splash and
    (batch, seq, heads, dim) for Tokamax.
    """

    def lse_of(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        lse = (q.float() * k.float()).sum(-1)
        return lse.transpose(1, 2).contiguous() if model_layout else lse

    @torch.library.custom_op(f"pallas::{name}", mutates_args=())
    def forward(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q * k + v, lse_of(q, k)

    @forward.register_fake
    def _(q, k, v):
        return torch.empty_like(q), lse_of(q, k)

    backward_name = name.replace("_fwd", "_bwd")

    @torch.library.custom_op(f"pallas::{backward_name}", mutates_args=())
    def backward_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del v, output, lse
        return grad_output * k, grad_output * q, grad_output.clone()

    @backward_op.register_fake
    def _(q, k, v, output, lse, grad_output):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs, *output)

    def backward(ctx, grad_output, grad_lse):
        del grad_lse
        return backward_op(*ctx.saved_tensors, grad_output)

    forward.register_autograd(backward, setup_context=setup_context)
    return forward


_STOCK_STYLE_SPLASH = _register_splash_like_op(
    "splash_attn_fwd_0123456789abcdef", model_layout=False
)
# Same stock-style name pattern; returns Tokamax's model layout.
_MODEL_LAYOUT_SPLASH = _register_splash_like_op(
    "splash_attn_fwd_fedcba9876543210", model_layout=True
)


class _SplashCore:
    """Calls a Splash-like op the way ``layers.N.attention`` does."""

    def __init__(self, op, model_layout: bool) -> None:
        self.op = op
        self.model_layout = model_layout

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        tokens, dim = x.shape
        heads = x.reshape(1, tokens, _HEADS, dim // _HEADS)
        if not self.model_layout:
            heads = heads.transpose(1, 2)
        output, _ = self.op(heads, heads * 0.5, heads)
        if not self.model_layout:
            output = output.transpose(1, 2)
        return output.reshape(tokens, dim)


def _splash_cores():
    """Layers 0 and 2 use kernel layout, layer 1 model layout."""
    return (
        _SplashCore(_STOCK_STYLE_SPLASH, model_layout=False),
        _SplashCore(_MODEL_LAYOUT_SPLASH, model_layout=True),
        _SplashCore(_STOCK_STYLE_SPLASH, model_layout=False),
    )


class _DenseBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.weight.to(x.dtype))


class _DenseToy(nn.Module):
    def __init__(self, dim: int = 8, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_DenseBlock(dim) for _ in range(n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _Attention(nn.Module):
    """A projection, optionally followed by an attention-kernel stand-in."""

    def __init__(self, dim: int, core=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x @ self.weight.to(x.dtype)
        return q if self.core is None else self.core(q)


def _router(num_expert_groups: int | None) -> TokenChoiceTopKRouter:
    """Upstream's router: softmax top-2 of 4 experts, optionally in 2 groups.

    With groups, the router keeps the best of 2 groups and then takes the
    top-2 of its experts, so it runs three top-ks in all.
    """
    return TokenChoiceTopKRouter.Config(
        num_experts=_EXPERTS,
        gate=RouterGateLinear.Config(in_features=_DIM, out_features=_EXPERTS),
        num_expert_groups=num_expert_groups,
        num_limited_groups=None if num_expert_groups is None else 1,
        top_k=_TOP_K,
        score_func="softmax",
    ).build()


class _MoE(nn.Module):
    """Upstream routing, then the real TPU local reorder and grouped reduce."""

    def __init__(self, dim: int, num_expert_groups: int | None) -> None:
        super().__init__()
        self.router = _router(num_expert_groups)
        self.expert = nn.Parameter(torch.randn(dim, dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top_scores, expert_ids, _ = self.router(x)
        routed, token_indices, routed_scores = local_reorder(
            x, top_scores, expert_ids, _TOP_K
        )
        expert_out = (routed @ self.expert.to(x.dtype)).float()
        weighted = (expert_out * routed_scores.reshape(-1, 1)).to(x.dtype)
        return stable_grouped_reduce(weighted, token_indices, _TOP_K)


class _Block(nn.Module):
    def __init__(self, dim: int, core, num_expert_groups: int | None) -> None:
        super().__init__()
        self.attention = _Attention(dim, core)
        self.moe = _MoE(dim, num_expert_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        return x + self.moe(x)


class _MoEToy(nn.Module):
    def __init__(
        self, cores=(None, None), num_expert_groups: int | None = None
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _Block(_DIM, core, num_expert_groups) for core in cores
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _trace_toy(model: nn.Module):
    """Trace one training step of ``model``; returns (traced, input)."""
    annotate_module_fqns(model)

    def step(x: torch.Tensor) -> list[torch.Tensor]:
        loss = model(x).float().sum()
        named_params = [
            (name, parameter)
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        ]
        grads = compute_parameter_gradients(loss, named_params)
        updated = [
            parameter - 0.01 * grad
            for (_, parameter), grad in zip(named_params, grads, strict=True)
        ]
        return [loss, *updated]

    x = torch.randn(_TOKENS, _DIM, dtype=torch.bfloat16)
    return minimal_fx_tracer(step, module=model)(x), x


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


def _trace_simple_fsdp_graph() -> torch.fx.GraphModule:
    """Trace a two-rank FSDP toy on a fake process group."""
    assert not dist.is_initialized()
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
    try:
        mesh = DeviceMesh("cpu", [0, 1])
        with torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True):
            model = _DenseToy()
            annotate_module_fqns(model)
            model = _shard_every_module(model, mesh, param_dtype=torch.bfloat16)
            inputs = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)

            def step(inputs: torch.Tensor) -> list[torch.Tensor]:
                loss = model(inputs).float().sum()
                named_params = list(model.named_parameters())
                return [loss, *compute_parameter_gradients(loss, named_params)]

            traced = minimal_fx_tracer(step, module=model)(inputs)
    finally:
        dist.destroy_process_group()
    return traced.gm


class TestFullRematWithSaves(unittest.TestCase):
    def test_zero2_keeps_all_gather_outputs_saved(self) -> None:
        for reshard_after_forward, expected in (
            ("never", CheckpointPolicy.MUST_SAVE),
            ("always", CheckpointPolicy.MUST_RECOMPUTE),
        ):
            with self.subTest(reshard_after_forward=reshard_after_forward):
                gm = _trace_simple_fsdp_graph()
                unshards = _find_fsdp_unshard_save_nodes(gm)
                self.assertTrue(unshards)

                # No policy saves of its own: only the FSDP contract applies.
                full_remat_with_saves(
                    gm,
                    SimpleNamespace(parallelism=_parallelism(reshard_after_forward)),
                    save_nodes=(),
                )

                self.assertEqual(
                    {node.meta["recompute"] for node in unshards}, {expected}
                )


def _router_topks(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    return [
        node
        for node in gm.graph.nodes
        if node.target is torch.ops.aten.topk.default
        and not _is_backward_node(node)
        and node.meta["custom"]["module_fqn"].endswith(".moe.router")
    ]


def _hand_built_routing_graph(
    with_topk: bool, with_routes: bool
) -> torch.fx.GraphModule:
    """A layer-0 graph with a router top-k, a marked route sort, or both."""
    graph = torch.fx.Graph()
    scores = graph.placeholder("scores")
    outputs = []
    if with_topk:
        topk = graph.call_function(torch.ops.aten.topk.default, args=(scores, 2))
        topk.meta["custom"] = {"module_fqn": "layers.0.moe.router"}
        ids = graph.call_function(operator.getitem, args=(topk, 1))
        ids.meta["custom"] = {"module_fqn": "layers.0.moe.router"}
        ids.meta["val"] = torch.empty(8, 2, dtype=torch.int64, device="meta")
        outputs.append(ids)
    if with_routes:
        routes = graph.call_function(torch.ops.aten.argsort.default, args=(scores,))
        routes.meta["custom"] = {
            "module_fqn": "layers.0.moe",
            REMAT_REGION: MOE_ROUTING,
        }
        routes.meta["val"] = torch.empty(16, dtype=torch.int64, device="meta")
        outputs.append(routes)
    graph.output(tuple(outputs))
    return torch.fx.GraphModule(nn.Module(), graph)


class TestRoutingRegion(unittest.TestCase):
    def test_selects_the_marked_sort_in_every_layer(self) -> None:
        torch.manual_seed(0)
        traced, _ = _trace_toy(_MoEToy())

        routes = select_route_permutations(traced.gm)

        # The hint is merged into the module FQN, not a replacement for it, so
        # each marked node still knows its layer.
        self.assertEqual(sorted({_get_layer_id(node) for node in routes}), [0, 1])
        for node in routes:
            self.assertEqual(node.meta["custom"][REMAT_REGION], MOE_ROUTING)

    def test_selects_the_last_topk_of_each_router(self) -> None:
        for num_expert_groups, topks_per_router in ((None, 1), (2, 3)):
            with self.subTest(num_expert_groups=num_expert_groups):
                torch.manual_seed(0)
                traced, _ = _trace_toy(_MoEToy(num_expert_groups=num_expert_groups))
                topks = _router_topks(traced.gm)
                self.assertEqual(len(topks), 2 * topks_per_router)

                decisions = select_routing_decisions(traced.gm)

                self.assertEqual(
                    [node.args for node in decisions],
                    [(topks[topks_per_router - 1], 1), (topks[-1], 1)],
                )
                self.assertEqual(
                    [node.meta["custom"]["module_fqn"] for node in decisions],
                    ["layers.0.moe.router", "layers.1.moe.router"],
                )
                for node in decisions:
                    self.assertEqual(node.meta["val"].shape, (_TOKENS, _TOP_K))
                    self.assertEqual(node.meta["val"].dtype, torch.int64)

    def test_unexpected_router_topk_raises(self) -> None:
        graph = torch.fx.Graph()
        scores = graph.placeholder("scores")
        topk = graph.call_function(torch.ops.aten.topk.default, args=(scores, 2))
        topk.meta["custom"] = {"module_fqn": "layers.0.moe.router"}
        indices = graph.call_function(operator.getitem, args=(topk, 1))
        # 1-D ids are not what the selector was written for.
        indices.meta["val"] = torch.empty(16, dtype=torch.int64, device="meta")
        graph.output(indices)
        gm = torch.fx.GraphModule(nn.Module(), graph)

        with self.assertRaisesRegex(ValueError, "does not have one 2-D integer"):
            select_routing_decisions(gm)

    def test_selects_top_k_ids_and_permutation_of_every_moe_layer(self) -> None:
        torch.manual_seed(0)
        traced, _ = _trace_toy(_MoEToy(num_expert_groups=2))

        routing = select_routing(traced.gm)

        self.assertEqual(
            set(routing),
            {
                *select_routing_decisions(traced.gm),
                *select_route_permutations(traced.gm),
            },
        )
        self.assertEqual(sorted({_get_layer_id(node) for node in routing}), [0, 1])

    def test_needs_both_in_the_same_layers(self) -> None:
        # A saved permutation next to recomputed top-k ids could disagree on a
        # tie, so finding only one of the two is an error.
        for with_topk, with_routes in ((False, True), (True, False)):
            with self.subTest(with_topk=with_topk, with_routes=with_routes):
                with self.assertRaisesRegex(ValueError, "every MoE layer needs both"):
                    select_routing(_hand_built_routing_graph(with_topk, with_routes))
        self.assertEqual(len(select_routing(_hand_built_routing_graph(True, True))), 2)


def _selective_config(
    save_regions: tuple[str, ...], last_n_layers: dict[str, int] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        tpu_config=SimpleNamespace(
            graph_trainer_save_regions=save_regions,
            graph_trainer_save_last_n_layers=last_n_layers or {},
        ),
        parallelism=_parallelism("always"),
    )


# Routing and Splash, both in every layer.
_ROUTING_SPLASH = ("moe.routing", "attention.inner_attention")


class TestSelectSaves(unittest.TestCase):
    def test_region_without_candidates_raises(self) -> None:
        traced, _ = _trace_toy(_DenseToy())

        with self.assertRaisesRegex(ValueError, "'moe.routing' matched no"):
            select_saves(traced.gm, [ROUTING_REGION], {})

    def test_last_n_layers_keeps_only_the_last_layers(self) -> None:
        torch.manual_seed(0)
        traced, _ = _trace_toy(_MoEToy(_splash_cores()))
        routing = select_routing(traced.gm)
        splash = select_splash_attention(traced.gm)
        regions = [ROUTING_REGION, SPLASH_ATTENTION_REGION]

        for n_layers, saved_layers in (
            (1, [2]),
            (2, [1, 2]),
            (3, [0, 1, 2]),
            (5, [0, 1, 2]),
        ):
            with self.subTest(n_layers=n_layers):
                chosen = select_saves(
                    traced.gm, regions, {"attention.inner_attention": n_layers}
                )
                # A region without a limit is saved in every layer.
                self.assertEqual(chosen - set(splash), set(routing))
                self.assertEqual(
                    sorted(_get_layer_id(node) for node in chosen & set(splash)),
                    saved_layers,
                )

    def test_limit_needs_values_inside_the_layers(self) -> None:
        traced, _ = _trace_toy(_DenseToy())
        outside = next(
            node
            for node in traced.gm.graph.nodes
            if node.op == "call_function" and _get_layer_id(node) == _NOT_IN_LAYERS
        )
        region = SaveRegion(name="test.outside", select=lambda gm: [outside])

        self.assertEqual(select_saves(traced.gm, [region], {}), {outside})
        with self.assertRaisesRegex(ValueError, "outside the transformer layers"):
            select_saves(traced.gm, [region], {"test.outside": 1})

    def test_config_values_are_checked(self) -> None:
        for save_regions, last_n_layers, expected in (
            (_ROUTING_SPLASH, {}, [ROUTING_REGION, SPLASH_ATTENTION_REGION]),
            (
                ("moe.routing",),
                {"attention.inner_attention": 2},
                [ROUTING_REGION, SPLASH_ATTENTION_REGION],
            ),
            # Listing a region only in last_n_layers is enough.
            ((), {"attention.inner_attention": 2}, [SPLASH_ATTENTION_REGION]),
        ):
            with self.subTest(save_regions=save_regions, last_n_layers=last_n_layers):
                self.assertEqual(
                    resolve_save_regions(save_regions, last_n_layers), expected
                )
        for save_regions, last_n_layers, message in (
            ((), {}, "needs a region"),
            (("moe.nope",), {}, "Unknown save regions"),
            ((), {"moe.nope": 1}, "Unknown save regions"),
            (("moe.routing",) * 2, {}, "Duplicate"),
            (
                ("attention.inner_attention",),
                {"attention.inner_attention": 2},
                "in both",
            ),
            ((), {"attention.inner_attention": 0}, "positive integer"),
            ((), {"attention.inner_attention": True}, "positive integer"),
            ((), {"attention.inner_attention": 2.0}, "positive integer"),
        ):
            with self.subTest(save_regions=save_regions, last_n_layers=last_n_layers):
                with self.assertRaisesRegex(ValueError, message):
                    resolve_save_regions(save_regions, last_n_layers)


class TestTpuSelectivePolicy(unittest.TestCase):
    def _run(
        self,
        model: nn.Module,
        save_regions: tuple[str, ...],
        last_n_layers: dict[str, int] | None = None,
    ) -> set[torch.fx.Node]:
        """Apply tpu_selective, remat and the TPU remat schedule to ``model``.

        Checks that the chosen values are saved rather than recomputed, that
        every layer still gets its entry barrier, and that the transformed
        graph computes exactly what the traced graph did. Returns the chosen
        nodes.
        """
        last_n_layers = last_n_layers or {}
        n_layers = len(model.layers)
        traced, x = _trace_toy(model)
        expected = [
            output.detach().clone()
            for output in run_traced(traced, module=model)(x)
        ]
        remove_parameter_gradient_markers_pass(traced.gm, traced.example_inputs)
        chosen = select_saves(
            traced.gm,
            resolve_save_regions(save_regions, last_n_layers),
            last_n_layers,
        )

        MEMORY_POLICY_REGISTRY["tpu_selective"](
            traced.gm, config=_selective_config(save_regions, last_n_layers)
        )
        for node in chosen:
            self.assertIs(node.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        selective_activation_remat_pass(traced.gm)
        apply_tpu_remat_schedule(
            traced.gm,
            traced.example_inputs,
            num_static_inputs=traced.num_static_inputs,
        )

        names = {node.name for node in traced.gm.graph.nodes}
        self.assertTrue(any(name.endswith("_recomputed") for name in names))
        self.assertFalse({f"{node.name}_recomputed" for node in chosen} & names)
        barriers = [
            node
            for node in traced.gm.graph.nodes
            if node.op == "call_function" and node.target is _optimization_barrier
        ]
        self.assertEqual(len(barriers), n_layers)
        actual = run_traced(traced, module=model)(x)
        for actual_output, expected_output in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_output, expected_output, rtol=0, atol=0
            )
        return chosen

    def test_saves_routing_in_every_layer(self) -> None:
        torch.manual_seed(0)
        chosen = self._run(_MoEToy(), ("moe.routing",))

        routes = {
            node
            for node in chosen
            if node.meta["custom"].get(REMAT_REGION) == MOE_ROUTING
        }
        self.assertEqual(sorted({_get_layer_id(node) for node in routes}), [0, 1])
        self.assertEqual(
            sorted(node.meta["custom"]["module_fqn"] for node in chosen - routes),
            ["layers.0.moe.router", "layers.1.moe.router"],
        )

    def test_saves_routing_and_every_splash_by_default(self) -> None:
        torch.manual_seed(0)
        chosen = self._run(_MoEToy(_splash_cores()), _ROUTING_SPLASH)

        splash = [node for node in chosen if _is_splash_node(node)]
        self.assertEqual(sorted(_get_layer_id(node) for node in splash), [0, 1, 2])
        # Plus the routing of every layer.
        routes = [
            node
            for node in chosen
            if node.meta["custom"].get(REMAT_REGION) == MOE_ROUTING
        ]
        self.assertEqual(sorted({_get_layer_id(node) for node in routes}), [0, 1, 2])

    def test_last_n_layers_saves_only_the_last_splash(self) -> None:
        # Recomputing layers 0-1 while layer 2's backward reads its saved
        # Splash output directly is the case the remat entry barrier must
        # handle.
        torch.manual_seed(0)
        chosen = self._run(
            _MoEToy(_splash_cores()),
            ("moe.routing",),
            last_n_layers={"attention.inner_attention": 1},
        )

        splash = [node for node in chosen if _is_splash_node(node)]
        self.assertEqual([_get_layer_id(node) for node in splash], [2])

    def test_saved_routing_skips_the_router_topk_recompute(self) -> None:
        # Backward needs each router's expert ids to scatter the gate-score
        # gradients. Unless they are saved, it re-runs every top-k behind
        # them: three per router with group-limited routing, three layers.
        for save_regions, last_n_layers, recomputed_topks in (
            (("attention.inner_attention",), {}, 9),
            (_ROUTING_SPLASH, {}, 0),
            # Routing saved in the last layer only: layers 0 and 1 re-run theirs.
            (("attention.inner_attention",), {"moe.routing": 1}, 6),
        ):
            with self.subTest(save_regions=save_regions, last_n_layers=last_n_layers):
                torch.manual_seed(0)
                chosen = self._run(
                    _MoEToy(_splash_cores(), num_expert_groups=2),
                    save_regions,
                    last_n_layers,
                )

                # The chosen nodes are in the graph that _run transformed.
                graph = next(iter(chosen)).graph
                recomputed = [
                    node
                    for node in graph.nodes
                    if node.target is torch.ops.aten.topk.default
                    and node.name.endswith("_recomputed")
                ]
                self.assertEqual(len(recomputed), recomputed_topks)


def _is_splash_node(node: torch.fx.Node) -> bool:
    return isinstance(node.target, torch._ops.OpOverload) and (
        is_splash_forward_op_name(node.target.name())
    )


class TestSplashAttentionRegion(unittest.TestCase):
    def test_matches_stock_and_tokamax_forward_ops_only(self) -> None:
        self.assertEqual(
            torch.ops.pallas.splash_attn_fwd_0123456789abcdef.default.name(),
            "pallas::splash_attn_fwd_0123456789abcdef",
        )
        for name, expected in (
            ("pallas::splash_attn_fwd_0123456789abcdef", True),
            ("pallas::tokamax_splash_attn_fwd", True),
            ("pallas::splash_attn_bwd_0123456789abcdef", False),
            ("pallas::tokamax_splash_attn_bwd", False),
            ("pallas::splash_attn_fwd_0123", False),
            ("other::tokamax_splash_attn_fwd", False),
        ):
            with self.subTest(name=name):
                self.assertEqual(is_splash_forward_op_name(name), expected)

    def test_selects_both_layouts_once_per_layer(self) -> None:
        torch.manual_seed(0)
        traced, _ = _trace_toy(_MoEToy(_splash_cores()))

        splash = select_splash_attention(traced.gm)

        self.assertEqual([_get_layer_id(node) for node in splash], [0, 1, 2])
        self.assertEqual(
            [node.target.name() for node in splash],
            [
                "pallas::splash_attn_fwd_0123456789abcdef",
                "pallas::splash_attn_fwd_fedcba9876543210",
                "pallas::splash_attn_fwd_0123456789abcdef",
            ],
        )
        # As on real Qwen3 traces, the op carries the attention module's FQN.
        self.assertEqual(
            [node.meta["custom"]["module_fqn"] for node in splash],
            [f"layers.{layer}.attention" for layer in range(3)],
        )

    def test_unexpected_splash_outputs_raise(self) -> None:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        splash = graph.call_function(
            torch.ops.pallas.splash_attn_fwd_0123456789abcdef.default,
            args=(x, x, x),
        )
        # An FP32 output is not what the selector was written for.
        splash.meta["val"] = (
            torch.empty(1, 2, 8, 4, dtype=torch.float32, device="meta"),
            torch.empty(1, 2, 8, dtype=torch.float32, device="meta"),
        )
        graph.output(splash)
        gm = torch.fx.GraphModule(nn.Module(), graph)

        with self.assertRaisesRegex(ValueError, "does not return"):
            select_splash_attention(gm)


if __name__ == "__main__":
    absltest.main()
