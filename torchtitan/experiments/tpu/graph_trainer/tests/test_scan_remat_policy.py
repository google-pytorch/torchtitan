"""The scan body saves its expert counts, and nothing else changes.

The counts are the one thing the body's full remat no longer redoes in
backward. Everything here is checked against the plain budget-0.0 partition
it replaces: the gradients bit for bit, and the backward graph for what it
stopped (and did not stop) recomputing.
"""

import unittest
from unittest import mock

from absl.testing import absltest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch._functorch import config as functorch_config, partitioners
from torch._higher_order_ops.scan import scan
from torch.fx.experimental.proxy_tensor import make_fx

# Imported for the same reason as in `test_layer_stack`: production scans run
# with it installed, so these do too.
from torchtitan.experiments.tpu import scan_backward_patch  # noqa: F401  pylint: disable=unused-import,g-bad-import-order
from torchtitan.experiments.tpu.graph_trainer import scan_remat_policy
from torchtitan.experiments.tpu.graph_trainer.scan_remat_policy import (
    count_collectives,
    scan_body_remat_window,
)
from torchtitan.models.common.moe import _batch_wise_load_balance_loss

SEED = 1234
N_LAYERS = 3
TOKENS = 16
DIM = 8
EXPERTS = 4
TOP_K = 2


def _init_single_rank_gloo(port: int):
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://localhost:{port}",
            world_size=1,
            rank=0,
        )


def _counts(scores: torch.Tensor) -> torch.Tensor:
    """The batch-wise aux loss's global counts, as `moe.py` computes them."""
    ids = scores.topk(TOP_K, dim=-1).indices
    counts = torch.histc(
        ids.reshape(-1).float(), bins=EXPERTS, min=0, max=EXPERTS
    ).to(scores.dtype)
    return funcol.all_reduce(counts, "sum", dist.group.WORLD)


def _aux_term(scores: torch.Tensor) -> torch.Tensor:
    """An aux-loss-like term, so the counts reach the gradient."""
    counts = _counts(scores)
    return (counts * scores.sum(dim=0)).sum() / counts.sum()


def _routing_counts(scores: torch.Tensor) -> torch.Tensor:
    """Expert parallelism's routing counts and their exchange.

    `_tokens_per_expert` and `_token_count_exchange`: a `histc` too, but
    exchanged with an `all_to_all_single`, not all-reduced.
    """
    ids = scores.topk(TOP_K, dim=-1).indices
    counts = torch.histc(
        ids.reshape(-1).float(), bins=EXPERTS, min=0, max=EXPERTS
    ).to(torch.int64)
    return funcol.all_to_all_single(
        counts.view(1, -1), None, None, dist.group.WORLD
    )


def _routing_term(scores: torch.Tensor) -> torch.Tensor:
    """Keeps the routing count exchange live without a gradient through it."""
    return _routing_counts(scores).sum().to(scores.dtype) * 0.0


class _Mesh:
    """Just what `_all_reduce_expert_counts` asks of a `DeviceMesh`."""

    def __init__(self, group):
        self._group = group

    def get_group(self):
        return self._group

    def size(self):
        return dist.get_world_size(self._group)


def _real_aux_term(dp_group, cp_group):
    """The real batch-wise aux loss, its in-place all-reduces over two groups."""

    def term(scores: torch.Tensor) -> torch.Tensor:
        # As the router builds it: a one-hot [T, E] map of each token's experts.
        ids = scores.topk(TOP_K, dim=-1).indices
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(
            -1, ids, True
        )
        return _batch_wise_load_balance_loss(
            scores.unsqueeze(0),
            routing_map.unsqueeze(0),
            TOP_K,
            0.01,
            dp_mesh=_Mesh(dp_group),
            cp_mesh=_Mesh(cp_group),
        )

    return term


def _body(h: torch.Tensor, w: torch.Tensor, *, extra):
    """A router, plus `extra(scores)` if given."""
    scores = torch.softmax(h @ w, dim=-1)
    out = torch.tanh(h + scores @ w.T)
    if extra is not None:
        out = out + extra(scores)
    return out, out.new_zeros(())


def _run(window, *, extra=_aux_term):
    """Forward and backward one scan; returns grads and the body's partitions."""
    torch._dynamo.reset()
    torch.manual_seed(SEED)
    h = torch.randn(TOKENS, DIM, requires_grad=True)
    w = torch.randn(N_LAYERS, DIM, EXPERTS, requires_grad=True)

    captured = []
    partition = partitioners.min_cut_rematerialization_partition

    def capture(joint, inputs, **kwargs):
        forward, backward = partition(joint, inputs, **kwargs)
        captured.append((functorch_config.activation_memory_budget, backward))
        return forward, backward

    def body(carry, x):
        return _body(carry, x, extra=extra)

    with mock.patch.object(
        partitioners, "min_cut_rematerialization_partition", capture
    ):
        with window():
            carry, _ = scan(body, h, w)
    carry.square().sum().backward()
    return (carry.detach(), h.grad, w.grad), captured


def _legacy_window():
    return functorch_config.patch(activation_memory_budget=0.0)


def _has_count_source(graph_module) -> bool:
    return any(
        scan_remat_policy._is_count_source(n) for n in graph_module.graph.nodes
    )


def _ops(graph_module) -> list[str]:
    return [str(n.target) for n in graph_module.graph.nodes if n.op == "call_function"]


class TestCountCollectives(unittest.TestCase):
    """Only an all-reduce whose operand *is* a count is picked up."""

    def setUp(self):
        _init_single_rank_gloo(12391)

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_finds_the_count_all_reduce_and_not_the_others(self):
        def f(scores, weight):
            counts = funcol.wait_tensor(_counts(scores))
            # Collectives on values that are not counts: a weight, and a count
            # once it has been mixed with something else.
            gathered = funcol.wait_tensor(
                funcol.all_reduce(weight, "sum", dist.group.WORLD)
            )
            mixed = funcol.wait_tensor(
                funcol.all_reduce(counts * weight.sum(), "sum", dist.group.WORLD)
            )
            return counts, gathered, mixed

        graph = make_fx(f)(
            torch.randn(TOKENS, EXPERTS).softmax(-1), torch.randn(EXPERTS)
        ).graph
        found = count_collectives(graph)

        self.assertTrue(found, "the count all-reduce was not found")
        # The count's own all_reduce and its wait(s) -- tracing may emit more
        # than one wait for it -- and nothing for the weight or the mix.
        self.assertEqual(
            {str(n.target) for n in found},
            {"_c10d_functional.all_reduce.default", "_c10d_functional.wait_tensor.default"},
        )
        all_reduces = [
            n for n in found if n.target is torch.ops._c10d_functional.all_reduce.default
        ]
        self.assertEqual(len(all_reduces), 1, found)
        (all_reduce,) = all_reduces
        self.assertIn(
            all_reduce.args[0].target,
            (torch.ops.aten._to_copy.default, torch.ops.aten.histc.default),
        )

    def test_ignores_the_expert_parallel_routing_count_exchange(self):
        """Also born at a `histc`, but all-to-all'd: not the aux loss's counts."""

        def f(scores):
            return funcol.wait_tensor(_routing_counts(scores))

        graph = make_fx(f)(torch.randn(TOKENS, EXPERTS).softmax(-1)).graph
        self.assertIn(
            "_c10d_functional.all_to_all_single.default",
            [str(n.target) for n in graph.nodes],
            "the exchange was not traced -- test is vacuous",
        )
        self.assertEqual(count_collectives(graph), [])

    def test_finds_the_ep_count_all_gather(self):
        """TPU EP exchanges `histc` counts via `all_gather_into_tensor`."""

        def f(scores):
            ids = scores.topk(TOP_K, dim=-1).indices
            counts = torch.histc(
                ids.reshape(-1).float(), bins=EXPERTS, min=0, max=EXPERTS
            ).to(torch.int32)
            return funcol.all_gather_tensor(
                counts, gather_dim=0, group=dist.group.WORLD
            )

        graph = make_fx(f)(torch.randn(TOKENS, EXPERTS).softmax(-1)).graph
        found = count_collectives(graph)
        self.assertEqual(
            {str(n.target) for n in found},
            {
                "_c10d_functional.all_gather_into_tensor.default",
                "_c10d_functional.wait_tensor.default",
            },
        )


class TestScanBodyRematWindow(unittest.TestCase):

    def setUp(self):
        _init_single_rank_gloo(12392)

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_gradients_are_bitwise_identical(self):
        """The whole point: the counts move, the math does not."""
        want, _ = _run(_legacy_window)
        got, _ = _run(scan_body_remat_window)
        for name, g, w in zip(("out", "h.grad", "w.grad"), got, want):
            torch.testing.assert_close(g, w, rtol=0, atol=0, msg=name)

    def test_backward_no_longer_recomputes_the_counts(self):
        _, legacy = _run(_legacy_window)
        _, captured = _run(scan_body_remat_window)
        self.assertTrue(captured, "the scan body was never partitioned")

        # Sanity: before, backward redid the histogram and its all-reduce.
        legacy_ops = _ops(legacy[0][1])
        self.assertIn("aten.histc.default", legacy_ops)
        self.assertIn("_c10d_functional.all_reduce.default", legacy_ops)

        budget, backward = captured[0]
        self.assertEqual(budget, 1.0)
        ops = _ops(backward)
        self.assertNotIn("aten.histc.default", ops)
        self.assertNotIn("_c10d_functional.all_reduce.default", ops)
        # ... while the rest of the body is still recomputed.
        self.assertIn("aten.mm.default", ops)
        self.assertIn("aten.tanh.default", ops)

    def test_the_real_aux_loss_over_two_groups(self):
        """`_batch_wise_load_balance_loss` itself, not a stand-in.

        Its counts go through an in-place `dist.all_reduce` per mesh axis --
        here DP and CP, two distinct groups -- which Dynamo and
        functionalization turn into a chain of functional all-reduces with
        copies between them. All of it has to be found and saved.
        """
        extra = _real_aux_term(dist.new_group([0]), dist.new_group([0]))
        want, legacy = _run(_legacy_window, extra=extra)
        got, captured = _run(scan_body_remat_window, extra=extra)

        legacy_ops = _ops(legacy[0][1])
        self.assertEqual(
            legacy_ops.count("_c10d_functional.all_reduce.default"),
            2,
            "expected one count all-reduce per group -- test is vacuous",
        )
        # The real loss counts by summing the routing map, not with `histc`.
        self.assertTrue(
            _has_count_source(legacy[0][1]),
            "expected budget 0.0 to recompute the counts -- test is vacuous",
        )
        budget, backward = captured[0]
        self.assertEqual(budget, 1.0)
        ops = _ops(backward)
        self.assertFalse(_has_count_source(backward))
        self.assertNotIn("_c10d_functional.all_reduce.default", ops)
        for name, g, w in zip(("out", "h.grad", "w.grad"), got, want):
            torch.testing.assert_close(g, w, rtol=0, atol=0, msg=name)

    def test_a_body_without_counts_is_partitioned_as_before(self):
        """No counts, no tags: the old budget-0.0 path, untouched."""
        want, legacy = _run(_legacy_window, extra=None)
        got, captured = _run(scan_body_remat_window, extra=None)
        self.assertEqual(captured[0][0], 0.0)
        self.assertEqual(_ops(captured[0][1]), _ops(legacy[0][1]))
        for name, g, w in zip(("out", "h.grad", "w.grad"), got, want):
            torch.testing.assert_close(g, w, rtol=0, atol=0, msg=name)

    def test_an_expert_parallel_body_without_aux_loss_is_as_before(self):
        """EP's routing count exchange alone does not leave the 0.0 path."""
        want, legacy = _run(_legacy_window, extra=_routing_term)
        got, captured = _run(scan_body_remat_window, extra=_routing_term)
        self.assertEqual(captured[0][0], 0.0)
        self.assertEqual(_ops(captured[0][1]), _ops(legacy[0][1]))
        for name, g, w in zip(("out", "h.grad", "w.grad"), got, want):
            torch.testing.assert_close(g, w, rtol=0, atol=0, msg=name)

    def test_the_window_closes(self):
        default_budget = functorch_config.activation_memory_budget
        partition = partitioners.min_cut_rematerialization_partition
        _run(scan_body_remat_window)
        self.assertEqual(functorch_config.activation_memory_budget, default_budget)
        self.assertIs(partitioners.min_cut_rematerialization_partition, partition)

    def test_backward_starts_with_forward_collective_group(self):
        """Recomputed FSDP parameter gathers stay ahead of EP combine backward."""
        fsdp_group = dist.new_group([0])
        ep_group = dist.new_group([0])

        class _CombineLike(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, gate):
                ctx.save_for_backward(gate)
                return x * gate

            @staticmethod
            def backward(ctx, grad_out):
                (gate,) = ctx.saved_tensors
                # Mirrors `_CombineTokens.backward`: an EP all-gather on
                # `grad_out` that depends only on `tangents_*`, before the
                # first op that consumes a recomputed forward tensor (`gate`).
                gathered_grad = funcol.all_gather_tensor(
                    grad_out, gather_dim=0, group=ep_group
                )
                return gathered_grad * gate, gathered_grad * gate

        def ep_layer(carry, w_shard):
            w_full = funcol.all_gather_tensor(
                w_shard, gather_dim=0, group=fsdp_group
            )
            gate = torch.tanh(carry @ w_full)
            out = carry + _CombineLike.apply(carry, gate)
            return out, out.new_zeros(())

        def run_with(window):
            torch._dynamo.reset()
            torch.manual_seed(SEED)
            h = torch.randn(TOKENS, DIM, requires_grad=True)
            w = torch.randn(N_LAYERS, DIM, DIM, requires_grad=True)
            partitioned = []

            with window():
                current = partitioners.min_cut_rematerialization_partition

                def wrapped(joint, inputs, *args, **kwargs):
                    fw_gm, bw_gm = current(joint, inputs, *args, **kwargs)
                    partitioned.append((fw_gm, bw_gm))
                    return fw_gm, bw_gm

                with mock.patch.object(
                    partitioners, "min_cut_rematerialization_partition", wrapped
                ):
                    carry, _ = scan(ep_layer, h, w)
            carry.square().sum().backward()
            return (carry.detach(), h.grad, w.grad), partitioned[-1]

        want, (legacy_fw, legacy_bw) = run_with(_legacy_window)
        got, (fw_gm, bw_gm) = run_with(scan_body_remat_window)

        def collective_groups(gm):
            return [
                group
                for node in gm.graph.nodes
                if (group := scan_remat_policy._collective_group_name(node))
                is not None
            ]

        self.assertEqual(collective_groups(legacy_fw)[0], fsdp_group.group_name)
        self.assertEqual(collective_groups(legacy_bw)[0], ep_group.group_name)
        self.assertEqual(collective_groups(fw_gm)[0], fsdp_group.group_name)
        self.assertEqual(collective_groups(bw_gm)[0], fsdp_group.group_name)
        for name, g, w in zip(("out", "h.grad", "w.grad"), got, want):
            torch.testing.assert_close(g, w, rtol=0, atol=0, msg=name)


if __name__ == "__main__":
    absltest.main()
