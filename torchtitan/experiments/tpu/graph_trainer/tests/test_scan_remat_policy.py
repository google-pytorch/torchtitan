"""The scan body saves its tagged regions, and nothing else changes.

Checked against a plain budget-0.0 partition (recompute everything): the
gradients bit for bit, and the backward graph for what it stopped (and did
not stop) recomputing.
"""

import unittest
from unittest import mock

from absl.testing import absltest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch._functorch import config as functorch_config, partitioners
from torch._higher_order_ops import partitioner as hop_partitioner
from torch._higher_order_ops.scan import scan

# Imported for the same reason as in `test_layer_stack`: production scans run
# with it installed, so these do too.
from torchtitan.experiments.tpu import scan_backward_patch  # noqa: F401  pylint: disable=unused-import,g-bad-import-order
from torchtitan.experiments.tpu.graph_trainer import scan_remat_policy
from torchtitan.experiments.tpu.graph_trainer.scan_remat_policy import (
    scan_body_remat_window,
)
from torchtitan.experiments.tpu.remat_regions import (
    annotate_moe_expert_counts,
    annotate_region,
    ATTENTION_INNER_ATTENTION,
    MOE_EXPERT_COUNTS,
)
from torchtitan.models.common import moe
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
    with annotate_region(MOE_EXPERT_COUNTS):
        return funcol.wait_tensor(funcol.all_reduce(counts, "sum", dist.group.WORLD))


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


def _attention_term(scores: torch.Tensor) -> torch.Tensor:
    """Stands in for splash: a region whose output its backward reads."""
    with annotate_region(ATTENTION_INNER_ATTENTION):
        return torch.sigmoid(scores).sum(dim=-1, keepdim=True)


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


def _ops(graph_module) -> list[str]:
    return [str(n.target) for n in graph_module.graph.nodes if n.op == "call_function"]


def _both(*terms):
    return lambda scores: sum(t(scores) for t in terms)


class TestScanBodyRematWindow(unittest.TestCase):

    def setUp(self):
        _init_single_rank_gloo(12392)
        # `LayerStack.install` does this for a scanned model.
        annotate_moe_expert_counts()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _check_same_grads(self, got, want):
        for name, g, w in zip(("out", "h.grad", "w.grad"), got, want):
            torch.testing.assert_close(g, w, rtol=0, atol=0, msg=name)

    def test_saves_the_counts_and_recomputes_the_rest(self):
        want, legacy = _run(_legacy_window)
        got, captured = _run(scan_body_remat_window)
        self.assertTrue(captured, "the scan body was never partitioned")
        self._check_same_grads(got, want)

        # Sanity: before, backward redid the histogram and its all-reduce.
        legacy_ops = _ops(legacy[0][1])
        self.assertIn("aten.histc.default", legacy_ops)
        self.assertIn("_c10d_functional.all_reduce.default", legacy_ops)

        budget, backward = captured[0]
        self.assertEqual(budget, 1.0)
        ops = _ops(backward)
        self.assertNotIn("_c10d_functional.all_reduce.default", ops)
        # ... while the rest of the body is still recomputed.
        self.assertIn("aten.mm.default", ops)
        self.assertIn("aten.tanh.default", ops)

    def test_the_real_aux_loss_over_two_groups(self):
        """`_batch_wise_load_balance_loss` itself, with its in-place all-reduces."""
        extra = _real_aux_term(dist.new_group([0]), dist.new_group([0]))
        want, legacy = _run(_legacy_window, extra=extra)
        got, captured = _run(scan_body_remat_window, extra=extra)
        self.assertEqual(
            _ops(legacy[0][1]).count("_c10d_functional.all_reduce.default"),
            2,
            "expected one count all-reduce per group -- test is vacuous",
        )
        self.assertNotIn(
            "_c10d_functional.all_reduce.default", _ops(captured[0][1])
        )
        self._check_same_grads(got, want)

    def test_the_counts_are_wrapped_once(self):
        wrapped = moe._all_reduce_expert_counts
        annotate_moe_expert_counts()
        self.assertIs(moe._all_reduce_expert_counts, wrapped)
        self.assertFalse(hasattr(wrapped.__wrapped__, "__wrapped__"))

    def test_a_body_without_regions_recomputes_everything(self):
        want, legacy = _run(_legacy_window, extra=None)
        got, captured = _run(scan_body_remat_window, extra=None)
        self.assertEqual(_ops(captured[0][1]), _ops(legacy[0][1]))
        self._check_same_grads(got, want)

    def test_the_expert_parallel_count_exchange_is_recomputed(self):
        """Only the aux loss's counts are saved, not EP's `all_to_all`."""
        def routing_scaled(scores):
            # Backward reads the exchanged counts, so it must get them somehow.
            return (_routing_counts(scores).sum().to(scores.dtype) * scores).sum()

        extra = _both(_aux_term, routing_scaled)
        want, _ = _run(_legacy_window, extra=extra)
        got, captured = _run(scan_body_remat_window, extra=extra)
        ops = _ops(captured[0][1])
        self.assertIn("_c10d_functional.all_to_all_single.default", ops)
        self.assertNotIn("_c10d_functional.all_reduce.default", ops)
        self._check_same_grads(got, want)

    def test_a_selected_region_is_saved(self):
        want, _ = _run(_legacy_window, extra=_attention_term)
        _, unselected = _run(scan_body_remat_window, extra=_attention_term)
        got, selected = _run(
            lambda: scan_body_remat_window({ATTENTION_INNER_ATTENTION}),
            extra=_attention_term,
        )
        self.assertIn("aten.sigmoid.default", _ops(unselected[0][1]))
        self.assertNotIn("aten.sigmoid.default", _ops(selected[0][1]))
        self._check_same_grads(got, want)

    def test_raises_when_the_region_meta_is_lost(self):
        with mock.patch.object(scan_remat_policy, "_keeping_meta", lambda fn: fn):
            with self.assertRaisesRegex(Exception, "no remat_region meta"):
                _run(scan_body_remat_window)

    def test_refuses_an_unknown_save_region(self):
        with self.assertRaises(ValueError):
            with scan_body_remat_window({"moe.routing"}):
                pass

    def test_the_window_closes(self):
        before = (
            functorch_config.activation_memory_budget,
            partitioners.min_cut_rematerialization_partition,
            hop_partitioner.create_bw_fn,
            torch.func.functionalize,
        )
        _run(scan_body_remat_window)
        after = (
            functorch_config.activation_memory_budget,
            partitioners.min_cut_rematerialization_partition,
            hop_partitioner.create_bw_fn,
            torch.func.functionalize,
        )
        self.assertEqual(after, before)


if __name__ == "__main__":
    absltest.main()
