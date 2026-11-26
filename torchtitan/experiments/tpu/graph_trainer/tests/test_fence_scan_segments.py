# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where the scan-segment fence goes.

Two segments, in the order the joint graph has them:

    fwd_a -> fwd_b -> bwd_b -> bwd_a

Fences go before fwd_b and before bwd_a; fwd_b -> bwd_b is left alone.
"""

import operator

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu.graph_trainer.passes.fence_scan_segments import (
    fence_scan_segments,
)

_SCAN = torch.ops.higher_order.scan


def _fence(values):
    return values


class FenceScanSegmentsTest(absltest.TestCase):

    def setUp(self):
        super().setUp()
        self.graph = torch.fx.Graph()
        self.x = self._placeholder("x", 4, 4)
        self.w = self._placeholder("w", 6, 4, 4)

    def _placeholder(self, name, *shape):
        node = self.graph.placeholder(name)
        node.meta["val"] = torch.empty(shape, device="meta")
        return node

    def _scan(self, carry, *, bwd, saved=None):
        """A scan over ``w``; returns (scan, carry out, saved stack out)."""
        xs = [self.w] + ([saved] if saved is not None else [])
        node = self.graph.call_function(_SCAN, (None, [carry], xs, ()))
        node.meta["autograd_backward"] = bwd
        outs = []
        for i, shape in enumerate([(4, 4), (6, 4, 4)]):
            out = self.graph.call_function(operator.getitem, (node, i))
            out.meta["val"] = torch.empty(shape, device="meta")
            out.meta["autograd_backward"] = bwd
            outs.append(out)
        return node, *outs

    def _segments(self):
        fwd_a, h, sa = self._scan(self.x, bwd=False)
        fwd_b, h, sb = self._scan(h, bwd=False)
        bwd_b, g, gw_b = self._scan(h, bwd=True, saved=sb)
        bwd_a, g, gw_a = self._scan(g, bwd=True, saved=sa)
        self.graph.output((g, gw_a, gw_b))
        gm = torch.fx.GraphModule(torch.nn.Module(), self.graph)
        gm = fence_scan_segments(gm, barrier=_fence)
        fences = list(gm.graph.find_nodes(op="call_function", target=_fence))
        return (fwd_a, fwd_b, bwd_b, bwd_a), fences

    def test_fences_each_same_direction_boundary(self):
        (_, fwd_b, bwd_b, bwd_a), fences = self._segments()
        self.assertLen(fences, 2)
        order = {n: i for i, n in enumerate(self.graph.nodes)}
        self.assertLess(order[fences[0]], order[fwd_b])
        self.assertLess(order[bwd_b], order[fences[1]])
        self.assertLess(order[fences[1]], order[bwd_a])

    def test_forward_fence_skips_the_saved_stack(self):
        (fwd_a, fwd_b, _, _), fences = self._segments()
        # fwd_a's carry and fwd_b's weights; the stack saved for backward,
        # which may be on host, stays out.
        values = fences[0].args[0]
        self.assertEqual(values[0].args, (fwd_a, 0))
        self.assertEqual(values[1:], [self.w])
        for arg in (fwd_b.args[1][0], fwd_b.args[2][0]):
            self.assertIs(arg.args[0], fences[0])

    def test_backward_fence_holds_every_gradient(self):
        (_, _, bwd_b, bwd_a), fences = self._segments()
        values = fences[1].args[0]
        # bwd_b's carry and weight gradient, then bwd_a's weights. The saved
        # stack bwd_a reads comes from a scan and stays out.
        self.assertEqual([v.args[0] for v in values[:2]], [bwd_b, bwd_b])
        self.assertEqual(values[2:], [self.w])
        self.assertIs(bwd_a.args[1][0].args[0], fences[1])
        self.assertIsNot(bwd_a.args[2][1].args[0], fences[1])

    def test_one_scan_per_direction_is_untouched(self):
        _, h, s = self._scan(self.x, bwd=False)
        _, g, _ = self._scan(h, bwd=True, saved=s)
        self.graph.output(g)
        gm = fence_scan_segments(
            torch.fx.GraphModule(torch.nn.Module(), self.graph), barrier=_fence
        )
        self.assertFalse(gm.graph.find_nodes(op="call_function", target=_fence))

    def test_no_pair_raises_when_last_n_splits_the_layers(self):
        _, h, s = self._scan(self.x, bwd=False)
        _, g, _ = self._scan(h, bwd=True, saved=s)
        self.graph.output(g)
        gm = torch.fx.GraphModule(torch.nn.Module(), self.graph)
        with self.assertRaisesRegex(RuntimeError, "no pair of scans"):
            fence_scan_segments(gm, barrier=_fence, expects_segments=True)

    def test_no_scan_boundary_is_silent_without_scans(self):
        # The optimizer and grad-norm graphs have no scan.
        self.graph.output(self.x)
        gm = torch.fx.GraphModule(torch.nn.Module(), self.graph)
        fence_scan_segments(gm, barrier=_fence, expects_segments=True)


if __name__ == "__main__":
    absltest.main()
