"""In-place writes to graph inputs must survive as outputs plus a copy back."""

import unittest

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.experiments.tpu.graph_trainer.input_mutations import (
    CopyBackInputMutations,
    functionalize_input_mutations,
)


def _step(counts, bias, x):
    y = x + bias
    counts.add_(y.sum(0))
    counts.add_(1.0)
    return [y * counts.sum()]


class TestInputMutations(unittest.TestCase):
    def test_writes_become_outputs_and_are_copied_back(self):
        counts, bias, x = torch.zeros(4), torch.randn(4), torch.randn(3, 4)
        want_counts = counts.clone()
        (want,) = _step(want_counts, bias, x)

        gm = make_fx(_step)(torch.zeros(4), bias, x)
        positions = functionalize_input_mutations(gm)
        self.assertEqual(positions, [0])
        self.assertFalse(
            [n for n in gm.graph.nodes if str(n.target).endswith("_.Tensor")]
        )
        # Run on fresh inputs: the graph itself no longer writes them.
        before = counts.clone()
        outs = gm(counts, bias, x)
        torch.testing.assert_close(counts, before)
        torch.testing.assert_close(outs[0], want)

        (got,) = CopyBackInputMutations(gm, positions)(counts, bias, x)
        torch.testing.assert_close(got, want)
        torch.testing.assert_close(counts, want_counts)

    def test_write_through_a_view_is_refused(self):
        def step(counts, x):
            counts[0].add_(x.sum())
            return [x]

        gm = make_fx(step)(torch.zeros(2, 4), torch.randn(4))
        with self.assertRaisesRegex(NotImplementedError, "through a view"):
            functionalize_input_mutations(gm)


if __name__ == "__main__":
    unittest.main()
