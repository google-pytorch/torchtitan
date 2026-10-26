"""Unit tests for fsdp_shard_dims's funcol monkey-patch (the eager SimpleFSDP lanes)."""

import unittest
from unittest import mock
from absl.testing import absltest
import torch
import torch.distributed._functional_collectives as funcol
from torchtitan.experiments.tpu.graph_trainer import fsdp_shard_dims


class TestFuncolMovedim(unittest.TestCase):

    def test_patch_installed(self):
        self.assertIs(funcol.all_gather_single, fsdp_shard_dims._movedim_all_gather_tensor)
        self.assertIs(funcol.all_gather_tensor, fsdp_shard_dims._movedim_all_gather_tensor)
        self.assertIs(
            funcol.reduce_scatter_single, fsdp_shard_dims._movedim_reduce_scatter_tensor
        )
        self.assertIs(
            funcol.reduce_scatter_tensor, fsdp_shard_dims._movedim_reduce_scatter_tensor
        )

    def test_all_gather_transposes_onto_dim_zero(self):
        # 4 ranks, gather along dim 1 of a (2, 3, 4) tensor
        t = torch.arange(24).reshape(2, 3, 4).float()
        with mock.patch.object(fsdp_shard_dims, "_orig_all_gather_single") as mock_ag:
            mock_ag.side_effect = lambda x, gather_dim, group, tag="": torch.cat([x] * 4, dim=0)
            out = funcol.all_gather_single(t, gather_dim=1, group="mesh")
            self.assertEqual(mock_ag.call_args.kwargs["gather_dim"], 0)
            self.assertEqual(out.shape, (2, 12, 4))
            self.assertTrue(torch.equal(out, torch.cat([t] * 4, dim=1)))

    def test_reduce_scatter_transposes_onto_dim_zero(self):
        # 4 ranks, scatter along dim 1 of a (2, 12, 4) tensor
        t = torch.arange(96).reshape(2, 12, 4).float()
        with mock.patch.object(fsdp_shard_dims, "_orig_reduce_scatter_single") as mock_rs:
            mock_rs.side_effect = lambda x, reduceOp, scatter_dim, group, tag="": x[: x.shape[0] // 4]
            out = funcol.reduce_scatter_single(t, reduceOp="sum", scatter_dim=1, group="mesh")
            self.assertEqual(mock_rs.call_args.kwargs["scatter_dim"], 0)
            self.assertEqual(out.shape, (2, 3, 4))

    def test_dim_zero_passes_through_without_transpose(self):
        t = torch.arange(12).reshape(3, 4).float()
        with mock.patch.object(fsdp_shard_dims, "_orig_all_gather_single") as mock_ag:
            mock_ag.return_value = t
            out = funcol.all_gather_single(t, gather_dim=0, group="mesh")
            self.assertIs(mock_ag.call_args.args[0], t)
            self.assertIs(out, t)


if __name__ == "__main__":
    absltest.main()
