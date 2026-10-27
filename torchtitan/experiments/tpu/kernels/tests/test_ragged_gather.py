# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the TensorCore ragged gather kernel."""

from absl.testing import absltest
from absl.testing import parameterized
import torch

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import ragged_gather as rg

_TOP_K = 4


def _routing(generator, num_tokens, rows, valid):
  """Token id per row such that no token receives more than `_TOP_K` rows."""
  token_ids = torch.arange(num_tokens).repeat_interleave(_TOP_K)
  token_ids = token_ids[torch.randperm(token_ids.numel(), generator=generator)]
  return token_ids[:rows].to(torch.int32), torch.tensor([valid])


class RaggedGatherTest(
    base_device_test.BaseAcceleratorDeviceTest, parameterized.TestCase
):

  def _to(self, *tensors):
    return [t.to(self.accelerator_device) for t in tensors]

  @parameterized.named_parameters(
      ("partial", 512, 1024, 777),
      ("empty", 512, 1024, 0),
      ("full", 512, 1024, 1024),
      ("single_row", 64, 256, 1),
      # GATHER_BLOCK_SIZE is 1536: multiple full blocks in both 2D and fallback paths.
      ("multi_block_aligned", 2048, 4608, 4321),
      ("multi_block_remainder", 2048, 5000, 4321),
  )
  def test_gather_zeroes_rows_past_valid_count(self, num_tokens, rows, valid):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(num_tokens, 256, generator=g).to(torch.bfloat16)
    token_ids, valid_1 = _routing(g, num_tokens, rows, valid)
    expected = x[token_ids.long()]
    expected[valid:] = 0

    x_d, ids_d, valid_d = self._to(x, token_ids, valid_1)
    out = rg.ragged_gather(rg.to_token_3d(x_d), ids_d, valid_d)

    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

  def test_inverse_permute_scatter(self):
    g = torch.Generator().manual_seed(2)
    n, r, valid = 2048, 640, 500
    perm = torch.randperm(n, generator=g).to(torch.int32)
    values = torch.randn(r, generator=g)
    expected = torch.zeros(n)
    expected[perm[:valid].long()] = values[:valid]

    out = rg.inverse_permute_scatter(*self._to(perm, values, torch.tensor([valid])))

    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

  def test_rejects_hidden_not_multiple_of_128(self):
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      rg.to_token_3d(torch.zeros(4, 100))


if __name__ == "__main__":
  absltest.main()
