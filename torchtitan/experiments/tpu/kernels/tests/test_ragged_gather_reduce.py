# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the TensorCore ragged gather-reduce kernel."""

from absl.testing import absltest
from absl.testing import parameterized
import torch

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import ragged_gather_reduce as rgr

_TOP_K = 4


def _routing(generator, num_tokens, rows, valid):
  """Token id per row such that no token receives more than `_TOP_K` rows."""
  token_ids = torch.arange(num_tokens).repeat_interleave(_TOP_K)
  token_ids = token_ids[torch.randperm(token_ids.numel(), generator=generator)]
  return token_ids[:rows].to(torch.int32), torch.tensor([valid])


def _expected_reduce(x, scale, token_ids, valid, num_tokens):
  if scale is not None:
    x = (x.float() * scale.reshape(-1, 1)).to(x.dtype)
  out = torch.zeros(num_tokens, x.shape[1])
  for i in range(valid):
    out[token_ids[i]] += x[i].float()
  return out.to(x.dtype)


class RaggedGatherReduceTest(
    base_device_test.BaseAcceleratorDeviceTest, parameterized.TestCase
):

  @parameterized.product(
      shape=[
          (256, 1024, 901),
          (256, 1024, 0),
          (256, 1024, 1024),
          (64, 256, 1),
          # REDUCE_BLOCK_SIZE is 896: several full blocks plus a remainder.
          (2048, 5000, 4321),
      ],
      with_scale=[True, False],
  )
  def test_matches_reference(self, shape, with_scale):
    num_tokens, rows, valid = shape
    g = torch.Generator().manual_seed(1)
    x = torch.randn(rows, 256, generator=g).to(torch.bfloat16)
    scale = torch.rand(rows, generator=g) if with_scale else None
    token_ids, valid_1 = _routing(g, num_tokens, rows, valid)
    expected = _expected_reduce(x, scale, token_ids, valid, num_tokens)
    # Rows past the valid count may be unwritten expert output; the kernel
    # must not read them.
    x[valid:] = float("nan")
    if scale is not None:
      scale[valid:] = float("nan")

    device = self.accelerator_device
    ids_d, valid_d = token_ids.to(device), valid_1.to(device)
    meta = rgr.ragged_gather_reduce_metadata(
        ids_d, valid_d, top_k=_TOP_K, num_out_tokens=num_tokens
    )
    out = rgr.ragged_gather_reduce(
        x.to(device),
        scale.to(device) if with_scale else None,
        valid_d,
        meta,
        num_out_tokens=num_tokens,
        top_k=_TOP_K,
    )

    self.assertEqual(tuple(out.shape), (num_tokens, 2, 128))
    torch.testing.assert_close(
        out.cpu().reshape(num_tokens, 256), expected, rtol=1e-2, atol=1e-2
    )

  def test_rejects_hidden_not_multiple_of_128(self):
    meta = rgr.ragged_gather_reduce_metadata(
        torch.zeros(4, dtype=torch.int32),
        torch.tensor([4]),
        top_k=1,
        num_out_tokens=4,
    )
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      rgr.ragged_gather_reduce(
          torch.zeros(4, 100), None, torch.tensor([4]), meta,
          num_out_tokens=4, top_k=1,
      )


if __name__ == "__main__":
  absltest.main()
