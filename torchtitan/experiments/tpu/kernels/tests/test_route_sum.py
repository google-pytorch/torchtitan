# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the fixed-K route sum."""

import torch
from absl.testing import absltest

from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels.route_sum import route_sum


def _token_major_sum(src, rows, top_k):
  return (
      torch.index_select(src, 0, rows)
      .reshape(-1, top_k, src.shape[1])
      .float()
      .sum(dim=1)
      .to(src.dtype)
  )


class RouteSumTest(base_device_test.BaseAcceleratorDeviceTest):

  def test_matches_token_major_sum(self):
    # On TPU, wide exponents make the FP32 sums depend on their association,
    # so the bitwise comparison checks the reduction order, eager and compiled.
    on_tpu = self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU
    modes = [(route_sum, _token_major_sum)]
    if on_tpu:
      modes.append(
          tuple(
              torch.compile(f, backend="tpu", dynamic=False)
              for f in modes[0]
          )
      )
    generator = torch.Generator().manual_seed(1357)
    tokens, dim = 256, 512
    for top_k in (3, 4, 8):
      src = torch.randn(tokens * top_k, dim, generator=generator)
      if on_tpu:
        src *= torch.exp2(
            torch.randint(-20, 21, src.shape, generator=generator)
        )
      src = src.bfloat16().to(self.accelerator_device)
      rows = torch.randperm(tokens * top_k, generator=generator)
      rows = rows.to(self.accelerator_device)
      for actual_fn, expected_fn in modes:
        actual = actual_fn(src, rows, top_k).cpu()
        expected = expected_fn(src, rows, top_k).cpu()
        tol = dict(rtol=0, atol=0) if on_tpu else {}
        torch.testing.assert_close(actual, expected, **tol)


if __name__ == "__main__":
  absltest.main()
