# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the TPU route-score gradient reduction."""

import torch
from absl.testing import absltest

from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.experiments.tpu.kernels.route_score import route_score_grad


class RouteScoreTest(base_device_test.BaseAcceleratorDeviceTest):

  def test_matches_fp32_reference(self):
    generator = torch.Generator().manual_seed(2468)
    grad_output = torch.randn(
        128, 256, dtype=torch.bfloat16, generator=generator
    )
    routed_output = torch.randn(
        128, 256, dtype=torch.bfloat16, generator=generator
    )
    expected = (grad_output.float() * routed_output.float()).sum(dim=1)

    actual = route_score_grad(
        grad_output.to(self.accelerator_device),
        routed_output.to(self.accelerator_device),
    )

    self.assertEqual(actual.dtype, torch.float32)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)

  def test_rejects_unsupported_contracts(self):
    with self.assertRaisesRegex(ValueError, "rank-2"):
      route_score_grad(
          torch.ones(8, dtype=torch.bfloat16),
          torch.ones(8, dtype=torch.bfloat16),
      )
    with self.assertRaisesRegex(ValueError, "same shape"):
      route_score_grad(
          torch.ones(8, 4, dtype=torch.bfloat16),
          torch.ones(7, 4, dtype=torch.bfloat16),
      )
    with self.assertRaisesRegex(ValueError, "both be bfloat16"):
      route_score_grad(torch.ones(8, 4), torch.ones(8, 4))


if __name__ == "__main__":
  absltest.main()
