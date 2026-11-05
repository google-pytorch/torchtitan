# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the fused TPU QK-norm + rotate-half RoPE kernel."""

import torch
import torch.nn.functional as F
from absl.testing import absltest
from absl.testing import parameterized

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.bld_layout import cossin_rotate
from torchtitan.experiments.tpu.kernels.qk_norm_rope import qk_norm_rope

_EPS = 1e-6


def _reference(x, weight, cos, sin):
  y = F.rms_norm(x, (x.shape[-1],), weight, _EPS)
  return cossin_rotate(y, cos.unsqueeze(2), sin.unsqueeze(2))


class QKNormRopeTest(
    base_device_test.BaseAcceleratorDeviceTest, parameterized.TestCase
):

  @parameterized.named_parameters(("queries", 16), ("keys", 4))
  def test_matches_unfused_graph(self, heads):
    b, l, d = 2, 2048, 128
    g = torch.Generator().manual_seed(heads)
    x = (torch.randn(b, l, heads, d, generator=g) * 3).bfloat16()
    weight = (1 + 0.1 * torch.randn(d, generator=g)).bfloat16()
    angle = torch.rand(b, l, d // 2, generator=g) * 100
    cos, sin = angle.cos(), angle.sin()
    grad = torch.randn(b, l, heads, d, generator=g).bfloat16()

    def run(fn):
      xs, ws = (
          t.to(self.accelerator_device).requires_grad_() for t in (x, weight)
      )
      cs, ss, gs = (t.to(self.accelerator_device) for t in (cos, sin, grad))
      out = fn(xs, ws, cs, ss)
      out.backward(gs)
      return out.cpu(), xs.grad.cpu(), ws.grad.cpu()

    expected = run(_reference)
    actual = run(lambda *a: qk_norm_rope(*a, _EPS))
    for name, e, a in zip(("out", "dx", "dw"), expected, actual):
      self.assertTrue(torch.equal(a, e), f"{name} not bitwise equal")


if __name__ == "__main__":
  absltest.main()
