# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stock `ComplexRoPE` inside a compiled `scan` body.

Under compile `ComplexRoPE` rotates in real arithmetic, so the scan body never
sees `view_as_complex`, whose functionalized copy has no complex autograd.
"""

from absl.testing import absltest
import torch
from torch._higher_order_ops.scan import scan

from torchtitan.models.common import ComplexRoPE

_HEAD_DIM = 8


def _inputs():
  torch.manual_seed(0)
  cache = torch.polar(
      torch.ones(4, _HEAD_DIM // 2), torch.randn(4, _HEAD_DIM // 2)
  ).view(4, 1, _HEAD_DIM // 2)
  weight = torch.randn(4, 2, _HEAD_DIM)
  return cache, weight


def _body(cache):
  def body(carry, x):
    rotated, _ = ComplexRoPE.apply_rotary_emb(carry * x, carry * x, cache)
    return rotated.clone(), rotated.sum().reshape(1)

  return body


class ComplexRoPEInScanTest(absltest.TestCase):

  def test_grads_match_an_eager_loop(self):
    cache, weight = _inputs()
    xs = torch.linspace(0.5, 1.5, 3).reshape(3, 1)

    def scanned(w):
      carry, ys = scan(_body(cache), w, xs)
      return carry.sum() + ys.sum()

    def looped(w):
      body, total = _body(cache), 0
      for x in xs:
        w, y = body(w, x)
        total = total + y.sum()
      return w.sum() + total

    w1 = weight.clone().requires_grad_(True)
    torch.compile(scanned, backend="aot_eager", fullgraph=True)(w1).backward()
    w2 = weight.clone().requires_grad_(True)
    looped(w2).backward()
    torch.testing.assert_close(w1.grad, w2.grad)


if __name__ == "__main__":
  absltest.main()
