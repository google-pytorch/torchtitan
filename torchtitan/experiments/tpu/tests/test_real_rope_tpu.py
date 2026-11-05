# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the real-arithmetic complex RoPE override.

`ComplexRoPE` rotates adjacent dimension pairs; `CosSinRoPE` rotates halves.
Drifting from one to the other silently changes what an existing checkpoint
means, so the parity test compares against the complex multiply itself rather
than against the other RoPE.

That parity is asserted bit for bit, which is a claim about the arithmetic the
accelerator performs, so these run on the accelerator rather than on CPU.
"""

from absl.testing import absltest
import torch
from torch._higher_order_ops.scan import scan

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.overrides.real_rope import (
    TpuRealArithmeticComplexRoPE,
    tpu_real_arithmetic_complex_rope,
)
from torchtitan.models.common import ComplexRoPE


def _scan_rope_backward(apply_rotary_emb):
  """Run `apply_rotary_emb` inside a `scan` body and backprop through it.

  Must be compiled: eager `scan` does not re-trace its body through
  functionalization, so it never reaches the failure this override exists for.
  """
  torch.manual_seed(0)
  head_dim = 8
  cache = torch.polar(
      torch.ones(4, head_dim // 2), torch.randn(4, head_dim // 2)
  ).view(1, 4, 1, head_dim // 2)
  weight = torch.randn(2, 4, 2, head_dim, requires_grad=True)

  def body(carry, x):
    rotated, _ = apply_rotary_emb(carry * x, carry * x, cache)
    return rotated.clone(), rotated.sum().reshape(1)

  def step(w):
    carry, ys = scan(body, w, torch.ones(3, 1))
    return carry.sum() + ys.sum()

  torch.compile(step, backend="aot_eager", fullgraph=True)(weight).backward()
  return weight.grad


class TpuRealArithmeticComplexRoPETest(base_device_test.BaseAcceleratorDeviceTest):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    head_dim = 64
    shape = (2, 16, 4, head_dim)
    device = self.accelerator_device
    self.query = torch.randn(*shape, device=device)
    self.key = torch.randn(*shape, device=device)
    self.rope_cache = torch.polar(
        torch.ones(16, head_dim // 2, device=device),
        torch.randn(16, head_dim // 2, device=device),
    ).view(1, 16, 1, head_dim // 2)

  def _complex_reference(self, x):
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_ * self.rope_cache).flatten(3).type_as(x)

  def test_override_preserves_config_and_builds_tpu_rope(self):
    source = ComplexRoPE.Config(dim=8, max_context_length=16, theta=5000.0)
    replacement = tpu_real_arithmetic_complex_rope(source)

    self.assertIsInstance(replacement, TpuRealArithmeticComplexRoPE.Config)
    self.assertEqual(replacement.dim, source.dim)
    self.assertEqual(
        replacement.max_context_length, source.max_context_length
    )
    self.assertEqual(replacement.theta, source.theta)
    self.assertIsInstance(replacement.build(), TpuRealArithmeticComplexRoPE)

  def test_matches_the_complex_multiply_bit_for_bit(self):
    query_out, key_out = TpuRealArithmeticComplexRoPE.apply_rotary_emb(
        self.query, self.key, self.rope_cache
    )
    torch.testing.assert_close(
        query_out, self._complex_reference(self.query), rtol=0, atol=0
    )
    torch.testing.assert_close(
        key_out, self._complex_reference(self.key), rtol=0, atol=0
    )

  def test_output_dtype_and_shape_match_input(self):
    query = self.query.bfloat16()
    query_out, _ = TpuRealArithmeticComplexRoPE.apply_rotary_emb(
        query, query, self.rope_cache
    )
    self.assertEqual(query_out.dtype, torch.bfloat16)
    self.assertEqual(query_out.shape, query.shape)

  def test_upstream_complex_rope_cannot_be_traced_inside_scan(self):
    """The reason this override exists.

    When `view_as_complex_copy` gains an autograd formula upstream this test
    fails, which is the signal to delete the whole override.
    """
    with self.assertRaisesRegex(Exception, "view_as_complex_copy"):
      _scan_rope_backward(ComplexRoPE.apply_rotary_emb)

  def test_the_override_can_be_traced_inside_scan(self):
    grad = _scan_rope_backward(TpuRealArithmeticComplexRoPE.apply_rotary_emb)
    self.assertIsNotNone(grad)
    self.assertTrue(torch.isfinite(grad).all())


if __name__ == "__main__":
  absltest.main()
