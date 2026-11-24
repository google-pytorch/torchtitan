# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the count-bounded expert SwiGLU kernel."""

from absl.testing import absltest
import torch
import torch.nn.functional as F
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.kernels import expert_swiglu as expert_swiglu_lib
from torchtitan.experiments.tpu.testing import base_device_test


def _eager_swiglu(
    gate_RF: torch.Tensor, up_RF: torch.Tensor, row_scale_R: torch.Tensor
) -> torch.Tensor:
  """TpuGroupedExperts' eager path: silu(gate) * up in BF16, scaled in FP32."""
  hidden_RF = F.silu(gate_RF) * up_RF
  return (hidden_RF.float() * row_scale_R.reshape(-1, 1)).to(hidden_RF.dtype)


def _relative_l2_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
  return float(
      (actual.double() - expected.double()).norm()
      / expected.double().norm().clamp_min(1e-12)
  )


class ExpertSwigluTest(base_device_test.BaseAcceleratorDeviceTest):

  def _require_tpu(self) -> None:
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("The Pallas expert SwiGLU kernel requires TPU.")

  def test_tiling_divides_rows_and_features_and_fits_vmem(self):
    vmem_limit = 32 * 1024 * 1024
    for is_bwd in (False, True):
      with self.subTest(is_bwd=is_bwd):
        block, f_chunk = expert_swiglu_lib._choose_swiglu_tiling(
            4096, 1024, 2, vmem_limit, is_bwd=is_bwd
        )
        self.assertEqual(4096 % block, 0)
        self.assertEqual(1024 % f_chunk, 0)
        self.assertEqual(f_chunk % 128, 0)
    # Rows or features that aren't multiples of 128 can't tile.
    self.assertIsNone(
        expert_swiglu_lib._choose_swiglu_tiling(
            4096, 1000, 2, vmem_limit, is_bwd=False
        )
    )
    self.assertIsNone(
        expert_swiglu_lib._choose_swiglu_tiling(
            1000, 1024, 2, vmem_limit, is_bwd=False
        )
    )

  def test_falls_back_to_eager_off_tpu_or_without_row_scale(self):
    torch.manual_seed(0)
    gate = torch.randn(256, 256, dtype=torch.bfloat16)
    up = torch.randn(256, 256, dtype=torch.bfloat16)
    scale = torch.rand(256)
    counts = torch.tensor([100, 28], dtype=torch.int32)
    torch.testing.assert_close(
        expert_swiglu_lib.expert_swiglu(gate, up, counts, row_scale_R=scale),
        _eager_swiglu(gate, up, scale),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        expert_swiglu_lib.expert_swiglu(gate, up, counts),
        F.silu(gate) * up,
        rtol=0,
        atol=0,
    )

  def test_matches_eager_on_routed_rows_and_zeroes_the_aligned_tail(self):
    """Rows past sum(counts) are stale (NaN here) and must not leak anywhere."""
    self._require_tpu()
    dev = self.accelerator_device
    num_rows, f_dim = 4096, 512
    for count, row_align in (
        (0, 512),
        (1, 512),
        (700, 512),
        (1024, 1024),
        (2600, 1024),
        (4096, 512),
    ):
      with self.subTest(count=count, row_align=row_align):
        generator = torch.Generator().manual_seed(count)
        rows = torch.arange(num_rows).reshape(-1, 1)
        stale = rows >= count
        nan = torch.tensor(float("nan"))
        gate = torch.randn(num_rows, f_dim, generator=generator)
        up = torch.randn(num_rows, f_dim, generator=generator)
        grad = torch.randn(num_rows, f_dim, generator=generator)
        gate = torch.where(stale, nan, gate).to(torch.bfloat16).to(dev)
        up = torch.where(stale, nan, up).to(torch.bfloat16).to(dev)
        grad = torch.where(stale, nan, grad).to(torch.bfloat16).to(dev)
        scale = (0.1 + torch.rand(num_rows, generator=generator)).to(dev)
        quarter = count // 4
        counts_e = torch.tensor(
            [quarter, quarter, quarter, count - 3 * quarter],
            dtype=torch.int32,
            device=dev,
        )
        aligned = min(-(-count // row_align) * row_align, num_rows)

        gate_k = gate.clone().requires_grad_(True)
        up_k = up.clone().requires_grad_(True)
        scale_k = scale.clone().requires_grad_(True)
        out = expert_swiglu_lib.expert_swiglu(
            gate_k, up_k, counts_e, row_scale_R=scale_k, row_align=row_align
        )
        dgate, dup, dscale = torch.autograd.grad(
            out, (gate_k, up_k, scale_k), grad
        )

        # Rows in [count, aligned) are zero; dscale is zero past count.
        for name, value in (("out", out), ("dgate", dgate), ("dup", dup)):
          self.assertTrue(
              bool((value[count:aligned] == 0).all()), f"{name} tail"
          )
        self.assertTrue(bool((dscale[count:] == 0).all()))
        if count == 0:
          continue

        gate_r = gate[:count].clone().requires_grad_(True)
        up_r = up[:count].clone().requires_grad_(True)
        scale_r = scale[:count].clone().requires_grad_(True)
        out_r = _eager_swiglu(gate_r, up_r, scale_r)
        dgate_r, dup_r, dscale_r = torch.autograd.grad(
            out_r, (gate_r, up_r, scale_r), grad[:count]
        )
        torch.testing.assert_close(out[:count], out_r)
        torch.testing.assert_close(dgate[:count], dgate_r)
        torch.testing.assert_close(dup[:count], dup_r)
        self.assertLessEqual(
            _relative_l2_error(dscale[:count], dscale_r), 1e-4
        )


if __name__ == "__main__":
  absltest.main()
