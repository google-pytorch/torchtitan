# Copyright 2026 Google LLC

"""Tests for Mamba-3 Pallas and SSD-3 chunk scan kernel on TPU."""

from absl import logging
from absl.testing import absltest
import torch
import torch.nn.functional as F
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.examples.mamba3.kernels import mamba3_pallas


class Mamba3PallasKernelTest(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for the Mamba-3 Pallas SSD-3 chunk scan kernel on TPU."""

  def test_forward_shape_and_execution(self):
    device = self.accelerator_device
    b, seq_len, nheads, head_dim = 2, 32, 4, 32
    d_state, chunk_size = 16, 16

    torch.manual_seed(42)
    v = torch.randn(
        b, seq_len, nheads, head_dim, device=device, dtype=torch.float32
    )
    b_mat = torch.randn(
        b, seq_len, nheads, d_state, device=device, dtype=torch.float32
    )
    c_mat = torch.randn(
        b, seq_len, nheads, d_state, device=device, dtype=torch.float32
    )
    b_bias = torch.ones(nheads, d_state, device=device, dtype=torch.float32)
    c_bias = torch.ones(nheads, d_state, device=device, dtype=torch.float32)
    dt = F.softplus(
        torch.randn(b, seq_len, nheads, device=device, dtype=torch.float32)
    )
    adt = (
        -F.softplus(
            torch.randn(b, seq_len, nheads, device=device, dtype=torch.float32)
        )
        * dt
    )
    trap_raw = torch.randn(
        b, seq_len, nheads, device=device, dtype=torch.float32
    )
    angle_raw = torch.randn(
        b,
        seq_len,
        nheads,
        d_state // 2,
        device=device,
        dtype=torch.float32,
    )
    d_vec = torch.ones(nheads, device=device, dtype=torch.float32)

    out = mamba3_pallas.ssd3_chunk_scan_pallas(
        v=v,
        b=b_mat,
        c=c_mat,
        b_bias=b_bias,
        c_bias=c_bias,
        adt=adt,
        dt=dt,
        trap_raw=trap_raw,
        angle_raw=angle_raw,
        d=d_vec,
        chunk_size=chunk_size,
    )

    self.assertEqual(out.shape, (b, seq_len, nheads, head_dim))
    self.assertTrue(torch.all(torch.isfinite(out)))
    logging.info("Mamba-3 forward execution test passed.")

  def test_backward_gradient_computation(self):
    device = self.accelerator_device
    b, seq_len, nheads, head_dim = 2, 32, 4, 32
    d_state, chunk_size = 16, 16

    torch.manual_seed(42)
    v = torch.randn(
        b,
        seq_len,
        nheads,
        head_dim,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    b_mat = torch.randn(
        b,
        seq_len,
        nheads,
        d_state,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    c_mat = torch.randn(
        b,
        seq_len,
        nheads,
        d_state,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    b_bias = torch.ones(nheads, d_state, device=device, dtype=torch.float32)
    c_bias = torch.ones(nheads, d_state, device=device, dtype=torch.float32)
    dt = F.softplus(
        torch.randn(b, seq_len, nheads, device=device, dtype=torch.float32)
    )
    adt = (
        -F.softplus(
            torch.randn(b, seq_len, nheads, device=device, dtype=torch.float32)
        )
        * dt
    )
    trap_raw = torch.randn(
        b, seq_len, nheads, device=device, dtype=torch.float32
    )
    angle_raw = torch.randn(
        b,
        seq_len,
        nheads,
        d_state // 2,
        device=device,
        dtype=torch.float32,
    )
    d_vec = torch.ones(nheads, device=device, dtype=torch.float32)

    out = mamba3_pallas.ssd3_chunk_scan_pallas(
        v=v,
        b=b_mat,
        c=c_mat,
        b_bias=b_bias,
        c_bias=c_bias,
        adt=adt,
        dt=dt,
        trap_raw=trap_raw,
        angle_raw=angle_raw,
        d=d_vec,
        chunk_size=chunk_size,
    )
    loss = out.sum()
    loss.backward()

    self.assertIsNotNone(v.grad)
    self.assertIsNotNone(b_mat.grad)
    self.assertIsNotNone(c_mat.grad)
    self.assertTrue(torch.all(torch.isfinite(v.grad)))
    self.assertTrue(torch.all(torch.isfinite(b_mat.grad)))
    self.assertTrue(torch.all(torch.isfinite(c_mat.grad)))
    logging.info("Mamba-3 backward gradient computation test passed.")


if __name__ == "__main__":
  absltest.main()
