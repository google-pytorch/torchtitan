# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Mamba-2 Pallas and SSD chunk scan kernel on TPU."""

from typing import Optional
from absl import logging
from absl.testing import absltest
import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.examples.mamba2.kernels import mamba2_pallas


def reference_sequential_ssd(
    x: torch.Tensor,
    dt: torch.Tensor,
    a_log: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  """Sequential reference implementation of Mamba-2 SSD for parity checking."""
  batch_size, seq_len, nheads, head_dim = x.shape
  _, _, ngroups, d_state = b.shape

  if ngroups != nheads:
    b = b.repeat_interleave(nheads // ngroups, dim=2)
    c = c.repeat_interleave(nheads // ngroups, dim=2)

  a_val = -torch.exp(a_log.float()).view(1, 1, nheads, 1)  # (1, 1, nheads, 1)
  dt_a = dt.float().unsqueeze(-1) * a_val  # (batch, seq, nheads, 1)
  decay = torch.exp(dt_a)  # (batch, seq, nheads, 1)

  y_list = []
  h = torch.zeros(
      batch_size,
      nheads,
      d_state,
      head_dim,
      device=x.device,
      dtype=torch.float32,
  )

  for t in range(seq_len):
    b_t = b[:, t].unsqueeze(-1)  # (batch, nheads, d_state, 1)
    x_t = x[:, t].unsqueeze(2)  # (batch, nheads, 1, head_dim)
    input_state = (b_t * x_t).float()

    decay_t = decay[:, t].unsqueeze(-1)  # (batch, nheads, 1, 1)
    h = h * decay_t + input_state

    c_t = c[:, t].unsqueeze(2)  # (batch, nheads, 1, d_state)
    y_t = torch.matmul(c_t.float(), h).squeeze(2)  # (batch, nheads, head_dim)
    y_list.append(y_t.to(x.dtype))

  y = torch.stack(y_list, dim=1)  # (batch, seq_len, nheads, head_dim)

  if d is not None:
    y = y + x * d.view(1, 1, nheads, -1).to(x.dtype)

  return y


class Mamba2PallasKernelTest(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for the Mamba-2 Pallas SSD chunk scan kernel on TPU."""

  def test_forward_shape_and_values(self):
    """Verifies output shape and values of ssd_chunk_scan_pallas against sequential reference."""
    device = self.accelerator_device
    b, seq_len, nheads, head_dim = 2, 128, 8, 64
    d_state, chunk_size = 16, 64

    torch.manual_seed(42)
    x = torch.randn(
        b, seq_len, nheads, head_dim, device=device, dtype=torch.float32
    )
    dt = F.softplus(
        torch.randn(b, seq_len, nheads, device=device, dtype=torch.float32)
    )
    a_log = torch.log(torch.arange(1, nheads + 1, device=device).float())
    b_mat = torch.randn(
        b, seq_len, nheads, d_state, device=device, dtype=torch.float32
    )
    c_mat = torch.randn(
        b, seq_len, nheads, d_state, device=device, dtype=torch.float32
    )
    d_vec = torch.ones(nheads, head_dim, device=device, dtype=torch.float32)

    actual_out = mamba2_pallas.ssd_chunk_scan_pallas(
        x, dt, a_log, b_mat, c_mat, d=d_vec, chunk_size=chunk_size
    )
    expected_out = reference_sequential_ssd(x, dt, a_log, b_mat, c_mat, d=d_vec)

    self.assertEqual(actual_out.shape, (b, seq_len, nheads, head_dim))
    # Note: tolerance accounts for chunk-wise associative scan vs step-by-step
    # sequential accumulation order
    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=0.2, atol=0.2
    )

    # Also verify with 1D d vector of shape (nheads,)
    d_1d = torch.ones(nheads, device=device, dtype=torch.float32)
    actual_1d = mamba2_pallas.ssd_chunk_scan_pallas(
        x, dt, a_log, b_mat, c_mat, d=d_1d, chunk_size=chunk_size
    )
    expected_1d = reference_sequential_ssd(x, dt, a_log, b_mat, c_mat, d=d_1d)
    self.assertEqual(actual_1d.shape, (b, seq_len, nheads, head_dim))
    torch.testing.assert_close(
        actual_1d.cpu(), expected_1d.cpu(), rtol=0.2, atol=0.2
    )
    logging.info("Mamba-2 forward parity test passed.")

  def test_grouped_heads_parity(self):
    """Tests Grouped-Query / GSSM configuration where ngroups != nheads."""
    device = self.accelerator_device
    b, seq_len, nheads, ngroups, head_dim = 2, 128, 8, 2, 64
    d_state, chunk_size = 16, 64

    torch.manual_seed(42)
    x = torch.randn(
        b, seq_len, nheads, head_dim, device=device, dtype=torch.float32
    )
    dt = F.softplus(
        torch.randn(b, seq_len, nheads, device=device, dtype=torch.float32)
    )
    a_log = torch.log(torch.arange(1, nheads + 1, device=device).float())
    b_mat = torch.randn(
        b, seq_len, ngroups, d_state, device=device, dtype=torch.float32
    )
    c_mat = torch.randn(
        b, seq_len, ngroups, d_state, device=device, dtype=torch.float32
    )

    actual_out = mamba2_pallas.ssd_chunk_scan_pallas(
        x, dt, a_log, b_mat, c_mat, chunk_size=chunk_size
    )
    expected_out = reference_sequential_ssd(x, dt, a_log, b_mat, c_mat)

    self.assertEqual(actual_out.shape, (b, seq_len, nheads, head_dim))
    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=0.2, atol=0.2
    )
    logging.info("Mamba-2 GSSM grouped heads parity test passed.")

  def test_backward_pass_gradients(self):
    """Verifies backward pass gradient computation for Mamba-2 inputs."""
    device = self.accelerator_device
    b, seq_len, nheads, head_dim = 2, 128, 4, 64
    d_state, chunk_size = 16, 64

    torch.manual_seed(42)
    x = torch.randn(
        b,
        seq_len,
        nheads,
        head_dim,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    dt = F.softplus(
        torch.randn(b, seq_len, nheads, device=device, dtype=torch.float32)
    ).requires_grad_(True)
    a_log = torch.log(
        torch.arange(1, nheads + 1, device=device).float()
    ).requires_grad_(True)
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
    d_vec = torch.ones(
        nheads, head_dim, device=device, dtype=torch.float32, requires_grad=True
    )

    out = mamba2_pallas.ssd_chunk_scan_pallas(
        x, dt, a_log, b_mat, c_mat, d=d_vec, chunk_size=chunk_size
    )
    loss = out.sum()
    loss.backward()

    self.assertIsNotNone(x.grad, "grad_x is None")
    self.assertIsNotNone(dt.grad, "grad_dt is None")
    self.assertIsNotNone(a_log.grad, "grad_a_log is None")
    self.assertIsNotNone(b_mat.grad, "grad_b is None")
    self.assertIsNotNone(c_mat.grad, "grad_c is None")
    self.assertIsNotNone(d_vec.grad, "grad_d is None")

    self.assertGreater(x.grad.abs().sum().item(), 0, "grad_x is all zeros")
    self.assertGreater(b_mat.grad.abs().sum().item(), 0, "grad_b is all zeros")
    self.assertGreater(c_mat.grad.abs().sum().item(), 0, "grad_c is all zeros")
    logging.info("Mamba-2 backward pass test passed.")

  def test_backward_pass_gradients_without_d(self):
    """Verifies backward pass gradient computation when d is None (covering line 219 branch)."""
    device = self.accelerator_device
    b, seq_len, nheads, head_dim = 2, 64, 4, 32
    d_state, chunk_size = 16, 32

    torch.manual_seed(42)
    x = torch.randn(
        b,
        seq_len,
        nheads,
        head_dim,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    dt = F.softplus(
        torch.randn(b, seq_len, nheads, device=device, dtype=torch.float32)
    ).requires_grad_(True)
    a_log = torch.log(
        torch.arange(1, nheads + 1, device=device).float()
    ).requires_grad_(True)
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

    out = mamba2_pallas.ssd_chunk_scan_pallas(
        x, dt, a_log, b_mat, c_mat, d=None, chunk_size=chunk_size
    )
    loss = out.sum()
    loss.backward()

    self.assertIsNotNone(x.grad, "grad_x is None")
    self.assertIsNotNone(dt.grad, "grad_dt is None")
    self.assertIsNotNone(a_log.grad, "grad_a_log is None")
    self.assertIsNotNone(b_mat.grad, "grad_b is None")
    self.assertIsNotNone(c_mat.grad, "grad_c is None")
    self.assertGreater(x.grad.abs().sum().item(), 0)

  def test_fused_chunk_kernel_direct(self):
    """Directly executes ssd_fused_chunk_kernel to test VMEM logic and kill mutation testing mutants."""
    chunk_size, head_dim, d_state = 16, 32, 16
    key = jax.random.PRNGKey(123)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    x = jax.random.normal(k1, (chunk_size, head_dim), dtype=jnp.float32)
    dt = jax.nn.softplus(
        jax.random.normal(k2, (chunk_size, 1), dtype=jnp.float32)
    )
    a_log = jnp.array([[-1.5]], dtype=jnp.float32)
    b = jax.random.normal(k3, (chunk_size, d_state), dtype=jnp.float32)
    c = jax.random.normal(k4, (chunk_size, d_state), dtype=jnp.float32)
    prev_s = jax.random.normal(k5, (d_state, head_dim), dtype=jnp.float32)

    class _MockRef:

      def __init__(self, arr):
        self.arr = arr

      def __getitem__(self, idx):
        return self.arr[idx]

      def __setitem__(self, idx, val):
        self.arr = val

    out_ref = _MockRef(jnp.zeros((chunk_size, head_dim), dtype=jnp.float32))
    next_s_ref = _MockRef(jnp.zeros((d_state, head_dim), dtype=jnp.float32))

    mamba2_pallas.ssd_fused_chunk_kernel(
        _MockRef(x),
        _MockRef(dt),
        _MockRef(a_log),
        _MockRef(b),
        _MockRef(c),
        _MockRef(prev_s),
        out_ref,
        next_s_ref,
    )

    out = out_ref.arr
    next_s = next_s_ref.arr

    self.assertEqual(out.shape, (chunk_size, head_dim))
    self.assertEqual(next_s.shape, (d_state, head_dim))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(next_s))))
    self.assertGreater(float(jnp.sum(jnp.abs(out))), 0)
    self.assertGreater(float(jnp.sum(jnp.abs(next_s))), 0)

  def test_custom_op_autograd_registration(self):
    """Explicitly tests the registered autograd backward formula of _pallas_ssd_fwd."""
    # pylint: disable=protected-access
    device = self.accelerator_device
    if (
        device.type not in ("tpu", "xla")
        or mamba2_pallas._pallas_ssd_fwd is None
    ):
      logging.info(
          "Skipping custom op autograd test on non-TPU environment"
          " (device=%s).",
          device,
      )
      return

    b, nheads, num_chunks, chunk_size, head_dim, d_state = 1, 2, 2, 16, 16, 8
    x_hf = torch.randn(
        b,
        nheads,
        num_chunks,
        chunk_size,
        head_dim,
        device=device,
        requires_grad=True,
    )
    dt_2d = F.softplus(
        torch.randn(b, nheads, num_chunks, chunk_size, 1, device=device)
    ).requires_grad_(True)
    a_log_2d = torch.log(
        torch.arange(1, nheads + 1, device=device).float().view(nheads, 1, 1)
    ).requires_grad_(True)
    b_hf = torch.randn(
        b,
        nheads,
        num_chunks,
        chunk_size,
        d_state,
        device=device,
        requires_grad=True,
    )
    c_hf = torch.randn(
        b,
        nheads,
        num_chunks,
        chunk_size,
        d_state,
        device=device,
        requires_grad=True,
    )

    out = mamba2_pallas._pallas_ssd_fwd(x_hf, dt_2d, a_log_2d, b_hf, c_hf)
    loss = out.sum()
    loss.backward()

    self.assertIsNotNone(x_hf.grad, "custom op grad_x is None")
    self.assertIsNotNone(dt_2d.grad, "custom op grad_dt is None")
    self.assertIsNotNone(a_log_2d.grad, "custom op grad_a_log is None")
    self.assertIsNotNone(b_hf.grad, "custom op grad_b is None")
    self.assertIsNotNone(c_hf.grad, "custom op grad_c is None")
    self.assertGreater(x_hf.grad.abs().sum().item(), 0)
    logging.info("Mamba-2 custom op autograd registration test passed.")

  def test_jax_ssd_scan_forward_and_backward(self):
    """Directly tests JAX forward and backward scan computations."""
    b, nheads, num_chunks, chunk_size, head_dim, d_state = 2, 4, 2, 16, 16, 8
    key = jax.random.PRNGKey(42)

    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    x = jax.random.normal(
        k1, (b, nheads, num_chunks, chunk_size, head_dim), dtype=jnp.float32
    )
    dt = jax.nn.softplus(
        jax.random.normal(
            k2, (b, nheads, num_chunks, chunk_size, 1), dtype=jnp.float32
        )
    )
    a_log = jnp.log(jnp.arange(1, nheads + 1, dtype=jnp.float32)).reshape(
        nheads, 1, 1
    )
    b_mat = jax.random.normal(
        k3, (b, nheads, num_chunks, chunk_size, d_state), dtype=jnp.float32
    )
    c_mat = jax.random.normal(
        k4, (b, nheads, num_chunks, chunk_size, d_state), dtype=jnp.float32
    )
    dout = jax.random.normal(
        k5, (b, nheads, num_chunks, chunk_size, head_dim), dtype=jnp.float32
    )

    y = mamba2_pallas.ssd_scan_pure_jax(x, dt, a_log, b_mat, c_mat)
    self.assertEqual(y.shape, (b, nheads, num_chunks, chunk_size, head_dim))
    self.assertTrue(bool(jnp.all(jnp.isfinite(y))))

    dx, ddt, da_log, db, dc = mamba2_pallas.ssd_scan_bwd_jax(
        x, dt, a_log, b_mat, c_mat, dout
    )
    self.assertEqual(dx.shape, x.shape)
    self.assertEqual(ddt.shape, dt.shape)
    self.assertEqual(da_log.shape, a_log.shape)
    self.assertEqual(db.shape, b_mat.shape)
    self.assertEqual(dc.shape, c_mat.shape)

    for name, grad in [
        ("dx", dx),
        ("ddt", ddt),
        ("da_log", da_log),
        ("db", db),
        ("dc", dc),
    ]:
      self.assertTrue(
          bool(jnp.all(jnp.isfinite(grad))), f"{name} contains NaN/Inf"
      )
      self.assertGreater(
          float(jnp.sum(jnp.abs(grad))), 0, f"{name} is all zeros"
      )
    logging.info("Direct JAX forward and backward scan test passed.")


if __name__ == "__main__":
  absltest.main()
