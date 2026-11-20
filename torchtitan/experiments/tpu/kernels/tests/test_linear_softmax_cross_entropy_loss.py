"""Tests for linear_softmax_cross_entropy_loss."""

import functools
import importlib
import inspect
from types import SimpleNamespace
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy
import torch
from torchtitan.components import loss
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.experiments.tpu.kernels import linear_softmax_cross_entropy_loss


class LinearSoftmaxCrossEntropyLossUnitTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # Reduced losses: the custom op's one-element cotangent is passed to
      # Tokamax as a scalar.
      ("sum", "sum", [1.0], ()),
      ("mean", "mean", [1.0], ()),
      # One loss per row: the (B,) cotangent is passed through unchanged.
      ("none", "none", [0.5, -2.0], (2,)),
  )
  def test_backward_cotangent_shape(self, reduction, dout_values, dout_shape):
    dout = jnp.asarray(dout_values, dtype=jnp.float32)
    lse = jnp.zeros((2,), dtype=jnp.float32)
    x = jnp.zeros((2, 4), dtype=jnp.bfloat16)
    labels = jnp.zeros((2,), dtype=jnp.int32)
    weights = jnp.zeros((4, 8), dtype=jnp.bfloat16)
    kernel = importlib.import_module(
        "tokamax._src.ops.linear_softmax_cross_entropy_loss."
        "pallas_mosaic_tpu_kernel"
    )

    with mock.patch.object(
        kernel,
        "linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu",
        return_value=(jnp.zeros_like(x), jnp.zeros_like(weights)),
    ) as backward:
      # pylint: disable=protected-access
      linear_softmax_cross_entropy_loss._my_bwd(
          dout, lse, x, labels, weights, reduction, "mosaic_tpu", 1, 1, 1
      )

    passed = backward.call_args.args[0]
    self.assertEqual(passed.shape, dout_shape)
    self.assertEqual(numpy.asarray(passed).reshape(-1).tolist(), dout_values)

  def test_reduction_none_xla_backward_matches_autograd(self):
    # The XLA reference path runs on CPU: check the per-row backward's values.
    torch.manual_seed(0)
    rows, hidden, vocab = 6, 8, 16
    x = torch.randn(rows, hidden)
    weights = torch.randn(hidden, vocab) / 4
    labels = torch.randint(0, vocab, (rows,))
    dout = torch.randn(rows)

    xr = x.clone().requires_grad_(True)
    wr = weights.clone().requires_grad_(True)
    logits = xr @ wr
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    (loss * dout).sum().backward()
    lse = torch.logsumexp(logits.detach(), dim=-1)

    to_jax = lambda t: jnp.asarray(t.numpy())
    # On a TPU host JAX runs on the TPU, whose default matmul precision is
    # bf16; compare against the FP32 autograd reference in FP32.
    with jax.default_matmul_precision("float32"):
      # pylint: disable=protected-access
      grad_x, grad_w = linear_softmax_cross_entropy_loss._my_bwd(
          to_jax(dout),
          to_jax(lse),
          to_jax(x),
          to_jax(labels.to(torch.int32)),
          to_jax(weights),
          "none",
          "xla",
          rows,
          hidden,
          vocab,
      )
    torch.testing.assert_close(
        torch.from_numpy(numpy.asarray(grad_x)), xr.grad, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        torch.from_numpy(numpy.asarray(grad_w)), wr.grad, rtol=1e-4, atol=1e-5
    )

  def test_backward_uses_single_buffer_when_double_buffering_exceeds_vmem(self):
    dout = jnp.ones((1,), dtype=jnp.float32)
    lse = jnp.zeros((1024,), dtype=jnp.float32)
    x = jnp.zeros((1024, 2048), dtype=jnp.bfloat16)
    labels = jnp.zeros((1024,), dtype=jnp.int32)
    weights = jnp.zeros((2048, 1280), dtype=jnp.bfloat16)
    kernel = importlib.import_module(
        "tokamax._src.ops.linear_softmax_cross_entropy_loss."
        "pallas_mosaic_tpu_kernel"
    )
    orig_bwd = inspect.unwrap(
        kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu
    )
    self.assertTrue(hasattr(kernel, "_calculate_bwd_vmem_bytes"))
    self.assertIn("buffer_count", inspect.signature(orig_bwd).parameters)
    from jax.experimental.pallas import tpu as pltpu  # pylint: disable=g-import-not-at-top

    fake_info = SimpleNamespace(vmem_capacity_bytes=64 * 1024 * 1024)
    calls = []

    @functools.wraps(orig_bwd)
    def spy_bwd(*args, **kwargs):
      calls.append((args, kwargs))
      return jnp.zeros_like(args[2]), jnp.zeros_like(args[4])

    with (
        mock.patch.object(pltpu, "get_tpu_info", return_value=fake_info),
        mock.patch.object(
            kernel,
            "linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu",
            new=spy_bwd,
        ),
        mock.patch.object(
            linear_softmax_cross_entropy_loss.logger, "info"
        ) as log_info,
        mock.patch.object(
            linear_softmax_cross_entropy_loss.logger, "warning"
        ) as log_warn,
    ):
      # pylint: disable=protected-access
      linear_softmax_cross_entropy_loss._logged_bwd_buffer_events.clear()
      # Default tile (1024, 512, 2048) bf16 on TPU7x (64 MiB VMEM, 57.6 MiB
      # limit) estimates 56.4 MiB at (2, 2) and stays on default buffering.
      linear_softmax_cross_entropy_loss._my_bwd(
          dout,
          lse,
          x[:, :512],
          labels,
          weights[:512, :2048],
          "sum",
          "mosaic_tpu",
          1024,
          512,
          2048,
      )
      self.assertNotIn("buffer_count", calls[-1][1])
      self.assertEqual(log_info.call_count, 0)
      self.assertEqual(log_warn.call_count, 0)
      # Intermediate tile (1024, 1024, 1280) bf16 exceeds 57.6 MiB at (2, 2)
      # (61.4 MiB) and fits at (2, 1) (52.4 MiB), logged once at info level.
      linear_softmax_cross_entropy_loss._my_bwd(
          dout,
          lse,
          x[:, :1024],
          labels,
          weights[:1024, :1280],
          "sum",
          "mosaic_tpu",
          1024,
          1024,
          1280,
      )
      linear_softmax_cross_entropy_loss._my_bwd(
          dout,
          lse,
          x[:, :1024],
          labels,
          weights[:1024, :1280],
          "sum",
          "mosaic_tpu",
          1024,
          1024,
          1280,
      )
      self.assertEqual(calls[-1][1].get("buffer_count"), (2, 1))
      self.assertEqual(log_info.call_count, 1)
      # Single-H benchmark tile (1024, 2048, 640) bf16 exceeds 57.6 MiB at
      # both (2, 2) (72.9 MiB) and (2, 1) (59.9 MiB), and fits at 1 (46.9 MiB).
      linear_softmax_cross_entropy_loss._my_bwd(
          dout, lse, x, labels, weights, "sum", "mosaic_tpu", 1024, 2048, 640
      )
      linear_softmax_cross_entropy_loss._my_bwd(
          dout, lse, x, labels, weights, "sum", "mosaic_tpu", 1024, 2048, 640
      )
      self.assertEqual(calls[-1][1].get("buffer_count"), 1)
      self.assertEqual(log_info.call_count, 2)
      self.assertEqual(log_warn.call_count, 0)
      # In fp32, (1024, 2048, 640) exceeds 57.6 MiB even at buffer_count=1
      # (90.9 MiB at (2, 2), 77.9 MiB at (2, 1), 64.9 MiB at 1), so it returns
      # 1 and logs a warning once.
      linear_softmax_cross_entropy_loss._my_bwd(
          dout,
          lse,
          x.astype(jnp.float32),
          labels,
          weights.astype(jnp.float32),
          "sum",
          "mosaic_tpu",
          1024,
          2048,
          640,
      )
      linear_softmax_cross_entropy_loss._my_bwd(
          dout,
          lse,
          x.astype(jnp.float32),
          labels,
          weights.astype(jnp.float32),
          "sum",
          "mosaic_tpu",
          1024,
          2048,
          640,
      )
      self.assertEqual(calls[-1][1].get("buffer_count"), 1)
      self.assertEqual(log_info.call_count, 2)
      self.assertEqual(log_warn.call_count, 1)
      # Missing _calculate_bwd_vmem_bytes logs a warning once and keeps (2, 2).
      with mock.patch.object(kernel, "_calculate_bwd_vmem_bytes", None):
        linear_softmax_cross_entropy_loss._my_bwd(
            dout, lse, x, labels, weights, "sum", "mosaic_tpu", 1024, 2048, 640
        )
        linear_softmax_cross_entropy_loss._my_bwd(
            dout, lse, x, labels, weights, "sum", "mosaic_tpu", 1024, 2048, 640
        )
      self.assertNotIn("buffer_count", calls[-1][1])
      self.assertEqual(log_warn.call_count, 2)


class LinearSoftmaxCrossEntropyLossTest(
    base_device_test.BaseAcceleratorDeviceTest
):
  """Tests for LinearSoftmaxCrossEntropyLoss."""

  def setUp(self):
    super().setUp()
    if self.accelerator_device.type != "tpu":
      self.skipTest("Pallas LCE kernel requires a TPU device.")

  @parameterized.parameters(
      (torch.float32, "sum", "mosaic_tpu", 1e-4, 1e-5),
      (torch.float32, "sum", "xla", 1e-4, 1e-5),
      (torch.float32, "mean", "mosaic_tpu", 1e-4, 1e-5),
      (torch.float32, "mean", "xla", 1e-4, 1e-5),
      (torch.bfloat16, "sum", "mosaic_tpu", 1.6e-2, 1e-5),
      (torch.bfloat16, "sum", "xla", 1.6e-2, 1e-5),
      (torch.bfloat16, "mean", "mosaic_tpu", 1.6e-2, 1e-5),
      (torch.bfloat16, "mean", "xla", 1.6e-2, 1e-5),
  )
  def test_parity(self, dtype, reduction, implementation, rtol, atol):
    device = self.accelerator_device
    b, s, h, v = 32, 32, 128, 1024  # Batch, SeqLen, Hidden, Vocab

    # Generate on CPU first
    x = torch.randn(b, s, h, dtype=dtype)
    weights = torch.randn(h, v, dtype=dtype)
    labels = torch.randint(0, v, (b, s), dtype=torch.int64)

    # Move to TPU
    x_tpu = x.to(device)
    weights_tpu = weights.to(device)
    labels_tpu = labels.to(device)

    # Reference Math on TPU
    # x is (B, S, H), weights is (H, V)
    # logits = x @ weights -> (B, S, V)
    # labels is (B, S)
    logits = x_tpu @ weights_tpu

    if reduction == "sum":
      expected_out = loss.cross_entropy_loss(
          logits.view(-1, v), labels_tpu.view(-1)
      )
    else:
      expected_out = torch.nn.functional.cross_entropy(
          logits.view(-1, v).float(),
          labels_tpu.view(-1),
          reduction="mean",
      )

    # TPU Pallas
    # tokamax_loss expects x as (B, H) and labels as (B,).
    # We need to flatten x and labels before passing to tokamax_loss.
    x_flat = x_tpu.view(-1, h)
    labels_flat = labels_tpu.view(-1)

    actual_out = (
        linear_softmax_cross_entropy_loss.linear_softmax_cross_entropy_loss(
            x_flat,
            labels_flat,
            weights_tpu,
            reduction=reduction,
            implementation=implementation,
            b_block_size=128,
            h_block_size=128,
            v_block_size=128,
        )
    )

    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=rtol, atol=atol
    )
    logging.info(
        "Parity %s test passed for %s. Output: %s, Expected: %s",
        reduction,
        dtype,
        actual_out,
        expected_out,
    )

  @parameterized.parameters(
      (torch.float32, "sum", "mosaic_tpu", 1e-4, 1e-5, 1e-2, 1e-2),
      (torch.float32, "sum", "xla", 1e-4, 1e-5, 1e-2, 1e-2),
      (torch.float32, "mean", "mosaic_tpu", 1e-4, 1e-5, 1e-2, 1e-2),
      (torch.float32, "mean", "xla", 1e-4, 1e-5, 1e-2, 1e-2),
      (torch.bfloat16, "sum", "mosaic_tpu", 1.6e-2, 1e-5, 5e-2, 0.5),
      (torch.bfloat16, "sum", "xla", 1.6e-2, 1e-5, 2.0, 1.0),
      (torch.bfloat16, "mean", "mosaic_tpu", 1.6e-2, 1e-5, 5e-2, 1e-2),
      (torch.bfloat16, "mean", "xla", 1.6e-2, 1e-5, 5e-2, 1e-2),
      # reduction="none": one loss per row, backward with a random per-row
      # cotangent. bf16 grad tolerances as in the bf16 cases above: the
      # reference computes the logits in bf16 too.
      (torch.float32, "none", "mosaic_tpu", 1e-4, 1e-5, 1e-2, 1e-2),
      (torch.float32, "none", "xla", 1e-4, 1e-5, 1e-2, 1e-2),
      (torch.bfloat16, "none", "mosaic_tpu", 1.6e-2, 1e-2, 5e-2, 0.5),
      (torch.bfloat16, "none", "xla", 1.6e-2, 1e-2, 5e-2, 0.5),
  )
  def test_gradient_new(
      self,
      dtype,
      reduction,
      implementation,
      loss_rtol,
      loss_atol,
      grad_rtol,
      grad_atol,
  ):
    device = self.accelerator_device

    b, h, v = 1024, 128, 1024

    x = torch.randn(b, h, dtype=dtype)
    weights = torch.randn(h, v, dtype=dtype)
    labels = torch.randint(0, v, (b,), dtype=torch.int64)

    x_tpu = x.detach().clone().to(device).requires_grad_(True)
    weights_tpu = weights.detach().clone().to(device).requires_grad_(True)
    labels_tpu = labels.to(device)
    # reduction="none" returns one loss per row; back-propagate a random
    # per-row cotangent through it.
    dout_tpu = (
        torch.randn(b, dtype=torch.float32).to(device)
        if reduction == "none"
        else None
    )

    def backward(out):
      (out if dout_tpu is None else (out * dout_tpu).sum()).backward()

    # Reference Math on TPU
    logits = x_tpu @ weights_tpu
    expected_out = torch.nn.functional.cross_entropy(
        logits.float(),
        labels_tpu,
        reduction=reduction,
    )
    backward(expected_out)
    expected_grad_x = x_tpu.grad.clone()
    expected_grad_weights = weights_tpu.grad.clone()

    # Zero out grads
    x_tpu.grad.zero_()
    weights_tpu.grad.zero_()

    # TPU Pallas
    def run_custom(x, l, w):
      return (
          linear_softmax_cross_entropy_loss.linear_softmax_cross_entropy_loss(
              x,
              l,
              w,
              reduction=reduction,
              implementation=implementation,
              b_block_size=128,
              v_block_size=128,
          )
      )

    actual_out = run_custom(x_tpu, labels_tpu, weights_tpu)
    if reduction == "none":
      self.assertEqual(tuple(actual_out.shape), (b,))
    backward(actual_out)
    actual_grad_x = x_tpu.grad
    actual_grad_weights = weights_tpu.grad

    # Compare

    self.assertEqual(actual_grad_x.dtype, x_tpu.dtype)
    self.assertEqual(actual_grad_weights.dtype, weights_tpu.dtype)

    torch.testing.assert_close(
        actual_out.detach().cpu(),
        expected_out.detach().cpu(),
        rtol=loss_rtol,
        atol=loss_atol,
    )

    # Try creating a new tensor and copying
    actual_grad_x_new = torch.empty_like(actual_grad_x).copy_(actual_grad_x)
    expected_grad_x_new = torch.empty_like(expected_grad_x).copy_(
        expected_grad_x
    )

    actual_grad_x_cpu = actual_grad_x_new.cpu()
    expected_grad_x_cpu = expected_grad_x_new.cpu()

    torch.testing.assert_close(
        actual_grad_x_cpu, expected_grad_x_cpu, rtol=grad_rtol, atol=grad_atol
    )
    torch.testing.assert_close(
        actual_grad_weights.cpu(),
        expected_grad_weights.cpu(),
        rtol=grad_rtol,
        atol=grad_atol,
    )
    logging.info(
        "Gradient %s test passed for %s.",
        reduction,
        dtype,
    )

  @parameterized.named_parameters(
      # name, dtype, soft_cap, output_multiplier, loss_rtol, grad_rtol,
      # grad_atol; tolerances as in test_linear_softmax_ce_softcap.py.
      ("fp32_cap20_m1", torch.float32, 20.0, 1.0, 1e-4, 1e-2, 1e-2),
      ("fp32_cap20_m0196", torch.float32, 20.0, 0.196116, 1e-4, 1e-2, 1e-2),
      ("bf16_cap20_m1", torch.bfloat16, 20.0, 1.0, 2e-3, 5e-2, 5e-2),
  )
  def test_gradient_reduction_none_soft_cap(
      self, dtype, soft_cap, output_multiplier, loss_rtol, grad_rtol, grad_atol
  ):
    """Per-row loss and backward through the soft-capped kernel."""
    device = self.accelerator_device
    b, h, v = 1024, 128, 1024
    gen = torch.Generator().manual_seed(1234)
    # Scale so the logits reach the tanh's nonlinear range.
    x = (torch.randn(b, h, generator=gen) * 4.0).to(dtype)
    weights = (torch.randn(h, v, generator=gen) * 0.25).to(dtype)
    labels = torch.randint(0, v, (b,), generator=gen)
    dout = torch.randn(b, generator=gen)

    # FP32 reference on the same rounded operands, on CPU.
    x_ref = x.clone().requires_grad_(True)
    w_ref = weights.clone().requires_grad_(True)
    logits = x_ref.float() @ w_ref.float()
    logits = soft_cap * torch.tanh(logits * output_multiplier / soft_cap)
    expected = torch.nn.functional.cross_entropy(
        logits, labels, reduction="none"
    )
    (expected * dout).sum().backward()

    x_tpu = x.to(device).requires_grad_(True)
    w_tpu = weights.to(device).requires_grad_(True)
    actual = (
        linear_softmax_cross_entropy_loss.linear_softmax_cross_entropy_loss(
            x_tpu,
            labels.to(device),
            w_tpu,
            reduction="none",
            implementation="mosaic_tpu",
            b_block_size=128,
            h_block_size=128,
            v_block_size=128,
            soft_cap=soft_cap,
            output_multiplier=output_multiplier,
        )
    )
    self.assertEqual(tuple(actual.shape), (b,))
    (actual * dout.to(device)).sum().backward()

    torch.testing.assert_close(
        actual.detach().cpu().float(),
        expected.detach(),
        rtol=loss_rtol,
        atol=1e-3,
    )
    for actual_grad, expected_grad in (
        (x_tpu.grad, x_ref.grad),
        (w_tpu.grad, w_ref.grad),
    ):
      torch.testing.assert_close(
          actual_grad.cpu().float(),
          expected_grad.float(),
          rtol=grad_rtol,
          atol=grad_atol,
      )

  def test_single_hidden_block_non_pow2_vocab_tile_640(self):
    """Tests single-H-block Mosaic LCE with non-power-of-2 v_block_size=640."""
    device = self.accelerator_device
    if device.type != "tpu":
      self.skipTest("Mosaic TPU kernel test requires a TPU device.")
    from jax.experimental.pallas import tpu as pltpu  # pylint: disable=g-import-not-at-top

    # Tokamax sets the kernel's VMEM limit (90% of capacity) itself, so the
    # exact recipe tile runs on any chip with >= 64 MiB VMEM (on TPU7x via the
    # `buffer_count=1` fallback).
    vmem_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    if vmem_bytes >= 64 * 1024 * 1024:
      # Exact Qwen3.5 35B benchmark tile: (1024, 2048, 640) at H=2048.
      b, h, v = 2048, 2048, 1280
      b_blk, h_blk, v_blk = 1024, 2048, 640
    else:
      # Fits 16-32 MiB default scoped VMEM while exercising num_h_blocks == 1
      # and non-power-of-2 v_block_size == 640.
      b, h, v = 512, 512, 1280
      b_blk, h_blk, v_blk = 256, 512, 640

    torch.manual_seed(0)
    x = torch.randn(b, h, dtype=torch.bfloat16, device=device)
    weights = torch.randn(h, v, dtype=torch.bfloat16, device=device) * (h**-0.5)
    labels = torch.randint(0, v, (b,), dtype=torch.int64, device=device)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weights.detach().clone().requires_grad_(True)
    ref_out = torch.nn.functional.cross_entropy(
        (x_ref @ w_ref).float(), labels, reduction="sum"
    )
    ref_out.backward()

    x_act = x.detach().clone().requires_grad_(True)
    w_act = weights.detach().clone().requires_grad_(True)
    act_out = (
        linear_softmax_cross_entropy_loss.linear_softmax_cross_entropy_loss(
            x_act,
            labels,
            w_act,
            reduction="sum",
            implementation="mosaic_tpu",
            b_block_size=b_blk,
            h_block_size=h_blk,
            v_block_size=v_blk,
        )
    )
    act_out.backward()

    torch.testing.assert_close(
        act_out.cpu(),
        ref_out.detach().cpu().view_as(act_out.cpu()),
        rtol=1.6e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        x_act.grad.cpu(), x_ref.grad.cpu(), rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(
        w_act.grad.cpu(), w_ref.grad.cpu(), rtol=5e-2, atol=0.5
    )


if __name__ == "__main__":
  absltest.main()
