"""Tests for linear_softmax_cross_entropy_loss."""

import functools
import importlib
import inspect
import os
from types import SimpleNamespace
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import torch
from torchtitan.components import loss
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import linear_softmax_cross_entropy_loss


class LinearSoftmaxCrossEntropyLossUnitTest(absltest.TestCase):

  def test_backward_normalizes_singleton_cotangent(self):
    dout = jnp.ones((1,), dtype=jnp.float32)
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
          dout,
          lse,
          x,
          labels,
          weights,
          "sum",
          "mosaic_tpu",
          1,
          1,
          1,
      )

    self.assertEqual(backward.call_args.args[0].shape, ())

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
    if (
        not hasattr(kernel, "_calculate_bwd_vmem_bytes")
        or "buffer_count" not in inspect.signature(orig_bwd).parameters
    ):
      self.skipTest("Installed tokamax does not expose buffer_count.")
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
    ):
      # pylint: disable=protected-access
      linear_softmax_cross_entropy_loss._my_bwd(
          dout, lse, x, labels, weights, "sum", "mosaic_tpu", 1024, 2048, 640
      )
      self.assertEqual(calls[-1][1].get("buffer_count"), 1)
      linear_softmax_cross_entropy_loss._my_bwd(
          dout[:1],
          lse[:128],
          x[:128, :128],
          labels[:128],
          weights[:128, :128],
          "sum",
          "mosaic_tpu",
          128,
          128,
          128,
      )
      self.assertNotIn("buffer_count", calls[-1][1])


class LinearSoftmaxCrossEntropyLossTest(
    base_device_test.BaseAcceleratorDeviceTest
):
  """Tests for LinearSoftmaxCrossEntropyLoss."""

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

    # Reference Math on TPU
    logits = x_tpu @ weights_tpu
    expected_out = torch.nn.functional.cross_entropy(
        logits.float(),
        labels_tpu,
        reduction=reduction,
    )
    expected_out.backward()
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
    actual_out.backward()
    actual_grad_x = x_tpu.grad
    actual_grad_weights = weights_tpu.grad

    # Compare

    self.assertEqual(actual_grad_x.dtype, x_tpu.dtype)
    self.assertEqual(actual_grad_weights.dtype, weights_tpu.dtype)

    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=loss_rtol, atol=loss_atol
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

  def test_single_hidden_block_non_pow2_vocab_tile_640(self):
    """Tests single-H-block Mosaic LCE with non-power-of-2 v_block_size=640."""
    device = self.accelerator_device
    if device.type != "tpu":
      self.skipTest("Mosaic TPU kernel test requires a TPU device.")
    from jax.experimental.pallas import tpu as pltpu  # pylint: disable=g-import-not-at-top

    vmem_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    has_64m_scoped = "65472" in os.environ.get("LIBTPU_INIT_ARGS", "")
    if vmem_bytes >= 64 * 1024 * 1024 and has_64m_scoped:
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
