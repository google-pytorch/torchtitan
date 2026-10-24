"""Accelerator tests for the Pallas depthwise causal conv1d + SiLU kernels."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import causal_conv1d


def _reference(x, weight):
  """``silu(conv1d)`` with left zero padding, in the inputs' dtype."""
  width = weight.shape[-1]
  y = F.conv1d(
      F.pad(x.transpose(1, 2), (width - 1, 0)), weight, groups=x.shape[-1]
  )
  return F.silu(y.transpose(1, 2))


def _relative_l2_error(actual, expected):
  return float((actual - expected).norm() / expected.norm())


class CausalConv1dTest(base_device_test.BaseAcceleratorDeviceTest):

  @parameterized.named_parameters(
      dict(testcase_name="_one_block", batch=2, seq_len=64, channels=128),
      # Several channel blocks, a batch of 3 (dw accumulates across it) and a
      # sequence that is not a power of 2.
      dict(testcase_name="_blocks", batch=3, seq_len=1032, channels=384),
  )
  def test_matches_conv1d_forward_and_backward(self, batch, seq_len, channels):
    torch.manual_seed(0)
    x = torch.randn(batch, seq_len, channels, dtype=torch.float64)
    weight = torch.randn(channels, 1, 4, dtype=torch.float64) * 0.5
    grad = torch.randn(batch, seq_len, channels, dtype=torch.float64)
    x_ref = x.clone().requires_grad_()
    w_ref = weight.clone().requires_grad_()
    out_ref = _reference(x_ref, w_ref)
    grads_ref = torch.autograd.grad(out_ref, (x_ref, w_ref), grad)

    dev = self.accelerator_device
    x_dev = x.to(dev, torch.bfloat16).requires_grad_()
    w_dev = weight.to(dev, torch.bfloat16).requires_grad_()
    self.assertTrue(causal_conv1d.supports(x_dev))
    out = causal_conv1d.causal_conv_silu(x_dev, w_dev)
    grads = torch.autograd.grad(
        out, (x_dev, w_dev), grad.to(dev, torch.bfloat16)
    )

    self.assertEqual(out.dtype, torch.bfloat16)
    self.assertLess(_relative_l2_error(out.double().cpu(), out_ref), 1e-2)
    for name, actual, expected in zip(("x", "weight"), grads, grads_ref):
      self.assertEqual(actual.dtype, torch.bfloat16, name)
      self.assertLess(
          _relative_l2_error(actual.double().cpu(), expected), 1e-2, name
      )

  def test_supports(self):
    dev = self.accelerator_device
    self.assertFalse(causal_conv1d.supports(torch.zeros(1, 8, 96, device=dev)))
    self.assertFalse(causal_conv1d.supports(torch.zeros(1, 6, 128, device=dev)))
    self.assertFalse(causal_conv1d.supports(torch.zeros(1, 8, 128)))


if __name__ == "__main__":
  absltest.main()
