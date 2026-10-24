"""Accelerator tests for the Pallas gated RMSNorm kernels."""

from absl.testing import absltest
import torch
import torch.nn.functional as F

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import rms_norm_gated


def _reference(x, gate, weight, eps):
  """``qwen3_5.gdn.RMSNormGated.forward``."""
  dtype = x.dtype
  xf = x.float()
  xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
  out = (weight.float() * xf).to(dtype)
  return (out * F.silu(gate.float())).to(dtype)


def _relative_l2_error(actual, expected):
  return float((actual - expected).norm() / expected.norm())


class RmsNormGatedTest(base_device_test.BaseAcceleratorDeviceTest):

  def test_matches_reference_forward_and_backward(self):
    torch.manual_seed(0)
    shape = (4096, 4, 128)
    x = torch.randn(*shape)
    gate = torch.randn(*shape)
    weight = 1 + 0.1 * torch.randn(128)
    grad = torch.randn(*shape)
    inputs_ref = [t.clone().requires_grad_() for t in (x, gate, weight)]
    out_ref = _reference(*inputs_ref, 1e-6)
    grads_ref = torch.autograd.grad(out_ref, inputs_ref, grad)

    dev = self.accelerator_device
    x_dev = x.to(dev, torch.bfloat16).requires_grad_()
    gate_dev = gate.to(dev, torch.bfloat16).requires_grad_()
    weight_dev = weight.to(dev).requires_grad_()
    self.assertTrue(rms_norm_gated.supports(x_dev))
    out = rms_norm_gated.rms_norm_gated(x_dev, gate_dev, weight_dev, 1e-6)
    grads = torch.autograd.grad(
        out, (x_dev, gate_dev, weight_dev), grad.to(dev, torch.bfloat16)
    )

    self.assertEqual(out.dtype, torch.bfloat16)
    self.assertLess(_relative_l2_error(out.float().cpu(), out_ref), 1e-2)
    names = ("x", "gate", "weight")
    for name, actual, expected in zip(names, grads, grads_ref):
      self.assertEqual(actual.shape, expected.shape, name)
      self.assertLess(
          _relative_l2_error(actual.float().cpu(), expected), 1e-2, name
      )

  def test_supports(self):
    dev = self.accelerator_device
    self.assertFalse(rms_norm_gated.supports(torch.zeros(100, 128, device=dev)))
    self.assertFalse(rms_norm_gated.supports(torch.zeros(2048, 96, device=dev)))
    self.assertFalse(rms_norm_gated.supports(torch.zeros(2048, 128)))


if __name__ == "__main__":
  absltest.main()
