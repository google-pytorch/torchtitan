"""Accelerator tests for the Pallas gated RMSNorm kernels."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.distributed.tensor import DTensor
import torch.nn.functional as F
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import rms_norm_gated


def _reference(x, gate, weight, eps):
  """``qwen3_5.gdn.RMSNormGated.forward``."""
  dtype = x.dtype
  xf = x.float()
  xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
  out = (weight.float() * xf).to(dtype)
  return (out * F.silu(gate.float())).to(dtype)


def _reference_fp64(x, gate, weight, eps):
  """FP64 reference on exact input values (isolates kernel rounding)."""
  xd = x.double()
  xd = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)
  return weight.double() * xd * F.silu(gate.double())


def _relative_l2_error(actual, expected):
  return float(
      (actual.double() - expected.double()).norm() / expected.double().norm()
  )


class RmsNormGatedTest(
    base_device_test.BaseAcceleratorDeviceTest, parameterized.TestCase
):

  def _require_tpu(self) -> None:
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("Pallas rms_norm_gated requires TPU.")

  @parameterized.named_parameters(
      dict(
          testcase_name="_all_bf16",
          shape=(4096, 4, 128),
          x_dtype=torch.bfloat16,
          gate_dtype=torch.bfloat16,
          w_dtype=torch.bfloat16,
          tol=1e-2,
      ),
      dict(
          testcase_name="_bf16_fp32_weight",
          shape=(4096, 4, 128),
          x_dtype=torch.bfloat16,
          gate_dtype=torch.bfloat16,
          w_dtype=torch.float32,
          tol=1e-2,
      ),
      dict(
          testcase_name="_bf16_fp32_gate",
          shape=(2048, 2, 128),
          x_dtype=torch.bfloat16,
          gate_dtype=torch.float32,
          w_dtype=torch.float32,
          tol=1e-2,
      ),
      dict(
          testcase_name="_all_fp32_dim256",
          shape=(4096, 2, 256),
          x_dtype=torch.float32,
          gate_dtype=torch.float32,
          w_dtype=torch.float32,
          tol=1e-5,
      ),
  )
  def test_matches_reference_forward_and_backward(
      self,
      shape: tuple[int, ...],
      x_dtype: torch.dtype,
      gate_dtype: torch.dtype,
      w_dtype: torch.dtype,
      tol: float,
  ):
    self._require_tpu()
    torch.manual_seed(0)
    dim = shape[-1]
    x = torch.randn(*shape, dtype=x_dtype)
    gate = torch.randn(*shape, dtype=gate_dtype)
    weight = (1.0 + 0.1 * torch.randn(dim)).to(w_dtype)
    grad = torch.randn(*shape, dtype=x_dtype)

    inputs_ref = [t.clone().requires_grad_() for t in (x, gate, weight)]
    out_ref = _reference(*inputs_ref, 1e-6)
    grads_ref = torch.autograd.grad(out_ref, inputs_ref, grad)

    dev = self.accelerator_device
    x_dev = x.to(dev).requires_grad_()
    gate_dev = gate.to(dev).requires_grad_()
    weight_dev = weight.to(dev).requires_grad_()
    self.assertTrue(rms_norm_gated.supports(x_dev, gate_dev, weight_dev))
    out = rms_norm_gated.rms_norm_gated(x_dev, gate_dev, weight_dev, 1e-6)
    grads = torch.autograd.grad(
        out, (x_dev, gate_dev, weight_dev), grad.to(dev)
    )

    self.assertEqual(out.dtype, x_dtype)
    self.assertLess(_relative_l2_error(out.cpu(), out_ref), tol)
    for name, actual, expected in zip(
        ("x", "gate", "weight"), grads, grads_ref, strict=True
    ):
      self.assertEqual(actual.shape, expected.shape, name)
      self.assertEqual(actual.dtype, expected.dtype, name)
      self.assertLess(_relative_l2_error(actual.cpu(), expected), tol, name)

  def test_large_row_biased_bf16_weight_gradient_accumulates_in_fp32(self):
    self._require_tpu()
    torch.manual_seed(7)
    shape = (65536, 1, 128)
    x = (0.3 + torch.randn(*shape)).to(torch.bfloat16)
    gate = (0.5 + torch.randn(*shape)).to(torch.bfloat16)
    weight = (1.0 + 0.2 * torch.randn(128)).to(torch.float32)
    grad = (0.2 + 0.5 * torch.randn(*shape)).to(torch.bfloat16)

    x_ref = x.clone().double().requires_grad_()
    gate_ref = gate.clone().double().requires_grad_()
    w_ref = weight.clone().double().requires_grad_()
    out_ref = _reference_fp64(x_ref, gate_ref, w_ref, 1e-6)
    _, _, dw_ref = torch.autograd.grad(
        out_ref, (x_ref, gate_ref, w_ref), grad.double()
    )

    dev = self.accelerator_device
    x_dev = x.to(dev).requires_grad_()
    gate_dev = gate.to(dev).requires_grad_()
    w_dev = weight.to(dev).requires_grad_()
    out = rms_norm_gated.rms_norm_gated(x_dev, gate_dev, w_dev, 1e-6)
    _, _, dw = torch.autograd.grad(out, (x_dev, gate_dev, w_dev), grad.to(dev))
    self.assertLess(_relative_l2_error(dw.cpu(), dw_ref), 2e-3)

  def test_non_contiguous_partial_grads_and_fullgraph_compile(self):
    self._require_tpu()
    self.addCleanup(torch._dynamo.reset)  # pylint: disable=protected-access
    torch.manual_seed(11)
    dev = self.accelerator_device
    full = torch.randn(2048, 2, 256, device=dev, dtype=torch.bfloat16)
    x_nc = full[:, :, :128].detach().requires_grad_(True)
    gate_nc = full[:, :, 128:].detach().requires_grad_(False)
    w = torch.ones(128, device=dev, dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(2048, 2, 128, device=dev, dtype=torch.bfloat16)
    self.assertFalse(x_nc.is_contiguous())

    out = rms_norm_gated.rms_norm_gated(x_nc, gate_nc, w, 1e-6)
    dx, dw = torch.autograd.grad(out, (x_nc, w), dy)
    self.assertEqual(dx.shape, x_nc.shape)
    self.assertEqual(dw.shape, w.shape)

    def step(x_in, g_in, w_in, dy_in):
      y = rms_norm_gated.rms_norm_gated(x_in, g_in, w_in, 1e-6)
      return torch.autograd.grad(y, (x_in, g_in, w_in), dy_in)

    compiled = torch.compile(step, fullgraph=True, backend="tpu")
    x_c = x_nc.detach().contiguous().requires_grad_(True)
    g_c = gate_nc.detach().contiguous().requires_grad_(True)
    w_c = w.detach().requires_grad_(True)
    grads_c = compiled(x_c, g_c, w_c, dy)
    grads_e = step(x_c, g_c, w_c, dy)
    for g_comp, g_eager in zip(grads_c, grads_e, strict=True):
      torch.testing.assert_close(g_comp.cpu(), g_eager.cpu())

  def test_supports_and_validation(self):
    x_cpu = torch.zeros(2048, 128)
    w_cpu = torch.ones(128)
    self.assertFalse(rms_norm_gated.supports(x_cpu, x_cpu, w_cpu))
    with self.assertRaisesRegex(ValueError, "Unsupported inputs"):
      rms_norm_gated.rms_norm_gated(x_cpu, x_cpu, w_cpu, 1e-6)

    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      return
    dev = self.accelerator_device
    x_ok = torch.zeros(2048, 128, device=dev, dtype=torch.bfloat16)
    w_ok = torch.ones(128, device=dev, dtype=torch.bfloat16)
    self.assertTrue(rms_norm_gated.supports(x_ok, x_ok, w_ok))

    bad_cases = [
        (torch.zeros(100, 128, device=dev, dtype=torch.bfloat16), None, w_ok),
        (torch.zeros(0, 128, device=dev, dtype=torch.bfloat16), None, w_ok),
        (
            torch.zeros(2048, 96, device=dev, dtype=torch.bfloat16),
            None,
            torch.ones(96, device=dev, dtype=torch.bfloat16),
        ),
        (
            torch.zeros(2048, 1024, device=dev, dtype=torch.bfloat16),
            None,
            torch.ones(1024, device=dev, dtype=torch.bfloat16),
        ),
        (
            x_ok,
            torch.zeros(2048, 256, device=dev, dtype=torch.bfloat16),
            w_ok,
        ),
        (x_ok, x_ok, torch.ones(256, device=dev, dtype=torch.bfloat16)),
        (
            torch.zeros(2048, 128, device=dev, dtype=torch.float16),
            None,
            torch.ones(128, device=dev, dtype=torch.float16),
        ),
    ]
    for x_bad, g_bad, w_bad in bad_cases:
      g_in = x_bad if g_bad is None else g_bad
      self.assertFalse(rms_norm_gated.supports(x_bad, g_in, w_bad))
      with self.assertRaisesRegex(ValueError, "Unsupported inputs"):
        rms_norm_gated.rms_norm_gated(x_bad, g_in, w_bad, 1e-6)

    with self.assertRaisesRegex(ValueError, "eps must be non-negative"):
      rms_norm_gated.rms_norm_gated(x_ok, x_ok, w_ok, -1e-6)

    fake_dtensor = mock.MagicMock(spec=DTensor)
    self.assertFalse(rms_norm_gated.supports(fake_dtensor, x_ok, w_ok))
    self.assertFalse(rms_norm_gated.supports(x_ok, fake_dtensor, w_ok))
    self.assertFalse(rms_norm_gated.supports(x_ok, x_ok, fake_dtensor))


if __name__ == "__main__":
  absltest.main()
