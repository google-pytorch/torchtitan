"""Accelerator tests for the Pallas depthwise causal conv1d + SiLU kernels."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.distributed.tensor import DTensor
import torch.nn.functional as F
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import causal_conv1d


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
  """``silu(conv1d)`` with left zero padding, in the inputs' dtype."""
  width = weight.shape[-1]
  y = F.conv1d(
      F.pad(x.transpose(1, 2), (width - 1, 0)), weight, groups=x.shape[-1]
  )
  return F.silu(y.transpose(1, 2))


def _relative_l2_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
  denom = expected.norm().clamp_min(1e-8)
  return float((actual - expected).norm() / denom)


class CausalConv1dTest(base_device_test.BaseAcceleratorDeviceTest):

  def _require_tpu(self) -> torch.device:
    dev = self.accelerator_device
    if dev.type != "tpu":
      self.skipTest("Pallas causal_conv1d kernels are TPU-only.")
    return dev

  @parameterized.named_parameters(
      dict(
          testcase_name="_bf16_one_block",
          batch=2,
          seq_len=64,
          channels=128,
          width=4,
          dtype=torch.bfloat16,
      ),
      # Several channel blocks, a batch of 3 (dw accumulates across it) and a
      # sequence that is not a power of 2.
      dict(
          testcase_name="_bf16_blocks",
          batch=3,
          seq_len=1032,
          channels=384,
          width=4,
          dtype=torch.bfloat16,
      ),
      dict(
          testcase_name="_bf16_min_tile_w1",
          batch=1,
          seq_len=8,
          channels=128,
          width=1,
          dtype=torch.bfloat16,
      ),
      dict(
          testcase_name="_bf16_w3",
          batch=2,
          seq_len=16,
          channels=128,
          width=3,
          dtype=torch.bfloat16,
      ),
      dict(
          testcase_name="_fp32_exact",
          batch=2,
          seq_len=64,
          channels=256,
          width=4,
          dtype=torch.float32,
      ),
  )
  def test_matches_conv1d_forward_and_backward(
      self,
      batch: int,
      seq_len: int,
      channels: int,
      width: int,
      dtype: torch.dtype,
  ):
    dev = self._require_tpu()
    prev_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    self.addCleanup(torch.set_float32_matmul_precision, prev_prec)

    torch.manual_seed(0)
    x = torch.randn(batch, seq_len, channels, dtype=torch.float64)
    weight = torch.randn(channels, 1, width, dtype=torch.float64) * 0.5
    grad = torch.randn(batch, seq_len, channels, dtype=torch.float64)
    x_ref = x.clone().requires_grad_()
    w_ref = weight.clone().requires_grad_()
    out_ref = _reference(x_ref, w_ref)
    grads_ref = torch.autograd.grad(out_ref, (x_ref, w_ref), grad)

    x_dev = x.to(dev, dtype).requires_grad_()
    w_dev = weight.to(dev, dtype).requires_grad_()
    self.assertTrue(causal_conv1d.supports(x_dev, w_dev))
    out = causal_conv1d.causal_conv_silu(x_dev, w_dev)
    grads = torch.autograd.grad(out, (x_dev, w_dev), grad.to(dev, dtype))

    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    self.assertEqual(out.dtype, dtype)
    self.assertLess(_relative_l2_error(out.double().cpu(), out_ref), tol)
    for name, actual, expected in zip(
        ("x", "weight"), grads, grads_ref, strict=True
    ):
      self.assertEqual(actual.dtype, dtype, name)
      self.assertLess(
          _relative_l2_error(actual.double().cpu(), expected), tol, name
      )

  def test_bf16_biased_dw_accumulation_and_mixed_fp32_weight(self):
    """Tests fp32 dw accumulation over 16K biased tokens and fp32 weights."""
    dev = self._require_tpu()
    torch.manual_seed(1)
    batch, seq_len, channels, width = 8, 2048, 128, 4
    x = (torch.randn(batch, seq_len, channels) + 0.5).bfloat16()
    weight = (0.5 * torch.randn(channels, 1, width)).float()
    grad = (torch.randn(batch, seq_len, channels) + 0.25).bfloat16()

    x_ref = x.double().requires_grad_()
    w_ref = weight.double().requires_grad_()
    out_ref = _reference(x_ref, w_ref)
    dx_ref, dw_ref = torch.autograd.grad(out_ref, (x_ref, w_ref), grad.double())

    x_dev = x.to(dev).requires_grad_()
    w_dev = weight.to(dev).requires_grad_()
    self.assertTrue(causal_conv1d.supports(x_dev, w_dev))
    out = causal_conv1d.causal_conv_silu(x_dev, w_dev)
    dx, dw = torch.autograd.grad(out, (x_dev, w_dev), grad.to(dev))
    self.assertEqual(dx.dtype, torch.bfloat16)
    self.assertEqual(dw.dtype, torch.float32)
    self.assertLess(_relative_l2_error(dx.double().cpu(), dx_ref), 1e-2)
    self.assertLess(_relative_l2_error(dw.double().cpu(), dw_ref), 2e-3)

  def test_ctx8k_strip_on_tpu7x(self):
    """Tests S=8192 bf16 strips on TPU7x (skipped on smaller-VMEM TPUs)."""
    dev = self._require_tpu()
    torch.manual_seed(2)
    x_dev = torch.randn(2, 8192, 128, device=dev, dtype=torch.bfloat16)
    w_dev = torch.randn(128, 1, 4, device=dev, dtype=torch.bfloat16) * 0.5
    if not causal_conv1d.supports(x_dev, w_dev):
      self.skipTest("S=8192 bf16 causal_conv1d requires TPU >= v7 VMEM.")
    x_ref = x_dev.double().cpu().requires_grad_()
    w_ref = w_dev.double().cpu().requires_grad_()
    grad = torch.randn(2, 8192, 128, dtype=torch.bfloat16)
    out_ref = _reference(x_ref, w_ref)
    dx_ref, dw_ref = torch.autograd.grad(out_ref, (x_ref, w_ref), grad.double())

    x_in = x_dev.detach().requires_grad_()
    w_in = w_dev.detach().requires_grad_()
    out = causal_conv1d.causal_conv_silu(x_in, w_in)
    dx, dw = torch.autograd.grad(out, (x_in, w_in), grad.to(dev))
    self.assertLess(_relative_l2_error(out.double().cpu(), out_ref), 1e-2)
    self.assertLess(_relative_l2_error(dx.double().cpu(), dx_ref), 1e-2)
    self.assertLess(_relative_l2_error(dw.double().cpu(), dw_ref), 2e-3)

  def test_noncontiguous_partial_grads_and_fullgraph_compile(self):
    """Tests non-contiguous views, single-leaf grads, and torch.compile."""
    dev = self._require_tpu()
    self.addCleanup(torch._dynamo.reset)  # pylint: disable=protected-access
    torch.manual_seed(3)
    base_x = torch.randn(2, 32, 256, device=dev, dtype=torch.bfloat16)
    x_slice = base_x[:, :16, :128]
    self.assertFalse(x_slice.is_contiguous())
    weight = torch.randn(128, 1, 4, device=dev, dtype=torch.bfloat16) * 0.5
    grad = torch.randn(2, 16, 128, device=dev, dtype=torch.bfloat16)

    for x_req in (True, False):
      x_in = x_slice.detach().requires_grad_(x_req)
      w_in = weight.detach().requires_grad_(not x_req)
      leaf = x_in if x_req else w_in
      (g_actual,) = torch.autograd.grad(
          causal_conv1d.causal_conv_silu(x_in, w_in), leaf, grad
      )
      x_ref = x_slice.detach().double().cpu().requires_grad_(x_req)
      w_ref = weight.detach().double().cpu().requires_grad_(not x_req)
      leaf_ref = x_ref if x_req else w_ref
      (g_ref,) = torch.autograd.grad(
          _reference(x_ref, w_ref), leaf_ref, grad.double().cpu()
      )
      self.assertLess(_relative_l2_error(g_actual.double().cpu(), g_ref), 1e-2)

    compiled = torch.compile(
        causal_conv1d.causal_conv_silu, backend="aot_eager", fullgraph=True
    )
    x_c = x_slice.detach().requires_grad_()
    w_c = weight.detach().requires_grad_()
    out_c = compiled(x_c, w_c)
    out_e = causal_conv1d.causal_conv_silu(x_c, w_c)
    torch.testing.assert_close(out_c.cpu(), out_e.cpu())

  def test_supports_and_validation(self):
    cpu_x = torch.zeros(1, 8, 128, dtype=torch.bfloat16)
    cpu_w = torch.zeros(128, 1, 4, dtype=torch.bfloat16)
    self.assertFalse(causal_conv1d.supports(cpu_x, cpu_w))
    with self.assertRaisesRegex(ValueError, "Unsupported inputs"):
      causal_conv1d.causal_conv_silu(cpu_x, cpu_w)

    dev = self.accelerator_device
    w_ok = torch.zeros(128, 1, 4, device=dev, dtype=torch.bfloat16)
    for bad_x, bad_w in (
        (torch.zeros(1, 8, 96, device=dev, dtype=torch.bfloat16), w_ok[:96]),
        (torch.zeros(1, 6, 128, device=dev, dtype=torch.bfloat16), w_ok),
        (torch.zeros(0, 8, 128, device=dev, dtype=torch.bfloat16), w_ok),
        (torch.zeros(1, 0, 128, device=dev, dtype=torch.bfloat16), w_ok),
        (torch.zeros(1, 16384, 128, device=dev, dtype=torch.bfloat16), w_ok),
        (torch.zeros(1, 8, 128, device=dev, dtype=torch.float64), w_ok),
        (
            torch.zeros(1, 8, 128, device=dev, dtype=torch.bfloat16),
            torch.zeros(128, 2, 4, device=dev, dtype=torch.bfloat16),
        ),
        (
            torch.zeros(1, 8, 128, device=dev, dtype=torch.bfloat16),
            torch.zeros(128, 1, 9, device=dev, dtype=torch.bfloat16),
        ),
    ):
      self.assertFalse(causal_conv1d.supports(bad_x, bad_w))

    fake_dtensor = mock.MagicMock(spec=DTensor)
    self.assertFalse(causal_conv1d.supports(fake_dtensor, w_ok))
    self.assertFalse(causal_conv1d.supports(cpu_x.to(dev), fake_dtensor))

    with mock.patch.object(causal_conv1d, "_tpu_generation", return_value=5):
      self.assertEqual(
          causal_conv1d._max_supported_seq_len(torch.bfloat16),  # pylint: disable=protected-access
          2048,
      )
      self.assertEqual(
          causal_conv1d._max_supported_seq_len(torch.float32),  # pylint: disable=protected-access
          1024,
      )
    with mock.patch.object(causal_conv1d, "_tpu_generation", return_value=7):
      self.assertEqual(
          causal_conv1d._max_supported_seq_len(torch.bfloat16),  # pylint: disable=protected-access
          8192,
      )
      self.assertEqual(
          causal_conv1d._max_supported_seq_len(torch.float32),  # pylint: disable=protected-access
          4096,
      )


if __name__ == "__main__":
  absltest.main()
