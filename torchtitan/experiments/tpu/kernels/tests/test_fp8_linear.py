"""Accelerator tests for the TPU-native FP8 linear layer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import fp8_linear

_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.detach().float().cpu(), want.detach().float().cpu()
    return ((got - want).norm() / want.norm()).item()


def _row_dequantized(x: torch.Tensor, qdtype: torch.dtype) -> torch.Tensor:
    """``x`` through per-row absmax FP8 quantization, back in FP32 (on CPU)."""
    x = x.detach().float().cpu()
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = (
        torch.where(amax > 0, amax, torch.ones_like(amax))
        / torch.finfo(qdtype).max
    )
    return (x / scale).to(qdtype).float() * scale


def _to_torch(x: jax.Array) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x.astype(jnp.float32)))


class Fp8RowQuantizeTest(base_device_test.BaseAcceleratorDeviceTest):
    """The row-quantize Pallas kernel, run by JAX directly."""

    # Not a multiple of the 256-row tile, so the last tile is partial.
    _ROWS, _COLS = 300, 256

    def _x(self):
        keys = jax.random.split(jax.random.key(0), 2)
        x = jax.random.normal(keys[0], (self._ROWS, self._COLS), jnp.bfloat16)
        col_mult = jax.random.uniform(
            keys[1], (1, self._COLS), jnp.float32, 0.5, 2.0
        )
        return x, col_mult

    def test_matches_reference(self):
        x, col_mult = self._x()
        v = _to_torch(x) * _to_torch(col_mult)
        for jdtype, tdtype in (
            (jnp.float8_e4m3fn, torch.float8_e4m3fn),
            (jnp.float8_e5m2, torch.float8_e5m2),
        ):
            q, scale = fp8_linear.row_quantize(x, jdtype, col_mult=col_mult)
            self.assertEqual(q.dtype, jdtype)
            self.assertEqual(scale.shape, (self._ROWS, 1))
            want_scale = v.abs().amax(dim=-1, keepdim=True) / torch.finfo(
                tdtype
            ).max
            # TPU division may differ from CPU's in the last bit.
            torch.testing.assert_close(
                _to_torch(scale), want_scale, rtol=1e-6, atol=0
            )
            got = _to_torch(q) * _to_torch(scale)
            self.assertLess(_rel_err(got, _row_dequantized(v, tdtype)), 2e-2)

    def test_zero_and_tiny_rows_stay_finite(self):
        x, _ = self._x()
        x = x.at[3].set(0).at[5].multiply(1e-37)
        q, scale = fp8_linear.row_quantize(x, jnp.float8_e4m3fn)
        q, scale = _to_torch(q), _to_torch(scale)
        self.assertTrue(torch.isfinite(q).all())
        self.assertEqual(float(q[3].abs().max()), 0.0)
        # Both rows get the floored scale instead of a zero one.
        self.assertEqual(float(scale[3]), torch.finfo(torch.float32).tiny)
        self.assertEqual(float(scale[5]), torch.finfo(torch.float32).tiny)


class Fp8LinearTest(base_device_test.BaseAcceleratorDeviceTest):

    def _inputs(self, lead=(2, 96), k=256, n=384):
        torch.manual_seed(0)
        x = torch.randn(*lead, k, dtype=torch.bfloat16)
        w = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
        return x, w

    def test_forward_matches_dequantized_reference(self):
        x, w = self._inputs()
        device = self.accelerator_device
        y = fp8_linear.fp8_linear(x.to(device), w.to(device)).cpu()

        # The kernel multiplies exact FP8 values and applies both scales to
        # the FP32 accumulator. XLA's FP32 -> FP8 convert may round a few
        # elements one FP8 ulp differently from torch's, so compare in norm.
        want = (
            _row_dequantized(x, torch.float8_e4m3fn)
            @ _row_dequantized(w, torch.float8_e4m3fn).T
        )
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertEqual(y.shape, (2, 96, 384))
        self.assertLess(_rel_err(y, want), 5e-3)

    def test_forward_is_close_to_bf16(self):
        x, w = self._inputs()
        device = self.accelerator_device
        y = fp8_linear.fp8_linear(x.to(device), w.to(device)).cpu().float()
        want = x.float() @ w.float().T
        rel = (y - want).norm() / want.norm()
        self.assertLess(rel.item(), 0.05)

    def test_all_zero_rows_stay_zero(self):
        x, w = self._inputs(lead=(64,))
        x[3] = 0
        device = self.accelerator_device
        y = fp8_linear.fp8_linear(x.to(device), w.to(device)).cpu()
        self.assertTrue(torch.isfinite(y.float()).all())
        torch.testing.assert_close(y[3], torch.zeros_like(y[3]))

    def test_tiny_rows_stay_finite(self):
        # amax / 448 of a 1e-37-scaled row is subnormal in FP32; TPUs flush it
        # to zero, so without a scale floor the row would quantize to NaN.
        x, w = self._inputs(lead=(64,))
        x[3] = x[3] * 1e-37
        device = self.accelerator_device
        x_d, w_d = x.to(device).requires_grad_(), w.to(device)
        y = fp8_linear.fp8_linear(x_d, w_d)
        dy = torch.randn_like(y) * 1e-30
        (dx,) = torch.autograd.grad(y, (x_d,), dy)
        self.assertTrue(torch.isfinite(y.float()).all())
        self.assertTrue(torch.isfinite(dx.float()).all())
        self.assertLess(y[3].float().abs().max().item(), 1e-30)

    def test_forward_only_backward_is_bf16(self):
        x, w = self._inputs(lead=(128,))
        device = self.accelerator_device
        dy = torch.randn(128, 384, dtype=torch.bfloat16)

        x_dev = x.to(device).requires_grad_()
        w_dev = w.to(device).requires_grad_()
        y = fp8_linear.fp8_linear(x_dev, w_dev, fp8_grad_input=False)
        dx, dw = torch.autograd.grad(y, (x_dev, w_dev), dy.to(device))

        torch.testing.assert_close(
            dx.cpu().float(), dy.float() @ w.float(), rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            dw.cpu().float(), dy.float().T @ x.float(), rtol=1e-2, atol=5e-2
        )

    @parameterized.parameters(False, True)
    def test_fp8_grad_input(self, fp8_grad_weight):
        x, w = self._inputs(lead=(128,))
        device = self.accelerator_device
        dy = torch.randn(128, 384, dtype=torch.bfloat16)

        x_dev = x.to(device).requires_grad_()
        w_dev = w.to(device).requires_grad_()
        y = fp8_linear.fp8_linear(x_dev, w_dev, fp8_grad_weight=fp8_grad_weight)
        dx, dw = torch.autograd.grad(y, (x_dev, w_dev), dy.to(device))

        # dx = dy @ w with w's channel scale folded into the e5m2 dy.
        w_amax = w.float().abs().amax(dim=1, keepdim=True)
        w_scale = w_amax / _E4M3_MAX
        w_q = (w.float() / w_scale).to(torch.float8_e4m3fn).float()
        want_dx = (
            _row_dequantized(dy.float() * w_scale.T, torch.float8_e5m2) @ w_q
        )
        self.assertLess(_rel_err(dx, want_dx), 5e-3)

        exact_dw = dy.float().T @ x.float()
        rel = (dw.cpu().float() - exact_dw).norm() / exact_dw.norm()
        if fp8_grad_weight:
            self.assertLess(rel.item(), 0.1)
        else:
            torch.testing.assert_close(
                dw.cpu().float(), exact_dw, rtol=1e-2, atol=5e-2
            )


if __name__ == "__main__":
    absltest.main()
