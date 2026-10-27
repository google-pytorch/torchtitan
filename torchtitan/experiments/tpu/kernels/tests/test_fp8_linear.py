"""Accelerator tests for the TPU-native FP8 linear layer."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import fp8_linear

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

if __name__ == "__main__":
    absltest.main()
