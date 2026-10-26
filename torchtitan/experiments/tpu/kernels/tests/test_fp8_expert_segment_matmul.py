"""Accelerator tests for the FP8 expert grouped matmuls."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import fp8_expert_segment_matmul as fp8_esm

_TILING = (128, 0, 0)
_GROUP_SIZES = (0, 103, 17, 136, 0, 256)


def _dequantized(x: torch.Tensor, dims, qdtype=torch.float8_e4m3fn):
    """``x`` through absmax FP8 quantization over ``dims``, in FP32 on CPU."""
    x = x.detach().float().cpu()
    amax = x.abs().amax(dim=dims, keepdim=True)
    scale = (
        torch.where(amax > 0, amax, torch.ones_like(amax))
        / torch.finfo(qdtype).max
    )
    return (x / scale).to(qdtype).float() * scale


def _grouped_reference(lhs, rhs, group_sizes):
    """FP32 ``lhs[rows of g] @ rhs[g]`` for every group ``g``."""
    out, start = [], 0
    for g, size in enumerate(group_sizes):
        out.append(lhs[start : start + size] @ rhs[g])
        start += size
    return torch.cat(out)


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.detach().float().cpu(), want.detach().float().cpu()
    return ((got - want).norm() / want.norm()).item()


def _to_torch(x: jax.Array) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x.astype(jnp.float32)))


class Fp8GmmTest(base_device_test.BaseAcceleratorDeviceTest):
    """The JAX-level GEMMs, run by JAX directly."""

    def _operands(self, k=256, n=256):
        keys = jax.random.split(jax.random.key(0), 3)
        m = sum(_GROUP_SIZES)
        lhs = jax.random.normal(keys[0], (m, k), jnp.bfloat16)
        rhs = jax.random.normal(
            keys[1], (len(_GROUP_SIZES), k, n), jnp.bfloat16
        ) * jnp.bfloat16(0.05)
        dout = jax.random.normal(keys[2], (m, n), jnp.bfloat16)
        sizes = jnp.array(_GROUP_SIZES, jnp.int32)
        return lhs, rhs, dout, sizes

    def test_fwd_matches_dequantized_reference(self):
        lhs, rhs, _, sizes = self._operands()
        out = fp8_esm.fp8_gmm_fwd(lhs, rhs, sizes, _TILING)

        want = _grouped_reference(
            _dequantized(_to_torch(lhs), dims=1),
            _dequantized(_to_torch(rhs), dims=(0, 1)),
            _GROUP_SIZES,
        )
        self.assertEqual(out.dtype, jnp.bfloat16)
        self.assertLess(_rel_err(_to_torch(out), want), 5e-3)

    def test_bwd_is_close_to_f32(self):
        lhs, rhs, dout, sizes = self._operands()
        dlhs, drhs = fp8_esm.fp8_gmm_bwd(
            dout, lhs, rhs, sizes, _TILING, _TILING
        )

        lhs_r = _to_torch(lhs).requires_grad_()
        rhs_r = _to_torch(rhs).requires_grad_()
        want = _grouped_reference(lhs_r, rhs_r, _GROUP_SIZES)
        want_dlhs, want_drhs = torch.autograd.grad(
            want, (lhs_r, rhs_r), _to_torch(dout)
        )
        self.assertEqual(drhs.dtype, jnp.bfloat16)
        self.assertLess(_rel_err(_to_torch(dlhs), want_dlhs), 0.1)
        self.assertLess(_rel_err(_to_torch(drhs), want_drhs), 0.1)
        # Experts that received no rows get exactly zero weight gradient.
        for g, size in enumerate(_GROUP_SIZES):
            if size == 0:
                self.assertEqual(float(jnp.abs(drhs[g]).max()), 0.0)


if __name__ == "__main__":
    absltest.main()
