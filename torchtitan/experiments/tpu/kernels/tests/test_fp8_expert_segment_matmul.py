"""Accelerator tests for the FP8 expert grouped matmuls."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import fp8_expert_segment_matmul as fp8_esm

_TILING = (128, 0, 0)
_TILINGS = dict(fwd_tiling=_TILING, dlhs_tiling=_TILING, drhs_tiling=_TILING)
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


class Fp8SwigluKernelTest(base_device_test.BaseAcceleratorDeviceTest):
    """The SwiGLU Pallas kernels, run by JAX directly."""

    # Not a multiple of the 256-row tile, so the last tile is partial.
    _ROWS, _HIDDEN = 600, 256

    def _gate_up(self):
        key = jax.random.key(2)
        return jax.random.normal(
            key, (self._ROWS, 2 * self._HIDDEN), jnp.bfloat16
        )

    def test_swiglu_quantize_matches_xla(self):
        gate_up = self._gate_up()
        q, scale = fp8_esm.swiglu_quantize(gate_up, fused=True)
        want_q, want_scale = fp8_esm.swiglu_quantize(gate_up, fused=False)
        self.assertEqual(q.dtype, jnp.float8_e4m3fn)
        self.assertEqual(scale.shape, (self._ROWS, 1))
        np.testing.assert_allclose(
            np.asarray(scale), np.asarray(want_scale), rtol=1e-6
        )
        # The FP32 -> FP8 rounding may differ by one FP8 ulp per element.
        got = _to_torch(q) * _to_torch(scale)
        want = _to_torch(want_q) * _to_torch(want_scale)
        self.assertLess(_rel_err(got, want), 1e-2)

        gate, up = _to_torch(gate_up).chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        self.assertLess(_rel_err(got, _dequantized(hidden, dims=1)), 1e-2)

    def test_swiglu_grad_matches_autodiff(self):
        gate_up = self._gate_up()
        d_hidden = jax.random.normal(
            jax.random.key(3), (self._ROWS, self._HIDDEN), jnp.bfloat16
        )
        got = fp8_esm.swiglu_grad(gate_up, d_hidden, fused=True)
        unfused = fp8_esm.swiglu_grad(gate_up, d_hidden, fused=False)
        self.assertEqual(got.dtype, jnp.bfloat16)
        self.assertEqual(got.shape, gate_up.shape)
        np.testing.assert_array_equal(
            np.asarray(got.astype(jnp.float32)),
            np.asarray(unfused.astype(jnp.float32)),
        )

        gu_r = _to_torch(gate_up).requires_grad_()
        g_r, u_r = gu_r.chunk(2, dim=-1)
        (want,) = torch.autograd.grad(
            F.silu(g_r) * u_r, (gu_r,), _to_torch(d_hidden)
        )
        self.assertLess(_rel_err(_to_torch(got), want), 1e-2)


class Fp8ExpertSegmentMatmulTest(base_device_test.BaseAcceleratorDeviceTest):

    def _operands(self, k=256, n=256, group_sizes=_GROUP_SIZES):
        torch.manual_seed(0)
        m = sum(group_sizes)
        lhs = torch.randn(m, k, dtype=torch.bfloat16)
        rhs = torch.randn(len(group_sizes), k, n, dtype=torch.bfloat16) * 0.05
        dout = torch.randn(m, n, dtype=torch.bfloat16)
        sizes = torch.tensor(group_sizes, dtype=torch.int32)
        return lhs, rhs, dout, sizes

    def _to_dev(self, *tensors, grad=False):
        return [
            t.to(self.accelerator_device).requires_grad_(grad) for t in tensors
        ]

    def test_forward_matches_dequantized_reference(self):
        lhs, rhs, _, sizes = self._operands()
        lhs_d, rhs_d, sizes_d = self._to_dev(lhs, rhs, sizes)
        out = fp8_esm.fp8_segment_matmul_pallas_2d(
            lhs_d, rhs_d, sizes_d, **_TILINGS
        ).cpu()

        # lhs is scaled per row; rhs per output column, shared by all experts.
        want = _grouped_reference(
            _dequantized(lhs, dims=1),
            _dequantized(rhs, dims=(0, 1)),
            _GROUP_SIZES,
        )
        self.assertEqual(out.dtype, torch.bfloat16)
        # XLA's FP32 -> FP8 convert may round a few elements one FP8 ulp
        # differently from torch's, so compare in norm.
        self.assertLess(_rel_err(out, want), 5e-3)

    def test_backward_is_close_to_bf16(self):
        lhs, rhs, dout, sizes = self._operands()
        lhs_d, rhs_d = self._to_dev(lhs, rhs, grad=True)
        (sizes_d,) = self._to_dev(sizes)
        out = fp8_esm.fp8_segment_matmul_pallas_2d(
            lhs_d, rhs_d, sizes_d, **_TILINGS
        )
        dlhs, drhs = torch.autograd.grad(
            out, (lhs_d, rhs_d), dout.to(self.accelerator_device)
        )

        lhs_r, rhs_r = (
            lhs.float().requires_grad_(),
            rhs.float().requires_grad_(),
        )
        want = _grouped_reference(lhs_r, rhs_r, _GROUP_SIZES)
        want_dlhs, want_drhs = torch.autograd.grad(
            want, (lhs_r, rhs_r), dout.float()
        )

        self.assertEqual(drhs.dtype, torch.bfloat16)
        self.assertLess(_rel_err(dlhs, want_dlhs), 0.1)
        self.assertLess(_rel_err(drhs, want_drhs), 0.1)
        # Experts that received no rows get exactly zero weight gradient.
        drhs = drhs.cpu()
        for g, size in enumerate(_GROUP_SIZES):
            if size == 0:
                torch.testing.assert_close(drhs[g], torch.zeros_like(drhs[g]))

    def test_tiny_gradient_rows_stay_finite(self):
        # A dout row whose (dout * rhs_scale) amax / 57344 is subnormal in FP32
        # gets a zero scale on TPU (flush-to-zero) unless the scale is floored;
        # the row then quantizes to inf/NaN and, through dW's per-column
        # scales, poisons every expert's weight gradient.
        lhs, rhs, dout, sizes = self._operands()
        dout[5] = dout[5] * 1e-32
        lhs_d, rhs_d = self._to_dev(lhs, rhs, grad=True)
        (sizes_d,) = self._to_dev(sizes)
        out = fp8_esm.fp8_segment_matmul_pallas_2d(
            lhs_d, rhs_d, sizes_d, **_TILINGS
        )
        dlhs, drhs = torch.autograd.grad(
            out, (lhs_d, rhs_d), dout.to(self.accelerator_device)
        )
        self.assertTrue(torch.isfinite(dlhs.float()).all())
        self.assertTrue(torch.isfinite(drhs.float()).all())

        lhs_r, rhs_r = (
            lhs.float().requires_grad_(),
            rhs.float().requires_grad_(),
        )
        want = _grouped_reference(lhs_r, rhs_r, _GROUP_SIZES)
        want_drhs = torch.autograd.grad(want, (rhs_r,), dout.float())[0]
        self.assertLess(_rel_err(drhs, want_drhs), 0.1)

    def test_paired_matches_unpaired(self):
        lhs, rhs_a, dout_a, sizes = self._operands()
        _, rhs_b, dout_b, _ = self._operands(n=256)
        rhs_b = rhs_b.flip(0) * 0.5
        dout_b = dout_b.flip(0)
        dev = self.accelerator_device

        def unpaired():
            x, a, b = self._to_dev(lhs, rhs_a, rhs_b, grad=True)
            out_a = fp8_esm.fp8_segment_matmul_pallas_2d(
                x, a, sizes.to(dev), **_TILINGS
            )
            out_b = fp8_esm.fp8_segment_matmul_pallas_2d(
                x, b, sizes.to(dev), **_TILINGS
            )
            grads = torch.autograd.grad(
                (out_a, out_b), (x, a, b), (dout_a.to(dev), dout_b.to(dev))
            )
            return (out_a, out_b), grads

        def paired():
            x, a, b = self._to_dev(lhs, rhs_a, rhs_b, grad=True)
            out_a, out_b = fp8_esm.fp8_paired_segment_matmul_pallas_2d(
                x, a, b, sizes.to(dev), **_TILINGS
            )
            grads = torch.autograd.grad(
                (out_a, out_b), (x, a, b), (dout_a.to(dev), dout_b.to(dev))
            )
            return (out_a, out_b), grads

        # Column scales are per column, so concatenating [a | b] leaves every
        # forward operand unchanged. dInput is one GEMM over 2N instead of two
        # GEMMs and a sum, so it only matches to rounding.
        (pa, pb), (pdx, pda, pdb) = paired()
        (ua, ub), (udx, uda, udb) = unpaired()
        torch.testing.assert_close(pa.cpu(), ua.cpu())
        torch.testing.assert_close(pb.cpu(), ub.cpu())
        torch.testing.assert_close(pda.cpu(), uda.cpu())
        torch.testing.assert_close(pdb.cpu(), udb.cpu())
        # dY is quantized per row over [a | b] instead of per half.
        self.assertLess(_rel_err(pdx, udx), 0.06)

    def test_paired_unsplit_is_concatenation(self):
        lhs, rhs, _, sizes = self._operands()
        dev = self.accelerator_device
        x, a, b = self._to_dev(lhs, rhs, rhs.flip(0))
        split = fp8_esm.fp8_paired_segment_matmul_pallas_2d(
            x, a, b, sizes.to(dev), **_TILINGS
        )
        joint = fp8_esm.fp8_paired_segment_matmul_pallas_2d(
            x, a, b, sizes.to(dev), split=False, **_TILINGS
        )
        torch.testing.assert_close(
            joint.cpu(), torch.cat(split, dim=-1).cpu(), rtol=0, atol=0
        )


if __name__ == "__main__":
    absltest.main()
