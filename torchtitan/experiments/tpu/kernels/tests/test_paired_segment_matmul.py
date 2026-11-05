"""Accelerator parity tests for the paired expert gate/up kernel."""

import math

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import expert_segment_matmul
from torchtitan.experiments.tpu.kernels import paired_segment_matmul


class PairedSegmentMatmulTest(base_device_test.BaseAcceleratorDeviceTest):
    def test_forward_and_backward_parity(self):
        device = self.accelerator_device
        group_sizes = torch.tensor([0, 103, 0, 17, 136, 0], dtype=torch.int32)
        total_m = int(group_sizes.sum().item())
        k, n = 128, 256

        def bounded_values(shape, modulus, scale):
            values = torch.arange(
                math.prod(shape), device=device, dtype=torch.float32
            )
            return (
                values.remainder(modulus)
                .sub(modulus // 2)
                .mul(scale)
                .reshape(shape)
                .bfloat16()
            )

        lhs_ref = bounded_values((total_m, k), 17, 0.003).requires_grad_()
        rhs1_ref = bounded_values((6, k, n), 19, 0.002).requires_grad_()
        rhs3_ref = bounded_values((6, k, n), 23, 0.0015).requires_grad_()
        grad1 = bounded_values((total_m, n), 29, 0.002)
        grad3 = bounded_values((total_m, n), 31, 0.001)
        group_sizes_dev = group_sizes.to(device)

        out1_ref = expert_segment_matmul.segment_matmul_pallas_2d(
            lhs_ref,
            rhs1_ref,
            group_sizes_dev,
            tile_m=128,
            tgmm_tiling=(128, 0, 0),
        )
        out3_ref = expert_segment_matmul.segment_matmul_pallas_2d(
            lhs_ref,
            rhs3_ref,
            group_sizes_dev,
            tile_m=128,
            tgmm_tiling=(128, 0, 0),
        )
        grads_ref = torch.autograd.grad(
            (out1_ref, out3_ref),
            (lhs_ref, rhs1_ref, rhs3_ref),
            (grad1, grad3),
        )

        lhs = lhs_ref.detach().clone().requires_grad_(True)
        rhs1 = rhs1_ref.detach().clone().requires_grad_(True)
        rhs3 = rhs3_ref.detach().clone().requires_grad_(True)
        out1, out3 = paired_segment_matmul.paired_segment_matmul_pallas_2d(
            lhs,
            rhs1,
            rhs3,
            group_sizes_dev,
            tile_m=128,
            tgmm_tile_m=128,
        )
        grads = torch.autograd.grad(
            (out1, out3),
            (lhs, rhs1, rhs3),
            (grad1, grad3),
        )

        torch.testing.assert_close(out1, out1_ref, rtol=0, atol=0)
        torch.testing.assert_close(out3, out3_ref, rtol=0, atol=0)
        torch.testing.assert_close(grads[0], grads_ref[0], rtol=0, atol=0)
        torch.testing.assert_close(grads[1], grads_ref[1], rtol=0, atol=0)
        torch.testing.assert_close(grads[2], grads_ref[2], rtol=0, atol=0)

    @staticmethod
    def _bounded_values(shape, modulus, scale, device):
        """Small exactly-representable bf16 values, so comparisons can be exact."""
        values = torch.arange(math.prod(shape), device=device, dtype=torch.float32)
        return (
            values.remainder(modulus)
            .sub(modulus // 2)
            .mul(scale)
            .reshape(shape)
            .bfloat16()
        )

    def _paired_vs_unpaired(self, group_sizes, k, n, *, tile_m, tgmm_tile_k):
        """Paired kernel with a K-tiled dWeight against the unpaired kernels.

        Every dWeight element is the same FP32 reduction over the group's rows
        whichever K slab it lands in, so the comparison is bit-exact.
        """
        device = self.accelerator_device
        total_m = int(group_sizes.sum().item())
        groups = group_sizes.shape[0]

        lhs_ref = self._bounded_values((total_m, k), 17, 0.003, device).requires_grad_()
        rhs1_ref = self._bounded_values((groups, k, n), 19, 0.002, device).requires_grad_()
        rhs3_ref = self._bounded_values((groups, k, n), 23, 0.0015, device).requires_grad_()
        grad1 = self._bounded_values((total_m, n), 29, 0.002, device)
        grad3 = self._bounded_values((total_m, n), 31, 0.001, device)
        group_sizes_dev = group_sizes.to(device)
        ref_tiling = (tile_m, tgmm_tile_k, 0)

        out1_ref = expert_segment_matmul.segment_matmul_pallas_2d(
            lhs_ref, rhs1_ref, group_sizes_dev, tile_m=tile_m, tgmm_tiling=ref_tiling
        )
        out3_ref = expert_segment_matmul.segment_matmul_pallas_2d(
            lhs_ref, rhs3_ref, group_sizes_dev, tile_m=tile_m, tgmm_tiling=ref_tiling
        )
        grads_ref = torch.autograd.grad(
            (out1_ref, out3_ref), (lhs_ref, rhs1_ref, rhs3_ref), (grad1, grad3)
        )

        lhs = lhs_ref.detach().clone().requires_grad_(True)
        rhs1 = rhs1_ref.detach().clone().requires_grad_(True)
        rhs3 = rhs3_ref.detach().clone().requires_grad_(True)
        out1, out3 = paired_segment_matmul.paired_segment_matmul_pallas_2d(
            lhs,
            rhs1,
            rhs3,
            group_sizes_dev,
            tile_m=tile_m,
            tgmm_tile_m=tile_m,
            tgmm_tile_k=tgmm_tile_k,
        )
        grads = torch.autograd.grad((out1, out3), (lhs, rhs1, rhs3), (grad1, grad3))

        torch.testing.assert_close(out1, out1_ref, rtol=0, atol=0)
        torch.testing.assert_close(out3, out3_ref, rtol=0, atol=0)
        for got, want in zip(grads, grads_ref):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_k_tiled_dweight_parity(self):
        self._paired_vs_unpaired(
            torch.tensor([0, 103, 0, 17, 136, 0], dtype=torch.int32),
            256,
            256,
            tile_m=128,
            tgmm_tile_k=128,
        )

    def test_k_tiled_dweight_at_qwen3_235b_like_shape(self):
        """The K-tiled dWeight at a 235B-*like* dim 2048 x moe_hidden 768.

        Half of Qwen3-235B's 4096 x 1536 on each axis, so the 8:3 ratio and the
        four K slabs are the production ones, but every buffer is a quarter of
        the size. That is deliberate: the scoped VMEM limit is generation
        dependent -- 64 MB on v6e/v7x but only 16 MB on v5e -- and at the true
        235B shape the surrounding forward and dLHS matmuls want 29.50 MB, so
        they fail to compile on v5e with CompileTimeScopedVmemOom.

        Measured on v5e, halving both dims leaves real headroom on both axes:

          scoped VMEM   8.75 MB needed vs the 16 MB default   (1.8x)
                        verified: passes at 10 MB, fails at 8 MB
          HBM           138 MiB of live tensors vs 15.75 GiB  (>100x)

        The tiling under test stays load-bearing at this size rather than
        becoming a formality: with full K the dWeight wants 27.50 MB, over
        v5e's limit, so ``tgmm_tile_k=0`` here fails to compile while
        ``tgmm_tile_k=512`` runs. This is the smallest 235B-ratio shape that
        keeps that true.

        This drives the paired *backward* op directly rather than going through
        the forward: the paired forward's two output slabs need a raised
        ``xla_tpu_scoped_vmem_limit_kib`` (which the 235B recipe sets), and that
        is unrelated to the dWeight tiling under test here.
        """
        device = self.accelerator_device
        group_sizes = torch.tensor([300, 0, 812, 424], dtype=torch.int32)
        k, n, tile_m, tgmm_tile_k = 2048, 768, 256, 512
        total_m = int(group_sizes.sum().item())
        groups = group_sizes.shape[0]

        lhs = self._bounded_values((total_m, k), 17, 0.003, device).requires_grad_()
        rhs1 = self._bounded_values((groups, k, n), 19, 0.002, device).requires_grad_()
        rhs3 = self._bounded_values((groups, k, n), 23, 0.0015, device).requires_grad_()
        grad1 = self._bounded_values((total_m, n), 29, 0.002, device)
        grad3 = self._bounded_values((total_m, n), 31, 0.001, device)
        group_sizes_dev = group_sizes.to(device)

        # Reference: the unpaired kernels' autograd, K-tiled the same way.
        ref_tiling = (tile_m, tgmm_tile_k, 0)
        out1 = expert_segment_matmul.segment_matmul_pallas_2d(
            lhs, rhs1, group_sizes_dev, tile_m=tile_m, tgmm_tiling=ref_tiling
        )
        out3 = expert_segment_matmul.segment_matmul_pallas_2d(
            lhs, rhs3, group_sizes_dev, tile_m=tile_m, tgmm_tiling=ref_tiling
        )
        want_lhs, want_rhs1, want_rhs3 = torch.autograd.grad(
            (out1, out3), (lhs, rhs1, rhs3), (grad1, grad3)
        )

        got_lhs, got_rhs1, got_rhs3 = paired_segment_matmul._paired_grouped_mm_bwd(
            grad1,
            grad3,
            lhs.detach(),
            rhs1.detach(),
            rhs3.detach(),
            group_sizes_dev,
            tile_m,
            tile_m,
            tgmm_tile_k,
        )
        torch.testing.assert_close(got_rhs1, want_rhs1, rtol=0, atol=0)
        torch.testing.assert_close(got_rhs3, want_rhs3, rtol=0, atol=0)
        torch.testing.assert_close(got_lhs, want_lhs, rtol=0, atol=0)

    def test_dinput_tiles_shared_by_many_groups(self):
        """dInput adds the W3 term in place, seeding each row tile once.

        With tile_m=128 the first tile holds rows of four groups (5, 60, 3 and
        the head of the 300-row group), runs of empty groups sit between them,
        one group spans three tiles, and the last group is a single row, so the
        kernel revisits tiles across group boundaries and must seed each one
        exactly once.
        """
        self._paired_vs_unpaired(
            torch.tensor([5, 0, 60, 3, 0, 300, 0, 0, 143, 1], dtype=torch.int32),
            128,
            256,
            tile_m=128,
            tgmm_tile_k=0,
        )

    def test_dinput_rounds_like_two_matmuls_with_random_values(self):
        """dInput equals bf16(bf16(dH1 W1^T) + bf16(dH3 W3^T)) bit for bit.

        Random bf16 inputs make both products round, so accumulating the W3
        term in FP32 before a single rounding (instead of rounding it to bf16
        first, as two separate matmuls and an add do) would be caught here.
        """
        device = self.accelerator_device
        generator = torch.Generator().manual_seed(0)
        group_sizes = torch.tensor([37, 0, 219, 12, 0, 244], dtype=torch.int32)
        total_m = int(group_sizes.sum().item())
        groups, k, n, tile_m = group_sizes.shape[0], 256, 512, 128

        def randn(shape):
            return torch.randn(shape, generator=generator).bfloat16().to(device)

        lhs = randn((total_m, k)).requires_grad_()
        rhs1 = randn((groups, k, n)).requires_grad_()
        rhs3 = randn((groups, k, n)).requires_grad_()
        grad1 = randn((total_m, n))
        grad3 = randn((total_m, n))
        group_sizes_dev = group_sizes.to(device)

        ref_tiling = (tile_m, 0, 0)
        out1 = expert_segment_matmul.segment_matmul_pallas_2d(
            lhs, rhs1, group_sizes_dev, tile_m=tile_m, tgmm_tiling=ref_tiling
        )
        out3 = expert_segment_matmul.segment_matmul_pallas_2d(
            lhs, rhs3, group_sizes_dev, tile_m=tile_m, tgmm_tiling=ref_tiling
        )
        (want_lhs,) = torch.autograd.grad((out1, out3), (lhs,), (grad1, grad3))

        got_lhs, _, _ = paired_segment_matmul._paired_grouped_mm_bwd(
            grad1,
            grad3,
            lhs.detach(),
            rhs1.detach(),
            rhs3.detach(),
            group_sizes_dev,
            tile_m,
            tile_m,
            0,
        )
        torch.testing.assert_close(got_lhs, want_lhs, rtol=0, atol=0)

    def test_fp32_dweight_widens_the_bf16_dweight_bit_for_bit(self):
        device = self.accelerator_device
        generator = torch.Generator().manual_seed(0)
        group_sizes = torch.tensor(
            [37, 0, 219, 12, 0, 244], dtype=torch.int32, device=device
        )
        groups, m, k, n = 6, 512, 256, 384

        def randn(*shape):
            return torch.randn(shape, generator=generator).bfloat16().to(device)

        args = (
            randn(m, n),
            randn(m, n),
            randn(m, k),
            randn(groups, k, n),
            randn(groups, k, n),
            group_sizes,
            128,
            128,
            128,
        )
        want = paired_segment_matmul._paired_grouped_mm_bwd(*args)
        got = paired_segment_matmul._paired_grouped_mm_bwd_fp32_dweight(*args)
        torch.testing.assert_close(got[0], want[0], rtol=0, atol=0)
        for got_rhs, want_rhs in zip(got[1:], want[1:]):
            self.assertEqual(got_rhs.dtype, torch.float32)
            torch.testing.assert_close(got_rhs, want_rhs.float(), rtol=0, atol=0)

    def test_rejects_k_tile_that_does_not_divide_k(self):
        device = self.accelerator_device
        lhs = torch.ones((128, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        rhs = torch.ones((2, 128, 256), device=device, dtype=torch.bfloat16, requires_grad=True)
        sizes = torch.tensor([64, 64], device=device, dtype=torch.int32)
        out1, out3 = paired_segment_matmul.paired_segment_matmul_pallas_2d(
            lhs, rhs, rhs, sizes, tile_m=128, tgmm_tile_m=128, tgmm_tile_k=48
        )
        with self.assertRaisesRegex(ValueError, "tile_k must divide K"):
            torch.autograd.grad((out1, out3), (rhs,), (out1, out3))

    def test_rejects_all_empty_groups(self):
        device = self.accelerator_device
        with self.assertRaisesRegex(ValueError, "at least one routed row"):
            paired_segment_matmul.paired_segment_matmul_pallas_2d(
                torch.empty((0, 128), device=device, dtype=torch.bfloat16),
                torch.empty((4, 128, 256), device=device, dtype=torch.bfloat16),
                torch.empty((4, 128, 256), device=device, dtype=torch.bfloat16),
                torch.zeros(4, device=device, dtype=torch.int32),
                tile_m=128,
                tgmm_tile_m=128,
            )


if __name__ == "__main__":
    absltest.main()
