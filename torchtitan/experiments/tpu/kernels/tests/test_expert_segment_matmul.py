"""Accelerator tests for the MegaBlox expert grouped matmul."""

from absl.testing import absltest
from absl.testing import parameterized
import torch

from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.experiments.tpu.kernels import expert_segment_matmul


class ExpertSegmentMatmulTest(base_device_test.BaseAcceleratorDeviceTest):
    def test_dweight_is_stored_in_the_weight_dtype_and_rounds_once(self):
        """The BF16 dWeight equals the exact FP32 dWeight rounded once.

        Small integer inputs keep every FP32 partial sum exact whatever order
        the kernel accumulates in, so the reference is unambiguous. The sums
        reach past 256, where BF16 can no longer hold every integer, so the
        final store does round.
        """
        device = self.accelerator_device
        generator = torch.Generator().manual_seed(0)
        sizes = [64, 64]
        group_sizes = torch.tensor(sizes, dtype=torch.int32, device=device)

        def integers(shape):
            return torch.randint(-32, 33, shape, generator=generator).bfloat16()

        lhs = integers((128, 128))
        dout = integers((128, 128))
        rhs = integers((2, 128, 128))

        _, grad_rhs = expert_segment_matmul._grouped_mm_bwd(
            dout.to(device),
            lhs.to(device),
            rhs.to(device),
            group_sizes,
            128,
            0,
            0,
            128,
            128,
            128,
            False,
        )

        want_fp32 = torch.stack(
            [
                rows_lhs.float().T @ rows_dout.float()
                for rows_lhs, rows_dout in zip(
                    lhs.split(sizes), dout.split(sizes)
                )
            ]
        )
        want = want_fp32.to(torch.bfloat16)
        self.assertGreater(int((want.float() != want_fp32).sum()), 0)
        self.assertEqual(grad_rhs.dtype, torch.bfloat16)
        torch.testing.assert_close(grad_rhs.cpu(), want, rtol=0, atol=0)

    def test_transpose_rhs_forward_and_backward_parity(self):
        device = self.accelerator_device
        group_sizes = torch.tensor([64, 64], dtype=torch.int32, device=device)
        lhs_ref = torch.randn(
            128, 128, dtype=torch.bfloat16, device=device
        ).requires_grad_()
        rhs_ref = torch.randn(
            2, 128, 256, dtype=torch.bfloat16, device=device
        ).requires_grad_()
        grad_output = torch.randn(128, 256, dtype=torch.bfloat16, device=device)

        out_ref = expert_segment_matmul.segment_matmul_pallas_2d(
            lhs_ref,
            rhs_ref,
            group_sizes,
            tile_m=128,
            tgmm_tiling=(128, 0, 0),
        )
        grads_ref = torch.autograd.grad(out_ref, (lhs_ref, rhs_ref), grad_output)

        lhs = lhs_ref.detach().clone().requires_grad_()
        rhs = rhs_ref.detach().transpose(-2, -1).contiguous().requires_grad_()
        out = expert_segment_matmul.segment_matmul_pallas_2d(
            lhs,
            rhs,
            group_sizes,
            tile_m=128,
            tgmm_tiling=(128, 0, 0),
            transpose_rhs=True,
        )
        grads = torch.autograd.grad(out, (lhs, rhs), grad_output)

        out = out.cpu()
        out_ref = out_ref.cpu()
        grad_lhs, grad_rhs = (grad.cpu() for grad in grads)
        grad_lhs_ref, grad_rhs_ref = (grad.cpu() for grad in grads_ref)

        torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
        # The transposed contraction can round BF16 partial sums in a different
        # order. Keep the tolerance to roughly one BF16 ULP at unit scale.
        torch.testing.assert_close(
            grad_lhs, grad_lhs_ref, rtol=8e-3, atol=1e-3
        )
        torch.testing.assert_close(
            grad_rhs.transpose(-2, -1),
            grad_rhs_ref,
            rtol=8e-3,
            atol=1e-3,
        )

    @parameterized.named_parameters(
        # (group sizes, row count, tile_m, dWeight M tile, transpose_rhs)
        # 136 routed rows end 8 rows into the second 128-row tile, and the
        # third tile holds no routed row at all.
        ("partial_and_inactive_tiles", [0, 91, 0, 45], 384, 128, 128, False),
        ("empty_last_group", [0, 91, 45, 0], 384, 128, 128, False),
        # 300 routed rows: 128-row forward tiles 2 full, 1 partial, 3 empty;
        # 256-row dWeight tiles 1 full, 1 partial, 1 empty.
        ("dweight_tile_larger_than_tile_m", [37, 0, 200, 63], 768, 128, 256, False),
        ("transpose_rhs", [37, 0, 200, 63], 768, 128, 256, True),
    )
    def test_rows_past_the_routed_count_are_ignored(
        self, sizes, num_rows, tile_m, tgmm_tile_m, transpose_rhs
    ):
        """Rows past sum(group_sizes) reach no routed row's output or dWeight.

        Under dropless EP the receive buffer has a static worst-case row count
        and its rows past the routed count hold stale data; here they hold NaN.
        The reference zeroes them and gives them to the last group instead,
        the way TpuGroupedExperts used to pad. Zero rows add exactly nothing,
        so the comparison is bit-exact.
        """
        device = self.accelerator_device
        generator = torch.Generator().manual_seed(0)
        valid_m = sum(sizes)
        tail_m = num_rows - valid_m
        groups, k, n = len(sizes), 128, 256

        def randn(shape):
            return torch.randn(shape, generator=generator).bfloat16().to(device)

        lhs_valid = randn((valid_m, k))
        grad_valid = randn((valid_m, n))
        rhs = randn((groups, n, k) if transpose_rhs else (groups, k, n))

        def run(tail_value, group_sizes):
            def with_tail(rows):
                tail = torch.full(
                    (tail_m, rows.shape[1]),
                    tail_value,
                    device=device,
                    dtype=torch.bfloat16,
                )
                return torch.cat([rows, tail])

            lhs = with_tail(lhs_valid).requires_grad_()
            weights = rhs.clone().requires_grad_()
            out = expert_segment_matmul.segment_matmul_pallas_2d(
                lhs,
                weights,
                torch.tensor(group_sizes, dtype=torch.int32).to(device),
                tile_m=tile_m,
                tgmm_tiling=(tgmm_tile_m, 0, 0),
                transpose_rhs=transpose_rhs,
            )
            grad_lhs, grad_rhs = torch.autograd.grad(
                out, (lhs, weights), with_tail(grad_valid)
            )
            return out[:valid_m], grad_lhs[:valid_m], grad_rhs

        want = run(0.0, sizes[:-1] + [sizes[-1] + tail_m])
        got = run(float("nan"), sizes)

        for name, actual, expected in zip(("out", "dX", "dW"), got, want):
            self.assertTrue(bool(torch.isfinite(actual).all()), name)
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0,
                atol=0,
                msg=lambda message, name=name: f"{name}: {message}",
            )


if __name__ == "__main__":
    absltest.main()
