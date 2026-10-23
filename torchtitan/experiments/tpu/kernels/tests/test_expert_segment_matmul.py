"""Accelerator tests for the MegaBlox expert grouped matmul."""

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu import base_device_test
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


if __name__ == "__main__":
    absltest.main()
