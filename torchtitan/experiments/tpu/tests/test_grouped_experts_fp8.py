"""CPU tests for the configuration checks of FP8 grouped experts."""

from unittest import mock

from absl.testing import absltest
import torch
from torchtitan.experiments.tpu.components.grouped_experts import TpuGroupedExperts


def _fp8_experts_config(**overrides) -> TpuGroupedExperts.Config:
  return TpuGroupedExperts.Config(
      dim=256,
      hidden_dim=256,
      num_experts=4,
      use_pallas_grouped_mm=True,
      use_fp8_grouped_mm=True,
      **overrides,
  )


class TpuGroupedExpertsFp8ConfigTest(absltest.TestCase):

  def test_fp8_path_rejects_row_scale(self):
    # The fused SwiGLU + W2 op cannot apply routing scores before W2; it must
    # fail instead of silently dropping them.
    experts = TpuGroupedExperts(_fp8_experts_config())
    x = torch.randn(512, 256, dtype=torch.bfloat16)
    sizes = torch.full((4,), 128, dtype=torch.int32)
    with self.assertRaisesRegex(NotImplementedError, "row_scale_R"):
      experts(x, sizes, row_scale_R=torch.ones(512))

  def test_fp8_requires_silu(self):
    # The fused SwiGLU + W2 op always computes silu(gate) * up.
    with self.assertRaisesRegex(ValueError, "silu"):
      TpuGroupedExperts(_fp8_experts_config(activation="gelu_tanh"))

  def test_fp8_path_rejects_ep(self):
    # The FP8 path grows the last expert's group over the pad rows, which
    # would pull an EP receive buffer's stale tail rows into that expert.
    experts = TpuGroupedExperts(_fp8_experts_config())

    class ParallelDimsWithEp:

      @staticmethod
      def get_optional_mesh(name):
        return object() if name == "ep" else None

    with self.assertRaisesRegex(ValueError, "FP8 grouped matmul requires EP=1"):
      experts._parallelize(ParallelDimsWithEp())

  def test_fp8_rows_are_padded_to_the_fp8_m_tiles(self):
    # The BF16 tiles of qwen3_moe_235b_scan align rows to 512, but the FP8
    # W1/W3 dWeight GEMM's default M tile is 1024: 512 rows must reach the
    # FP8 kernels as 1024, with the 512 zero rows in the last expert.
    experts = TpuGroupedExperts(
        _fp8_experts_config(
            pallas_grouped_mm_tile_m=256,
            pallas_grouped_mm_tgmm_tiling=(512, 2048, 0),
        )
    )
    self.assertEqual(experts.fp8_w13_tilings[2][0], 1024)
    seen = {}

    def fake_fp8_forward(x, w1, w2, w3, sizes):
      del w1, w2, w3
      seen["rows"], seen["sizes"] = x.shape[0], sizes.tolist()
      seen["tail"] = x[512:].float().abs().max().item()
      return torch.zeros(x.shape[0], 256, dtype=torch.bfloat16)

    x = torch.randn(512, 256, dtype=torch.bfloat16)
    sizes = torch.full((4,), 128, dtype=torch.int32)
    with mock.patch.object(
        experts, "_fp8_experts_forward", side_effect=fake_fp8_forward
    ):
      out = experts(x, sizes)
    self.assertEqual(out.shape, (512, 256))
    self.assertEqual(seen["rows"], 1024)
    self.assertEqual(seen["sizes"], [128, 128, 128, 640])
    self.assertEqual(seen["tail"], 0.0)

  def test_in_kernel_lhs_quant_is_opt_in(self):
    # GMM v2 in-kernel quantization needs an FP8 x FP8 MXU (v7x), so turning
    # on FP8 experts alone must work on earlier TPUs too.
    self.assertFalse(_fp8_experts_config().fp8_w13_fused_lhs_quant)


if __name__ == "__main__":
  absltest.main()
