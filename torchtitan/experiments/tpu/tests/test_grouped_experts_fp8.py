"""CPU tests for the configuration checks of FP8 grouped experts."""

from absl.testing import absltest
import torch
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts


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


if __name__ == "__main__":
  absltest.main()
