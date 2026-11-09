"""Device tests for how TpuGroupedExperts pads the routed rows for its kernels.

The Pallas kernels need a row count divisible by tile_m and by both dWeight
TGMM M tiles. The experts pad the routed rows to that and leave
num_tokens_per_expert_E alone, since the kernels skip rows past its sum. At
EP=1 that must be bit-identical to appending zero rows to the last expert, as
the experts did before: zero rows add exactly nothing.
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch

from torchtitan.experiments.tpu.components.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.testing import base_device_test


class TpuGroupedExpertsRowPaddingTest(base_device_test.BaseAcceleratorDeviceTest):

  @parameterized.named_parameters(
      # (paired, dWeight M tile, padded row count, pass row_scale_R)
      ("unpaired", False, 128, 384, True),
      ("paired", True, 128, 384, True),
      # A dWeight tile larger than tile_m: 300 rows pad to 512, not 384.
      ("unpaired_dweight_tile_256", False, 256, 512, True),
      ("paired_dweight_tile_256", True, 256, 512, True),
      ("unpaired_no_row_scale", False, 256, 512, False),
      ("paired_no_row_scale", True, 256, 512, False),
  )
  def test_unaligned_rows_match_zero_rows_in_the_last_expert(
      self, paired, tgmm_tile_m, padded_rows, use_row_scale
  ):
    """300 routed rows with tile_m=128, against zero rows in the last expert.

    The reference passes the same rows followed by zero rows that belong to
    the last expert, ``padded_rows`` rows in all, so the experts add no
    padding of their own. That is how the experts used to pad, except that
    they padded to a multiple of tile_m only, so a 256-row dWeight tile
    failed on the 384 rows it got.
    """
    device = self.accelerator_device
    counts = [37, 0, 150, 113]
    num_rows, dim, hidden = sum(counts), 128, 256
    experts = TpuGroupedExperts.Config(
        dim=dim,
        hidden_dim=hidden,
        num_experts=len(counts),
        use_pallas_grouped_mm=True,
        use_pallas_paired_grouped_mm=paired,
        pallas_grouped_mm_tile_m=128,
        pallas_grouped_mm_tgmm_tiling=(tgmm_tile_m, 0, 0),
        use_kernel_native_weight_layout=True,
    ).build()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
      for param in experts.parameters():
        # Kernel-native (E, K, N) layout: K, the input dimension, is dim -2.
        fan_in = param.shape[-2]
        param.copy_(torch.randn(param.shape, generator=generator) / fan_in**0.5)
    experts = experts.to(device)
    weights = (experts.w1_EDF, experts.w2_EFD, experts.w3_EDF)

    x = torch.randn(num_rows, dim, generator=generator).to(device)
    row_scale = torch.rand(num_rows, generator=generator).to(device)
    grad_out = torch.randn(num_rows, dim, generator=generator).to(device)

    def run(zero_rows):
      x_leaf = x.clone().requires_grad_()
      scale_leaf = row_scale.clone().requires_grad_()
      x_rows, scale_rows = x_leaf, scale_leaf
      if zero_rows:
        x_rows = torch.cat([x_leaf, x.new_zeros(zero_rows, dim)])
        scale_rows = torch.cat([scale_leaf, row_scale.new_zeros(zero_rows)])
      counts_e = torch.tensor(
          counts[:-1] + [counts[-1] + zero_rows], dtype=torch.int32
      ).to(device)
      out = experts(x_rows, counts_e, scale_rows if use_row_scale else None)
      out = out[:num_rows]
      inputs = (x_leaf, *weights) + ((scale_leaf,) if use_row_scale else ())
      return (out, *torch.autograd.grad(out, inputs, grad_out))

    want = run(padded_rows - num_rows)
    got = run(0)

    names = ("out", "dx", "dw1", "dw2", "dw3", "drow_scale")
    self.assertEqual(len(got), 5 + use_row_scale)
    for name, actual, expected in zip(names, got, want):
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
