"""Device tests for TpuGroupedExperts' in-kernel EP chunk dWeight accumulation.

Under EP token chunking every chunk runs the experts on the same unsharded
weights. ``TpuGroupedExperts.forward(dweight_accumulators=...)`` adds each
chunk's dWeights onto running sums inside the TGMM kernels. It must give the
same forward values and dInputs as the plain Pallas path, and weight gradients
that match the autograd sums of the per-chunk dWeights (the path it replaces)
up to the per-chunk BF16 rounding those sums carry.
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch

from torchtitan.experiments.tpu.components.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.testing import base_device_test


# Routed rows per expert in each chunk. The second chunk routes nothing to
# expert 0, and no chunk's row count is a multiple of the 128-row tiles, so
# the experts pad every chunk.
_CHUNK_COUNTS = ([37, 0, 150, 113], [0, 64, 20, 30], [10, 10, 10, 10])


class TpuGroupedExpertsChunkDWeightTest(base_device_test.BaseAcceleratorDeviceTest):

  @parameterized.named_parameters(
      # (paired W1+W3 kernel, routing weights before W2, weight dtype)
      ("unpaired", False, True, torch.bfloat16),
      ("paired", True, True, torch.bfloat16),
      ("unpaired_no_row_scale", False, False, torch.bfloat16),
      # FP32 unsharded weights, as with mixed_precision_param=float32.
      ("paired_fp32_weights", True, True, torch.float32),
  )
  def test_in_kernel_sums_match_autograd_sums(
      self, paired, use_row_scale, weight_dtype
  ):
    device = self.accelerator_device
    num_experts, dim, hidden = len(_CHUNK_COUNTS[0]), 128, 256
    experts = TpuGroupedExperts.Config(
        dim=dim,
        hidden_dim=hidden,
        num_experts=num_experts,
        use_pallas_grouped_mm=True,
        use_pallas_paired_grouped_mm=paired,
        pallas_grouped_mm_tile_m=128,
        pallas_grouped_mm_tgmm_tiling=(128, 0, 0),
        use_kernel_native_weight_layout=True,
    ).build()
    self.assertTrue(experts.supports_chunk_dweight_accumulation())
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
      for param in experts.parameters():
        fan_in = param.shape[-2]  # kernel-native (E, K, N)
        param.copy_(torch.randn(param.shape, generator=generator) / fan_in**0.5)
    experts = experts.to(device=device, dtype=weight_dtype)
    params = (experts.w1_EDF, experts.w2_EFD, experts.w3_EDF)

    chunks = []
    for counts in _CHUNK_COUNTS:
      rows = sum(counts)
      chunks.append((
          torch.randn(rows, dim, generator=generator).to(device),
          torch.rand(rows, generator=generator).to(device),
          torch.tensor(counts, dtype=torch.int32).to(device),
          torch.randn(rows, dim, generator=generator).to(device),
      ))

    def run(in_kernel):
      xs = [x.clone().requires_grad_() for x, _, _, _ in chunks]
      weights = experts.unshard_weights()
      outs = []
      if in_kernel:
        accs = experts.chunk_dweight_accumulators(weights, accumulate_fp32=True)
        chunk_weights = tuple(w.bfloat16() for w in weights)
        for i, (x, (_, scale, counts, _)) in enumerate(zip(xs, chunks)):
          out, accs = experts(
              x,
              counts,
              scale if use_row_scale else None,
              pre_unsharded_weights=chunk_weights,
              dweight_accumulators=accs,
              last_chunk=i == len(chunks) - 1,
          )
          outs.append(out)
        self.assertIsNone(accs)
      else:
        # The autograd path: FP32 weights, so each chunk's BF16 dWeight is
        # widened before autograd sums the chunks.
        wide = tuple(w.float() for w in weights)
        for x, (_, scale, counts, _) in zip(xs, chunks):
          outs.append(
              experts(
                  x,
                  counts,
                  scale if use_row_scale else None,
                  pre_unsharded_weights=wide,
              )
          )
      grads = torch.autograd.grad(
          outs, xs + list(params), [g for _, _, _, g in chunks]
      )
      return outs, grads[: len(xs)], grads[len(xs) :]

    outs, grad_xs, dweights = run(in_kernel=True)
    ref_outs, ref_grad_xs, ref_dweights = run(in_kernel=False)

    for name, got, want in (
        *(("out", g, w) for g, w in zip(outs, ref_outs)),
        *(("dx", g, w) for g, w in zip(grad_xs, ref_grad_xs)),
    ):
      torch.testing.assert_close(
          got, want, rtol=0, atol=0, msg=lambda m, name=name: f"{name}: {m}"
      )
    for name, got, want, param in zip(("dw1", "dw2", "dw3"), dweights, ref_dweights, params):
      self.assertEqual(got.dtype, param.dtype, name)
      self.assertTrue(bool(torch.isfinite(got).all()), name)
      # The autograd sums round each chunk's dWeight to BF16 before adding;
      # the in-kernel sums don't. Allow a few BF16 ulps of the largest value.
      scale = want.float().abs().max().item()
      torch.testing.assert_close(
          got.float(),
          want.float(),
          rtol=2e-2,
          atol=2e-2 * scale,
          msg=lambda m, name=name: f"{name}: {m}",
      )

  def test_requires_pre_unsharded_weights(self):
    experts = TpuGroupedExperts.Config(
        dim=128,
        hidden_dim=256,
        num_experts=2,
        use_pallas_grouped_mm=True,
        use_kernel_native_weight_layout=True,
    ).build()
    accs = experts.chunk_dweight_accumulators(
        experts.unshard_weights(), accumulate_fp32=True
    )
    with self.assertRaisesRegex(ValueError, "pre_unsharded_weights"):
      experts(
          torch.zeros(128, 128),
          torch.tensor([64, 64], dtype=torch.int32),
          dweight_accumulators=accs,
      )


if __name__ == "__main__":
  absltest.main()
