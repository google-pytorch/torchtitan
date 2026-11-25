"""TPU parity test: fused-layout RoE experts vs. sort-gather + GMMs + unsort.

Runs one EP rank's Ring-of-Experts experts on a single device both ways and
checks the partial combine and every gradient match bit for bit.
"""

from absl.testing import absltest
import torch
import torch.nn.functional as F
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.testing import base_device_test


class RoEFusedGmmTpuTest(base_device_test.BaseAcceleratorDeviceTest):

  def _run(
      self,
      *,
      transpose_w2: bool,
      g_tokens=2048,
      d=1024,
      f=512,
      num_experts=16,
      ep_size=4,
      ep_rank=1,
      top_k=4,
      cap=3072,
      tile_m=512,
      tgmm_tile_k=0
  ):
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("Pallas RoE kernels require TPU.")
    # pylint: disable=g-import-not-at-top
    from torchtitan.experiments.tpu.kernels import ragged_gather_tc as rgt
    from torchtitan.experiments.tpu.kernels import roe_fused_gmm as rfg
    from torchtitan.experiments.tpu.kernels.expert_segment_matmul import (
        segment_matmul_pallas_2d,
    )
    from torchtitan.experiments.tpu.kernels.paired_segment_matmul import (
        paired_segment_matmul_pallas_2d,
    )
    # pylint: enable=g-import-not-at-top

    dev = self.accelerator_device
    gen = torch.Generator().manual_seed(0)
    e = num_experts // ep_size
    reduce_block = 896
    gather_block = 1024
    ids = torch.stack([
        torch.randperm(num_experts, generator=gen)[:top_k]
        for _ in range(g_tokens)
    ]).to(torch.int32)
    rank = torch.tensor([ep_rank], dtype=torch.int32)
    (
        counts_e,
        token_ids_sorted,
        window_slot,
        start_count,
        src_rows,
        tokens,
        blocks,
    ) = rgt.roe_compute_routing(
        ids.reshape(-1).to(dev),
        rank.to(dev),
        num_experts=num_experts,
        ep_size=ep_size,
        top_k=top_k,
        num_out_tokens=g_tokens,
        cap=cap,
        reduce_block_size=reduce_block,
    )
    del window_slot
    count = int(start_count[1].cpu())
    self.assertGreater(count, 0)
    self.assertLessEqual(count, cap)

    def rnd(shape, scale):
      return (torch.randn(shape, generator=gen) * scale).bfloat16().to(dev)

    x0 = rnd((g_tokens, d // 128, 128), 1.0)
    w1_0 = rnd((e, d, f), 0.03)
    w3_0 = rnd((e, d, f), 0.03)
    w2_0 = rnd((e, d, f) if transpose_w2 else (e, f, d), 0.03)
    scale0 = torch.rand((cap,), generator=gen).to(dev)
    grad_out = rnd((g_tokens, d // 128, 128), 1.0)

    def leaves():
      return [
          t.detach().clone().requires_grad_(True)
          for t in (x0, w1_0, w3_0, w2_0, scale0)
      ]

    # Reference: the existing RoE path.
    x, w1, w3, w2, scale = leaves()
    routed = rgt.roe_sort_gather(
        x,
        token_ids_sorted,
        start_count,
        src_rows,
        tokens,
        blocks,
        cap=cap,
        top_k=top_k,
        gather_block_size=gather_block,
        reduce_block_size=reduce_block,
    )
    gate, up = paired_segment_matmul_pallas_2d(
        routed,
        w1,
        w3,
        counts_e,
        tile_m=tile_m,
        tgmm_tile_m=tile_m,
        tgmm_tile_k=tgmm_tile_k,
    )
    hidden = F.silu(gate) * up
    hidden = (hidden.float() * scale.reshape(-1, 1)).to(hidden.dtype)
    y = segment_matmul_pallas_2d(
        hidden,
        w2,
        counts_e,
        tile_m=tile_m,
        tgmm_tiling=(tile_m, 0, 0),
        transpose_rhs=transpose_w2,
    )
    ref = rgt.roe_unsort_reduce(
        y,
        token_ids_sorted,
        start_count,
        src_rows,
        tokens,
        blocks,
        num_out_tokens=g_tokens,
        top_k=top_k,
        gather_block_size=gather_block,
        reduce_block_size=reduce_block,
        out_3d=True,
    )
    ref_grads = torch.autograd.grad(ref, (x, w1, w3, w2, scale), grad_out)

    # Fused W1/W3 dispatch-gather path (as wired in TpuGroupedExperts).
    x, w1, w3, w2, scale = leaves()
    gate, up = rfg.roe_fused_w13(
        x,
        w1,
        w3,
        token_ids_sorted,
        start_count,
        counts_e,
        src_rows,
        tokens,
        blocks,
        cap=cap,
        top_k=top_k,
        gather_block_size=512,
        reduce_block_size=reduce_block,
        tile_m=tile_m,
        tgmm_tile_m=tile_m,
        tgmm_tile_k=tgmm_tile_k,
    )
    hidden = F.silu(gate) * up
    hidden = (hidden.float() * scale.reshape(-1, 1)).to(hidden.dtype)
    y_fused = segment_matmul_pallas_2d(
        hidden,
        w2,
        counts_e,
        tile_m=tile_m,
        tgmm_tiling=(tile_m, 0, 0),
        transpose_rhs=transpose_w2,
    )
    out = rgt.roe_unsort_reduce(
        y_fused,
        token_ids_sorted,
        start_count,
        src_rows,
        tokens,
        blocks,
        num_out_tokens=g_tokens,
        top_k=top_k,
        gather_block_size=gather_block,
        reduce_block_size=reduce_block,
        out_3d=True,
    )
    grads = torch.autograd.grad(out, (x, w1, w3, w2, scale), grad_out)

    self.assertEqual(out.shape, ref.shape)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=0, atol=0)
    valid = torch.arange(cap) < count
    for name, got, want in zip(
        ("dx", "dw1", "dw3", "dw2", "dscale"), grads, ref_grads
    ):
      got, want = got.cpu(), want.cpu()
      if name == "dscale":
        got, want = got[valid], want[valid]
      torch.testing.assert_close(got, want, rtol=0, atol=0, msg=name)

  def test_parity(self):
    self._run(transpose_w2=False)

  def test_parity_transposed_w2(self):
    self._run(transpose_w2=True)

  def test_parity_qwen_dims(self):
    self._run(
        transpose_w2=False,
        g_tokens=4096,
        d=4096,
        f=1024,
        num_experts=64,
        ep_size=16,
        top_k=10,
        cap=3072,
        tile_m=256,
        tgmm_tile_k=1024,
    )


if __name__ == "__main__":
  absltest.main()
