"""Tests for fill_indices."""

import torch
from absl.testing import absltest
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.experiments.tpu.kernels.fill_indices import fill_indices

def fill_indices_cpu_original(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
):
  """Original Loop-based Implementation copied from models/moe/kernels.py"""
  permuted_indices = torch.full(
      (max_len,),
      -1,
      dtype=torch.int32,
      device=tokens_per_expert_group.device,
  )
  for e in range(experts_per_rank):
    write_start = write_offsets[e].item()
    for r in range(num_ranks):
      i = r * experts_per_rank + e
      start_index = start_index_values[i].item()
      length = tokens_per_expert_group[i].item()
      if length > 0:
        end_idx = min(write_start + length, max_len)
        if write_start < end_idx:
          permuted_indices[write_start:end_idx] = torch.arange(
              start_index,
              start_index + (end_idx - write_start),
              dtype=torch.int32,
              device=tokens_per_expert_group.device,
          )
      write_start += length
  return permuted_indices


class TestFillIndices(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for fill_indices."""

  def setUp(self):
    super().setUp()
    self.epr = 4
    self.nr = 2
    self.ml = 20
    self.tpeg = torch.tensor([4, 2, 1, 3, 1, 2, 3, 4], dtype=torch.int32)
    self.siv = (torch.cumsum(self.tpeg, 0) - self.tpeg).to(torch.int32)
    m_sizes = torch.tensor([8, 4, 4, 8], dtype=torch.int32)
    self.wo = (torch.cumsum(m_sizes, 0) - m_sizes).to(torch.int32)

  def test_parity(self):
    """Test functionality matches the CPU reference on default config."""
    device = self.accelerator_device
    tpeg_tpu = self.tpeg.to(device)
    siv_tpu = self.siv.to(device)
    wo_tpu = self.wo.to(device)

    res_orig = fill_indices_cpu_original(
        self.tpeg, self.siv, self.wo, self.epr, self.nr, self.ml
    )
    res_tpu = fill_indices(
        tpeg_tpu, siv_tpu, wo_tpu, self.epr, self.nr, self.ml
    )
    res_tpu_cpu = res_tpu.cpu()

    self.assertTrue(
        torch.equal(res_orig, res_tpu_cpu),
        msg="TPU result does not match CPU reference",
    )

  def test_truncation(self):
    """Tests functionality when written offsets exceed max sequence length."""
    device = self.accelerator_device
    ml_short = 10
    tpeg_tpu = self.tpeg.to(device)
    siv_tpu = self.siv.to(device)
    wo_tpu = self.wo.to(device)

    res_orig = fill_indices_cpu_original(
        self.tpeg, self.siv, self.wo, self.epr, self.nr, ml_short
    )
    res_tpu = fill_indices(
        tpeg_tpu, siv_tpu, wo_tpu, self.epr, self.nr, ml_short
    )
    res_tpu_cpu = res_tpu.cpu()

    self.assertTrue(
        torch.equal(res_orig, res_tpu_cpu),
        msg=f"TPU result does not match CPU reference with max_len={ml_short}",
    )
    self.assertEqual(res_tpu_cpu.size(0), ml_short)

  def test_empty(self):
    """Tests functionality with empty uninitialized input arrays."""
    device = self.accelerator_device
    tpeg_empty = torch.zeros_like(self.tpeg)
    siv_empty = torch.zeros_like(self.siv)
    wo_empty = torch.zeros_like(self.wo)

    tpeg_tpu = tpeg_empty.to(device)
    siv_tpu = siv_empty.to(device)
    wo_tpu = wo_empty.to(device)

    res_orig = fill_indices_cpu_original(
        tpeg_empty, siv_empty, wo_empty, self.epr, self.nr, self.ml
    )
    res_tpu = fill_indices(
        tpeg_tpu, siv_tpu, wo_tpu, self.epr, self.nr, self.ml
    )
    res_tpu_cpu = res_tpu.cpu()

    self.assertTrue(
        torch.equal(res_orig, res_tpu_cpu),
        msg="TPU result does not match CPU reference for empty inputs",
    )

  def test_random_config(self):
    """Tests functionality on randomly initialized tensor configurations."""
    device = self.accelerator_device
    torch.manual_seed(42)
    epr = 16
    nr = 8
    ml = 500

    tpeg = torch.randint(0, 10, (epr * nr,), dtype=torch.int32)
    siv = (torch.cumsum(tpeg, 0) - tpeg).to(torch.int32)
    m_sizes = torch.randint(20, 40, (epr,), dtype=torch.int32)
    wo = (torch.cumsum(m_sizes, 0) - m_sizes).to(torch.int32)

    tpeg_tpu = tpeg.to(device)
    siv_tpu = siv.to(device)
    wo_tpu = wo.to(device)

    res_orig = fill_indices_cpu_original(tpeg, siv, wo, epr, nr, ml)
    res_tpu = fill_indices(tpeg_tpu, siv_tpu, wo_tpu, epr, nr, ml)
    res_tpu_cpu = res_tpu.cpu()

    self.assertTrue(
        torch.equal(res_orig, res_tpu_cpu),
        msg="TPU result does not match CPU reference for large random inputs",
    )

  def test_real_4_experts_config(self):
    """Tests real configuration logged from Qwen3 with 4 experts."""
    device = self.accelerator_device
    tpeg = torch.tensor([512, 512, 512, 512], dtype=torch.int32)
    siv = torch.tensor([0, 512, 1024, 1536], dtype=torch.int32)
    wo = torch.tensor([0, 512, 1024, 1536], dtype=torch.int32)
    epr = 4
    nr = 1
    ml = 2080

    tpeg_tpu = tpeg.to(device)
    siv_tpu = siv.to(device)
    wo_tpu = wo.to(device)

    res_orig = fill_indices_cpu_original(tpeg, siv, wo, epr, nr, ml)
    res_tpu = fill_indices(tpeg_tpu, siv_tpu, wo_tpu, epr, nr, ml)
    res_tpu_cpu = res_tpu.cpu()

    self.assertTrue(
        torch.equal(res_orig, res_tpu_cpu),
        msg="TPU result does not match CPU reference for real config 1",
    )

  def test_real_16_experts_config(self):
    """Tests real configuration logged from Qwen3 with 16 experts."""
    device = self.accelerator_device
    tpeg = torch.tensor([
        297, 215, 297, 215, 297, 215, 297, 215, 297, 215, 297, 215, 297, 215, 297, 215
    ], dtype=torch.int32)
    siv = torch.tensor([
        0, 297, 512, 809, 1024, 1321, 1536, 1833, 2048, 2345, 2560, 2857, 3072, 3369, 3584, 3881
    ], dtype=torch.int32)
    wo = torch.tensor([
        0, 304, 520, 824, 1040, 1344, 1560, 1864, 2080, 2384, 2600, 2904, 3120, 3424, 3640, 3944
    ], dtype=torch.int32)
    epr = 16
    nr = 1
    ml = 4224

    tpeg_tpu = tpeg.to(device)
    siv_tpu = siv.to(device)
    wo_tpu = wo.to(device)

    res_orig = fill_indices_cpu_original(tpeg, siv, wo, epr, nr, ml)
    res_tpu = fill_indices(tpeg_tpu, siv_tpu, wo_tpu, epr, nr, ml)
    res_tpu_cpu = res_tpu.cpu()

    self.assertTrue(
        torch.equal(res_orig, res_tpu_cpu),
        msg="TPU result does not match CPU reference for real config 2",
    )


if __name__ == "__main__":
  absltest.main()
