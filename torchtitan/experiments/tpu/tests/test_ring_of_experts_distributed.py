"""Multi-device parity test for the Ring-of-Experts EP transport.

Each rank holds a shard of its EP group's tokens and runs the transport with
the real collectives and the TPU kernels; the result and the gradients must
match a single-device reference computed over the group's full batch.
"""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu import grouped_experts
from torchtitan.experiments.tpu import ring_of_experts
from torchtitan.experiments.tpu import token_dispatcher

_NUM_TOKENS = 256  # per rank
_TOP_K = 4
_NUM_EXPERTS = 16
_DIM = 256


def _global_inputs(world_size):
  """Same inputs on every rank: one shard of tokens per rank."""
  g = torch.Generator().manual_seed(1234)
  x = torch.randn(world_size, _NUM_TOKENS, _DIM, generator=g).bfloat16()
  scores = torch.rand(world_size, _NUM_TOKENS, _TOP_K, generator=g)
  ids = torch.stack([
      torch.randperm(_NUM_EXPERTS, generator=g)[:_TOP_K]
      for _ in range(world_size * _NUM_TOKENS)
  ]).reshape(world_size, _NUM_TOKENS, _TOP_K)
  scales = (torch.rand(_NUM_EXPERTS, _DIM, generator=g) + 0.5).bfloat16()
  grad_out = torch.randn(world_size, _NUM_TOKENS, _DIM, generator=g).bfloat16()
  return x, scores, ids, scales, grad_out


def _scale_experts(routed_RD, group_sizes_e, scales_eD):
  """Per-expert row scaling; rows past the groups become NaN.

  Stands in for the grouped matmuls, which leave rows past `sum(group_sizes)`
  unwritten, so a correct transport never reads them.
  """
  num_rows = routed_RD.shape[0]
  rows = torch.arange(num_rows, device=routed_RD.device)
  ends = torch.cumsum(group_sizes_e.to(torch.int64), 0)
  e_idx = torch.searchsorted(ends, rows, right=True).clamp(
      max=group_sizes_e.numel() - 1
  )
  # FP32 math rounded once to BF16, like the grouped matmuls (whose weight
  # gradients also accumulate in FP32).
  scaled = (routed_RD.float() * scales_eD.float()[e_idx]).to(routed_RD.dtype)
  valid = (rows < ends[-1]).unsqueeze(-1)
  return torch.where(valid, scaled, torch.full_like(scaled, float("nan")))


def _reference(x, scores, ids, scales, grad_out):
  """out[t] = sum_k scores[t, k] * y[t, k], with y = bf16(x[t] * scales[e]).

  The expert outputs `y` are rounded to BF16 exactly as `_scale_experts`
  produces them; everything after that is FP32, like the transport.
  """
  y = (x.unsqueeze(1) * scales[ids]).float()  # BF16 multiply, as the expert
  y = y.requires_grad_()
  scores = scores.clone().requires_grad_()
  out = (scores.unsqueeze(-1) * y).sum(1)
  grad_y, grad_scores = torch.autograd.grad(
      out, (y, scores), grad_out.float()
  )
  # Chain rule through y = x * scale in FP32.
  grad_x = (grad_y * scales[ids].float()).sum(1)
  grad_scales = torch.zeros(scales.shape, dtype=torch.float32)
  grad_scales.index_add_(
      0,
      ids.reshape(-1),
      (grad_y * x.float().unsqueeze(1)).reshape(-1, x.shape[-1]),
  )
  return out.detach(), (grad_x, grad_scores, grad_scales)


def _parity_worker(device, rank, world_size, ep_size, num_chunks):
  mesh = init_device_mesh(
      device.type,
      (world_size // ep_size, ep_size),
      mesh_dim_names=("efsdp", "ep"),
  )
  ep_mesh = mesh["ep"]
  ep_rank = ep_mesh.get_local_rank()
  group = slice(rank - ep_rank, rank - ep_rank + ep_size)
  num_local = _NUM_EXPERTS // ep_size
  experts = slice(ep_rank * num_local, (ep_rank + 1) * num_local)

  x, scores, ids, scales, grad_out = _global_inputs(world_size)
  ref_out, (ref_dx, ref_dscores, ref_dscales) = _reference(
      x[group].reshape(-1, _DIM),
      scores[group].reshape(-1, _TOP_K),
      ids[group].reshape(-1, _TOP_K),
      scales,
      grad_out[group].reshape(-1, _DIM),
  )

  x_d = x[rank].to(device).requires_grad_()
  scores_d = scores[rank].to(device).requires_grad_()
  scales_d = scales[experts].to(device).requires_grad_()
  chunk = _NUM_TOKENS // num_chunks
  outs = []
  for c in range(num_chunks):
    rows = slice(c * chunk, (c + 1) * chunk)
    routed, sizes, meta = ring_of_experts.ring_of_experts_dispatch(
        x_d[rows],
        scores_d[rows],
        ids[rank][rows].to(device),
        num_experts=_NUM_EXPERTS,
        ep_mesh=ep_mesh,
        capacity_factor=None,
    )
    outs.append(
        ring_of_experts.ring_of_experts_combine(
            _scale_experts(routed, sizes, scales_d), meta
        )
    )
  out = torch.cat(outs)
  dx, dscores, dscales = torch.autograd.grad(
      out, (x_d, scores_d, scales_d), grad_out[rank].to(device)
  )

  local = slice(ep_rank * _NUM_TOKENS, (ep_rank + 1) * _NUM_TOKENS)
  # BF16 tokens and expert outputs; FP32 per-token sums.
  tol = dict(rtol=2e-2, atol=2e-2)
  torch.testing.assert_close(out.float().cpu(), ref_out[local], **tol)
  torch.testing.assert_close(dx.float().cpu(), ref_dx[local], **tol)
  torch.testing.assert_close(dscores.cpu(), ref_dscores[local], **tol)
  # Each expert's scale sees every token of the EP group routed to it.
  torch.testing.assert_close(
      dscales.float().cpu(), ref_dscales[experts], rtol=2e-2, atol=2e-1
  )
  logging.info("[rank %d] Ring-of-Experts parity passed.", rank)


_HIDDEN = 128


def _moe_block(num_local_experts, ep_mesh, num_chunks, w1, w2, w3):
  """The Qwen3-235B scan recipe's MoE block over the given expert weights.

  Pallas paired grouped matmuls, kernel-native weight layout, 256-row tiles,
  and the fused route-score backward, as in `qwen3_moe_235b_scan`; the TGMM
  tiles span the full (small) dims.
  """
  config = token_dispatcher.TpuRoutedExperts.Config(
      inner_experts=grouped_experts.TpuGroupedExperts.Config(
          dim=_DIM,
          hidden_dim=_HIDDEN,
          num_experts=num_local_experts,
          use_pallas_grouped_mm=True,
          use_pallas_paired_grouped_mm=True,
          pallas_grouped_mm_tile_m=256,
          pallas_grouped_mm_tgmm_tiling=(256, 0, 0),
          use_kernel_native_weight_layout=True,
      ),
      token_dispatcher=token_dispatcher.TpuAllToAllTokenDispatcher.Config(
          num_experts=_NUM_EXPERTS,
          top_k=_TOP_K,
          fuse_route_score_backward=True,
      ),
      num_moe_token_chunks=num_chunks,
  )
  block = config.build()
  with torch.no_grad():
    for name, weight in (("w1_EDF", w1), ("w2_EFD", w2), ("w3_EDF", w3)):
      getattr(block.inner_experts, name).copy_(weight)
  block.token_dispatcher.wire_meshes(ep_mesh=ep_mesh)
  return block


def _moe_block_parity_worker(device, rank, world_size, ep_size, num_chunks):
  """EP and no-EP runs of the production MoE block with the same weights.

  The EP block holds this rank's slice of the experts and dispatches its own
  tokens over the EP group. The no-EP block holds every expert and runs the
  whole group's batch, so its weight gradients see the same routes. This
  covers dispatch, the grouped matmuls, combine and chunking; it does not
  cover how the scan trainer shards stacked expert weights.
  """
  mesh = init_device_mesh(
      device.type,
      (world_size // ep_size, ep_size),
      mesh_dim_names=("efsdp", "ep"),
  )
  ep_mesh = mesh["ep"]
  ep_rank = ep_mesh.get_local_rank()
  group = slice(rank - ep_rank, rank - ep_rank + ep_size)
  num_local = _NUM_EXPERTS // ep_size
  experts = slice(ep_rank * num_local, (ep_rank + 1) * num_local)

  x, scores, ids, _, grad_out = _global_inputs(world_size)
  g = torch.Generator().manual_seed(5678)
  w1 = (torch.randn(_NUM_EXPERTS, _DIM, _HIDDEN, generator=g) / _DIM**0.5)
  w2 = (torch.randn(_NUM_EXPERTS, _HIDDEN, _DIM, generator=g) / _HIDDEN**0.5)
  w3 = (torch.randn(_NUM_EXPERTS, _DIM, _HIDDEN, generator=g) / _DIM**0.5)
  w1, w2, w3 = (w.bfloat16() for w in (w1, w2, w3))

  def run(block, x_TD, scores_TK, ids_TK, grad_TD):
    block = block.to(device)
    x_TD = x_TD.to(device).requires_grad_()
    scores_TK = scores_TK.to(device).requires_grad_()
    ids_TK = ids_TK.to(device)
    counts_E = torch.bincount(ids_TK.reshape(-1), minlength=_NUM_EXPERTS)
    out = block(x_TD, scores_TK, ids_TK, counts_E)
    weights = (
        block.inner_experts.w1_EDF,
        block.inner_experts.w2_EFD,
        block.inner_experts.w3_EDF,
    )
    grads = torch.autograd.grad(
        out, (x_TD, scores_TK, *weights), grad_TD.to(device)
    )
    return out.cpu(), [grad.cpu() for grad in grads]

  ep_out, ep_grads = run(
      _moe_block(
          num_local, ep_mesh, num_chunks, w1[experts], w2[experts], w3[experts]
      ),
      x[rank],
      scores[rank],
      ids[rank],
      grad_out[rank],
  )
  ref_out, ref_grads = run(
      _moe_block(_NUM_EXPERTS, None, 1, w1, w2, w3),
      x[group].reshape(-1, _DIM),
      scores[group].reshape(-1, _TOP_K),
      ids[group].reshape(-1, _TOP_K),
      grad_out[group].reshape(-1, _DIM),
  )

  local = slice(ep_rank * _NUM_TOKENS, (ep_rank + 1) * _NUM_TOKENS)
  expected = {
      "out": ref_out[local],
      "dx": ref_grads[0][local],
      "dscores": ref_grads[1][local],
      "dw1": ref_grads[2][experts],
      "dw2": ref_grads[3][experts],
      "dw3": ref_grads[4][experts],
  }
  actual = dict(zip(expected, [ep_out, *ep_grads]))
  diffs = {
      name: (
          (actual[name].float() - expected[name].float()).abs().max().item(),
          expected[name].float().abs().max().item(),
      )
      for name in expected
  }
  report = ", ".join(
      f"{name} {d:.3g}/{m:.3g}" for name, (d, m) in diffs.items()
  )
  logging.info("[rank %d] max |EP - no-EP| / max |no-EP|: %s", rank, report)
  for name in expected:
    # Both sides round the same BF16 expert outputs; only the order of the
    # FP32 top-k sums and weight-gradient accumulations can differ.
    atol = 2e-2
    if num_chunks > 1 and name.startswith("dw"):
      # Each chunk's weight gradient leaves the grouped matmul rounded to
      # BF16 before the chunks are summed, where the no-EP block rounds the
      # sum once: allow a couple of BF16 ulps at the gradient's scale.
      atol = max(atol, 2.0**-7 * diffs[name][1])
    torch.testing.assert_close(
        actual[name].float(),
        expected[name].float(),
        rtol=2e-2,
        atol=atol,
        msg=lambda m, name=name: f"{name}: {m}\n[rank {rank}] {report}",
    )
  logging.info("[rank %d] MoE block EP/no-EP parity passed.", rank)


class RingOfExpertsDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):

  @parameterized.named_parameters(
      ("ep_all", 0, 1),
      ("ep_half", 2, 1),
      ("ep_half_chunked", 2, 2),
  )
  def test_matches_single_device_reference(self, ep_divisor, num_chunks):
    # ep_divisor 0 uses every device for EP; 2 makes two EP groups.
    ep_size = self.num_devices // ep_divisor if ep_divisor else self.num_devices
    if ep_size < 2 or _NUM_EXPERTS % ep_size:
      self.skipTest(f"needs an EP degree >= 2 dividing {_NUM_EXPERTS}")
    distributed.run_distributed(
        self.num_devices,
        self.accelerator_device_type,
        _parity_worker,
        ep_size,
        num_chunks,
    )

  @parameterized.named_parameters(
      ("ep_all", 0, 1),
      ("ep_half", 2, 1),
      ("ep_all_chunked", 0, 4),
      ("ep_half_chunked", 2, 4),
  )
  def test_moe_block_matches_no_ep_with_the_same_weights(
      self, ep_divisor, num_chunks
  ):
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("the Pallas grouped matmuls run on TPU only")
    ep_size = self.num_devices // ep_divisor if ep_divisor else self.num_devices
    if ep_size < 2 or _NUM_EXPERTS % ep_size:
      self.skipTest(f"needs an EP degree >= 2 dividing {_NUM_EXPERTS}")
    distributed.run_distributed(
        self.num_devices,
        self.accelerator_device_type,
        _moe_block_parity_worker,
        ep_size,
        num_chunks,
    )


if __name__ == "__main__":
  absltest.main()
