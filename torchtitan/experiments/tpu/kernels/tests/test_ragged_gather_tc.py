"""Unit and accelerator tests for TensorCore ragged gather / gather-reduce kernels."""

from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.experiments.tpu.kernels import ragged_gather_tc


def _relative_l2_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
  return float(
      (actual.double() - expected.double()).norm()
      / expected.double().norm().clamp_min(1e-12)
  )


def _roe_reference(
    x_all_GD: torch.Tensor,
    ids_flat_M: torch.Tensor,
    scores_flat_M: torch.Tensor,
    *,
    ep_rank: int,
    num_experts: int,
    ep_size: int,
    top_k: int,
    weighted: bool,
) -> torch.Tensor:
  """Pure-PyTorch FP32 reference for one EP rank's gather + (weighted) reduce."""
  g_tokens, d_dim = x_all_GD.shape
  e_local = num_experts // ep_size
  lo = ep_rank * e_local
  hi = lo + e_local
  out_f32 = torch.zeros(
      (g_tokens, d_dim), device=x_all_GD.device, dtype=torch.float32
  )
  token_ids_M = (
      torch.arange(ids_flat_M.numel(), device=x_all_GD.device, dtype=torch.long)
      // top_k
  )
  is_local_M = (ids_flat_M >= lo) & (ids_flat_M < hi)
  for slot in is_local_M.nonzero(as_tuple=False).flatten().tolist():
    tok = int(token_ids_M[slot].item())
    row = x_all_GD[tok].float()
    if weighted:
      row = row * scores_flat_M[slot].float()
    out_f32[tok] = out_f32[tok] + row
  return out_f32.to(x_all_GD.dtype)


class RaggedGatherTcTest(base_device_test.BaseAcceleratorDeviceTest):

  def _require_tpu(self) -> None:
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("Pallas ragged_gather_tc kernels require TPU.")

  def test_tpu_generation_device_name_parsing(self):
    self.addCleanup(ragged_gather_tc._tpu_generation.cache_clear)
    cases = (
        ("TPU v5 lite", 5),
        ("TPU v6e", 6),
        ("tpu7x", 7),
        ("TPU-7", 7),
        ("TPU 7", 7),
        ("unknown accelerator", None),
    )
    for dev_name, expected in cases:
      ragged_gather_tc._tpu_generation.cache_clear()
      fake_tpu = SimpleNamespace(
          get_device_properties=lambda n=dev_name: SimpleNamespace(name=n)
      )
      with (
          self.subTest(dev_name=dev_name),
          mock.patch.object(torch, "tpu", fake_tpu, create=True),
      ):
        self.assertEqual(ragged_gather_tc._tpu_generation(), expected)

    ragged_gather_tc._tpu_generation.cache_clear()
    failing_tpu = SimpleNamespace(
        get_device_properties=mock.Mock(side_effect=RuntimeError("no device"))
    )
    with mock.patch.object(torch, "tpu", failing_tpu, create=True):
      self.assertIsNone(ragged_gather_tc._tpu_generation())

  def test_packed_stable_sort_packed_and_wide_key_fallback(self):
    # Empty and singleton inputs.
    empty_k, empty_p = ragged_gather_tc.packed_stable_sort(
        jnp.zeros((0,), dtype=jnp.int32), max_key=7
    )
    self.assertEqual(empty_k.shape, (0,))
    self.assertEqual(empty_p.shape, (0,))

    single_k, single_p = ragged_gather_tc.packed_stable_sort(
        jnp.array([5], dtype=jnp.int32), max_key=7
    )
    np.testing.assert_array_equal(np.asarray(single_k), np.array([5]))
    np.testing.assert_array_equal(np.asarray(single_p), np.array([0]))

    # 1) Packed <= 32-bit path (pos_bits=6, key_bits=3 -> 9 <= 32).
    rng = np.random.default_rng(0)
    keys_small = rng.integers(0, 8, size=64, dtype=np.int32)
    sorted_k, sorted_p = ragged_gather_tc.packed_stable_sort(
        jnp.asarray(keys_small), max_key=7
    )
    expected_p = np.argsort(keys_small, kind="stable")
    np.testing.assert_array_equal(np.asarray(sorted_p), expected_p)
    np.testing.assert_array_equal(np.asarray(sorted_k), keys_small[expected_p])

    # 2) Wide > 32-bit fallback (pos_bits=6, key_bits=29 -> 35 > 32).
    max_wide = (1 << 28) + 7
    palette = np.array([0, 13, 1 << 27, max_wide], dtype=np.int32)
    keys_wide = palette[rng.integers(0, len(palette), size=48)]
    sorted_wk, sorted_wp = ragged_gather_tc.packed_stable_sort(
        jnp.asarray(keys_wide), max_key=int(max_wide)
    )
    expected_wp = np.argsort(keys_wide, kind="stable")
    np.testing.assert_array_equal(np.asarray(sorted_wp), expected_wp)
    np.testing.assert_array_equal(np.asarray(sorted_wk), keys_wide[expected_wp])

    with self.assertRaisesRegex(ValueError, "block_size .* must be > top_k"):
      ragged_gather_tc._ragged_gather_reduce_tc_layout(
          num_indices=64, top_k=8, block_size=8
      )
    empty_meta = ragged_gather_tc.ragged_gather_reduce_tc_metadata(
        jnp.zeros((0,), dtype=jnp.int32),
        0,
        0,
        top_k=4,
        block_size=128,
        num_out_tokens=8,
    )
    self.assertEqual(empty_meta.num_indices, 0)

  def test_multi_block_gather_and_four_buffer_reduce_ring(self):
    """Exercises >= 6 gather steps, >= 8 reduce blocks (4-buffer wrap), and rem > 0."""
    self._require_tpu()
    dev = self.accelerator_device
    torch.manual_seed(42)

    num_experts = 8
    ep_size = 2
    top_k = 4
    num_out_tokens = 320
    d_dim = 256
    gather_block_size = 128
    reduce_block_size = 128

    # Bias ~65% of slots to rank 0 (experts 0..3) so count ~ 830 (> 6 * 128 = 768)
    # and reduce blocks >= 8 (> 4-buffer ring).
    M = num_out_tokens * top_k  # 1280
    ids_list = []
    for t in range(num_out_tokens):
      if t % 3 == 0:
        ids_list.append([0, 1, 2, 3])
      elif t % 3 == 1:
        ids_list.append([0, 2, 4, 6])
      else:
        ids_list.append([1, 3, 5, 7])
    ids_flat_M = torch.tensor(ids_list, device=dev, dtype=torch.int32).reshape(
        -1
    )
    ep_rank_1 = torch.zeros((1,), device=dev, dtype=torch.int32)

    # Cap = 950 leaves rem = 950 % 128 = 54 != 0 and count < cap.
    cap = 950
    (
        local_group_sizes_e,
        token_ids_sorted_M,
        window_slot_order_cap,
        start_count_2,
        src_rows,
        tokens,
        blocks,
    ) = ragged_gather_tc.roe_compute_routing(
        ids_flat_M,
        ep_rank_1,
        num_experts=num_experts,
        ep_size=ep_size,
        top_k=top_k,
        num_out_tokens=num_out_tokens,
        cap=cap,
        reduce_block_size=reduce_block_size,
    )
    count = int(start_count_2[1].item())
    self.assertGreaterEqual(count, 6 * gather_block_size + 1)
    self.assertLess(count, cap)
    self.assertEqual(int(local_group_sizes_e.sum().item()), count)
    # Verify >= 6 active reduce blocks so the 4-buffer ring wraps around.
    num_active_blocks = int((blocks[1::4] > 0).sum().item())
    self.assertGreaterEqual(num_active_blocks, 6)

    x_ref = (
        torch.randn(num_out_tokens, d_dim, device=dev, dtype=torch.bfloat16)
        * 0.25
    ).requires_grad_()
    scores_ref = (
        0.1 + torch.rand(M, device=dev, dtype=torch.float32)
    ).requires_grad_()
    grad_out = (
        torch.randn(num_out_tokens, d_dim, device=dev, dtype=torch.bfloat16)
        * 0.25
    )

    # 1) Weighted gather -> identity -> roe_unsort_combine (with NaN tail rows).
    out_ref = _roe_reference(
        x_ref,
        ids_flat_M,
        scores_ref,
        ep_rank=0,
        num_experts=num_experts,
        ep_size=ep_size,
        top_k=top_k,
        weighted=True,
    )
    dx_ref, dscores_ref = torch.autograd.grad(
        out_ref, (x_ref, scores_ref), grad_out
    )

    x = x_ref.detach().clone().requires_grad_(True)
    scores = scores_ref.detach().clone().requires_grad_(True)
    gathered_RD = ragged_gather_tc.roe_sort_gather(
        x,
        token_ids_sorted_M,
        start_count_2,
        src_rows,
        tokens,
        blocks,
        cap=cap,
        top_k=top_k,
        gather_block_size=gather_block_size,
        reduce_block_size=reduce_block_size,
    )
    # Inject NaNs into [count : cap) while preserving autograd on [:count].
    nan_tail = torch.full(
        (cap - count, d_dim), float("nan"), device=dev, dtype=torch.bfloat16
    )
    routed_with_nan_tail = torch.cat([gathered_RD[:count], nan_tail], dim=0)
    out = ragged_gather_tc.roe_unsort_combine(
        routed_with_nan_tail,
        scores,
        token_ids_sorted_M,
        window_slot_order_cap,
        start_count_2,
        src_rows,
        tokens,
        blocks,
        num_out_tokens=num_out_tokens,
        top_k=top_k,
        gather_block_size=gather_block_size,
        reduce_block_size=reduce_block_size,
    )
    dx, dscores = torch.autograd.grad(out, (x, scores), grad_out)
    self.assertTrue(bool(torch.isfinite(out).all()))
    self.assertTrue(bool(torch.isfinite(dx).all()))
    self.assertTrue(bool(torch.isfinite(dscores).all()))
    self.assertLessEqual(_relative_l2_error(out, out_ref), 1e-2)
    self.assertLessEqual(_relative_l2_error(dx, dx_ref), 1e-2)
    self.assertLessEqual(_relative_l2_error(dscores, dscores_ref), 1e-2)

    # 2) Unweighted roe_unsort_reduce (4-buffer ring without weights + NaN tail).
    x_u_ref = x_ref.detach().clone().requires_grad_(True)
    out_u_ref = _roe_reference(
        x_u_ref,
        ids_flat_M,
        scores_ref.detach(),
        ep_rank=0,
        num_experts=num_experts,
        ep_size=ep_size,
        top_k=top_k,
        weighted=False,
    )
    (dx_u_ref,) = torch.autograd.grad(out_u_ref, (x_u_ref,), grad_out)

    x_u = x_ref.detach().clone().requires_grad_(True)
    gathered_u = ragged_gather_tc.roe_sort_gather(
        x_u,
        token_ids_sorted_M,
        start_count_2,
        src_rows,
        tokens,
        blocks,
        cap=cap,
        top_k=top_k,
        gather_block_size=gather_block_size,
        reduce_block_size=reduce_block_size,
    )
    routed_u = torch.cat([gathered_u[:count], nan_tail], dim=0)
    out_u = ragged_gather_tc.roe_unsort_reduce(
        routed_u,
        token_ids_sorted_M,
        start_count_2,
        src_rows,
        tokens,
        blocks,
        num_out_tokens=num_out_tokens,
        top_k=top_k,
        gather_block_size=gather_block_size,
        reduce_block_size=reduce_block_size,
    )
    (dx_u,) = torch.autograd.grad(out_u, (x_u,), grad_out)
    self.assertTrue(bool(torch.isfinite(out_u).all()))
    self.assertTrue(bool(torch.isfinite(dx_u).all()))
    self.assertLessEqual(_relative_l2_error(out_u, out_u_ref), 1e-2)
    self.assertLessEqual(_relative_l2_error(dx_u, dx_u_ref), 1e-2)

    # 3) Exact count == cap with non-multiple rem (count % 128 != 0) hits the
    # last-block `step_idx == num_blocks - 1` remainder branch in ragged_gather_tc.
    self.assertNotEqual(count % gather_block_size, 0)
    (
        _,
        tok_exact,
        _,
        sc_exact,
        sr_exact,
        tk_exact,
        blk_exact,
    ) = ragged_gather_tc.roe_compute_routing(
        ids_flat_M,
        ep_rank_1,
        num_experts=num_experts,
        ep_size=ep_size,
        top_k=top_k,
        num_out_tokens=num_out_tokens,
        cap=count,
        reduce_block_size=reduce_block_size,
    )
    gathered_exact = ragged_gather_tc.roe_sort_gather(
        x_ref.detach(),
        tok_exact,
        sc_exact,
        sr_exact,
        tk_exact,
        blk_exact,
        cap=count,
        top_k=top_k,
        gather_block_size=gather_block_size,
        reduce_block_size=reduce_block_size,
    )
    torch.testing.assert_close(
        gathered_exact, gathered_RD.detach()[:count], rtol=0, atol=0
    )

  def test_pallas_scale_and_dscore_full_and_partial_blocks_mask_tail_nans(self):
    """Exercises Pallas _ragged_scale_and_dscore_tc _full_block + _partial_block."""
    self._require_tpu()
    dev = self.accelerator_device
    torch.manual_seed(7)

    # cap = 2048 (multiple of 1024) enables the Pallas _scale_dscore_kernel.
    # Route exactly 1337 slots to rank 0 so block 0 (0..1024) runs _full_block
    # and block 1 (1024..1337) runs _partial_block with NaNs in [1337:2048).
    cap = 2048
    num_out_tokens = 512
    top_k = 4
    num_experts = 8
    ep_size = 2
    d_dim = 256
    target_count = 1337

    M = num_out_tokens * top_k  # 2048
    ids_np = np.full(M, 4, dtype=np.int32)  # rank 1 expert
    ids_np[:target_count] = np.arange(target_count, dtype=np.int32) % 4
    # Permute within each token's top_k=4 so no token has duplicate expert IDs
    # if we construct per-token distinct choices: 334 tokens get 4 local experts
    # (1336) and token 334 gets 1 local + 3 remote experts (total = 1337).
    ids_grid = np.zeros((num_out_tokens, top_k), dtype=np.int32)
    ids_grid[:334] = np.array([0, 1, 2, 3], dtype=np.int32)
    ids_grid[334] = np.array([0, 4, 5, 6], dtype=np.int32)
    ids_grid[335:] = np.array([4, 5, 6, 7], dtype=np.int32)
    del ids_np
    ids_flat_M = torch.tensor(
        ids_grid.reshape(-1), device=dev, dtype=torch.int32
    )
    ep_rank_1 = torch.zeros((1,), device=dev, dtype=torch.int32)

    (
        _,
        token_ids_sorted_M,
        window_slot_order_cap,
        start_count_2,
        src_rows,
        tokens,
        blocks,
    ) = ragged_gather_tc.roe_compute_routing(
        ids_flat_M,
        ep_rank_1,
        num_experts=num_experts,
        ep_size=ep_size,
        top_k=top_k,
        num_out_tokens=num_out_tokens,
        cap=cap,
        reduce_block_size=896,
    )
    self.assertEqual(int(start_count_2[1].item()), target_count)

    valid_routed = (
        torch.randn(target_count, d_dim, device=dev, dtype=torch.bfloat16) * 0.3
    )
    nan_tail = torch.full(
        (cap - target_count, d_dim),
        float("nan"),
        device=dev,
        dtype=torch.bfloat16,
    )
    routed = torch.cat([valid_routed, nan_tail], dim=0).requires_grad_()
    routed_clean = torch.cat(
        [valid_routed, torch.zeros_like(nan_tail)], dim=0
    ).requires_grad_()

    scores = (
        0.1 + torch.rand(M, device=dev, dtype=torch.float32)
    ).requires_grad_()
    scores_ref = scores.detach().clone().requires_grad_(True)
    grad_out = (
        torch.randn(num_out_tokens, d_dim, device=dev, dtype=torch.bfloat16)
        * 0.3
    )

    out = ragged_gather_tc.roe_unsort_combine(
        routed,
        scores,
        token_ids_sorted_M,
        window_slot_order_cap,
        start_count_2,
        src_rows,
        tokens,
        blocks,
        num_out_tokens=num_out_tokens,
        top_k=top_k,
        zero_pad_tail=False,
    )
    drouted, dscores = torch.autograd.grad(out, (routed, scores), grad_out)

    # Compare against zero_pad_tail=True (pure-JAX reference branch in
    # _ragged_scale_and_dscore_tc) on zero-padded clean inputs.
    out_ref = ragged_gather_tc.roe_unsort_combine(
        routed_clean,
        scores_ref,
        token_ids_sorted_M,
        window_slot_order_cap,
        start_count_2,
        src_rows,
        tokens,
        blocks,
        num_out_tokens=num_out_tokens,
        top_k=top_k,
        zero_pad_tail=True,
    )
    drouted_ref, dscores_ref = torch.autograd.grad(
        out_ref, (routed_clean, scores_ref), grad_out
    )

    self.assertTrue(bool(torch.isfinite(out).all()))
    self.assertTrue(bool(torch.isfinite(drouted[:target_count]).all()))
    self.assertTrue(bool(torch.isfinite(dscores).all()))
    torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        drouted[:target_count], drouted_ref[:target_count], rtol=1e-3, atol=1e-3
    )
    self.assertLessEqual(_relative_l2_error(dscores, dscores_ref), 1e-4)

  def test_roe_gather_row_scales_parity_nan_masking_and_fx_trace(self):
    self._require_tpu()
    dev = self.accelerator_device
    torch.manual_seed(99)

    M = 64
    cap = 128
    count = 37
    perm = torch.randperm(M, device=dev, dtype=torch.int32)
    window_slot_order_cap = torch.zeros(cap, device=dev, dtype=torch.int32)
    window_slot_order_cap[:count] = perm[:count]
    start_count_2 = torch.tensor([0, count], device=dev, dtype=torch.int32)

    for score_dtype in (torch.float32, torch.bfloat16):
      with self.subTest(score_dtype=score_dtype):
        scores = (
            0.2 + torch.rand(M, device=dev, dtype=score_dtype)
        ).requires_grad_()
        scores_ref = scores.detach().clone().requires_grad_(True)

        row_scales = ragged_gather_tc.roe_gather_row_scales(
            scores, window_slot_order_cap, start_count_2
        )
        self.assertEqual(row_scales.dtype, torch.float32)
        self.assertEqual(row_scales.shape, (cap,))

        valid_mask = torch.arange(cap, device=dev) < count
        expected_scales = torch.where(
            valid_mask,
            scores_ref.float().index_select(0, window_slot_order_cap.long()),
            torch.zeros((), device=dev, dtype=torch.float32),
        )
        torch.testing.assert_close(row_scales, expected_scales, rtol=0, atol=0)

        grad_valid = torch.randn(count, device=dev, dtype=torch.float32)
        grad_with_nan = torch.cat(
            [
                grad_valid,
                torch.full(
                    (cap - count,),
                    float("nan"),
                    device=dev,
                    dtype=torch.float32,
                ),
            ],
            dim=0,
        )
        grad_clean = torch.cat(
            [
                grad_valid,
                torch.zeros(cap - count, device=dev, dtype=torch.float32),
            ],
            dim=0,
        )
        (dscores,) = torch.autograd.grad(row_scales, (scores,), grad_with_nan)
        (dscores_ref,) = torch.autograd.grad(
            expected_scales, (scores_ref,), grad_clean
        )
        self.assertEqual(dscores.dtype, score_dtype)
        self.assertTrue(bool(torch.isfinite(dscores).all()))
        torch.testing.assert_close(dscores, dscores_ref, rtol=0, atol=0)

    # Zero valid rows (count == 0).
    zero_sc = torch.tensor([0, 0], device=dev, dtype=torch.int32)
    scores_z = torch.ones(
        M, device=dev, dtype=torch.float32, requires_grad=True
    )
    scales_z = ragged_gather_tc.roe_gather_row_scales(
        scores_z, window_slot_order_cap, zero_sc
    )
    (dscores_z,) = torch.autograd.grad(
        scales_z,
        (scores_z,),
        torch.full((cap,), float("nan"), device=dev, dtype=torch.float32),
    )
    torch.testing.assert_close(
        scales_z, torch.zeros_like(scales_z), rtol=0, atol=0
    )
    torch.testing.assert_close(
        dscores_z, torch.zeros_like(dscores_z), rtol=0, atol=0
    )

    # Verify make_fx fake-tensor trace lowers cleanly with zero index_add ops.
    def _step(s, w, sc):
      r = ragged_gather_tc.roe_gather_row_scales(s, w, sc)
      (ds,) = torch.autograd.grad(r.sum(), (s,))
      return r, ds

    gm = make_fx(_step, tracing_mode="fake")(
        scores_z, window_slot_order_cap, start_count_2
    )
    op_names = [
        str(n.target) for n in gm.graph.nodes if n.op == "call_function"
    ]
    self.assertFalse(any("index_add" in name for name in op_names))
    self.assertFalse(any("local_scalar_dense" in name for name in op_names))


if __name__ == "__main__":
  absltest.main()
