"""Tests for Tokamax splash attention on TPU."""

from collections.abc import Callable
import functools
import logging
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.nn import attention
import torch.nn.functional as F
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.experiments.tpu.distributed.headtail import headtail_positions
from torchtitan.experiments.tpu.kernels import splash_tbwd
from torchtitan.experiments.tpu.kernels import tokamax_splash_attention


def _packed_segment_ids(document_lengths_per_row) -> torch.Tensor:
  """(B, S) int32 segment ids of rows packing documents of the given lengths."""
  return torch.stack([
      torch.repeat_interleave(
          torch.arange(len(lengths), dtype=torch.int32), torch.tensor(lengths)
      )
      for lengths in document_lengths_per_row
  ])


# Packed rows of 2048 tokens whose document boundaries fall inside head-tail
# chunks and 128-token blocks.
_DOCUMENTS = ((300, 517, 77, 1154), (1000, 48, 1000))
# KV all-gather CP shards: one KV block, then shards spanning several Q and KV
# blocks with GQA / MQA, packed documents and a window. (Tokamax has only the
# fused backward: SplashConfig rejects use_fused_bwd_kernel=False.)
_CP_SHARD_CASES = (
    dict(testcase_name="cp2", cp_size=2),
    dict(testcase_name="cp4", cp_size=4),
    dict(testcase_name="cp2_window", cp_size=2, local_window_size=200),
    dict(testcase_name="cp4_window", cp_size=4, local_window_size=200),
    dict(
        testcase_name="cp2_blocks_gqa_documents",
        cp_size=2,
        seq_len=2048,
        n_heads=4,
        n_kv_heads=2,
        block=128,
        documents=_DOCUMENTS,
    ),
    dict(
        testcase_name="cp4_blocks_mqa_window_documents",
        cp_size=4,
        seq_len=2048,
        n_heads=4,
        n_kv_heads=1,
        block=128,
        local_window_size=64,
        documents=_DOCUMENTS,
    ),
    dict(
        testcase_name="cp4_blocks_documents",
        cp_size=4,
        seq_len=2048,
        n_heads=4,
        n_kv_heads=4,
        block=128,
        documents=_DOCUMENTS,
    ),
)


class HeadTailMaskTest(parameterized.TestCase):
  """KV all-gather CP masks: stock Tokamax masks with head-tail Q rows."""

  @parameterized.product(cp_size=(1, 2, 4), local_window_size=(None, 5))
  def test_mask_matches_dense_reference(self, cp_size, local_window_size):
    seq_len = 32
    mask = tokamax_splash_attention._make_mask(
        (seq_len, seq_len), True, local_window_size, cp_size
    )
    # Row i holds the query at original position headtail_positions(...)[i].
    q_pos = headtail_positions(seq_len, cp_size)[:, None]
    kv_pos = np.arange(seq_len)[None, :]
    expected = q_pos >= kv_pos
    if local_window_size is not None:
      expected &= q_pos - kv_pos <= local_window_size
    np.testing.assert_array_equal(mask[:, :], expected)

  @parameterized.parameters(None, 5)
  def test_mask_equality_follows_q_order(self, local_window_size):
    make_mask = functools.partial(
        tokamax_splash_attention._make_mask, (32, 32), True, local_window_size
    )
    mask, same = make_mask(cp_size=2), make_mask(cp_size=2)
    self.assertEqual(mask, same)
    self.assertEqual(hash(mask), hash(same))
    for cp_size in (1, 4):
      self.assertNotEqual(mask, make_mask(cp_size=cp_size))

  def test_non_causal_mask_is_full(self):
    mask = tokamax_splash_attention._make_mask((32, 32), False, None, 2)
    self.assertTrue(np.all(mask[:, :]))


def _fake_tpu_info(generation=7, mxu_column_size=256, vmem_capacity_bytes=64 << 20, smem_capacity_bytes=1 << 20):
  return types.SimpleNamespace(
      generation=generation,
      mxu_column_size=mxu_column_size,
      vmem_capacity_bytes=vmem_capacity_bytes,
      smem_capacity_bytes=smem_capacity_bytes,
  )


class SplashTbwdSupportTest(parameterized.TestCase):
  """splash_tbwd.unsupported_reason: hard caps and the VMEM / SMEM estimates (no device needed).

  Runs without TPU info, so the limits are the TPU7x defaults (60 MiB VMEM, 1 MiB - 16 KiB SMEM).
  """

  def setUp(self):
    super().setUp()
    self.real_tpu_info = splash_tbwd._tpu_info  # pylint: disable=protected-access
    patcher = mock.patch.object(splash_tbwd, "_tpu_info", return_value=None)
    patcher.start()
    self.addCleanup(patcher.stop)

  @parameterized.named_parameters(
      # (B, S, H, N, D)
      ("qwen3_235b", (16, 4096, 64, 4, 128)),
      ("group1_dim64", (2, 4096, 16, 16, 64)),
      ("group16_seq10240", (1, 10240, 32, 2, 128)),
      ("muse_glimmer_seq16384", (1, 16384, 32, 2, 128)),
      ("tile256", (1, 4352, 8, 2, 128)),
      ("dim512_group1", (1, 4096, 4, 4, 512)),
  )
  def test_supported(self, shape):
    b, s, h, n, d = shape
    self.assertIsNone(
        splash_tbwd.unsupported_reason((b, s, h, d), (b, s, n, d), (b, s, n, d), jnp.dtype(jnp.bfloat16))
    )

  @parameterized.named_parameters(
      ("float32", (2, 4096, 16, 1, 128), "float32", "bf16"),
      ("group32", (2, 4096, 32, 1, 128), "bfloat16", "q heads per kv head"),
      ("dim48", (2, 4096, 16, 1, 48), "bfloat16", "head_dim"),
      ("seq4100", (2, 4100, 16, 1, 128), "bfloat16", "multiple of 128"),
      ("vmem_dim256_group16", (2, 4096, 16, 1, 256), "bfloat16", "VMEM"),
      ("vmem_seq17920", (1, 17920, 32, 2, 128), "bfloat16", "VMEM"),
      ("smem_batch64_seq16384", (64, 16384, 4, 4, 64), "bfloat16", "SMEM"),
  )
  def test_unsupported(self, shape, dtype, expected):
    b, s, h, n, d = shape
    reason = splash_tbwd.unsupported_reason((b, s, h, d), (b, s, n, d), (b, s, n, d), jnp.dtype(dtype))
    self.assertIn(expected, reason or "")

  @parameterized.named_parameters(
      # (B, S, H, N, D), expected (stream_kv, single_buffer_kv)
      ("qwen3_235b", (16, 4096, 64, 4, 128), (False, False)),
      ("group16_seq10240", (1, 10240, 32, 2, 128), (False, True)),
      ("muse_glimmer_seq16384", (1, 16384, 32, 2, 128), (True, True)),
      ("group16_dim64_seq32768", (1, 32768, 32, 2, 64), (True, False)),
  )
  def test_kv_modes(self, shape, expected):
    """K / V stay resident (the old, slightly faster layout) unless only streaming them fits."""
    b, s, h, n, d = shape
    block = splash_tbwd.block_size(s)
    self.assertEqual(splash_tbwd._kv_modes(d, h // n, s, block, None), expected)  # pylint: disable=protected-access

  @parameterized.named_parameters(("v_dim64", 64), ("v_dim256", 256))
  def test_v_head_dim_must_match(self, v_head_dim):
    """The V / O / dO / dV blocks are sized from q's head dim: a different V head dim gave NaN or a truncated dV."""
    b, s, h, n, d = 1, 256, 2, 1, 128
    reason = splash_tbwd.unsupported_reason((b, s, h, d), (b, s, n, d), (b, s, n, v_head_dim), jnp.dtype(jnp.bfloat16))
    self.assertIn("needs v of k's shape", reason or "")

  def test_limits_follow_tpu_info(self):
    """The VMEM / SMEM limits are the attached core's capacities less fixed headroom."""
    # Needs ~49 MiB of VMEM streamed (fits TPU7x's 60 MiB limit, not a 44 MiB one) and a 6.6 KiB schedule.
    shape = ((1, 10240, 32, 128), (1, 10240, 2, 128), (1, 10240, 2, 128), jnp.dtype(jnp.bfloat16))
    for info, expected in (
        (_fake_tpu_info(), None),
        (_fake_tpu_info(vmem_capacity_bytes=48 << 20), "VMEM"),
        (_fake_tpu_info(smem_capacity_bytes=20 << 10), "SMEM"),
    ):
      with mock.patch.object(splash_tbwd, "_tpu_info", return_value=info):
        reason = splash_tbwd.unsupported_reason(*shape)
      if expected is None:
        self.assertIsNone(reason)
      else:
        self.assertIn(expected, reason or "")
    with mock.patch.object(splash_tbwd, "_tpu_info", return_value=_fake_tpu_info()):
      self.assertEqual(splash_tbwd._vmem_limit_bytes(), 60 << 20)  # pylint: disable=protected-access
      self.assertEqual(splash_tbwd._smem_budget_bytes(), (1 << 20) - (16 << 10))  # pylint: disable=protected-access

  @parameterized.named_parameters(
      ("tpu7x", _fake_tpu_info(), None),
      ("v6e_not_validated", _fake_tpu_info(generation=6), "validated on TPU generations"),
      ("v5e_mxu128", _fake_tpu_info(generation=5, mxu_column_size=128), "256-wide MXU"),
      ("no_tpu_info", None, "no TPU info"),
  )
  def test_hardware_gate(self, info, expected):
    with mock.patch.object(splash_tbwd, "_tpu_info", return_value=info):
      reason = splash_tbwd.hardware_unsupported_reason()
    if expected is None:
      self.assertIsNone(reason)
    else:
      self.assertIn(expected, reason or "")

  @parameterized.named_parameters(
      ("supported", 128, _fake_tpu_info(), None),
      ("v_dim64", 64, _fake_tpu_info(), "needs v of k's shape"),
      ("v6e", 128, _fake_tpu_info(generation=6), "validated on TPU generations"),
  )
  def test_wrapper_gate(self, v_head_dim, info, expected):
    """tokamax_splash_attention routes to splash_tbwd only with v of k's shape, on validated hardware."""
    q = jax.ShapeDtypeStruct((1, 256, 2, 128), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((1, 256, 1, 128), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, 256, 1, v_head_dim), jnp.bfloat16)
    with mock.patch.object(splash_tbwd, "_tpu_info", return_value=info):
      reason = tokamax_splash_attention._transposed_bwd_unsupported_reason(  # pylint: disable=protected-access
          q, k, v, is_causal=True
      )
    if expected is None:
      self.assertIsNone(reason)
    else:
      self.assertIn(expected, reason or "")

  def test_tpu_info_unavailable(self):
    """Without TPU info, pltpu.get_tpu_info raises and _tpu_info returns None (TPU7x default limits)."""
    with mock.patch.object(
        splash_tbwd.pltpu, "get_tpu_info", side_effect=ValueError("Unsupported TPU device kind: cpu")
    ):
      self.assertIsNone(self.real_tpu_info())
    self.assertEqual(splash_tbwd._vmem_limit_bytes(), 60 << 20)  # pylint: disable=protected-access


class SplashTbwdScheduleTest(parameterized.TestCase):
  """splash_tbwd.build_schedule against a brute-force tile list (no device)."""

  @staticmethod
  def _reference(seg, block, window=None):
    """Active tiles in (b, qi, kj) order, their kinds and accumulator bounds."""
    num_b, s = seg.shape
    nb = s // block
    blocks = seg.reshape(num_b, nb, block)
    tiles = []
    for b in range(num_b):
      for qi in range(nb):
        for kj in range(qi + 1):
          q_ids, kv_ids = blocks[b, qi], blocks[b, kj]
          if q_ids.max() < kv_ids.min() or kv_ids.max() < q_ids.min():
            continue
          # q - k over the tile, from qi * block - (kj * block + block - 1) to its maximum.
          min_dist = qi * block - (kj * block + block - 1)
          max_dist = qi * block + block - 1 - kj * block
          if window is not None and min_dist > window:
            continue
          single_doc = (
              q_ids.min() == q_ids.max() == kv_ids.min() == kv_ids.max()
          )
          full = kj < qi and single_doc and (window is None or max_dist <= window)
          tiles.append((b, qi, kj, 1 if full else 2))
    bounds = {}
    for name, run in (("q", lambda t: t[:2]), ("b", lambda t: t[0])):
      runs = [run(t) for t in tiles]
      bounds[f"{name}_bounds_start"] = [
          int(i == 0 or runs[i - 1] != r) for i, r in enumerate(runs)
      ]
      bounds[f"{name}_bounds_end"] = [
          int(i == len(runs) - 1 or runs[i + 1] != r)
          for i, r in enumerate(runs)
      ]
    return tiles, bounds

  @parameterized.named_parameters(
      ("packed_512", _DOCUMENTS, 512),
      ("packed_256", ((300, 517, 77, 898), (1000, 48, 744)), 256),
      ("one_document", ((2048,), (2048,)), 512),
      ("unsorted_ids", None, 128),
      ("window_200_block128", _DOCUMENTS, 128, 200),
      ("window_1023_block256", ((2048,), (2048,)), 256, 1023),
      ("window_511_block512", _DOCUMENTS, 512, 511),
  )
  def test_matches_brute_force(self, documents, block, window=None):
    if documents is None:
      # Ids that repeat out of order: the [min, max] test stays conservative.
      seg = np.array([[0] * 128 + [1] * 128 + [0] * 128 + [2] * 128], np.int32)
    else:
      seg = _packed_segment_ids(documents).numpy()
    sched = {
        k: np.asarray(v)
        for k, v in splash_tbwd.build_schedule(jnp.asarray(seg), block, window).items()
    }
    tiles, bounds = self._reference(seg, block, window)
    n = len(tiles)
    num_b, s = seg.shape
    self.assertEqual(int(sched["num_active"]), n)
    self.assertLen(sched["b"], splash_tbwd._num_steps(num_b, s, block))
    got = list(
        zip(*(sched[k][:n].tolist() for k in ("b", "qi", "kj", "kind")))
    )
    self.assertEqual(got, tiles)
    for name, want in bounds.items():
      self.assertEqual(sched[name][:n].tolist(), want, name)
    # Padding steps: kind 0, all bounds 0.
    self.assertTrue(np.all(sched["kind"][n:] == 0))
    for name in bounds:
      self.assertTrue(np.all(sched[name][n:] == 0), name)


class SplashAttentionTokamaxTest(base_device_test.BaseAcceleratorDeviceTest):

  def check_forward_parity(
      self,
      splash_fn: Callable[
          [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
      ],
      b: int,
      n_heads: int,
      n_kv_heads: int,
      seq_len: int,
      head_dim: int,
  ):
    device = self.accelerator_device

    # Generates inputs in (B, S, H, D) layout standard
    q = torch.randn(b, seq_len, n_heads, head_dim, dtype=torch.float32)
    k = torch.randn(b, seq_len, n_kv_heads, head_dim, dtype=torch.float32)
    v = torch.randn(b, seq_len, n_kv_heads, head_dim, dtype=torch.float32)
    q_tpu, k_tpu, v_tpu = q.to(device), k.to(device), v.to(device)

    # Expands kv_head dim if necessary (head dim is at axis 2 now)
    n_rep = n_heads // n_kv_heads
    k_exp = k_tpu.repeat_interleave(n_rep, dim=2)
    v_exp = v_tpu.repeat_interleave(n_rep, dim=2)

    # Reference SDPA Math expects (B, H, S, D)
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      expected_out = F.scaled_dot_product_attention(
          q_tpu.transpose(1, 2),
          k_exp.transpose(1, 2),
          v_exp.transpose(1, 2),
          is_causal=True,
      )
      expected_out = expected_out.transpose(1, 2)

    actual_out = splash_fn(q_tpu, k_tpu, v_tpu)

    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=5e-2, atol=5e-2
    )

  def test_mha_parity(self):
    splash_fn = functools.partial(
        tokamax_splash_attention.splash_sdpa, is_causal=True, enable_gqa=False
    )

    self.check_forward_parity(
        splash_fn=splash_fn,
        b=2,
        n_heads=8,
        n_kv_heads=8,
        seq_len=128,
        head_dim=64,
    )

  def test_mqa_parity(self):
    splash_fn = functools.partial(
        tokamax_splash_attention.splash_sdpa, is_causal=True, enable_gqa=True
    )

    self.check_forward_parity(
        splash_fn=splash_fn,
        b=2,
        n_heads=8,
        n_kv_heads=2,
        seq_len=128,
        head_dim=64,
    )

  def test_mqa_parity_one_kv_head(self):
    splash_fn = functools.partial(
        tokamax_splash_attention.splash_sdpa, is_causal=True, enable_gqa=True
    )

    self.check_forward_parity(
        splash_fn=splash_fn,
        b=2,
        n_heads=8,
        n_kv_heads=1,
        seq_len=128,
        head_dim=64,
    )

  def test_block_sizes_parity(self):
    # Block sizes reach this kernel through the process-global TokamaxConfig, not
    # through per-call kwargs - splash_sdpa never forwards the per-call values. This
    # test previously passed them per-call, so it ran at the 512 defaults and
    # asserted nothing about block sizes; set them the way the kernel actually reads
    # them, and restore afterwards since the config is process-global.
    cfg = tokamax_splash_attention.TokamaxConfig
    saved = (cfg.block_q, cfg.block_kv, cfg.block_kv_compute)
    self.addCleanup(
        lambda: cfg.update(
            block_q=saved[0], block_kv=saved[1], block_kv_compute=saved[2]
        )
    )
    cfg.update(block_q=128, block_kv=128, block_kv_compute=128)

    splash_fn = functools.partial(
        tokamax_splash_attention.splash_sdpa,
        is_causal=True,
        enable_gqa=False,
    )

    self.check_forward_parity(
        splash_fn=splash_fn,
        b=2,
        n_heads=8,
        n_kv_heads=8,
        seq_len=128,
        head_dim=64,
    )

  @parameterized.named_parameters(
      ("gqa_shapes_flag_false_raises", 8, 2, False, True),
      ("gqa_shapes_flag_true_ok", 8, 2, True, False),
      ("mha_shapes_flag_false_ok", 8, 8, False, False),
      ("mha_shapes_flag_true_raises", 8, 8, True, True),
  )
  def test_enable_gqa_is_validated(self, n_heads, n_kv_heads, flag, should_raise):
    """enable_gqa must be enforced, not ignored.

    The stock splash_sdpa enforces `enable_gqa == (n_heads != n_kv_heads)`, and the
    two kernels are interchangeable via a config flag - so accepting a wrong value
    here would make them disagree on identical input: stock raises, tokamax runs on.
    Shapes are token-major (B, S, H, D); validation happens before any device work.
    """
    b, seq_len, head_dim = 2, 128, 64
    mk = lambda h: torch.zeros(b, seq_len, h, head_dim, dtype=torch.bfloat16)
    q, k, v = mk(n_heads), mk(n_kv_heads), mk(n_kv_heads)

    if should_raise:
      with self.assertRaisesRegex(ValueError, "enable_gqa"):
        tokamax_splash_attention.splash_sdpa(q, k, v, enable_gqa=flag)
    else:
      try:
        tokamax_splash_attention.splash_sdpa(q, k, v, enable_gqa=flag)
      except ValueError as e:
        self.assertNotIn("enable_gqa", str(e))
      except Exception:  # pylint: disable=broad-except
        pass  # got past validation; device/kernel errors are not this test's concern

  def test_custom_scale_backward(self):
    """Regression: dq must carry the chain-rule factor for the custom-scale
    q pre-multiply in the backward (it was off by default_scale/scale)."""
    device = self.accelerator_device
    torch.manual_seed(11)
    b, s, h, d = 2, 128, 4, 64
    scale = 0.05
    q = torch.randn(b, s, h, d, dtype=torch.float32)
    k = torch.randn(b, s, h, d, dtype=torch.float32)
    v = torch.randn(b, s, h, d, dtype=torch.float32)

    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    v_ref = v.clone().requires_grad_(True)
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          q_ref.transpose(1, 2),
          k_ref.transpose(1, 2),
          v_ref.transpose(1, 2),
          is_causal=True,
          scale=scale,
      )
    out_ref.sum().backward()

    q_t = q.to(device).requires_grad_(True)
    k_t = k.to(device).requires_grad_(True)
    v_t = v.to(device).requires_grad_(True)
    out = tokamax_splash_attention.splash_sdpa(
        q_t, k_t, v_t, is_causal=True, scale=scale
    )
    out.sum().backward()

    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad),
        ("dk", k_t.grad, k_ref.grad),
        ("dv", v_t.grad, v_ref.grad),
    ):
      self.assertIsNotNone(got, f"{name} is None")
      torch.testing.assert_close(
          got.cpu(), want, rtol=5e-2, atol=5e-2, msg=name
      )

  def test_gpu_available(self):
    device = self.accelerator_device
    self.assertEqual(device.type, "tpu")

  def test_mha_backward(self):
    """Gradients from splash attention backward match reference SDPA."""
    device = self.accelerator_device
    b, n_heads, seq_len, head_dim = 2, 8, 128, 64

    # (B, S, H, D)
    q = torch.randn(
        b, seq_len, n_heads, head_dim, dtype=torch.float32, device=device
    )
    k = torch.randn(
        b, seq_len, n_heads, head_dim, dtype=torch.float32, device=device
    )
    v = torch.randn(
        b, seq_len, n_heads, head_dim, dtype=torch.float32, device=device
    )

    # Reference backward via math SDPA (expects B, H, S, D).
    q_ref = q.transpose(1, 2).clone().requires_grad_(True)
    k_ref = k.transpose(1, 2).clone().requires_grad_(True)
    v_ref = v.transpose(1, 2).clone().requires_grad_(True)
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          q_ref, k_ref, v_ref, is_causal=True
      )
    out_ref.sum().backward()

    # Splash backward.
    q_test = q.clone().requires_grad_(True)
    k_test = k.clone().requires_grad_(True)
    v_test = v.clone().requires_grad_(True)
    out_test = tokamax_splash_attention.splash_sdpa(
        q_test, k_test, v_test, is_causal=True, enable_gqa=False
    )

    out_test.sum().backward()

    # Confirm the backward kernel was invoked and produced non-zero gradients.
    self.assertIsNotNone(
        q_test.grad, "grad_q is None — splash backward not called"
    )
    self.assertIsNotNone(
        k_test.grad, "grad_k is None — splash backward not called"
    )
    self.assertIsNotNone(
        v_test.grad, "grad_v is None — splash backward not called"
    )
    self.assertGreater(q_test.grad.abs().sum().item(), 0, "grad_q is all zeros")
    self.assertGreater(k_test.grad.abs().sum().item(), 0, "grad_k is all zeros")
    self.assertGreater(v_test.grad.abs().sum().item(), 0, "grad_v is all zeros")

    # Transpose the reference gradients back to compare with our raw (B, S, H, D) gradients!
    torch.testing.assert_close(
        q_test.grad.cpu(),
        q_ref.grad.transpose(1, 2).cpu(),
        rtol=5e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        k_test.grad.cpu(),
        k_ref.grad.transpose(1, 2).cpu(),
        rtol=5e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        v_test.grad.cpu(),
        v_ref.grad.transpose(1, 2).cpu(),
        rtol=5e-2,
        atol=5e-2,
    )
    logging.info("MHA backward test passed.")

  def test_mqa_backward(self):
    """Gradients from splash GQA backward match reference SDPA."""
    device = self.accelerator_device
    b, n_heads, n_kv_heads, seq_len, head_dim = 2, 8, 2, 128, 64
    n_rep = n_heads // n_kv_heads

    # (B, S, H, D)
    q = torch.randn(
        b, seq_len, n_heads, head_dim, dtype=torch.float32, device=device
    )
    k = torch.randn(
        b, seq_len, n_kv_heads, head_dim, dtype=torch.float32, device=device
    )
    v = torch.randn(
        b, seq_len, n_kv_heads, head_dim, dtype=torch.float32, device=device
    )

    # Reference backward (expand k/v to full heads and transpose to B, H, S, D).
    q_ref = q.transpose(1, 2).clone().requires_grad_(True)
    k_ref = k.transpose(1, 2).clone().requires_grad_(True)
    v_ref = v.transpose(1, 2).clone().requires_grad_(True)
    k_exp = k_ref.repeat_interleave(n_rep, dim=1)
    v_exp = v_ref.repeat_interleave(n_rep, dim=1)
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          q_ref, k_exp, v_exp, is_causal=True
      )
    out_ref.sum().backward()
    k_ref_grad = k_ref.grad.transpose(1, 2)
    v_ref_grad = v_ref.grad.transpose(1, 2)

    # Splash GQA backward.
    q_test = q.clone().requires_grad_(True)
    k_test = k.clone().requires_grad_(True)
    v_test = v.clone().requires_grad_(True)
    out_test = tokamax_splash_attention.splash_sdpa(
        q_test, k_test, v_test, is_causal=True, enable_gqa=True
    )

    out_test.sum().backward()

    self.assertIsNotNone(
        q_test.grad, "grad_q is None — splash backward not called"
    )
    self.assertIsNotNone(
        k_test.grad, "grad_k is None — splash backward not called"
    )
    self.assertIsNotNone(
        v_test.grad, "grad_v is None — splash backward not called"
    )
    self.assertGreater(q_test.grad.abs().sum().item(), 0, "grad_q is all zeros")
    self.assertGreater(k_test.grad.abs().sum().item(), 0, "grad_k is all zeros")
    self.assertGreater(v_test.grad.abs().sum().item(), 0, "grad_v is all zeros")

    torch.testing.assert_close(
        q_test.grad.cpu(),
        q_ref.grad.transpose(1, 2).cpu(),
        rtol=5e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        k_test.grad.cpu(), k_ref_grad.cpu(), rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(
        v_test.grad.cpu(), v_ref_grad.cpu(), rtol=5e-2, atol=5e-2
    )
    logging.info("MQA/GQA backward test passed.")

  def _set_tokamax_config(self, **values):
    cfg = tokamax_splash_attention.TokamaxConfig
    for name, value in values.items():
      self.addCleanup(setattr, cfg, name, getattr(cfg, name))
      setattr(cfg, name, value)

  @parameterized.named_parameters(
      ("mha_scan_bwd", 8, dict(use_vmap_bwd=False)),
      ("mha_vmap_bwd", 8, dict(use_vmap_bwd=True)),
      ("gqa_scan_bwd", 2, dict(use_vmap_bwd=False)),
      ("gqa_vmap_bwd", 2, dict(use_vmap_bwd=True)),
      ("mqa_scan_bwd", 1, dict(use_vmap_bwd=False)),
      ("mqa_vmap_bwd", 1, dict(use_vmap_bwd=True)),
      ("gqa_vmap_bwd_stacked", 2, dict(use_vmap_bwd=True, num_stacked_q_heads=4)),
      ("gqa_loop_fwd_stacked", 2, dict(use_vmap_bwd=True, use_batch_loop_fwd=True, num_stacked_q_heads=4)),
      ("mqa_loop_fwd", 1, dict(use_vmap_bwd=True, use_batch_loop_fwd=True)),
  )
  def test_segment_ids_parity(self, n_kv_heads, config):
    """Packed-document forward and gradients match a block-causal SDPA reference."""
    device = self.accelerator_device
    torch.manual_seed(7)
    b, n_heads, seq_len, head_dim = 2, 8, 256, 64
    n_rep = n_heads // n_kv_heads
    self._set_tokamax_config(**config)

    # (B, S, H, D). Row 0 packs two documents, row 1 packs four with a
    # boundary inside a 128-token block, so a kernel that only honored the
    # static causal mask would visibly disagree with the reference.
    q = torch.randn(b, seq_len, n_heads, head_dim, dtype=torch.float32)
    k = torch.randn(b, seq_len, n_kv_heads, head_dim, dtype=torch.float32)
    v = torch.randn(b, seq_len, n_kv_heads, head_dim, dtype=torch.float32)
    segment_ids = torch.zeros(b, seq_len, dtype=torch.int32)
    segment_ids[0, 128:] = 1
    segment_ids[1, 40:100] = 1
    segment_ids[1, 100:200] = 2
    segment_ids[1, 200:] = 3

    # (B, 1, S, S): same document AND causal.
    document_mask = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    attn_mask = (document_mask & causal_mask).unsqueeze(1)

    q_ref = q.transpose(1, 2).clone().requires_grad_(True)
    k_ref = k.transpose(1, 2).clone().requires_grad_(True)
    v_ref = v.transpose(1, 2).clone().requires_grad_(True)
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          q_ref,
          k_ref.repeat_interleave(n_rep, dim=1),
          v_ref.repeat_interleave(n_rep, dim=1),
          attn_mask=attn_mask,
          is_causal=False,
      )
    out_ref.sum().backward()

    q_t = q.to(device).requires_grad_(True)
    k_t = k.to(device).requires_grad_(True)
    v_t = v.to(device).requires_grad_(True)
    out = tokamax_splash_attention.splash_sdpa(
        q_t,
        k_t,
        v_t,
        segment_ids=segment_ids.to(device),
        is_causal=True,
        enable_gqa=n_kv_heads != n_heads,
    )
    out.sum().backward()

    torch.testing.assert_close(
        out.detach().cpu(), out_ref.detach().transpose(1, 2), rtol=5e-2, atol=5e-2
    )
    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad),
        ("dk", k_t.grad, k_ref.grad),
        ("dv", v_t.grad, v_ref.grad),
    ):
      self.assertIsNotNone(got, f"{name} is None")
      torch.testing.assert_close(
          got.cpu(), want.transpose(1, 2), rtol=5e-2, atol=5e-2, msg=name
      )

    if config.get("num_stacked_q_heads", 1) != 1 or config.get("use_batch_loop_fwd"):
      # Stacking only regroups query rows of the same matmuls, and per-example
      # calls only drop tiles with no same-document pair, so the output must
      # equal the unstacked, vmapped kernel's bit for bit.
      self._set_tokamax_config(num_stacked_q_heads=1, use_batch_loop_fwd=False)
      out_unstacked = tokamax_splash_attention.splash_sdpa(
          q_t.detach(),
          k_t.detach(),
          v_t.detach(),
          segment_ids=segment_ids.to(device),
          is_causal=True,
          enable_gqa=n_kv_heads != n_heads,
      )
      self.assertTrue(torch.equal(out.detach().cpu(), out_unstacked.cpu()))

    # Guard against a vacuous pass: without segment_ids the kernel must differ
    # from the packed reference.
    out_unpacked = tokamax_splash_attention.splash_sdpa(
        q_t.detach(),
        k_t.detach(),
        v_t.detach(),
        is_causal=True,
        enable_gqa=n_kv_heads != n_heads,
    )
    self.assertFalse(
        torch.allclose(
            out_unpacked.cpu(),
            out_ref.detach().transpose(1, 2),
            rtol=5e-2,
            atol=5e-2,
        )
    )

  def _packed_grads(self, q, k, v, grad_out, segment_ids, **attn):
    q, k, v = (x.to(self.accelerator_device).requires_grad_(True) for x in (q, k, v))
    out = tokamax_splash_attention.splash_sdpa(
        q, k, v, segment_ids=segment_ids, enable_gqa=q.shape[2] != k.shape[2], **attn
    )
    out.backward(grad_out.to(self.accelerator_device))
    return [x.grad.float().cpu() for x in (q, k, v)]

  @parameterized.named_parameters(
      dict(testcase_name="group16", n_heads=16, n_kv_heads=1, head_dim=128),
      dict(testcase_name="group8", n_heads=16, n_kv_heads=2, head_dim=128),
      dict(testcase_name="group2_dim64", n_heads=8, n_kv_heads=4, head_dim=64),
      dict(testcase_name="group5_dim192", n_heads=10, n_kv_heads=2, head_dim=192),
      dict(testcase_name="group16_no_segment_ids", n_heads=16, n_kv_heads=1, head_dim=128, packed=False),
      dict(testcase_name="group8_scale", n_heads=16, n_kv_heads=2, head_dim=128, scale=0.05),
      # 1792 tokens: 256-token tiles; 1920 tokens: 128-token tiles.
      dict(testcase_name="tile256", n_heads=8, n_kv_heads=2, head_dim=128,
           documents=((300, 517, 77, 898), (1000, 48, 744)), block=256),
      dict(testcase_name="tile128_dim96", n_heads=4, n_kv_heads=2, head_dim=96,
           documents=((300, 517, 77, 1026), (1000, 48, 872)), block=128),
      # Sliding windows: inside a tile, one tile, several tiles (Muse Glimmer: 2047, Gemma 4: 1023).
      dict(testcase_name="window200", n_heads=16, n_kv_heads=2, head_dim=128, window=200),
      dict(testcase_name="window511_no_segment_ids", n_heads=16, n_kv_heads=1, head_dim=128, packed=False,
           window=511),
      dict(testcase_name="window1023", n_heads=8, n_kv_heads=4, head_dim=128, window=1023),
      # A window across 256-token tiles; Gemma 4's sliding layers (head_dim 256, softmax scale 1.0, so the
      # non-default-scale q rounding, window 1023).
      dict(testcase_name="window300_tile256", n_heads=8, n_kv_heads=2, head_dim=128,
           documents=((300, 517, 77, 898), (1000, 48, 744)), block=256, window=300),
      dict(testcase_name="gemma4_window1023_dim256_scale1", n_heads=4, n_kv_heads=2, head_dim=256, scale=1.0,
           window=1023),
  )
  def test_transposed_bwd_matches_reference(
      self, n_heads, n_kv_heads, head_dim, packed=True, documents=_DOCUMENTS, block=None, scale=None,
      window=None,
  ):
    """use_transposed_bwd (splash_tbwd) gradients match dense f32 attention and Tokamax's backward.

    _DOCUMENTS yields both tile kinds: boundaries inside 512-token blocks (masked tiles) and a 1154-token document
    spanning whole blocks below the diagonal (full tiles, no element mask). Also covers G not a multiple of 8,
    head dims other than 128, no segment ids, a non-default scale and the smaller tiles chosen for seq_len
    not a multiple of 512.
    """
    torch.manual_seed(7)
    if block is not None:
      blocks = ("block_q", "block_kv", "block_kv_compute", "block_q_dkv", "block_kv_dkv", "block_kv_dkv_compute")
      self._set_tokamax_config(**{name: block for name in blocks})
    segment_ids = _packed_segment_ids(documents)
    b, seq_len = segment_ids.shape
    q, grad_out = (torch.randn(b, seq_len, n_heads, head_dim, dtype=torch.bfloat16) for _ in range(2))
    k, v = (torch.randn(b, seq_len, n_kv_heads, head_dim, dtype=torch.bfloat16) for _ in range(2))

    # Dense f32 reference, [B, H, S, D]: causal and same-document.
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    if window is not None:
      mask = mask & ~torch.ones(seq_len, seq_len, dtype=torch.bool).tril(-window - 1)
    if packed:
      mask = (segment_ids[:, :, None] == segment_ids[:, None, :]) & mask
    else:
      mask = mask.expand(b, seq_len, seq_len)
    ref = [x.float().transpose(1, 2).requires_grad_(True) for x in (q, k, v)]
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          ref[0],
          *(x.repeat_interleave(n_heads // n_kv_heads, dim=1) for x in ref[1:]),
          attn_mask=mask[:, None],
          scale=scale,
      )
    out_ref.backward(grad_out.float().transpose(1, 2))
    dense = [x.grad.transpose(1, 2) for x in ref]

    segment_ids = segment_ids.to(self.accelerator_device) if packed else None
    self._check_transposed_bwd(q, k, v, grad_out, segment_ids, dense, scale=scale, local_window_size=window)

  def _dense_causal_grads(self, q, k, v, grad_out, scale=None):
    """dq, dk, dv of dense f32 causal attention (one document), [B, S, heads, D]."""
    group = q.shape[2] // k.shape[2]
    ref = [x.float().transpose(1, 2).requires_grad_(True) for x in (q, k, v)]
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          ref[0], *(x.repeat_interleave(group, dim=1) for x in ref[1:]), is_causal=True, scale=scale
      )
    out_ref.backward(grad_out.float().transpose(1, 2))
    return [x.grad.transpose(1, 2) for x in ref]

  def _assert_runs_transposed_bwd(self, q, k, v):
    """The wrapper routes these shapes to splash_tbwd (else the test compares Tokamax with itself).

    Skips on a TPU splash_tbwd does not run on; an unsupported shape fails.
    """
    hardware_reason = splash_tbwd.hardware_unsupported_reason()
    if hardware_reason is not None:
      self.skipTest(f"splash_tbwd does not run here: {hardware_reason}")
    as_jax = lambda x: types.SimpleNamespace(shape=tuple(x.shape), dtype=jnp.dtype(jnp.bfloat16))
    reason = tokamax_splash_attention._transposed_bwd_unsupported_reason(  # pylint: disable=protected-access
        as_jax(q), as_jax(k), as_jax(v), is_causal=True
    )
    self.assertIsNone(reason)

  def _check_transposed_bwd(
      self, q, k, v, grad_out, segment_ids, dense, scale=None, max_error_ratio=1.05, max_norm_bias=1e-3,
      local_window_size=None,
  ):
    """splash_tbwd's gradients are no further from `dense` than Tokamax's backward is, with no scale bias."""
    attn = dict(scale=scale, local_window_size=local_window_size)
    tokamax = self._packed_grads(q, k, v, grad_out, segment_ids, **attn)
    self._assert_runs_transposed_bwd(q, k, v)
    self._set_tokamax_config(use_transposed_bwd=True)
    got = self._packed_grads(q, k, v, grad_out, segment_ids, **attn)
    for name, g, d, t in zip(("dq", "dk", "dv"), got, dense, tokamax):
      # Both kernels round P / dS to bf16 for the MXU (~3-5e-3 off dense f32, growing with head_dim), so bound
      # splash_tbwd's error by Tokamax's own on the same inputs: no worse than Tokamax, with 5% slack so a small
      # Tokamax accuracy gain does not fail the test. TPU7x: P is the forward's (dv error equals Tokamax's) and
      # dq / dk are ~0.90x Tokamax's at these logits.
      error = ((g - d).norm() / d.norm()).item()
      tokamax_error = ((t - d).norm() / d.norm()).item()
      self.assertLess(error, max_error_ratio * tokamax_error, name)
      if max_norm_bias is not None:
        # No systematic scale difference from Tokamax's backward (TPU7x: <= 3.1e-4 here). A log2(e) or scale ratio
        # that did not match the forward's bf16 rounding biased every gradient by 1-2e-3.
        self.assertLess(abs((g.norm() / t.norm()).item() - 1), max_norm_bias, name)

  @parameterized.named_parameters(
      ("causal", None, 2048), ("window300", 300, 2048), ("window300_tile256", 300, 2304)
  )
  def test_transposed_bwd_stream_kv_matches_resident(self, window, s):
    """Streaming K / V one kv block per step gives bit-identical gradients to keeping them resident."""
    reason = splash_tbwd.hardware_unsupported_reason()
    if reason is not None:
      self.skipTest(reason)
    b, h, n, d = 2, 8, 2, 128
    keys = jax.random.split(jax.random.key(3), 5)
    q, out, do = (jax.random.normal(kk, (b, s, h, d), jnp.bfloat16) for kk in keys[:3])
    k, v = (jax.random.normal(kk, (b, s, n, d), jnp.bfloat16) for kk in keys[3:])
    lse2 = jnp.full((b, h, s), 12.0, jnp.float32)
    seg = jnp.asarray(np.repeat(np.arange(8), s // 8)[None].repeat(b, 0), jnp.int32)
    grads = {
        stream: jax.jit(
            functools.partial(
                splash_tbwd.transposed_attention_bwd, q_scale=d**-0.5, window=window, stream_kv=stream
            )
        )(q, k, v, out, lse2, do, seg)
        for stream in (False, True)
    }
    for name, resident, streamed in zip(("dq", "dk", "dv"), grads[False], grads[True]):
      np.testing.assert_array_equal(np.asarray(resident), np.asarray(streamed), err_msg=name)

  @parameterized.named_parameters(("global", None), ("window2047", 2047))
  def test_transposed_bwd_long_sequence(self, window):
    """At Muse Glimmer's 16k tokens, 32 q / 2 kv heads (fits only with K / V streamed), splash_tbwd matches Tokamax."""
    torch.manual_seed(7)
    segment_ids = _packed_segment_ids(((5000, 3000, 8384),))
    q, grad_out = (torch.randn(1, 16384, 32, 128, dtype=torch.bfloat16) for _ in range(2))
    k, v = (torch.randn(1, 16384, 2, 128, dtype=torch.bfloat16) for _ in range(2))
    self._assert_runs_transposed_bwd(q, k, v)
    segment_ids = segment_ids.to(self.accelerator_device)
    tokamax = self._packed_grads(q, k, v, grad_out, segment_ids, local_window_size=window)
    self._set_tokamax_config(use_transposed_bwd=True)
    got = self._packed_grads(q, k, v, grad_out, segment_ids, local_window_size=window)
    for name, g, t in zip(("dq", "dk", "dv"), got, tokamax):
      # Both round P / dS to bf16 for the MXU (~3-5e-3 off dense f32 each).
      self.assertLess(((g - t).norm() / t.norm()).item(), 1e-2, name)

  @parameterized.named_parameters(
      dict(testcase_name="logit_std8", logit_std=8.0),
      dict(testcase_name="sink_logit20", sink_logit=20.0),
      dict(testcase_name="sink_logit50", sink_logit=50.0),
      dict(testcase_name="sink_logit100", sink_logit=100.0),
      # Non-default scales (the forward rounds q twice before Q.K). The sink is built at the default scale, so its
      # logit is scale_ratio * sink_logit (68.5 here).
      dict(testcase_name="scale0p8_logit_std8", logit_std=8.0, scale_ratio=0.8),
      dict(testcase_name="scale1p37_sink_logit68", sink_logit=50.0, scale_ratio=1.37),
  )
  def test_transposed_bwd_large_logits(self, logit_std=1.0, sink_logit=None, scale_ratio=None):
    """splash_tbwd stays at Tokamax's accuracy with trained-model logits: large, or one attention-sink key.

    P must be rebuilt from q rounded exactly as the forward rounded it; folding rounded constants into an f32 Q.K
    instead leaves an error in the logits that grows with them, invisible at initialization (logit std ~1).
    TPU7x, error vs dense f32 as a multiple of Tokamax's (dq / dk / dv), with q rounded as the forward does vs
    with the rounded constants folded into Q.K (before):
      logit std ~8:        0.99 / 0.99 / 1.00   (before 1.63 / 1.63 / 2.79)
      sink 20 / 50 / 100:  <= 1.02 / 1.02 / 1.00   (before dv 2.6 / 6.7 / 13.3)
    """
    torch.manual_seed(7)
    b, seq_len, n_heads, n_kv_heads, head_dim = 1, 512, 4, 1, 128
    scale = None if scale_ratio is None else scale_ratio / head_dim**0.5
    # logit = q.k / sqrt(head_dim): with q, k ~ N(0, c^2) its std is c^2 (times scale_ratio).
    c = (logit_std / (scale_ratio or 1.0))**0.5
    q = torch.randn(b, seq_len, n_heads, head_dim) * c
    k = torch.randn(b, seq_len, n_kv_heads, head_dim) * c
    if sink_logit is not None:
      # Key 0 shares a direction with every query, with a per-query strength in [0, 1]: its logit spans
      # 0..sink_logit, so P does not saturate on it everywhere (dq would vanish).
      direction = torch.ones(head_dim)
      q = q + torch.rand(b, seq_len, n_heads, 1) * direction
      k[:, 0] += direction * (sink_logit / head_dim**0.5)
    q, k = q.bfloat16(), k.bfloat16()
    v = torch.randn(b, seq_len, n_kv_heads, head_dim, dtype=torch.bfloat16)
    grad_out = torch.randn(b, seq_len, n_heads, head_dim, dtype=torch.bfloat16)
    dense = self._dense_causal_grads(q, k, v, grad_out, scale=scale)
    self._check_transposed_bwd(q, k, v, grad_out, None, dense, scale=scale, max_error_ratio=1.1, max_norm_bias=None)

  @parameterized.named_parameters(
      ("logit724", 8.0),
      ("logit11585", 32.0),
      # Non-default scale: the forward rounds q * bf16(scale / default) and then * bf16(1 / sqrt(D)); one rounding of
      # q * (their product) differs by a bf16 step here (dV 1.914 at 1.03125; ~2e19 at 31.625 on CPU interpret).
      # These pass on TPU main too; there the large-logit scale cases above catch the single rounding.
      ("scale005_value1", 1.03125, 0.05),
      ("scale005_value32", 31.625, 0.05),
  )
  def test_transposed_bwd_rebuilds_forward_softmax(self, value, scale=None):
    """P matches the forward's logsumexp even where every logit is huge.

    q = k = value everywhere, V = 1, dO = 1 at query 0 only. Query 0 attends to key 0 alone, so P[0, 0] = 1 in each
    of the 2 q heads and dV[0, 0] = 2, everything else 0. Before the fix splash_tbwd returned 0.664 (value 8) and
    4.2e-8 (value 32).
    """
    q = torch.full((1, 128, 2, 128), value, dtype=torch.bfloat16)
    k = torch.full((1, 128, 1, 128), value, dtype=torch.bfloat16)
    v = torch.ones(1, 128, 1, 128, dtype=torch.bfloat16)
    grad_out = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    grad_out[:, 0] = 1
    expected_dv = torch.zeros(1, 128, 1, 128)
    expected_dv[:, 0] = 2
    tokamax = self._packed_grads(q, k, v, grad_out, None, scale=scale)
    torch.testing.assert_close(tokamax[2], expected_dv)
    self._assert_runs_transposed_bwd(q, k, v)
    self._set_tokamax_config(use_transposed_bwd=True)
    got = self._packed_grads(q, k, v, grad_out, None, scale=scale)
    torch.testing.assert_close(got[2], expected_dv)

  @parameterized.named_parameters(
      ("non_causal", dict(is_causal=False)),
      ("group32", dict(n_heads=32)),
      ("group1", dict(n_kv_heads=16)),
  )
  def test_transposed_bwd_falls_back_to_tokamax(self, attn):
    """Attention splash_tbwd does not support (or is slower at) takes Tokamax's backward, with a warning."""
    torch.manual_seed(7)
    n_heads = attn.pop("n_heads", 16)
    n_kv_heads = attn.pop("n_kv_heads", 1)
    segment_ids = _packed_segment_ids(_DOCUMENTS).to(self.accelerator_device)
    q, grad_out = (torch.randn(2, 2048, n_heads, 128, dtype=torch.bfloat16) for _ in range(2))
    k, v = (torch.randn(2, 2048, n_kv_heads, 128, dtype=torch.bfloat16) for _ in range(2))
    tokamax = self._packed_grads(q, k, v, grad_out, segment_ids, **attn)
    self._set_tokamax_config(use_transposed_bwd=True)
    tokamax_splash_attention._warn_transposed_bwd_fallback.cache_clear()
    with self.assertLogs(level="WARNING") as logged:
      got = self._packed_grads(q, k, v, grad_out, segment_ids, **attn)
    self.assertIn("using the Tokamax backward instead of splash_tbwd", "\n".join(logged.output))
    for g, t in zip(got, tokamax):
      torch.testing.assert_close(g, t, rtol=0, atol=0)

  def test_config_change_retraces_kernel(self):
    """A TokamaxConfig change must not reuse a kernel compiled for the old one.

    jax_op caches compiled kernels by argument values, so the config has to be
    passed as op arguments; read inside the op, two calls with equal shapes
    would share one compiled kernel whatever the config said.
    """
    from unittest import mock  # pylint: disable=g-import-not-at-top

    device = self.accelerator_device
    cfg = tokamax_splash_attention.TokamaxConfig
    saved = cfg.use_vmap_bwd
    self.addCleanup(setattr, cfg, "use_vmap_bwd", saved)

    # A shape no other test uses, so the first call below must trace too.
    q, k, v = (
        torch.randn(1, 1024, 2, 64, device=device).requires_grad_(True)
        for _ in range(3)
    )
    real = tokamax_splash_attention._make_splash_attention_fn
    seen = []

    def record(*args, **kwargs):
      seen.append(kwargs["use_vmap_bwd"])
      return real(*args, **kwargs)

    with mock.patch.object(
        tokamax_splash_attention, "_make_splash_attention_fn", record
    ):
      for use_vmap_bwd in (False, True):
        cfg.use_vmap_bwd = use_vmap_bwd
        tokamax_splash_attention.splash_sdpa(q, k, v).sum().backward()

    # fwd + bwd each traced once per config value.
    self.assertEqual(sorted(seen), [False, False, True, True])

  @parameterized.named_parameters(*_CP_SHARD_CASES)
  def test_cp_shard_parity(
      self,
      cp_size,
      local_window_size=None,
      seq_len=512,
      n_heads=2,
      n_kv_heads=2,
      block=None,
      documents=None,
  ):
    """Head-tail CP Q shards against the full K/V match full attention.

    KV all-gather CP gives rank r the r-th slice of the head-tail sequence; the
    kernel must mask those queries by their original positions (causal or
    sliding window, and by document) in every Q and KV block. dK and dV add up
    over the ranks.
    """
    device = self.accelerator_device
    torch.manual_seed(0)
    if block is not None:
      cfg = tokamax_splash_attention.TokamaxConfig
      for name in (
          "block_q",
          "block_kv",
          "block_kv_compute",
          "block_q_dkv",
          "block_kv_dkv",
          "block_kv_dkv_compute",
      ):
        self.addCleanup(setattr, cfg, name, getattr(cfg, name))
        setattr(cfg, name, block)

    # (B, S, H, D), the layout this kernel takes.
    b, head_dim = (1 if documents is None else len(documents)), 128
    q, grad_out = (torch.randn(b, seq_len, n_heads, head_dim) for _ in range(2))
    k, v = (torch.randn(b, seq_len, n_kv_heads, head_dim) for _ in range(2))
    segment_ids = None if documents is None else _packed_segment_ids(documents)
    pos = torch.arange(seq_len)
    allowed = pos[:, None] >= pos[None, :]
    if local_window_size is not None:
      allowed &= pos[:, None] - pos[None, :] <= local_window_size
    if segment_ids is not None:
      same_document = segment_ids[:, :, None] == segment_ids[:, None, :]
      allowed = (allowed & same_document)[:, None]
    q_ref, k_ref, v_ref = (
        x.transpose(1, 2).clone().requires_grad_(True) for x in (q, k, v)
    )
    n_rep = n_heads // n_kv_heads
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          q_ref,
          k_ref.repeat_interleave(n_rep, dim=1),
          v_ref.repeat_interleave(n_rep, dim=1),
          attn_mask=allowed,
      )
    out_ref.backward(grad_out.transpose(1, 2))
    out_ref = out_ref.detach().transpose(1, 2)
    dq_ref, dk_ref, dv_ref = (
        x.grad.transpose(1, 2) for x in (q_ref, k_ref, v_ref)
    )

    k_test = k.to(device).requires_grad_(True)
    v_test = v.to(device).requires_grad_(True)
    shards = torch.from_numpy(headtail_positions(seq_len, cp_size)).long()
    for r, rows in enumerate(shards.reshape(cp_size, -1)):
      q_test = q[:, rows].to(device).requires_grad_(True)
      out = tokamax_splash_attention.splash_sdpa(
          q_test,
          k_test,
          v_test,
          segment_ids=None if segment_ids is None else segment_ids.to(device),
          cp_rank=torch.tensor([r], dtype=torch.int32, device=device),
          local_window_size=local_window_size,
          enable_gqa=n_heads != n_kv_heads,
      )
      out.backward(grad_out[:, rows].to(device))
      for name, got, want in (
          ("out", out, out_ref),
          ("dq", q_test.grad, dq_ref),
      ):
        torch.testing.assert_close(
            got.cpu(),
            want[:, rows],
            rtol=5e-2,
            atol=5e-2,
            msg=lambda m, name=name, r=r: f"{name} of rank {r}: {m}",
        )
    torch.testing.assert_close(k_test.grad.cpu(), dk_ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_test.grad.cpu(), dv_ref, rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
  absltest.main()
