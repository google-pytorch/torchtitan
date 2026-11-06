"""Tests for Tokamax splash attention on TPU."""

from collections.abc import Callable
import functools
import logging

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
from torch.nn import attention
import torch.nn.functional as F
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.headtail import headtail_positions
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
