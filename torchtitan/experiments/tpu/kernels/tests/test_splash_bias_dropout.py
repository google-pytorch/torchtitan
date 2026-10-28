# Copyright 2026 Google LLC

"""Tests for splash attention with bias + PRNG dropout on TPU.

Conventions follow test_tokamax_splash_attention.py: absltest +
BaseAcceleratorDeviceTest, (B, S, H, D) layouts, SDPA MATH-backend
references. `SbdConfig` values ride as static op arguments (part of the
trace cache key), so config changes are safe at any shape; the per-key
test reuses the full-rank test's shape on purpose to lock that in.
"""

import functools
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.nn import attention
import torch.nn.functional as F
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import unflatten_shim
from torchtitan.experiments.tpu.kernels import splash_bias_dropout
from torchtitan.experiments.tpu.kernels import tokamax_splash_attention


def _sdpa_ref(q, k, v, attn_mask=None, is_causal=False):
  """Math-backend SDPA reference on CPU (torch_tpu overrides SDPA on-device
  with a fused op whose backward cannot produce attn_mask grads); inputs
  and outputs are CPU (B, S, H, D) tensors."""
  with attention.sdpa_kernel([attention.SDPBackend.MATH]):
    out = F.scaled_dot_product_attention(
        q.cpu().transpose(1, 2),
        k.cpu().transpose(1, 2),
        v.cpu().transpose(1, 2),
        attn_mask=attn_mask.cpu() if attn_mask is not None else None,
        is_causal=is_causal,
    )
  return out.transpose(1, 2)


def _causal_bias_mask(bias_hss, seq_len, device, dtype=torch.float32):
  """Additive attn_mask (H, S, S): bias + -inf above the diagonal."""
  causal = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
  causal.masked_fill_(
      torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), 1),
      float("-inf"),
  )
  return bias_hss.to(dtype) + causal[None]


class SplashBiasDropoutTest(base_device_test.BaseAcceleratorDeviceTest):

  def _qkv(self, b, s, h, h_kv, d, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, s, h, d, generator=g, dtype=torch.float32).to(device)
    k = torch.randn(b, s, h_kv, d, generator=g, dtype=torch.float32).to(device)
    v = torch.randn(b, s, h_kv, d, generator=g, dtype=torch.float32).to(device)
    return q, k, v

  def _expand_kv(self, k, v, n_heads):
    n_rep = n_heads // k.shape[2]
    return k.repeat_interleave(n_rep, dim=2), v.repeat_interleave(n_rep, dim=2)

  # -------- plain causal attention (no bias, no dropout) --------

  def test_mha_parity(self):
    device = self.accelerator_device
    q, k, v = self._qkv(2, 512, 4, 4, 64, device)
    expected = _sdpa_ref(q, k, v, is_causal=True)
    actual = splash_bias_dropout.sbd_splash_attention(q, k, v)
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=5e-2, atol=5e-2
    )

  def test_gqa_parity(self):
    device = self.accelerator_device
    q, k, v = self._qkv(2, 512, 4, 2, 64, device)
    k_exp, v_exp = self._expand_kv(k, v, 4)
    expected = _sdpa_ref(q, k_exp, v_exp, is_causal=True)
    actual = splash_bias_dropout.sbd_splash_attention(q, k, v)
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=5e-2, atol=5e-2
    )

  def test_mqa_parity(self):
    device = self.accelerator_device
    q, k, v = self._qkv(2, 512, 4, 1, 64, device, seed=29)
    k_exp, v_exp = self._expand_kv(k, v, 4)
    expected = _sdpa_ref(q, k_exp, v_exp, is_causal=True)
    actual = splash_bias_dropout.sbd_splash_attention(q, k, v)
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=5e-2, atol=5e-2
    )

    # Backward: the reference expands KV inside the graph so autograd folds
    # the per-group gradients back onto the single KV head.
    q_ref = q.cpu().clone().requires_grad_(True)
    k_ref = k.cpu().clone().requires_grad_(True)
    v_ref = v.cpu().clone().requires_grad_(True)
    out_ref = _sdpa_ref(
        q_ref,
        k_ref.repeat_interleave(4, dim=2),
        v_ref.repeat_interleave(4, dim=2),
        is_causal=True,
    )
    out_ref.sum().backward()

    q_t = q.clone().requires_grad_(True)
    k_t = k.clone().requires_grad_(True)
    v_t = v.clone().requires_grad_(True)
    out = splash_bias_dropout.sbd_splash_attention(q_t, k_t, v_t)
    out.sum().backward()
    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad),
        ("dk", k_t.grad, k_ref.grad),
        ("dv", v_t.grad, v_ref.grad),
    ):
      self.assertIsNotNone(got, f"{name} is None")
      torch.testing.assert_close(
          got.cpu(), want.cpu(), rtol=8e-2, atol=8e-2, msg=name
      )

  # -------- packed-document segment ids --------

  def _segment_ids_case(self, h_kv, causal, use_window=False):
    """Kernel with segment_ids vs SDPA over the dense document(+causal) mask,
    forward and backward; also checks the ids actually changed the output
    (a vacuous pass -- ids silently dropped -- would still match the
    unmasked reference otherwise)."""
    device = self.accelerator_device
    b, s, h, d = 2, 512, 4, 64
    q, k, v = self._qkv(b, s, h, h_kv, d, device, seed=41)
    segment_ids = torch.zeros(b, s, dtype=torch.int32)
    segment_ids[0, 128:] = 1  # one boundary on a block edge
    segment_ids[1, 40:100] = 1  # three boundaries inside blocks
    segment_ids[1, 100:200] = 2
    segment_ids[1, 200:] = 3
    allowed = segment_ids[:, :, None] == segment_ids[:, None, :]  # (B, Sq, Skv)
    if causal:
      allowed &= torch.tril(torch.ones(s, s, dtype=torch.bool))[None]
    window = 64 if use_window else None
    if use_window:
      i = torch.arange(s)
      allowed &= ((i[:, None] - i[None, :]) <= window) & (i[:, None] >= i[None, :])
    mask = torch.zeros(b, 1, s, s, dtype=torch.float32)
    mask.masked_fill_(~allowed[:, None], float("-inf"))

    def reference(q_, k_, v_):
      k_e, v_e = self._expand_kv(k_, v_, h)
      return _sdpa_ref(q_, k_e, v_e, attn_mask=mask)

    q_ref = q.cpu().clone().requires_grad_(True)
    k_ref = k.cpu().clone().requires_grad_(True)
    v_ref = v.cpu().clone().requires_grad_(True)
    out_ref = reference(q_ref, k_ref, v_ref)
    out_ref.sum().backward()

    q_t = q.clone().requires_grad_(True)
    k_t = k.clone().requires_grad_(True)
    v_t = v.clone().requires_grad_(True)
    out = splash_bias_dropout.sbd_splash_attention(
        q_t, k_t, v_t, segment_ids=segment_ids.to(device), is_causal=causal,
        local_window_size=window,
    )
    out.sum().backward()
    torch.testing.assert_close(out.cpu(), out_ref.cpu(), rtol=5e-2, atol=5e-2)
    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad), ("dk", k_t.grad, k_ref.grad), ("dv", v_t.grad, v_ref.grad),
    ):
      self.assertIsNotNone(got, f"{name} is None")
      torch.testing.assert_close(got.cpu(), want.cpu(), rtol=8e-2, atol=8e-2, msg=name)

    # Guard against a vacuous pass: the same call without ids must differ.
    out_no_ids = splash_bias_dropout.sbd_splash_attention(
        q, k, v, is_causal=causal, local_window_size=window
    )
    self.assertGreater(
        float((out_no_ids.cpu() - out.cpu()).abs().max()), 1e-2,
        "segment_ids had no effect on the output",
    )

  def test_segment_ids_mha_causal(self):
    self._segment_ids_case(h_kv=4, causal=True)

  def test_segment_ids_mha_bidirectional(self):
    self._segment_ids_case(h_kv=4, causal=False)

  def test_segment_ids_gqa_causal(self):
    self._segment_ids_case(h_kv=2, causal=True)

  def test_segment_ids_mqa_causal(self):
    # The kernel tiles kv segment ids differently on its rank-reduced entry.
    self._segment_ids_case(h_kv=1, causal=True)

  def test_segment_ids_window(self):
    self._segment_ids_case(h_kv=4, causal=True, use_window=True)

  def test_segment_ids_with_dropout(self):
    """Dropout on top of segment ids must not disturb the document mask.

    Deterministic form (a mean-over-seeds check is a coin flip at 262k
    elements): the second document's values carry a +1000 offset, so any
    cross-document leakage would show up as a large first-document output;
    with dropout active that output must stay within the undropped,
    unoffset output's range, while within-document outputs do change with
    the seed.
    """
    device = self.accelerator_device
    b, s, h, d = 2, 512, 4, 64
    q, k, v = self._qkv(b, s, h, h, d, device, seed=43)
    segment_ids = torch.zeros(b, s, dtype=torch.int32)
    segment_ids[:, 256:] = 1
    seg = segment_ids.to(device)
    v_offset = v.clone()
    v_offset[:, 256:] += 1000.0  # document 2 only
    base = splash_bias_dropout.sbd_splash_attention(q, k, v, segment_ids=seg)
    splash_bias_dropout.SbdConfig.update(dropout_rate=0.1)
    try:
      out0 = splash_bias_dropout.sbd_splash_attention(q, k, v_offset, segment_ids=seg, dropout_seed=0)
      out1 = splash_bias_dropout.sbd_splash_attention(q, k, v_offset, segment_ids=seg, dropout_seed=1)
    finally:
      splash_bias_dropout.SbdConfig.update(dropout_rate=0.0)
    # Document 1's queries never see document 2's +1000: bounded by the
    # undropped output scaled by the worst-case dropout rescale (1 / 0.9).
    doc1_bound = float(base[:, :256].abs().max()) / 0.9 + 1e-3
    self.assertLess(float(out0[:, :256].abs().max()), doc1_bound, "cross-document leakage under dropout")
    self.assertLess(float(out1[:, :256].abs().max()), doc1_bound, "cross-document leakage under dropout")
    # Dropout is active: a different seed gives a different output.
    self.assertGreater(float((out0 - out1).abs().max()), 1e-3, "seeds gave identical masks")

  # -------- attention-logit soft-cap (Gemma-2) --------

  @parameterized.named_parameters(("cap50", 50.0), ("cap2", 2.0), ("cap0p5", 0.5))
  def test_attn_logits_soft_cap(self, cap):
    """cap * tanh(scaled logits / cap), forward and backward, vs a math reference.

    The adapter hands HF's `softcap` kwarg to the kernel through
    SbdConfig.attn_logits_soft_cap. A cap of 0.5 on N(0, 1)-ish logits is far
    from the identity, so a cap that is silently skipped fails this test.
    """
    device = self.accelerator_device
    b, s, h, d = 1, 512, 4, 64
    q, k, v = self._qkv(b, s, h, h, d, device, seed=79)
    scale = d ** -0.5
    q_ref, k_ref, v_ref = (t.cpu().clone().requires_grad_(True) for t in (q, k, v))
    scores = torch.einsum("bqhd,bkhd->bhqk", q_ref, k_ref) * scale
    scores = cap * torch.tanh(scores / cap)
    scores = scores.masked_fill(~torch.ones(s, s, dtype=torch.bool).tril(), float("-inf"))
    out_ref = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, -1), v_ref)
    out_ref.sum().backward()
    splash_bias_dropout.SbdConfig.attn_logits_soft_cap = cap
    try:
      q_t, k_t, v_t = (t.clone().requires_grad_(True) for t in (q, k, v))
      out = splash_bias_dropout.sbd_splash_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
      out.sum().backward()
    finally:
      splash_bias_dropout.SbdConfig.attn_logits_soft_cap = None
    torch.testing.assert_close(out.cpu(), out_ref, rtol=5e-2, atol=5e-2)
    for name, got, want in (("dq", q_t.grad, q_ref.grad), ("dk", k_t.grad, k_ref.grad),
                            ("dv", v_t.grad, v_ref.grad)):
      torch.testing.assert_close(got.cpu(), want, rtol=8e-2, atol=8e-2, msg=name)
    # Guard against a vacuous pass: the uncapped output must differ.
    uncapped = splash_bias_dropout.sbd_splash_attention(q, k, v, is_causal=True, scale=scale)
    if cap <= 2.0:
      self.assertGreater(float((uncapped.cpu() - out.cpu()).abs().max()), 1e-2, "cap had no effect")

  # -------- attention sinks (gpt-oss, Flint) --------

  @parameterized.named_parameters(
      ("mha", 4, None, False),
      ("gqa", 2, None, False),
      ("mqa", 1, None, False),
      ("gqa_window", 2, 128, False),
      ("gqa_base2", 2, None, True),
  )
  def test_attention_sinks(self, h_kv, window, base2):
    """One learned logit per head as an extra softmax column (HF's ``s_aux``):
    output, dq/dk/dv and the sink gradient vs a math reference."""
    device = self.accelerator_device
    b, s, h, d = 2, 512, 4, 64
    q, k, v = self._qkv(b, s, h, h_kv, d, device, seed=83)
    k, v = k.to(device), v.to(device)
    g = torch.Generator().manual_seed(84)
    # Sinks near the logits' scale, so the sink takes a sizeable share.
    sinks = (torch.randn(h, generator=g) * 2.0).to(device)
    cot = torch.randn(b, s, h, d, generator=g)
    scale = d ** -0.5

    q_ref, k_ref, v_ref, s_ref = (
        t.cpu().clone().requires_grad_(True) for t in (q, k, v, sinks)
    )
    k_e, v_e = self._expand_kv(k_ref, v_ref, h)
    scores = torch.einsum("bqhd,bkhd->bhqk", q_ref, k_e) * scale
    keep = torch.ones(s, s, dtype=torch.bool).tril()
    if window is not None:
      keep &= ~torch.ones(s, s, dtype=torch.bool).tril(-window - 1)
    scores = scores.masked_fill(~keep, float("-inf"))
    col = s_ref.reshape(1, h, 1, 1).expand(b, h, s, 1)
    probs = torch.softmax(torch.cat([scores, col], -1), -1)[..., :-1]
    out_ref = torch.einsum("bhqk,bkhd->bqhd", probs, v_e)
    (out_ref * cot).sum().backward()

    old_base2 = splash_bias_dropout.SbdConfig.use_base2_exp
    splash_bias_dropout.SbdConfig.use_base2_exp = base2
    try:
      q_t, k_t, v_t, s_t = (t.clone().requires_grad_(True) for t in (q, k, v, sinks))
      out = splash_bias_dropout.sbd_splash_attention(
          q_t, k_t, v_t, is_causal=True, scale=scale, local_window_size=window,
          sinks=s_t,
      )
      (out * cot.to(device)).sum().backward()
      no_sink = splash_bias_dropout.sbd_splash_attention(
          q, k, v, is_causal=True, scale=scale, local_window_size=window
      )
    finally:
      splash_bias_dropout.SbdConfig.use_base2_exp = old_base2
    torch.testing.assert_close(out.detach().cpu(), out_ref.detach(), rtol=5e-2, atol=5e-2)
    for name, got, want in (("dq", q_t.grad, q_ref.grad), ("dk", k_t.grad, k_ref.grad),
                            ("dv", v_t.grad, v_ref.grad)):
      torch.testing.assert_close(got.cpu(), want, rtol=8e-2, atol=8e-2, msg=name)
    torch.testing.assert_close(s_t.grad.cpu(), s_ref.grad, rtol=5e-2, atol=5e-2,
                               msg="dsinks")
    # Guard against a vacuous pass: the sinks must change the output.
    self.assertGreater(float((no_sink.cpu() - out.detach().cpu()).abs().max()), 1e-1,
                       "sinks had no effect")

  # -------- Gemma-4 attention shapes --------

  @parameterized.named_parameters(
      # Global (full-attention) layers: head_dim 512 with very few KV heads --
      # 2 on Gemma-4-26B, 1 (MQA) on Gemma-4-12B. No other model in the lane
      # reaches head_dim 512, so these are the shapes most likely to hit a
      # VMEM or tiling limit.
      ("global_hd512_gqa", 512, 2, None),
      ("global_hd512_mqa", 512, 1, None),
      # Sliding layers: head_dim 256 under a window shorter than the sequence.
      ("sliding_hd256_window", 256, 4, 256),
  )
  def test_gemma4_shapes(self, head_dim, h_kv, window):
    device = self.accelerator_device
    b, s, h = 1, 1024, 8
    q, k, v = self._qkv(b, s, h, h_kv, head_dim, device, seed=71)
    segment_ids = torch.zeros(b, s, dtype=torch.int32)
    segment_ids[:, 640:] = 1
    allowed = (segment_ids[:, :, None] == segment_ids[:, None, :]) & torch.tril(
        torch.ones(s, s, dtype=torch.bool))[None]
    if window is not None:
      i = torch.arange(s)
      allowed &= (i[:, None] - i[None, :]) <= window
    mask = torch.zeros(b, 1, s, s, dtype=torch.float32)
    mask.masked_fill_(~allowed[:, None], float("-inf"))
    scale = 1.0  # Gemma-4 attention runs with scaling=1.0 (QK-norm upstream)

    def reference(q_, k_, v_):
      k_e, v_e = self._expand_kv(k_, v_, h)
      with attention.sdpa_kernel([attention.SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(
            q_.transpose(1, 2), k_e.transpose(1, 2), v_e.transpose(1, 2),
            attn_mask=mask, scale=scale)
      return out.transpose(1, 2)

    # Inputs scaled down so scale=1.0 logits stay in a range bf16-free fp32
    # comparisons can resolve (QK-norm keeps real Gemma activations O(1)).
    q, k = q * 0.1, k * 0.1
    q_ref, k_ref, v_ref = (t.cpu().clone().requires_grad_(True) for t in (q, k, v))
    out_ref = reference(q_ref, k_ref, v_ref)
    out_ref.sum().backward()

    q_t, k_t, v_t = (t.clone().requires_grad_(True) for t in (q, k, v))
    out = splash_bias_dropout.sbd_splash_attention(
        q_t, k_t, v_t, segment_ids=segment_ids.to(device), is_causal=True,
        scale=scale, local_window_size=window,
    )
    out.sum().backward()
    torch.testing.assert_close(out.cpu(), out_ref.cpu(), rtol=5e-2, atol=5e-2)
    for name, got, want in (("dq", q_t.grad, q_ref.grad), ("dk", k_t.grad, k_ref.grad),
                            ("dv", v_t.grad, v_ref.grad)):
      torch.testing.assert_close(got.cpu(), want.cpu(), rtol=8e-2, atol=8e-2, msg=name)

  # -------- layouts --------

  @parameterized.named_parameters(
      ("mha", 4), ("gqa", 2),
      # MQA takes the kernel's rank-reduced entry, whose dk/dv come back
      # without a head axis -- the one place the two layouts need different
      # reshapes rather than a transpose.
      ("mqa", 1),
  )
  def test_heads_first_layout_matches_torch_standard(self, h_kv):
    """`heads_first=True` is the same computation on the same bits.

    It is the layout HF hands attention and the kernel's own, so it skips the
    wrapper's transposes -- the result has to be identical, forward and
    backward, or the saving is a bug.
    """
    device = self.accelerator_device
    b, s, h, d = 2, 512, 4, 64
    q, k, v = self._qkv(b, s, h, h_kv, d, device, seed=53)
    segment_ids = torch.zeros(b, s, dtype=torch.int32)
    segment_ids[:, 256:] = 1
    seg = segment_ids.to(device)

    def run(heads_first):
      made = [
          (t.transpose(1, 2).contiguous() if heads_first else t.clone())
          for t in (q, k, v)
      ]
      qq, kk, vv = [t.requires_grad_(True) for t in made]
      out = splash_bias_dropout.sbd_splash_attention(
          qq, kk, vv, segment_ids=seg, heads_first=heads_first
      )
      out.sum().backward()
      grads = [t.grad for t in (qq, kk, vv)]
      if heads_first:
        out = out.transpose(1, 2)
        grads = [g.transpose(1, 2) for g in grads]
      return out.cpu(), [g.cpu() for g in grads]

    out_bshd, grads_bshd = run(False)
    out_bhsd, grads_bhsd = run(True)
    torch.testing.assert_close(out_bhsd, out_bshd, rtol=0, atol=0)
    for name, a, b_ in zip(("dq", "dk", "dv"), grads_bhsd, grads_bshd):
      torch.testing.assert_close(a, b_, rtol=0, atol=0, msg=name)

  # -------- document-aware block skipping --------

  def _blocks(self, blk):
    return dict(
        block_q=blk, block_kv=blk, block_kv_compute=blk,
        block_q_dkv=blk, block_kv_dkv=blk, block_kv_dkv_compute=blk,
    )

  def test_block_skipping_with_dropout_is_bit_identical(self):
    """Skipping must not disturb the dropout mask.

    The forward's grid is compacted while the backward's dkv grid stays dense
    and merely zeroes tiles, and the PRNG mask is keyed on the ABSOLUTE
    (q-block, kv-block) indices -- so compaction has to carry those indices
    through unchanged or the two passes would drop different entries. With the
    same seed, skipping on and off must agree bit for bit, forward and
    backward; a shifted key would show up here and nowhere else (the
    dropout-free cases pass either way).
    """
    device = self.accelerator_device
    # Blocks must be a multiple of the kernel's 256-token dropout quantum, so
    # this case runs 1024 tokens in 256-blocks: a 4x4 block grid, 4 documents.
    b, s, h, d, blk = 2, 1024, 4, 64, 256
    q, k, v = self._qkv(b, s, h, h, d, device, seed=67)
    docs = torch.arange(s) // blk
    seg = docs[None, :].repeat(b, 1).to(torch.int32).to(device)

    def run(skipping):
      splash_bias_dropout.SbdConfig.update(
          dropout_rate=0.1, document_aware_skipping=skipping, **self._blocks(blk)
      )
      try:
        q_t = q.clone().requires_grad_(True)
        k_t = k.clone().requires_grad_(True)
        v_t = v.clone().requires_grad_(True)
        out = splash_bias_dropout.sbd_splash_attention(
            q_t, k_t, v_t, segment_ids=seg, dropout_seed=4242
        )
        out.sum().backward()
      finally:
        splash_bias_dropout.SbdConfig.update(
            dropout_rate=0.0, document_aware_skipping=True, **self._blocks(512)
        )
      return out.cpu(), [t.grad.cpu() for t in (q_t, k_t, v_t)]

    out_on, grads_on = run(True)
    out_off, grads_off = run(False)
    # Dropout really is on: the undropped output differs.
    out_plain = splash_bias_dropout.sbd_splash_attention(q, k, v, segment_ids=seg)
    self.assertGreater(
        float((out_on - out_plain.cpu()).abs().max()), 1e-3, "dropout did not fire"
    )
    torch.testing.assert_close(out_on, out_off, rtol=0, atol=0)
    for name, a, b_ in zip(("dq", "dk", "dv"), grads_on, grads_off):
      torch.testing.assert_close(a, b_, rtol=0, atol=0, msg=name)

  @parameterized.named_parameters(("mha", 4), ("gqa", 2), ("mqa", 1))
  def test_dq_reduction_steps_3_matches_default(self, h_kv):
    """`dq_reduction_steps=3` (in-kernel dq reduction + dynamic dkv list).

    With grouped KV the dkv pass accumulates dk/dv across a group's q heads
    through aliased outputs. At one kv block (the 512-block case) consecutive
    q heads hit the same dk/dv block on consecutive grid steps, which Pallas
    neither writes back nor re-fetches -- the read-back used to return the
    uninitialized buffer, so dk/dv were off by O(10) for every GQA/MQA shape
    there. Against the default: MHA at one kv block is bit-identical; grouped
    KV, and four kv blocks (the 128-block case, which also compacts the dkv
    list by document), differ only in fp32 summation order (~1e-6 on v6e).
    """
    device = self.accelerator_device
    b, s, h, d = 2, 512, 4, 64
    q, k, v = self._qkv(b, s, h, h_kv, d, device, seed=71)
    segment_ids = torch.zeros(b, s, dtype=torch.int32)
    segment_ids[0, 128:] = 1
    segment_ids[1, 40:300] = 1
    seg = segment_ids.to(device)

    prev_steps = splash_bias_dropout.SbdConfig.dq_reduction_steps

    def run(steps, blk):
      splash_bias_dropout.SbdConfig.update(
          dq_reduction_steps=steps, **self._blocks(blk)
      )
      try:
        q_t, k_t, v_t = [t.clone().requires_grad_(True) for t in (q, k, v)]
        out = splash_bias_dropout.sbd_splash_attention(
            q_t, k_t, v_t, segment_ids=seg
        )
        out.sum().backward()
      finally:
        splash_bias_dropout.SbdConfig.update(
            dq_reduction_steps=prev_steps, **self._blocks(512)
        )
      return [t.grad.cpu() for t in (q_t, k_t, v_t)]

    # 1, 2 and 4 kv blocks: at <= 2 the grouped-KV dk/dv are reduced over q
    # heads outside the kernel (the aliased accumulation would race the DMA
    # pipeline), above that they accumulate in-kernel through the alias.
    for blk in (512, 256, 128):
      tol = 0.0 if (blk == 512 and h_kv == h) else 1e-5
      for name, a, b_ in zip(("dq", "dk", "dv"), run(3, blk), run(0, blk)):
        torch.testing.assert_close(
            a, b_, rtol=tol, atol=tol, msg=f"{name} (block {blk})"
        )

  def test_segment_ids_padding_layout_parity(self):
    """Right-padded ids (the encoder shape) with blocks smaller than the
    sequence.

    The skipping test is a non-decreasing packed stream; a padding mask is
    non-INcreasing (real tokens 1, padding 0), and a block that straddles the
    boundary spans [0, 1], which makes the interval test conservative rather
    than exact. Conservative is safe -- it only declines to skip -- but the
    claim needs a measurement, not an argument.
    """
    device = self.accelerator_device
    b, s, h, d, blk = 2, 512, 4, 64, 128
    q, k, v = self._qkv(b, s, h, h, d, device, seed=61)
    segment_ids = torch.ones(b, s, dtype=torch.int32)
    segment_ids[0, 300:] = 0  # boundary inside a block
    segment_ids[1, 256:] = 0  # boundary on a block edge
    allowed = segment_ids[:, :, None] == segment_ids[:, None, :]
    allowed &= torch.tril(torch.ones(s, s, dtype=torch.bool))[None]
    mask = torch.zeros(b, 1, s, s, dtype=torch.float32)
    mask.masked_fill_(~allowed[:, None], float("-inf"))

    q_ref = q.cpu().clone().requires_grad_(True)
    k_ref = k.cpu().clone().requires_grad_(True)
    v_ref = v.cpu().clone().requires_grad_(True)
    out_ref = _sdpa_ref(q_ref, k_ref, v_ref, attn_mask=mask)
    out_ref.sum().backward()

    splash_bias_dropout.SbdConfig.update(**self._blocks(blk))
    try:
      q_t = q.clone().requires_grad_(True)
      k_t = k.clone().requires_grad_(True)
      v_t = v.clone().requires_grad_(True)
      out = splash_bias_dropout.sbd_splash_attention(
          q_t, k_t, v_t, segment_ids=segment_ids.to(device)
      )
      out.sum().backward()
    finally:
      splash_bias_dropout.SbdConfig.update(**self._blocks(512))
    torch.testing.assert_close(out.cpu(), out_ref.cpu(), rtol=5e-2, atol=5e-2)
    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad), ("dk", k_t.grad, k_ref.grad),
        ("dv", v_t.grad, v_ref.grad),
    ):
      torch.testing.assert_close(
          got.cpu(), want.cpu(), rtol=8e-2, atol=8e-2, msg=name
      )

  def test_document_aware_mask_info_compacts_grid(self):
    """The grid metadata itself, with no kernel in the way.

    An end-to-end parity test cannot tell skipping from masking -- both are
    exact -- so the thing that has to be measured is the step count: this is
    the firing audit for `_document_aware_mask_info`.
    """
    import jax.numpy as jnp  # local: the rest of this suite is torch-only
    import numpy as np

    def build(seq, blk):
      return splash_bias_dropout._build_kernel(
          seq, True, 1.0, -1, False, False, blk, blk, blk, blk, blk, blk,
          False, True, None, 0.0, True, 4, True, 0, False, True,
      )[1]

    def steps(mi, seg, blk, is_dkv):
      out = splash_bias_dropout._document_aware_mask_info(
          mi, seg, blk, blk, is_dkv=is_dkv
      )
      if is_dkv:  # dense grid: tiles are zeroed, the grid is not compacted
        return int((np.asarray(out.block_mask) > 0).sum())
      return int(out.num_active_blocks[0])

    # The lane's op-point: 8 x 2048-token documents in one 16k stream, block
    # 1024. 136 causal tiles, 24 of which can hold a same-document pair.
    kernel = build(16384, 1024)
    seg = jnp.repeat(jnp.arange(8, dtype=jnp.int32), 2048)
    self.assertEqual(int(kernel.fwd_mask_info.num_active_blocks[0]), 136)
    self.assertEqual(steps(kernel.fwd_mask_info, seg, 1024, False), 24)
    self.assertEqual(int((np.asarray(kernel.dkv_mask_info.block_mask) > 0).sum()), 136)
    self.assertEqual(steps(kernel.dkv_mask_info, seg, 1024, True), 24)

    # One document is the no-op case: every causal tile stays.
    seg_one = jnp.zeros(16384, jnp.int32)
    self.assertEqual(steps(kernel.fwd_mask_info, seg_one, 1024, False), 136)
    self.assertEqual(steps(kernel.dkv_mask_info, seg_one, 1024, True), 136)

    # Row order survives compaction (find_bounds reads neighbours), and the
    # dead tail reads as a row end.
    out = splash_bias_dropout._document_aware_mask_info(
        kernel.fwd_mask_info, seg, 1024, 1024, is_dkv=False
    )
    n = int(out.num_active_blocks[0])
    rows = np.asarray(out.active_rows)
    self.assertTrue((np.diff(rows[:n]) >= 0).all(), rows[:n])
    self.assertEqual(int(rows[n]), -1)

  def test_document_aware_mask_info_has_no_gather(self):
    """The schedule builder lowers with no XLA gather or scatter.

    On TPU7x XLA offloads large gathers to SparseCore, whose lowering fails
    for schedule-sized index arrays from 16k tokens up.
    """
    import jax
    import jax.numpy as jnp

    kernel = splash_bias_dropout._build_kernel(
        16384, True, 1.0, -1, False, False, 1024, 1024, 1024, 1024, 1024, 1024,
        False, True, None, 0.0, True, 4, True, 0, False, True, 3,
    )[1]
    seg = jnp.repeat(jnp.arange(8, dtype=jnp.int32), 2048)
    for name, mi, is_dkv in (("fwd", kernel.fwd_mask_info, False),
                             ("dkv", kernel.dkv_mask_info, True)):
      hlo = jax.jit(
          lambda s, mi=mi, is_dkv=is_dkv: splash_bias_dropout._document_aware_mask_info(
              mi, s, 1024, 1024, is_dkv=is_dkv)
      ).lower(seg).as_text()
      self.assertNotIn("gather", hlo, name)
      self.assertNotIn("scatter", hlo, name)

  def test_segment_ids_block_skipping_parity(self):
    """Skipping is exact: the kernel agrees with the dense mask, and with
    itself when the skipping is turned off.

    Needs blocks SMALLER than the sequence -- at the suite's default 512
    blocks a 512-token case is one tile and nothing can be skipped.
    """
    device = self.accelerator_device
    b, s, h, d, blk = 2, 512, 4, 64, 128
    q, k, v = self._qkv(b, s, h, h, d, device, seed=47)
    docs = torch.arange(s) // blk  # 4 documents, one per block row
    segment_ids = docs[None, :].repeat(b, 1).to(torch.int32)
    allowed = (segment_ids[:, :, None] == segment_ids[:, None, :]) & torch.tril(
        torch.ones(s, s, dtype=torch.bool)
    )[None]
    mask = torch.zeros(b, 1, s, s, dtype=torch.float32)
    mask.masked_fill_(~allowed[:, None], float("-inf"))

    q_ref = q.cpu().clone().requires_grad_(True)
    k_ref = k.cpu().clone().requires_grad_(True)
    v_ref = v.cpu().clone().requires_grad_(True)
    out_ref = _sdpa_ref(q_ref, k_ref, v_ref, attn_mask=mask)
    out_ref.sum().backward()

    seg = segment_ids.to(device)
    grads = {}
    outs = {}
    for skipping in (True, False):
      splash_bias_dropout.SbdConfig.update(
          document_aware_skipping=skipping, **self._blocks(blk)
      )
      try:
        q_t = q.clone().requires_grad_(True)
        k_t = k.clone().requires_grad_(True)
        v_t = v.clone().requires_grad_(True)
        out = splash_bias_dropout.sbd_splash_attention(
            q_t, k_t, v_t, segment_ids=seg
        )
        out.sum().backward()
      finally:
        splash_bias_dropout.SbdConfig.update(
            document_aware_skipping=True, **self._blocks(512)
        )
      outs[skipping] = out.cpu()
      grads[skipping] = [t.grad.cpu() for t in (q_t, k_t, v_t)]
      torch.testing.assert_close(
          outs[skipping], out_ref.cpu(), rtol=5e-2, atol=5e-2,
          msg=f"forward, skipping={skipping}",
      )
      for name, got, want in zip(
          ("dq", "dk", "dv"), grads[skipping],
          (q_ref.grad, k_ref.grad, v_ref.grad),
      ):
        torch.testing.assert_close(
            got, want.cpu(), rtol=8e-2, atol=8e-2,
            msg=f"{name}, skipping={skipping}",
        )

    # A skipped tile contributes exactly zero, so dropping it must not move
    # a single bit of the result.
    torch.testing.assert_close(
        outs[True], outs[False], rtol=0, atol=0, msg="skipping changed the output"
    )
    for name, a, b_ in zip(("dq", "dk", "dv"), grads[True], grads[False]):
      torch.testing.assert_close(
          a, b_, rtol=0, atol=0, msg=f"skipping changed {name}"
      )

  def test_noncausal_parity(self):
    device = self.accelerator_device
    q, k, v = self._qkv(2, 512, 4, 4, 64, device, seed=31)
    expected = _sdpa_ref(q, k, v, is_causal=False)
    actual = splash_bias_dropout.sbd_splash_attention(q, k, v, is_causal=False)
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=5e-2, atol=5e-2
    )

    q_ref = q.cpu().clone().requires_grad_(True)
    k_ref = k.cpu().clone().requires_grad_(True)
    v_ref = v.cpu().clone().requires_grad_(True)
    out_ref = _sdpa_ref(q_ref, k_ref, v_ref, is_causal=False)
    out_ref.sum().backward()

    q_t = q.clone().requires_grad_(True)
    k_t = k.clone().requires_grad_(True)
    v_t = v.clone().requires_grad_(True)
    out = splash_bias_dropout.sbd_splash_attention(q_t, k_t, v_t, is_causal=False)
    out.sum().backward()
    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad),
        ("dk", k_t.grad, k_ref.grad),
        ("dv", v_t.grad, v_ref.grad),
    ):
      self.assertIsNotNone(got, f"{name} is None")
      torch.testing.assert_close(
          got.cpu(), want.cpu(), rtol=8e-2, atol=8e-2, msg=name
      )

  def _assert_matches_tokamax(self, q, k, v, name, **kwargs):
    """Forward AND backward bit-exact against the stock tokamax wrapper."""
    q_a = q.clone().requires_grad_(True)
    k_a = k.clone().requires_grad_(True)
    v_a = v.clone().requires_grad_(True)
    out_a = splash_bias_dropout.sbd_splash_attention(q_a, k_a, v_a, **kwargs)
    out_a.sum().backward()

    q_b = q.clone().requires_grad_(True)
    k_b = k.clone().requires_grad_(True)
    v_b = v.clone().requires_grad_(True)
    out_b = tokamax_splash_attention.splash_sdpa(q_b, k_b, v_b, **kwargs)
    out_b.sum().backward()

    torch.testing.assert_close(
        out_a.detach().cpu(), out_b.detach().cpu(), rtol=0.0, atol=0.0, msg=name
    )
    for gname, ga, gb in (
        ("dq", q_a.grad, q_b.grad),
        ("dk", k_a.grad, k_b.grad),
        ("dv", v_a.grad, v_b.grad),
    ):
      self.assertIsNotNone(ga, f"{name}:{gname} is None")
      torch.testing.assert_close(
          ga.cpu(), gb.cpu(), rtol=0.0, atol=0.0, msg=f"{name}:{gname}"
      )

  # TODO(jerryx99): Update tokamax pin. This test requires tokamax pin
  # 8eadbf6cfdd9 otherwise fails with ValueError: too many values to unpack
  # (expected 8)
  @unittest.skip("Requires tokamax pin 8eadbf6cfdd9")
  def test_matches_tokamax(self):
    """The no-regression contract: forward and backward parity with the
    stock tokamax wrapper across its full user surface."""
    device = self.accelerator_device
    q, k, v = self._qkv(2, 512, 4, 4, 64, device, seed=3)
    self._assert_matches_tokamax(q, k, v, "causal", is_causal=True)
    self._assert_matches_tokamax(q, k, v, "scale", is_causal=True, scale=0.05)
    self._assert_matches_tokamax(
        q, k, v, "window", is_causal=True, local_window_size=128
    )
    q1, k1, v1 = self._qkv(2, 512, 4, 1, 64, device, seed=5)
    self._assert_matches_tokamax(q1, k1, v1, "mqa", is_causal=True)

  def test_sliding_window_parity(self):
    device = self.accelerator_device
    s, w = 512, 128
    q, k, v = self._qkv(2, s, 4, 4, 64, device, seed=37)
    i = torch.arange(s)[:, None]
    j = torch.arange(s)[None, :]
    banded = torch.zeros(s, s)
    banded.masked_fill_(~((j <= i) & (j >= i - w)), float("-inf"))

    q_ref = q.cpu().clone().requires_grad_(True)
    k_ref = k.cpu().clone().requires_grad_(True)
    v_ref = v.cpu().clone().requires_grad_(True)
    out_ref = _sdpa_ref(q_ref, k_ref, v_ref, attn_mask=banded)
    out_ref.sum().backward()

    q_t = q.clone().requires_grad_(True)
    k_t = k.clone().requires_grad_(True)
    v_t = v.clone().requires_grad_(True)
    out = splash_bias_dropout.sbd_splash_attention(
        q_t, k_t, v_t, local_window_size=w
    )
    out.sum().backward()

    torch.testing.assert_close(
        out.detach().cpu(), out_ref.detach().cpu(), rtol=5e-2, atol=5e-2
    )
    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad),
        ("dk", k_t.grad, k_ref.grad),
        ("dv", v_t.grad, v_ref.grad),
    ):
      self.assertIsNotNone(got, f"{name} is None")
      torch.testing.assert_close(
          got.cpu(), want.cpu(), rtol=8e-2, atol=8e-2, msg=name
      )

  # ---------------- full-rank bias (+ exact dbias) ----------------

  def test_fullrank_bias_forward_and_backward(self):
    device = self.accelerator_device
    b, s, h, d = 2, 512, 4, 64
    q, k, v = self._qkv(b, s, h, h, d, device, seed=7)
    g = torch.Generator().manual_seed(11)
    bias = (torch.randn(h, s, s, generator=g) * 0.5).to(device)

    # Reference: SDPA with additive causal+bias mask, autograd on CPU.
    q_ref = q.cpu().clone().requires_grad_(True)
    k_ref = k.cpu().clone().requires_grad_(True)
    v_ref = v.cpu().clone().requires_grad_(True)
    bias_ref = bias.cpu().clone().requires_grad_(True)
    mask = _causal_bias_mask(bias_ref, s, torch.device("cpu"))
    out_ref = _sdpa_ref(q_ref, k_ref, v_ref, attn_mask=mask)
    out_ref.sum().backward()

    q_t = q.clone().requires_grad_(True)
    k_t = k.clone().requires_grad_(True)
    v_t = v.clone().requires_grad_(True)
    bias_t = bias.clone().requires_grad_(True)
    out = splash_bias_dropout.sbd_splash_attention(q_t, k_t, v_t, bias_t)
    out.sum().backward()

    torch.testing.assert_close(
        out.detach().cpu(), out_ref.detach().cpu(), rtol=5e-2, atol=5e-2
    )
    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad),
        ("dk", k_t.grad, k_ref.grad),
        ("dv", v_t.grad, v_ref.grad),
        ("dbias", bias_t.grad, bias_ref.grad),
    ):
      self.assertIsNotNone(got, f"{name} is None")
      torch.testing.assert_close(
          got.cpu(), want.cpu(), rtol=8e-2, atol=8e-2, msg=name
      )

  def test_unflatten_shim_leaves_batch_shared_bias_alone(self):
    """Under the trainer's unflatten shim only q/k/v are token streams.

    The (H, S, S) bias is batch-shared and must not be unflattened (H is not
    divisible by S here, so doing so would raise). Flat [T, H, D] inputs come
    back flat, and documented 4-D inputs come back 4-D.
    """
    device = self.accelerator_device
    b, s, h, d = 2, 512, 4, 64
    q, k, v = self._qkv(b, s, h, h, d, device, seed=7)
    g = torch.Generator().manual_seed(11)
    bias = (torch.randn(h, s, s, generator=g) * 0.5).to(device)
    want = splash_bias_dropout.sbd_splash_attention(q, k, v, bias)

    with unflatten_shim.set_unflatten_seq_len(s):
      got_flat = splash_bias_dropout.sbd_splash_attention(
          q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), bias
      )
      got_4d = splash_bias_dropout.sbd_splash_attention(q, k, v, bias)

    self.assertEqual(tuple(got_flat.shape), (b * s, h, d))
    self.assertEqual(tuple(got_4d.shape), (b, s, h, d))
    torch.testing.assert_close(got_flat.cpu(), want.flatten(0, 1).cpu())
    torch.testing.assert_close(got_4d.cpu(), want.cpu())

  # ---------------- per-key bias (+ trainable rank-reduced dbias) --------

  def test_perkey_bias_dbias(self):
    device = self.accelerator_device
    # Deliberately the SAME shape as the full-rank bias test: the per-key
    # config must retrace via the static args, not via a shape change.
    b, s, h, d = 2, 512, 4, 64
    rows = splash_bias_dropout.PER_KEY_ROWS
    splash_bias_dropout.SbdConfig.update(
        bias_per_key_rows=rows, bias_per_key_dbias=True
    )
    try:
      q, k, v = self._qkv(b, s, h, h, d, device, seed=13)
      g = torch.Generator().manual_seed(17)
      row = (torch.randn(h, s, generator=g) * 0.5).to(device)
      bias_pk = row[:, None, :].expand(h, rows, s).contiguous()

      q_ref = q.cpu().clone().requires_grad_(True)
      k_ref = k.cpu().clone().requires_grad_(True)
      v_ref = v.cpu().clone().requires_grad_(True)
      bias_pk_ref = bias_pk.cpu().clone().requires_grad_(True)
      # The kernel contract: row 0 is the logical bias, broadcast over q.
      mask = _causal_bias_mask(
          bias_pk_ref[:, 0:1, :].expand(h, s, s), s, torch.device("cpu")
      )
      out_ref = _sdpa_ref(q_ref, k_ref, v_ref, attn_mask=mask)
      out_ref.sum().backward()

      q_t = q.clone().requires_grad_(True)
      k_t = k.clone().requires_grad_(True)
      v_t = v.clone().requires_grad_(True)
      bias_t = bias_pk.clone().requires_grad_(True)
      out = splash_bias_dropout.sbd_splash_attention(q_t, k_t, v_t, bias_t)
      out.sum().backward()

      torch.testing.assert_close(
          out.detach().cpu(), out_ref.detach().cpu(), rtol=5e-2, atol=5e-2
      )
      self.assertIsNotNone(bias_t.grad, "per-key dbias is None")
      # Row 0 carries the rank-reduced gradient; pad rows are exactly zero.
      # The reference's autograd concentrates its grad in row 0 as well
      # (rows 1.. are unused by the reference mask).
      torch.testing.assert_close(
          bias_t.grad[:, 0, :].cpu(),
          bias_pk_ref.grad[:, 0, :].cpu(),
          rtol=8e-2,
          atol=8e-2,
      )
      self.assertEqual(
          bias_t.grad[:, 1:, :].abs().sum().item(), 0.0,
          "per-key dbias pad rows must be exactly zero",
      )
    finally:
      splash_bias_dropout.SbdConfig.update(
          bias_per_key_rows=0, bias_per_key_dbias=False
      )

  # ---------------- PRNG dropout ----------------

  def test_dropout_determinism_and_grads(self):
    device = self.accelerator_device
    b, s, h, d = 4, 512, 4, 64
    splash_bias_dropout.SbdConfig.update(dropout_rate=0.1)
    try:
      q, k, v = self._qkv(b, s, h, h, d, device, seed=23)

      out_a = splash_bias_dropout.sbd_splash_attention(
          q, k, v, dropout_seed=1234
      )
      out_b = splash_bias_dropout.sbd_splash_attention(
          q, k, v, dropout_seed=1234
      )
      out_c = splash_bias_dropout.sbd_splash_attention(
          q, k, v, dropout_seed=999
      )
      # Same seed => identical; different seed => different masks.
      torch.testing.assert_close(out_a.cpu(), out_b.cpu(), rtol=0.0, atol=0.0)
      self.assertGreater(
          (out_a - out_c).abs().sum().item(), 0.0,
          "different dropout seeds produced identical outputs",
      )
      # Dropout keeps the output in the no-dropout ballpark (fold-rescale
      # preserves scale): mean magnitude within 20% of the reference.
      ref = _sdpa_ref(q, k, v, is_causal=True)
      self.assertLess(
          (out_a.abs().mean() - ref.abs().mean()).abs().item(),
          0.2 * ref.abs().mean().item(),
      )

      # Gradients flow and are finite through the dropout path.
      q_t = q.clone().requires_grad_(True)
      k_t = k.clone().requires_grad_(True)
      v_t = v.clone().requires_grad_(True)
      out = splash_bias_dropout.sbd_splash_attention(
          q_t, k_t, v_t, dropout_seed=1234
      )
      out.sum().backward()
      for name, grad in (("dq", q_t.grad), ("dk", k_t.grad), ("dv", v_t.grad)):
        self.assertIsNotNone(grad, f"{name} is None")
        self.assertTrue(
            torch.isfinite(grad).all().item(), f"{name} has non-finite values"
        )
        self.assertGreater(grad.abs().sum().item(), 0.0, f"{name} is all zeros")
    finally:
      splash_bias_dropout.SbdConfig.update(dropout_rate=0.0)


if __name__ == "__main__":
  absltest.main()
