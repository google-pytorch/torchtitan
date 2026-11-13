"""CPU unit tests for ``experiments/tpu/components/fused_qk_norm_rope.py``."""

import copy
import pickle
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
import torch.nn.functional as F
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.components import bld_layout
from torchtitan.experiments.tpu.components import fused_qk_norm_rope
from torchtitan.experiments.tpu.components import qkv as tpu_qkv
from torchtitan.experiments.tpu.components import unflatten_shim
from torchtitan.experiments.tpu.examples import gemma4
from torchtitan.experiments.tpu.overrides import bf16_rope
from torchtitan.experiments.tpu.overrides import real_rope
from torchtitan.models import common
from torchtitan.models import llama3
from torchtitan.models import qwen3

_DIM = 128
_HEAD_DIM = 32
_BATCH = 2
_SEQ_LEN = 16
_MAX_LEN = 64


def _fake_qk_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
  """CPU reference standing in for the Pallas kernel to verify module wiring."""
  y = F.rms_norm(x, (x.shape[-1],), weight, eps)
  rotary_half = x.shape[-1] // 2
  cs_shape = (*x.shape[:-2], 1, rotary_half)
  return bld_layout.cossin_rotate(
      y, cos.float().reshape(cs_shape), sin.float().reshape(cs_shape)
  )


def _build_qwen3_block(*, use_bf16_rope: bool = False) -> nn.Module:
  """Builds a single-layer Qwen3 block with packed QKV and QK RMSNorm."""
  rope_cfg = common.CosSinRoPE.Config(
      dim=_HEAD_DIM, max_context_length=_MAX_LEN
  )
  if use_bf16_rope:
    rope_cfg = bf16_rope.tpu_activation_dtype_cossin_rope(rope_cfg)
  cfg = qwen3._build_qwen3_layers(  # pylint: disable=protected-access
      n_layers=1,
      dim=_DIM,
      n_heads=4,
      n_kv_heads=2,
      head_dim=_HEAD_DIM,
      hidden_dim=256,
      fuse_qkv=True,
      attn_backend="sdpa",
      rope=rope_cfg,
  )[0]
  cfg = tpu_qkv.TpuPackedQKVConverter.Config().build().convert(cfg)
  block = cfg.build()
  for param in block.parameters():
    nn.init.normal_(param, std=0.05)
  with torch.no_grad():
    block.attention.q_norm.weight.copy_(1.0 + 0.2 * torch.randn(_HEAD_DIM))
    block.attention.k_norm.weight.copy_(0.7 - 0.15 * torch.randn(_HEAD_DIM))
  block.attention.q_norm.eps = 1e-6
  block.attention.k_norm.eps = 1e-5
  return block


class FusedQkNormRopeComponentTest(parameterized.TestCase):
  """Tests module eligibility, forward/backward wiring, and fallbacks."""

  def _mock_kernel(self):
    fake_mod = types.SimpleNamespace(
        supports=mock.Mock(return_value=True),
        qk_norm_rope=mock.Mock(side_effect=_fake_qk_norm_rope),
    )
    return (
        mock.patch.object(
            fused_qk_norm_rope,
            "qk_norm_rope_kernel",
            fake_mod,
        ),
        fake_mod,
    )

  def test_patches_only_eligible_attention_and_warns_when_zero(self):
    """Patches CosSinRoPE+RMSNorm blocks and skips BF16 or ComplexRoPE ones."""
    eligible = _build_qwen3_block(use_bf16_rope=False)
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(eligible), 1)
    # Repeated calls are idempotent and preserve the count.
    rope_cls_once = type(eligible.attention.rope)
    norm_cls_once = type(eligible.attention.q_norm)
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(eligible), 1)
    self.assertIs(type(eligible.attention.rope), rope_cls_once)
    self.assertIs(type(eligible.attention.q_norm), norm_cls_once)
    self.assertIs(pickle.loads(pickle.dumps(rope_cls_once)), rope_cls_once)  # pylint: disable=g-unsafe-pickle-load
    self.assertIs(pickle.loads(pickle.dumps(norm_cls_once)), norm_cls_once)  # pylint: disable=g-unsafe-pickle-load

    ineligible_bf16 = _build_qwen3_block(use_bf16_rope=True)
    with self.assertLogs(fused_qk_norm_rope.logger, level="WARNING") as cm:
      self.assertEqual(
          fused_qk_norm_rope.apply_fused_qk_norm_rope(ineligible_bf16), 0
      )
    self.assertIn("0 eligible attention modules", cm.output[0])

    llama_rope = real_rope.tpu_real_arithmetic_complex_rope(
        common.ComplexRoPE.Config(dim=_HEAD_DIM, max_context_length=_MAX_LEN)
    )
    llama_cfg = llama3._build_llama3_layers(  # pylint: disable=protected-access
        n_layers=1,
        dim=_DIM,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=256,
        fuse_qkv=True,
        attn_backend="sdpa",
        rope=llama_rope,
    )[0]
    llama_block = llama_cfg.build()
    with self.assertLogs(fused_qk_norm_rope.logger, level="WARNING"):
      self.assertEqual(
          fused_qk_norm_rope.apply_fused_qk_norm_rope(llama_block), 0
      )

  @parameterized.named_parameters(
      ("bld_3d", True),
      ("folded_2d", False),
  )
  def test_qwen3_fused_forward_and_backward_match_unfused(self, use_bld: bool):
    """Verifies forward and backward parity on both 3-D BLD and 2-D blocks."""
    torch.manual_seed(0)
    ref_block = _build_qwen3_block()
    pat_block = copy.deepcopy(ref_block)
    if use_bld:
      self.assertEqual(bld_layout.apply_bld_layout(pat_block), 1)
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(pat_block), 1)
    self.assertEqual(set(pat_block.state_dict()), set(ref_block.state_dict()))

    x_ref = torch.randn(_BATCH * _SEQ_LEN, _DIM, requires_grad=True)
    x_pat = x_ref.detach().clone().requires_grad_(True)
    dy = torch.randn(_BATCH * _SEQ_LEN, _DIM)
    positions = torch.arange(_SEQ_LEN).repeat(_BATCH)

    patch_ctx, fake_mod = self._mock_kernel()
    with unflatten_shim.set_unflatten_seq_len(_SEQ_LEN):
      out_ref = ref_block(x_ref, None, positions)
      out_ref.backward(dy)
      with patch_ctx:
        out_pat = pat_block(x_pat, None, positions)
        out_pat.backward(dy)

    self.assertEqual(fake_mod.qk_norm_rope.call_count, 2)
    torch.testing.assert_close(out_pat, out_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x_pat.grad, x_ref.grad, rtol=1e-5, atol=1e-5)
    for name, p_ref in ref_block.named_parameters():
      p_pat = dict(pat_block.named_parameters())[name]
      torch.testing.assert_close(
          p_pat.grad, p_ref.grad, rtol=1e-5, atol=1e-5, msg=name
      )

  def test_2d_path_handles_sequence_gathering_qkv_projection(self):
    """Preserves output token count when QKV projection all-gathers tokens."""
    torch.manual_seed(1)
    ref_block = _build_qwen3_block()
    orig_qkv = ref_block.attention.qkv_linear

    class _GatheringQKV(nn.Module):
      """Simulates AllGatherFusedQKVLinear expanding T_shard -> 2 * T_shard."""

      def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

      def forward(self, x_shard: torch.Tensor):
        return self.inner(torch.cat([x_shard, x_shard], dim=0))

    ref_block.attention.qkv_linear = _GatheringQKV(orig_qkv)
    ref_block.attention.wo = nn.Linear(_DIM, _DIM, bias=False)
    pat_block = copy.deepcopy(ref_block)
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(pat_block), 1)

    x_shard = torch.randn(_SEQ_LEN, _DIM, requires_grad=True)
    positions = torch.arange(2 * _SEQ_LEN)
    patch_ctx, fake_mod = self._mock_kernel()
    with patch_ctx:
      out_pat = pat_block.attention(x_shard, None, positions)
    out_ref = ref_block.attention(x_shard.detach(), None, positions)
    self.assertEqual(fake_mod.qk_norm_rope.call_count, 2)
    self.assertEqual(out_pat.shape, (2 * _SEQ_LEN, _DIM))
    torch.testing.assert_close(out_pat, out_ref, rtol=1e-5, atol=1e-5)

  def test_gemma4_sliding_and_global_keqv_with_dict_masks(self):
    """Verifies Gemma 4 sliding + global k_eq_v layers with dict masks."""
    torch.manual_seed(2)
    layer_cfgs = gemma4._build_gemma4_layers(  # pylint: disable=protected-access
        n_layers=2,
        dim=_DIM,
        hidden_dim=256,
        n_heads=4,
        n_kv_heads=2,
        n_global_kv_heads=2,
        head_dim=_HEAD_DIM,
        global_head_dim=_HEAD_DIM * 2,
        norm_eps=1e-6,
        max_seq_len=_MAX_LEN,
        sliding_window_size=8,
        global_attn_every_n_layers=2,
        use_k_eq_v_on_global=True,
        attn_backend="sdpa",
    )
    ref_layers = nn.ModuleList([cfg.build() for cfg in layer_cfgs])
    for param in ref_layers.parameters():
      nn.init.normal_(param, std=0.05)

    class _MaskRecordingInnerAttn(nn.Module):
      """Records the dispatched mask sentinel before forwarding None to SDPA."""

      def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.seen_masks: list[object] = []

      def forward(self, *args, **kwargs):
        self.seen_masks.append(kwargs.get("attention_masks"))
        kwargs["attention_masks"] = None
        return self.inner(*args, **kwargs)

    for layer in ref_layers:
      layer.attention.inner_attention = _MaskRecordingInnerAttn(
          layer.attention.inner_attention
      )

    pat_layers = copy.deepcopy(ref_layers)
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(pat_layers), 2)
    self.assertEqual(set(pat_layers.state_dict()), set(ref_layers.state_dict()))

    total_tokens = _BATCH * _SEQ_LEN
    x_ref = torch.randn(total_tokens, _DIM, requires_grad=True)
    x_pat = x_ref.detach().clone().requires_grad_(True)
    dy = torch.randn_like(x_ref)
    positions = torch.arange(_SEQ_LEN).repeat(_BATCH)
    sliding_sentinel = object()
    basic_sentinel = object()
    masks = {
        "sliding_window_mask": sliding_sentinel,
        "basic_mask": basic_sentinel,
    }

    with unflatten_shim.set_unflatten_seq_len(_SEQ_LEN):
      h_ref = x_ref
      for layer in ref_layers:
        h_ref = layer(h_ref, masks, positions)
      h_ref.backward(dy)

      patch_ctx, fake_mod = self._mock_kernel()
      with patch_ctx:
        h_pat = x_pat
        for pat_layer in pat_layers:
          for cls in (
              type(pat_layer.attention),
              type(pat_layer.attention.rope),
              type(pat_layer.attention.q_norm),
              type(pat_layer.attention.k_norm),
          ):
            self.assertIs(pickle.loads(pickle.dumps(cls)), cls)  # pylint: disable=g-unsafe-pickle-load
          h_pat = pat_layer(h_pat, masks, positions)
        h_pat.backward(dy)

    self.assertEqual(
        pat_layers[0].attention.inner_attention.seen_masks, [sliding_sentinel]
    )
    self.assertEqual(
        pat_layers[1].attention.inner_attention.seen_masks, [basic_sentinel]
    )

    self.assertEqual(fake_mod.qk_norm_rope.call_count, 4)
    torch.testing.assert_close(h_pat, h_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x_pat.grad, x_ref.grad, rtol=1e-5, atol=1e-5)
    for (name, p_ref), (_, p_pat) in zip(
        ref_layers.named_parameters(),
        pat_layers.named_parameters(),
        strict=True,
    ):
      torch.testing.assert_close(
          p_pat.grad, p_ref.grad, rtol=1e-5, atol=1e-5, msg=name
      )

  @parameterized.named_parameters(
      ("bld", True),
      ("non_bld", False),
  )
  def test_layer_stack_unregistered_norm_weights_still_patch_and_run(
      self, use_bld: bool
  ):
    """Supports LayerStack functional_call after leaf norm weights are cleared."""
    block = _build_qwen3_block()
    ref = copy.deepcopy(block)
    saved_params = dict(block.named_parameters())
    # Simulate LayerStack.unregister_and_allocate() deleting leaf parameters
    # before workarounds.apply_patches runs.
    del block.attention.q_norm._parameters["weight"]  # pylint: disable=protected-access
    del block.attention.k_norm._parameters["weight"]  # pylint: disable=protected-access

    if use_bld:
      self.assertEqual(bld_layout.apply_bld_layout(block), 1)
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(block), 1)

    x = torch.randn(_BATCH * _SEQ_LEN, _DIM)
    positions = torch.arange(_SEQ_LEN).repeat(_BATCH)
    patch_ctx, fake_mod = self._mock_kernel()
    with unflatten_shim.set_unflatten_seq_len(_SEQ_LEN):
      with patch_ctx:
        out = torch.func.functional_call(
            block, saved_params, (x, None, positions)
        )
      out_ref = ref(x, None, positions)
    self.assertEqual(fake_mod.qk_norm_rope.call_count, 2)
    torch.testing.assert_close(out, out_ref, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("bld", True),
      ("non_bld", False),
  )
  def test_post_patch_deepcopy_preserves_wiring_and_parity(
      self, use_bld: bool
  ):
    """Deepcopying a patched block keeps wiring, norms, and parity intact."""
    torch.manual_seed(4)
    ref_block = _build_qwen3_block()
    pat_block = copy.deepcopy(ref_block)
    if use_bld:
      self.assertEqual(bld_layout.apply_bld_layout(pat_block), 1)
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(pat_block), 1)

    copied = copy.deepcopy(pat_block)
    self.assertTrue(
        fused_qk_norm_rope._is_patched(copied.attention)  # pylint: disable=protected-access
    )
    if use_bld:
      attn_cls = type(copied.attention)
      self.assertIs(pickle.loads(pickle.dumps(attn_cls)), attn_cls)  # pylint: disable=g-unsafe-pickle-load
    else:
      norms = copied.attention.rope._fused_qk_norms  # pylint: disable=protected-access
      self.assertIs(norms[0], copied.attention.q_norm)
      self.assertIs(norms[1], copied.attention.k_norm)

    x_ref = torch.randn(_BATCH * _SEQ_LEN, _DIM, requires_grad=True)
    x_copy = x_ref.detach().clone().requires_grad_(True)
    dy = torch.randn(_BATCH * _SEQ_LEN, _DIM)
    positions = torch.arange(_SEQ_LEN).repeat(_BATCH)

    patch_ctx, fake_mod = self._mock_kernel()
    with unflatten_shim.set_unflatten_seq_len(_SEQ_LEN):
      out_ref = ref_block(x_ref, None, positions)
      out_ref.backward(dy)
      with patch_ctx:
        out_copy = copied(x_copy, None, positions)
        out_copy.backward(dy)

    self.assertEqual(fake_mod.qk_norm_rope.call_count, 2)
    torch.testing.assert_close(out_copy, out_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x_copy.grad, x_ref.grad, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("bld_fused", True, True),
      ("bld_fallback", True, False),
      ("non_bld_fused", False, True),
      ("non_bld_fallback", False, False),
  )
  def test_norm_weight_property_is_read_once_per_norm(
      self, use_bld: bool, supports_fused: bool
  ):
    """SimpleFSDP @property weights are read once per layer on fast & fallback."""
    torch.manual_seed(5)
    ref_block = _build_qwen3_block()
    pat_block = copy.deepcopy(ref_block)
    if use_bld:
      self.assertEqual(bld_layout.apply_bld_layout(pat_block), 1)
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(pat_block), 1)

    def _wrap_counting_weight(norm: nn.RMSNorm) -> list[int]:
      reads = [0]

      class _CountingRMSNorm(type(norm)):

        @property
        def weight(self):
          reads[0] += 1
          return self._parameters["weight"]

      norm.__class__ = _CountingRMSNorm
      return reads

    q_reads = _wrap_counting_weight(pat_block.attention.q_norm)
    k_reads = _wrap_counting_weight(pat_block.attention.k_norm)

    x = torch.randn(_BATCH * _SEQ_LEN, _DIM)
    positions = torch.arange(_SEQ_LEN).repeat(_BATCH)
    patch_ctx, fake_mod = self._mock_kernel()
    fake_mod.supports.return_value = supports_fused
    with unflatten_shim.set_unflatten_seq_len(_SEQ_LEN):
      out_ref = ref_block(x, None, positions)
      with patch_ctx:
        out_pat = pat_block(x, None, positions)

    self.assertEqual(q_reads[0], 1)
    self.assertEqual(k_reads[0], 1)
    self.assertEqual(fake_mod.qk_norm_rope.call_count, 2 if supports_fused else 0)
    torch.testing.assert_close(out_pat, out_ref, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("bld", True),
      ("non_bld", False),
  )
  def test_bf16_x_fp32_weight_runs_fused_and_matches_unfused_dtype(
      self, use_bld: bool
  ):
    """BF16 input + FP32 weight runs fused and matches F.rms_norm BF16 dtype."""
    torch.manual_seed(6)
    block = _build_qwen3_block()
    if use_bld:
      self.assertEqual(bld_layout.apply_bld_layout(block), 1)
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(block), 1)
    self.assertEqual(block.attention.q_norm.weight.dtype, torch.float32)
    self.assertEqual(block.attention.k_norm.weight.dtype, torch.float32)

    if use_bld:
      xq = torch.randn(_BATCH, _SEQ_LEN, 4, _HEAD_DIM, dtype=torch.bfloat16)
      xk = torch.randn(_BATCH, _SEQ_LEN, 2, _HEAD_DIM, dtype=torch.bfloat16)
      cache_ref = block.attention.rope._reshape_cache(  # pylint: disable=protected-access
          xq.reshape(_BATCH * _SEQ_LEN, -1, _HEAD_DIM),
          torch.arange(_SEQ_LEN).repeat(_BATCH),
      ).view(_BATCH, _SEQ_LEN, 1, 2 * _HEAD_DIM)
    else:
      xq = torch.randn(_BATCH * _SEQ_LEN, 4, _HEAD_DIM, dtype=torch.bfloat16)
      xk = torch.randn(_BATCH * _SEQ_LEN, 2, _HEAD_DIM, dtype=torch.bfloat16)
      cache_ref = block.attention.rope._reshape_cache(  # pylint: disable=protected-access
          xq, torch.arange(_SEQ_LEN).repeat(_BATCH)
      )
    positions = torch.arange(_SEQ_LEN).repeat(_BATCH)

    xq_ref = F.rms_norm(
        xq, (_HEAD_DIM,), block.attention.q_norm.weight, block.attention.q_norm.eps
    )
    xk_ref = F.rms_norm(
        xk, (_HEAD_DIM,), block.attention.k_norm.weight, block.attention.k_norm.eps
    )
    q_ref, k_ref = common.CosSinRoPE.apply_rotary_emb(xq_ref, xk_ref, cache_ref)
    self.assertEqual(q_ref.dtype, torch.bfloat16)
    self.assertEqual(k_ref.dtype, torch.bfloat16)

    patch_ctx, fake_mod = self._mock_kernel()
    with patch_ctx:
      if use_bld:
        q_out, k_out = block.attention._apply_qk_norm_and_rope(  # pylint: disable=protected-access
            xq, xk, positions
        )
      else:
        q_out, k_out = block.attention.rope(xq, xk, positions)

    self.assertEqual(fake_mod.qk_norm_rope.call_count, 2)
    self.assertEqual(q_out.dtype, q_ref.dtype)
    self.assertEqual(k_out.dtype, k_ref.dtype)
    torch.testing.assert_close(q_out, q_ref, rtol=0, atol=0)
    torch.testing.assert_close(k_out, k_ref, rtol=0, atol=0)

  def test_non_bld_rope_inverse_and_key_none_safety(self):
    """Non-BLD rope mixin forwards inverse=True unfused and rejects key=None."""
    block = _build_qwen3_block()
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(block), 1)
    xq = torch.randn(_BATCH * _SEQ_LEN, 4, _HEAD_DIM)
    xk = torch.randn(_BATCH * _SEQ_LEN, 2, _HEAD_DIM)
    positions = torch.arange(_SEQ_LEN).repeat(_BATCH)

    with self.assertRaisesRegex(RuntimeError, "key"):
      block.attention.rope(xq, positions=positions)
    with self.assertRaisesRegex(RuntimeError, "key"):
      block.attention.rope(xq, None, positions, inverse=True)

    patch_ctx, fake_mod = self._mock_kernel()
    with (
        patch_ctx,
        mock.patch.object(
            common.CosSinRoPE,
            "apply_rotary_emb",
            wraps=common.CosSinRoPE.apply_rotary_emb,
        ) as spy,
    ):
      with self.assertRaises(NotImplementedError):
        block.attention.rope(xq, xk, positions, inverse=True)

    fake_mod.qk_norm_rope.assert_not_called()
    self.assertEqual(spy.call_count, 1)
    self.assertTrue(spy.call_args.kwargs["inverse"])
    expected_q = F.rms_norm(
        xq, (_HEAD_DIM,), block.attention.q_norm.weight, block.attention.q_norm.eps
    )
    expected_k = F.rms_norm(
        xk, (_HEAD_DIM,), block.attention.k_norm.weight, block.attention.k_norm.eps
    )
    torch.testing.assert_close(spy.call_args.args[0], expected_q, rtol=0, atol=0)
    torch.testing.assert_close(spy.call_args.args[1], expected_k, rtol=0, atol=0)

  def test_shared_rope_skips_non_bld_patching_and_tightens_is_patched(self):
    """Sharing one RoPE across attention modules skips non-BLD patching."""
    torch.manual_seed(7)
    block0 = _build_qwen3_block()
    block1 = _build_qwen3_block()
    block1.attention.rope = block0.attention.rope
    model = nn.ModuleList([block0, block1])
    ref = copy.deepcopy(model)

    with self.assertLogs(fused_qk_norm_rope.logger, level="WARNING"):
      self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(model), 0)
    self.assertFalse(
        fused_qk_norm_rope._is_patched(block0.attention)  # pylint: disable=protected-access
    )
    self.assertFalse(
        fused_qk_norm_rope._is_patched(block1.attention)  # pylint: disable=protected-access
    )
    self.assertIs(type(block0.attention.rope), common.CosSinRoPE)
    self.assertIs(type(block0.attention.q_norm), common.RMSNorm)

    x = torch.randn(_BATCH * _SEQ_LEN, _DIM)
    positions = torch.arange(_SEQ_LEN).repeat(_BATCH)
    out = block1(block0(x, None, positions), None, positions)
    out_ref = ref[1](ref[0](x, None, positions), None, positions)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)

    # Even if an already-patched rope is later aliased onto another attention
    # module, _is_patched returns True only for the owning (q_norm, k_norm).
    solo = _build_qwen3_block()
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(solo), 1)
    other = _build_qwen3_block()
    other.attention.rope = solo.attention.rope
    self.assertTrue(
        fused_qk_norm_rope._is_patched(solo.attention)  # pylint: disable=protected-access
    )
    self.assertFalse(
        fused_qk_norm_rope._is_patched(other.attention)  # pylint: disable=protected-access
    )

  def test_mismatched_rope_cache_width_falls_back_cleanly(self):
    """Returns None instead of slicing wrong offsets when cache width differs."""
    block = _build_qwen3_block()
    # Build a CosSinRoPE with dim = 2 * head_dim so cache.shape[-1] == 4 * hd.
    block.attention.rope = common.CosSinRoPE.Config(
        dim=_HEAD_DIM * 2, max_context_length=_MAX_LEN
    ).build()
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(block), 1)
    xq = torch.randn(_BATCH * _SEQ_LEN, 4, _HEAD_DIM)
    xk = torch.randn(_BATCH * _SEQ_LEN, 2, _HEAD_DIM)
    positions = torch.arange(_SEQ_LEN).repeat(_BATCH)
    patch_ctx, fake_mod = self._mock_kernel()
    with patch_ctx:
      res = fused_qk_norm_rope._try_fused_qk_norm_rope(  # pylint: disable=protected-access
          block.attention.rope,
          block.attention.q_norm,
          block.attention.k_norm,
          xq,
          xk,
          positions,
      )
    self.assertIsNone(res)
    fake_mod.qk_norm_rope.assert_not_called()

  @parameterized.named_parameters(
      ("bld_and_fused", True, True, 1),
      ("non_bld_and_fused", False, True, 1),
      ("tp_skips_bld_keeps_fused", True, True, 2),
      ("bld_only", True, False, 1),
  )
  def test_workarounds_apply_patches_combinations(
      self, bld_enabled: bool, fused_enabled: bool, tp_degree: int
  ):
    """Verifies orthogonal BLD and fused QK-norm+RoPE dispatch in workarounds."""
    block = _build_qwen3_block()
    job_config = types.SimpleNamespace(
        parallelism=types.SimpleNamespace(
            context_parallel_degree=1, tensor_parallel_degree=tp_degree
        ),
        splash_attention_kernel=types.SimpleNamespace(
            use_splash_attention_kernel=False,
            use_tokamax_splash_attention_kernel=False,
        ),
        loss_kernel=types.SimpleNamespace(use_loss_kernel=False),
        qwen3=types.SimpleNamespace(
            use_gmm_kernel=False, use_fill_indices_kernel=False
        ),
        tpu_config=types.SimpleNamespace(
            apply_rope_complex_workaround=False,
            bld_block_layout=bld_enabled,
            fused_qk_norm_rope=fused_enabled,
        ),
    )
    workarounds.apply_patches(block, job_config)
    self.assertEqual(
        fused_qk_norm_rope._is_patched(block.attention),  # pylint: disable=protected-access
        fused_enabled,
    )
    self.assertEqual(
        isinstance(block, bld_layout._BLDBlock),  # pylint: disable=protected-access
        bld_enabled and tp_degree == 1,
    )


if __name__ == "__main__":
  absltest.main()
