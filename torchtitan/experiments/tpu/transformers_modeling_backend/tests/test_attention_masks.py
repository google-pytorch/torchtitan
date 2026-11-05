"""The splash adapter's mask derivation against HF's own mask functions (CPU).

The parity tests run ``_segment_ids`` / ``_fallback_mask`` on both the TPU and
the CPU side, so a bug in the derivation itself is invisible to them. These
compare it with ``transformers.masking_utils``: every (query, key) pair of the
dense mask the SDPA fallback builds must match what HF's mask functions allow.
"""

from types import SimpleNamespace

from absl.testing import absltest
from absl.testing import parameterized
import torch
from transformers import masking_utils as hf

from torchtitan.experiments.tpu.transformers_modeling_backend import attention


def _module(attn_mask_type="block_causal", sliding_window=None):
  return SimpleNamespace(config=SimpleNamespace(attn_mask_type=attn_mask_type),
                         sliding_window=sliding_window)


def _hf_mask(mask_fn, batch, seq):
  """HF's mask function evaluated on every (q, kv) pair: (B, 1, S, S) bool.

  HF's combinators expect tensor indices (they are written for vmap).
  """
  t = torch.tensor
  return torch.tensor([[[[bool(mask_fn(t(b), t(0), t(q), t(kv))) for kv in range(seq)]
                         for q in range(seq)]] for b in range(batch)])


def _ours(module, positions, is_causal, window=None, padding=None):
  batch, seq = positions.shape
  q = torch.zeros(batch, 1, seq, 8)
  mask, fb_causal = attention._fallback_mask(
      module, q, q, padding, {"position_ids": positions}, is_causal, window)
  assert fb_causal is False
  return mask


# Packed streams as torchtitan's packer produces them: every document (and
# every split chunk of one) restarts its positions at 0.
_POSITIONS = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 3, 4, 5]])


class SegmentIdsTest(absltest.TestCase):

  def test_documents_match_hf_packed_sequence_indices(self):
    ours = attention.document_segment_ids(_POSITIONS)
    theirs = hf.find_packed_sequence_indices(_POSITIONS)
    same_ours = ours[:, :, None] == ours[:, None, :]
    same_theirs = theirs[:, :, None] == theirs[:, None, :]
    self.assertTrue(torch.equal(same_ours, same_theirs))

  def test_precomputed_segment_ids_win_over_positions(self):
    seg = torch.ones_like(_POSITIONS)
    got = attention._segment_ids(_module(), None, {"tpu_segment_ids": seg,
                                                  "position_ids": _POSITIONS})
    self.assertTrue(torch.equal(got, seg))

  def test_missing_positions_raise_under_document_masking(self):
    with self.assertRaisesRegex(ValueError, "position_ids or tpu_segment_ids"):
      attention._segment_ids(_module(), None, {})

  def test_plain_causal_needs_no_positions(self):
    self.assertIsNone(attention._segment_ids(_module("causal"), None, {}))

  def test_padding_gets_segment_zero(self):
    padding = torch.ones_like(_POSITIONS)
    padding[:, -3:] = 0
    seg = attention._segment_ids(_module(), padding, {"position_ids": _POSITIONS})
    self.assertTrue((seg[:, -3:] == 0).all())
    self.assertTrue((seg[:, :-3] > 0).all())


class FallbackMaskTest(parameterized.TestCase):

  def test_document_causal(self):
    packed = hf.find_packed_sequence_indices(_POSITIONS)
    expected = _hf_mask(hf.and_masks(hf.causal_mask_function,
                                     hf.packed_sequence_mask_function(packed)), 1, 16)
    self.assertTrue(torch.equal(_ours(_module(), _POSITIONS, True), expected))

  def test_document_bidirectional(self):
    packed = hf.find_packed_sequence_indices(_POSITIONS)
    expected = _hf_mask(hf.and_masks(hf.bidirectional_mask_function,
                                     hf.packed_sequence_mask_function(packed)), 1, 16)
    self.assertTrue(torch.equal(_ours(_module(), _POSITIONS, False), expected))

  @parameterized.parameters(1, 2, 3, 5)
  def test_causal_sliding_window(self, window):
    # One document, so only causality and the window shape the mask; the
    # boundary keys (q - k == window - 1 kept, q - k == window dropped) are
    # what the off-by-one moved.
    positions = torch.arange(16)[None]
    expected = _hf_mask(hf.sliding_window_causal_mask_function(window), 1, 16)
    self.assertTrue(torch.equal(_ours(_module(), positions, True, window), expected))

  @parameterized.parameters(1, 2, 3)
  def test_bidirectional_sliding_window(self, window):
    positions = torch.arange(16)[None]
    expected = _hf_mask(hf.and_masks(hf.bidirectional_mask_function,
                                     hf.sliding_window_bidirectional_overlay(window)), 1, 16)
    self.assertTrue(torch.equal(_ours(_module(), positions, False, window), expected))

  def test_windowed_documents(self):
    packed = hf.find_packed_sequence_indices(_POSITIONS)
    expected = _hf_mask(hf.and_masks(hf.sliding_window_causal_mask_function(2),
                                     hf.packed_sequence_mask_function(packed)), 1, 16)
    self.assertTrue(torch.equal(_ours(_module(), _POSITIONS, True, 2), expected))

  def test_padding(self):
    # Real tokens see real tokens of their own document only. Padded rows
    # attend only to padding (no all-masked row), which the loss ignores, so
    # they are left out of the comparison.
    padding = torch.ones_like(_POSITIONS)
    padding[:, -4:] = 0
    packed = hf.find_packed_sequence_indices(_POSITIONS)
    expected = _hf_mask(hf.and_masks(hf.causal_mask_function,
                                     hf.packed_sequence_mask_function(packed),
                                     hf.padding_mask_function(padding.bool())), 1, 16)
    got = _ours(_module(), _POSITIONS, True, padding=padding)
    self.assertTrue(torch.equal(got[..., :-4, :], expected[..., :-4, :]))


class KernelArgumentsTest(parameterized.TestCase):

  def test_window_from_kwarg_wins(self):
    # Mistral / Mixtral hand the window only as a keyword argument.
    self.assertEqual(attention._window(_module(), {"sliding_window": 4096}), 4096)
    self.assertEqual(attention._window(_module(sliding_window=7), {}), 7)
    self.assertIsNone(attention._window(_module(sliding_window="x"), {}))
    self.assertIsNone(attention._window(_module(), {"sliding_window": None}))

  def test_bidirectional_window_falls_back(self):
    # A stand-in query on "tpu" so the device check passes and the window
    # check is what decides.
    reason = attention._unsupported(
        _module(), SimpleNamespace(shape=(1, 1, 256, 8), device=SimpleNamespace(type="tpu")),
        SimpleNamespace(shape=(1, 1, 256, 8)), None, 0.0, False, 128)
    self.assertIn("bidirectional sliding window", reason)
    self.assertIsNone(attention._unsupported(
        _module(), SimpleNamespace(shape=(1, 1, 256, 8), device=SimpleNamespace(type="tpu")),
        SimpleNamespace(shape=(1, 1, 256, 8)), None, 0.0, True, 128))

  @parameterized.parameters((2048, 2048, 1024, 1024), (1536, 1536, 1024, 512),
                            (2048, 1024, 2048, 1024), (192, 192, 1024, None))
  def test_pick_block(self, q_len, kv_len, preferred, expected):
    self.assertEqual(attention._pick_block(q_len, kv_len, preferred), expected)

  def test_kernel_kwargs_come_from_module_config(self):
    module = SimpleNamespace(config=SimpleNamespace(
        splash_kernel_kwargs={"block_q": 1024, "use_fused_bwd_kernel": True}))
    self.assertEqual(attention._kernel_kwargs(module, 1536, 1536),
                     {"block_q": 512, "use_fused_bwd_kernel": True})
    self.assertEqual(attention._kernel_kwargs(SimpleNamespace(), 2048, 2048), {})


class PostTrainingWorkshopConfigsTest(absltest.TestCase):
  """Tests for Gemma 4 post-training workshop configurations."""

  def test_gemma4_cpt_and_sft_configs_build(self):
    """Verifies CPT and SFT workshop configs construct with expected settings."""
    from torchtitan.experiments.tpu.transformers_modeling_backend import config_registry  # pylint: disable=g-import-not-at-top

    cpt_cfg = config_registry.gemma4_e2b_cpt_codeparrot()
    self.assertEqual(cpt_cfg.hf_model, "google/gemma-4-E2B")
    self.assertTrue(cpt_cfg.checkpoint.enable)
    self.assertTrue(cpt_cfg.checkpoint.initial_load_in_hf)
    self.assertTrue(cpt_cfg.checkpoint.last_save_in_hf)
    self.assertTrue(cpt_cfg.checkpoint.last_save_model_only)

    cpt_plain = config_registry.gemma4_e2b_cpt_codeparrot_plain_ce()
    self.assertFalse(cpt_plain.model_spec.model.fused_loss)

    sft_cfg = config_registry.gemma4_e2b_sft_alpaca()
    self.assertEqual(sft_cfg.hf_model, "google/gemma-4-E2B-it")
    self.assertEqual(sft_cfg.model_spec.flavor, "sft_full")
    self.assertTrue(sft_cfg.checkpoint.last_save_in_hf)

    sft_plain = config_registry.gemma4_e2b_sft_alpaca_plain_ce()
    self.assertFalse(sft_plain.model_spec.model.fused_loss)

  def test_format_alpaca_sample(self):
    """Verifies Alpaca row formatting into user/assistant chat messages."""
    from torchtitan.experiments.tpu.transformers_modeling_backend import config_registry  # pylint: disable=g-import-not-at-top

    msgs = config_registry._format_alpaca_sample({  # pylint: disable=protected-access
        "instruction": "Sort a list.",
        "input": "[3, 1, 2]",
        "output": "sorted([3, 1, 2])",
    })
    self.assertLen(msgs, 2)
    self.assertEqual(msgs[0]["role"], "user")
    self.assertIn("### Input:\n[3, 1, 2]", msgs[0]["content"])
    self.assertEqual(
        msgs[1], {"role": "assistant", "content": "sorted([3, 1, 2])"}
    )


if __name__ == "__main__":
  absltest.main()
