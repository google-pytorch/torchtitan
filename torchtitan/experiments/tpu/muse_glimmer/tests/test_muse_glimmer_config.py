"""CPU-only tests for the TPU Muse Glimmer attention and config wiring.

Checks the sliding-window mask semantics, the splash-attention window
convention, the registered layer patterns (SWA/RoPE vs. global/NoPE) and the
TPU model registry. CPU vs. device parity lives in test_muse_glimmer.py.
"""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.nn.attention import flex_attention
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.config.manager import ConfigManager
from torchtitan.config.override import apply_overrides
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.graph_trainer.tpu_memory_policy import resolve_save_regions
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
import torchtitan.experiments.tpu.muse_glimmer as muse_glimmer_tpu
from torchtitan.experiments.tpu.muse_glimmer import attention as tpu_attention
from torchtitan.experiments.tpu.muse_glimmer import config_registry
from torchtitan.experiments.tpu.muse_glimmer.infra.parallelize import parallelize_muse_glimmer
from torchtitan.experiments.tpu.overrides import fused_loss
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
from torchtitan.models.common import attention as common_attention
from torchtitan.models.common.attention import segment_ids_from_positions
from torchtitan.models.muse_glimmer import model as upstream_model
from torchtitan.models.muse_glimmer.model import MuseGlimmerModel


_FUSED_LOSS_OVERRIDE_PATH = (
    "torchtitan.experiments.tpu.overrides.fused_loss."
    "fused_loss_soft_capped_lm_head"
)


def _get_model_config(model_name: str = "testmodel") -> MuseGlimmerModel.Config:
  """Retrieves a registered Muse Glimmer configuration from the TPU registry.

  Defaults for 'testmodel':
    - vocab_size = 2048, max_seq_len = 128, dim = 128
    - n_layers = 4, n_heads = 16, n_kv_heads = 8, head_dim = 16
    - window pattern = [4, 4, 4, global]; layer 3 is global + NoPE

  Defaults for '30B':
    - vocab_size = 202048, dim = 6656, n_layers = 52
    - n_heads = 32, n_kv_heads = 2, head_dim = 128
    - window pattern = [2048, 2048, 2048, global] (39 SWA, 13 global)
  """
  return muse_glimmer_tpu.muse_glimmer_configs[model_name]


class MuseGlimmerConfigTest(parameterized.TestCase):
  """Device-independent tests of the TPU attention and config wiring."""

  def test_sliding_window_mask_semantics(self):
    """Tests that query i attends to keys in [i - window + 1, i]."""
    mask = tpu_attention.sliding_window_causal_mask(
        seq_len=5, window_size=2, device=torch.device("cpu")
    )
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(mask, expected)

  def test_window_size_one_attends_only_to_self(self):
    """Tests that a window of 1 reduces to the identity mask."""
    mask = tpu_attention.sliding_window_causal_mask(
        seq_len=4, window_size=1, device=torch.device("cpu")
    )
    torch.testing.assert_close(mask, torch.eye(4, dtype=torch.bool))

  def test_rejects_invalid_window(self):
    """Tests that a non-positive window size is rejected."""
    with self.assertRaisesRegex(ValueError, "window_size"):
      tpu_attention.SlidingWindowScaledDotProductInnerAttention.Config(
          window_size=0
      ).build()

  def test_sliding_window_changes_output(self):
    """Guards against the window being silently ignored (upstream SDPA bug)."""
    batch, seq_len, n_heads, head_dim = 1, 16, 2, 8
    generator = torch.Generator().manual_seed(0)
    q, k, v = (
        torch.randn(batch, seq_len, n_heads, head_dim, generator=generator)
        for _ in range(3)
    )
    windowed = tpu_attention.SlidingWindowScaledDotProductInnerAttention.Config(
        window_size=3
    ).build()(q, k, v)
    full = tpu_attention.SlidingWindowScaledDotProductInnerAttention.Config(
        window_size=None
    ).build()(q, k, v)
    # The first `window` positions see the same keys either way.
    torch.testing.assert_close(windowed[:, :3], full[:, :3])
    self.assertFalse(torch.allclose(windowed[:, 3:], full[:, 3:]))

  def test_splash_patch_matches_inner_attention_class_name(self):
    """Tests that the splash-attention patch will pick up the inner attention."""
    # workarounds.use_splash_attention_patch matches on this substring.
    self.assertIn(
        "ScaledDotProduct",
        tpu_attention.SlidingWindowScaledDotProductInnerAttention.__name__,
    )

  def test_testmodel_layer_pattern(self):
    """Tests the testmodel SWA/global and RoPE/NoPE layer pattern."""
    config = _get_model_config()
    windows = [layer.attention.window_size for layer in config.layers]
    has_rope = [layer.attention.rope is not None for layer in config.layers]
    self.assertEqual(
        windows, [muse_glimmer_tpu.TESTMODEL_WINDOW_SIZE] * 3 + [None]
    )
    self.assertEqual(has_rope, [True, True, True, False])
    for layer in config.layers:
      inner = layer.attention.inner_attention
      self.assertIsInstance(
          inner,
          tpu_attention.SlidingWindowScaledDotProductInnerAttention.Config,
      )
      self.assertEqual(inner.window_size, layer.attention.window_size)

  def test_splash_window_is_one_less_than_muse_window(self):
    """Tests the Muse window W -> splash window W - 1 conversion."""
    config = _get_model_config()
    splash_windows = [
        layer.attention.inner_attention.build().sliding_window_size
        for layer in config.layers
    ]
    window = muse_glimmer_tpu.TESTMODEL_WINDOW_SIZE
    self.assertEqual(splash_windows, [window - 1] * 3 + [None])

  def test_30b_layer_pattern(self):
    """Tests the 30B iRoPE layer pattern (39 SWA + 13 global/NoPE layers)."""
    config = _get_model_config("30B")
    self.assertLen(config.layers, 52)
    windows = [layer.attention.window_size for layer in config.layers]
    self.assertEqual(windows.count(None), 13)
    self.assertEqual(windows.count(2048), 39)
    for layer in config.layers:
      attention = layer.attention
      self.assertEqual(
          attention.inner_attention.window_size, attention.window_size
      )
      # iRoPE: the global layers are exactly the NoPE layers.
      self.assertEqual(attention.window_size is None, attention.rope is None)

  def test_model_registry_uses_tpu_parallelize(self):
    """Tests that every flavor uses the TPU parallelize fn and attention."""
    for flavor in ("testmodel", "debugmodel", "30B"):
      with self.subTest(flavor=flavor):
        spec = muse_glimmer_tpu.model_registry(flavor)
        self.assertEqual(spec.name, "muse_glimmer")
        self.assertIs(spec.parallelize_fn, parallelize_muse_glimmer)
        self.assertIsNone(spec.pipelining_fn)
        for layer in spec.model.layers:
          self.assertIsInstance(
              layer.attention.inner_attention,
              tpu_attention.SlidingWindowScaledDotProductInnerAttention.Config,
          )

  def test_model_registry_rejects_unknown_flavor(self):
    """Tests that an unknown flavor is rejected."""
    with self.assertRaisesRegex(ValueError, "Unsupported Muse Glimmer flavor"):
      muse_glimmer_tpu.model_registry("not_a_flavor")

  def test_model_registry_rejects_long_testmodel_seq_len(self):
    """Tests that seq_len above the testmodel context length is rejected."""
    with self.assertRaisesRegex(ValueError, "exceeds max context length"):
      muse_glimmer_tpu.model_registry(
          "testmodel",
          seq_len=muse_glimmer_tpu.TESTMODEL_MAX_CONTEXT_LENGTH + 1,
      )

  def test_model_registry_uses_tpu_outer_attention(self):
    """Tests that every layer uses the positions-forwarding TPU attention."""
    for flavor in ("testmodel", "debugmodel", "30B"):
      with self.subTest(flavor=flavor):
        for layer in muse_glimmer_tpu.model_registry(flavor).model.layers:
          self.assertIsInstance(
              layer.attention, tpu_attention.TpuMuseGlimmerAttention.Config
          )

  def test_packed_mask_matches_upstream_flex_document_mask(self):
    """Tests the SDPA packed mask against upstream's flex mask_mods."""
    # Three packed documents of lengths 5, 3 and 4 in one row.
    positions = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3])
    seq_len = positions.shape[0]
    segment_ids_BL = segment_ids_from_positions(positions).reshape(1, seq_len)
    for window_size in (3, None):
      with self.subTest(window_size=window_size):
        mask_mods = [
            common_attention.get_causal_mask_mod(),
            common_attention.get_efficient_causal_mask_mod_for_packed_document(
                positions
            ),
        ]
        if window_size is not None:
          mask_mods.append(
              common_attention.get_sliding_window_mask_mod(window_size)
          )
        expected = flex_attention.create_mask(
            flex_attention.and_masks(*mask_mods),
            1,
            1,
            seq_len,
            seq_len,
            device="cpu",
        )
        actual = tpu_attention.packed_sliding_window_causal_mask(
            seq_len,
            window_size,
            torch.device("cpu"),
            segment_ids_BL=segment_ids_BL,
        )
        self.assertTrue(torch.equal(actual, expected))

  def _build_attention(self, layer_id: int) -> torch.nn.Module:
    attention = _get_model_config().layers[layer_id].attention.build()
    attention.init_states(buffer_device=torch.device("cpu"))
    return attention

  def test_attention_isolates_packed_documents(self):
    """Tests that a packed document's output equals running it on its own."""
    torch.manual_seed(0)
    dim = _get_model_config().dim
    len_a, len_b = 7, 5
    x = torch.randn(len_a + len_b, dim)
    positions = torch.cat([torch.arange(len_a), torch.arange(len_b)])
    # Layer 0: sliding window + RoPE. Layer 3: global + NoPE.
    for layer_id in (0, 3):
      with self.subTest(layer_id=layer_id):
        attention = self._build_attention(layer_id)
        with torch.no_grad():
          with set_unflatten_seq_len(len_a + len_b):
            packed = attention(x, None, positions)
          with set_unflatten_seq_len(len_a):
            alone_a = attention(x[:len_a], None, torch.arange(len_a))
          with set_unflatten_seq_len(len_b):
            alone_b = attention(x[len_a:], None, torch.arange(len_b))
        torch.testing.assert_close(packed[:len_a], alone_a)
        torch.testing.assert_close(packed[len_a:], alone_b)

  def test_tpu_attention_matches_upstream_on_single_document(self):
    """Guards TpuMuseGlimmerAttention.forward against drifting from upstream."""
    torch.manual_seed(0)
    dim = _get_model_config().dim
    batch, seq_len = 2, 8
    x = torch.randn(batch * seq_len, dim)
    positions = torch.arange(seq_len).repeat(batch)  # One document per row.
    for layer_id in (0, 3):
      with self.subTest(layer_id=layer_id):
        tpu_module = self._build_attention(layer_id)
        tpu_config = _get_model_config().layers[layer_id].attention
        upstream_module = upstream_model.Attention.Config(**{
            field.name: getattr(tpu_config, field.name)
            for field in dataclasses.fields(upstream_model.Attention.Config)
        }).build()
        upstream_module.init_states(buffer_device=torch.device("cpu"))
        upstream_module.load_state_dict(tpu_module.state_dict())
        with torch.no_grad(), set_unflatten_seq_len(seq_len):
          torch.testing.assert_close(
              tpu_module(x, None, positions),
              upstream_module(x, None, positions),
          )

  def test_loss_kernel_rejects_soft_capped_lm_head(self):
    """Tests that the fused loss patch refuses SoftCappedLinear."""
    model = torch.nn.Module()
    model.lm_head = _get_model_config().lm_head.build()
    self.assertIsInstance(model.lm_head, upstream_model.SoftCappedLinear)
    with self.assertRaisesRegex(ValueError, "use_loss_kernel"):
      workarounds.use_output_projection_patch(model)

  def test_loss_kernel_still_wraps_plain_linear_lm_head(self):
    """Tests that a plain nn.Linear lm_head is still wrapped."""
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(8, 16, bias=False)
    workarounds.use_output_projection_patch(model)
    self.assertIsInstance(model.lm_head, workarounds.PallasLossLinear)

  def test_fused_ce_recipe_carries_soft_cap_into_loss(self):
    """Tests the *_fused_ce recipe: wrapper head + capped fused loss."""
    base = config_registry.muse_glimmer_debugmodel()
    base_head = base.model_spec.model.lm_head
    self.assertIsInstance(base_head, upstream_model.SoftCappedLinear.Config)
    self.assertIsNotNone(base_head.output_soft_cap_temp)

    config = config_registry.muse_glimmer_debugmodel_fused_ce()
    self.assertIsInstance(config.loss, PallasCrossEntropyLoss.Config)
    self.assertEqual(config.loss.soft_cap, base_head.output_soft_cap_temp)
    self.assertEqual(
        config.loss.output_multiplier, base_head.output_multiplier
    )
    self.assertIn(
        "torchtitan.experiments.tpu.overrides.fused_loss."
        "fused_loss_soft_capped_lm_head",
        config.override.imports,
    )

    apply_overrides(config.override, config)
    head = config.model_spec.model.lm_head
    self.assertIsInstance(head, fused_loss.FusedLossLinearWrapper.Config)
    self.assertEqual(head.soft_cap, base_head.output_soft_cap_temp)
    self.assertEqual(head.output_multiplier, base_head.output_multiplier)

    with torch.device("meta"):
      model = config.model_spec.model.build()
    self.assertIsInstance(model.lm_head, fused_loss.FusedLossLinearWrapper)
    # The checkpoint key is unchanged, so plain-CE checkpoints load.
    self.assertEqual(
        [k for k in model.state_dict() if k.endswith("lm_head.weight")],
        ["lm_head.weight"],
    )

  def _build_fused_ce_model(self, config):
    apply_overrides(config.override, config)
    with torch.device("meta"):
      return config.model_spec.model.build()

  def test_fused_ce_recipe_passes_logit_transform_check(self):
    """Tests that the *_fused_ce recipe's head and loss agree."""
    config = config_registry.muse_glimmer_debugmodel_fused_ce()
    model = self._build_fused_ce_model(config)
    workarounds._check_fused_loss_logit_transform(model, config)

  def test_logit_transform_check_rejects_mismatched_multiplier(self):
    """Tests the flavor-swap trap: the loss keeps a stale multiplier."""
    config = config_registry.muse_glimmer_debugmodel_fused_ce()
    model = self._build_fused_ce_model(config)
    config.loss.output_multiplier = 0.196
    with self.assertRaisesRegex(ValueError, "logit transform mismatch"):
      workarounds._check_fused_loss_logit_transform(model, config)

  def test_logit_transform_check_rejects_untransformed_loss(self):
    """Tests that a default (uncapped) fused loss is rejected for Muse."""
    config = config_registry.muse_glimmer_debugmodel_fused_ce()
    model = self._build_fused_ce_model(config)
    config.loss = PallasCrossEntropyLoss.Config()
    with self.assertRaisesRegex(ValueError, "logit transform mismatch"):
      workarounds._check_fused_loss_logit_transform(model, config)

  def test_logit_transform_check_rejects_non_fused_loss(self):
    """Tests that the wrapper head is rejected without a fused loss."""
    config = config_registry.muse_glimmer_debugmodel_fused_ce()
    model = self._build_fused_ce_model(config)
    config.loss = CrossEntropyLoss.Config()
    with self.assertRaisesRegex(ValueError, "PallasCrossEntropyLoss"):
      workarounds._check_fused_loss_logit_transform(model, config)

  @parameterized.parameters(
      "muse_glimmer_30b_full_logits",
      "muse_glimmer_30b_fused_ce",
      "reg_trainer_muse_glimmer_30b_fused_ce",
  )
  def test_30b_recipes_share_upstream_model_and_flags(self, name):
    """Tests the 30B recipes: upstream model/data settings, 4x4x4 flags."""
    config = getattr(config_registry, name)()
    self.assertEqual(config.model_spec.flavor, "30B")
    self.assertEqual(config.training.max_context_length, 16384)
    self.assertEqual(
        config.training.num_tokens_per_microbatch_per_dp_rank, 16384
    )
    flags = config.libtpu_init_flags
    self.assertEqual(flags["xla_tpu_dvfs_p_state"], "7")
    self.assertEqual(
        flags["xla_tpu_enable_sparse_core_hierarchical_reduce_scatter"],
        "true",
    )
    self.assertTrue(config.splash_attention_kernel.use_splash_attention_kernel)
    self.assertFalse(
        config.splash_attention_kernel.use_tokamax_splash_attention_kernel
    )

  @parameterized.parameters(
      ("muse_glimmer_30b_full_logits", True, False, "fused"),
      ("muse_glimmer_30b_fused_ce", True, True, "fused"),
      ("reg_trainer_muse_glimmer_30b_fused_ce", False, True, "foreach"),
  )
  def test_30b_recipe_trainer_loss_and_optimizer_selection(
      self, name, graph_trainer, fused, optimizer_implementation
  ):
    """Tests each 30B recipe's trainer, loss and AdamW implementation."""
    config = getattr(config_registry, name)()
    self.assertEqual(config.tpu_config.use_graph_trainer, graph_trainer)
    self.assertEqual(
        isinstance(config.loss, PallasCrossEntropyLoss.Config), fused
    )
    self.assertEqual(
        _FUSED_LOSS_OVERRIDE_PATH in config.override.imports, fused
    )
    self.assertEqual(
        config.optimizer.implementation, optimizer_implementation
    )

  @parameterized.parameters(
      (
          "muse_glimmer_30b_full_logits",
          "tpu_selective",
          ("attention.inner_attention",),
      ),
      ("muse_glimmer_30b_fused_ce", "full", ()),
  )
  def test_30b_graph_trainer_recipes_use_zero3_and_remat_policy(
      self, name, memory_policy, save_regions
  ):
    """Tests Zero-3, the remat policy and the FSDP schedule."""
    config = getattr(config_registry, name)()
    self.assertEqual(config.parallelism.fsdp_reshard_after_forward, "always")
    self.assertEqual(
        config.tpu_config.graph_trainer_memory_policy, memory_policy
    )
    self.assertEqual(config.tpu_config.graph_trainer_save_regions, save_regions)
    self.assertEqual(config.tpu_config.graph_trainer_save_last_n_layers, {})
    self.assertNotEqual(config.tpu_config.graph_trainer_fsdp_schedule, "none")

  @parameterized.parameters(
      # Opt out to plain full rematerialization.
      (
          (
              "--tpu_config.graph_trainer_memory_policy=full",
              "--tpu_config.graph_trainer_save_regions",
          ),
          "full",
          {},
      ),
      # Splash saved in the last 26 layers only: no extra peak HBM.
      (
          (
              "--tpu_config.graph_trainer_save_regions",
              "--tpu_config.graph_trainer_save_last_n_layers",
              "attention.inner_attention",
              "26",
          ),
          "tpu_selective",
          {"attention.inner_attention": 26},
      ),
  )
  def test_30b_full_logits_readme_memory_overrides(
      self, flags, memory_policy, last_n_layers
  ):
    """Tests the memory-policy overrides documented in the recipe README."""
    config = ConfigManager().parse_args([
        "--module=torchtitan.experiments.tpu.muse_glimmer",
        "--config=muse_glimmer_30b_full_logits",
        *flags,
    ])
    tpu_config = config.tpu_config
    self.assertEqual(tpu_config.graph_trainer_memory_policy, memory_policy)
    self.assertEqual(tpu_config.graph_trainer_save_regions, ())
    self.assertEqual(tpu_config.graph_trainer_save_last_n_layers, last_n_layers)
    if memory_policy == "tpu_selective":
      # The policy accepts the combination (each region listed once).
      resolve_save_regions(
          tpu_config.graph_trainer_save_regions, last_n_layers
      )

  def test_30b_fused_ce_recipe_passes_logit_transform_check(self):
    """Tests that the 30B head and loss agree on cap and 0.196 multiplier."""
    config = config_registry.muse_glimmer_30b_fused_ce()
    self.assertAlmostEqual(config.loss.output_multiplier, 0.19611613513)
    self.assertIsNotNone(config.loss.soft_cap)
    model = self._build_fused_ce_model(config)
    self.assertIsInstance(model.lm_head, fused_loss.FusedLossLinearWrapper)
    workarounds._check_fused_loss_logit_transform(model, config)


if __name__ == "__main__":
  absltest.main()
