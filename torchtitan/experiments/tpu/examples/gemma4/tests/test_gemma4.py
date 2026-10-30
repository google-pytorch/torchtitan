"""Unit and numerical tests for Gemma 4 in torchtitan.experiments.tpu.examples.gemma4."""

import copy
from functools import partial
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.components.loss import (
    ChunkedLossWrapper,
    CrossEntropyLoss,
    cross_entropy_loss,
)
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest as BaseDeviceTest
from torchtitan.experiments.tpu.examples.gemma4 import (
    _build_flavor_config,
    gemma4_configs,
)
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch
from torchtitan.experiments.tpu.examples.gemma4.config_registry import (
    gemma4_26b_a4b,
    gemma4_31b,
    gemma4_debugmodel,
    gemma4_moe_debug,
    gemma4_moe_testmodel,
)
from torchtitan.experiments.tpu.examples.gemma4.model import (
    Gemma4Attention,
    Gemma4Model,
    Gemma4RoPE,
    SlidingWindowScaledDotProductAttention,
)
from torchtitan.experiments.tpu.examples.gemma4.state_dict_adapter import (
    Gemma4StateDictAdapter,
)
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
from torchtitan.models.common.attention import (
    QKVLinear,
    ScaledDotProductInnerAttention,
)
from torchtitan.models.common.param_init import skip_param_init
from torchtitan.models.common.linear import Linear, RouterGateLinear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.muse_glimmer.model import SoftCappedLinear
import transformers


_NORM_INIT = {"weight": nn.init.ones_}
_LINEAR_INIT = {"weight": partial(nn.init.trunc_normal_, std=0.02)}
_EXPERTS_INIT = {
    "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
    "w2_EDF": partial(nn.init.trunc_normal_, std=0.02),
    "w3_EFD": partial(nn.init.trunc_normal_, std=0.02),
}


class TestTpuGroupedExpertsActivation(BaseDeviceTest):
  """Tests gelu_tanh activation support in TpuGroupedExperts."""

  def test_gelu_tanh_vs_silu_forward(self):
    num_experts, dim, hidden_dim = 4, 16, 32
    x = torch.randn(8, dim, device=self.accelerator_device, dtype=torch.float32)
    num_tokens_per_expert = torch.tensor(
        [2, 2, 2, 2], device=self.accelerator_device, dtype=torch.int32
    )

    cfg_gelu = TpuGroupedExperts.Config(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        use_pallas_grouped_mm=False,
        activation="gelu_tanh",
        param_init=_EXPERTS_INIT,
    )
    experts_gelu = cfg_gelu.build().to(self.accelerator_device)
    experts_gelu.init_states()

    out_gelu = experts_gelu(x, num_tokens_per_expert)

    # Compute reference gelu_tanh in bfloat16 (matching GroupedExperts compute)
    expected_chunks = []
    for i, chunk in enumerate(x.split([2, 2, 2, 2], dim=0)):
      c_bf16 = chunk.bfloat16()
      w1 = experts_gelu.w1_EFD[i].bfloat16().t()
      w2 = experts_gelu.w2_EDF[i].bfloat16().t()
      w3 = experts_gelu.w3_EFD[i].bfloat16().t()
      h = F.gelu(c_bf16 @ w1, approximate="tanh") * (c_bf16 @ w3)
      expected_chunks.append((h @ w2).type_as(x))
    expected = torch.cat(expected_chunks, dim=0)
    torch.testing.assert_close(out_gelu, expected, atol=2e-2, rtol=2e-2)

  def test_converter_propagates_gemma4_gelu_tanh(self):
    config = copy.deepcopy(gemma4_configs["testmodel_moe"])
    converter = TpuGroupedExpertsConverter(TpuGroupedExpertsConverter.Config())
    config = converter.convert(config)
    for layer in config.layers:
      if layer.moe is not None:
        inner_experts = layer.moe.routed_experts.inner_experts
        self.assertIsInstance(inner_experts, TpuGroupedExperts.Config)
        self.assertEqual(inner_experts.activation, "gelu_tanh")
        self.assertIsInstance(layer.moe.router.gate, RouterGateLinear.Config)

  def test_config_registry_valid(self):
    for fn in (
        gemma4_debugmodel,
        gemma4_moe_testmodel,
        gemma4_moe_debug,
        gemma4_31b,
        gemma4_26b_a4b,
    ):
      cfg = fn()
      self.assertEqual(cfg.model_spec.name, "gemma4")
      self.assertIn(cfg.model_spec.flavor, gemma4_configs)
      self.assertFalse(cfg.model_spec.model.enable_weight_tying)

  def test_build_flavor_config_weight_tying(self):
    tied = _build_flavor_config("testmodel")
    untied = _build_flavor_config("testmodel", enable_weight_tying=False)
    self.assertTrue(tied.enable_weight_tying)
    self.assertFalse(untied.enable_weight_tying)
    with torch.device("meta"):
      tied_model = tied.build()
      untied_model = untied.build()
    self.assertIs(tied_model.tok_embeddings.weight, tied_model.lm_head.weight)
    self.assertIsNot(
        untied_model.tok_embeddings.weight, untied_model.lm_head.weight
    )
    # An untied embedding must get a real init (std=0.02 matching HF
    # initializer_range so sqrt(dim) scaling stays O(1)), not skip_param_init.
    self.assertIsNot(
        untied.tok_embeddings.param_init["weight"], skip_param_init
    )
    self.assertEqual(
        untied.tok_embeddings.param_init["weight"].keywords["std"], 0.02
    )

    # Verify factory-backed flavors (debugmodel, 31B, etc.) return isolated
    # sub-config instances so building one config does not mutate another.
    dbg_tied = _build_flavor_config("debugmodel", enable_weight_tying=True)
    init_tied = dbg_tied.tok_embeddings.param_init
    dbg_untied = _build_flavor_config(
        "debugmodel", enable_weight_tying=False, enable_sp=False
    )
    self.assertIsNot(dbg_tied.tok_embeddings, dbg_untied.tok_embeddings)
    self.assertIsNot(dbg_tied.norm, dbg_untied.norm)
    self.assertIsNot(dbg_tied.lm_head, dbg_untied.lm_head)
    self.assertIs(dbg_tied.tok_embeddings.param_init, init_tied)
    self.assertIs(
        dbg_tied.tok_embeddings.param_init["weight"], skip_param_init
    )

  def test_silu_delegates_to_super_and_gelu_uses_grouped_mm_seam(self):
    from unittest import mock

    num_experts, dim, hidden_dim = 4, 16, 32
    x = torch.randn(8, dim, device=self.accelerator_device, dtype=torch.float32)
    num_tokens_per_expert = torch.tensor(
        [2, 2, 2, 2], device=self.accelerator_device, dtype=torch.int32
    )

    cfg_silu = TpuGroupedExperts.Config(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        use_pallas_grouped_mm=False,
        activation="silu",
        param_init=_EXPERTS_INIT,
    )
    experts_silu = cfg_silu.build().to(self.accelerator_device)
    experts_silu.init_states()
    with mock.patch.object(
        experts_silu,
        "_native_experts_forward",
        wraps=experts_silu._native_experts_forward,
    ) as spy_native:
      _ = experts_silu(x, num_tokens_per_expert)
      spy_native.assert_not_called()

    cfg_gelu = TpuGroupedExperts.Config(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        use_pallas_grouped_mm=False,
        activation="gelu_tanh",
        param_init=_EXPERTS_INIT,
    )
    experts_gelu = cfg_gelu.build().to(self.accelerator_device)
    experts_gelu.init_states()
    with mock.patch.object(
        experts_gelu,
        "_grouped_mm",
        wraps=experts_gelu._grouped_mm,
    ) as spy_grouped_mm:
      _ = experts_gelu(x, num_tokens_per_expert)
      self.assertEqual(spy_grouped_mm.call_count, 3)

  def test_context_parallel_rejected(self):
    from types import SimpleNamespace
    from torchtitan.experiments.tpu.examples.gemma4.infra.parallelize import (
        parallelize_gemma4,
    )

    cfg = gemma4_debugmodel()
    cfg.parallelism.context_parallel_degree = 2
    with self.assertRaisesRegex(
        NotImplementedError, "Context parallelism is not supported"
    ):
      cfg.model_spec.model.update_from_config(config=cfg)

    with torch.device("meta"):
      model = gemma4_debugmodel().model_spec.model.build()
    pdims = SimpleNamespace(cp_enabled=True, cp=2, dp_shard=1)
    with self.assertRaisesRegex(
        NotImplementedError, "Context parallelism is not supported"
    ):
      parallelize_gemma4(
          model,
          parallel_dims=pdims,
          training=cfg.training,
          parallelism=cfg.parallelism,
          compile_config=cfg.compile,
          ac_config=None,
          dump_folder="",
      )

  def test_moe_spmd_safe_applied_when_ep_disabled(self):
    from types import SimpleNamespace
    from unittest import mock
    from torchtitan.experiments.tpu.examples.gemma4.infra import (
        parallelize as gemma4_parallelize,
    )

    cfg = gemma4_moe_testmodel()
    cfg.compile.enable = True
    cfg.parallelism.use_simple_fsdp = False

    for ep_enabled, expected_calls in ((False, 2), (True, 0)):
      with torch.device("meta"):
        model = cfg.model_spec.model.build()
      pdims = SimpleNamespace(
          cp_enabled=False,
          ep_enabled=ep_enabled,
          tp_enabled=False,
          dp_shard=1,
          ep=2 if ep_enabled else 1,
      )
      with (
          mock.patch.object(
              gemma4_parallelize, "apply_spmd_safe"
          ) as mock_spmd_safe,
          mock.patch.object(
              gemma4_parallelize,
              "_parallelize_gemma4",
              side_effect=lambda m, **kw: m,
          ),
      ):
        gemma4_parallelize.parallelize_gemma4(
            model,
            parallel_dims=pdims,
            training=cfg.training,
            parallelism=cfg.parallelism,
            compile_config=cfg.compile,
            ac_config=None,
            dump_folder="",
            skip_dp=True,
        )
      self.assertEqual(mock_spmd_safe.call_count, expected_calls)


class TestGemma4StateDictAdapter(absltest.TestCase):
  """Tests HF <-> TorchTitan state_dict conversion for Gemma 4."""

  def test_dense_state_dict_roundtrip(self):
    config = _build_flavor_config("testmodel")
    with torch.device("meta"):
      model = config.build()
    tt_sd = {}
    for k, v in model.state_dict().items():
      tt_sd[k] = torch.randn(v.shape, dtype=torch.float32)
    if config.enable_weight_tying:
      tt_sd["lm_head.weight"] = tt_sd["tok_embeddings.weight"]

    adapter = Gemma4StateDictAdapter(config, hf_assets_path=None)
    hf_sd = adapter.to_hf(tt_sd)

    self.assertIn("model.language_model.embed_tokens.weight", hf_sd)
    self.assertIn("model.language_model.norm.weight", hf_sd)
    self.assertIn(
        "model.language_model.layers.0.self_attn.q_proj.weight", hf_sd
    )
    self.assertIn(
        "model.language_model.layers.0.self_attn.k_proj.weight", hf_sd
    )
    self.assertIn(
        "model.language_model.layers.0.self_attn.v_proj.weight", hf_sd
    )
    self.assertIn("model.language_model.layers.0.layer_scalar", hf_sd)
    # Global K=V layer (layer 1 in testmodel)
    self.assertIn(
        "model.language_model.layers.1.self_attn.k_proj.weight", hf_sd
    )
    self.assertNotIn(
        "model.language_model.layers.1.self_attn.v_proj.weight", hf_sd
    )

    roundtrip_sd = adapter.from_hf(hf_sd)
    self.assertEqual(set(roundtrip_sd.keys()), set(tt_sd.keys()))
    for k in tt_sd:
      torch.testing.assert_close(roundtrip_sd[k], tt_sd[k])

  def test_moe_state_dict_roundtrip_and_vision_skip(self):
    config = _build_flavor_config("testmodel_moe")
    with torch.device("meta"):
      model = config.build()
    tt_sd = {}
    for k, v in model.state_dict().items():
      tt_sd[k] = torch.randn(v.shape, dtype=torch.float32)
    if config.enable_weight_tying:
      tt_sd["lm_head.weight"] = tt_sd["tok_embeddings.weight"]

    adapter = Gemma4StateDictAdapter(config, hf_assets_path=None)
    hf_sd = adapter.to_hf(tt_sd)

    self.assertIn(
        "model.language_model.layers.0.experts.gate_up_proj", hf_sd
    )
    self.assertIn("model.language_model.layers.0.experts.down_proj", hf_sd)
    self.assertIn("model.language_model.layers.0.router.proj.weight", hf_sd)
    self.assertIn("model.language_model.layers.0.router.scale", hf_sd)
    self.assertIn(
        "model.language_model.layers.0.router.per_expert_scale", hf_sd
    )
    self.assertIn(
        "model.language_model.layers.0.post_feedforward_layernorm_1.weight",
        hf_sd,
    )
    self.assertIn(
        "model.language_model.layers.0.post_feedforward_layernorm_2.weight",
        hf_sd,
    )
    self.assertIn(
        "model.language_model.layers.0.pre_feedforward_layernorm_2.weight",
        hf_sd,
    )

    # Add dummy vision/audio keys and verify they are ignored on from_hf
    hf_sd["model.vision_tower.encoder.layers.0.weight"] = torch.randn(4, 4)
    hf_sd["model.audio_tower.encoder.layers.0.weight"] = torch.randn(4, 4)
    hf_sd["model.embed_vision.weight"] = torch.randn(4, 4)
    hf_sd["model.embed_audio.weight"] = torch.randn(4, 4)
    hf_sd["multi_modal_projector.weight"] = torch.randn(4, 4)

    roundtrip_sd = adapter.from_hf(hf_sd)
    self.assertEqual(set(roundtrip_sd.keys()), set(tt_sd.keys()))
    for k in tt_sd:
      torch.testing.assert_close(roundtrip_sd[k], tt_sd[k])

  def test_fused_qkv_rejected(self):
    config = _build_flavor_config("debugmodel", use_fused_qkv=True)
    with self.assertRaisesRegex(
        NotImplementedError, "does not support FusedQKVLinear"
    ):
      Gemma4StateDictAdapter(config, hf_assets_path=None)

  def test_tied_hf_checkpoint_loads_into_untied_recipe(self):
    """Tied Gemma 4 HF checkpoints (no lm_head.weight) load into untied recipes."""
    tied_config = _build_flavor_config("testmodel", enable_weight_tying=True)
    untied_config = _build_flavor_config("testmodel", enable_weight_tying=False)
    model = untied_config.build()
    model.init_states()

    # Build a tied HF checkpoint (which omits lm_head.weight).
    tied_adapter = Gemma4StateDictAdapter(tied_config, hf_assets_path=None)
    hf_sd = tied_adapter.to_hf(
        {k: torch.randn_like(v) for k, v in model.state_dict().items()}
    )
    self.assertNotIn("lm_head.weight", hf_sd)
    self.assertIn("model.language_model.embed_tokens.weight", hf_sd)

    untied_adapter = Gemma4StateDictAdapter(untied_config, hf_assets_path=None)
    converted_sd = untied_adapter.from_hf(hf_sd)
    self.assertIn("lm_head.weight", converted_sd)
    torch.testing.assert_close(
        converted_sd["lm_head.weight"],
        hf_sd["model.language_model.embed_tokens.weight"],
    )
    missing, unexpected = model.load_state_dict(converted_sd, strict=False)
    self.assertEmpty(missing)
    self.assertEmpty(unexpected)
    # The real on-disk flow (to_hf placeholder -> dcp.load -> from_hf) is
    # covered by TestGemma4HFReferenceParity.
    # test_tied_hf_safetensors_load_into_untied_recipe.


class TestGemma4SoftCappedLoss(absltest.TestCase):
  """Tests that Gemma 4 logit soft-capping survives the chunked-loss path."""

  def test_chunked_loss_matches_full_cross_entropy_with_soft_cap(self):
    torch.manual_seed(0)
    config = _build_flavor_config("testmodel", enable_weight_tying=False)
    self.assertIsInstance(config.lm_head, SoftCappedLinear.Config)
    self.assertEqual(config.lm_head.output_soft_cap_temp, 30.0)

    full_model = config.build()
    full_model.init_states()
    # Amplify lm_head weights so pre-cap logits exceed 30.0 and tanh is active.
    with torch.no_grad():
      full_model.lm_head.weight.mul_(50.0)
    chunked_model = copy.deepcopy(full_model)
    chunked_model._skip_lm_head = True

    batch_size, seq_len = 2, 16
    tokens = torch.randint(0, config.vocab_size, (batch_size * seq_len,))
    positions = torch.arange(seq_len).repeat(batch_size)
    labels = torch.randint(0, config.vocab_size, (batch_size * seq_len,))
    valid_tokens = torch.tensor(batch_size * seq_len, dtype=torch.int64)

    full_loss_fn = CrossEntropyLoss.Config().build()
    chunked_loss_fn = ChunkedLossWrapper.Config(num_chunks=4).build()
    chunked_loss_fn.set_lm_head(chunked_model.lm_head)

    with set_unflatten_seq_len(seq_len):
      logits = full_model(tokens, positions=positions)
      self.assertLessEqual(logits.detach().abs().max().item(), 30.0 + 1e-5)
      full_loss, _ = full_loss_fn(logits, labels, valid_tokens)
      full_loss.backward()

      hidden = chunked_model(tokens, positions=positions)
      chunked_loss, _ = chunked_loss_fn(hidden, labels, valid_tokens)
      chunked_loss.backward()

    torch.testing.assert_close(chunked_loss, full_loss, atol=1e-5, rtol=1e-5)
    for (name, p_full), (_, p_chunk) in zip(
        full_model.named_parameters(), chunked_model.named_parameters()
    ):
      torch.testing.assert_close(
          p_chunk.grad, p_full.grad, atol=1e-5, rtol=1e-5, msg=name
      )

  def test_legacy_loss_kernel_rejects_soft_capped_lm_head(self):
    config = _build_flavor_config("testmodel")
    model = nn.Module()
    model.lm_head = config.lm_head.build()
    self.assertIsInstance(model.lm_head, SoftCappedLinear)
    with self.assertRaisesRegex(ValueError, "use_loss_kernel"):
      workarounds.use_output_projection_patch(model)


class TestGemma4SdpaSlidingWindow(absltest.TestCase):
  """Sliding layers must apply the HF window on the plain SDPA backend."""

  _WINDOW = 4
  _SEQ_LEN = 12  # Longer than the window, so the window actually masks keys.
  _DIM = 32

  def _attention_with_mask(self, attn, mask_LL):
    """Copy of ``attn`` whose inner attention uses an explicit boolean mask."""
    ref = copy.deepcopy(attn)

    def masked_sdpa(
        q, k, v, *, scale=None, enable_gqa=False, **unused_kwargs
    ):
      q_4d = q.unflatten(0, (-1, self._SEQ_LEN)).transpose(1, 2)
      k_4d = k.unflatten(0, (-1, self._SEQ_LEN)).transpose(1, 2)
      v_4d = v.unflatten(0, (-1, self._SEQ_LEN)).transpose(1, 2)
      return (
          F.scaled_dot_product_attention(
              q_4d,
              k_4d,
              v_4d,
              attn_mask=mask_LL,
              scale=scale,
              enable_gqa=enable_gqa,
          )
          .transpose(1, 2)
          .flatten(0, 1)
      )

    ref.inner_attention.forward = masked_sdpa
    return ref

  def test_sliding_window_applied_beyond_window(self):
    head_dim = 8
    cfg = Gemma4Attention.Config(
        dim=self._DIM,
        n_heads=4,
        n_kv_heads=2,
        head_dim=head_dim,
        is_sliding=True,
        sliding_window_size=self._WINDOW,
        qk_norm=RMSNorm.Config(
            normalized_shape=head_dim, eps=1e-6, param_init=_NORM_INIT
        ),
        v_norm=RMSNorm.Config(
            normalized_shape=head_dim, eps=1e-6, elementwise_affine=False
        ),
        qkv_linear=QKVLinear.Config(
            head_dim=head_dim,
            wq=Linear.Config(
                in_features=self._DIM, out_features=32, param_init=_LINEAR_INIT
            ),
            wkv=Linear.Config(
                in_features=self._DIM, out_features=16, param_init=_LINEAR_INIT
            ),
        ),
        wo=Linear.Config(
            in_features=32, out_features=self._DIM, param_init=_LINEAR_INIT
        ),
        inner_attention=ScaledDotProductInnerAttention.Config(),
        rope=Gemma4RoPE.Config(
            dim=head_dim, max_context_length=self._SEQ_LEN, theta=10000.0
        ),
    )
    torch.manual_seed(0)
    attn = cfg.build()
    attn.init_states()

    # HF Gemma 4: query q attends to key kv iff 0 <= q - kv < sliding_window.
    dist = (
        torch.arange(self._SEQ_LEN)[:, None]
        - torch.arange(self._SEQ_LEN)[None, :]
    )
    hf_mask = (dist >= 0) & (dist < self._WINDOW)
    causal_mask = dist >= 0

    x = torch.randn(2 * self._SEQ_LEN, self._DIM)
    positions = torch.arange(self._SEQ_LEN).repeat(2)
    with torch.no_grad(), set_unflatten_seq_len(self._SEQ_LEN):
      actual = attn(x, attention_masks=None, positions=positions).view(
          2, self._SEQ_LEN, self._DIM
      )
      expected = self._attention_with_mask(attn, hf_mask)(
          x, attention_masks=None, positions=positions
      ).view(2, self._SEQ_LEN, self._DIM)
      full_causal = self._attention_with_mask(attn, causal_mask)(
          x, attention_masks=None, positions=positions
      ).view(2, self._SEQ_LEN, self._DIM)

    torch.testing.assert_close(actual, expected)
    # Rows that see fewer than W tokens are unaffected by the window; every
    # later row must differ from full causal attention.
    w = self._WINDOW
    torch.testing.assert_close(actual[:, :w], full_causal[:, :w])
    row_diff = (actual - full_causal).abs().amax(dim=(0, 2))
    self.assertTrue(
        bool((row_diff[w:] > 1e-4).all()),
        f"Rows >= window match full causal attention: {row_diff[w:]}",
    )
    self.assertIsInstance(
        attn.inner_attention, SlidingWindowScaledDotProductAttention
    )

  def test_flavor_sliding_layers_use_windowed_sdpa(self):
    config = _build_flavor_config("testmodel", attn_backend="sdpa")
    with torch.device("meta"):
      model = config.build()
    for layer in model.layers.values():
      inner = layer.attention.inner_attention
      if layer.attention.is_sliding:
        self.assertIsInstance(inner, SlidingWindowScaledDotProductAttention)
        self.assertEqual(
            inner.sliding_window_size,
            layer.attention.sliding_window_size - 1,
        )
      else:
        self.assertIs(type(inner), ScaledDotProductInnerAttention)


class TestGemma4HFReferenceParity(BaseDeviceTest):
  """Single-chip parity of TorchTitan Gemma4Model against HuggingFace Gemma4ForCausalLM."""

  def setUp(self):
    super().setUp()
    if not hasattr(transformers, "Gemma4ForCausalLM"):
      self.skipTest(
          "transformers.Gemma4ForCausalLM requires transformers >= 5.0"
      )

  @staticmethod
  def _build_hf_reference(enable_moe: bool, sliding_window: int):
    hf_config = transformers.Gemma4TextConfig(
        vocab_size=2048,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_global_key_value_heads=8,
        head_dim=16,
        global_head_dim=32,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        sliding_window=sliding_window,
        layer_types=["sliding_attention", "full_attention"],
        final_logit_softcapping=30.0,
        hidden_size_per_layer_input=0,
        attention_k_eq_v=True,
        num_kv_shared_layers=0,
        tie_word_embeddings=True,
        pad_token_id=None,
        enable_moe_block=enable_moe,
        num_experts=8 if enable_moe else None,
        top_k_experts=2 if enable_moe else None,
        moe_intermediate_size=128 if enable_moe else None,
        attn_implementation="sdpa",
    )
    hf_model = transformers.Gemma4ForCausalLM(hf_config).float().eval()
    # Perturb unit-initialized norms/scales/buffers so scaling bugs are caught.
    with torch.no_grad():
      for name, param in hf_model.named_parameters():
        if "norm" in name or "scale" in name:
          param.copy_(1.0 + 0.1 * torch.randn_like(param))
      for layer in hf_model.model.layers:
        layer.layer_scalar.fill_(0.85)
    return hf_model

  @parameterized.named_parameters(
      dict(
          testcase_name="dense",
          flavor="testmodel",
          enable_moe=False,
          logit_atol=5e-2,
          logit_rtol=1e-2,
          loss_atol=5e-3,
          loss_rtol=1e-3,
          grad_atol=8e-2,
          grad_rtol=2e-2,
      ),
      dict(
          testcase_name="moe",
          flavor="testmodel_moe",
          enable_moe=True,
          logit_atol=1e-1,
          logit_rtol=2e-2,
          loss_atol=5e-3,
          loss_rtol=2e-3,
          grad_atol=8e-2,
          grad_rtol=2e-2,
      ),
  )
  def test_one_step_matches_hf(
      self,
      flavor: str,
      enable_moe: bool,
      logit_atol: float,
      logit_rtol: float,
      loss_atol: float,
      loss_rtol: float,
      grad_atol: float,
      grad_rtol: float,
  ):
    use_cpu_safe_histc_patch()
    torch.manual_seed(42)

    # Use a sliding window smaller than seq_len so layer 0 actively masks keys
    # beyond the window while layer 1 attends globally.
    sliding_window = 8
    batch_size, seq_len = 2, 16

    hf_model = self._build_hf_reference(
        enable_moe=enable_moe, sliding_window=sliding_window
    )

    tt_config: Gemma4Model.Config = _build_flavor_config(flavor)
    tt_config.sliding_window_size = sliding_window
    for layer_cfg in tt_config.layers:
      if layer_cfg.attention.is_sliding:
        layer_cfg.attention.sliding_window_size = sliding_window
    if enable_moe:
      converter = TpuGroupedExpertsConverter(
          TpuGroupedExpertsConverter.Config()
      )
      tt_config = converter.convert(tt_config)
      for layer_cfg in tt_config.layers:
        if layer_cfg.moe is not None:
          layer_cfg.moe.routed_experts.inner_experts.use_pallas_grouped_mm = (
              False
          )

    tt_model = tt_config.build().to(self.accelerator_device)
    tt_model.init_states()

    adapter = Gemma4StateDictAdapter(tt_config, hf_assets_path=None)
    converted_sd = adapter.from_hf(
        {k: v.detach().clone() for k, v in hf_model.state_dict().items()}
    )
    missing, unexpected = tt_model.load_state_dict(converted_sd, strict=False)
    self.assertEmpty(missing)
    self.assertEmpty(unexpected)

    tokens_cpu = torch.randint(0, tt_config.vocab_size, (batch_size, seq_len))
    labels_cpu = torch.randint(0, tt_config.vocab_size, (batch_size, seq_len))
    tokens_dev = tokens_cpu.to(self.accelerator_device)
    labels_dev = labels_cpu.to(self.accelerator_device)
    positions_dev = torch.arange(
        seq_len, device=self.accelerator_device
    ).repeat(batch_size)
    num_tokens = labels_cpu.numel()

    with set_unflatten_seq_len(seq_len):
      tt_logits = tt_model(
          tokens_dev.view(-1), positions=positions_dev
      ).view(batch_size, seq_len, -1)
    tt_loss = cross_entropy_loss(tt_logits, labels_dev) / num_tokens
    tt_loss.backward()

    hf_logits = hf_model(input_ids=tokens_cpu).logits
    hf_loss = cross_entropy_loss(hf_logits, labels_cpu) / num_tokens
    hf_loss.backward()

    labels = dict(
        test_label=f"TorchTitan Gemma4Model ({self.accelerator_device.type})",
        ref_label="HF Gemma4ForCausalLM (CPU)",
    )
    test_utils.check_equivalence(
        tt_logits.detach().float().cpu(),
        hf_logits.detach().float(),
        atol=logit_atol,
        rtol=logit_rtol,
        check_name="Logits",
        **labels,
    )
    test_utils.check_equivalence(
        tt_loss.detach().float().cpu(),
        hf_loss.detach().float(),
        atol=loss_atol,
        rtol=loss_rtol,
        check_name="Loss",
        **labels,
    )

    tt_grads = {
        name: param.grad.detach().float().cpu()
        for name, param in tt_model.named_parameters()
        if param.grad is not None
    }
    raw_hf_grads = {
        name: param.grad.detach().float().cpu()
        for name, param in hf_model.named_parameters()
        if param.grad is not None
    }
    hf_grads_in_tt_keys = adapter.from_hf(raw_hf_grads)
    # When weights are tied, both models accumulate the embedding + lm_head
    # gradient into the single shared parameter (tok_embeddings.weight in TT).
    if tt_config.enable_weight_tying:
      hf_grads_in_tt_keys.pop("lm_head.weight", None)

    self.assertEqual(set(tt_grads.keys()), set(hf_grads_in_tt_keys.keys()))
    for name in sorted(tt_grads.keys()):
      test_utils.check_equivalence(
          tt_grads[name],
          hf_grads_in_tt_keys[name],
          atol=grad_atol,
          rtol=grad_rtol,
          check_name=f"Gradient ({name})",
          **labels,
      )

  @parameterized.named_parameters(
      dict(testcase_name="single_file", max_shard_size=None),
      # Force two shards so save_pretrained writes model.safetensors.index.json.
      dict(testcase_name="sharded_with_index", max_shard_size="1MB"),
  )
  def test_tied_hf_safetensors_load_into_untied_recipe(self, max_shard_size):
    """Real initial_load_in_hf path: tied HF safetensors -> untied Gemma4Model.

    Mirrors ``CheckpointManager._load_checkpoint(from_hf=True)``:
    ``to_hf`` builds the placeholder, ``dcp.load`` fills it from disk via
    ``HuggingFaceStorageReader``, and ``from_hf`` maps back. Gemma 4 HF
    checkpoints are tied, so ``lm_head.weight`` is absent on disk and the
    adapter must neither request it from DCP nor leave it unset.
    """
    import os
    import shutil
    import tempfile
    from safetensors import safe_open
    import torch.distributed.checkpoint as dcp

    torch.manual_seed(0)
    hf_model = self._build_hf_reference(enable_moe=False, sliding_window=8)
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
    save_kwargs = {"safe_serialization": True}
    if max_shard_size is not None:
      save_kwargs["max_shard_size"] = max_shard_size
    hf_model.save_pretrained(tmp_dir, **save_kwargs)

    has_index = os.path.exists(
        os.path.join(tmp_dir, "model.safetensors.index.json")
    )
    self.assertEqual(has_index, max_shard_size is not None)
    # Sanity: the on-disk checkpoint really has no lm_head.weight.
    on_disk_keys = set()
    for fname in os.listdir(tmp_dir):
      if fname.endswith(".safetensors"):
        with safe_open(os.path.join(tmp_dir, fname), framework="pt") as f:
          on_disk_keys.update(f.keys())
    self.assertNotIn("lm_head.weight", on_disk_keys)

    tt_config: Gemma4Model.Config = _build_flavor_config(
        "testmodel", enable_weight_tying=False
    )
    tt_config.sliding_window_size = 8
    for layer_cfg in tt_config.layers:
      if layer_cfg.attention.is_sliding:
        layer_cfg.attention.sliding_window_size = 8
    tt_model = tt_config.build()
    tt_model.init_states()
    self.assertIsNot(tt_model.tok_embeddings.weight, tt_model.lm_head.weight)

    adapter = Gemma4StateDictAdapter(tt_config, hf_assets_path=tmp_dir)
    state_dict = {k: v.detach().clone() for k, v in tt_model.state_dict().items()}
    hf_placeholder = adapter.to_hf(state_dict)
    # The placeholder must match the on-disk layout exactly (DCP is strict):
    # text-only Gemma4ForCausalLM keys, and no lm_head.weight.
    self.assertEqual(set(hf_placeholder), on_disk_keys)
    dcp.load(
        hf_placeholder,
        storage_reader=adapter.get_hf_storage_reader(tmp_dir),
    )
    loaded = adapter.from_hf(hf_placeholder)
    missing, unexpected = tt_model.load_state_dict(loaded, strict=True)
    self.assertEmpty(missing)
    self.assertEmpty(unexpected)

    hf_embed = hf_model.model.embed_tokens.weight.detach()
    torch.testing.assert_close(tt_model.tok_embeddings.weight.detach(), hf_embed)
    torch.testing.assert_close(tt_model.lm_head.weight.detach(), hf_embed)

    # Loaded untied model must produce the same logits as the tied HF model.
    batch_size, seq_len = 2, 16
    tokens = torch.randint(0, tt_config.vocab_size, (batch_size, seq_len))
    positions = torch.arange(seq_len).repeat(batch_size)
    with torch.no_grad(), set_unflatten_seq_len(seq_len):
      tt_logits = tt_model(tokens.view(-1), positions=positions).view(
          batch_size, seq_len, -1
      )
      hf_logits = hf_model(input_ids=tokens).logits
    torch.testing.assert_close(tt_logits, hf_logits, atol=5e-2, rtol=1e-2)


if __name__ == "__main__":
  absltest.main()
