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
    IGNORE_INDEX,
    cross_entropy_loss,
)
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest as BaseDeviceTest
from torchtitan.experiments.tpu.examples.gemma4 import (
    _build_flavor_config,
    _build_gemma4_layers,
    gemma4_configs,
)
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch
from torchtitan.experiments.tpu.examples.gemma4.config_registry import (
    gemma4_26b_a4b,
    gemma4_31b,
    gemma4_debugmodel,
    gemma4_debugmodel_fused_ce,
    gemma4_moe_debug,
    gemma4_moe_debug_fused_ce,
    gemma4_moe_debug_pallas,
    gemma4_moe_testmodel,
)
from torchtitan.experiments.tpu.examples.gemma4.infra.parallelize import (
    configure_gemma4_simple_fsdp_shard_dims,
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
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.overrides.fused_loss import (
    FusedLossLinearWrapper,
)
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig
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
      self.assertIsInstance(inner, SlidingWindowScaledDotProductAttention)
      if layer.attention.is_sliding:
        self.assertEqual(
            inner.sliding_window_size,
            layer.attention.sliding_window_size - 1,
        )
      else:
        self.assertIsNone(inner.sliding_window_size)

  def test_packed_document_masking_on_sliding_and_global_layers(self):
    """Packed positions (resetting to 0 at doc boundaries) mask cross-doc keys."""
    head_dim = 8
    for is_sliding, window in ((True, self._WINDOW), (False, None)):
      cfg = Gemma4Attention.Config(
          dim=self._DIM,
          n_heads=4,
          n_kv_heads=2,
          head_dim=head_dim,
          is_sliding=is_sliding,
          sliding_window_size=window,
          qk_norm=RMSNorm.Config(
              normalized_shape=head_dim, eps=1e-6, param_init=_NORM_INIT
          ),
          v_norm=RMSNorm.Config(
              normalized_shape=head_dim, eps=1e-6, elementwise_affine=False
          ),
          qkv_linear=QKVLinear.Config(
              head_dim=head_dim,
              wq=Linear.Config(
                  in_features=self._DIM,
                  out_features=32,
                  param_init=_LINEAR_INIT,
              ),
              wkv=Linear.Config(
                  in_features=self._DIM,
                  out_features=16,
                  param_init=_LINEAR_INIT,
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

      # Two packed documents of length 6 in a 12-token sequence (doc boundary
      # at index 6 is inside the 4-token sliding window for tokens 6..8).
      doc_split = 6
      positions_1d = torch.cat(
          [torch.arange(doc_split), torch.arange(self._SEQ_LEN - doc_split)]
      )
      positions = positions_1d.repeat(2)
      seg_1d = torch.tensor(
          [1] * doc_split + [2] * (self._SEQ_LEN - doc_split)
      )
      same_doc = seg_1d[:, None] == seg_1d[None, :]
      dist = (
          torch.arange(self._SEQ_LEN)[:, None]
          - torch.arange(self._SEQ_LEN)[None, :]
      )
      unpacked_mask = dist >= 0
      if is_sliding and window is not None:
        unpacked_mask = unpacked_mask & (dist < window)
      packed_mask = unpacked_mask & same_doc

      x = torch.randn(2 * self._SEQ_LEN, self._DIM)
      with torch.no_grad(), set_unflatten_seq_len(self._SEQ_LEN):
        actual = attn(x, attention_masks=None, positions=positions).view(
            2, self._SEQ_LEN, self._DIM
        )
        expected = self._attention_with_mask(attn, packed_mask)(
            x, attention_masks=None, positions=positions
        ).view(2, self._SEQ_LEN, self._DIM)
        unpacked = self._attention_with_mask(attn, unpacked_mask)(
            x, attention_masks=None, positions=positions
        ).view(2, self._SEQ_LEN, self._DIM)

      torch.testing.assert_close(actual, expected)
      # Document 1 (rows :doc_split) is identical; document 2 must differ when
      # cross-document keys are masked out.
      torch.testing.assert_close(
          actual[:, :doc_split], unpacked[:, :doc_split]
      )
      doc2_diff = (actual[:, doc_split:] - unpacked[:, doc_split:]).abs().amax()
      self.assertGreater(doc2_diff.item(), 1e-4)


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


# Splash / tokamax attention patches vs. SDPA on Gemma 4 attention layers across
# both debugmodel head dimensions (64 sliding / 128 global) and production
# 31B / 26B-A4B head dimensions (256 sliding / 512 global):
# - layer 0 is sliding-window (window 128, full RoPE, Splash LocalMask),
# - layer 5 is global (K=V, partial RoPE 0.25, Splash CausalMask).
_SPLASH_TEST_CONFIGS = [
    dict(
        testcase_name=(
            f"_{kernel}_{layer_name}"
            if (head_dim, global_head_dim) == (64, 128)
            else (
                f"_{kernel}_{layer_name}"
                f"_hd{head_dim if layer_id == 0 else global_head_dim}"
            )
        ),
        kernel=kernel,
        layer_id=layer_id,
        head_dim=head_dim,
        global_head_dim=global_head_dim,
    )
    for kernel in ("splash", "splash_batch_native", "tokamax")
    for head_dim, global_head_dim in ((64, 128), (256, 512))
    for layer_name, layer_id in (("SWA_ROPE", 0), ("GLOBAL_KEQV", 5))
]


class TestGemma4SplashAttentionParity(BaseDeviceTest):
  """TPU-only checks that Splash and Tokamax patches match Gemma 4 SDPA on packed docs."""

  def setUp(self):
    super().setUp()
    # torch_tpu defaults fp32 matmuls to a single bf16 pass ("medium"); run
    # the non-attention matmuls (wq/wk/wv/wo) at full fp32 so the comparison
    # isolates the attention kernel numerics.
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    self.addCleanup(torch.set_float32_matmul_precision, previous_precision)

  def _check_relative_l2(
      self,
      test: torch.Tensor,
      reference: torch.Tensor,
      max_rel_error: float,
      check_name: str,
  ) -> None:
    """Checks ``||test - reference|| / ||reference|| <= max_rel_error``."""
    test_f = test.detach().float().cpu()
    ref_f = reference.detach().float().cpu()
    rel_error = (
        torch.linalg.vector_norm(test_f - ref_f)
        / torch.linalg.vector_norm(ref_f).clamp_min(1e-12)
    ).item()
    self.assertLessEqual(
        rel_error,
        max_rel_error,
        f"{check_name}: relative L2 error {rel_error:.3e} >"
        f" {max_rel_error:.1e}",
    )

  @parameterized.parameters(gemma4_31b, gemma4_26b_a4b)
  def test_production_recipes_enable_splash_attention(self, recipe):
    cfg = recipe()
    self.assertTrue(cfg.splash_attention_kernel.use_splash_attention_kernel)
    self.assertFalse(
        cfg.splash_attention_kernel.use_tokamax_splash_attention_kernel
    )

  @parameterized.named_parameters(
      dict(testcase_name="splash", kernel="splash"),
      dict(testcase_name="tokamax", kernel="tokamax"),
  )
  def test_patch_replaces_every_layer(self, kernel: str):
    """Both Splash and Tokamax patches replace/rebind every Gemma 4 layer.

    Verifies that on a meta-device ``debugmodel`` (5 sliding-window layers +
    1 global K=V layer), the patch takes effect on all 6 layers and preserves
    per-layer ``sliding_window_size`` (127 on sliding layers, None on global).
    """
    with torch.device("meta"):
      model = gemma4_configs["debugmodel"].build()
    if kernel == "tokamax":
      workarounds.use_tokamax_splash_attention_patch(
          model, SplashAttentionKernelConfig()
      )
    else:
      workarounds.use_splash_attention_patch(
          model, SplashAttentionKernelConfig()
      )
    layers = list(model.layers.values())
    self.assertLen(layers, 6)
    for idx, layer in enumerate(layers):
      inner = layer.attention.inner_attention
      expected_window = 127 if idx < 5 else None
      if kernel == "tokamax":
        self.assertIn("forward", vars(inner))
        self.assertEqual(inner.sliding_window_size, expected_window)
      else:
        self.assertNotIn("ScaledDotProduct", type(inner).__name__)
        self.assertEqual(inner.inner.local_window_size, expected_window)

  @parameterized.named_parameters(*_SPLASH_TEST_CONFIGS)
  def test_gemma4_splash_attention_matches_sdpa(
      self,
      kernel: str,
      layer_id: int,
      head_dim: int = 64,
      global_head_dim: int = 128,
  ):
    """Tests Splash and tokamax attention patches against SDPA on packed docs.

    Exercises ``use_splash_attention_patch`` (both document-aware block
    skipping and batch-native masking) and
    ``use_tokamax_splash_attention_patch`` on a ``Gemma4Attention`` layer across
    both debugmodel head dimensions (64 sliding / 128 global) and production
    31B / 26B-A4B head dimensions (256 sliding / 512 global): layer 0 has
    ``sliding_window_size = 128`` (patched to ``127`` on ``inner_attention``,
    exercising Splash ``LocalMask``), and layer 5 is global ``K=V`` with
    partial RoPE 0.25 (exercising ``CausalMask``). The input has packed
    documents with resets inside the 128-token window so both windowing and
    ``segment_ids`` masking are active.
    """
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("Splash and tokamax attention kernels only run on TPU.")

    batch, seq_len, dim = 2, 512, 256
    layers = _build_gemma4_layers(
        n_layers=6,
        dim=dim,
        hidden_dim=512,
        n_heads=4,
        n_kv_heads=2,
        n_global_kv_heads=2,
        head_dim=head_dim,
        global_head_dim=global_head_dim,
        norm_eps=1e-6,
        max_seq_len=seq_len,
        sliding_window_size=128,
    )
    attention_config = layers[layer_id].attention

    torch.manual_seed(0)
    attention_cpu = attention_config.build()
    attention_cpu.init_states(buffer_device=torch.device("cpu"))
    attention_cpu = attention_cpu.cpu()
    # Move Q/K norm weights slightly off all-ones so per-channel scaling is
    # non-trivial in both forward and backward.
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
      for name, param in attention_cpu.named_parameters():
        if "norm" in name:
          param.copy_(
              1.0 + 0.1 * torch.randn(param.shape, generator=gen).to(param)
          )
    attention_device = copy.deepcopy(attention_cpu).to(self.accelerator_device)

    # On v5e-1 (16 MiB default scoped VMEM limit), fp32 splash_mha_dkv with
    # head_dim=512 and 512x512 tiles uses 16.34 MiB (Splash) / 18.62 MiB
    # (Tokamax); using 256x256 tiles halves fp32 VMEM (~8.5 MiB) and exercises
    # a 2x2 tile grid at seq_len=512.
    block_size = 512 if (head_dim, global_head_dim) == (64, 128) else 256
    sa_config = SplashAttentionKernelConfig(
        sa_block_q=block_size,
        sa_block_kv=block_size,
        sa_block_kv_compute=block_size,
        sa_block_q_dkv=block_size,
        sa_block_kv_dkv=block_size,
        sa_block_kv_dkv_compute=block_size,
        sa_block_q_dq=block_size,
        sa_block_kv_dq=block_size,
        sa_use_document_aware_block_skipping=(kernel == "splash"),
    )
    if kernel == "tokamax":
      workarounds.use_tokamax_splash_attention_patch(
          attention_device, sa_config
      )
    elif kernel in ("splash", "splash_batch_native"):
      workarounds.use_splash_attention_patch(attention_device, sa_config)
    else:
      self.fail(f"Unknown kernel: {kernel}")

    # Row 0 packs 3 documents (boundaries at 150 and 350, both inside the
    # 128-token window of subsequent queries); row 1 is a single document.
    doc_lengths_row0 = (150, 200, seq_len - 350)
    positions_row0 = torch.cat([torch.arange(n) for n in doc_lengths_row0])
    positions_row1 = torch.arange(seq_len)
    positions_cpu = torch.cat([positions_row0, positions_row1]).to(torch.long)
    positions_device = positions_cpu.to(self.accelerator_device)

    x_cpu, x_device = self._create_input_tensor_cpu_device(
        size=(batch * seq_len, dim), requires_grad=True
    )

    with set_unflatten_seq_len(seq_len):
      y_cpu = attention_cpu(x_cpu, attention_masks=None, positions=positions_cpu)
      y_device = attention_device(
          x_device, attention_masks=None, positions=positions_device
      )

    # Splash/tokamax run QK^T and PV in bf16 on the TPU MXU, vs. fp32 SDPA on
    # CPU. Because Gemma 4 applies unit-RMS qk_norm with scale=1.0, pre-softmax
    # logits QK^T have std sqrt(head_dim) (8..11.3 at head_dim 64/128 vs.
    # 16..22.6 at head_dim 256/512), increasing bf16 rounding at head_dim
    # 256/512 (measured ~1.05e-2 fwd / ~3.44e-2 grad vs. >1e-1 unmasked
    # negative control).
    fwd_max_rel_error = (
        1e-2 if (head_dim, global_head_dim) == (64, 128) else 1.2e-2
    )
    bwd_max_rel_error = (
        2.5e-2 if (head_dim, global_head_dim) == (64, 128) else 4e-2
    )
    self._check_relative_l2(
        y_device,
        y_cpu,
        max_rel_error=fwd_max_rel_error,
        check_name=f"Gemma4Attention ({kernel}) forward",
    )

    grad_out_cpu, grad_out_device = self._create_input_tensor_cpu_device(
        size=y_cpu.shape
    )
    y_cpu.backward(grad_out_cpu)
    y_device.backward(grad_out_device)

    self._check_relative_l2(
        x_device.grad,
        x_cpu.grad,
        max_rel_error=bwd_max_rel_error,
        check_name=f"Gemma4Attention ({kernel}) input grad",
    )
    for (name, p_cpu), (_, p_device) in zip(
        attention_cpu.named_parameters(), attention_device.named_parameters()
    ):
      self._check_relative_l2(
          p_device.grad,
          p_cpu.grad,
          max_rel_error=bwd_max_rel_error,
          check_name=f"Gemma4Attention ({kernel}) param grad ({name})",
      )

    # Failure control: if the kernel ignored the sliding window or the
    # packed-document boundaries (plain full-sequence causal SDPA with the same
    # RoPE positions), the output would shift by >10% relative L2, well above
    # the 1e-2 tolerance checked above.
    unmasked_cpu = copy.deepcopy(attention_cpu)
    unmasked_cpu.inner_attention = ScaledDotProductInnerAttention.Config().build()
    with torch.no_grad(), set_unflatten_seq_len(seq_len):
      y_unmasked = unmasked_cpu(
          x_cpu, attention_masks=None, positions=positions_cpu
      )
    unmasked_diff = (
        torch.linalg.vector_norm(y_device.detach().float().cpu() - y_unmasked)
        / torch.linalg.vector_norm(y_unmasked)
    ).item()
    self.assertGreater(
        unmasked_diff,
        0.1,
        "Unmasked causal attention is too close to the windowed/packed output;"
        " the 1e-2 check above would not catch a broken mask.",
    )


class TestGemma4FusedCrossEntropyConfig(parameterized.TestCase):
  """Config-level checks for Gemma 4 soft-capped fused Pallas cross-entropy."""

  @parameterized.parameters(
      gemma4_debugmodel_fused_ce,
      gemma4_moe_debug_fused_ce,
      gemma4_31b,
      gemma4_26b_a4b,
  )
  def test_recipe_wires_soft_capped_pallas_cross_entropy(self, recipe):
    cfg = recipe()
    self.assertIsInstance(cfg.loss, PallasCrossEntropyLoss.Config)
    self.assertEqual(cfg.loss.soft_cap, 30.0)
    self.assertEqual(cfg.loss.output_multiplier, 1.0)
    self.assertIn(
        "torchtitan.experiments.tpu.overrides.fused_loss."
        "fused_loss_soft_capped_lm_head",
        cfg.override.imports,
    )

  def test_override_replaces_lm_head_with_fused_loss_linear_wrapper(self):
    from torchtitan.config import apply_overrides

    cfg = gemma4_debugmodel_fused_ce()
    apply_overrides(cfg.override, cfg)
    lm_head_cfg = cfg.model_spec.model.lm_head
    self.assertIsInstance(lm_head_cfg, FusedLossLinearWrapper.Config)
    self.assertEqual(lm_head_cfg.soft_cap, 30.0)
    self.assertEqual(lm_head_cfg.output_multiplier, 1.0)

    with torch.device("meta"):
      model = cfg.model_spec.model.build()
    self.assertIsInstance(model.lm_head, FusedLossLinearWrapper)
    self.assertTrue(model.lm_head.returns_loss_operands)
    self.assertEqual(tuple(model.lm_head.weight.shape), (2048, 256))


class TestGemma4FusedCrossEntropyParity(BaseDeviceTest):
  """TPU parity of PallasCrossEntropyLoss(soft_cap=30.0) vs SoftCappedLinear + CrossEntropyLoss."""

  def setUp(self):
    super().setUp()
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    self.addCleanup(torch.set_float32_matmul_precision, previous_precision)

  def _check_relative_l2(
      self,
      test: torch.Tensor,
      reference: torch.Tensor,
      max_rel_error: float,
      check_name: str,
  ) -> None:
    test_f = test.detach().float().cpu()
    ref_f = reference.detach().float().cpu()
    rel_error = (
        torch.linalg.vector_norm(test_f - ref_f)
        / torch.linalg.vector_norm(ref_f).clamp_min(1e-12)
    ).item()
    self.assertLessEqual(
        rel_error,
        max_rel_error,
        f"{check_name}: relative L2 error {rel_error:.3e} >"
        f" {max_rel_error:.1e}",
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="float32",
          dtype=torch.float32,
          loss_atol=1e-3,
          loss_rtol=1e-3,
          # Pallas Mosaic TPU kernel uses MXU 1-pass bf16-input dot products
          # internally (measured ~2.9e-3 vs CPU IEEE FP32; negative control with
          # soft_cap=None diverges by >5e-2).
          grad_rel_l2=5e-3,
      ),
      dict(
          testcase_name="bfloat16",
          dtype=torch.bfloat16,
          loss_atol=1.5e-2,
          loss_rtol=5e-3,
          grad_rel_l2=1.5e-2,
      ),
  )
  def test_fused_cross_entropy_matches_eager_soft_capped_linear(
      self,
      dtype: torch.dtype,
      loss_atol: float,
      loss_rtol: float,
      grad_rel_l2: float,
  ):
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("PallasCrossEntropyLoss Mosaic kernel only runs on TPU.")

    torch.manual_seed(42)
    num_tokens, dim, vocab_size = 1024, 256, 2048
    soft_cap = 30.0

    eager_head_cfg = SoftCappedLinear.Config(
        in_features=dim,
        out_features=vocab_size,
        output_soft_cap_temp=soft_cap,
        param_init={"weight": partial(nn.init.normal_, mean=0.0, std=1.5)},
    )
    fused_head_cfg = FusedLossLinearWrapper.Config(
        in_features=dim,
        out_features=vocab_size,
        soft_cap=soft_cap,
        output_multiplier=1.0,
        param_init={"weight": partial(nn.init.normal_, mean=0.0, std=1.5)},
    )

    eager_head = eager_head_cfg.build().to(dtype=dtype)
    eager_head.init_states()
    fused_head = fused_head_cfg.build().to(
        device=self.accelerator_device, dtype=dtype
    )
    fused_head.load_state_dict(eager_head.state_dict())

    # Scale hidden states so pre-cap logits (z = x @ w.T) have std ~24 and
    # exercise the non-linear saturation region of 30 * tanh(z / 30).
    x_cpu = torch.randn(num_tokens, dim, dtype=dtype, requires_grad=True)
    x_tpu = x_cpu.detach().to(self.accelerator_device).requires_grad_(True)
    labels_cpu = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.long)
    # Include IGNORE_INDEX (-100) positions to verify masked token handling.
    labels_cpu[::16] = IGNORE_INDEX
    labels_tpu = labels_cpu.to(self.accelerator_device)
    valid_tokens_cpu = (labels_cpu != IGNORE_INDEX).sum()
    valid_tokens_tpu = valid_tokens_cpu.to(self.accelerator_device)

    eager_loss_fn = CrossEntropyLoss.Config().build()
    fused_loss_fn = PallasCrossEntropyLoss.Config(
        b_block_size=1024,
        h_block_size=256,
        v_block_size=2048,
        soft_cap=soft_cap,
        output_multiplier=1.0,
    ).build()

    logits_cpu = eager_head(x_cpu)
    loss_cpu, _ = eager_loss_fn(logits_cpu, labels_cpu, valid_tokens_cpu)
    loss_cpu.backward()

    operands_tpu = fused_head(x_tpu)
    loss_tpu, _ = fused_loss_fn(operands_tpu, labels_tpu, valid_tokens_tpu)
    loss_tpu.backward()

    test_utils.check_equivalence(
        loss_tpu.detach().float().cpu(),
        loss_cpu.detach().float().cpu(),
        atol=loss_atol,
        rtol=loss_rtol,
        check_name=f"Fused CE loss ({dtype})",
    )
    self._check_relative_l2(
        x_tpu.grad,
        x_cpu.grad,
        max_rel_error=grad_rel_l2,
        check_name=f"Fused CE hidden grad ({dtype})",
    )
    self._check_relative_l2(
        fused_head.weight.grad,
        eager_head.weight.grad,
        max_rel_error=grad_rel_l2,
        check_name=f"Fused CE lm_head.weight grad ({dtype})",
    )

    # Negative control: running PallasCrossEntropyLoss without soft_cap (None)
    # on the same inputs must diverge well beyond the test tolerances.
    uncapped_loss_fn = PallasCrossEntropyLoss.Config(
        b_block_size=1024,
        h_block_size=256,
        v_block_size=2048,
        soft_cap=None,
        output_multiplier=1.0,
    ).build()
    x_uncapped = x_cpu.detach().to(self.accelerator_device).requires_grad_(True)
    fused_head.zero_grad()
    loss_uncapped, _ = uncapped_loss_fn(
        fused_head(x_uncapped), labels_tpu, valid_tokens_tpu
    )
    loss_uncapped.backward()
    loss_diff = (loss_uncapped.detach().float().cpu() - loss_cpu.detach().float()).abs().item()
    self.assertGreater(
        loss_diff,
        0.5,
        f"Uncapped loss differed by only {loss_diff:.3e}; soft-cap check is not sensitive.",
    )
    grad_uncapped_rel = (
        torch.linalg.vector_norm(x_uncapped.grad.detach().float().cpu() - x_cpu.grad.detach().float())
        / torch.linalg.vector_norm(x_cpu.grad.detach().float())
    ).item()
    self.assertGreater(
        grad_uncapped_rel,
        0.1,
        f"Uncapped grad relative L2 was only {grad_uncapped_rel:.3e}; soft-cap check is not sensitive.",
    )


class TestGemma4MoEPallasKernels(BaseDeviceTest):
  """Tests MoE Pallas kernels, kernel-native weights, and route-score fusion for Gemma 4."""

  def setUp(self):
    super().setUp()
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    self.addCleanup(torch.set_float32_matmul_precision, previous_precision)

  def _check_relative_l2(
      self,
      test: torch.Tensor,
      reference: torch.Tensor,
      max_rel_error: float,
      check_name: str,
  ) -> None:
    test_f = test.detach().float().cpu()
    ref_f = reference.detach().float().cpu()
    rel_error = (
        torch.linalg.vector_norm(test_f - ref_f)
        / torch.linalg.vector_norm(ref_f).clamp_min(1e-12)
    ).item()
    self.assertLessEqual(
        rel_error,
        max_rel_error,
        f"{check_name}: relative L2 error {rel_error:.3e} >"
        f" {max_rel_error:.1e}",
    )

  @parameterized.parameters(gemma4_moe_debug_pallas, gemma4_26b_a4b)
  def test_moe_pallas_config_enables_all_five_levers(self, recipe):
    cfg = recipe()
    model_cfg = cfg.model_spec.model
    dispatchers = list(
        model_cfg.traverse(TpuAllToAllTokenDispatcher.Config)
    )
    self.assertNotEmpty(dispatchers)
    for _, dispatcher, _, _ in dispatchers:
      self.assertTrue(dispatcher.fuse_route_score_backward)

    experts_list = list(model_cfg.traverse(TpuGroupedExperts.Config))
    self.assertNotEmpty(experts_list)
    for _, experts, _, _ in experts_list:
      self.assertEqual(experts.activation, "gelu_tanh")
      self.assertTrue(experts.use_pallas_grouped_mm)
      self.assertTrue(experts.use_pallas_paired_grouped_mm)
      self.assertEqual(experts.pallas_grouped_mm_tile_m, 256)
      self.assertEqual(experts.pallas_grouped_mm_tile_k, 0)
      self.assertEqual(experts.pallas_grouped_mm_tile_n, 0)
      self.assertEqual(experts.pallas_grouped_mm_tgmm_tiling, (256, 0, 0))
      self.assertTrue(experts.use_kernel_native_weight_layout)

  def test_kernel_native_weight_layout_state_dict_adapter_round_trip(self):
    """Kernel-native expert storage (w1_EDF, w2_EFD, w3_EDF) round-trips through HF state_dict."""
    torch.manual_seed(0)
    cfg = _build_flavor_config(
        "testmodel_moe",
        converters=[
            TpuTokenDispatcherConverter.Config(),
            TpuGroupedExpertsConverter.Config(),
        ],
        enable_weight_tying=False,
    )
    for _, experts, _, _ in cfg.traverse(TpuGroupedExperts.Config):
      experts.use_pallas_grouped_mm = True
      experts.use_pallas_paired_grouped_mm = True
      experts.pallas_grouped_mm_tile_m = 256
      experts.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)
      experts.use_kernel_native_weight_layout = True

    model = cfg.build()
    model.init_states()

    inner = model.layers["0"].moe.routed_experts.inner_experts
    # Native in-memory parameters are transposed: (E=8, D=128, F=128).
    self.assertTrue(hasattr(inner, "w1_EDF"))
    self.assertTrue(hasattr(inner, "w2_EFD"))
    self.assertTrue(hasattr(inner, "w3_EDF"))
    self.assertFalse(hasattr(inner, "w1_EFD"))

    # Fill w1_EDF with an asymmetric tensor so a missing transpose is caught even when D == F.
    for layer in model.layers.values():
      exp = layer.moe.routed_experts.inner_experts
      with torch.no_grad():
        exp.w1_EDF.copy_(torch.randn_like(exp.w1_EDF))
        exp.w2_EFD.copy_(torch.randn_like(exp.w2_EFD))
        exp.w3_EDF.copy_(torch.randn_like(exp.w3_EDF))

    expected_w1_edf = inner.w1_EDF.detach().clone()
    expected_w2_efd = inner.w2_EFD.detach().clone()
    expected_w3_edf = inner.w3_EDF.detach().clone()

    # state_dict() post-hook must expose canonical keys (w1_EFD, w2_EDF, w3_EFD)
    # with transposed values.
    tt_sd = model.state_dict()
    self.assertIn(
        "layers.0.moe.routed_experts.inner_experts.w1_EFD", tt_sd
    )
    self.assertNotIn(
        "layers.0.moe.routed_experts.inner_experts.w1_EDF", tt_sd
    )
    torch.testing.assert_close(
        tt_sd["layers.0.moe.routed_experts.inner_experts.w1_EFD"],
        expected_w1_edf.transpose(-2, -1),
    )

    adapter = Gemma4StateDictAdapter(cfg, hf_assets_path=None)
    hf_sd = adapter.to_hf(tt_sd)
    self.assertIn(
        "model.language_model.layers.0.experts.gate_up_proj", hf_sd
    )
    self.assertIn("model.language_model.layers.0.experts.down_proj", hf_sd)

    # Zero out model weights and reload from converted HF state dict.
    with torch.no_grad():
      for p in model.parameters():
        p.zero_()
    reloaded_tt_sd = adapter.from_hf(hf_sd)
    missing, unexpected = model.load_state_dict(reloaded_tt_sd, strict=True)
    self.assertEmpty(missing)
    self.assertEmpty(unexpected)

    torch.testing.assert_close(inner.w1_EDF, expected_w1_edf)
    torch.testing.assert_close(inner.w2_EFD, expected_w2_efd)
    torch.testing.assert_close(inner.w3_EDF, expected_w3_edf)

  def test_configure_simple_fsdp_shard_dims_with_kernel_native_weights(self):
    """configure_gemma4_simple_fsdp_shard_dims shards uneven experts on dim 1 under native layout."""
    from types import SimpleNamespace

    cfg = gemma4_moe_debug_pallas().model_spec.model
    # Set E=8 (not divisible by dp_shard=6), D=192 and F=96 (both divisible by 6).
    cfg.vocab_size = 192
    cfg.dim = 192
    cfg.tok_embeddings.num_embeddings = 192
    cfg.tok_embeddings.embedding_dim = 192
    cfg.lm_head.in_features = 192
    cfg.lm_head.out_features = 192
    for layer in cfg.layers:
      experts = layer.moe.routed_experts.inner_experts
      experts.num_experts = 8
      experts.dim = 192
      experts.hidden_dim = 96

    with torch.device("meta"):
      model = cfg.build()
    pdims = SimpleNamespace(dp_shard=6, ep_enabled=False)
    configure_gemma4_simple_fsdp_shard_dims(model, pdims)
    for layer in model.layers.values():
      inner = layer.moe.routed_experts.inner_experts
      self.assertEqual(getattr(inner, "_fsdp_shard_dim", None), 1)

  def test_gemma4_moe_pallas_kernels_match_native_reference(self):
    """Full Gemma4MoE with Pallas grouped MM, paired W1/W3, native weights, and fused route-score bwd."""
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("MoE Pallas grouped matmul kernels only run on TPU.")

    use_cpu_safe_histc_patch()
    torch.manual_seed(123)

    ref_cfg = copy.deepcopy(gemma4_moe_debug().model_spec.model.layers[0].moe)
    cand_cfg = copy.deepcopy(
        gemma4_moe_debug_pallas().model_spec.model.layers[0].moe
    )

    ref_moe = ref_cfg.build().to(self.accelerator_device)
    ref_moe.init_states()
    # Perturb unit-initialized router and norm scales so scaling bugs are caught.
    with torch.no_grad():
      ref_moe.pre_feedforward_layernorm_2.weight.copy_(
          1.0 + 0.1 * torch.randn_like(ref_moe.pre_feedforward_layernorm_2.weight)
      )
      ref_moe.router.scale.copy_(
          1.0 + 0.1 * torch.randn_like(ref_moe.router.scale)
      )
      ref_moe.router.per_expert_scale.copy_(
          1.0 + 0.1 * torch.randn_like(ref_moe.router.per_expert_scale)
      )

    cand_moe = cand_cfg.build().to(self.accelerator_device)
    cand_moe.init_states()
    missing, unexpected = cand_moe.load_state_dict(
        ref_moe.state_dict(), strict=True
    )
    self.assertEmpty(missing)
    self.assertEmpty(unexpected)

    # 512 tokens * top_k=2 = 1024 routed tokens (4 tiles of tile_m=256).
    num_tokens, dim = 512, 256
    x_ref = torch.randn(
        num_tokens, dim, device=self.accelerator_device, requires_grad=True
    )
    x_cand = x_ref.detach().clone().requires_grad_(True)
    grad_out = torch.randn(num_tokens, dim, device=self.accelerator_device)

    y_ref = ref_moe(x_ref)
    y_cand = cand_moe(x_cand)

    self._check_relative_l2(
        y_cand,
        y_ref,
        max_rel_error=1e-2,
        check_name="Gemma4MoE Pallas forward",
    )

    y_ref.backward(grad_out)
    y_cand.backward(grad_out)

    self._check_relative_l2(
        x_cand.grad,
        x_ref.grad,
        max_rel_error=2e-2,
        check_name="Gemma4MoE Pallas input grad",
    )

    ref_grads = {
        name: p.grad.detach()
        for name, p in ref_moe.named_parameters()
        if p.grad is not None
    }
    cand_grads_raw = {
        name: p.grad.detach()
        for name, p in cand_moe.named_parameters()
        if p.grad is not None
    }
    # Map kernel-native parameter gradient names/shapes back to canonical layout.
    native_to_canonical = {
        "routed_experts.inner_experts.w1_EDF": (
            "routed_experts.inner_experts.w1_EFD"
        ),
        "routed_experts.inner_experts.w2_EFD": (
            "routed_experts.inner_experts.w2_EDF"
        ),
        "routed_experts.inner_experts.w3_EDF": (
            "routed_experts.inner_experts.w3_EFD"
        ),
    }
    cand_grads = {}
    for name, g in cand_grads_raw.items():
      if name in native_to_canonical:
        cand_grads[native_to_canonical[name]] = g.transpose(-2, -1).contiguous()
      else:
        cand_grads[name] = g

    self.assertEqual(set(cand_grads.keys()), set(ref_grads.keys()))
    for name in sorted(ref_grads.keys()):
      self._check_relative_l2(
          cand_grads[name],
          ref_grads[name],
          max_rel_error=2.5e-2,
          check_name=f"Gemma4MoE Pallas param grad ({name})",
      )

    # Negative control: if the Pallas path ran silu instead of gelu_tanh on the
    # same weights and inputs, the output would shift by >10% relative L2.
    silu_cfg = copy.deepcopy(cand_cfg)
    silu_cfg.routed_experts.inner_experts.activation = "silu"
    silu_moe = silu_cfg.build().to(self.accelerator_device)
    silu_moe.init_states()
    silu_moe.load_state_dict(ref_moe.state_dict(), strict=True)
    with torch.no_grad():
      y_silu = silu_moe(x_ref.detach())
    silu_rel_diff = (
        torch.linalg.vector_norm((y_silu - y_ref).float())
        / torch.linalg.vector_norm(y_ref.float())
    ).item()
    self.assertGreater(
        silu_rel_diff,
        0.1,
        f"SiLU MoE output differed by only {silu_rel_diff:.3e} from GeLU-tanh;"
        " the 1e-2 check above would not catch a wrong activation.",
    )


if __name__ == "__main__":
  absltest.main()
