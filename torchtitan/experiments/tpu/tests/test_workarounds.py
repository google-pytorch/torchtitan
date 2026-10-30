"""Unit tests for TPU workarounds in torchtitan.experiments.tpu.workarounds."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torchtitan.experiments.tpu import workarounds


class WorkaroundsTest(parameterized.TestCase):
  """Unit tests for TPU workarounds."""

  # NOTE: `test_rope_tpu_optimization_patch` and
  # `test_benchmark_rope_gather_vs_2d_indexing` were removed alongside
  # `workarounds.use_rope_tpu_optimization_patch` in the d771abb0 upstream
  # import. Upstream's `_reshape_for_broadcast` now performs the same 2D
  # advanced indexing over the folded `[T]` token stream, so there is no longer
  # a patched-vs-original pair to compare.

  def test_wrapper_conversion_positions_to_segment_ids(self):
    """Patched splash attention converts positions to segment ids."""
    import unittest.mock
    from torchtitan.experiments.tpu import workarounds
    from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
    from torchtitan.models.common.attention import GQAttention as Attention
    from torchtitan.experiments.tpu.llama3 import llama3_configs

    config = llama3_configs["testmodel"]

    captured_segment_ids = []

    def mock_splash_sdpa_capture(
        q,
        k,
        v,
        segment_ids=None,
        scale=None,
        is_causal=True,
        local_window_size=None,
        enable_gqa=False,
        **kwargs,
    ):
      captured_segment_ids.append(segment_ids)
      return torch.zeros_like(q)

    with unittest.mock.patch(
        "torchtitan.experiments.tpu.kernels.splash_attention.splash_sdpa",
        side_effect=mock_splash_sdpa_capture,
    ):
      attention_config = config.layers[0].attention
      attention_block = Attention(config=attention_config).to("cpu")

      patch_kwargs = {"block_q": 128}
      workarounds.use_splash_attention_patch(attention_block, patch_kwargs)

      # TT-UNFLATTEN-SHIM: the patch wraps the kernel in `UnflattenShim`, so
      # `inner_attention` is the shim and the kernel is its `.inner`. Remove the
      # kernel's CPU fallback so the patched path actually runs here.
      attention_block.inner_attention.inner.original_module = None

      # Upstream d771abb0 folds the batch axis away: the decoder now passes a
      # packed `[T, D]` token stream and `[T]` positions. The splash kernel is
      # still `[B, S, H, D]`, so the shim has to be active for this seam to
      # work at all -- that is what it is for.
      #
      # Both documents share one sequence (B=1, S=T). Splitting them across two
      # batch rows would make the unflattening itself separate them, and the
      # positions-to-segment-ids conversion under test would have nothing to do.
      num_tokens = 128
      tokens = torch.randn(
          num_tokens,
          config.dim,
          requires_grad=False,
          device="cpu",
      )

      # Create positions for two segments of length 64
      positions = torch.cat([torch.arange(64), torch.arange(64)]).to("cpu")
      kwargs = {"positions": positions}

      with set_unflatten_seq_len(num_tokens):
        out = attention_block(tokens, None, **kwargs)

      self.assertEqual(
          len(captured_segment_ids),
          1,
          "splash_sdpa was not called, patch might have failed.",
      )
      segment_ids = captured_segment_ids[0]

      self.assertIsNotNone(
          segment_ids,
          "segment_ids was None, meaning document packing logic didn't"
          " trigger.",
      )

      # Expected: 64 zeros followed by 64 ones, on the one unflattened row.
      expected_segment_ids = torch.cat([
          torch.zeros(64, dtype=torch.int32),
          torch.ones(64, dtype=torch.int32),
      ]).unsqueeze(0).to("cpu")
      torch.testing.assert_close(segment_ids, expected_segment_ids)

  def test_pallas_loss_linear_preserves_state_dict_keys(self):
    """The loss-kernel wrapper must not rename the output-projection parameter.

    use_output_projection_patch replaces lm_head with PallasLossLinear. If the
    wrapper holds the Linear as a child module the parameter becomes
    `lm_head.original_linear.weight`, which the model state dict adapters cannot
    map -- loading pretrained HF safetensors then dies with
    KeyError: 'lm_head.original_linear.weight'.
    """
    import torch.nn as nn
    from torchtitan.experiments.tpu import workarounds

    class _TinyModel(nn.Module):

      def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(8, 16, bias=False)

    model = _TinyModel()
    before = sorted(model.state_dict().keys())
    original_weight = model.lm_head.weight

    workarounds.use_output_projection_patch(model)

    self.assertIsInstance(model.lm_head, workarounds.PallasLossLinear)
    self.assertEqual(
        sorted(model.state_dict().keys()),
        before,
        "loss-kernel patch renamed the output projection parameter",
    )
    # The Parameter object must be the same one, so FSDP shards it in place and
    # weight tying (lm_head.weight is tok_embeddings.weight) still holds.
    self.assertIs(model.lm_head.weight, original_weight)
    self.assertIsInstance(model.lm_head.weight, nn.Parameter)

    # And the wrapper still does its job: return (x, W^T) for the Pallas loss.
    x = torch.randn(2, 3, 8)
    out, weight_t = model.lm_head(x)
    torch.testing.assert_close(out, x)
    self.assertEqual(tuple(weight_t.shape), (8, 16))

  def test_fused_loss_matches_unfused_cross_entropy(self):
    """The deferred LM-head path preserves loss and gradient numerics."""
    import sys
    import types
    from unittest import mock

    import torch.nn.functional as F
    from torchtitan.components import loss as components_loss
    from torchtitan.experiments.tpu import kernels
    from torchtitan.experiments.tpu.loss import pallas_cross_entropy_loss
    from torchtitan.experiments.tpu.overrides.fused_loss import (
        fused_loss_lm_head,
    )
    from torchtitan.models.common import Linear

    torch.manual_seed(0)
    linear_config = Linear.Config(
        in_features=8,
        out_features=16,
        bias=False,
    )
    unfused_head = linear_config.build()
    fused_head = fused_loss_lm_head(linear_config).build()
    fused_head.load_state_dict(unfused_head.state_dict())

    # Upstream d771abb0 folds the batch axis away: `cross_entropy_loss` takes
    # `pred[T, V]` and `labels[T]`.
    unfused_hidden = torch.randn(8, 8, requires_grad=True)
    fused_hidden = unfused_hidden.detach().clone().requires_grad_(True)
    labels = torch.tensor(
        [1, 4, components_loss.IGNORE_INDEX, 2, 5, 3, 0, 7]
    )

    expected_loss = components_loss.cross_entropy_loss(
        unfused_head(unfused_hidden), labels
    )

    # Kernel parity is covered by test_linear_softmax_cross_entropy_loss. Stub
    # only its TPU-specific dispatch here so this CPU test covers the adapter,
    # weight orientation, sum reduction, and ignored-label behavior together.
    def reference_linear_loss(x, kernel_labels, weights, **unused_kwargs):
      return F.cross_entropy(
          x.float() @ weights.float(),
          kernel_labels,
          reduction="sum",
      )

    kernel_module_name = (
        "torchtitan.experiments.tpu.kernels."
        "linear_softmax_cross_entropy_loss"
    )
    fake_kernel_module = types.ModuleType(kernel_module_name)
    fake_kernel_module.linear_softmax_cross_entropy_loss = reference_linear_loss
    with (
        mock.patch.dict(sys.modules, {kernel_module_name: fake_kernel_module}),
        mock.patch.object(
            kernels,
            "linear_softmax_cross_entropy_loss",
            fake_kernel_module,
            create=True,
        ),
    ):
      actual_loss = pallas_cross_entropy_loss(
          fused_head(fused_hidden),
          labels,
          implementation="xla",
      )

    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-5)

    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(
        fused_hidden.grad,
        unfused_hidden.grad,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        fused_head.weight.grad,
        unfused_head.weight.grad,
        rtol=1e-5,
        atol=1e-5,
    )

  def test_pallas_loss_linear_patches_output_and_output_transform(self):
    """Key preservation must also hold for `output` / `output_transform`."""
    import torch.nn as nn
    from torchtitan.experiments.tpu import workarounds

    for attr in ("output", "output_transform"):
      model = nn.Module()
      setattr(model, attr, nn.Linear(4, 6, bias=False))
      before = sorted(model.state_dict().keys())

      workarounds.use_output_projection_patch(model)

      self.assertIsInstance(getattr(model, attr), workarounds.PallasLossLinear)
      self.assertEqual(
          sorted(model.state_dict().keys()), before, f"{attr} was renamed"
      )

  def test_loss_kernel_patch_keeps_hf_state_dict_adapter_working(self):
    """A patched model must still convert through the HF state dict adapter.

    This is the end-to-end regression: apply_patches runs before
    checkpointer.load(), so if the loss-kernel patch renames the output
    projection, loading pretrained HF safetensors dies in to_hf with
    KeyError: 'lm_head.original_linear.weight'.
    """
    from torchtitan.models.llama3 import llama3_configs
    from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
    from torchtitan.experiments.tpu import workarounds

    # Upstream d771abb0 made `llama3_configs[flavor]` a
    # `(builder, max_context_length)` pair with a required keyword-only
    # `seq_len`.
    build_config, max_context_length = llama3_configs["debugmodel"]
    config = build_config(attn_backend="sdpa", seq_len=max_context_length)
    adapter = Llama3StateDictAdapter(config, None)

    with torch.device("meta"):
      model = config.build()
    workarounds.use_output_projection_patch(model)
    self.assertIsInstance(model.lm_head, workarounds.PallasLossLinear)

    # Must not raise, and must produce the HF-side name.
    hf_state_dict = adapter.to_hf(dict(model.state_dict()))
    self.assertIn("lm_head.weight", hf_state_dict)

  def test_loss_kernel_patch_preserves_weight_initialization(self):
    """A patched model must still have its output projection initialized.

    apply_patches runs before the trainer materializes weights, and
    Module.init_states initializes by recursing over children(). A wrapper that
    adopts the Parameter instead of keeping the Linear as a child drops out of
    that walk, leaving lm_head.weight all zeros for any run that does not load a
    checkpoint -- i.e. every from-scratch pretraining run, which then NaNs on
    the first step. SFT-from-pretrained cannot catch this, because the
    checkpoint load
    overwrites the zeros.
    """
    from torchtitan.models.llama3 import llama3_configs
    from torchtitan.experiments.tpu import workarounds

    # Upstream d771abb0 made `llama3_configs[flavor]` a
    # `(builder, max_context_length)` pair with a required keyword-only
    # `seq_len`.
    build_config, max_context_length = llama3_configs["debugmodel"]
    config = build_config(attn_backend="sdpa", seq_len=max_context_length)

    def build(patch):
      torch.manual_seed(0)
      with torch.device("meta"):
        model = config.build()
      if patch:
        workarounds.use_output_projection_patch(model)
      model.to_empty(device="cpu")
      model.init_states()
      return model.lm_head.weight

    baseline = build(patch=False)
    patched = build(patch=True)

    self.assertTrue(torch.isfinite(patched).all(), "lm_head weight is not finite")
    self.assertGreater(
        patched.float().std().item(), 0.0, "lm_head weight was never initialized"
    )
    torch.testing.assert_close(patched, baseline)

  def _tokamax_patch_fixture(self):
    """A module with one ScaledDotProductInnerAttention, and a populated splash config."""
    from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig  # pylint: disable=g-import-not-at-top
    from torchtitan.models.common.attention import ScaledDotProductInnerAttention  # pylint: disable=g-import-not-at-top

    class _Tiny(torch.nn.Module):

      def __init__(self):
        super().__init__()
        attn = ScaledDotProductInnerAttention.__new__(ScaledDotProductInnerAttention)
        torch.nn.Module.__init__(attn)
        self.attn = attn

    sa = SplashAttentionKernelConfig(
        use_tokamax_splash_attention_kernel=True,
        sa_block_q=1024,
        sa_block_kv=1024,
        sa_block_kv_compute=512,
        sa_block_q_dkv=2048,
        sa_block_kv_dkv=2048,
        sa_block_kv_dkv_compute=1024,
        sa_q_layout="SEQ_MINOR",
        sa_k_layout="SEQ_MINOR",
        sa_v_layout="HEAD_DIM_MINOR",
    )
    sa.use_vmap_bwd = True
    return _Tiny(), sa

  def test_tokamax_patch_forwards_splash_config(self):
    """The splash config must actually reach TokamaxConfig.

    Regression test. SplashAttentionKernelConfig declares its fields sa_-prefixed;
    reading them unprefixed returns None for every one, and TokamaxConfig.update
    skips None, so the patch silently kept tokamax's library defaults and the kernel
    ran at the wrong tiles. Nothing failed - it was just slower, which is exactly why
    this needs a test rather than a runtime error.
    """
    from torchtitan.experiments.tpu.kernels.tokamax_splash_attention import TokamaxConfig  # pylint: disable=g-import-not-at-top

    model, sa = self._tokamax_patch_fixture()
    self.assertEqual(TokamaxConfig.block_q, 512, "precondition: class default")
    sa.sa_num_stacked_q_heads = 16
    sa.use_batch_loop_fwd = True
    sa.use_batch_loop_bwd = True
    for name in ("num_stacked_q_heads", "use_batch_loop_fwd", "use_batch_loop_bwd"):
      self.addCleanup(setattr, TokamaxConfig, name, getattr(TokamaxConfig, name))

    workarounds.use_tokamax_splash_attention_patch(model, sa)

    self.assertEqual(TokamaxConfig.block_q, 1024)
    self.assertEqual(TokamaxConfig.block_kv, 1024)
    self.assertEqual(TokamaxConfig.block_kv_compute, 512)
    self.assertEqual(TokamaxConfig.block_q_dkv, 2048)
    self.assertEqual(TokamaxConfig.block_kv_dkv, 2048)
    self.assertEqual(TokamaxConfig.block_kv_dkv_compute, 1024)
    self.assertEqual(TokamaxConfig.q_layout, "SEQ_MINOR")
    self.assertEqual(TokamaxConfig.k_layout, "SEQ_MINOR")
    self.assertEqual(TokamaxConfig.v_layout, "HEAD_DIM_MINOR")
    self.assertTrue(TokamaxConfig.use_vmap_bwd)
    self.assertEqual(TokamaxConfig.num_stacked_q_heads, 16)
    self.assertTrue(TokamaxConfig.use_batch_loop_fwd)
    self.assertTrue(TokamaxConfig.use_batch_loop_bwd)

  def test_tokamax_patch_raises_when_a_splash_config_resolves_nothing(self):
    """A real splash config whose fields resolve to nothing must fail loudly."""
    model, _ = self._tokamax_patch_fixture()

    class _BrokenSplash:  # exposes sa_ fields, so it IS a splash config
      use_tokamax_splash_attention_kernel = True
      sa_block_q = None
      sa_block_kv = None

    with self.assertRaisesRegex(ValueError, "no splash_attention_kernel values"):
      workarounds.use_tokamax_splash_attention_patch(model, _BrokenSplash())

  def test_tokamax_patch_accepts_a_config_with_no_splash_fields(self):
    """An empty config means "use the kernel defaults" and must not raise.

    Some parity tests pass a bare `class MockConfig: pass`. The guard above has
    to
    distinguish that from a real splash config read incorrectly - the
    discriminator is
    whether sa_-prefixed attributes exist at all. Caught by CI, not by me.
    """
    model, _ = self._tokamax_patch_fixture()

    class _NotASplashConfig:
      pass

    workarounds.use_tokamax_splash_attention_patch(model, _NotASplashConfig())

  def test_tokamax_patch_drops_kwargs_the_kernel_does_not_accept(self):
    """SPLASH_KEYS is the union over splash implementations.

    tokamax's splash_sdpa accepts a strict subset - it has no
    use_document_aware_block_skipping - so forwarding the whole set made every run
    die at first compile with "Unexpected keyword arguments".
    """
    import inspect  # pylint: disable=g-import-not-at-top
    from torchtitan.experiments.tpu.kernels.tokamax_splash_attention import splash_sdpa  # pylint: disable=g-import-not-at-top

    accepted = set(inspect.signature(splash_sdpa).parameters)
    self.assertNotIn("use_document_aware_block_skipping", accepted)
    self.assertIn("segment_ids", accepted)
    # every key the patch may forward must be one the callee accepts
    self.assertContainsSubset(
        workarounds.SPLASH_KEYS & accepted, accepted,
        "the forwarded set must be a subset of the callee signature",
    )

  def test_tokamax_patch_forwards_segment_ids(self):
    """Packed-document segment IDs must reach the tokamax kernel.

    Regression test. The patch filters forward kwargs against tokamax
    splash_sdpa's signature, which used to lack segment_ids, so the IDs that
    GQAttention passes were silently dropped and packed documents attended
    across their boundaries. Also covers the positions-only fallback.
    """
    import functools  # pylint: disable=g-import-not-at-top
    import unittest.mock  # pylint: disable=g-import-not-at-top
    from torchtitan.experiments.tpu.kernels import tokamax_splash_attention  # pylint: disable=g-import-not-at-top

    captured = []
    real_splash_sdpa = tokamax_splash_attention.splash_sdpa

    # functools.wraps keeps the real signature visible to the patch's filter.
    @functools.wraps(real_splash_sdpa)
    def capture(q, k, v, *, segment_ids=None, **kwargs):
      del k, v, kwargs
      captured.append(segment_ids)
      return torch.zeros_like(q)

    # The patch writes the process-global TokamaxConfig; restore it so tests
    # that assert its class defaults are order-independent.
    config_cls = tokamax_splash_attention.TokamaxConfig
    saved = {k: v for k, v in vars(config_cls).items() if not k.startswith("_")}
    self.addCleanup(lambda: [setattr(config_cls, k, v) for k, v in saved.items()])

    model, sa = self._tokamax_patch_fixture()
    with unittest.mock.patch.object(
        tokamax_splash_attention, "splash_sdpa", capture
    ):
      workarounds.use_tokamax_splash_attention_patch(model, sa)

    q = k = v = torch.zeros(2, 128, 2, 8)
    positions = torch.cat([torch.arange(64), torch.arange(64)]).expand(2, -1)
    expected = torch.cat(
        [torch.zeros(64, dtype=torch.int32), torch.ones(64, dtype=torch.int32)]
    ).expand(2, -1)
    # A transposed copy: same values, non-contiguous strides.
    non_contiguous = expected.t().contiguous().t()
    self.assertFalse(non_contiguous.is_contiguous())

    model.attn(q, k, v, segment_ids=non_contiguous, positions=positions)
    model.attn(q, k, v, positions=positions)
    model.attn(q, k, v)

    self.assertLen(captured, 3)
    torch.testing.assert_close(captured[0], expected)
    self.assertTrue(captured[0].is_contiguous())
    torch.testing.assert_close(captured[1], expected)
    self.assertIsNone(captured[2])

  def test_tokamax_patch_warns_on_unsupported_block_skipping(self):
    """Requesting document-aware block skipping must not be silently dropped."""
    from torchtitan.experiments.tpu.kernels import tokamax_splash_attention  # pylint: disable=g-import-not-at-top

    config_cls = tokamax_splash_attention.TokamaxConfig
    saved = {k: v for k, v in vars(config_cls).items() if not k.startswith("_")}
    self.addCleanup(lambda: [setattr(config_cls, k, v) for k, v in saved.items()])

    model, sa = self._tokamax_patch_fixture()
    self.assertTrue(sa.sa_use_document_aware_block_skipping, "precondition")
    with self.assertLogs(workarounds.logger, level="WARNING") as logs:
      workarounds.use_tokamax_splash_attention_patch(model, sa)
    self.assertTrue(
        any("use_document_aware_block_skipping" in m for m in logs.output)
    )

    model, sa = self._tokamax_patch_fixture()
    sa.sa_use_document_aware_block_skipping = False
    with self.assertNoLogs(workarounds.logger, level="WARNING"):
      workarounds.use_tokamax_splash_attention_patch(model, sa)

  @parameterized.parameters("ulysses", "all_gather")
  def test_splash_attention_patch_plugs_into_cp_attention(self, strategy):
    """The Splash patch keeps the CP wrapper and plugs the kernel in as its local attention."""
    from torchtitan.experiments.tpu import workarounds
    from torchtitan.models.common.attention import GQAttention as Attention
    from torchtitan.models.common.attention import ScaledDotProductInnerAttention
    from torchtitan.experiments.tpu.cp_attention import tpu_cp_inner_attention
    from torchtitan.experiments.tpu.kernels.splash_attention import SplashAttention
    from torchtitan.experiments.tpu.qwen3 import qwen3_configs
    import copy

    cp_cls = tpu_cp_inner_attention(ScaledDotProductInnerAttention, strategy)
    config = copy.deepcopy(qwen3_configs["testmodel"])
    config.layers[0].attention.inner_attention = cp_cls.Config()
    attention_block = Attention(config=config.layers[0].attention).to("cpu")

    workarounds.use_splash_attention_patch(attention_block, {"block_q": 128})
    inner = attention_block.inner_attention
    self.assertIsInstance(inner, cp_cls)
    splash = inner._local_attn_override.inner  # Inside the UnflattenShim.
    self.assertIsInstance(splash, SplashAttention)
    self.assertIn("local_attention=", inner.extra_repr())
    self.assertIn("SplashAttention", repr(inner))
    # On CPU or short sequences the kernel falls back to the CP reference path,
    # not to the CP module itself (which would recurse).
    self.assertEqual(splash.original_module, inner.local_forward)

  def test_splash_patch_warns_on_cp_block_skipping(self):
    """KV all-gather CP with stock Splash warns when block skipping is left on."""
    import copy
    import types
    from torchtitan.experiments.tpu import workarounds
    from torchtitan.experiments.tpu.cp_attention import tpu_cp_inner_attention
    from torchtitan.experiments.tpu.qwen3 import qwen3_configs
    from torchtitan.models.common.attention import GQAttention as Attention
    from torchtitan.models.common.attention import ScaledDotProductInnerAttention

    cp_cls = tpu_cp_inner_attention(
        ScaledDotProductInnerAttention, "all_gather"
    )
    config = copy.deepcopy(qwen3_configs["testmodel"])
    config.layers[0].attention.inner_attention = cp_cls.Config()

    attn_block = Attention(config=config.layers[0].attention).to("cpu")
    with self.assertLogs(workarounds.logger, level="WARNING") as logs:
      workarounds.use_splash_attention_patch(
          attn_block,
          types.SimpleNamespace(sa_use_document_aware_block_skipping=True),
      )
    self.assertTrue(
        any("use_document_aware_block_skipping" in m for m in logs.output)
    )

    attn_block = Attention(config=config.layers[0].attention).to("cpu")
    with self.assertNoLogs(workarounds.logger, level="WARNING"):
      workarounds.use_splash_attention_patch(
          attn_block,
          types.SimpleNamespace(sa_use_document_aware_block_skipping=False),
      )

  def test_qwen3_ulysses_cp_recipe_and_validation(self):
    """The Ulysses CP recipe applies ContextParallelTransform; TPU rules validate it."""
    from torchtitan.experiments.tpu.cp_attention import (
        UlyssesCPScaledDotProductInnerAttention,
        revert_tpu_context_parallel,
        validate_tpu_context_parallel,
    )
    from torchtitan.experiments.tpu.qwen3 import config_registry
    from torchtitan.models.common.attention import ScaledDotProductInnerAttention

    config = config_registry.qwen3_debugmodel_ulysses_cp2()
    self.assertEqual(config.parallelism.context_parallel_degree, 2)
    self.assertEqual(config.parallelism.spmd_backend, "spmd_types")
    self.assertIsNone(config.parallelism.context_parallel_load_balancer)
    for layer in config.model_spec.model.layers:
      self.assertIsInstance(
          layer.attention.inner_attention,
          UlyssesCPScaledDotProductInnerAttention.Config,
      )

    # Allow n_kv_heads < cp when cp % n_kv_heads == 0 (e.g. Qwen3.5-397B n_kv_heads=2, cp=4).
    config.parallelism.context_parallel_degree = 4
    for layer in config.model_spec.model.layers:
      layer.attention.n_heads = 32
      layer.attention.n_kv_heads = 2
    validate_tpu_context_parallel(config.model_spec.model, config.parallelism)

    # Reject n_kv_heads indivisible with cp.
    for layer in config.model_spec.model.layers:
      layer.attention.n_kv_heads = 3
    with self.assertRaisesRegex(ValueError, r"n_kv_heads \(3\)"):
      validate_tpu_context_parallel(config.model_spec.model, config.parallelism)
    for layer in config.model_spec.model.layers:
      layer.attention.n_kv_heads = 2

    # Ulysses attends over the full sequence, so it rejects a load balancer.
    config.parallelism.context_parallel_load_balancer = "headtail"
    with self.assertRaisesRegex(ValueError, "context_parallel_load_balancer"):
      validate_tpu_context_parallel(config.model_spec.model, config.parallelism)

    revert_tpu_context_parallel(config.model_spec.model)
    for layer in config.model_spec.model.layers:
      self.assertIs(
          type(layer.attention.inner_attention),
          ScaledDotProductInnerAttention.Config,
      )

  def test_tpu_cp_inner_attention_lookup(self):
    """SDPA and Varlen map to TPU CP attentions; Flex is rejected (not on TPU)."""
    from torchtitan.experiments.tpu.cp_attention import (
        KVAllGatherCPScaledDotProductInnerAttention,
        UlyssesCPVarlenInnerAttention,
        tpu_cp_inner_attention,
    )
    from torchtitan.models.common.attention import (
        FlexInnerAttention,
        ScaledDotProductInnerAttention,
        VarlenInnerAttention,
    )

    self.assertIs(
        tpu_cp_inner_attention(ScaledDotProductInnerAttention, "all_gather"),
        KVAllGatherCPScaledDotProductInnerAttention,
    )
    self.assertIs(
        tpu_cp_inner_attention(VarlenInnerAttention, "ulysses"),
        UlyssesCPVarlenInnerAttention,
    )
    with self.assertRaisesRegex(
        ValueError, "Flex attention does not run on TPU"
    ):
      tpu_cp_inner_attention(FlexInnerAttention, "all_gather")

  def test_splash_attention_auto_block_size_and_fallback(self):
    """auto_block_size picks a dividing block; the fallback forwards CP kwargs."""
    from torchtitan.experiments.tpu.kernels.splash_attention import (
        SplashAttention,
        splash_block_size_for,
    )

    self.assertEqual(splash_block_size_for(512), 512)
    self.assertEqual(splash_block_size_for(1024), 512)
    self.assertEqual(splash_block_size_for(768), 256)
    self.assertEqual(splash_block_size_for(384), 128)
    self.assertEqual(splash_block_size_for(64), 0)

    seen = {}

    def fallback(q, k, v, **kwargs):
      seen.update(kwargs)
      return q + k + v

    attn = SplashAttention(original_module=fallback, auto_block_size=True)
    q = torch.ones(256, 2, 4)  # Folded (T, H, D) token stream.
    cp_rank = torch.tensor([1], dtype=torch.int32)
    out = attn(q, q, q, cp_rank=cp_rank)
    torch.testing.assert_close(out, q * 3)
    self.assertIs(seen["cp_rank"], cp_rank)
    self.assertTrue(seen["is_causal"])

    # The Pallas kernel has no LSE epilogue: out_transform is rejected rather
    # than silently dropped (meta tensors take the kernel path, not fallback).
    meta_q = torch.empty(256, 2, 4, device="meta")
    with self.assertRaisesRegex(NotImplementedError, "out_transform"):
      attn(meta_q, meta_q, meta_q, out_transform=lambda out, lse: out)

  def test_cp_without_cp_recipe_is_rejected(self):
    """As on GPU, cp > 1 on a recipe without a CP transform fails core validation."""
    from torchtitan.experiments.tpu.qwen3 import config_registry

    config = config_registry.qwen3_debugmodel()
    config.parallelism.context_parallel_degree = 2
    config.parallelism.spmd_backend = "spmd_types"
    with self.assertRaisesRegex(ValueError, "CPInnerAttention"):
      config.__post_init__()

  def test_wait_helper_waits_async_tensors_once(self):
    """`_wait` leaves traced (already waited) funcol results alone and waits eager ones once."""
    from unittest import mock
    import torch.distributed._functional_collectives as funcol
    from torchtitan.experiments.tpu import cp_attention

    x = torch.ones(4, 2)
    with mock.patch.object(
        funcol, "wait_tensor", side_effect=AssertionError("double wait")
    ):
      self.assertIs(cp_attention._wait(x), x)

    act = mock.create_autospec(funcol.AsyncCollectiveTensor, instance=True)
    act.wait.return_value = x
    self.assertIs(cp_attention._wait(act), x)
    act.wait.assert_called_once_with()

  def test_tpu_selective_ac_recomputes_saved_collectives(self):
    """TPUSelectiveAC is core SelectiveAC minus the collectives core saves."""
    from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
    from torchtitan.experiments.tpu.activation_checkpoint import (
        TPUSelectiveAC,
        tpu_selective_ac_config,
    )

    c10d = torch.ops._c10d_functional
    collectives = {
        c10d.all_to_all_single.default,
        c10d.reduce_scatter_tensor.default,
    }
    core_ops = SelectiveAC.Config().build().get_save_ops()
    self.assertLessEqual(collectives, core_ops)
    self.assertEqual(
        TPUSelectiveAC.Config().build().get_save_ops(), core_ops - collectives
    )

    # CP parallelize functions switch plain SelectiveAC (keeping its fields)
    # and leave every other AC config alone.
    selective = SelectiveAC.Config(preserve_rng_state=True)
    switched = tpu_selective_ac_config(selective)
    self.assertIs(type(switched), TPUSelectiveAC.Config)
    self.assertTrue(switched.preserve_rng_state)
    self.assertIsInstance(switched.build(), TPUSelectiveAC)
    for other in (None, FullAC.Config(), switched):
      self.assertIs(tpu_selective_ac_config(other), other)

  def test_ulysses_cp_repeats_kv_heads(self):
    """UlyssesCPScaledDotProductInnerAttention repeats KV heads when cp > n_kv_heads."""
    from types import SimpleNamespace
    from unittest import mock
    import spmd_types as spmd
    from torchtitan.experiments.tpu import cp_attention
    from torchtitan.experiments.tpu.cp_attention import (
        UlyssesCPScaledDotProductInnerAttention,
    )

    q = torch.randn(4, 32, 8)
    k = torch.randn(4, 2, 8)
    v = torch.randn(4, 2, 8)
    captured = {}

    def fake_redistribute(x, group, *, src, dst):
      if src == spmd.S(0) and dst == spmd.S(1):
        self.assertEqual(0, x.shape[1] % group.size())
        return x.reshape(
            x.shape[0] * group.size(),
            x.shape[1] // group.size(),
            *x.shape[2:],
        )
      if src == spmd.S(1) and dst == spmd.S(0):
        return x.reshape(
            x.shape[0] // group.size(),
            x.shape[1] * group.size(),
            *x.shape[2:],
        )
      raise AssertionError(f"unexpected redistribute: {src} -> {dst}")

    def fake_local_attn(q_in, k_in, v_in, **kw):
      captured["q_shape"] = q_in.shape
      captured["k_shape"] = k_in.shape
      captured["v_shape"] = v_in.shape
      captured["enable_gqa"] = kw.get("enable_gqa")
      return q_in

    kernel = UlyssesCPScaledDotProductInnerAttention(
        UlyssesCPScaledDotProductInnerAttention.Config()
    )
    kernel.set_local_attention(fake_local_attn)
    with (
        mock.patch.object(
            cp_attention,
            "spmd_mesh_group",
            return_value=SimpleNamespace(size=lambda: 4),
        ),
        mock.patch.object(spmd, "redistribute", side_effect=fake_redistribute),
    ):
      out = kernel.forward(q, k, v, enable_gqa=True)

    self.assertEqual((16, 8, 8), captured["q_shape"])
    self.assertEqual((16, 1, 8), captured["k_shape"])
    self.assertEqual((16, 1, 8), captured["v_shape"])
    self.assertTrue(captured["enable_gqa"])
    self.assertEqual((4, 32, 8), out.shape)

  def test_all_gather_cp_config_and_numerical_parity(self):
    """KVAllGatherCPInnerAttention configures headtail load balancer and matches CP=1."""
    from types import SimpleNamespace
    from unittest import mock
    import spmd_types as spmd
    from torchtitan.experiments.tpu import cp_attention
    from torchtitan.experiments.tpu.cp_attention import (
        KVAllGatherCPScaledDotProductInnerAttention,
        reorder_sequence,
        revert_tpu_context_parallel,
        validate_tpu_context_parallel,
    )
    from torchtitan.experiments.tpu.headtail import headtail_positions
    from torchtitan.experiments.tpu.qwen3 import config_registry
    from torchtitan.models.common.attention import ScaledDotProductInnerAttention

    # 1. Verify the all_gather CP recipe and reverting it for cp=1.
    config = config_registry.qwen3_debugmodel_allgather_cp2()
    self.assertEqual(config.parallelism.spmd_backend, "spmd_types")
    self.assertEqual(
        config.parallelism.context_parallel_load_balancer, "headtail"
    )
    for layer in config.model_spec.model.layers:
      self.assertIsInstance(
          layer.attention.inner_attention,
          KVAllGatherCPScaledDotProductInnerAttention.Config,
      )
      # The debug model runs without Splash, so the recipe opts in to the
      # reference attention.
      self.assertTrue(layer.attention.inner_attention.allow_reference_attention)
    # KV All-Gather does not shard heads across CP, so arbitrary n_kv_heads works.
    config.parallelism.context_parallel_degree = 4
    for layer in config.model_spec.model.layers:
      layer.attention.n_heads = 32
      layer.attention.n_kv_heads = 3
    validate_tpu_context_parallel(config.model_spec.model, config.parallelism)

    revert_tpu_context_parallel(config.model_spec.model)
    for layer in config.model_spec.model.layers:
      self.assertIs(
          type(layer.attention.inner_attention),
          ScaledDotProductInnerAttention.Config,
      )

    # 2. Verify reorder_sequence round-trip and head-tail pairing.
    x = torch.arange(16).reshape(1, 16, 1)
    reord = reorder_sequence(x, cp_size=4, seq_dim=1, to_contiguous=False)
    expected_reord = torch.tensor(
        [0, 1, 14, 15, 2, 3, 12, 13, 4, 5, 10, 11, 6, 7, 8, 9]
    ).reshape(1, 16, 1)
    torch.testing.assert_close(reord, expected_reord)
    restored = reorder_sequence(reord, cp_size=4, seq_dim=1, to_contiguous=True)
    torch.testing.assert_close(restored, x)
    # The Splash masks put their Q rows in the same order.
    self.assertEqual(
        headtail_positions(16, 4).tolist(), expected_reord.flatten().tolist()
    )

    # 3. Verify forward + backward numerical parity against CP=1 (with packed documents).
    torch.manual_seed(42)
    cp = 4
    seq_len = 32
    n_heads, n_kv_heads, head_dim = 8, 2, 16
    q_full = torch.randn(seq_len, n_heads, head_dim, requires_grad=True)
    k_full = torch.randn(seq_len, n_kv_heads, head_dim, requires_grad=True)
    v_full = torch.randn(seq_len, n_kv_heads, head_dim, requires_grad=True)
    do_full = torch.randn(seq_len, n_heads, head_dim)
    positions_full = torch.cat([torch.arange(16), torch.arange(16)])

    ref_attn = KVAllGatherCPScaledDotProductInnerAttention(
        KVAllGatherCPScaledDotProductInnerAttention.Config()
    )
    with mock.patch.object(
        cp_attention,
        "spmd_mesh_group",
        return_value=SimpleNamespace(size=lambda: 1, rank=lambda: 0),
    ), mock.patch.object(
        spmd, "redistribute", side_effect=lambda t, *a, **kw: t
    ):
      out_ref = ref_attn(
          q_full,
          k_full,
          v_full,
          enable_gqa=True,
          positions=positions_full,
      )
      out_ref.backward(do_full)

    q_reord = reorder_sequence(
        q_full.detach(), cp_size=cp, seq_dim=0, to_contiguous=False
    )
    k_reord = reorder_sequence(
        k_full.detach(), cp_size=cp, seq_dim=0, to_contiguous=False
    )
    v_reord = reorder_sequence(
        v_full.detach(), cp_size=cp, seq_dim=0, to_contiguous=False
    )
    do_reord = reorder_sequence(
        do_full, cp_size=cp, seq_dim=0, to_contiguous=False
    )
    pos_reord = reorder_sequence(
        positions_full, cp_size=cp, seq_dim=0, to_contiguous=False
    )

    out_shards = []
    dq_shards = []
    dk_total = torch.zeros_like(k_reord)
    dv_total = torch.zeros_like(v_reord)
    shard_len = seq_len // cp

    for r in range(cp):
      s_slice = slice(r * shard_len, (r + 1) * shard_len)
      q_r = q_reord[s_slice].clone().requires_grad_(True)
      k_r = k_reord[s_slice].clone().requires_grad_(True)
      v_r = v_reord[s_slice].clone().requires_grad_(True)
      k_full_r = k_reord.clone().requires_grad_(True)
      v_full_r = v_reord.clone().requires_grad_(True)
      pos_r = pos_reord[s_slice]

      def fake_redistribute(
          x_in,
          group,
          *,
          src,
          dst,
          k_r=k_r,
          v_r=v_r,
          k_full_r=k_full_r,
          v_full_r=v_full_r,
          pos_r=pos_r,
          **kwargs,
      ):
        del group, src, dst, kwargs
        if x_in is k_r:
          return k_full_r
        if x_in is v_r:
          return v_full_r
        if x_in is pos_r:
          return pos_reord
        raise AssertionError("Unexpected tensor in redistribute")

      attn_r = KVAllGatherCPScaledDotProductInnerAttention(
          KVAllGatherCPScaledDotProductInnerAttention.Config()
      )
      cp_attention._CP_RANK_CACHE.clear()
      with mock.patch.object(
          cp_attention,
          "spmd_mesh_group",
          return_value=SimpleNamespace(size=lambda: cp, rank=lambda r=r: r),
      ), mock.patch.object(spmd, "redistribute", side_effect=fake_redistribute):
        out_r = attn_r(
            q_r,
            k_r,
            v_r,
            enable_gqa=True,
            positions=pos_r,
        )
        out_r.backward(do_reord[s_slice])
      out_shards.append(out_r.detach())
      dq_shards.append(q_r.grad.detach())
      dk_total += k_full_r.grad.detach()
      dv_total += v_full_r.grad.detach()

    out_cp = reorder_sequence(
        torch.cat(out_shards, dim=0),
        cp_size=cp,
        seq_dim=0,
        to_contiguous=True,
    )
    dq_cp = reorder_sequence(
        torch.cat(dq_shards, dim=0),
        cp_size=cp,
        seq_dim=0,
        to_contiguous=True,
    )
    dk_cp = reorder_sequence(
        dk_total, cp_size=cp, seq_dim=0, to_contiguous=True
    )
    dv_cp = reorder_sequence(
        dv_total, cp_size=cp, seq_dim=0, to_contiguous=True
    )
    torch.testing.assert_close(out_cp, out_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dq_cp, q_full.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dk_cp, k_full.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dv_cp, v_full.grad, rtol=1e-5, atol=1e-5)

  def test_all_gather_cp_needs_a_kernel_off_cpu(self):
    """Off CPU, KV all-gather CP runs its reference attention only on opt-in."""
    from unittest import mock
    from torchtitan.experiments.tpu import cp_attention
    from torchtitan.experiments.tpu.cp_attention import (
        KVAllGatherCPScaledDotProductInnerAttention,
    )

    config_cls = KVAllGatherCPScaledDotProductInnerAttention.Config
    # Past the kernel check, forward stops at the missing CP mesh.
    self.enter_context(
        mock.patch.object(cp_attention, "spmd_mesh_group", return_value=None)
    )
    q = torch.empty(8, 2, 4, device="meta")

    attn = KVAllGatherCPScaledDotProductInnerAttention(config_cls())
    with self.assertRaisesRegex(RuntimeError, "no attention kernel"):
      attn(q, q, q)
    attn.set_local_attention(lambda *args, **kwargs: None)
    with self.assertRaisesRegex(RuntimeError, "active CP mesh axis"):
      attn(q, q, q)

    attn = KVAllGatherCPScaledDotProductInnerAttention(
        config_cls(allow_reference_attention=True)
    )
    with self.assertRaisesRegex(RuntimeError, "active CP mesh axis"):
      attn(q, q, q)

  def test_all_gather_cp_reference_attention_limits(self):
    """The reference attention needs cp_rank for a CP shard and has no windows."""
    from torchtitan.experiments.tpu.cp_attention import (
        KVAllGatherCPScaledDotProductInnerAttention,
        KVAllGatherCPVarlenInnerAttention,
    )

    q = torch.randn(4, 2, 8)
    kv = torch.randn(8, 2, 8)  # Q is one of cp=2 shards of the K/V sequence.
    sdpa = KVAllGatherCPScaledDotProductInnerAttention(
        KVAllGatherCPScaledDotProductInnerAttention.Config()
    )
    with self.assertRaisesRegex(ValueError, "cp_rank is required"):
      sdpa.local_forward(q, kv, kv)
    out = sdpa.local_forward(
        q, kv, kv, cp_rank=torch.tensor([1], dtype=torch.int32)
    )
    self.assertEqual(q.shape, out.shape)

    # SDPA models with a sliding window set it on the inner attention module.
    sdpa.sliding_window_size = 3
    with self.assertRaisesRegex(NotImplementedError, "sliding windows"):
      sdpa.local_forward(q, q, q)
    varlen = KVAllGatherCPVarlenInnerAttention(
        KVAllGatherCPVarlenInnerAttention.Config(window_size=(3, 0))
    )
    with self.assertRaisesRegex(NotImplementedError, "sliding windows"):
      varlen.local_forward(q, q, q)

  def test_all_gather_cp_requires_headtail_inputs(self):
    """cp_shard rejects any load balancer but head-tail, which forward assumes."""
    from types import SimpleNamespace
    from torchtitan.experiments.tpu.cp_attention import (
        KVAllGatherCPInnerAttention,
    )

    cp_mesh = SimpleNamespace(
        device_type="cpu", get_local_rank=lambda: 0, size=lambda dim: 2
    )
    with self.assertRaisesRegex(ValueError, "must be 'headtail'"):
      KVAllGatherCPInnerAttention.cp_shard(
          {"input": torch.arange(8)}, None, cp_mesh, None, None
      )

  def test_cp_with_tp_or_ep_on_fsdp2_needs_spmd_types_fsdp(self):
    """CP + TP/EP with FSDP2 on torch without spmd_types FSDP fails clearly."""
    from types import SimpleNamespace
    from unittest import mock
    from torchtitan.experiments.tpu import cp_attention
    from torchtitan.experiments.tpu.qwen3 import config_registry

    config = config_registry.qwen3_debugmodel_allgather_cp2()
    model, parallelism = config.model_spec.model, config.parallelism
    for layer in model.layers:
      layer.attention.n_heads = 8
      layer.attention.n_kv_heads = 2
    without = mock.patch.object(cp_attention, "FSDPParam", SimpleNamespace())
    with_it = mock.patch.object(
        cp_attention,
        "FSDPParam",
        SimpleNamespace(_resolve_spmd_types_for_storage=None),
    )

    for tp, ep in ((2, 1), (1, 2)):
      parallelism.tensor_parallel_degree = tp
      parallelism.expert_parallel_degree = ep
      parallelism.use_simple_fsdp = False
      with without, self.assertRaisesRegex(NotImplementedError, "FSDP2"):
        cp_attention.validate_tpu_context_parallel(model, parallelism)
      with with_it:
        cp_attention.validate_tpu_context_parallel(model, parallelism)
      # SimpleFSDP does not go through resolve_fsdp_mesh.
      parallelism.use_simple_fsdp = True
      with without:
        cp_attention.validate_tpu_context_parallel(model, parallelism)

    # CP alone is covered by install_tpu_cp_patches.
    parallelism.tensor_parallel_degree = parallelism.expert_parallel_degree = 1
    parallelism.use_simple_fsdp = False
    with without:
      cp_attention.validate_tpu_context_parallel(model, parallelism)


class RaiseDynamoRecompileLimitsTest(parameterized.TestCase):
  """The recompile limit covers train + validation compiles of every layer."""

  def setUp(self):
    super().setUp()
    import torch._dynamo.config as dynamo_config  # pylint: disable=g-import-not-at-top

    saved = (dynamo_config.recompile_limit,
             dynamo_config.accumulated_recompile_limit)

    def restore():
      (dynamo_config.recompile_limit,
       dynamo_config.accumulated_recompile_limit) = saved

    self.addCleanup(restore)
    dynamo_config.recompile_limit = 8
    dynamo_config.accumulated_recompile_limit = 256
    self.dynamo_config = dynamo_config

  def test_num_layers_covers_train_and_validation_compiles(self):
    workarounds.raise_dynamo_recompile_limits_for_num_layers(42)
    self.assertEqual(self.dynamo_config.recompile_limit, 2 * 42 + 16)
    self.assertEqual(self.dynamo_config.accumulated_recompile_limit,
                     8 * (2 * 42 + 16))

  def test_config_without_layers_keeps_floor(self):
    import types  # pylint: disable=g-import-not-at-top

    workarounds.raise_dynamo_recompile_limits(types.SimpleNamespace())
    self.assertEqual(self.dynamo_config.recompile_limit, 64)


if __name__ == "__main__":
  absltest.main()
