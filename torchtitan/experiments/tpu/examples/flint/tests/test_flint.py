"""Unit and numerical tests for Flint in torchtitan.experiments.tpu.examples.flint."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest as BaseDeviceTest
from torchtitan.experiments.tpu.examples.flint import (
    build_flavor_config,
    flint_layer_types,
)
from torchtitan.experiments.tpu.examples.flint.config_registry import (
    flint_40b,
    flint_debugmodel,
    flint_moe_testmodel,
)
from torchtitan.experiments.tpu.examples.flint.model import FlintSinkAttention
from torchtitan.experiments.tpu.examples.flint.state_dict_adapter import (
    FlintStateDictAdapter,
)
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch
from torchtitan.models.utils import get_nparams_and_active_nparams
import transformers

# layer_types of the released RecursiveSI/flint config.json.
_CHECKPOINT_FULL_LAYERS = (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48)


def _reference_attention(q, k, v, sinks, window, segment_ids, scale):
  """Per-query loop: softmax over the allowed keys plus the sink logit."""
  b, s, h, d = q.shape
  rep = h // k.shape[2]
  k = k.repeat_interleave(rep, dim=2)
  v = v.repeat_interleave(rep, dim=2)
  out = torch.zeros(b, s, h, d, dtype=torch.float64)
  for bi in range(b):
    for qi in range(s):
      keys = [
          ki
          for ki in range(qi + 1)
          if (window is None or qi - ki < window)
          and (segment_ids is None or segment_ids[bi, ki] == segment_ids[bi, qi])
      ]
      logits = torch.einsum(
          "hd,khd->hk", q[bi, qi].double(), k[bi, keys].double()
      ) * scale
      if sinks is not None:
        logits = torch.cat([logits, sinks.double()[:, None]], dim=1)
      p = torch.softmax(logits, dim=1)[:, : len(keys)]
      out[bi, qi] = torch.einsum("hk,khd->hd", p, v[bi, keys].double())
  return out


class TestFlintSinkAttention(parameterized.TestCase):
  """Mask and sink semantics of the reference attention, with controls."""

  def _inputs(self):
    g = torch.Generator().manual_seed(0)
    b, s, h, kv, d = 2, 24, 4, 2, 8
    q = torch.randn(b, s, h, d, generator=g)
    k = torch.randn(b, s, kv, d, generator=g)
    v = torch.randn(b, s, kv, d, generator=g)
    sinks = torch.randn(h, generator=g) * 2.0
    # Row 0: documents of 10 and 14 tokens; row 1: one document.
    positions = torch.cat([torch.arange(10), torch.arange(14), torch.arange(24)])
    return q, k, v, sinks, positions

  @parameterized.named_parameters(
      ("full", None), ("window_5", 5), ("window_1", 1)
  )
  def test_matches_reference(self, window):
    q, k, v, sinks, positions = self._inputs()
    b, s = q.shape[:2]
    attn = FlintSinkAttention(FlintSinkAttention.Config(window_size=window))
    with set_unflatten_seq_len(s):
      out = attn(
          q.reshape(b * s, *q.shape[2:]),
          k.reshape(b * s, *k.shape[2:]),
          v.reshape(b * s, *v.shape[2:]),
          sinks=sinks,
          positions=positions,
          scale=0.3,
      ).reshape(b, s, *q.shape[2:])
    seg = (torch.cumsum((positions == 0).int(), 0) - 1).reshape(b, s)
    ref = _reference_attention(q, k, v, sinks, window, seg, 0.3)
    torch.testing.assert_close(out.double(), ref, atol=1e-5, rtol=1e-5)

  def test_controls(self):
    """Sinks, the window and the document mask each move the output."""
    q, k, v, sinks, positions = self._inputs()
    b, s = q.shape[:2]

    def run(window, sinks, positions):
      attn = FlintSinkAttention(FlintSinkAttention.Config(window_size=window))
      return attn(q, k, v, sinks=sinks, positions=positions, scale=0.3)

    base = run(5, sinks, positions)
    self.assertGreater(float((base - run(5, None, positions)).abs().max()), 1e-1)
    self.assertGreater(float((base - run(None, sinks, positions)).abs().max()), 1e-1)
    unpacked = torch.arange(s).repeat(b)
    self.assertGreater(float((base - run(5, sinks, unpacked)).abs().max()), 1e-1)

  def test_window_conventions(self):
    attn = FlintSinkAttention(FlintSinkAttention.Config(window_size=513))
    self.assertEqual(attn.window_size, 513)
    self.assertEqual(attn.sliding_window_size, 512)  # splash LocalMask: keys back


class TestFlintSplashKernel(BaseDeviceTest):
  """The sbd Splash path vs the float32 reference on a packed batch (TPU)."""

  @parameterized.named_parameters(("sliding", 129), ("full", None))
  def test_kernel_matches_reference(self, window):
    if self.accelerator_device.type != "tpu":
      self.skipTest("Pallas kernel: TPU only")
    g = torch.Generator().manual_seed(1)
    b, s, h, kv, d = 2, 1024, 8, 2, 128
    q, k, v = (
        torch.randn(b, s, n, d, generator=g).to(torch.bfloat16)
        for n in (h, kv, kv)
    )
    sinks = torch.randn(h, generator=g) * 2.0
    cot = torch.randn(b, s, h, d, generator=g)
    # Three documents per row, boundaries inside the window.
    positions = torch.cat([
        torch.arange(300), torch.arange(500), torch.arange(224),
        torch.arange(1024),
    ])

    def run(kernel, use_sinks):
      attn = FlintSinkAttention(
          FlintSinkAttention.Config(window_size=window, use_splash_kernel=kernel)
      )
      dev = self.accelerator_device
      qd, kd, vd, sd = (
          t.clone().to(dev).requires_grad_(True) for t in (q, k, v, sinks)
      )
      out = attn(
          qd, kd, vd, sinks=sd if use_sinks else None,
          positions=positions.to(dev), scale=d**-0.5,
      )
      (out.float() * cot.to(dev)).sum().backward()
      grads = [t.grad.float().cpu() for t in (qd, kd, vd)]
      grads.append(sd.grad.float().cpu() if use_sinks else None)
      return out.float().cpu(), grads

    ref, ref_g = run(False, True)
    out, out_g = run(True, True)

    def rel(a, b):
      return float((a - b).norm() / b.norm())

    self.assertLess(rel(out, ref), 1e-2)
    for name, a, r in zip(("dq", "dk", "dv", "dsinks"), out_g, ref_g):
      self.assertLess(rel(a, r), 2.5e-2, name)
    # Control: without the sinks the kernel output moves well past the bound.
    no_sink, _ = run(True, False)
    self.assertGreater(rel(no_sink, ref), 1e-1)


class TestFlintFlavors(absltest.TestCase):

  def test_40b_matches_checkpoint(self):
    config = build_flavor_config("40B")
    self.assertLen(config.layers, 49)
    self.assertEqual(
        flint_layer_types(49),
        [
            "full_attention" if i in _CHECKPOINT_FULL_LAYERS else "sliding_attention"
            for i in range(49)
        ],
    )
    for i, layer in enumerate(config.layers):
      attention = layer.attention
      full = i in _CHECKPOINT_FULL_LAYERS
      self.assertEqual(attention.inner_attention.window_size, None if full else 513)
      self.assertEqual(attention.rope is None, full)  # NoPE on full layers
      self.assertEqual((attention.n_heads, attention.n_kv_heads, attention.head_dim), (32, 4, 128))
      self.assertEqual(layer.moe is None, i in (0, 1))
      if layer.moe is not None:
        self.assertEqual(layer.moe.num_experts, 256)
        self.assertEqual(layer.moe.router.top_k, 6)
        self.assertEqual(layer.moe.router.score_func, "sigmoid")
        self.assertTrue(layer.moe.router.route_norm)
        self.assertEqual(layer.moe.shared_experts.w1.out_features, 1024)
        self.assertEqual(layer.moe.routed_experts.inner_experts.hidden_dim, 512)
      else:
        self.assertEqual(layer.feed_forward.w1.out_features, 6144)
    with torch.device("meta"):
      model = config.build()
    nparams, active = get_nparams_and_active_nparams(model)
    self.assertEqual(nparams, 39_785_287_456)  # HF: 39.8B total
    self.assertEqual(active, 2_515_488_544)  # 2.5B active, embedding excluded

  def test_recipes_build(self):
    for recipe in (flint_debugmodel, flint_moe_testmodel, flint_40b):
      config = recipe()
      experts = list(config.model_spec.model.traverse(TpuGroupedExperts.Config))
      self.assertNotEmpty(experts, recipe.__name__)
    cfg = flint_40b()
    self.assertEqual(cfg.training.num_tokens_per_train_step, 512 * 2048)


class TestFlintStateDictAdapter(absltest.TestCase):

  def test_roundtrip_and_checkpoint_layout(self):
    config = build_flavor_config("testmodel")
    model = config.build()
    model.init_states()
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    adapter = FlintStateDictAdapter(config, hf_assets_path=None)
    hf_sd = adapter.to_hf(sd)
    # The released checkpoint's layout: per-expert weights, softmax_offset.
    self.assertIn("model.layers.1.self_attn.softmax_offset", hf_sd)
    self.assertIn("model.layers.1.mlp.experts.7.gate_proj.weight", hf_sd)
    self.assertIn("model.layers.1.mlp.shared_expert.down_proj.weight", hf_sd)
    self.assertIn("model.layers.0.mlp.gate_proj.weight", hf_sd)
    back = adapter.from_hf(hf_sd)
    params = dict(model.named_parameters())
    self.assertEqual(set(back), set(params))
    for key, value in back.items():
      torch.testing.assert_close(value, sd[key], atol=0, rtol=0, msg=key)

  def test_fused_hf_layout(self):
    """transformers >= 5.21 in-memory names: fused experts and ``sinks``."""
    config = build_flavor_config("testmodel")
    model = config.build()
    model.init_states()
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    prefix = "layers.1.moe.routed_experts.inner_experts"
    fused = {
        "model.layers.1.mlp.experts.gate_up_proj": torch.cat(
            [sd[f"{prefix}.w1_EFD"], sd[f"{prefix}.w3_EFD"]], dim=1
        ),
        "model.layers.1.mlp.experts.down_proj": sd[f"{prefix}.w2_EDF"],
        "model.layers.1.self_attn.sinks": sd["layers.1.attention.sinks"],
    }
    back = FlintStateDictAdapter(config, hf_assets_path=None).from_hf(fused)
    for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
      torch.testing.assert_close(back[f"{prefix}.{name}"], sd[f"{prefix}.{name}"])
    torch.testing.assert_close(
        back["layers.1.attention.sinks"], sd["layers.1.attention.sinks"]
    )


class TestFlintHFReferenceParity(BaseDeviceTest):
  """One forward + backward of TorchTitan Flint vs HF FlintForCausalLM."""

  def setUp(self):
    super().setUp()
    if not hasattr(transformers, "FlintForCausalLM"):
      self.skipTest("transformers.FlintForCausalLM needs transformers >= 5.21")

  @staticmethod
  def _build_hf_reference(sliding_window: int):
    hf_config = transformers.FlintConfig(
        vocab_size=2048,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=128,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        mlp_only_layers=[0],
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
        sliding_window=sliding_window,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=False,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
    )
    hf_config._attn_implementation = "eager"
    hf_model = transformers.FlintForCausalLM(hf_config).float().eval()
    with torch.no_grad():
      for name, param in hf_model.named_parameters():
        # Unit-initialized norms would hide a dropped scale; sinks near the
        # logits' scale take a sizeable share of the softmax.
        if "norm" in name:
          param.copy_(1.0 + 0.1 * torch.randn_like(param))
        elif name.endswith("sinks"):
          param.copy_(torch.randn_like(param))
    return hf_model

  @parameterized.named_parameters(
      dict(testcase_name="grouped_experts", tpu_experts=False),
      dict(testcase_name="tpu_grouped_experts", tpu_experts=True),
  )
  def test_one_step_matches_hf(self, tpu_experts: bool):
    use_cpu_safe_histc_patch()
    torch.manual_seed(42)
    sliding_window = 8  # < seq_len, so the sliding layers mask
    batch_size, seq_len = 2, 16

    hf_model = self._build_hf_reference(sliding_window)
    tt_config = build_flavor_config("testmodel")
    if tpu_experts:
      tt_config = TpuGroupedExpertsConverter.Config().build().convert(tt_config)
      for _, experts, _, _ in tt_config.traverse(TpuGroupedExperts.Config):
        experts.use_pallas_grouped_mm = False  # runs on CPU too
    tt_model = tt_config.build().to(self.accelerator_device)
    tt_model.init_states()

    adapter = FlintStateDictAdapter(tt_config, hf_assets_path=None)
    converted = adapter.from_hf(
        {k: v.detach().clone() for k, v in hf_model.state_dict().items()}
    )
    missing, unexpected = tt_model.load_state_dict(converted, strict=False)
    buffers = {name for name, _ in tt_model.named_buffers()}
    self.assertEmpty(set(missing) - buffers)
    self.assertEmpty(unexpected)

    tokens_cpu = torch.randint(0, 2048, (batch_size, seq_len))
    labels_cpu = torch.randint(0, 2048, (batch_size, seq_len))
    positions = torch.arange(seq_len, device=self.accelerator_device).repeat(batch_size)
    num_tokens = labels_cpu.numel()

    with set_unflatten_seq_len(seq_len):
      tt_logits = tt_model(
          tokens_cpu.view(-1).to(self.accelerator_device), positions=positions
      ).view(batch_size, seq_len, -1)
    tt_loss = cross_entropy_loss(tt_logits, labels_cpu.to(self.accelerator_device)) / num_tokens
    tt_loss.backward()

    hf_logits = hf_model(input_ids=tokens_cpu).logits
    hf_loss = cross_entropy_loss(hf_logits, labels_cpu) / num_tokens
    hf_loss.backward()

    labels = dict(
        test_label=f"TorchTitan Flint ({self.accelerator_device.type})",
        ref_label="HF FlintForCausalLM (CPU)",
    )
    test_utils.check_equivalence(
        tt_logits.detach().float().cpu(), hf_logits.detach().float(),
        atol=1e-1, rtol=2e-2, check_name="Logits", **labels,
    )
    test_utils.check_equivalence(
        tt_loss.detach().float().cpu(), hf_loss.detach().float(),
        atol=5e-3, rtol=2e-3, check_name="Loss", **labels,
    )
    tt_grads = {
        n: p.grad.detach().float().cpu()
        for n, p in tt_model.named_parameters() if p.grad is not None
    }
    hf_grads = adapter.from_hf({
        n: p.grad.detach().float().cpu()
        for n, p in hf_model.named_parameters() if p.grad is not None
    })
    self.assertEqual(set(tt_grads), set(hf_grads))
    for name in sorted(tt_grads):
      test_utils.check_equivalence(
          tt_grads[name], hf_grads[name], atol=8e-2, rtol=2e-2,
          check_name=f"Gradient ({name})", **labels,
      )


if __name__ == "__main__":
  absltest.main()
