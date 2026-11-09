"""DeepSeek-V3 on TPU: HF reference parity, weight mapping, flavors, masking.

The reference is HF transformers' ``DeepseekV3ForCausalLM`` built in-process
from a tiny config that covers what the 671B model uses: one dense and one MoE
layer, MLA with the q LoRA path and unequal q/k (48) and v (24) head dims, YaRN
RoPE (interleaved pairs), sigmoid routing with group-limited top-k, the expert
score-correction bias, normalized and scaled routing weights and a shared
expert. MTP is not covered: HF transformers does not implement it.

Weights reach the TorchTitan model the way ``--checkpoint.initial_load_in_hf``
does: ``save_pretrained`` writes the on-disk (per-expert) safetensors layout,
and upstream's ``DeepSeekV3StateDictAdapter.from_hf`` maps it.
"""

import os
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
import torchtitan.experiments.tpu.deepseek_v3 as deepseek_v3_tpu
from torchtitan.experiments.tpu.masked_sdpa import SlidingWindowScaledDotProductInnerAttention
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch
import torchtitan.models.common as common_models
from torchtitan.models.common.attention import ScaledDotProductInnerAttention
import torchtitan.models.deepseek_v3 as deepseek_v3_models
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter

try:
  import transformers  # pylint: disable=g-import-not-at-top
except ImportError:  # pragma: no cover
  transformers = None

# Tiny geometry shared by the HF reference and the TorchTitan config.
_TINY = dict(
    vocab_size=2048,
    dim=128,
    n_layers=2,
    n_dense_layers=1,
    n_heads=8,
    q_lora_rank=48,
    kv_lora_rank=32,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=24,
    dense_hidden_dim=256,
    moe_hidden_dim=64,
    num_experts=8,
    num_shared_experts=1,
    top_k=3,
    num_expert_groups=4,
    num_limited_groups=2,
    route_scale=2.5,
    rope_theta=10000.0,
    rope_factor=40.0,
    original_seq_len=4096,
    max_context_length=128,
)

# Two rows of 16 tokens, each packing three documents (6, 3, 7 tokens).
_DOC_LENGTHS = (6, 3, 7)
_SEQ_LEN = sum(_DOC_LENGTHS)
_BATCH = 2


def _tiny_hf_config():
  t = _TINY
  return transformers.DeepseekV3Config(
      vocab_size=t["vocab_size"],
      hidden_size=t["dim"],
      intermediate_size=t["dense_hidden_dim"],
      moe_intermediate_size=t["moe_hidden_dim"],
      num_hidden_layers=t["n_layers"],
      first_k_dense_replace=t["n_dense_layers"],
      num_attention_heads=t["n_heads"],
      num_key_value_heads=t["n_heads"],
      q_lora_rank=t["q_lora_rank"],
      kv_lora_rank=t["kv_lora_rank"],
      qk_nope_head_dim=t["qk_nope_head_dim"],
      qk_rope_head_dim=t["qk_rope_head_dim"],
      v_head_dim=t["v_head_dim"],
      n_routed_experts=t["num_experts"],
      n_shared_experts=t["num_shared_experts"],
      num_experts_per_tok=t["top_k"],
      n_group=t["num_expert_groups"],
      topk_group=t["num_limited_groups"],
      routed_scaling_factor=t["route_scale"],
      norm_topk_prob=True,
      max_position_embeddings=t["max_context_length"],
      rope_interleave=True,
      rope_parameters=dict(
          rope_type="yarn",
          rope_theta=t["rope_theta"],
          factor=t["rope_factor"],
          beta_fast=32.0,
          beta_slow=1.0,
          original_max_position_embeddings=t["original_seq_len"],
          mscale=1.0,
          mscale_all_dim=1.0,
      ),
      tie_word_embeddings=False,
      attn_implementation="sdpa",
  )


def _tiny_titan_config(
    num_mtp_layers: int = 0, **overrides
) -> deepseek_v3_models.DeepSeekV3Model.Config:
  """The TorchTitan side of ``_tiny_hf_config``, built like the 671B flavor.

  The aux loss is off: HF has none, and it only adds a router gradient.
  The expert-bias balancer stays on, so ``expert_bias_E`` exists and carries
  HF's ``e_score_correction_bias``. ``overrides`` replace ``_TINY`` entries.
  """
  t = {**_TINY, **overrides}
  # pylint: disable=protected-access
  layers = deepseek_v3_models._build_dsv3_layers(
      n_layers=t["n_layers"],
      n_dense_layers=t["n_dense_layers"],
      dim=t["dim"],
      n_heads=t["n_heads"],
      q_lora_rank=t["q_lora_rank"],
      kv_lora_rank=t["kv_lora_rank"],
      qk_nope_head_dim=t["qk_nope_head_dim"],
      qk_rope_head_dim=t["qk_rope_head_dim"],
      v_head_dim=t["v_head_dim"],
      mscale=1.0,
      dense_hidden_dim=t["dense_hidden_dim"],
      moe_hidden_dim=t["moe_hidden_dim"],
      num_experts=t["num_experts"],
      num_shared_experts=t["num_shared_experts"],
      router_top_k=t["top_k"],
      router_score_func="sigmoid",
      router_num_expert_groups=t["num_expert_groups"],
      router_num_limited_groups=t["num_limited_groups"],
      router_route_scale=t["route_scale"],
      router_route_norm=True,
      aux_loss_coeff=None,
      attn_backend="sdpa",
      moe_comm_backend="standard",
      non_blocking_capacity_factor=None,
      rope=common_models.ComplexRoPE.Config(
          dim=t["qk_rope_head_dim"],
          max_context_length=t["max_context_length"],
          theta=t["rope_theta"],
          scaling="yarn",
          rope_factor=t["rope_factor"],
          beta_fast=32.0,
          beta_slow=1.0,
          original_seq_len=t["original_seq_len"],
      ),
  )
  dim, vocab_size = t["dim"], t["vocab_size"]
  config = deepseek_v3_models.DeepSeekV3Model.Config(
      vocab_size=vocab_size,
      dim=dim,
      tok_embeddings=common_models.Embedding.Config(
          num_embeddings=vocab_size,
          embedding_dim=dim,
          param_init=deepseek_v3_models._EMBEDDING_INIT,
      ),
      norm=common_models.RMSNorm.Config(
          normalized_shape=dim,
          eps=1e-6,
          param_init=deepseek_v3_models._NORM_INIT,
      ),
      lm_head=common_models.Linear.Config(
          in_features=dim,
          out_features=vocab_size,
          param_init=deepseek_v3_models._output_linear_init(dim),
      ),
      layers=layers,
      mtp_layers=deepseek_v3_models._build_mtp_layers(
          layers[-1], dim=dim, num_mtp_layers=num_mtp_layers
      ),
  )
  # pylint: enable=protected-access
  return deepseek_v3_tpu.use_tpu_attention(config)


def _hf_to_checkpoint_layout(state_dict):
  """HF transformers v5 in-memory keys -> the on-disk checkpoint keys.

  v5 fuses each MoE layer's experts into ``experts.gate_up_proj`` ``[E, 2F,
  D]`` (gate rows first) and ``experts.down_proj`` ``[E, D, F]``; checkpoints
  keep one ``experts.<e>.{gate,up,down}_proj.weight`` per expert, which is
  what the adapter reads. Used for gradients; checked against
  ``save_pretrained`` on the weights.
  """
  out = {}
  for key, value in state_dict.items():
    if key.endswith("mlp.experts.gate_up_proj"):
      prefix = key[: -len("gate_up_proj")]
      gate, up = value.chunk(2, dim=1)
      for e in range(value.shape[0]):
        out[f"{prefix}{e}.gate_proj.weight"] = gate[e]
        out[f"{prefix}{e}.up_proj.weight"] = up[e]
    elif key.endswith("mlp.experts.down_proj"):
      prefix = key[: -len("down_proj")]
      for e in range(value.shape[0]):
        out[f"{prefix}{e}.down_proj.weight"] = value[e]
    else:
      out[key] = value
  return out


def _packed_positions() -> torch.Tensor:
  return torch.cat([torch.arange(n) for n in _DOC_LENGTHS]).repeat(_BATCH)


def assert_gradients(test, actual, reference, bound):
  """Per-tensor relative L2 ``||g - g_ref|| / ||g_ref||`` <= ``bound``.

  An absolute floor (1e-6) applies only where ``||g_ref|| < 1e-4``.

  Returns:
    ``(worst relative L2, name)``.
  """
  test.assertEqual(set(actual), set(reference))
  worst = (0.0, "")
  for name, expected in reference.items():
    expected = expected.float().cpu()
    got = actual[name].float().cpu()
    norm = expected.norm().item()
    error = (got - expected).norm().item()
    if norm < 1e-4:
      test.assertLessEqual(error, 1e-6, name)
    else:
      relative = error / norm
      test.assertLessEqual(relative, bound, name)
      worst = max(worst, (relative, name))
  return worst


class _Bf16GroupedMmDefaultPrecision(TorchDispatchMode):
  """Runs bf16 ``torch._grouped_mm`` at DEFAULT precision on TPU.

  libtpu 0.0.48 rejects HIGHEST for the bf16 ragged dot behind
  ``torch._grouped_mm`` (the upstream experts always run bf16). Only those
  ops (forward and backward) are scoped; every fp32 matmul stays HIGHEST.
  """

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    if (
        func == torch.ops.aten._grouped_mm.default
        and args[0].device.type == "tpu"
        and args[0].dtype == torch.bfloat16
    ):
      with torch.tpu.precision(torch.tpu.Precision.DEFAULT):
        return func(*args, **(kwargs or {}))
    return func(*args, **(kwargs or {}))


class TestDeepSeekV3HFReferenceParity(BaseAcceleratorDeviceTest):
  """One forward + backward vs HF ``DeepseekV3ForCausalLM`` on packed rows."""

  def setUp(self):
    super().setUp()
    if transformers is None or not hasattr(
        transformers, "DeepseekV3ForCausalLM"
    ):
      self.skipTest("transformers with DeepseekV3ForCausalLM is required")
    use_cpu_safe_histc_patch()
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    self.addCleanup(torch.set_float32_matmul_precision, previous)
    if self.accelerator_device.type == "tpu":
      self.enter_context(_Bf16GroupedMmDefaultPrecision())

  def _build_reference(self):
    torch.manual_seed(0)
    hf_model = transformers.DeepseekV3ForCausalLM(_tiny_hf_config())
    hf_model = hf_model.float().eval()
    with torch.no_grad():
      for name, param in hf_model.named_parameters():
        if "norm" in name:
          # Unit-initialized: a dropped multiply would be invisible.
          param.copy_(1.0 + 0.1 * torch.randn_like(param))
        elif name.endswith("mlp.gate.weight"):
          param.copy_(0.1 * torch.randn_like(param))
      for name, buffer in hf_model.named_buffers():
        if name.endswith("e_score_correction_bias"):
          # Moves some top-k choices away from the unbiased ranking.
          buffer.copy_(0.05 * torch.randn_like(buffer))
    return hf_model

  def test_forward_backward_matches_hf(self):
    hf_model = self._build_reference()
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
    hf_model.save_pretrained(tmp_dir, safe_serialization=True)
    from safetensors.torch import load_file  # pylint: disable=g-import-not-at-top

    on_disk = {}
    for fname in os.listdir(tmp_dir):
      if fname.endswith(".safetensors"):
        on_disk.update(load_file(os.path.join(tmp_dir, fname)))
    # The gradient re-layout helper reproduces save_pretrained's layout.
    in_memory = _hf_to_checkpoint_layout(hf_model.state_dict())
    self.assertEqual(set(in_memory), set(on_disk))
    for key, value in on_disk.items():
      torch.testing.assert_close(in_memory[key], value, atol=0, rtol=0)

    config = _tiny_titan_config()
    model = config.build().to(self.accelerator_device)
    model.init_states(buffer_device=self.accelerator_device)
    model.eval()
    adapter = DeepSeekV3StateDictAdapter(config, None)
    # Keys both ways: what the model asks for is exactly what is on disk ...
    self.assertEqual(set(adapter.to_hf(model.state_dict())), set(on_disk))
    # ... and everything on disk lands in the model.
    missing, unexpected = model.load_state_dict(
        adapter.from_hf(on_disk), strict=False
    )
    self.assertEmpty(missing)
    self.assertEmpty(unexpected)

    routes = ({}, {})

    def record_hf(name):
      def hook(module, inputs, output):
        routes[0][name] = output[2].detach().sort(dim=-1).values

      return hook

    def record_titan(name):
      def hook(module, inputs, output):
        routes[1][name] = output[1].detach().cpu().sort(dim=-1).values

      return hook

    for i, layer in enumerate(hf_model.model.layers):
      if hasattr(layer.mlp, "gate"):
        layer.mlp.gate.register_forward_hook(record_hf(str(i)))
    for name, layer in model.layers.items():
      if layer.moe_enabled:
        layer.moe.router.register_forward_hook(record_titan(name))

    torch.manual_seed(1)
    num_tokens = _BATCH * _SEQ_LEN
    tokens = torch.randint(0, _TINY["vocab_size"], (num_tokens,))
    labels = torch.randint(0, _TINY["vocab_size"], (num_tokens,))
    positions = _packed_positions()

    # Reference: each packed document is its own sequence.
    hf_logits, start = [], 0
    hf_routes = {}
    for _ in range(_BATCH):
      for length in _DOC_LENGTHS:
        hf_logits.append(
            hf_model(input_ids=tokens[start : start + length][None]).logits[0]
        )
        for name, value in routes[0].items():
          hf_routes.setdefault(name, []).append(value)
        start += length
    hf_logits = torch.cat(hf_logits)
    hf_loss = F.cross_entropy(hf_logits, labels)
    hf_loss.backward()

    with set_unflatten_seq_len(_SEQ_LEN):
      logits = model(
          tokens.to(self.accelerator_device),
          positions.to(self.accelerator_device),
      )
    loss = F.cross_entropy(logits.float(), labels.to(self.accelerator_device))
    loss.backward()

    for name, value in routes[1].items():
      torch.testing.assert_close(
          value, torch.cat(hf_routes[name]), atol=0, rtol=0
      )
    self.assertNotEmpty(routes[1])

    labels_kw = dict(
        test_label=f"TorchTitan DeepSeekV3 ({self.accelerator_device.type})",
        ref_label="HF DeepseekV3ForCausalLM (CPU)",
    )
    logit_error = (logits.detach().cpu() - hf_logits.detach()).abs().max()
    loss_error = (loss.detach().cpu() - hf_loss.detach()).abs()
    test_utils.check_equivalence(
        logits.detach().float().cpu(),
        hf_logits.detach(),
        atol=2e-2,
        rtol=1e-2,
        check_name="Logits",
        **labels_kw,
    )
    test_utils.check_equivalence(
        loss.detach().float().cpu(),
        hf_loss.detach(),
        atol=2e-3,
        rtol=1e-3,
        check_name="Loss",
        **labels_kw,
    )

    hf_grads = adapter.from_hf(
        _hf_to_checkpoint_layout({
            name: param.grad
            for name, param in hf_model.named_parameters()
            if param.grad is not None
        })
    )
    grads = {
        name: param.grad
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    self.assertLen(grads, len(list(model.parameters())))
    worst = assert_gradients(self, grads, hf_grads, bound=2e-2)
    print(
        f"DeepSeek-V3 vs HF ({self.accelerator_device.type}): max |logit"
        f" diff| {logit_error:.3e}, |loss diff| {loss_error:.3e}, worst"
        f" gradient relative L2 {worst[0]:.3e} ({worst[1]})"
    )
    # Negative control: the bound rejects zeroed and sign-flipped gradients.
    for multiplier in (0.0, -1.0):
      with self.assertRaises(AssertionError):
        assert_gradients(
            self,
            {n: g * multiplier for n, g in hf_grads.items()},
            hf_grads,
            bound=2e-2,
        )


class TestDeepSeekV3PackedDocumentMask(BaseAcceleratorDeviceTest):
  """The document-masked SDPA at MLA's real head dims (q/k 192, v 128)."""

  def test_matches_per_document_attention_and_fails_without_mask(self):
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    self.addCleanup(torch.set_float32_matmul_precision, previous)
    device = self.accelerator_device
    torch.manual_seed(0)
    heads, qk_dim, v_dim = 4, 192, 128
    num_tokens = _BATCH * _SEQ_LEN
    scale = qk_dim**-0.5
    q, k = (torch.randn(num_tokens, heads, qk_dim) for _ in range(2))
    v = torch.randn(num_tokens, heads, v_dim)
    positions = _packed_positions()

    # Independent reference on CPU: causal SDPA per document.
    ref_inputs = [t.clone().requires_grad_() for t in (q, k, v)]
    pieces, start = [], 0
    for _ in range(_BATCH):
      for length in _DOC_LENGTHS:
        qq, kk, vv = (
            t[start : start + length].transpose(0, 1) for t in ref_inputs
        )
        pieces.append(
            F.scaled_dot_product_attention(
                qq, kk, vv, is_causal=True, scale=scale
            ).transpose(0, 1)
        )
        start += length
    expected = torch.cat(pieces)
    expected_grads = torch.autograd.grad(expected.square().sum(), ref_inputs)

    inputs = [t.to(device).requires_grad_() for t in (q, k, v)]
    attention = SlidingWindowScaledDotProductInnerAttention.Config().build()
    with set_unflatten_seq_len(_SEQ_LEN):
      actual = attention(*inputs, scale=scale, positions=positions.to(device))
      # Control: the stock SDPA ignores positions (one causal problem/row).
      unmasked = ScaledDotProductInnerAttention.Config().build()(
          *inputs, scale=scale, positions=positions.to(device)
      )
    self.assertEqual(actual.shape, (num_tokens, heads, v_dim))
    torch.testing.assert_close(
        actual.detach().cpu(), expected.detach(), atol=1e-5, rtol=1e-5
    )
    actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
    assert_gradients(
        self,
        dict(enumerate(actual_grads)),
        dict(enumerate(expected_grads)),
        bound=1e-4,
    )
    relative = (
        (unmasked.detach().cpu() - expected.detach()).norm()
        / expected.detach().norm()
    ).item()
    self.assertGreater(relative, 0.1)


class TestDeepSeekV3SplashAttention(BaseAcceleratorDeviceTest):
  """Splash / tokamax replace MLA's document-masked SDPA with the same result.

  The kernels' numerics, segment ids included, are ``kernels/tests``'. What is
  DeepSeek-V3-specific: MLA is its own attention module, its q/k head dim
  (``qk_nope + qk_rope`` = 192) differs from its v head dim (128), its softmax
  scale carries the YaRN mscale factor, and it threads ``positions`` /
  ``segment_ids`` to the inner attention itself. So one MLA attention at the
  real head dims runs on a packed batch (documents 200 / 180 / 132 and 300 /
  212 in two 512-token rows), patched on the chip vs the document-masked SDPA
  on CPU at "highest", comparing the output and every gradient.

  Bounds: relative L2 <= 1e-2 forward, <= 2.5e-2 per gradient (measured on
  v6e: 2.0e-3 forward, worst gradient 3.1e-3, both kernels).

  Controls on the reference, each outside the forward bound, so a kernel that
  dropped the segment ids or the YaRN-scaled softmax scale fails here: without
  the document mask it moves 7.2e-1; with the plain ``qk_head_dim**-0.5``
  scale (1.87x smaller logits) 3.3e-2.
  """

  _DOCS = ((200, 180, 132), (300, 212))
  _SEQ = 512

  def _attention(self):
    torch.manual_seed(0)
    config = _tiny_titan_config(
        dim=256,
        n_heads=4,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        max_context_length=self._SEQ,
    )
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    attention = model.layers["0"].attention
    with torch.no_grad():
      for norm in (attention.q_norm, attention.kv_norm):
        norm.weight.mul_(1.0 + 0.1 * torch.randn_like(norm.weight))
    return attention

  def _run(self, attention, x, positions, device):
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.models.common.attention import segment_ids_from_positions

    attention = attention.to(device)
    x = x.to(device).requires_grad_()
    positions = positions.to(device)
    with set_unflatten_seq_len(self._SEQ):
      out = attention(
          x,
          None,
          positions=positions,
          segment_ids=segment_ids_from_positions(positions),
      )
    loss = (out.float() * self._probe.to(device)).sum()
    params = dict(attention.named_parameters())
    grads = torch.autograd.grad(loss, [x, *params.values()])
    grads = dict(zip(["x", *params], grads))
    return out.detach().float().cpu(), grads

  @parameterized.named_parameters(
      dict(testcase_name="splash", kernel="splash"),
      dict(testcase_name="tokamax", kernel="tokamax"),
  )
  def test_matches_masked_sdpa_at_real_head_dims(self, kernel):
    if self.accelerator_device.type != "tpu":
      self.skipTest("Splash / tokamax are TPU Pallas kernels")
    # pylint: disable=g-import-not-at-top
    import copy
    from torchtitan.experiments.tpu import workarounds
    from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig
    # pylint: enable=g-import-not-at-top

    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    self.addCleanup(torch.set_float32_matmul_precision, previous)
    attention = self._attention()
    num_tokens = len(self._DOCS) * self._SEQ
    torch.manual_seed(1)
    x = torch.randn(num_tokens, 256)
    self._probe = torch.randn(num_tokens, 256)
    positions = torch.cat([torch.arange(n) for row in self._DOCS for n in row])

    expected, expected_grads = self._run(
        copy.deepcopy(attention), x, positions, torch.device("cpu")
    )

    patched = copy.deepcopy(attention)
    if kernel == "tokamax":
      workarounds.use_tokamax_splash_attention_patch(
          patched, SplashAttentionKernelConfig()
      )
      self.assertIn("forward", vars(patched.inner_attention))
    else:
      workarounds.use_splash_attention_patch(
          patched, SplashAttentionKernelConfig()
      )
      self.assertNotIn("ScaledDotProduct", type(patched.inner_attention).__name__)
    actual, actual_grads = self._run(
        patched, x, positions, self.accelerator_device
    )
    forward_error = ((actual - expected).norm() / expected.norm()).item()
    self.assertLessEqual(forward_error, 1e-2)
    worst = assert_gradients(self, actual_grads, expected_grads, bound=2.5e-2)
    print(
        f"[{kernel}] forward rel L2 {forward_error:.3e}; worst gradient rel L2"
        f" {worst[0]:.3e} ({worst[1]})"
    )

    # Controls on the reference: no document mask, then the plain scale.
    unmasked = copy.deepcopy(attention)
    unmasked.inner_attention = ScaledDotProductInnerAttention.Config().build()
    plain_scale = copy.deepcopy(attention)
    plain_scale.softmax_scale = plain_scale.qk_head_dim**-0.5
    for name, control, floor in (
        ("unmasked", unmasked, 0.1),
        ("plain scale", plain_scale, 2e-2),
    ):
      moved, _ = self._run(control, x, positions, torch.device("cpu"))
      relative = ((moved - expected).norm() / expected.norm()).item()
      print(f"[{kernel}] control {name}: reference moves {relative:.3e}")
      self.assertGreater(relative, floor, name)


class TestDeepSeekV3FusedCrossEntropy(BaseAcceleratorDeviceTest):
  """The ``*_fused_ce`` recipes and the fused kernel at DeepSeek-V3's vocab.

  The kernel's numerics are ``kernels/tests``' at tile-multiple vocabularies.
  DeepSeek-V3's 129280 is not a multiple of the default 2048-wide vocab tile
  (63 full tiles + 256), so the test runs the fused loss at the real hidden
  size and vocabulary against fp32 ``F.cross_entropy`` on full logits, with a
  half of the labels in the partial last tile. Bounds and measurement in
  the assertions.

  Control: the same reference with the partial tile's weight rows zeroed (a
  kernel that padded the tail instead of reading it) moves the loss far
  outside the bound.
  """

  @parameterized.named_parameters(
      dict(testcase_name="debugmodel", recipe="deepseek_v3_debugmodel_fused_ce"),
      dict(testcase_name="proxy", recipe="deepseek_v3_671b_2l_16e_fused_ce"),
      dict(
          testcase_name="proxy_graph_trainer",
          recipe="graph_trainer_deepseek_v3_671b_2l_16e_fused_ce",
      ),
  )
  def test_recipe_swaps_lm_head_for_fused_loss(self, recipe):
    # pylint: disable=g-import-not-at-top
    from torchtitan.config import apply_overrides
    from torchtitan.experiments.tpu import workarounds
    from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
    from torchtitan.experiments.tpu.overrides.fused_loss import FusedLossLinearWrapper
    # pylint: enable=g-import-not-at-top

    config = getattr(deepseek_v3_tpu, recipe)()
    self.assertIsInstance(config.loss, PallasCrossEntropyLoss.Config)
    self.assertIsNone(config.loss.soft_cap)
    self.assertEqual(config.loss.output_multiplier, 1.0)
    apply_overrides(config.override, config)
    self.assertIsInstance(
        config.model_spec.model.lm_head, FusedLossLinearWrapper.Config
    )
    with torch.device("meta"):
      model = config.model_spec.model.build()
    self.assertIsInstance(model.lm_head, FusedLossLinearWrapper)
    # Same checkpoint key, so plain-CE checkpoints and the HF adapter still fit.
    self.assertEqual(
        [k for k in model.state_dict() if k.endswith("lm_head.weight")],
        ["lm_head.weight"],
    )
    workarounds._check_fused_loss_logit_transform(model, config)  # pylint: disable=protected-access

  def test_matches_fp32_cross_entropy_at_real_vocab(self):
    if self.accelerator_device.type != "tpu":
      self.skipTest("The fused loss is a TPU Mosaic kernel")
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.loss import pallas_cross_entropy_loss

    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    self.addCleanup(torch.set_float32_matmul_precision, previous)
    device = self.accelerator_device
    tokens, hidden, vocab, full_tiles = 1024, 7168, 129280, 63 * 2048
    torch.manual_seed(0)
    # Trainer dtypes: bf16 activations and (all-gathered) bf16 head weight.
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    weight = (
        torch.randn(vocab, hidden, device=device) * hidden**-0.5
    ).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (tokens,), device=device)
    labels[::2] = torch.randint(full_tiles, vocab, (tokens // 2,), device=device)

    def reference(x, weight):
      logits = x.float() @ weight.float().t()
      return F.cross_entropy(logits, labels, reduction="sum")

    ref_inputs = [t.detach().clone().requires_grad_() for t in (x, weight)]
    expected = reference(*ref_inputs)
    expected_grads = torch.autograd.grad(expected, ref_inputs)

    inputs = [t.detach().clone().requires_grad_() for t in (x, weight)]
    actual = pallas_cross_entropy_loss((inputs[0], inputs[1].t()), labels)
    actual_grads = torch.autograd.grad(actual, inputs)

    loss_error = abs(actual.item() - expected.item()) / abs(expected.item())
    print(f"fused CE loss rel {loss_error:.3e}")
    # Measured on v6e: loss 7.8e-8 relative, worst gradient rel L2 2.3e-4.
    self.assertLessEqual(loss_error, 1e-5)
    worst = assert_gradients(
        self,
        {"x": actual_grads[0], "weight": actual_grads[1]},
        {"x": expected_grads[0], "weight": expected_grads[1]},
        bound=2e-3,
    )
    print(f"fused CE worst gradient rel L2 {worst[0]:.3e} ({worst[1]})")
    with torch.no_grad():
      tail_zeroed = weight.clone()
      tail_zeroed[full_tiles:] = 0
      dropped_tail = reference(x, tail_zeroed)
    control = abs(dropped_tail.item() - expected.item()) / abs(expected.item())
    # Measured 2.3e-3, 230x the loss bound.
    print(f"fused CE control (no partial tile): loss moves {control:.3e}")
    self.assertGreater(control, 1e-3)


class TestDeepSeekV3CompiledBiasBalancer(BaseAcceleratorDeviceTest):
  """The auxiliary-loss-free balancer survives per-block compile.

  ``MoE.forward`` accumulates ``tokens_per_expert_E`` in place; the optimizer
  pre-hook turns the counts into an ``expert_bias_E`` update. If the compiled
  block dropped the in-place add, the counts would stay zero and the bias
  would never move.
  """

  def test_counter_and_bias_update_under_compile(self):
    if self.accelerator_device.type != "tpu":
      self.skipTest("Validates TorchTPU per-block compile (backend='tpu')")
    from torchtitan.components.optimizer import register_moe_load_balancing_hook  # pylint: disable=g-import-not-at-top

    torch.manual_seed(0)
    config = _tiny_titan_config()
    model = config.build().to(self.accelerator_device)
    model.init_states(buffer_device=self.accelerator_device)
    model.train()
    torch._dynamo.config.capture_scalar_outputs = True
    for layer in model.layers.values():
      layer.compile(backend="tpu", fullgraph=True, dynamic=False)
    num_tokens = _BATCH * _SEQ_LEN
    tokens = torch.randint(0, _TINY["vocab_size"], (num_tokens,))
    with set_unflatten_seq_len(_SEQ_LEN):
      logits = model(
          tokens.to(self.accelerator_device),
          _packed_positions().to(self.accelerator_device),
      )
    logits.float().square().mean().backward()

    moes = [layer.moe for layer in model.layers.values() if layer.moe_enabled]
    self.assertNotEmpty(moes)
    for moe in moes:
      self.assertEqual(
          moe.tokens_per_expert_E.sum().item(), num_tokens * _TINY["top_k"]
      )
      self.assertEqual(moe.expert_bias_E.abs().sum().item(), 0.0)

    hooks = []

    class _Optimizers:

      def register_step_pre_hook(self, hook):
        hooks.append(hook)

    class _ParallelDims:
      ep_enabled = False
      tp = 1

      def get_optional_mesh(self, name):
        del name
        return None

    register_moe_load_balancing_hook(_Optimizers(), [model], _ParallelDims())
    self.assertLen(hooks, 1)
    hooks[0]()
    for moe in moes:
      self.assertGreater(moe.expert_bias_E.abs().sum().item(), 0.0)
      self.assertEqual(moe.tokens_per_expert_E.abs().sum().item(), 0.0)


class TestDeepSeekV3GraphTrainerBiasBalancer(BaseAcceleratorDeviceTest):
  """The auxiliary-loss-free balancer survives the GraphTrainer compile.

  GraphTrainer traces forward + loss + backward into one FX graph
  (``minimal_fx_tracer``) and lowers it with ``tpu_backend_compile_pass``. The
  executable returns values only, so ``MoE.tokens_per_expert_E.add_`` reaches
  the live buffer only because the pass returns the buffer and copies it back.
  Without that the counts stay zero, the optimizer pre-hook sees no load, and
  ``expert_bias_E`` never moves (measured: this test fails with the pass's
  buffer-update lifting removed).
  """

  def test_counter_and_bias_update_on_the_compiled_graph(self):
    if self.accelerator_device.type != "tpu":
      self.skipTest("Validates the TPU joint-graph compiler")
    # pylint: disable=g-import-not-at-top
    from torchtitan.components.optimizer.optimizer import default_adamw
    from torchtitan.components.optimizer.optimizer import register_moe_load_balancing_hook
    from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
    from torchtitan.experiments.graph_trainer.make_fx_tracer import run_traced
    from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import tpu_backend_compile_pass
    # pylint: enable=g-import-not-at-top

    torch.manual_seed(0)
    model = _tiny_titan_config().build()
    model.init_states(buffer_device=torch.device("cpu"))
    model = model.to(self.accelerator_device).train()
    optimizer = default_adamw(lr=1e-3, implementation="foreach").build(
        model_parts=[model]
    )

    class _ParallelDims:
      ep_enabled = False
      tp = 1

      def get_optional_mesh(self, name):
        del name
        return None

    register_moe_load_balancing_hook(optimizer, [model], _ParallelDims())
    num_tokens = _BATCH * _SEQ_LEN
    tokens = torch.randint(0, _TINY["vocab_size"], (num_tokens,))
    labels = torch.randint(0, _TINY["vocab_size"], (num_tokens,))
    tokens = tokens.to(self.accelerator_device)
    labels = labels.to(self.accelerator_device)
    positions = _packed_positions().to(self.accelerator_device)

    def step(inputs, targets, positions):
      # The tracer swaps the module's parameters for graph inputs, so read them
      # inside the traced function.
      loss = F.cross_entropy(model(inputs, positions).float(), targets)
      return loss, torch.autograd.grad(loss, tuple(model.parameters()))

    moes = [layer.moe for layer in model.layers.values() if layer.moe_enabled]
    self.assertNotEmpty(moes)
    with set_unflatten_seq_len(_SEQ_LEN):
      traced = minimal_fx_tracer(step, module=model)(tokens, labels, positions)
      # Tracing runs on fake tensors; the live counters must still be zero.
      for moe in moes:
        self.assertEqual(moe.tokens_per_expert_E.abs().sum().item(), 0.0)
      traced.gm = tpu_backend_compile_pass(traced.gm, traced.example_inputs)
      run = run_traced(traced, module=model)
      previous_bias = [moe.expert_bias_E.clone() for moe in moes]
      for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, grads = run(tokens, labels, positions)
        self.assertTrue(torch.isfinite(loss).item())
        for param, grad in zip(model.parameters(), grads, strict=True):
          param.grad = grad
        for moe in moes:
          self.assertEqual(
              moe.tokens_per_expert_E.sum().item(), num_tokens * _TINY["top_k"]
          )
        optimizer.step()
        for moe, before in zip(moes, previous_bias, strict=True):
          self.assertEqual(moe.tokens_per_expert_E.abs().sum().item(), 0.0)
          self.assertGreater(
              (moe.expert_bias_E - before).abs().sum().item(), 0.0
          )
        previous_bias = [moe.expert_bias_E.clone() for moe in moes]


class TestDeepSeekV3StateDictAdapter(absltest.TestCase):

  def test_round_trip_with_mtp(self):
    config = _tiny_titan_config(num_mtp_layers=1)
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    state = model.state_dict()
    adapter = DeepSeekV3StateDictAdapter(config, None)
    hf_state = adapter.to_hf(state)
    # The MTP layer is HF layer index n_layers (DeepSeek's layer 61 at 671B).
    mtp_index = _TINY["n_layers"]
    self.assertIn(f"model.layers.{mtp_index}.eh_proj.weight", hf_state)
    self.assertIn(
        f"model.layers.{mtp_index}.mlp.experts.0.gate_proj.weight", hf_state
    )
    restored = adapter.from_hf(hf_state)
    self.assertEqual(set(restored), set(state))
    for key in state:
      torch.testing.assert_close(restored[key], state[key], atol=0, rtol=0)


class TestDeepSeekV3Flavors(parameterized.TestCase):

  def test_every_flavor_uses_document_masked_sdpa(self):
    for flavor, config in deepseek_v3_tpu.deepseekv3_configs.items():
      for layer in [*config.layers, *config.mtp_layers]:
        inner = layer.attention.inner_attention
        self.assertIsInstance(
            inner, SlidingWindowScaledDotProductInnerAttention.Config, flavor
        )
        self.assertIsNone(inner.window_size, flavor)
    spec = deepseek_v3_tpu.model_registry("debugmodel", num_mtp_layers=1)
    self.assertLen(spec.model.mtp_layers, 1)
    self.assertIsInstance(
        spec.model.mtp_layers[0].attention.inner_attention,
        SlidingWindowScaledDotProductInnerAttention.Config,
    )
    self.assertIs(
        spec.parallelize_fn, deepseek_v3_tpu.parallelize_deepseek_v3
    )
    self.assertIsNone(spec.pipelining_fn)

  def test_671b_matches_reference_config(self):
    """Upstream 671B vs the DeepSeek-V3 checkpoint config.

    ``transformers.DeepseekV3Config()`` defaults are DeepSeek-V3's
    ``config.json`` values; RoPE is YaRN factor 40 over 4096 original
    positions, beta 32 / 1, mscale = mscale_all_dim = 1 (checkpoint
    ``rope_scaling``).
    """
    if transformers is None:
      self.skipTest("transformers is required")
    hf = transformers.DeepseekV3Config()
    config = deepseek_v3_tpu.deepseekv3_configs["671B"]
    self._check_widths(config, hf)
    self.assertLen(config.layers, hf.num_hidden_layers)
    for i, layer in enumerate(config.layers):
      self.assertEqual(layer.moe is None, i < hf.first_k_dense_replace, i)

  def test_671b_31l_proxy(self):
    if transformers is None:
      self.skipTest("transformers is required")
    hf = transformers.DeepseekV3Config()
    spec = deepseek_v3_tpu.model_registry("671B-31L", num_mtp_layers=1)
    config = spec.model
    self.assertEqual(spec.flavor, "671B-31L")
    self._check_widths(config, hf)
    self.assertLen(config.layers, deepseek_v3_tpu.PROXY_31L_N_LAYERS)
    self.assertEqual(
        [layer.moe is None for layer in config.layers],
        [True] * 3 + [False] * 28,
    )
    self.assertLen(config.mtp_layers, 1)
    self.assertIsNotNone(config.mtp_layers[0].moe)

  def test_671b_2l_16e_measurement_proxy(self):
    """671B's widths with 1 dense + 1 MoE layer and 16 routed experts."""
    if transformers is None:
      self.skipTest("transformers is required")
    spec = deepseek_v3_tpu.model_registry(
        deepseek_v3_tpu.REDUCED_671B_FLAVOR, seq_len=4096
    )
    config = spec.model
    self.assertEqual(spec.flavor, "671B-2L-16E")
    self._check_widths(config, transformers.DeepseekV3Config(n_routed_experts=16))
    self.assertEqual(
        [layer.moe is None for layer in config.layers], [True, False]
    )
    self.assertEmpty(config.mtp_layers)
    self.assertEqual(config.layers[0].attention.rope.max_context_length, 4096)
    with torch.device("meta"):
      model = config.build()
    params, _ = config.get_nparams_and_flops(model, 4096)
    self.assertAlmostEqual(params / 1e9, 3.373, places=3)

  def _check_widths(self, config, hf):
    self.assertEqual(config.vocab_size, hf.vocab_size)
    self.assertEqual(config.dim, hf.hidden_size)
    self.assertFalse(config.enable_weight_tying)
    for layer in config.layers:
      attn = layer.attention
      self.assertEqual(attn.n_heads, hf.num_attention_heads)
      self.assertEqual(attn.q_lora_rank, hf.q_lora_rank)
      self.assertEqual(attn.kv_lora_rank, hf.kv_lora_rank)
      self.assertEqual(attn.qk_nope_head_dim, hf.qk_nope_head_dim)
      self.assertEqual(attn.qk_rope_head_dim, hf.qk_rope_head_dim)
      self.assertEqual(attn.v_head_dim, hf.v_head_dim)
      self.assertEqual(attn.mscale, 1.0)
      self.assertEqual(attn.rope.scaling, "yarn")
      self.assertEqual(attn.rope.theta, 10000.0)
      self.assertEqual(attn.rope.rope_factor, 40.0)
      self.assertEqual(attn.rope.original_seq_len, 4096)
      self.assertEqual((attn.rope.beta_fast, attn.rope.beta_slow), (32.0, 1.0))
      if layer.moe is None:
        self.assertEqual(layer.feed_forward.w1.out_features, hf.intermediate_size)
        continue
      moe = layer.moe
      router = moe.router
      self.assertEqual(moe.num_experts, hf.n_routed_experts)
      self.assertEqual(router.top_k, hf.num_experts_per_tok)
      self.assertEqual(router.score_func, "sigmoid")
      self.assertEqual(router.num_expert_groups, hf.n_group)
      self.assertEqual(router.num_limited_groups, hf.topk_group)
      self.assertEqual(router.route_scale, hf.routed_scaling_factor)
      self.assertEqual(router.route_norm, hf.norm_topk_prob)
      self.assertEqual(
          moe.routed_experts.inner_experts.hidden_dim, hf.moe_intermediate_size
      )
      self.assertEqual(
          moe.shared_experts.w1.out_features,
          hf.moe_intermediate_size * hf.n_shared_experts,
      )
      # Upstream's balancing: auxiliary-loss-free expert bias + aux loss.
      self.assertEqual(moe.load_balance_coeff, 1e-3)

  def test_671b_parameter_counts(self):
    """671B total / 37B active (DeepSeek-V3 report), MTP excluded."""
    config = deepseek_v3_tpu.deepseekv3_configs["671B"]
    with torch.device("meta"):
      model = config.build()
    nparams, _ = config.get_nparams_and_flops(model, seq_len=4096)
    self.assertAlmostEqual(nparams / 1e9, 671.0, delta=1.0)


if __name__ == "__main__":
  absltest.main()
