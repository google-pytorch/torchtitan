"""Tests for Qwen3.5 (text-only Dense and MoE) on CPU and TPU devices.

Covers:
1. HuggingFace reference parity (``Qwen3_5ForCausalLM`` and
   ``Qwen3_5MoeForCausalLM``) through ``Qwen35StateDictAdapter`` for forward
   logits, cross-entropy loss, and every parameter gradient (with negative
   controls), for dense, MoE, and MoE with ``TpuGroupedExpertsConverter``;
   plus the adapter round trip.
2. ``chunked_gated_delta`` against the exact per-token recurrence, also with
   packed documents, and the slice-based causal convolution's backward against
   autograd of ``conv1d``, also with packed documents.
3. Row isolation in the GatedDeltaNet and packed-document isolation in the
   causal convolution, the GatedDeltaNet, the attention and the whole model
   (with negative controls), plus Splash parity on TPU.
4. Flavor, registry and recipe checks.
"""

import copy

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.muse_glimmer.attention import SlidingWindowScaledDotProductInnerAttention
import torchtitan.experiments.tpu.qwen3_5 as qwen3_5_tpu
from torchtitan.experiments.tpu.qwen3_5 import config_registry
from torchtitan.experiments.tpu.qwen3_5 import gated_delta
from torchtitan.experiments.tpu.qwen3_5 import model as tpu_model
from torchtitan.experiments.tpu.qwen3_5.infra.parallelize import parallelize_qwen3_5
from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch
from torchtitan.models.common.attention import segment_ids_from_positions
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from transformers import (
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
)


def _recurrent_gated_delta_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
) -> torch.Tensor:
  """Per-token reference for the gated delta rule recurrence in fp32.

  With ``(bs, seq_len)`` ``segment_ids``, the state restarts from zero at the
  first token of every document.
  """
  q = gated_delta._l2norm(q.float()) * (q.shape[-1] ** -0.5)  # pylint: disable=protected-access
  k = gated_delta._l2norm(k.float())  # pylint: disable=protected-access
  v, g, beta = v.float(), g.float(), beta.float()
  bs, seq_len, n_heads, d_k = k.shape
  state = torch.zeros(
      bs, n_heads, d_k, v.shape[-1], dtype=torch.float32, device=q.device
  )
  outs = []
  for t in range(seq_len):
    if segment_ids is not None and t > 0:
      new_document = segment_ids[:, t] != segment_ids[:, t - 1]
      state = torch.where(
          new_document.to(q.device)[:, None, None, None],
          torch.zeros_like(state),
          state,
      )
    state = state * g[:, t].exp()[..., None, None]
    kv_mem = (state * k[:, t, :, :, None]).sum(-2)
    delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
    state = state + k[:, t, :, :, None] * delta[:, :, None, :]
    outs.append((state * q[:, t, :, :, None]).sum(-2))
  return torch.stack(outs, dim=1)


def _packed_positions(doc_lens: tuple[int, ...]) -> torch.Tensor:
  """Positions of documents of ``doc_lens`` tokens packed into one row."""
  return torch.cat([torch.arange(n) for n in doc_lens])


def _relative_l2_error(test: torch.Tensor, reference: torch.Tensor) -> float:
  test_f = test.detach().float().cpu()
  ref_f = reference.detach().float().cpu()
  denom = torch.linalg.vector_norm(ref_f).clamp_min(1e-6)
  return (torch.linalg.vector_norm(test_f - ref_f) / denom).item()


class Qwen35Test(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for Qwen3.5 hybrid GatedDeltaNet + Attention + MoE on CPU and TPU."""

  def setUp(self):
    super().setUp()
    use_cpu_safe_histc_patch()
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    self.addCleanup(torch.set_float32_matmul_precision, previous_precision)

  def _is_tpu(self) -> bool:
    return self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU

  def _build_hf_reference(self, is_moe: bool) -> torch.nn.Module:
    # The geometry of upstream `debugmodel` (first 4 layers) / `debugmodel_moe`
    # with vocab_size=2048 for fast CPU execution.
    common_kwargs = dict(
        vocab_size=2048,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [3, 3, 2],
        },
        attn_implementation="sdpa",
    )
    torch.manual_seed(42)
    if is_moe:
      hf_cfg = Qwen3_5MoeTextConfig(
          **common_kwargs,
          moe_intermediate_size=256,
          shared_expert_intermediate_size=256,
          num_experts=8,
          num_experts_per_tok=2,
          tie_word_embeddings=False,
      )
      hf_model = Qwen3_5MoeForCausalLM(hf_cfg).float().eval()
    else:
      hf_cfg = Qwen3_5TextConfig(
          **common_kwargs,
          intermediate_size=512,
          tie_word_embeddings=True,
      )
      hf_model = Qwen3_5ForCausalLM(hf_cfg).float().eval()

    # Perturb unit/zero-initialized parameters so dropped scales cannot hide.
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
      for name, param in hf_model.named_parameters():
        if "norm" in name or "A_log" in name or "dt_bias" in name:
          param.copy_(0.2 * torch.randn(param.shape, generator=gen))
    return hf_model

  @parameterized.named_parameters(
      dict(
          testcase_name="_dense",
          flavor="debugmodel",
          use_tpu_grouped_experts=False,
      ),
      dict(
          testcase_name="_moe",
          flavor="debugmodel_moe",
          use_tpu_grouped_experts=False,
      ),
      dict(
          testcase_name="_moe_tpu_grouped_experts",
          flavor="debugmodel_moe",
          use_tpu_grouped_experts=True,
      ),
  )
  def test_qwen3_5_hf_reference_parity(
      self, flavor: str, use_tpu_grouped_experts: bool
  ):
    """Tests forward logits, loss, and all gradients against HF Qwen3.5."""
    is_moe = flavor == "debugmodel_moe"
    hf_model = self._build_hf_reference(is_moe=is_moe)
    converters = (
        [TpuGroupedExpertsConverter.Config()] if use_tpu_grouped_experts else []
    )
    tt_config = qwen3_5_tpu.model_registry(
        flavor, seq_len=128, converters=converters
    ).model
    tt_config.vocab_size = 2048
    tt_config.tok_embeddings.num_embeddings = 2048
    tt_config.lm_head.out_features = 2048
    tt_config.enable_weight_tying = not is_moe
    tt_config.layers = tt_config.layers[:4]
    adapter = Qwen35StateDictAdapter(tt_config, None)

    tt_model = tt_config.build()
    tt_model.init_states(buffer_device=torch.device("cpu"))

    # Check state_dict mapping both ways.
    tt_sd = adapter.from_hf(copy.deepcopy(hf_model.state_dict()))
    missing, unexpected = tt_model.load_state_dict(tt_sd, strict=False)
    self.assertEmpty(missing)
    self.assertEmpty(unexpected)

    if not use_tpu_grouped_experts:
      round_trip = adapter.from_hf(adapter.to_hf(tt_model.state_dict()))
      self.assertEqual(
          set(round_trip.keys()), set(tt_model.state_dict().keys())
      )
      for k, v in tt_model.state_dict().items():
        torch.testing.assert_close(round_trip[k], v, msg=f"Round-trip key {k}")

    tt_model = tt_model.to(self.accelerator_device)
    if tt_config.enable_weight_tying:
      tt_model.tok_embeddings.weight = tt_model.lm_head.weight
      self.assertIs(tt_model.tok_embeddings.weight, tt_model.lm_head.weight)

    batch, seq_len = 2, 16
    gen = torch.Generator().manual_seed(99)
    tokens_2d = torch.randint(
        0, tt_config.vocab_size, (batch, seq_len), generator=gen
    )
    labels_2d = torch.randint(
        0, tt_config.vocab_size, (batch, seq_len), generator=gen
    )
    positions_1d = torch.arange(seq_len, dtype=torch.long).repeat(batch)

    # Reference forward + backward on CPU in fp32.
    hf_model.zero_grad()
    hf_logits = hf_model(tokens_2d).logits
    hf_loss = F.cross_entropy(
        hf_logits.view(-1, tt_config.vocab_size), labels_2d.view(-1)
    )
    hf_loss.backward()

    # TorchTitan forward + backward on accelerator_device in packed [T] layout.
    tt_model.zero_grad()
    tokens_dev = tokens_2d.view(-1).to(self.accelerator_device)
    labels_dev = labels_2d.view(-1).to(self.accelerator_device)
    positions_dev = positions_1d.to(self.accelerator_device)

    # libtpu 0.0.48 rejects HIGHEST matmul precision for bf16/fp32 ragged dot
    # inside torch._grouped_mm on TPU; use "high" precision when running MoE on
    # TPU.
    precision = "high" if (self._is_tpu() and is_moe) else "highest"
    prev_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
      with set_unflatten_seq_len(seq_len):
        tt_logits = tt_model(tokens_dev, positions=positions_dev)
        tt_loss = F.cross_entropy(
            tt_logits.float().view(-1, tt_config.vocab_size), labels_dev
        )
        tt_loss.backward()
    finally:
      torch.set_float32_matmul_precision(prev_prec)

    # GroupedExperts runs expert matmuls in bfloat16 even when the surrounding
    # model is float32 on CPU, so MoE uses bfloat16 tolerances on both CPU and
    # TPU.
    logit_atol = 5e-2 if (self._is_tpu() or is_moe) else 5e-3
    logit_rtol = 1e-2 if (self._is_tpu() or is_moe) else 5e-3
    loss_atol = 5e-3 if (self._is_tpu() or is_moe) else 1e-5
    loss_rtol = 1e-3 if (self._is_tpu() or is_moe) else 1e-5
    grad_rel_bound = 6e-2 if (self._is_tpu() or is_moe) else 5e-4

    test_utils.check_equivalence(
        tt_logits.cpu().view(batch, seq_len, -1),
        hf_logits.detach(),
        atol=logit_atol,
        rtol=logit_rtol,
        check_name="HF Parity Logits",
    )
    test_utils.check_equivalence(
        tt_loss.cpu(),
        hf_loss.detach(),
        atol=loss_atol,
        rtol=loss_rtol,
        check_name="HF Parity Loss",
    )

    hf_grads = {
        name: param.grad.detach().clone()
        for name, param in hf_model.named_parameters()
        if param.grad is not None
    }
    expected_tt_grads = adapter.from_hf(hf_grads)
    if tt_config.enable_weight_tying:
      expected_tt_grads.pop("lm_head.weight", None)

    actual_tt_grads = {
        name: param.grad.detach().float().cpu()
        for name, param in tt_model.named_parameters()
        if param.grad is not None
    }

    self.assertEqual(set(actual_tt_grads.keys()), set(expected_tt_grads.keys()))
    for name, ref_grad in expected_tt_grads.items():
      rel_err = _relative_l2_error(actual_tt_grads[name], ref_grad)
      self.assertLessEqual(
          rel_err,
          grad_rel_bound,
          msg=f"Gradient relative L2 error too high for {name}: {rel_err:.3e}",
      )
      # Negative controls: zeroed and sign-flipped gradients must exceed bound.
      if torch.linalg.vector_norm(ref_grad).item() > 1e-4:
        self.assertGreater(
            _relative_l2_error(torch.zeros_like(ref_grad), ref_grad),
            grad_rel_bound,
        )
        self.assertGreater(
            _relative_l2_error(-ref_grad, ref_grad),
            grad_rel_bound,
        )

  @parameterized.named_parameters(
      # Several chunks, the last one padded.
      dict(testcase_name="_random", correlated_keys=False, chunk_size=16),
      # Nearly parallel keys with beta near 1 and little decay push the UT
      # transform's triangular system toward its worst conditioning.
      dict(
          testcase_name="_correlated_chunk64",
          correlated_keys=True,
          chunk_size=64,
      ),
      dict(
          testcase_name="_correlated_chunk128",
          correlated_keys=True,
          chunk_size=128,
      ),
  )
  def test_chunked_gated_delta_matches_recurrent(
      self, correlated_keys: bool, chunk_size: int
  ):
    """Tests chunked_gated_delta against the per-token recurrent reference."""
    torch.manual_seed(0)
    if correlated_keys:
      bs, seq_len, n_heads, d_k, d_v = 1, 160, 2, 32, 32
      base = torch.randn(bs, 1, n_heads, d_k)
      k = base + 0.05 * torch.randn(bs, seq_len, n_heads, d_k)
      g_raw = torch.full((bs, seq_len, n_heads), -4.0)
      beta_raw = torch.full((bs, seq_len, n_heads), 4.0)
    else:
      bs, seq_len, n_heads, d_k, d_v = 2, 50, 4, 16, 16
      k = torch.randn(bs, seq_len, n_heads, d_k)
      g_raw = torch.randn(bs, seq_len, n_heads)
      beta_raw = torch.randn(bs, seq_len, n_heads)
    q = torch.randn(bs, seq_len, n_heads, d_k)
    v = torch.randn(bs, seq_len, n_heads, d_v)
    leaves_cpu = tuple(x.requires_grad_() for x in (q, k, v, g_raw, beta_raw))
    leaves_dev = tuple(
        x.detach().to(self.accelerator_device).requires_grad_()
        for x in leaves_cpu
    )

    def run(fn, q, k, v, g_raw, beta_raw, **kwargs):
      return fn(q, k, v, -F.softplus(g_raw), torch.sigmoid(beta_raw), **kwargs)

    expected = run(_recurrent_gated_delta_reference, *leaves_cpu)
    out_dev = run(
        gated_delta.chunked_gated_delta, *leaves_dev, chunk_size=chunk_size
    )
    tol = 5e-3 if self._is_tpu() else 1e-4
    test_utils.check_equivalence(
        out_dev.cpu(),
        expected,
        atol=tol,
        rtol=tol,
        check_name="Chunked vs Recurrent GDN Forward",
    )
    grad_out = torch.randn_like(expected)
    expected.backward(grad_out)
    out_dev.backward(grad_out.to(self.accelerator_device))
    grad_tol = 2e-2 if self._is_tpu() else 5e-4
    for name, x_cpu, x_dev in zip(
        ("q", "k", "v", "g", "beta"), leaves_cpu, leaves_dev, strict=True
    ):
      self.assertLessEqual(
          _relative_l2_error(x_dev.grad, x_cpu.grad),
          grad_tol,
          msg=f"GDN {name} grad mismatch",
      )

  def test_causal_conv_silu_matches_autograd(self):
    """Tests the slice-based causal conv backward against autograd/conv1d."""
    torch.manual_seed(0)
    channels, kernel_size = 8, 4
    x = torch.randn(3, 37, channels, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(
        channels, 1, kernel_size, dtype=torch.float64, requires_grad=True
    )
    grad = torch.randn(3, 37, channels, dtype=torch.float64)
    # pylint: disable=protected-access
    conv = gated_delta._causal_conv(x, weight)
    expected_conv = F.conv1d(
        F.pad(x.transpose(1, 2), (kernel_size - 1, 0)),
        weight,
        groups=channels,
    ).transpose(1, 2)
    # pylint: enable=protected-access
    torch.testing.assert_close(conv, expected_conv)
    out = gated_delta.causal_conv_silu(x, weight)
    expected = F.silu(expected_conv)
    torch.testing.assert_close(out, expected)
    grads = torch.autograd.grad(out, (x, weight), grad)
    expected_grads = torch.autograd.grad(expected, (x, weight), grad)
    for actual_grad, expected_grad in zip(grads, expected_grads, strict=True):
      torch.testing.assert_close(actual_grad, expected_grad)

    # On the accelerator in bf16, as the model runs it.
    dev = self.accelerator_device
    x_bf16 = x.detach().to(dev, torch.bfloat16).requires_grad_()
    w_bf16 = weight.detach().to(dev, torch.bfloat16).requires_grad_()
    out_bf16 = gated_delta.causal_conv_silu(x_bf16, w_bf16)
    grads_bf16 = torch.autograd.grad(
        out_bf16, (x_bf16, w_bf16), grad.to(dev, torch.bfloat16)
    )
    self.assertLess(_relative_l2_error(out_bf16.double().cpu(), expected), 1e-2)
    for actual_grad, expected_grad in zip(
        grads_bf16, expected_grads, strict=True
    ):
      self.assertEqual(actual_grad.dtype, torch.bfloat16)
      self.assertLess(
          _relative_l2_error(actual_grad.double().cpu(), expected_grad), 2e-2
      )

  @parameterized.named_parameters(
      dict(testcase_name="_seq1", seq_len=1),
      dict(testcase_name="_seq2", seq_len=2),
      dict(testcase_name="_seq3", seq_len=3),
  )
  def test_causal_conv_silu_sequences_shorter_than_kernel(self, seq_len: int):
    """Tests the conv backward when taps reach before the sequence start."""
    torch.manual_seed(0)
    channels, kernel_size = 8, 4
    x = torch.randn(
        2, seq_len, channels, dtype=torch.float64, requires_grad=True
    )
    weight = torch.randn(
        channels, 1, kernel_size, dtype=torch.float64, requires_grad=True
    )
    grad = torch.randn(2, seq_len, channels, dtype=torch.float64)
    expected = F.silu(
        F.conv1d(
            F.pad(x.transpose(1, 2), (kernel_size - 1, 0)),
            weight,
            groups=channels,
        ).transpose(1, 2)
    )
    out = gated_delta.causal_conv_silu(x, weight)
    torch.testing.assert_close(out, expected)
    grads = torch.autograd.grad(out, (x, weight), grad)
    expected_grads = torch.autograd.grad(expected, (x, weight), grad)
    for actual_grad, expected_grad in zip(grads, expected_grads, strict=True):
      torch.testing.assert_close(actual_grad, expected_grad)

  @parameterized.named_parameters(
      # Documents start mid-chunk and at chunk starts (16 and 48 with 16-token
      # chunks), one has a single token, some span several chunks, and the
      # last chunk is padded.
      dict(testcase_name="_chunk16", chunk_size=16),
      dict(testcase_name="_chunk64", chunk_size=64),
  )
  def test_chunked_gated_delta_resets_at_documents(self, chunk_size: int):
    """Tests chunked_gated_delta on packed documents against the recurrence."""
    torch.manual_seed(0)
    doc_lens = ((5, 11, 1, 31, 22), (70,))
    bs, seq_len, n_heads, d_k, d_v = len(doc_lens), 70, 2, 16, 16
    segment_ids = segment_ids_from_positions(
        torch.stack([_packed_positions(lens) for lens in doc_lens])
    )
    q = torch.randn(bs, seq_len, n_heads, d_k)
    k = torch.randn(bs, seq_len, n_heads, d_k)
    v = torch.randn(bs, seq_len, n_heads, d_v)
    g_raw = torch.randn(bs, seq_len, n_heads)
    beta_raw = torch.randn(bs, seq_len, n_heads)
    leaves_cpu = tuple(x.requires_grad_() for x in (q, k, v, g_raw, beta_raw))
    leaves_dev = tuple(
        x.detach().to(self.accelerator_device).requires_grad_()
        for x in leaves_cpu
    )

    def run(fn, q, k, v, g_raw, beta_raw, **kwargs):
      return fn(q, k, v, -F.softplus(g_raw), torch.sigmoid(beta_raw), **kwargs)

    expected = run(
        _recurrent_gated_delta_reference, *leaves_cpu, segment_ids=segment_ids
    )
    out_dev = run(
        gated_delta.chunked_gated_delta,
        *leaves_dev,
        chunk_size=chunk_size,
        segment_ids=segment_ids.to(self.accelerator_device),
    )
    tol = 5e-3 if self._is_tpu() else 1e-4
    test_utils.check_equivalence(
        out_dev.cpu(),
        expected,
        atol=tol,
        rtol=tol,
        check_name="Packed Chunked vs Recurrent GDN Forward",
    )
    grad_out = torch.randn_like(expected)
    expected.backward(grad_out)
    out_dev.backward(grad_out.to(self.accelerator_device))
    grad_tol = 2e-2 if self._is_tpu() else 5e-4
    for name, x_cpu, x_dev in zip(
        ("q", "k", "v", "g", "beta"), leaves_cpu, leaves_dev, strict=True
    ):
      self.assertLessEqual(
          _relative_l2_error(x_dev.grad, x_cpu.grad),
          grad_tol,
          msg=f"Packed GDN {name} grad mismatch",
      )

    # Negative control: without segment_ids, the state of each document leaks
    # into the first tokens of the next one.
    with torch.no_grad():
      leaky = run(
          gated_delta.chunked_gated_delta, *leaves_dev, chunk_size=chunk_size
      )
    starts = [5, 16, 17, 48]
    self.assertGreater(
        _relative_l2_error(leaky[0, starts], expected[0, starts]),
        0.1,
        msg="Negative control failed: no state leak without segment_ids",
    )

  def test_causal_conv_resets_at_documents(self):
    """Tests that causal conv1d taps into an earlier document read zeros."""
    torch.manual_seed(0)
    # Documents shorter than the 4-tap kernel, including a 1-token document.
    doc_lens = (3, 1, 6, 2, 8)
    seq_len, channels = sum(doc_lens), 8
    segment_ids = segment_ids_from_positions(_packed_positions(doc_lens))
    segment_ids = segment_ids.expand(2, -1).to(self.accelerator_device)
    x = torch.randn(
        2, seq_len, channels, device=self.accelerator_device
    ).requires_grad_()
    weight = torch.randn(
        channels, 1, 4, device=self.accelerator_device
    ).requires_grad_()

    out = gated_delta._causal_conv(x, weight, segment_ids)  # pylint: disable=protected-access
    starts = [sum(doc_lens[:i]) for i in range(len(doc_lens))]
    expected = torch.cat(
        [
            gated_delta._causal_conv(x[:, s : s + n], weight)  # pylint: disable=protected-access
            for s, n in zip(starts, doc_lens)
        ],
        dim=1,
    )
    tol = 5e-3 if self._is_tpu() else 1e-6
    test_utils.check_equivalence(
        out.detach().cpu(),
        expected.detach().cpu(),
        atol=tol,
        rtol=tol,
        check_name="Packed Causal Conv Forward",
    )
    grad_out = torch.randn_like(out)
    grads = torch.autograd.grad(out, (x, weight), grad_out)
    expected_grads = torch.autograd.grad(
        expected, (x, weight), grad_out, retain_graph=True
    )
    for name, grad, expected_grad in zip(
        ("x", "weight"), grads, expected_grads, strict=True
    ):
      test_utils.check_equivalence(
          grad.cpu(),
          expected_grad.cpu(),
          atol=tol,
          rtol=tol,
          check_name=f"Packed Causal Conv {name} Grad",
      )
    # The slice-based backward of causal_conv_silu restarts at documents too.
    out_silu = gated_delta.causal_conv_silu(x, weight, segment_ids)
    expected_silu = F.silu(expected)
    test_utils.check_equivalence(
        out_silu.detach().cpu(),
        expected_silu.detach().cpu(),
        atol=tol,
        rtol=tol,
        check_name="Packed Causal Conv SiLU Forward",
    )
    grads_silu = torch.autograd.grad(out_silu, (x, weight), grad_out)
    expected_silu_grads = torch.autograd.grad(
        expected_silu, (x, weight), grad_out
    )
    for name, grad, expected_grad in zip(
        ("x", "weight"), grads_silu, expected_silu_grads, strict=True
    ):
      test_utils.check_equivalence(
          grad.cpu(),
          expected_grad.cpu(),
          atol=tol,
          rtol=tol,
          check_name=f"Packed Causal Conv SiLU {name} Grad",
      )
    # Negative control: without segment_ids, the taps cross documents.
    with torch.no_grad():
      leaky = gated_delta._causal_conv(x, weight)  # pylint: disable=protected-access
    self.assertGreater(
        _relative_l2_error(leaky, expected),
        0.1,
        msg="Negative control failed: conv taps did not cross documents",
    )

  def test_causal_conv_adds_bf16_taps_in_fp32(self):
    """Tests that bf16 causal conv1d rounds an fp32 tap sum once, as F.conv1d."""
    torch.manual_seed(0)
    seq_len, channels, kernel_size = 64, 32, 4
    x = torch.randn(2, seq_len, channels).bfloat16()
    weight = torch.randn(channels, 1, kernel_size).bfloat16()
    # The products of bf16 values are exact in fp32; F.conv1d adds them in fp32.
    expected = F.conv1d(
        F.pad(x.float().transpose(1, 2), (kernel_size - 1, 0)),
        weight.float(),
        groups=channels,
    ).transpose(1, 2)
    out = gated_delta._causal_conv(  # pylint: disable=protected-access
        x.to(self.accelerator_device), weight.to(self.accelerator_device)
    ).cpu()
    self.assertEqual(out.dtype, torch.bfloat16)
    # Rounding once to bf16 is off by at most half an ulp, 2**-8 of the value.
    bound = expected.abs() * 2**-8 + 1e-5
    self.assertTrue(
        bool(((out.float() - expected).abs() <= bound).all()),
        msg="bf16 causal conv1d taps did not add in fp32",
    )
    # Negative control: adding the bf16 products in bf16 exceeds the bound.
    x_pad = F.pad(x, (0, 0, kernel_size - 1, 0))
    bf16_sum = sum(
        x_pad[:, k : k + seq_len] * weight[:, 0, k] for k in range(kernel_size)
    )
    self.assertFalse(
        bool(((bf16_sum.float() - expected).abs() <= bound).all()),
        msg="Negative control failed: the bf16 sum is within the bound",
    )

  def test_gated_delta_net_isolates_batch_sequences(self):
    """Tests that B=2 packed sequences in GatedDeltaNet equal two B=1 runs."""
    config = qwen3_5_tpu.model_registry("debugmodel", seq_len=128).model
    gdn = config.layers[0].delta_net.build()
    gdn.init_states(buffer_device=torch.device("cpu"))
    gdn = gdn.to(self.accelerator_device)

    seq_len = 16
    x0 = torch.randn(seq_len, config.dim, device=self.accelerator_device)
    x1 = torch.randn(seq_len, config.dim, device=self.accelerator_device)
    x_b2 = torch.cat([x0, x1], dim=0)

    with torch.no_grad():
      with set_unflatten_seq_len(seq_len):
        out0 = gdn(x0)
        out1 = gdn(x1)
        out_b2 = gdn(x_b2)
      # Negative control: if treated as 1 sequence of length 2*seq_len, the
      # second half changes significantly because conv1d and state bleed across.
      with set_unflatten_seq_len(2 * seq_len):
        out_bleed = gdn(x_b2)

    tol = 5e-3 if self._is_tpu() else 1e-5
    test_utils.check_equivalence(
        out_b2[:seq_len].cpu(),
        out0.cpu(),
        atol=tol,
        rtol=tol,
        check_name="GDN Batch Slice 0",
    )
    test_utils.check_equivalence(
        out_b2[seq_len:].cpu(),
        out1.cpu(),
        atol=tol,
        rtol=tol,
        check_name="GDN Batch Slice 1",
    )
    self.assertGreater(
        _relative_l2_error(out_bleed[seq_len:], out1),
        0.05,
        msg="Negative control failed: GDN bleed did not change slice 1",
    )

  def test_gated_delta_net_isolates_packed_documents(self):
    """Tests that documents packed into GatedDeltaNet rows equal single runs."""
    config = qwen3_5_tpu.model_registry("debugmodel", seq_len=128).model
    gdn = config.layers[0].delta_net.build()
    gdn.init_states(buffer_device=torch.device("cpu"))
    gdn = gdn.to(self.accelerator_device)

    # Two rows of 80 tokens. With 64-token chunks, documents start mid-chunk
    # and at a chunk start (64), and some span two chunks.
    rows = ((30, 34, 16), (50, 30))
    seq_len = 80
    doc_lens = [n for lens in rows for n in lens]
    starts = [sum(doc_lens[:i]) for i in range(len(doc_lens))]
    x = torch.randn(
        len(rows) * seq_len, config.dim, device=self.accelerator_device
    )
    positions = torch.cat([_packed_positions(lens) for lens in rows]).to(
        self.accelerator_device
    )

    with torch.no_grad():
      with set_unflatten_seq_len(seq_len):
        out_packed = gdn(x, positions=positions)
        out_leaky = gdn(x)
      # Without set_unflatten_seq_len, the [T] token stream is one row.
      out_stream = gdn(x, positions=positions)
      expected = []
      for s, n in zip(starts, doc_lens):
        with set_unflatten_seq_len(n):
          expected.append(gdn(x[s : s + n]))
      expected = torch.cat(expected)

    tol = 5e-3 if self._is_tpu() else 1e-5
    test_utils.check_equivalence(
        out_packed.cpu(),
        expected.cpu(),
        atol=tol,
        rtol=tol,
        check_name="GDN Packed Rows",
    )
    test_utils.check_equivalence(
        out_stream.cpu(),
        expected.cpu(),
        atol=tol,
        rtol=tol,
        check_name="GDN Packed Stream",
    )
    # Negative control: without positions, the convolution and the state leak
    # into the first tokens of each document that does not start a row.
    inner_starts = [30, 64, 130]
    self.assertGreater(
        _relative_l2_error(out_leaky[inner_starts], expected[inner_starts]),
        0.1,
        msg="Negative control failed: no GDN leak without positions",
    )

  def test_model_isolates_packed_documents(self):
    """Tests that documents packed into rows equal the documents run alone.

    Covers ``positions`` reaching every layer: the attention masks other
    documents, and the GatedDeltaNet restarts its convolution and state, in the
    forward pass and in every parameter gradient.
    """
    config = qwen3_5_tpu.model_registry("debugmodel", seq_len=128).model
    vocab_size = 512
    config.vocab_size = vocab_size
    config.tok_embeddings.num_embeddings = vocab_size
    config.lm_head.out_features = vocab_size
    config.layers = config.layers[:4]  # 3 GatedDeltaNet + 1 attention layers.
    torch.manual_seed(0)
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    # Moderate decays, as in the HF parity test: A_log's init reaches log(16),
    # with which a GatedDeltaNet forgets within a token or two (hiding state
    # leaks) and the fp32 chunk-cumulative decays grow to O(1000).
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
      for name, param in model.named_parameters():
        if "A_log" in name or "dt_bias" in name:
          param.copy_(0.2 * torch.randn(param.shape, generator=gen))
    model = model.to(self.accelerator_device)
    if config.enable_weight_tying:
      model.tok_embeddings.weight = model.lm_head.weight

    rows = ((30, 34, 16), (50, 30))
    seq_len = 80
    doc_lens = [n for lens in rows for n in lens]
    starts = [sum(doc_lens[:i]) for i in range(len(doc_lens))]
    gen = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, vocab_size, (len(rows) * seq_len,), generator=gen)
    loss_weights = torch.randn(len(rows) * seq_len, vocab_size, generator=gen)
    positions = torch.cat([_packed_positions(lens) for lens in rows])

    def forward_backward(tokens, positions, loss_weights, row_len):
      model.zero_grad()
      with set_unflatten_seq_len(row_len):
        logits = model(
            tokens.to(self.accelerator_device),
            positions=positions.to(self.accelerator_device),
        )
      (
          logits.float() * loss_weights.to(self.accelerator_device)
      ).sum().backward()
      grads = {
          name: param.grad.detach().float().cpu()
          for name, param in model.named_parameters()
          if param.grad is not None
      }
      return logits.detach().float().cpu(), grads

    logits_packed, grads_packed = forward_backward(
        tokens, positions, loss_weights, seq_len
    )
    logits_docs, grads_docs = [], {}
    for s, n in zip(starts, doc_lens):
      logits, grads = forward_backward(
          tokens[s : s + n], torch.arange(n), loss_weights[s : s + n], n
      )
      logits_docs.append(logits)
      for name, grad in grads.items():
        grads_docs[name] = grads_docs.get(name, 0) + grad
    logits_docs = torch.cat(logits_docs)

    tol = 5e-2 if self._is_tpu() else 1e-4
    test_utils.check_equivalence(
        logits_packed,
        logits_docs,
        atol=tol,
        rtol=tol,
        check_name="Packed Documents Logits",
    )
    grad_rel_bound = 2e-2 if self._is_tpu() else 5e-4
    self.assertEqual(set(grads_packed), set(grads_docs))
    for name, grad in grads_docs.items():
      self.assertLessEqual(
          _relative_l2_error(grads_packed[name], grad),
          grad_rel_bound,
          msg=f"Packed documents gradient mismatch for {name}",
      )

    # Negative control: positions that run on across documents leak attention
    # and GatedDeltaNet state into every document that does not start a row.
    later = torch.cat([
        torch.arange(s, s + n)
        for s, n in zip(starts, doc_lens)
        if s % seq_len != 0
    ])
    with torch.no_grad(), set_unflatten_seq_len(seq_len):
      logits_unpacked = model(
          tokens.to(self.accelerator_device),
          positions=torch.arange(seq_len)
          .repeat(len(rows))
          .to(self.accelerator_device),
      )
    leak = _relative_l2_error(logits_unpacked.cpu()[later], logits_docs[later])
    self.assertGreater(
        leak,
        0.01,
        msg="Negative control failed: unpacked positions did not leak",
    )

  def test_attention_isolates_packed_documents(self):
    """Tests that TpuQwen35Attention isolates packed documents via positions."""
    config = qwen3_5_tpu.model_registry("debugmodel", seq_len=128).model
    attn = config.layers[3].attention.build()
    attn.init_states(buffer_device=torch.device("cpu"))
    attn = attn.to(self.accelerator_device)

    len_a, len_b = 7, 9
    seq_len = len_a + len_b
    x = torch.randn(seq_len, config.dim, device=self.accelerator_device)
    positions_packed = torch.cat([
        torch.arange(len_a, device=self.accelerator_device),
        torch.arange(len_b, device=self.accelerator_device),
    ])
    positions_unpacked = torch.arange(seq_len, device=self.accelerator_device)

    with torch.no_grad():
      with set_unflatten_seq_len(seq_len):
        out_packed = attn(x, None, positions_packed)
        out_unpacked = attn(x, None, positions_unpacked)
      with set_unflatten_seq_len(len_a):
        out_a = attn(
            x[:len_a],
            None,
            torch.arange(len_a, device=self.accelerator_device),
        )
      with set_unflatten_seq_len(len_b):
        out_b = attn(
            x[len_a:],
            None,
            torch.arange(len_b, device=self.accelerator_device),
        )

    tol = 5e-3 if self._is_tpu() else 1e-5
    test_utils.check_equivalence(
        out_packed[:len_a].cpu(),
        out_a.cpu(),
        atol=tol,
        rtol=tol,
        check_name="Packed Doc A",
    )
    test_utils.check_equivalence(
        out_packed[len_a:].cpu(),
        out_b.cpu(),
        atol=tol,
        rtol=tol,
        check_name="Packed Doc B",
    )
    self.assertGreater(
        _relative_l2_error(out_unpacked[len_a:], out_b),
        0.1,
        msg="Negative control failed: unpacked positions did not change Doc B",
    )

  def test_splash_attention_matches_sdpa_on_packed_documents(self):
    """Tests Splash Attention patch on TpuQwen35Attention vs CPU SDPA."""
    if not self._is_tpu():
      self.skipTest("Splash Attention is a TPU-only Pallas kernel.")
    config = qwen3_5_tpu.model_registry("debugmodel").model
    batch, seq_len = 2, 256
    attn_cpu = config.layers[3].attention.build()
    attn_cpu.init_states(buffer_device=torch.device("cpu"))
    attn_cpu = attn_cpu.cpu()
    attn_tpu = copy.deepcopy(attn_cpu).to(self.accelerator_device)

    sa_config = SplashAttentionKernelConfig(
        use_splash_attention_kernel=True,
        sa_block_q=128,
        sa_block_kv=128,
        sa_block_kv_compute=128,
        sa_block_q_dkv=128,
        sa_block_kv_dkv=128,
        sa_block_kv_dkv_compute=128,
        sa_block_q_dq=128,
        sa_block_kv_dq=128,
    )
    workarounds.use_splash_attention_patch(attn_tpu, sa_config)

    x_cpu, x_tpu = self._create_input_tensor_cpu_device(
        (batch * seq_len, config.dim), requires_grad=True
    )
    positions_cpu = torch.cat([
        torch.arange(100),
        torch.arange(seq_len - 100),
        torch.arange(seq_len),
    ])
    positions_tpu = positions_cpu.to(self.accelerator_device)

    with set_unflatten_seq_len(seq_len):
      out_cpu = attn_cpu(x_cpu, None, positions_cpu)
      out_tpu = attn_tpu(x_tpu, None, positions_tpu)
      with torch.no_grad():
        out_cpu_unpacked = attn_cpu(
            x_cpu, None, torch.arange(seq_len).repeat(batch)
        )

    self.assertGreater(_relative_l2_error(out_cpu_unpacked, out_cpu), 0.1)
    self.assertLessEqual(_relative_l2_error(out_tpu, out_cpu), 1e-2)

    grad_out = torch.randn_like(out_cpu)
    out_cpu.backward(grad_out)
    out_tpu.backward(grad_out.to(self.accelerator_device))
    self.assertLessEqual(_relative_l2_error(x_tpu.grad, x_cpu.grad), 2.5e-2)

  def test_flavors_and_registry(self):
    """Tests the 35B-A3B geometry and the TPU modules of every flavor."""
    moe_35b = qwen3_5_tpu.model_registry("35B-A3B").model
    self.assertLen(moe_35b.layers, 40)
    self.assertEqual(moe_35b.dim, 2048)
    self.assertEqual(moe_35b.vocab_size, 248320)
    attn_layers = [l for l in moe_35b.layers if l.attention is not None]
    self.assertLen(attn_layers, 10)
    self.assertEqual(attn_layers[0].attention.head_dim, 256)
    self.assertEqual(attn_layers[0].attention.n_heads, 16)
    self.assertEqual(attn_layers[0].attention.n_kv_heads, 2)
    self.assertEqual(moe_35b.layers[0].moe.num_experts, 256)
    self.assertEqual(moe_35b.layers[0].moe.router.top_k, 8)

    for flavor in ("debugmodel", "debugmodel_moe", "35B-A3B"):
      with self.subTest(flavor=flavor):
        spec = qwen3_5_tpu.model_registry(flavor)
        self.assertIs(spec.parallelize_fn, parallelize_qwen3_5)
        self.assertIsNone(spec.pipelining_fn)
        self.assertIsNone(spec.model.vision_encoder)
        for layer in spec.model.layers:
          self.assertIsInstance(
              layer, tpu_model.TpuQwen35TransformerBlock.Config
          )
          if layer.attention is not None:
            self.assertIsInstance(
                layer.attention, tpu_model.TpuQwen35Attention.Config
            )
            self.assertIsInstance(
                layer.attention.inner_attention,
                SlidingWindowScaledDotProductInnerAttention.Config,
            )
          else:
            self.assertIsInstance(
                layer.delta_net, gated_delta.TpuGatedDeltaNet.Config
            )
            self.assertIsInstance(
                layer.delta_net.inner_gated_delta_net,
                gated_delta.TpuInnerGatedDeltaNet.Config,
            )
            for conv in (
                layer.delta_net.conv_q,
                layer.delta_net.conv_k,
                layer.delta_net.conv_v,
            ):
              self.assertIn("weight", conv.param_init or {})
          if layer.moe is not None:
            self.assertIsNone(layer.moe.load_balance_coeff)

    with self.assertRaisesRegex(ValueError, "attn_backend='sdpa'"):
      qwen3_5_tpu.model_registry("debugmodel", attn_backend="flex")
    with self.assertRaisesRegex(ValueError, "exceeds max context length"):
      qwen3_5_tpu.model_registry("debugmodel", seq_len=262145)

  def test_35b_recipe(self):
    """Tests the 35B-A3B recipe, also as parsed from the command line."""
    cfg = config_registry.qwen35_moe_35b()
    self.assertEqual(cfg.model_spec.flavor, "35B-A3B")
    self.assertEqual(cfg.training.dtype, "bfloat16")
    self.assertTrue(cfg.parallelism.use_simple_fsdp)
    self.assertEqual(cfg.parallelism.fsdp_reshard_after_forward, "always")
    with torch.device("meta"):
      model = cfg.model_spec.model.build()
    self.assertLen(model.layers, 40)
    self.assertIsInstance(
        model.layers["0"].moe.routed_experts.inner_experts, TpuGroupedExperts
    )

    parsed = ConfigManager().parse_args([
        "--module=torchtitan.experiments.tpu.qwen3_5",
        "--config=qwen35_moe_35b",
        "--training.num_tokens_per_microbatch_per_dp_rank=8192",
    ])
    self.assertEqual(parsed.model_spec.flavor, "35B-A3B")
    self.assertEqual(
        parsed.training.num_tokens_per_microbatch_per_dp_rank, 8192
    )


if __name__ == "__main__":
  absltest.main()
