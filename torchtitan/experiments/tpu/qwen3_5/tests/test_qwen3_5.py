"""Tests for Qwen3.5 (text-only Dense and MoE) on CPU and TPU devices.

Covers:
1. HuggingFace reference parity (``Qwen3_5ForCausalLM`` and
   ``Qwen3_5MoeForCausalLM``) through ``Qwen35StateDictAdapter`` for forward
   logits, cross-entropy loss, and every parameter gradient (with negative
   controls), for dense, MoE, and MoE with ``TpuGroupedExpertsConverter``;
   plus the adapter round trip.
2. ``chunked_gated_delta`` against the exact per-token recurrence, also with
   packed documents, and the slice-based causal convolution's backward against
   autograd of ``conv1d``, also with packed documents, and in bf16 against the
   accuracy of the autograd it replaces.
3. Row isolation in the GatedDeltaNet and packed-document isolation in the
   causal convolution, the GatedDeltaNet, the attention and the whole model
   (with negative controls), plus Splash parity on TPU.
4. Flavor, registry, recipe and MFU-accounting checks.
5. Expert parallelism: ``TpuQwen35TokenDispatcher`` EP=2 parity with EP=1 and
   a static-shape FX trace, and per-mesh optimizer param groups.
"""

import copy
import dataclasses
import functools
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.fx.experimental.proxy_tensor import make_fx
import torch.nn.functional as F
from torchtitan.components.optimizer import default_adamw
from torchtitan.config.manager import ConfigManager
from torchtitan.config.override import apply_overrides
from torchtitan.config.override import OverrideConfig
from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
    annotate_module_fqns,
    compute_parameter_gradients,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    tag_with_memory_policy_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.graph_trainer import qwen3_5 as gt_qwen3_5
from torchtitan.experiments.tpu.graph_trainer import tpu_graph_trainer as tpu_gt_mod
from torchtitan.experiments.tpu.graph_trainer.config import apply_tpu_graph_trainer_model_transforms
from torchtitan.experiments.tpu.graph_trainer.graph_split import GraphSplitter
from torchtitan.experiments.tpu.graph_trainer.passes.insert_ep_chunk_barriers import (
    apply_ep_chunk_barrier_pass,
)
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    _optimization_barrier,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer
from torchtitan.experiments.tpu.graph_trainer.tpu_memory_policy import (
    EP_DISPATCH_REGION,
    ROUTING_REGION,
    select_ep_dispatch,
    select_route_permutations,
)
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.kernels import ragged_gather_tc
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.muse_glimmer.attention import SlidingWindowScaledDotProductInnerAttention
import torchtitan.experiments.tpu.qwen3_5 as qwen3_5_tpu
from torchtitan.experiments.tpu.qwen3_5 import config_registry
from torchtitan.experiments.tpu.qwen3_5 import gated_delta
from torchtitan.experiments.tpu.qwen3_5 import model as tpu_model
from torchtitan.experiments.tpu.qwen3_5 import tokamax_gated_delta
from torchtitan.experiments.tpu.qwen3_5 import token_dispatcher as qwen35_token_dispatcher
from torchtitan.experiments.tpu.qwen3_5.infra.parallelize import parallelize_qwen3_5
from torchtitan.experiments.tpu.token_dispatcher import TpuAllToAllTokenDispatcher
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
      dict(
          testcase_name="_moe_pallas_paired_grouped_mm",
          flavor="debugmodel_moe",
          use_tpu_grouped_experts=True,
          use_pallas_paired_grouped_mm=True,
          tile_m=256,
      ),
      dict(
          testcase_name="_moe_pallas_paired_grouped_mm_tile512",
          flavor="debugmodel_moe",
          use_tpu_grouped_experts=True,
          use_pallas_paired_grouped_mm=True,
          tile_m=512,
      ),
  )
  def test_qwen3_5_hf_reference_parity(
      self,
      flavor: str,
      use_tpu_grouped_experts: bool,
      use_pallas_paired_grouped_mm: bool = False,
      tile_m: int = 256,
  ):
    """Tests forward logits, loss, and all gradients against HF Qwen3.5."""
    if use_pallas_paired_grouped_mm and not self._is_tpu():
      self.skipTest("Pallas grouped matmuls are TPU-only.")
    is_moe = flavor == "debugmodel_moe"
    hf_model = self._build_hf_reference(is_moe=is_moe)
    converters = (
        [TpuGroupedExpertsConverter.Config()] if use_tpu_grouped_experts else []
    )
    tt_config = qwen3_5_tpu.model_registry(
        flavor, seq_len=128, converters=converters
    ).model
    if use_pallas_paired_grouped_mm:
      # The expert kernels of `_use_tpu_moe_kernels`, on canonical weights so
      # that the gradients compare by name.
      for _, experts, _, _ in tt_config.traverse(TpuGroupedExperts.Config):
        experts.use_pallas_grouped_mm = True
        experts.use_pallas_paired_grouped_mm = True
        experts.pallas_grouped_mm_tile_m = tile_m
        experts.pallas_grouped_mm_tgmm_tiling = (tile_m, 0, 0)
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

  def test_causal_conv_silu_partial_input_grads(self):
    """Tests the conv backward when only the input or the weight needs grad."""
    torch.manual_seed(0)
    channels, kernel_size = 8, 4
    x = torch.randn(2, 11, channels, dtype=torch.float64)
    weight = torch.randn(channels, 1, kernel_size, dtype=torch.float64)
    grad = torch.randn(2, 11, channels, dtype=torch.float64)
    for name, x_grad in (("x", True), ("weight", False)):
      with self.subTest(grad_of=name):
        x_in = x.clone().requires_grad_(x_grad)
        w_in = weight.clone().requires_grad_(not x_grad)
        leaf = x_in if x_grad else w_in
        expected = F.silu(
            F.conv1d(
                F.pad(x_in.transpose(1, 2), (kernel_size - 1, 0)),
                w_in,
                groups=channels,
            ).transpose(1, 2)
        )
        (actual_grad,) = torch.autograd.grad(
            gated_delta.causal_conv_silu(x_in, w_in), leaf, grad
        )
        (expected_grad,) = torch.autograd.grad(expected, leaf, grad)
        torch.testing.assert_close(actual_grad, expected_grad)

  @parameterized.named_parameters(
      dict(testcase_name="_rows", packed=False),
      dict(testcase_name="_packed_documents", packed=True),
  )
  def test_causal_conv_silu_bf16_backward_accuracy(self, packed: bool):
    """Tests the bf16 conv backward against fp64 over 16K-token dw sums.

    The slice-based backward must be as accurate as the autograd it replaces,
    which multiplies in fp32 and sums dw in fp32. Adding the dw products in
    bf16 is not: torch_tpu disallows XLA excess precision, so a bf16 sum rounds
    every add.
    """
    torch.manual_seed(0)
    bs, seq_len, channels, kernel_size = 8, 2048, 128, 4
    # Biased activations and gradients (correlated, as in the model) make the
    # dw sums grow with the number of tokens.
    x = (torch.randn(bs, seq_len, channels) + 0.5).bfloat16()
    weight = (0.5 * torch.randn(channels, 1, kernel_size)).bfloat16()
    grad = (torch.randn(bs, seq_len, channels) + 0.25).bfloat16()
    segment_ids = None
    if packed:
      doc_lens = (700, 1, 3, 500, 844)
      segment_ids = segment_ids_from_positions(_packed_positions(doc_lens))
      segment_ids = segment_ids.expand(bs, -1)

    def conv_grads(fn, device, dtype):
      x_in = x.to(device, dtype).requires_grad_()
      w_in = weight.to(device, dtype).requires_grad_()
      seg = None if segment_ids is None else segment_ids.to(device)
      return torch.autograd.grad(
          fn(x_in, w_in, seg), (x_in, w_in), grad.to(device, dtype)
      )

    def autograd_conv_silu(x_in, w_in, seg):
      return F.silu(gated_delta._causal_conv(x_in, w_in, seg))  # pylint: disable=protected-access

    # fp64 on the bf16 values: only the backward's own rounding counts.
    # Call `_CausalConvSilu.apply` directly so whole-row slice backward is not
    # bypassed on TPU when `causal_conv1d` is installed.
    expected_grads = conv_grads(autograd_conv_silu, "cpu", torch.float64)
    dev = self.accelerator_device
    grads = conv_grads(
        gated_delta._CausalConvSilu.apply, dev, torch.bfloat16  # pylint: disable=protected-access
    )
    autograd_grads = conv_grads(autograd_conv_silu, dev, torch.bfloat16)
    for name, grad_bf16, autograd_bf16, expected_grad in zip(
        ("x", "weight"), grads, autograd_grads, expected_grads, strict=True
    ):
      self.assertEqual(grad_bf16.dtype, torch.bfloat16)
      err = _relative_l2_error(grad_bf16, expected_grad)
      autograd_err = _relative_l2_error(autograd_bf16, expected_grad)
      self.assertLessEqual(
          err,
          1.25 * autograd_err,
          msg=f"d{name}: {err:.2e} vs autograd {autograd_err:.2e}",
      )
      self.assertLess(err, 1e-2, msg=f"d{name}: {err:.2e}")

  def test_causal_conv_silu_dispatches_to_pallas_when_supported(self):
    """Tests whole-row TPU dispatch to Pallas causal_conv1d and fallbacks."""
    dev = self.accelerator_device
    x = torch.randn(2, 64, 128, device=dev, dtype=torch.bfloat16)
    weight = torch.randn(128, 1, 4, device=dev, dtype=torch.bfloat16) * 0.5
    seg = segment_ids_from_positions(_packed_positions((24, 40))).expand(2, -1)
    seg = seg.to(dev)

    if self._is_tpu() and gated_delta.causal_conv1d is not None:
      with mock.patch.object(
          gated_delta.causal_conv1d,
          "causal_conv_silu",
          wraps=gated_delta.causal_conv1d.causal_conv_silu,
      ) as pallas_spy:
        out_pallas = gated_delta.causal_conv_silu(x, weight, None)
        self.assertEqual(pallas_spy.call_count, 1)
        _ = gated_delta.causal_conv_silu(x, weight, seg)
        _ = gated_delta.causal_conv_silu(x[:, :7, :], weight, None)
        self.assertEqual(pallas_spy.call_count, 1)
      out_slice = gated_delta._CausalConvSilu.apply(x, weight, None)  # pylint: disable=protected-access
      self.assertLess(_relative_l2_error(out_pallas, out_slice), 5e-3)
    else:
      out_fallback = gated_delta.causal_conv_silu(x, weight, None)
      out_slice = gated_delta._CausalConvSilu.apply(x, weight, None)  # pylint: disable=protected-access
      torch.testing.assert_close(out_fallback, out_slice)

  def test_tpu_rms_norm_gated_dispatches_to_pallas_when_supported(self):
    """Tests TpuRMSNormGated Pallas dispatch at >=2048 rows and fallbacks."""
    dev = self.accelerator_device
    torch.manual_seed(17)
    x_cpu = torch.randn(512, 4, 128, dtype=torch.bfloat16)
    gate_cpu = torch.randn(512, 4, 128, dtype=torch.bfloat16)
    w_cpu = (1.0 + 0.1 * torch.randn(128)).to(torch.bfloat16)
    dy_cpu = torch.randn(512, 4, 128, dtype=torch.bfloat16)

    ref_norm = gated_delta.RMSNormGated(
        gated_delta.RMSNormGated.Config(dim=128)
    )
    with torch.no_grad():
      ref_norm.weight.copy_(w_cpu)
    ref_norm = ref_norm.to(dev)
    x_ref = x_cpu.to(dev).requires_grad_(True)
    gate_ref = gate_cpu.to(dev).requires_grad_(True)
    out_ref = ref_norm(x_ref, gate_ref)
    grads_ref = torch.autograd.grad(
        out_ref, (x_ref, gate_ref, ref_norm.weight), dy_cpu.to(dev)
    )

    tpu_norm = gated_delta.TpuRMSNormGated(
        gated_delta.TpuRMSNormGated.Config(dim=128)
    )
    with torch.no_grad():
      tpu_norm.weight.copy_(w_cpu)
    tpu_norm = tpu_norm.to(dev)
    x_dev = x_cpu.to(dev).requires_grad_(True)
    gate_dev = gate_cpu.to(dev).requires_grad_(True)

    # Verify nn.Module forward hooks run (not bypassed by free helper).
    hook_calls = 0

    def _post_hook(_mod, _inp, _out):
      nonlocal hook_calls
      hook_calls += 1

    handle = tpu_norm.register_forward_hook(_post_hook)
    self.addCleanup(handle.remove)

    if self._is_tpu() and gated_delta.rms_norm_gated is not None:
      with mock.patch.object(
          gated_delta.rms_norm_gated,
          "rms_norm_gated",
          wraps=gated_delta.rms_norm_gated.rms_norm_gated,
      ) as pallas_spy:
        out_dev = tpu_norm(x_dev, gate_dev)
        self.assertEqual(pallas_spy.call_count, 1)
        # Rows = 256 * 4 = 1024 (< 2048 block size) falls back cleanly.
        _ = tpu_norm(x_dev[:256], gate_dev[:256])
        self.assertEqual(pallas_spy.call_count, 1)
    else:
      out_dev = tpu_norm(x_dev, gate_dev)
      _ = tpu_norm(x_dev[:256], gate_dev[:256])
    self.assertEqual(hook_calls, 2)

    grads_dev = torch.autograd.grad(
        out_dev, (x_dev, gate_dev, tpu_norm.weight), dy_cpu.to(dev)
    )
    self.assertLess(_relative_l2_error(out_dev, out_ref), 1e-2)
    for g_actual, g_expected in zip(grads_dev, grads_ref, strict=True):
      self.assertLess(_relative_l2_error(g_actual, g_expected), 1e-2)

  def test_causal_conv_silu_backward_avoids_scatter_adds(self):
    """Tests that the conv backward does not add slices into zero buffers.

    Autograd of the slice-based forward adds every tap's input gradient into
    a zero buffer of the padded input (``slice_backward``), which XLA emits as
    slow ``dynamic-update-slice`` adds. The traced backward of
    ``causal_conv_silu`` must have no such op, with or without documents.
    """
    scatter_ops = {
        torch.ops.aten.slice_backward.default,
        torch.ops.aten.slice_scatter.default,
        torch.ops.aten.index_put.default,
        torch.ops.aten.scatter_add.default,
    }
    x = torch.randn(2, 16, 8, requires_grad=True)
    weight = torch.randn(8, 1, 4, requires_grad=True)
    grad = torch.randn(2, 16, 8)
    segment_ids = segment_ids_from_positions(_packed_positions((5, 11)))

    def autograd_conv_silu(x_in, w_in, seg):
      return F.silu(gated_delta._causal_conv(x_in, w_in, seg))  # pylint: disable=protected-access

    def backward_ops(conv_silu, seg):
      def grads(x_in, w_in, grad_in):
        return torch.autograd.grad(
            conv_silu(x_in, w_in, seg), (x_in, w_in), grad_in
        )

      graph = make_fx(grads)(x, weight, grad).graph
      return {node.target for node in graph.nodes if node.op == "call_function"}

    for seg in (None, segment_ids.expand(2, -1)):
      with self.subTest(packed=seg is not None):
        self.assertEmpty(
            backward_ops(gated_delta.causal_conv_silu, seg) & scatter_ops
        )
        # Negative control: autograd of the forward does use them.
        self.assertNotEmpty(backward_ops(autograd_conv_silu, seg) & scatter_ops)

  @parameterized.named_parameters(
      # Slow decays: every token's gradient matters.
      dict(testcase_name="_slow_decays", max_decay_rate=0.2),
      # Qwen3.5's initialization: A_log = log(U(0.01, 16)).
      dict(testcase_name="_init_decays", max_decay_rate=16.0),
  )
  def test_chunked_gated_delta_compiled_grads_match_eager(
      self, max_decay_rate: float
  ):
    """Tests chunked_gated_delta's gradients under torch.compile vs eager.

    TpuGraphTrainer compiles the joint graph. On TPU, the gradient through a
    slice of the decays' cumsum (`cumsum(-1)[..., -1:]`, the chunk totals)
    that is also used elementwise came out wrong when compiled (dg relative
    error 3.8, and NaN losses at step 1 on the debug model), while eager was
    right.
    """
    torch.manual_seed(0)
    self.addCleanup(torch._dynamo.reset)  # pylint: disable=protected-access
    dev = self.accelerator_device
    shape = (2, 128, 4)
    q, k, v = (torch.randn(*shape, 128, device=dev) for _ in range(3))
    rate = torch.empty(shape[-1], device=dev).uniform_(0.01, max_decay_rate)
    g = -rate * F.softplus(torch.randn(*shape, device=dev) + 1.0)
    beta = torch.sigmoid(torch.randn(*shape, device=dev))
    grad_out = torch.randn(*shape, 128, device=dev)

    def run(fn):
      inputs = [x.detach().clone().requires_grad_() for x in (q, k, v, g, beta)]
      out = fn(*inputs)
      return [out, *torch.autograd.grad(out, inputs, grad_out)]

    backend = "tpu" if self._is_tpu() else "aot_eager"
    expected = run(gated_delta.chunked_gated_delta)
    actual = run(
        torch.compile(
            gated_delta.chunked_gated_delta, backend=backend, fullgraph=True
        )
    )
    for name, a_t, e_t in zip(
        ("out", "dq", "dk", "dv", "dg", "dbeta"), actual, expected, strict=True
    ):
      self.assertTrue(torch.isfinite(a_t).all().item(), msg=name)
      # TPU7x / v5e: <= 5e-7 with the fix.
      self.assertLessEqual(
          _relative_l2_error(a_t, e_t), 1e-4, msg=f"compiled {name}"
      )

  def test_causal_conv_silu_compiles_fullgraph(self):
    """Tests causal_conv_silu under torch.compile(fullgraph=True).

    The per-layer compile mode compiles each transformer block with
    ``fullgraph=True``, so the custom autograd.Function must trace without
    graph breaks.
    """
    torch.manual_seed(0)
    self.addCleanup(torch._dynamo.reset)  # pylint: disable=protected-access
    x = torch.randn(2, 16, 8, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(8, 1, 4, dtype=torch.float64, requires_grad=True)
    grad = torch.randn(2, 16, 8, dtype=torch.float64)
    segment_ids = segment_ids_from_positions(_packed_positions((5, 11)))
    compiled = torch.compile(
        gated_delta.causal_conv_silu, backend="aot_eager", fullgraph=True
    )
    for seg in (None, segment_ids.expand(2, -1)):
      with self.subTest(packed=seg is not None):
        out = compiled(x, weight, seg)
        expected = gated_delta.causal_conv_silu(x, weight, seg)
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(
            torch.autograd.grad(out, (x, weight), grad),
            torch.autograd.grad(expected, (x, weight), grad),
        )

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
    # Config.reset_at_documents=False (pretraining) ignores positions in both
    # forward and backward: whole rows match gdn(x, positions=None).
    gdn_no_reset = dataclasses.replace(
        config.layers[0].delta_net, reset_at_documents=False
    ).build()
    gdn_no_reset.init_states(buffer_device=torch.device("cpu"))
    gdn_no_reset.load_state_dict(gdn.state_dict())
    gdn_no_reset = gdn_no_reset.to(self.accelerator_device)
    self.assertFalse(gdn_no_reset.reset_at_documents)

    grad_out = torch.randn_like(out_leaky)
    x_ref = x.detach().clone().requires_grad_(True)
    x_no_reset = x.detach().clone().requires_grad_(True)
    gdn.zero_grad()
    gdn_no_reset.zero_grad()
    with set_unflatten_seq_len(seq_len):
      out_ref_bwd = gdn(x_ref)
      out_no_reset = gdn_no_reset(x_no_reset, positions=positions)
    out_ref_bwd.backward(grad_out)
    out_no_reset.backward(grad_out)

    test_utils.check_equivalence(
        out_no_reset.detach().cpu(),
        out_ref_bwd.detach().cpu(),
        atol=tol,
        rtol=tol,
        check_name="GDN Without Document Resets",
    )
    grad_tol = 2e-2 if self._is_tpu() else 5e-4
    self.assertLessEqual(
        _relative_l2_error(x_no_reset.grad, x_ref.grad),
        grad_tol,
        msg="GDN reset_at_documents=False input grad mismatch",
    )
    ref_grads = dict(gdn.named_parameters())
    for name, param in gdn_no_reset.named_parameters():
      self.assertLessEqual(
          _relative_l2_error(param.grad, ref_grads[name].grad),
          grad_tol,
          msg=f"GDN reset_at_documents=False param grad mismatch for {name}",
      )

  def test_per_row_segment_ids(self):
    """Tests per-row 1-based document ids capped at max_num_segments."""
    positions = torch.cat([
        _packed_positions((2, 1, 3, 2)),  # row 0: 4 documents
        _packed_positions((5, 3)),  # row 1: 2 documents
    ])
    stream_ids = segment_ids_from_positions(positions).reshape(2, 8)
    torch.testing.assert_close(
        gated_delta.per_row_segment_ids(stream_ids, 3),
        torch.tensor(
            [[1, 1, 2, 3, 3, 3, 3, 3], [1, 1, 1, 1, 1, 2, 2, 2]],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        gated_delta.per_row_segment_ids(stream_ids, 8)[0],
        torch.tensor([1, 1, 2, 3, 3, 3, 4, 4], dtype=torch.int32),
    )

  def test_gated_delta_net_document_ids_reach_conv_and_recurrence(self):
    """Tests that the kernel's document ids drive the conv and recurrence."""
    config = qwen3_5_tpu.model_registry("debugmodel", seq_len=128).model
    gdn = config.layers[0].delta_net.build()
    gdn.init_states(buffer_device=torch.device("cpu"))
    gdn = gdn.to(self.accelerator_device)
    dev = self.accelerator_device
    # One 80-token row of 4 documents; a kernel that keeps 2 apart must merge
    # documents 2-4 in both the convolution and the recurrence, exactly as if
    # they were one document.
    seq_len, cap = 80, 2
    x = torch.randn(seq_len, config.dim, device=dev)
    positions = _packed_positions((20, 15, 25, 20)).to(dev)
    merged = _packed_positions((20, 60)).to(dev)
    with torch.no_grad(), set_unflatten_seq_len(seq_len):
      expected = gdn(x, positions=merged)
      uncapped = gdn(x, positions=positions)
      gdn.inner_gated_delta_net.kernel.document_ids = functools.partial(
          gated_delta.per_row_segment_ids, max_num_segments=cap
      )
      capped = gdn(x, positions=positions)

    tol = 5e-3 if self._is_tpu() else 1e-5
    test_utils.check_equivalence(
        capped.cpu(),
        expected.cpu(),
        atol=tol,
        rtol=tol,
        check_name="GDN Capped Document Ids",
    )
    # Negative control: keeping every document apart differs at the merges.
    self.assertGreater(
        _relative_l2_error(uncapped[35:39], expected[35:39]),
        0.1,
        msg="Negative control failed: capping the documents changed nothing",
    )

  def test_use_tokamax_gated_delta(self):
    """Tests the recipe helper and override resolution for Tokamax GDN."""
    cfg = config_registry.qwen35_moe_35b()
    self.assertNotIn(
        config_registry.TOKAMAX_GATED_DELTA_OVERRIDE, cfg.override.imports
    )
    config_registry._use_tokamax_gated_delta(cfg)  # pylint: disable=protected-access
    config_registry._use_tokamax_gated_delta(cfg)  # pylint: disable=protected-access
    self.assertEqual(
        cfg.override.imports.count(
            config_registry.TOKAMAX_GATED_DELTA_OVERRIDE
        ),
        1,
    )
    # Explicit kwargs replace a bare entry, and a bare call preserves existing
    # kwargs rather than appending a second entry that resets them.
    config_registry._use_tokamax_gated_delta(  # pylint: disable=protected-access
        cfg, min_log_decay=-4.0, max_num_segments=24
    )
    config_registry._use_tokamax_gated_delta(cfg)  # pylint: disable=protected-access
    self.assertEqual(
        cfg.override.imports,
        [
            config_registry.FUSED_LOSS_OVERRIDE,
            (
                config_registry.TOKAMAX_GATED_DELTA_OVERRIDE,
                {"min_log_decay": -4.0, "max_num_segments": 24},
            ),
        ],
    )
    apply_overrides(
        OverrideConfig(imports=[cfg.override.imports[-1]]),
        cfg,
    )
    kernels = [
        layer.delta_net.inner_gated_delta_net.kernel
        for layer in cfg.model_spec.model.layers
        if layer.delta_net is not None
    ]
    self.assertLen(kernels, 30)
    for kernel_cfg in kernels:
      self.assertEqual(
          type(kernel_cfg).__qualname__, "TokamaxGatedDeltaKernel.Config"
      )
      self.assertEqual(kernel_cfg.min_log_decay, -4.0)
      self.assertEqual(kernel_cfg.max_num_segments, 24)
    # On a TPU that runs the Mosaic kernels, building the recipe's model fails
    # at once (before initialization or checkpoint loading) without the KDA op.
    with (
        mock.patch.object(
            tokamax_gated_delta, "supports_current_tpu", return_value=True
        ),
        mock.patch.object(
            tokamax_gated_delta,
            "_import_kda",
            return_value=(None, ImportError("no KDA")),
        ),
    ):
      with self.assertRaisesRegex(ImportError, r"tokamax>=0\.0\.17"):
        kernels[0].build()

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
    config = qwen3_5_tpu.model_registry("debugmodel", seq_len=256).model
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
        self.assertIs(
            spec.post_optimizer_build_fn,
            qwen3_5_tpu.post_optimizer_build_qwen3_5,
        )
        self.assertIsInstance(spec.model, tpu_model.TpuQwen35Model.Config)
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
            self.assertIsInstance(
                layer.delta_net.norm, gated_delta.TpuRMSNormGated.Config
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

  def _assert_uses_tpu_moe_kernels(
      self, model_config, *, tile_m: int = 256
  ) -> None:
    """Asserts that `_use_tpu_moe_kernels` reached every MoE layer.

    Args:
      model_config: The model config to check.
      tile_m: The expected grouped-matmul and dWeight row tile.
    """
    num_moe = sum(layer.moe is not None for layer in model_config.layers)
    experts = [
        c for _, c, _, _ in model_config.traverse(TpuGroupedExperts.Config)
    ]
    dispatchers = [
        c
        for _, c, _, _ in model_config.traverse(
            TpuAllToAllTokenDispatcher.Config
        )
    ]
    # A traversal that stops matching would silently drop the kernels.
    self.assertGreater(num_moe, 0)
    self.assertLen(experts, num_moe)
    self.assertLen(dispatchers, num_moe)
    for experts_config in experts:
      self.assertTrue(experts_config.use_pallas_grouped_mm)
      self.assertTrue(experts_config.use_pallas_paired_grouped_mm)
      self.assertTrue(experts_config.use_kernel_native_weight_layout)
      self.assertEqual(experts_config.pallas_grouped_mm_tile_m, tile_m)
      self.assertEqual(
          experts_config.pallas_grouped_mm_tgmm_tiling, (tile_m, 0, 0)
      )
    for dispatcher in dispatchers:
      self.assertTrue(dispatcher.fuse_route_score_backward)

  def test_tpu_moe_kernels_debug_recipe(self):
    """Tests that the debug recipe selects the 35B recipe's expert kernels."""
    cfg = config_registry.qwen35_moe_debugmodel_tpu_moe_kernels()
    self._assert_uses_tpu_moe_kernels(cfg.model_spec.model)

  def test_35b_recipe(self):
    """Tests the 35B-A3B recipe, also as parsed from the command line."""
    cfg = config_registry.qwen35_moe_35b()
    self.assertEqual(cfg.model_spec.flavor, "35B-A3B")
    self.assertEqual(cfg.training.dtype, "float32")
    self.assertEqual(cfg.training.mixed_precision_param, "bfloat16")
    self.assertEqual(cfg.training.mixed_precision_reduce, "float32")
    self.assertTrue(cfg.parallelism.use_simple_fsdp)
    self.assertEqual(cfg.parallelism.fsdp_reshard_after_forward, "always")
    self.assertTrue(cfg.compile.compile_grad_norm)
    self.assertEqual(cfg.tpu_config.eager_mode, "DEFER_AND_FUSE")
    self._assert_uses_tpu_moe_kernels(cfg.model_spec.model)
    splash = cfg.splash_attention_kernel
    self.assertTrue(splash.use_splash_attention_kernel)
    self.assertEqual(
        (
            splash.sa_block_q,
            splash.sa_block_kv,
            splash.sa_block_q_dkv,
            splash.sa_block_kv_dkv,
            splash.sa_block_q_dq,
            splash.sa_block_kv_dq,
        ),
        (1024,) * 6,
    )
    self.assertIsInstance(cfg.loss, PallasCrossEntropyLoss.Config)
    self.assertEqual(
        (cfg.loss.b_block_size, cfg.loss.h_block_size, cfg.loss.v_block_size),
        (1024, 512, 2048),
    )
    self.assertIn(config_registry.FUSED_LOSS_OVERRIDE, cfg.override.imports)
    self.assertEqual(
        cfg.libtpu_init_flags, {"xla_tpu_scoped_vmem_limit_kib": "65472"}
    )
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
    self.assertIsInstance(parsed.loss, PallasCrossEntropyLoss.Config)
    self.assertEqual(
        parsed.libtpu_init_flags, {"xla_tpu_scoped_vmem_limit_kib": "65472"}
    )

  def test_35b_benchmark_graph_trainer_config(self):
    """Tests the benchmark recipe as train.py hands it to TpuGraphTrainer."""
    gt_cfg = config_registry.qwen35_moe_35b_split_tokamax_benchmark()
    self.assertIsInstance(gt_cfg, TpuGraphTrainer.Config)
    self.assertIs(type(gt_cfg)._owner, TpuGraphTrainer)
    apply_tpu_graph_trainer_model_transforms(gt_cfg)
    self.assertIs(
        gt_cfg.model_spec.parallelize_fn, gt_qwen3_5.parallelize_qwen3_5
    )
    self.assertIsNone(gt_cfg.model_spec.pipelining_fn)
    self.assertEqual(gt_cfg.compile.memory_policy, "tpu_selective")
    self.assertIsInstance(gt_cfg.loss, PallasCrossEntropyLoss.Config)
    self.assertEqual(
        (
            gt_cfg.loss.b_block_size,
            gt_cfg.loss.h_block_size,
            gt_cfg.loss.v_block_size,
        ),
        (1024, 2048, 640),
    )
    # The conversion re-wraps the model config: the kernels and GDN reset
    # setting must survive it.
    self._assert_uses_tpu_moe_kernels(gt_cfg.model_spec.model, tile_m=512)
    self.assertEqual(
        [
            d.reset_at_documents
            for _, d, _, _ in gt_cfg.model_spec.model.traverse(
                gated_delta.TpuGatedDeltaNet.Config
            )
        ],
        [False] * 30,
    )
    splitter = GraphSplitter.from_trainer_config(
        gt_cfg,
        optimizer_in_graph=(gt_cfg.compile.optimizer_compile == "in_graph"),
    )
    self.assertTrue(splitter.enabled)
    self.assertEqual(splitter.n_layers, 40)

  def test_graph_trainer_debug_recipe(self):
    """Tests the GraphTrainer debug recipe's split and selective remat."""
    gt_cfg = config_registry.graph_trainer_qwen35_moe_debugmodel()
    self.assertIsInstance(gt_cfg, TpuGraphTrainer.Config)
    apply_tpu_graph_trainer_model_transforms(gt_cfg)
    self.assertIs(
        gt_cfg.model_spec.parallelize_fn, gt_qwen3_5.parallelize_qwen3_5
    )
    self.assertEqual(gt_cfg.model_spec.flavor, "debugmodel_moe")
    self.assertNotIn(
        config_registry.TOKAMAX_GATED_DELTA_OVERRIDE, gt_cfg.override.imports
    )
    compile_cfg = gt_cfg.compile
    self.assertEqual(compile_cfg.mode, "aot_fx_trace")
    self.assertEqual(compile_cfg.memory_policy, "tpu_selective")
    self.assertEqual(compile_cfg.save_regions, ("moe.routing",))
    self.assertEqual(compile_cfg.graph_split.forward_layers_per_executable, 1)
    self.assertEqual(compile_cfg.graph_split.backward_layers_per_executable, 1)
    splitter = GraphSplitter.from_trainer_config(
        gt_cfg,
        optimizer_in_graph=(compile_cfg.optimizer_compile == "in_graph"),
    )
    self.assertTrue(splitter.enabled)
    self.assertEqual(splitter.n_layers, 4)
    parsed = ConfigManager().parse_args([
        "--module=torchtitan.experiments.tpu.qwen3_5",
        "--config=graph_trainer_qwen35_moe_debugmodel",
    ])
    self.assertIsInstance(parsed, TpuGraphTrainer.Config)

  def test_graph_trainer_parallelize_rejects_tp_cp_pp(self):
    """Tests that TP, CP, and PP fail before the model is touched."""
    for name in ("tp", "cp", "pp"):
      enabled = dict(
          tp_enabled=False,
          cp_enabled=False,
          pp_enabled=False,
          ep_enabled=False,
      )
      enabled[f"{name}_enabled"] = True
      with (
          self.subTest(name),
          self.assertRaisesRegex(NotImplementedError, "TP, CP, or PP"),
      ):
        gt_qwen3_5.parallelize_qwen3_5(
            None,
            parallel_dims=SimpleNamespace(**enabled),
            training=None,
            parallelism=None,
            compile_config=None,
            ac_config=None,
            dump_folder="",
        )

  def test_35b_recipes(self):
    """Tests the 35B-A3B recipes, their TPU kernels and MFU accounting."""
    gt_spec = gt_qwen3_5.model_registry("35B-A3B")
    self.assertEqual(gt_spec.name, "qwen3_5")
    self.assertIs(gt_spec.parallelize_fn, gt_qwen3_5.parallelize_qwen3_5)

    self.assertNotIn(
        config_registry.TOKAMAX_GATED_DELTA_OVERRIDE,
        config_registry.qwen35_moe_35b().override.imports,
    )

    # The benchmark recipe: ZeRO-3, graph split (2 layers per executable),
    # fused AdamW outside the graph, Tokamax GatedDeltaNet, Pallas MoE / CE
    # kernels, and Splash attention.
    bench_cfg = config_registry.qwen35_moe_35b_split_tokamax_benchmark()
    self.assertIsInstance(bench_cfg, TpuGraphTrainer.Config)
    self.assertEqual(bench_cfg.model_spec.flavor, "35B-A3B")

    # Pretraining: no GatedDeltaNet packed-document resets (RL keeps them).
    def resets(cfg):
      return [
          d.reset_at_documents
          for _, d, _, _ in cfg.model_spec.model.traverse(
              gated_delta.TpuGatedDeltaNet.Config
          )
      ]

    self.assertEqual(resets(bench_cfg), [False] * 30)
    self.assertEqual(resets(config_registry.qwen35_moe_35b()), [True] * 30)
    re_enabled_cfg = config_registry.qwen35_moe_35b_split_tokamax_benchmark()
    apply_overrides(
        OverrideConfig(
            imports=[
                "torchtitan.experiments.tpu.qwen3_5.gated_delta."
                "gated_delta_document_resets"
            ]
        ),
        re_enabled_cfg,
    )
    self.assertEqual(resets(re_enabled_cfg), [True] * 30)
    self.assertEqual(
        bench_cfg.override.imports,
        [
            config_registry.FUSED_LOSS_OVERRIDE,
            config_registry.TOKAMAX_GATED_DELTA_OVERRIDE,
        ],
    )
    self._assert_uses_tpu_moe_kernels(bench_cfg.model_spec.model, tile_m=512)
    self.assertEqual(bench_cfg.parallelism.fsdp_reshard_after_forward, "always")
    self.assertEqual(bench_cfg.parallelism.expert_parallel_degree, 1)
    self.assertTrue(bench_cfg.parallelism.enable_tpu_topo_mesh)
    self.assertEqual(
        bench_cfg.training.num_tokens_per_microbatch_per_dp_rank, 16 * 2048
    )
    self.assertEqual(bench_cfg.training.dtype, "float32")
    self.assertEqual(bench_cfg.training.mixed_precision_param, "bfloat16")
    self.assertEqual(bench_cfg.training.mixed_precision_reduce, "float32")
    self.assertEqual(bench_cfg.metrics.log_freq, 5)
    self.assertEqual(
        bench_cfg.xla_flags, {"xla_enable_hlo_sharding_v3": "false"}
    )
    self.assertEqual(
        bench_cfg.libtpu_init_flags["xla_tpu_scoped_vmem_limit_kib"], "65472"
    )
    self.assertEqual(bench_cfg.libtpu_init_flags["xla_tpu_dvfs_p_state"], "7")
    compile_cfg = bench_cfg.compile
    self.assertTrue(compile_cfg.enable)
    self.assertEqual(compile_cfg.mode, "aot_fx_trace")
    self.assertEqual(compile_cfg.memory_policy, "tpu_selective")
    self.assertEqual(
        compile_cfg.save_regions,
        ("moe.routing", "attention.inner_attention"),
    )
    self.assertFalse(compile_cfg.enable_remat_barriers)
    self.assertEqual(compile_cfg.fsdp_barrier_policy, "none")
    self.assertTrue(compile_cfg.graph_split.enabled)
    self.assertTrue(compile_cfg.graph_split.forward_isolate_prefix)
    self.assertEqual(compile_cfg.graph_split.forward_layers_per_executable, 2)
    self.assertEqual(compile_cfg.graph_split.backward_layers_per_executable, 2)
    self.assertEqual(
        compile_cfg.graph_split.backward_partition_mode, "fsdp_pipeline"
    )
    self.assertEqual(compile_cfg.optimizer_compile, "eager")
    self.assertTrue(compile_cfg.compile_grad_norm)
    self.assertFalse(bench_cfg.tpu_config.enable_amp)
    self.assertFalse(bench_cfg.tpu_config.enable_mfu_benchmark)
    self.assertEqual(bench_cfg.optimizer.implementation, "fused")
    self.assertTrue(
        bench_cfg.splash_attention_kernel.use_splash_attention_kernel
    )
    self.assertIsInstance(bench_cfg.loss, PallasCrossEntropyLoss.Config)
    self.assertEqual(
        (
            bench_cfg.loss.b_block_size,
            bench_cfg.loss.h_block_size,
            bench_cfg.loss.v_block_size,
        ),
        (1024, 2048, 640),
    )

    # Apply both recipe overrides (`tokamax_gated_delta` imports KDA lazily).
    apply_overrides(bench_cfg.override, bench_cfg)
    gdn_kernels = [
        layer.delta_net.inner_gated_delta_net.kernel
        for layer in bench_cfg.model_spec.model.layers
        if layer.delta_net is not None
    ]
    self.assertLen(gdn_kernels, 30)
    for kernel_cfg in gdn_kernels:
      self.assertEqual(
          type(kernel_cfg).__qualname__, "TokamaxGatedDeltaKernel.Config"
      )
    with torch.device("meta"):
      meta_model = bench_cfg.model_spec.model.build()
    workarounds._check_fused_loss_logit_transform(meta_model, bench_cfg)  # pylint: disable=protected-access
    inner_experts = meta_model.layers["0"].moe.routed_experts.inner_experts
    self.assertIsInstance(inner_experts, TpuGroupedExperts)

    # 35B-A3B parameter and MFU accounting. Active params exclude the input
    # embedding and count the top-8 of 256 routed experts. FLOPs per token are
    # 6 * active params, plus 10 quadratic attention layers
    # (6 * 16 heads * (256 + 256) * seq_len each), plus 30 delta-rule layers
    # (6 * 3 * 32 value heads * 128 * 128 each).
    seq_len = bench_cfg.training.max_context_length
    self.assertEqual(seq_len, 2048)
    total_params, flops_per_token = (
        bench_cfg.model_spec.model.get_nparams_and_flops(meta_model, seq_len)
    )
    self.assertEqual(total_params, 34_660_610_688)
    self.assertEqual(
        flops_per_token,
        6 * 2_946_429_568
        + 10 * 6 * 16 * (256 + 256) * seq_len
        + 30 * 6 * 3 * 32 * 128 * 128,
    )

    # The recipes parse from the command line.
    parsed = ConfigManager().parse_args([
        "--module=torchtitan.experiments.tpu.qwen3_5",
        "--config=qwen35_moe_35b_split_tokamax_benchmark",
        "--training.steps=20",
        "--training.max_context_length=8192",
        "--training.num_tokens_per_microbatch_per_dp_rank=65536",
    ])
    self.assertIsInstance(parsed, TpuGraphTrainer.Config)
    self.assertEqual(parsed.training.steps, 20)
    self.assertEqual(parsed.training.max_context_length, 8192)
    self.assertEqual(
        parsed.training.num_tokens_per_microbatch_per_dp_rank, 65536
    )
    self.assertEqual(parsed.model_spec.flavor, "35B-A3B")
    self.assertEqual(parsed.model_spec.max_context_length, 262144)
    self.assertEqual(resets(parsed), [False] * 30)
    self.assertEqual(
        (
            parsed.loss.b_block_size,
            parsed.loss.h_block_size,
            parsed.loss.v_block_size,
        ),
        (1024, 2048, 640),
    )
    self._assert_uses_tpu_moe_kernels(parsed.model_spec.model, tile_m=512)

    # The README's 96K split=1 + selective save_last_n_layers flags also parse.
    parsed_96k = ConfigManager().parse_args([
        "--module=torchtitan.experiments.tpu.qwen3_5",
        "--config=qwen35_moe_35b_split_tokamax_benchmark",
        "--compile.graph_split.forward_layers_per_executable=1",
        "--compile.graph_split.backward_layers_per_executable=1",
        "--compile.save_regions",
        "moe.routing",
        "--compile.save_last_n_layers",
        "attention.inner_attention",
        "2",
    ])
    apply_tpu_graph_trainer_model_transforms(parsed_96k)
    self.assertEqual(
        parsed_96k.compile.graph_split.forward_layers_per_executable, 1
    )
    self.assertEqual(
        parsed_96k.compile.graph_split.backward_layers_per_executable, 1
    )
    self.assertEqual(parsed_96k.compile.save_regions, ("moe.routing",))
    self.assertEqual(
        parsed_96k.compile.save_last_n_layers,
        {"attention.inner_attention": 2},
    )

    # Expert Parallelism (EP > 1) recipes and model config updates.
    for ep_recipe in (
        config_registry.qwen35_moe_35b_split_tokamax_ep_benchmark,
    ):
      ep_cfg = ep_recipe()
      self.assertIsInstance(ep_cfg, TpuGraphTrainer.Config)
      self.assertEqual(ep_cfg.training.dtype, "float32")
      self.assertEqual(ep_cfg.parallelism.expert_parallel_degree, 4)
      self.assertFalse(ep_cfg.compile.compile_grad_norm)
      self.assertEqual(ep_cfg.tpu_config.eager_mode, "DEFER_AND_FUSE")
      self.assertEqual(ep_cfg.compile.memory_policy, "tpu_selective")
      self.assertEqual(
          ep_cfg.compile.save_regions,
          ("moe.routing", "attention.inner_attention"),
      )
      self.assertEqual(ep_cfg.compile.save_last_n_layers, {})
      self.assertEqual(
          ep_cfg.compile.graph_split.backward_partition_mode, "by_layer"
      )
      self.assertEqual(
          ep_cfg.compile.graph_split.forward_layers_per_executable, 1
      )
      self.assertEqual(
          ep_cfg.compile.graph_split.backward_layers_per_executable, 1
      )
      self.assertEqual(
          ep_cfg.libtpu_init_flags.get(
              "xla_tpu_enable_sparse_core_reduce_scatter_v2"
          ),
          "true",
      )
      self.assertEqual(
          ep_cfg.libtpu_init_flags.get(
              "xla_tpu_enable_layer_scheduler_for_dependent_collectives"
          ),
          "true",
      )
      self.assertNotIn(
          "xla_tpu_enable_async_all_to_all", ep_cfg.libtpu_init_flags
      )
      self.assertIs(
          ep_cfg.model_spec.post_optimizer_build_fn,
          qwen3_5_tpu.post_optimizer_build_qwen3_5,
      )
      apply_tpu_graph_trainer_model_transforms(ep_cfg)
      ep_cfg.model_spec.model.update_from_config(
          config=ep_cfg, global_batch_size=8, local_global_batch_size=8
      )
      ep_dispatchers = list(
          ep_cfg.model_spec.model.traverse(
              qwen35_token_dispatcher.TpuQwen35TokenDispatcher.Config
          )
      )
      self.assertLen(ep_dispatchers, len(ep_cfg.model_spec.model.layers))
      self.assertTrue(
          all(
              d.fuse_route_score_backward
              and d.comm_mode == "auto"
              and d.roe_use_pallas_tc
              for _, d, _, _ in ep_dispatchers
          )
      )
      self.assertFalse(hasattr(qwen35_token_dispatcher, "EP_CAPACITY_FACTOR"))
      self.assertTrue(
          all(
              not hasattr(d, "capacity_factor") for _, d, _, _ in ep_dispatchers
          )
      )
      ep_routed = list(
          ep_cfg.model_spec.model.traverse(
              qwen35_token_dispatcher.TpuQwen35RoutedExperts.Config
          )
      )
      self.assertLen(ep_routed, len(ep_cfg.model_spec.model.layers))
      self.assertTrue(
          all(
              r.num_token_chunks == 4 and r.max_inflight_chunks == 2
              for _, r, _, _ in ep_routed
          )
      )
      ep_experts = list(
          ep_cfg.model_spec.model.traverse(TpuGroupedExperts.Config)
      )
      self.assertTrue(
          all(
              e.allow_ep_paired_grouped_mm and e.use_pallas_paired_grouped_mm
              for _, e, _, _ in ep_experts
          )
      )

    parsed_ep = ConfigManager().parse_args([
        "--module=torchtitan.experiments.tpu.qwen3_5",
        "--config=qwen35_moe_35b_split_tokamax_ep_benchmark",
        "--parallelism.expert_parallel_degree=8",
    ])
    self.assertIsInstance(parsed_ep, TpuGraphTrainer.Config)
    self.assertEqual(parsed_ep.parallelism.expert_parallel_degree, 8)
    self.assertFalse(parsed_ep.compile.compile_grad_norm)
    apply_tpu_graph_trainer_model_transforms(parsed_ep)
    parsed_ep.model_spec.model.update_from_config(
        config=parsed_ep, global_batch_size=8, local_global_batch_size=8
    )
    parsed_ep_routed = list(
        parsed_ep.model_spec.model.traverse(
            qwen35_token_dispatcher.TpuQwen35RoutedExperts.Config
        )
    )
    self.assertTrue(
        all(
            r.num_token_chunks == 8 and r.max_inflight_chunks == 2
            for _, r, _, _ in parsed_ep_routed
        )
    )

    # Under EP > 1 on TPU, TpuQwen35Model.parallelize upgrades unconfigured
    # TpuGroupedExperts (use_pallas_grouped_mm=False) to unpaired Pallas
    # (tile_m=256), leaves CPU EP meshes on GroupedExperts.forward, and
    # preserves explicit tiling when use_pallas_grouped_mm is already True.
    def _build_debug_ep_model(tile_m: int | None = None):
      dbg_cfg = config_registry.qwen35_moe_debugmodel()
      if tile_m is not None:
        config_registry._use_tpu_moe_kernels(dbg_cfg, tile_m=tile_m)  # pylint: disable=protected-access
      dbg_cfg.parallelism.expert_parallel_degree = 2
      dbg_cfg.model_spec.model.layers = dbg_cfg.model_spec.model.layers[:1]
      dbg_cfg.model_spec.model.update_from_config(
          config=dbg_cfg, global_batch_size=4, local_global_batch_size=4
      )
      with torch.device("meta"):
        return dbg_cfg.model_spec.model.build()

    def _fake_parallel_dims(
        ep_device_type: str | None, local_rank: int | None = None
    ):
      if ep_device_type is None:
        ep_mesh = None
      elif local_rank is None:
        ep_mesh = SimpleNamespace(device_type=ep_device_type)
      else:
        ep_mesh = SimpleNamespace(
            device_type=ep_device_type,
            size=lambda: 2,
            get_local_rank=lambda: local_rank,
            get_group=lambda: None,
        )
      return SimpleNamespace(
          get_optional_mesh=lambda name: ep_mesh if name == "ep" else None
      )

    with mock.patch.object(tpu_model.Qwen35Model, "parallelize", autospec=True):
      cpu_ep_model = _build_debug_ep_model()
      cpu_ep_model.parallelize(_fake_parallel_dims("cpu"))
      cpu_experts = cpu_ep_model.layers["0"].moe.routed_experts.inner_experts
      self.assertFalse(cpu_experts.use_pallas_grouped_mm)

      tpu_ep_model = _build_debug_ep_model()
      tpu_ep_model.parallelize(_fake_parallel_dims("tpu"))
      tpu_experts = tpu_ep_model.layers["0"].moe.routed_experts.inner_experts
      self.assertTrue(tpu_experts.use_pallas_grouped_mm)
      self.assertFalse(tpu_experts.use_pallas_paired_grouped_mm)
      self.assertEqual(tpu_experts.pallas_grouped_mm_tile_m, 256)
      self.assertEqual(tpu_experts.pallas_grouped_mm_tgmm_tiling, (256, 0, 0))

      preset_ep_model = _build_debug_ep_model(tile_m=512)
      fake_pdims = _fake_parallel_dims("tpu", local_rank=1)
      preset_ep_model.parallelize(fake_pdims)
      preset_experts = preset_ep_model.layers[
          "0"
      ].moe.routed_experts.inner_experts
      self.assertTrue(preset_experts.use_pallas_grouped_mm)
      self.assertFalse(preset_experts.use_pallas_paired_grouped_mm)
      self.assertEqual(preset_experts.pallas_grouped_mm_tile_m, 512)
      self.assertEqual(
          preset_experts.pallas_grouped_mm_tgmm_tiling, (512, 0, 0)
      )
      self.assertIn("_ep_local_rank", preset_ep_model._buffers)  # pylint: disable=protected-access
      self.assertEqual(preset_ep_model._ep_local_rank.device.type, "meta")  # pylint: disable=protected-access
      preset_ep_model.to_empty(device=self.accelerator_device)
      preset_ep_model.init_weights(buffer_device=self.accelerator_device)
      self.assertEqual(int(preset_ep_model._ep_local_rank.item()), 1)  # pylint: disable=protected-access
      self.assertNotIn("_ep_local_rank", preset_ep_model.state_dict())
      bound_disp = preset_ep_model.layers[
          "0"
      ].moe.routed_experts.token_dispatcher
      bound_disp.wire_meshes(ep_mesh=fake_pdims.get_optional_mesh("ep"))
      x_probe = torch.zeros(2, 4, device=self.accelerator_device)
      self.assertIs(
          bound_disp._get_ep_local_rank_tensor(x_probe),  # pylint: disable=protected-access
          preset_ep_model._ep_local_rank,  # pylint: disable=protected-access
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="_ep2_balanced",
          ep_size=2,
          skew_mode="balanced",
          use_grouped_experts_module=False,
      ),
      dict(
          testcase_name="_ep2_extreme_skew_all_rank0",
          ep_size=2,
          skew_mode="all_rank0",
          use_grouped_experts_module=False,
      ),
      dict(
          testcase_name="_ep4_skewed_and_empty_experts",
          ep_size=4,
          skew_mode="skewed_ep4",
          use_grouped_experts_module=False,
      ),
      dict(
          testcase_name="_ep2_real_grouped_experts",
          ep_size=2,
          skew_mode="all_rank0",
          use_grouped_experts_module=True,
      ),
  )
  def test_ep_token_dispatcher_dropless_parity_and_static_fx_trace(
      self,
      ep_size: int,
      skew_mode: str,
      use_grouped_experts_module: bool,
  ):
    """Tests dropless TpuQwen35TokenDispatcher parity vs EP=1 and FX trace."""
    torch.manual_seed(123 + ep_size)
    dev = self.accelerator_device
    num_experts = 8
    e_local = num_experts // ep_size
    top_k = 2
    tokens_per_rank = 6
    dim = 128 if use_grouped_experts_module else 16
    hidden_dim = 128 if use_grouped_experts_module else 24

    w1_full = (
        0.1 * torch.randn(num_experts, hidden_dim, dim, device=dev)
    ).requires_grad_()
    w2_full = (
        0.1 * torch.randn(num_experts, dim, hidden_dim, device=dev)
    ).requires_grad_()
    w3_full = (
        0.1 * torch.randn(num_experts, hidden_dim, dim, device=dev)
    ).requires_grad_()

    xs_ref = [
        torch.randn(tokens_per_rank, dim, device=dev, requires_grad=True)
        for _ in range(ep_size)
    ]
    scores_ref = [
        (0.2 + torch.rand(tokens_per_rank, top_k, device=dev)).requires_grad_()
        for _ in range(ep_size)
    ]
    if skew_mode == "all_rank0":
      # 100% of routed tokens on every rank target rank 0's experts (0..e-1),
      # leaving rank 1 with 0 valid tokens and expert (e-1) unrouted on rank 1.
      idxs = [
          torch.tensor(
              [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]],
              dtype=torch.int64,
              device=dev,
          ),
          torch.tensor(
              [[0, 1], [0, 2], [1, 2], [0, 1], [1, 2], [0, 2]],
              dtype=torch.int64,
              device=dev,
          ),
      ]
    elif skew_mode == "skewed_ep4":
      idxs = [
          torch.tensor(
              [[0, 1], [0, 2], [0, 6], [1, 2], [0, 1], [2, 7]],
              dtype=torch.int64,
              device=dev,
          ),
          torch.tensor(
              [[0, 1], [0, 1], [2, 6], [0, 7], [1, 6], [0, 2]],
              dtype=torch.int64,
              device=dev,
          ),
          torch.tensor(
              [[0, 2], [1, 6], [0, 1], [6, 7], [0, 2], [1, 7]],
              dtype=torch.int64,
              device=dev,
          ),
          torch.tensor(
              [[0, 1], [0, 2], [0, 1], [1, 2], [6, 7], [0, 6]],
              dtype=torch.int64,
              device=dev,
          ),
      ]
    else:
      idxs = [
          torch.tensor(
              [[0, 4], [1, 5], [2, 6], [0, 5], [1, 7], [4, 5]],
              dtype=torch.int64,
              device=dev,
          ),
          torch.tensor(
              [[0, 1], [4, 7], [5, 7], [0, 4], [1, 5], [4, 7]],
              dtype=torch.int64,
              device=dev,
          ),
      ]

    def expert_mlp_ref(h, group_sizes, w1, w2, w3):
      total_valid = int(group_sizes.sum().item())
      splits = list(h[:total_valid].split(group_sizes.tolist(), dim=0))
      outs = []
      for e_idx, chunk in enumerate(splits):
        if chunk.shape[0] == 0:
          outs.append(chunk)
        else:
          gate = F.silu(chunk @ w1[e_idx].t())
          up = chunk @ w3[e_idx].t()
          outs.append((gate * up) @ w2[e_idx].t())
      valid_out = (
          torch.cat(outs, dim=0) if outs else h.new_zeros((0, h.shape[1]))
      )
      if valid_out.shape[0] < h.shape[0]:
        return F.pad(valid_out, (0, 0, 0, h.shape[0] - valid_out.shape[0]))
      return valid_out

    def count_tokens(indices_tk):
      return torch.histc(
          indices_tk.view(-1).float(),
          bins=num_experts,
          min=0,
          max=num_experts - 1,
      ).to(torch.int32)

    ref_disp = TpuAllToAllTokenDispatcher.Config(
        num_experts=num_experts,
        top_k=top_k,
        fuse_route_score_backward=True,
    ).build()
    x_cat = torch.cat(xs_ref, dim=0)
    idx_cat = torch.cat(idxs, dim=0)
    scores_cat = torch.cat(scores_ref, dim=0)
    counts_cat = count_tokens(idx_cat)
    h_ref, gs_ref, meta_ref = ref_disp.dispatch(
        x_cat, scores_cat, idx_cat, counts_cat
    )
    if use_grouped_experts_module:
      ref_experts = TpuGroupedExperts.Config(
          dim=dim,
          hidden_dim=hidden_dim,
          num_experts=num_experts,
          use_pallas_grouped_mm=self._is_tpu(),
          pallas_grouped_mm_tile_m=128,
          pallas_grouped_mm_tgmm_tiling=(128, 0, 0),
      ).build()
      ref_experts.to(dev)
      with torch.no_grad():
        ref_experts.w1_EFD.copy_(w1_full)
        ref_experts.w2_EDF.copy_(w2_full)
        ref_experts.w3_EFD.copy_(w3_full)
      y_ref = ref_experts(h_ref, gs_ref)
    else:
      ref_experts = None
      y_ref = expert_mlp_ref(h_ref, gs_ref, w1_full, w2_full, w3_full)
    out_ref_all = ref_disp.combine(y_ref, meta_ref, x_cat)
    outs_ref = list(out_ref_all.split(tokens_per_rank, dim=0))
    grad_outs = [torch.randn_like(o) for o in outs_ref]
    sum(
        (o * go).sum() for o, go in zip(outs_ref, grad_outs, strict=True)
    ).backward()

    def ref_ragged_a2a(srcs, out_rows, in_offs, send_szs, out_offs, recv_szs):
      num_blocks = in_offs[0].numel()
      u = num_blocks // ep_size
      results = []
      for dst in range(ep_size):
        buf = srcs[dst].new_zeros((out_rows, *srcs[dst].shape[1:]))
        occupied = torch.zeros(out_rows, dtype=torch.bool)
        for src in range(ep_size):
          for k in range(u):
            b_send = dst * u + k
            b_recv = src * u + k
            sz = int(send_szs[src][b_send].item())
            recv_sz = int(recv_szs[dst][b_recv].item())
            self.assertEqual(sz, recv_sz)
            if sz > 0:
              in_off = int(in_offs[src][b_send].item())
              out_off = int(out_offs[src][b_send].item())
              self.assertFalse(
                  bool(occupied[out_off : out_off + sz].any()),
                  msg="Overlapping ragged_all_to_all destination slices",
              )
              occupied[out_off : out_off + sz] = True
              buf[out_off : out_off + sz] = srcs[src][in_off : in_off + sz]
        results.append(buf)
      return results

    xs = [x.detach().clone().requires_grad_(True) for x in xs_ref]
    scores = [s.detach().clone().requires_grad_(True) for s in scores_ref]
    w1_ep = [
        t.detach().clone().requires_grad_(True)
        for t in w1_full.detach().chunk(ep_size, dim=0)
    ]
    w2_ep = [
        t.detach().clone().requires_grad_(True)
        for t in w2_full.detach().chunk(ep_size, dim=0)
    ]
    w3_ep = [
        t.detach().clone().requires_grad_(True)
        for t in w3_full.detach().chunk(ep_size, dim=0)
    ]
    ep_experts_mods = []
    if use_grouped_experts_module:
      for r in range(ep_size):
        mod = TpuGroupedExperts.Config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=e_local,
            use_pallas_grouped_mm=self._is_tpu(),
            pallas_grouped_mm_tile_m=128,
            pallas_grouped_mm_tgmm_tiling=(128, 0, 0),
        ).build()
        mod.to(dev)
        with torch.no_grad():
          mod.w1_EFD.copy_(w1_ep[r])
          mod.w2_EDF.copy_(w2_ep[r])
          mod.w3_EFD.copy_(w3_ep[r])
        ep_experts_mods.append(mod)

    disp = []
    for r in range(ep_size):
      d = qwen35_token_dispatcher.TpuQwen35TokenDispatcher.Config(
          num_experts=num_experts,
          top_k=top_k,
          fuse_route_score_backward=True,
          comm_mode="ragged_all_to_all",
      ).build()
      d.wire_meshes(
          ep_mesh=SimpleNamespace(  # pyrefly: ignore[bad-argument-type]
              size=lambda sz=ep_size: sz,
              get_local_rank=lambda rk=r: rk,
              get_group=lambda: None,
              device_type=dev.type,
          )
      )
      disp.append(d)

    counts = [count_tokens(idxs[r]) for r in range(ep_size)]
    gathered_counts = torch.cat(counts, dim=0)
    for r in range(ep_size):
      disp[r]._gather_expert_counts = (  # pylint: disable=protected-access
          lambda lc, pg, sz, gc=gathered_counts: gc
      )

    # Pass 1 + 2 for dispatch forward through real _RaggedAllToAll.apply.
    disp_args = [None] * ep_size
    for r in range(ep_size):
      with mock.patch.object(
          qwen35_token_dispatcher,
          "_ragged_all_to_all_impl",
          side_effect=lambda op, rows, io, ss, oo, rs, m, p, rk=r: (
              disp_args.__setitem__(rk, (op.detach(), rows, io, ss, oo, rs))
              or op.detach().new_zeros((rows, *op.shape[1:]))
          ),
      ):
        disp[r].dispatch(xs[r], scores[r], idxs[r], counts[r])

    disp_recvs = ref_ragged_a2a(
        [a[0] for a in disp_args],
        disp_args[0][1],
        [a[2] for a in disp_args],
        [a[3] for a in disp_args],
        [a[4] for a in disp_args],
        [a[5] for a in disp_args],
    )
    hs, gss, metas = [], [], []
    for r in range(ep_size):
      with mock.patch.object(
          qwen35_token_dispatcher,
          "_ragged_all_to_all_impl",
          side_effect=lambda op, rows, io, ss, oo, rs, m, p, rv=disp_recvs[
              r
          ]: rv.detach().clone(),
      ):
        h_r, gs_r, meta_r = disp[r].dispatch(
            xs[r], scores[r], idxs[r], counts[r]
        )
      hs.append(h_r)
      gss.append(gs_r)
      metas.append(meta_r)
      # Verify unpadded group_sizes matches global counts for rank r's experts.
      expected_gs = counts_cat[r * e_local : (r + 1) * e_local]
      torch.testing.assert_close(gs_r.cpu(), expected_gs.cpu())

    hs_leaf = [h.detach().clone().requires_grad_(True) for h in hs]
    if use_grouped_experts_module:
      ys = [ep_experts_mods[r](hs_leaf[r], gss[r]) for r in range(ep_size)]
    else:
      ys = [
          expert_mlp_ref(hs_leaf[r], gss[r], w1_ep[r], w2_ep[r], w3_ep[r])
          for r in range(ep_size)
      ]
    ys_leaf = [y.detach().clone().requires_grad_(True) for y in ys]

    # Pass 1 + 2 for combine forward through real _RaggedAllToAll.apply.
    comb_args = [None] * ep_size
    for r in range(ep_size):
      with mock.patch.object(
          qwen35_token_dispatcher,
          "_ragged_all_to_all_impl",
          side_effect=lambda op, rows, io, ss, oo, rs, m, p, rk=r: (
              comb_args.__setitem__(rk, (op.detach(), rows, io, ss, oo, rs))
              or op.detach().new_zeros((rows, *op.shape[1:]))
          ),
      ):
        disp[r].combine(ys_leaf[r], metas[r], xs[r])

    comb_recvs = ref_ragged_a2a(
        [a[0] for a in comb_args],
        comb_args[0][1],
        [a[2] for a in comb_args],
        [a[3] for a in comb_args],
        [a[4] for a in comb_args],
        [a[5] for a in comb_args],
    )
    outs_ep = []
    for r in range(ep_size):
      with mock.patch.object(
          qwen35_token_dispatcher,
          "_ragged_all_to_all_impl",
          side_effect=lambda op, rows, io, ss, oo, rs, m, p, rv=comb_recvs[
              r
          ]: rv.detach().clone(),
      ):
        outs_ep.append(disp[r].combine(ys_leaf[r], metas[r], xs[r]))

    tol = 2e-2 if (self._is_tpu() or use_grouped_experts_module) else 1e-5
    for r in range(ep_size):
      test_utils.check_equivalence(
          outs_ep[r].cpu(),
          outs_ref[r].cpu(),
          atol=tol,
          rtol=tol,
          check_name=f"dropless EP{ep_size} out{r}",
      )

    # Propagate backward through _RaggedAllToAll.backward on combine -> experts -> dispatch.
    comb_bwd_args = [None] * ep_size
    dscores_ep = []
    for r in range(ep_size):
      with mock.patch.object(
          qwen35_token_dispatcher,
          "_ragged_all_to_all_impl",
          side_effect=lambda op, rows, io, ss, oo, rs, m, p, rk=r: (
              comb_bwd_args.__setitem__(rk, (op.detach(), rows, io, ss, oo, rs))
              or op.detach().new_zeros((rows, *op.shape[1:]))
          ),
      ):
        _, ds_r = torch.autograd.grad(
            outs_ep[r],
            (ys_leaf[r], scores[r]),
            grad_outs[r],
            retain_graph=True,
        )
        dscores_ep.append(ds_r)

    dys = ref_ragged_a2a(
        [a[0] for a in comb_bwd_args],
        comb_bwd_args[0][1],
        [a[2] for a in comb_bwd_args],
        [a[3] for a in comb_bwd_args],
        [a[4] for a in comb_bwd_args],
        [a[5] for a in comb_bwd_args],
    )

    dhs = []
    dw1s, dw2s, dw3s = [], [], []
    for r in range(ep_size):
      if use_grouped_experts_module:
        mod = ep_experts_mods[r]
        dh_r, dw1_r, dw2_r, dw3_r = torch.autograd.grad(
            ys[r], (hs_leaf[r], mod.w1_EFD, mod.w2_EDF, mod.w3_EFD), dys[r]
        )
      else:
        raw_grads = torch.autograd.grad(
            ys[r],
            (hs_leaf[r], w1_ep[r], w2_ep[r], w3_ep[r]),
            dys[r],
            allow_unused=True,
        )
        dh_r, dw1_r, dw2_r, dw3_r = (
            g if g is not None else torch.zeros_like(t)
            for g, t in zip(
                raw_grads,
                (hs_leaf[r], w1_ep[r], w2_ep[r], w3_ep[r]),
                strict=True,
            )
        )
      dhs.append(dh_r)
      dw1s.append(dw1_r)
      dw2s.append(dw2_r)
      dw3s.append(dw3_r)

    disp_bwd_args = [None] * ep_size
    for r in range(ep_size):
      with mock.patch.object(
          qwen35_token_dispatcher,
          "_ragged_all_to_all_impl",
          side_effect=lambda op, rows, io, ss, oo, rs, m, p, rk=r: (
              disp_bwd_args.__setitem__(rk, (op.detach(), rows, io, ss, oo, rs))
              or op.detach().new_zeros((rows, *op.shape[1:]))
          ),
      ):
        _ = torch.autograd.grad(hs[r], (xs[r],), dhs[r], retain_graph=True)

    grad_routed_inputs = ref_ragged_a2a(
        [a[0] for a in disp_bwd_args],
        disp_bwd_args[0][1],
        [a[2] for a in disp_bwd_args],
        [a[3] for a in disp_bwd_args],
        [a[4] for a in disp_bwd_args],
        [a[5] for a in disp_bwd_args],
    )
    dxs_ep = []
    for r in range(ep_size):
      with mock.patch.object(
          qwen35_token_dispatcher,
          "_ragged_all_to_all_impl",
          side_effect=lambda op, rows, io, ss, oo, rs, m, p, gri=grad_routed_inputs[
              r
          ]: gri.detach().clone(),
      ):
        (dx_r,) = torch.autograd.grad(hs[r], (xs[r],), dhs[r])
        dxs_ep.append(dx_r)

    for r in range(ep_size):
      self.assertLessEqual(
          _relative_l2_error(dxs_ep[r], xs_ref[r].grad),
          tol,
          msg=f"EP{ep_size} backward mismatch on dx{r}",
      )
      self.assertLessEqual(
          _relative_l2_error(dscores_ep[r], scores_ref[r].grad),
          tol,
          msg=f"EP{ep_size} backward mismatch on dscores{r}",
      )
    expected_w1_grad = (
        ref_experts.w1_EFD.grad if ref_experts is not None else w1_full.grad
    )
    expected_w2_grad = (
        ref_experts.w2_EDF.grad if ref_experts is not None else w2_full.grad
    )
    expected_w3_grad = (
        ref_experts.w3_EFD.grad if ref_experts is not None else w3_full.grad
    )
    for name, actual, expected in (
        ("dw1", torch.cat(dw1s, dim=0), expected_w1_grad),
        ("dw2", torch.cat(dw2s, dim=0), expected_w2_grad),
        ("dw3", torch.cat(dw3s, dim=0), expected_w3_grad),
    ):
      self.assertLessEqual(
          _relative_l2_error(actual, expected),
          tol,
          msg=f"EP{ep_size} backward mismatch on {name}",
      )

    # Verify static-shape make_fx + FakeTensorMode tracing with zero host syncs
    # or scatter ops, and verify TPU emits tpu._all_gather_dim +
    # tpu._ragged_all_to_all with zero c10d_functional collectives.
    mock_pg = (
        SimpleNamespace(group_name="ep_trace_pg", size=lambda: ep_size)
        if self._is_tpu()
        else None
    )
    trace_disp = qwen35_token_dispatcher.TpuQwen35TokenDispatcher.Config(
        num_experts=num_experts,
        top_k=top_k,
        fuse_route_score_backward=True,
        comm_mode="ragged_all_to_all",
    ).build()
    trace_disp.wire_meshes(
        ep_mesh=SimpleNamespace(  # pyrefly: ignore[bad-argument-type]
            size=lambda: ep_size,
            get_local_rank=lambda: 0,
            get_group=lambda: mock_pg,
            device_type=dev.type,
        )
    )

    def trace_step(x_in, idx_in, scores_in, counts_in):
      h_disp, gs_disp, meta = trace_disp.dispatch(
          x_in, scores_in, idx_in, counts_in
      )
      out_disp = trace_disp.combine(h_disp * 2.0, meta, x_in)
      loss = out_disp.sum()
      dx, dscores = torch.autograd.grad(loss, (x_in, scores_in))
      return h_disp, gs_disp, out_disp, dx, dscores

    gm = make_fx(trace_step, tracing_mode="fake")(
        xs[0].detach().requires_grad_(True),
        idxs[0],
        scores[0].detach().requires_grad_(True),
        counts[0],
    )
    op_names = [
        str(n.target) for n in gm.graph.nodes if n.op == "call_function"
    ]
    self.assertFalse(
        any("local_scalar_dense" in name for name in op_names),
        msg=f"Unexpected host scalar sync in traced graph: {op_names}",
    )
    self.assertFalse(
        any("repeat_interleave" in name for name in op_names),
        msg=f"Unexpected dynamic repeat_interleave in traced graph: {op_names}",
    )
    scatter_ops = {
        "slice_backward",
        "slice_scatter",
        "index_put",
        "index_add",
        "scatter_add",
    }
    self.assertFalse(
        any(any(s in name for s in scatter_ops) for name in op_names),
        msg=f"Unexpected scatter op in traced graph: {op_names}",
    )
    if self._is_tpu():
      self.assertTrue(
          any("_all_gather_dim" in name for name in op_names),
          msg=f"Expected tpu._all_gather_dim in TPU trace: {op_names}",
      )
      self.assertEqual(
          sum("_ragged_all_to_all" in name for name in op_names),
          4,
          msg=(
              f"Expected 4 _ragged_all_to_all calls (2 fwd + 2 bwd): {op_names}"
          ),
      )
      self.assertFalse(
          any("c10d_functional" in name for name in op_names),
          msg=f"Unexpected c10d_functional collective in TPU trace: {op_names}",
      )

    num_routed = tokens_per_rank * top_k
    out_node = next(n for n in gm.graph.nodes if n.op == "output")
    out_meta = [arg.meta["val"] for arg in out_node.args[0]]
    self.assertEqual(tuple(out_meta[0].shape), (ep_size * num_routed, dim))
    self.assertEqual(tuple(out_meta[1].shape), (e_local,))
    self.assertEqual(tuple(out_meta[2].shape), (tokens_per_rank, dim))
    self.assertEqual(tuple(out_meta[3].shape), (tokens_per_rank, dim))
    self.assertEqual(tuple(out_meta[4].shape), (tokens_per_rank, top_k))

  def test_split_optimizer_groups_by_mesh(self):
    """Tests mesh-splitting helpers preserve param_names and state_dict FQNs."""

    class _TwoPartToy(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.dense = torch.nn.Parameter(torch.ones(4))
        self.expert = torch.nn.Parameter(torch.ones(4) * 2.0)

    for split_fn in (
        qwen3_5_tpu._split_optimizer_groups_by_mesh,  # pylint: disable=protected-access
        tpu_gt_mod._split_optimizers_by_mesh,  # pylint: disable=protected-access
    ):
      toy = _TwoPartToy()
      toy.dense._spec = type("Spec", (), {"mesh": object()})()
      toy.expert._spec = type("Spec", (), {"mesh": object()})()
      opt_container = default_adamw(lr=1e-3, implementation="for-loop").build(
          model_parts=[toy]
      )
      split_fn(opt_container)
      opt = opt_container.optimizers[0]
      self.assertLen(opt.param_groups, 2)
      self.assertIs(opt.param_groups[0]["params"][0], toy.dense)
      self.assertEqual(opt.param_groups[0]["param_names"], ["dense"])
      self.assertIs(opt.param_groups[1]["params"][0], toy.expert)
      self.assertEqual(opt.param_groups[1]["param_names"], ["expert"])
      toy.dense.grad = torch.ones_like(toy.dense)
      toy.expert.grad = torch.ones_like(toy.expert)
      opt_container.step()
      sd = opt_container.state_dict()
      self.assertTrue(any("dense" in k for k in sd))
      self.assertTrue(any("expert" in k for k in sd))

  def test_ep_ragged_all_to_all_detaches_operand_and_caches_rank(self):
    """Tests TPU ragged_all_to_all receives detached operand & rank buffer."""
    dev = self.accelerator_device
    disp = qwen35_token_dispatcher.TpuQwen35TokenDispatcher.Config(
        num_experts=4, top_k=2
    ).build()
    fake_pg = SimpleNamespace(group_name="ep_pg", size=lambda: 2)
    fake_mesh = SimpleNamespace(
        size=lambda: 2,
        get_local_rank=lambda: 1,
        get_group=lambda: fake_pg,
        device_type=dev.type,
    )
    disp.wire_meshes(ep_mesh=fake_mesh)  # pyrefly: ignore[bad-argument-type]
    self.assertIsNone(disp._ep_local_rank_buf)  # pylint: disable=protected-access

    x_probe = torch.zeros(3, 4, device=dev)
    rank_t = disp._get_ep_local_rank_tensor(x_probe)  # pylint: disable=protected-access
    self.assertEqual(int(rank_t.item()), 1)
    # Re-wiring ep_mesh invalidates the cached fallback rank buffer.
    disp.wire_meshes(
        ep_mesh=SimpleNamespace(  # pyrefly: ignore[bad-argument-type]
            size=lambda: 2,
            get_local_rank=lambda: 0,
            get_group=lambda: fake_pg,
            device_type=dev.type,
        )
    )
    self.assertIsNone(disp._ep_local_rank_buf)  # pylint: disable=protected-access

    if self._is_tpu():
      operand = torch.randn(4, 8, device=dev, requires_grad=True)
      zeros_idx = torch.zeros(4, dtype=torch.int32, device=dev)
      seen_requires_grad = []

      def _spy_r2a(op, out, io, ss, oo, rs, pg_name):
        del io, ss, oo, rs
        self.assertEqual(pg_name, "ep_pg")
        seen_requires_grad.append(op.requires_grad)
        return out.clone()

      with mock.patch.object(
          torch.tpu.ops, "ragged_all_to_all", side_effect=_spy_r2a
      ):
        y = disp._ragged_all_to_all(  # pylint: disable=protected-access
            operand,
            8,
            zeros_idx,
            zeros_idx,
            zeros_idx,
            zeros_idx,
            zeros_idx,
            zeros_idx,
            zeros_idx,
            zeros_idx,
            fake_pg,
        )
        self.assertEqual(tuple(y.shape), (8, 8))
        (dx,) = torch.autograd.grad(y.sum(), (operand,))
        self.assertEqual(tuple(dx.shape), (4, 8))
      self.assertEqual(seen_requires_grad, [False, False])

  def test_scale_dscore_d_chunk_fits_vmem(self):
    """Tests the dScore kernel's feature chunk against the VMEM budget."""
    mib = 2**20
    d_chunk = ragged_gather_tc._scale_dscore_d_chunk  # pylint: disable=protected-access
    # TPU7x (64 MiB): the 35B-A3B hidden size stays one chunk; 4096 (397B,
    # which overflowed VMEM at 65.16 of 63.94 MiB) splits in two in bf16.
    self.assertEqual(d_chunk(2048, 1024, 2, 64 * mib), 2048)
    self.assertEqual(d_chunk(4096, 1024, 2, 64 * mib), 2048)
    self.assertEqual(d_chunk(4096, 1024, 4, 64 * mib), 1024)
    # No 128-multiple chunk fits: the caller falls back to XLA.
    self.assertIsNone(d_chunk(128, 1024, 2, 1 * mib))

  @parameterized.named_parameters(
      dict(
          testcase_name="_ep2_balanced",
          ep_size=2,
          top_k=2,
          tokens_per_rank=6,
          skew_mode="balanced",
          use_paired_experts=False,
      ),
      dict(
          testcase_name="_ep2_extreme_skew_all_rank0",
          ep_size=2,
          top_k=2,
          tokens_per_rank=6,
          skew_mode="all_rank0",
          use_paired_experts=False,
      ),
      dict(
          testcase_name="_ep4_skewed_and_empty_experts",
          ep_size=4,
          top_k=4,
          tokens_per_rank=6,
          skew_mode="skewed_ep4",
          use_paired_experts=False,
      ),
      dict(
          testcase_name="_ep2_paired_experts_and_dscore_kernel",
          ep_size=2,
          top_k=4,
          tokens_per_rank=128,
          skew_mode="all_rank0",
          use_paired_experts=True,
      ),
      # The dScore kernel's feature-chunked grid (as wide hidden sizes such as
      # 4096 need for VMEM): two 128-wide chunks of a 256-wide hidden size,
      # accumulating the dScore row sums across chunks.
      dict(
          testcase_name="_ep2_paired_experts_and_dscore_kernel_d_chunks",
          ep_size=2,
          top_k=4,
          tokens_per_rank=128,
          skew_mode="all_rank0",
          use_paired_experts=True,
          dim=256,
          dscore_d_chunk=128,
      ),
  )
  def test_ep_ring_of_experts_dropless_parity_and_fx_trace(
      self,
      ep_size: int,
      top_k: int,
      tokens_per_rank: int,
      skew_mode: str,
      use_paired_experts: bool,
      dim: int | None = None,
      dscore_d_chunk: int | None = None,
  ):
    """Tests Ring-of-Experts dropless parity, paired GMM, and static FX trace.

    The paired cases on TPU also check the weights-on-activation combine
    (`roe_weights_on_activation`): outputs and the scores' gradient.
    """
    torch.manual_seed(321 + ep_size + top_k)
    dev = self.accelerator_device
    num_experts = 8
    e_local = num_experts // ep_size
    if dim is None:
      dim = 128 if (use_paired_experts or self._is_tpu()) else 16
    if dscore_d_chunk is not None:
      self.enter_context(
          mock.patch.object(
              ragged_gather_tc,
              "_scale_dscore_d_chunk",
              return_value=dscore_d_chunk,
          )
      )
    hidden_dim = 128 if (use_paired_experts or self._is_tpu()) else 24

    auto_disp = qwen35_token_dispatcher.TpuQwen35TokenDispatcher.Config(
        num_experts=num_experts,
        top_k=2,
        comm_mode="auto",
    ).build()
    self.assertEqual(auto_disp._resolve_comm_mode(2), "ring_of_experts")  # pylint: disable=protected-access
    self.assertEqual(auto_disp._resolve_comm_mode(4), "ragged_all_to_all")  # pylint: disable=protected-access

    w1_full = (
        0.1 * torch.randn(num_experts, hidden_dim, dim, device=dev)
    ).requires_grad_()
    w2_full = (
        0.1 * torch.randn(num_experts, dim, hidden_dim, device=dev)
    ).requires_grad_()
    w3_full = (
        0.1 * torch.randn(num_experts, hidden_dim, dim, device=dev)
    ).requires_grad_()

    xs_ref = [
        torch.randn(tokens_per_rank, dim, device=dev, requires_grad=True)
        for _ in range(ep_size)
    ]
    scores_ref = [
        (0.2 + torch.rand(tokens_per_rank, top_k, device=dev)).requires_grad_()
        for _ in range(ep_size)
    ]
    if skew_mode == "all_rank0":
      # Route 100% of global tokens to rank 0's local experts (0 .. e_local-1),
      # so rank 0 hits full capacity (count == cap) and rank 1 has count == 0.
      base_row = torch.arange(top_k, dtype=torch.int64, device=dev) % e_local
      idxs = [
          base_row.unsqueeze(0).expand(tokens_per_rank, top_k).contiguous()
          for _ in range(ep_size)
      ]
    elif skew_mode == "skewed_ep4":
      # Leave expert 3 (rank 1) and expert 4 (rank 2) completely unrouted.
      allowed = torch.tensor([0, 1, 2, 5, 6, 7], dtype=torch.int64, device=dev)
      idxs = []
      for r in range(ep_size):
        rows = []
        for t in range(tokens_per_rank):
          perm = torch.randperm(allowed.numel(), device=dev)[:top_k]
          rows.append(allowed[perm])
        idxs.append(torch.stack(rows, dim=0))
    else:
      idxs = []
      for _ in range(ep_size):
        logits = torch.rand(tokens_per_rank, num_experts, device=dev)
        idxs.append(torch.topk(logits, k=top_k, dim=-1).indices)

    def expert_mlp_ref(h, group_sizes, w1, w2, w3):
      total_valid = int(group_sizes.sum().item())
      splits = list(h[:total_valid].split(group_sizes.tolist(), dim=0))
      outs = []
      for e_idx, chunk in enumerate(splits):
        if chunk.shape[0] == 0:
          outs.append(chunk)
        else:
          gate = F.silu(chunk @ w1[e_idx].t())
          up = chunk @ w3[e_idx].t()
          outs.append((gate * up) @ w2[e_idx].t())
      valid_out = (
          torch.cat(outs, dim=0) if outs else h.new_zeros((0, h.shape[1]))
      )
      if valid_out.shape[0] < h.shape[0]:
        return F.pad(valid_out, (0, 0, 0, h.shape[0] - valid_out.shape[0]))
      return valid_out

    def count_tokens(indices_tk):
      return torch.histc(
          indices_tk.view(-1).float(),
          bins=num_experts,
          min=0,
          max=num_experts - 1,
      ).to(torch.int32)

    ref_disp = TpuAllToAllTokenDispatcher.Config(
        num_experts=num_experts,
        top_k=top_k,
        fuse_route_score_backward=True,
    ).build()
    x_cat = torch.cat(xs_ref, dim=0)
    idx_cat = torch.cat(idxs, dim=0)
    scores_cat = torch.cat(scores_ref, dim=0)
    counts_cat = count_tokens(idx_cat)
    h_ref, gs_ref, meta_ref = ref_disp.dispatch(
        x_cat, scores_cat, idx_cat, counts_cat
    )
    if use_paired_experts:
      ref_experts = TpuGroupedExperts.Config(
          dim=dim,
          hidden_dim=hidden_dim,
          num_experts=num_experts,
          use_pallas_grouped_mm=self._is_tpu(),
          use_pallas_paired_grouped_mm=self._is_tpu(),
          pallas_grouped_mm_tile_m=128,
          pallas_grouped_mm_tgmm_tiling=(128, 0, 0),
      ).build()
      ref_experts.to(dev)
      with torch.no_grad():
        ref_experts.w1_EFD.copy_(w1_full)
        ref_experts.w2_EDF.copy_(w2_full)
        ref_experts.w3_EFD.copy_(w3_full)
      y_ref = ref_experts(h_ref, gs_ref)
    else:
      ref_experts = None
      y_ref = expert_mlp_ref(h_ref, gs_ref, w1_full, w2_full, w3_full)
    out_ref_all = ref_disp.combine(y_ref, meta_ref, x_cat)
    outs_ref = list(out_ref_all.split(tokens_per_rank, dim=0))
    grad_outs = [torch.randn_like(o) for o in outs_ref]
    sum(
        (o * go).sum() for o, go in zip(outs_ref, grad_outs, strict=True)
    ).backward()

    xs = [x.detach().clone().requires_grad_(True) for x in xs_ref]
    scores = [s.detach().clone().requires_grad_(True) for s in scores_ref]
    w1_ep = [
        t.detach().clone().requires_grad_(True)
        for t in w1_full.detach().chunk(ep_size, dim=0)
    ]
    w2_ep = [
        t.detach().clone().requires_grad_(True)
        for t in w2_full.detach().chunk(ep_size, dim=0)
    ]
    w3_ep = [
        t.detach().clone().requires_grad_(True)
        for t in w3_full.detach().chunk(ep_size, dim=0)
    ]
    ep_experts_mods = []
    if use_paired_experts:
      for r in range(ep_size):
        mod = TpuGroupedExperts.Config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=e_local,
            use_pallas_grouped_mm=self._is_tpu(),
            use_pallas_paired_grouped_mm=self._is_tpu(),
            allow_ep_paired_grouped_mm=True,
            pallas_grouped_mm_tile_m=128,
            pallas_grouped_mm_tgmm_tiling=(128, 0, 0),
        ).build()
        mod.to(dev)
        with torch.no_grad():
          mod.w1_EFD.copy_(w1_ep[r])
          mod.w2_EDF.copy_(w2_ep[r])
          mod.w3_EFD.copy_(w3_ep[r])
        ep_experts_mods.append(mod)

    disp = []
    for r in range(ep_size):
      d = qwen35_token_dispatcher.TpuQwen35TokenDispatcher.Config(
          num_experts=num_experts,
          top_k=top_k,
          fuse_route_score_backward=True,
          comm_mode="ring_of_experts",
      ).build()
      d.wire_meshes(
          ep_mesh=SimpleNamespace(  # pyrefly: ignore[bad-argument-type]
              size=lambda sz=ep_size: sz,
              get_local_rank=lambda rk=r: rk,
              get_group=lambda: None,
              device_type=dev.type,
          )
      )
      disp.append(d)

    counts = [count_tokens(idxs[r]) for r in range(ep_size)]
    ids_cat_i32 = idx_cat.to(torch.int32).contiguous()
    x_cat_det = torch.cat([x.detach() for x in xs], dim=0).contiguous()
    scores_cat_det = torch.cat([s.detach() for s in scores], dim=0).contiguous()

    def _mock_ag_fwd(op, _mesh, _pg):
      if op.dtype in (torch.int32, torch.int64):
        return ids_cat_i32.clone()
      if op.shape[-1] == dim:
        return x_cat_det.clone()
      return scores_cat_det.clone()

    hs, gss, metas = [], [], []
    with mock.patch.object(
        qwen35_token_dispatcher, "_ep_all_gather_impl", side_effect=_mock_ag_fwd
    ):
      for r in range(ep_size):
        h_r, gs_r, meta_r = disp[r].dispatch(
            xs[r], scores[r], idxs[r], counts[r]
        )
        hs.append(h_r)
        gss.append(gs_r)
        metas.append(meta_r)
        expected_gs = counts_cat[r * e_local : (r + 1) * e_local]
        torch.testing.assert_close(gs_r.cpu(), expected_gs.cpu())

    hs_leaf = [h.detach().clone().requires_grad_(True) for h in hs]
    if use_paired_experts:
      ys = [ep_experts_mods[r](hs_leaf[r], gss[r]) for r in range(ep_size)]
    else:
      ys = [
          expert_mlp_ref(hs_leaf[r], gss[r], w1_ep[r], w2_ep[r], w3_ep[r])
          for r in range(ep_size)
      ]
    ys_leaf = [y.detach().clone().requires_grad_(True) for y in ys]

    comb_partials = [None] * ep_size
    for r in range(ep_size):
      with mock.patch.object(
          qwen35_token_dispatcher,
          "_ep_reduce_scatter_sum_impl",
          side_effect=lambda op, _m, _p, rk=r: (
              comb_partials.__setitem__(rk, op.detach().clone())
              or op.detach().new_zeros((tokens_per_rank, dim))
          ),
      ):
        disp[r].combine(ys_leaf[r], metas[r], xs[r])
    comb_recvs = list(
        torch.stack(comb_partials, dim=0).sum(dim=0).split(tokens_per_rank, 0)
    )
    outs_ep = []
    for r in range(ep_size):
      with mock.patch.object(
          qwen35_token_dispatcher,
          "_ep_reduce_scatter_sum_impl",
          side_effect=lambda op, _m, _p, rv=comb_recvs[r]: rv.detach().clone(),
      ):
        outs_ep.append(disp[r].combine(ys_leaf[r], metas[r], xs[r]))

    tol = 2e-2 if (self._is_tpu() or use_paired_experts) else 1e-5
    for r in range(ep_size):
      self.assertTrue(bool(torch.isfinite(outs_ep[r]).all()))
      test_utils.check_equivalence(
          outs_ep[r].cpu(),
          outs_ref[r].cpu(),
          atol=tol,
          rtol=tol,
          check_name=f"RoE EP{ep_size} out{r}",
      )

    grad_outs_cat = torch.cat(grad_outs, dim=0).contiguous()
    dscores_partials = [None] * ep_size
    dys = []
    for r in range(ep_size):
      with (
          mock.patch.object(
              qwen35_token_dispatcher,
              "_ep_all_gather_impl",
              side_effect=lambda op, _m, _p: grad_outs_cat.clone(),
          ),
          mock.patch.object(
              qwen35_token_dispatcher,
              "_ep_reduce_scatter_sum_impl",
              side_effect=lambda op, _m, _p, rk=r: (
                  dscores_partials.__setitem__(rk, op.detach().clone())
                  or op.detach().new_zeros((tokens_per_rank, top_k))
              ),
          ),
      ):
        dy_r, _ = torch.autograd.grad(
            outs_ep[r],
            (ys_leaf[r], scores[r]),
            grad_outs[r],
            retain_graph=True,
        )
        dys.append(dy_r)
    dscores_ep = list(
        torch.stack(dscores_partials, dim=0)
        .sum(dim=0)
        .split(tokens_per_rank, 0)
    )

    dhs, dw1s, dw2s, dw3s = [], [], [], []
    for r in range(ep_size):
      if use_paired_experts:
        mod = ep_experts_mods[r]
        dh_r, dw1_r, dw2_r, dw3_r = torch.autograd.grad(
            ys[r], (hs_leaf[r], mod.w1_EFD, mod.w2_EDF, mod.w3_EFD), dys[r]
        )
      else:
        raw_grads = torch.autograd.grad(
            ys[r],
            (hs_leaf[r], w1_ep[r], w2_ep[r], w3_ep[r]),
            dys[r],
            allow_unused=True,
        )
        dh_r, dw1_r, dw2_r, dw3_r = (
            g if g is not None else torch.zeros_like(t)
            for g, t in zip(
                raw_grads,
                (hs_leaf[r], w1_ep[r], w2_ep[r], w3_ep[r]),
                strict=True,
            )
        )
      dhs.append(dh_r)
      dw1s.append(dw1_r)
      dw2s.append(dw2_r)
      dw3s.append(dw3_r)

    dx_partials = [None] * ep_size
    for r in range(ep_size):
      with mock.patch.object(
          qwen35_token_dispatcher,
          "_ep_reduce_scatter_sum_impl",
          side_effect=lambda op, _m, _p, rk=r: (
              dx_partials.__setitem__(rk, op.detach().clone())
              or op.detach().new_zeros((tokens_per_rank, dim))
          ),
      ):
        _ = torch.autograd.grad(hs[r], (xs[r],), dhs[r])
    dxs_ep = list(
        torch.stack(dx_partials, dim=0).sum(dim=0).split(tokens_per_rank, 0)
    )

    for r in range(ep_size):
      self.assertTrue(bool(torch.isfinite(dxs_ep[r]).all()))
      self.assertTrue(bool(torch.isfinite(dscores_ep[r]).all()))
      self.assertLessEqual(
          _relative_l2_error(dxs_ep[r], xs_ref[r].grad),
          tol,
          msg=f"RoE EP{ep_size} backward mismatch on dx{r}",
      )
      self.assertLessEqual(
          _relative_l2_error(dscores_ep[r], scores_ref[r].grad),
          tol,
          msg=f"RoE EP{ep_size} backward mismatch on dscores{r}",
      )
    expected_w1_grad = (
        ref_experts.w1_EFD.grad if ref_experts is not None else w1_full.grad
    )
    expected_w2_grad = (
        ref_experts.w2_EDF.grad if ref_experts is not None else w2_full.grad
    )
    expected_w3_grad = (
        ref_experts.w3_EFD.grad if ref_experts is not None else w3_full.grad
    )
    for name, actual, expected in (
        ("dw1", torch.cat(dw1s, dim=0), expected_w1_grad),
        ("dw2", torch.cat(dw2s, dim=0), expected_w2_grad),
        ("dw3", torch.cat(dw3s, dim=0), expected_w3_grad),
    ):
      self.assertTrue(bool(torch.isfinite(actual).all()))
      self.assertLessEqual(
          _relative_l2_error(actual, expected),
          tol,
          msg=f"RoE EP{ep_size} backward mismatch on {name}",
      )

    if use_paired_experts and self._is_tpu():
      # Weights on activation (`roe_weights_on_activation`, the default of
      # the Pallas path in TpuQwen35RoutedExperts): each buffer row's routing
      # weight scales the experts' hidden activation (`row_scale_R`), and the
      # combine is the unweighted `roe_unsort_reduce`. The scores' gradient
      # then flows through the row scales, one buffer row per (token, k) slot.
      row_scales = [
          metas[r].expert_row_scale_R.detach().clone().requires_grad_(True)
          for r in range(ep_size)
      ]
      woa_ys = [
          ep_experts_mods[r](hs_leaf[r], gss[r], row_scale_R=row_scales[r])
          for r in range(ep_size)
      ]
      woa_metas = [
          dataclasses.replace(m, row_scale_applied=True) for m in metas
      ]
      woa_partials = [None] * ep_size
      for r in range(ep_size):
        with mock.patch.object(
            qwen35_token_dispatcher,
            "_ep_reduce_scatter_sum_impl",
            side_effect=lambda op, _m, _p, rk=r: (
                woa_partials.__setitem__(rk, op.detach().clone())
                or op.detach().new_zeros((tokens_per_rank, dim))
            ),
        ):
          disp[r].combine(woa_ys[r], woa_metas[r], xs[r])
      woa_recvs = list(
          torch.stack(woa_partials, dim=0).sum(dim=0).split(tokens_per_rank, 0)
      )
      dscores_flat = torch.zeros(
          ep_size * tokens_per_rank * top_k, dtype=torch.float32, device=dev
      )
      for r in range(ep_size):
        with mock.patch.object(
            qwen35_token_dispatcher,
            "_ep_reduce_scatter_sum_impl",
            side_effect=lambda op, _m, _p, rv=woa_recvs[r]: rv.detach().clone(),
        ):
          out_woa = disp[r].combine(woa_ys[r], woa_metas[r], xs[r])
        self.assertTrue(bool(torch.isfinite(out_woa).all()))
        # The weight scales the bf16 hidden activation before w2 here, but the
        # fp32 sum of expert outputs in the reference, so compare in norm.
        self.assertLessEqual(
            _relative_l2_error(out_woa, outs_ref[r]),
            tol,
            msg=f"RoE EP{ep_size} weights-on-activation out{r}",
        )
        with mock.patch.object(
            qwen35_token_dispatcher,
            "_ep_all_gather_impl",
            side_effect=lambda op, _m, _p: grad_outs_cat.clone(),
        ):
          (drow_r,) = torch.autograd.grad(
              out_woa, (row_scales[r],), grad_outs[r]
          )
        cap = drow_r.shape[0]
        valid = (
            (torch.arange(cap, device=dev) < metas[r].start_count_2[1])
            .nonzero()
            .squeeze(-1)
        )
        dscores_flat.index_add_(
            0,
            metas[r].window_slot_order_cap[valid].long(),
            drow_r[valid].float(),
        )
      expected_dscores = torch.cat([s.grad for s in scores_ref], dim=0)
      self.assertLessEqual(
          _relative_l2_error(
              dscores_flat.reshape(expected_dscores.shape), expected_dscores
          ),
          tol,
          msg=f"RoE EP{ep_size} weights-on-activation dscores",
      )

    # Verify static-shape make_fx + FakeTensorMode trace for Ring-of-Experts and
    # confirm MOE_ROUTING excludes roe_compute_routing so tpu_selective saves
    # only ids_all_GK (not per-chunk src_rows).
    mock_pg = (
        SimpleNamespace(group_name="ep_roe_pg", size=lambda: ep_size)
        if self._is_tpu()
        else None
    )
    trace_disp = qwen35_token_dispatcher.TpuQwen35TokenDispatcher.Config(
        num_experts=num_experts,
        top_k=top_k,
        fuse_route_score_backward=True,
        comm_mode="ring_of_experts",
    ).build()
    trace_disp.wire_meshes(
        ep_mesh=SimpleNamespace(  # pyrefly: ignore[bad-argument-type]
            size=lambda: ep_size,
            get_local_rank=lambda: 0,
            get_group=lambda: mock_pg,
            device_type=dev.type,
        )
    )

    def trace_roe_step(x_in, idx_in, scores_in, counts_in):
      h_disp, gs_disp, meta = trace_disp.dispatch(
          x_in, scores_in, idx_in, counts_in
      )
      out_disp = trace_disp.combine(h_disp * 2.0, meta, x_in)
      loss = out_disp.sum()
      dx, dscores = torch.autograd.grad(loss, (x_in, scores_in))
      return h_disp, gs_disp, out_disp, dx, dscores

    with torch.fx.traceback.preserve_node_meta():
      gm = make_fx(trace_roe_step, tracing_mode="fake")(
          xs[0].detach().requires_grad_(True),
          idxs[0],
          scores[0].detach().requires_grad_(True),
          counts[0],
      )
    op_names = [
        str(n.target) for n in gm.graph.nodes if n.op == "call_function"
    ]
    self.assertFalse(any("local_scalar_dense" in name for name in op_names))
    ep_disp_saved = select_ep_dispatch(gm)
    self.assertGreaterEqual(len(ep_disp_saved), 2)
    route_saved = select_route_permutations(gm)
    self.assertGreaterEqual(len(route_saved), 1)
    if self._is_tpu():
      self.assertTrue(any("_all_gather_dim" in name for name in op_names))
      self.assertTrue(any("_reduce_scatter_dim" in name for name in op_names))
      self.assertFalse(any("c10d_functional" in name for name in op_names))
      self.assertFalse(
          any("roe_routing_metadata" in str(n.target) for n in route_saved),
          msg=f"roe_compute_routing must not be saved by moe.routing: {route_saved}",
      )

  def test_ep_chunked_routed_experts_and_chunk_barriers(self):
    """Tests sequence-chunked TpuQwen35RoutedExperts parity and FX chunk barriers."""
    torch.manual_seed(999)
    dev = (
        self.accelerator_device
        if self.accelerator_device.type in ("cpu", "tpu")
        else torch.device("cpu")
    )
    num_experts = 4
    ep_size = 2
    e_local = num_experts // ep_size
    top_k = 2
    tokens = 16
    dim = 128 if dev.type == "tpu" else 16
    hidden_dim = 128 if dev.type == "tpu" else 24

    def _build_routed(num_chunks: int):
      routed = qwen35_token_dispatcher.TpuQwen35RoutedExperts.Config(
          num_token_chunks=num_chunks,
          max_inflight_chunks=2,
          inner_experts=TpuGroupedExperts.Config(
              dim=dim,
              hidden_dim=hidden_dim,
              num_experts=e_local,
              use_pallas_grouped_mm=(dev.type == "tpu"),
              use_pallas_paired_grouped_mm=(dev.type == "tpu"),
              allow_ep_paired_grouped_mm=True,
              pallas_grouped_mm_tile_m=128,
              pallas_grouped_mm_tgmm_tiling=(128, 0, 0),
          ),
          token_dispatcher=qwen35_token_dispatcher.TpuQwen35TokenDispatcher.Config(
              num_experts=num_experts,
              top_k=top_k,
              comm_mode="ring_of_experts",
          ),
      ).build()
      routed.to(dev)
      routed.token_dispatcher.wire_meshes(
          ep_mesh=SimpleNamespace(  # pyrefly: ignore[bad-argument-type]
              size=lambda: ep_size,
              get_local_rank=lambda: 0,
              get_group=lambda: None,
              device_type=dev.type,
          )
      )
      return routed

    routed_c1 = _build_routed(1)
    routed_c4 = _build_routed(4)
    with torch.no_grad():
      for p in routed_c1.inner_experts.parameters():
        p.normal_(0.0, 0.1)
      routed_c4.inner_experts.w1_EFD.copy_(routed_c1.inner_experts.w1_EFD)
      routed_c4.inner_experts.w2_EDF.copy_(routed_c1.inner_experts.w2_EDF)
      routed_c4.inner_experts.w3_EFD.copy_(routed_c1.inner_experts.w3_EFD)

    x1 = torch.randn(tokens, dim, device=dev, requires_grad=True)
    s1 = (0.2 + torch.rand(tokens, top_k, device=dev)).requires_grad_()
    ids = torch.topk(
        torch.rand(tokens, num_experts, device=dev), k=top_k, dim=-1
    ).indices
    counts = torch.histc(
        ids.view(-1).float(), bins=num_experts, min=0, max=num_experts - 1
    ).to(torch.int32)
    go = torch.randn(tokens, dim, device=dev)

    out1 = routed_c1(x1, s1, ids, counts)
    (out1 * go).sum().backward()

    x4 = x1.detach().clone().requires_grad_(True)
    s4 = s1.detach().clone().requires_grad_(True)
    unshard_calls = []
    orig_fwd = routed_c4.inner_experts.forward

    def _spy_inner(*args, **kwargs):
      unshard_calls.append(
          (
              bool(kwargs.get("unshard_only", False)),
              kwargs.get("pre_unsharded_weights") is not None,
          )
      )
      return orig_fwd(*args, **kwargs)

    with mock.patch.object(
        routed_c4.inner_experts, "forward", side_effect=_spy_inner
    ):
      out4 = routed_c4(x4, s4, ids, counts)
    (out4 * go).sum().backward()

    # Verify weights are unsharded once outside the 4-chunk loop and reused 4x.
    self.assertEqual(
        unshard_calls,
        [(True, False)] + [(False, True)] * 4,
    )
    # GroupedExperts computes in BF16 on both CPU and TPU.
    tol = 2e-2
    test_utils.check_equivalence(
        out4.cpu(), out1.cpu(), atol=tol, rtol=tol, check_name="chunked out"
    )
    self.assertLessEqual(_relative_l2_error(x4.grad, x1.grad), tol)
    self.assertLessEqual(_relative_l2_error(s4.grad, s1.grad), tol)
    for p4, p1 in zip(
        routed_c4.inner_experts.parameters(),
        routed_c1.inner_experts.parameters(),
        strict=True,
    ):
      self.assertLessEqual(_relative_l2_error(p4.grad, p1.grad), tol)

    # Trace a 1-layer wrapper on CPU and verify apply_ep_chunk_barrier_pass
    # inserts (4 - 2) = 2 forward + 2 backward barriers inside layer 0 after
    # selective_activation_remat_pass while preserving exact outputs/gradients.
    cpu_dev = torch.device("cpu")

    class _ToyBlock(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.routed_experts = qwen35_token_dispatcher.TpuQwen35RoutedExperts.Config(
            num_token_chunks=4,
            max_inflight_chunks=2,
            inner_experts=TpuGroupedExperts.Config(
                dim=16,
                hidden_dim=24,
                num_experts=e_local,
                allow_ep_paired_grouped_mm=True,
            ),
            token_dispatcher=qwen35_token_dispatcher.TpuQwen35TokenDispatcher.Config(
                num_experts=num_experts,
                top_k=top_k,
                comm_mode="ring_of_experts",
            ),
        ).build()
        self.routed_experts.token_dispatcher.wire_meshes(
            ep_mesh=SimpleNamespace(  # pyrefly: ignore[bad-argument-type]
                size=lambda: ep_size,
                get_local_rank=lambda: 0,
                get_group=lambda: None,
                device_type="cpu",
            )
        )

      def forward(self, x_in, s_in, ids_in, c_in):
        return self.routed_experts(x_in, s_in, ids_in, c_in)

    class _ToyModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleDict({"0": _ToyBlock()})

      def forward(self, x_in, s_in, ids_in, c_in):
        return self.layers["0"](x_in, s_in, ids_in, c_in)

    toy = _ToyModel().to(cpu_dev)
    with torch.no_grad():
      for p in toy.parameters():
        p.normal_(0.0, 0.1)
    annotate_module_fqns(toy)
    x_cpu = torch.randn(tokens, 16, device=cpu_dev, requires_grad=True)
    s_cpu = (0.2 + torch.rand(tokens, top_k, device=cpu_dev)).requires_grad_()
    ids_cpu = ids.cpu()
    counts_cpu = counts.cpu()

    def _toy_step(x_in, s_in, ids_in, c_in):
      loss = toy(x_in, s_in, ids_in, c_in).float().sum()
      named_params = list(toy.named_parameters())
      return [loss, *compute_parameter_gradients(loss, named_params)]

    traced = minimal_fx_tracer(_toy_step, module=toy)(
        x_cpu, s_cpu, ids_cpu, counts_cpu
    )
    ref_outputs = run_traced(traced, module=toy)(
        x_cpu, s_cpu, ids_cpu, counts_cpu
    )

    fake_cfg = SimpleNamespace(
        compile=SimpleNamespace(
            memory_policy="full",
            full_recompute_save_ops="",
            save_regions=(),
            save_last_n_layers={},
            graph_split=SimpleNamespace(enabled=True),
        ),
        parallelism=SimpleNamespace(
            fsdp_reshard_after_forward="always",
            pipeline_parallel_degree=1,
        ),
    )
    gm = tag_with_memory_policy_pass(traced.gm, config=fake_cfg)
    gm = selective_activation_remat_pass(gm)
    gm = apply_ep_chunk_barrier_pass(gm)
    traced.gm = gm

    barriers = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is _optimization_barrier
    ]
    fwd_barriers = [n for n in barriers if not _is_backward_node(n)]
    bwd_barriers = [n for n in barriers if _is_backward_node(n)]
    self.assertLen(fwd_barriers, 2)
    self.assertLen(bwd_barriers, 2)
    self.assertTrue(all(_get_layer_id(n) == 0 for n in barriers))

    got_outputs = run_traced(traced, module=toy)(
        x_cpu, s_cpu, ids_cpu, counts_cpu
    )
    for got, ref in zip(got_outputs, ref_outputs, strict=True):
      torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
  absltest.main()
