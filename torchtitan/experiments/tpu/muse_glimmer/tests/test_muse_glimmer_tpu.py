"""Tests for Muse Glimmer (text-only) model components on GPU and TPU devices.

This file contains tests to ensure parity between CPU and GPU/TPU device
implementations of various Muse Glimmer model layers (sliding-window SDPA,
Attention, EmbeddingWithNorm, SoftCappedLinear output head and
TransformerBlock) and full model training, plus TPU-only checks that the Splash
and tokamax attention patches match the SDPA reference on packed documents.

CPU-only config / registry checks live in test_muse_glimmer_config.py.
"""

import copy
import math

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu import workarounds
import torchtitan.experiments.tpu.muse_glimmer as muse_glimmer_tpu
from torchtitan.experiments.tpu.muse_glimmer import attention as tpu_attention
from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
from torchtitan.models.common.attention import segment_ids_from_positions
from torchtitan.models.muse_glimmer.model import MuseGlimmerModel


def _get_model_config(model_name: str = "testmodel") -> MuseGlimmerModel.Config:
  """Retrieves a registered Muse Glimmer configuration from the TPU registry.

  Defaults for 'testmodel':
    - vocab_size = 2048, max_seq_len = 128, dim = 128
    - n_layers = 4, n_heads = 16, n_kv_heads = 8, head_dim = 16
    - window pattern = [4, 4, 4, global]; layer 3 is global + NoPE
  """
  return muse_glimmer_tpu.muse_glimmer_configs[model_name]


def _naive_sliding_window_attention(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHV: torch.Tensor,
    window_size: int | None,
    scale: float,
    segment_ids_BL: torch.Tensor | None = None,
) -> torch.Tensor:
  """Reference attention: explicit scores, ``0 <= q - k < window`` mask.

  With ``segment_ids_BL``, queries also only see keys of their own document.
  """
  n_rep = q_BLHK.shape[2] // k_BLHK.shape[2]
  k_BLHK = k_BLHK.repeat_interleave(n_rep, dim=2)
  v_BLHV = v_BLHV.repeat_interleave(n_rep, dim=2)
  scores_BHLL = torch.einsum("blhk,bmhk->bhlm", q_BLHK, k_BLHK) * scale
  seq_len = q_BLHK.shape[1]
  distance = (
      torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
  )
  allowed = distance >= 0
  if window_size is not None:
    allowed &= distance < window_size
  if segment_ids_BL is not None:
    same_document_BLL = segment_ids_BL[:, :, None] == segment_ids_BL[:, None, :]
    allowed = allowed[None] & same_document_BLL
    allowed = allowed[:, None]  # Broadcast over heads.
  scores_BHLL = scores_BHLL.masked_fill(~allowed, float("-inf"))
  probs_BHLL = torch.softmax(scores_BHLL, dim=-1)
  return torch.einsum("bhlm,bmhv->blhv", probs_BHLL, v_BLHV)


SDPA_TEST_CONFIGS = [
    dict(testcase_name="_SWA", window_size=3),
    dict(testcase_name="_GLOBAL", window_size=None),
]

# Layer 0 is a sliding-window layer with RoPE; layer 3 is global with NoPE.
LAYER_TEST_CONFIGS = [
    dict(testcase_name="_SWA_ROPE", layer_id=0),
    dict(testcase_name="_GLOBAL_NOPE", layer_id=3),
]

# Splash / tokamax attention patches vs. SDPA, on the debugmodel's attention
# (head_dim 64, window 128): layer 0 is sliding-window + RoPE (Splash
# LocalMask), layer 3 global + NoPE (CausalMask).
SPLASH_TEST_CONFIGS = [
    dict(
        testcase_name=f"_{kernel}_{layer_name}",
        kernel=kernel,
        layer_id=layer_id,
    )
    for kernel in ("splash", "splash_batch_native", "tokamax")
    for layer_name, layer_id in (("SWA_ROPE", 0), ("GLOBAL_NOPE", 3))
]

FULL_MODEL_TEST_CONFIGS = [
    dict(
        testcase_name="_TESTMODEL",
        model_name="testmodel",
        loss_atol=1e-1,
        # Adam (lr 0.01) amplifies the bf16-pass attention error, so the step-2
        # loss drifts further on TPU: 0.33 off on the OSS torch_tpu stack.
        loss_atol_tpu=5e-1,
        loss_rtol=1e-2,
        grad_atol=2.5,  # Grad noise builds up over multiple steps
        grad_atol_tpu=5.0,  # Plus bf16-pass attention on TPU (see _attn_tol)
        grad_rtol=1e-2,
        param_atol=8e-2,
        param_rtol=1e-2,
    ),
]


class MuseGlimmerTest(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for Muse Glimmer model components on CPU and accelerator devices.

  This class contains tests to ensure parity between CPU and accelerator device
  implementations of various Muse Glimmer model layers and full model training.
  """

  def setUp(self):
    super().setUp()
    # torch_tpu defaults fp32 matmuls to a single bf16 pass ("medium"); run
    # the non-attention matmuls at full fp32 so those checks stay strict.
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    self.addCleanup(torch.set_float32_matmul_precision, previous_precision)

  def _is_tpu(self) -> bool:
    return self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU

  def _attn_tol(self, strict: float, tpu: float) -> float:
    """Tolerance for checks whose computation goes through attention.

    torch_tpu routes every F.scaled_dot_product_attention call (even with only
    the MATH backend enabled) to its own SHLO kernel, whose matmuls ignore
    ``torch.set_float32_matmul_precision`` and so run at bf16-pass precision.
    Muse Glimmer's large post-QK-norm query scale (~43.8 / sqrt(head_dim))
    amplifies that error, so these checks need looser tolerances on TPU only.

    Args:
      strict: Tolerance on CPU/GPU.
      tpu: Tolerance on TPU.

    Returns:
      The tolerance for the current device.
    """
    return tpu if self._is_tpu() else strict

  def _setup_module_cpu_device_from_config(self, module_config):
    """Builds a module on CPU with initialized states plus a device copy.

    Unlike ``_setup_module_cpu_device``, this does not call
    ``.to(dtype=...)``: ``nn.Module.to`` casts complex tensors too, which would
    drop the imaginary part of the complex RoPE cache.

    Args:
      module_config: A torchtitan ``Module.Config``.

    Returns:
      A ``(module_cpu, module_device)`` tuple with identical weights.
    """
    module_cpu = module_config.build()
    module_cpu.init_states(buffer_device=torch.device("cpu"))
    module_cpu = module_cpu.cpu()
    module_device = copy.deepcopy(module_cpu).to(self.accelerator_device)
    return module_cpu, module_device

  def _check_relative_l2(self, test, reference, max_rel_error, check_name):
    """Checks ``||test - reference|| / ||reference|| <= max_rel_error``."""
    test = test.detach().float().cpu()
    reference = reference.detach().float().cpu()
    rel_error = (
        torch.linalg.vector_norm(test - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    ).item()
    print(
        f"[{check_name}] relative L2 error: {rel_error:.3e} (max"
        f" {max_rel_error:.1e}); reference max |x|:"
        f" {reference.abs().max().item():.3e}"
    )
    self.assertLessEqual(
        rel_error,
        max_rel_error,
        f"{check_name}: relative L2 error {rel_error:.3e} >"
        f" {max_rel_error:.1e}",
    )

  def _check_attention_parity(
      self, test, reference, *, tol, tpu_max_rel_error, check_name
  ):
    """Parity check for tensors computed through attention.

    CPU/GPU: per-element ``atol = rtol = tol``. TPU: whole-tensor relative L2
    error <= ``tpu_max_rel_error``. The bf16-pass SDPA error (see
    ``_attn_tol``) lands on a few elements, so a per-element bound loose enough
    for them would hide real bugs; relative L2 still catches a wrong mask, scale
    or layout (those give O(1) error).

    Args:
      test: Device tensor.
      reference: CPU reference tensor.
      tol: Per-element tolerance on CPU/GPU.
      tpu_max_rel_error: Maximum relative L2 error on TPU.
      check_name: Name for failure messages.
    """
    if self._is_tpu():
      self._check_relative_l2(
          test, reference, tpu_max_rel_error, check_name=check_name
      )
    else:
      test_utils.check_equivalence(
          test.cpu(), reference, atol=tol, rtol=tol, check_name=check_name
      )

  def _perturb_gain_norms(self, *modules):
    """Moves gain-centered norm weights off their init of 0 (identically)."""
    for module in modules:
      generator = torch.Generator().manual_seed(1234)
      with torch.no_grad():
        for name, param in module.named_parameters():
          if "norm" in name:
            param.copy_(
                0.1 * torch.randn(param.shape, generator=generator).to(param)
            )

  @parameterized.named_parameters(*SDPA_TEST_CONFIGS)
  def test_muse_glimmer_sliding_window_sdpa_cpu_device_parity(
      self, window_size
  ):
    """Tests windowed SDPA vs. an explicit masked-softmax reference (GQA)."""
    batch, seq_len, n_heads, n_kv_heads, head_dim = 2, 16, 4, 2, 8
    scale = 1.0 / math.sqrt(head_dim)

    # CPU and DEVICE setup
    inner_cpu = tpu_attention.SlidingWindowScaledDotProductInnerAttention.Config(
        window_size=window_size
    ).build()
    inner_device = copy.deepcopy(inner_cpu).to(self.accelerator_device)
    q_cpu, q_device = self._create_input_tensor_cpu_device(
        (batch, seq_len, n_heads, head_dim)
    )
    k_cpu, k_device = self._create_input_tensor_cpu_device(
        (batch, seq_len, n_kv_heads, head_dim)
    )
    v_cpu, v_device = self._create_input_tensor_cpu_device(
        (batch, seq_len, n_kv_heads, head_dim)
    )
    expected = _naive_sliding_window_attention(
        q_cpu, k_cpu, v_cpu, window_size, scale
    )

    with self.subTest(name="ForwardPass4D"):
      out_device = inner_device(
          q_device, k_device, v_device, scale=scale, enable_gqa=True
      )
      self._check_attention_parity(
          out_device,
          expected,
          tol=5e-3,
          tpu_max_rel_error=1e-2,
          check_name="SDPA 4D",
      )

    with self.subTest(name="PackedDocuments4D"):
      # Row 0 packs documents of lengths 7 and 9; row 1 is one document.
      positions_BL = torch.stack([
          torch.cat([torch.arange(7), torch.arange(seq_len - 7)]),
          torch.arange(seq_len),
      ])
      expected_packed = _naive_sliding_window_attention(
          q_cpu,
          k_cpu,
          v_cpu,
          window_size,
          scale,
          segment_ids_BL=segment_ids_from_positions(positions_BL),
      )
      out_device = inner_device(
          q_device,
          k_device,
          v_device,
          scale=scale,
          enable_gqa=True,
          positions=positions_BL.to(self.accelerator_device),
      )
      self._check_attention_parity(
          out_device,
          expected_packed,
          tol=5e-3,
          tpu_max_rel_error=1e-2,
          check_name="SDPA Packed 4D",
      )

    with self.subTest(name="ForwardPassFolded3D"):
      # Muse Glimmer feeds (B*L, H, D) tensors; the shim restores the 4D view.
      with set_unflatten_seq_len(seq_len):
        out_cpu = inner_cpu(
            q_cpu.reshape(-1, n_heads, head_dim),
            k_cpu.reshape(-1, n_kv_heads, head_dim),
            v_cpu.reshape(-1, n_kv_heads, head_dim),
            scale=scale,
            enable_gqa=True,
        )
      test_utils.check_equivalence(
          out_cpu,
          expected.reshape(-1, n_heads, head_dim),
          atol=1e-5,
          rtol=1e-5,
          check_name="SDPA Folded 3D",
      )

  @parameterized.named_parameters(*LAYER_TEST_CONFIGS)
  def test_muse_glimmer_attention_cpu_device_parity(self, layer_id):
    """Tests the CPU vs. DEVICE parity of the Attention layer."""
    config = _get_model_config()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    attention_cpu, attention_device = (
        self._setup_module_cpu_device_from_config(
            config.layers[layer_id].attention
        )
    )
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch * seq_len, config.dim), requires_grad=True
    )
    positions_cpu = torch.arange(seq_len, dtype=torch.long).repeat(batch)
    positions_device = positions_cpu.to(self.accelerator_device)

    with self.subTest(name="ForwardPass"):
      # Forward pass
      with set_unflatten_seq_len(seq_len):
        out_cpu = attention_cpu(x_cpu, None, positions_cpu)
        out_device = attention_device(x_device, None, positions_device)
      self._check_attention_parity(
          out_device,
          out_cpu,
          tol=5e-3,
          tpu_max_rel_error=1e-2,
          check_name="Attention Forward",
      )
      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()
        self._check_attention_parity(
            x_device.grad,
            x_cpu.grad,
            tol=2e-2,
            tpu_max_rel_error=5e-2,
            check_name="Attention Input Grad",
        )
        for (name, p_cpu), p_device in zip(
            attention_cpu.named_parameters(), attention_device.parameters()
        ):
          self._check_attention_parity(
              p_device.grad,
              p_cpu.grad,
              tol=5e-2,
              tpu_max_rel_error=5e-2,
              check_name=f"Attention Param Grad: {name}",
          )

  def test_muse_glimmer_embedding_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the EmbeddingWithNorm layer."""
    config = _get_model_config()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    embedding_cpu, embedding_device = (
        self._setup_module_cpu_device_from_config(config.tok_embeddings)
    )
    tokens_cpu = torch.randint(
        0, config.vocab_size, (batch * seq_len,), device="cpu"
    )
    tokens_device = tokens_cpu.to(self.accelerator_device)

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = embedding_cpu(tokens_cpu)
      out_device = embedding_device(tokens_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=1e-4,
          rtol=1e-4,
          check_name="Embedding Forward",
      )
      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()
        for (name, p_cpu), p_device in zip(
            embedding_cpu.named_parameters(), embedding_device.parameters()
        ):
          test_utils.check_equivalence(
              p_device.grad.cpu(),
              p_cpu.grad,
              atol=1e-3,
              rtol=1e-3,
              check_name=f"Embedding Param Grad: {name}",
          )

  def test_muse_glimmer_output_head_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the SoftCappedLinear output head."""
    config = _get_model_config()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    head_cpu, head_device = self._setup_module_cpu_device_from_config(
        config.lm_head
    )
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch * seq_len, config.dim), requires_grad=True
    )

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = head_cpu(x_cpu)
      out_device = head_device(x_device)
      # SoftCappedLinear always returns float32 logits.
      self.assertEqual(out_device.dtype, torch.float32)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=5e-3,
          rtol=5e-3,
          check_name="Output Head Forward",
      )
      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()
        test_utils.check_equivalence(
            x_device.grad.cpu(),
            x_cpu.grad,
            atol=5e-3,
            rtol=5e-3,
            check_name="Output Head Input Grad",
        )
        for (name, p_cpu), p_device in zip(
            head_cpu.named_parameters(), head_device.parameters()
        ):
          test_utils.check_equivalence(
              p_device.grad.cpu(),
              p_cpu.grad,
              atol=2e-2,
              rtol=2e-2,
              check_name=f"Output Head Param Grad: {name}",
          )

  @parameterized.named_parameters(*LAYER_TEST_CONFIGS)
  def test_muse_glimmer_transformer_block_cpu_device_parity(self, layer_id):
    """Tests the CPU vs. DEVICE parity of the post-norm TransformerBlock."""
    config = _get_model_config()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    transformer_block_cpu, transformer_block_device = (
        self._setup_module_cpu_device_from_config(config.layers[layer_id])
    )
    self._perturb_gain_norms(transformer_block_cpu, transformer_block_device)
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch * seq_len, config.dim), requires_grad=True
    )
    positions_cpu = torch.arange(seq_len, dtype=torch.long).repeat(batch)
    positions_device = positions_cpu.to(self.accelerator_device)

    with self.subTest(name="ForwardPass"):
      # Forward pass
      with set_unflatten_seq_len(seq_len):
        out_cpu = transformer_block_cpu(x_cpu, None, positions_cpu)
        out_device = transformer_block_device(x_device, None, positions_device)
      self._check_attention_parity(
          out_device,
          out_cpu,
          tol=5e-3,
          tpu_max_rel_error=1e-2,
          check_name="TransformerBlock Forward",
      )
      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()
        grads = [("Input Grad", x_device.grad, x_cpu.grad)] + [
            (f"Param Grad: {name}", p_device.grad, p_cpu.grad)
            for (name, p_cpu), p_device in zip(
                transformer_block_cpu.named_parameters(),
                transformer_block_device.parameters(),
            )
        ]
        for label, grad_device, grad_cpu in grads:
          check_name = f"TransformerBlock {label}"
          if self._is_tpu():
            # The eps=1e-8 post-norms rescale the small attention/FFN
            # outputs, so grads are large and bf16-pass attention error (see
            # _attn_tol) shows up in a few elements; compare whole tensors.
            self._check_relative_l2(
                grad_device,
                grad_cpu,
                max_rel_error=5e-2,
                check_name=check_name,
            )
          else:
            tol = 2e-2 if label == "Input Grad" else 5e-2
            test_utils.check_equivalence(
                grad_device.cpu(),
                grad_cpu,
                atol=tol,
                rtol=tol,
                check_name=check_name,
            )

  @parameterized.named_parameters(*SPLASH_TEST_CONFIGS)
  def test_muse_glimmer_splash_attention_matches_sdpa(self, kernel, layer_id):
    """Tests a Splash-patched Attention against the SDPA one on packed docs.

    Covers the Muse window ``W`` -> Splash ``LocalMask`` ``W - 1`` conversion at
    a context longer than the window, and packed-document masking through the
    ``positions`` that ``TpuMuseGlimmerAttention`` passes to the kernel. The
    reference is the unpatched module on CPU (window + document mask in SDPA).

    Args:
      kernel: ``splash`` (document-aware block skipping),
        ``splash_batch_native`` (batch-native masking) or ``tokamax``.
      layer_id: Debugmodel layer whose attention config is used.
    """
    if not self._is_tpu():
      self.skipTest("Splash and tokamax attention are TPU-only Pallas kernels.")
    config = _get_model_config("debugmodel")
    window_size = config.layers[layer_id].attention.window_size
    batch, seq_len = 2, 512
    self.assertTrue(window_size is None or window_size < seq_len)

    attention_cpu, attention_device = (
        self._setup_module_cpu_device_from_config(
            config.layers[layer_id].attention
        )
    )
    sa_config = SplashAttentionKernelConfig(
        sa_use_document_aware_block_skipping=(kernel == "splash")
    )
    if kernel == "tokamax":
      workarounds.use_tokamax_splash_attention_patch(
          attention_device, sa_config
      )
    elif kernel in ("splash", "splash_batch_native"):
      workarounds.use_splash_attention_patch(attention_device, sa_config)
    else:
      raise ValueError(f"Unexpected kernel: {kernel!r}")

    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch * seq_len, config.dim), requires_grad=True
    )
    # Row 0 packs documents of lengths 200, 180 and 132, so boundaries fall
    # inside the window; row 1 is a single document.
    positions_cpu = torch.cat([
        torch.arange(200),
        torch.arange(180),
        torch.arange(seq_len - 380),
        torch.arange(seq_len),
    ])
    positions_device = positions_cpu.to(self.accelerator_device)

    with set_unflatten_seq_len(seq_len):
      out_cpu = attention_cpu(x_cpu, None, positions_cpu)
      out_device = attention_device(x_device, None, positions_device)
      # Sensitivity: without the document mask the reference moves by far more
      # than the tolerance, so a kernel ignoring documents would fail below.
      with torch.no_grad():
        out_cpu_unpacked = attention_cpu(
            x_cpu, None, torch.arange(seq_len).repeat(batch)
        )
    unpacked_rel_diff = (
        torch.linalg.vector_norm(out_cpu_unpacked - out_cpu)
        / torch.linalg.vector_norm(out_cpu)
    ).item()
    self.assertGreater(unpacked_rel_diff, 0.1)
    self._check_relative_l2(
        out_device, out_cpu, 1e-2, check_name=f"{kernel} Attention Forward"
    )

    grad_out = torch.randn(
        out_cpu.shape, generator=torch.Generator().manual_seed(0)
    )
    out_cpu.backward(grad_out)
    out_device.backward(grad_out.to(self.accelerator_device))
    self._check_relative_l2(
        x_device.grad, x_cpu.grad, 2.5e-2, check_name=f"{kernel} Input Grad"
    )
    for (name, p_cpu), p_device in zip(
        attention_cpu.named_parameters(), attention_device.parameters()
    ):
      self._check_relative_l2(
          p_device.grad,
          p_cpu.grad,
          2.5e-2,
          check_name=f"{kernel} Param Grad: {name}",
      )

  @parameterized.named_parameters(*FULL_MODEL_TEST_CONFIGS)
  def test_muse_glimmer_full_model_training_steps_cpu_device_parity(
      self,
      model_name,
      loss_atol,
      loss_atol_tpu,
      loss_rtol,
      grad_atol,
      grad_atol_tpu,
      grad_rtol,
      param_atol,
      param_rtol,
  ):
    """Tests CPU vs. DEVICE parity of full model after 3 training steps."""
    config = _get_model_config(model_name)
    batch, seq_len = 2, 16  # seq_len > window so SWA is exercised.

    model_cpu = MuseGlimmerModel(config)
    model_cpu.init_weights(buffer_device=torch.device("cpu"))
    model_cpu = model_cpu.cpu()

    self._run_full_model_training_steps_parity_test(
        model_cpu=model_cpu,
        loss_atol=self._attn_tol(loss_atol, loss_atol_tpu),
        loss_rtol=loss_rtol,
        grad_atol=self._attn_tol(grad_atol, grad_atol_tpu),
        grad_rtol=grad_rtol,
        param_atol=param_atol,
        param_rtol=param_rtol,
        vocab_size=config.vocab_size,
        batch=batch,
        seq_len=seq_len,
    )


if __name__ == "__main__":
  absltest.main()
