"""Tests for Llama3 model components on GPU and TPU devices.

This file contains tests to ensure parity between CPU and GPU/TPU device
implementations of various Llama3 model layers (including Embedding, Attention,
FeedForward, and TransformerBlock) as well as full model testing.
"""

from absl.testing import absltest
import torch
from torch import nn
import torch.nn.functional as F
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import test_utils
import torchtitan.experiments.tpu.llama3 as llama3_tpu
from torchtitan.models.common.attention import GQAttention as Attention, QKVLinear, ScaledDotProductInnerAttention
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock


def init_linear_weights(m: nn.Module):
  """Applies deterministic normal initialization to linear weights for testing."""
  for name, p in m.named_parameters():
    if "weight" in name:
      nn.init.normal_(p, std=0.02)
    elif "bias" in name:
      nn.init.zeros_(p)


def _get_model_config(model_name: str = "testmodel") -> Llama3Model.Config:
  """Retrieves a registered Llama3 configuration dynamically from TPU registry.

  Defaults to 'testmodel' (which specifies dim=64, n_heads=8, n_layers=1,
  vocab_size=128, max_seq_len=512).
  """
  return llama3_tpu.llama3_configs[model_name]


class Llama3Test(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for Llama3 model components on CPU and accelerator devices.

  This class contains tests to ensure parity between CPU and accelerator device
  implementations of various Llama3 model layers and full model testing.
  """

  def test_llama_embedding_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the Embedding layer."""
    config = _get_model_config()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    embedding_cpu, embedding_device = self._setup_module_cpu_device(
        nn.Embedding,
        init_fn=lambda m: nn.init.normal_(m.weight),
        num_embeddings=config.vocab_size,
        embedding_dim=config.dim,
    )

    tokens_cpu = torch.randint(
        0,
        config.vocab_size,
        (batch, seq_len),
        device="cpu",
        requires_grad=False,
    )
    tokens_device = tokens_cpu.to(self.accelerator_device)

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = embedding_cpu(tokens_cpu)
      out_device = embedding_device(tokens_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=1e-8,
          rtol=1e-8,
          check_name="Embedding Forward",
      )

      with self.subTest(name="BackwardPass"):
        # Backward pass
        # Add requires_grad=True for output to calculate gradients.
        out_cpu.requires_grad_(True)
        out_device.requires_grad_(True)
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()

        test_utils.check_equivalence(
            embedding_device.weight.grad.cpu(),
            embedding_cpu.weight.grad,
            atol=1e-9,
            rtol=1e-9,
            check_name="Embedding Weight Grad"
        )

  def test_view_as_complex_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of torch.view_as_complex."""

    batch, seq_len, dim = 4, 8, 64

    # CPU setup - last dimension must be 2 for view_as_complex
    x_cpu = torch.randn(
        batch, seq_len, dim, 2, device="cpu", requires_grad=True
    )

    # DEVICE setup
    x_device = (
        x_cpu.detach().clone().to(self.accelerator_device).requires_grad_(True)
    )
    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = torch.view_as_complex(x_cpu)
      out_device = torch.view_as_complex(x_device)

      # Check output parity
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=1e-9,
          rtol=1e-9,
          check_name="view_as_complex Forward"
      )

      # Verify output is complex
      assert out_cpu.is_complex()
      assert out_device.is_complex()

      # Verify shape: last dimension should be collapsed
      assert out_cpu.shape == (batch, seq_len, dim)
      assert out_device.shape == (batch, seq_len, dim)

      with self.subTest(name="BackwardPass"):
        # Create complex gradient by doing some operation
        loss_cpu = (out_cpu * (2 + 3j)).abs().sum()
        loss_device = (out_device * (2 + 3j)).abs().sum()

        loss_cpu.backward()
        loss_device.backward()

        # Check gradient parity
        test_utils.check_equivalence(
            x_device.grad.cpu(),
            x_cpu.grad,
            atol=1e-6,
            rtol=1e-6,
            check_name="view_as_complex Input Grad"
        )

  def test_apply_rotary_emb_cpu_device_parity(self):
    """Tests CPU vs. DEVICE parity for apply_rotary_emb function."""

    batch, seq_len, n_heads, head_dim = 8, 128, 16, 64

    # CPU setup
    xq_cpu = torch.randn(
        batch, seq_len, n_heads, head_dim, device="cpu", requires_grad=True
    )
    xk_cpu = torch.randn(
        batch, seq_len, n_heads, head_dim, device="cpu", requires_grad=True
    )
    theta = torch.randn(seq_len, head_dim // 2, device="cpu")
    freqs_cis_cpu = torch.exp(1j * theta)

    # DEVICE setup
    xq_device = (
        xq_cpu.detach().clone().to(self.accelerator_device).requires_grad_(True)
    )
    xk_device = (
        xk_cpu.detach().clone().to(self.accelerator_device).requires_grad_(True)
    )
    freqs_cis_device = freqs_cis_cpu.to(self.accelerator_device)

    with self.subTest(name="ForwardPass"):
      # Reshape freqs_cis to be broadcastable: [1, seq_len, 1, head_dim // 2]
      freqs_cis_cpu_reshaped = freqs_cis_cpu.view(1, seq_len, 1, head_dim // 2)
      freqs_cis_device_reshaped = freqs_cis_device.view(1, seq_len, 1, head_dim // 2)

      xq_out_cpu, xk_out_cpu = ComplexRoPE.apply_rotary_emb(xq_cpu, xk_cpu, freqs_cis_cpu_reshaped)
      xq_out_device, xk_out_device = ComplexRoPE.apply_rotary_emb(
          xq_device, xk_device, freqs_cis_device_reshaped
      )

      test_utils.check_equivalence(
          xq_out_device.cpu(),
          xq_out_cpu,
          atol=1e-7,
          rtol=1e-7,
          check_name="RoPE xq Forward",
      )
      test_utils.check_equivalence(
          xk_out_device.cpu(),
          xk_out_cpu,
          atol=1e-7,
          rtol=1e-7,
          check_name="RoPE xk Forward",
      )

      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = xq_out_cpu.sum() + xk_out_cpu.sum()
        loss_device = xq_out_device.sum() + xk_out_device.sum()

        loss_cpu.backward()
        loss_device.backward()

        test_utils.check_equivalence(
            xq_device.grad.cpu(),
            xq_cpu.grad,
            atol=1e-7,
            rtol=1e-7,
            check_name="RoPE xq Grad",
        )
        test_utils.check_equivalence(
            xk_device.grad.cpu(),
            xk_cpu.grad,
            atol=1e-7,
            rtol=1e-7,
            check_name="RoPE xk Grad",
        )

  def test_llama_attention_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the Attention layer."""
    config = _get_model_config()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    attention_config = config.layers[0].attention
    attention_cpu, attention_device = self._setup_module_cpu_device(
        Attention, init_fn=init_linear_weights, config=attention_config
    )
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch * seq_len, config.dim), requires_grad=True
    )
    with self.subTest(name="ForwardPass"):
      # Forward pass
      from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
      positions_cpu = torch.arange(seq_len, dtype=torch.long, device="cpu").repeat(batch)
      positions_device = positions_cpu.to(self.accelerator_device)
      with set_unflatten_seq_len(seq_len):
        out_cpu = attention_cpu(x_cpu, None, positions_cpu)
        out_device = attention_device(x_device, None, positions_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=1e-3,
          rtol=1e-3,
          check_name="Attention Forward",
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
            atol=1e-3,
            rtol=1e-3,
            check_name="Attention Input Grad",
        )
        for (name, p_cpu), p_device in zip(
            attention_cpu.named_parameters(), attention_device.parameters()
        ):
          test_utils.check_equivalence(
              p_device.grad.cpu(),
              p_cpu.grad,
              atol=5e-2,
              rtol=5e-2,
              check_name=f"Attention Param Grad: {name}",
          )

  def test_llama_feedforward_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the FeedForward layer."""
    config = _get_model_config()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    ffn_config = config.layers[0].feed_forward
    feedforward_cpu, feedforward_device = self._setup_module_cpu_device(
        FeedForward,
        init_fn=init_linear_weights,
        config=ffn_config,
    )

    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch * seq_len, config.dim), requires_grad=True
    )

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = feedforward_cpu(x_cpu)
      out_device = feedforward_device(x_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=1e-4,
          rtol=1e-4,
          check_name="FFN Forward",
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
            atol=1e-4,
            rtol=1e-4,
            check_name="FFN Input Grad",
        )
        for (name, p_cpu), p_dev in zip(
            feedforward_cpu.named_parameters(), feedforward_device.parameters()
        ):
          test_utils.check_equivalence(
              p_dev.grad.cpu(),
              p_cpu.grad,
              atol=5e-3,
              rtol=5e-3,
              check_name=f"FFN Param Grad: {name}",
          )

  def test_llama_transformer_block_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the TransformerBlock layer."""

    config = _get_model_config()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    block_config = config.layers[0]
    transformer_block_cpu, transformer_block_device = (
        self._setup_module_cpu_device(
            Llama3TransformerBlock,
            init_fn=init_linear_weights,
            config=block_config,
        )
    )
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch * seq_len, config.dim), requires_grad=True
    )
    with self.subTest(name="ForwardPass"):
      # Forward pass
      from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
      positions_cpu = torch.arange(seq_len, dtype=torch.long, device="cpu").repeat(batch)
      positions_device = positions_cpu.to(self.accelerator_device)
      with set_unflatten_seq_len(seq_len):
        out_cpu = transformer_block_cpu(x_cpu, None, positions_cpu)
        out_device = transformer_block_device(x_device, None, positions_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=2e-4,
          rtol=1e-5,
          check_name="TransformerBlock Forward",
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
            atol=1e-3,
            rtol=1e-3,
            check_name="TransformerBlock Input Grad",
        )
        for (name, p_cpu), p_dev in zip(
            transformer_block_cpu.named_parameters(),
            transformer_block_device.parameters(),
        ):
          # Default strict tolerances
          # Looser tolerances are needed throughout due to changes made
          # in cl/843793821.
          param_atol = 1e-3
          param_rtol = 1e-3
          # Relax atol for Attention params to cover BF16 noise floor.
          if "attention" in name:
            param_atol = 2e-2
          elif "feed_forward" in name:
            param_atol = 5e-3

          test_utils.check_equivalence(
              p_dev.grad.cpu(),
              p_cpu.grad,
              atol=param_atol,
              rtol=param_rtol,
              check_name=f"TransformerBlock Param Grad: {name}",
          )

  def test_llama_full_model_training_steps_cpu_device_parity(self):
    """Tests CPU vs. DEVICE parity of full Llama model after 3 training steps.

    Uses baseclass function wrapper for full model training steps parity test.
    """
    config = _get_model_config()
    batch, seq_len = 2, 8

    self._run_full_model_training_steps_parity_test(
        model_cpu=Llama3Model(config).cpu(),
        batch=batch,
        seq_len=seq_len,
        vocab_size=config.vocab_size,
        loss_atol=5e-3,
        loss_rtol=1e-2,
        grad_atol=0.1,  # Grad noise builds up over multiple steps
        grad_rtol=1e-2,
        param_atol=5e-2,
        param_rtol=1e-2,
    )

if __name__ == "__main__":
  absltest.main()
