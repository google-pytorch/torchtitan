
"""Tests for DeepSeekV3 model components on GPU and TPU devices.

This file contains tests to ensure parity between CPU and GPU/TPU device
implementations of various DeepSeekV3 model layers (including Embedding, and
Attention) as well as full model testing.
"""

from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
import torchtitan.experiments.tpu.deepseek_v3 as deepseek_v3_tpu
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch
from torchtitan.models.common.rope import RoPE
from torchtitan.models.deepseek_v3.model import Attention, DeepSeekV3Model, DeepSeekV3TransformerBlock


def _get_model_config(model_name: str = "testmodel") -> DeepSeekV3Model.Config:
  """Retrieves a registered DeepSeek V3 configuration dynamically from TPU registry.

  Defaults for 'testmodel' (Dense):
    - vocab_size = 2048, max_seq_len = 128, dim = 128
    - n_layers = 3, n_dense_layers = 3, n_heads = 8, dense_hidden_dim = 256
    - head_dim = 16, rope_theta = 10000.0, kv_lora_rank = 512

  Defaults for 'testmodel_moe' (MoE):
    - vocab_size = 2048, max_seq_len = 128, dim = 128
    - n_layers = 3, n_dense_layers = 1, n_heads = 8, dense_hidden_dim = 256
    - head_dim = 16, rope_theta = 10000.0, kv_lora_rank = 512
    - moe_hidden_dim = 128, num_experts = 8, router_top_k = 2
  """
  return deepseek_v3_tpu.deepseekv3_configs[model_name]


FULL_MODEL_TEST_CONFIGS = [
    dict(
        testcase_name="_DENSE",
        model_name="testmodel",
        loss_atol=1e-1,
        loss_rtol=1e-2,
        grad_atol=1.0,  # Grad noise builds up over multiple steps
        grad_rtol=1e-2,
        param_atol=8e-2,
        param_rtol=1e-2,
        is_moe=False,
    ),
    dict(
        testcase_name="_MOE",
        model_name="testmodel_moe",
        # May need looser tolerances for MOE
        loss_atol=1e-1,
        loss_rtol=1e-2,
        grad_atol=1.0,  # Grad noise builds up over multiple steps
        grad_rtol=1e-2,
        param_atol=8e-2,
        param_rtol=1e-2,
        is_moe=True,
    ),
]


class DeepseekV3Test(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for DeepSeekV3 model components on CPU and accelerator devices.

  This class contains tests to ensure parity between CPU and accelerator device
  implementations of various DeepSeekV3 model layers and full model testing.
  """

  def setUp(self):
    super().setUp()
    # Enable CPU histc workaround.
    use_cpu_safe_histc_patch()



  def test_deepseek_embedding_cpu_device_parity(self):
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
          out_cpu, out_device.cpu(), atol=1e-5, rtol=1e-5
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
            embedding_cpu.weight.grad,
            embedding_device.weight.grad.cpu(),
            atol=1e-9,
            rtol=1e-9,
        )

  def test_apply_rotary_emb_cpu_device_parity(self):
    """Tests CPU vs. DEVICE parity for apply_rotary_emb function."""
    config = _get_model_config()
    batch, seq_len = 2, 8
    head_dim = 16
    n_heads = 8

    # CPU setup
    x_cpu = torch.randn(
        batch, seq_len, n_heads, head_dim, device="cpu", requires_grad=True
    )
    rope_module = config.layers[0].attention.rope.build()
    freqs_cis_cpu = rope_module.cache[:seq_len]

    # DEVICE setup
    x_device = (
        x_cpu.detach().clone().to(self.accelerator_device).requires_grad_(True)
    )
    freqs_cis_device = freqs_cis_cpu.to(self.accelerator_device)

    with self.subTest(name="ForwardPass"):
      # Forward pass
      from torchtitan.models.common.rope import ComplexRoPE

      y_cpu = ComplexRoPE.apply_rotary_emb(x_cpu, x_cpu, freqs_cis_cpu)[0]
      y_device = ComplexRoPE.apply_rotary_emb(x_device, x_device, freqs_cis_device)[0]

      test_utils.check_equivalence(y_cpu, y_device.cpu(), atol=1e-5, rtol=1e-5)

      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = y_cpu.sum()
        loss_device = y_device.sum()

        loss_cpu.backward()
        loss_device.backward()

        test_utils.check_equivalence(
            x_cpu.grad, x_device.grad.cpu(), atol=1e-5, rtol=1e-5
        )

  def test_deepseek_attention_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the Attention layer."""
    config = _get_model_config()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    attention_cpu, attention_device = self._setup_module_cpu_device(
        Attention,
        init_fn=lambda m: None,
        config=config.layers[0].attention,
    )
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch * seq_len, config.dim), requires_grad=True
    )
    positions_cpu = torch.arange(seq_len, dtype=torch.long).repeat(batch)
    positions_device = positions_cpu.to(self.accelerator_device)

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = attention_cpu(x_cpu, None, positions_cpu)
      out_device = attention_device(x_device, None, positions_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=3e-2,
          rtol=3e-2,
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
            atol=1e-1,
            rtol=1e-1,
            check_name="Attention Input Grad",
        )

        for (name, p_cpu), p_device in zip(
            attention_cpu.named_parameters(), attention_device.parameters()
        ):
          test_utils.check_equivalence(
              p_device.grad.cpu(),
              p_cpu.grad,
              atol=1.0,
              rtol=4e-1,
              check_name=f"Attention Param Grad: {name}",
          )

  @parameterized.named_parameters(*FULL_MODEL_TEST_CONFIGS)
  def test_deepseek_transformer_block_cpu_device_parity(
      self,
      model_name,
      loss_atol,
      loss_rtol,
      grad_atol,
      grad_rtol,
      param_atol,
      param_rtol,
      is_moe,  # pylint: disable=unused-argument
  ):
    """Tests the CPU vs. DEVICE parity of the TransformerBlock layer."""
    config = _get_model_config(model_name)
    batch, seq_len, layer_id = 2, 8, 0
    block_config = config.layers[layer_id]
    transformer_block_cpu, transformer_block_device = (
        self._setup_module_cpu_device(
            DeepSeekV3TransformerBlock,
            lambda m: m.init_states(buffer_device=torch.device("cpu")),
            config=block_config,
        )
    )
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch * seq_len, config.dim), requires_grad=True
    )
    positions_cpu = torch.arange(seq_len, dtype=torch.long).repeat(batch)
    positions_device = positions_cpu.to(self.accelerator_device)

    out_cpu = transformer_block_cpu(x_cpu, None, positions_cpu)
    out_device = transformer_block_device(x_device, None, positions_device)
    test_utils.check_equivalence(
        out_device.cpu(),
        out_cpu,
        atol=3e-2,
        rtol=3e-2,
        check_name="TransformerBlock Forward",
    )

    out_cpu.sum().backward()
    out_device.sum().backward()
    test_utils.check_equivalence(
        x_device.grad.cpu(),
        x_cpu.grad,
        atol=8e-3,
        rtol=8e-3,
        check_name="TransformerBlock Input Grad",
    )
    for (name, p_cpu), p_device in zip(
        transformer_block_cpu.named_parameters(),
        transformer_block_device.parameters(),
    ):
      param_atol = 5e-3
      param_rtol = 5e-3
      # Relax tolerances for Attention layers to cover noise from new kernel.
      if "attention" in name:
        param_atol = 3e-1
        param_rtol = 3e-1
      test_utils.check_equivalence(
          p_device.grad.cpu(),
          p_cpu.grad,
          atol=param_atol,
          rtol=param_rtol,
          check_name=f"TransformerBlock Param Grad: {name}",
      )

  @parameterized.named_parameters(*FULL_MODEL_TEST_CONFIGS)
  def test_deepseek_complete_model_training_steps_cpu_device_parity(
      self,
      model_name,
      loss_atol,
      loss_rtol,
      grad_atol,
      grad_rtol,
      param_atol,
      param_rtol,
      is_moe,  # pylint: disable=unused-argument
  ):
    """Tests CPU vs DEVICE parity of full DeepSeek model after 3 training steps."""
    config = _get_model_config(model_name)
    batch, seq_len = 2, 8

    model_cpu = DeepSeekV3Model(config)
    model_cpu.init_weights(buffer_device=torch.device("cpu"))
    model_cpu = model_cpu.cpu()

    self._run_full_model_training_steps_parity_test(
        model_cpu=model_cpu,
        batch=batch,
        seq_len=seq_len,
        loss_atol=loss_atol,
        loss_rtol=loss_rtol,
        grad_atol=grad_atol,
        grad_rtol=grad_rtol,
        param_atol=param_atol,
        param_rtol=param_rtol,
        vocab_size=config.vocab_size,
    )


if __name__ == "__main__":
  absltest.main()
