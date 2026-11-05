"""Tests for Qwen3 Dense model components on GPU and TPU devices.

This file contains tests to ensure parity between CPU and GPU/TPU device
implementations of various Qwen3 Dense model layers (Embedding, Attention,
FeedForward, and TransformerBlock) and full model training.

NOTE: This file currently verifies the Qwen3 Dense and MoE implementation.
"""

import copy
from typing import Any, Dict, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import test_utils
import torchtitan.experiments.tpu.qwen3 as qwen3_tpu
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.rope import RoPE
from torchtitan.models.qwen3.model import Qwen3Model, Qwen3TransformerBlock


def _get_model_config(model_name: str = "testmodel") -> Qwen3Model.Config:
  """Retrieves a registered Qwen3 configuration dynamically from TPU registry.

  Defaults for 'testmodel' (Dense):
    - vocab_size = 2048, max_seq_len = 128, dim = 128
    - n_layers = 3, n_heads = 8, n_kv_heads = 8, hidden_dim = 256
    - head_dim = 16, rope_theta = 1000000.0, enable_weight_tying = True

  Defaults for 'testmodel_moe' (MoE):
    - vocab_size = 2048, max_seq_len = 128, dim = 128
    - n_layers = 3, n_heads = 8, n_kv_heads = 8, hidden_dim = 256
    - head_dim = 16, rope_theta = 1000000.0
    - moe_inter_dim = 128, num_experts = 8, top_k = 2
  """
  return qwen3_tpu.qwen3_configs[model_name]


FULL_MODEL_TEST_CONFIGS = [
    dict(
        testcase_name="_DENSE",
        model_name="testmodel",
        loss_atol=1e-1,
        loss_rtol=1e-2,
        grad_atol=2.5,  # Grad noise builds up over multiple steps (before: 1.0 at head; after: 2.5)
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
        is_moe=True,  # Used for skipping MOE TPU tests.
    ),
]


class Qwen3Test(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for Qwen3 model components (Dense and MoE) on CPU and accelerator devices.

  This class contains tests to ensure parity between CPU and accelerator device
  implementations of various Qwen3 model layers and full model training.
  """

  def setUp(self):
    super().setUp()
    # Enable CPU histc workaround.
    use_cpu_safe_histc_patch()

  def _create_rope_input_tensors_cpu_device(
      self,
      config: Qwen3Model.Config,
      batch: int = 2,
      seq_len: int = 8,
      requires_grad: bool = True,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates input tensors (x, rope_cache) necessary for the Attention module."""
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch, seq_len, config.dim), requires_grad=requires_grad
    )

    rope_module = config.rope.build()
    rope_cache_cpu = rope_module.cache[:seq_len]
    rope_cache_device = rope_cache_cpu.to(self.accelerator_device)

    return x_cpu, x_device, rope_cache_cpu, rope_cache_device

  def test_qwen_embedding_cpu_device_parity(self):
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
        0, config.vocab_size, (batch, seq_len), device="cpu"
    )
    tokens_device = tokens_cpu.to(self.accelerator_device)

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = embedding_cpu(tokens_cpu)
      out_device = embedding_device(tokens_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=1e-5,
          rtol=1e-5,
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
            atol=1e-5,
            rtol=1e-5,
            check_name="Embedding Weight Grad",
        )

  def test_qwen_attention_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the Attention layer."""
    config = _get_model_config()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    attention_cpu, attention_device = self._setup_module_cpu_device(
        GQAttention,
        init_fn=lambda m: None,
        config=config.layers[0].attention,
    )
    x_cpu, x_device = self._create_input_tensor_cpu_device((batch * seq_len, config.dim), requires_grad=True)
    positions_cpu = torch.arange(seq_len, dtype=torch.long, device="cpu").repeat(batch)
    positions_device = positions_cpu.to(self.accelerator_device)

    with self.subTest(name="ForwardPass"):
      # Forward pass
      from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
      with set_unflatten_seq_len(seq_len):
        out_cpu = attention_cpu(x_cpu, None, positions_cpu)
        out_device = attention_device(x_device, None, positions_device)
      # Note looser tolerance.
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=5e-3,
          rtol=5e-3,
          check_name="Attention Forward",
      )
      with self.subTest(name="BackwardPass"):
        # Backward pass
        # requires_grad=True per _create_rope_input_tensors_cpu_device().
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()
        # Gradient check (Note looser tolerance).
        test_utils.check_equivalence(
            x_device.grad.cpu(),
            x_cpu.grad,
            atol=2e-2,
            rtol=2e-2,
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

  def test_qwen_rmsnorm_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the RMSNorm layer."""
    config = _get_model_config()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    # Manual setup kept ad-hoc for simplicity.
    norm_cpu = nn.RMSNorm(config.dim, eps=config.norm.eps).cpu()
    norm_device = copy.deepcopy(norm_cpu).to(self.accelerator_device)
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch, seq_len, config.dim), requires_grad=True
    )

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = norm_cpu(x_cpu)
      out_device = norm_device(x_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=1e-5,
          rtol=1e-5,
          check_name="RMSNorm Forward",
      )
      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()
        # Gradient check (Note looser tolerance).
        test_utils.check_equivalence(
            x_device.grad.cpu(),
            x_cpu.grad,
            atol=5e-4,
            rtol=5e-4,
            check_name="RMSNorm Input Grad",
        )
        for (name, p_cpu), p_device in zip(
            norm_cpu.named_parameters(), norm_device.parameters()
        ):
          test_utils.check_equivalence(
              p_device.grad.cpu(),
              p_cpu.grad,
              atol=5e-4,
              rtol=5e-4,
              check_name=f"RMSNorm Param Grad: {name}",
          )

  def test_qwen_dense_feedforward_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the Dense FeedForward layer."""
    config = _get_model_config()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    feedforward_cpu, feedforward_device = self._setup_module_cpu_device(
        FeedForward,
        init_fn=lambda m: None,
        config=config.layers[0].feed_forward,
    )
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch * seq_len, config.dim), requires_grad=True
    )

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = feedforward_cpu(x_cpu)
      out_device = feedforward_device(x_device)
      # Note looser tolerance.
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=2e-3,
          rtol=2e-3,
          check_name="Dense FFN Forward",
      )
      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()
        # Gradient check (Note looser tolerance).
        test_utils.check_equivalence(
            x_device.grad.cpu(),
            x_cpu.grad,
            atol=5e-3,
            rtol=5e-3,
            check_name="Dense FFN Input Grad",
        )
        for (name, p_cpu), p_device in zip(
            feedforward_cpu.named_parameters(),
            feedforward_device.parameters(),
        ):
          test_utils.check_equivalence(
              p_device.grad.cpu(),
              p_cpu.grad,
              atol=2e-2,
              rtol=2e-2,
              check_name=f"Dense FFN Param Grad: {name}",
          )

  def test_qwen_moe_feedforward_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the raw MoE layer."""
    config = _get_model_config("testmodel_moe")
    batch, seq_len = 2, 8

    # CPU setup (uses standard config but cpu device)
    moe_cpu = MoE(config.layers[0].moe)
    # Initialize experts to bounded values to avoid raw memory garbage
    with torch.no_grad():
      for param in moe_cpu.parameters():
        param.uniform_(-0.02, 0.02)
    moe_cpu = moe_cpu.cpu()

    # Device setup (uses identical config but mapped to device)
    moe_device = MoE(config.layers[0].moe)
    moe_device.load_state_dict(moe_cpu.state_dict())
    moe_device = moe_device.to(self.accelerator_device)
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch * seq_len, config.dim), requires_grad=True
    )

    with self.subTest(name="ForwardPass"):
      # Forward pass
      from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
      with set_unflatten_seq_len(seq_len):
        out_cpu = moe_cpu(x_cpu)
        out_device = moe_device(x_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=3e-2,
          rtol=3e-2,
          check_name="MoE Forward",
      )

      with self.subTest(name="BackwardPass"):
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()

        test_utils.check_equivalence(
            x_device.grad.cpu(),
            x_cpu.grad,
            atol=3e-2,
            rtol=3e-2,
            check_name="MoE Input Grad",
        )
        for (name, p_cpu), p_device in zip(
            moe_cpu.named_parameters(), moe_device.parameters()
        ):
          test_utils.check_equivalence(
              p_device.grad.cpu(),
              p_cpu.grad,
              atol=5e-3,
              rtol=5e-3,
              check_name=f"MoE Param Grad: {name}",
          )

  @parameterized.named_parameters(*FULL_MODEL_TEST_CONFIGS)
  def test_qwen_transformer_block_cpu_device_parity(
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
            Qwen3TransformerBlock,
            lambda m: m.init_states(buffer_device=torch.device("cpu")),
            config=block_config,
        )
    )
    x_cpu, x_device = self._create_input_tensor_cpu_device((batch * seq_len, config.dim), requires_grad=True)
    positions_cpu = torch.arange(seq_len, dtype=torch.long, device="cpu").repeat(batch)
    positions_device = positions_cpu.to(self.accelerator_device)

    from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
    with set_unflatten_seq_len(seq_len):
      out_cpu = transformer_block_cpu(x_cpu, None, positions_cpu)
      out_device = transformer_block_device(x_device, None, positions_device)
    test_utils.check_equivalence(
        out_device.cpu(),
        out_cpu,
        atol=5e-3,
        rtol=5e-3,
        check_name="TransformerBlock Forward",
    )

    out_cpu.sum().backward()
    out_device.sum().backward()
    test_utils.check_equivalence(
        x_device.grad.cpu(),
        x_cpu.grad,
        atol=1e-1 if is_moe else 2e-3,
        rtol=1e-1 if is_moe else 2e-3,
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
        param_atol = 5e-2
        param_rtol = 5e-2
      elif "moe" in name:
        # MoE routing flips cause large grad diffs
        param_atol = 1.0
        param_rtol = 1e-1
      test_utils.check_equivalence(
          p_device.grad.cpu(),
          p_cpu.grad,
          atol=param_atol,
          rtol=param_rtol,
          check_name=f"TransformerBlock Param Grad: {name}",
      )

  @parameterized.named_parameters(*FULL_MODEL_TEST_CONFIGS)
  def test_qwen_full_model_training_steps_cpu_device_parity(
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
    """Tests CPU vs. DEVICE parity of full model after 3 training steps."""

    config = _get_model_config(model_name)
    batch, seq_len = 2, 8

    # For MoE's it is important to initialize model before passing to test
    # helper.
    model_cpu = Qwen3Model(config)
    model_cpu.init_weights(buffer_device=torch.device("cpu"))
    model_cpu = model_cpu.cpu()

    self._run_full_model_training_steps_parity_test(
        model_cpu=model_cpu,
        # Tolerances for Qwen3 are looser than Llama3.
        loss_atol=loss_atol,
        loss_rtol=loss_rtol,
        grad_atol=grad_atol,  # Grad noise builds up over multiple steps
        grad_rtol=grad_rtol,
        param_atol=param_atol,
        param_rtol=param_rtol,
        vocab_size=config.vocab_size,
        batch=batch,
        seq_len=seq_len,
    )


if __name__ == "__main__":
  absltest.main()
