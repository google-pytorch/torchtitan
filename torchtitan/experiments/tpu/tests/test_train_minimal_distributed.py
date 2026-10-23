"""Tests for the minimal distributed training setup (includes parity checks for generic minimal trainer)."""

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_distributed_device_test


class TrainMinimalDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests train_minimal execution in distributed settings."""

  @parameterized.named_parameters([
      # deepseek_v3
      dict(
          testcase_name="deepseek_v3_tp",
          module="deepseek_v3",
          config="deepseek_v3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          skip_devices=[
              # TODO: b/482057603 - Issue with mixing DTensor and Tensor.
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
              device_type.AcceleratorDeviceType.TPU,
          ],
          loss_atol=7e-1,
          loss_rtol=5e-3,
          grad_atol=3e-1,
          grad_rtol=5e-3,
          param_atol=5e-3,
          param_rtol=5e-3,
      ),
      dict(
          testcase_name="deepseek_v3_fsdp",
          module="deepseek_v3",
          config="deepseek_v3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=[
              # TODO(alexanderb14): Error initializing torch.distributed
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
              # b/487028245 - Timing out w/Math Backend
              device_type.AcceleratorDeviceType.TPU,
          ],
          loss_atol=2e-1,
          loss_rtol=5e-3,
          grad_atol=2e-8,
          grad_rtol=5e-3,
          param_atol=5e-3,
          param_rtol=5e-3,
      ),
      # llama3
      dict(
          testcase_name="llama3_tp",
          module="llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          loss_atol=2e-2,
          loss_rtol=1e-3,
          grad_atol=2e-2,
          grad_rtol=1e-3,
          param_atol=1e-2,
          param_rtol=1e-3,
      ),
      dict(
          testcase_name="llama3_fsdp",
          module="llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=[
              # TODO(jerryx99): Investigate numerical divergence b/540005818
              device_type.AcceleratorDeviceType.CPU,
          ],
          loss_atol=1e-3,
          loss_rtol=1e-3,
          grad_atol=8e-4,
          grad_rtol=1e-3,
          param_atol=5e-3,
          param_rtol=1e-3,
      ),
      dict(
          testcase_name="llama3_simplefsdp",
          module="llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--parallelism.use_simple_fsdp"],
          skip_devices=[
              # TODO(jerryx99): Investigate numerical divergence b/540005818
              device_type.AcceleratorDeviceType.CPU,
          ],
          loss_atol=1e-3,
          loss_rtol=1e-3,
          grad_atol=8e-4,
          grad_rtol=1e-3,
          param_atol=5e-3,
          param_rtol=1e-3,
      ),
      # qwen3
      dict(
          testcase_name="qwen3_tp",
          module="qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          loss_atol=2e-2,
          loss_rtol=1e-3,
          grad_atol=2e-2,
          grad_rtol=1e-3,
          param_atol=1e-2,
          param_rtol=1e-3,
      ),
      dict(
          testcase_name="qwen3_fsdp",
          module="qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=[
             # TODO(jerryx99): Investigate numerical divergence b/540005818
              device_type.AcceleratorDeviceType.TPU,
          ],
          loss_atol=1e-3,
          loss_rtol=1e-3,
          grad_atol=2e-3,
          grad_rtol=1e-3,
          param_atol=5e-3,
          param_rtol=1e-3,
      ),
      dict(
          testcase_name="qwen3_simplefsdp",
          module="qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--parallelism.use_simple_fsdp"],
          skip_devices=[
              # TODO(jerryx99): Investigate numerical divergence b/540005818
              device_type.AcceleratorDeviceType.TPU,
          ],
          loss_atol=1e-3,
          loss_rtol=1e-3,
          grad_atol=2e-3,
          grad_rtol=1e-3,
          param_atol=5e-3,
          param_rtol=1e-3,
      ),
      # gemma4
      dict(
          testcase_name="gemma4_tp",
          module="examples.gemma4",
          config="gemma4_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          loss_atol=2e-2,
          loss_rtol=1e-3,
          grad_atol=5e-2,
          grad_rtol=5e-3,
          param_atol=1e-2,
          param_rtol=1e-3,
      ),
      dict(
          testcase_name="gemma4_fsdp",
          module="examples.gemma4",
          config="gemma4_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          loss_atol=1e-3,
          loss_rtol=1e-3,
          grad_atol=5e-3,
          grad_rtol=1e-3,
          param_atol=5e-3,
          param_rtol=1e-3,
      ),
      dict(
          testcase_name="gemma4_simplefsdp",
          module="examples.gemma4",
          config="gemma4_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--parallelism.use_simple_fsdp"],
          loss_atol=1e-3,
          loss_rtol=1e-3,
          grad_atol=5e-3,
          grad_rtol=1e-3,
          param_atol=5e-3,
          param_rtol=1e-3,
      ),
  ])
  def test_train_minimal_distributed_parity(
      self,
      module,
      config,
      data_parallel_shard_degree,
      tensor_parallel_degree,
      skip_devices=None,
      loss_atol=5e-3,
      loss_rtol=5e-3,  # Default atol/rtols
      grad_atol=5e-3,
      grad_rtol=5e-3,
      param_atol=5e-3,
      param_rtol=5e-3,
      optimizer_implementation="foreach",
      enable_compile=False,
      args=None,
  ):
    """Runs a distributed parameter and gradient parity test against a CPU reference.

    This explicitly tests the generic minimal training script
    (experiments/tpu/train_minimal.py).
    This test executes a few training steps on a mock distributed topology (like
    FSDP),
    recording the loss, gradients, and parameter states. It then mathematically
    verifies
    that a standard single-device CPU model executing identical inputs achieves
    the perfectly equivalent numerical values, proving the distributed
    implementation
    is correct.
    """
    # Upstream replaced seq_len/local_batch_size with a token budget:
    # num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len = 2 * 128.
    self._run_trainer_distributed_parity_test(
        config_args=[
            f"--module=torchtitan.experiments.tpu.{module}",
            f"--config={config}",
            f"--optimizer.implementation={optimizer_implementation}",
            "--hf_assets_path=tests/assets/tokenizer",
            "--dataloader.dataset_path=tests/assets/c4_test",
            "--training.max_context_length=128",
            "--dataloader.dataset=c4_test",
            "--training.steps=3",
            "--training.mixed_precision_param=float32",
            "--training.mixed_precision_reduce=float32",
            "--training.num_tokens_per_microbatch_per_dp_rank=256",
        ] + (args or []),
        tensor_parallel_degree=tensor_parallel_degree,
        data_parallel_shard_degree=data_parallel_shard_degree,
        skip_devices=skip_devices,
        loss_atol=loss_atol,
        loss_rtol=loss_rtol,
        grad_atol=grad_atol,
        grad_rtol=grad_rtol,
        param_atol=param_atol,
        param_rtol=param_rtol,
        enable_compile=enable_compile,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
