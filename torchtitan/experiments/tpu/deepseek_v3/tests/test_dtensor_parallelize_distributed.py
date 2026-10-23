"""Distributed tests for the DeepSeekV3 DTensor TP."""

from absl import logging
from absl.testing import absltest
import torch
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.config.configs import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu.base_distributed_device_test import InputDistribution
import torchtitan.experiments.tpu.deepseek_v3 as deepseek_v3_tpu
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch
from torchtitan.models.deepseek_v3.model import DeepSeekV3Model
import torchtitan.models.deepseek_v3.parallelize as deepseek_v3_dtensor_parallelize


# Constants for test parameters
# To use OVERRIDEABLE SDP on TPU, need to ensure local batch size is
# >= 4, i.e. TEST_BATCH_SIZE / world_size >= 4.
TEST_BATCH_SIZE = 32

# Need to make sure this is large enough for world size for sequence sharding.
# To use OVERRIDEABLE SDP on TPU, sequence length needs to be multiple of 512.
TEST_SEQ_LEN = 128

# Constants for training parameters
TEST_TRAINING_STEPS = 3


def _get_deepseek_v3_model_config(
    model_name: str = "testmodel",
) -> DeepSeekV3Model.Config:
  """Retrieves a registered DeepSeek V3 configuration from TPU registry.

  Defaults to 'testmodel' (which specifies dim=128, n_layers=3,
  n_dense_layers=3, vocab_size=2048).
  """
  return deepseek_v3_tpu.deepseekv3_configs[model_name]


def _verify_dtensor_deepseek_v3_non_moe_tp_forward_worker(
    device: torch.device, rank: int, world_size: int
):
  """Verifies numerical equivalence of forward pass of DeepSeekV3 model with DTensor TP."""
  # Apply histc CPU workaround in worker process. Doesn't propagate from parent.
  use_cpu_safe_histc_patch()

  def apply_tp_wrapper(model):
    parallel_dims = ParallelDims(
        dp_shard=1,
        dp_replicate=1,
        cp=1,
        tp=world_size,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    # pytype: disable=wrong-arg-types
    deepseek_v3_dtensor_parallelize.parallelize_deepseekv3(
        model,
        ac_config=SelectiveAC.Config(),
        compile_config=CompileConfig(),
        dump_folder="",
        parallel_dims=parallel_dims,
        parallelism=ParallelismConfig(tensor_parallel_degree=world_size),
        training=TrainingConfig(),
    )
    # pytype: enable=wrong-arg-types

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=DeepSeekV3Model,
      model_args=_get_deepseek_v3_model_config(),
      parallelism_func=apply_tp_wrapper,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=True,
  )

  runner.run_forward_parity(TEST_BATCH_SIZE, TEST_SEQ_LEN, atol=2e-1, rtol=1e-1)


def _verify_dtensor_deepseek_v3_tp_backward_worker(
    device: torch.device, rank: int, world_size: int
):
  """Worker function to run forward andbackward pass on model after apply_tp is called."""
  # Apply histc CPU workaround in worker process. Doesn't propagate from parent.
  use_cpu_safe_histc_patch()

  def apply_tp_wrapper(model):
    parallel_dims = ParallelDims(
        dp_shard=1,
        dp_replicate=1,
        cp=1,
        tp=world_size,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    # pytype: disable=wrong-arg-types
    deepseek_v3_dtensor_parallelize.parallelize_deepseekv3(
        model,
        ac_config=SelectiveAC.Config(),
        compile_config=CompileConfig(),
        dump_folder="",
        parallel_dims=parallel_dims,
        parallelism=ParallelismConfig(tensor_parallel_degree=world_size),
        training=TrainingConfig(),
    )
    # pytype: enable=wrong-arg-types

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=DeepSeekV3Model,
      model_args=_get_deepseek_v3_model_config(),
      parallelism_func=apply_tp_wrapper,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=True,
  )
  # This test is already flaky at head with default tolerances of
  # (atol_loss=1e-3, rtol_loss=1e-3, atol_grad=1e-3, rtol_grad=1e-3).
  # TODO(jerryx99): b/540005818 to audit tolerances
  runner.run_backward_parity(
      num_steps=TEST_TRAINING_STEPS,
      batch_size=TEST_BATCH_SIZE,
      seq_len=TEST_SEQ_LEN,
      atol_loss=1e-2,
      rtol_loss=1e-2,
      atol_grad=1e-2,
      rtol_grad=1e-2,
  )


def _verify_dtensor_deepseek_v3_fsdp_training_loop_worker(
    device: torch.device,
    rank: int,
    world_size: int,
):
  """Worker function to call the FSDP numerical equivalence helper on DeepSeekV3 model."""
  # Apply histc CPU workaround in worker process. Doesn't propagate from parent.
  use_cpu_safe_histc_patch()

  def apply_fsdp_wrapper(model):
    parallel_dims = ParallelDims(
        dp_shard=world_size,
        dp_replicate=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    # pytype: disable=wrong-arg-types
    deepseek_v3_dtensor_parallelize.parallelize_deepseekv3(
        model,
        ac_config=SelectiveAC.Config(),
        compile_config=CompileConfig(),
        dump_folder="",
        parallel_dims=parallel_dims,
        parallelism=ParallelismConfig(data_parallel_shard_degree=world_size),
        training=TrainingConfig(),
    )
    # pytype: enable=wrong-arg-types

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=DeepSeekV3Model,
      model_args=_get_deepseek_v3_model_config(),
      parallelism_func=apply_fsdp_wrapper,
      input_distribution=InputDistribution.SPLIT_BATCH,
      use_meta_init=True,
  )

  runner.run_backward_parity(
      num_steps=TEST_TRAINING_STEPS,
      batch_size=TEST_BATCH_SIZE,
      seq_len=TEST_SEQ_LEN,
      atol_loss=3,  # TODO(tanvibajpai): investigate tolerance issues b/495494788
      rtol_loss=3,
      atol_grad=9e-1,
      rtol_grad=9e-1,
  )


class DeepSeekV3DTensorParallelizeTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests the DTensor-based `apply_non_moe_tp` in a distributed environment."""

  # TP Tests are run with loss_parallel=False.
  def test_apply_non_moe_tp_forward_equivalence_distributed(self):
    """Launches test to verify numerical equivalence of the DTensor parallel model."""
    logging.info(
        "Launching DTensor TP forward numerical equivalence test with %d"
        " devices on DeepSeekV3 model.",
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_dtensor_deepseek_v3_non_moe_tp_forward_worker,
    )  # pytype: disable=wrong-arg-types
    logging.info(
        "Distributed DTensor TP forward numerical equivalence test finished."
    )

  def test_apply_non_moe_tp_backward_equivalence_distributed(self):
    """Tests if the backward pass runs on model after apply_tp."""
    logging.info(
        "Launching DTensor TP backward numerical equivalence test with %d"
        " devices on DeepSeekV3 model.",
    )
    distributed.run_distributed(
        self.num_devices,
        self.accelerator_device_type,
        _verify_dtensor_deepseek_v3_tp_backward_worker,
    )
    logging.info(
        "Distributed DTensor TP backward numerical equivalence test finished."
    )

  def test_apply_fsdp_full_training_loop_equivalence_distributed(self):
    """Verifies numerical equivalence of a full training loop with FSDP."""
    logging.info(
        "Launching FSDP training equivalence test with %d devices on DeepSeekV3"
        " model.",
        self.num_devices,
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_dtensor_deepseek_v3_fsdp_training_loop_worker,
    )  # pytype: disable=wrong-arg-types
    logging.info("Distributed FSDP training equivalence test finished.")


if __name__ == "__main__":
  absltest.main()
