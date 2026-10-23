"""Distributed tests for the Qwen3 DTensor TP."""

from absl import logging
from absl.testing import absltest
import torch
from torch.distributed.device_mesh import DeviceMesh
import torch.nn as nn
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.config.configs import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu.base_distributed_device_test import InputDistribution
import torchtitan.experiments.tpu.qwen3 as qwen3_tpu
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.experiments.tpu.qwen3.infra import parallelize as qwen3_dtensor_parallelize



# Constants for test parameters
# To use OVERRIDEABLE SDP on TPU, need to ensure local batch size is
# >= 4, i.e. TEST_BATCH_SIZE / world_size >= 4.
TEST_BATCH_SIZE = 32

# Need to make sure this is large enough for world size for sequence sharding.
# To use OVERRIDEABLE SDP on TPU, sequence length needs to be multiple of 512.
TEST_SEQ_LEN = 128

# Constants for training parameters
TEST_TRAINING_STEPS = 3


def _get_qwen3_model_config(model_name: str = "testmodel") -> Qwen3Model.Config:
  """Retrieves a registered Qwen3 configuration from TPU registry.

  Defaults to 'testmodel' (which specifies dim=128, head_dim=16, n_layers=3,
  vocab_size=2048, max_seq_len=128).
  """
  return qwen3_tpu.qwen3_configs[model_name]


def _verify_dtensor_qwen3_non_moe_tp_forward_worker(
    device: torch.device, rank: int, world_size: int
):
  """Verifies numerical equivalence of forward pass of Qwen3 model with DTensor TP."""

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
    qwen3_dtensor_parallelize.parallelize_qwen3(
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
      model_class=Qwen3Model,
      model_args=_get_qwen3_model_config(),
      parallelism_func=apply_tp_wrapper,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=True,
  )

  runner.run_forward_parity(TEST_BATCH_SIZE, TEST_SEQ_LEN, atol=5e-1, rtol=5e-1)


def _verify_dtensor_qwen3_tp_backward_worker(
    device: torch.device, rank: int, world_size: int
):
  """Worker function to run forward and backward pass on model after apply_tp is called."""

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
    qwen3_dtensor_parallelize.parallelize_qwen3(
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
      model_class=Qwen3Model,
      model_args=_get_qwen3_model_config(),
      parallelism_func=apply_tp_wrapper,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=True,
  )
  runner.run_backward_parity(
      num_steps=1,
      batch_size=TEST_BATCH_SIZE,
      seq_len=TEST_SEQ_LEN,
      rtol_loss=5e-3,
      atol_loss=5e-3,
  )


def _verify_dtensor_qwen3_fsdp_training_loop_worker(
    device: torch.device,
    rank: int,
    world_size: int,
):
  """Worker function to call the FSDP numerical equivalence helper on Qwen3 model."""

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
    qwen3_dtensor_parallelize.parallelize_qwen3(
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
      model_class=Qwen3Model,
      model_args=_get_qwen3_model_config(),
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


class Qwen3DTensorParallelizeTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests the DTensor-based `apply_non_moe_tp` in a distributed environment."""

  def test_apply_non_moe_tp_forward_equivalence_distributed(self):
    """Launches test to verify numerical equivalence of the DTensor parallel model."""
    logging.info(
        "Launching DTensor TP forward numerical equivalence test with %d"
        " devices on Qwen3 model.",
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_dtensor_qwen3_non_moe_tp_forward_worker,
    )  # pytype: disable=wrong-arg-types
    logging.info(
        "Distributed DTensor TP forward numerical equivalence test finished."
    )

  def test_apply_non_moe_tp_backward_equivalence_distributed(self):
    """Tests if the backward pass runs on model after apply_tp."""
    logging.info(
        "Launching DTensor TP backward numerical equivalence test with %d"
        " devices on Qwen3 model.",
    )
    distributed.run_distributed(
        self.num_devices,
        self.accelerator_device_type,
        _verify_dtensor_qwen3_tp_backward_worker,
    )
    logging.info(
        "Distributed DTensor TP backward numerical equivalence test finished."
    )

  def test_apply_fsdp_full_training_loop_equivalence_distributed(self):
    """Verifies numerical equivalence of a full training loop with FSDP."""
    logging.info(
        "Launching FSDP training equivalence test with %d devices on Qwen3"
        " model.",
        self.num_devices,
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_dtensor_qwen3_fsdp_training_loop_worker,
    )  # pytype: disable=wrong-arg-types
    logging.info("Distributed FSDP training equivalence test finished.")


if __name__ == "__main__":
  absltest.main()
