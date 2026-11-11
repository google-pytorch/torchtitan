"""Distributed tests for Muse Glimmer TPU parallelization (TP, FSDP2, SimpleFSDP)."""

import copy

from absl import logging
from absl.testing import absltest
import torch
from torch.distributed.tensor import DTensor
from torchtitan.config.configs import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.testing import base_distributed_device_test
from torchtitan.experiments.tpu.distributed import distributed
from torchtitan.experiments.tpu.testing.base_distributed_device_test import InputDistribution
import torchtitan.experiments.tpu.muse_glimmer as muse_glimmer_tpu
from torchtitan.experiments.tpu.muse_glimmer.infra import parallelize as muse_glimmer_parallelize
from torchtitan.models.muse_glimmer.model import MuseGlimmerModel
from torchtitan.models.muse_glimmer.sharding import set_muse_glimmer_sharding_config

# To use OVERRIDEABLE SDP on TPU, the local batch size must be >= 4, i.e.
# TEST_BATCH_SIZE / world_size >= 4.
TEST_BATCH_SIZE = 32
# Longer than the testmodel sliding window, and <= its max context length.
TEST_SEQ_LEN = 128
TEST_TRAINING_STEPS = 3


def _get_model_config() -> MuseGlimmerModel.Config:
  """Returns a copy of the TPU ``testmodel`` config with TP sharding attached.

  The trainer attaches the sharding config via
  ``MuseGlimmerModel.Config.update_from_config``; without it
  ``Module.parallelize`` is a no-op under spmd_types. Deepcopy keeps the shared
  registry entry unchanged. SP is off because the test feeds unsharded inputs.
  """
  config = copy.deepcopy(muse_glimmer_tpu.muse_glimmer_configs["testmodel"])
  set_muse_glimmer_sharding_config(config, enable_sp=False)
  return config


def _parallel_dims(*, world_size: int, tp: int = 1, dp_shard: int = 1) -> ParallelDims:
  return ParallelDims(
      dp_shard=dp_shard,
      dp_replicate=1,
      cp=1,
      tp=tp,
      pp=1,
      ep=1,
      world_size=world_size,
  )


def _gather_dtensor_output(module, args, output):
  """Forward hook: returns full logits so the runner's plain-tensor loss works.

  With real TP sharding, ``lm_head`` emits vocab-sharded DTensor logits (loss
  parallel layout). ``full_tensor`` is differentiable.
  """
  del module, args  # Unused.
  if isinstance(output, DTensor):
    return output.full_tensor()
  return output


def _tp_wrapper(world_size: int, parallel_dims: ParallelDims):
  def apply_tp(model):
    # Under spmd_types ``parallelize_muse_glimmer`` applies FSDP even with no
    # DP axis (it is what turns the annotated params into DTensors), and FSDP's
    # MixedPrecisionPolicy would then cast params to bf16. Keep fp32 so this
    # TP-only test still compares against the fp32 CPU reference at the
    # tolerance it always used.
    training = TrainingConfig()
    import os as _os
    if _os.environ.get("MUSE_DBG_FP32", "1") == "1":
      training.mixed_precision_param = "float32"
      training.mixed_precision_reduce = "float32"
    print("[GRADDBG] TP wrapper mixed_precision_param=%s" % training.mixed_precision_param, flush=True)
    # pytype: disable=wrong-arg-types
    model = muse_glimmer_parallelize.parallelize_muse_glimmer(
        model,
        ac_config=(None if _os.environ.get("MUSE_DBG_NOAC") == "1" else SelectiveAC.Config()),
        compile_config=CompileConfig(),
        dump_folder="",
        parallel_dims=parallel_dims,
        parallelism=ParallelismConfig(tensor_parallel_degree=world_size),
        training=training,
    )
    # pytype: enable=wrong-arg-types
    model.register_forward_hook(_gather_dtensor_output)
    return model

  return apply_tp


def _verify_tp_forward_worker(device: torch.device, rank: int, world_size: int):
  """Forward parity of the TP model against a single-device reference."""
  parallel_dims = _parallel_dims(world_size=world_size, tp=world_size)
  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=MuseGlimmerModel,
      model_args=_get_model_config(),
      parallelism_func=_tp_wrapper(world_size, parallel_dims),
      parallel_dims=parallel_dims,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=True,
  )
  runner.run_forward_parity(TEST_BATCH_SIZE, TEST_SEQ_LEN, atol=5e-1, rtol=5e-1)


def _verify_tp_backward_worker(device: torch.device, rank: int, world_size: int):
  """Loss/grad parity of the TP model against a single-device reference."""
  parallel_dims = _parallel_dims(world_size=world_size, tp=world_size)
  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=MuseGlimmerModel,
      model_args=_get_model_config(),
      parallelism_func=_tp_wrapper(world_size, parallel_dims),
      parallel_dims=parallel_dims,
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


def _verify_fsdp_training_loop_worker(
    device: torch.device, rank: int, world_size: int
):
  """Multi-step loss/grad parity with FSDP2."""

  parallel_dims = _parallel_dims(world_size=world_size, dp_shard=world_size)

  def apply_fsdp(model):
    # pytype: disable=wrong-arg-types
    return muse_glimmer_parallelize.parallelize_muse_glimmer(
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
      model_class=MuseGlimmerModel,
      model_args=_get_model_config(),
      parallelism_func=apply_fsdp,
      parallel_dims=parallel_dims,
      input_distribution=InputDistribution.SPLIT_BATCH,
      use_meta_init=True,
  )
  runner.run_backward_parity(
      num_steps=TEST_TRAINING_STEPS,
      batch_size=TEST_BATCH_SIZE,
      seq_len=TEST_SEQ_LEN,
      atol_loss=3,  # Matches qwen3; see b/495494788.
      rtol_loss=3,
      atol_grad=9e-1,
      rtol_grad=9e-1,
  )


def _verify_simple_fsdp_training_loop_worker(
    device: torch.device, rank: int, world_size: int
):
  """Multi-step loss/grad parity with SimpleFSDP (the Ironwood target path)."""

  parallel_dims = _parallel_dims(world_size=world_size, dp_shard=world_size)

  def apply_simple_fsdp(model):
    # Under spmd_types ``parallelize_muse_glimmer`` applies FSDP even with no
    # DP axis (it is what turns the annotated params into DTensors), and FSDP's
    # MixedPrecisionPolicy would then cast params to bf16. Keep fp32 so this
    # TP-only test still compares against the fp32 CPU reference at the
    # tolerance it always used.
    training = TrainingConfig()
    training.mixed_precision_param = "float32"
    training.mixed_precision_reduce = "float32"
    # pytype: disable=wrong-arg-types
    return muse_glimmer_parallelize.parallelize_muse_glimmer(
        model,
        ac_config=SelectiveAC.Config(),
        compile_config=CompileConfig(),
        dump_folder="",
        parallel_dims=parallel_dims,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=world_size, use_simple_fsdp=True
        ),
        training=training,
    )
    # pytype: enable=wrong-arg-types

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=MuseGlimmerModel,
      model_args=_get_model_config(),
      parallelism_func=apply_simple_fsdp,
      parallel_dims=parallel_dims,
      input_distribution=InputDistribution.SPLIT_BATCH,
      use_meta_init=False,
  )
  runner.run_backward_parity(
      num_steps=TEST_TRAINING_STEPS,
      batch_size=TEST_BATCH_SIZE,
      seq_len=TEST_SEQ_LEN,
      atol_loss=1,
      rtol_loss=1,
      atol_grad=9e-1,
      rtol_grad=9e-1,
  )


class MuseGlimmerParallelizeTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests ``parallelize_muse_glimmer`` in a distributed environment."""

  def _run(self, func, label: str):
    logging.info(
        "Launching Muse Glimmer %s test with %d devices.",
        label,
        self.num_devices,
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=func,
    )  # pytype: disable=wrong-arg-types
    logging.info("Muse Glimmer %s test finished.", label)

  def test_tp_forward_equivalence_distributed(self):
    if self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
      # TODO: b/567185409 - TP forward hangs on TPU until the test timeout (a
      # worker is stuck in process_group_tpu item() -> copy to CPU). Possibly
      # the rank-0 reference forward running with no barrier while the other
      # ranks are in TP collectives. TP backward passes on TPU.
      self.skipTest("TP forward hangs on TPU: b/567185409")
    self._run(_verify_tp_forward_worker, "TP forward")

  def test_tp_backward_equivalence_noac_distributed(self):
    import os as _os
    _os.environ["MUSE_DBG_NOAC"] = "1"
    _os.environ["GRADDBG_HOOKS"] = "1"
    try:
      distributed.run_distributed(
          num_devices=self.num_devices,
          accelerator_device_type=self.accelerator_device_type,
          func=_verify_tp_backward_worker,
      )  # type: ignore
    finally:
      _os.environ.pop("MUSE_DBG_NOAC", None)
      _os.environ.pop("GRADDBG_HOOKS", None)

  def test_tp_backward_equivalence_hooks_distributed(self):
    import os as _os
    _os.environ["GRADDBG_HOOKS"] = "1"
    try:
      distributed.run_distributed(
          num_devices=self.num_devices,
          accelerator_device_type=self.accelerator_device_type,
          func=_verify_tp_backward_worker,
      )  # type: ignore
    finally:
      _os.environ.pop("GRADDBG_HOOKS", None)

  def test_tp_backward_equivalence_bf16_params_distributed(self):
    import os as _os
    _os.environ["MUSE_DBG_FP32"] = "0"
    try:
      distributed.run_distributed(
          num_devices=self.num_devices,
          accelerator_device_type=self.accelerator_device_type,
          func=_verify_tp_backward_worker,
      )  # type: ignore
    finally:
      _os.environ["MUSE_DBG_FP32"] = "1"

  def test_tp_backward_equivalence_distributed(self):
    self._run(_verify_tp_backward_worker, "TP backward")

  def test_fsdp_full_training_loop_equivalence_distributed(self):
    self._run(_verify_fsdp_training_loop_worker, "FSDP2 training loop")

  def test_simple_fsdp_full_training_loop_equivalence_distributed(self):
    self._run(_verify_simple_fsdp_training_loop_worker, "SimpleFSDP training loop")


if __name__ == "__main__":
  absltest.main()
