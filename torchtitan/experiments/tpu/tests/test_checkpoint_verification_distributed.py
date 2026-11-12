"""Tests for checkpoint verification in Llama3 models."""

import os
import shutil
import tempfile

from absl import logging
from absl.testing import absltest
import torch
import torch.multiprocessing as mp
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.testing import base_distributed_device_test
import torchtitan.experiments.tpu.train
from torchtitan.experiments.tpu.train import TPUTrainer, start_trainer



class StartTrainerWithLossCapture:

  def __init__(self, output_path: str):
    self.output_path = output_path
    self.__name__ = "StartTrainerWithLossCapture"
    self.__qualname__ = "StartTrainerWithLossCapture"

  def __call__(self, config):
    losses = []

    original_init = TPUTrainer.__init__

    def patched_init(this, job_config):
      original_init(this, job_config)
      original_log = this.metrics_processor.log

      def patched_log(step, global_avg_loss, *args, **kwargs):
        original_log(step, global_avg_loss, *args, **kwargs)
        losses.append((step, global_avg_loss))

      this.metrics_processor.log = patched_log

    TPUTrainer.__init__ = patched_init  # pyrefly: ignore[bad-assignment]

    start_trainer(config)

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
      torch.save(losses, self.output_path)


class TrainCheckpointVerificationTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests the checkpoint verification.

  The test verifies that checkpointing produces identical loss to a continuous
  run. Specifically, the test runs:
  1. A continuous run for 5 steps.
  2. A checkpointed run, saving every 2 steps, for 2 steps. Then another
  checkpointed run, loading from the previous checkpoint, and running for
  additional 3 steps (total 5 steps).

  The test then compares the losses from the continuous run to the
  checkpointed run, and asserts that they are almost equal.
  """

  def test_llama3_fsdp_checkpoint_verification(self):
    """Verifies that checkpointing produces identical loss to a continuous run."""
    if self.accelerator_device_type not in [
        device_type.AcceleratorDeviceType.TPU,
        device_type.AcceleratorDeviceType.CUDA,
    ]:
      self.skipTest("This test is specifically for TPU or GPU.")
      return

    test_tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, test_tmp_dir, ignore_errors=True)
    checkpoint_dir = os.path.join(test_tmp_dir, "test_verification_checkpoint")
    continuous_output = os.path.join(test_tmp_dir, "continuous_losses.pt")
    checkpointed_output = os.path.join(test_tmp_dir, "checkpointed_losses.pt")

    # Upstream replaced seq_len/local_batch_size with a token budget:
    # num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len = 4 * 128.
    base_args = [
        "--module", "torchtitan.experiments.tpu.llama3",
        "--config", "llama3_debugmodel",
        "--dataloader.dataset_path=tests/assets/c4_test",
        "--hf_assets_path=tests/assets/tokenizer",
        "--training.max_context_length=128",
        "--training.num_tokens_per_microbatch_per_dp_rank=512",
        "--debug.deterministic",
        "--debug.seed=42",
    ]

    # Step 1: Continuous Run (5 steps)
    logging.info("Running Continuous Baseline")
    continuous_args = base_args + ["--training.steps=5"]
    self._test_train_distributed(
        config_args=continuous_args,
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        start_trainer=StartTrainerWithLossCapture(continuous_output),
    )

    # Step 2: Checkpointed Run - Save (2 steps)
    logging.info("Running Checkpointed Save")
    save_args = base_args + [
        "--training.steps=2",
        "--checkpoint.enable",
        "--checkpoint.interval=2",
        f"--checkpoint.folder={checkpoint_dir}",
    ]

    self._test_train_distributed(
        config_args=save_args,
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        start_trainer=torchtitan.experiments.tpu.train.start_trainer,  # pyrefly: ignore[bad-argument-type]
    )

    # Step 3: Checkpointed Run - Load & Resume (steps 3-5)
    logging.info("Running Checkpointed Load & Resume")
    load_args = base_args + [
        "--training.steps=5",
        "--checkpoint.enable",
        "--checkpoint.interval=2",
        f"--checkpoint.folder={checkpoint_dir}",
    ]
    self._test_train_distributed(
        config_args=load_args,
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        start_trainer=StartTrainerWithLossCapture(checkpointed_output),
    )

    # Step 4: Compare losses
    continuous_losses = torch.load(continuous_output)
    checkpointed_losses = torch.load(checkpointed_output)

    logging.info(f"Continuous losses: {continuous_losses}")
    logging.info(f"Checkpointed losses: {checkpointed_losses}")

    cont_dict = {step: loss for step, loss in continuous_losses}
    chk_dict = {step: loss for step, loss in checkpointed_losses}

    for step in range(3, 6):
      self.assertIn(step, cont_dict)
      self.assertIn(step, chk_dict)
      self.assertAlmostEqual(
          cont_dict[step],
          chk_dict[step],
          places=4,
          msg=f"Loss mismatch at step {step}",
      )

    logging.info("Checkpoint verification passed!")


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
