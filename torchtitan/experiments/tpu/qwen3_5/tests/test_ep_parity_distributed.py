# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EP > 1 vs single-device parity for Qwen3.5 MoE on TPU (verification only).

Runs ``qwen35_moe_debugmodel`` (8 experts, top-2, 4 layers) for 3 steps on all
local TPU chips with FSDP and ``expert_parallel_degree`` in {1, 2, 4, 8}, then
replays the recorded inputs on a single-device CPU model (EP = 1) loaded with
the device run's initial weights, and checks loss, gradients and parameters at
every step.

Unlike the unit tests in ``test_qwen3_5_tpu.py``, nothing is mocked on the device
side: EP > 1 runs the real ``torch.tpu.ops.ragged_all_to_all`` dispatch and
combine (forward and backward), the ``all_gather_dim`` count exchange, and the
unpaired Pallas grouped matmuls with unpadded group sizes (``R = EP * T * k``
rows, of which only ``sum(group_sizes)`` are written). The EP = 1 case is the
baseline for TPU-vs-CPU numerics with the same tolerances.
"""

from absl.testing import absltest
from absl.testing import parameterized
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.testing import base_distributed_device_test

_CPU_CONFIG_FN = (
    "torchtitan.experiments.tpu.qwen3_5.tests.test_ep_parity_distributed:"
    "cpu_reference_config"
)


def cpu_reference_config(cfg):
  """Single-device CPU reference: EP = 1 (the device run's EP is kept in args)."""
  cfg.parallelism.expert_parallel_degree = 1
  return cfg


class EpParityDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):

  @parameterized.named_parameters(
      ("_ep1", 1),
      ("_ep2", 2),
      ("_ep4", 4),
      ("_ep8", 8),
  )
  def test_qwen3_5_moe_ep_matches_single_device(self, ep_degree: int):
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("Runs the real TPU ragged_all_to_all.")
    if self.num_devices % ep_degree:
      self.skipTest(f"EP={ep_degree} needs a multiple of {ep_degree} chips.")
    self._run_trainer_distributed_parity_test(
        config_args=[
            "--module=torchtitan.experiments.tpu.qwen3_5",
            "--config=qwen35_moe_debugmodel",
            # TrainerMinimal (used by the parity harness) builds the optimizer
            # without the model's per-mesh param-group split, and foreach
            # AdamW can't mix the dense (fsdp) and expert (efsdp, ep) meshes
            # in one group; the per-parameter loop can.
            "--optimizer.implementation=for-loop",
            "--hf_assets_path=tests/assets/tokenizer",
            "--dataloader.dataset_path=tests/assets/c4_test",
            "--training.max_context_length=128",
            "--dataloader.dataset=c4_test",
            "--training.steps=3",
            "--training.mixed_precision_param=float32",
            "--training.mixed_precision_reduce=float32",
            "--training.num_tokens_per_microbatch_per_dp_rank=256",
            "--parallelism.use_simple_fsdp",
            f"--parallelism.expert_parallel_degree={ep_degree}",
        ],
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        # Same tolerances as the dense qwen3_5_simplefsdp parity case.
        loss_atol=5e-3,
        loss_rtol=5e-3,
        grad_atol=1e-2,
        grad_rtol=5e-3,
        param_atol=1e-2,
        param_rtol=5e-3,
        cpu_config_fn=_CPU_CONFIG_FN,
    )


if __name__ == "__main__":
  absltest.main()
