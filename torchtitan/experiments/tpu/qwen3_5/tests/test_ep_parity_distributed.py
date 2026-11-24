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
native ``torch._grouped_mm`` expert matmuls with unpadded group sizes
(``R = EP * T * k`` rows, of which only ``sum(group_sizes)`` are written). The
EP = 1 case is the baseline for TPU-vs-CPU numerics with the same tolerances.
"""

from absl.testing import absltest
from absl.testing import parameterized
from torchtitan.experiments.tpu.components.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.testing import base_distributed_device_test

_CPU_CONFIG_FN = (
    "torchtitan.experiments.tpu.qwen3_5.tests.test_ep_parity_distributed:"
    "cpu_reference_config"
)
_CPU_UNFUSED_EXPERTS_CONFIG_FN = (
    "torchtitan.experiments.tpu.qwen3_5.tests.test_ep_parity_distributed:"
    "cpu_reference_config_torch_grouped_mm"
)


# Routing weights before W2 pass per-row scales, which only the Pallas grouped
# matmuls take, so those cases select them explicitly.
_PALLAS_GMM_CONFIG = "--config=qwen35_moe_debugmodel_pallas_gmm"


def cpu_reference_config(cfg):
  """Single-device CPU reference: EP = 1 (the device run's EP is kept in args).

  Also runs the experts on native ``torch._grouped_mm``: the Pallas grouped
  matmuls are TPU-only.
  """
  cfg.parallelism.expert_parallel_degree = 1
  for _, experts, _, _ in cfg.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.use_pallas_grouped_mm = False
  return cfg


def cpu_reference_config_torch_grouped_mm(cfg):
  """CPU reference at EP = 1 with `torch._grouped_mm` instead of Pallas."""
  cfg = cpu_reference_config(cfg)
  for _, experts, _, _ in cfg.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.use_pallas_paired_grouped_mm = False
  return cfg


_PARITY_ARGS = [
    # TrainerMinimal (used by the parity harness) builds the optimizer
    # without the model's per-mesh param-group split, and foreach AdamW
    # can't mix the dense (fsdp) and expert (efsdp, ep) meshes in one group;
    # the per-parameter loop can.
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
]
# CP on TPU: the Tokamax Splash kernel on 128-token blocks. The debug model's
# RoPE table holds 256 positions, and each of the two CP ranks gets 128 of
# them, the shortest the Splash kernel takes (as in
# test_train_minimal_distributed's qwen3_5 CP cases).
_CP_SPLASH_ARGS = [
    "--training.max_context_length=256",
    "--training.num_tokens_per_microbatch_per_dp_rank=512",
    "--splash_attention_kernel.use_tokamax_splash_attention_kernel",
    "--splash_attention_kernel.sa_block_q=128",
    "--splash_attention_kernel.sa_block_kv=128",
    "--splash_attention_kernel.sa_block_kv_compute=128",
    "--splash_attention_kernel.sa_block_q_dkv=128",
    "--splash_attention_kernel.sa_block_kv_dkv=128",
    "--splash_attention_kernel.sa_block_kv_dkv_compute=128",
]
# Same tolerances as the dense qwen3_5_simplefsdp parity case.
_PARITY_TOLERANCES = dict(
    loss_atol=5e-3,
    loss_rtol=5e-3,
    grad_atol=1e-2,
    grad_rtol=5e-3,
    param_atol=1e-2,
    param_rtol=5e-3,
)


class EpParityDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):

  @parameterized.named_parameters(
      ("_ep1", 1, []),
      # Under `auto` on debugmodel_moe (top_k=2): EP=2 uses Ring-of-Experts;
      # EP=4 and EP=8 use ragged_all_to_all.
      ("_ep2_auto_roe", 2, []),
      ("_ep4_auto_ragged", 4, []),
      ("_ep8_auto_ragged", 8, []),
      # Explicit Ring-of-Experts at EP=4 and EP=8 (ep_size > top_k).
      ("_ep4_roe", 4, ["--tpu_config.ep_transport=ring_of_experts"]),
      ("_ep8_roe", 8, ["--tpu_config.ep_transport=ring_of_experts"]),
      # Explicit ragged_all_to_all at EP=2, where `auto` picks Ring-of-Experts.
      # This two-device subgroup hung on TPU7x with libtpu older than 0.0.51,
      # whose SparseCore ragged-all-to-all didn't support subslice programs.
      ("_ep2_ragged", 2, ["--tpu_config.ep_transport=ragged_all_to_all"]),
      # Routing weights applied before W2 (`scale_before_down_proj`) under both
      # Ring-of-Experts (EP=2 via auto) and ragged_all_to_all (EP=4 via auto).
      (
          "_ep2_scale_before_w2",
          2,
          [_PALLAS_GMM_CONFIG, "--tpu_config.ep_scale_before_down_proj=True"],
      ),
      (
          "_ep4_scale_before_w2",
          4,
          [_PALLAS_GMM_CONFIG, "--tpu_config.ep_scale_before_down_proj=True"],
      ),
      # Bounded expert buffer below the EP=4 worst case (4 * 512 = 2048 rows)
      # on ragged_all_to_all (via auto; 3.0 -> 1536 rows) and Ring-of-Experts
      # (2.0 -> 1024 rows; it rounds capped buffers up to a multiple of 1024,
      # so 3.0 would round up to the worst case).
      ("_ep4_cap3_ragged", 4, ["--tpu_config.ep_capacity_factor=3.0"]),
      (
          "_ep4_cap2_roe",
          4,
          [
              _PALLAS_GMM_CONFIG,
              "--tpu_config.ep_transport=ring_of_experts",
              "--tpu_config.ep_scale_before_down_proj=True",
              "--tpu_config.ep_capacity_factor=2.0",
          ],
      ),
      ("_ep2_chunk4", 2, ["--tpu_config.ep_token_chunks=4"]),
      (
          "_ep4_chunk4_w2",
          4,
          [
              _PALLAS_GMM_CONFIG,
              "--tpu_config.ep_token_chunks=4",
              "--tpu_config.ep_scale_before_down_proj=True",
          ],
      ),
  )
  def test_qwen3_5_moe_ep_matches_single_device(
      self, ep_degree: int, extra_args: list[str]
  ):
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("Runs the real TPU EP collectives and Pallas kernels.")
    if self.num_devices % ep_degree:
      self.skipTest(f"EP={ep_degree} needs a multiple of {ep_degree} chips.")
    config = next(
        (arg for arg in extra_args if arg.startswith("--config=")),
        "--config=qwen35_moe_debugmodel",
    )
    extra_args = [arg for arg in extra_args if not arg.startswith("--config=")]
    self._run_trainer_distributed_parity_test(
        config_args=[
            "--module=torchtitan.experiments.tpu.qwen3_5",
            config,
            *_PARITY_ARGS,
            f"--parallelism.expert_parallel_degree={ep_degree}",
            *extra_args,
        ],
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        cpu_config_fn=_CPU_CONFIG_FN,
        **_PARITY_TOLERANCES,
    )

  @parameterized.named_parameters(
      # Paired W1+W3 kernel on EP's static-size receive buffers (whose rows
      # past the routed count are stale), with ragged_all_to_all on the
      # topology-aware mesh.
      ("_ep4_ragged_topo_mesh", 4, ["--parallelism.enable_tpu_topo_mesh"]),
      # Chunked Ring-of-Experts with routing weights before W2 on the
      # topology-aware mesh: the 397B recipe's combination.
      (
          "_ep2_chunk4_w2_topo_mesh",
          2,
          [
              "--tpu_config.ep_token_chunks=4",
              "--tpu_config.ep_scale_before_down_proj=True",
              "--parallelism.enable_tpu_topo_mesh",
          ],
      ),
  )
  def test_qwen3_5_moe_ep_paired_experts_match_single_device(
      self, ep_degree: int, extra_args: list[str]
  ):
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("Runs the real TPU EP collectives and Pallas kernels.")
    if self.num_devices % ep_degree:
      self.skipTest(f"EP={ep_degree} needs a multiple of {ep_degree} chips.")
    self._run_trainer_distributed_parity_test(
        config_args=[
            "--module=torchtitan.experiments.tpu.qwen3_5",
            "--config=qwen35_moe_debugmodel_pallas_experts",
            *_PARITY_ARGS,
            f"--parallelism.expert_parallel_degree={ep_degree}",
            *extra_args,
        ],
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        cpu_config_fn=_CPU_UNFUSED_EXPERTS_CONFIG_FN,
        **_PARITY_TOLERANCES,
    )

  @parameterized.named_parameters(
      ("_ep2_ulysses", "qwen35_moe_debugmodel_ulysses_cp2", 2),
      ("_ep2_all_gather", "qwen35_moe_debugmodel_allgather_cp2", 2),
      # EP = 4 > top_k = 2: `auto` takes ragged_all_to_all, and EP spans
      # dp_shard x cp ranks rather than only the CP group.
      ("_ep4_ulysses", "qwen35_moe_debugmodel_ulysses_cp2", 4),
  )
  def test_qwen3_5_moe_ep_cp2_matches_single_device(
      self, config: str, ep_degree: int
  ):
    """EP with context parallelism (CP = 2) against the CPU reference.

    Head-parallel GatedDeltaNet CP and CP full attention shard each sequence
    over the CP group, and EP dispatches every rank's token shard over the EP
    group, which ``ParallelDims`` carves out of ``dp_shard * cp``. On TPU the
    run is compiled per block with the Splash kernel (eager TPU CP with
    ``dp_shard > 1`` is unreliable, see ``parallelize_qwen3_5``); on CPU it
    runs eagerly over gloo.
    """
    on_tpu = (
        self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU
    )
    if self.num_devices % 4:
      # EP comes out of dp_shard * cp, i.e. all devices here.
      self.skipTest("EP x CP=2 needs a multiple of 4 devices.")
    self._run_trainer_distributed_parity_test(
        config_args=[
            "--module=torchtitan.experiments.tpu.qwen3_5",
            f"--config={config}",
            *_PARITY_ARGS,
            f"--parallelism.expert_parallel_degree={ep_degree}",
            *(_CP_SPLASH_ARGS if on_tpu else []),
        ],
        data_parallel_shard_degree=None,
        tensor_parallel_degree=1,
        context_parallel_degree=2,
        enable_compile=on_tpu,
        cpu_config_fn=_CPU_CONFIG_FN,
        **_PARITY_TOLERANCES,
    )

if __name__ == "__main__":
  absltest.main()
