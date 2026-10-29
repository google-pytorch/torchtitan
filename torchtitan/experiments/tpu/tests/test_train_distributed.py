"""Tests for the minimal distributed training setup for Llama3 models."""

import os
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_distributed_device_test
import torchtitan.experiments.tpu.train


class TrainDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests the distributed training setup."""

  @parameterized.named_parameters([
      # deepseek_v3
      dict(
          testcase_name="deepseek_v3_tp",
          module="torchtitan.experiments.tpu.deepseek_v3",
          config="deepseek_v3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          skip_devices=[
              # TODO: b/482057603 - Issue with mixing DTensor and Tensor.
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="deepseek_v3_tp_compile",
          module="torchtitan.experiments.tpu.deepseek_v3",
          config="deepseek_v3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          enable_compile=True,
          skip_devices=[
              # TODO: b/482057603 - Issue with mixing DTensor and Tensor.
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="deepseek_v3_fsdp",
          module="torchtitan.experiments.tpu.deepseek_v3",
          config="deepseek_v3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=[
              # b/487028245 - Timing out w/Math Backend
              device_type.AcceleratorDeviceType.TPU,
          ],
      ),
      dict(
          testcase_name="deepseek_v3_fsdp_compile",
          module="torchtitan.experiments.tpu.deepseek_v3",
          config="deepseek_v3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          enable_compile=True,
          skip_devices=[
              # b/487028245 - Timing out w/Math Backend
              device_type.AcceleratorDeviceType.TPU,
              # b/501466433 - Bug with torch.compile
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      # flux
      dict(
          testcase_name="flux_tp",
          module="torchtitan.experiments.tpu.flux",
          config="flux_testmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          dataset_path="tests/assets/cc12m_test",
          dataset="cc12m-test",
      ),
      dict(
          testcase_name="flux_tp_compile",
          module="torchtitan.experiments.tpu.flux",
          config="flux_testmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          dataset_path="tests/assets/cc12m_test",
          dataset="cc12m-test",
          enable_compile=True,
          skip_devices=[
              # b/501466433 - Bug with torch.compile (not reproducible on TPU
              # with flux_testmodel).
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="flux_fsdp",
          module="torchtitan.experiments.tpu.flux",
          config="flux_testmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          dataset_path="tests/assets/cc12m_test",
          dataset="cc12m-test",
      ),
      dict(
          testcase_name="flux_fsdp_compile",
          module="torchtitan.experiments.tpu.flux",
          config="flux_testmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          dataset_path="tests/assets/cc12m_test",
          dataset="cc12m-test",
          enable_compile=True,
          skip_devices=[
              # b/501466433 - Bug with torch.compile (not reproducible on TPU
              # with flux_testmodel).
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      # llama3
      dict(
          testcase_name="llama3_tp",
          module="torchtitan.experiments.tpu.llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
      dict(
          testcase_name="llama3_tp_compile",
          module="torchtitan.experiments.tpu.llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          enable_compile=True,
          skip_devices=[
              # b/501466433 - Bug with torch.compile
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="llama3_fsdp",
          module="torchtitan.experiments.tpu.llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
      ),
      dict(
          testcase_name="llama3_simplefsdp",
          module="torchtitan.experiments.tpu.llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--parallelism.use_simple_fsdp"],
      ),
      dict(
          testcase_name="llama3_fsdp_compile",
          module="torchtitan.experiments.tpu.llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          enable_compile=True,
          skip_devices=[
              # b/501466433 - Bug with torch.compile
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="llama3_fsdp_checkpoint",
          module="torchtitan.experiments.tpu.llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          checkpoint_enabled=True,
          skip_devices=[
              # TODO: remove once libtpu version is updated
              device_type.AcceleratorDeviceType.TPU,
          ],
      ),
      dict(
          testcase_name="llama3_fsdp_use_loss_kernel",
          module="torchtitan.experiments.tpu.llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          # seq_len -> max_context_length. The token budget has to be
          # restated too: default_args sizes it for a 128-token context, and a
          # token budget is absolute, so inheriting it here would ask for a
          # sub-1 microbatch. 4 * 1024 preserves the pre-import 4 sequences.
          args=[
              "--training.dtype=bfloat16",
              "--loss_kernel.use_loss_kernel",
              "--training.max_context_length=1024",
              "--training.num_tokens_per_microbatch_per_dp_rank=4096",
              "--loss_kernel.loss_b_block_size=128",
              "--loss_kernel.loss_h_block_size=128",
              "--loss_kernel.loss_v_block_size=128",
          ],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="llama3_fsdp_use_splash_attention_kernel",
          module="torchtitan.experiments.tpu.llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--splash_attention_kernel.use_splash_attention_kernel"],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      # qwen3
      dict(
          testcase_name="qwen3_tp",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
      dict(
          testcase_name="qwen3_tp_compile",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          enable_compile=True,
          skip_devices=[
              # b/501466433 - Bug with torch.compile
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="qwen3_fsdp",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
      ),
      dict(
          testcase_name="qwen3_simplefsdp",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--parallelism.use_simple_fsdp"],
      ),
      dict(
          testcase_name="qwen3_cp_all_gather",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel_allgather_cp2",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=1,
          context_parallel_degree=-1,
      ),
      dict(
          testcase_name="qwen3_cp_ulysses",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel_ulysses_cp2",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=1,
          context_parallel_degree=-1,
      ),
      dict(
          testcase_name="qwen3_simplefsdp_cp2_all_gather",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel_allgather_cp2",
          data_parallel_shard_degree=None,
          tensor_parallel_degree=1,
          context_parallel_degree=2,
          args=["--parallelism.use_simple_fsdp"],
      ),
      dict(
          testcase_name="qwen3_fsdp_compile",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          enable_compile=True,
          skip_devices=[
              # b/501466433 - Bug with torch.compile
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="qwen3_fsdp_use_loss_kernel",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          # seq_len -> max_context_length. The token budget has to be
          # restated too: default_args sizes it for a 128-token context, and a
          # token budget is absolute, so inheriting it here would ask for a
          # sub-1 microbatch. 4 * 1024 preserves the pre-import 4 sequences.
          args=[
              "--training.dtype=bfloat16",
              "--loss_kernel.use_loss_kernel",
              "--training.max_context_length=1024",
              "--training.num_tokens_per_microbatch_per_dp_rank=4096",
              "--loss_kernel.loss_b_block_size=128",
              "--loss_kernel.loss_h_block_size=128",
              "--loss_kernel.loss_v_block_size=128",
          ],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="qwen3_fsdp_use_splash_attention_kernel",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--splash_attention_kernel.use_splash_attention_kernel"],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="qwen3_fsdp_moe_use_gmm_kernel",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_moe_testmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--qwen3.use_gmm_kernel",
          ],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      # TODO(alexanderb14): fill_indices is not in OSS code, update this test once we decide to add it back or remove it.
      #   dict(
      #       testcase_name="qwen3_fsdp_moe_use_gmm_and_fill_indices_kernels",
      #       module="torchtitan.experiments.tpu.qwen3",
      #       config="qwen3_debugmodel",
      #       data_parallel_shard_degree=-1,
      #       tensor_parallel_degree=1,
      #       args=[
      #           "--model.flavor=testmodel_moe",  # NOTE: this model has use_grouped_mm=True
      #           "--qwen3.use_gmm_kernel",
      #           "--qwen3.use_fill_indices_kernel",
      #       ],
      #       skip_devices=[
      #           device_type.AcceleratorDeviceType.CPU,
      #           device_type.AcceleratorDeviceType.CUDA,
      #       ],
      #   ),
      # gemma4
      dict(
          testcase_name="gemma4_tp",
          module="torchtitan.experiments.tpu.examples.gemma4",
          config="gemma4_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
      dict(
          testcase_name="gemma4_fsdp",
          module="torchtitan.experiments.tpu.examples.gemma4",
          config="gemma4_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
      ),
      dict(
          testcase_name="gemma4_simplefsdp",
          module="torchtitan.experiments.tpu.examples.gemma4",
          config="gemma4_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--parallelism.use_simple_fsdp"],
      ),
      dict(
          testcase_name="gemma4_moe_fsdp",
          module="torchtitan.experiments.tpu.examples.gemma4",
          config="gemma4_moe_testmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
      ),
      dict(
          testcase_name="gemma4_fsdp_checkpoint",
          module="torchtitan.experiments.tpu.examples.gemma4",
          config="gemma4_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          checkpoint_enabled=True,
      ),
      # flint: no TP (the per-head sinks are replicated, as in HF Flint).
      dict(
          testcase_name="flint_fsdp",
          module="torchtitan.experiments.tpu.examples.flint",
          config="flint_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          # 512 > the debug flavor's 129-token window, so the window masks.
          args=[
              "--parallelism.no-use_simple_fsdp",
              "--training.max_context_length=512",
              "--training.num_tokens_per_microbatch_per_dp_rank=1024",
          ],
      ),
      dict(
          testcase_name="flint_simplefsdp",
          module="torchtitan.experiments.tpu.examples.flint",
          config="flint_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--training.max_context_length=512",
              "--training.num_tokens_per_microbatch_per_dp_rank=1024",
          ],
      ),
      # muse_glimmer (text-only). TP is capped at 2: the debugmodel has 2 KV
      # heads (see muse_glimmer_fsdp_tp2).
      dict(
          testcase_name="muse_glimmer_fsdp",
          module="torchtitan.experiments.tpu.muse_glimmer",
          config="muse_glimmer_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          # The debugmodel config defaults to SimpleFSDP; use FSDP2 here.
          # Context 256 > the debugmodel's 128-token sliding window, so SWA
          # layers actually mask. The token budget keeps 4 sequences per rank.
          args=[
              "--parallelism.no-use_simple_fsdp",
              "--training.max_context_length=256",
              "--training.num_tokens_per_microbatch_per_dp_rank=1024",
          ],
      ),
      dict(
          testcase_name="muse_glimmer_simplefsdp",
          module="torchtitan.experiments.tpu.muse_glimmer",
          config="muse_glimmer_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--parallelism.use_simple_fsdp",
              "--training.max_context_length=256",
              "--training.num_tokens_per_microbatch_per_dp_rank=1024",
          ],
      ),
      # Splash / tokamax on the sliding-window (LocalMask) and global layers.
      # Context 512 (one 512 splash block) is 4x the 128-token window, and the
      # packed C4 batches exercise document masking.
      dict(
          testcase_name="muse_glimmer_simplefsdp_use_splash_attention_kernel",
          module="torchtitan.experiments.tpu.muse_glimmer",
          config="muse_glimmer_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--parallelism.use_simple_fsdp",
              "--training.max_context_length=512",
              "--training.num_tokens_per_microbatch_per_dp_rank=2048",
              "--splash_attention_kernel.use_splash_attention_kernel",
          ],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name=(
              "muse_glimmer_simplefsdp_use_tokamax_splash_attention_kernel"
          ),
          module="torchtitan.experiments.tpu.muse_glimmer",
          config="muse_glimmer_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--parallelism.use_simple_fsdp",
              "--training.max_context_length=512",
              "--training.num_tokens_per_microbatch_per_dp_rank=2048",
              "--splash_attention_kernel.use_tokamax_splash_attention_kernel",
          ],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      # Fused Pallas CE with the SoftCappedLinear head's soft-cap applied
      # in-kernel (1024 tokens per rank = one b_block).
      dict(
          testcase_name="muse_glimmer_simplefsdp_fused_ce",
          module="torchtitan.experiments.tpu.muse_glimmer",
          config="muse_glimmer_debugmodel_fused_ce",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--parallelism.use_simple_fsdp",
              "--training.max_context_length=256",
              "--training.num_tokens_per_microbatch_per_dp_rank=1024",
          ],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          # TP=2 divides the debugmodel's 2 KV heads. dp_shard is inferred:
          # 1 with 2 ranks (CPU/GPU), 4 on v5e 4x2x1.
          testcase_name="muse_glimmer_fsdp_tp2",
          module="torchtitan.experiments.tpu.muse_glimmer",
          config="muse_glimmer_debugmodel",
          data_parallel_shard_degree=None,
          tensor_parallel_degree=2,
          args=[
              "--parallelism.no-use_simple_fsdp",
              "--training.max_context_length=256",
              "--training.num_tokens_per_microbatch_per_dp_rank=1024",
          ],
      ),
      # qwen3_5 (hybrid GatedDeltaNet + Attention, Dense and MoE).
      dict(
          testcase_name="qwen3_5_fsdp",
          module="torchtitan.experiments.tpu.qwen3_5",
          config="qwen35_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--parallelism.no-use_simple_fsdp"],
      ),
      dict(
          testcase_name="qwen3_5_simplefsdp",
          module="torchtitan.experiments.tpu.qwen3_5",
          config="qwen35_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--parallelism.use_simple_fsdp"],
      ),
      dict(
          testcase_name="qwen3_5_moe_simplefsdp",
          module="torchtitan.experiments.tpu.qwen3_5",
          config="qwen35_moe_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--parallelism.use_simple_fsdp"],
      ),
  ])
  def test_train_distributed(
      self,
      module,
      config,
      data_parallel_shard_degree,
      tensor_parallel_degree,
      data_parallel_replicate_degree=None,
      context_parallel_degree=None,
      skip_devices=None,
      optimizer_implementation="foreach",
      enable_compile=False,
      checkpoint_enabled=False,
      dataset_path="tests/assets/c4_test",
      dataset="c4_test",
      args: list[str] | None = None,
  ):
    # Upstream replaced seq_len/local_batch_size with a token budget:
    # num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len, with
    # seq_len = 128 here, so 1 * 128 for flux and 4 * 128 otherwise.
    default_args = [
        f"--module={module}",
        f"--config={config}",
        f"--optimizer.implementation={optimizer_implementation}",
        f"--dataloader.dataset_path={dataset_path}",
        f"--dataloader.dataset={dataset}",
        "--hf_assets_path=tests/assets/tokenizer",
        "--training.max_context_length=128",
        "--training.steps=3",
        "--training.mixed_precision_param=float32",
        "--training.mixed_precision_reduce=float32",
        (
            "--training.num_tokens_per_microbatch_per_dp_rank="
            f"{128 if module.endswith('flux') else 512}"
        ),
    ]
    if not args:
      args = []

    # Combined args: default args first, then test case specific args.
    # This allows test cases to override default arguments.
    combined_args = default_args + args

    if checkpoint_enabled:
      test_tmp_dir = tempfile.mkdtemp()
      self.addCleanup(shutil.rmtree, test_tmp_dir, ignore_errors=True)
      combined_args.extend([
          "--checkpoint.enable",
          "--checkpoint.interval=2",
          f"--checkpoint.folder={os.path.join(test_tmp_dir, 'checkpoint')}",
      ])

    self._test_train_distributed(
        config_args=combined_args,
        data_parallel_shard_degree=data_parallel_shard_degree,
        tensor_parallel_degree=tensor_parallel_degree,
        data_parallel_replicate_degree=data_parallel_replicate_degree,
        context_parallel_degree=context_parallel_degree,
        skip_devices=skip_devices,
        start_trainer=torchtitan.experiments.tpu.train.start_trainer,  # pyrefly: ignore[bad-argument-type]
        enable_compile=enable_compile,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
