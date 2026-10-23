"""Tests for the minimal distributed training setup under TpuGraphTrainer."""

import json
import os
import shutil
import tempfile
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from torchtitan.components import metrics
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_distributed_device_test
import torchtitan.experiments.tpu.train


_MODELS = [
    dict(
        testcase_name="qwen3",
        module="torchtitan.experiments.tpu.qwen3",
        config="qwen3_debugmodel",
    ),
    dict(
        testcase_name="llama3",
        module="torchtitan.experiments.tpu.llama3",
        config="llama3_debugmodel",
    ),
]


def _train_args(
    module: str,
    config: str,
    optimizer_implementation: str = "foreach",
    dataset_path: str = "tests/assets/c4_test",
    dataset: str = "c4_test",
    *,
    use_splash_attention_kernel: bool = True,
    use_graph_trainer: bool = True,
) -> list[str]:
  """The flags every case here shares, so the arms differ only in what they test.

  ``use_splash_attention_kernel=False`` keeps the model on its plain SDPA
  inner attention (models whose Splash parity is not yet established).
  ``use_graph_trainer=False`` leaves the recipe on the regular TPU trainer with
  per-block ``torch.compile``, which is the baseline the graph trainer is
  compared against.
  """
  # Upstream replaced seq_len/local_batch_size with a token budget:
  # num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len = 4 * 128.
  args = [
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
      "--training.num_tokens_per_microbatch_per_dp_rank=512",
  ]
  if use_splash_attention_kernel:
    args.append("--splash_attention_kernel.use_splash_attention_kernel")
  if use_graph_trainer:
    args += [
        "--tpu_config.use_graph_trainer",
        "--tpu_config.compile_mode=aot_fx_trace",
    ]
  return args


class _RecordLosses:
  """Runs the trainer, saving rank 0's reported loss for every step.

  Wraps ``MetricsProcessor.log`` because every step's reduced loss passes
  through it, so the record is what the run reported rather than something
  recomputed beside it. Spawned workers get this by pickle, so it has to be a
  module-level class with the name attributes ``torchrun`` looks for.
  """

  def __init__(self, output_path: str):
    self.output_path = output_path
    self.__name__ = "_RecordLosses"
    self.__qualname__ = "_RecordLosses"

  def __call__(self, config):
    history = []
    original = metrics.MetricsProcessor.log

    def log(processor, step, global_avg_loss, global_max_loss, *args, **kwargs):
      history.append([step, global_avg_loss])
      return original(
          processor, step, global_avg_loss, global_max_loss, *args, **kwargs
      )

    with mock.patch.object(metrics.MetricsProcessor, "log", log):
      torchtitan.experiments.tpu.train.start_trainer(config)

    if int(os.environ.get("RANK", "0")) == 0:
      with open(self.output_path, "w") as f:
        json.dump(history, f)


class TpuGraphTrainerDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests the distributed training setup under TpuGraphTrainer."""

  # The `hsdp` case is not about a model. It leaves
  # `data_parallel_shard_degree` unset so torchtitan fills it from what
  # replicate leaves -- the mesh is 2-D for any even device count, which is the
  # only shape the in-body gather has to tell apart from a 1-D one.
  @parameterized.named_parameters([
      dict(
          testcase_name="qwen3_simple_fsdp",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--tpu_config.use_simple_fsdp",
          ],
      ),
      dict(
          testcase_name="llama3_simple_fsdp",
          module="torchtitan.experiments.tpu.llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--tpu_config.use_simple_fsdp",
          ],
      ),
      dict(
          testcase_name="llama3_simple_fsdp_scan_layers_hsdp",
          module="torchtitan.experiments.tpu.llama3",
          config="llama3_debugmodel",
          # Unset, so FSDP gets the remaining num_devices / 2.
          data_parallel_shard_degree=None,
          tensor_parallel_degree=1,
          data_parallel_replicate_degree=2,
          args=[
              "--tpu_config.use_simple_fsdp",
              "--tpu_config.use_scan_layers",
          ],
          # TODO(b/574966859): Broken by cl/1001117754 `RuntimeError:
          # traverse_and_compile(): expected Pallas kernels and scans in the
          # same program to have the same device assignment`
          skip_reason=(
              "b/574966859: broken by cl/1001117754 (scan + Pallas device"
              " assignment mismatch under HSDP)"
          ),
      ),
      # Muse Glimmer (text-only). Context 256 exceeds the debugmodel's
      # 128-token sliding window, so the SWA layers actually mask; the token
      # budget keeps 4 sequences per rank. No scan case: the SWA/RoPE and
      # global/NoPE layers differ, so they cannot be stacked. No TP case:
      # b/567185409.
      dict(
          testcase_name="muse_glimmer_simple_fsdp",
          module="torchtitan.experiments.tpu.muse_glimmer",
          config="muse_glimmer_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--tpu_config.use_simple_fsdp",
              "--training.max_context_length=256",
              "--training.num_tokens_per_microbatch_per_dp_rank=1024",
          ],
      ),
      # Gemma 4 (examples/gemma4). Context 256 exceeds the debugmodel's
      # 128-token sliding window, so the sliding layers actually mask; the
      # token budget keeps 4 sequences per rank. Plain SDPA: Splash/tokamax
      # parity on Gemma 4 is its own PR. No scan case: the sliding (head_dim
      # 32, QKVLinear) and global (head_dim 64, QKVLinearKeqV) layers cannot
      # be stacked. No TP case: the wrapper is SimpleFSDP/HSDP only.
      dict(
          testcase_name="gemma4_simple_fsdp",
          module="torchtitan.experiments.tpu.examples.gemma4",
          config="gemma4_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          use_splash_attention_kernel=False,
          args=[
              "--tpu_config.use_simple_fsdp",
              "--training.max_context_length=256",
              "--training.num_tokens_per_microbatch_per_dp_rank=1024",
          ],
      ),
      dict(
          testcase_name="gemma4_moe_simple_fsdp",
          module="torchtitan.experiments.tpu.examples.gemma4",
          config="gemma4_moe_debug",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          use_splash_attention_kernel=False,
          args=[
              "--tpu_config.use_simple_fsdp",
              "--training.max_context_length=256",
              "--training.num_tokens_per_microbatch_per_dp_rank=1024",
          ],
      ),
  ])
  def test_graph_trainer_distributed(
      self,
      module,
      config,
      data_parallel_shard_degree,
      tensor_parallel_degree,
      data_parallel_replicate_degree=None,
      skip_devices=None,
      optimizer_implementation="foreach",
      enable_compile=True,
      dataset_path="tests/assets/c4_test",
      dataset="c4_test",
      args: list[str] | None = None,
      skip_reason: str | None = None,
      use_splash_attention_kernel: bool = True,
  ):
    if skip_reason:
      self.skipTest(skip_reason)

    combined_args = _train_args(
        module,
        config,
        optimizer_implementation,
        dataset_path,
        dataset,
        use_splash_attention_kernel=use_splash_attention_kernel,
    ) + (args or [])

    self._test_train_distributed(
        config_args=combined_args,
        data_parallel_shard_degree=data_parallel_shard_degree,
        tensor_parallel_degree=tensor_parallel_degree,
        data_parallel_replicate_degree=data_parallel_replicate_degree,
        skip_devices=skip_devices,
        start_trainer=torchtitan.experiments.tpu.train.start_trainer,  # pyrefly: ignore[bad-argument-type]
        enable_compile=enable_compile,
    )

  # Both models, because the flag reaches them through a config rewrite with no
  # model-specific code, and running it end to end on two models is what holds
  # that true. qwen3 is also the MoE arm.
  @parameterized.named_parameters(_MODELS)
  def test_scan_matches_the_layer_loop(self, module, config):
    """The scan must train to the same losses as the layer loop it replaces.

    Every step, not just the first: a scan that drops the gradient on its init
    carry leaves step 1 exact and only diverges once the optimizer has stepped
    on the wrong grads.
    """
    tempdir = self.create_tempdir().full_path
    losses = {}
    for arm, extra in (("loop", []), ("scan", ["--tpu_config.use_scan_layers"])):
      path = os.path.join(tempdir, f"{arm}.json")
      self._test_train_distributed(
          config_args=_train_args(module, config)
          + ["--tpu_config.use_simple_fsdp"]
          + extra,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          start_trainer=_RecordLosses(path),  # pyrefly: ignore[bad-argument-type]
          enable_compile=True,
      )
      with open(path) as f:
        losses[arm] = json.load(f)

    self.assertNotEmpty(losses["loop"])
    # Bit-identical on llama3. qwen3's MoE reorders its accumulation between the
    # arms, so the last mantissa bits drift once the optimizer has stepped; a
    # severed carry moves the loss by percent, far outside this.
    for (step, scan), (_, loop) in zip(losses["scan"], losses["loop"]):
      self.assertAlmostEqual(scan, loop, delta=abs(loop) * 1e-5, msg=f"step {step}")

  def test_gemma4_graph_trainer_matches_the_layer_compile(self):
    """The graph trainer must train Gemma 4 to the losses of the eager lane.

    The PR-1 lane is the regular TPU trainer with per-block
    ``torch.compile(dynamic=False)`` and SimpleFSDP; this PR adds the
    ``aot_fx_trace`` joint graph on the same recipe. Same model config, data,
    seed, fp32 params and reductions, so the two arms differ only in how the
    step is captured. Every step is compared, not just the first: a wrapper
    that mis-shards an uneven parameter (the vocabulary over 8 ranks, say) or
    drops a gradient leaves step 1 exact and only diverges once the optimizer
    has stepped.
    """
    module = "torchtitan.experiments.tpu.examples.gemma4"
    config = "gemma4_debugmodel"
    # Context 256 > the 128-token sliding window, 4 sequences per rank.
    shape = [
        "--training.max_context_length=256",
        "--training.num_tokens_per_microbatch_per_dp_rank=1024",
    ]
    layer_compile_args = _train_args(
        module,
        config,
        use_splash_attention_kernel=False,
        use_graph_trainer=False,
    )
    graph_trainer_args = _train_args(
        module, config, use_splash_attention_kernel=False
    )
    arms = {
        "layer_compile": [
            *layer_compile_args,
            "--parallelism.use_simple_fsdp",
            *shape,
        ],
        "graph_trainer": [
            *graph_trainer_args,
            "--tpu_config.use_simple_fsdp",
            *shape,
        ],
        "control": [
            *graph_trainer_args,
            "--tpu_config.use_simple_fsdp",
            *shape,
            "--lr_scheduler.warmup_steps=3",
        ],
    }
    tempdir = self.create_tempdir().full_path
    losses = {}
    for arm, config_args in arms.items():
      path = os.path.join(tempdir, f"{arm}.json")
      self._test_train_distributed(
          config_args=config_args,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          start_trainer=_RecordLosses(path),  # pyrefly: ignore[bad-argument-type]
          enable_compile=True,
      )
      with open(path) as f:
        losses[arm] = json.load(f)

    self.assertNotEmpty(losses["layer_compile"])
    self.assertLen(losses["graph_trainer"], len(losses["layer_compile"]))
    # Both arms are fp32 end to end; what differs is XLA's fusion and
    # reduction order between one joint program and per-block programs.
    # Measured on v5e-8 (tpu_4x2x1): |diff| <= 2.9e-6 absolute, 3.7e-7
    # relative, over the 3 steps (7.674659 vs 7.674662 at step 1). A dropped
    # gradient or a mis-sharded parameter moves the loss by percent, far
    # outside this.
    for (step, graph), (_, layer) in zip(
        losses["graph_trainer"], losses["layer_compile"]
    ):
      logging.info(
          "gemma4 step %d: graph_trainer loss %.6f, layer_compile loss %.6f,"
          " |diff| %.2e (%.2e relative)",
          step,
          graph,
          layer,
          abs(graph - layer),
          abs(graph - layer) / abs(layer),
      )
      self.assertAlmostEqual(
          graph, layer, delta=abs(layer) * 1e-5, msg=f"step {step}"
      )
    control_losses = dict(losses["control"])
    layer_losses = dict(losses["layer_compile"])
    for step in (2, 3):
      self.assertGreater(
          abs(control_losses[step] - layer_losses[step]),
          abs(layer_losses[step]) * 1e-5,
          msg=f"control, step {step}",
      )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
