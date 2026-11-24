"""Tests for the minimal distributed training setup under TpuGraphTrainer."""

import json
import math
import os
import shutil
import tempfile
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from torchtitan.components import metrics
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.testing import base_distributed_device_test
import torchtitan.experiments.tpu.train


_MODELS = [
    dict(
        testcase_name="qwen3",
        module="torchtitan.experiments.tpu.qwen3",
        config="graph_trainer_qwen3_debugmodel",
    ),
    dict(
        testcase_name="llama3",
        module="torchtitan.experiments.tpu.llama3",
        config="graph_trainer_llama3_debugmodel",
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
) -> list[str]:
  """The flags every case here shares, so the arms differ only in what they test.

  ``use_splash_attention_kernel=False`` keeps the model on its plain SDPA
  inner attention (models whose Splash parity is not yet established).
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


class _RecordEpDropStats:
  """Runs the trainer, saving rank 0's losses and logged EP drop counts.

  Wraps ``TpuGraphTrainer._maybe_log_ep_drop_stats``, which returns the
  batch-mesh sum it logged, so the record is what the run reported. A
  module-level class for the same pickling reason as ``_RecordLosses``.
  """

  def __init__(self, output_path: str):
    self.output_path = output_path
    self.__name__ = "_RecordEpDropStats"
    self.__qualname__ = "_RecordEpDropStats"

  def __call__(self, config):
    from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer  # pylint: disable=g-import-not-at-top

    losses = []
    drop_stats = []
    original_log = metrics.MetricsProcessor.log
    original_drop_stats = TpuGraphTrainer._maybe_log_ep_drop_stats

    def log(processor, step, global_avg_loss, global_max_loss, *args, **kwargs):
      losses.append([step, global_avg_loss])
      return original_log(
          processor, step, global_avg_loss, global_max_loss, *args, **kwargs
      )

    def log_drop_stats(trainer):
      stats = original_drop_stats(trainer)
      if stats is not None:
        drop_stats.append([trainer.step, stats.tolist()])
      return stats

    with mock.patch.object(metrics.MetricsProcessor, "log", log):
      with mock.patch.object(
          TpuGraphTrainer, "_maybe_log_ep_drop_stats", log_drop_stats
      ):
        torchtitan.experiments.tpu.train.start_trainer(config)

    if int(os.environ.get("RANK", "0")) == 0:
      with open(self.output_path, "w") as f:
        json.dump({"losses": losses, "ep_drop_stats": drop_stats}, f)


# EP=2 flags shared by the Qwen3.5 MoE GraphTrainer EP tests.
_QWEN35_MOE_EP2 = [
    "--parallelism.expert_parallel_degree=2",
    "--compile.no-compile_grad_norm",
    "--compile.graph_split.backward_partition_mode=by_layer",
]


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
          config="graph_trainer_qwen3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--parallelism.use_simple_fsdp",
          ],
      ),
      dict(
          testcase_name="llama3_simple_fsdp",
          module="torchtitan.experiments.tpu.llama3",
          config="graph_trainer_llama3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--parallelism.use_simple_fsdp",
          ],
      ),
      dict(
          testcase_name="llama3_simple_fsdp_scan_layers_hsdp",
          module="torchtitan.experiments.tpu.llama3",
          config="graph_trainer_llama3_debugmodel",
          # Unset, so FSDP gets the remaining num_devices / 2.
          data_parallel_shard_degree=None,
          tensor_parallel_degree=1,
          data_parallel_replicate_degree=2,
          args=[
              "--parallelism.use_simple_fsdp",
              "--compile.use_scan_layers",
          ],
      ),
      # The HSDP mesh over the topology-ordered rank permutation, as the
      # recipes run it. Every multi-axis mesh failed spmd_types' orthogonality
      # check under that permutation (b/587033846). TPU only: the flag is a
      # no-op without torch.tpu.topology_aware_mesh.
      dict(
          testcase_name="llama3_simple_fsdp_scan_layers_hsdp_topo_mesh",
          module="torchtitan.experiments.tpu.llama3",
          config="graph_trainer_llama3_debugmodel",
          data_parallel_shard_degree=None,
          tensor_parallel_degree=1,
          data_parallel_replicate_degree=2,
          args=[
              "--parallelism.use_simple_fsdp",
              "--parallelism.enable_tpu_topo_mesh",
              "--compile.use_scan_layers",
          ],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      # Muse Glimmer (text-only). Context 256 exceeds the debugmodel's
      # 128-token sliding window, so the SWA layers actually mask; the token
      # budget keeps 4 sequences per rank. No scan case: the SWA/RoPE and
      # global/NoPE layers differ, so they cannot be stacked. No TP case:
      # b/567185409.
      dict(
          testcase_name="muse_glimmer_simple_fsdp",
          module="torchtitan.experiments.tpu.muse_glimmer",
          config="graph_trainer_muse_glimmer_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--parallelism.use_simple_fsdp",
              "--training.max_context_length=256",
              "--training.num_tokens_per_microbatch_per_dp_rank=1024",
          ],
      ),
      # Gemma 4 (examples/gemma4). Context 256 exceeds the debugmodel's
      # 128-token sliding window, so the sliding layers actually mask; the
      # token budget keeps 4 sequences per rank. Plain SDPA: Splash/tokamax
      # parity on Gemma 4 is its own PR. No scan case: the sliding (head_dim
      # 32, FusedQKVLinear) and global (head_dim 64, FusedQKVLinearKeqV)
      # layers cannot be stacked. No TP case: the wrapper is SimpleFSDP/HSDP
      # only.
      dict(
          testcase_name="gemma4_simple_fsdp",
          module="torchtitan.experiments.tpu.examples.gemma4",
          config="graph_trainer_gemma4_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          use_splash_attention_kernel=False,
          args=[
              "--parallelism.use_simple_fsdp",
              "--parallelism.fsdp_reshard_after_forward=always",
              "--compile.enable_remat_barriers",
              "--compile.fsdp_barrier_policy=interleave",
              "--training.max_context_length=256",
              "--training.num_tokens_per_microbatch_per_dp_rank=1024",
          ],
      ),
      dict(
          testcase_name="gemma4_moe_simple_fsdp",
          module="torchtitan.experiments.tpu.examples.gemma4",
          config="graph_trainer_gemma4_moe_debug",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          use_splash_attention_kernel=False,
          args=[
              "--parallelism.use_simple_fsdp",
              "--parallelism.fsdp_reshard_after_forward=always",
              "--compile.enable_remat_barriers",
              "--compile.fsdp_barrier_policy=interleave",
              "--training.max_context_length=256",
              "--training.num_tokens_per_microbatch_per_dp_rank=1024",
          ],
      ),
      dict(
          testcase_name="gemma4_graph_split_offload",
          module="torchtitan.experiments.tpu.examples.gemma4",
          config="graph_trainer_gemma4_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          use_splash_attention_kernel=False,
          args=[
              "--parallelism.use_simple_fsdp",
              "--parallelism.fsdp_reshard_after_forward=always",
              "--compile.memory_policy=full_and_offload",
              "--compile.graph_split.enabled",
              "--compile.graph_split.forward_layers_per_executable=1",
              "--compile.graph_split.forward_isolate_prefix",
              "--compile.graph_split.backward_partition_mode=by_layer",
              "--compile.graph_split.backward_layers_per_executable=1",
              "--compile.graph_split.cpu_offload_prefetch_executables=1",
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
    for arm, extra in (("loop", []), ("scan", ["--compile.use_scan_layers"])):
      path = os.path.join(tempdir, f"{arm}.json")
      self._test_train_distributed(
          config_args=_train_args(module, config)
          + ["--parallelism.use_simple_fsdp"]
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
    # Context 256 > the 128-token sliding window, 4 sequences per rank.
    shape = [
        "--training.max_context_length=256",
        "--training.num_tokens_per_microbatch_per_dp_rank=1024",
    ]
    layer_compile_args = _train_args(
        module,
        "gemma4_debugmodel",
        use_splash_attention_kernel=False,
    )
    graph_trainer_args = _train_args(
        module,
        "graph_trainer_gemma4_debugmodel",
        use_splash_attention_kernel=False,
    )
    arms = {
        "layer_compile": [
            *layer_compile_args,
            "--parallelism.use_simple_fsdp",
            *shape,
        ],
        "graph_trainer": [
            *graph_trainer_args,
            "--parallelism.use_simple_fsdp",
            *shape,
        ],
        "control": [
            *graph_trainer_args,
            "--parallelism.use_simple_fsdp",
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
    # Measured on v5e-8 (tpu_4x2x1) with full AC on both arms: |diff| <= 4.3e-6
    # absolute, 5.9e-7 relative, over the 3 steps (7.674668 vs 7.674666 at step 1). A dropped
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

  def _train_qwen3_5_moe(self, start_trainer, extra_args: list[str]) -> None:
    """Trains graph_trainer_qwen35_moe_debugmodel with ``extra_args`` appended."""
    self._test_train_distributed(
        config_args=_train_args(
            "torchtitan.experiments.tpu.qwen3_5",
            "graph_trainer_qwen35_moe_debugmodel",
            use_splash_attention_kernel=False,
        )
        + extra_args,
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        start_trainer=start_trainer,  # pyrefly: ignore[bad-argument-type]
        enable_compile=True,
    )

  # The EP tests below run one EP=2 arm per test instead of looping over the
  # arms in one test. Every EP=2 arm compiles its own programs, which takes
  # minutes on the test TPUs, and absltest shards by test method, so a test
  # that ran every arm would not fit in a 15-minute test shard.
  @parameterized.named_parameters(
      dict(testcase_name="unchunked", ep2_args=[]),
      dict(testcase_name="chunk4", ep2_args=["--tpu_config.ep_token_chunks=4"]),
  )
  def test_qwen3_5_moe_graph_trainer_ep2_matches_ep1(self, ep2_args):
    """GraphTrainer Qwen3.5 MoE at EP=2 (unchunked or 4-chunk RoE) matches EP=1."""
    tempdir = self.create_tempdir().full_path
    step1_loss = {}
    for arm, extra_args in (("ep1", []), ("ep2", _QWEN35_MOE_EP2 + ep2_args)):
      path = os.path.join(tempdir, f"{arm}.json")
      self._train_qwen3_5_moe(
          _RecordLosses(path), ["--training.steps=1"] + extra_args
      )
      with open(path) as f:
        step1_loss[arm] = dict(json.load(f))[1]
    logging.info("qwen3_5 MoE GraphTrainer step-1 loss by arm: %s", step1_loss)
    self.assertTrue(math.isfinite(step1_loss["ep2"]))
    self.assertLess(
        abs(step1_loss["ep2"] - step1_loss["ep1"]),
        abs(step1_loss["ep1"]) * 1e-3,
    )

  def test_qwen3_5_moe_graph_trainer_ep2_fsdp_pipeline_matches_by_layer(self):
    """EP=2 with fsdp_pipeline backward matches by_layer over three steps.

    fsdp_pipeline only moves each backward executable's FSDP all-gathers one
    executable earlier and its gradient reduce-scatters one later, so the
    losses after two optimizer updates match by_layer's.
    """
    tempdir = self.create_tempdir().full_path
    losses = {}
    for mode in ("by_layer", "fsdp_pipeline"):
      path = os.path.join(tempdir, f"{mode}.json")
      self._train_qwen3_5_moe(
          _RecordLosses(path),
          ["--training.steps=3", "--debug.seed=0"]
          + _QWEN35_MOE_EP2
          + [f"--compile.graph_split.backward_partition_mode={mode}"],
      )
      with open(path) as f:
        losses[mode] = {int(k): v for k, v in json.load(f)}
    logging.info("qwen3_5 MoE GraphTrainer EP=2 losses by mode: %s", losses)
    for step in (1, 2, 3):
      self.assertTrue(math.isfinite(losses["fsdp_pipeline"][step]))
      self.assertLess(
          abs(losses["fsdp_pipeline"][step] - losses["by_layer"][step]),
          abs(losses["by_layer"][step]) * 1e-3,
          msg=f"step {step}",
      )

  @parameterized.named_parameters(
      # Ragged all-to-all caps at int(1024 * 0.5) = 512 rows, half the balanced
      # load, so at least half the routes drop.
      dict(
          testcase_name="ragged_half",
          extra_args=[
              "--tpu_config.ep_transport=ragged_all_to_all",
              "--tpu_config.ep_capacity_factor=0.5",
          ],
          drops="at_least_half",
      ),
      # `auto` picks Ring-of-Experts at EP=2, which rounds the 512-row cap up
      # to the Pallas block, 1024 rows, i.e. the balanced load. With 2048
      # routes over two ranks each holding 1024, fewer than half can drop (only
      # if every route went to one rank would half).
      dict(
          testcase_name="roe_half_rounded",
          extra_args=["--tpu_config.ep_capacity_factor=0.5"],
          drops="under_half",
      ),
      # 2.0 is the EP=2 worst case, so nothing drops.
      dict(
          testcase_name="roe_worst_case",
          extra_args=["--tpu_config.ep_capacity_factor=2.0"],
          drops="none",
      ),
      # 4 token chunks, each capped separately at int(256 * 0.5) = 128 rows, so
      # at least half drop; the stats carry the sum over chunks. (Ring-of-Experts
      # would round each chunk's cap up to 1024 rows, above the chunk's 512-row
      # worst case, and drop nothing.)
      dict(
          testcase_name="ragged_chunk4_half",
          extra_args=[
              "--tpu_config.ep_transport=ragged_all_to_all",
              "--tpu_config.ep_token_chunks=4",
              "--tpu_config.ep_capacity_factor=0.5",
          ],
          drops="at_least_half",
      ),
      # Dropless: nothing is collected.
      dict(testcase_name="dropless", extra_args=[], drops=None),
  )
  def test_qwen3_5_moe_graph_trainer_ep2_capacity_reports_drops(
      self, extra_args, drops
  ):
    """Capacity-bounded EP=2 reports every MoE layer's drops from the step.

    A traced step does not write module state back, so the counts are a step
    output. A capacity-bounded run must count every rank's routes on every
    layer at steps 1 and 2; a dropless run must not collect them at all.
    Per rank: 512 tokens * top_k 2 = 1024 routes, worst case 2 * 1024 rows.
    """
    tokens_per_rank = 512  # _train_args' per-rank token budget.
    path = os.path.join(self.create_tempdir().full_path, "record.json")
    self._train_qwen3_5_moe(
        _RecordEpDropStats(path),
        _QWEN35_MOE_EP2 + ["--training.steps=2"] + extra_args,
    )
    with open(path) as f:
      record = json.load(f)
    logging.info("qwen3_5 MoE EP=2 capacity record: %s", record)

    for step, loss in record["losses"]:
      self.assertTrue(math.isfinite(loss), msg=f"step {step}")
    if drops is None:
      self.assertEqual(record["ep_drop_stats"], [])
      return
    self.assertEqual([step for step, _ in record["ep_drop_stats"]], [1, 2])
    for step, rows in record["ep_drop_stats"]:
      msg = f"step {step}"
      self.assertNotEmpty(rows, msg=msg)
      for dropped, routes in rows:
        # Every rank's tokens times top_k, on every MoE layer.
        self.assertEqual(routes, self.num_devices * tokens_per_rank * 2, msg=msg)
        if drops == "none":
          self.assertEqual(dropped, 0, msg=msg)
        elif drops == "at_least_half":
          self.assertGreaterEqual(2 * dropped, routes, msg=msg)
          self.assertLess(dropped, routes, msg=msg)
        else:  # "under_half"
          self.assertGreaterEqual(dropped, 0, msg=msg)
          self.assertLess(2 * dropped, routes, msg=msg)

  def test_qwen3_5_moe_graph_trainer_tpu_moe_kernels_ep2_matches_ep1(self):
    """GraphTrainer with the split recipes' MoE kernels: EP=2 matches EP=1.

    Paired W1+W3 Pallas grouped matmuls on kernel-native expert weights and
    the fused route-score backward (`_use_tpu_moe_kernels`), under EP=2
    Ring-of-Experts with 4 token chunks and routing weights before W2 on the
    topology-aware mesh (the 397B recipe's combination). All 3 steps are
    compared, so dWeights count too.
    """
    tempdir = self.create_tempdir().full_path
    losses = {}
    common = [
        "--training.steps=3",
        # The fused route-score backward needs BF16 expert outputs.
        "--training.mixed_precision_param=bfloat16",
    ]
    ep2_args = _QWEN35_MOE_EP2 + [
        "--parallelism.enable_tpu_topo_mesh",
        "--tpu_config.ep_token_chunks=4",
        "--tpu_config.ep_scale_before_down_proj=True",
    ]
    for arm, extra_args in (("ep1", []), ("ep2", ep2_args)):
      path = os.path.join(tempdir, f"{arm}.json")
      self._test_train_distributed(
          config_args=_train_args(
              "torchtitan.experiments.tpu.qwen3_5",
              "graph_trainer_qwen35_moe_debugmodel_tpu_moe_kernels",
              use_splash_attention_kernel=False,
          )
          + common
          + extra_args,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          start_trainer=_RecordLosses(path),  # pyrefly: ignore[bad-argument-type]
          enable_compile=True,
      )
      with open(path) as f:
        losses[arm] = dict(json.load(f))
    logging.info(
        "qwen3_5 MoE TPU-kernel GraphTrainer losses by arm: %s", losses
    )
    for step in (1, 2, 3):
      self.assertTrue(math.isfinite(losses["ep2"][step]), msg=f"step {step}")
      self.assertLess(
          abs(losses["ep2"][step] - losses["ep1"][step]),
          abs(losses["ep1"][step]) * 2e-3,
          msg=f"step {step}",
      )

  def test_qwen3_5_graph_trainer_matches_eager(self):
    """The graph-split GraphTrainer must train Qwen3.5 MoE to eager's losses.

    The eager lane is the regular TPU trainer on `qwen35_moe_debugmodel`
    (SimpleFSDP, selective AC, no compile); the GraphTrainer lane is
    `graph_trainer_qwen35_moe_debugmodel`: the same model through
    `_use_graph_split`, as the 35B recipes use it (one layer per forward and
    backward executable, `tpu_selective` saving only `moe.routing`). Same model
    config, data, seed, fp32 params and reductions, and foreach AdamW, so the
    arms differ only in how the step is captured and executed. Every step is
    compared, not just the first (see the Gemma 4 test).
    """
    module = "torchtitan.experiments.tpu.qwen3_5"
    arms = {
        "eager": (
            _train_args(
                module,
                "qwen35_moe_debugmodel",
                use_splash_attention_kernel=False,
            ),
            False,
        ),
        "graph_trainer": (
            _train_args(
                module,
                "graph_trainer_qwen35_moe_debugmodel",
                use_splash_attention_kernel=False,
            ),
            True,
        ),
        "control": (
            _train_args(
                module,
                "graph_trainer_qwen35_moe_debugmodel",
                use_splash_attention_kernel=False,
            )
            + ["--lr_scheduler.warmup_steps=3"],
            True,
        ),
    }
    tempdir = self.create_tempdir().full_path
    losses = {}
    for arm, (config_args, enable_compile) in arms.items():
      path = os.path.join(tempdir, f"{arm}.json")
      self._test_train_distributed(
          config_args=config_args + ["--parallelism.use_simple_fsdp"],
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          start_trainer=_RecordLosses(path),  # pyrefly: ignore[bad-argument-type]
          enable_compile=enable_compile,
      )
      with open(path) as f:
        losses[arm] = json.load(f)

    self.assertNotEmpty(losses["eager"])
    self.assertLen(losses["graph_trainer"], len(losses["eager"]))
    for (step, graph), (_, eager) in zip(
        losses["graph_trainer"], losses["eager"]
    ):
      logging.info(
          "qwen3_5 step %d: graph_trainer loss %.6f, eager loss %.6f,"
          " |diff| %.2e (%.2e relative)",
          step,
          graph,
          eager,
          abs(graph - eager),
          abs(graph - eager) / abs(eager),
      )
      self.assertAlmostEqual(
          graph, eager, delta=abs(eager) * 1e-4, msg=f"step {step}"
      )
    control_losses = dict(losses["control"])
    eager_losses = dict(losses["eager"])
    for step in (2, 3):
      self.assertGreater(
          abs(control_losses[step] - eager_losses[step]),
          abs(eager_losses[step]) * 1e-4,
          msg=f"control, step {step}",
      )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
