"""TPU training recipes for Flint."""

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.examples.flint import (
    graph_trainer_model_registry,
    model_registry,
)
from torchtitan.experiments.tpu.graph_trainer.config import (
    TpuGraphTrainerCompileConfig,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.examples.flint.model import FlintSinkAttention
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    TPUConfig,
    TpuTrainerCompileConfig,
    TPUTrainerConfig,
)

__all__ = [
    "flint_40b",
    "flint_40b_proxy",
    "flint_40b_v002",
    "flint_40b_v003",
    "flint_40b_v004_mb16",
    "flint_40b_v004_mb32",
    "flint_40b_v005_gt",
    "flint_40b_v005_gt_mb16",
    "flint_40b_v005_gt_rb",
    "flint_40b_v005_gt_split",
    "flint_40b_v007",
    "flint_40b_v008",
    "flint_40b_v009",
    "flint_40b_v009_gmm",
    "flint_40b_v011",
    "flint_40b_v011_n40",
    "graph_trainer_flint_debugmodel",
    "flint_debugmodel",
    "flint_moe_testmodel",
]

_FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"
)

# Longer than the debug flavor's 129-token window, so the window masks.
_DEBUGMODEL_CONTEXT_LENGTH = 512
_40B_CONTEXT_LENGTH = 2048
# 512 sequences of 2048 tokens per optimizer step, the global batch of the HF
# backend's Flint runs and of the perfsim flint_40b reference. Microbatching
# (num_tokens_per_microbatch_per_dp_rank) changes the per-chip working set but
# not what one optimizer step computes.
_40B_GLOBAL_BATCH = 512


def _tpu_moe_converters():
  return [
      TpuTokenDispatcherConverter.Config(),
      TpuGroupedExpertsConverter.Config(),
  ]


def flint_debugmodel() -> TPUTrainerConfig:
  """Flint debug model (8 layers, dim 256, 8 experts) on the SDPA path."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_registry(
          "debugmodel",
          max_context_length=_DEBUGMODEL_CONTEXT_LENGTH,
          converters=_tpu_moe_converters(),
      ),
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=4 * _DEBUGMODEL_CONTEXT_LENGTH,
          max_context_length=_DEBUGMODEL_CONTEXT_LENGTH,
          steps=10,
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(use_simple_fsdp=True),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=FullAC.Config(),
  )


def flint_moe_testmodel() -> TPUTrainerConfig:
  """The four-layer parity-test flavor (one of each layer kind)."""
  config = flint_debugmodel()
  config.model_spec = model_registry(
      "testmodel", max_context_length=128, converters=_tpu_moe_converters()
  )
  config.training.max_context_length = 128
  config.training.num_tokens_per_microbatch_per_dp_rank = 4 * 128
  return config


def flint_40b() -> TPUTrainerConfig:
  """Flint 40B (49 layers, 256 experts, top-6) with SimpleFSDP.

  The onboarding baseline: reference attention (sinks + sliding window in
  float32 math, no Pallas kernel), plain cross entropy, torch grouped mm,
  full activation checkpointing and per-layer ``torch.compile``. Target
  topology tpu7x-2x2x2 (8 chips, 16 ranks).
  """
  seq = _40B_CONTEXT_LENGTH
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_registry(
          "40B", max_context_length=seq, converters=_tpu_moe_converters()
      ),
      optimizer=default_adamw(lr=3e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=5,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          max_norm=1.0,
          num_tokens_per_microbatch_per_dp_rank=4 * seq,
          num_tokens_per_train_step=_40B_GLOBAL_BATCH * seq,
          max_context_length=seq,
          steps=20,
          mixed_precision_reduce="float32",
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          enable_tpu_topo_mesh=True,
          data_parallel_shard_degree=-1,
      ),
      checkpoint=CheckpointManager.Config(enable=False),
      activation_checkpoint=FullAC.Config(),
      compile=TpuTrainerCompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      tpu_config=TPUConfig(
          enable_mfu_benchmark=True,
          mfu_benchmark_start_step=10,
      ),
      validator=Validator.Config(enable=False),
  )


def flint_40b_proxy() -> TPUTrainerConfig:
  """``flint_40b`` truncated to its first 6 layers (2 dense, 4 MoE including
  the full-attention layer 5): full width, fits one v6e-4 host."""
  config = flint_40b()
  config.model_spec = model_registry(
      "40B",
      max_context_length=_40B_CONTEXT_LENGTH,
      num_layers=6,
      converters=_tpu_moe_converters(),
  )
  config.training.num_tokens_per_train_step = -1
  config.training.num_tokens_per_microbatch_per_dp_rank = 2 * _40B_CONTEXT_LENGTH
  return config


def _use_splash_kernel(config: TPUTrainerConfig) -> None:
  for _, attention, _, _ in config.model_spec.model.traverse(
      FlintSinkAttention.Config
  ):
    attention.use_splash_kernel = True


def flint_40b_v002() -> TPUTrainerConfig:
  """``flint_40b`` with attention on the sbd Splash kernel (sinks in-kernel)."""
  config = flint_40b()
  _use_splash_kernel(config)
  return config


def flint_40b_proxy_v002() -> TPUTrainerConfig:
  """``flint_40b_proxy`` with the Splash kernel (local pre-dispatch check)."""
  config = flint_40b_proxy()
  _use_splash_kernel(config)
  return config


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Fused linear + cross entropy: the LM head hands (hidden, weight.T) to the
  loss, which never materializes the [tokens, 150144] logits."""
  config.loss = PallasCrossEntropyLoss.Config()
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)


def flint_40b_v003() -> TPUTrainerConfig:
  """``flint_40b_v002`` plus the fused (Pallas) cross entropy."""
  config = flint_40b_v002()
  _enable_pallas_cross_entropy(config)
  return config


def flint_40b_proxy_v003() -> TPUTrainerConfig:
  config = flint_40b_proxy_v002()
  _enable_pallas_cross_entropy(config)
  return config


def flint_40b_v004_mb16() -> TPUTrainerConfig:
  """``flint_40b_v003`` with 16-sequence microbatches (2 per optimizer step)."""
  config = flint_40b_v003()
  config.training.num_tokens_per_microbatch_per_dp_rank = 16 * _40B_CONTEXT_LENGTH
  return config


def flint_40b_v004_mb32() -> TPUTrainerConfig:
  """``flint_40b_v003`` with the whole per-rank batch (32 sequences) in one
  microbatch: no gradient accumulation."""
  config = flint_40b_v003()
  config.training.num_tokens_per_microbatch_per_dp_rank = 32 * _40B_CONTEXT_LENGTH
  return config


def _to_graph_trainer(config: TPUTrainerConfig) -> TpuGraphTrainer.Config:
  """The same recipe on TPU GraphTrainer: forward, loss and backward traced
  into one program per step with SimpleFSDP collectives inside it, so XLA can
  schedule parameter all-gathers and gradient reduce-scatters across layers.
  Full remat matches the per-layer recipe's full AC."""
  graph_config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      config, graph_trainer_model_registry
  )
  graph_config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
      memory_policy="full",
  )
  # Shardy HLO sharding v3 (default in libtpu 0.0.48) segfaults in
  # buildTensorFactorShardings on GraphTrainer.
  graph_config.xla_flags = tpu_flags.merge_flags(
      graph_config.xla_flags, xla_enable_hlo_sharding_v3=False
  )
  return graph_config


def graph_trainer_flint_debugmodel() -> TpuGraphTrainer.Config:
  return _to_graph_trainer(flint_debugmodel())


def flint_40b_v005_gt() -> TpuGraphTrainer.Config:
  """``flint_40b_v004_mb32`` (no gradient accumulation) on TPU GraphTrainer."""
  return _to_graph_trainer(flint_40b_v004_mb32())


def flint_40b_v005_gt_mb16() -> TpuGraphTrainer.Config:
  """``flint_40b_v004_mb16`` on TPU GraphTrainer (fallback if mb32 does not
  fit the whole-step program)."""
  return _to_graph_trainer(flint_40b_v004_mb16())


def flint_40b_proxy_gt() -> TpuGraphTrainer.Config:
  """``flint_40b_proxy_v003`` on GraphTrainer (local pre-dispatch check)."""
  config = flint_40b_proxy_v003()
  config.tpu_config.mfu_benchmark_start_step = 3
  return _to_graph_trainer(config)


def flint_40b_v005_gt_rb() -> TpuGraphTrainer.Config:
  """``flint_40b_v005_gt`` with remat barriers: each layer's recompute waits
  for the previous layer's parameter gradients, so XLA cannot hoist all
  recomputes and keep their workspaces live at once (the unbarriered
  whole-step program needed 388.7 GiB of temporaries at mb32)."""
  config = flint_40b_v005_gt()
  config.compile.enable_remat_barriers = True
  return config


def flint_40b_v005_gt_split() -> TpuGraphTrainer.Config:
  """``flint_40b_v005_gt`` split into one executable per layer (forward) and
  an FSDP-pipelined per-layer backward, the Qwen3-235B recipe's layout: the
  executable boundaries bound activation lifetimes."""
  config = flint_40b_v005_gt()
  split = config.compile.graph_split
  split.enabled = True
  split.forward_layers_per_executable = 1
  split.forward_isolate_prefix = True
  split.backward_partition_mode = "fsdp_pipeline"
  split.backward_layers_per_executable = 1
  config.parallelism.fsdp_reshard_after_forward = "always"
  return config


def flint_40b_proxy_gt_rb() -> TpuGraphTrainer.Config:
  config = flint_40b_proxy_gt()
  config.compile.enable_remat_barriers = True
  return config


def flint_40b_proxy_gt_split() -> TpuGraphTrainer.Config:
  config = flint_40b_proxy_gt()
  split = config.compile.graph_split
  split.enabled = True
  split.forward_layers_per_executable = 1
  split.forward_isolate_prefix = True
  split.backward_partition_mode = "fsdp_pipeline"
  split.backward_layers_per_executable = 1
  config.parallelism.fsdp_reshard_after_forward = "always"
  return config


# The graph-splitter flag set of the Qwen3-235B recipe: latency-hiding layer
# scheduling of SparseCore-aggregated FSDP collectives, both SparseCores for
# reduce-scatter, and async all-gathers kept as all-gathers.
_GRAPH_SPLIT_COLLECTIVE_FLAGS = tpu_flags.merge_flags(
    tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
    tpu_flags.DUAL_SPARSECORE_REDUCE_SCATTER,
    {"xla_tpu_prefer_async_allgather_to_allreduce": True},
)


def flint_40b_v007() -> TpuGraphTrainer.Config:
  """``flint_40b_v005_gt_split`` with the graph-split collective flags."""
  config = flint_40b_v005_gt_split()
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags, _GRAPH_SPLIT_COLLECTIVE_FLAGS
  )
  return config


def flint_40b_v008() -> TpuGraphTrainer.Config:
  """``flint_40b_v007`` at TPU7x performance state 7."""
  config = flint_40b_v007()
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags, tpu_flags.TPU7X_THROUGHPUT_P_STATE
  )
  return config


def _use_pallas_grouped_mm(config) -> None:
  """Pallas grouped mm for the 256 experts, W1/W3 paired. Tiles are the HF
  backend's Flint sweep winners on tpu7x-2x2x2 (2026-10-28, tile_m 512)."""
  for _, experts, _, _ in config.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.use_pallas_grouped_mm = True
    experts.use_pallas_paired_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 512
    experts.pallas_grouped_mm_tgmm_tiling = (512, 2048, 0)
    experts.pallas_grouped_mm_w2_tgmm_tiling = (512, 512, 0)
    experts.pallas_grouped_mm_w2_transposed_tgmm_tiling = (512, 0, 2048)


def flint_40b_v009_gmm() -> TpuGraphTrainer.Config:
  """``flint_40b_v008`` plus Pallas grouped mm (canonical weight layout)."""
  config = flint_40b_v008()
  _use_pallas_grouped_mm(config)
  return config


def flint_40b_v009() -> TpuGraphTrainer.Config:
  """``flint_40b_v009_gmm`` with the expert weights stored in the grouped-mm
  kernels' native [E, K, N] layout, so the forward does not transpose them
  (checkpoints keep the canonical layout via the module's state-dict hooks)."""
  config = flint_40b_v009_gmm()
  for _, experts, _, _ in config.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.use_kernel_native_weight_layout = True
  return config


def _save_attention_last_n_layers(config, num_layers: int) -> None:
  """Keep the Splash output (and its LSE) of the last ``num_layers`` layers
  for backward instead of recomputing it; every other value is still
  recomputed (the full-remat policy otherwise)."""
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_last_n_layers = {"attention.inner_attention": num_layers}


def flint_40b_v011() -> TpuGraphTrainer.Config:
  """``flint_40b_v009`` saving the attention output of the last 32 layers
  (~0.5 GiB per layer per device at 32 x 2048 tokens; v009 peaks at 69 GiB)."""
  config = flint_40b_v009()
  _save_attention_last_n_layers(config, 32)
  return config


def flint_40b_v011_n40() -> TpuGraphTrainer.Config:
  """``flint_40b_v011`` saving the last 40 layers (dose arm; ~20 GiB)."""
  config = flint_40b_v009()
  _save_attention_last_n_layers(config, 40)
  return config


def flint_40b_proxy_gt_split_save_attn() -> TpuGraphTrainer.Config:
  """``flint_40b_proxy_gt_split`` saving attention in all 6 layers (local
  pre-dispatch check of ``flint_40b_v011``)."""
  config = flint_40b_proxy_gt_split()
  _save_attention_last_n_layers(config, 6)
  return config
