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
from torchtitan.experiments.tpu.examples.flint import model_registry
from torchtitan.experiments.tpu.examples.flint.model import FlintSinkAttention
from torchtitan.experiments.tpu.grouped_experts import (
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
    "flint_debugmodel",
    "flint_moe_testmodel",
]

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
