"""TPU configuration registry for Qwen3.5 (Dense and MoE) models."""

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import (
    CommConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.tpu.datasets import (
    TPUGrainDataLoader,
    c4_test_dataloader,
)
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExpertsConverter
from torchtitan.experiments.tpu.qwen3_5 import model_registry
from torchtitan.experiments.tpu.token_dispatcher import TpuTokenDispatcherConverter
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig, TPUTrainerConfig

_TPU_MOE_CONVERTERS = [
    TpuTokenDispatcherConverter.Config(),
    TpuGroupedExpertsConverter.Config(),
]


def qwen35_debugmodel() -> TPUTrainerConfig:
  """Debug config for hybrid Qwen3.5 (dense) on TPU."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry("debugmodel", seq_len=256),
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * 256,
          max_context_length=256,
          steps=10,
      ),
      dataloader=c4_test_dataloader(),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
      ),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(preserve_rng_state=False),
      compile=CompileConfig(enable=False, backend="tpu"),
      validator=Validator.Config(enable=False),
      tpu_config=TPUConfig(enable_amp=False),
  )


def qwen35_moe_debugmodel() -> TPUTrainerConfig:
  """Debug config for 4-layer hybrid Qwen3.5 MoE on TPU."""
  cfg = qwen35_debugmodel()
  cfg.model_spec = model_registry(
      "debugmodel_moe", seq_len=256, converters=_TPU_MOE_CONVERTERS
  )
  return cfg


def qwen35_moe_35b() -> TPUTrainerConfig:
  """Qwen3.5-35B-A3B (40 hybrid layers, 256 experts) on the regular trainer.

  Fits one tpu7x-2x2x1 host (4 chips, 8 ranks) and up: bf16 parameters,
  gradients and AdamW state (8 bytes per parameter), SimpleFSDP resharding
  after forward (ZeRO-3), full activation checkpointing, and torch_tpu's
  DEFER_AND_FUSE eager mode with a compiled gradient norm. 32K tokens per rank
  (16 sequences of 2048).
  """
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./assets/hf/Qwen3.5-35B-A3B",
      # Logging syncs the host on the loss; every 5 steps lets the host
      # dispatch the next step while the current one runs.
      metrics=MetricsProcessor.Config(log_freq=5),
      model_spec=model_registry("35B-A3B", converters=_TPU_MOE_CONVERTERS),
      optimizer=default_adamw(lr=3e-4, implementation="foreach"),
      # A 2-step warmup to the peak LR diverged the randomly initialized
      # 40-layer model (loss 12.77 -> 6.45 -> 8.48 -> NaN at step 4 on
      # tpu7x-2x2x1). Match the upstream Qwen3-30B-A3B schedule (LR 3e-4, 200
      # warmup steps); short benchmark runs clamp the warmup to
      # `training.steps`, as the Qwen3 TPU benchmark recipes do.
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=200,
          decay_ratio=0.8,
          decay_type="cosine",
          min_lr_factor=0.1,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=16 * 2048,
          max_context_length=2048,
          steps=10,
          dtype="bfloat16",
          mixed_precision_param="bfloat16",
          mixed_precision_reduce="float32",
      ),
      dataloader=TPUGrainDataLoader.Config(dataset="c4"),
      parallelism=ParallelismConfig(
          expert_parallel_degree=1,
          fsdp_reshard_after_forward="always",
          use_simple_fsdp=True,
      ),
      comm=CommConfig(train_timeout_seconds=600),
      activation_checkpoint=FullAC.Config(preserve_rng_state=False),
      compile=CompileConfig(enable=False, backend="tpu"),
      validator=Validator.Config(enable=False),
      tpu_config=TPUConfig(
          enable_amp=False,
          eager_mode="DEFER_AND_FUSE",
          compile_grad_norm=True,
      ),
  )
