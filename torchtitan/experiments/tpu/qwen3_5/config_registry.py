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
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.datasets import (
    TPUGrainDataLoader,
    c4_test_dataloader,
)
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.qwen3_5 import model_registry
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
)

_TPU_MOE_CONVERTERS = [
    TpuTokenDispatcherConverter.Config(),
    TpuGroupedExpertsConverter.Config(),
]

FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"
)


def _enable_pallas_cross_entropy(
    cfg: TPUTrainerConfig,
    *,
    b_block_size: int = 1024,
    h_block_size: int = 512,
    v_block_size: int = 2048,
) -> None:
  """Selects the Pallas fused linear cross-entropy loss and LM-head wrapper."""
  cfg.loss = PallasCrossEntropyLoss.Config(
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
  )
  if FUSED_LOSS_OVERRIDE not in cfg.override.imports:
    cfg.override.imports.append(FUSED_LOSS_OVERRIDE)


def _use_tpu_moe_kernels(cfg: TPUTrainerConfig) -> None:
  """Selects the Qwen3-30B-A3B TPU recipe's MoE kernels.

  Pallas paired grouped matmuls with kernel-native expert weights, and dScore
  in one fused FP32 reduction instead of saving an FP32 copy of every expert
  output. These are per-model-config settings, so a recipe that replaces
  `model_spec` must apply them again.

  Args:
    cfg: The recipe to update in place.
  """
  assert cfg.model_spec is not None
  experts = list(cfg.model_spec.model.traverse(TpuGroupedExperts.Config))
  dispatchers = list(
      cfg.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config)
  )
  if not experts or not dispatchers:
    raise ValueError(
        "The TPU MoE kernels need TpuGroupedExperts and a TPU token "
        "dispatcher (_TPU_MOE_CONVERTERS)"
    )
  for _, experts_config, _, _ in experts:
    experts_config.use_pallas_grouped_mm = True
    experts_config.use_pallas_paired_grouped_mm = True
    experts_config.pallas_grouped_mm_tile_m = 256
    experts_config.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)
    experts_config.use_kernel_native_weight_layout = True
  for _, dispatcher, _, _ in dispatchers:
    dispatcher.fuse_route_score_backward = True


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


def qwen35_moe_debugmodel_fused_ce() -> TPUTrainerConfig:
  """Debug config for 4-layer hybrid Qwen3.5 MoE with Pallas fused CE loss."""
  cfg = qwen35_moe_debugmodel()
  cfg.training.dtype = "bfloat16"
  cfg.training.mixed_precision_param = "bfloat16"
  cfg.training.mixed_precision_reduce = "bfloat16"
  cfg.training.max_context_length = 256
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 1024
  _enable_pallas_cross_entropy(
      cfg, b_block_size=1024, h_block_size=128, v_block_size=512
  )
  return cfg


def qwen35_moe_debugmodel_tpu_moe_kernels() -> TPUTrainerConfig:
  """Debug config for 4-layer hybrid Qwen3.5 MoE with the TPU MoE kernels.

  The routed experts of `qwen35_moe_35b` (`_use_tpu_moe_kernels`): Pallas
  paired grouped matmuls on kernel-native expert weights, and the fused
  route-score backward.
  """
  cfg = qwen35_moe_debugmodel()
  _use_tpu_moe_kernels(cfg)
  return cfg


def qwen35_moe_35b() -> TPUTrainerConfig:
  """Qwen3.5-35B-A3B (40 hybrid layers, 256 experts) on the regular trainer.

  Fits one tpu7x-2x2x1 host (4 chips, 8 ranks) and up: bf16 parameters,
  gradients and AdamW state (8 bytes per parameter), SimpleFSDP resharding
  after forward (ZeRO-3), full activation checkpointing, Pallas paired grouped
  matmuls for the experts, Splash attention, Pallas fused linear cross-entropy,
  and torch_tpu's DEFER_AND_FUSE eager mode with a compiled gradient norm.
  32K tokens per rank (16 sequences of 2048).
  """
  cfg = TPUTrainerConfig(
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
      # Full attention in Qwen3.5 35B-A3B has head_dim=256 (2x Qwen3's 128):
      # Splash's backward overflows the default 32 MiB of scoped VMEM with
      # 2048 blocks, so use 1024.
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_block_q=1024,
          sa_block_kv=1024,
          sa_block_dkv=1024,
          sa_block_kv_compute=512,
          sa_block_q_dkv=1024,
          sa_block_kv_dkv=1024,
          sa_block_kv_dkv_compute=512,
          sa_block_q_dq=1024,
          sa_block_kv_dq=1024,
      ),
      # The largest scoped VMEM limit TPU7x accepts (65536 is rejected); the
      # default is 32 MiB.
      libtpu_init_flags=tpu_flags.merge_flags(
          {"xla_tpu_scoped_vmem_limit_kib": 65472}
      ),
      tpu_config=TPUConfig(
          enable_amp=False,
          eager_mode="DEFER_AND_FUSE",
          compile_grad_norm=True,
      ),
  )
  _use_tpu_moe_kernels(cfg)
  _enable_pallas_cross_entropy(cfg)
  return cfg
