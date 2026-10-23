from functools import partial
from typing import cast

from torch import nn
from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.examples.gemma4 import (
    Gemma4Model,
    model_registry,
)
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.models.muse_glimmer.model import SoftCappedLinear

_FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss."
    "fused_loss_soft_capped_lm_head"
)


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Select the soft-capped fused Pallas loss and its LM-head override."""
  assert config.model_spec is not None
  model_config = cast(Gemma4Model.Config, config.model_spec.model)
  lm_head = cast(SoftCappedLinear.Config, model_config.lm_head)
  config.loss = PallasCrossEntropyLoss.Config(
      soft_cap=lm_head.output_soft_cap_temp,
      output_multiplier=lm_head.output_multiplier,
  )
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)


def gemma4_debugmodel() -> TPUTrainerConfig:
  """Gemma4 dense debug model configuration for TPU."""
  model_spec = model_registry(
      "debugmodel", attn_backend="sdpa", enable_weight_tying=False
  )
  model_config = cast(Gemma4Model.Config, model_spec.model)
  # _EMBEDDING_INIT uses std=0.02 so that Gemma4Embedding's sqrt(dim) scaling
  # stays O(1) on large models (31B dim=5376, 26B-A4B dim=2816) for convergence.
  # In the backward pass, layers[0].attention_norm (RMSNorm) divides by
  # RMS(x_0) ≈ std * sqrt(dim) while Gemma4Embedding multiplies by sqrt(dim),
  # scaling tok_embeddings.weight.grad by 1/std (50x at std=0.02) and amplifying
  # bf16-vs-fp32 absolute gradient differences beyond distributed parity test
  # tolerances. Override to std=1.0 (like qwen3_debugmodel) for debugmodel.
  model_config.tok_embeddings.param_init = {
      "weight": partial(nn.init.normal_, mean=0.0, std=1.0)
  }
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=3e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=4 * 512,
          max_context_length=512,
          steps=10,
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          pipeline_parallel_schedule="Interleaved1F1B"
      ),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=FullAC.Config(),
      validator=Validator.Config(
          freq=5,
          steps=10,
      ),
  )


def gemma4_debugmodel_fused_ce() -> TPUTrainerConfig:
  """Gemma4 dense debug model with fused Pallas cross-entropy and soft-cap."""
  cfg = gemma4_debugmodel()
  _enable_pallas_cross_entropy(cfg)
  return cfg


def gemma4_moe_testmodel() -> TPUTrainerConfig:
  """Gemma4 MoE test model configuration for TPU."""
  cfg = gemma4_debugmodel()
  cfg.model_spec = model_registry(
      "testmodel_moe",
      attn_backend="sdpa",
      enable_weight_tying=False,
  )
  cfg.training.max_context_length = 128
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 4 * 128
  return cfg


def gemma4_moe_debug() -> TPUTrainerConfig:
  """Gemma4 MoE debug model configuration for TPU."""
  cfg = gemma4_debugmodel()
  cfg.model_spec = model_registry(
      "debugmodel_moe",
      attn_backend="sdpa",
      converters=[
          TpuTokenDispatcherConverter.Config(),
          TpuGroupedExpertsConverter.Config(),
      ],
      enable_weight_tying=False,
  )
  return cfg


def gemma4_moe_debug_fused_ce() -> TPUTrainerConfig:
  """Gemma4 MoE debug model with fused Pallas cross-entropy and soft-cap."""
  cfg = gemma4_moe_debug()
  _enable_pallas_cross_entropy(cfg)
  return cfg


def gemma4_31b() -> TPUTrainerConfig:
  """Gemma4 31B dense model configuration for TPU."""
  model_spec = model_registry(
      "31B", attn_backend="sdpa", enable_weight_tying=False
  )
  cfg = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./assets/hf/gemma-4-31B",
      model_spec=model_spec,
      optimizer=default_adamw(lr=3e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=200,
          decay_ratio=0.8,
          decay_type="cosine",
          min_lr_factor=0.1,
      ),
      training=TrainingConfig(
          max_norm=1.0,
          num_tokens_per_microbatch_per_dp_rank=2 * 4096,
          max_context_length=4096,
          steps=15,
          mixed_precision_reduce="float32",
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          enable_tpu_topo_mesh=True,
          data_parallel_shard_degree=-1,
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=False,
      ),
      activation_checkpoint=FullAC.Config(),
      compile=CompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      validator=Validator.Config(
          enable=False,
          freq=5,
          steps=10,
      ),
  )
  _enable_pallas_cross_entropy(cfg)
  return cfg


def gemma4_26b_a4b() -> TPUTrainerConfig:
  """Gemma4 26B-A4B MoE model configuration for TPU."""
  cfg = gemma4_31b()
  cfg.hf_assets_path = "./assets/hf/gemma-4-26B-A4B"
  cfg.model_spec = model_registry(
      "26B-A4B",
      attn_backend="sdpa",
      converters=[
          TpuTokenDispatcherConverter.Config(),
          TpuGroupedExpertsConverter.Config(),
      ],
      enable_weight_tying=False,
  )
  _enable_pallas_cross_entropy(cfg)
  return cfg
