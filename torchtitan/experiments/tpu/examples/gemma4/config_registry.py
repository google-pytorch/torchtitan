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
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.examples.gemma4 import (
    Gemma4Model,
    model_registry,
)
from torchtitan.experiments.tpu.graph_trainer.config import (
    GraphSplitterConfig,
    TpuGraphTrainerCompileConfig,
)
from torchtitan.experiments.tpu.graph_trainer.gemma4 import (
    model_registry as graph_trainer_model_registry,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    TpuTrainerCompileConfig,
    TPUTrainerConfig,
)
from torchtitan.models.muse_glimmer.model import SoftCappedLinear


# Host-offload copy-insertion flags shared with the Qwen3 235B graph-splitter
# offload recipe: keep activation D2H/H2D copies off SparseCore and use region
# analysis to eliminate defensive activation snapshots across donated aliases.
_GEMMA4_GRAPH_SPLIT_OFFLOAD_FLAGS = tpu_flags.merge_flags(
    {
        "xla_tpu_enable_offloading_copy_to_sparsecore": False,
        "xla_tpu_copy_insertion_use_region_analysis": True,
        "xla_tpu_copy_insertion_use_region_analysis_limit": 1000,
    },
)

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


def _enable_moe_pallas_kernels(config: TPUTrainerConfig) -> None:
  """Enable Pallas grouped MM, paired W1/W3, native weight layout, and fused route-score bwd."""
  assert config.model_spec is not None
  model_config = cast(Gemma4Model.Config, config.model_spec.model)
  dispatchers = list(
      model_config.traverse(TpuAllToAllTokenDispatcher.Config)
  )
  if not dispatchers:
    raise ValueError(
        "Gemma4 MoE Pallas configuration requires TpuTokenDispatcherConverter"
    )
  for _, dispatcher, _, _ in dispatchers:
    dispatcher.fuse_route_score_backward = True

  experts_list = list(model_config.traverse(TpuGroupedExperts.Config))
  if not experts_list:
    raise ValueError(
        "Gemma4 MoE Pallas configuration requires TpuGroupedExpertsConverter"
    )
  for _, experts, _, _ in experts_list:
    experts.use_pallas_grouped_mm = True
    experts.use_pallas_paired_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 256
    experts.pallas_grouped_mm_tile_k = 0
    experts.pallas_grouped_mm_tile_n = 0
    experts.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)
    experts.use_kernel_native_weight_layout = True


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
      xla_flags=tpu_flags.merge_flags(
          xla_enable_hlo_sharding_v3=False,
      ),
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


def gemma4_moe_debug_pallas() -> TPUTrainerConfig:
  """Gemma4 MoE debug model with Pallas grouped MM, paired W1/W3, native weights, and fused route-score backward."""
  cfg = gemma4_moe_debug()
  _enable_moe_pallas_kernels(cfg)
  return cfg


def graph_trainer_gemma4_debugmodel() -> TpuGraphTrainer.Config:
  """Gemma4 dense debug model configuration using TPU GraphTrainer."""
  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      gemma4_debugmodel(), graph_trainer_model_registry
  )
  # Match gemma4_debugmodel's FullAC on the joint graph.
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
      memory_policy="full",
  )
  return config


def graph_trainer_gemma4_moe_debug() -> TpuGraphTrainer.Config:
  """Gemma4 MoE debug model configuration using TPU GraphTrainer."""
  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      gemma4_moe_debug(), graph_trainer_model_registry
  )
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
      memory_policy="full",
  )
  return config


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
      # Disable Shardy HLO sharding v3 (default since libtpu 0.0.48), matching
      # Qwen3 and Muse Glimmer recipes.
      xla_flags=tpu_flags.merge_flags(
          xla_enable_hlo_sharding_v3=False,
      ),
      compile=TpuTrainerCompileConfig(
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
  _enable_moe_pallas_kernels(cfg)
  return cfg


def _configure_graph_trainer_full_ac(
    base_cfg: TPUTrainerConfig,
) -> TpuGraphTrainer.Config:
  """Configure unsplit GraphTrainer ZeRO-3 with full activation checkpointing."""
  cfg = TpuGraphTrainer.Config.from_tpu_trainer_config(
      base_cfg, graph_trainer_model_registry
  )
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  cfg.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
      memory_policy="full",
      enable_remat_barriers=True,
      fsdp_barrier_policy="interleave",
  )
  return cfg


def _configure_graph_split(
    base_cfg: TPUTrainerConfig, *, offload: bool = False
) -> TpuGraphTrainer.Config:
  """Configure per-layer GraphTrainer splitting with full or host-offload AC."""
  cfg = TpuGraphTrainer.Config.from_tpu_trainer_config(
      base_cfg, graph_trainer_model_registry
  )
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  cfg.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
      memory_policy="full_and_offload" if offload else "full",
      optimizer_compile="eager",
      enable_remat_barriers=False,
      fsdp_barrier_policy="none",
      graph_split=GraphSplitterConfig(
          enabled=True,
          forward_layers_per_executable=1,
          forward_isolate_prefix=True,
          backward_partition_mode="by_layer",
          backward_layers_per_executable=1,
          cpu_offload_prefetch_executables=1 if offload else 0,
      ),
  )
  if offload:
    cfg.libtpu_init_flags = tpu_flags.merge_flags(
        cfg.libtpu_init_flags,
        _GEMMA4_GRAPH_SPLIT_OFFLOAD_FLAGS,
    )
  return cfg


def gemma4_31b_graph_trainer() -> TpuGraphTrainer.Config:
  """Gemma4 31B unsplit GraphTrainer ZeRO-3 recipe with full AC."""
  return _configure_graph_trainer_full_ac(gemma4_31b())


def gemma4_31b_graph_split() -> TpuGraphTrainer.Config:
  """Gemma4 31B per-layer graph-split GraphTrainer recipe with full AC."""
  return _configure_graph_split(gemma4_31b(), offload=False)


def gemma4_31b_graph_split_offload() -> TpuGraphTrainer.Config:
  """Gemma4 31B per-layer graph-split GraphTrainer recipe with full+offload AC."""
  return _configure_graph_split(gemma4_31b(), offload=True)


def gemma4_26b_a4b_graph_trainer() -> TpuGraphTrainer.Config:
  """Gemma4 26B-A4B unsplit GraphTrainer ZeRO-3 recipe with full AC."""
  return _configure_graph_trainer_full_ac(gemma4_26b_a4b())


def gemma4_26b_a4b_graph_split() -> TpuGraphTrainer.Config:
  """Gemma4 26B-A4B per-layer graph-split GraphTrainer recipe with full AC."""
  return _configure_graph_split(gemma4_26b_a4b(), offload=False)


def gemma4_26b_a4b_graph_split_offload() -> TpuGraphTrainer.Config:
  """Gemma4 26B-A4B per-layer graph-split GraphTrainer recipe with full+offload AC."""
  return _configure_graph_split(gemma4_26b_a4b(), offload=True)
