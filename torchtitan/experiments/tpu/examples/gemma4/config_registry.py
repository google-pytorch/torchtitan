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
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.examples.gemma4 import (
    Gemma4Model,
    model_registry,
)
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig, TPUTrainerConfig


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
      # GraphTrainer ignores eager AC and reads TPUConfig.graph_trainer_memory_policy;
      # set "full" there too so --tpu_config.use_graph_trainer matches FullAC.
      activation_checkpoint=FullAC.Config(),
      tpu_config=TPUConfig(
          graph_trainer_memory_policy="full",
      ),
      xla_flags=tpu_flags.merge_flags(
          xla_enable_hlo_sharding_v3=False,
      ),
      validator=Validator.Config(
          freq=5,
          steps=10,
      ),
  )


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


def gemma4_31b() -> TPUTrainerConfig:
  """Gemma4 31B dense model configuration for TPU."""
  model_spec = model_registry(
      "31B", attn_backend="sdpa", enable_weight_tying=False
  )
  return TPUTrainerConfig(
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
      # GraphTrainer ignores eager AC and reads TPUConfig.graph_trainer_memory_policy;
      # set "full" there too so --tpu_config.use_graph_trainer matches FullAC.
      activation_checkpoint=FullAC.Config(),
      tpu_config=TPUConfig(
          graph_trainer_memory_policy="full",
      ),
      # Disable Shardy HLO sharding v3 (default since libtpu 0.0.48), matching
      # Qwen3 and Muse Glimmer recipes.
      xla_flags=tpu_flags.merge_flags(
          xla_enable_hlo_sharding_v3=False,
      ),
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
  return cfg


def _configure_graph_trainer_full_ac(cfg: TPUTrainerConfig) -> TPUTrainerConfig:
  """Configure unsplit GraphTrainer ZeRO-3 with full activation checkpointing."""
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  cfg.tpu_config.use_graph_trainer = True
  cfg.tpu_config.compile_mode = "aot_fx_trace"
  cfg.tpu_config.graph_trainer_memory_policy = "full"
  cfg.tpu_config.enable_graph_trainer_remat_schedule = True
  cfg.tpu_config.graph_trainer_fsdp_schedule = "interleave"
  return cfg


def _configure_graph_split(
    cfg: TPUTrainerConfig, *, offload: bool = False
) -> TPUTrainerConfig:
  """Configure per-layer GraphTrainer splitting with full or host-offload AC."""
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  cfg.tpu_config.use_graph_trainer = True
  cfg.tpu_config.compile_mode = "aot_fx_trace"
  cfg.tpu_config.use_graph_split = True
  cfg.tpu_config.graph_trainer_memory_policy = (
      "full_and_offload" if offload else "full"
  )
  cfg.tpu_config.enable_graph_trainer_remat_schedule = False
  cfg.tpu_config.graph_trainer_fsdp_schedule = "none"
  cfg.tpu_config.compile_optimizer = False
  cfg.tpu_config.optimizer_in_graph = False
  cfg.tpu_config.graph_split.forward_layers_per_executable = 1
  cfg.tpu_config.graph_split.forward_isolate_prefix = True
  cfg.tpu_config.graph_split.backward_partition_mode = "by_layer"
  cfg.tpu_config.graph_split.backward_layers_per_executable = 1
  if offload:
    cfg.tpu_config.graph_split.cpu_offload_prefetch_executables = 1
    cfg.libtpu_init_flags = tpu_flags.merge_flags(
        cfg.libtpu_init_flags,
        _GEMMA4_GRAPH_SPLIT_OFFLOAD_FLAGS,
    )
  return cfg


def gemma4_31b_graph_trainer() -> TPUTrainerConfig:
  """Gemma4 31B unsplit GraphTrainer ZeRO-3 recipe with full AC."""
  return _configure_graph_trainer_full_ac(gemma4_31b())


def gemma4_31b_graph_split() -> TPUTrainerConfig:
  """Gemma4 31B per-layer graph-split GraphTrainer recipe with full AC."""
  return _configure_graph_split(gemma4_31b(), offload=False)


def gemma4_31b_graph_split_offload() -> TPUTrainerConfig:
  """Gemma4 31B per-layer graph-split GraphTrainer recipe with full+offload AC."""
  return _configure_graph_split(gemma4_31b(), offload=True)


def gemma4_26b_a4b_graph_trainer() -> TPUTrainerConfig:
  """Gemma4 26B-A4B unsplit GraphTrainer ZeRO-3 recipe with full AC."""
  return _configure_graph_trainer_full_ac(gemma4_26b_a4b())


def gemma4_26b_a4b_graph_split() -> TPUTrainerConfig:
  """Gemma4 26B-A4B per-layer graph-split GraphTrainer recipe with full AC."""
  return _configure_graph_split(gemma4_26b_a4b(), offload=False)


def gemma4_26b_a4b_graph_split_offload() -> TPUTrainerConfig:
  """Gemma4 26B-A4B per-layer graph-split GraphTrainer recipe with full+offload AC."""
  return _configure_graph_split(gemma4_26b_a4b(), offload=True)
