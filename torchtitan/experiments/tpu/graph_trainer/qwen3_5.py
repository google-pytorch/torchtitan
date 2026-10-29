"""TPU GraphTrainer integration for Qwen3.5 (text-only Dense and MoE)."""

import dataclasses

import torch.nn as nn
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    annotate_moe_ep_regions,
    apply_simple_fsdp,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.tpu import qwen3_5 as tpu_qwen3_5
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import configure_simple_fsdp_shard_dims
from torchtitan.models.common.moe import configure_moe_aux_loss_reduction
from torchtitan.models.qwen3_5.model import Qwen35Model
from torchtitan.protocols.model_spec import ModelSpec


def parallelize_qwen3_5(
    model: Qwen35Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> nn.Module:
  """Applies FQN annotation, SimpleFSDP, and GraphTrainer compile to Qwen3.5."""
  del ac_config
  # Upstream's eager parallelize_qwen3_5 rejects CP too (the GatedDeltaNet
  # needs the whole sequence); TP is untested on TPU, the TPU model_registry
  # has no pipelining_fn, and EP requires mesh-split optimizer support.
  if (
      parallel_dims.tp_enabled
      or parallel_dims.cp_enabled
      or parallel_dims.pp_enabled
      or parallel_dims.ep_enabled
  ):
    raise NotImplementedError(
        "TPU GraphTrainer Qwen3.5 does not support TP, CP, PP or EP."
    )

  if any(layer.moe is not None for layer in model.config.layers):
    annotate_moe_ep_regions()
  annotate_module_fqns(model)

  if parallelism.spmd_backend == "spmd_types":
    model.parallelize(parallel_dims)

  configure_moe_aux_loss_reduction(model, parallel_dims)
  configure_simple_fsdp_shard_dims(model, parallel_dims)
  model = apply_simple_fsdp(
      model, parallel_dims=parallel_dims, training=training
  )  # pyrefly: ignore[bad-assignment]

  model = apply_compile(
      model,
      compile_config=compile_config,
      parallelism=parallelism,
      parallel_dims=parallel_dims,
      dump_folder=dump_folder,
  )
  return model


def model_registry(flavor: str) -> ModelSpec:
  """GraphTrainer model registry for TPU text-only Qwen3.5 flavors."""
  return dataclasses.replace(
      tpu_qwen3_5.model_registry(flavor),
      parallelize_fn=parallelize_qwen3_5,
  )
