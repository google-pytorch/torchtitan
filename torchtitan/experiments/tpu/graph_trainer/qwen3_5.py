"""TPU GraphTrainer integration for Qwen3.5 (text-only Dense and MoE)."""

import dataclasses

from torch.fx.traceback import annotate_fn
import torch.nn as nn
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.experiments.graph_trainer.common_utils import (
    _MODULE_FQN,
    annotate_module_fqns,
    annotate_moe_ep_regions,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.tpu import qwen3_5 as tpu_qwen3_5
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import configure_simple_fsdp_shard_dims
from torchtitan.experiments.tpu.qwen3_5.infra.parallelize import apply_qwen3_5_simple_fsdp
from torchtitan.models.common.moe import configure_moe_aux_loss_reduction
from torchtitan.models.qwen3_5.model import Qwen35Model
from torchtitan.protocols.model_spec import ModelSpec


def annotate_expert_unshard_fqns(model: nn.Module) -> None:
  """Tags each module's ``unshard_weights`` with its FQN, like ``forward``.

  ``annotate_module_fqns`` wraps every module's ``forward``. EP token chunking
  unshards the routed experts' weights once per layer through
  ``TpuGroupedExperts.unshard_weights`` instead, outside ``forward``; wrapping
  that entry point the same way gives its SimpleFSDP all-gathers the same
  ``module_fqn``, which SimpleFSDP bucketing orders and groups them by.
  """
  for fqn, module in model.named_modules():
    if fqn and callable(getattr(module, "unshard_weights", None)):
      module.unshard_weights = annotate_fn({_MODULE_FQN: fqn})(
          module.unshard_weights
      )


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
  # CP recipes run the GatedDeltaNet head-parallel (qwen3_5/cp_gated_delta.py;
  # upstream's eager parallelize_qwen3_5 rejects CP because its GatedDeltaNet
  # needs the whole sequence). TP is untested on TPU and the TPU model_registry
  # has no pipelining_fn; EP runs with SimpleFSDP only.
  if parallel_dims.tp_enabled or parallel_dims.pp_enabled:
    raise NotImplementedError(
        "TPU GraphTrainer Qwen3.5 does not support TP or PP."
    )
  if parallel_dims.ep_enabled and not parallelism.use_simple_fsdp:
    raise NotImplementedError(
        "TPU GraphTrainer Qwen3.5 EP requires SimpleFSDP (use_simple_fsdp=True)."
    )

  if any(layer.moe is not None for layer in model.config.layers):
    annotate_moe_ep_regions()
  annotate_module_fqns(model)
  annotate_expert_unshard_fqns(model)

  model.parallelize(parallel_dims)

  configure_moe_aux_loss_reduction(model, parallel_dims)
  configure_simple_fsdp_shard_dims(model, parallel_dims)
  model = apply_qwen3_5_simple_fsdp(
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
      pipelining_fn=None,
  )
