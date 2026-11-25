"""TPU GraphTrainer integration for Qwen3.5 (text-only Dense and MoE)."""

import dataclasses
from dataclasses import dataclass

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
from torchtitan.experiments.tpu.qwen3_5.model import TpuQwen35Model
from torchtitan.models.qwen3_5.model import Qwen35Model


class TpuGraphTrainerQwen35Model(TpuQwen35Model):
  """``TpuQwen35Model`` that parallelizes itself for the TPU GraphTrainer.

  The model stays the TPU Qwen3.5 model (TPU EP wiring in
  ``Config.update_from_config``, CP input sharding); only ``parallelize``
  changes. It does not mix in upstream's ``GraphTrainerModel``, whose
  ``init_states`` disables SimpleFSDP's parametrization.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TpuQwen35Model.Config):
    pass

  def parallelize(
      self,
      *,
      parallel_dims: ParallelDims,
      training: TrainingConfig,
      parallelism: ParallelismConfig,
      compile_config: GraphTrainerCompileConfig,
      ac_config: ActivationCheckpointingConfig | None = None,
      dump_folder: str,
      skip_dp: bool = False,
  ) -> nn.Module:
    if skip_dp:
      raise ValueError("GraphTrainer models do not support skip_dp=True.")
    return parallelize_qwen3_5(
        self,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,  # type: ignore[arg-type]
        dump_folder=dump_folder,
    )

  def pipeline(self, **kwargs):
    raise NotImplementedError(
        "TPU GraphTrainer Qwen3.5 does not support TP or PP."
    )


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
  # has no pipelining; EP runs with SimpleFSDP only.
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

  model._parallelize(parallel_dims)

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


def model_registry(flavor: str) -> TpuGraphTrainerQwen35Model.Config:
  """GraphTrainer model registry for TPU text-only Qwen3.5 flavors."""
  base = tpu_qwen3_5.model_registry(flavor)
  return TpuGraphTrainerQwen35Model.Config(
      **{
          field.name: getattr(base, field.name)
          for field in dataclasses.fields(base)
      }
  )


model_registry._model_config_cls = TpuGraphTrainerQwen35Model.Config  # type: ignore[attr-defined]
