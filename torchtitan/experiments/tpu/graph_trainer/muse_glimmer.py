"""Muse Glimmer on the TPU GraphTrainer path.

Wraps upstream ``torchtitan.experiments.graph_trainer.muse_glimmer`` with the
TPU Muse Glimmer model registry and FSDP policy:

* The model config comes from ``experiments.tpu.muse_glimmer.model_registry``,
  so every layer keeps the window-aware SDPA inner attention that the Splash
  patch (``workarounds.apply_patches``, run by ``TpuGraphTrainer`` before
  ``parallelize``)
  turns into causal / sliding-window ``LocalMask`` kernels. This also makes the
  TPU-only ``testmodel`` flavor available.
* Before upstream's SimpleFSDP setup, vocab-sized weights whose dim 0 does not
  divide the FSDP degree switch to dim-1 sharding
  (``configure_muse_glimmer_simple_fsdp_shard_dims``).

Only SimpleFSDP / HSDP is supported here. Upstream's GraphTrainer TP path goes
through ``Module._parallelize`` with ``spmd_types``, which the fork does not use,
and TP forward currently hangs on TPU (b/567185409).
"""

import dataclasses
from dataclasses import dataclass

from torch import nn
from torchtitan.config import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.muse_glimmer.model import GraphTrainerMuseGlimmerModel
from torchtitan.experiments.tpu import muse_glimmer as tpu_muse_glimmer
from torchtitan.experiments.tpu.muse_glimmer.infra.parallelize import configure_muse_glimmer_simple_fsdp_shard_dims


class TpuGraphTrainerMuseGlimmerModel(GraphTrainerMuseGlimmerModel):
  """GraphTrainer Muse Glimmer with TPU uneven-vocab SimpleFSDP policy."""

  @dataclass(kw_only=True, slots=True)
  class Config(GraphTrainerMuseGlimmerModel.Config):
    pass

  def parallelize(
      self,
      *,
      training: TrainingConfig,
      parallel_dims: ParallelDims,
      **kwargs,
  ) -> nn.Module:
    return parallelize_muse_glimmer(
        self,
        training=training,
        parallel_dims=parallel_dims,
        **kwargs,
    )

  def pipeline(self, **kwargs):
    raise NotImplementedError(
        "TPU GraphTrainer Muse Glimmer supports SimpleFSDP/HSDP only; TP and"
        " PP are not supported."
    )


def parallelize_muse_glimmer(
    model: GraphTrainerMuseGlimmerModel,
    *,
    training: TrainingConfig,
    parallel_dims: ParallelDims,
    **kwargs,
) -> nn.Module:
  """Applies the TPU uneven-vocab FSDP policy, then upstream's setup."""
  # Upstream's GraphTrainerMuseGlimmerModel.parallelize already rejects CP, and
  # the TPU model_registry only accepts text-only flavors. Upstream applies TP
  # through Module._parallelize (spmd_types), which this fork does not use and
  # which hangs on TPU (b/567185409); PP is not supported here.
  if parallel_dims.tp_enabled or parallel_dims.pp_enabled:
    raise NotImplementedError(
        "TPU GraphTrainer Muse Glimmer supports SimpleFSDP/HSDP only; TP and"
        " PP are not supported."
    )
  configure_muse_glimmer_simple_fsdp_shard_dims(model, parallel_dims)
  return GraphTrainerMuseGlimmerModel.parallelize(
      model,
      training=training,
      parallel_dims=parallel_dims,
      **kwargs,
  )


def model_registry(flavor: str) -> TpuGraphTrainerMuseGlimmerModel.Config:
  """Returns the GraphTrainer model config for a TPU Muse Glimmer flavor."""
  base = tpu_muse_glimmer.model_registry(flavor)
  return TpuGraphTrainerMuseGlimmerModel.Config(**{
      field.name: getattr(base, field.name)
      for field in dataclasses.fields(base)
  })


model_registry._model_config_cls = TpuGraphTrainerMuseGlimmerModel.Config  # type: ignore[attr-defined]
