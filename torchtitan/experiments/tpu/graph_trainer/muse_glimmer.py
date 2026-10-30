"""Muse Glimmer on the TPU GraphTrainer path.

Wraps upstream ``torchtitan.experiments.graph_trainer.muse_glimmer`` with the
TPU Muse Glimmer model registry and FSDP policy:

* The model config comes from ``experiments.tpu.muse_glimmer.model_registry``,
  so every layer keeps the window-aware SDPA inner attention that the Splash
  patch (``workarounds.apply_patches``, run by ``TpuGraphTrainer`` before
  ``parallelize_fn``)
  turns into causal / sliding-window ``LocalMask`` kernels. This also makes the
  TPU-only ``testmodel`` flavor available.
* Before upstream's SimpleFSDP setup, vocab-sized weights whose dim 0 does not
  divide the FSDP degree switch to dim-1 sharding
  (``configure_muse_glimmer_simple_fsdp_shard_dims``).

Only SimpleFSDP / HSDP is supported here. Upstream's GraphTrainer TP path goes
through ``Module.parallelize`` with ``spmd_types``, which the fork does not use,
and TP forward currently hangs on TPU (b/567185409).
"""

import dataclasses

from torch import nn
from torchtitan.config import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.muse_glimmer.model import GraphTrainerMuseGlimmerModel
from torchtitan.experiments.graph_trainer.muse_glimmer.parallelize import parallelize_muse_glimmer as graph_trainer_parallelize_muse_glimmer
from torchtitan.experiments.tpu import muse_glimmer as tpu_muse_glimmer
from torchtitan.experiments.tpu.muse_glimmer.infra.parallelize import configure_muse_glimmer_simple_fsdp_shard_dims
from torchtitan.protocols.model_spec import ModelSpec


def parallelize_muse_glimmer(
    model: GraphTrainerMuseGlimmerModel,
    *,
    training: TrainingConfig,
    parallel_dims: ParallelDims,
    **kwargs,
) -> nn.Module:
  """Applies the TPU uneven-vocab FSDP policy, then upstream's setup."""
  # Upstream's parallelize_muse_glimmer already rejects CP, and the TPU
  # model_registry only accepts text-only flavors. Upstream applies TP through
  # Module.parallelize (spmd_types), which this fork does not use and which
  # hangs on TPU (b/567185409); PP has no pipelining_fn here.
  if parallel_dims.tp_enabled or parallel_dims.pp_enabled:
    raise NotImplementedError(
        "TPU GraphTrainer Muse Glimmer supports SimpleFSDP/HSDP only; TP and"
        " PP are not supported."
    )
  configure_muse_glimmer_simple_fsdp_shard_dims(model, parallel_dims)
  return graph_trainer_parallelize_muse_glimmer(
      model,
      training=training,
      parallel_dims=parallel_dims,
      **kwargs,
  )


def model_registry(flavor: str) -> ModelSpec:
  """Returns the GraphTrainer ``ModelSpec`` for a TPU Muse Glimmer flavor.

  ``to_graph_trainer_config`` only takes the model class, ``parallelize_fn``
  and ``pipelining_fn`` from this spec; the field values come from the recipe's
  own model config.

  Args:
    flavor: ``testmodel`` or a text-only upstream flavor.

  Returns:
    A spec with a ``GraphTrainerMuseGlimmerModel.Config`` model and the TPU
    ``parallelize_muse_glimmer``.
  """
  model_spec = tpu_muse_glimmer.model_registry(flavor)
  # Same field-for-field re-wrap as upstream graph_trainer.muse_glimmer's
  # model_registry and the TPU qwen3 wrapper: only the class changes
  # (GraphTrainerMuseGlimmerModel.Config adds no fields; its model runs
  # init_states with SimpleFSDP parametrization disabled), so the TPU attention
  # swapped in by the TPU registry is kept.
  model_config = GraphTrainerMuseGlimmerModel.Config(**{
      field.name: getattr(model_spec.model, field.name)
      for field in dataclasses.fields(model_spec.model)
  })
  return dataclasses.replace(
      model_spec,
      model=model_config,
      parallelize_fn=parallelize_muse_glimmer,
      pipelining_fn=None,
  )
