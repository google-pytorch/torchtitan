"""Muse Glimmer on the TPU GraphTrainer path.

Wraps upstream ``torchtitan.experiments.graph_trainer.muse_glimmer`` with the
TPU Muse Glimmer model registry and FSDP policy:

* The model config comes from ``experiments.tpu.muse_glimmer.model_registry``,
  so every layer keeps the window-aware SDPA inner attention that the Splash
  patch (``workarounds.apply_patches``, run by ``TpuGraphTrainer`` after init)
  turns into causal / sliding-window ``LocalMask`` kernels. This also makes the
  TPU-only ``testmodel`` flavor available.
* Before sharding, vocab-sized weights whose dim 0 does not divide the FSDP
  degree switch to dim-1 sharding (``configure_muse_glimmer_simple_fsdp_shard_dims``
  on top of the shared ``configure_fsdp_shard_dims``).

Only FSDP / HSDP is supported here. Upstream's GraphTrainer TP path goes
through ``Module.parallelize`` with ``spmd_types``, which the fork does not use,
and TP forward currently hangs on TPU (b/567185409).
"""

import dataclasses

from torch import nn
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import annotate_module_fqns
from torchtitan.experiments.graph_trainer.muse_glimmer.model import GraphTrainerMuseGlimmerModel
from torchtitan.experiments.tpu import muse_glimmer as tpu_muse_glimmer
from torchtitan.experiments.tpu.graph_trainer.parallelize import parallelize_tpu_graph_trainer
from torchtitan.experiments.tpu.graph_trainer.fsdp_shard_dims import configure_fsdp_shard_dims
from torchtitan.experiments.tpu.muse_glimmer.infra.parallelize import configure_muse_glimmer_simple_fsdp_shard_dims
from torchtitan.protocols.model_spec import ModelSpec


def _configure_shard_dims(model: nn.Module, parallel_dims: ParallelDims) -> None:
  configure_fsdp_shard_dims(model, parallel_dims)
  configure_muse_glimmer_simple_fsdp_shard_dims(model, parallel_dims)


def parallelize_muse_glimmer(
    model: GraphTrainerMuseGlimmerModel,
    *,
    parallel_dims: ParallelDims,
    **kwargs,
) -> nn.Module:
  """Applies the TPU uneven-vocab FSDP policy, then the shared TPU setup."""
  # Upstream's parallelize_muse_glimmer rejects CP (the shard-dims hook does
  # too), and the TPU model_registry only accepts text-only flavors. Upstream
  # applies TP through Module.parallelize (spmd_types), which this fork does
  # not use and which hangs on TPU (b/567185409); PP has no pipelining_fn here.
  if parallel_dims.tp_enabled or parallel_dims.pp_enabled:
    raise NotImplementedError(
        "TPU GraphTrainer Muse Glimmer supports FSDP/HSDP only; TP and PP are"
        " not supported."
    )
  return parallelize_tpu_graph_trainer(
      model,
      annotate=annotate_module_fqns,
      configure_shard_dims=_configure_shard_dims,
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
  # (GraphTrainerMuseGlimmerModel.Config adds no fields), so the TPU attention
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
