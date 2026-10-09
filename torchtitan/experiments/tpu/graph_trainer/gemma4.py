"""Gemma 4 on the TPU GraphTrainer path (``--tpu_config.use_graph_trainer``).

``train.py`` needs a GraphTrainer registry per model. This one reuses
``experiments.tpu.examples.gemma4``'s model config unchanged (with
``enable_weight_tying=False``) and swaps in ``parallelize_gemma4``, which
annotates module FQNs, configures uneven-dimension SimpleFSDP shard dims
(``configure_gemma4_simple_fsdp_shard_dims``: the 262144-token vocabulary and
the 128-expert banks switch to a divisible hidden axis when the FSDP degree
does not divide dim 0), applies SimpleFSDP, and hands the model to
GraphTrainer's ``apply_compile``.

There is no ``GraphTrainerGemma4Model``: upstream's GraphTrainer models wrap
``init_states`` in ``disable_active_parametrization()``, but every Gemma 4
module initializes through ``param_init`` / ``named_parameters()``, which
bypasses SimpleFSDP's parametrization.

Only SimpleFSDP / HSDP is supported here.

Scan (``--tpu_config.use_scan_layers``) is not supported without
homogeneous-group partitioning: the sliding and global layers have different
``head_dim`` / ``n_kv_heads`` and different QKV projections (``QKVLinear`` vs
``QKVLinearKeqV``), so ``check_layers_are_stackable`` refuses them. Graph split
is the large-size path for Gemma 4.
"""

import dataclasses

from torch import nn

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    apply_simple_fsdp,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.tpu.examples import gemma4 as tpu_gemma4
from torchtitan.experiments.tpu.examples.gemma4.infra.parallelize import (
    configure_gemma4_simple_fsdp_shard_dims,
)
from torchtitan.models.common.moe import configure_moe_aux_loss_reduction
from torchtitan.protocols.model_spec import ModelSpec


def parallelize_gemma4(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> nn.Module:
  """Applies SimpleFSDP (TPU uneven-dim policy) and graph capture to Gemma 4.

  ``ac_config`` is accepted for the ``parallelize_fn`` protocol but unused: the
  graph trainer rematerializes its joint graph with
  ``tpu_config.graph_trainer_memory_policy`` instead of eager AC wrappers.
  """
  del ac_config
  unsupported = [
      name
      for name, enabled in (
          ("tensor", parallel_dims.tp_enabled),
          ("context", parallel_dims.cp_enabled),
          ("pipeline", parallel_dims.pp_enabled),
          ("expert", parallel_dims.ep_enabled),
      )
      if enabled
  ]
  if unsupported:
    raise NotImplementedError(
        "TPU GraphTrainer Gemma 4 supports SimpleFSDP/HSDP only; "
        f"{", ".join(unsupported)} parallelism is not supported."
    )
  # SimpleFSDP wraps tok_embeddings with a read-only `weight` property, which
  # is incompatible with Decoder.init_states re-tying tok_embeddings.weight
  # (same check as the eager TPU path in examples/gemma4/infra/parallelize.py).
  if getattr(model, "enable_weight_tying", False):
    raise ValueError(
        "SimpleFSDP requires an untied Gemma4 model; build it with "
        "model_registry(..., enable_weight_tying=False)."
    )

  annotate_module_fqns(model)

  if parallelism.spmd_backend == "spmd_types":
    model.parallelize(parallel_dims)

  configure_moe_aux_loss_reduction(model, parallel_dims)

  # Apply simple_fsdp unconditionally. The `fsdp` mesh always exists with a
  # real backend (see ParallelDims._mesh_exist), even at degree 1, so that
  # MixedPrecisionPolicy's param_dtype cast still applies in single-chip runs.
  configure_gemma4_simple_fsdp_shard_dims(model, parallel_dims)
  model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)

  return apply_compile(
      model,
      compile_config=compile_config,
      parallelism=parallelism,
      parallel_dims=parallel_dims,
      dump_folder=dump_folder,
  )


def model_registry(flavor: str) -> ModelSpec:
  """Returns the GraphTrainer ``ModelSpec`` for a TPU Gemma 4 flavor.

  ``to_graph_trainer_config`` only takes the model class, ``parallelize_fn``
  and ``pipelining_fn`` from this spec; the field values (weight tying,
  converters, sharding) come from the recipe's own model config.

  Args:
    flavor: a flavor of ``experiments.tpu.examples.gemma4.model_registry``
      (``testmodel``, ``testmodel_moe``, ``debugmodel``, ``debugmodel_moe``,
      ``31B``, ``26B-A4B``).

  Returns:
    The TPU Gemma 4 spec with the graph-trainer ``parallelize_gemma4`` and no
    ``pipelining_fn``.
  """
  return dataclasses.replace(
      tpu_gemma4.model_registry(flavor, enable_weight_tying=False),
      parallelize_fn=parallelize_gemma4,
      pipelining_fn=None,
  )


__all__ = [
    "model_registry",
    "parallelize_gemma4",
]
