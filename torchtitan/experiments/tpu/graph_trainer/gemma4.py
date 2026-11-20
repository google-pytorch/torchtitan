"""Gemma 4 on the TPU GraphTrainer path (``TpuGraphTrainer.Config``).

GraphTrainer recipes convert a base ``TPUTrainerConfig`` via
``TpuGraphTrainer.Config.from_tpu_trainer_config`` using this module's
``model_registry``. This registry reuses ``experiments.tpu.examples.gemma4``'s
model config unchanged (with ``enable_weight_tying=False``) and swaps in
``parallelize_gemma4``, which annotates module FQNs, configures
uneven-dimension SimpleFSDP shard dims
(``configure_gemma4_simple_fsdp_shard_dims``: the 262144-token vocabulary and
the 128-expert banks switch to a divisible hidden axis when the FSDP degree
does not divide dim 0), applies SimpleFSDP, and hands the model to
GraphTrainer's ``apply_compile``.

Only SimpleFSDP / HSDP is supported here.

Scan (``--compile.use_scan_layers``) is not supported without
homogeneous-group partitioning: the sliding and global layers have different
``head_dim`` / ``n_kv_heads`` and different QKV projections (``QKVLinear``
vs ``QKVLinearKeqV``), so ``check_layers_are_stackable`` refuses them.
Graph split is the large-size path for Gemma 4.
"""

import dataclasses
from dataclasses import dataclass

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
from torchtitan.experiments.tpu.examples.gemma4.model import Gemma4Model
from torchtitan.models.common.moe import configure_moe_aux_loss_reduction


class TpuGraphTrainerGemma4Model(Gemma4Model):
  """Gemma 4 model subclass owning the TPU GraphTrainer ``parallelize`` path."""

  @dataclass(kw_only=True, slots=True)
  class Config(Gemma4Model.Config):
    pass

  def parallelize(
      self,
      *,
      parallel_dims: ParallelDims | None = None,
      parallelism_context: ParallelDims | None = None,
      training: TrainingConfig,
      parallelism: ParallelismConfig,
      compile_config: GraphTrainerCompileConfig,
      ac_config: ActivationCheckpointingConfig | None = None,
      dump_folder: str,
      skip_dp: bool = False,
  ) -> nn.Module:
    if skip_dp:
      raise ValueError("GraphTrainer models do not support skip_dp=True.")
    if parallel_dims is None:
      parallel_dims = parallelism_context
    assert parallel_dims is not None
    return parallelize_gemma4(
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
        "TPU GraphTrainer Gemma 4 supports SimpleFSDP/HSDP only; "
        "pipeline parallelism is not supported."
    )


def parallelize_gemma4(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims | None = None,
    parallelism_context: ParallelDims | None = None,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> nn.Module:
  """Applies SimpleFSDP (TPU uneven-dim policy) and graph capture to Gemma 4.

  ``ac_config`` is accepted for the ``parallelize`` protocol but unused: the
  graph trainer rematerializes its joint graph with ``compile.memory_policy``
  instead of eager AC wrappers.
  """
  if parallel_dims is None:
    parallel_dims = parallelism_context
  assert parallel_dims is not None
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
        f"{', '.join(unsupported)} parallelism is not supported."
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

  model._parallelize(parallel_dims)

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


def model_registry(flavor: str) -> TpuGraphTrainerGemma4Model.Config:
  """Returns the GraphTrainer model config for a TPU Gemma 4 flavor."""
  base = tpu_gemma4.model_registry(flavor, enable_weight_tying=False)
  return TpuGraphTrainerGemma4Model.Config(
      **{
          field.name: getattr(base, field.name)
          for field in dataclasses.fields(base)
      }
  )


model_registry._model_config_cls = TpuGraphTrainerGemma4Model.Config  # type: ignore[attr-defined]


__all__ = [
    "TpuGraphTrainerGemma4Model",
    "model_registry",
    "parallelize_gemma4",
]
