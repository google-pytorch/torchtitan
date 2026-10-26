"""Parallelization for Muse Glimmer models on TPU.

Upstream ``torchtitan.models.muse_glimmer.parallelize.parallelize_muse_glimmer``
hard-requires ``parallelism.spmd_backend == "spmd_types"``, which the Google3
fork deliberately does not use (it pins ``partial_dtensor``; see
``UPSTREAM_IMPORT_NOTES.md`` D1/D7). This module is the TPU-owned replacement,
mirroring upstream ``models/qwen3/parallelize.py``'s ``partial_dtensor`` path:

  TP (``Module.parallelize``) -> AC -> per-block compile -> SimpleFSDP / FSDP2.

Scope: text-only flavors with FSDP/HSDP and TP. CP, PP and the multimodal
(vision) flavors are rejected explicitly until they are validated on TPU.
"""

import dataclasses

import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.graph_trainer.fsdp_shard_dims import configure_fsdp_shard_dims
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.tools.logging import logger

# Muse Glimmer's ``tok_embeddings`` is an ``EmbeddingWithNorm``; its weight
# lives on ``tok_embeddings.embedding``, which the generic
# ``configure_fsdp_shard_dims`` (it looks at ``tok_embeddings.weight``)
# does not see.
_EMBEDDING_FQN = "tok_embeddings.embedding"


def configure_muse_glimmer_simple_fsdp_shard_dims(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
) -> list[str]:
  """Selects SimpleFSDP dim 1 for vocab-sized weights with an uneven dim 0.

  Muse Glimmer's vocab (202048) is not divisible by large FSDP degrees (e.g. 128
  on tpu7x-4x4x4). ``lm_head`` (and any layer stacks) are handled by the shared
  ``configure_fsdp_shard_dims``; this adds the same rule for the
  embedding weight nested inside ``EmbeddingWithNorm``.

  Args:
    model: The Muse Glimmer model (or a pipeline stage of it).
    parallel_dims: Parallel dimensions; the FSDP degree is ``dp_shard``.

  Returns:
    FQNs of the modules switched to dim-1 sharding.

  Raises:
    NotImplementedError: If context parallelism is enabled.
    ValueError: If neither dim 0 nor dim 1 is divisible by the FSDP degree.
  """
  if parallel_dims.cp_enabled:
    raise NotImplementedError(
        "Context parallelism is not supported for Muse Glimmer on TPU yet."
    )
  fsdp_degree = parallel_dims.dp_shard
  if fsdp_degree <= 1:
    return []

  hidden_axis_modules: list[str] = []
  try:
    embedding = model.get_submodule(_EMBEDDING_FQN)
  except AttributeError:
    embedding = None  # Absent on this pipeline stage.
  if embedding is not None:
    weight = embedding.weight
    if weight.shape[0] % fsdp_degree != 0:
      # Same rule as the shared helper: dim 1 must divide, or be narrower than
      # the FSDP degree (SimpleFSDP then replicates it).
      if weight.ndim != 2 or (
          weight.shape[1] % fsdp_degree != 0 and weight.shape[1] >= fsdp_degree
      ):
        raise ValueError(
            f"Cannot FSDP-shard {_EMBEDDING_FQN}.weight with shape "
            f"{tuple(weight.shape)} over {fsdp_degree} ranks"
        )
      # apply_simple_fsdp's documented per-submodule shard-axis hook.
      setattr(embedding, "_fsdp_shard_dim", 1)
      hidden_axis_modules.append(_EMBEDDING_FQN)

  hidden_axis_modules.extend(
      configure_fsdp_shard_dims(model, parallel_dims)
  )

  if hidden_axis_modules:
    logger.info(
        "Using SimpleFSDP dimension 1 for uneven Muse Glimmer modules: %s",
        hidden_axis_modules,
    )
  return hidden_axis_modules


def _validate_supported(
    model: nn.Module, parallel_dims: torchtitan.distributed.ParallelDims
) -> None:
  if getattr(model, "vision_encoder", None) is not None:
    raise NotImplementedError(
        "Multimodal Muse Glimmer flavors are not supported on TPU yet; use a"
        " text-only flavor."
    )
  if parallel_dims.cp_enabled:
    raise NotImplementedError(
        "Context parallelism is not supported for Muse Glimmer on TPU yet."
    )
  if parallel_dims.pp_enabled:
    raise NotImplementedError(
        "Pipeline parallelism is not supported for Muse Glimmer on TPU yet."
    )
  if parallel_dims.tp_enabled and hasattr(model, "layers"):
    for layer_id, layer in model.layers.items():
      n_kv_heads = layer.attention.n_kv_heads
      if n_kv_heads % parallel_dims.tp != 0:
        raise ValueError(
            f"Layer {layer_id}: n_kv_heads ({n_kv_heads}) must be divisible by"
            f" the TP degree ({parallel_dims.tp})."
        )


def parallelize_muse_glimmer(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config: "torchtitan.distributed.activation_checkpoint.ActivationCheckpointingConfig | None",
    dump_folder: str,
    skip_dp: bool = False,
    tpu_compile_mode: str = "layer",
    **kwargs,
) -> nn.Module:
  """Parallelizes a text-only Muse Glimmer model for TPU.

  Args:
    model: The ``MuseGlimmerModel`` to parallelize (may be on meta device).
    parallel_dims: Parallel dimensions for this run.
    training: Training config (mixed-precision dtypes, CPU offload).
    parallelism: Parallelism config; ``use_simple_fsdp`` selects SimpleFSDP
      instead of FSDP2.
    compile_config: ``torch.compile`` config.
    ac_config: Activation checkpointing config, or None.
    dump_folder: Dump folder for AC debugging artifacts.
    skip_dp: Skip data parallelism (e.g. for inference).
    tpu_compile_mode: One of ``layer``, ``fullgraph`` or ``aot_fx_trace``.
    **kwargs: Ignored; accepted for signature compatibility with train.py.

  Returns:
    The parallelized model.
  """
  del kwargs  # Unused.
  _validate_supported(model, parallel_dims)

  effective_compile_config = compile_config
  if compile_config.enable and hasattr(model, "layers"):
    for layer in model.layers.values():
      apply_spmd_safe(layer)
  if tpu_compile_mode in ("fullgraph", "aot_fx_trace") and compile_config.enable:
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  # Under partial_dtensor, Module.parallelize is only needed for TP; it applies
  # the sharding configs set by MuseGlimmerModel.Config.update_from_config.
  if parallel_dims.tp_enabled:
    model.parallelize(parallel_dims)

  if ac_config is not None:
    ac_config.build(dump_folder=dump_folder).apply(model)

  if (
      effective_compile_config.enable
      and "model" in effective_compile_config.components
  ):
    apply_compile(
        model,
        compile_config=effective_compile_config,
        parallel_dims=parallel_dims,
    )

  if not skip_dp and (
      parallel_dims.dp_shard_enabled or parallel_dims.dp_replicate_enabled
  ):
    if parallelism.use_simple_fsdp:
      configure_muse_glimmer_simple_fsdp_shard_dims(model, parallel_dims)
      model = apply_simple_fsdp(
          model, parallel_dims=parallel_dims, training=training
      )
    else:
      dp_mesh_names = (
          ["dp_replicate", "fsdp"]
          if parallel_dims.dp_replicate_enabled
          else ["fsdp"]
      )
      apply_fsdp_to_decoder(
          model,
          parallel_dims.get_mesh(dp_mesh_names),
          param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
              training.mixed_precision_param
          ],
          reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
              training.mixed_precision_reduce
          ],
          pp_enabled=parallel_dims.pp_enabled,
          cpu_offload=training.enable_cpu_offload,
          reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
          dp_mesh_dims=None,
          enable_symm_mem=parallelism.enable_fsdp_symm_mem,
      )

  if tpu_compile_mode == "fullgraph" and compile_config.enable:
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")
  return model
