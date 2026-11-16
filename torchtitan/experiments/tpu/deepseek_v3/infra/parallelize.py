"""Parallelization for DeepSeek-V3 on TPU (eager / per-block compile path)."""

import dataclasses

import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.distributed.spmd_utils import apply_spmd_safe
from torchtitan.experiments.tpu.graph_trainer.simple_fsdp import (
    install_tpu_replicate_computation,
)
from torchtitan.experiments.tpu.qwen3_sharding import (
    configure_qwen_simple_fsdp_shard_dims,
)
from torchtitan.models.deepseek_v3 import parallelize as dtensor_parallelize
from torchtitan.tools.logging import logger


def configure_deepseek_v3_simple_fsdp_shard_dims(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
) -> None:
  """Pick SimpleFSDP shard axes that divide the FSDP degree.

  On top of the vocab and expert-bank handling in
  ``configure_qwen_simple_fsdp_shard_dims``, moves weight-only modules whose
  dim 0 does not divide the FSDP degree to dim 1. In DeepSeek-V3 these are
  ``attention.wkv_a`` (576 = kv_lora_rank 512 + rope dim 64 rows) and
  ``moe.router.gate`` (one row per expert, 160 in 236B). Left on dim 0 they
  would be sharded unevenly (padded gathers) or, once dim 0 is smaller than
  the degree, silently replicated.

  Args:
    model: The model, before ``apply_simple_fsdp``.
    parallel_dims: Parallel dimensions; FSDP degree is ``dp_shard * cp``.
  """
  fsdp_degree = parallel_dims.dp_shard * parallel_dims.cp
  if fsdp_degree <= 1:
    return
  configure_qwen_simple_fsdp_shard_dims(model, parallel_dims)

  moved = []
  for fqn, module in model.named_modules():
    if hasattr(module, "_fsdp_shard_dim"):
      continue
    params = dict(module.named_parameters(recurse=False))
    weight = params.get("weight")
    if set(params) != {"weight"} or weight.ndim != 2:
      continue
    if weight.shape[0] % fsdp_degree == 0:
      continue
    if weight.shape[1] % fsdp_degree != 0:
      logger.warning(
          "%s.weight %s divides the FSDP degree %d on neither axis; SimpleFSDP"
          " will shard it unevenly on dim 0.",
          fqn,
          tuple(weight.shape),
          fsdp_degree,
      )
      continue
    module._fsdp_shard_dim = 1
    moved.append(fqn)
  if moved:
    logger.info(
        "Sharding %d uneven DeepSeek-V3 weights on dim 1 (e.g. %s)",
        len(moved),
        moved[0],
    )


def parallelize_deepseekv3(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config: "torchtitan.distributed.activation_checkpoint.ActivationCheckpointingConfig",
    dump_folder: str,
    skip_dp: bool = False,
    tpu_compile_mode: str = "layer",
    **kwargs,
) -> nn.Module:
  """Parallelizes DeepSeek-V3 for TPU.

  Delegates to the upstream parallelizer. With ``parallelism.use_simple_fsdp``
  it skips upstream data parallelism and applies SimpleFSDP with the TPU shard
  axes and collectives instead, as the Qwen3 TPU parallelizer does.

  Args:
    model: The ``DeepSeekV3Model`` to parallelize (may be on meta device).
    parallel_dims: Parallel dimensions for this run.
    training: Training config (mixed-precision dtypes).
    parallelism: Parallelism config; ``use_simple_fsdp`` selects SimpleFSDP
      instead of FSDP2.
    compile_config: ``torch.compile`` config.
    ac_config: Activation checkpointing config.
    dump_folder: Dump folder for AC debugging artifacts.
    skip_dp: Skip data parallelism.
    tpu_compile_mode: One of ``layer``, ``fullgraph`` or ``aot_fx_trace``.
    **kwargs: Ignored; accepted for signature compatibility with train.py.

  Returns:
    The parallelized model.
  """
  del kwargs  # Unused.
  effective_compile_config = compile_config
  if compile_config.enable and hasattr(model, "layers"):
    from torchtitan.models.common.moe import MoE  # pylint: disable=g-import-not-at-top

    for layer in model.layers.values():
      if not any(isinstance(m, MoE) for m in layer.modules()):
        apply_spmd_safe(layer)
  if tpu_compile_mode in ("fullgraph", "aot_fx_trace") and compile_config.enable:
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  use_simple_fsdp = parallelism.use_simple_fsdp and not skip_dp
  dtensor_parallelize.parallelize_deepseekv3(
      model,
      parallel_dims=parallel_dims,
      training=training,
      parallelism=parallelism,
      compile_config=effective_compile_config,
      ac_config=ac_config,
      dump_folder=dump_folder,
      skip_dp=skip_dp or use_simple_fsdp,
  )
  if use_simple_fsdp:
    # After upstream has installed its wrappers, before SimpleFSDP picks the
    # parameter shard axes.
    configure_deepseek_v3_simple_fsdp_shard_dims(model, parallel_dims)
    model = apply_simple_fsdp(
        model, parallel_dims=parallel_dims, training=training
    )
    # Creates process groups, so every rank runs it here, right after sharding.
    for holder in model.modules():
      if hasattr(holder, "build_split_meshes"):
        holder.build_split_meshes()
    install_tpu_replicate_computation(model)

  if tpu_compile_mode == "fullgraph" and compile_config.enable:
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")
  return model
