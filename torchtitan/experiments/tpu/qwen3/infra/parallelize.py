"""Parallelization utilities for Qwen3 models on TPU."""

import dataclasses
import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.models.qwen3 import parallelize as dtensor_parallelize

from torchtitan.experiments.graph_trainer.common_utils import (
    apply_simple_fsdp,
)
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import (
    configure_simple_fsdp_shard_dims,
)
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.tools.logging import logger


def configure_qwen_simple_fsdp_shard_dims(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
) -> None:
  """Use divisible hidden axes for uneven Qwen parameters under SimpleFSDP."""
  fsdp_degree = parallel_dims.dp_shard * parallel_dims.cp
  if fsdp_degree <= 1:
    return

  hidden_axis_modules = configure_simple_fsdp_shard_dims(model, parallel_dims)
  stacks = any(hasattr(m, "plan_shard_dims") for m in model.modules())

  # With EP, apply_simple_fsdp separately wraps each expert bank on the EDP
  # mesh and owns its shard-axis selection.
  if not parallel_dims.ep_enabled:
    qwen_expert_modules = [
        (module_fqn, module)
        for module_fqn, module in model.named_modules()
        if isinstance(module, TpuGroupedExperts)
    ]
    for module_fqn, module in qwen_expert_modules:
      if module.num_experts % fsdp_degree == 0:
        continue

      parameters = list(module.named_parameters(recurse=False))
      if not parameters:
        if stacks:
          continue  # Stacking took them; the holder above shards them instead.
        raise ValueError(f"{module_fqn} has no direct parameters to FSDP-shard")
      incompatible = [
          f"{name}{tuple(parameter.shape)}"
          for name, parameter in parameters
          if (
              parameter.ndim != 3
              or parameter.shape[0] != module.num_experts
              or parameter.shape[1] % fsdp_degree != 0
          )
      ]
      if incompatible:
        raise ValueError(
            f"Cannot FSDP-shard {module_fqn} on Qwen's hidden dimension 1 "
            f"over {fsdp_degree} ranks; incompatible expert parameters: "
            f"{incompatible}"
        )
      # See the documented apply_simple_fsdp hook above.
      setattr(module, "_fsdp_shard_dim", 1)
      hidden_axis_modules.append(module_fqn)

  if hidden_axis_modules:
    logger.info(
        "Using SimpleFSDP dimension 1 for %d uneven Qwen modules",
        len(hidden_axis_modules),
    )


def parallelize_qwen3(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config: "torchtitan.distributed.activation_checkpoint.ActivationCheckpointing.Config",
    dump_folder: str,
    skip_dp: bool = False,
    tpu_compile_mode: str = "layer",
    **kwargs,
) -> nn.Module:
  """Parallelizes the Qwen3 model, optionally applying Simple FSDP.

  This wrapper delegates to the upstream Qwen3 parallelize function. If
  `--parallelism.use_simple_fsdp` is set, it skips upstream Data Parallelism
  and instead applies the TPU-specific Simple FSDP implementation.
  """
  compile_mode = tpu_compile_mode
  effective_compile_config = compile_config
  if effective_compile_config.enable and hasattr(model, "layers"):
    from torchtitan.models.common.moe import MoE

    for layer in model.layers.values():
      has_moe = any(isinstance(m, MoE) for m in layer.modules())
      if not has_moe:
        apply_spmd_safe(layer)

  if compile_mode in ("fullgraph", "aot_fx_trace") and compile_config.enable:
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  if parallelism.use_simple_fsdp:
    # Apply all other parallelisms but skip data parallelism
    dtensor_parallelize.parallelize_qwen3(
          model,  # pyrefly: ignore[bad-argument-type]
          ac_config=ac_config,
          compile_config=effective_compile_config,
          dump_folder=dump_folder,
          parallel_dims=parallel_dims,
          parallelism=parallelism,
          skip_dp=True,
          training=training,
      )

    # Configure expert sharding after upstream parallelization has installed
    # any wrappers, but before SimpleFSDP selects parameter shard dimensions.
    configure_qwen_simple_fsdp_shard_dims(model, parallel_dims)
    model = apply_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=training,
    )
    # Creates process groups, so every rank runs it here, right after sharding.
    for holder in model.modules():
      if hasattr(holder, "shard_split_stacks"):
        holder.shard_split_stacks()
  else:
    dtensor_parallelize.parallelize_qwen3(
          model,  # pyrefly: ignore[bad-argument-type]
          ac_config=ac_config,
          compile_config=effective_compile_config,
          dump_folder=dump_folder,
          parallel_dims=parallel_dims,
          parallelism=parallelism,
          skip_dp=skip_dp,
          training=training,
      )
  if compile_mode == "fullgraph":
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")
  return model
