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
from torchtitan.experiments.tpu.activation_checkpoint import tpu_selective_ac_config
from torchtitan.experiments.tpu.graph_trainer.simple_fsdp import (
    install_tpu_replicate_computation,
)
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import configure_simple_fsdp_shard_dims
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
      # W2's dim 1 (F=1536) gives a 3-row stripe at 512 ranks, which no TPU
      # tile fits. Its trailing D axis gives 8 rows, but the gather then stores
      # W2 as [E, D, F], so the grouped matmul reads it as a transposed RHS
      # instead of paying a full relayout copy.
      # TODO: remove the trailing-W2/transposed-RHS path if GraphSplitter
      # aligns with Scan's flattened dim-0 expert sharding, or if FlexShard
      # provides tile-aligned expert shards.
      use_trailing_w2 = (
          module.use_kernel_native_weight_layout
          and any(
              name == "w2_EFD"
              and parameter.ndim == 3
              and parameter.shape[2] % fsdp_degree == 0
              for name, parameter in parameters
          )
      )
      shard_dims: dict[str, int] = {}
      incompatible: list[str] = []
      for name, parameter in parameters:
        if parameter.ndim != 3 or parameter.shape[0] != module.num_experts:
          incompatible.append(f"{name}{tuple(parameter.shape)}")
          continue
        if use_trailing_w2 and name == "w2_EFD":
          shard_dims[name] = 2
        elif parameter.shape[1] % fsdp_degree == 0:
          shard_dims[name] = 1
        else:
          incompatible.append(f"{name}{tuple(parameter.shape)}")
      if incompatible:
        raise ValueError(
            f"Cannot FSDP-shard {module_fqn} over {fsdp_degree} ranks; "
            f"incompatible expert parameters: "
            f"{incompatible}"
        )
      # See the documented apply_simple_fsdp hook above.
      if use_trailing_w2:
        setattr(module, "_fsdp_shard_dim", shard_dims)
        module._use_transposed_w2_rhs = True
      else:
        setattr(module, "_fsdp_shard_dim", 1)
      hidden_axis_modules.append(module_fqn)

  if hidden_axis_modules:
    logger.info(
        "Using non-leading SimpleFSDP dimensions for %d uneven Qwen modules",
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
  if parallel_dims.cp_enabled:
    # Ulysses CP's all-to-alls are collectives that core SelectiveAC saves,
    # which torch-tpu cannot compile; see TPUSelectiveAC.
    ac_config = tpu_selective_ac_config(ac_config)
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
      if hasattr(holder, "build_split_meshes"):
        holder.build_split_meshes()
    install_tpu_replicate_computation(model)
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
