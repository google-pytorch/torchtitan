"""Parallelization utilities for Gemma4 models on TPU."""

import dataclasses
import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.compile import _maybe_regional_inductor_backend
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    apply_simple_fsdp,
)
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.models.common.moe import (
    GroupedExperts,
    MoE,
    configure_moe_aux_loss_reduction,
)
from torchtitan.tools.logging import logger


class _Gemma4SelectiveAC(SelectiveAC):
  """SelectiveAC that avoids saving un-waited AsyncCollectiveTensors across forward/backward."""

  def get_save_ops(self) -> set:
    return super().get_save_ops() - {
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        torch.ops._c10d_functional.all_to_all_single.default,
    }


def _apply_ac_gemma4(
    model: nn.Module,
    ac_config: "torchtitan.distributed.activation_checkpoint.ActivationCheckpointing.Config",
    dump_folder: str,
) -> None:
  if isinstance(ac_config, SelectiveAC.Config):
    _Gemma4SelectiveAC(ac_config, dump_folder=dump_folder).apply(model)
  else:
    ac_config.build(dump_folder=dump_folder).apply(model)


def _apply_compile_gemma4(
    model: nn.Module, compile_config: torchtitan.config.CompileConfig
) -> None:
  """Apply torch.compile to each Gemma4TransformerBlock with dynamic=False.

  Gemma4 mixes sliding-window layers (smaller head_dim) and global attention
  layers (larger head_dim) using the same Gemma4TransformerBlock class. Leaving
  `dynamic` at None causes TorchDynamo's automatic dynamic shapes to mark the
  differing head_dim / rope_cache dimensions as SymInt on the first global
  layer, which is unsupported by the TPU backend.
  """
  torch._dynamo.config.capture_scalar_outputs = True
  torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = (
      True  # pyrefly: ignore [bad-assignment]
  )
  backend = _maybe_regional_inductor_backend(model, compile_config.backend)
  # pyrefly: ignore [missing-attribute]
  for _, transformer_block in model.layers.named_children():
    transformer_block.compile(backend=backend, fullgraph=True, dynamic=False)
  logger.info(
      "Compiling each Gemma4TransformerBlock with torch.compile (dynamic=False)"
  )


def _parallelize_gemma4(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config: "torchtitan.distributed.activation_checkpoint.ActivationCheckpointing.Config | None",
    dump_folder: str,
    skip_dp: bool = False,
) -> nn.Module:
  """Apply DTensor/SPMD TP, EP, AC, compile, and FSDP to Gemma4."""
  if parallel_dims.cp_enabled:
    raise NotImplementedError(
        "Context parallelism is not supported for Gemma4 on TPU yet."
    )
  model_compile_enabled = (
      compile_config.enable and "model" in compile_config.components
  )

  if (
      parallelism.spmd_backend == "spmd_types"
      or parallel_dims.tp_enabled
      or parallel_dims.ep_enabled
  ):
    model.parallelize(parallel_dims)

  configure_moe_aux_loss_reduction(model, parallel_dims)

  if ac_config is not None:
    _apply_ac_gemma4(model, ac_config, dump_folder=dump_folder)

  if model_compile_enabled:
    _apply_compile_gemma4(model, compile_config)

  if skip_dp:
    return model

  if parallelism.spmd_backend == "spmd_types":
    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
  else:
    dp_mesh_names = (
        ["dp_replicate", "fsdp"]
        if parallel_dims.dp_replicate_enabled
        else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
    dp_mesh_dims = None
    edp_mesh = None
    edp_mesh_dims = None
    if parallel_dims.ep_enabled:
      edp_mesh_names = (
          ["dp_replicate", "efsdp"]
          if parallel_dims.dp_replicate_enabled
          else ["efsdp"]
      )
      edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

  apply_fsdp_to_decoder(
      model,
      dp_mesh,
      param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
          training.mixed_precision_param
      ],
      reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
          training.mixed_precision_reduce
      ],
      pp_enabled=parallel_dims.pp_enabled,
      cpu_offload=training.enable_cpu_offload,
      reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
      ep_degree=parallel_dims.ep,
      edp_mesh=edp_mesh,
      dp_mesh_dims=dp_mesh_dims,
      edp_mesh_dims=edp_mesh_dims,
      enable_symm_mem=parallelism.enable_fsdp_symm_mem,
  )

  return model


def configure_gemma4_simple_fsdp_shard_dims(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
) -> None:
  """Use divisible hidden axes for uneven Gemma4 parameters under SimpleFSDP."""
  fsdp_degree = parallel_dims.dp_shard
  if fsdp_degree <= 1:
    return

  stacks = [m for m in model.modules() if hasattr(m, "plan_shard_dims")]
  for holder in stacks:
    holder._fsdp_shard_dim = holder.plan_shard_dims(fsdp_degree)

  hidden_axis_modules: list[str] = []
  for module_fqn in ("tok_embeddings", "lm_head"):
    module = getattr(model, module_fqn, None)
    if module is None:
      continue
    weight = module.weight
    if weight.shape[0] % fsdp_degree == 0:
      continue
    if weight.ndim != 2 or weight.shape[1] % fsdp_degree != 0:
      raise ValueError(
          f"Cannot FSDP-shard {module_fqn}.weight with shape "
          f"{tuple(weight.shape)} over {fsdp_degree} ranks"
      )
    setattr(module, "_fsdp_shard_dim", 1)
    hidden_axis_modules.append(module_fqn)

  if not parallel_dims.ep_enabled:
    expert_modules = [
        (module_fqn, module)
        for module_fqn, module in model.named_modules()
        if isinstance(module, (TpuGroupedExperts, GroupedExperts))
    ]
    for module_fqn, module in expert_modules:
      if module.num_experts % fsdp_degree == 0:
        continue

      parameters = list(module.named_parameters(recurse=False))
      if not parameters:
        if stacks:
          continue
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
            f"Cannot FSDP-shard {module_fqn} on Gemma4's hidden dimension 1 "
            f"over {fsdp_degree} ranks; incompatible expert parameters: "
            f"{incompatible}"
        )
      setattr(module, "_fsdp_shard_dim", 1)
      hidden_axis_modules.append(module_fqn)

  if hidden_axis_modules:
    logger.info(
        "Using SimpleFSDP dimension 1 for %d uneven Gemma4 modules",
        len(hidden_axis_modules),
    )


def parallelize_gemma4(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config: "torchtitan.distributed.activation_checkpoint.ActivationCheckpointing.Config | None",
    dump_folder: str,
    skip_dp: bool = False,
    tpu_compile_mode: str = "layer",
    **kwargs,
) -> nn.Module:
  """Parallelizes the Gemma4 model, optionally applying Simple FSDP."""
  if parallel_dims.cp_enabled:
    raise NotImplementedError(
        "Context parallelism is not supported for Gemma4 on TPU yet."
    )
  compile_mode = tpu_compile_mode
  effective_compile_config = compile_config
  if effective_compile_config.enable and hasattr(model, "layers"):
    for layer in model.layers.values():
      has_moe = any(isinstance(m, MoE) for m in layer.modules())
      if not (has_moe and parallel_dims.ep_enabled):
        apply_spmd_safe(layer)

  if compile_mode in ("fullgraph", "aot_fx_trace") and compile_config.enable:
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  if parallelism.use_simple_fsdp:
    # SimpleFSDP wraps tok_embeddings with a read-only `weight` property, which
    # is incompatible with Decoder.init_states re-tying tok_embeddings.weight.
    if getattr(model, "enable_weight_tying", False):
      raise ValueError(
          "SimpleFSDP requires an untied Gemma4 model; build it with "
          "model_registry(..., enable_weight_tying=False)."
      )

    _parallelize_gemma4(
        model,
        ac_config=None,
        compile_config=dataclasses.replace(
            effective_compile_config, enable=False
        ),
        dump_folder=dump_folder,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        skip_dp=True,
        training=training,
    )

    configure_gemma4_simple_fsdp_shard_dims(model, parallel_dims)
    model = apply_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=training,
    )
    if ac_config is not None:
      _apply_ac_gemma4(model, ac_config, dump_folder=dump_folder)
    if (
        effective_compile_config.enable
        and "model" in effective_compile_config.components
    ):
      _apply_compile_gemma4(model, effective_compile_config)
  else:
    _parallelize_gemma4(
        model,
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
