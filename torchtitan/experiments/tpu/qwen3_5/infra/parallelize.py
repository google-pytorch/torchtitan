"""Parallelization for Qwen3.5 models on TPU."""

import dataclasses
import logging

import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torch.distributed.fsdp import MixedPrecisionPolicy
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder, resolve_fsdp_mesh, resolve_sparse_fsdp_mesh
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.experiments.tpu.components.activation_checkpoint import tpu_selective_ac_config
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import configure_simple_fsdp_shard_dims
logger = logging.getLogger(__name__)
from torchtitan.tools.utils import device_type

# Before `torchtitan.models.qwen3_5`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu.qwen3_5 import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.experiments.tpu.distributed.spmd_utils import apply_spmd_safe
from torchtitan.models.common.moe import configure_moe_aux_loss_reduction, MoE
from torchtitan.models.qwen3_5 import parallelize as dtensor_parallelize


def apply_qwen3_5_simple_fsdp(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
) -> nn.Module:
  """Applies SimpleFSDP to Qwen3.5, using ``edp_mesh`` for routed experts.

  Upstream ``apply_simple_fsdp`` checks ``isinstance(block, TransformerBlock)``
  before wrapping ``moe.routed_experts.inner_experts`` on ``edp_mesh``. Since
  ``Qwen35TransformerBlock`` subclasses ``Module`` (and may be wrapped in a
  ``CheckpointWrapper``), we wrap every ``MoE`` submodule's ``inner_experts`` on
  ``edp_mesh`` first and then delegate the remaining parameters to
  ``apply_simple_fsdp``.

  TODO: Upstream pytorch/torchtitan already makes ``Qwen35TransformerBlock``
  subclass ``TransformerBlock``, and its ``apply_simple_fsdp`` walks
  ``model.modules()`` (so it also sees checkpoint-wrapped blocks). Once that
  reaches this fork in an upstream sync, delete this wrapper and call
  ``apply_simple_fsdp`` directly.
  """
  if parallel_dims.ep_enabled:
    if parallel_dims.dp_replicate_enabled:
      dp_mode = (
          "hybrid_shard"
          if (parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled)
          else "replicate"
      )
      edp_mesh_names = ["dp_replicate", "efsdp"]
    else:
      dp_mode = "fully_shard"
      edp_mesh_names = ["efsdp"]
    edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
    assert edp_mesh is not None
    efsdp_degree = edp_mesh["efsdp"].size()
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            training.mixed_precision_param
        ],
        reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            training.mixed_precision_reduce
        ],
    )
    non_dp_mesh = parallel_dims.get_optional_mesh("ep")

    for module_fqn, moe in model.named_modules():
      if not isinstance(moe, MoE):
        continue
      inner_experts = moe.routed_experts.inner_experts
      if inner_experts.num_experts % (efsdp_degree * parallel_dims.ep) == 0:
        experts_shard_dim = 0
      else:
        incompatible = [
            f"{name}{tuple(parameter.shape)}"
            for name, parameter in inner_experts.named_parameters(recurse=False)
            if parameter.ndim != 3 or parameter.shape[1] % efsdp_degree != 0
        ]
        if incompatible:
          raise ValueError(
              f"Cannot FSDP-shard {module_fqn}.routed_experts.inner_experts on "
              f"dimension 1 over {efsdp_degree} efsdp ranks; incompatible "
              f"expert parameters: {incompatible}"
          )
        experts_shard_dim = 1

      moe.routed_experts.inner_experts = data_parallel(
          inner_experts,
          edp_mesh,
          dp_mode,
          mp_policy=mp_policy,
          shard_dim=experts_shard_dim,
          non_dp_mesh=non_dp_mesh,
      )

  return apply_simple_fsdp(
      model,
      parallel_dims=parallel_dims,
      training=training,
  )


def _parallelize_text_qwen3_5_with_cp(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config: "torchtitan.distributed.activation_checkpoint.ActivationCheckpointingConfig | None",
    dump_folder: str,
    skip_dp: bool,
) -> nn.Module:
  """Upstream ``parallelize_qwen3_5`` for a text-only model, allowing CP.

  Upstream rejects context parallelism because its GatedDeltaNet needs the
  whole sequence; the TPU CP recipes run it head-parallel instead
  (``cp_gated_delta``) and validate that in ``validate_tpu_context_parallel``.
  The steps are upstream's: sharding annotations, activation checkpointing,
  per-block compile and FSDP2 (on the ``(dp_shard, cp)`` mesh).

  Keep in sync with the text-only path of
  ``torchtitan/models/qwen3_5/parallelize.py::parallelize_qwen3_5``.
  """
  if model.vision_encoder is not None:
    raise NotImplementedError(
        "Qwen3.5 context parallelism on TPU is text-only."
    )
  model.parallelize(parallel_dims)  # pyrefly: ignore[not-callable]
  # Not in upstream's text path (which rejects CP): raises for MoE auxiliary
  # losses, which would treat each CP rank's token shard as whole sequences.
  configure_moe_aux_loss_reduction(model, parallel_dims)
  if ac_config is not None:
    ac_config.build(dump_folder=dump_folder).apply(model)
  if compile_config.enable and "model" in compile_config.components:
    apply_compile(
        model, compile_config=compile_config, parallel_dims=parallel_dims
    )
  if skip_dp:
    return model
  dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
  edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
  apply_fsdp_to_decoder(
      model,  # pyrefly: ignore[bad-argument-type]
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
      symm_mem_scope=parallelism.fsdp_symm_mem_scope,
  )
  return model


def parallelize_qwen3_5(
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
  """Parallelizes a text-only Qwen3.5 model on TPU, optionally with SimpleFSDP.

  Delegates to upstream ``parallelize_qwen3_5`` (with context parallelism, to
  ``_parallelize_text_qwen3_5_with_cp``). With ``--parallelism.use_simple_fsdp``
  it skips upstream data parallelism and applies SimpleFSDP instead, as
  ``tpu/qwen3`` does.

  Args:
    model: The model to parallelize.
    parallel_dims: Parallel dimensions.
    training: Training configuration.
    parallelism: Parallelism configuration.
    compile_config: Compile configuration.
    ac_config: Activation checkpointing configuration.
    dump_folder: Folder for dumped artifacts.
    skip_dp: Whether to skip data parallelism.
    tpu_compile_mode: TPU compile mode (``layer``, ``fullgraph`` or
      ``aot_fx_trace``).
    **kwargs: Unused.

  Returns:
    The parallelized model.
  """
  del kwargs  # Unused.

  if (
      parallel_dims.cp_enabled
      and parallel_dims.dp_shard > 1
      and not compile_config.enable
      and device_type == "tpu"
  ):
    logger.warning(
        "Eager Qwen3.5 context parallelism with dp_shard > 1 can produce"
        " wrong or non-finite gradients on the current torch_tpu build; run"
        " it compiled (--compile.enable, or the TpuGraphTrainer recipes)."
    )
  if parallel_dims.cp_enabled:
    # TPU: the CP all-to-alls are collectives that core SelectiveAC saves,
    # which torch-tpu cannot compile; see TPUSelectiveAC (as in tpu/qwen3).
    ac_config = tpu_selective_ac_config(ac_config)
  effective_compile_config = compile_config
  if compile_config.enable and hasattr(model, "layers"):
    # TPU: torch_tpu `spmd_safe` for compiled blocks. Per-block compile also
    # wraps MoE blocks unless EP is on (as in gemma4): without it,
    # SplitCompiler splits a block at every SimpleFSDP/CP collective. The
    # whole-model modes keep MoE blocks unwrapped, including TpuGraphTrainer,
    # which drops `tpu_compile_mode` and sets `compile.mode="aot_fx_trace"`.
    per_block = tpu_compile_mode not in ("fullgraph", "aot_fx_trace") and (
        getattr(compile_config, "mode", None) != "aot_fx_trace"
    )
    for layer in model.layers.values():
      has_moe = any(isinstance(m, MoE) for m in layer.modules())
      if not has_moe or (per_block and not parallel_dims.ep_enabled):
        apply_spmd_safe(layer)
  if (
      tpu_compile_mode in ("fullgraph", "aot_fx_trace")
      and compile_config.enable
  ):
    # TPU: whole-model compile modes replace upstream's per-block compile.
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  parallelize = (
      _parallelize_text_qwen3_5_with_cp
      if parallel_dims.cp_enabled
      else dtensor_parallelize.parallelize_qwen3_5
  )
  parallelize(
      model,  # pyrefly: ignore[bad-argument-type]
      ac_config=ac_config,  # pyrefly: ignore[bad-argument-type]
      compile_config=effective_compile_config,
      dump_folder=dump_folder,
      parallel_dims=parallel_dims,
      parallelism=parallelism,
      # TPU: SimpleFSDP (below) replaces upstream's FSDP2.
      skip_dp=skip_dp or parallelism.use_simple_fsdp,
      training=training,
  )
  if parallelism.use_simple_fsdp:
    # TPU: SimpleFSDP instead of FSDP2, on shard dims that divide the mesh.
    configure_simple_fsdp_shard_dims(model, parallel_dims)
    model = apply_qwen3_5_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=training,
    )

  if tpu_compile_mode == "fullgraph" and compile_config.enable:
    # TPU: compile the whole sharded model once with the torch_tpu backend.
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")
  return model
