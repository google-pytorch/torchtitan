"""Parallelization for Qwen3.5 models on TPU."""

import dataclasses

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy
import torchtitan.config
import torchtitan.distributed
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import configure_simple_fsdp_shard_dims

# Before `torchtitan.models.qwen3_5`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu.qwen3_5 import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.models.common.moe import MoE
from torchtitan.models.qwen3_5 import parallelize as dtensor_parallelize


def apply_qwen3_5_simple_fsdp(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
) -> nn.Module:
  """Applies SimpleFSDP to Qwen3.5, wrapping routed experts on ``edp_mesh`` under EP.

  Upstream ``apply_simple_fsdp`` checks ``isinstance(block, TransformerBlock)``
  before wrapping ``moe.routed_experts.inner_experts`` on ``edp_mesh``. Since
  ``Qwen35TransformerBlock`` subclasses ``Module`` (and may be wrapped in a
  ``CheckpointWrapper``), we wrap every ``MoE`` submodule's ``inner_experts`` on
  ``edp_mesh`` first and then delegate the remaining parameters to
  ``apply_simple_fsdp``.
  """
  if parallel_dims.ep_enabled:
    use_spmd_types = get_spmd_backend() == "spmd_types"
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
    non_dp_mesh = (
        parallel_dims.get_optional_mesh("ep") if use_spmd_types else None
    )

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

  Delegates to upstream ``parallelize_qwen3_5``. With
  ``--parallelism.use_simple_fsdp`` it skips upstream data parallelism and
  applies SimpleFSDP instead, as ``tpu/qwen3`` does.

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

  effective_compile_config = compile_config
  if compile_config.enable and hasattr(model, "layers"):
    # TPU: torch_tpu `spmd_safe` for compiled non-MoE blocks, as in tpu/qwen3.
    for layer in model.layers.values():
      if not any(isinstance(m, MoE) for m in layer.modules()):
        apply_spmd_safe(layer)
  if (
      tpu_compile_mode in ("fullgraph", "aot_fx_trace")
      and compile_config.enable
  ):
    # TPU: whole-model compile modes replace upstream's per-block compile.
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  dtensor_parallelize.parallelize_qwen3_5(
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
