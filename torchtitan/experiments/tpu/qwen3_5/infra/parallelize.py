"""Parallelization for Qwen3.5 models on TPU."""

import dataclasses

import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.graph_trainer.fsdp_shard_dims import configure_fsdp_shard_dims

# Before `torchtitan.models.qwen3_5`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu.qwen3_5 import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.models.common.moe import MoE
from torchtitan.models.qwen3_5 import parallelize as dtensor_parallelize


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
    configure_fsdp_shard_dims(model, parallel_dims)
    model = apply_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=training,
    )

  if tpu_compile_mode == "fullgraph" and compile_config.enable:
    # TPU: compile the whole sharded model once with the torch_tpu backend.
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")
  return model
