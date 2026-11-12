# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Parallelization strategies for Mamba-2 architecture on TPU."""

from torchtitan.config import CompileConfig
from torchtitan.config import ParallelismConfig
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.config import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.fsdp import resolve_fsdp_mesh
from .model import Mamba2LM


def parallelize_mamba2(
    model: Mamba2LM,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    skip_dp: bool = False,
):
  """Applies FSDP, compilation, and activation checkpointing to Mamba-2."""
  if parallel_dims.tp_enabled:
    raise NotImplementedError(
        "Tensor parallelism is not yet supported for Mamba-2."
    )
  if parallel_dims.cp_enabled:
    raise NotImplementedError(
        "Context parallelism is not yet supported for Mamba-2."
    )

  if ac_config is not None:
    ac_config.build(dump_folder=dump_folder).apply(model)

  model_compile_enabled = (
      compile_config.enable and "model" in compile_config.components
  )
  if model_compile_enabled:
    apply_compile(model, compile_config)

  if skip_dp:
    return model

  dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)

  apply_fsdp_to_decoder(
      model,
      dp_mesh,
      param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
      reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
      pp_enabled=parallel_dims.pp_enabled,
      cpu_offload=training.enable_cpu_offload,
      reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
      dp_mesh_dims=dp_mesh_dims,
      enable_symm_mem=parallelism.enable_fsdp_symm_mem,
  )

  return model
