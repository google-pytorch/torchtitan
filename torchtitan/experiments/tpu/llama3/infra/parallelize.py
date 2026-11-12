"""Parallelization utilities for Llama models on TPU.

This module is used in distributed training tests configured with
`model_name="llama3_tpu".
"""

import dataclasses
import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.graph_trainer.common_utils import (
    apply_simple_fsdp,
)
from torchtitan.experiments.tpu.graph_trainer.simple_fsdp import (
    install_tpu_replicate_computation,
)
from torchtitan.experiments.tpu.distributed.spmd_utils import apply_spmd_safe
from torchtitan.models.llama3 import parallelize as dtensor_parallelize
from torchtitan.tools.logging import logger


def parallelize_llama(
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
  """Parallelizes the Llama model based on the provided configuration and applies TPU-specific workarounds.

  Args:
    model: The nn.Module to be parallelized.
    parallel_dims: Contains the world mesh and other parallelization dimensions.
    training: Training configuration.
    parallelism: Parallelism configuration.
    compile_config: Compile configuration.
    ac_config: Activation checkpointing configuration.
    dump_folder: Folder for dumped artifacts.
    skip_dp: Whether to skip data parallelism wrapping.
    tpu_compile_mode: TPU compile mode ('layer' or 'fullgraph').
  """
  # TODO b/499075866: re-enable CPU offload for TPU torchtitan run.
  if training.enable_cpu_offload:
    logger.warning(
        "CPU offload is not supported for TPU torchtitan run, setting it to False."
    )
    training.enable_cpu_offload = False

  compile_mode = tpu_compile_mode
  effective_compile_config = compile_config
  if compile_mode in ("fullgraph", "aot_fx_trace") and compile_config.enable:
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  if effective_compile_config.enable and hasattr(model, "layers"):
    for layer in model.layers.values():
      apply_spmd_safe(layer)

  if parallelism.use_simple_fsdp:
    # Apply all other parallelisms but skip data parallelism
    dtensor_parallelize.parallelize_llama(
          model,  # pyrefly: ignore[bad-argument-type]
          ac_config=ac_config,
          compile_config=effective_compile_config,
          dump_folder=dump_folder,
          parallel_dims=parallel_dims,
          parallelism=parallelism,
          training=training,
          skip_dp=True,
      )

    model = apply_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=training,
    )
    install_tpu_replicate_computation(model)
  else:
    dtensor_parallelize.parallelize_llama(
          model,  # pyrefly: ignore[bad-argument-type]
          ac_config=ac_config,
          compile_config=effective_compile_config,
          dump_folder=dump_folder,
          parallel_dims=parallel_dims,
          parallelism=parallelism,
          training=training,
          skip_dp=skip_dp,
      )
  if compile_mode == "fullgraph":
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")
  return model
