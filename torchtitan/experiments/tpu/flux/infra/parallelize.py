"""Parallelization utilities for Flux models on TPU."""

from torch import nn
import torchtitan.config
import torchtitan.distributed
import torchtitan.trainer


def parallelize_flux(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: torchtitan.trainer.Trainer.Config,
) -> nn.Module:
  """Parallelizes the Flux model based on the provided configuration.

  Args:
    model: The nn.Module to be parallelized.
    parallel_dims: Contains the world mesh and other parallelization dimensions.
    job_config: The job configuration.

  Returns:
    The parallelized nn.Module.
  """
  # For now, we delegate to the native implementation.
  # If TPU-specific optimizations are needed, we can implement them here.
  return model.parallelize(
      ac_config=job_config.activation_checkpoint,
      compile_config=job_config.compile,
      dump_folder="",
      parallel_dims=parallel_dims,
      parallelism=job_config.parallelism,
      training=job_config.training,
  )
