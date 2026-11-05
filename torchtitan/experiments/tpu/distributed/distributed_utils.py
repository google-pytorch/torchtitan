"""Utilities for initializing distributed TPU training."""

import os
from absl import logging

from torchtitan.experiments.tpu.distributed import cloud_distributed_utils


def maybe_init_distributed(num_devices: int | None = None) -> bool:
  """Initializes the TPU environment variables based on the platform."""
  # Use provided num_devices, or fall back to WORLD_SIZE env var
  if num_devices is None:
    num_devices = int(os.environ.get("WORLD_SIZE", "1"))

  if num_devices <= 1:
    return False

  if "TORCH_TPU_SLICEBUILDER_ADDRESSES" in os.environ:
    return True  # Already initialzed

  logging.info("Platform: Cloud. Initializing via cloud_distributed_utils.")
  return cloud_distributed_utils.maybe_init_distributed()


