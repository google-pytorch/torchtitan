"""Utilities for initializing distributed TPU training."""

import os
from absl import logging
import torch
import torch.distributed as dist

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




def init_single_rank_process_group() -> None:
  """Initialize a one-rank process group for a single-process run.

  spmd_types annotates the model inputs against ``ParallelDims``' device mesh,
  and torch builds that mesh from the default process group, so even a
  single-process run (a parity test's CPU reference, or a one-chip run outside
  torchrun) needs one. Uses a gloo group over an in-process store: no
  rendezvous, and no TPU backend, which refuses a group smaller than the host's
  chip count. Every mesh axis has size 1, so the spmd_types helpers never issue
  a collective on it. No-op if a group already exists.
  """
  if dist.is_initialized():
    return
  from torchtitan.distributed import utils as dist_utils  # pylint: disable=g-import-not-at-top

  # Same process-wide setup as dist_utils.init_distributed.
  dist_utils.enable_fp32_matmul_emulation_with_bf16x9()
  torch.autograd.set_multithreading_enabled(False)
  dist.init_process_group(
      backend="gloo", rank=0, world_size=1, store=dist.HashStore()
  )
