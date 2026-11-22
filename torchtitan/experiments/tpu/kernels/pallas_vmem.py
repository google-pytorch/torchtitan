"""TPU generation lookup and per-generation VMEM limits for Pallas kernels.

``vmem_limit_bytes()`` is the value kernels pass as
``pltpu.CompilerParams(vmem_limit_bytes=...)``. It depends on the TPU
generation parsed from ``torch.tpu.get_device_properties().name``.
"""

import functools
import re

import torch


@torch._dynamo.assume_constant_result
@functools.lru_cache(maxsize=1)
def tpu_generation() -> int | None:
  """Returns the TPU generation (e.g., 5, 6, 7), or ``None`` if unavailable."""
  get_props = getattr(
      getattr(torch, "tpu", None), "get_device_properties", None
  )
  if get_props is None:
    return None
  try:
    name = get_props().name.lower()
  except RuntimeError:
    return None
  match = re.search(r"(?:v|tpu[- ]?)(\d+)", name)
  return int(match.group(1)) if match else None


def vmem_limit_bytes() -> int:
  """Returns the per-kernel VMEM limit in bytes for the active TPU generation."""
  gen = tpu_generation()
  if gen is not None and gen >= 7:
    return 64 * 1024 * 1024
  if gen == 6:
    return 32 * 1024 * 1024
  return 16 * 1024 * 1024


__all__ = ["tpu_generation", "vmem_limit_bytes"]
