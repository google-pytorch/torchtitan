"""SPMD utilities for TPU."""

import torch
from torchtitan.tools.logging import logger

try:
  spmd_safe = torch.tpu.distributed.spmd_safe
except AttributeError as e:
  logger.warning(
      "Unable to find spmd_safe decorator in torch.tpu. Compiled modules "
      f"will fall back to skinny collective behavior.\nDetails: {e}"
  )
  def spmd_safe(func):
    return func


def apply_spmd_safe(mod: torch.nn.Module) -> torch.nn.Module:
  """Monkey patches a module's forward method with torch_tpu spmd_safe decorator."""
  orig_forward = mod.forward
  mod.forward = spmd_safe(orig_forward)
  return mod
