"""Enum defining the types of accelerators supported.

This module defines the `AcceleratorDeviceType` enum, which enumerates the
different hardware accelerators (TPU, GPU, CPU) that can be used for
computations.
"""
import enum


class AcceleratorDeviceType(enum.Enum):
  """Accelerator device type."""
  TPU = "tpu"
  CUDA = "cuda"
  CPU = "cpu"
