"""Provides utilities for running distributed training on TPU, GPU, and CPU.

run_distributed method can be used to run a function in a distributed manner
across multiple devices. It supports running on TPUs, GPUs, and CPUs.
"""

import os
from typing import Any, Callable, Tuple

from absl import logging
import torch
import torch.distributed as dist
from torchtitan.experiments.tpu import gdist
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import device_env
from torchtitan.experiments.tpu import distributed_utils
from torchtitan.tools import utils


def _ensure_device_type_set(
    accelerator_device_type: device_type.AcceleratorDeviceType):
  """Sets the device type for the current process."""
  if accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
    # We have to force TPU device to be registered here
    available_device_type = utils.device_type
    if available_device_type != "tpu":
      raise RuntimeError(
          "TPU requested but TPU device is not available."
      )
  elif accelerator_device_type == device_type.AcceleratorDeviceType.CUDA:
    if not torch.cuda.is_available():
      raise RuntimeError("GPU requested but CUDA is not available.")
    # No need to select CUDA: detection already reports it whenever it is
    # available, since _get_available_device_type() checks CUDA first.

  elif accelerator_device_type == device_type.AcceleratorDeviceType.CPU:
    # CPU is always available, so there is nothing to verify. Reaching here
    # means the caller already resolved CPU (test_utils.resolve_device_type or
    # gmain), and workers get it from the CPU-only environment set before the
    # spawn below.
    pass

  else:
    raise ValueError(f"Unsupported device type: {device_type}")


def _run_in_gdist_worker(
    accelerator_device_type: device_type.AcceleratorDeviceType,
    func: Callable[[torch.device, int, int, ...], None],  # pytype: disable=invalid-annotation
    run_init_process_group: bool,
    *worker_args: Tuple[Any, ...]):
  """Runs a function within a gdist worker process.

  This function sets up the distributed environment using `torch.distributed`
  based on environment variables populated by `gdist.torchrun` and then
  executes the provided function `func`.
  Note that it only works for singlehost GPU setup.

  Args:
    accelerator_device_type: The type of accelerator being used ('gpu' or 'cpu')
    func: The worker function to execute. It should accept `device`, `rank`,
      `world_size`, and any additional `worker_args`.
    run_init_process_group: Whether to initialize the process group.
    *worker_args: Additional arguments to pass to `func`.

  Raises:
    ValueError: If an unsupported `accelerator_device_type` is provided.
  """
  world_size = int(os.environ["WORLD_SIZE"])
  local_rank = int(os.environ["LOCAL_RANK"])
  nproc_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
  node_rank = int(os.environ["GROUP_RANK"])

  global_rank = node_rank * nproc_per_node + local_rank
  _ensure_device_type_set(accelerator_device_type)
  if accelerator_device_type == device_type.AcceleratorDeviceType.CUDA:
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized() and run_init_process_group:
      dist.init_process_group(
          backend="nccl",
          rank=global_rank,
          world_size=world_size,
      )
    func(device, global_rank, world_size, *worker_args)
    return
  elif accelerator_device_type == device_type.AcceleratorDeviceType.CPU:
    device = torch.device("cpu")
    if not dist.is_initialized() and run_init_process_group:
      dist.init_process_group(
          backend="gloo",
          rank=global_rank,
          world_size=world_size,
      )
    func(device, global_rank, world_size, *worker_args)
  elif accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
    # Bootstrap TORCH_TPU_SLICEBUILDER_ADDRESSES / TORCH_TPU_TOPOLOGY here, in
    # the worker: _maybe_init_distributed_on_tpu_vm() reads WORLD_SIZE from the
    # environment, which torchrun sets per worker and which is absent in the
    # spawning parent -- so the pre-spawn call cannot do it for us. Must precede
    # the first TPU device use, or the PJRT client is never created.
    distributed_utils.maybe_init_distributed()
    if not dist.is_initialized() and run_init_process_group:
      device = torch.device("tpu")

      dist.init_process_group(
          backend="tpu_dist",
          rank=global_rank,
          world_size=world_size,
      )
    else:
      device = torch.device("tpu")

    func(device, global_rank, world_size, *worker_args)
  else:
    raise ValueError(
        f"Unsupported accelerator device type: {accelerator_device_type}"
    )


def run_distributed(
    num_devices: int,
    accelerator_device_type: device_type.AcceleratorDeviceType,
    func: Callable,  # pylint: disable=g-bare-generic
    *worker_args: Any,
    run_init_process_group: bool = True,
) -> None:
  """Runs a function in a distributed manner across multiple devices.

  This function abstracts away the details of setting up a distributed
  environment and executing a function across multiple devices. It supports
  running on TPUs, GPUs, and CPUs.

  Args:
    num_devices: The number of devices to use. If 1, `func` is run on a
      single device without initializing a distributed process group.
    accelerator_device_type: The type of accelerator to use, one of 'tpu',
      'gpu', or 'cpu'.
    func: The worker function to execute on each process. The function
      signature must be `func(device: torch.Device, rank: int, world_size: int,
      *args)`, where `device` is the device assigned to the worker, `rank` is
      the global rank of the process, `world_size` is the total number of
      processes, and `*args` are arguments from `worker_args`.
    *worker_args: A tuple of arguments to pass to `func` after `device`,
      `rank`, and `world_size`.
    run_init_process_group: Whether to initialize the distributed process group.
      Defaults to True.

  Raises:
    RuntimeError: If `accelerator_device_type` is "cuda" but CUDA is not
      available, or if `accelerator_device_type` is an unexpected value
      when `num_devices` is 1.
  """
  if num_devices == 1:
    _ensure_device_type_set(accelerator_device_type)
    if accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
      accelerator_device = torch.device("tpu")
      logging.info("Using TPU device: %s", accelerator_device)
      func(accelerator_device, 0, 1, *worker_args)
    elif accelerator_device_type == device_type.AcceleratorDeviceType.CUDA:
      accelerator_device = torch.device("cuda:0")
      logging.info("Using GPU device: %s", accelerator_device)
      func(accelerator_device, 0, 1, *worker_args)
    elif accelerator_device_type == device_type.AcceleratorDeviceType.CPU:
      accelerator_device = torch.device("cpu")
      logging.info("Using CPU device: %s", accelerator_device)
      func(accelerator_device, 0, 1, *worker_args)
    else:
      raise RuntimeError(
          f"Unexpected flag value: {accelerator_device_type}"
      )
  else:
    if accelerator_device_type == device_type.AcceleratorDeviceType.CPU:
      # workaround for Google3: `backend=gloo` is only allowed in tests.
      os.environ["TEST_TARGET"] = "1"
      # Workers are spawned, so they re-detect their device from scratch and do
      # not inherit this process's device-type global. Hide the accelerator in
      # the environment they inherit instead, so their detection returns CPU.
      os.environ.update(device_env.cpu_only_env())
    if accelerator_device_type == device_type.AcceleratorDeviceType.CUDA:
      # Hide GPUs from TF so that it won't occupy any GPU memory.
      pass
    if accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
      # Construct the environment before spawning workers.
      distributed_utils.maybe_init_distributed(num_devices)
    try:
      # TODO(jialei777): Support returning results from gdist workers.
      gdist.torchrun(
          _run_in_gdist_worker,
          nproc_per_node=num_devices)(
              accelerator_device_type,
              func,
              run_init_process_group,
              *worker_args)
    finally:
      # Calling destroy_process_group() for cleanup once workers are done.
      if dist.is_initialized():
        dist.destroy_process_group()
