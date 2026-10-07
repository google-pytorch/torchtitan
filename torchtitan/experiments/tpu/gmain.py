"""Entry point for Torchtitan TPU experiments."""

import argparse
import os
import typing
from typing import Callable, List, Sequence, Tuple

from absl import flags
from absl import logging
from absl.flags import argparse_flags
import torch.multiprocessing as mp
import torchtitan.config
from torchtitan.experiments.tpu import distributed_utils as tpu_distributed_utils
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.tools import utils
import torchtitan.experiments.tpu.tpu_job_config as tpu_job_config_module

from absl import app

global_start_trainer_func = None


def _start_trainer(config: tpu_job_config_module.TPUTrainerConfig):
  """Starts the trainer with the specified eager mode if enabled."""
  global global_start_trainer_func  # pylint: disable=global-variable-not-assigned
  is_tpu = utils.device_type == "tpu"
  if is_tpu:
    # Render the config's XLA / libtpu flags into the environment before the
    # first TPU device use below: libtpu and the XLA compiler read them while
    # the PJRT client is created, and torch_tpu memoizes them from then on.
    tpu_flags.apply_init_flags(config)
    # Populates TORCH_TPU_SLICEBUILDER_ADDRESSES / TORCH_TPU_TOPOLOGY before any
    # TPU device is created. gmain does not spawn, so this is the per-worker
    # call the Cloud/GKE path requires: it derives the configuration from the
    # LOCAL_RANK / RANK / WORLD_SIZE that torchrun sets in each worker.
    # Idempotent -- returns early when a host-level launcher already populated
    # the env.
    tpu_distributed_utils.maybe_init_distributed()
  if is_tpu and config.tpu_config.eager_mode:
    import torch  # pylint: disable=g-import-not-at-top
    import torch_tpu  # pylint: disable=g-import-not-at-top,unused-import

    eager_mode_enum = getattr(
        torch.tpu.EagerMode, config.tpu_config.eager_mode  # pyrefly: ignore[missing-attribute]
    )
    with torch.tpu.set_eager_mode(eager_mode_enum):  # pyrefly: ignore[missing-attribute]
      global_start_trainer_func(config)
  else:
    global_start_trainer_func(config)


def _parse_flags(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
  """Parses command-line flags."""
  parser = argparse_flags.ArgumentParser(
      description="Minimal trainer for TorchTitan models.",
      inherited_absl_flags=flags.FLAGS,
      allow_abbrev=False,
  )
  parser.add_argument(
      "--device",
      help=(
          "Device the training is expected to run on. Optional, and checked"
          " rather than applied: the run fails if the detected device differs."
          " Select the device in the environment before launching --"
          " TORCH_DEVICE_BACKEND_AUTOLOAD=0 unregisters the TPU backend and"
          " CUDA_VISIBLE_DEVICES= hides GPUs, both yielding CPU."
      ),
      choices=["cpu", "cuda", "tpu"],
      default=None,
      required=False,
  )
  parser.add_argument(
      "--nproc_per_node",
      help="Number of devices to run the training on",
      type=int,
      default=1,
  )
  args, remaining_args = parser.parse_known_args(argv[1:])
  return args, remaining_args


def _verify_requested_device(requested: str | None) -> None:
  """Checks --device against the device that was actually detected.

  The flag reports; it does not select. Device selection is fixed before this
  process starts, when torch is imported and registers whichever accelerator
  the environment exposes -- by which time main() is far too late to change it.
  Checking here turns a stale or copy-pasted --device into an immediate error
  instead of a run that silently uses the wrong hardware.

  Args:
    requested: The value of --device, or None when it was not passed.

  Raises:
    ValueError: If the requested device is not the detected one.
  """
  if not requested:
    return
  detected = utils.device_type
  if requested == detected:
    return
  raise ValueError(
      f"--device={requested} was requested but {detected} was detected."
      " --device reports the device; it cannot select one, because the"
      " accelerator is fixed when torch is imported. Select the device in the"
      " environment before launching: TORCH_DEVICE_BACKEND_AUTOLOAD=0"
      " unregisters the TPU backend and CUDA_VISIBLE_DEVICES= hides GPUs, both"
      " of which yield CPU. Omit --device to accept whatever is detected."
  )


def main(parsed_args: Tuple[argparse.Namespace, List[str]]):
  args, remaining_args = parsed_args
  _verify_requested_device(args.device)
  config_manager = torchtitan.config.ConfigManager()
  config = typing.cast(
      tpu_job_config_module.TPUTrainerConfig,
      config_manager.parse_args(remaining_args),
  )

  _start_trainer(config)



def handle_main(
    start_trainer_func: Callable[
        [tpu_job_config_module.TPUTrainerConfig], None
    ],
):
  global global_start_trainer_func
  global_start_trainer_func = start_trainer_func
  app.run(main, flags_parser=_parse_flags)
