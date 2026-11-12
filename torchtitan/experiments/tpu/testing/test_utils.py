"""Shared utilities for accelerator device tests."""

import functools
import json
import os
import subprocess
import sys
from typing import Any
from absl import flags
from absl import logging
import torch
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.distributed import device_env


_DEVICE = flags.DEFINE_enum_class(
    "device",
    None,  # Default value
    device_type.AcceleratorDeviceType,
    help=(
        "Device to test on. Optional: when omitted, the detected accelerator"
        " is used. Pass cpu to force CPU; passing an accelerator that is not"
        " present is an error. Detection follows the environment, so"
        " TORCH_DEVICE_BACKEND_AUTOLOAD=0 (unregisters the TPU backend) and"
        " CUDA_VISIBLE_DEVICES= (hides GPUs) also select CPU."
    ),
    required=False,
)

_NUM_DEVICES = flags.DEFINE_integer(
    "num_devices",
    1,
    required=False,
    help="Number of devices to run the test on.",
)


def resolve_device_type() -> device_type.AcceleratorDeviceType:
  """Returns the device to test on, honoring --device if it was given.

  Omitting --device runs on whatever accelerator is detected. Passing
  --device=cpu forces CPU, which is always possible: it is a downgrade from
  whatever accelerator is present. Passing an accelerator that is not present
  is an error rather than a silent fallback, so a stale or copy-pasted value
  fails loudly instead of running on the wrong hardware.

  Raises:
    ValueError: If --device names an accelerator that was not detected.
  """
  # Imported here: tpu_utils pulls in torch device modules, which tests that
  # only need the flags should not pay for at import time.
  from torchtitan.tools import utils  # pylint: disable=g-import-not-at-top

  detected = device_type.AcceleratorDeviceType(utils.device_type)
  requested = _DEVICE.value
  if requested is None or requested == detected:
    return detected

  if requested == device_type.AcceleratorDeviceType.CPU:
    # Nothing in this process reads the device-type global after resolution --
    # tests use the returned value via self.accelerator_device -- so returning
    # CPU is enough here. Spawned workers do re-detect, so hide the accelerator
    # in the environment they inherit.
    os.environ.update(device_env.cpu_only_env())
    return requested

  raise ValueError(
      f"--device={requested.value} was requested but {detected.value} was"
      f" detected: no {requested.value} device is available. Omit --device to"
      " use the detected device, or pass --device=cpu to force CPU. Detection"
      " itself follows the environment, so TORCH_DEVICE_BACKEND_AUTOLOAD=0"
      " (unregisters the TPU backend) and CUDA_VISIBLE_DEVICES= (hides GPUs)"
      " also select CPU."
  )

EXPECTED_CPU_DTYPE = torch.float32
EXPECTED_DEVICE_DTYPE = torch.float32


def _skip_if_device(
    device_to_skip: device_type.AcceleratorDeviceType,
    reason: str,
    bug_id: int | None = None,
):
  """Skips the test if the target device matches device_to_skip.

  Args:
      device_to_skip: The device type to skip the test on.
      reason: The reason for skipping the test.
      bug_id: The bug ID associated with the skipping of the test.

  Returns:
      A decorator function that skips the test if the device matches.
  """

  def decorator(test_func):
    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
      # Format reason with bug ID if provided
      skip_reason = f"{reason} (b/{bug_id})" if bug_id else reason

      # Check if device matches and skip if so
      if _DEVICE.value == device_to_skip:
        args[0].skipTest(skip_reason)

      return test_func(*args, **kwargs)

    return wrapper

  return decorator


def skip_if_tpu(
    reason: str = "Test does not work on TPU", bug_id: int | None = None
):
  """Skips the test if the target device is TPU."""
  return _skip_if_device(device_type.AcceleratorDeviceType.TPU, reason, bug_id)


def skip_if_cpu(
    reason: str = "Test does not work on CPU", bug_id: int | None = None
):
  """Skips the test if the target device is CPU."""
  return _skip_if_device(device_type.AcceleratorDeviceType.CPU, reason, bug_id)


def skip_if_cuda(
    reason: str = "Test does not work on CUDA", bug_id: int | None = None
):
  """Skips the test if the target device is CUDA."""
  return _skip_if_device(device_type.AcceleratorDeviceType.CUDA, reason, bug_id)


def _get_difference_str(
    tensor_test: torch.Tensor,
    tensor_reference: torch.Tensor,
    test_label: str,
    ref_label: str,
) -> str:
  """Formats difference information."""
  # Calculate difference metrics
  abs_diff = torch.abs(tensor_reference - tensor_test)
  max_abs_diff = abs_diff.max().item()
  is_scalar = tensor_test.numel() == 1

  # Conditional string components
  if is_scalar:
    extra_info = (
        f"\n  {ref_label} Value: {tensor_reference.item()}"
        f"\n  {test_label} Value: {tensor_test.item()}"
    )
  else:
    extra_info = (
        f"\n  {ref_label} shape: {tensor_reference.shape}"
        f"\n  {test_label} shape: {tensor_test.shape}"
    )
  return f"Max Absolute Difference: {max_abs_diff}\n" + extra_info


def print_difference(
    tensor_test: torch.Tensor,
    tensor_reference: torch.Tensor,
):
  """Logs the difference between two tensors.

  Args:
      tensor_test: The tensor or scalar to validate (e.g. from parallel
        execution).
      tensor_reference: The reference tensor or scalar (e.g. from single device
        execution).
  """
  logging.info(
      _get_difference_str(tensor_test, tensor_reference, "Test", "Reference")
  )


def check_equivalence(
    tensor_test: torch.Tensor,
    tensor_reference: torch.Tensor,
    atol: float,
    rtol: float,
    check_name: str | None = None,
    step: int | None = None,
    test_label: str = "Test",
    ref_label: str = "Reference",
):
  """Helper for comparing tensors/losses.

  Args:
      tensor_test: The tensor or scalar to validate (e.g. from parallel
        execution).
      tensor_reference: The reference tensor or scalar (e.g. from single device
        execution).
      check_name: A descriptive name for the value being checked (e.g. "Loss").
      step: The current training step index, or None if not applicable.
      atol: Absolute tolerance for torch.allclose.
      rtol: Relative tolerance for torch.allclose.
      test_label: Label for the test tensor in error messages. Defaults to
        "Parallel".
      ref_label: Label for the reference tensor in error messages. Defaults to
        "Single Device".

  Raises:
      AssertionError: If the tensors are not numerically equivalent within
        tolerances.
  """

  if tensor_test.dtype != tensor_reference.dtype:
    tensor_test = tensor_test.to(tensor_reference.dtype)

  if not torch.allclose(tensor_test, tensor_reference, atol=atol, rtol=rtol):
    error_info = _get_difference_str(
        tensor_test, tensor_reference, test_label, ref_label
    )

    step_info = f"[Step {step}]" if step is not None else "[Check]"
    error_message = (
        f"\n{step_info} FAILED: {check_name} mismatch."
        f"\n  {error_info}"
        f"\n  Tolerance: {atol} / {rtol}"
    )
    raise AssertionError(error_message)


def run_in_subprocess(
    module_name: str,
    function_name: str,
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> None:
  """Runs a function in an isolated Python subprocess.

  This is necessary because PyTorch distributed and TPU states (like PjRtClient)
  are stored globally within the Python process. Re-initializing them for
  verification steps causes errors.

  Args:
      module_name: The name of the module containing the function.
      function_name: The name of the function to run.
      args: Positional arguments to pass to the function (must be
        JSON-serializable).
      kwargs: Keyword arguments to pass to the function (must be
        JSON-serializable).
      env: Environment for the subprocess. Defaults to inheriting this
        process's environment. Pass device_env.cpu_only_env() to make the
        subprocess detect CPU, which is the supported way to run a verification
        step on CPU while an accelerator is attached.

  Raises:
      subprocess.CalledProcessError: If the subprocess fails.
  """
  args = args or []
  kwargs = kwargs or {}

  if sys.executable is None:
    logging.warning(
        "sys.executable is None (likely embedded Python). Falling back to"
        " in-process execution which may leak state."
    )
    import importlib

    module = importlib.import_module(module_name)
    func = getattr(module, function_name)
    return func(*args, **kwargs)

  logging.info(
      "Running %s.%s in an isolated subprocess...", module_name, function_name
  )

  # Build a script that imports the function, decodes arguments, and executes it
  script = f"""
import sys
import json
import importlib

sys.path = {sys.path}

module_name = {repr(module_name)}
function_name = {repr(function_name)}
args = json.loads({repr(json.dumps(args))})
kwargs = json.loads({repr(json.dumps(kwargs))})

module = importlib.import_module(module_name)
func = getattr(module, function_name)
func(*args, **kwargs)
"""

  try:
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    logging.info("Isolated subprocess output:\n%s", result.stdout)
    if result.stderr:
      logging.warning("Isolated subprocess stderr:\n%s", result.stderr)
  # Wrap in try/except to reveal underlying Python crash traces when the
  # isolated subprocess fails.
  except subprocess.CalledProcessError as e:
    logging.error("Isolated subprocess failed! stdout:\n%s", e.stdout)
    if e.stderr:
      logging.error("Isolated subprocess failed! stderr:\n%s", e.stderr)
    raise
