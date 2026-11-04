"""Environment-level device selection, importable before torch.

Which accelerator a process can see is fixed when torch is imported: PyTorch's
backend autoload registers torch_tpu at that point, and CUDA reads its device
mask then too. Selecting CPU therefore has to happen *before* `import torch`,
which is why this module imports no torch -- it only computes and applies
environment variables, so a parent can configure a child it is about to spawn.

Preferring this over mutating a device-type global means the child's detection
is correct from the start, and every later reader of device_type agrees
by construction rather than by remembering to call a setter.
"""

import os
from typing import Mapping

# Stops PyTorch autoloading privateuse1 backend extensions, so torch_tpu never
# registers and no TPU accelerator is reported.
_BACKEND_AUTOLOAD = "TORCH_DEVICE_BACKEND_AUTOLOAD"

# An empty mask hides every GPU from CUDA.
_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


def cpu_only_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
  """Returns a copy of an environment in which only CPU will be detected.

  Pass the result to a subprocess, or apply it to this process so that spawned
  workers inherit it:

      subprocess.run(..., env=device_env.cpu_only_env())   # explicit child env
      os.environ.update(device_env.cpu_only_env())         # inherited by spawn

  The second form does not change the caller's own device, which was fixed when
  it imported torch; it only affects processes started afterwards.

  Args:
    base: Environment to copy. Defaults to the current process environment.

  Returns:
    A new environment mapping in which torch detects no accelerator.
  """
  env = dict(os.environ if base is None else base)
  env[_BACKEND_AUTOLOAD] = "0"
  env[_CUDA_VISIBLE_DEVICES] = ""
  return env
