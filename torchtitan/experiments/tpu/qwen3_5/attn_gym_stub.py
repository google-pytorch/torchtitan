"""Lets upstream ``torchtitan.models.qwen3_5`` import without ``attn_gym``.

Upstream ``torchtitan.models.qwen3_5.gdn`` imports the CUDA/Triton-only
``attn_gym`` package at module top level. Importing this module registers stub
``attn_gym`` / ``attn_gym.linear`` modules when the package is not installed,
so ``torchtitan.models.qwen3_5`` imports on CPU and TPU. The stub ops raise if
called; the TPU modules in ``gated_delta`` never call them. Import this module
before ``torchtitan.models.qwen3_5``.
"""

import importlib.util
import sys
import types


def _stub_op(name: str):
  if name.startswith("__"):
    raise AttributeError(name)

  def unavailable(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        f"attn_gym.linear.{name} is CUDA-only; Qwen3.5 on TPU runs "
        "TpuInnerGatedDeltaNet instead."
    )

  return unavailable


if "attn_gym" not in sys.modules and not importlib.util.find_spec("attn_gym"):
  for _name in ("attn_gym", "attn_gym.linear"):
    _module = types.ModuleType(_name)
    _module.__path__ = []  # type: ignore[attr-defined]
    _module.__getattr__ = _stub_op  # type: ignore[method-assign]
    sys.modules[_name] = _module
