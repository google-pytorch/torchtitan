"""Lets upstream ``torchtitan.models.qwen3_5`` import without ``attn_gym``.

Upstream ``torchtitan.models.qwen3_5.gdn`` imports the CUDA/Triton-only
``attn_gym`` package at module top level (``attn_gym.linear`` ops and, since
upstream #4738 / #4960, kernel autograd functions from submodules such as
``attn_gym.linear.gdn.impl.cudnn``, which it registers with
``spmd.register_local_autograd_function``). Importing this module installs an
import finder that fabricates any ``attn_gym`` module when the package is not
installed, so ``torchtitan.models.qwen3_5`` imports on CPU and TPU. Every
attribute of a stub module is a placeholder that raises if called; the TPU
modules in ``gated_delta`` never call them. Import this module before
``torchtitan.models.qwen3_5``.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types


def _stub_op(module_name: str, name: str):
  if name.startswith("__"):
    raise AttributeError(name)

  class _Unavailable:
    """Placeholder for a CUDA-only kernel or autograd function."""

    def __init__(self, *args, **kwargs):
      del args, kwargs
      raise RuntimeError(
          f"{module_name}.{name} is CUDA-only; Qwen3.5 on TPU runs "
          "TpuInnerGatedDeltaNet instead."
      )

    def __new__(cls, *args, **kwargs):
      del args, kwargs
      raise RuntimeError(
          f"{module_name}.{name} is CUDA-only; Qwen3.5 on TPU runs "
          "TpuInnerGatedDeltaNet instead."
      )

  _Unavailable.__name__ = _Unavailable.__qualname__ = name
  _Unavailable.__module__ = module_name
  return _Unavailable


class _StubModule(types.ModuleType):

  def __getattr__(self, name: str):
    return _stub_op(self.__name__, name)


class _AttnGymStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
  """Creates an empty package for every ``attn_gym`` (sub)module."""

  def find_spec(self, fullname, path=None, target=None):
    del path, target
    if fullname == "attn_gym" or fullname.startswith("attn_gym."):
      return importlib.machinery.ModuleSpec(
          fullname, self, is_package=True
      )
    return None

  def create_module(self, spec):
    module = _StubModule(spec.name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module

  def exec_module(self, module):
    del module


if "attn_gym" not in sys.modules and not importlib.util.find_spec("attn_gym"):
  sys.meta_path.insert(0, _AttnGymStubFinder())
