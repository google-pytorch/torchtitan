# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU runtime optimizations and policy wiring for FSDP2 CPU offload.

Stock PyTorch FSDP2 CPU offload (``CPUOffloadPolicy``) has four bottlenecks or
incompatibilities on TPU:

1. ``CPUOffloadPolicy()`` defaults to ``pin_memory=True``, which fails without a
   CUDA/host-pinned allocator and can deadlock non-blocking transfers against
   compiled activation-checkpointed blocks on ``torch_tpu`` (see ``b/499075866``).
   :class:`TPUCPUOffloadPolicy` defaults ``pin_memory=False``.
2. ``FSDPParam.all_gather_inputs`` copies each FSDP unit's fp32 master shard
   from CPU to device and casts it to ``param_dtype`` on *every* microbatch
   forward and post-reshard backward unshard, even though master weights only
   change on ``optimizer.step()``. Caching the cast device shard per
   ``FSDPParam`` (invalidated on ``optimizer.step()``, checkpoint load, and
   ``reset_sharded_param()``) reduces H2D shard transfers to once per step.
3. ``foreach_reduce`` copies each parameter's reduce-scattered fp32 gradient
   synchronously from device to CPU on *every* microbatch. Accumulating the
   cast/viewed sharded gradients in fp32 on device across microbatches and
   copying once per optimizer step into persistent pageable host buffers removes
   per-microbatch D2H stalls while preserving IEEE-754 addition order
   (``max_abs == 0.0`` bit-identical parity with stock FSDP2 CPU offload).
4. Under ``torchrun`` (which sets ``OMP_NUM_THREADS=1`` when unset), host-side
   ``clip_grad_norm_`` and ``AdamW.step()`` run single-threaded over offloaded
   master weights unless ``torch.set_num_threads`` is configured.
"""

from collections.abc import Callable
import dataclasses
import functools
import sys
import threading
from typing import Any
import weakref

import torch
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as fsdp_collectives
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as fsdp_param_group
from torch.distributed.fsdp._fully_shard._fsdp_param import (
    FSDPParam,
    ShardedState,
    _to_dtype_if_needed,
)

from torchtitan.tools import utils as tool_utils
from torchtitan.tools.logging import logger


@dataclasses.dataclass
class TPUCPUOffloadPolicy(CPUOffloadPolicy):
  """``CPUOffloadPolicy`` subclass defaulting ``pin_memory=False`` on TPU."""

  pin_memory: bool = False


@dataclasses.dataclass
class FSDPCPUOffloadStats:
  """Diagnostic counters for unit tests and profiling."""

  param_cache_hits: int = 0
  param_cache_misses: int = 0
  d2h_sync_calls: int = 0


@dataclasses.dataclass
class _ParamViewSpec:
  """Metadata locating one parameter's sharded gradient inside a flat run."""

  fsdp_param: Any
  rel_offset: int
  numel: int
  shape: torch.Size


@dataclasses.dataclass
class _DeviceGradRun:
  """Contiguous 1D device buffer accumulating sharded gradients for a run."""

  flat_accum: torch.Tensor
  specs: list[_ParamViewSpec]


_ORIG_CPU_OFFLOAD_POLICIES: dict[str, Any] = {}
_ORIG_ALL_GATHER_INPUTS: property | None = None
_ORIG_RESET_SHARDED_PARAM: Callable[..., Any] | None = None
_ORIG_CAST_AND_VIEW_SHARDED_GRADS: Callable[..., Any] | None = None
_ORIG_FOREACH_REDUCE_COLLECTIVES: Callable[..., Any] | None = None
_ORIG_FOREACH_REDUCE_PARAM_GROUP: Callable[..., Any] | None = None
_ORIG_COLLECTIVES_TO_DTYPE: Callable[..., Any] | None = None
_LEGACY_REDUCE_TLS = threading.local()

_POLICY_PATCH_MODULES: tuple[str, ...] = (
    "torchtitan.distributed.fsdp",
    "torchtitan.experiments.transformers_modeling_backend.parallelize",
    "torchtitan.models.flux.parallelize",
)

_RUNTIME_INSTALLED: bool = False
_PARAM_CACHE_ENABLED: bool = False
_DEVICE_GRAD_ACCUM_ENABLED: bool = False
_PARAM_CACHE_VERSION: int = 0
_CACHED_FSDP_PARAMS: weakref.WeakSet[Any] = weakref.WeakSet()
_ACTIVE_GROUP_GRADS: dict[tuple[int, ...], list[_DeviceGradRun]] = {}
_STATS = FSDPCPUOffloadStats()


def get_cpu_offload_stats() -> FSDPCPUOffloadStats:
  """Returns a snapshot copy of the current FSDP2 CPU offload counters."""
  return dataclasses.replace(_STATS)


def reset_cpu_offload_stats() -> None:
  """Resets diagnostic counters to zero."""
  _STATS.param_cache_hits = 0
  _STATS.param_cache_misses = 0
  _STATS.d2h_sync_calls = 0


def install_tpu_cpu_offload_policy(*, force: bool = False) -> None:
  """Rebinds TorchTitan's ``CPUOffloadPolicy`` references to ``pin_memory=False``.

  By default this activates only when ``torchtitan.tools.utils.device_type == "tpu"``;
  pass ``force=True`` in CPU/Gloo tests or explicit TPU offload configurations.
  """
  if not force and tool_utils.device_type != "tpu":
    return

  import torchtitan.distributed.fsdp as tt_fsdp  # pylint: disable=g-import-not-at-top

  del tt_fsdp  # imported to ensure module is present in sys.modules
  for mod_name in _POLICY_PATCH_MODULES:
    mod = sys.modules.get(mod_name)
    if mod is None or not hasattr(mod, "CPUOffloadPolicy"):
      continue
    current = getattr(mod, "CPUOffloadPolicy")
    if current is TPUCPUOffloadPolicy:
      continue
    if mod_name not in _ORIG_CPU_OFFLOAD_POLICIES:
      _ORIG_CPU_OFFLOAD_POLICIES[mod_name] = current
    setattr(mod, "CPUOffloadPolicy", TPUCPUOffloadPolicy)


def uninstall_tpu_cpu_offload_policy() -> None:
  """Restores upstream ``CPUOffloadPolicy`` bindings across TorchTitan modules."""
  for mod_name, orig_cls in list(_ORIG_CPU_OFFLOAD_POLICIES.items()):
    mod = sys.modules.get(mod_name)
    if mod is not None:
      setattr(mod, "CPUOffloadPolicy", orig_cls)
  _ORIG_CPU_OFFLOAD_POLICIES.clear()


def invalidate_param_cache() -> None:
  """Invalidates and frees all cached device-resident parameter shards."""
  global _PARAM_CACHE_VERSION
  _PARAM_CACHE_VERSION += 1
  for fsdp_param in list(_CACHED_FSDP_PARAMS):
    setattr(fsdp_param, "_tpu_offload_param_cache", None)
  _CACHED_FSDP_PARAMS.clear()


def clear_device_grads() -> None:
  """Drops any unmaterialized device-accumulated sharded gradients."""
  _ACTIVE_GROUP_GRADS.clear()


def _patched_all_gather_inputs(self: FSDPParam) -> list[torch.Tensor]:
  assert _ORIG_ALL_GATHER_INPUTS is not None
  orig_fget = _ORIG_ALL_GATHER_INPUTS.fget
  assert orig_fget is not None
  if (
      not _PARAM_CACHE_ENABLED
      or not self.offload_to_cpu
      or self.sharded_state != ShardedState.SHARDED
      or hasattr(self._sharded_local_tensor, "fsdp_pre_all_gather")  # pylint: disable=protected-access
  ):
    return orig_fget(self)

  cached_entry: tuple[int, list[torch.Tensor]] | None = getattr(
      self, "_tpu_offload_param_cache", None
  )
  if cached_entry is not None:
    cached_ver, cached_inputs = cached_entry
    if (
        cached_ver == _PARAM_CACHE_VERSION
        and cached_inputs
        and all(t.device == self.device for t in cached_inputs)
    ):
      _STATS.param_cache_hits += 1
      return list(cached_inputs)

  orig_inputs = orig_fget(self)
  setattr(
      self,
      "_tpu_offload_param_cache",
      (_PARAM_CACHE_VERSION, list(orig_inputs)),
  )
  _CACHED_FSDP_PARAMS.add(self)
  _STATS.param_cache_misses += 1
  return list(orig_inputs)


def _patched_reset_sharded_param(self: FSDPParam) -> None:
  setattr(self, "_tpu_offload_param_cache", None)
  assert _ORIG_RESET_SHARDED_PARAM is not None
  _ORIG_RESET_SHARDED_PARAM(self)


def _pack_sharded_grads_into_runs(
    reduce_output: torch.Tensor,
    fsdp_params: list[FSDPParam],
    sharded_grads: list[torch.Tensor],
) -> list[tuple[torch.Tensor, list[_ParamViewSpec]]]:
  """Packs ``sharded_grads`` into contiguous 1D device buffers grouped by dtype."""
  if not fsdp_params:
    return []

  first_dtype = sharded_grads[0].dtype
  total_grad_numel = sum(g.numel() for g in sharded_grads)
  can_view_reduce_output_directly = (
      total_grad_numel == reduce_output.numel()
      and all(g.dtype == first_dtype for g in sharded_grads)
      and not any(
          getattr(p, "needs_nonzero_dim_padding", False) for p in fsdp_params
      )
  )
  if can_view_reduce_output_directly:
    flat_view = _to_dtype_if_needed(reduce_output, first_dtype).reshape(-1)
    specs: list[_ParamViewSpec] = []
    offset = 0
    for fsdp_param, grad in zip(fsdp_params, sharded_grads):
      numel = grad.numel()
      specs.append(
          _ParamViewSpec(
              fsdp_param=fsdp_param,
              rel_offset=offset,
              numel=numel,
              shape=grad.shape,
          )
      )
      offset += numel
    return [(flat_view, specs)]

  runs: list[tuple[torch.Tensor, list[_ParamViewSpec]]] = []
  idx = 0
  num_params = len(fsdp_params)
  while idx < num_params:
    dtype = sharded_grads[idx].dtype
    run_flat_grads: list[torch.Tensor] = []
    specs = []
    offset = 0
    while idx < num_params and sharded_grads[idx].dtype == dtype:
      grad = sharded_grads[idx]
      numel = grad.numel()
      run_flat_grads.append(grad.contiguous().reshape(-1))
      specs.append(
          _ParamViewSpec(
              fsdp_param=fsdp_params[idx],
              rel_offset=offset,
              numel=numel,
              shape=grad.shape,
          )
      )
      offset += numel
      idx += 1
    flat_grad = (
        run_flat_grads[0]
        if len(run_flat_grads) == 1
        else torch.cat(run_flat_grads, dim=0)
    )
    runs.append((flat_grad, specs))
  return runs


def _is_device_grad_accum_eligible(
    fsdp_params: list[FSDPParam],
    *,
    legacy_path: bool = False,
) -> bool:
  if not (
      _DEVICE_GRAD_ACCUM_ENABLED
      and fsdp_params
      and all(p.offload_to_cpu for p in fsdp_params)
      and not any(
          getattr(p.sharded_param, "_post_accumulate_grad_hooks", None)
          for p in fsdp_params
      )
  ):
    return False
  if legacy_path and any(
      getattr(p, "needs_nonzero_dim_padding", False) for p in fsdp_params
  ):
    return False
  return True


def _accumulate_runs_on_device(
    fsdp_params: list[FSDPParam],
    runs: list[tuple[torch.Tensor, list[_ParamViewSpec]]],
) -> None:
  group_key = tuple(id(p) for p in fsdp_params)
  existing = _ACTIVE_GROUP_GRADS.get(group_key)
  if existing is None or len(existing) != len(runs):
    if existing is not None:
      # Fallback if run layout ever changes mid-step: flush prior runs first.
      materialize_host_grads()
    new_runs: list[_DeviceGradRun] = [
        _DeviceGradRun(flat_accum=flat_view.clone(), specs=specs)
        for flat_view, specs in runs
    ]
    _ACTIVE_GROUP_GRADS[group_key] = new_runs
  else:
    for run_state, (flat_view, _specs) in zip(existing, runs):
      run_state.flat_accum.add_(flat_view)


def _patched_cast_and_view_sharded_grads(
    reduce_output: torch.Tensor,
    fsdp_params: list[FSDPParam],
    *args: Any,
    **kwargs: Any,
) -> list[torch.Tensor]:
  assert _ORIG_CAST_AND_VIEW_SHARDED_GRADS is not None
  if not _is_device_grad_accum_eligible(fsdp_params):
    return _ORIG_CAST_AND_VIEW_SHARDED_GRADS(
        reduce_output, fsdp_params, *args, **kwargs
    )

  sharded_grads = _ORIG_CAST_AND_VIEW_SHARDED_GRADS(
      reduce_output, fsdp_params, *args, **kwargs
  )
  runs = _pack_sharded_grads_into_runs(reduce_output, fsdp_params, sharded_grads)
  _accumulate_runs_on_device(fsdp_params, runs)

  # Returning an empty list makes foreach_reduce's non-strict
  # `zip(fsdp_params, sharded_grads)` skip the per-microbatch D2H copy loop
  # while still recording `post_reduce_event` cleanly.
  return []


def _patched_foreach_reduce_legacy(
    fsdp_params: list[FSDPParam],
    unsharded_grads: list[torch.Tensor],
    *args: Any,
    **kwargs: Any,
) -> Any:
  """Pure-Python fallback hook for PyTorch < 2.15 lacking _cast_and_view_sharded_grads."""
  assert _ORIG_FOREACH_REDUCE_COLLECTIVES is not None
  if not _is_device_grad_accum_eligible(fsdp_params, legacy_path=True):
    return _ORIG_FOREACH_REDUCE_COLLECTIVES(
        fsdp_params, unsharded_grads, *args, **kwargs
    )

  orig_params = list(fsdp_params)
  mutable_params = list(fsdp_params)
  padded_sharded_numels = [
      p._sharded_param_data.numel()  # pylint: disable=protected-access
      for p in orig_params
  ]
  prev_ctx = getattr(_LEGACY_REDUCE_TLS, "ctx", None)
  _LEGACY_REDUCE_TLS.ctx = (
      orig_params,
      mutable_params,
      unsharded_grads,
      padded_sharded_numels,
  )
  try:
    return _ORIG_FOREACH_REDUCE_COLLECTIVES(
        mutable_params, unsharded_grads, *args, **kwargs
    )
  finally:
    _LEGACY_REDUCE_TLS.ctx = prev_ctx


def _patched_to_dtype_if_needed_legacy(
    tensor: torch.Tensor,
    dtype: torch.dtype | None,
) -> torch.Tensor:
  assert _ORIG_COLLECTIVES_TO_DTYPE is not None
  cast_output = _ORIG_COLLECTIVES_TO_DTYPE(tensor, dtype)
  ctx = getattr(_LEGACY_REDUCE_TLS, "ctx", None)
  if ctx is not None:
    orig_params, mutable_params, unsharded_grads, padded_sharded_numels = ctx
    total_numel = sum(padded_sharded_numels)
    if (
        len(unsharded_grads) == 0
        and cast_output.ndim == 1
        and cast_output.numel() == total_numel
    ):
      _LEGACY_REDUCE_TLS.ctx = None
      specs: list[_ParamViewSpec] = []
      offset = 0
      for fsdp_param, span in zip(orig_params, padded_sharded_numels):
        specs.append(
            _ParamViewSpec(
                fsdp_param=fsdp_param,
                rel_offset=offset,
                numel=fsdp_param.sharded_size.numel(),
                shape=fsdp_param.sharded_size,
            )
        )
        offset += span
      flat_view = cast_output.reshape(-1)
      _accumulate_runs_on_device(orig_params, [(flat_view, specs)])
      # Clearing mutable_params makes foreach_reduce's subsequent non-strict
      # `zip(fsdp_params, padded_unsharded_sizes)` skip per-microbatch D2H.
      mutable_params.clear()
  return cast_output


def _get_or_alloc_host_buffer(
    anchor_param: Any,
    run_idx: int,
    numel: int,
    dtype: torch.dtype,
) -> torch.Tensor:
  """Returns a persistent pageable CPU buffer owned by ``anchor_param``."""
  buf_map: dict[int, torch.Tensor] | None = getattr(
      anchor_param, "_tpu_offload_host_grad_buffers", None
  )
  if buf_map is None:
    buf_map = {}
    setattr(anchor_param, "_tpu_offload_host_grad_buffers", buf_map)
  buf = buf_map.get(run_idx)
  if buf is None or buf.numel() != numel or buf.dtype != dtype:
    buf = torch.empty((numel,), dtype=dtype, device="cpu", pin_memory=False)
    buf_map[run_idx] = buf
  return buf


def materialize_host_grads() -> None:
  """Copies accumulated device gradients to CPU ``sharded_param.grad`` once."""
  if not _ACTIVE_GROUP_GRADS:
    return

  needs_accel_sync = any(
      run.flat_accum.device.type != "cpu"
      for runs in _ACTIVE_GROUP_GRADS.values()
      for run in runs
  )
  if needs_accel_sync and torch.accelerator.is_available():
    torch.accelerator.synchronize()

  for runs in _ACTIVE_GROUP_GRADS.values():
    for run_idx, run in enumerate(runs):
      any_existing_grad = any(
          spec.fsdp_param.sharded_param.grad is not None for spec in run.specs
      )
      if not any_existing_grad:
        host_flat = _get_or_alloc_host_buffer(
            run.specs[0].fsdp_param,
            run_idx,
            run.flat_accum.numel(),
            run.flat_accum.dtype,
        )
        host_flat.copy_(run.flat_accum, non_blocking=False)
        for spec in run.specs:
          grad_view = host_flat.narrow(
              0, spec.rel_offset, spec.numel
          ).view(spec.shape)
          spec.fsdp_param.sharded_param.grad = (
              spec.fsdp_param.to_sharded_dtensor(grad_view)
          )
      else:
        cpu_flat = run.flat_accum.to(device="cpu", non_blocking=False)
        for spec in run.specs:
          grad_view = cpu_flat.narrow(
              0, spec.rel_offset, spec.numel
          ).view(spec.shape)
          sharded_param = spec.fsdp_param.sharded_param
          if sharded_param.grad is None:
            sharded_param.grad = spec.fsdp_param.to_sharded_dtensor(grad_view)
          else:
            sharded_param.grad._local_tensor.add_(grad_view)  # pylint: disable=protected-access

  _ACTIVE_GROUP_GRADS.clear()
  _STATS.d2h_sync_calls += 1


def install_tpu_fsdp_cpu_offload(
    *,
    enable_param_cache: bool = True,
    enable_device_grad_accum: bool = True,
    cpu_offload_num_threads: int = 0,
    force_policy: bool = True,
) -> None:
  """Installs TPU FSDP2 CPU offload policy and runtime hooks (idempotent)."""
  global _ORIG_ALL_GATHER_INPUTS, _ORIG_RESET_SHARDED_PARAM
  global _ORIG_CAST_AND_VIEW_SHARDED_GRADS
  global _ORIG_FOREACH_REDUCE_COLLECTIVES, _ORIG_FOREACH_REDUCE_PARAM_GROUP
  global _ORIG_COLLECTIVES_TO_DTYPE, _RUNTIME_INSTALLED
  global _PARAM_CACHE_ENABLED, _DEVICE_GRAD_ACCUM_ENABLED

  install_tpu_cpu_offload_policy(force=force_policy)
  _PARAM_CACHE_ENABLED = enable_param_cache
  _DEVICE_GRAD_ACCUM_ENABLED = enable_device_grad_accum

  if cpu_offload_num_threads > 0:
    if torch.get_num_threads() != cpu_offload_num_threads:
      torch.set_num_threads(cpu_offload_num_threads)
      logger.info(
          "TPU FSDP2 CPU offload: set torch.num_threads=%d",
          cpu_offload_num_threads,
      )

  if not _RUNTIME_INSTALLED:
    orig_prop = FSDPParam.__dict__.get("all_gather_inputs")
    if isinstance(orig_prop, property):
      _ORIG_ALL_GATHER_INPUTS = orig_prop
      FSDPParam.all_gather_inputs = property(_patched_all_gather_inputs)  # type: ignore[assignment]
      _ORIG_RESET_SHARDED_PARAM = FSDPParam.reset_sharded_param
      FSDPParam.reset_sharded_param = _patched_reset_sharded_param  # type: ignore[method-assign]
    else:
      logger.warning(
          "FSDPParam.all_gather_inputs property not found; skipping TPU FSDP2"
          " parameter shard cache."
      )
      _PARAM_CACHE_ENABLED = False

    if hasattr(fsdp_collectives, "_cast_and_view_sharded_grads"):
      _ORIG_CAST_AND_VIEW_SHARDED_GRADS = (
          fsdp_collectives._cast_and_view_sharded_grads  # pylint: disable=protected-access
      )
      fsdp_collectives._cast_and_view_sharded_grads = (  # pylint: disable=protected-access
          _patched_cast_and_view_sharded_grads
      )
    elif hasattr(fsdp_collectives, "foreach_reduce") and hasattr(
        fsdp_collectives, "_to_dtype_if_needed"
    ):
      _ORIG_FOREACH_REDUCE_COLLECTIVES = fsdp_collectives.foreach_reduce
      _ORIG_COLLECTIVES_TO_DTYPE = (
          fsdp_collectives._to_dtype_if_needed  # pylint: disable=protected-access
      )
      fsdp_collectives.foreach_reduce = _patched_foreach_reduce_legacy  # type: ignore[assignment]
      fsdp_collectives._to_dtype_if_needed = (  # pylint: disable=protected-access
          _patched_to_dtype_if_needed_legacy
      )
      if hasattr(fsdp_param_group, "foreach_reduce"):
        _ORIG_FOREACH_REDUCE_PARAM_GROUP = fsdp_param_group.foreach_reduce
        fsdp_param_group.foreach_reduce = _patched_foreach_reduce_legacy  # type: ignore[assignment]
    else:
      logger.warning(
          "Unsupported torch.distributed.fsdp._fully_shard._fsdp_collectives"
          " layout; leaving per-microbatch D2H gradient offload on stock FSDP2."
      )
      _DEVICE_GRAD_ACCUM_ENABLED = False

    _RUNTIME_INSTALLED = True

  if not enable_param_cache:
    invalidate_param_cache()
  if not enable_device_grad_accum:
    clear_device_grads()


def uninstall_tpu_fsdp_cpu_offload() -> None:
  """Restores stock PyTorch FSDP2 methods and clears cached state."""
  global _ORIG_ALL_GATHER_INPUTS, _ORIG_RESET_SHARDED_PARAM
  global _ORIG_CAST_AND_VIEW_SHARDED_GRADS
  global _ORIG_FOREACH_REDUCE_COLLECTIVES, _ORIG_FOREACH_REDUCE_PARAM_GROUP
  global _ORIG_COLLECTIVES_TO_DTYPE
  global _RUNTIME_INSTALLED, _PARAM_CACHE_ENABLED, _DEVICE_GRAD_ACCUM_ENABLED

  clear_device_grads()
  invalidate_param_cache()
  uninstall_tpu_cpu_offload_policy()

  if _RUNTIME_INSTALLED:
    if _ORIG_ALL_GATHER_INPUTS is not None:
      FSDPParam.all_gather_inputs = _ORIG_ALL_GATHER_INPUTS  # type: ignore[assignment]
      _ORIG_ALL_GATHER_INPUTS = None
    if _ORIG_RESET_SHARDED_PARAM is not None:
      FSDPParam.reset_sharded_param = _ORIG_RESET_SHARDED_PARAM  # type: ignore[method-assign]
      _ORIG_RESET_SHARDED_PARAM = None
    if _ORIG_CAST_AND_VIEW_SHARDED_GRADS is not None:
      fsdp_collectives._cast_and_view_sharded_grads = (  # pylint: disable=protected-access
          _ORIG_CAST_AND_VIEW_SHARDED_GRADS
      )
      _ORIG_CAST_AND_VIEW_SHARDED_GRADS = None
    if _ORIG_FOREACH_REDUCE_COLLECTIVES is not None:
      fsdp_collectives.foreach_reduce = _ORIG_FOREACH_REDUCE_COLLECTIVES
      _ORIG_FOREACH_REDUCE_COLLECTIVES = None
    if _ORIG_FOREACH_REDUCE_PARAM_GROUP is not None:
      fsdp_param_group.foreach_reduce = _ORIG_FOREACH_REDUCE_PARAM_GROUP
      _ORIG_FOREACH_REDUCE_PARAM_GROUP = None
    if _ORIG_COLLECTIVES_TO_DTYPE is not None:
      fsdp_collectives._to_dtype_if_needed = (  # pylint: disable=protected-access
          _ORIG_COLLECTIVES_TO_DTYPE
      )
      _ORIG_COLLECTIVES_TO_DTYPE = None
    _RUNTIME_INSTALLED = False

  _PARAM_CACHE_ENABLED = False
  _DEVICE_GRAD_ACCUM_ENABLED = False


def is_tpu_fsdp_cpu_offload_installed() -> bool:
  """Returns True when the FSDP2 CPU offload runtime hooks are installed."""
  return _RUNTIME_INSTALLED


def configure_tpu_fsdp_cpu_offload_from_job_config(job_config: Any) -> bool:
  """Configures TPU FSDP2 CPU offload from ``job_config`` prior to parallelizing.

  Returns True if FSDP2 CPU offload is active, or False if disabled (including
  when ``parallelism.use_simple_fsdp`` is enabled, which does not support CPU
  offload).
  """
  training = getattr(job_config, "training", None)
  if training is None or not getattr(training, "enable_cpu_offload", False):
    return False

  parallelism = getattr(job_config, "parallelism", None)
  tpu_config = getattr(job_config, "tpu_config", None)
  use_simple_fsdp = bool(
      getattr(parallelism, "use_simple_fsdp", False)
      or getattr(tpu_config, "use_simple_fsdp", False)
  )
  if use_simple_fsdp:
    logger.warning(
        "CPU offload is not supported with SimpleFSDP; setting"
        " training.enable_cpu_offload=False."
    )
    training.enable_cpu_offload = False
    return False

  enable_param_cache = (
      getattr(tpu_config, "enable_cpu_offload_param_cache", True)
      if tpu_config is not None
      else True
  )
  enable_device_grad_accum = (
      getattr(tpu_config, "enable_cpu_offload_device_grad_accum", True)
      if tpu_config is not None
      else True
  )
  cpu_offload_num_threads = (
      getattr(tpu_config, "cpu_offload_num_threads", 0)
      if tpu_config is not None
      else 0
  )
  install_tpu_fsdp_cpu_offload(
      enable_param_cache=enable_param_cache,
      enable_device_grad_accum=enable_device_grad_accum,
      cpu_offload_num_threads=cpu_offload_num_threads,
      force_policy=True,
  )
  return True


def wire_trainer_cpu_offload(
    optimizers: Any,
    checkpointer: Any = None,
) -> None:
  """Idempotently hooks ``optimizers`` and ``checkpointer`` lifecycle methods.

  * ``optimizers.zero_grad`` clears any unmaterialized device gradients.
  * ``optimizers.step`` materializes device gradients to CPU beforehand (no-op
    if already materialized by ``_clip_grad_norm``) and invalidates the device
    parameter shard cache afterwards.
  * ``checkpointer.load`` invalidates the parameter shard cache and clears
    device gradients after loading weights.
  """
  if optimizers is not None and not getattr(
      optimizers, "_tpu_cpu_offload_wired", False
  ):
    orig_zero_grad = optimizers.zero_grad
    orig_step = optimizers.step

    @functools.wraps(orig_zero_grad)
    def _wrapped_zero_grad(*args: Any, **kwargs: Any) -> Any:
      clear_device_grads()
      return orig_zero_grad(*args, **kwargs)

    @functools.wraps(orig_step)
    def _wrapped_step(*args: Any, **kwargs: Any) -> Any:
      materialize_host_grads()
      try:
        return orig_step(*args, **kwargs)
      finally:
        invalidate_param_cache()

    optimizers.zero_grad = _wrapped_zero_grad
    optimizers.step = _wrapped_step
    setattr(optimizers, "_tpu_cpu_offload_wired", True)

  if (
      checkpointer is not None
      and hasattr(checkpointer, "load")
      and not getattr(checkpointer, "_tpu_cpu_offload_wired", False)
  ):
    orig_load = checkpointer.load

    @functools.wraps(orig_load)
    def _wrapped_load(*args: Any, **kwargs: Any) -> Any:
      result = orig_load(*args, **kwargs)
      invalidate_param_cache()
      clear_device_grads()
      return result

    checkpointer.load = _wrapped_load
    setattr(checkpointer, "_tpu_cpu_offload_wired", True)
