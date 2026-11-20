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
   In addition, stock ``FSDPParam.all_gather_inputs`` issues
   ``sharded_param_data.to(device, non_blocking=True)`` on
   ``all_gather_copy_in_stream``, which is not ordered by
   ``all_gather_stream.wait_stream(...)`` on multi-rank ``torch_tpu`` and can
   read uninitialized HBM (`Inf`/`NaN`). :class:`TPUCPUOffloadPolicy` defaults
   ``pin_memory=False``, and ``_patched_all_gather_inputs`` performs synchronous
   ``non_blocking=False`` host-to-device staging regardless of whether shard
   caching is enabled.
2. ``FSDPParam.all_gather_inputs`` copies each FSDP unit's fp32 master shard
   from CPU to device and casts it to ``param_dtype`` on *every* microbatch
   forward and post-reshard backward unshard, even though master weights only
   change on ``optimizer.step()``. Opting into
   ``tpu_config.enable_cpu_offload_param_cache`` caches the cast device shard per
   ``FSDPParam`` (invalidated on ``optimizer.step()``, ``optimizer.zero_grad()``,
   checkpoint load, and ``reset_sharded_param()``), reducing H2D shard transfers
   to once per step at persistent HBM cost
   ``~sharded_param_numel * sizeof(param_dtype)``.
3. ``foreach_reduce`` copies each parameter's reduce-scattered gradient
   synchronously from device to CPU on *every* microbatch. Opting into
   ``tpu_config.enable_cpu_offload_device_grad_accum`` accumulates the
   cast/viewed sharded gradients on device across microbatches and copies once
   per optimizer step into persistent pageable host buffers, removing
   per-microbatch D2H stalls at persistent HBM cost
   ``~sharded_param_numel * sizeof(reduce_dtype)``.
4. Under ``torchrun`` (which sets ``OMP_NUM_THREADS=1`` when unset), host-side
   ``clip_grad_norm_`` and ``AdamW.step()`` run single-threaded over offloaded
   master weights unless ``torch.set_num_threads`` is configured via
   ``tpu_config.cpu_offload_num_threads``.
"""

from collections.abc import Callable, Iterable
import dataclasses
import functools
import sys
from typing import Any
import weakref

import torch
from torch import nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as fsdp_collectives
from torch.distributed.fsdp._fully_shard._fsdp_param import (
    FSDPParam,
    ShardedState,
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
  """Metadata locating one parameter's sharded gradient inside a flat buffer."""

  fsdp_param: Any
  rel_offset: int
  numel: int
  shape: torch.Size


@dataclasses.dataclass
class _FlatGradAccum:
  """Contiguous 1D device buffer accumulating uniform unpadded sharded grads."""

  flat_accum: torch.Tensor
  specs: list[_ParamViewSpec]


@dataclasses.dataclass
class _PerParamGradAccum:
  """Per-parameter device buffers accumulated via ``torch._foreach_add_``."""

  fsdp_params: list[Any]
  grad_accums: list[torch.Tensor]


_GradAccumState = _FlatGradAccum | _PerParamGradAccum

_ORIG_CPU_OFFLOAD_POLICIES: dict[str, Any] = {}
_ORIG_ALL_GATHER_INPUTS: property | None = None
_ORIG_RESET_SHARDED_PARAM: Callable[..., Any] | None = None
_ORIG_CAST_AND_VIEW_SHARDED_GRADS: Callable[..., Any] | None = None

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
_ACTIVE_GROUP_GRADS: dict[tuple[int, ...], _GradAccumState] = {}
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


def verify_tpu_fsdp_cpu_offload_no_pin_memory(
    model_parts: Iterable[nn.Module],
) -> None:
  """Raises ``RuntimeError`` if any offloaded ``FSDPParam`` has ``pin_memory=True``.

  Guards against parallelize modules that import PyTorch's ``CPUOffloadPolicy``
  directly without going through ``TPUCPUOffloadPolicy(pin_memory=False)``.
  """
  for part in model_parts:
    for mod in part.modules():
      if not isinstance(mod, FSDPModule):
        continue
      fsdp_state = mod._get_fsdp_state()  # pylint: disable=protected-access
      param_group = getattr(fsdp_state, "_fsdp_param_group", None)
      if param_group is None:
        continue
      for fsdp_param in param_group.fsdp_params:
        if fsdp_param.offload_to_cpu and fsdp_param.pin_memory:
          fqn = getattr(fsdp_param, "_param_fqn", "<unknown>")
          raise RuntimeError(
              f"FSDPParam {fqn!r} on {type(mod).__name__} was initialized with"
              " CPUOffloadPolicy(pin_memory=True) on TPU, which is unsupported"
              " (b/499075866). Use TPUCPUOffloadPolicy(pin_memory=False) or"
              " configure_tpu_fsdp_cpu_offload_from_job_config()."
          )


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


def _copy_and_cast_offloaded_shard(self: FSDPParam) -> list[torch.Tensor]:
  """Stages ``_sharded_param_data`` to ``self.device`` with ``non_blocking=False``.

  Stock ``FSDPParam.all_gather_inputs`` uses ``non_blocking=True`` inside
  ``foreach_all_gather``'s ``all_gather_copy_in_stream``. On multi-rank
  ``torch_tpu``, ``all_gather_stream.wait_stream(all_gather_copy_in_stream)``
  does not synchronize async host-to-device DMA from pageable host memory,
  corrupting unsharded weights with ``Inf``/``NaN``. Synchronous H2D staging
  avoids that cross-stream race in both default and shard-cached modes.
  """
  sharded_param_data = self._sharded_param_data.to(  # pylint: disable=protected-access
      self.device, non_blocking=False
  )
  if (
      self.param_dtype is not None
      and sharded_param_data.dtype != self.param_dtype
  ):
    sharded_param_data = sharded_param_data.to(dtype=self.param_dtype)
  return [sharded_param_data]


def _patched_all_gather_inputs(self: FSDPParam) -> list[torch.Tensor]:
  assert _ORIG_ALL_GATHER_INPUTS is not None
  orig_fget = _ORIG_ALL_GATHER_INPUTS.fget
  assert orig_fget is not None
  if (
      not self.offload_to_cpu
      or self.sharded_state != ShardedState.SHARDED
      or hasattr(self._sharded_local_tensor, "fsdp_pre_all_gather")  # pylint: disable=protected-access
  ):
    return orig_fget(self)

  if not _PARAM_CACHE_ENABLED:
    return _copy_and_cast_offloaded_shard(self)

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

  ag_inputs = _copy_and_cast_offloaded_shard(self)
  setattr(
      self,
      "_tpu_offload_param_cache",
      (_PARAM_CACHE_VERSION, list(ag_inputs)),
  )
  _CACHED_FSDP_PARAMS.add(self)
  _STATS.param_cache_misses += 1
  return list(ag_inputs)


def _patched_reset_sharded_param(self: FSDPParam) -> None:
  setattr(self, "_tpu_offload_param_cache", None)
  assert _ORIG_RESET_SHARDED_PARAM is not None
  _ORIG_RESET_SHARDED_PARAM(self)


def _is_device_grad_accum_eligible(fsdp_params: list[FSDPParam]) -> bool:
  return bool(
      _DEVICE_GRAD_ACCUM_ENABLED
      and fsdp_params
      and all(p.offload_to_cpu for p in fsdp_params)
      and not any(
          getattr(p.sharded_param, "_post_accumulate_grad_hooks", None)
          for p in fsdp_params
      )
  )


def _foreach_add_by_dtype(
    accums: list[torch.Tensor],
    grads: list[torch.Tensor],
) -> None:
  """Accumulates ``grads`` into ``accums`` in-place without ``torch.cat``."""
  buckets: dict[torch.dtype, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
  for acc, grad in zip(accums, grads):
    bucket = buckets.get(acc.dtype)
    if bucket is None:
      bucket = ([], [])
      buckets[acc.dtype] = bucket
    bucket[0].append(acc)
    bucket[1].append(grad)
  for bucket_accums, bucket_grads in buckets.values():
    torch._foreach_add_(bucket_accums, bucket_grads)


def _accumulate_sharded_grads_on_device(
    reduce_output: torch.Tensor,
    fsdp_params: list[FSDPParam],
    sharded_grads: list[torch.Tensor],
) -> None:
  if not fsdp_params:
    return

  group_key = tuple(id(p) for p in fsdp_params)
  first_dtype = sharded_grads[0].dtype
  total_grad_numel = sum(g.numel() for g in sharded_grads)
  can_view_reduce_output_directly = (
      reduce_output.dtype == first_dtype
      and total_grad_numel == reduce_output.numel()
      and all(g.dtype == first_dtype for g in sharded_grads)
      and not any(
          getattr(p, "needs_nonzero_dim_padding", False) for p in fsdp_params
      )
  )

  existing = _ACTIVE_GROUP_GRADS.get(group_key)
  if can_view_reduce_output_directly:
    flat_view = reduce_output.reshape(-1)
    if isinstance(existing, _FlatGradAccum) and (
        existing.flat_accum.numel() == flat_view.numel()
        and existing.flat_accum.dtype == flat_view.dtype
    ):
      existing.flat_accum.add_(flat_view)
      return
    if existing is not None:
      materialize_host_grads()
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
    _ACTIVE_GROUP_GRADS[group_key] = _FlatGradAccum(
        flat_accum=flat_view.clone(),
        specs=specs,
    )
    return

  if isinstance(existing, _PerParamGradAccum) and (
      len(existing.grad_accums) == len(sharded_grads)
      and all(
          a.shape == g.shape and a.dtype == g.dtype
          for a, g in zip(existing.grad_accums, sharded_grads)
      )
  ):
    _foreach_add_by_dtype(existing.grad_accums, sharded_grads)
    return

  if existing is not None:
    materialize_host_grads()
  _ACTIVE_GROUP_GRADS[group_key] = _PerParamGradAccum(
      fsdp_params=list(fsdp_params),
      grad_accums=[g.detach().clone() for g in sharded_grads],
  )


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
  _accumulate_sharded_grads_on_device(reduce_output, fsdp_params, sharded_grads)

  # Returning an empty list makes foreach_reduce's non-strict
  # `zip(fsdp_params, sharded_grads)` skip the per-microbatch D2H copy loop
  # while still recording `post_reduce_event` cleanly.
  return []


def _get_or_alloc_host_buffer(
    anchor_param: Any,
    buf_key: int,
    shape: torch.Size | tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
  """Returns a persistent pageable CPU buffer owned by ``anchor_param``."""
  buf_map: dict[int, torch.Tensor] | None = getattr(
      anchor_param, "_tpu_offload_host_grad_buffers", None
  )
  if buf_map is None:
    buf_map = {}
    setattr(anchor_param, "_tpu_offload_host_grad_buffers", buf_map)
  target_shape = torch.Size(shape)
  buf = buf_map.get(buf_key)
  if buf is None or buf.shape != target_shape or buf.dtype != dtype:
    buf = torch.empty(target_shape, dtype=dtype, device="cpu", pin_memory=False)
    buf_map[buf_key] = buf
  return buf


def _state_uses_accelerator(state: _GradAccumState) -> bool:
  if isinstance(state, _FlatGradAccum):
    return state.flat_accum.device.type != "cpu"
  return any(t.device.type != "cpu" for t in state.grad_accums)


def materialize_host_grads() -> None:
  """Copies accumulated device gradients to CPU ``sharded_param.grad`` once."""
  if not _ACTIVE_GROUP_GRADS:
    return

  needs_accel_sync = any(
      _state_uses_accelerator(state) for state in _ACTIVE_GROUP_GRADS.values()
  )
  if needs_accel_sync and torch.accelerator.is_available():
    torch.accelerator.synchronize()

  for state in _ACTIVE_GROUP_GRADS.values():
    if isinstance(state, _FlatGradAccum):
      any_existing_grad = any(
          spec.fsdp_param.sharded_param.grad is not None for spec in state.specs
      )
      if not any_existing_grad:
        host_flat = _get_or_alloc_host_buffer(
            state.specs[0].fsdp_param,
            0,
            (state.flat_accum.numel(),),
            state.flat_accum.dtype,
        )
        host_flat.copy_(state.flat_accum, non_blocking=False)
        for spec in state.specs:
          grad_view = host_flat.narrow(
              0, spec.rel_offset, spec.numel
          ).view(spec.shape)
          spec.fsdp_param.sharded_param.grad = (
              spec.fsdp_param.to_sharded_dtensor(grad_view)
          )
      else:
        cpu_flat = state.flat_accum.to(device="cpu", non_blocking=False)
        for spec in state.specs:
          grad_view = cpu_flat.narrow(
              0, spec.rel_offset, spec.numel
          ).view(spec.shape)
          sharded_param = spec.fsdp_param.sharded_param
          if sharded_param.grad is None:
            sharded_param.grad = spec.fsdp_param.to_sharded_dtensor(grad_view)
          else:
            sharded_param.grad._local_tensor.add_(grad_view)  # pylint: disable=protected-access
    else:
      for fsdp_param, dev_grad in zip(state.fsdp_params, state.grad_accums):
        sharded_param = fsdp_param.sharded_param
        if sharded_param.grad is None:
          host_buf = _get_or_alloc_host_buffer(
              fsdp_param,
              1,
              dev_grad.shape,
              dev_grad.dtype,
          )
          host_buf.copy_(dev_grad, non_blocking=False)
          sharded_param.grad = fsdp_param.to_sharded_dtensor(host_buf)
        else:
          cpu_grad = dev_grad.to(device="cpu", non_blocking=False)
          sharded_param.grad._local_tensor.add_(cpu_grad)  # pylint: disable=protected-access

  _ACTIVE_GROUP_GRADS.clear()
  _STATS.d2h_sync_calls += 1


def install_tpu_fsdp_cpu_offload(
    *,
    enable_param_cache: bool = False,
    enable_device_grad_accum: bool = False,
    cpu_offload_num_threads: int = 0,
    force_policy: bool = True,
) -> None:
  """Installs TPU FSDP2 CPU offload policy and runtime hooks (idempotent)."""
  global _ORIG_ALL_GATHER_INPUTS, _ORIG_RESET_SHARDED_PARAM
  global _ORIG_CAST_AND_VIEW_SHARDED_GRADS, _RUNTIME_INSTALLED
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
    else:
      logger.warning(
          "torch.distributed.fsdp._fully_shard._fsdp_collectives lacks"
          " _cast_and_view_sharded_grads; leaving per-microbatch D2H gradient"
          " offload on stock FSDP2."
      )
      _DEVICE_GRAD_ACCUM_ENABLED = False

    _RUNTIME_INSTALLED = True
  else:
    if _ORIG_ALL_GATHER_INPUTS is None:
      _PARAM_CACHE_ENABLED = False
    if _ORIG_CAST_AND_VIEW_SHARDED_GRADS is None:
      _DEVICE_GRAD_ACCUM_ENABLED = False

  if not _PARAM_CACHE_ENABLED:
    invalidate_param_cache()
  if not _DEVICE_GRAD_ACCUM_ENABLED:
    clear_device_grads()


def uninstall_tpu_fsdp_cpu_offload() -> None:
  """Restores stock PyTorch FSDP2 methods and clears cached state."""
  global _ORIG_ALL_GATHER_INPUTS, _ORIG_RESET_SHARDED_PARAM
  global _ORIG_CAST_AND_VIEW_SHARDED_GRADS
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
      getattr(tpu_config, "enable_cpu_offload_param_cache", False)
      if tpu_config is not None
      else False
  )
  enable_device_grad_accum = (
      getattr(tpu_config, "enable_cpu_offload_device_grad_accum", False)
      if tpu_config is not None
      else False
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

  * ``optimizers.zero_grad`` clears any unmaterialized device gradients and
    defensively invalidates cached parameter shards so stale shards can never
    persist across steps even if an inner optimizer was stepped directly.
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
      invalidate_param_cache()
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
