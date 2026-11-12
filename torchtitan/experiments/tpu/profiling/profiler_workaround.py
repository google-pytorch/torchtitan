# pylint: disable=protected-access
from typing import Any, Optional

import torch


import torchtitan.tools.logging as torchtitan_logging
import torchtitan.tools.profiler as torchtitan_profiler

_global_job_config = None


def init(job_config: Any) -> None:
  """Initialize the profiler workaround with the global job config."""
  global _global_job_config
  _global_job_config = job_config





# Save originals
_original_init = torchtitan_profiler.Profiler.__init__
_original_enter = torchtitan_profiler.Profiler.__enter__
_original_step = torchtitan_profiler.Profiler.step
_original_exit = torchtitan_profiler.Profiler.__exit__


def custom_init(
    self: torchtitan_profiler.Profiler, *args: Any, **kwargs: Any
) -> None:
  """Patched __init__ to initialize TPU-specific attributes."""
  _original_init(self, *args, **kwargs)
  self._patched_profiler: Optional[Any] = None
  self.tpu_step_num: int = self._global_step
  self.tpu_tracing: bool = False


def custom_enter(self: torchtitan_profiler.Profiler) -> torchtitan_profiler.Profiler:
  """Patched __enter__ to redirect to custom TPU profilers if configured."""
  cfg = self._config
  if not cfg.enable_profiling:
    return _original_enter(self)

  job_config = _global_job_config
  tpu_config = getattr(job_config, "tpu_config", None)
  use_jax = getattr(tpu_config, "use_jax_profiler", False)
  use_internal = False

  if use_jax or use_internal:
    patched_profiler = None

    if use_jax:
      try:
        # pylint: disable=g-import-not-at-top
        from torchtitan.experiments.tpu.profiling import jax_profiler
      except ImportError:
        jax_profiler = None

      if jax_profiler and jax_profiler.jax:
        patched_profiler = jax_profiler.JaxProfiler(self)
      else:
        torchtitan_logging.logger.warning(
            "JAX profiler requested, but JAX is not available."
        )

    if patched_profiler:
      self._patched_profiler = patched_profiler
      # Reset tpu_step_num to self._global_step to support .active() updates
      self.tpu_step_num = self._global_step
      self.tpu_tracing = False
      self.torch_profiler = None  # Disable native PyTorch profiler
      self.memory_profiler = self.build_memory_profiler(
          global_step=self._global_step,
          base_folder=self._base_folder,
          leaf_folder=self._leaf_folder,
      )
      torchtitan_logging.logger.info(
          "Custom TPU Profiler active (type: %s).",
          type(self._patched_profiler).__name__,
      )
      return self

  return _original_enter(self)


def custom_step(self: torchtitan_profiler.Profiler) -> None:
  """Patched step to manually schedule JAX/XProf profiling cycles."""
  if self._patched_profiler:
    self.tpu_step_num += 1
    cfg = self._config
    if cfg.profile_freq == 0:
      return
    cycle_step = self.tpu_step_num % cfg.profile_freq
    wait = cfg.profile_freq - cfg.profiler_warmup - cfg.profiler_active

    if cycle_step == wait + cfg.profiler_warmup:
      if not self.tpu_tracing:
        self._patched_profiler.start()
        self.tpu_tracing = True
    elif (
        self.tpu_tracing
        and cycle_step
        == (wait + cfg.profiler_warmup + cfg.profiler_active)
        % cfg.profile_freq
    ):
      if self.tpu_tracing:
        self._patched_profiler.stop()
        self.tpu_tracing = False

    # Also step the memory profiler if active
    if self.memory_profiler is not None:
      self.memory_profiler.step()
  else:
    _original_step(self)


def custom_exit(
    self: torchtitan_profiler.Profiler, exc_type: Any, exc_val: Any, exc_tb: Any
) -> None:
  """Patched __exit__ to ensure custom profilers are stopped."""
  if self._patched_profiler:
    if self.tpu_tracing:
      self._patched_profiler.stop()
      self.tpu_tracing = False
    # Clean up memory profiler context if active
    if self.memory_profiler is not None:
      if isinstance(exc_val, torch.OutOfMemoryError):
        self.memory_profiler.step(exit_ctx=True)
      self.memory_profiler = None
  else:
    _original_exit(self, exc_type, exc_val, exc_tb)


# Apply the patches
torchtitan_profiler.Profiler.__init__ = custom_init
torchtitan_profiler.Profiler.__enter__ = custom_enter
torchtitan_profiler.Profiler.step = custom_step
torchtitan_profiler.Profiler.__exit__ = custom_exit
