"""JAX profiling utilities for the pure-JAX experiment.

Mirrors torchtitan/experiments/tpu/jax_profiling.py but adapted for the
single-process JAX case (no torch.distributed dependency).

Usage in the training loop::

    with maybe_enable_profiling(job_config.profiling, base_folder=...) as profiler:
        for step in range(steps):
            # ... training step ...
            if profiler is not None:
                profiler.step()
"""

import contextlib
import logging
import os

import jax

import torchtitan.tools.profiler

ProfilingConfig = torchtitan.tools.profiler.Profiler.Config
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def maybe_enable_profiling(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
):
    """Context manager to conditionally enable JAX profiling based on config.

    Yields a ``JaxProfiler`` when profiling is enabled, or ``None`` otherwise.
    Call ``profiler.step()`` once per training step inside the context to let
    the profiler manage start/stop of JAX traces according to ``profile_freq``,
    ``profiler_warmup``, and ``profiler_active``.
    """
    enable_profiling = profiling_config.enable_profiling

    if not enable_profiling:
        yield None
        return

    trace_dir = os.path.join(base_folder, profiling_config.save_traces_folder)
    profile_freq = profiling_config.profile_freq
    warmup = profiling_config.profiler_warmup
    active = profiling_config.profiler_active

    class JaxProfiler:
        """Manages periodic JAX trace start/stop within a training loop."""

        def __init__(self, step_num: int) -> None:
            self.step_num = step_num
            self.tracing = False

        def step(self):
            """Advance the profiler by one step, starting or stopping a trace.

            A trace warmup begins when ``(step_num + warmup) % profile_freq == 0``
            (matching the TPU jax_profiling convention); the trace stops after
            ``active`` more steps.
            """
            self.step_num += 1

            # Active phase starts (warmup begins now, trace data captured for
            # next `active` steps after warmup).
            if self.step_num > 0 and (self.step_num + warmup) % profile_freq == 0:
                curr_trace_dir = os.path.join(
                    trace_dir, f"iteration_{self.step_num}"
                )
                os.makedirs(curr_trace_dir, exist_ok=True)
                logger.info(
                    'Starting JAX profiler warmup at step %d, trace starts at step %d',
                    self.step_num,
                    self.step_num + warmup,
                )
                jax.profiler.start_trace(curr_trace_dir)
                self.tracing = True

            # Active phase ends.
            elif self.tracing and (self.step_num - active) % profile_freq == 0:
                logger.info('Stopping JAX profiler before step %d', self.step_num)
                jax.profiler.stop_trace()
                self.tracing = False

    logger.info('JAX profiling active. Traces will be saved at %s', trace_dir)
    os.makedirs(trace_dir, exist_ok=True)

    profiler = JaxProfiler(global_step)
    try:
        yield profiler
    finally:
        if profiler.tracing:
            logger.info('Stopping JAX profiler at end of training.')
            jax.profiler.stop_trace()
