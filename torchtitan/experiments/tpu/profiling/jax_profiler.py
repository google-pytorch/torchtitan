import os as os
import torch
import torchtitan.tools.logging as torchtitan_logging
import torchtitan.tools.profiler as profiler_tool

try:
  import jax
except ImportError:
  jax = None

logger = torchtitan_logging.logger


class JaxProfiler:
  """Wrapper for JAX profiler."""

  def __init__(self, profiler_instance: profiler_tool.Profiler):
    self.prof = profiler_instance

  def start(self) -> None:
    if not jax:
      return
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
      # pytype: disable=attribute-error
      # pylint: disable=protected-access
      save_folder = self.prof._config.save_traces_folder
      if "://" in save_folder or os.path.isabs(save_folder):
        trace_dir = save_folder
      else:
        trace_dir = os.path.join(self.prof._base_folder, save_folder)
      curr_trace_dir = os.path.join(
          trace_dir,
          f"iteration_{self.prof.tpu_step_num}",
          f"rank_{torch.distributed.get_rank()}",
          self.prof._leaf_folder,
      )
      # pylint: enable=protected-access
      # pytype: enable=attribute-error
      logger.info(
          "Starting JAX profiler warmup at step %d. Trace dir: %s",
          # pytype: disable=attribute-error
          self.prof.tpu_step_num,
          # pytype: enable=attribute-error
          curr_trace_dir,
      )
      if "://" not in curr_trace_dir:
        os.makedirs(curr_trace_dir, exist_ok=True)
      jax.profiler.start_trace(curr_trace_dir)

  def stop(self) -> None:
    if jax:
      rank = int(os.environ.get("RANK", 0))
      if rank == 0:
        logger.info(
            "Stopping JAX profiler at step %d",
            # pytype: disable=attribute-error
            self.prof.tpu_step_num,
            # pytype: enable=attribute-error
        )
        jax.profiler.stop_trace()
