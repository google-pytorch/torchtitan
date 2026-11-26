"""Goodput event recording and monitoring (the global recorder and monitor).

See the package docstring for an overview. Public functions are re-exported by
`torchtitan.experiments.tpu.goodput`. The `maybe_*` names mark context managers
that may do nothing (goodput off).
"""

import contextlib
import datetime
import enum
import logging
import os
import time
from typing import Any, Callable, Iterator, Optional

import torch.distributed as dist
from torchtitan.experiments.tpu.goodput.config import GoodputConfig

logger = logging.getLogger(__name__)


class GoodputEvent(enum.Enum):
  """Recorder events; values match the library's `record_<value>_*_time`."""

  JOB = "job"
  TPU_INIT = "tpu_init"
  TRAINING_PREPARATION = "training_preparation"
  DATA_LOADING = "data_loading"


# Custom badput event types recorded by the graph trainer.
EVAL_BADPUT = "eval"

_recorder: Any = None
_job_name: str = ""
_preparation_open: bool = False
_warned_unknown_rank: bool = False
_warned_no_cloud_logger: bool = False


def _import_goodput():
  """Returns the library's `(goodput, monitoring)` modules, or None."""
  try:
    from ml_goodput_measurement import goodput  # pylint: disable=g-import-not-at-top
    from ml_goodput_measurement import monitoring  # pylint: disable=g-import-not-at-top
  except ImportError:
    return None
  except Exception as e:  # pylint: disable=broad-exception-caught
    # Installed but broken in this environment (e.g. a protobuf mismatch in its
    # Cloud Logging / Monitoring deps raises TypeError); disable goodput.
    logger.warning("Goodput: importing ml-goodput-measurement failed: %s", e)
    return None
  return goodput, monitoring


def _is_lead_process() -> bool:
  """Whether this process is global rank 0.

  Uses the process group once it exists. Before that, torchrun's `RANK` is
  used; if it is missing in a multi-process job the rank is unknown, so the
  process is not treated as the lead (avoids every rank writing events), and a
  warning is logged once.
  """
  global _warned_unknown_rank
  if dist.is_available() and dist.is_initialized():
    return dist.get_rank() == 0
  rank = os.environ.get("RANK")
  if rank is not None:
    return int(rank) == 0
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  if world_size <= 1:
    return True
  if not _warned_unknown_rank:
    _warned_unknown_rank = True
    logger.warning(
        "Goodput: RANK is unset and WORLD_SIZE=%d, so this process's rank is"
        " unknown; goodput is disabled for this run. Launch with torchrun or"
        " set RANK.",
        world_size,
    )
  return False


def _resolve_job_name(config: GoodputConfig) -> str:
  """Returns `--goodput.job-name`, or a timestamped default (with a warning)."""
  if config.job_name:
    return config.job_name
  name = "torchtitan_" + datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y%m%d%H%M%S"
  )
  logger.warning(
      "Goodput: --goodput.job-name is not set; using %s. Restarts of this run"
      " will be recorded as separate jobs.",
      name,
  )
  return name


def initialize(config: GoodputConfig) -> None:
  """Creates the global recorder on the lead process if recording is on.

  The library is imported only when recording is on. If it is missing or
  broken, or the recorder can't be created, a warning says why and every
  recording function stays a no-op.

  Args:
    config: The `--goodput.*` options.
  """
  global _recorder, _job_name, _preparation_open, _warned_no_cloud_logger
  _recorder = None
  _job_name = ""
  _preparation_open = False
  _warned_no_cloud_logger = False
  if not (config.enable_goodput_recording or config.enable_goodput_monitoring):
    return
  if not _is_lead_process():
    return
  _job_name = _resolve_job_name(config)
  if not config.enable_goodput_recording:
    return
  modules = _import_goodput()
  if modules is None:
    logger.warning(
        "Goodput: ml-goodput-measurement is unavailable; recording disabled."
    )
    return
  goodput, _ = modules
  try:
    _recorder = goodput.GoodputRecorder(
        job_name=_job_name,
        logger_name=f"goodput_{_job_name}",
        logging_enabled=True,
    )
    logger.info("Goodput: created GoodputRecorder for job %s", _job_name)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Goodput: could not create recorder: %s", e)


def enabled() -> bool:
  """Whether this process records goodput events."""
  return _recorder is not None


def shutdown() -> None:
  """Flushes buffered events and drops the global recorder."""
  global _recorder, _preparation_open
  if _recorder is not None:
    try:
      _recorder.flush()
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Goodput: flush failed: %s", e)
  _recorder = None
  _preparation_open = False


def _record(method_name: str, *args, **kwargs) -> None:
  """Calls `recorder.<method_name>` if recording; never raises."""
  if _recorder is None:
    return
  fn = getattr(_recorder, method_name, None)
  if fn is None:
    return
  try:
    fn(*args, **kwargs)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Goodput: %s failed: %s", method_name, e)


def record_start(event: GoodputEvent) -> None:
  """Records the start time of `event` (no-op if not recording)."""
  _record(f"record_{event.value}_start_time")


def record_end(event: GoodputEvent) -> None:
  """Records the end time of `event` (no-op if not recording)."""
  _record(f"record_{event.value}_end_time")


@contextlib.contextmanager
def maybe_record(event: GoodputEvent) -> Iterator[None]:
  """Records `event` around the block, if recording.

  The end is recorded only if the block doesn't raise, so a failed phase is
  not counted as completed.

  Args:
    event: The event to record.

  Yields:
    None.
  """
  record_start(event)
  yield
  record_end(event)


@contextlib.contextmanager
def _maybe_record_custom_badput(event_type: str) -> Iterator[None]:
  """Records a custom badput interval around the block (end only on success)."""
  _record(
      "record_custom_badput_event_start_time",
      custom_badput_event_type=event_type,
  )
  yield
  _record(
      "record_custom_badput_event_end_time",
      custom_badput_event_type=event_type,
  )


def _disable_if_not_lead_process() -> None:
  """Drops the recorder if the process group says this is not rank 0.

  `initialize()` runs before the process group exists and picks the lead
  process from torchrun's `RANK`, so non-lead processes have no recorder and
  never get here with one. This only fires if `RANK` disagreed with the
  process group's rank.
  """
  global _recorder
  if _recorder is not None and not _is_lead_process():
    logger.warning(
        "Goodput: RANK=%s but the process group rank is %d; disabling this"
        " process's recorder.",
        os.environ.get("RANK"),
        dist.get_rank(),
    )
    _recorder = None


def start_training_preparation() -> None:
  """Starts recording training preparation; call once the process group exists.

  Rechecks the lead process against the process group first. Preparation
  covers model build, parallelization and compilation setup; it ends at
  `end_training_preparation()`, which runs when a checkpoint load starts or at
  the first `record_step_start()`.
  """
  global _preparation_open
  _disable_if_not_lead_process()
  if _recorder is None:
    return
  record_start(GoodputEvent.TRAINING_PREPARATION)
  _preparation_open = True


def end_training_preparation() -> None:
  """Ends training preparation if it is open; later calls are no-ops."""
  global _preparation_open
  if _preparation_open:
    _preparation_open = False
    record_end(GoodputEvent.TRAINING_PREPARATION)


def record_step_start(step: int) -> None:
  """Records the start of training step `step`.

  The first call also ends training preparation, if it is still open.

  Args:
    step: The training step about to run.
  """
  end_training_preparation()
  _record("record_step_start_time", step)


def record_data_loading(iterator: Iterator[Any]) -> Iterator[Any]:
  """Wraps a batch iterator, recording each blocking `next()` as data loading.

  Exceptions from the wrapped iterator (e.g. dataloader exhaustion) propagate
  unchanged, without a data loading end entry.

  Args:
    iterator: The batch iterator to wrap.

  Yields:
    The batches of `iterator`, unchanged.
  """
  while True:
    record_start(GoodputEvent.DATA_LOADING)
    try:
      batch = next(iterator)
    except StopIteration:
      # The unpaired data_loading_start is harmless, since the
      # calculator ignores a start without an end.
      return
    record_end(GoodputEvent.DATA_LOADING)
    yield batch


def wrap_with_custom_badput(
    fn: Callable[..., Any],
    event_type: str,
    should_record: Optional[Callable[..., bool]] = None,
) -> Callable[..., Any]:
  """Returns `fn` wrapped in a custom badput interval.

  Args:
    fn: The function to wrap.
    event_type: The custom badput event type, e.g. `EVAL_BADPUT`.
    should_record: If given, `should_record(*args, **kwargs)` decides per call
      whether the call does blocking work worth recording.

  Returns:
    A wrapper that calls `fn` unchanged when not recording.
  """

  def wrapper(*args, **kwargs):
    if _recorder is None or (
        should_record is not None and not should_record(*args, **kwargs)
    ):
      return fn(*args, **kwargs)
    with _maybe_record_custom_badput(event_type):
      return fn(*args, **kwargs)

  return wrapper


def write_log_entry(entry: dict[str, Any]) -> None:
  """Writes a raw entry to the goodput log; never raises.

  For the checkpointer only: `GoodputRecorder` has no public checkpoint API
  yet (its `record_checkpoint_progress` is a stub), so checkpoint entries go
  through its private Cloud Logging writer. Switch to
  `record_checkpoint_save_stats` / `record_checkpoint_restore_stats` once the
  library has them.

  Args:
    entry: The log entry; `job_name` is added.
  """
  global _warned_no_cloud_logger
  if _recorder is None:
    return
  cloud_logger = getattr(_recorder, "_cloud_logger", None)
  if cloud_logger is None:
    # The library's private writer moved or logging is off: checkpoint entries
    # (both checkpoint buckets) are dropped, so say so once.
    if not _warned_no_cloud_logger:
      _warned_no_cloud_logger = True
      logger.warning(
          "Goodput: GoodputRecorder has no `_cloud_logger`; checkpoint save"
          " and restore badput are not recorded."
      )
    return
  try:
    cloud_logger.write_cloud_logging_entry({"job_name": _job_name, **entry})
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Goodput: writing a checkpoint entry failed: %s", e)


@contextlib.contextmanager
def maybe_monitor_goodput(
    config: GoodputConfig, default_tensorboard_dir: str
) -> Iterator[None]:
  """Runs the goodput uploader(s) around the block on the lead process.

  Starts the cumulative uploader (with badput breakdown and step time
  deviation) and, if enabled, the rolling-window uploader. On exit each is
  stopped, which does a final flush. Failures to start or stop are logged and
  swallowed; exceptions from the block propagate. A no-op unless
  `--goodput.enable-goodput-monitoring` is set.

  Args:
    config: The `--goodput.*` options.
    default_tensorboard_dir: Used when `config.tensorboard_dir` is empty.

  Yields:
    None.
  """
  if not config.enable_goodput_monitoring or not _is_lead_process():
    yield
    return
  monitor = None
  try:
    monitor = _construct_monitor(config, default_tensorboard_dir)
    if monitor is not None:
      monitor.start_goodput_uploader()
      logger.info(
          "Goodput: started cumulative goodput upload to TensorBoard and GCM"
          " for job %s.",
          _job_name,
      )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Goodput: could not start goodput monitoring: %s", e)
    monitor = None
  try:
    with _maybe_monitor_rolling_window(monitor, config):
      try:
        yield
      finally:
        if monitor is not None and config.final_flush_delay_seconds > 0:
          # Opt-in (see GoodputConfig.final_flush_delay_seconds). The recorder
          # was flushed just before this (see train.py); give Cloud Logging
          # time to make the last entries readable.
          time.sleep(config.final_flush_delay_seconds)
  finally:
    if monitor is not None:
      try:
        monitor.stop_goodput_uploader()
        logger.info(
            "Goodput: flushed final metrics and stopped cumulative goodput"
            " monitoring."
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Goodput: could not stop goodput monitoring: %s", e)


def _construct_monitor(config: GoodputConfig, default_tensorboard_dir: str):
  """Returns a `GoodputMonitor` for this job, or None without the library."""
  modules = _import_goodput()
  if modules is None:
    logger.warning(
        "Goodput: ml-goodput-measurement is unavailable; monitoring disabled."
    )
    return None
  _, monitoring = modules
  job_name = _job_name or _resolve_job_name(config)
  gcp_options = monitoring.GCPOptions(
      enable_gcp_goodput_metrics=config.enable_gcp_goodput_metrics,
  )
  # Step time deviation is computed and uploaded by the cumulative and
  # rolling-window uploaders; the separate step-deviation uploader (and its
  # include_step_deviation / step_deviation_interval_seconds options) is
  # deprecated and not used.
  return monitoring.GoodputMonitor(
      job_name=job_name,
      logger_name=f"goodput_{job_name}",
      tensorboard_dir=config.tensorboard_dir or default_tensorboard_dir,
      upload_interval=config.upload_interval_seconds,
      monitoring_enabled=True,
      include_badput_breakdown=True,
      configured_ideal_step_time=config.configured_ideal_step_time,
      gcp_options=gcp_options,
  )


@contextlib.contextmanager
def _maybe_monitor_rolling_window(monitor, config: GoodputConfig):
  """Runs the rolling-window uploader around the block, if enabled."""
  if monitor is None or not config.enable_rolling_window_goodput:
    yield
    return
  started = False
  try:
    monitor.start_rolling_window_goodput_uploader(
        list(config.rolling_windows_seconds)
    )
    started = True
    logger.info(
        "Goodput: started rolling window goodput upload to TensorBoard and GCM."
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Goodput: could not start rolling window uploader: %s", e)
  try:
    yield
  finally:
    if started:
      try:
        monitor.stop_rolling_window_goodput_uploader()
        logger.info(
            "Goodput: flushed final metrics and stopped rolling window goodput"
            " monitoring."
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Goodput: could not stop rolling window uploader: %s", e)
