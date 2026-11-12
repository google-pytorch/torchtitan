"""Goodput recording and monitoring for TorchTitan on TPU.

Wires the ML Goodput Measurement library (`ml-goodput-measurement`) into the
TPU graph trainer. The library has two halves:

* Recording: a `GoodputRecorder` on the lead process writes start/end
  timestamps (job, TPU init, training preparation, data loading, steps and
  custom badput events) to the Cloud Logging log `goodput_<job_name>`.
  `GoodputCheckpointer` adds PyTorch DCP checkpoint save/restore entries in the
  shape Orbax writes, so they land in the library's native
  `UNPRODUCTIVE_CHECKPOINT_SAVE_TIME` / `_RESTORE_TIME` buckets.
* Monitoring: a `GoodputMonitor` on the lead process reads that log in a
  background process, computes goodput/badput and uploads them to TensorBoard
  and Google Cloud Monitoring (`compute.googleapis.com/workload/*`).

Both are off by default (`--goodput.enable-goodput-recording`,
`--goodput.enable-goodput-monitoring`), because monitoring starts background
GCP traffic. Everything here is a no-op when goodput is disabled, when the
library is not installed, or on non-lead processes, and a goodput failure is
logged and swallowed instead of interrupting training.
"""

import contextlib
import datetime
import enum
import os
import time
from typing import Any, Callable, Iterator, Optional

import torch.distributed as dist
from torchtitan.experiments.tpu import goodput_config
from torchtitan.tools.logging import logger

# Re-exported so callers can keep using `goodput.GoodputConfig`.
GoodputConfig = goodput_config.GoodputConfig


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
_warned_unknown_rank: bool = False


def _import_goodput():
  """Returns the library's `(goodput, monitoring)` modules, or None."""
  try:
    from ml_goodput_measurement import goodput  # pylint: disable=g-import-not-at-top
    from ml_goodput_measurement import monitoring  # pylint: disable=g-import-not-at-top
  except ImportError:
    return None
  return goodput, monitoring


def is_lead_process() -> bool:
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
        " unknown before the process group starts; it will not record or"
        " monitor goodput until then.",
        world_size,
    )
  return False


def resolve_job_name(config: GoodputConfig) -> str:
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
  """Creates the global recorder on the lead process if recording is on."""
  global _recorder, _job_name
  _recorder = None
  _job_name = ""
  if not (config.enable_goodput_recording or config.enable_goodput_monitoring):
    return
  if not is_lead_process():
    return
  _job_name = resolve_job_name(config)
  if not config.enable_goodput_recording:
    return
  modules = _import_goodput()
  if modules is None:
    logger.warning(
        "Goodput: ml-goodput-measurement is not installed; recording disabled."
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


def get_job_name() -> str:
  return _job_name


def enabled() -> bool:
  """Whether this process records goodput events."""
  return _recorder is not None


def verify_lead_process() -> None:
  """Drops the recorder if the process group says this is not rank 0."""
  global _recorder
  if _recorder is not None and not is_lead_process():
    logger.warning(
        "Goodput: rank %d is not the lead process; disabling its recorder.",
        dist.get_rank(),
    )
    _recorder = None


def shutdown() -> None:
  """Flushes buffered events and drops the global recorder."""
  global _recorder
  if _recorder is not None:
    try:
      _recorder.flush()
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Goodput: flush failed: %s", e)
  _recorder = None


def record(method_name: str, *args, **kwargs) -> None:
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
  record(f"record_{event.value}_start_time")


def record_end(event: GoodputEvent) -> None:
  record(f"record_{event.value}_end_time")


@contextlib.contextmanager
def maybe_record(event: GoodputEvent) -> Iterator[None]:
  """Records `event` start, and its end only if the block doesn't raise."""
  record_start(event)
  yield
  record_end(event)


@contextlib.contextmanager
def maybe_record_custom_badput(event_type: str) -> Iterator[None]:
  """Records a custom badput interval around the block (end only on success)."""
  record(
      "record_custom_badput_event_start_time",
      custom_badput_event_type=event_type,
  )
  yield
  record(
      "record_custom_badput_event_end_time",
      custom_badput_event_type=event_type,
  )


def record_step_start(step: int) -> None:
  record("record_step_start_time", step)


def record_data_loading(iterator: Iterator[Any]) -> Iterator[Any]:
  """Wraps a batch iterator, recording each blocking `next()` as data loading.

  Exceptions from the wrapped iterator (e.g. dataloader exhaustion) propagate
  unchanged, without a data loading end entry.
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

  `should_record(*args, **kwargs)`, if given, decides per call whether the call
  does blocking work worth recording (e.g. only checkpointing steps).
  """

  def wrapper(*args, **kwargs):
    if _recorder is None or (
        should_record is not None and not should_record(*args, **kwargs)
    ):
      return fn(*args, **kwargs)
    with maybe_record_custom_badput(event_type):
      return fn(*args, **kwargs)

  return wrapper


def _utc_now() -> float:
  """UTC epoch seconds, comparable with the library's step start times."""
  return datetime.datetime.now(datetime.timezone.utc).timestamp()


def _write_entry(entry: dict[str, Any]) -> None:
  """Writes a raw entry to the goodput log; never raises.

  `GoodputRecorder` has no public checkpoint API yet (its
  `record_checkpoint_progress` is a stub), so checkpoint entries go through
  its private Cloud Logging writer. Switch to `record_checkpoint_save_stats` /
  `record_checkpoint_restore_stats` here once the library has them.
  """
  if _recorder is None:
    return
  cloud_logger = getattr(_recorder, "_cloud_logger", None)
  if cloud_logger is None:
    return
  try:
    cloud_logger.write_cloud_logging_entry({"job_name": _job_name, **entry})
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Goodput: writing a checkpoint entry failed: %s", e)


def record_checkpoint_save(
    *,
    step: int,
    directory: str,
    start_time: float,
    wait_for_prev_secs: float,
    get_old_steps_secs: float,
    checkpointer_blocking_secs: float,
    synchronous: bool,
) -> None:
  """Records one blocking checkpoint save, in the shape Orbax writes.

  The library's checkpoint calculator reads these keys (all of them are
  required by ml-goodput-measurement 0.2.3) and reports their total as
  `UNPRODUCTIVE_CHECKPOINT_SAVE_TIME`. It averages entries with the same step,
  so each step must get at most one entry.
  """
  total = wait_for_prev_secs + get_old_steps_secs + checkpointer_blocking_secs
  if total <= 0:
    return
  _write_entry({
      "event_type": "save",
      "step": int(step),
      "directory": directory,
      "synchronous": synchronous,
      "checkpoint_manager_blocking_start_time": start_time,
      "checkpoint_manager_blocking_duration_secs": total,
      "wait_for_prev_start_time": start_time,
      "wait_for_prev_duration_secs": wait_for_prev_secs,
      "get_old_steps_start_time": start_time + wait_for_prev_secs,
      "get_old_steps_duration_secs": get_old_steps_secs,
      "checkpointer_blocking_start_time": (
          start_time + wait_for_prev_secs + get_old_steps_secs
      ),
      "checkpointer_blocking_duration_secs": checkpointer_blocking_secs,
  })


def record_checkpoint_restore(
    *, step: int, directory: str, start_time: float, duration_secs: float
) -> None:
  """Records one checkpoint restore (`UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME`)."""
  if duration_secs <= 0:
    return
  _write_entry({
      "event_type": "restore",
      "step": int(step),
      "directory": directory,
      "checkpoint_manager_start_time": start_time,
      "checkpoint_manager_duration_secs": duration_secs,
      "checkpointer_start_time": start_time,
      "checkpointer_duration_secs": duration_secs,
  })


def _async_mode_name(checkpointer: Any) -> str:
  mode = getattr(checkpointer, "async_mode", "disabled")
  return str(getattr(mode, "value", mode)).lower()


class GoodputCheckpointer:
  """Records TorchTitan DCP checkpoint badput on a checkpointer instance.

  Wraps the instance's methods in place; the checkpointer class is unchanged.
  On the lead process, with recording on:

  * `save()` on a saving step writes one save entry for that step. The entry
    is split into the time waiting for the previous async upload
    (`maybe_wait_for_saving`), purging old checkpoints, and the rest
    (`dcp_save`: device-to-host copy, plus the write in `disabled` mode).
    This includes the final save (`last_step=True`).
  * `maybe_wait_for_staging()` (pinned-memory async mode), when a staging copy
    is still pending, writes a second save entry on the current step, so the
    stall that surfaces one step after `save()` is not lost. It is skipped at
    `close()`.
  * `load()` that restores a checkpoint writes one restore entry, tagged with
    the restored step. `before_load` runs first, so training preparation can
    end there and not overlap the restore.

  Entries are written only after the call returns normally.
  """

  # The checkpointer methods wrapped in place, and the attributes read.
  # test_goodput checks them against TorchTitan's `CheckpointManager`, so an
  # upstream rename fails presubmit instead of silently dropping a bucket.
  WRAPPED_METHODS = (
      "save",
      "load",
      "maybe_wait_for_staging",
      "maybe_wait_for_saving",
      "_purge_stale_checkpoints",
      "close",
  )
  READ_ATTRIBUTES = ("enable", "folder", "staging_future")

  def __init__(
      self,
      checkpointer: Any,
      get_step: Callable[[], int],
      before_load: Optional[Callable[[], None]] = None,
  ):
    self._checkpointer = checkpointer
    self._get_step = get_step
    self._before_load = before_load
    self._save_timing: Optional[dict[str, float]] = None
    self._closing = False
    self._warned_should_save = False
    self._original = {}
    wrappers = {
        "save": self._save,
        "load": self._load,
        "maybe_wait_for_staging": self._maybe_wait_for_staging,
        "maybe_wait_for_saving": self._maybe_wait_for_saving,
        "_purge_stale_checkpoints": self._purge_stale_checkpoints,
        "close": self._close,
    }
    for name in self.WRAPPED_METHODS:
      wrapper = wrappers[name]
      original = getattr(checkpointer, name, None)
      if original is None:
        # The checkpointer API may have changed upstream; without this method
        # the matching entries (or part of their breakdown) are not recorded.
        logger.warning(
            "Goodput: %s has no `%s`; checkpoint badput recording for it is"
            " disabled.",
            type(checkpointer).__name__,
            name,
        )
        continue
      self._original[name] = original
      setattr(checkpointer, name, wrapper)

  @property
  def _directory(self) -> str:
    return str(getattr(self._checkpointer, "folder", ""))

  def _is_recorded_save(self, curr_step: int, last_step: bool) -> bool:
    """Whether this `save()` call writes a checkpoint that is recorded."""
    checkpointer = self._checkpointer
    should_save = getattr(checkpointer, "_should_save", None)
    if (
        not enabled()
        or not getattr(checkpointer, "enable", False)
        or should_save is None
    ):
      return False
    try:
      # The final save (`last_step=True`) is recorded like any other; it falls
      # in the last step's window, which counts once `job_end_time` is read.
      # TODO: if `job_end_time` is not readable yet at the monitor's final
      # query, ml-goodput-measurement 0.2.3 still subtracts this save, so
      # productive time comes out low by its duration.
      return bool(should_save(curr_step, last_step))
    except TypeError as e:
      # `_should_save` is private upstream API; if its signature changes, skip
      # recording instead of breaking the save (test_goodput checks it).
      if not self._warned_should_save:
        self._warned_should_save = True
        logger.warning(
            "Goodput: could not call %s._should_save (%s); checkpoint save"
            " badput is not recorded.",
            type(checkpointer).__name__,
            e,
        )
      return False

  def _save(self, curr_step: int, last_step: bool = False) -> Any:
    checkpointer = self._checkpointer
    save = self._original["save"]
    if not self._is_recorded_save(curr_step, last_step):
      return save(curr_step, last_step)
    start = _utc_now()
    self._save_timing = {"wait_for_prev": 0.0, "get_old_steps": 0.0}
    try:
      result = save(curr_step, last_step)
      timing = self._save_timing
    finally:
      self._save_timing = None
    total = _utc_now() - start
    record_checkpoint_save(
        step=curr_step,
        directory=self._directory,
        start_time=start,
        wait_for_prev_secs=timing["wait_for_prev"],
        get_old_steps_secs=timing["get_old_steps"],
        checkpointer_blocking_secs=max(
            total - timing["wait_for_prev"] - timing["get_old_steps"], 0.0
        ),
        synchronous=_async_mode_name(checkpointer) == "disabled",
    )
    return result

  def _timed(self, name: str, key: str, *args, **kwargs) -> Any:
    start = _utc_now()
    result = self._original[name](*args, **kwargs)
    if self._save_timing is not None:
      self._save_timing[key] += _utc_now() - start
    return result

  def _maybe_wait_for_saving(self, *args, **kwargs) -> Any:
    return self._timed(
        "maybe_wait_for_saving", "wait_for_prev", *args, **kwargs
    )

  def _purge_stale_checkpoints(self, *args, **kwargs) -> Any:
    return self._timed(
        "_purge_stale_checkpoints", "get_old_steps", *args, **kwargs
    )

  def _maybe_wait_for_staging(self, *args, **kwargs) -> Any:
    wait = self._original["maybe_wait_for_staging"]
    pending = getattr(self._checkpointer, "staging_future", None) is not None
    if (
        not enabled()
        or not pending
        or self._closing
        or self._save_timing is not None
    ):
      return wait(*args, **kwargs)
    start = _utc_now()
    result = wait(*args, **kwargs)
    record_checkpoint_save(
        step=self._get_step(),
        directory=self._directory,
        start_time=start,
        wait_for_prev_secs=0.0,
        get_old_steps_secs=0.0,
        checkpointer_blocking_secs=_utc_now() - start,
        synchronous=False,
    )
    return result

  def _load(self, *args, **kwargs) -> Any:
    if self._before_load is not None:
      self._before_load()
    start = _utc_now()
    loaded = self._original["load"](*args, **kwargs)
    if loaded and enabled():
      record_checkpoint_restore(
          step=self._get_step(),
          directory=self._directory,
          start_time=start,
          duration_secs=_utc_now() - start,
      )
    return loaded

  def _close(self, *args, **kwargs) -> Any:
    self._closing = True
    return self._original["close"](*args, **kwargs)


@contextlib.contextmanager
def maybe_monitor_goodput(
    config: GoodputConfig, default_tensorboard_dir: str
) -> Iterator[None]:
  """Runs the goodput uploader(s) around the block on the lead process.

  Starts the cumulative uploader (with badput breakdown and step time
  deviation) and, if enabled, the rolling-window uploader. On exit each is
  stopped, which does a final flush. Failures to start or stop are logged and
  swallowed; exceptions from the block propagate.
  """
  if not config.enable_goodput_monitoring or not is_lead_process():
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
  modules = _import_goodput()
  if modules is None:
    logger.warning(
        "Goodput: ml-goodput-measurement is not installed; monitoring disabled."
    )
    return None
  _, monitoring = modules
  job_name = _job_name or resolve_job_name(config)
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
