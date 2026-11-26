"""Checkpoint save/restore badput for TorchTitan's DCP checkpointer.

Writes checkpoint entries in the shape Orbax writes, so they land in the
library's native `UNPRODUCTIVE_CHECKPOINT_SAVE_TIME` / `_RESTORE_TIME` buckets.
Use `wrap_checkpointer()`; `GoodputCheckpointer` is an implementation detail.
"""

import datetime
import logging
from typing import Any, Callable, NamedTuple, Optional

from torchtitan.experiments.tpu.goodput import goodput

logger = logging.getLogger(__name__)

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


def wrap_checkpointer(
    checkpointer: Any, get_step: Callable[[], int]
) -> Optional["GoodputCheckpointer"]:
  """Records checkpoint save/restore badput of `checkpointer`, if recording.

  Wraps the instance's methods in place; the checkpointer class is unchanged.
  A checkpoint load also ends training preparation, so the two don't overlap.

  Args:
    checkpointer: TorchTitan's `CheckpointManager` (or a compatible object).
    get_step: Returns the trainer's current step.

  Returns:
    The wrapper, or None if not recording or `checkpointer` can't be wrapped
    (a warning says why).
  """
  if not goodput.enabled() or checkpointer is None:
    return None
  if not hasattr(checkpointer, "_should_save"):
    logger.warning(
        "Goodput: %s has no `_should_save`; checkpoint save and restore badput"
        " are not recorded.",
        type(checkpointer).__name__,
    )
    return None
  return GoodputCheckpointer(checkpointer, get_step)


def _utc_now() -> float:
  """UTC epoch seconds, comparable with the library's step start times."""
  return datetime.datetime.now(datetime.timezone.utc).timestamp()


def _record_save(
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
  goodput.write_log_entry({
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


def _record_restore(
    *, step: int, directory: str, start_time: float, duration_secs: float
) -> None:
  """Records one checkpoint restore (`UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME`)."""
  if duration_secs <= 0:
    return
  goodput.write_log_entry({
      "event_type": "restore",
      "step": int(step),
      "directory": directory,
      "checkpoint_manager_start_time": start_time,
      "checkpoint_manager_duration_secs": duration_secs,
      "checkpointer_start_time": start_time,
      "checkpointer_duration_secs": duration_secs,
  })


def _async_mode_name(checkpointer: Any) -> str:
  """The checkpointer's async mode, e.g. `disabled` or `async`."""
  mode = getattr(checkpointer, "async_mode", "disabled")
  return str(getattr(mode, "value", mode)).lower()


class _StagingWait(NamedTuple):
  """A pinned-memory staging wait not yet written as a save entry.

  Held until the same step's save (which absorbs it) or the next step.
  """

  # The training step the wait happened on.
  step: int
  # When the wait started (UTC epoch seconds).
  start_time: float
  # How long the wait took, in seconds.
  duration_secs: float


class GoodputCheckpointer:
  """Records TorchTitan DCP checkpoint badput on a checkpointer instance.

  Wraps the instance's `WRAPPED_METHODS` in place. On the lead process, with
  recording on:

  * `save()` on a saving step writes one save entry for that step. The entry
    is split into the time waiting for the previous async upload
    (`maybe_wait_for_saving`), purging old checkpoints, and the rest
    (`dcp_save`: device-to-host copy, plus the write in `disabled` mode).
    This includes the final save (`last_step=True`).
  * `maybe_wait_for_staging()` (pinned-memory async mode), when a staging copy
    is still in flight (`staging_future` not done), records the wait on the
    current step, so the stall that surfaces one step after `save()` is not
    lost. If that step also saves, the wait is folded into its save entry
    (`wait_for_prev`), keeping one entry per step; otherwise it gets its own
    entry at the next step or at `close()`. Waits during `close()` are skipped.
  * `load()` first ends training preparation, so it doesn't overlap the
    restore; a load that restores a checkpoint writes one restore entry,
    tagged with the restored step.

  Entries are written only after the call returns normally.
  """

  def __init__(self, checkpointer: Any, get_step: Callable[[], int]):
    self._checkpointer = checkpointer
    self._get_step = get_step
    self._save_timing: Optional[dict[str, float]] = None
    self._closing = False
    self._warned_should_save = False
    self._pending_staging: Optional[_StagingWait] = None
    self._original = {}
    wrappers = {
        "save": self._save,
        "load": self._load,
        "maybe_wait_for_staging": self._maybe_wait_for_staging,
        "maybe_wait_for_saving": self._maybe_wait_for_saving,
        "_purge_stale_checkpoints": self._purge_stale_checkpoints,
        "close": self._close,
    }
    # Wrap the instance's methods in place, rather than subclassing or proxying:
    # - The trainer builds the checkpointer from its config type
    #   (`config.checkpoint.build(...)`), so a subclass would need every
    #   recipe's `checkpoint=` config swapped, per checkpoint backend.
    # - `save()` calls `self.maybe_wait_for_saving()` and
    #   `self._purge_stale_checkpoints()` itself; instance attributes shadow
    #   the class methods, so those inner calls are timed too, which a proxy
    #   object would miss.
    # - Only this instance changes, and only when recording is on.
    for name in WRAPPED_METHODS:
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
        not goodput.enabled()
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
    """Wraps `save()`: records one save entry on a saving step."""
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
    wait_for_prev = timing["wait_for_prev"]
    staging = self._pending_staging
    if staging is not None and staging.step == curr_step:
      # This step already waited for the previous staging copy; fold it into
      # this entry so the step keeps a single save entry.
      self._pending_staging = None
      start = min(start, staging.start_time)
      wait_for_prev += staging.duration_secs
    else:
      self._flush_pending_staging()
    _record_save(
        step=curr_step,
        directory=self._directory,
        start_time=start,
        wait_for_prev_secs=wait_for_prev,
        get_old_steps_secs=timing["get_old_steps"],
        checkpointer_blocking_secs=max(
            total - timing["wait_for_prev"] - timing["get_old_steps"], 0.0
        ),
        synchronous=_async_mode_name(checkpointer) == "disabled",
    )
    return result

  def _timed(self, name: str, key: str, *args, **kwargs) -> Any:
    """Calls the original `name`, adding its time to `key` during a save."""
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

  def _staging_pending(self) -> bool:
    """Whether a pinned-memory staging copy is still in flight."""
    future = getattr(self._checkpointer, "staging_future", None)
    if future is None:
      return False
    done = getattr(future, "done", None)
    return not (callable(done) and done())

  def _flush_pending_staging(self) -> None:
    """Writes the held staging wait, if any, as its own save entry."""
    staging, self._pending_staging = self._pending_staging, None
    if staging is None:
      return
    _record_save(
        step=staging.step,
        directory=self._directory,
        start_time=staging.start_time,
        wait_for_prev_secs=0.0,
        get_old_steps_secs=0.0,
        checkpointer_blocking_secs=staging.duration_secs,
        synchronous=False,
    )

  def _maybe_wait_for_staging(self, *args, **kwargs) -> Any:
    """Wraps `maybe_wait_for_staging()`: holds an in-flight wait's time."""
    wait = self._original["maybe_wait_for_staging"]
    if not goodput.enabled() or self._closing or self._save_timing is not None:
      return wait(*args, **kwargs)
    step = self._get_step()
    staging = self._pending_staging
    if staging is not None and staging.step != step:
      # An earlier step's wait that no save absorbed gets its own entry.
      self._flush_pending_staging()
      staging = None
    if not self._staging_pending():
      return wait(*args, **kwargs)
    start = _utc_now()
    result = wait(*args, **kwargs)
    secs = _utc_now() - start
    if staging is not None:
      # A second wait on the same step extends the held one.
      start, secs = staging.start_time, staging.duration_secs + secs
    self._pending_staging = _StagingWait(
        step=step, start_time=start, duration_secs=secs
    )
    return result

  def _load(self, *args, **kwargs) -> Any:
    """Wraps `load()`: ends preparation, then records a restore entry."""
    goodput.end_training_preparation()
    start = _utc_now()
    loaded = self._original["load"](*args, **kwargs)
    if loaded and goodput.enabled():
      _record_restore(
          step=self._get_step(),
          directory=self._directory,
          start_time=start,
          duration_secs=_utc_now() - start,
      )
    return loaded

  def _close(self, *args, **kwargs) -> Any:
    """Wraps `close()`: writes a held staging wait, then skips later waits."""
    if goodput.enabled():
      self._flush_pending_staging()
    self._closing = True
    return self._original["close"](*args, **kwargs)
