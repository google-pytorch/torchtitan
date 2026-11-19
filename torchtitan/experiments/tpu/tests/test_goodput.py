"""Tests for goodput recording and monitoring in the TPU graph trainer."""

import dataclasses
import inspect
import os
from unittest import mock

from absl.testing import absltest
from torchtitan.components import checkpointer as checkpointer_lib
from torchtitan.config.manager import custom_registry
from torchtitan.experiments.tpu import goodput
import tyro

_GoodputConfig = goodput.GoodputConfig


class _GoodputTestCase(absltest.TestCase):
  """Runs each test as global rank 0 of a fresh, uninitialized job."""

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "8"})
    )
    self.enter_context(
        mock.patch.object(goodput.dist, "is_initialized", return_value=False)
    )
    self.goodput_lib = mock.MagicMock()
    self.monitoring_lib = mock.MagicMock()
    self.enter_context(
        mock.patch.object(
            goodput,
            "_import_goodput",
            return_value=(self.goodput_lib, self.monitoring_lib),
        )
    )
    self.addCleanup(goodput.shutdown)
    self.sleep = self.enter_context(mock.patch.object(goodput.time, "sleep"))
    self.enter_context(
        mock.patch.object(goodput, "_warned_unknown_rank", False)
    )

  @property
  def recorder(self):
    return self.goodput_lib.GoodputRecorder.return_value

  @property
  def monitor(self):
    return self.monitoring_lib.GoodputMonitor.return_value


class InitializeTest(_GoodputTestCase):

  def test_disabled_by_default(self):
    goodput.initialize(goodput.GoodputConfig())
    self.assertFalse(goodput.enabled())
    goodput.record_start(goodput.GoodputEvent.JOB)
    self.goodput_lib.GoodputRecorder.assert_not_called()

  def test_creates_recorder_on_lead_process(self):
    goodput.initialize(
        goodput.GoodputConfig(enable_goodput_recording=True, job_name="run")
    )
    self.assertTrue(goodput.enabled())
    self.assertEqual(goodput.get_job_name(), "run")
    self.goodput_lib.GoodputRecorder.assert_called_once_with(
        job_name="run", logger_name="goodput_run", logging_enabled=True
    )

  def test_no_recorder_on_other_ranks(self):
    with mock.patch.dict(os.environ, {"RANK": "3"}):
      goodput.initialize(
          goodput.GoodputConfig(enable_goodput_recording=True, job_name="run")
      )
    self.assertFalse(goodput.enabled())
    self.goodput_lib.GoodputRecorder.assert_not_called()

  def test_no_recorder_when_multi_process_rank_is_unknown(self):
    with mock.patch.dict(os.environ, {"WORLD_SIZE": "8"}), mock.patch.object(
        goodput.logger, "warning"
    ) as warning:
      del os.environ["RANK"]
      goodput.initialize(goodput.GoodputConfig(enable_goodput_recording=True))
      self.assertFalse(goodput.is_lead_process())
    self.assertFalse(goodput.enabled())
    # Warned once, although the rank was checked twice.
    warning.assert_called_once()
    self.assertIn("RANK is unset", warning.call_args.args[0])

  def test_default_job_name_is_timestamped(self):
    goodput.initialize(goodput.GoodputConfig(enable_goodput_recording=True))
    self.assertStartsWith(goodput.get_job_name(), "torchtitan_")

  def test_missing_library_disables_recording(self):
    with mock.patch.object(goodput, "_import_goodput", return_value=None):
      goodput.initialize(goodput.GoodputConfig(enable_goodput_recording=True))
    self.assertFalse(goodput.enabled())

  def test_recorder_construction_failure_is_swallowed(self):
    self.goodput_lib.GoodputRecorder.side_effect = RuntimeError("no creds")
    goodput.initialize(goodput.GoodputConfig(enable_goodput_recording=True))
    self.assertFalse(goodput.enabled())

  def test_verify_lead_process_drops_recorder_of_other_ranks(self):
    goodput.initialize(goodput.GoodputConfig(enable_goodput_recording=True))
    with mock.patch.object(
        goodput.dist, "is_initialized", return_value=True
    ), mock.patch.object(goodput.dist, "get_rank", return_value=2):
      goodput.verify_lead_process()
    self.assertFalse(goodput.enabled())

  def test_shutdown_flushes(self):
    goodput.initialize(goodput.GoodputConfig(enable_goodput_recording=True))
    recorder = self.recorder
    goodput.shutdown()
    recorder.flush.assert_called_once()
    self.assertFalse(goodput.enabled())


class RecordTest(_GoodputTestCase):

  def setUp(self):
    super().setUp()
    goodput.initialize(
        goodput.GoodputConfig(enable_goodput_recording=True, job_name="run")
    )

  def test_record_start_and_end(self):
    goodput.record_start(goodput.GoodputEvent.TPU_INIT)
    goodput.record_end(goodput.GoodputEvent.TPU_INIT)
    self.recorder.record_tpu_init_start_time.assert_called_once()
    self.recorder.record_tpu_init_end_time.assert_called_once()

  def test_recorder_errors_are_swallowed(self):
    self.recorder.record_job_start_time.side_effect = RuntimeError("quota")
    goodput.record_start(goodput.GoodputEvent.JOB)

  def test_maybe_record_skips_end_on_exception(self):
    with self.assertRaises(ValueError):
      with goodput.maybe_record(goodput.GoodputEvent.TRAINING_PREPARATION):
        raise ValueError()
    self.recorder.record_training_preparation_start_time.assert_called_once()
    self.recorder.record_training_preparation_end_time.assert_not_called()

  def test_record_step_start(self):
    goodput.record_step_start(7)
    self.recorder.record_step_start_time.assert_called_once_with(7)

  def test_data_loading_wraps_each_batch(self):
    batches = list(goodput.record_data_loading(iter([1, 2])))
    self.assertEqual(batches, [1, 2])
    # One start per fetch, including the final one that hits StopIteration.
    self.assertEqual(self.recorder.record_data_loading_start_time.call_count, 3)
    self.assertEqual(self.recorder.record_data_loading_end_time.call_count, 2)

  def test_data_loading_propagates_errors(self):
    def failing():
      yield 1
      raise RuntimeError("dataloader exhausted")

    iterator = goodput.record_data_loading(failing())
    self.assertEqual(next(iterator), 1)
    with self.assertRaisesRegex(RuntimeError, "exhausted"):
      next(iterator)

  def test_custom_badput_wrapper(self):
    fn = mock.Mock(return_value="saved")
    wrapped = goodput.wrap_with_custom_badput(
        fn,
        goodput.EVAL_BADPUT,
        should_record=lambda curr_step, last_step=False: curr_step % 5 == 0,
    )
    self.assertEqual(wrapped(3), "saved")
    self.recorder.record_custom_badput_event_start_time.assert_not_called()
    self.assertEqual(wrapped(5, last_step=True), "saved")
    fn.assert_called_with(5, last_step=True)
    self.recorder.record_custom_badput_event_start_time.assert_called_once_with(
        custom_badput_event_type="eval"
    )
    self.recorder.record_custom_badput_event_end_time.assert_called_once_with(
        custom_badput_event_type="eval"
    )

  def test_custom_badput_wrapper_is_passthrough_when_disabled(self):
    goodput.shutdown()
    fn = mock.Mock(return_value=1)
    wrapped = goodput.wrap_with_custom_badput(fn, goodput.EVAL_BADPUT)
    self.assertEqual(wrapped(), 1)
    self.recorder.record_custom_badput_event_start_time.assert_not_called()


class _FakeCheckpointer:
  """Mimics the parts of TorchTitan's DCP checkpointer that are wrapped."""

  def __init__(self, async_mode="disabled", interval=5):
    self.enable = True
    self.folder = "/tmp/ckpt"
    self.async_mode = async_mode
    self.interval = interval
    self.staging_future = None
    self.calls = []
    self.restore_step = None

  def _should_save(self, curr_step, last_step=False):
    return last_step or curr_step % self.interval == 0

  def maybe_wait_for_saving(self):
    self.calls.append("wait_for_saving")

  def _purge_stale_checkpoints(self, saving_step):
    self.calls.append(f"purge_{saving_step}")

  def save(self, curr_step, last_step=False):
    if not self._should_save(curr_step, last_step):
      return False
    self.maybe_wait_for_saving()
    self._purge_stale_checkpoints(saving_step=curr_step)
    if self.async_mode == "async_with_pinned_mem":
      self.staging_future = object()
    self.calls.append(f"save_{curr_step}")
    return True

  def maybe_wait_for_staging(self):
    self.staging_future = None
    self.calls.append("wait_for_staging")

  def load(self, step=-1):
    self.calls.append("load")
    return self.restore_step is not None

  def close(self):
    self.maybe_wait_for_staging()
    self.calls.append("close")


class CheckpointerTest(_GoodputTestCase):

  def setUp(self):
    super().setUp()
    goodput.initialize(
        goodput.GoodputConfig(enable_goodput_recording=True, job_name="run")
    )
    self.step = 0
    self.before_load = mock.Mock()

  def _wrap(self, checkpointer):
    goodput.GoodputCheckpointer(
        checkpointer,
        get_step=lambda: self.step,
        before_load=self.before_load,
    )
    return checkpointer

  def _entries(self):
    writer = self.recorder._cloud_logger.write_cloud_logging_entry  # pylint: disable=protected-access
    return [c.args[0] for c in writer.call_args_list]

  def test_save_writes_native_entry_only_on_saving_steps(self):
    checkpointer = self._wrap(_FakeCheckpointer())
    self.assertFalse(checkpointer.save(3))
    self.assertEqual(self._entries(), [])
    self.assertTrue(checkpointer.save(5))
    (entry,) = self._entries()
    self.assertEqual(entry["event_type"], "save")
    self.assertEqual(entry["step"], 5)
    self.assertEqual(entry["directory"], "/tmp/ckpt")
    self.assertEqual(entry["job_name"], "run")
    self.assertTrue(entry["synchronous"])
    # Every key ml-goodput-measurement 0.2.3 reads must be present.
    for key in (
        "checkpoint_manager_blocking_duration_secs",
        "wait_for_prev_duration_secs",
        "get_old_steps_duration_secs",
        "checkpointer_blocking_duration_secs",
        "checkpoint_manager_blocking_start_time",
    ):
      self.assertIn(key, entry)
    self.assertAlmostEqual(
        entry["checkpoint_manager_blocking_duration_secs"],
        entry["wait_for_prev_duration_secs"]
        + entry["get_old_steps_duration_secs"]
        + entry["checkpointer_blocking_duration_secs"],
    )
    self.assertEqual(
        checkpointer.calls, ["wait_for_saving", "purge_5", "save_5"]
    )
    # No custom badput for checkpoints any more.
    self.recorder.record_custom_badput_event_start_time.assert_not_called()

  def test_async_save_is_not_synchronous(self):
    checkpointer = self._wrap(_FakeCheckpointer(async_mode="async"))
    checkpointer.save(5)
    (entry,) = self._entries()
    self.assertFalse(entry["synchronous"])

  def test_last_step_save_is_recorded(self):
    checkpointer = self._wrap(_FakeCheckpointer())
    self.assertTrue(checkpointer.save(7, last_step=True))
    (entry,) = self._entries()
    self.assertEqual(entry["step"], 7)
    self.assertIn("save_7", checkpointer.calls)

  def test_pinned_staging_wait_gets_its_own_step(self):
    checkpointer = self._wrap(
        _FakeCheckpointer(async_mode="async_with_pinned_mem")
    )
    checkpointer.save(5)
    self.step = 6
    checkpointer.maybe_wait_for_staging()  # pending: entry on step 6
    checkpointer.maybe_wait_for_staging()  # nothing pending: no entry
    save, staging = self._entries()
    self.assertEqual(save["step"], 5)
    self.assertEqual(staging["step"], 6)
    self.assertEqual(staging["wait_for_prev_duration_secs"], 0.0)
    self.assertFalse(staging["synchronous"])

  def test_staging_wait_at_close_is_not_recorded(self):
    checkpointer = self._wrap(
        _FakeCheckpointer(async_mode="async_with_pinned_mem")
    )
    checkpointer.save(5)
    checkpointer.close()
    self.assertEqual([e["step"] for e in self._entries()], [5])
    self.assertEqual(checkpointer.calls[-2:], ["wait_for_staging", "close"])

  def test_restore_entry_uses_restored_step(self):
    fake = _FakeCheckpointer()

    def load(step=-1):
      del step
      self.step = 50  # The trainer's step is restored from the checkpoint.
      return True

    fake.load = load
    checkpointer = self._wrap(fake)
    self.assertTrue(checkpointer.load())
    self.before_load.assert_called()
    (entry,) = self._entries()
    self.assertEqual(entry["event_type"], "restore")
    self.assertEqual(entry["step"], 50)
    self.assertEqual(
        entry["checkpoint_manager_duration_secs"],
        entry["checkpointer_duration_secs"],
    )

  def test_no_restore_entry_without_a_checkpoint(self):
    checkpointer = self._wrap(_FakeCheckpointer())
    self.assertFalse(checkpointer.load())
    self.before_load.assert_called_once()
    self.assertEqual(self._entries(), [])

  def test_failed_save_writes_nothing(self):
    fake = _FakeCheckpointer()

    def broken_save(curr_step, last_step=False):
      raise OSError("disk full")

    fake.save = broken_save
    checkpointer = self._wrap(fake)
    with self.assertRaises(OSError):
      checkpointer.save(5)
    self.assertEqual(self._entries(), [])

  def test_passthrough_on_other_ranks(self):
    goodput.shutdown()
    checkpointer = self._wrap(_FakeCheckpointer())
    self.assertTrue(checkpointer.save(5))
    self.assertEqual(self._entries(), [])

  def test_write_failure_is_swallowed(self):
    writer = self.recorder._cloud_logger.write_cloud_logging_entry  # pylint: disable=protected-access
    writer.side_effect = RuntimeError("quota")
    checkpointer = self._wrap(_FakeCheckpointer())
    self.assertTrue(checkpointer.save(5))

  def test_missing_method_warns_and_rest_still_records(self):
    fake = _FakeCheckpointer()
    # As if upstream renamed it.
    fake.maybe_wait_for_saving = None
    with mock.patch.object(goodput.logger, "warning") as warning:
      checkpointer = self._wrap(fake)
    warning.assert_called_once()
    self.assertIn("maybe_wait_for_saving", warning.call_args.args)
    # The original attribute is left alone; saves are still recorded.
    self.assertIsNone(checkpointer.maybe_wait_for_saving)
    fake.maybe_wait_for_saving = lambda: None
    self.assertTrue(checkpointer.save(5))
    (entry,) = self._entries()
    self.assertEqual(entry["wait_for_prev_duration_secs"], 0.0)

  def test_no_warning_when_all_methods_exist(self):
    with mock.patch.object(goodput.logger, "warning") as warning:
      self._wrap(_FakeCheckpointer())
    warning.assert_not_called()

  def test_real_checkpointer_has_wrapped_api(self):
    # An upstream rename should fail here instead of silently dropping a
    # checkpoint badput bucket at runtime.
    manager = checkpointer_lib.CheckpointManager
    for name in goodput.GoodputCheckpointer.WRAPPED_METHODS:
      self.assertTrue(callable(getattr(manager, name, None)), name)
    annotations = {}
    for cls in reversed(manager.__mro__):
      annotations.update(getattr(cls, "__annotations__", {}))
    for name in goodput.GoodputCheckpointer.READ_ATTRIBUTES:
      self.assertIn(name, annotations)
    # `_save` calls `_should_save(curr_step, last_step)`.
    inspect.signature(manager._should_save).bind(None, 5, False)  # pylint: disable=protected-access

  def test_should_save_signature_change_falls_back_to_plain_save(self):
    fake = _FakeCheckpointer()
    fake._should_save = lambda curr_step: True  # pylint: disable=protected-access

    def save(curr_step, last_step=False):
      del last_step
      fake.calls.append(f"save_{curr_step}")
      return True

    fake.save = save
    checkpointer = self._wrap(fake)
    with mock.patch.object(goodput.logger, "warning") as warning:
      self.assertTrue(checkpointer.save(5))
      self.assertTrue(checkpointer.save(10))
    self.assertEqual(self._entries(), [])
    self.assertIn("save_5", fake.calls)
    warning.assert_called_once()


class MonitorTest(_GoodputTestCase):

  def _config(self, **kwargs):
    return goodput.GoodputConfig(
        enable_goodput_recording=True,
        enable_goodput_monitoring=True,
        job_name="run",
        **kwargs,
    )

  def test_disabled_monitoring_does_nothing(self):
    with goodput.maybe_monitor_goodput(goodput.GoodputConfig(), "/tmp/tb"):
      pass
    self.monitoring_lib.GoodputMonitor.assert_not_called()

  def test_starts_and_stops_cumulative_uploader(self):
    config = self._config(tensorboard_dir="gs://bucket/tb")
    goodput.initialize(config)
    with goodput.maybe_monitor_goodput(config, "/tmp/tb"):
      self.monitor.start_goodput_uploader.assert_called_once()
      self.monitor.stop_goodput_uploader.assert_not_called()
    self.monitor.stop_goodput_uploader.assert_called_once()
    self.monitor.start_rolling_window_goodput_uploader.assert_not_called()
    kwargs = self.monitoring_lib.GoodputMonitor.call_args.kwargs
    self.assertEqual(kwargs["job_name"], "run")
    self.assertEqual(kwargs["logger_name"], "goodput_run")
    self.assertEqual(kwargs["tensorboard_dir"], "gs://bucket/tb")
    self.assertTrue(kwargs["include_badput_breakdown"])
    # Deprecated step-deviation uploader options are not passed; step time
    # deviation comes from the cumulative / rolling-window uploaders.
    self.assertNotIn("include_step_deviation", kwargs)
    self.assertNotIn("step_deviation_interval_seconds", kwargs)
    self.monitor.start_step_deviation_uploader.assert_not_called()
    self.monitoring_lib.GCPOptions.assert_called_once_with(
        enable_gcp_goodput_metrics=True
    )

  def test_no_wait_before_final_flush_by_default(self):
    config = self._config(enable_rolling_window_goodput=True)
    goodput.initialize(config)
    with goodput.maybe_monitor_goodput(config, "/tmp/tb"):
      pass
    self.sleep.assert_not_called()
    self.monitor.stop_goodput_uploader.assert_called_once()

  def test_waits_for_last_entries_before_final_flush(self):
    config = self._config(
        enable_rolling_window_goodput=True, final_flush_delay_seconds=7
    )
    goodput.initialize(config)
    order = mock.Mock()
    order.attach_mock(self.sleep, "sleep")
    order.attach_mock(self.monitor.stop_rolling_window_goodput_uploader, "roll")
    order.attach_mock(self.monitor.stop_goodput_uploader, "cumulative")
    with goodput.maybe_monitor_goodput(config, "/tmp/tb"):
      self.sleep.assert_not_called()
    self.assertEqual(
        order.mock_calls,
        [mock.call.sleep(7), mock.call.roll(), mock.call.cumulative()],
    )

  def test_no_wait_without_monitor(self):
    with goodput.maybe_monitor_goodput(goodput.GoodputConfig(), "/tmp/tb"):
      pass
    self.sleep.assert_not_called()

  def test_default_tensorboard_dir(self):
    config = self._config()
    goodput.initialize(config)
    with goodput.maybe_monitor_goodput(config, "/tmp/tb"):
      pass
    kwargs = self.monitoring_lib.GoodputMonitor.call_args.kwargs
    self.assertEqual(kwargs["tensorboard_dir"], "/tmp/tb")

  def test_rolling_window_uploader(self):
    config = self._config(
        enable_rolling_window_goodput=True, rolling_windows_seconds=(60, 120)
    )
    goodput.initialize(config)
    with goodput.maybe_monitor_goodput(config, "/tmp/tb"):
      self.monitor.start_rolling_window_goodput_uploader.assert_called_once_with(
          [60, 120]
      )
    self.monitor.stop_rolling_window_goodput_uploader.assert_called_once()
    self.monitor.stop_goodput_uploader.assert_called_once()

  def test_stops_uploader_when_training_fails(self):
    config = self._config()
    goodput.initialize(config)
    with self.assertRaises(ValueError):
      with goodput.maybe_monitor_goodput(config, "/tmp/tb"):
        raise ValueError()
    self.monitor.stop_goodput_uploader.assert_called_once()

  def test_start_failure_does_not_interrupt_training(self):
    self.monitor.start_goodput_uploader.side_effect = RuntimeError("no creds")
    config = self._config()
    goodput.initialize(config)
    ran = False
    with goodput.maybe_monitor_goodput(config, "/tmp/tb"):
      ran = True
    self.assertTrue(ran)
    self.monitor.stop_goodput_uploader.assert_not_called()

  def test_no_monitor_on_other_ranks(self):
    with mock.patch.dict(os.environ, {"RANK": "1"}):
      with goodput.maybe_monitor_goodput(self._config(), "/tmp/tb"):
        pass
    self.monitoring_lib.GoodputMonitor.assert_not_called()


class CommandLineTest(absltest.TestCase):

  @dataclasses.dataclass
  class _Config:
    goodput: _GoodputConfig = dataclasses.field(default_factory=_GoodputConfig)

  def test_flags(self):
    config = tyro.cli(
        self._Config,
        args=[
            "--goodput.enable-goodput-recording",
            "--goodput.enable-goodput-monitoring",
            "--goodput.job-name=my_run",
            "--goodput.tensorboard-dir=gs://bucket/tb",
            "--goodput.enable-rolling-window-goodput",
            "--goodput.rolling-windows-seconds",
            "60",
            "300",
        ],
        registry=custom_registry,
    )
    self.assertTrue(config.goodput.enable_goodput_recording)
    self.assertTrue(config.goodput.enable_goodput_monitoring)
    self.assertEqual(config.goodput.job_name, "my_run")
    self.assertEqual(config.goodput.tensorboard_dir, "gs://bucket/tb")
    self.assertTrue(config.goodput.enable_rolling_window_goodput)
    self.assertEqual(config.goodput.rolling_windows_seconds, (60, 300))

  def test_goodput_is_part_of_tpu_trainer_config(self):
    from torchtitan.experiments.tpu import tpu_job_config  # pylint: disable=g-import-not-at-top

    fields = {
        f.name for f in dataclasses.fields(tpu_job_config.TPUTrainerConfig)
    }
    self.assertIn("goodput", fields)


if __name__ == "__main__":
  absltest.main()
