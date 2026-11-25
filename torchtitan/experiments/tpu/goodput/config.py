"""Goodput options for TorchTitan on TPU (`--goodput.*`).

Kept apart from `goodput.py` so the job config can expose these options without
depending on the ML Goodput Measurement library, whose transitive dependencies
(Cloud Monitoring, TensorBoardX, boto3, ...) are only needed by the trainer.
"""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class GoodputConfig:
  """Goodput options, exposed on the CLI as `--goodput.*`."""

  # Record goodput events to Cloud Logging from the lead process.
  enable_goodput_recording: bool = False

  # Compute goodput from the recorded events and upload it to TensorBoard and
  # Google Cloud Monitoring from the lead process.
  enable_goodput_monitoring: bool = False

  # Goodput job name; events go to the Cloud Logging log `goodput_<job_name>`.
  # Use a new name for every run, and the same name when a run restarts or
  # resumes. Defaults to a timestamped name, which can't track restarts.
  job_name: str = ""

  # How often, in seconds, the monitor recomputes and uploads goodput.
  upload_interval_seconds: int = 30

  # Optional, as it defaults to a subfolder of TorchTitan's TensorBoard folder,
  # `<dump_folder>/<metrics.save_tb_folder>` (set in train.py).
  tensorboard_dir: str = ""

  # Optional ideal step time, in seconds, used for step time deviation.
  configured_ideal_step_time: Optional[float] = None

  # Upload goodput metrics to Google Cloud Monitoring.
  enable_gcp_goodput_metrics: bool = True

  # Also upload goodput over rolling windows (`interval_goodput` etc.).
  enable_rolling_window_goodput: bool = False

  # Rolling window sizes, in seconds.
  rolling_windows_seconds: tuple[int, ...] = (3600, 86400)

  # Optional wait at shutdown, after flushing the recorder and before the
  # monitor's final flush, so Cloud Logging entries written just before
  # (notably `job_end_time`) are readable by the final query. Off by default so
  # the application is never blocked on goodput at exit.
  # TODO: remove once ml-goodput-measurement handles the final-flush wait and
  # exposes its own timeout / opt-out to the application.
  final_flush_delay_seconds: float = 0.0
