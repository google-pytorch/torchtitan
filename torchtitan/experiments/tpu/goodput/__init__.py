"""Goodput recording and monitoring for TorchTitan on TPU.

Wires the ML Goodput Measurement library (`ml-goodput-measurement`):

* Recording: a `GoodputRecorder` on the lead process writes start/end
  timestamps (job, TPU init, training preparation, data loading, steps and
  custom badput events) to the Cloud Logging log `goodput_<job_name>`.
  `wrap_checkpointer()` adds PyTorch DCP checkpoint save/restore entries in the
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

Modules:

* `config`: `GoodputConfig` (`--goodput.*`), with no library dependency.
* `goodput`: the global recorder, event recording and the monitor.
* `checkpointer`: checkpoint save/restore badput.
"""

# pylint: disable=g-importing-member
from torchtitan.experiments.tpu.goodput.checkpointer import wrap_checkpointer
from torchtitan.experiments.tpu.goodput.config import GoodputConfig
from torchtitan.experiments.tpu.goodput.goodput import enabled
from torchtitan.experiments.tpu.goodput.goodput import EVAL_BADPUT
from torchtitan.experiments.tpu.goodput.goodput import GoodputEvent
from torchtitan.experiments.tpu.goodput.goodput import initialize
from torchtitan.experiments.tpu.goodput.goodput import maybe_monitor_goodput
from torchtitan.experiments.tpu.goodput.goodput import maybe_record
from torchtitan.experiments.tpu.goodput.goodput import record_data_loading
from torchtitan.experiments.tpu.goodput.goodput import record_end
from torchtitan.experiments.tpu.goodput.goodput import record_start
from torchtitan.experiments.tpu.goodput.goodput import record_step_start
from torchtitan.experiments.tpu.goodput.goodput import shutdown
from torchtitan.experiments.tpu.goodput.goodput import start_training_preparation
from torchtitan.experiments.tpu.goodput.goodput import wrap_with_custom_badput

# pylint: enable=g-importing-member
