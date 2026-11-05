"""Tests for the TPU / CPU Profiler."""

import os
import pathlib
import tempfile
from unittest import mock

import torch

from absl.testing import absltest
import jax

import torchtitan.experiments.tpu.base_device_test as base_device_test
import torchtitan.tools.profiler as torchtitan_profiler


class ProfilerAcceleratorTest(base_device_test.BaseAcceleratorDeviceTest):

  def test_profiler_runs_and_outputs_traces(self):
    device = self.accelerator_device
    self.assertEqual(device.type, "tpu", "This test is TPU-only")

    # 1. Setup a simple model to run some compute workloads on the TPU device.
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 16),
    ).to(device)

    # 2. Determine where to save the outputs. If the test runner provides
    # TEST_UNDECLARED_OUTPUTS_DIR, save there so they are kept as artifacts.
    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if outputs_dir:
      tmpdir = outputs_dir
      temp_dir_ctx = None
    else:
      temp_dir_ctx = tempfile.TemporaryDirectory()
      tmpdir = temp_dir_ctx.name

    try:
      # 3. Instantiate the profiler.
      # Setup a schedule to wait for 2 steps (4 - 1 - 1), warmup on step 2,
      # profile on step 3, and dump trace on step 4.
      profiler = torchtitan_profiler.Profiler(
          torchtitan_profiler.Profiler.Config(
              enable_profiling=True,
              profile_freq=4,
              profiler_warmup=1,
              profiler_active=1,
          ),
          global_step=0,
          base_folder=tmpdir,
      )

      # 4. Run the training steps.
      # Mock the distributed rank for a single-device environment.
      with mock.patch("torch.distributed.get_rank", return_value=0):
        with profiler:
          for _ in range(5):
            x = torch.randn(4, 16, device=device)
            _ = model(x)
            # For TPUs, force-synchronize execution so lazy compiler
            # operations finish executing inside the step 3 trace window.
            torch.accelerator.synchronize()
            profiler.step()

      # 5. Verify that trace files are created.
      # The CPU trace goes to `profile_traces/iteration_3/rank0_trace.json`.
      # On TPU, the TPU hardware trace (.xplane.pb) goes to `profile_traces/rank_0/plugins/profile/...`.

      cpu_trace_files = list(pathlib.Path(tmpdir).rglob("*.json*"))
      self.assertNotEmpty(
          cpu_trace_files, "Should have exported Chrome trace (.json)"
      )

      xplane_files = list(pathlib.Path(tmpdir).rglob("*.xplane.pb"))
      self.assertNotEmpty(
          xplane_files,
          "Should have exported XPlane trace (.xplane.pb) on TPU",
      )

      # Parse the XPlane trace and identify if it dumped any TPU events
      tpu_events_found = False
      for xplane_path in xplane_files:
        profile = jax.profiler.ProfileData.from_file(str(xplane_path))
        for plane in profile.planes:
          # In TPU profiles, planes can be named like "/device:TPU:0" or similar
          if plane.name.startswith("/device:TPU:") or plane.name == "TPU":
            for line in plane.lines:
              if any(True for _ in line.events):
                tpu_events_found = True
                break

      self.assertTrue(
          tpu_events_found,
          "Failed to find any TPU events in the exported trace",
      )
    finally:
      # 6. Clean up temporary directory if fallback was used.
      if temp_dir_ctx:
        temp_dir_ctx.cleanup()


if __name__ == "__main__":
  absltest.main()
