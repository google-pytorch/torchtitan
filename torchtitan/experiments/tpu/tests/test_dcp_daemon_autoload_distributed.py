"""Tests that DCP's async-checkpoint background process can start on TPU.

`async_with_pinned_mem` checkpointing saves from a CPU-only process that DCP
spawns lazily; that process runs `dist.init_process_group("gloo")` and
`dist.barrier()`. On TPU, `barrier()` queries the accelerator, which initializes
the TPU runtime in the child and fails because the trainer already holds the
TPU (b/576703287). TorchTitan works around this by disabling device-backend
autoload while the save is in flight (`_disable_device_autoload_for_dcp_daemon`
in components/checkpointer/dcp.py).

This test reproduces exactly that sequence -- a process holding the TPU spawns
a CPU-only child that initializes Gloo and calls `barrier()` -- with the
workaround active, without running training or a full checkpoint.
"""

import multiprocessing
import os
import socket

from absl.testing import absltest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchtitan.components.checkpointer import dcp as tt_dcp
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test

_AUTOLOAD_ENV = "TORCH_DEVICE_BACKEND_AUTOLOAD"


def _free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def _dcp_daemon_like_child(port: int, queue: multiprocessing.Queue) -> None:
  """What DCP's background process does before it saves anything."""
  try:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
    )
    dist.init_process_group(backend="gloo")
    dist.barrier()
    dist.destroy_process_group()
    queue.put(("ok", os.environ.get(_AUTOLOAD_ENV)))
  except BaseException as e:  # pylint: disable=broad-exception-caught
    queue.put(("error", f"{type(e).__name__}: {e}"))


class DcpDaemonAutoloadTest(base_device_test.BaseAcceleratorDeviceTest):

  def test_cpu_child_can_init_gloo_and_barrier_while_tpu_is_held(self):
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("Only relevant on TPU.")
    self.assertEqual(tt_dcp.device_type, "tpu")

    # Hold the TPU in this process, like the trainer does.
    x = torch.ones(16, device=self.accelerator_device)
    self.assertEqual(x.sum().item(), 16.0)

    prev = os.environ.get(_AUTOLOAD_ENV)
    restore = tt_dcp._disable_device_autoload_for_dcp_daemon()  # pylint: disable=protected-access
    self.assertIsNotNone(restore)
    try:
      self.assertEqual(os.environ.get(_AUTOLOAD_ENV), "0")
      ctx = mp.get_context("spawn")
      queue = ctx.Queue()
      child = ctx.Process(
          target=_dcp_daemon_like_child, args=(_free_port(), queue)
      )
      child.start()
      child.join(timeout=600)
    finally:
      restore()

    self.assertEqual(os.environ.get(_AUTOLOAD_ENV), prev)
    self.assertFalse(child.is_alive(), "child did not finish in time")
    self.assertFalse(queue.empty(), f"child exited ({child.exitcode}) silently")
    status, detail = queue.get(timeout=10)
    self.assertEqual(status, "ok", detail)
    self.assertEqual(detail, "0")  # The child started with autoload disabled.


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
