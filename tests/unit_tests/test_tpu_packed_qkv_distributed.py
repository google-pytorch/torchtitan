# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

from absl.testing import absltest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
import torch.multiprocessing as mp
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.testing import base_distributed_device_test
from torchtitan.experiments.tpu.distributed import distributed
from tests.unit_tests.test_tpu_packed_qkv import _build_fused_and_packed


def _full_tensor(tensor):
  return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _test_simple_fsdp_checkpoint_roundtrip(
    device: torch.device,
    _rank: int,
    world_size: int,
    checkpoint_dir: str,
) -> None:
  mesh = DeviceMesh(device.type, list(range(world_size)))
  fused, packed = (
      data_parallel(module.to(device), mesh, mode="fully_shard")
      for module in _build_fused_and_packed()
  )
  for module in (fused, packed):
    torch.manual_seed(1234)
    module.init_states()

  fused_state = fused.state_dict()
  packed_state = packed.state_dict()
  assert fused_state.keys() == packed_state.keys()
  for key in fused_state:
    torch.testing.assert_close(
        _full_tensor(packed_state[key]),
        _full_tensor(fused_state[key]),
        rtol=0,
        atol=0,
    )
  [(name, parameter)] = packed.named_parameters()
  assert name == "wqkv_packed.weight"
  assert isinstance(parameter, DTensor)

  dcp.save({"model": packed_state}, checkpoint_id=checkpoint_dir)
  dist.barrier()
  restored, _ = _build_fused_and_packed()
  restored = data_parallel(restored.to(device), mesh, mode="fully_shard")
  load_state = {"model": restored.state_dict()}
  dcp.load(load_state, checkpoint_id=checkpoint_dir)
  restored.load_state_dict(load_state["model"])
  for key, value in restored.state_dict().items():
    torch.testing.assert_close(
        _full_tensor(value),
        _full_tensor(packed_state[key]),
        rtol=0,
        atol=0,
    )


class TpuPackedQKVDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):

  def test_simple_fsdp_checkpoint_roundtrip(self) -> None:
    num_devices = self.num_devices
    if self.accelerator_device_type == device_type.AcceleratorDeviceType.CPU:
      num_devices = max(num_devices, 2)
    elif num_devices < 2:
      self.skipTest("SimpleFSDP checkpoint test requires at least two devices")

    with tempfile.TemporaryDirectory() as checkpoint_dir:
      distributed.run_distributed(
          num_devices,
          self.accelerator_device_type,
          _test_simple_fsdp_checkpoint_roundtrip,
          checkpoint_dir,
      )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
