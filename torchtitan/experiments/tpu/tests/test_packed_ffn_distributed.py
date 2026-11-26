# Copyright 2026 Google LLC

import os
import tempfile

from absl.testing import absltest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
import torch.multiprocessing as mp
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.distributed import distributed
from torchtitan.experiments.tpu.testing import base_distributed_device_test
from torchtitan.experiments.tpu.tests.test_packed_ffn import _build_fused_and_packed
from torchtitan.experiments.tpu.tests.test_packed_ffn import DIM
from torchtitan.experiments.tpu.tests.test_packed_ffn import HIDDEN_DIM


def _full_tensor(tensor):
  return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _assert_states_equal(actual, expected) -> None:
  assert actual.keys() == expected.keys()
  for key in expected:
    torch.testing.assert_close(
        _full_tensor(actual[key]),
        _full_tensor(expected[key]),
        rtol=0,
        atol=0,
    )


def _sharded_fused_and_packed(device: torch.device, mesh: DeviceMesh):
  fused, packed = (
      data_parallel(module.to(device), mesh, mode="fully_shard")
      for module in _build_fused_and_packed(bias=True)
  )
  return fused, packed


def _test_simple_fsdp_checkpoint_roundtrip(
    device: torch.device,
    _rank: int,
    world_size: int,
    checkpoint_dir: str,
) -> None:
  mesh = DeviceMesh(device.type, list(range(world_size)))
  fused, packed = _sharded_fused_and_packed(device, mesh)
  for module in (fused, packed):
    torch.manual_seed(1234)
    module.init_states()

  parameters = dict(packed.named_parameters())
  assert set(parameters) == {
      "w13_packed.weight",
      "w13_packed.bias",
      "w2.weight",
      "w2.bias",
  }
  assert all(isinstance(p, DTensor) for p in parameters.values())

  # Upstream's logical entries arrive sharded on dim 0; the load hook packs
  # them into the dim-0-sharded parameter.
  fused_state = fused.state_dict()
  packed.load_state_dict(fused_state)
  packed_state = packed.state_dict()
  _assert_states_equal(packed_state, fused_state)

  # The logical w1/w3 entries are what the checkpointer's ModelWrapper caches
  # for the whole run, so a matrix must stay sharded: on dim 1, because w1 and
  # w3 are split along dim 0 of the packed parameter. Each device then holds
  # every row and about 1 / world_size of the columns.
  max_local_columns = -(-DIM // world_size)
  for name in ("w1", "w3"):
    weight = packed_state[f"{name}.weight"]
    assert isinstance(weight, DTensor)
    assert tuple(weight.placements) == (Shard(1),), weight.placements
    local = weight.to_local()
    assert local.shape[0] == HIDDEN_DIM, local.shape
    assert local.shape[1] <= max_local_columns, local.shape
    bias = packed_state[f"{name}.bias"]
    assert isinstance(bias, DTensor)
    assert tuple(bias.placements) == (Replicate(),), bias.placements

  # Save the packed layout, then load it into the packed and into upstream's
  # layout. Both go through the load hook and DCP's resharding.
  packed_dir = os.path.join(checkpoint_dir, "packed")
  dcp.save({"model": packed_state}, checkpoint_id=packed_dir)
  dist.barrier()
  for restored in _sharded_fused_and_packed(device, mesh):
    load_state = {"model": restored.state_dict()}
    dcp.load(load_state, checkpoint_id=packed_dir)
    restored.load_state_dict(load_state["model"])
    _assert_states_equal(restored.state_dict(), packed_state)

  # And a checkpoint written by upstream's layout loads into the packed one.
  fused_dir = os.path.join(checkpoint_dir, "fused")
  dcp.save({"model": fused_state}, checkpoint_id=fused_dir)
  dist.barrier()
  _, restored = _sharded_fused_and_packed(device, mesh)
  load_state = {"model": restored.state_dict()}
  dcp.load(load_state, checkpoint_id=fused_dir)
  restored.load_state_dict(load_state["model"])
  _assert_states_equal(restored.state_dict(), fused_state)
  # `named_parameters()`, not attribute access: SimpleFSDP's parametrization
  # makes `module.w13_packed.weight` the gathered plain tensor.
  torch.testing.assert_close(
      _full_tensor(dict(restored.named_parameters())["w13_packed.weight"]),
      _full_tensor(parameters["w13_packed.weight"]),
      rtol=0,
      atol=0,
  )


class TpuPackedFeedForwardDistributedTest(
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
