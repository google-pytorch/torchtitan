"""The TPU SimpleFSDP override gathers and reduces non-leading shards natively.

Each check compares against DTensor's own redistribute, on ``--device`` across
``--num_devices`` ranks.
"""

import functools

from absl import logging
from absl.testing import absltest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.experiments.tpu.testing import base_distributed_device_test
from torchtitan.experiments.tpu.distributed import distributed
from torchtitan.experiments.tpu.graph_trainer.simple_fsdp import (
    install_tpu_replicate_computation,
    TpuReplicateComputation,
)
from torchtitan.models.common.linear import Linear


def nonleading_shard_matches_dtensor(device, rank, world_size, *, hybrid):
  """FSDP (or HSDP with a replicate axis) of a dim-1 shard matches DTensor."""
  if world_size == 1 or (hybrid and (world_size < 4 or world_size % 2)):
    logging.warning("skipped: needs 2 ranks, or an even mesh of 4 for HSDP")
    return
  if hybrid:
    shape, mode = (2, world_size // 2), "hybrid_shard"
    names = ("dp_replicate", "dp_shard")
  else:
    shape, names, mode = (world_size,), ("dp_shard",), "fully_shard"
  mesh = init_device_mesh(device.type, shape, mesh_dim_names=names)
  placements = ((Replicate(),) if hybrid else ()) + (Shard(1),)
  module = nn.Linear(2 * shape[-1], 4, bias=False, device=device)
  data_parallel(
      module,
      mesh,
      mode,
      shard_dim=1,
      mp_policy=MixedPrecisionPolicy(
          param_dtype=torch.bfloat16, reduce_dtype=torch.float32
      ),
  )
  install_tpu_replicate_computation(module)
  param = module._parameters["weight"]
  assert param.placements == placements, param.placements
  assert type(module._simple_fsdp_parametrization_list[0]) is TpuReplicateComputation

  want_shard = param.to_local().detach().clone().requires_grad_(True)
  want = (
      DTensor.from_local(want_shard, mesh, placements)
      .redistribute(
          placements=[Replicate()] * len(placements),
          forward_dtype=torch.bfloat16,
          backward_dtype=torch.float32,
      )
      .to_local(grad_placements=[Partial()] * len(placements))
  )
  got = module.weight

  torch.testing.assert_close(got, want, rtol=0, atol=0)
  for gathered in (want, got):
    (gathered.float().sum() * (rank + 1)).backward()
  torch.testing.assert_close(param.grad.to_local(), want_shard.grad, rtol=0, atol=0)

  # Without the list to patch, a non-leading shard must fail loudly instead of
  # silently keeping SimpleFSDP's copying path.
  del module._simple_fsdp_parametrization_list
  try:
    install_tpu_replicate_computation(module)
  except RuntimeError as err:
    assert "_simple_fsdp_parametrization_list" in str(err), err
  else:
    raise AssertionError("installing without a parametrization list passed")


def stacked_linear_shards_matrix_rows(device, rank, world_size, *, hybrid):
  """A stacked Linear ([N, F, D] weight, [N, F] bias) shards F, never N.

  N is smaller than the mesh, so sharding dim 0 would replicate the whole
  parameter on every rank (the FeedForward.w13 out-of-memory on PR #195).
  """
  del rank
  if world_size == 1 or (hybrid and (world_size < 4 or world_size % 2)):
    logging.warning("skipped: needs 2 ranks, or an even mesh of 4 for HSDP")
    return
  if hybrid:
    shape, mode = (2, world_size // 2), "hybrid_shard"
    names = ("dp_replicate", "dp_shard")
  else:
    shape, names, mode = (world_size,), ("dp_shard",), "fully_shard"
  mesh = init_device_mesh(device.type, shape, mesh_dim_names=names)
  placements = ((Replicate(),) if hybrid else ()) + (Shard(1),)
  num_linears, in_features, out_features = 2, 4, 2 * shape[-1]
  module = Linear.Config(
      in_features=in_features,
      out_features=out_features,
      num_linears=num_linears,
      bias=True,
  ).build()
  module.to(device)
  # No `shard_dim`: the model walk in `apply_simple_fsdp` passes none either.
  data_parallel(
      module,
      mesh,
      mode,
      mp_policy=MixedPrecisionPolicy(
          param_dtype=torch.bfloat16, reduce_dtype=torch.float32
      ),
  )
  install_tpu_replicate_computation(module)

  for name, full_shape in (
      ("weight", (num_linears, out_features, in_features)),
      ("bias", (num_linears, out_features)),
  ):
    param = module._parameters[name]
    assert isinstance(param, DTensor), type(param)
    assert param.placements == placements, (name, param.placements)
    assert tuple(param.shape) == full_shape, (name, param.shape)
    local_shape = list(full_shape)
    local_shape[1] //= shape[-1]
    assert list(param.to_local().shape) == local_shape, (name, param.to_local().shape)
  for computation in module._simple_fsdp_parametrization_list:
    assert type(computation) is TpuReplicateComputation, type(computation)

  param = module._parameters["weight"]
  want = (
      DTensor.from_local(param.to_local().detach(), mesh, placements)
      .redistribute(
          placements=[Replicate()] * len(placements),
          forward_dtype=torch.bfloat16,
      )
      .to_local()
  )
  torch.testing.assert_close(module.weight.detach(), want, rtol=0, atol=0)


class TestTpuSimpleFSDP(base_distributed_device_test.BaseDistributedDeviceTest):

  def _on_devices(self, worker):
    distributed.run_distributed(
        self.num_devices, self.accelerator_device_type, worker
    )

  def test_fsdp_nonleading_shard_matches_dtensor(self):
    self._on_devices(
        functools.partial(nonleading_shard_matches_dtensor, hybrid=False)
    )

  def test_hsdp_nonleading_shard_matches_dtensor(self):
    self._on_devices(
        functools.partial(nonleading_shard_matches_dtensor, hybrid=True)
    )

  def test_fsdp_stacked_linear_shards_matrix_rows(self):
    self._on_devices(
        functools.partial(stacked_linear_shards_matrix_rows, hybrid=False)
    )

  def test_hsdp_stacked_linear_shards_matrix_rows(self):
    self._on_devices(
        functools.partial(stacked_linear_shards_matrix_rows, hybrid=True)
    )


if __name__ == "__main__":
  absltest.main()
