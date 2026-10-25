# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The real Qwen3-235B model, placed on meshes wider than any slice we can book.

Everything that breaks Qwen3-235B above 128 ranks is a property of the mesh
width, not of the devices: an axis stops dividing, and either `shard_module`
replicates the parameter or pads it. Both train to a loss. A fake process
group reproduces all of it in a few seconds, so a 1024-chip regression does not
need 1024 chips to catch.

This builds the model on `meta` and runs the shard-axis planning and the FSDP
sharding the way `parallelize_qwen3` does, so every placement asserted here is
one the trainer would install.
"""

import contextlib

import torch
import torch.distributed as dist
from absl.testing import absltest
from torch.distributed.tensor import init_device_mesh, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.config import TrainingConfig
from torchtitan.experiments.tpu.graph_trainer.layer_stack import (
    with_tpu_stacked_layers,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_fsdp import (
    apply_tpu_fsdp,
    param_specs,
    SPECS_ATTR,
)
from torchtitan.experiments.tpu.qwen3 import config_registry
from torchtitan.experiments.tpu.qwen3.infra.parallelize import (
    configure_qwen_simple_fsdp_shard_dims,
)


class _FakeParallelDims:
  """Pure FSDP on one mesh axis -- the shape of a single-slice scaling run."""

  def __init__(self, mesh, fsdp_degree: int):
    self._mesh = mesh
    self.dp_shard = fsdp_degree
    self.cp = 1
    self.dp_replicate_enabled = False
    self.dp_shard_enabled = True
    self.cp_enabled = False
    self.ep_enabled = False
    self.tp_enabled = False

  def get_mesh(self, names):
    return self._mesh

  def get_optional_mesh(self, names):
    return None


@contextlib.contextmanager
def _placed_235b(world: int):
  """Qwen3-235B-A22B, stacked and FSDP-wrapped on a `world`-wide fake mesh."""
  dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=world)
  try:
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("fsdp",))
    spec = config_registry.qwen3_moe_235b_scan().model_spec
    with torch.device("meta"):
      model = with_tpu_stacked_layers(spec.model, use_scan=True).build()
    configure_qwen_simple_fsdp_shard_dims(model, _FakeParallelDims(mesh, world))
    yield apply_tpu_fsdp(
        model,
        parallel_dims=_FakeParallelDims(mesh, world),
        training=TrainingConfig(
            mixed_precision_param="bfloat16", mixed_precision_reduce="float32"
        ),
    )
  finally:
    dist.destroy_process_group()


class Qwen3_235BScaleTest(absltest.TestCase):

  def test_every_weight_shards_evenly_at_2048_ranks(self) -> None:
    # 2048 ranks is 1024 v7x chips. `Shard._split_tensor` pads a short tail
    # rather than raising, so an uneven axis here is a wrong-weights bug that
    # never surfaces as one.
    with _placed_235b(2048) as model:
      specs = param_specs(model)
      for name, param in model.named_parameters():
        self.assertIn(id(param), specs, name)
        spec = specs[id(param)]
        placement = spec.placements[-1]
        if isinstance(placement, Shard):
          self.assertEqual(
              spec.global_shape[placement.dim] % 2048, 0, f"{name} {placement}"
          )

  def test_nothing_large_is_replicated_at_2048_ranks(self) -> None:
    # The pre-fix failure mode: an expert stack's dim 1 is `num_experts` = 128,
    # `data_parallel` cannot split it past 128 ranks, so it replicates in
    # silence and per-rank weights go from 3.4 GiB to 423 GiB.
    with _placed_235b(2048) as model:
      specs = param_specs(model)
      for name, param in model.named_parameters():
        if isinstance(specs[id(param)].placements[-1], Replicate):
          # Only the per-layer norms, which are `(n_layers, dim)` at most.
          self.assertLess(param.numel(), 1 << 20, f"{name} {tuple(param.shape)}")

  def test_the_scan_body_gathers_what_was_sharded_at_2048_ranks(self) -> None:
    # The body sees one layer's row, so its dim 0 is the stack's dim 1. Off by
    # one reassembles the weight wrong and still trains.
    with _placed_235b(2048) as model:
      holders = [m for m in model.modules() if hasattr(m, "plan_shard_dims")]
      self.assertTrue(holders, "layer stacking did not run")
      for holder in holders:
        for name, spec in getattr(holder, SPECS_ATTR).items():
          placement = spec.placements[-1]
          row = spec.layer_slice().scan_spec()
          self.assertEqual(row.sharded, isinstance(placement, Shard), name)
          if row.sharded:
            self.assertEqual(row.gather_dim, placement.dim - 1, name)

  def test_128_ranks_still_shards_every_stack_on_dim_one(self) -> None:
    # Every published 235B number was taken here. If any stack moves off dim 1
    # at 128 ranks this stops being the same run, measured or not.
    with _placed_235b(128) as model:
      for holder in [m for m in model.modules() if hasattr(m, "plan_shard_dims")]:
        for name, spec in getattr(holder, SPECS_ATTR).items():
          self.assertEqual(spec.placements[-1], Shard(1), name)


if __name__ == "__main__":
  absltest.main()
