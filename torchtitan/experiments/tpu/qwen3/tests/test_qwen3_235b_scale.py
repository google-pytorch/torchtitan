# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The real Qwen3-235B model, placed on meshes wider than any slice we can book.

Everything that breaks Qwen3-235B above 128 ranks is a property of the mesh
width, not of the devices: an axis stops dividing, and either `data_parallel`
replicates the parameter or `Shard._split_tensor` pads it. Both train to a loss.
A fake process group reproduces all of it in a few seconds, so a 1024-chip
regression does not need 1024 chips to catch.

This builds the model on `meta` and runs the calls `parallelize_qwen3` makes
around SimpleFSDP, including `shard_split_stacks`, so every placement asserted
here is one the trainer would install.
"""

import contextlib

import torch
import torch.distributed as dist
from absl.testing import absltest
from torch.distributed.tensor import DTensor, init_device_mesh, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.config import TrainingConfig
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.graph_trainer.layer_stack import (
    with_tpu_stacked_layers,
)
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import plan_stacked_gathers
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
    model = apply_simple_fsdp(
        model,
        parallel_dims=_FakeParallelDims(mesh, world),
        training=TrainingConfig(
            mixed_precision_param="bfloat16", mixed_precision_reduce="float32"
        ),
    )
    for holder in _holders(model):
      holder.shard_split_stacks()
    yield model
  finally:
    dist.destroy_process_group()


def _holders(model):
  return [m for m in model.modules() if hasattr(m, "plan_shard_dims")]


class Qwen3_235BScaleTest(absltest.TestCase):

  def test_every_weight_shards_evenly_at_2048_ranks(self) -> None:
    # 2048 ranks is 1024 v7x chips. `Shard._split_tensor` pads a short tail
    # rather than raising, so an uneven axis here is a wrong-weights bug that
    # never surfaces as one.
    with _placed_235b(2048) as model:
      for name, param in model.named_parameters():
        self.assertIsInstance(param, DTensor, name)
        for i, placement in enumerate(param.placements):
          if isinstance(placement, Shard):
            size = param.device_mesh.size(i)
            self.assertEqual(
                param.shape[placement.dim] % size, 0, f"{name} {placement}"
            )

  def test_expert_stacks_are_split_at_2048_ranks(self) -> None:
    # 128 experts < 2048 ranks, so expert stacks go on a split mesh.
    with _placed_235b(2048) as model:
      self.assertTrue(any(h.split_meshes for h in _holders(model)))

  def test_nothing_large_is_replicated_at_2048_ranks(self) -> None:
    # The pre-fix failure mode: an expert stack's dim 1 is `num_experts` = 128,
    # `data_parallel` cannot split it past 128 ranks, so it replicates in
    # silence and per-rank weights go from 3.4 GiB to 423 GiB.
    with _placed_235b(2048) as model:
      for name, param in model.named_parameters():
        if all(isinstance(p, Replicate) for p in param.placements):
          # Only the per-layer norms, which are `(n_layers, dim)` at most.
          self.assertLess(param.numel(), 1 << 20, f"{name} {tuple(param.shape)}")

  def test_the_scan_body_gathers_what_was_sharded_at_2048_ranks(self) -> None:
    # The body sees one layer's row, so its dim 0 is the stack's dim 1. Off by
    # one reassembles the weight wrong and still trains.
    with _placed_235b(2048) as model:
      holders = _holders(model)
      self.assertTrue(holders, "layer stacking did not run")
      for holder in holders:
        for name, spec in plan_stacked_gathers(holder).items():
          placements = holder._parameters[name].placements
          dims = [p.dim for p in placements if isinstance(p, Shard)]
          self.assertEqual(spec.sharded, bool(dims), name)
          if spec.sharded:
            # A split stack gathers both shard axes as one, from the first.
            self.assertEqual(spec.gather_dim, dims[0] - 1, name)

  def test_128_ranks_still_shards_every_stack_on_dim_one(self) -> None:
    # Every published 235B number was taken here. If any stack moves off dim 1
    # at 128 ranks this stops being the same run, measured or not.
    with _placed_235b(128) as model:
      for holder in _holders(model):
        self.assertEqual(holder.split_meshes, {})
        for name, param in holder._parameters.items():
          self.assertEqual(param.placements[-1], Shard(1), name)


if __name__ == "__main__":
  absltest.main()
