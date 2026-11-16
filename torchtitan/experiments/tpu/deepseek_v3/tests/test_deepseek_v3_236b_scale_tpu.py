"""DeepSeek-V3 236B's SimpleFSDP placements on fake meshes of 128-512 ranks.

The recipe's model is built on ``meta`` and put through the two calls the TPU
GraphTrainer parallelizer makes before upstream's SimpleFSDP setup, so every
placement asserted here is one the trainer would install. An axis that does not
divide the FSDP degree does not fail: ``Shard._split_tensor`` pads it, or
``data_parallel`` replicates it, and the run trains anyway.
"""

import contextlib

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, init_device_mesh, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.config import TrainingConfig
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.deepseek_v3 import config_registry
from torchtitan.experiments.tpu.deepseek_v3.infra.parallelize import (
    configure_deepseek_v3_simple_fsdp_shard_dims,
)


class _FakeParallelDims:
  """Pure FSDP on one mesh axis."""

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
def _placed_236b(world: int):
  """The 4x4x4 recipe's 236B model, FSDP-wrapped on a `world`-wide fake mesh."""
  dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=world)
  try:
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("fsdp",))
    recipe = config_registry.deepseek_v3_236b_fsdp_activation_offload_4x4x4()
    with torch.device("meta"):
      model = recipe.model_spec.model.build()
    parallel_dims = _FakeParallelDims(mesh, world)
    configure_deepseek_v3_simple_fsdp_shard_dims(model, parallel_dims)
    yield apply_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=TrainingConfig(
            mixed_precision_param="bfloat16", mixed_precision_reduce="float32"
        ),
    )
  finally:
    dist.destroy_process_group()


class DeepSeekV3_236BScaleTest(parameterized.TestCase):

  @parameterized.parameters(128, 256, 512)
  def test_every_sharded_weight_divides_evenly(self, world: int) -> None:
    with _placed_236b(world) as model:
      for name, param in model.named_parameters():
        self.assertIsInstance(param, DTensor, name)
        placement = param.placements[-1]
        if isinstance(placement, Shard):
          self.assertEqual(
              param.shape[placement.dim] % world, 0, f"{name} {placement}"
          )

  @parameterized.parameters(128, 256, 512)
  def test_only_vectors_are_replicated(self, world: int) -> None:
    # At 256+ ranks the router gate (160 rows) is narrower than the mesh, and
    # `data_parallel` would replicate it in silence on dim 0. At 0.8M
    # parameters a size threshold misses it, so allow only 1-D (norm) weights.
    with _placed_236b(world) as model:
      for name, param in model.named_parameters():
        if isinstance(param.placements[-1], Replicate):
          self.assertEqual(param.ndim, 1, f"{name} {tuple(param.shape)}")

  def test_uneven_weights_move_to_dim_one_at_128_ranks(self) -> None:
    # The axes of the 4x4x4 recipe. Experts follow the shared Qwen policy (160
    # experts do not divide 128); wkv_a (576 rows) and the router gate (160
    # rows) are DeepSeek-V3's own.
    with _placed_236b(128) as model:
      params = dict(model.named_parameters())
      self.assertEqual(
          params["layers.1.attention.wkv_a.weight"].placements[-1], Shard(1)
      )
      self.assertEqual(
          params["layers.1.moe.router.gate.weight"].placements[-1], Shard(1)
      )
      self.assertEqual(params["tok_embeddings.weight"].placements[-1], Shard(0))
      expert_placements = {
          name: param.placements[-1]
          for name, param in params.items()
          if name.startswith("layers.1.moe.routed_experts.")
      }
      self.assertTrue(expert_placements)
      for name, placement in expert_placements.items():
        self.assertIsInstance(placement, Shard, name)
        self.assertNotEqual(placement.dim, 0, name)


if __name__ == "__main__":
  absltest.main()
