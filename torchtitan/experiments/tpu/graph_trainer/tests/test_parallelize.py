"""The shared TPU graph-trainer parallelize runs its steps in the one order that works."""

import unittest
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from torch import nn

from torchtitan.experiments.tpu.graph_trainer import parallelize as parallelize_lib


class _DummyHolder(nn.Module):
  """Stands in for ``LayerStackHolder``: records the shard-dim planning calls."""

  def __init__(self):
    super().__init__()
    self.flattened_for = None
    self.split_meshes_built = False

  def flatten_for(self, degree: int) -> list[str]:
    self.flattened_for = degree
    return []

  def plan_shard_dims(self, degree: int) -> dict[str, int]:
    return {"w": 1}

  def build_split_meshes(self) -> None:
    self.split_meshes_built = True


class _DummyModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.tok_embeddings = nn.Embedding(128256, 4096, device="meta")
    self.lm_head = nn.Linear(4096, 128256, bias=False, device="meta")
    self.layer_stacks = _DummyHolder()


class ParallelizeTest(unittest.TestCase):

  def test_shard_dims_are_planned_before_sharding_and_split_meshes_after(self):
    model = _DummyModel()
    seen: list[str] = []
    pdims = SimpleNamespace(dp_shard=512, cp=1, seq_len_divisor=1)
    training = SimpleNamespace(num_tokens_per_microbatch_per_dp_rank=4096)

    def fake_apply_tpu_fsdp(m, *, parallel_dims, training):
      del parallel_dims, training
      # The stacks and the vocab weights must already carry their axes.
      self.assertEqual(m.layer_stacks.flattened_for, 512)
      self.assertEqual(m.layer_stacks._fsdp_shard_dim, {"w": 1})
      self.assertEqual(m.tok_embeddings._fsdp_shard_dim, 1)
      self.assertEqual(m.lm_head._fsdp_shard_dim, 1)
      self.assertFalse(m.layer_stacks.split_meshes_built)
      seen.append("fsdp")
      return m

    def fake_apply_compile(m, **kwargs):
      del kwargs
      self.assertTrue(m.layer_stacks.split_meshes_built)
      seen.append("compile")
      return m

    with (
        mock.patch.object(parallelize_lib, "apply_tpu_fsdp", fake_apply_tpu_fsdp),
        mock.patch.object(parallelize_lib, "apply_compile", fake_apply_compile),
        mock.patch.object(parallelize_lib, "configure_moe_aux_loss_reduction"),
    ):
      out = parallelize_lib.parallelize_tpu_graph_trainer(
          model,
          annotate=lambda m: seen.append("annotate"),
          training=training,
          parallel_dims=pdims,
          parallelism=SimpleNamespace(spmd_backend="dtensor"),
          compile_config=SimpleNamespace(),
          ac_config=SimpleNamespace(),
          dump_folder="",
      )
    self.assertIs(out, model)
    self.assertEqual(seen, ["annotate", "fsdp", "compile"])

  def test_rejects_a_token_count_the_mesh_cannot_split(self):
    pdims = SimpleNamespace(dp_shard=1, cp=1, seq_len_divisor=4)
    with self.assertRaisesRegex(ValueError, "divisible"):
      parallelize_lib.parallelize_tpu_graph_trainer(
          _DummyModel(),
          annotate=lambda m: None,
          training=SimpleNamespace(num_tokens_per_microbatch_per_dp_rank=6),
          parallel_dims=pdims,
          parallelism=SimpleNamespace(spmd_backend="dtensor"),
          compile_config=SimpleNamespace(),
          ac_config=SimpleNamespace(),
          dump_folder="",
      )


if __name__ == "__main__":
  absltest.main()
