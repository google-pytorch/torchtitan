"""CPU tests for Muse Glimmer's SimpleFSDP shard-dim selection.

The 202048-token vocab does not divide large FSDP degrees (e.g. 128 on
tpu7x-4x4x4), so the embedding and lm_head fall back to dim-1 sharding. The
end-to-end tests use the small debugmodel, whose vocab always divides, so this
is the only coverage of that fallback.
"""

from absl.testing import absltest
from torch import nn
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.experiments.tpu.muse_glimmer.infra.parallelize import configure_muse_glimmer_simple_fsdp_shard_dims


def _model(*, vocab_size: int, dim: int = 8) -> nn.Module:
  """Mimics Muse Glimmer's vocab-sized module layout."""
  model = nn.Module()
  model.tok_embeddings = nn.Module()
  model.tok_embeddings.embedding = nn.Embedding(vocab_size, dim)
  model.lm_head = nn.Linear(dim, vocab_size, bias=False)
  return model


def _parallel_dims(fsdp_degree: int) -> ParallelDims:
  return ParallelDims(
      dp_replicate=1,
      dp_shard=fsdp_degree,
      cp=1,
      tp=1,
      pp=1,
      ep=1,
      world_size=fsdp_degree,
      enable_sequence_parallel=False,
  )


class SimpleFSDPShardingTest(absltest.TestCase):

  def test_uses_hidden_axis_when_vocabulary_is_uneven(self) -> None:
    model = _model(vocab_size=6)

    modified = configure_muse_glimmer_simple_fsdp_shard_dims(
        model, _parallel_dims(4)
    )

    self.assertEqual(modified, ["tok_embeddings.embedding", "lm_head"])
    self.assertEqual(model.tok_embeddings.embedding._fsdp_shard_dim, 1)
    self.assertEqual(model.lm_head._fsdp_shard_dim, 1)

  def test_keeps_default_axis_when_vocabulary_is_even(self) -> None:
    model = _model(vocab_size=8)

    modified = configure_muse_glimmer_simple_fsdp_shard_dims(
        model, _parallel_dims(4)
    )

    self.assertEmpty(modified)
    self.assertFalse(hasattr(model.tok_embeddings.embedding, "_fsdp_shard_dim"))
    self.assertFalse(hasattr(model.lm_head, "_fsdp_shard_dim"))

  def test_noop_without_fsdp(self) -> None:
    model = _model(vocab_size=6)

    self.assertEmpty(
        configure_muse_glimmer_simple_fsdp_shard_dims(model, _parallel_dims(1))
    )

  def test_rejects_when_hidden_axis_is_not_divisible(self) -> None:
    model = _model(vocab_size=6, dim=6)

    with self.assertRaisesRegex(ValueError, "tok_embeddings.embedding"):
      configure_muse_glimmer_simple_fsdp_shard_dims(model, _parallel_dims(4))

  def test_skips_modules_absent_on_pipeline_stage(self) -> None:
    model = nn.Module()
    model.lm_head = nn.Linear(8, 6, bias=False)

    modified = configure_muse_glimmer_simple_fsdp_shard_dims(
        model, _parallel_dims(4)
    )

    self.assertEqual(modified, ["lm_head"])

  def test_rejects_context_parallelism(self) -> None:
    model = _model(vocab_size=6)
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=2,
        cp=2,
        tp=1,
        pp=1,
        ep=1,
        world_size=4,
        enable_sequence_parallel=False,
    )

    with self.assertRaisesRegex(NotImplementedError, "Context parallelism"):
      configure_muse_glimmer_simple_fsdp_shard_dims(model, parallel_dims)

  def test_production_shapes_on_ironwood_4x4x4(self) -> None:
    # 30B: vocab 202048, dim 6656; tpu7x-4x4x4 = 64 chips x 2 = 128 FSDP ranks.
    fsdp_degree = 128
    vocab_size, dim = 202048, 6656
    self.assertNotEqual(vocab_size % fsdp_degree, 0)
    self.assertEqual(dim % fsdp_degree, 0)


if __name__ == "__main__":
  absltest.main()
