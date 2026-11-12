"""Tests for how ``loss.global_vocab_size`` is filled on the TPU lane.

spmd_types loss-parallel CE sees only the local vocab shard, so it needs the
full vocab size. It is filled from the *final* model config right before the
trainer builds the model and the loss (``set_global_vocab_size``) rather than
snapshotted into the job config at construction, so it cannot go stale when the
model vocab changes afterwards (``train_minimal`` widening it to the
tokenizer's, or a derived recipe overriding ``vocab_size``).
"""

import contextlib
import dataclasses
from unittest import mock

from absl.testing import absltest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.experiments.tpu.components.loss import set_global_vocab_size
from torchtitan.experiments.tpu.llama3 import config_registry


@contextlib.contextmanager
def _fake_tp_mesh(tp: int):
  dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=tp)
  try:
    # The dense SPMD mesh the loss asserts types against: dp / cp / tp.
    mesh = init_device_mesh("cpu", (1, 1, tp), mesh_dim_names=("dp", "cp", "tp"))
    with set_current_spmd_mesh(mesh):
      yield
  finally:
    dist.destroy_process_group()


class _Model:

  def __init__(self, vocab_size):
    self.vocab_size = vocab_size


class SetGlobalVocabSizeTest(absltest.TestCase):

  def test_fills_from_model(self):
    cfg = CrossEntropyLoss.Config()
    set_global_vocab_size(cfg, _Model(4096))
    self.assertEqual(cfg.global_vocab_size, 4096)

  def test_fills_the_chunked_wrapper_inner_loss(self):
    cfg = ChunkedLossWrapper.Config()
    set_global_vocab_size(cfg, _Model(4096))
    self.assertEqual(cfg.loss_fn.global_vocab_size, 4096)

  def test_keeps_an_explicit_value_and_rejects_a_mismatch(self):
    cfg = CrossEntropyLoss.Config(global_vocab_size=4096)
    set_global_vocab_size(cfg, _Model(4096))
    self.assertEqual(cfg.global_vocab_size, 4096)
    with self.assertRaisesRegex(ValueError, "does not match"):
      set_global_vocab_size(cfg, _Model(2048))

  def test_configs_without_a_vocab_are_left_alone(self):
    @dataclasses.dataclass
    class Mse:
      pass

    set_global_vocab_size(Mse(), _Model(4096))  # no field: no-op
    cfg = CrossEntropyLoss.Config()
    set_global_vocab_size(cfg, object())  # model without a vocab: no-op
    self.assertIsNone(cfg.global_vocab_size)


class JobConfigVocabSizeTest(absltest.TestCase):

  def test_post_init_does_not_snapshot_the_vocab(self):
    config = config_registry.llama3_debugmodel()
    self.assertIsNone(config.loss.global_vocab_size)

  def test_tokenizer_widening_is_seen_by_the_loss_under_tp(self):
    # `train_minimal` widens `model_config.vocab_size` to the tokenizer's after
    # the config is built. A vocab snapshotted at construction would make the
    # loss-parallel CE reject the (now larger) local shard; filling from the
    # final model config does not.
    config = config_registry.llama3_debugmodel()
    model_config = config.model_spec.model
    old_vocab = model_config.vocab_size
    model_config.vocab_size = 2 * old_vocab  # what the widening does
    tp = 2
    set_global_vocab_size(config.loss, model_config)
    loss_fn = config.loss.build(compile_config=config.compile)
    stale_loss_fn = CrossEntropyLoss(
        CrossEntropyLoss.Config(global_vocab_size=old_vocab)
    )
    tokens = 8
    local_logits = torch.randn(tokens, model_config.vocab_size // tp)
    labels = torch.randint(0, model_config.vocab_size, (tokens,))
    with _fake_tp_mesh(tp):
      loss, _ = loss_fn(local_logits, labels)
      self.assertEqual(loss.shape, ())
      with self.assertRaisesRegex(ValueError, "expected local vocab size"):
        stale_loss_fn(local_logits, labels)

  def test_derived_recipe_overriding_vocab_is_seen_by_the_loss_under_tp(self):
    # A recipe derived from another that changes `vocab_size` after the base
    # config was constructed.
    config = config_registry.llama3_debugmodel()
    config.model_spec.model.vocab_size = 4096
    config.parallelism.tensor_parallel_degree = 2
    set_global_vocab_size(config.loss, config.model_spec.model)
    self.assertEqual(config.loss.global_vocab_size, 4096)


class _StopInit(Exception):
  pass


class TrainerFillsBeforeBaseInitTest(absltest.TestCase):
  """`Trainer.__init__` builds `loss_fn` from `config.loss` (and binds it into the
  PP schedule), so both TPU trainers must fill `global_vocab_size` before
  calling it."""

  def _captured_vocab(self, trainer_cls, base_cls, config):
    seen = {}

    def fake_base_init(self_, cfg):
      seen["vocab"] = cfg.loss.global_vocab_size
      raise _StopInit()

    with mock.patch.object(base_cls, "__init__", fake_base_init):
      with self.assertRaises(_StopInit):
        trainer_cls(config)
    return seen["vocab"]

  def test_tpu_trainer_fills_before_trainer_init(self):
    from torchtitan.experiments.tpu import train as tpu_train
    import torchtitan.train

    config = config_registry.llama3_debugmodel()
    self.assertIsNone(config.loss.global_vocab_size)
    self.assertEqual(
        self._captured_vocab(tpu_train.TPUTrainer, torchtitan.train.Trainer, config),
        config.model_spec.model.vocab_size,
    )

  def test_tpu_graph_trainer_fills_before_graph_trainer_init(self):
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
    from torchtitan.experiments.tpu.graph_trainer import tpu_graph_trainer

    config = config_registry.graph_trainer_llama3_debugmodel()
    self.assertIsNone(config.loss.global_vocab_size)
    self.assertEqual(
        self._captured_vocab(tpu_graph_trainer.TpuGraphTrainer, GraphTrainer, config),
        config.model_spec.model.vocab_size,
    )


if __name__ == "__main__":
  absltest.main()
