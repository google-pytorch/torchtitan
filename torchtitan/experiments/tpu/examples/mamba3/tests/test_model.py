# Copyright 2026 Google LLC

"""Unit tests for Mamba-3 TPU model architecture and configurations."""

from absl.testing import absltest
import torch
from torchtitan.experiments.tpu.examples import mamba3
from torchtitan.experiments.tpu.examples.mamba3 import model


class Mamba3ModelTest(absltest.TestCase):

  def test_instantiation_and_forward(self):
    config = model.Mamba3Config(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_state=32,
        chunk_size=16,
    )
    mamba3_model = model.Mamba3LM(config)
    mamba3_model.init_weights()

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = mamba3_model(input_ids)
    self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))

  def test_model_backward(self):
    """Verifies end-to-end model backward pass computes valid gradients."""
    config = model.Mamba3Config(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_state=32,
        chunk_size=16,
    )
    mamba3_model = model.Mamba3LM(config)
    mamba3_model.init_weights()

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = mamba3_model(input_ids)
    loss = logits.sum()
    loss.backward()

    for name, param in mamba3_model.named_parameters():
      if param.requires_grad:
        self.assertIsNotNone(param.grad, f"Param {name} has no gradient")
        self.assertTrue(
            torch.all(torch.isfinite(param.grad)),
            f"Param {name} has NaN or Inf gradient",
        )

  def test_weight_tying(self):
    config = model.Mamba3Config(
        vocab_size=128,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_state=16,
        chunk_size=16,
        enable_weight_tying=True,
    )
    mamba3_model = model.Mamba3LM(config)
    mamba3_model.init_weights()
    self.assertIs(
        mamba3_model.lm_head.weight, mamba3_model.tok_embeddings.weight
    )

  def test_mimo_model(self):
    """Verifies end-to-end model execution with MIMO recurrence."""
    config = model.Mamba3Config(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_state=32,
        is_mimo=True,
        mimo_rank=2,
        rope_fraction=0.5,
        chunk_size=16,
    )
    mamba3_model = model.Mamba3LM(config)
    mamba3_model.init_weights()

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = mamba3_model(input_ids)
    self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
    loss = logits.sum()
    loss.backward()

    self.assertIsNotNone(mamba3_model.tok_embeddings.weight.grad)
    self.assertGreater(
        mamba3_model.tok_embeddings.weight.grad.abs().sum().item(), 0
    )

  def test_model_registry_and_configs(self):
    spec = mamba3.model_registry("2.7B")
    self.assertEqual(spec.name, "mamba3")
    self.assertEqual(spec.flavor, "2.7B")

    cfg_2_7b_256k = mamba3.mamba3_2_7b_256k()
    self.assertEqual(cfg_2_7b_256k.model_spec.name, "mamba3")
    self.assertEqual(cfg_2_7b_256k.model_spec.flavor, "2.7B")


if __name__ == "__main__":
  absltest.main()
