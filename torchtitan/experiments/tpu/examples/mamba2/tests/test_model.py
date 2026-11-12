# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Mamba-2 TPU model architecture and configurations."""

from absl.testing import absltest
import torch
from torchtitan.experiments.tpu.examples import mamba2
from torchtitan.experiments.tpu.examples.mamba2 import model


class Mamba2ModelTest(absltest.TestCase):

  def test_instantiation_and_forward(self):
    config = model.Mamba2Config(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_state=16,
        d_conv=4,
        chunk_size=16,
    )
    mamba2_model = model.Mamba2LM(config)
    mamba2_model.init_weights()

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = mamba2_model(input_ids)
    self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))

  def test_model_backward(self):
    """Verifies end-to-end model backward pass computes valid gradients."""
    config = model.Mamba2Config(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_state=16,
        d_conv=4,
        chunk_size=16,
    )
    mamba2_model = model.Mamba2LM(config)
    mamba2_model.init_weights()

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = mamba2_model(input_ids)
    loss = logits.sum()
    loss.backward()

    for name, param in mamba2_model.named_parameters():
      if param.requires_grad:
        self.assertIsNotNone(param.grad, f"Param {name} has no gradient")
        self.assertTrue(
            torch.all(torch.isfinite(param.grad)),
            f"Param {name} has NaN or Inf gradient",
        )

  def test_weight_tying(self):
    config = model.Mamba2Config(
        vocab_size=128,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_state=8,
        d_conv=4,
        chunk_size=16,
        enable_weight_tying=True,
    )
    mamba2_model = model.Mamba2LM(config)
    mamba2_model.init_weights()
    self.assertIs(
        mamba2_model.lm_head.weight, mamba2_model.tok_embeddings.weight
    )

  def test_grouped_heads_model(self):
    """Verifies end-to-end model execution with GSSM grouped heads (n_groups != n_heads)."""
    config = model.Mamba2Config(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_groups=2,
        d_state=16,
        d_conv=4,
        chunk_size=16,
    )
    mamba2_model = model.Mamba2LM(config)
    mamba2_model.init_weights()

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = mamba2_model(input_ids)
    self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
    loss = logits.sum()
    loss.backward()

    self.assertIsNotNone(mamba2_model.tok_embeddings.weight.grad)
    self.assertGreater(
        mamba2_model.tok_embeddings.weight.grad.abs().sum().item(), 0
    )

  def test_model_registry_and_configs(self):
    spec = mamba2.model_registry("debugmodel")
    self.assertEqual(spec.name, "mamba2")
    self.assertEqual(spec.flavor, "debugmodel")

    debug_trainer_cfg = mamba2.mamba2_debug()
    self.assertEqual(debug_trainer_cfg.model_spec.name, "mamba2")

    cfg_130m = mamba2.mamba2_130m()
    self.assertEqual(cfg_130m.model_spec.flavor, "130M")


if __name__ == "__main__":
  absltest.main()
