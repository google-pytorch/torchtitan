# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import cast
from unittest import mock

import torch
from absl.testing import absltest
from torch import nn

from torchtitan.config import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.tpu.graph_trainer import qwen3 as graph_trainer_qwen3
from torchtitan.experiments.tpu.graph_trainer.layer_stack import LayerStackHolder
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.qwen3.infra.parallelize import (
    configure_qwen_simple_fsdp_shard_dims,
)


def _model(
    *, hidden_dim: int = 8, vocab_size: int = 8, kernel_native: bool = False
) -> nn.Module:
  model = nn.Module()
  model.tok_embeddings = nn.Embedding(vocab_size, 8)
  model.lm_head = nn.Linear(8, vocab_size, bias=False)
  layer = nn.Module()
  layer.moe = nn.Module()
  layer.moe.experts = TpuGroupedExperts.Config(
      dim=8,
      hidden_dim=hidden_dim,
      num_experts=2,
      use_pallas_grouped_mm=kernel_native,
      use_kernel_native_weight_layout=kernel_native,
  ).build()
  model.layers = nn.ModuleDict({"0": layer})
  return model


def _wrapped_model() -> nn.Module:
  model = _model()
  wrapper = nn.Module()
  wrapper.inner = model.layers["0"]
  model.layers["0"] = wrapper
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
  )


class SimpleFSDPShardingTest(absltest.TestCase):

  def test_uses_hidden_axis_when_leading_axis_is_uneven(self) -> None:
    model = _model()

    configure_qwen_simple_fsdp_shard_dims(model, _parallel_dims(4))

    self.assertEqual(model.layers["0"].moe.experts._fsdp_shard_dim, 1)

  def test_keeps_default_axis_when_leading_axis_is_even(self) -> None:
    model = _model()

    configure_qwen_simple_fsdp_shard_dims(model, _parallel_dims(2))

    self.assertFalse(hasattr(model.layers["0"].moe.experts, "_fsdp_shard_dim"))

  def test_native_w2_uses_its_tile_aligned_trailing_axis(self) -> None:
    model = _model(hidden_dim=6, kernel_native=True)

    configure_qwen_simple_fsdp_shard_dims(model, _parallel_dims(4))

    experts = model.layers["0"].moe.experts
    self.assertEqual(
        experts._fsdp_shard_dim,
        {"w1_EDF": 1, "w2_EFD": 2, "w3_EDF": 1},
    )
    self.assertTrue(experts._use_transposed_w2_rhs)

  def test_finds_experts_beneath_layer_wrappers(self) -> None:
    model = _wrapped_model()

    configure_qwen_simple_fsdp_shard_dims(model, _parallel_dims(4))

    self.assertEqual(
        model.layers["0"].inner.moe.experts._fsdp_shard_dim,
        1,
    )

  def test_plans_an_axis_per_stack_when_layers_are_stacked(self) -> None:
    model = _model()
    model.layer_stacks = LayerStackHolder()
    # Layer axis in front, so dim 1 is num_experts and dim 2 is the hidden dim.
    model.layer_stacks.register_parameter(
        "moe__experts__w1_EDF", nn.Parameter(torch.empty(2, 2, 8, 8))
    )
    # The expert modules are empty shells once stacking has taken their weights.
    model.layers["0"].moe.experts._parameters.clear()

    configure_qwen_simple_fsdp_shard_dims(model, _parallel_dims(4))

    # 2 experts can't split over 4 ranks, so the stack is flattened to
    # [2, 2*8, 8] and shards on dim 1 instead of the hidden dim.
    self.assertEqual(
        model.layer_stacks._fsdp_shard_dim, {"moe__experts__w1_EDF": 1}
    )
    self.assertEqual(
        tuple(model.layer_stacks.moe__experts__w1_EDF.shape), (2, 16, 8)
    )

  def test_rejects_an_empty_expert_module_when_nothing_is_stacked(self) -> None:
    model = _model()
    model.layers["0"].moe.experts._parameters.clear()

    with self.assertRaisesRegex(ValueError, "no direct parameters"):
      configure_qwen_simple_fsdp_shard_dims(model, _parallel_dims(4))

  def test_rejects_when_hidden_axis_is_not_divisible(self) -> None:
    model = _model(hidden_dim=6)

    with self.assertRaisesRegex(ValueError, "moe.experts.*incompatible expert"):
      configure_qwen_simple_fsdp_shard_dims(model, _parallel_dims(4))

  def test_uses_hidden_axis_when_vocabulary_is_uneven(self) -> None:
    model = _model(vocab_size=6)

    configure_qwen_simple_fsdp_shard_dims(model, _parallel_dims(4))

    self.assertEqual(model.tok_embeddings._fsdp_shard_dim, 1)
    self.assertEqual(model.lm_head._fsdp_shard_dim, 1)

  def test_graph_trainer_configures_shard_dims_before_simple_fsdp(self) -> None:
    model = _model()
    parallel_dims = _parallel_dims(4)
    training = cast(
        TrainingConfig,
        SimpleNamespace(mixed_precision_reduce="bfloat16"),
    )

    with (
        mock.patch.object(
            graph_trainer_qwen3,
            "graph_trainer_parallelize_qwen3",
            return_value=model,
        ) as upstream_parallelize,
        mock.patch.object(
            graph_trainer_qwen3,
            "install_tpu_replicate_computation",
        ) as install_tpu_replicate,
    ):
      result = graph_trainer_qwen3.parallelize_qwen3(
          cast(GraphTrainerQwen3Model, model),
          training=training,
          parallel_dims=parallel_dims,
      )

    self.assertIs(result, model)
    self.assertEqual(model.layers["0"].moe.experts._fsdp_shard_dim, 1)
    upstream_parallelize.assert_called_once_with(
        model,
        training=training,
        parallel_dims=parallel_dims,
    )
    install_tpu_replicate.assert_called_once_with(model)


if __name__ == "__main__":
  absltest.main()
