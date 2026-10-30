# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the standalone compiled TPU gradient-norm path."""

import contextlib
from types import SimpleNamespace
from unittest import mock

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from torchtitan.config import ParallelismConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.tpu import (
  base_distributed_device_test,
  distributed,
  grad_norm,
)
from torchtitan.experiments.tpu.graph_trainer import tpu_graph_trainer
from torchtitan.experiments.tpu.graph_trainer.config import (
  apply_tpu_graph_trainer_model_transforms,
)
from torchtitan.experiments.tpu.qwen3.config_registry import (
  qwen3_8b,
  qwen3_moe_30b_fsdp_only_full_ac_benchmark,
  reg_trainer_qwen3_8b,
)
from torchtitan.experiments.tpu.train import TPUTrainer
from torchtitan.experiments.tpu.tpu_job_config import (
  TPUTrainerConfig,
  TpuTrainerCompileConfig,
)
from torchtitan.models.llama3.config_registry import model_registry


class CompiledGradNormUnitTest(absltest.TestCase):
  def test_both_tpu_trainers_use_shared_compiled_norm(self) -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    expected = torch.tensor(2.0)
    for trainer_cls in (TPUTrainer, tpu_graph_trainer.TpuGraphTrainer):
      with self.subTest(trainer=trainer_cls.__name__):
        compiled_norm = mock.Mock()
        compiled_norm.clip_grad_norm_.return_value = expected
        trainer = SimpleNamespace(
            _compiled_grad_norm=compiled_norm,
            model_parts=[model],
            config=SimpleNamespace(training=SimpleNamespace(max_norm=1.0)),
        )

        actual = trainer_cls._clip_grad_norm(trainer)

        self.assertIs(actual, expected)
        compiled_norm.clip_grad_norm_.assert_called_once_with(
            list(model.parameters()),
            1.0,
        )

  def test_recipes_enable_compiled_grad_norm(self) -> None:
    for recipe in (
        reg_trainer_qwen3_8b,
        qwen3_moe_30b_fsdp_only_full_ac_benchmark,
    ):
      with self.subTest(recipe=recipe.__name__):
        self.assertTrue(recipe().compile.compile_grad_norm)

  def test_unsupported_training_topologies_are_rejected(self) -> None:
    cases = (
        ({"expert_parallel_degree": 2}, "EP1 and PP1"),
        ({"pipeline_parallel_degree": 2}, "EP1 and PP1"),
    )
    for parallelism, message in cases:
      with self.subTest(parallelism=parallelism), self.assertRaisesRegex(
          ValueError, message
      ):
        # Upstream d771abb0 made `model_spec` required; the topology checks
        # under test do not look at it.
        TPUTrainerConfig(
            model_spec=model_registry("debugmodel", attn_backend="sdpa"),
            parallelism=ParallelismConfig(**parallelism),
            compile=TpuTrainerCompileConfig(compile_grad_norm=True),
        )

    config = qwen3_8b()
    config.compile.compile_grad_norm = True
    config.compile.optimizer_compile = "in_graph"
    with self.assertRaisesRegex(ValueError, "optimizer_compile='in_graph'"):
      apply_tpu_graph_trainer_model_transforms(config)


def _run_clip_parity(
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
  del rank
  mesh = torch.distributed.DeviceMesh(
      device.type,
      torch.arange(world_size).reshape(2, world_size // 2),
  )
  max_norm = 5.0

  def make_parameters() -> list[torch.Tensor]:
    parameters = []
    for size, scale, placements in (
        (2 * world_size + 1, 1.0, [Shard(0), Replicate()]),
        (3 * world_size + 1, 0.5, [Replicate(), Shard(0)]),
        (1, 2.0, [Replicate(), Replicate()]),
    ):
      parameter = torch.nn.Parameter(
          distribute_tensor(
              torch.zeros(size, device=device),
              mesh,
              placements,
          )
      )
      parameter.grad = distribute_tensor(
          scale * torch.arange(1, size + 1, device=device),
          mesh,
          placements,
      )
      parameters.append(parameter)
    return parameters

  reference = make_parameters()
  candidate = make_parameters()

  reference_norm = dist_utils.clip_grad_norm_(
      reference,
      max_norm,
      foreach=True,
  )
  traced_graphs = []

  def capture_graph(gm, _example_inputs, *, is_fwd):
    assert is_fwd is True
    traced_graphs.append(gm)
    return gm

  compile_pass = (
      mock.patch.object(
          grad_norm,
          "tpu_backend_compile_pass",
          side_effect=capture_graph,
      )
      if device.type == "cpu"
      else contextlib.nullcontext()
  )
  with compile_pass:
    compiled_norm = grad_norm.CompiledGradNorm()
    candidate_norm = compiled_norm.clip_grad_norm_(
        candidate,
        max_norm,
    )
  if device.type == "cpu":
    targets = [str(node.target) for node in traced_graphs[0].graph.nodes]
    assert any("all_reduce" in target for target in targets)
    assert not any("all_gather" in target for target in targets)

  torch.testing.assert_close(candidate_norm, reference_norm)
  for candidate_parameter, reference_parameter in zip(
      candidate, reference, strict=True
  ):
    torch.testing.assert_close(
        candidate_parameter.grad.to_local(),
        reference_parameter.grad.to_local(),
    )

  candidate[0].grad = candidate[0].grad.redistribute(
      mesh,
      [Replicate(), Shard(0)],
  )
  try:
    compiled_norm.clip_grad_norm_(candidate, max_norm)
  except ValueError as error:
    assert "gradient metadata changed" in str(error)
  else:
    raise AssertionError("changed DTensor placements were not rejected")


class CompiledGradNormDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):

  def test_matches_dtensor_clip_grad_norm(self) -> None:
    num_devices = self.num_devices
    if self.accelerator_device_type.value == "cpu":
      num_devices = 4
    elif num_devices < 4:
      self.skipTest("2-D gradient-norm parity requires at least four devices")
    distributed.run_distributed(
        num_devices,
        self.accelerator_device_type,
        _run_clip_parity,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
