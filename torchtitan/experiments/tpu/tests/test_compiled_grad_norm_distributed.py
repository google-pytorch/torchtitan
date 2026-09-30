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
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from torchtitan.config import ParallelismConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.tpu import (
  base_distributed_device_test,
  distributed,
  grad_norm,
)
from torchtitan.experiments.tpu.graph_trainer import tpu_graph_trainer
from torchtitan.experiments.tpu.qwen3.config_registry import (
  qwen3_moe_30b_fsdp_only_full_ac_benchmark,
  reg_trainer_qwen3_8b,
)
from torchtitan.experiments.tpu.train import TPUTrainer
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig, TPUTrainerConfig
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
        self.assertTrue(recipe().tpu_config.compile_grad_norm)

  def test_unsupported_training_topologies_are_rejected(self) -> None:
    cases = (
        ({"expert_parallel_degree": 2}, False, "EP1 and PP1"),
        ({"pipeline_parallel_degree": 2}, False, "EP1 and PP1"),
        ({}, True, "optimizer_in_graph"),
    )
    for parallelism, optimizer_in_graph, message in cases:
      with self.subTest(parallelism=parallelism), self.assertRaisesRegex(
          ValueError, message
      ):
        # Upstream d771abb0 made `model_spec` required; the topology checks
        # under test do not look at it.
        TPUTrainerConfig(
            model_spec=model_registry("debugmodel", attn_backend="sdpa"),
            parallelism=ParallelismConfig(**parallelism),
            tpu_config=TPUConfig(
                compile_grad_norm=True,
                optimizer_in_graph=optimizer_in_graph,
            ),
        )


def _run_clip_parity(
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
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

  # In the same workers, since spawning them is most of the test's time.
  _run_mixed_mesh_clip_parity(device, rank, world_size)


def _run_mixed_mesh_clip_parity(
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
  """Gradients on two meshes: norm and clip must match a whole-tensor reference.

  Stacks stored on a split mesh (graph_trainer/layer_stack.py) put some
  gradients on an ``(E, ranks // E)`` split of the FSDP mesh and the rest on
  the FSDP mesh itself. ``get_total_norm`` and the foreach clip both refuse to
  mix meshes, so the norm is taken per mesh and combined.
  """
  del rank
  mesh = torch.distributed.init_device_mesh(
      device.type, (world_size,), mesh_dim_names=("fsdp",)
  )
  # pylint: disable-next=protected-access
  split = mesh._unflatten(0, (2, world_size // 2), ("fsdp_outer", "fsdp_inner"))

  def make_parameters() -> list[torch.Tensor]:
    parameters = []
    for shape, scale, target_mesh, placements in (
        ((2 * world_size,), 1.0, mesh, [Shard(0)]),
        ((3,), 2.0, mesh, [Replicate()]),
        ((2, world_size, 3), 0.5, split, [Shard(0), Shard(1)]),
        ((2, world_size), -3.0, split, [Shard(0), Shard(1)]),
    ):
      numel = torch.Size(shape).numel()
      whole = scale * torch.arange(1, numel + 1, device=device).view(shape)
      parameter = torch.nn.Parameter(
          distribute_tensor(torch.zeros_like(whole), target_mesh, placements)
      )
      parameter.grad = distribute_tensor(whole, target_mesh, placements)
      parameters.append(parameter)
    # A plain tensor too, as a parameter no parallelism touched would be.
    plain = torch.nn.Parameter(torch.zeros(5, device=device))
    plain.grad = torch.arange(1.0, 6.0, device=device)
    parameters.append(plain)
    return parameters

  def whole(t: torch.Tensor) -> torch.Tensor:
    return t.full_tensor() if isinstance(t, DTensor) else t

  for norm_type in (2.0, 1.0, 3.0, float("inf")):
    parameters = make_parameters()
    grads = [p.grad for p in parameters]
    assert len(grad_norm._mesh_groups(grads)) == 3  # pylint: disable=protected-access
    want = torch.linalg.vector_norm(
        torch.cat([whole(g).flatten() for g in grads]), norm_type
    )
    got = grad_norm.global_grad_norm(grads, norm_type)
    assert not isinstance(got, DTensor), type(got)
    torch.testing.assert_close(got, want, msg=f"norm_type={norm_type}")

  max_norm = 5.0
  reference = make_parameters()
  want_norm = torch.linalg.vector_norm(
      torch.cat([whole(p.grad).flatten() for p in reference])
  )
  coef = torch.clamp(max_norm / (want_norm + 1e-6), max=1.0)
  assert coef < 1.0, "nothing is clipped, so this checks nothing"
  want_grads = [whole(p.grad) * coef for p in reference]

  candidate = make_parameters()
  compile_pass = (
      mock.patch.object(
          grad_norm,
          "tpu_backend_compile_pass",
          side_effect=lambda gm, _example_inputs, *, is_fwd: gm,
      )
      if device.type == "cpu"
      else contextlib.nullcontext()
  )
  with compile_pass:
    got_norm = grad_norm.CompiledGradNorm().clip_grad_norm_(candidate, max_norm)
  torch.testing.assert_close(got_norm, want_norm)
  for i, (parameter, want_grad) in enumerate(
      zip(candidate, want_grads, strict=True)
  ):
    got_grad = parameter.grad
    if isinstance(got_grad, DTensor):
      # Clipping scales in place; it must not move the gradient.
      assert got_grad.device_mesh == reference[i].grad.device_mesh
      assert got_grad.placements == reference[i].grad.placements
    torch.testing.assert_close(whole(got_grad), want_grad, msg=f"grad {i}")

  # Non-compiled grad_norm.clip_grad_norm_ (used when compile_grad_norm=False)
  # must also handle gradients across multiple meshes.
  eager_candidate = make_parameters()
  eager_norm = grad_norm.clip_grad_norm_(
      eager_candidate, max_norm, foreach=True
  )
  torch.testing.assert_close(eager_norm, want_norm)
  for i, (parameter, want_grad) in enumerate(
      zip(eager_candidate, want_grads, strict=True)
  ):
    got_grad = parameter.grad
    if isinstance(got_grad, DTensor):
      assert got_grad.device_mesh == reference[i].grad.device_mesh
      assert got_grad.placements == reference[i].grad.placements
    torch.testing.assert_close(
        whole(got_grad), want_grad, msg=f"eager grad {i}"
    )

  # optimizer_in_graph clipping (_in_graph_clip_grads) must also handle
  # gradients across multiple meshes.
  in_graph_grads = tuple(p.grad for p in make_parameters())
  clipped_in_graph = tpu_graph_trainer._in_graph_clip_grads(  # pylint: disable=protected-access
      in_graph_grads, max_norm
  )
  for i, (got_grad, want_grad) in enumerate(
      zip(clipped_in_graph, want_grads, strict=True)
  ):
    if isinstance(got_grad, DTensor):
      assert got_grad.device_mesh == reference[i].grad.device_mesh
      assert got_grad.placements == reference[i].grad.placements
    torch.testing.assert_close(
        whole(got_grad), want_grad, msg=f"in_graph grad {i}"
    )


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
