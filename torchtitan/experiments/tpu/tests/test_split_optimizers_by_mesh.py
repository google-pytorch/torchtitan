# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for splitting optimizer param groups by device mesh.

Under expert parallelism the routed experts live on an ``(efsdp, ep)`` mesh and
dense parameters on ``(fsdp,)``. A fake two-rank process group is enough: the
split only reads each DTensor's mesh, and the optimizer math here is
rank-local.
"""

import dataclasses
import unittest
from unittest import mock

import torch
import torch.distributed as dist
from absl.testing import absltest
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.components.optimizer import (
    LRSchedulersContainer,
    OptimizersContainer,
    ParamGroupConfig,
    default_adamw,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.experiments.tpu import utils as tpu_utils
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.qwen3.config_registry import (
    graph_trainer_qwen3_debugmodel,
    qwen3_debugmodel,
)
from torchtitan.experiments.tpu.train import TPUTrainer
import torchtitan.trainer


def _replicated(local: torch.Tensor, mesh: DeviceMesh) -> DTensor:
  return DTensor.from_local(
      local, mesh, [Replicate()] * mesh.ndim, run_check=False
  )


def _model(**params: torch.Tensor) -> nn.Module:
  """A module whose parameters are ``params``, in keyword order."""
  model = nn.Module()
  for name, tensor in params.items():
    model.register_parameter(name, nn.Parameter(tensor))
  return model


def _ep_model(dense_mesh: DeviceMesh, expert_mesh: DeviceMesh) -> nn.Module:
  """The expert weight sits between two dense ones in parameter order."""
  return _model(
      embed=_replicated(torch.full((4,), 1.0), dense_mesh),
      experts=_replicated(torch.full((2, 4), 2.0), expert_mesh),
      norm=_replicated(torch.full((4,), 3.0), dense_mesh),
  )


def _build_optimizers(model: nn.Module) -> OptimizersContainer:
  config = default_adamw(lr=0.1, implementation="for-loop")
  return config.build(model_parts=[model])


def _options(group: dict) -> dict:
  return {
      k: v for k, v in group.items() if k not in ("params", "param_names")
  }


def _ids(params) -> list[int]:
  return [id(param) for param in params]


class _FakeTwoRankTestCase(absltest.TestCase):
  """Runs on a fake two-rank process group with a dense and an expert mesh."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if dist.is_initialized():
      raise unittest.SkipTest("a process group is already initialized")
    from torch.testing._internal.distributed.fake_pg import FakeStore  # pylint: disable=g-import-not-at-top

    dist.init_process_group("fake", rank=0, world_size=2, store=FakeStore())
    cls.dense_mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("fsdp",))
    cls.expert_mesh = init_device_mesh(
        "cpu", (1, 2), mesh_dim_names=("efsdp", "ep")
    )

  @classmethod
  def tearDownClass(cls):
    dist.destroy_process_group()
    super().tearDownClass()


@unittest.skipUnless(dist.is_available(), "torch.distributed is unavailable")
class SplitOptimizersByMeshTest(_FakeTwoRankTestCase):

  def test_one_mesh_leaves_the_groups_alone(self) -> None:
    cases = {
        "dtensors": _model(
            embed=_replicated(torch.ones(4), self.dense_mesh),
            norm=_replicated(torch.ones(4), self.dense_mesh),
        ),
        "plain_tensors": _model(embed=torch.ones(4), norm=torch.ones(4)),
    }
    for name, model in cases.items():
      with self.subTest(name):
        optimizers = _build_optimizers(model)
        (optimizer,) = optimizers.optimizers
        (group,) = optimizer.param_groups
        params = list(group["params"])

        self.assertFalse(tpu_utils.split_optimizers_by_mesh(optimizers))

        self.assertLen(optimizer.param_groups, 1)
        self.assertIs(optimizer.param_groups[0], group)
        self.assertEqual(_ids(group["params"]), _ids(params))
        self.assertEqual(group["param_names"], ["embed", "norm"])

  def test_equal_meshes_count_as_one_mesh(self) -> None:
    """FSDP2 with TP gives each ``fully_shard`` call its own, equal mesh."""
    same_mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("fsdp",))
    self.assertIsNot(same_mesh, self.dense_mesh)
    self.assertEqual(same_mesh, self.dense_mesh)
    optimizers = _build_optimizers(
        _model(
            embed=_replicated(torch.ones(4), self.dense_mesh),
            norm=_replicated(torch.ones(4), same_mesh),
        )
    )

    self.assertFalse(tpu_utils.split_optimizers_by_mesh(optimizers))

    self.assertLen(optimizers.optimizers[0].param_groups, 1)

  def test_two_meshes_split_into_one_group_per_mesh(self) -> None:
    model = _ep_model(self.dense_mesh, self.expert_mesh)
    optimizers = _build_optimizers(model)
    (optimizer,) = optimizers.optimizers
    options = _options(optimizer.param_groups[0])

    self.assertTrue(tpu_utils.split_optimizers_by_mesh(optimizers))

    dense, expert = optimizer.param_groups
    self.assertEqual(_ids(dense["params"]), _ids([model.embed, model.norm]))
    self.assertEqual(dense["param_names"], ["embed", "norm"])
    self.assertEqual(_ids(expert["params"]), _ids([model.experts]))
    self.assertEqual(expert["param_names"], ["experts"])
    self.assertEqual(_options(dense), options)
    self.assertEqual(_options(expert), options)

  def test_plain_tensors_share_one_group(self) -> None:
    model = _model(
        scale=torch.ones(4),
        embed=_replicated(torch.ones(4), self.dense_mesh),
        bias=torch.zeros(4),
    )
    optimizers = _build_optimizers(model)

    self.assertTrue(tpu_utils.split_optimizers_by_mesh(optimizers))

    plain, dense = optimizers.optimizers[0].param_groups
    self.assertEqual(_ids(plain["params"]), _ids([model.scale, model.bias]))
    self.assertEqual(plain["param_names"], ["scale", "bias"])
    self.assertEqual(_ids(dense["params"]), _ids([model.embed]))
    self.assertEqual(dense["param_names"], ["embed"])

  def test_second_call_changes_nothing(self) -> None:
    optimizers = _build_optimizers(
        _ep_model(self.dense_mesh, self.expert_mesh)
    )
    (optimizer,) = optimizers.optimizers
    self.assertTrue(tpu_utils.split_optimizers_by_mesh(optimizers))
    groups = list(optimizer.param_groups)
    contents = [(_ids(g["params"]), list(g["param_names"])) for g in groups]

    self.assertFalse(tpu_utils.split_optimizers_by_mesh(optimizers))

    self.assertLen(optimizer.param_groups, len(groups))
    for group, before in zip(optimizer.param_groups, groups, strict=True):
      self.assertIs(group, before)
    self.assertEqual(
        [(_ids(g["params"]), g["param_names"]) for g in optimizer.param_groups],
        contents,
    )

  def test_param_names_length_mismatch_raises_before_any_change(self) -> None:
    model = _ep_model(self.dense_mesh, self.expert_mesh)
    # One AdamW with two groups: the first needs a split, the second is bad.
    config = OptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r"^(embed|experts)$",
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 0.1},
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 0.2},
            ),
        ],
        implementation="for-loop",
    )
    optimizers = config.build(model_parts=[model])
    (optimizer,) = optimizers.optimizers
    first, second = optimizer.param_groups
    second["param_names"] = []

    with self.assertRaisesRegex(
        ValueError, r"param_names length \(0\) != params length \(1\)"
    ):
      tpu_utils.split_optimizers_by_mesh(optimizers)

    self.assertLen(optimizer.param_groups, 2)
    self.assertIs(optimizer.param_groups[0], first)
    self.assertIs(optimizer.param_groups[1], second)
    self.assertEqual(first["param_names"], ["embed", "experts"])

  def test_refuses_to_split_after_lr_schedulers_are_built(self) -> None:
    optimizers = _build_optimizers(
        _ep_model(self.dense_mesh, self.expert_mesh)
    )
    LRSchedulersContainer.Config(warmup_steps=1).build(
        optimizers=optimizers, training_steps=4
    )

    with self.assertRaisesRegex(RuntimeError, "before the LR schedulers"):
      tpu_utils.split_optimizers_by_mesh(optimizers)

    self.assertLen(optimizers.optimizers[0].param_groups, 1)

  def _step(self, model: nn.Module, optimizers: OptimizersContainer) -> None:
    for i, param in enumerate(model.parameters()):
      param.grad = _replicated(
          torch.full(param.shape, 0.1 * (i + 1)), param.device_mesh
      )
    optimizers.step()

  def _assert_state_dicts_equal(self, actual: dict, expected: dict) -> None:
    self.assertEqual(sorted(actual), sorted(expected))
    for key, want in expected.items():
      got = actual[key]
      if isinstance(want, torch.Tensor):
        torch.testing.assert_close(
            got.to_local() if isinstance(got, DTensor) else got,
            want.to_local() if isinstance(want, DTensor) else want,
            rtol=0,
            atol=0,
            msg=key,
        )
      else:
        self.assertEqual(got, want, msg=key)

  def test_optimizer_state_dict_does_not_depend_on_the_split(self) -> None:
    """Checkpoints are FQN-keyed, so a split changes neither save nor load."""
    model = _ep_model(self.dense_mesh, self.expert_mesh)
    optimizers = _build_optimizers(model)
    self._step(model, optimizers)
    saved = optimizers.state_dict()
    self.assertIn("state.norm.exp_avg", saved)
    self.assertIn("param_groups.experts.lr", saved)

    self.assertTrue(tpu_utils.split_optimizers_by_mesh(optimizers))
    self._assert_state_dicts_equal(optimizers.state_dict(), saved)

    resumed = _build_optimizers(_ep_model(self.dense_mesh, self.expert_mesh))
    self.assertTrue(tpu_utils.split_optimizers_by_mesh(resumed))
    resumed.load_state_dict(saved)
    self._assert_state_dicts_equal(resumed.state_dict(), saved)

  def test_wrapper_runs_the_model_hook_then_splits(self) -> None:
    for name, with_model_hook in (("model_hook", True), ("no_hook", False)):
      with self.subTest(name):
        model = _ep_model(self.dense_mesh, self.expert_mesh)
        optimizers = _build_optimizers(model)
        (optimizer,) = optimizers.optimizers
        groups_seen_by_hook = []
        model_hook = mock.Mock(
            side_effect=lambda opts, *_: groups_seen_by_hook.append(
                len(opts.optimizers[0].param_groups)
            )
        )

        hook = tpu_utils.with_split_optimizers_by_mesh(
            model_hook if with_model_hook else None
        )
        hook(optimizers, [model], mock.sentinel.parallel_dims)

        if with_model_hook:
          model_hook.assert_called_once_with(
              optimizers, [model], mock.sentinel.parallel_dims
          )
          self.assertEqual(groups_seen_by_hook, [1])
        self.assertLen(optimizer.param_groups, 2)


@unittest.skipUnless(dist.is_available(), "torch.distributed is unavailable")
class TpuTrainersSplitOptimizersTest(_FakeTwoRankTestCase):
  """Both TPU trainers split inside the hook the base trainer runs first.

  The base ``__init__`` is replaced by the optimizer section of upstream
  ``Trainer.__init__``: build optimizers, run the model spec's
  ``post_optimizer_build_fn``, then build the LR schedulers.
  """

  def _graph_trainer_config(self) -> TpuGraphTrainer.Config:
    config = graph_trainer_qwen3_debugmodel()
    config.compile.mode = "aot_fx_trace"
    config.compile.optimizer_compile = "in_graph"
    return config

  def test_both_trainers_split_before_building_lr_schedulers(self) -> None:
    cases = (
        (TPUTrainer, torchtitan.trainer.Trainer, qwen3_debugmodel),
        (TpuGraphTrainer, GraphTrainer, self._graph_trainer_config),
    )
    for trainer_cls, base_cls, make_config in cases:
      with self.subTest(trainer=trainer_cls.__name__):
        model = _ep_model(self.dense_mesh, self.expert_mesh)
        optimizers = _build_optimizers(model)
        (optimizer,) = optimizers.optimizers
        groups_seen_by_model_hook = []
        groups_at_lr_scheduler_build = []
        model_hook = mock.Mock(
            side_effect=lambda opts, *_: groups_seen_by_model_hook.append(
                len(opts.optimizers[0].param_groups)
            )
        )
        config = make_config()
        config.model_spec = dataclasses.replace(
            config.model_spec, post_optimizer_build_fn=model_hook
        )

        def base_init(trainer, config, model=model, optimizers=optimizers):
          trainer.config = config
          trainer.model_parts = [model]
          trainer.optimizers = optimizers
          hook = config.model_spec.post_optimizer_build_fn
          if hook is not None:
            hook(optimizers, trainer.model_parts, mock.sentinel.parallel_dims)
          groups_at_lr_scheduler_build.append(
              len(optimizers.optimizers[0].param_groups)
          )
          trainer.lr_schedulers = config.lr_scheduler.build(
              optimizers=optimizers, training_steps=config.training.steps
          )

        with (
            mock.patch.object(base_cls, "__init__", base_init),
            mock.patch.object(
                tpu_utils,
                "split_optimizers_by_mesh",
                wraps=tpu_utils.split_optimizers_by_mesh,
            ) as split,
        ):
          trainer = trainer_cls(config)

        model_hook.assert_called_once_with(
            optimizers, [model], mock.sentinel.parallel_dims
        )
        self.assertEqual(groups_seen_by_model_hook, [1])
        split.assert_called_once_with(optimizers)
        self.assertEqual(groups_at_lr_scheduler_build, [2])
        (scheduler,) = trainer.lr_schedulers.schedulers
        self.assertLen(scheduler.base_lrs, 2)
        for group in optimizer.param_groups:
          self.assertIn("initial_lr", group)
        if trainer_cls is TpuGraphTrainer:
          # The in-graph optimizer path marks every split group capturable.
          self.assertTrue(
              all(group["capturable"] for group in optimizer.param_groups)
          )


if __name__ == "__main__":
  absltest.main()
