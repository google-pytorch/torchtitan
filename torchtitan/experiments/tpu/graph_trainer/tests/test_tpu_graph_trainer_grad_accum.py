"""CPU tests for gradient accumulation in TpuGraphTrainer."""

import copy
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.nn import functional as F
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.experiments.tpu.graph_trainer import tpu_graph_trainer
from torchtitan.experiments.tpu.graph_trainer.tests.test_tpu_graph_trainer_unit_tpu import (
    _run_grad_accum_step,
    _traced_trainer,
)


class _ShardedModel(nn.Module):
  """Params are DTensors gathered in forward, as SimpleFSDP does."""

  def __init__(self, mesh) -> None:
    super().__init__()
    self.embed = nn.Parameter(
        distribute_tensor(torch.randn(128, 32), mesh, [Shard(0)])
    )
    self.linear = nn.Parameter(
        distribute_tensor(torch.randn(32, 32) / 32**0.5, mesh, [Shard(0)])
    )
    # A plain-tensor param next to the DTensors, as an unsharded param would be.
    self.scale = nn.Parameter(torch.ones(32))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    embed = self.embed.redistribute(placements=[Replicate()]).to_local()
    linear = self.linear.redistribute(placements=[Replicate()]).to_local()
    h = torch.relu(F.embedding(x, embed) @ linear.T) * self.scale
    return h @ embed.T


def _full(t: torch.Tensor) -> torch.Tensor:
  return t.full_tensor() if isinstance(t, DTensor) else t


class GradAccumulationTest(absltest.TestCase):

  def test_dtensor_params_match_full_batch(self) -> None:
    """GA=2 matches eager full-batch grads when params are DTensors."""
    dist.init_process_group(
        "gloo", store=dist.HashStore(), rank=0, world_size=1
    )
    self.addCleanup(dist.destroy_process_group)
    model = _ShardedModel(init_device_mesh("cpu", (1,)))
    inputs = torch.randint(0, 128, (4, 16))
    labels = torch.randint(0, 128, (4, 16))

    ref = copy.deepcopy(model)
    loss = F.cross_entropy(
        ref(inputs).view(-1, 128), labels.view(-1), reduction="sum"
    )
    (loss / labels.numel()).backward()

    trainer = _traced_trainer(
        model, inputs[:2], labels[:2], torch.tensor(float(labels.numel()))
    )
    _run_grad_accum_step(trainer, model, inputs, labels, 2)
    for (name, got), want in zip(
        model.named_parameters(), ref.parameters(), strict=True
    ):
      self.assertIs(type(got.grad), type(want.grad), name)
      torch.testing.assert_close(
          _full(got.grad), _full(want.grad), rtol=1e-5, atol=1e-6, msg=name
      )

  def test_in_graph_optimizer_rejects_grad_accumulation(self) -> None:
    config = ConfigManager().parse_args([
        "--module=torchtitan.experiments.tpu.qwen3",
        "--config=graph_trainer_qwen3_debugmodel",
    ])
    config.compile.optimizer_compile = "in_graph"

    def fake_init(trainer, config):
      del config
      trainer.optimizers = SimpleNamespace(optimizers=[])
      trainer.gradient_accumulation_steps = steps

    with (
        mock.patch.object(GraphTrainer, "__init__", fake_init),
        mock.patch.object(
            tpu_graph_trainer, "_split_optimizers_by_mesh", return_value=False
        ),
    ):
      steps = 2
      with self.assertRaisesRegex(ValueError, "gradient accumulation"):
        tpu_graph_trainer.TpuGraphTrainer(config)
      steps = 1
      trainer = tpu_graph_trainer.TpuGraphTrainer(config)
      self.assertTrue(trainer._optimizer_in_graph)

if __name__ == "__main__":
  absltest.main()
