"""Unit tests for TpuGraphTrainer (tracing, loss override, and config wiring)."""

from types import SimpleNamespace
import typing
from unittest import mock

from absl.testing import absltest
import torch
from torch import nn
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.graph_trainer import make_fx_tracer
from torchtitan.experiments.graph_trainer.common_utils import (
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    remove_parameter_gradient_markers_pass,
)
from torchtitan.experiments.tpu.graph_trainer import mfu_benchmark
from torchtitan.experiments.tpu.graph_trainer.qwen3 import (
    model_registry as tpu_graph_trainer_model_registry,
)
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.experiments.tpu.graph_trainer import tpu_graph_trainer
from torchtitan.experiments.tpu.loss import build_cross_entropy_loss
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig


class DummyModule(nn.Module):
  """Simple synthetic module for fast FX tracing checks."""

  def __init__(self, dim: int = 32, vocab_size: int = 128) -> None:
    super().__init__()
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.linear = nn.Linear(dim, dim)
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.tok_embeddings(x)
    return self.lm_head(torch.relu(self.linear(h)))


def _make_dummy_fwd_bwd_step(loss_fn: nn.Module, model: nn.Module):
  """Returns a joint forward+loss+backward step function for tracing verification."""

  def fwd_bwd_fn(x, target):
    pred = model(x)
    loss = loss_fn(pred.view(-1, 128), target.view(-1))
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params)
    return [loss] + list(grads)

  return fwd_bwd_fn


class MfuBenchmarkTest(absltest.TestCase):

  def test_mfu_benchmark_bracket(self) -> None:
    metrics = SimpleNamespace(
        parallel_dims=SimpleNamespace(non_data_parallel_size=1),
        num_flops_per_token=2e9,
        gpu_peak_flops=100e12,
        should_log=lambda step: step in (5, 7),
    )
    messages: list[str] = []
    with (
        mock.patch.object(
            mfu_benchmark.time,
            "perf_counter",
            side_effect=(10.0, 17.0),
        ),
        mock.patch.object(
            mfu_benchmark.logger,
            "info",
            side_effect=lambda message, *args: messages.append(message % args),
        ),
    ):
      benchmark = mfu_benchmark.MfuBenchmark(metrics, 5, 7)
      benchmark.record(5, 100)
      benchmark.record(6, 200)
      benchmark.record(7, 300)

    self.assertEqual(
        messages,
        [
            "[mfu_benchmark] start_step=5 final_step=7 measured_steps=2 "
            "wall=7.000000s mean_step_time=3.500000s tokens_per_rank=200 "
            "tflops=0.057143 mfu=0.057143%"
        ],
    )


class TpuGraphTrainerUnitTest(BaseAcceleratorDeviceTest):
  """Fast single-process unit tests for TpuGraphTrainer components cleanly without mocks."""

  def setUp(self) -> None:
    super().setUp()
    self.config_manager = ConfigManager()
    # Upstream replaced seq_len/local_batch_size with a token budget:
    # num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len = 2 * 16.
    self.config = typing.cast(
        TPUTrainerConfig,
        self.config_manager.parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            "--training.num_tokens_per_microbatch_per_dp_rank=32",
            "--training.max_context_length=16",
            "--compile.mode=fullgraph",
        ]),
    )

  def test_from_tpu_trainer_config_target_cls(self) -> None:
    """The classmethod binds the final TpuGraphTrainer config type."""
    gt_cfg = tpu_graph_trainer.TpuGraphTrainer.Config.from_tpu_trainer_config(
        self.config,
        tpu_graph_trainer_model_registry,
    )
    self.assertIsInstance(
        gt_cfg,
        tpu_graph_trainer.TpuGraphTrainer.Config,
        "from_tpu_trainer_config must produce TpuGraphTrainer.Config.",
    )
    assert self.config.model_spec is not None
    assert gt_cfg.model_spec is not None
    self.assertEqual(gt_cfg.model_spec.name, self.config.model_spec.name)

  def test_pallas_ce_loss_override(self) -> None:
    """Verifies build_cross_entropy_loss returns PallasCrossEntropyLoss when use_loss_kernel=True."""
    # Upstream replaced seq_len/local_batch_size with a token budget:
    # num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len = 2 * 16.
    config = typing.cast(
        TPUTrainerConfig,
        self.config_manager.parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            "--training.num_tokens_per_microbatch_per_dp_rank=32",
            "--training.max_context_length=16",
            "--compile.mode=fullgraph",
            "--loss-kernel.use-loss-kernel",
        ]),
    )

    loss_fn = build_cross_entropy_loss(config)
    self.assertIsInstance(
        loss_fn,
        PallasCrossEntropyLoss,
        "build_cross_entropy_loss must produce PallasCrossEntropyLoss when use_loss_kernel is set.",
    )

  def test_make_fx_tracing_on_synthetic_model(self) -> None:
    """Verifies make_fx joint tracing and execution on a synthetic fwd+loss+bwd step."""
    device = self.accelerator_device
    model = DummyModule(dim=32, vocab_size=128).to(device)
    loss_fn = nn.CrossEntropyLoss()
    fwd_bwd_fn = _make_dummy_fwd_bwd_step(loss_fn, model)

    inputs = torch.randint(0, 128, (2, 16), device=device)
    labels = torch.randint(0, 128, (2, 16), device=device)

    traced_step = make_fx_tracer.minimal_fx_tracer(fwd_bwd_fn, module=model)(
        inputs, labels
    )
    self.assertIsInstance(
        traced_step.gm,
        torch.fx.GraphModule,
        "traced_step.gm must be a valid FX GraphModule.",
    )
    num_params = len(list(model.parameters()))
    self.assertEqual(
        traced_step.num_flat_outputs,
        1 + num_params,
        "Traced graph outputs must match 1 loss + parameter gradients.",
    )

    bw_nodes = [
        node for node in traced_step.gm.graph.nodes
        if "backward" in str(node.target) or node.meta.get("autograd_backward", False)
    ]
    self.assertNotEmpty(bw_nodes, "Traced graph must contain backward pass nodes.")

    outputs = make_fx_tracer.run_traced(traced_step, module=model)(
        inputs, labels
    )
    self.assertLen(outputs, 1 + num_params)
    loss = outputs[0]
    grads = outputs[1:]
    self.assertEqual(loss.ndim, 0, "Loss output must be a scalar tensor.")
    for p, g in zip(model.parameters(), grads):
      self.assertEqual(
          g.shape, p.shape, "Gradient shape must match parameter shape."
      )

  def test_fwd_bwd_adamw_grad_clipping_tracing(self) -> None:
    """Verifies that fwd_bwd_adamw traces and executes cleanly with max_norm > 0."""
    device = self.accelerator_device
    model = DummyModule(dim=32, vocab_size=128).to(device)

    def loss_fn(pred, labels, global_valid_tokens):
      return nn.CrossEntropyLoss()(
          pred.view(-1, 128), labels.view(-1)
      ) / global_valid_tokens

    param_fqns = []
    for name, p in model.named_parameters():
      if not p.requires_grad:
        continue
      param_fqns.append(name)
      parent_path, _, leaf = name.rpartition(".")
      parent = model.get_submodule(parent_path) if parent_path else model
      parent.register_buffer(
          f"_optstate_{leaf}_exp_avg", torch.zeros_like(p), persistent=True
      )
      parent.register_buffer(
          f"_optstate_{leaf}_exp_avg_sq", torch.zeros_like(p), persistent=True
      )

    model.register_buffer(
        "_optstate_step",
        torch.zeros((), dtype=torch.float32, device=device),
        persistent=True,
    )
    model.register_buffer(
        "_optstate_lr",
        torch.tensor(1e-3, dtype=torch.float32, device=device),
        persistent=False,
    )

    fwd_bwd_adamw = tpu_graph_trainer._make_fwd_bwd_adamw_fn(
        model,
        loss_fn=loss_fn,
        param_fqns=param_fqns,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        max_norm=1.0,
        capturable=False,
    )

    inputs = torch.randint(0, 128, (2, 16), device=device)
    labels = torch.randint(0, 128, (2, 16), device=device)
    valid_tokens = torch.tensor(32.0, device=device)

    traced_step = make_fx_tracer.minimal_fx_tracer(fwd_bwd_adamw, module=model)(
        inputs, labels, valid_tokens, {}
    )
    self.assertIsInstance(
        traced_step.gm,
        torch.fx.GraphModule,
        "traced_step.gm must be a valid FX GraphModule when max_norm > 0.",
    )
    marker_nodes = [
        node
        for node in traced_step.gm.graph.nodes
        if node.target is torch.ops.aten.alias.default
        and PARAMETER_GRADIENT_FQNS_META in node.meta.get("custom", {})
    ]
    self.assertSequenceEqual(
        [
            node.meta["custom"][PARAMETER_GRADIENT_FQNS_META][0]
            for node in marker_nodes
        ],
        param_fqns,
    )
    graph_outputs = set(
        traced_step.gm.graph.find_nodes(op="output")[0].all_input_nodes
    )
    self.assertTrue(all(node not in graph_outputs for node in marker_nodes))

    remove_parameter_gradient_markers_pass(
        traced_step.gm, traced_step.example_inputs
    )
    tagged_fqns = {
        fqn
        for node in traced_step.gm.graph.nodes
        for fqn in node.meta.get("custom", {}).get(
            PARAMETER_GRADIENT_FQNS_META, ()
        )
    }
    self.assertSetEqual(tagged_fqns, set(param_fqns))

    outputs = make_fx_tracer.run_traced(traced_step, module=model)(
        inputs, labels, valid_tokens, {}
    )
    self.assertNotEmpty(outputs)
    loss = outputs[0]
    self.assertEqual(loss.ndim, 0, "Loss output must be a scalar tensor.")



if __name__ == "__main__":
  absltest.main()
