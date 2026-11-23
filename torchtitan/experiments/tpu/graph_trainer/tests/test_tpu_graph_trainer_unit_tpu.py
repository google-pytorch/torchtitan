"""Unit tests for TpuGraphTrainer (tracing, loss override, and config wiring)."""

import contextlib
import copy
from types import SimpleNamespace
import typing
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.nn import functional as F
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.graph_trainer import make_fx_tracer
from torchtitan.experiments.graph_trainer.common_utils import (
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    remove_parameter_gradient_markers_pass,
)
from torchtitan.experiments.graph_trainer.trainer import FwdBwdOutputs
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.experiments.tpu.graph_trainer import mfu_benchmark
from torchtitan.experiments.tpu.graph_trainer.qwen3 import (
    model_registry as tpu_graph_trainer_model_registry,
)
from torchtitan.experiments.tpu.testing.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.experiments.tpu.graph_trainer import tpu_graph_trainer
from torchtitan.experiments.tpu.components.loss import build_cross_entropy_loss
from torchtitan.experiments.tpu.components.loss import PallasCrossEntropyLoss
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


class _SharedGradModel(nn.Module):
  """The traced graph returns one tensor as the grad of both a and b."""

  def __init__(self) -> None:
    super().__init__()
    self.a = nn.Parameter(torch.randn(128, 128))
    self.b = nn.Parameter(torch.randn(128, 128))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.embedding(x, self.a + self.b)


class _ShardedModel(nn.Module):
  """DTensor params gathered in forward, as SimpleFSDP does, plus a plain one.

  a and b get one grad tensor back, which run_traced wraps once per param.
  """

  def __init__(self, mesh) -> None:
    super().__init__()
    self.a = nn.Parameter(
        distribute_tensor(torch.randn(128, 32), mesh, [Shard(0)])
    )
    self.b = nn.Parameter(
        distribute_tensor(torch.randn(128, 32), mesh, [Shard(0)])
    )
    self.scale = nn.Parameter(torch.ones(32))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    embed = (self.a + self.b).redistribute(placements=[Replicate()]).to_local()
    return (F.embedding(x, embed) * self.scale) @ embed.T


def _loss(model: nn.Module, inputs, labels, global_valid_tokens):
  pred = model(inputs)
  # log_softmax + nll_loss, not F.cross_entropy: eager can pick a fused
  # cross-entropy kernel that the traced graph decomposes, which rounds
  # differently. This way both paths run the same ops.
  logp = F.log_softmax(pred.view(-1, 128), dim=-1)
  loss = F.nll_loss(logp, labels.view(-1), reduction="sum")
  return loss / global_valid_tokens


def _traced_trainer(model: nn.Module, batch_size: int, device):
  """A bare TpuGraphTrainer holding one traced fwd+bwd step on (batch_size, 16)."""

  def fwd_bwd(inputs, labels, global_valid_tokens, extra_kwargs):
    loss = _loss(model, inputs, labels, global_valid_tokens)
    params = [p for _, p in model.named_parameters(remove_duplicate=False)]
    grads = list(torch.autograd.grad(loss, params))
    # Same output structure as make_fwd_bwd_step, which the trainer reads.
    return FwdBwdOutputs(loss, grads, [], {})

  inputs = torch.randint(0, 128, (batch_size, 16), device=device)
  labels = torch.randint(0, 128, (batch_size, 16), device=device)
  trainer = object.__new__(tpu_graph_trainer.TpuGraphTrainer)
  trainer._traced_step = make_fx_tracer.minimal_fx_tracer(
      fwd_bwd, module=model
  )(inputs, labels, torch.tensor(1.0, device=device), {})
  trainer.train_context = contextlib.nullcontext
  trainer._collect_moe_routing_metrics = False
  trainer._ep_drop_stats_enabled = False
  return trainer


def _check_grad_accum_step(test, trainer, model, num_microbatches, device):
  """Runs one batch-4 step as num_microbatches calls; grads must equal eager's."""
  inputs = torch.randint(0, 128, (4, 16), device=device)
  labels = torch.randint(0, 128, (4, 16), device=device)
  tokens = torch.tensor(float(labels.numel()), device=device)
  params = tuple(p for _, p in model.named_parameters(remove_duplicate=False))
  for x, y in zip(
      inputs.chunk(num_microbatches), labels.chunk(num_microbatches)
  ):
    trainer._make_fx_forward_backward_step(model, x, y, tokens, params, {})

  ref = copy.deepcopy(model)
  ref.zero_grad(set_to_none=True)
  _loss(ref, inputs, labels, tokens).backward()
  for (name, got), want in zip(
      model.named_parameters(), ref.parameters(), strict=True
  ):
    test.assertIs(type(got.grad), type(want.grad), name)
    if isinstance(want.grad, DTensor):
      got, want = got.grad.full_tensor(), want.grad.full_tensor()
    else:
      got, want = got.grad, want.grad
    # A callable msg keeps assert_close's mismatch report; a string replaces it.
    torch.testing.assert_close(
        got, want, rtol=1e-5, atol=1e-6, msg=lambda m: f"{name}: {m}"
    )


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

  def test_make_fwd_bwd_adamw_fn_returns_loss_metrics(self) -> None:
    device = self.accelerator_device
    model = DummyModule().to(device)

    def loss_fn(pred, labels, *, global_valid_tokens):
      log_prob = pred.mean(dim=-1)
      loss = torch.nn.functional.cross_entropy(
          pred.view(-1, 128), labels.view(-1)
      ) / global_valid_tokens
      return loss, {"log_prob": log_prob.detach()}

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
    outputs = make_fx_tracer.run_traced(traced_step, module=model)(
        inputs, labels, valid_tokens, {}
    )
    self.assertIsInstance(outputs, tpu_graph_trainer.FwdBwdAdamWOutputs)
    self.assertEqual(outputs.loss.ndim, 0, "Loss output must be a scalar tensor.")
    self.assertLen(outputs.params, len(param_fqns))
    self.assertEqual(set(outputs.loss_metrics), {"log_prob"})
    self.assertEqual(
        outputs.loss_metrics["log_prob"].shape, model(inputs).mean(dim=-1).shape
    )


class GradAccumulationTest(BaseAcceleratorDeviceTest):
  """Grads add up over a step's fwd+bwd calls to the full-batch grad."""

  def setUp(self):
    super().setUp()
    # Compare at full f32 matmul precision, as the gemma4 TPU tests do; a
    # reduced default would put rounding above the 1e-5 tolerance.
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    self.addCleanup(torch.set_float32_matmul_precision, previous)

  @parameterized.named_parameters(
      ("ga1", "tied", 1, True),
      ("ga2", "tied", 2, True),
      ("ga2_zeroed_not_none", "tied", 2, False),
      ("ga2_shared_grad_tensor", "shared_grad", 2, True),
  )
  def test_grads_match_full_batch(
      self, model_kind: str, num_microbatches: int, set_to_none: bool
  ) -> None:
    """Two steps, each batch 4 split into microbatches, match eager batch 4."""
    device = self.accelerator_device
    if model_kind == "tied":
      model = DummyModule(dim=32, vocab_size=128).to(device)
      # One param under two names: the graph returns its full grad twice.
      model.lm_head.weight = model.tok_embeddings.weight
    else:
      model = _SharedGradModel().to(device)
    trainer = _traced_trainer(model, 4 // num_microbatches, device)
    if model_kind == "shared_grad":
      grad_a, grad_b = trainer._traced_step.gm.graph.output_node().args[0][1:3]
      self.assertIs(grad_a, grad_b, "the traced graph no longer aliases grads")
    for _ in range(2):
      model.zero_grad(set_to_none=set_to_none)
      _check_grad_accum_step(self, trainer, model, num_microbatches, device)

  def test_dtensor_grads_match_full_batch(self) -> None:
    """GA=2 with DTensor and plain params, on CPU (needs a process group)."""
    dist.init_process_group(
        "gloo", store=dist.HashStore(), rank=0, world_size=1
    )
    self.addCleanup(dist.destroy_process_group)
    model = _ShardedModel(init_device_mesh("cpu", (1,)))
    trainer = _traced_trainer(model, 2, "cpu")
    _check_grad_accum_step(self, trainer, model, 2, "cpu")

  @parameterized.parameters(1, 2)
  def test_in_graph_optimizer_rejects_grad_accumulation(self, steps) -> None:
    """The in-graph step runs AdamW per call, so GA > 1 must raise."""
    config = ConfigManager().parse_args([
        "--module=torchtitan.experiments.tpu.qwen3",
        "--config=graph_trainer_qwen3_debugmodel",
    ])
    config.compile.optimizer_compile = "in_graph"

    def fake_init(trainer, config):
      del config
      trainer.model_parts = []
      trainer.optimizers = SimpleNamespace(optimizers=[])
      trainer.gradient_accumulation_steps = steps

    with mock.patch.object(GraphTrainer, "__init__", fake_init):
      if steps > 1:
        with self.assertRaisesRegex(ValueError, "gradient accumulation"):
          tpu_graph_trainer.TpuGraphTrainer(config)
      else:
        tpu_graph_trainer.TpuGraphTrainer(config)


if __name__ == "__main__":
  absltest.main()
