"""Unit tests for separate-graph compiled AdamW optimizer on TPU / CPU."""

from unittest import mock

from absl.testing import absltest
import torch
from torch import nn
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.experiments.tpu.compiled_optimizers import separate_graph
from torchtitan.experiments.tpu.compiled_optimizers.separate_graph import apply_compiled_optimizer


class DummyOptimizersContainer:

  def __init__(self, optimizer: torch.optim.Optimizer):
    self.optimizers = [optimizer]

  def step(self):
    for opt in self.optimizers:
      opt.step()


class DummyTrainer:

  def __init__(self, optimizer: torch.optim.Optimizer):
    self.optimizers = DummyOptimizersContainer(optimizer)


class SeparateGraphCompilerConstructionTest(absltest.TestCase):

  def test_static_compiler_is_instantiated_before_graph_arguments(self):
    graph_module = torch.fx.symbolic_trace(lambda value: (value + 1,))
    example_inputs = (torch.ones(1),)
    compiler = mock.Mock()
    compiler_class = mock.Mock(return_value=compiler)

    with mock.patch.object(separate_graph, "StaticCompiler", compiler_class):
      separate_graph._tpu_backend_compile_pass(graph_module, example_inputs)

    compiler_class.assert_called_once_with()
    compiler.assert_called_once_with(graph_module, list(example_inputs))


class SeparateGraphOptimizerTest(BaseAcceleratorDeviceTest):

  def test_compiled_optimizer_step(self):
    device = self.accelerator_device
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    trainer = DummyTrainer(opt)

    # Apply compiled optimizer patch
    apply_compiled_optimizer(trainer)

    # Initial weights snapshot
    initial_weights = [p.clone() for p in model.parameters()]

    # Run 3 training steps
    for _ in range(3):
      x = torch.randn(4, 16, device=device)
      target = torch.randn(4, 8, device=device)
      loss = ((model(x) - target) ** 2).sum()
      loss.backward()
      trainer.optimizers.step()
      opt.zero_grad()

    # Verify parameters were updated
    for p_init, p_curr in zip(initial_weights, model.parameters(), strict=True):
      self.assertFalse(
          torch.equal(p_init, p_curr),
          "Model weights should be updated after compiled optimizer step.",
      )

    # Verify optimizer states exist and were updated
    for p in model.parameters():
      state = opt.state[p]
      self.assertIn("exp_avg", state)
      self.assertIn("exp_avg_sq", state)
      self.assertIn("step", state)
      self.assertGreater(
          state["step"].item(),
          0,
          "Optimizer step counter should be greater than 0.",
      )

  def test_compiled_optimizer_numerical_correctness(self):
    """Verify that compiled AdamW step yields identical results to eager AdamW."""
    device = self.accelerator_device

    # Create reference eager baseline and test model with compiled optimizer
    # model_ref & opt_eager for the reference eager baseline.
    # model_test & opt_compiled for the test model running with the compiled optimizer.
    torch.manual_seed(42)
    model_ref = nn.Linear(16, 16).to(device)
    opt_eager = torch.optim.AdamW(
        model_ref.parameters(), lr=1e-3, weight_decay=1e-2
    )

    torch.manual_seed(42)
    model_test = nn.Linear(16, 16).to(device)
    opt_compiled = torch.optim.AdamW(
        model_test.parameters(), lr=1e-3, weight_decay=1e-2
    )
    trainer_compiled = DummyTrainer(opt_compiled)
    apply_compiled_optimizer(trainer_compiled)

    # Run for 3 steps and verify identical outputs and optimizer state
    for step in range(3):
      torch.manual_seed(100 + step)
      x = torch.randn(4, 16, device=device)

      # Eager step
      loss_eager = (model_ref(x) ** 2).sum()
      loss_eager.backward()
      opt_eager.step()
      opt_eager.zero_grad()

      # Compiled step
      loss_compiled = (model_test(x) ** 2).sum()
      loss_compiled.backward()
      trainer_compiled.optimizers.step()
      opt_compiled.zero_grad()

      # Compare parameters
      for p_eager, p_compiled in zip(
          model_ref.parameters(),
          model_test.parameters(),
          strict=True,
      ):
        torch.testing.assert_close(p_compiled, p_eager, rtol=1e-4, atol=1e-4)

      # Compare optimizer states (exp_avg, exp_avg_sq, step)
      for p_eager, p_compiled in zip(
          model_ref.parameters(),
          model_test.parameters(),
          strict=True,
      ):
        st_eager = opt_eager.state[p_eager]
        st_compiled = opt_compiled.state[p_compiled]
        torch.testing.assert_close(
            st_compiled["exp_avg"], st_eager["exp_avg"], rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            st_compiled["exp_avg_sq"],
            st_eager["exp_avg_sq"],
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            st_compiled["step"],
            st_eager["step"],
            rtol=1e-4,
            atol=1e-4,
            check_device=False,
        )


if __name__ == "__main__":
  absltest.main()
