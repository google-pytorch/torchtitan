"""Base class and helper functions for single-device accelerator tests."""

import copy
from typing import Any, Callable, Dict, Tuple, Type

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torch import optim
import torchtitan.config
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.testing import test_utils
from torchtitan.experiments.tpu.testing import numerical_validation


class BaseAcceleratorDeviceTest(parameterized.TestCase):
  """Base class for tests that need to run on different accelerator devices.

  This class sets up `self.accelerator_device` based on the `--device` flag,
  supporting 'tpu', 'cuda', and 'cpu'.
  """

  def setUp(self):
    super().setUp()
    torch.compiler.reset()
    self.accelerator_device_type = test_utils.resolve_device_type()
    if self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
      self.accelerator_device = torch.device("tpu")
      logging.info("Using TPU device: %s", self.accelerator_device)
    elif self.accelerator_device_type == device_type.AcceleratorDeviceType.CUDA:
      if not torch.cuda.is_available():
        self.fail("GPU requested but CUDA not available.")
      self.accelerator_device = torch.device("cuda:0")
      logging.info("Using GPU device: %s", self.accelerator_device)
    elif self.accelerator_device_type == device_type.AcceleratorDeviceType.CPU:
      self.accelerator_device = torch.device("cpu")
      logging.info("Using CPU device: %s", self.accelerator_device)
    else:
      raise RuntimeError(
          f"Unexpected flag value: {self.accelerator_device_type}"
      )

    # Set torch and torch.cuda seeds to ensure deterministic behavior.
    # Note that tests would set it by default to 301, unless overridden:
    # https://github.com/abseil/abseil-py/blob/main/absl/testing/absltest.py
    if absltest.FLAGS["test_random_seed"].present:
      seed = absltest.FLAGS.test_random_seed
    else:
      seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe to call even if not using GPU
    print(f"Torch initial seed: {torch.initial_seed()}")

  def test_gpu_available(self):
    if self.accelerator_device_type == "cuda":
      if not torch.cuda.is_available():
        self.fail("GPU requested but CUDA not available.")

  def _setup_module_cpu_device(
      self,
      module_class: Type[nn.Module],
      init_fn: Callable[[nn.Module], None],
      *module_args: Any,
      **module_kwargs: Any
  ) -> Tuple[nn.Module, nn.Module]:
    """Initializes a PyTorch Module on CPU and creates an identical copy on device.

    Args:
      module_class: PyTorch nn.Module class to instantiate (e.g., Attention).
      init_fn: function/lambda that takes the newly created CPU module
        (nn.Module) as its only argument and applies module-specific weight
        initialization logic.
      *module_args: Positional arguments passed directly to the module_class.
      **module_kwargs: Keyword arguments passed directly to the module_class.

    Returns:
        A tuple containing the initialized (module_cpu, module_device).
    """
    # Instantiate component on CPU.
    module_cpu = module_class(*module_args, **module_kwargs).cpu()
    init_fn(module_cpu)
    # Create an identical copy on the accelerator device.
    module_device = (
        copy.deepcopy(module_cpu)
        .to(dtype=test_utils.EXPECTED_DEVICE_DTYPE, device=self.accelerator_device)
    )

    cpu_weight = next(module_cpu.parameters())
    assert cpu_weight.dtype == test_utils.EXPECTED_CPU_DTYPE, (
        f"CPU Dtype ERROR: Expected {test_utils.EXPECTED_CPU_DTYPE}, but found"
        f" {cpu_weight.dtype}"
    )

    device_weight = next(module_device.parameters())
    assert device_weight.dtype == test_utils.EXPECTED_DEVICE_DTYPE, (
        f"DEVICE Dtype ERROR: Expected {test_utils.EXPECTED_DEVICE_DTYPE} after moving to"
        f" device, but found {device_weight.dtype}. Check device"
        " initialization/casting."
    )

    return module_cpu, module_device

  def _create_input_tensor_cpu_device(
      self, size: Tuple[int, ...], requires_grad: bool = False
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates matched CPU and Device input tensors for parity testing.

    Args:
        size: Desired shape of the input tensor (e.g., (batch, seq_len, dim)).
        requires_grad: If True, tracks the gradient for the input tensor.
          Defaults to False.

    Returns:
        A tuple containing the initialized (cpu_tensor, device_tensor) pair.
    """
    x_cpu = torch.randn(size, device="cpu", requires_grad=requires_grad)
    x_device = (
        x_cpu.detach()
        .clone()
        .to(dtype=test_utils.EXPECTED_DEVICE_DTYPE, device=self.accelerator_device)
        .requires_grad_(requires_grad)
    )
    return x_cpu, x_device

  def _run_full_model_training_steps_parity_test(
      self,
      model_cpu: nn.Module,
      n_steps: int = 3,
      loss_atol: float = 1e-4,
      loss_rtol: float = 1e-4,
      grad_atol: float = 1e-5,
      grad_rtol: float = 1e-5,
      param_atol: float = 1e-4,
      param_rtol: float = 1e-4,
      vocab_size: int = 32,
      batch: int = 2,
      seq_len: int = 8,
      loss_fn: torch.nn.Module | None = None
  ):
    """Encapsulates the full model training loop for numerical parity testing.

    Executes training mechanics (Forward, Backward, Optimizer Step) using a
    standard loss function (CrossEntropy by default) to verify parity.

    Args:
      model_cpu: The PyTorch nn.Module instance, already initialized on CPU.
      n_steps: The number of training steps to run. Defaults to 3.
      loss_atol: Absolute tolerance for loss check.
      loss_rtol: Relative tolerance for loss check.
      grad_atol: Absolute tolerance for gradient parity check.
      grad_rtol: Relative tolerance for gradient parity check.
      param_atol: Absolute tolerance for parameter parity check.
      param_rtol: Relative tolerance for parameter parity check.
      vocab_size: Model's vocab size for token generation. Defaults to 32.
      batch: Batch size for token generation. Defaults to 2.
      seq_len: Sequence length for token generation. Defaults to 8.
      loss_fn: Loss function to use for training. Defaults to Cross Entropy loss if not provided.
    """
    # Set up loss function if not provided.
    if loss_fn is None:
      loss_fn = torch.nn.CrossEntropyLoss()

    # DEVICE setup
    model_device = copy.deepcopy(model_cpu).to(self.accelerator_device)

    # Ensure parameters require gradients.
    for p in model_cpu.parameters():
      p.requires_grad_(True)
    for p in model_device.parameters():
      p.requires_grad_(True)

    # Optimizer setup
    optimizer_cpu = optim.Adam(model_cpu.parameters(), lr=0.01)
    optimizer_device = optim.Adam(model_device.parameters(), lr=0.01)

    from torchtitan.experiments.tpu.components.unflatten_shim import set_unflatten_seq_len
    from torchtitan.models.common.decoder import Decoder

    is_decoder = isinstance(model_cpu, Decoder)

    # Run 3 training steps
    for step in range(n_steps):
      # Generate new tokens for each step
      token_shape = (batch * seq_len,) if is_decoder else (batch, seq_len)
      tokens_cpu = torch.randint(0, vocab_size, token_shape, device="cpu")
      tokens_device = tokens_cpu.to(self.accelerator_device)

      with self.subTest(name=f"Step_{step}_ForwardPass"):
        # Zero gradients
        optimizer_cpu.zero_grad()
        optimizer_device.zero_grad()
        # Forward pass
        positions_cpu = torch.arange(
            seq_len, dtype=torch.long, device="cpu"
        ).repeat(batch)
        positions_device = positions_cpu.to(self.accelerator_device)
        with set_unflatten_seq_len(seq_len):
          if is_decoder:
            out_cpu = model_cpu(tokens_cpu, positions_cpu)
            out_device = model_device(tokens_device, positions_device)
          else:
            out_cpu = model_cpu(tokens_cpu)
            out_device = model_device(tokens_device)
          loss_cpu = loss_fn(out_cpu.view(-1, vocab_size), tokens_cpu.view(-1))
          loss_device = loss_fn(
              out_device.view(-1, vocab_size), tokens_device.view(-1)
          )

          test_utils.check_equivalence(
              tensor_test=loss_device.cpu(),
              tensor_reference=loss_cpu,
              check_name="Loss Function parity",
              step=step,
              atol=loss_atol,
              rtol=loss_rtol,
          )

          with self.subTest(name=f"Step_{step}_BackwardPass"):
            # Backward pass
            loss_cpu.backward()
            loss_device.backward()
          # Gradient check
          for i, (p_cpu, p_device) in enumerate(
              zip(model_cpu.parameters(), model_device.parameters())
          ):
            # Handle potential None gradients
            if p_cpu.grad is not None and p_device.grad is not None:
              test_utils.check_equivalence(
                  tensor_test=p_device.grad.cpu(),
                  tensor_reference=p_cpu.grad,
                  check_name=f"Gradient parity param {i}",
                  step=step,
                  atol=grad_atol,
                  rtol=grad_rtol,
                  test_label="Device",
                  ref_label="CPU",
              )
            elif not (p_cpu.grad is None and p_device.grad is None):
              # Fail if one is None and the other is not
              self.fail(
                  f"Gradient mismatch: CPU grad is {p_cpu.grad is not None},"
                  f" Device grad is {p_device.grad is not None} at step {step},"
                  f" param {i}."
              )
        # Optimizer step
        optimizer_cpu.step()
        optimizer_device.step()

    # Final Check: params are in sync after all training steps
    with self.subTest(name="Final_Parameter_Check"):
      for i, (p_cpu, p_device) in enumerate(
          zip(model_cpu.parameters(), model_device.parameters())
      ):
        test_utils.check_equivalence(
            tensor_test=p_device.cpu(),
            tensor_reference=p_cpu,
            check_name=f"Final parameter mismatch for param {i}",
            step=n_steps,
            atol=param_atol,
            rtol=param_rtol,
            test_label="Device",
            ref_label="CPU",
        )

  def _run_trainer_device_parity_test(
      self,
      job_config,
      loss_atol: float = 1e-3,
      loss_rtol: float = 1e-3,
      grad_atol: float = 1e-3,
      grad_rtol: float = 1e-3,
      param_atol: float = 5e-3,
      param_rtol: float = 5e-3,
  ):

    # Setup recorder callback
    recorder = numerical_validation.StateRecorderCallback(
        capture_fn=numerical_validation.capture_local_state
    )

    trainer_device = torchtitan.experiments.tpu.train_minimal.TrainerMinimal(
        device=self.accelerator_device,
        rank=0,
        world_size=1,
        job_config=job_config,
        step_callback=recorder.on_step,
    )

    # Save initial weights to ensure CPU model has same init weights as device.
    initial_state_dict = {
        k: v.cpu().clone()
        for k, v in trainer_device.model.state_dict().items()
    }

    # Phase 1: Run Device Training and record the state at each step
    trainer_device.train()
    # Get the recorded history
    recorded_history = recorder.get_history()

    # Setup validator callback with the recorded history.
    validator = numerical_validation.StateValidatorCallback(
        recorded_history=recorded_history,
        capture_fn=numerical_validation.capture_local_state,
        loss_atol=loss_atol, loss_rtol=loss_rtol,
        grad_atol=grad_atol, grad_rtol=grad_rtol,
        param_atol=param_atol, param_rtol=param_rtol,
    )

    with validator:
      trainer_cpu = torchtitan.experiments.tpu.train_minimal.TrainerMinimal(
          torch.device("cpu"),
          rank=0,
          world_size=1,
          job_config=job_config,
          step_callback=validator.on_step,
      )

      # Load initial weights
      trainer_cpu.model.load_state_dict(initial_state_dict)

      # Phase 2: Train the CPU model and validate against the recorded history.
      trainer_cpu.train()
