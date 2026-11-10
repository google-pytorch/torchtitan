"""Shared utilities for accelerator device tests."""

import os
from typing import Any, Callable, Dict

import torch
from torch import nn
from torchtitan.experiments.tpu.testing import test_utils


def clean_parameter_name(name: str) -> str:
  """Removes wrapper prefixes (like activation checkpointing) from parameter names.

  This ensures that weights and gradients can be compared and loaded between
  models that have checkpointing enabled and those that do not.

  Args:
      name: The original parameter name.

  Returns:
      The cleaned parameter name.
  """
  return name.replace("_checkpoint_wrapped_module.", "")


CaptureFn = Callable[[nn.Module, torch.Tensor], Dict[str, Any]]


def capture_local_state(model: nn.Module, loss: torch.Tensor) -> Dict[str, Any]:
  """Captures local model state (CPU/GPU/TPU single device).

  Args:
      model: The model to capture the state of.
      loss: The current loss value.

  Returns:
      A dictionary containing the loss, gradients, and parameters of the model
      on the local device.
  """
  state = {
      "loss": loss.detach().cpu(),
      "grads": {},
      "params": {},
  }
  for name, param in model.named_parameters():
    state["params"][name] = param.detach().cpu().clone()
    if param.grad is not None:
      state["grads"][name] = param.grad.detach().cpu().clone()

  return state


class StateRecorderCallback:
  """Generic callback to record model state history during training."""

  def __init__(self, capture_fn: CaptureFn = capture_local_state):
    self.history: Dict[int, Any] = {}
    self.capture_fn = capture_fn

  def on_step(
      self,
      step: int,
      model: nn.Module,
      loss: torch.Tensor,
      inputs: torch.Tensor = None,
      labels: torch.Tensor = None,
      positions: torch.Tensor = None,
  ):
    """Records the state at the current step."""
    self.history[step] = self.capture_fn(model, loss)
    if inputs is not None:
      self.history[step]["inputs"] = inputs.detach().cpu().clone()
    if labels is not None:
      self.history[step]["labels"] = labels.detach().cpu().clone()
    # Packed streams carry document boundaries only in positions; a replay
    # without them reads the whole stream as one sequence.
    if positions is not None:
      self.history[step]["positions"] = positions.detach().cpu().clone()

  def get_history(self) -> Dict[int, Any]:
    """Returns the recorded state history."""
    return self.history


class StateValidatorCallback:
  """Generic callback to validate current model state against device history."""

  # The callback must be used within a 'with' context to enable error
  # collection and handling. It records the validation errors and raises them
  # as an exception upon exiting the context.
  #
  # Example:
  #   with StateValidatorCallback(...) as validator:
  #     # Run training and validation logic
  #     trainer.train(v.on_step(...))
  def __init__(
      self,
      recorded_history: Dict[int, Any],
      capture_fn: CaptureFn = capture_local_state,
      init_state: Dict[str, Any] = None,
      loss_atol: float = 1e-3,
      loss_rtol: float = 1e-3,
      grad_atol: float = 1e-3,
      grad_rtol: float = 1e-3,
      param_atol: float = 1e-3,
      param_rtol: float = 1e-3,
      current_label: str = "CPU Reference",
      recorded_label: str = "Device",
  ):
    self.recorded_history = recorded_history
    self.capture_fn = capture_fn
    self.init_state = init_state
    self.loss_atol = loss_atol
    self.loss_rtol = loss_rtol
    self.grad_atol = grad_atol
    self.grad_rtol = grad_rtol
    self.param_atol = param_atol
    self.param_rtol = param_rtol

    # Labels for error messages
    self.current_label = current_label
    self.recorded_label = recorded_label
    self.errors = []
    self.in_context = False

  def __enter__(self):
    """Enters the context."""
    self.in_context = True
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Exits the context and raises any collected errors."""
    self.in_context = False
    if self.errors:
      raise AssertionError("\n".join(self.errors))

  def on_step(
      self,
      step: int,
      model: nn.Module,
      loss: torch.Tensor,
      inputs: torch.Tensor = None,
      labels: torch.Tensor = None,
      positions: torch.Tensor = None,
  ):
    """Validates the current state against the reference history."""
    del positions  # Only the recorder needs the batch.
    if not self.in_context:
      raise RuntimeError("on_step can only be called within a 'with' context.")
    if step not in self.recorded_history:
      raise ValueError(f"Step {step} missing from reference history.")

    current_state = self.capture_fn(model, loss)
    recorded_state = self.recorded_history[step]

    try:
      test_utils.check_equivalence(
          tensor_test=current_state["loss"],
          tensor_reference=recorded_state["loss"],
          check_name="Loss",
          step=step,
          atol=self.loss_atol,
          rtol=self.loss_rtol,
          test_label=self.current_label,
          ref_label=self.recorded_label,
      )
    except AssertionError as e:
      self.errors.append(str(e))

    for name, rec_grad in recorded_state["grads"].items():
      # Strip activation checkpointing wrappers to align parameter names
      cleaned_name = clean_parameter_name(name)
      if cleaned_name in current_state["grads"]:
        try:
          test_utils.check_equivalence(
              tensor_test=current_state["grads"][cleaned_name],
              tensor_reference=rec_grad,
              check_name=f"Gradient ({name})",
              step=step,
              atol=self.grad_atol,
              rtol=self.grad_rtol,
              test_label=self.current_label,
              ref_label=self.recorded_label,
          )
        except AssertionError as e:
          self.errors.append(str(e))

    for name, rec_param in recorded_state["params"].items():
      # Strip activation checkpointing wrappers to align parameter names
      cleaned_name = clean_parameter_name(name)
      if cleaned_name in current_state["params"]:
        try:
          test_utils.check_equivalence(
              tensor_test=current_state["params"][cleaned_name],
              tensor_reference=rec_param,
              check_name=f"Parameter ({name})",
              step=step,
              atol=self.param_atol,
              rtol=self.param_rtol,
              test_label=self.current_label,
              ref_label=self.recorded_label,
          )
        except AssertionError as e:
          self.errors.append(str(e))
