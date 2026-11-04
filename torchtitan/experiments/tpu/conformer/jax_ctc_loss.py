"""JAX CTC Loss implementation for Conformer on TPU via torch_tpu."""

import logging
import jax
import jax.numpy as jnp
import optax
import torch
import torch_tpu  # pylint: disable=unused-import
from torch.tpu import pallas  # pyrefly: ignore[missing-import]


def ctc_loss_forward_jax(logits: jax.Array, labels: jax.Array) -> jax.Array:
  """Computes CTC loss using optax.

  Args:
    logits: logit tensor, shape (B, T, C)
    labels: label tensor, shape (B, N)

  Returns:
    Scalar loss.
  """
  vocab_size = logits.shape[-1]

  # Ignore index for padding (matches PyTorch's default -100)
  ignore_mask = labels == -100

  # Replace -100 with 0 and modulo positive label IDs by vocab_size
  safe_labels = jnp.where(ignore_mask, 0, labels)
  safe_labels = (safe_labels % vocab_size).astype(jnp.int32)
  label_paddings = ignore_mask.astype(jnp.float32)

  # Assuming inputs are not padded at the end (logit_paddings is all zeros)
  logit_paddings = jnp.zeros(logits.shape[:-1], dtype=jnp.float32)

  # Compute Optax CTC loss: returns (B,) loss array
  loss_per_seq = optax.losses.ctc_loss(
      logits, logit_paddings, safe_labels, label_paddings, blank_id=0
  )

  # Return the sum of loss across sequences in the batch
  return jnp.sum(loss_per_seq)


def ctc_loss_backward_jax(
    logits: jax.Array, labels: jax.Array, d_loss: jax.Array
) -> jax.Array:
  """Computes CTC loss gradient w.r.t logits using JAX autodiff.

  Args:
    logits: logit tensor, shape (B, T, C)
    labels: label tensor, shape (B, N)
    d_loss: downstream gradient (scalar)

  Returns:
    Gradient w.r.t logits, shape (B, T, C)
  """

  def surrogate(logits_local):
    return ctc_loss_forward_jax(logits_local, labels) * d_loss

  # We only need gradient w.r.t logits (argnum 0)
  grad_logits = jax.grad(surrogate)(logits)
  return grad_logits


# Register the forward and backward passes as PyTorch custom ops.
# We use the "conformer" namespace.
try:
  ctc_loss_fwd = pallas.jax_op(
      "conformer::jax_ctc_loss_fwd",
      ctc_loss_forward_jax,
  )
  ctc_loss_bwd = pallas.jax_op(
      "conformer::jax_ctc_loss_bwd",
      ctc_loss_backward_jax,
  )

  # Register the backward pass as the autograd for the forward pass.
  def setup_context(ctx, inputs, output):
    del output  # Unused
    logits, labels = inputs
    ctx.save_for_backward(logits, labels)

  def backward(ctx, d_loss):
    logits, labels = ctx.saved_tensors
    # d_loss is the gradient of the loss w.r.t the output of forward
    # (which is a scalar).
    grad_logits = ctc_loss_bwd(logits, labels, d_loss)
    return grad_logits, None  # None for labels (non-differentiable)

  ctc_loss_fwd.register_autograd(backward, setup_context=setup_context)

except Exception as e:  # pylint: disable=broad-except
  # Fallback or logging if registration fails (e.g. during non-tpu import)
  logging.warning("Failed to register JAX CTC loss custom ops: %s", e)
  ctc_loss_fwd = None
  ctc_loss_bwd = None


def jax_ctc_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
  """PyTorch entry point for JAX CTC loss.

  Args:
    logits: logit tensor, shape (B, T, C)
    labels: label tensor, shape (B, N)

  Returns:
    Scalar loss.

  Raises:
    RuntimeError: If the JAX CTC loss custom op was not registered.
  """
  if ctc_loss_fwd is None:
    raise RuntimeError("JAX CTC loss custom op was not registered.")
  # Cast labels to int32 to match JAX TPU default integer type (i32)
  # and avoid StableHLO refinement mismatch (i32 vs i64).
  return ctc_loss_fwd(logits, labels.to(torch.int32))

