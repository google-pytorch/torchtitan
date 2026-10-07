"""Tokamax-backed grouped/ragged matmul (segment_matmul) kernel for TPU.

Unlike segment_matmul implementations that call `group_sizes.tolist()` (a D2H
sync, which deadlocks on TPU), this kernel takes `group_sizes` as a runtime
tensor.
"""

from typing import Tuple, cast

import jax
import jax.numpy as jnp
import torch
import torch_tpu  # pylint: disable=unused-import
from torch.tpu import pallas  # pyrefly: ignore[missing-import]
import tokamax


def _build_ragged_dot_kernels():
  """Build forward + backward Pallas megablox kernels with TENSOR group_sizes.

  Returns ``(fwd_op, bwd_op)`` where each is a ``CustomOpDef`` registered via
  ``pallas.jax_op``. The ops accept ``group_sizes`` as a runtime tensor input
  rather than a closure constant, so a single compiled kernel handles every
  routing pattern (no recompile per routing change).

  Construction is wrapped in ``jax.default_device(jax.local_devices()[0])`` —
  see module docstring for why.
  """
  with jax.default_device(jax.local_devices()[0]):

    @pallas.jax_op("tt_segment_matmul::ragged_dot_fwd")
    def fwd_op(
        lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array
    ) -> jax.Array:
      return tokamax.ragged_dot(
          lhs, rhs, group_sizes=group_sizes, implementation="mosaic"
      )

    @pallas.jax_op("tt_segment_matmul::ragged_dot_bwd")
    def bwd_op(
        dout: jax.Array,
        lhs: jax.Array,
        rhs: jax.Array,
        group_sizes: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
      # grad_lhs = dout @ rhs^T  per group → reuse ragged_dot with rhs^T
      grad_lhs = tokamax.ragged_dot(
          dout,
          jnp.swapaxes(rhs, 1, 2),
          group_sizes=group_sizes,
          implementation="mosaic",
      )
      # grad_rhs = lhs^T @ dout per group → use the megablox tgmm primitive,
      # which tokamax doesn't expose directly. Fall back to jax megablox here;
      # this is still tensor-group_sizes (no D2H), so the multi-host invariant
      # holds.
      from jax.experimental.pallas.ops.tpu.megablox import ops as jax_gmm_ops  # pylint: disable=g-import-not-at-top
      grad_rhs = jax_gmm_ops.backend.tgmm(
          jnp.swapaxes(lhs, 0, 1),
          dout,
          group_sizes=group_sizes,
          preferred_element_type=jnp.float32,
          tiling=(128, 128, 128),
          interpret=False,
      )
      return grad_lhs, grad_rhs

  return fwd_op, bwd_op


_FWD_OP, _BWD_OP = None, None


def _get_ops():
  global _FWD_OP, _BWD_OP
  if _FWD_OP is None:
    _FWD_OP, _BWD_OP = _build_ragged_dot_kernels()
  return _FWD_OP, _BWD_OP


class _RaggedDotAutograd(torch.autograd.Function):
  """Autograd Function for grouped matmul with TENSOR ``group_sizes``."""

  @staticmethod
  def forward(ctx, lhs, rhs, group_sizes):
    ctx.save_for_backward(lhs, rhs, group_sizes)
    fwd_op, _ = _get_ops()
    return cast(torch.Tensor, fwd_op(lhs, rhs, group_sizes))

  @staticmethod
  def backward(ctx, grad_output):  # pyrefly: ignore[bad-override]
    lhs, rhs, group_sizes = ctx.saved_tensors
    _, bwd_op = _get_ops()
    grad_lhs, grad_rhs = cast(
        Tuple[torch.Tensor, torch.Tensor],
        bwd_op(grad_output, lhs, rhs, group_sizes),
    )
    return grad_lhs, grad_rhs, None


def segment_matmul_pallas_2d(
    lhs: torch.Tensor, rhs: torch.Tensor, group_sizes: torch.Tensor
) -> torch.Tensor:
  """Per-track ragged dot. ``group_sizes`` is a TPU int32 tensor (no .tolist()).

  Args:
    lhs: ``(total_tokens, hidden_dim)`` — token activations sorted by expert
      assignment.
    rhs: ``(num_experts, hidden_dim, output_dim)`` — per-expert weights.
    group_sizes: ``(num_experts,)`` int32 TPU tensor. Sum must equal
      ``total_tokens``. May be padded with zeros for empty experts.

  Returns:
    ``(total_tokens, output_dim)`` activations.
  """
  if group_sizes.dtype != torch.int32:
    group_sizes = group_sizes.to(torch.int32)
  return _RaggedDotAutograd.apply(lhs, rhs, group_sizes)
