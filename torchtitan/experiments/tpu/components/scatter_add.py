# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Row-gather backward registration for ``torchtitan::deterministic_scatter_add`` on TPU.

``LocalTokenDispatcher.combine``, ``AllToAllTokenDispatcher.combine``, and
``TorchAOTokenDispatcher.combine`` invoke ``deterministic_scatter_add`` with a
row-broadcast 2D index tensor ``token_indices.reshape(-1, 1).expand(-1, D)``.
Upstream's default backward ``torch.gather(grad_output, dim=0, index=index)``
lowers on XLA/TPU to a 2D elementwise ``HloGather`` over ``N * D`` scalar
coordinates (~4.22 s per MoE layer on TPU v7x at ``(N=40960, D=2048)`` bf16),
and materializes a dense ``(N, D)`` ``int64`` tensor under ``save_on_cpu()``.

When ``index`` is row-broadcast across columns (``stride(1) == 0`` or ``D == 1``
with ``D > 0``), saving the 1D row index ``index[:, 0]`` in ``_setup_context``
and computing ``grad_output.index_select(0, index)`` in ``_backward`` lowers to
a contiguous row-slice ``HloGather`` (~4.4 ms per layer on TPU v7x) with
bit-identical gradients (``max_abs == 0.0``), while preserving ``torch.gather``
for arbitrary non-row-broadcast 2D ``index`` inputs.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from torchtitan.ops.scatter_add import deterministic_scatter_add

_ORIG_SETUP_CONTEXT_FN: Callable[..., Any] | None = None
_ORIG_BACKWARD_FN: Callable[..., Any] | None = None
_INSTALLED: bool = False


def _is_row_broadcast_2d(
    out: torch.Tensor,
    index: torch.Tensor,
    src: torch.Tensor,
) -> bool:
  """Returns True when ``index`` broadcasts a 1D row index across all columns."""
  if (
      out.ndim != 2
      or index.ndim != 2
      or index.shape != src.shape
      or index.shape[1] != out.shape[1]
      or index.shape[1] == 0
  ):
    return False
  if isinstance(index, DTensor) and not all(
      placement.is_replicate() for placement in index.placements
  ):
    return False
  return (
      index.stride(1) == 0
      or index.shape[1] == 1
      or index.shape[0] == 0
  )


def _tpu_setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output: torch.Tensor,
) -> None:
  del output
  if not ctx.needs_input_grad[2]:
    ctx.save_for_backward()
    return
  out, index, src = inputs
  if _is_row_broadcast_2d(out, index, src):
    index = index[:, 0]
  ctx.save_for_backward(index)


def _tpu_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, None, torch.Tensor | None]:
  grad_out = grad_output if ctx.needs_input_grad[0] else None
  if not ctx.needs_input_grad[2]:
    grad_src = None
  else:
    (index,) = ctx.saved_tensors  # pyrefly: ignore[missing-attribute]
    if index.ndim == 1 and grad_output.ndim == 2:
      grad_src = grad_output.index_select(0, index)
    else:
      grad_src = torch.gather(grad_output, dim=0, index=index)
  return grad_out, None, grad_src


def install_tpu_scatter_add_row_gather() -> None:
  """Installs the row-gather backward on ``deterministic_scatter_add`` (idempotent)."""
  global _ORIG_SETUP_CONTEXT_FN, _ORIG_BACKWARD_FN, _INSTALLED
  if (
      _INSTALLED
      and deterministic_scatter_add._setup_context_fn is _tpu_setup_context  # pylint: disable=protected-access
      and deterministic_scatter_add._backward_fn is _tpu_backward  # pylint: disable=protected-access
  ):
    return
  if _ORIG_SETUP_CONTEXT_FN is None and _ORIG_BACKWARD_FN is None:
    _ORIG_SETUP_CONTEXT_FN = deterministic_scatter_add._setup_context_fn  # pylint: disable=protected-access
    _ORIG_BACKWARD_FN = deterministic_scatter_add._backward_fn  # pylint: disable=protected-access
  deterministic_scatter_add.register_autograd(
      _tpu_backward,
      setup_context=_tpu_setup_context,
  )
  _INSTALLED = True


def uninstall_tpu_scatter_add_row_gather() -> None:
  """Restores upstream's original autograd registration (test-only).

  Warning:
    Intended strictly for unit tests comparing installed vs. uninstalled runs.
    Do not toggle installation between a forward pass and its corresponding
    backward pass: PyTorch's ``CustomOpDef`` dispatches ``_setup_context_fn``
    at forward time and ``_backward_fn`` at backward time.
  """
  global _INSTALLED
  if not _INSTALLED:
    return
  if _ORIG_BACKWARD_FN is not None:
    deterministic_scatter_add.register_autograd(
        _ORIG_BACKWARD_FN,
        setup_context=_ORIG_SETUP_CONTEXT_FN,
    )
  _INSTALLED = False


def is_tpu_scatter_add_row_gather_installed() -> bool:
  """Returns True when the row-gather backward is currently active."""
  return (
      _INSTALLED
      and deterministic_scatter_add._setup_context_fn is _tpu_setup_context  # pylint: disable=protected-access
      and deterministic_scatter_add._backward_fn is _tpu_backward  # pylint: disable=protected-access
  )
