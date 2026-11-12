# Copyright 2026 The TorchTitan Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Linear Softmax Cross-Entropy Loss wrapped for TPU.

The identity case (``soft_cap=None``, ``output_multiplier=1.0``) dispatches to
the installed Tokamax kernel exactly as before. A configured logit transform
(Muse Glimmer's ``SoftCappedLinear`` LM head: ``cap * tanh(z * mul / cap)``)
dispatches to the vendored copy in ``linear_softmax_ce_softcap_kernel``, which
applies the transform to each logits tile in-kernel (see that module's doc).
"""

import hashlib
import inspect

import jax
import tokamax
import torch
from torch_tpu._internal.pallas import pallas
from torchtitan.tools.logging import logger


def _has_logit_transform(soft_cap, output_multiplier) -> bool:
  return soft_cap is not None or output_multiplier != 1.0


_logged_bwd_buffer_events: set[tuple[object, ...]] = set()


def _select_bwd_buffer_count(
    kernel,
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
    dtype,
) -> tuple[int, int] | int | None:
  """Picks the backward kernel's gradient buffering so its tiles fit in VMEM.

  ``buffer_count`` controls how the backward kernel moves its two fp32
  gradient blocks (``x_grad`` of shape (B, H) and ``w_grad`` of shape (H, V))
  between HBM and VMEM. Each block is read, accumulated into and written
  back, so it has two sides:
    (in, out): ``in`` VMEM copies for reading the next block ahead of time,
      ``out`` copies for writing finished blocks back while compute
      continues. An int ``n`` means ``(n, 1)``.
    (2, 2): reads and write-backs both overlap compute (Tokamax's default).
    (2, 1): reads overlap; each write-back must finish before the kernel moves
      on. One fewer gradient copy in VMEM.
    1: no overlap at all. Smallest, slowest.
  Each extra copy costs one fp32 gradient tile per block, so ``(2, 2)`` needs
  two more of each than ``1``. Tokamax may lower the count further when a
  block is revisited too soon to buffer safely.

  Tries ``(2, 2)``, ``(2, 1)``, then ``1``, and returns the first that fits
  under the kernel's VMEM limit (90% of capacity). Returns ``None`` when
  ``(2, 2)`` fits, so Tokamax's default is passed through unchanged. If even
  ``1`` exceeds the estimate, returns ``1`` and logs a warning, since the
  kernel will likely fail to compile.

  The VMEM use comes from Tokamax's ``_calculate_bwd_vmem_bytes``, which
  over-counts: besides the (B, V) logits temporary (which can overlap the
  previous V-step's gradient matmuls via the 2-slot scratch buffer), it
  reserves space for both the (H, V) weight-gradient product and the (B, H)
  input-gradient product as if both were live at once. The kernel computes
  those two gradient products one after another, so only the larger of the
  two needs space, and the estimate here drops the smaller one.

  For example, on TPU7x (57.6 MiB limit) with bf16 inputs:
    (1024, 512, 2048): 56.4 MiB at (2, 2) -> keeps the default.
    (1024, 1024, 1280): 61.4 MiB at (2, 2), 52.4 MiB at (2, 1) -> (2, 1).
    (1024, 2048, 640): 72.9 MiB at (2, 2), 59.9 MiB at (2, 1), 46.9 MiB at 1
      -> 1.

  Logs once per tile and dtype when it changes the buffering, and keeps the
  default (with a warning) if the estimate is unavailable.
  """
  calc_fn = getattr(kernel, "_calculate_bwd_vmem_bytes", None)
  if calc_fn is None:
    event_key = ("missing_calc_fn",)
    if event_key not in _logged_bwd_buffer_events:
      _logged_bwd_buffer_events.add(event_key)
      logger.warning(
          "Tokamax kernel is missing _calculate_bwd_vmem_bytes; keeping"
          " default (2, 2) gradient buffering."
      )
    return None
  try:
    from jax.experimental.pallas import tpu as pltpu  # pylint: disable=g-import-not-at-top

    limit_fn = getattr(kernel, "_get_vmem_limit_bytes", None)
    vmem_limit = (
        limit_fn()
        if limit_fn is not None
        else int(0.90 * pltpu.get_tpu_info().vmem_capacity_bytes)
    )
    h_alloc = 1 << (h_block_size - 1).bit_length()
    non_overlapping_grad_dot_bytes = min(
        4 * h_alloc * v_block_size,
        4 * b_block_size * h_alloc,
    )

    def estimate_vmem(buf_count: tuple[int, int] | int) -> int:
      return (
          calc_fn(
              b_block_size,
              h_block_size,
              v_block_size,
              dtype=dtype,
              buffer_count=buf_count,
          )
          - non_overlapping_grad_dot_bytes
      )

    vmem_22 = estimate_vmem((2, 2))
    if vmem_22 <= vmem_limit:
      return None
    chosen: tuple[int, int] | int = (
        (2, 1) if estimate_vmem((2, 1)) <= vmem_limit else 1
    )
    vmem_chosen = estimate_vmem(chosen)
    event_key = (
        "buffer_count",
        b_block_size,
        h_block_size,
        v_block_size,
        str(jax.numpy.dtype(dtype)),
        chosen,
    )
    if event_key not in _logged_bwd_buffer_events:
      _logged_bwd_buffer_events.add(event_key)
      if vmem_chosen > vmem_limit:
        logger.warning(
            "Fused CE backward tiles (%d, %d, %d) (%s) exceed VMEM limit"
            " %.1f MiB even at buffer_count=1 (est. %.1f MiB); kernel will"
            " likely fail to compile (use bfloat16 inputs or smaller tiles).",
            b_block_size,
            h_block_size,
            v_block_size,
            jax.numpy.dtype(dtype),
            vmem_limit / (1024 * 1024),
            vmem_chosen / (1024 * 1024),
        )
      else:
        logger.info(
            "Fused CE backward tiles (%d, %d, %d) (%s) exceed VMEM limit"
            " %.1f MiB at buffer_count=(2, 2) (est. %.1f MiB); using"
            " buffer_count=%s (est. %.1f MiB).",
            b_block_size,
            h_block_size,
            v_block_size,
            jax.numpy.dtype(dtype),
            vmem_limit / (1024 * 1024),
            vmem_22 / (1024 * 1024),
            chosen,
            vmem_chosen / (1024 * 1024),
        )
    return chosen
  except Exception:  # pylint: disable=broad-exception-caught
    # Keep (2, 2) buffering, but say why: a Tokamax API change here silently
    # disables the fallback and surfaces as a compile-time VMEM OOM.
    logger.warning(
        "Could not estimate the fused CE backward VMEM for tiles (%d, %d, %d);"
        " keeping the default (2, 2) gradient buffering.",
        b_block_size,
        h_block_size,
        v_block_size,
        exc_info=True,
    )
    return None


def _my_fwd(
    x,
    labels,
    weights,
    reduction,
    implementation,
    b_block_size,
    h_block_size,
    v_block_size,
    soft_cap=None,
    output_multiplier=1.0,
):
  if _has_logit_transform(soft_cap, output_multiplier):
    from torchtitan.experiments.tpu.kernels import linear_softmax_ce_softcap_kernel as softcap_kernel  # pylint: disable=g-import-not-at-top

    fwd_fn = (
        softcap_kernel.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu
    )
    return fwd_fn(
        x,
        labels,
        weights,
        b_block_size=b_block_size,
        h_block_size=h_block_size,
        v_block_size=v_block_size,
        reduction=reduction,
        soft_cap=soft_cap,
        output_multiplier=output_multiplier,
    )

  if implementation == "xla":
    from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference  # pylint: disable=g-import-not-at-top

    loss, lse = reference.linear_softmax_cross_entropy_loss_fwd_reference(
        x, labels, weights, reduction=reduction
    )
    return loss, lse

  from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu_kernel as kernel

  loss, lse = kernel.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
      x,
      labels,
      weights,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
      reduction=reduction,
  )
  return loss, lse


def _my_bwd(
    dout,
    lse,
    x,
    labels,
    weights,
    reduction,
    implementation,
    b_block_size,
    h_block_size,
    v_block_size,
    soft_cap=None,
    output_multiplier=1.0,
):
  # The Torch custom op exposes the reduced loss as a one-element tensor, but
  # Tokamax's backward API consumes its cotangent as a scalar. Whole-graph XLA
  # can fold that distinction away; an executable boundary preserves the
  # one-element shape, so normalize it explicitly before entering Tokamax.
  dout = jax.numpy.reshape(dout, ())
  if _has_logit_transform(soft_cap, output_multiplier):
    from torchtitan.experiments.tpu.kernels import linear_softmax_ce_softcap_kernel as softcap_kernel  # pylint: disable=g-import-not-at-top

    grad_fn = (
        softcap_kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu
    )
    grad_x, grad_w = grad_fn(
        dout,
        lse,
        x,
        labels,
        weights,
        b_block_size=b_block_size,
        h_block_size=h_block_size,
        v_block_size=v_block_size,
        reduction=reduction,
        soft_cap=soft_cap,
        output_multiplier=output_multiplier,
    )
    # Same autograd dtype contract as the identity paths below.
    return grad_x.astype(x.dtype), grad_w.astype(weights.dtype)

  if implementation == "xla":
    from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference  # pylint: disable=g-import-not-at-top

    grad_x, grad_w = reference.linear_softmax_cross_entropy_loss_bwd_reference(
        dout, lse, x, labels, weights, reduction=reduction
    )
    # Match the custom-op schema and ordinary linear autograd: Tokamax may
    # accumulate in FP32, but gradients cross this boundary in the input dtype.
    return grad_x.astype(x.dtype), grad_w.astype(weights.dtype)

  from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu_kernel as kernel  # pylint: disable=g-import-not-at-top

  bwd_fn = kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu
  bwd_kwargs = {}
  bwd_dtype = jax.numpy.result_type(x.dtype, weights.dtype)
  if "buffer_count" in inspect.signature(inspect.unwrap(bwd_fn)).parameters:
    buf_count = _select_bwd_buffer_count(
        kernel, b_block_size, h_block_size, v_block_size, bwd_dtype
    )
    if buf_count is not None:
      bwd_kwargs["buffer_count"] = buf_count
  grad_x, grad_w = bwd_fn(
      dout,
      lse,
      x,
      labels,
      weights,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
      reduction=reduction,
      **bwd_kwargs,
  )
  # Keep the Pallas and reference implementations on the same autograd dtype
  # contract. The pinned Tokamax revision separately fixes where dout is
  # applied so this cast occurs after the BF16 gradient GEMMs.
  return grad_x.astype(x.dtype), grad_w.astype(weights.dtype)


_loss_op_name_cache = {}


@torch._dynamo.assume_constant_result
def _get_loss_op_names(
    reduction,
    implementation,
    b_block_size,
    h_block_size,
    v_block_size,
    soft_cap=None,
    output_multiplier=1.0,
):
  cache_key = (
      reduction,
      implementation,
      b_block_size,
      h_block_size,
      v_block_size,
  )
  if _has_logit_transform(soft_cap, output_multiplier):
    # Only extend the key for a configured transform so the identity case keeps
    # its historical op names (and compilation-cache entries).
    cache_key = cache_key + (soft_cap, output_multiplier)
  if cache_key in _loss_op_name_cache:
    return _loss_op_name_cache[cache_key]

  compiled_fwd = pallas.custom_jax_kernel(
      _my_fwd, static_argnums=(3, 4, 5, 6, 7, 8, 9)
  )
  compiled_bwd = pallas.custom_jax_kernel(
      _my_bwd, static_argnums=(5, 6, 7, 8, 9, 10, 11)
  )

  def fwd_wrapped(x, labels, weights):
    return compiled_fwd(
        x,
        labels,
        weights,
        reduction,
        implementation,
        b_block_size,
        h_block_size,
        v_block_size,
        soft_cap,
        output_multiplier,
    )

  def bwd_wrapped(dout, lse, x, labels, weights):
    return compiled_bwd(
        dout,
        lse,
        x,
        labels,
        weights,
        reduction,
        implementation,
        b_block_size,
        h_block_size,
        v_block_size,
        soft_cap,
        output_multiplier,
    )

  # Truncate SHA-256 to 16 hex characters (64 bits) for op name readability.
  identifier = hashlib.sha256(str(cache_key).encode("utf-8")).hexdigest()[:16]
  op_fwd_name = f"linear_loss_fwd_{identifier}"
  op_bwd_name = f"linear_loss_bwd_{identifier}"

  torch_fwd_fn = torch.library.custom_op(
      f"pallas::{op_fwd_name}",
      fwd_wrapped,
      mutates_args=(),
      schema="(Tensor x, Tensor labels, Tensor weights) -> (Tensor, Tensor)",
  )
  torch_fwd_fn.register_fake(
      lambda x, labels, weights: (
          torch.empty(1, dtype=torch.float32, device=x.device),
          torch.empty(x.shape[0], dtype=torch.float32, device=x.device),
      )
  )

  torch_bwd_fn = torch.library.custom_op(
      f"pallas::{op_bwd_name}",
      bwd_wrapped,
      mutates_args=(),
      schema=(
          "(Tensor dout, Tensor lse, Tensor x, Tensor labels, Tensor weights)"
          " -> (Tensor, Tensor)"
      ),
  )
  torch_bwd_fn.register_fake(
      lambda dout, lse, x, labels, weights: (
          torch.empty_like(x),
          torch.empty_like(weights),
      )
  )

  _loss_op_name_cache[cache_key] = (op_fwd_name, op_bwd_name)
  return op_fwd_name, op_bwd_name


class TorchLinearLoss(torch.autograd.Function):
  """Linear Softmax Cross-Entropy Loss autograd function."""

  @staticmethod
  def forward(
      ctx,
      x,
      labels,
      weights,
      reduction="sum",
      implementation="mosaic_tpu",
      b_block_size=1024,
      h_block_size=512,
      v_block_size=2048,
      soft_cap=None,
      output_multiplier=1.0,
  ):
    labels_int32 = labels.to(torch.int32)

    fwd_name, _ = _get_loss_op_names(
        reduction,
        implementation,
        b_block_size,
        h_block_size,
        v_block_size,
        soft_cap,
        output_multiplier,
    )
    torch_fwd_fn = getattr(torch.ops.pallas, fwd_name)
    loss, lse = torch_fwd_fn(x, labels_int32, weights)
    ctx.save_for_backward(x, labels_int32, weights, lse)
    ctx.reduction = reduction
    ctx.implementation = implementation
    ctx.b_block_size = b_block_size
    ctx.h_block_size = h_block_size
    ctx.v_block_size = v_block_size
    ctx.soft_cap = soft_cap
    ctx.output_multiplier = output_multiplier

    return loss

  @staticmethod
  def backward(ctx, grad_output):  # pyrefly: ignore[bad-override]
    x, labels, weights, lse = ctx.saved_tensors
    reduction = ctx.reduction
    implementation = ctx.implementation
    b_block_size = ctx.b_block_size
    h_block_size = ctx.h_block_size
    v_block_size = ctx.v_block_size

    _, bwd_name = _get_loss_op_names(
        reduction,
        implementation,
        b_block_size,
        h_block_size,
        v_block_size,
        ctx.soft_cap,
        ctx.output_multiplier,
    )
    torch_bwd_fn = getattr(torch.ops.pallas, bwd_name)
    grad_x, grad_w = torch_bwd_fn(grad_output, lse, x, labels, weights)

    return grad_x, None, grad_w, None, None, None, None, None, None, None


def linear_softmax_cross_entropy_loss(
    x: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    reduction: str = "sum",
    implementation: str = "mosaic_tpu",
    b_block_size: int = 1024,
    h_block_size: int = 512,
    v_block_size: int = 2048,
    soft_cap: float | None = None,
    output_multiplier: float = 1.0,
) -> torch.Tensor:
  """Linear Softmax Cross-Entropy Loss.

  Args:
      x: The last layer output in the dimension of (B, H) where B is batch size
        and H is the hidden dimension.
      labels: The ground truth labels index in the dimension of (B,).
      weights: The linear projection weight matrix in the dimension of (H, V)
        where V is the dimension of the output logits aka vocabulary size.
      reduction: The reduction method for the cross entropy loss. Can be set to
        "sum" or "mean" explicitly.
      implementation: By default "None" will be used to pick the best available
        backend. Can be set to "xla" or "mosaic_tpu" explicitly.
      b_block_size: Block size for batch dimension.
      h_block_size: Block size for hidden dimension.
      v_block_size: Block size for vocab dimension.
      soft_cap: Optional tanh soft-cap on the logits, applied in-kernel as
        ``soft_cap * tanh(logits * output_multiplier / soft_cap)`` (Muse
        Glimmer's ``SoftCappedLinear``). ``None`` leaves the logits uncapped.
      output_multiplier: Multiplier on the logits (applied before the cap).
        ``1.0`` together with ``soft_cap=None`` is the unchanged Tokamax path.

  Returns:
      The Cross-Entropy loss.

  Raises:
      ValueError: If a logit transform is combined with a non-Mosaic
        ``implementation``.
  """
  if _has_logit_transform(soft_cap, output_multiplier) and (
      implementation != "mosaic_tpu"
  ):
    raise ValueError(
        "soft_cap / output_multiplier are only supported with"
        f" implementation='mosaic_tpu', got {implementation!r}"
    )
  return TorchLinearLoss.apply(
      x,
      labels.to(torch.int32),
      weights,
      reduction,
      implementation,
      b_block_size,
      h_block_size,
      v_block_size,
      soft_cap,
      output_multiplier,
  )
