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

# Workaround for https://github.com/openxla/tokamax/issues/240
import jaxtyping
# 1. Save the real decorator
_real_jaxtyped = jaxtyping.jaxtyped


# 2. Create a dummy pass-through
def _dummy_jaxtyped(*args, **kwargs):
  def decorator(f):
    return f

  if len(args) == 1 and callable(args[0]):
    return args[0]
  return decorator

# 3. Swap the dummy in, import tokamax, and swap the real one back
jaxtyping.jaxtyped = _dummy_jaxtyped
import tokamax  # pylint: disable=g-import-not-at-top
jaxtyping.jaxtyped = _real_jaxtyped

import jax  # pylint: disable=g-import-not-at-top, g-bad-import-order
import torch
from torch_tpu._internal.pallas import pallas


def _has_logit_transform(soft_cap, output_multiplier) -> bool:
  return soft_cap is not None or output_multiplier != 1.0


def _bwd_vmem_exceeds_capacity(
    kernel,
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
    dtype,
) -> bool:
  """Whether default (2, 2) grad buffering exceeds physical VMEM capacity."""
  calc_fn = getattr(kernel, "_calculate_bwd_vmem_bytes", None)
  if calc_fn is None:
    return False
  try:
    from jax.experimental.pallas import tpu as pltpu  # pylint: disable=g-import-not-at-top

    vmem_22 = calc_fn(
        b_block_size,
        h_block_size,
        v_block_size,
        dtype=dtype,
        buffer_count=(2, 2),
    )
    return vmem_22 > int(0.95 * pltpu.get_tpu_info().vmem_capacity_bytes)
  except Exception:  # pylint: disable=broad-exception-caught
    return False


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
  if "buffer_count" in inspect.signature(
      bwd_fn
  ).parameters and _bwd_vmem_exceeds_capacity(
      kernel, b_block_size, h_block_size, v_block_size, x.dtype
  ):
    # Large single-H tiles such as (1024, 2048, 640) fit TPU7x VMEM at 51.9 MiB
    # with single-buffered grad accumulators, whereas jax >= 0.12's (2, 2)
    # triple-buffering reaches 77.9 MiB and OOMs VMEM at compile time.
    bwd_kwargs["buffer_count"] = 1
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
