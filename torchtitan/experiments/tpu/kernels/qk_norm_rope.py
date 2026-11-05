# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 QK-norm + rotate-half RoPE with BF16 HBM traffic on TPU.

``qk_norm_rope(x, w, cos, sin, eps)`` equals
``cossin_rotate(F.rms_norm(x, (D,), w, eps), cos, sin)`` bit for bit, forward
and backward. Unfused, XLA materializes several FP32 copies of the
[B, L, H, D] activation between the norm statistics, the norm, the rotation
and the rotation transpose. Here every elementwise pass is a Pallas kernel
with BF16 inputs and outputs and FP32 math in VMEM.

The reductions (the per-row sum of squares, and in backward the per-row
``sum(dy * w * n)`` and the weight gradient) stay in XLA, written exactly as
TorchTPU lowers ``rms_norm`` and its backward, so they keep XLA's summation
order. The kernels repeat the remaining FP32 operations of that lowering:

* norm: ``y = bf16((x * r) * w)`` with ``r = rsqrt(sum(x * x) / D + eps)``;
* rotation: ``bf16(y1 * c - y2 * s)`` and ``bf16(y2 * c + y1 * s)``;
* rotation transpose: ``dy = bf16(g1 * c + g2 * s | g2 * c - g1 * s)``;
* input gradient: ``bf16((dy * w - n * m) * r)`` with ``n = x * r`` and
  ``m = sum(dy * w * n) / D``.

The kernels run head-dimension-major with the sequence on lanes
([B, H, D, L]), the layout XLA gives the QKV projection feeding
sequence-minor Splash, so the transposes around them are bitcasts.
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas


def _block(n: int, target: int) -> int:
  b = min(n, target)
  while n % b:
    b //= 2
  return b


def _per_head(ref, body):
  def step(i, carry):
    body(i)
    return carry

  jax.lax.fori_loop(0, ref.shape[1], step, 0)


def _norm_rope_kernel(x_ref, r_ref, w_ref, c_ref, s_ref, o_ref):
  h = x_ref.shape[2] // 2
  w = jnp.broadcast_to(w_ref[...], x_ref.shape[2:])
  c, s = c_ref[0], s_ref[0]

  def body(i):
    xf = x_ref[0, i].astype(jnp.float32)
    y = ((xf * r_ref[0, i]) * w).astype(jnp.bfloat16).astype(jnp.float32)
    y1, y2 = y[:h], y[h:]
    # a - b is exactly a + (-b), and (-a) * b is exactly -(a * b).
    o_ref[0, i, :h] = (y1 * c - y2 * s).astype(o_ref.dtype)
    o_ref[0, i, h:] = (y2 * c + y1 * s).astype(o_ref.dtype)

  _per_head(x_ref, body)


def _rope_t_kernel(g_ref, c_ref, s_ref, dy_ref):
  h = g_ref.shape[2] // 2
  c, s = c_ref[0], s_ref[0]

  def body(i):
    g = g_ref[0, i].astype(jnp.float32)
    g1, g2 = g[:h], g[h:]
    dy_ref[0, i, :h] = (g1 * c + g2 * s).astype(dy_ref.dtype)
    dy_ref[0, i, h:] = (g2 * c - g1 * s).astype(dy_ref.dtype)

  _per_head(g_ref, body)


def _norm_dx_kernel(x_ref, dy_ref, r_ref, m_ref, w_ref, dx_ref):
  w = jnp.broadcast_to(w_ref[...], x_ref.shape[2:])

  def body(i):
    r = r_ref[0, i]
    n = x_ref[0, i].astype(jnp.float32) * r
    gw = dy_ref[0, i].astype(jnp.float32) * w
    dx_ref[0, i] = ((gw - n * m_ref[0, i]) * r).astype(dx_ref.dtype)

  _per_head(x_ref, body)


def _call(kernel, x, operands, hb, lt):
  """Runs ``kernel`` on [B, H, D, L] blocks; ``x`` gives the [B, L, H, D] shape.

  ``operands`` are (array, kind) pairs; kind is "x" ([B, L, H, D]), "row"
  ([B, L, H]), "cs" ([B, L, D / 2]) or "w" ([D]).
  """
  b, l, hh, d = x.shape
  hb, lt = _block(hh, hb), _block(l, lt)
  specs = {
      "x": pl.BlockSpec((1, hb, d, lt), lambda i, j, k: (i, j, 0, k)),
      "row": pl.BlockSpec((1, hb, 1, lt), lambda i, j, k: (i, j, 0, k)),
      "cs": pl.BlockSpec((1, d // 2, lt), lambda i, j, k: (i, 0, k)),
      "w": pl.BlockSpec((d, 1), lambda i, j, k: (0, 0)),
  }
  layouts = {
      "x": lambda a: jnp.transpose(a, (0, 2, 3, 1)),
      "row": lambda a: jnp.transpose(a, (0, 2, 1))[:, :, None, :],
      "cs": lambda a: jnp.transpose(a, (0, 2, 1)),
      "w": lambda a: a.astype(jnp.float32)[:, None],
  }
  out = pl.pallas_call(
      kernel,
      grid=(b, hh // hb, l // lt),
      in_specs=[specs[kind] for _, kind in operands],
      out_specs=specs["x"],
      out_shape=jax.ShapeDtypeStruct((b, hh, d, l), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel")
      ),
  )(*(layouts[kind](a) for a, kind in operands))
  return jnp.transpose(out, (0, 3, 1, 2))


@pallas.jax_op("tt_attn::qk_norm_rope")
def _norm_rope(
    x: jax.Array,
    r: jax.Array,
    weight: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    hb: int,
    lt: int,
) -> jax.Array:
  return _call(
      _norm_rope_kernel,
      x,
      ((x, "x"), (r, "row"), (weight, "w"), (cos, "cs"), (sin, "cs")),
      hb,
      lt,
  )


@pallas.jax_op("tt_attn::rope_transpose")
def _rope_transpose(
    grad: jax.Array, cos: jax.Array, sin: jax.Array, hb: int, lt: int
) -> jax.Array:
  return _call(
      _rope_t_kernel, grad, ((grad, "x"), (cos, "cs"), (sin, "cs")), hb, lt
  )


@pallas.jax_op("tt_attn::rms_norm_input_grad")
def _rms_norm_input_grad(
    x: jax.Array,
    dy: jax.Array,
    r: jax.Array,
    m: jax.Array,
    weight: jax.Array,
    hb: int,
    lt: int,
) -> jax.Array:
  return _call(
      _norm_dx_kernel,
      x,
      ((x, "x"), (dy, "x"), (r, "row"), (m, "row"), (weight, "w")),
      hb,
      lt,
  )


def _inv_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
  # TorchTPU's rms_norm lowering: rsqrt(sum(x * x) * (1 / D) + eps) in FP32.
  xf = x.float()
  return torch.rsqrt((xf * xf).sum(-1) * (1.0 / x.shape[-1]) + eps)


class _QKNormRope(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x, weight, cos, sin, eps, hb, lt):
    ctx.save_for_backward(x, weight, cos, sin)
    ctx.cfg = (eps, hb, lt)
    return _norm_rope(x, _inv_rms(x, eps), weight, cos, sin, hb, lt)

  @staticmethod
  def backward(ctx, grad):
    x, weight, cos, sin = ctx.saved_tensors
    eps, hb, lt = ctx.cfg
    r = _inv_rms(x, eps)
    dy = _rope_transpose(grad.contiguous(), cos, sin, hb, lt)
    # TorchTPU's rms_norm backward reductions, in its operation order.
    dyf = dy.float()
    n = x.float() * r.unsqueeze(-1)
    m = (dyf * weight.float() * n).sum(-1) * (1.0 / x.shape[-1])
    dw = (dyf * n).sum((0, 1, 2)).to(weight.dtype)
    dx = _rms_norm_input_grad(x, dy, r, m, weight, hb, lt)
    return dx, dw, None, None, None, None, None


def qk_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    *,
    heads_per_block: int = 16,
    seq_block: int = 1024,
) -> torch.Tensor:
  """``cossin_rotate(F.rms_norm(x, (D,), weight, eps), cos, sin)`` on TPU.

  Args:
    x: BF16 [B, L, H, D] queries or keys.
    weight: [D] RMSNorm weight.
    cos: FP32 [B, L, D/2] cosines (the first half of the CosSinRoPE cos).
    sin: FP32 [B, L, D/2] sines.
    eps: RMSNorm epsilon.
    heads_per_block: Heads per kernel grid step.
    seq_block: Sequence positions per kernel grid step.
  """
  return _QKNormRope.apply(
      x, weight, cos, sin, float(eps), heads_per_block, seq_block
  )


__all__ = ["qk_norm_rope"]
