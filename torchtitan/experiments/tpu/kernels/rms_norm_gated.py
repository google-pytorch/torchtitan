"""Pallas TPU kernels for Qwen3.5's gated RMSNorm (``RMSNormGated``).

GatedDeltaNet normalizes every ``[T, H, 128]`` head output in FP32, scales it by
a learned weight, rounds to the input dtype and multiplies by ``silu(gate)``
(``qwen3_5.gdn.RMSNormGated``). XLA splits this into several elementwise and
reduction fusions with FP32 intermediates in HBM; under full rematerialization
they run twice forward and once backward per layer. These kernels process a
block of rows in VMEM per pass, so each pass reads its inputs and writes its
outputs once. Numerics follow ``RMSNormGated.forward``.
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

_BLOCK_ROWS = 2048


def _silu_and_grad(g: jax.Array) -> tuple[jax.Array, jax.Array]:
  sig = jax.nn.sigmoid(g)
  return g * sig, sig * (1.0 + g * (1.0 - sig))


def _fwd_kernel(eps, x_ref, g_ref, w_ref, o_ref):
  x = x_ref[...].astype(jnp.float32)
  rstd = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
  normed = (w_ref[...] * (x * rstd)).astype(o_ref.dtype).astype(jnp.float32)
  silu, _ = _silu_and_grad(g_ref[...].astype(jnp.float32))
  o_ref[...] = (normed * silu).astype(o_ref.dtype)


def _bwd_kernel(eps, x_ref, g_ref, w_ref, dy_ref, dx_ref, dg_ref, dw_ref):
  x = x_ref[...].astype(jnp.float32)
  w = w_ref[...]
  rstd = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
  n = x * rstd
  scaled = (w * n).astype(x_ref.dtype).astype(jnp.float32)
  silu, dsilu = _silu_and_grad(g_ref[...].astype(jnp.float32))
  dy = dy_ref[...].astype(jnp.float32)
  dg_ref[...] = (dy * scaled * dsilu).astype(dg_ref.dtype)
  d_scaled = dy * silu
  dw_ref[...] = jnp.sum(d_scaled * n, axis=0, keepdims=True)[None]
  dn = d_scaled * w
  dx = rstd * (dn - n * jnp.mean(dn * n, axis=-1, keepdims=True))
  dx_ref[...] = dx.astype(dx_ref.dtype)


def _row_spec(dim: int) -> pl.BlockSpec:
  return pl.BlockSpec((_BLOCK_ROWS, dim), lambda i: (i, 0))


def rms_norm_gated_fwd_jax(
    x: jax.Array, gate: jax.Array, w: jax.Array, eps: float
) -> jax.Array:
  rows, dim = x.shape
  return pl.pallas_call(
      lambda *refs: _fwd_kernel(eps, *refs),
      grid=(rows // _BLOCK_ROWS,),
      in_specs=[
          _row_spec(dim),
          _row_spec(dim),
          pl.BlockSpec((1, dim), lambda i: (0, 0)),
      ],
      out_specs=_row_spec(dim),
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
      name="rms_norm_gated_fwd",
  )(x, gate, w)


def rms_norm_gated_bwd_jax(
    x: jax.Array, gate: jax.Array, w: jax.Array, dy: jax.Array, eps: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
  rows, dim = x.shape
  blocks = rows // _BLOCK_ROWS
  dx, dg, dw_parts = pl.pallas_call(
      lambda *refs: _bwd_kernel(eps, *refs),
      grid=(blocks,),
      in_specs=[
          _row_spec(dim),
          _row_spec(dim),
          pl.BlockSpec((1, dim), lambda i: (0, 0)),
          _row_spec(dim),
      ],
      out_specs=[
          _row_spec(dim),
          _row_spec(dim),
          pl.BlockSpec((1, 1, dim), lambda i: (i, 0, 0)),
      ],
      out_shape=[
          jax.ShapeDtypeStruct(x.shape, x.dtype),
          jax.ShapeDtypeStruct(gate.shape, gate.dtype),
          jax.ShapeDtypeStruct((blocks, 1, dim), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
      name="rms_norm_gated_bwd",
  )(x, gate, w, dy)
  return dx, dg, jnp.sum(dw_parts, axis=(0, 1))


@pallas.jax_op("tt_qwen35::rms_norm_gated_fwd")
def _fwd_op(
    x: jax.Array, gate: jax.Array, w: jax.Array, eps: float
) -> jax.Array:
  return rms_norm_gated_fwd_jax(x, gate, w, eps)


@pallas.jax_op("tt_qwen35::rms_norm_gated_bwd")
def _bwd_op(
    x: jax.Array, gate: jax.Array, w: jax.Array, dy: jax.Array, eps: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
  return rms_norm_gated_bwd_jax(x, gate, w, dy, eps)


def _setup_context(ctx, inputs, output):
  del output
  x, gate, w, eps = inputs
  ctx.save_for_backward(x, gate, w)
  ctx.eps = eps


def _backward(ctx, dy):
  x, gate, w = ctx.saved_tensors
  dx, dg, dw = _bwd_op(x, gate, w, dy.contiguous(), ctx.eps)
  return dx, dg, dw.reshape(w.shape), None


_fwd_op.register_autograd(_backward, setup_context=_setup_context)


def supports(x: torch.Tensor) -> bool:
  """Whether the kernels take ``x`` (TPU, 128-wide rows, whole row blocks)."""
  rows = x.numel() // x.shape[-1] if x.ndim else 0
  return (
      x.device.type == "tpu"
      and x.shape[-1] % 128 == 0
      and rows % _BLOCK_ROWS == 0
  )


def rms_norm_gated(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
  """``RMSNormGated`` over the last dim of ``x``; ``gate`` has ``x``'s shape."""
  dim = x.shape[-1]
  w = weight.to(torch.float32).reshape(1, dim)
  out = _fwd_op(
      x.reshape(-1, dim).contiguous(),
      gate.reshape(-1, dim).to(x.dtype).contiguous(),
      w,
      float(eps),
  )
  return out.reshape(x.shape)
