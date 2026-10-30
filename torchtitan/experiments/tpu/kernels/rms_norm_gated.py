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
from torch.distributed.tensor import DTensor
from torch_tpu._internal import pallas

_BLOCK_ROWS = 2048
_BLOCK_DIM = 128
_MAX_DIM = 512
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


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

  @pl.when(pl.program_id(0) == 0)
  def _():
    dw_ref[...] = jnp.zeros_like(dw_ref)

  dw_ref[...] += jnp.sum(d_scaled * n, axis=0, keepdims=True)
  dn = d_scaled * w
  dx = rstd * (dn - n * jnp.mean(dn * n, axis=-1, keepdims=True))
  dx_ref[...] = dx.astype(dx_ref.dtype)


def _block_rows(dim: int, *dtypes) -> int:
  """Row block size (``<= 2048``, power of 2) keeping tile buffers <= 1 MiB."""
  elem_bytes = max(jnp.dtype(dt).itemsize for dt in dtypes)
  # Keep each (block_rows, dim) buffer <= 1 MiB so 5 double-buffered backward
  # arrays (x, gate, dy, dx, dg = 10 buffers) stay <= 10 MiB, fitting TPU
  # v5e/v6e's 16 MiB default scoped VMEM while keeping 2048 rows at dim=128.
  block = _BLOCK_ROWS
  while block > 128 and block * dim * elem_bytes > 1024 * 1024:
    block //= 2
  return block


def _row_spec(block_rows: int, dim: int) -> pl.BlockSpec:
  return pl.BlockSpec((block_rows, dim), lambda i: (i, 0))


def rms_norm_gated_fwd_jax(
    x: jax.Array, gate: jax.Array, w: jax.Array, eps: float
) -> jax.Array:
  rows, dim = x.shape
  block_rows = _block_rows(dim, x.dtype, gate.dtype)
  row_spec = _row_spec(block_rows, dim)
  return pl.pallas_call(
      lambda *refs: _fwd_kernel(eps, *refs),
      grid=(rows // block_rows,),
      in_specs=[
          row_spec,
          row_spec,
          pl.BlockSpec((1, dim), lambda i: (0, 0)),
      ],
      out_specs=row_spec,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
      name="rms_norm_gated_fwd",
  )(x, gate, w)


def rms_norm_gated_bwd_jax(
    x: jax.Array, gate: jax.Array, w: jax.Array, dy: jax.Array, eps: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns ``(dx, dg, dw_1D)`` accumulated in FP32 across row blocks."""
  rows, dim = x.shape
  block_rows = _block_rows(dim, x.dtype, gate.dtype, dy.dtype)
  row_spec = _row_spec(block_rows, dim)
  return pl.pallas_call(
      lambda *refs: _bwd_kernel(eps, *refs),
      grid=(rows // block_rows,),
      in_specs=[
          row_spec,
          row_spec,
          pl.BlockSpec((1, dim), lambda i: (0, 0)),
          row_spec,
      ],
      out_specs=[
          row_spec,
          row_spec,
          pl.BlockSpec((1, dim), lambda i: (0, 0)),
      ],
      out_shape=[
          jax.ShapeDtypeStruct(x.shape, x.dtype),
          jax.ShapeDtypeStruct(gate.shape, gate.dtype),
          jax.ShapeDtypeStruct((1, dim), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
      name="rms_norm_gated_bwd",
  )(x, gate, w, dy)


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
  ctx.set_materialize_grads(False)
  x, gate, w, eps = inputs
  ctx.save_for_backward(x, gate, w)
  ctx.eps = eps


def _backward(ctx, dy):
  need_dx, need_dg, need_dw, _ = ctx.needs_input_grad
  if dy is None or not (need_dx or need_dg or need_dw):
    return None, None, None, None
  x, gate, w = ctx.saved_tensors
  dx, dg, dw = _bwd_op(x, gate, w, dy.contiguous(), ctx.eps)
  return (
      dx if need_dx else None,
      dg if need_dg else None,
      dw.reshape(w.shape) if need_dw else None,
      None,
  )


_fwd_op.register_autograd(_backward, setup_context=_setup_context)


def supports(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor) -> bool:
  """Whether ``(x, gate, weight)`` satisfy the Pallas kernel limits."""
  if (
      isinstance(x, DTensor)
      or isinstance(gate, DTensor)
      or isinstance(weight, DTensor)
  ):
    return False
  if (
      x.device.type != "tpu"
      or gate.device != x.device
      or weight.device != x.device
  ):
    return False
  if (
      x.dtype not in _SUPPORTED_DTYPES
      or gate.dtype not in _SUPPORTED_DTYPES
      or weight.dtype not in _SUPPORTED_DTYPES
  ):
    return False
  if x.ndim < 2 or gate.shape != x.shape:
    return False
  dim = x.shape[-1]
  if dim <= 0 or dim > _MAX_DIM or dim % _BLOCK_DIM != 0:
    return False
  if weight.shape != (dim,):
    return False
  rows = x.numel() // dim
  return rows > 0 and rows % _BLOCK_ROWS == 0


def rms_norm_gated(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
  """``RMSNormGated`` over the last dim of ``x``; ``gate`` has ``x``'s shape."""
  if not supports(x, gate, weight):
    raise ValueError(
        "Unsupported inputs for Pallas rms_norm_gated:"
        f" x={tuple(x.shape)} ({x.dtype}, {x.device}),"
        f" gate={tuple(gate.shape)} ({gate.dtype}, {gate.device}),"
        f" weight={tuple(weight.shape)} ({weight.dtype}, {weight.device})"
    )
  if eps < 0:
    raise ValueError(f"eps must be non-negative, got {eps}")
  dim = x.shape[-1]
  w = weight.to(torch.float32).reshape(1, dim).contiguous()
  out = _fwd_op(
      x.reshape(-1, dim).contiguous(),
      gate.reshape(-1, dim).contiguous(),
      w,
      float(eps),
  )
  return out.reshape(x.shape)
