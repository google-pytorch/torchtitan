"""Pallas TPU kernels for GatedDeltaNet's depthwise causal ``conv1d`` + SiLU.

Qwen3.5's GatedDeltaNet runs a width-4 depthwise causal convolution followed by
SiLU over the ``[B, S, C]`` query, key and value projections. Written as shifted
slices in PyTorch (``qwen3_5.gated_delta._causal_conv``), XLA lowers every
sequence shift to a sublane-misaligned copy: on TPU7x the forward and backward
of one ``[8, 8192, 4096]`` bf16 convolution take ~6 ms, about 7x the HBM
traffic bound. These kernels keep a ``[S, 128]`` column strip in VMEM, shift it
with sublane rotations and fuse the SiLU (and its derivative), so each pass
reads and writes every element once.

Only whole rows are supported (no packed-document resets); see
``qwen3_5.gated_delta.causal_conv_silu`` for the dispatch.
"""


import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

# Channels per kernel block (one lane tile). A [8192, 128] fp32 strip is 4 MiB,
# so the backward's handful of live strips stays well inside scoped VMEM.
_BLOCK_C = 128


def _shift_down(x: jax.Array, shift: int) -> jax.Array:
  """``y[t] = x[t - shift]`` along axis 0, zeros for ``t < shift``."""
  if shift == 0:
    return x
  rolled = pltpu.roll(x, shift=shift, axis=0)
  rows = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
  return jnp.where(rows >= shift, rolled, 0.0)


def _shift_up(x: jax.Array, shift: int) -> jax.Array:
  """``y[t] = x[t + shift]`` along axis 0, zeros for ``t >= S - shift``."""
  if shift == 0:
    return x
  seq_len = x.shape[0]
  rolled = pltpu.roll(x, shift=seq_len - shift, axis=0)
  rows = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
  return jnp.where(rows < seq_len - shift, rolled, 0.0)


def _conv(x: jax.Array, w: jax.Array) -> jax.Array:
  """Causal depthwise conv of a ``[S, Cb]`` fp32 strip with ``[W, Cb]`` taps."""
  width = w.shape[0]
  acc = x * w[width - 1 : width]
  for k in range(width - 1):
    acc = acc + _shift_down(x, width - 1 - k) * w[k : k + 1]
  return acc


def _fwd_kernel(x_ref, w_ref, o_ref):
  z = _conv(x_ref[0].astype(jnp.float32), w_ref[...])
  o_ref[0] = (z * jax.nn.sigmoid(z)).astype(o_ref.dtype)


def _bwd_kernel(x_ref, w_ref, g_ref, dx_ref, dw_ref):
  b = pl.program_id(1)
  w = w_ref[...]
  width = w.shape[0]
  x = x_ref[0].astype(jnp.float32)
  z = _conv(x, w)
  sig = jax.nn.sigmoid(z)
  gz = g_ref[0].astype(jnp.float32) * (sig * (1.0 + z * (1.0 - sig)))
  # dx[t] = sum_k w[k] * gz[t + W - 1 - k]: an anti-causal convolution.
  dx = gz * w[width - 1 : width]
  for k in range(width - 1):
    dx = dx + _shift_up(gz, width - 1 - k) * w[k : k + 1]
  dx_ref[0] = dx.astype(dx_ref.dtype)
  # dw[k] = sum_t x[t - (W - 1 - k)] * gz[t], accumulated over the batch.
  dw = jnp.concatenate(
      [
          jnp.sum(_shift_down(x, width - 1 - k) * gz, axis=0, keepdims=True)
          for k in range(width)
      ],
      axis=0,
  )

  @pl.when(b == 0)
  def _():
    dw_ref[...] = jnp.zeros_like(dw_ref)

  dw_ref[...] += dw


def _specs(batch: int, seq_len: int, channels: int, width: int):
  del batch
  strip = pl.BlockSpec((1, seq_len, _BLOCK_C), lambda c, b: (b, 0, c))
  taps = pl.BlockSpec((width, _BLOCK_C), lambda c, b: (0, c))
  grid = (channels // _BLOCK_C,)
  return strip, taps, grid


def causal_conv_silu_fwd_jax(x: jax.Array, w_WC: jax.Array) -> jax.Array:
  """``silu(causal_conv(x))`` for ``x`` ``[B, S, C]``, fp32 taps ``[W, C]``."""
  batch, seq_len, channels = x.shape
  width = w_WC.shape[0]
  strip, taps, grid = _specs(batch, seq_len, channels, width)
  return pl.pallas_call(
      _fwd_kernel,
      grid=(*grid, batch),
      in_specs=[strip, taps],
      out_specs=strip,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel")
      ),
      name="causal_conv1d_silu_fwd",
  )(x, w_WC)


def causal_conv_silu_bwd_jax(
    x: jax.Array, w_WC: jax.Array, g: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Returns ``(dx, dw_WC)`` for the output gradient ``g`` ``[B, S, C]``."""
  batch, seq_len, channels = x.shape
  width = w_WC.shape[0]
  strip, taps, grid = _specs(batch, seq_len, channels, width)
  return pl.pallas_call(
      _bwd_kernel,
      grid=(*grid, batch),
      in_specs=[strip, taps, strip],
      out_specs=[strip, taps],
      out_shape=[
          jax.ShapeDtypeStruct(x.shape, x.dtype),
          jax.ShapeDtypeStruct(w_WC.shape, jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary")
      ),
      name="causal_conv1d_silu_bwd",
  )(x, w_WC, g)


@pallas.jax_op("tt_qwen35::causal_conv1d_silu_fwd")
def _fwd_op(x: jax.Array, w_WC: jax.Array) -> jax.Array:
  return causal_conv_silu_fwd_jax(x, w_WC)


@pallas.jax_op("tt_qwen35::causal_conv1d_silu_bwd")
def _bwd_op(
    x: jax.Array, w_WC: jax.Array, g: jax.Array
) -> tuple[jax.Array, jax.Array]:
  return causal_conv_silu_bwd_jax(x, w_WC, g)


def _setup_context(ctx, inputs, output):
  del output
  x, w_WC = inputs
  ctx.save_for_backward(x, w_WC)


def _backward(ctx, grad_out):
  x, w_WC = ctx.saved_tensors
  dx, dw_WC = _bwd_op(x, w_WC, grad_out.contiguous())
  return dx, dw_WC


_fwd_op.register_autograd(_backward, setup_context=_setup_context)


def supports(x_BSC: torch.Tensor) -> bool:
  """Whether the kernels take ``x_BSC`` (TPU, lane- and sublane-aligned)."""
  return (
      x_BSC.device.type == "tpu"
      and x_BSC.ndim == 3
      and x_BSC.shape[-1] % _BLOCK_C == 0
      and x_BSC.shape[1] % 8 == 0
  )


def causal_conv_silu(
    x_BSC: torch.Tensor, weight_C1W: torch.Tensor
) -> torch.Tensor:
  """``silu`` of a depthwise causal ``conv1d`` over whole ``[B, S, C]`` rows.

  Args:
    x_BSC: ``[B, S, C]`` input with ``C % 128 == 0`` and ``S % 8 == 0``.
    weight_C1W: ``[C, 1, W]`` depthwise ``Conv1d`` weight. The taps run in
      fp32, as in ``F.conv1d``'s accumulation.

  Returns:
    ``[B, S, C]`` output in ``x_BSC.dtype``.
  """
  w_WC = weight_C1W[:, 0, :].t().to(torch.float32).contiguous()
  return _fwd_op(x_BSC.contiguous(), w_WC)
