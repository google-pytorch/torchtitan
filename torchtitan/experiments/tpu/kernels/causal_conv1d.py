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

import re

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch
from torch.distributed.tensor import DTensor
from torch_tpu._internal import pallas

# Channels per kernel block (one lane tile). At S=8192 on TPU7x, one [8192, 128]
# fp32 strip is 4 MiB, so keeping S bounded by TPU generation and dtype keeps
# the backward's live strips inside scoped VMEM.
_BLOCK_C = 128
_MAX_WIDTH = 8
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


_tpu_gen_cache: list[int | None] = []


@torch._dynamo.assume_constant_result
def _tpu_generation() -> int | None:
  """Attached TPU generation (e.g. 5 for v5e/v5p, 6 for v6e, 7 for TPU7x)."""
  if _tpu_gen_cache:
    return _tpu_gen_cache[0]
  gen: int | None = None
  get_props = getattr(
      getattr(torch, "tpu", None), "get_device_properties", None
  )
  if get_props is not None:
    try:
      name = get_props().name.lower()
      match = re.search(r"(?:v|tpu[- ]?)(\d+)", name)
      gen = int(match.group(1)) if match else None
    except RuntimeError:
      gen = None
  _tpu_gen_cache.append(gen)
  return gen


def _max_supported_seq_len(dtype: torch.dtype) -> int:
  """Maximum whole-strip ``seq_len`` that fits default scoped VMEM."""
  gen = _tpu_generation()
  if gen is not None and gen >= 7:
    # TPU7x has >= 32 MiB default scoped VMEM (63.94 MiB in 35B recipes).
    return 8192 if dtype == torch.bfloat16 else 4096
  # TPU v5/v6 default to 16 MiB scoped VMEM.
  return 2048 if dtype == torch.bfloat16 else 1024


def _shift_down(x: jax.Array, shift: int, rows: jax.Array) -> jax.Array:
  """``y[t] = x[t - shift]`` along axis 0, zeros for ``t < shift``."""
  if shift == 0:
    return x
  rolled = pltpu.roll(x, shift=shift, axis=0)
  return jnp.where(rows >= shift, rolled, 0.0)


def _shift_up(x: jax.Array, shift: int, rows: jax.Array) -> jax.Array:
  """``y[t] = x[t + shift]`` along axis 0, zeros for ``t >= S - shift``."""
  if shift == 0:
    return x
  seq_len = x.shape[0]
  rolled = pltpu.roll(x, shift=seq_len - shift, axis=0)
  return jnp.where(rows < seq_len - shift, rolled, 0.0)


def _conv(x: jax.Array, w: jax.Array, rows: jax.Array) -> jax.Array:
  """Causal depthwise conv of a ``[S, Cb]`` fp32 strip with ``[W, Cb]`` taps."""
  width = w.shape[0]
  acc = x * w[width - 1 : width]
  for k in range(width - 1):
    acc = acc + _shift_down(x, width - 1 - k, rows) * w[k : k + 1]
  return acc


def _fwd_kernel(x_ref, w_ref, o_ref):
  x = x_ref[0].astype(jnp.float32)
  rows = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
  z = _conv(x, w_ref[...], rows)
  o_ref[0] = (z * jax.nn.sigmoid(z)).astype(o_ref.dtype)


def _bwd_kernel(x_ref, w_ref, g_ref, dx_ref, dw_ref):
  b = pl.program_id(1)
  w = w_ref[...]
  width = w.shape[0]
  x = x_ref[0].astype(jnp.float32)
  rows = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
  z = _conv(x, w, rows)
  sig = jax.nn.sigmoid(z)
  gz = g_ref[0].astype(jnp.float32) * (sig * (1.0 + z * (1.0 - sig)))
  # dx[t] = sum_k w[k] * gz[t + W - 1 - k]: an anti-causal convolution.
  dx = gz * w[width - 1 : width]
  for k in range(width - 1):
    dx = dx + _shift_up(gz, width - 1 - k, rows) * w[k : k + 1]
  dx_ref[0] = dx.astype(dx_ref.dtype)
  # dw[k] = sum_t x[t - (W - 1 - k)] * gz[t], accumulated over the batch.
  dw = jnp.concatenate(
      [
          jnp.sum(
              _shift_down(x, width - 1 - k, rows) * gz, axis=0, keepdims=True
          )
          for k in range(width)
      ],
      axis=0,
  )

  @pl.when(b == 0)
  def _():
    dw_ref[...] = jnp.zeros_like(dw_ref)

  dw_ref[...] += dw


def _specs(batch: int, seq_len: int, channels: int, width: int):
  """BlockSpecs and grid ``(channels // 128, batch)``.

  ``batch`` is the innermost grid axis so ``dw_ref``'s block index ``(0, c)``
  stays constant across ``b = 0 .. batch - 1`` in backward, keeping ``dw_ref``
  resident in VMEM across the batch before a single HBM writeback.
  """
  strip = pl.BlockSpec((1, seq_len, _BLOCK_C), lambda c, b: (b, 0, c))
  taps = pl.BlockSpec((width, _BLOCK_C), lambda c, b: (0, c))
  grid = (channels // _BLOCK_C, batch)
  return strip, taps, grid


def causal_conv_silu_fwd_jax(x: jax.Array, w_WC: jax.Array) -> jax.Array:
  """``silu(causal_conv(x))`` for ``x`` ``[B, S, C]``, fp32 taps ``[W, C]``."""
  batch, seq_len, channels = x.shape
  width = w_WC.shape[0]
  strip, taps, grid = _specs(batch, seq_len, channels, width)
  return pl.pallas_call(
      _fwd_kernel,
      grid=grid,
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
      grid=grid,
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
  ctx.set_materialize_grads(False)
  x, w_WC = inputs
  ctx.save_for_backward(x, w_WC)


def _backward(ctx, grad_out):
  need_dx, need_dw = ctx.needs_input_grad
  if grad_out is None or not (need_dx or need_dw):
    return None, None
  x, w_WC = ctx.saved_tensors
  dx, dw_WC = _bwd_op(x, w_WC, grad_out.contiguous())
  return dx if need_dx else None, dw_WC if need_dw else None


_fwd_op.register_autograd(_backward, setup_context=_setup_context)


def supports(x_BSC: torch.Tensor, weight_C1W: torch.Tensor) -> bool:
  """Whether ``x_BSC`` and ``weight_C1W`` satisfy the Pallas kernel limits."""
  if isinstance(x_BSC, DTensor) or isinstance(weight_C1W, DTensor):
    return False
  if x_BSC.device.type != "tpu" or weight_C1W.device != x_BSC.device:
    return False
  if (
      x_BSC.dtype not in _SUPPORTED_DTYPES
      or weight_C1W.dtype not in _SUPPORTED_DTYPES
  ):
    return False
  if x_BSC.ndim != 3 or weight_C1W.ndim != 3:
    return False
  batch, seq_len, channels = x_BSC.shape
  width = weight_C1W.shape[2]
  max_s = _max_supported_seq_len(x_BSC.dtype)
  return (
      batch > 0
      and 0 < seq_len <= max_s
      and seq_len % 8 == 0
      and channels > 0
      and channels % _BLOCK_C == 0
      and weight_C1W.shape == (channels, 1, width)
      and 1 <= width <= min(seq_len, _MAX_WIDTH)
  )


def causal_conv_silu(
    x_BSC: torch.Tensor, weight_C1W: torch.Tensor
) -> torch.Tensor:
  """``silu`` of a depthwise causal ``conv1d`` over whole ``[B, S, C]`` rows.

  Args:
    x_BSC: ``[B, S, C]`` TPU tensor with ``C % 128 == 0`` and ``S % 8 == 0``.
    weight_C1W: ``[C, 1, W]`` depthwise ``Conv1d`` weight. The taps run in fp32,
      as in ``F.conv1d``'s accumulation.

  Returns:
    ``[B, S, C]`` output in ``x_BSC.dtype``.
  """
  if not supports(x_BSC, weight_C1W):
    raise ValueError(
        "Unsupported inputs for Pallas causal_conv_silu:"
        f" x={tuple(x_BSC.shape)} ({x_BSC.dtype}, {x_BSC.device}),"
        f" weight={tuple(weight_C1W.shape)} ({weight_C1W.dtype},"
        f" {weight_C1W.device})"
    )
  w_WC = weight_C1W[:, 0, :].t().to(torch.float32).contiguous()
  return _fwd_op(x_BSC.contiguous(), w_WC)
