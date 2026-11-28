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

import math
import os

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch
from torch.distributed.tensor import DTensor
from torch_tpu._internal import pallas
from torchtitan.experiments.tpu.kernels import pallas_vmem

# Channels per kernel block (one lane tile). At S=8192 on TPU7x, one [8192, 128]
# fp32 strip is 4 MiB, so keeping S bounded by TPU generation and dtype keeps
# the backward's live strips inside scoped VMEM; longer sequences are tiled into
# <= max_s strips with an 8-row VREG-aligned halo.
_BLOCK_C = 128
_HALO = 8
_MIN_SEQ_TILE = 512
_MAX_WIDTH = 8
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


def _vmem_limit_bytes() -> int | None:
  """Scoped VMEM limit for both kernels: 90% of the core's VMEM.

  Set explicitly so the kernels do not depend on the caller's
  ``xla_tpu_scoped_vmem_limit_kib`` (the backward keeps several fp32
  ``[seq_len, 128]`` strips live). Returns ``None`` when TPU device info is
  unavailable (e.g. CPU shape evaluation).
  """
  try:
    return int(0.9 * pltpu.get_tpu_info().vmem_capacity_bytes)
  except (AttributeError, RuntimeError, ValueError):
    return None


def _max_supported_seq_len(dtype: torch.dtype, width: int = 4) -> int:
  """Maximum whole-strip ``seq_len`` the kernels support (conservative).

  The bounds were measured against the default scoped VMEM (16 MiB on v5/v6,
  >= 32 MiB on TPU7x); the kernels request ``_vmem_limit_bytes()``, which is
  larger, so the bounds keep headroom.
  """
  gen = pallas_vmem.tpu_generation()
  if gen is not None and gen >= 7:
    # TPU7x has >= 32 MiB default scoped VMEM (63.94 MiB in 35B recipes);
    # wider kernels (> 4 taps) keep more rolled strips live in backward.
    base = 8192 if dtype == torch.bfloat16 else 4096
  else:
    # TPU v5/v6 default to 16 MiB scoped VMEM.
    # TODO: Benchmark and raise the v5/v6 sequence-length caps now
    # that _vmem_limit_bytes() requests 90% of physical VMEM.
    base = 2048 if dtype == torch.bfloat16 else 1024
  return base // 2 if width > 4 else base


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
  # `[S, 1]` row ids broadcast across lanes; a full `[S, 128]` int32 iota
  # would be one more live strip.
  rows = jax.lax.broadcasted_iota(jnp.int32, (x.shape[0], 1), 0)
  z = _conv(x, w_ref[...], rows)
  o_ref[0] = (z * jax.nn.sigmoid(z)).astype(o_ref.dtype)


def _bwd_kernel(x_ref, w_ref, g_ref, dx_ref, dw_ref):
  b = pl.program_id(1)
  w = w_ref[...]
  width = w.shape[0]
  x = x_ref[0].astype(jnp.float32)
  # `[S, 1]` row ids broadcast across lanes; a full `[S, 128]` int32 iota
  # would be one more live strip.
  rows = jax.lax.broadcasted_iota(jnp.int32, (x.shape[0], 1), 0)
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


def _select_seq_tile(seq_len: int, max_s: int) -> int | None:
  """Returns the sequence tile length ``<= max_s`` dividing ``seq_len``, or ``None``."""
  if seq_len <= 0 or seq_len % _HALO != 0:
    return None
  if seq_len <= max_s:
    return seq_len
  if os.getenv("TT_DISABLE_TILED_CONV1D") == "1":
    return None
  tile_s = math.gcd(seq_len, max_s)
  if tile_s < _MIN_SEQ_TILE or tile_s % _HALO != 0:
    return None
  return tile_s


def _shift_down_halo(
    x: jax.Array,
    x_prev8: jax.Array,
    shift: int,
    rows8: jax.Array,
    prev_scale: jax.Array | float,
) -> jax.Array:
  """``y[r] = x[r - shift]`` using 8-row predecessor halo ``x_prev8``."""
  if shift == 0:
    return x
  rolled = pltpu.roll(x, shift=shift, axis=0)
  prev_rolled = pltpu.roll(x_prev8, shift=shift, axis=0) * prev_scale
  head8 = jnp.where(rows8 < shift, prev_rolled, rolled[:_HALO])
  if x.shape[0] == _HALO:
    return head8
  return jnp.concatenate([head8, rolled[_HALO:]], axis=0)


def _shift_up_halo(
    gz: jax.Array,
    gz_next8: jax.Array,
    shift: int,
    rows8: jax.Array,
    next_scale: jax.Array | float,
) -> jax.Array:
  """``y[r] = gz[r + shift]`` using 8-row successor halo ``gz_next8``."""
  if shift == 0:
    return gz
  tile_s = gz.shape[0]
  rolled = pltpu.roll(gz, shift=tile_s - shift, axis=0)
  next_rolled = pltpu.roll(gz_next8, shift=_HALO - shift, axis=0) * next_scale
  tail8 = jnp.where(
      rows8 >= _HALO - shift, next_rolled, rolled[tile_s - _HALO :]
  )
  if tile_s == _HALO:
    return tail8
  return jnp.concatenate([rolled[: tile_s - _HALO], tail8], axis=0)


def _conv_halo(
    x: jax.Array,
    x_prev8: jax.Array,
    w: jax.Array,
    rows8: jax.Array,
    prev_scale: jax.Array | float,
) -> jax.Array:
  """Causal depthwise conv of ``x`` with 8-row predecessor halo ``x_prev8``."""
  width = w.shape[0]
  acc = x * w[width - 1 : width]
  for k in range(width - 1):
    shifted = _shift_down_halo(x, x_prev8, width - 1 - k, rows8, prev_scale)
    acc = acc + shifted * w[k : k + 1]
  return acc


def _fwd_tiled_kernel(x_prev_ref, x_ref, w_ref, o_ref):
  t = pl.program_id(2)
  x = x_ref[0].astype(jnp.float32)
  x_prev8 = x_prev_ref[0].astype(jnp.float32)
  rows8 = jax.lax.broadcasted_iota(jnp.int32, (_HALO, 1), 0)
  prev_scale = jnp.where(t > 0, jnp.float32(1.0), jnp.float32(0.0))
  z = _conv_halo(x, x_prev8, w_ref[...], rows8, prev_scale)
  o_ref[0] = (z * jax.nn.sigmoid(z)).astype(o_ref.dtype)


def _bwd_tiled_kernel(
    x_prev_ref, x_ref, x_next_ref, w_ref, g_ref, g_next_ref, dx_ref, dw_ref
):
  b = pl.program_id(1)
  t = pl.program_id(2)
  num_tiles = pl.num_programs(2)
  w = w_ref[...]
  width = w.shape[0]
  x = x_ref[0].astype(jnp.float32)
  x_prev8 = x_prev_ref[0].astype(jnp.float32)
  x_next8 = x_next_ref[0].astype(jnp.float32)
  rows8 = jax.lax.broadcasted_iota(jnp.int32, (_HALO, 1), 0)
  prev_scale = jnp.where(t > 0, jnp.float32(1.0), jnp.float32(0.0))
  next_scale = jnp.where(
      t < (num_tiles - 1), jnp.float32(1.0), jnp.float32(0.0)
  )

  z = _conv_halo(x, x_prev8, w, rows8, prev_scale)
  sig = jax.nn.sigmoid(z)
  gz = g_ref[0].astype(jnp.float32) * (sig * (1.0 + z * (1.0 - sig)))

  # Compute gz on the 8-row successor halo using x's last 8 rows as predecessor.
  tile_s = x.shape[0]
  z_next8 = _conv_halo(x_next8, x[tile_s - _HALO :], w, rows8, 1.0)
  sig_next8 = jax.nn.sigmoid(z_next8)
  gz_next8 = g_next_ref[0].astype(jnp.float32) * (
      sig_next8 * (1.0 + z_next8 * (1.0 - sig_next8))
  )

  dx = gz * w[width - 1 : width]
  for k in range(width - 1):
    dx = (
        dx
        + _shift_up_halo(gz, gz_next8, width - 1 - k, rows8, next_scale)
        * w[k : k + 1]
    )
  dx_ref[0] = dx.astype(dx_ref.dtype)

  dw = jnp.concatenate(
      [
          jnp.sum(
              _shift_down_halo(x, x_prev8, width - 1 - k, rows8, prev_scale)
              * gz,
              axis=0,
              keepdims=True,
          )
          for k in range(width)
      ],
      axis=0,
  )

  @pl.when((b == 0) & (t == 0))
  def _():
    dw_ref[...] = jnp.zeros_like(dw_ref)

  dw_ref[...] += dw


def _specs(batch: int, seq_len: int, channels: int, width: int):
  """BlockSpecs and grid ``(channels // 128, batch)``."""
  strip = pl.BlockSpec((1, seq_len, _BLOCK_C), lambda c, b: (b, 0, c))
  taps = pl.BlockSpec((width, _BLOCK_C), lambda c, b: (0, c))
  grid = (channels // _BLOCK_C, batch)
  return strip, taps, grid


def _tiled_specs(
    batch: int, seq_len: int, tile_s: int, channels: int, width: int
):
  """BlockSpecs and grid ``(channels // 128, batch, seq_len // tile_s)``."""
  num_tiles = seq_len // tile_s
  blocks8_per_tile = tile_s // _HALO
  total_blocks8 = seq_len // _HALO
  strip = pl.BlockSpec((1, tile_s, _BLOCK_C), lambda c, b, t: (b, t, c))
  prev_halo = pl.BlockSpec(
      (1, _HALO, _BLOCK_C),
      lambda c, b, t: (b, jnp.maximum(t * blocks8_per_tile - 1, 0), c),
  )
  next_halo = pl.BlockSpec(
      (1, _HALO, _BLOCK_C),
      lambda c, b, t: (
          b,
          jnp.minimum((t + 1) * blocks8_per_tile, total_blocks8 - 1),
          c,
      ),
  )
  taps = pl.BlockSpec((width, _BLOCK_C), lambda c, b, t: (0, c))
  grid = (channels // _BLOCK_C, batch, num_tiles)
  return prev_halo, strip, next_halo, taps, grid


def _jax_max_seq_len(dtype: jnp.dtype, width: int) -> int:
  torch_dtype = (
      torch.bfloat16 if jnp.dtype(dtype) == jnp.bfloat16 else torch.float32
  )
  return _max_supported_seq_len(torch_dtype, width)


def causal_conv_silu_fwd_jax(x: jax.Array, w_WC: jax.Array) -> jax.Array:
  """``silu(causal_conv(x))`` for ``x`` ``[B, S, C]``, fp32 taps ``[W, C]``."""
  batch, seq_len, channels = x.shape
  width = w_WC.shape[0]
  max_s = _jax_max_seq_len(x.dtype, width)
  if seq_len <= max_s:
    strip, taps, grid = _specs(batch, seq_len, channels, width)
    return pl.pallas_call(
        _fwd_kernel,
        grid=grid,
        in_specs=[strip, taps],
        out_specs=strip,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=_vmem_limit_bytes(),
        ),
        name="causal_conv1d_silu_fwd",
    )(x, w_WC)
  tile_s = _select_seq_tile(seq_len, max_s)
  assert tile_s is not None
  prev_halo, strip, _, taps, grid = _tiled_specs(
      batch, seq_len, tile_s, channels, width
  )
  return pl.pallas_call(
      _fwd_tiled_kernel,
      grid=grid,
      in_specs=[prev_halo, strip, taps],
      out_specs=strip,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel"),
          vmem_limit_bytes=_vmem_limit_bytes(),
      ),
      name="causal_conv1d_silu_tiled_fwd",
  )(x, x, w_WC)


def causal_conv_silu_bwd_jax(
    x: jax.Array, w_WC: jax.Array, g: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Returns ``(dx, dw_WC)`` for the output gradient ``g`` ``[B, S, C]``."""
  batch, seq_len, channels = x.shape
  width = w_WC.shape[0]
  max_s = _jax_max_seq_len(x.dtype, width)
  if seq_len <= max_s:
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
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=_vmem_limit_bytes(),
        ),
        name="causal_conv1d_silu_bwd",
    )(x, w_WC, g)
  tile_s = _select_seq_tile(seq_len, max_s)
  assert tile_s is not None
  prev_halo, strip, next_halo, taps, grid = _tiled_specs(
      batch, seq_len, tile_s, channels, width
  )
  return pl.pallas_call(
      _bwd_tiled_kernel,
      grid=grid,
      in_specs=[prev_halo, strip, next_halo, taps, strip, next_halo],
      out_specs=[strip, taps],
      out_shape=[
          jax.ShapeDtypeStruct(x.shape, x.dtype),
          jax.ShapeDtypeStruct(w_WC.shape, jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          vmem_limit_bytes=_vmem_limit_bytes(),
      ),
      name="causal_conv1d_silu_tiled_bwd",
  )(x, x, x, w_WC, g, g)


def _shift_down_halo_int(
    s: jax.Array,
    s_prev8: jax.Array,
    shift: int,
    rows8: jax.Array,
    has_prev: jax.Array | bool,
) -> jax.Array:
  """``y[r] = s[r - shift]`` for int32 segment ids, ``-1`` before row start."""
  if shift == 0:
    return s
  rolled = pltpu.roll(s, shift=shift, axis=0)
  prev_rolled = jnp.where(
      has_prev, pltpu.roll(s_prev8, shift=shift, axis=0), jnp.int32(-1)
  )
  head8 = jnp.where(rows8 < shift, prev_rolled, rolled[:_HALO])
  if s.shape[0] == _HALO:
    return head8
  return jnp.concatenate([head8, rolled[_HALO:]], axis=0)


def _shift_up_halo_int(
    s: jax.Array,
    s_next8: jax.Array,
    shift: int,
    rows8: jax.Array,
    has_next: jax.Array | bool,
) -> jax.Array:
  """``y[r] = s[r + shift]`` for int32 segment ids, ``-1`` past row end."""
  if shift == 0:
    return s
  tile_s = s.shape[0]
  rolled = pltpu.roll(s, shift=tile_s - shift, axis=0)
  next_rolled = jnp.where(
      has_next, pltpu.roll(s_next8, shift=_HALO - shift, axis=0), jnp.int32(-1)
  )
  tail8 = jnp.where(
      rows8 >= _HALO - shift, next_rolled, rolled[tile_s - _HALO :]
  )
  if tile_s == _HALO:
    return tail8
  return jnp.concatenate([rolled[: tile_s - _HALO], tail8], axis=0)


def _conv_halo_seg(
    x: jax.Array,
    x_prev8: jax.Array,
    seg: jax.Array,
    seg_prev8: jax.Array,
    w: jax.Array,
    rows8: jax.Array,
    has_prev: jax.Array | bool,
) -> jax.Array:
  """Segmented causal depthwise conv of ``x`` with 8-row predecessor halo."""
  width = w.shape[0]
  prev_scale = jnp.where(has_prev, jnp.float32(1.0), jnp.float32(0.0))
  acc = x * w[width - 1 : width]
  for k in range(width - 1):
    shift = width - 1 - k
    shifted_x = _shift_down_halo(x, x_prev8, shift, rows8, prev_scale)
    shifted_seg = _shift_down_halo_int(seg, seg_prev8, shift, rows8, has_prev)
    acc = acc + jnp.where(shifted_seg == seg, shifted_x, 0.0) * w[k : k + 1]
  return acc


def _fwd_tiled_seg_kernel(
    x_prev_ref, x_ref, seg_prev_ref, seg_ref, w_ref, o_ref
):
  t = pl.program_id(2)
  x = x_ref[0].astype(jnp.float32)
  x_prev8 = x_prev_ref[0].astype(jnp.float32)
  seg = seg_ref[0]
  seg_prev8 = seg_prev_ref[0]
  rows8 = jax.lax.broadcasted_iota(jnp.int32, (_HALO, 1), 0)
  z = _conv_halo_seg(x, x_prev8, seg, seg_prev8, w_ref[...], rows8, t > 0)
  o_ref[0] = (z * jax.nn.sigmoid(z)).astype(o_ref.dtype)


def _bwd_tiled_seg_kernel(
    x_prev_ref,
    x_ref,
    x_next_ref,
    seg_prev_ref,
    seg_ref,
    seg_next_ref,
    w_ref,
    g_ref,
    g_next_ref,
    dx_ref,
    dw_ref,
):
  b = pl.program_id(1)
  t = pl.program_id(2)
  num_tiles = pl.num_programs(2)
  w = w_ref[...]
  width = w.shape[0]
  x = x_ref[0].astype(jnp.float32)
  x_prev8 = x_prev_ref[0].astype(jnp.float32)
  x_next8 = x_next_ref[0].astype(jnp.float32)
  seg = seg_ref[0]
  seg_prev8 = seg_prev_ref[0]
  seg_next8 = seg_next_ref[0]
  rows8 = jax.lax.broadcasted_iota(jnp.int32, (_HALO, 1), 0)
  has_prev = t > 0
  has_next = t < (num_tiles - 1)
  prev_scale = jnp.where(has_prev, jnp.float32(1.0), jnp.float32(0.0))
  next_scale = jnp.where(has_next, jnp.float32(1.0), jnp.float32(0.0))

  z = _conv_halo_seg(x, x_prev8, seg, seg_prev8, w, rows8, has_prev)
  sig = jax.nn.sigmoid(z)
  gz = g_ref[0].astype(jnp.float32) * (sig * (1.0 + z * (1.0 - sig)))

  tile_s = x.shape[0]
  z_next8 = _conv_halo_seg(
      x_next8,
      x[tile_s - _HALO :],
      seg_next8,
      seg[tile_s - _HALO :],
      w,
      rows8,
      True,
  )
  sig_next8 = jax.nn.sigmoid(z_next8)
  gz_next8 = g_next_ref[0].astype(jnp.float32) * (
      sig_next8 * (1.0 + z_next8 * (1.0 - sig_next8))
  )

  dx = gz * w[width - 1 : width]
  dw_list = []
  for k in range(width):
    shift = width - 1 - k
    if shift > 0:
      up_gz = _shift_up_halo(gz, gz_next8, shift, rows8, next_scale)
      up_seg = _shift_up_halo_int(seg, seg_next8, shift, rows8, has_next)
      dx = dx + jnp.where(up_seg == seg, up_gz, 0.0) * w[k : k + 1]
      dn_x = _shift_down_halo(x, x_prev8, shift, rows8, prev_scale)
      dn_seg = _shift_down_halo_int(seg, seg_prev8, shift, rows8, has_prev)
      dn_x = jnp.where(dn_seg == seg, dn_x, 0.0)
    else:
      dn_x = x
    dw_list.append(jnp.sum(dn_x * gz, axis=0, keepdims=True))
  dx_ref[0] = dx.astype(dx_ref.dtype)
  dw = jnp.concatenate(dw_list, axis=0)

  @pl.when((b == 0) & (t == 0))
  def _():
    dw_ref[...] = jnp.zeros_like(dw_ref)

  dw_ref[...] += dw


def _seg_tiled_specs(
    batch: int, seq_len: int, tile_s: int, channels: int, width: int
):
  num_tiles = seq_len // tile_s
  blocks8_per_tile = tile_s // _HALO
  total_blocks8 = seq_len // _HALO
  strip = pl.BlockSpec((1, tile_s, _BLOCK_C), lambda c, b, t: (b, t, c))
  prev_halo = pl.BlockSpec(
      (1, _HALO, _BLOCK_C),
      lambda c, b, t: (b, jnp.maximum(t * blocks8_per_tile - 1, 0), c),
  )
  next_halo = pl.BlockSpec(
      (1, _HALO, _BLOCK_C),
      lambda c, b, t: (
          b,
          jnp.minimum((t + 1) * blocks8_per_tile, total_blocks8 - 1),
          c,
      ),
  )
  seg_strip = pl.BlockSpec((1, tile_s, _BLOCK_C), lambda c, b, t: (b, t, 0))
  seg_prev = pl.BlockSpec(
      (1, _HALO, _BLOCK_C),
      lambda c, b, t: (b, jnp.maximum(t * blocks8_per_tile - 1, 0), 0),
  )
  seg_next = pl.BlockSpec(
      (1, _HALO, _BLOCK_C),
      lambda c, b, t: (
          b,
          jnp.minimum((t + 1) * blocks8_per_tile, total_blocks8 - 1),
          0,
      ),
  )
  taps = pl.BlockSpec((width, _BLOCK_C), lambda c, b, t: (0, c))
  grid = (channels // _BLOCK_C, batch, num_tiles)
  return prev_halo, strip, next_halo, seg_prev, seg_strip, seg_next, taps, grid


def causal_conv_silu_seg_fwd_jax(
    x: jax.Array, w_WC: jax.Array, seg_BS: jax.Array
) -> jax.Array:
  batch, seq_len, channels = x.shape
  width = w_WC.shape[0]
  max_s = min(_jax_max_seq_len(x.dtype, width), 4096)
  tile_s = _select_seq_tile(seq_len, max_s)
  if tile_s is None:
    tile_s = seq_len
  seg_3d = jnp.broadcast_to(
      seg_BS.astype(jnp.int32)[:, :, None], (batch, seq_len, _BLOCK_C)
  )
  prev_halo, strip, _, seg_prev, seg_strip, _, taps, grid = _seg_tiled_specs(
      batch, seq_len, tile_s, channels, width
  )
  return pl.pallas_call(
      _fwd_tiled_seg_kernel,
      grid=grid,
      in_specs=[prev_halo, strip, seg_prev, seg_strip, taps],
      out_specs=strip,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel"),
          vmem_limit_bytes=_vmem_limit_bytes(),
      ),
      name="causal_conv1d_silu_seg_fwd",
  )(x, x, seg_3d, seg_3d, w_WC)


def causal_conv_silu_seg_bwd_jax(
    x: jax.Array, w_WC: jax.Array, seg_BS: jax.Array, g: jax.Array
) -> tuple[jax.Array, jax.Array]:
  batch, seq_len, channels = x.shape
  width = w_WC.shape[0]
  max_s = min(_jax_max_seq_len(x.dtype, width), 4096)
  tile_s = _select_seq_tile(seq_len, max_s)
  if tile_s is None:
    tile_s = seq_len
  seg_3d = jnp.broadcast_to(
      seg_BS.astype(jnp.int32)[:, :, None], (batch, seq_len, _BLOCK_C)
  )
  prev_halo, strip, next_halo, seg_prev, seg_strip, seg_next, taps, grid = (
      _seg_tiled_specs(batch, seq_len, tile_s, channels, width)
  )
  return pl.pallas_call(
      _bwd_tiled_seg_kernel,
      grid=grid,
      in_specs=[
          prev_halo,
          strip,
          next_halo,
          seg_prev,
          seg_strip,
          seg_next,
          taps,
          strip,
          next_halo,
      ],
      out_specs=[strip, taps],
      out_shape=[
          jax.ShapeDtypeStruct(x.shape, x.dtype),
          jax.ShapeDtypeStruct(w_WC.shape, jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          vmem_limit_bytes=_vmem_limit_bytes(),
      ),
      name="causal_conv1d_silu_seg_bwd",
  )(x, x, x, seg_3d, seg_3d, seg_3d, w_WC, g, g)


@pallas.jax_op("tt_qwen35::causal_conv1d_silu_fwd")
def _fwd_op(x: jax.Array, w_WC: jax.Array) -> jax.Array:
  return causal_conv_silu_fwd_jax(x, w_WC)


@pallas.jax_op("tt_qwen35::causal_conv1d_silu_bwd")
def _bwd_op(
    x: jax.Array, w_WC: jax.Array, g: jax.Array
) -> tuple[jax.Array, jax.Array]:
  return causal_conv_silu_bwd_jax(x, w_WC, g)


@pallas.jax_op("tt_qwen35::causal_conv1d_silu_seg_fwd")
def _fwd_seg_op(x: jax.Array, w_WC: jax.Array, seg_BS: jax.Array) -> jax.Array:
  return causal_conv_silu_seg_fwd_jax(x, w_WC, seg_BS)


@pallas.jax_op("tt_qwen35::causal_conv1d_silu_seg_bwd")
def _bwd_seg_op(
    x: jax.Array, w_WC: jax.Array, seg_BS: jax.Array, g: jax.Array
) -> tuple[jax.Array, jax.Array]:
  return causal_conv_silu_seg_bwd_jax(x, w_WC, seg_BS, g)


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


def _setup_seg_context(ctx, inputs, output):
  del output
  ctx.set_materialize_grads(False)
  x, w_WC, seg_BS = inputs
  ctx.save_for_backward(x, w_WC, seg_BS)


def _backward_seg(ctx, grad_out):
  need_dx, need_dw, _ = ctx.needs_input_grad
  if grad_out is None or not (need_dx or need_dw):
    return None, None, None
  x, w_WC, seg_BS = ctx.saved_tensors
  dx, dw_WC = _bwd_seg_op(x, w_WC, seg_BS, grad_out.contiguous())
  return dx if need_dx else None, dw_WC if need_dw else None, None


_fwd_seg_op.register_autograd(_backward_seg, setup_context=_setup_seg_context)


def unsupported_reason(
    x_BSC: torch.Tensor,
    weight_C1W: torch.Tensor,
    segment_ids_BS: torch.Tensor | None = None,
) -> str | None:
  """Reason ``x_BSC`` and ``weight_C1W`` cannot use Pallas, or ``None``."""
  if (
      isinstance(x_BSC, DTensor)
      or isinstance(weight_C1W, DTensor)
      or isinstance(segment_ids_BS, DTensor)
  ):
    return "DTensor inputs are not supported"
  if x_BSC.device.type != "tpu" or weight_C1W.device != x_BSC.device:
    return (
        f"inputs must be on the same TPU device, got {x_BSC.device} and"
        f" {weight_C1W.device}"
    )
  if segment_ids_BS is not None:
    if os.getenv("TT_DISABLE_SEG_CONV1D") == "1":
      return (
          "segmented Pallas causal_conv1d disabled by TT_DISABLE_SEG_CONV1D=1"
      )
    if segment_ids_BS.device != x_BSC.device:
      return "segment_ids_BS must be on the same TPU device as x_BSC"
    if segment_ids_BS.shape != x_BSC.shape[:2]:
      return (
          f"segment_ids_BS shape {tuple(segment_ids_BS.shape)} must match"
          f" {tuple(x_BSC.shape[:2])}"
      )
    if segment_ids_BS.dtype not in (torch.int32, torch.int64):
      return f"unsupported segment_ids_BS dtype {segment_ids_BS.dtype}"
  if (
      x_BSC.dtype not in _SUPPORTED_DTYPES
      or weight_C1W.dtype not in _SUPPORTED_DTYPES
  ):
    return (
        f"unsupported dtype x={x_BSC.dtype}, weight={weight_C1W.dtype}"
        f" (supported: {_SUPPORTED_DTYPES})"
    )
  if x_BSC.ndim != 3 or weight_C1W.ndim != 3:
    return (
        f"expected 3D x_BSC and weight_C1W, got {x_BSC.ndim}D and"
        f" {weight_C1W.ndim}D"
    )
  batch, seq_len, channels = x_BSC.shape
  width = weight_C1W.shape[2]
  max_s = _max_supported_seq_len(x_BSC.dtype, width)
  if segment_ids_BS is not None:
    max_s = min(max_s, 4096)
  if batch <= 0:
    return f"batch size must be positive, got {batch}"
  if _select_seq_tile(seq_len, max_s) is None:
    return (
        f"seq_len={seq_len} must be a positive multiple of {_HALO} and either"
        f" <= {max_s} or have gcd(seq_len, {max_s}) >= {_MIN_SEQ_TILE} for"
        f" dtype={x_BSC.dtype}, width={width}"
    )
  if channels <= 0 or channels % _BLOCK_C != 0:
    return f"channels={channels} must be a positive multiple of {_BLOCK_C}"
  if weight_C1W.shape != (channels, 1, width):
    return (
        f"weight shape {tuple(weight_C1W.shape)} must match"
        f" (channels={channels}, 1, width={width})"
    )
  if not 1 <= width <= min(seq_len, _MAX_WIDTH):
    return (
        f"kernel width={width} must be in [1, min(seq_len={seq_len},"
        f" {_MAX_WIDTH})]"
    )
  return None


def supports(
    x_BSC: torch.Tensor,
    weight_C1W: torch.Tensor,
    segment_ids_BS: torch.Tensor | None = None,
) -> bool:
  """Whether ``x_BSC`` and ``weight_C1W`` satisfy the Pallas kernel limits."""
  return unsupported_reason(x_BSC, weight_C1W, segment_ids_BS) is None


def causal_conv_silu(
    x_BSC: torch.Tensor,
    weight_C1W: torch.Tensor,
    segment_ids_BS: torch.Tensor | None = None,
) -> torch.Tensor:
  """``silu`` of a depthwise causal ``conv1d`` over ``[B, S, C]`` rows."""
  reason = unsupported_reason(x_BSC, weight_C1W, segment_ids_BS)
  if reason is not None:
    raise ValueError(
        f"Unsupported inputs for Pallas causal_conv_silu ({reason}):"
        f" x={tuple(x_BSC.shape)} ({x_BSC.dtype}, {x_BSC.device}),"
        f" weight={tuple(weight_C1W.shape)} ({weight_C1W.dtype},"
        f" {weight_C1W.device})"
    )
  w_WC = weight_C1W[:, 0, :].t().to(torch.float32).contiguous()
  if segment_ids_BS is not None:
    seg_i32 = segment_ids_BS.to(torch.int32).contiguous()
    return _fwd_seg_op(x_BSC.contiguous(), w_WC, seg_i32)
  return _fwd_op(x_BSC.contiguous(), w_WC)
