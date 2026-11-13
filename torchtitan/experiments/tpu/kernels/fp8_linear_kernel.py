"""Pure-JAX Pallas TPU kernels for fused FP8 linear forward and backward."""

from __future__ import annotations

import dataclasses
import functools
import itertools
from typing import Literal, overload

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_CORE_AXIS_NAME = "core"

FP8_DTYPE = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8_DTYPE).max)
_QUANT_EPS = 1e-34
BLOCK_ALIGNMENT = 128
SX_LANES = 128

VMEM_LIMIT_BYTES = 64 * 1024 * 1024
VMEM_BUDGET_BYTES = 52 * 1024 * 1024
DLHS_VMEM_BUDGET_BYTES = 62 * 1024 * 1024

MAX_DRHS_OUT_TILE_ELEMS = 2_200_000
MAX_DRHS_COTANGENT_BLOCK_ELEMS = 3_500_000
MAX_DRHS_MEASURED_COTANGENT_ELEMS = 4096 * 1024
MAX_DRHS_ACTIVATION_BLOCK_ELEMS = 4 * 1024 * 1024
_MIN_DRHS_OUT_TILE_ELEMS = 1_000_000
_DEAD_SCRATCH_SHAPE = (8, 128)

_MEASURED_DRHS_TILES = (
    (4096, 1024, 1024),
    (4096, 1792, 512),
    (2048, 1024, 1024),
    (2048, 1792, 512),
    (1024, 1024, 2048),
    (1024, 1024, 1536),
    (1024, 512, 3072),
)

Residuals = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


@dataclasses.dataclass(frozen=True)
class TileConfig:
  """Block sizes `(block_m, block_k, block_n)`, each a multiple of `BLOCK_ALIGNMENT`."""

  block_m: int
  block_k: int
  block_n: int

  def __post_init__(self):
    for name in ("block_m", "block_k", "block_n"):
      value = getattr(self, name)
      if value < BLOCK_ALIGNMENT or value % BLOCK_ALIGNMENT != 0:
        raise ValueError(
            f"{name} must be a positive multiple of {BLOCK_ALIGNMENT}, got"
            f" {value}."
        )


def _tensorcore_mesh() -> pltpu.TensorCoreMesh:
  if jax.__version_info__ < (0, 11, 0):
    return pltpu.create_tensorcore_mesh(axis_name=_CORE_AXIS_NAME)
  return pltpu.TensorCoreMesh(axis_name=_CORE_AXIS_NAME)


def quantize_rows_fp8(
    x: jax.Array, eps: float = _QUANT_EPS
) -> tuple[jax.Array, jax.Array]:
  """Quantizes `x` to `FP8_DTYPE` with float32 per-row absmax scale `(*x.shape[:-1], 1)`."""
  max_abs = jnp.maximum(
      jnp.max(jnp.abs(x), axis=-1, keepdims=True).astype(jnp.float32), eps
  )
  scale = max_abs / FP8_MAX
  inv = (FP8_MAX / max_abs).astype(x.dtype)
  return (x * inv).astype(FP8_DTYPE), scale


def quantize_cols_fp8(
    g: jax.Array, eps: float = _QUANT_EPS
) -> tuple[jax.Array, jax.Array]:
  """Quantizes `g` to `FP8_DTYPE` with float32 per-column absmax scale `g.shape[1:]`."""
  max_abs = jnp.maximum(jnp.max(jnp.abs(g), axis=0).astype(jnp.float32), eps)
  scale = max_abs / FP8_MAX
  inv = (FP8_MAX / max_abs).astype(g.dtype)
  return (g * inv).astype(FP8_DTYPE), scale


def fp8_matmul_reference(
    lhs: jax.Array, rhs: jax.Array
) -> tuple[jax.Array, Residuals]:
  """Plain-XLA reference for `(m, k) @ (k, n)` returning `(out, residuals)`."""
  lhs_q, lhs_scale = quantize_rows_fp8(lhs)
  rhs_q, rhs_scale = quantize_cols_fp8(rhs)
  acc = jnp.matmul(lhs_q, rhs_q, preferred_element_type=jnp.float32)
  out = (acc * (lhs_scale * rhs_scale)).astype(lhs.dtype)
  return out, (lhs_q, lhs_scale, rhs_q, rhs_scale)


def fp8_dlhs_reference(
    dout: jax.Array, rhs_q: jax.Array, rhs_scale: jax.Array
) -> jax.Array:
  """Plain-XLA reference for `dlhs = dout @ rhs^T` of shape `(m, k)`."""
  dout_q, dout_scale = quantize_rows_fp8(dout * rhs_scale.astype(dout.dtype))
  acc = jax.lax.dot_general(
      dout_q,
      rhs_q,
      dimension_numbers=(((1,), (1,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  return (acc * dout_scale).astype(dout.dtype)


def fp8_drhs_reference(
    lhs_q: jax.Array, lhs_scale: jax.Array, dout: jax.Array
) -> jax.Array:
  """Plain-XLA reference for `drhs = lhs^T @ dout` of shape `(k, n)`."""
  dout_q, dout_scale = quantize_cols_fp8(dout * lhs_scale.astype(dout.dtype))
  acc = jax.lax.dot_general(
      lhs_q,
      dout_q,
      dimension_numbers=(((0,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  return (acc * dout_scale).astype(dout.dtype)


def _divisors(dim: int) -> list[int]:
  """Multiples of `BLOCK_ALIGNMENT` that divide `dim`."""
  return [
      block
      for block in range(BLOCK_ALIGNMENT, dim + 1, BLOCK_ALIGNMENT)
      if dim % block == 0
  ]


def pick_tile(dim: int, candidates: tuple[int, ...]) -> int:
  """Returns the first aligned candidate dividing `dim`, else a power-of-two fallback."""
  for c in candidates:
    if (
        BLOCK_ALIGNMENT <= c <= dim
        and c % BLOCK_ALIGNMENT == 0
        and dim % c == 0
    ):
      return c
  c = BLOCK_ALIGNMENT
  while c * 2 <= dim and dim % (c * 2) == 0:
    c *= 2
  return c


def tile_candidates(dim: int, candidates: tuple[int, ...]) -> tuple[int, ...]:
  """Returns aligned candidates dividing `dim` in preference order."""
  seen: list[int] = []
  for c in candidates:
    if (
        BLOCK_ALIGNMENT <= c <= dim
        and c % BLOCK_ALIGNMENT == 0
        and dim % c == 0
        and c not in seen
    ):
      seen.append(c)
  return tuple(seen) or (pick_tile(dim, candidates),)


def forward_vmem_bytes(
    bm: int, bk: int, bn: int, k: int, n: int, *, lhs_itemsize: int
) -> int:
  """Estimates scoped VMEM footprint in bytes for the forward kernel."""
  grid_k = -(-k // bk)
  grid_n = -(-n // bn)
  w_buffers = 1 if (grid_k == 1 and grid_n == 1) else 2
  x_block = bm * bk * lhs_itemsize * 2
  w_block = bk * bn * w_buffers
  o_block = bm * bn * lhs_itemsize * 2
  xq_cache = grid_k * bm * bk
  sx_cache = grid_k * bm * 4
  f32_tmp = bm * bn * 4 * (2 if grid_k > 1 else 1)
  return x_block + w_block + o_block + xq_cache + sx_cache + f32_tmp


def fit_fwd_tiles(
    m: int, k: int, n: int, *, lhs_itemsize: int
) -> tuple[int, int, int]:
  """Selects `(bm, bk, bn)` within `VMEM_BUDGET_BYTES` for the forward kernel."""
  bks = tile_candidates(k, (k, 1024, 512, 256))
  preference = (
      (1024, n),
      (512, n),
      (256, n),
      (1024, 1024),
      (1024, 512),
      (512, 1024),
      (512, 512),
      (512, 256),
      (256, 512),
      (256, 256),
  )
  for bk in bks:
    for bm, bn in preference:
      if (
          bm > m
          or m % bm
          or bn > n
          or n % bn
          or bn < BLOCK_ALIGNMENT
          or bn % BLOCK_ALIGNMENT
      ):
        continue
      if (
          forward_vmem_bytes(bm, bk, bn, k, n, lhs_itemsize=lhs_itemsize)
          <= VMEM_BUDGET_BYTES
      ):
        return bm, bk, bn
  bm = pick_tile(m, (256, 128))
  bn = pick_tile(n, (256, 128))
  for bk in tile_candidates(k, (k, 1024, 512, 256, 128)):
    if (
        forward_vmem_bytes(bm, bk, bn, k, n, lhs_itemsize=lhs_itemsize)
        <= VMEM_BUDGET_BYTES
    ):
      return bm, bk, bn
  return bm, pick_tile(k, (256, 128)), bn


def get_fwd_tiles(
    m: int, k: int, n: int, *, lhs_dtype: jax.typing.DTypeLike = jnp.bfloat16
) -> TileConfig:
  """Returns forward `TileConfig` for `(m, k, n)`."""
  bm, bk, bn = fit_fwd_tiles(
      m, k, n, lhs_itemsize=jnp.dtype(lhs_dtype).itemsize
  )
  return TileConfig(block_m=bm, block_k=bk, block_n=bn)


def dlhs_vmem_bytes(
    bm: int, bk: int, bn: int, k: int, n: int, *, cotangent_itemsize: int
) -> int:
  """Estimates scoped VMEM footprint in bytes for the dLHS kernel."""
  grid_k = -(-k // bk)
  grid_n = -(-n // bn)
  w_buffers = 1 if (grid_k == 1 and grid_n == 1) else 2
  g_block = bm * bn * cotangent_itemsize * 2
  o_block = bm * bk * cotangent_itemsize * 2
  gq = bm * bn
  dot_out = bm * bk * 4
  w_block = bk * bn * w_buffers
  accum = bm * bk * 4 if grid_n > 1 else 0
  gq_cache = grid_n * bm * bn if grid_k > 1 else 0
  g_folded = bm * bn * cotangent_itemsize if cotangent_itemsize > 2 else 0
  return (
      g_block + o_block + gq + dot_out + w_block + accum + gq_cache + g_folded
  )


def fit_dlhs_tiles(
    m: int, k: int, n: int, *, cotangent_itemsize: int
) -> tuple[int, int, int]:
  """Selects `(bm, bk, bn)` within `DLHS_VMEM_BUDGET_BYTES` for the dLHS kernel."""
  bns = tile_candidates(n, (n, 2048, 1024, 512, 256, 128))
  bks = tile_candidates(k, (k, 1536, 1024, 512, 256, 128))
  bms = tile_candidates(m, (1024, 512, 256, 128))
  for bn, bk, bm in itertools.product(bns, bks, bms):
    if (
        dlhs_vmem_bytes(bm, bk, bn, k, n, cotangent_itemsize=cotangent_itemsize)
        <= DLHS_VMEM_BUDGET_BYTES
    ):
      return bm, bk, bn
  return bms[-1], bks[-1], bns[-1]


def get_dlhs_tiles(
    m: int, k: int, n: int, *, lhs_dtype: jax.typing.DTypeLike = jnp.bfloat16
) -> TileConfig:
  """Returns dLHS `TileConfig` for `(m, k, n)`."""
  bm, bk, bn = fit_dlhs_tiles(
      m, k, n, cotangent_itemsize=jnp.dtype(lhs_dtype).itemsize
  )
  return TileConfig(block_m=bm, block_k=bk, block_n=bn)


def fit_drhs_tiles(
    m: int, k: int, n: int, *, cotangent_itemsize: int = 2
) -> tuple[int, int, int]:
  """Selects `(bm, bk, bn)` with VMEM-bounded activation, cotangent, and output blocks."""
  g_elems = lambda bm, bn: bm * bn * cotangent_itemsize // 2

  for bm, bk, bn in _MEASURED_DRHS_TILES:
    if (
        m % bm == 0
        and k % bk == 0
        and n % bn == 0
        and bm * bk <= MAX_DRHS_ACTIVATION_BLOCK_ELEMS
        and bk * bn <= MAX_DRHS_OUT_TILE_ELEMS
        and g_elems(bm, bn) <= MAX_DRHS_MEASURED_COTANGENT_ELEMS
    ):
      return bm, bk, bn

  m_cands = ((m,) if m <= 4096 else ()) + (4096, 2048, 1024, 512, 256, 128)
  bms = tile_candidates(m, m_cands)
  bns = tuple(reversed(_divisors(n))) or (BLOCK_ALIGNMENT,)
  bks = tuple(reversed(_divisors(k))) or (BLOCK_ALIGNMENT,)
  for bm in bms:
    for bn in bns:
      if g_elems(bm, bn) > MAX_DRHS_COTANGENT_BLOCK_ELEMS:
        continue
      for bk in bks:
        if (
            bm * bk > MAX_DRHS_ACTIVATION_BLOCK_ELEMS
            or bk * bn > MAX_DRHS_OUT_TILE_ELEMS
        ):
          continue
        if bk * bn < _MIN_DRHS_OUT_TILE_ELEMS and (bk, bn) != (k, n):
          break
        return bm, bk, bn

  best = None
  for bm in _divisors(m):
    for bk in _divisors(k):
      if bm * bk > MAX_DRHS_ACTIVATION_BLOCK_ELEMS:
        break
      for bn in _divisors(n):
        if (
            bk * bn > MAX_DRHS_OUT_TILE_ELEMS
            or g_elems(bm, bn) > MAX_DRHS_COTANGENT_BLOCK_ELEMS
        ):
          break
        key = ((m // bm) * (k // bk) * (n // bn), -(bk * bn))
        if best is None or key < best[0]:
          best = (key, (bm, bk, bn))
  if best is not None:
    return best[1]
  return bms[-1], bks[-1], bns[-1]


def get_drhs_tiles(
    m: int, k: int, n: int, *, lhs_dtype: jax.typing.DTypeLike = jnp.bfloat16
) -> TileConfig:
  """Returns dRHS `TileConfig` for `(m, k, n)`."""
  bm, bk, bn = fit_drhs_tiles(
      m, k, n, cotangent_itemsize=jnp.dtype(lhs_dtype).itemsize
  )
  return TileConfig(block_m=bm, block_k=bk, block_n=bn)


def _fused_fp8_matmul_kernel(
    x_hbm,
    w_hbm,
    sw_hbm,
    o_hbm,
    *residual_and_scratch,
    bm: int,
    bk: int,
    bn: int,
    return_xq: bool,
):
  if return_xq:
    xq_hbm, sxq_hbm, accum_vmem, xq_cache, sx_cache = residual_and_scratch
  else:
    xq_hbm = sxq_hbm = None
    accum_vmem, xq_cache, sx_cache = residual_and_scratch

  m, k = x_hbm.shape
  _, n = w_hbm.shape
  grid = (pl.cdiv(m, bm), pl.cdiv(n, bn), pl.cdiv(k, bk))

  x_spec = pl.BlockSpec((bm, bk), lambda a, b, c: (a, c))
  w_spec = pl.BlockSpec((bk, bn), lambda a, b, c: (c, b))
  sw_spec = pl.BlockSpec((bn,), lambda a, b, c: (b,))
  o_spec = pl.BlockSpec((bm, bn), lambda a, b, c: (a, b))
  xq_out_spec = pl.BlockSpec((bm, bk), lambda a, b, c: (a, c))
  sxq_out_spec = pl.BlockSpec((bm, SX_LANES), lambda a, b, c: (a, 0))

  def _load_xq(
      step: pltpu.PipelineStep,
      x_vmem,
      xq_out_vmem,
      sxq_out_vmem,
  ) -> tuple[jax.Array, jax.Array]:
    nind = step.index[1]
    cind = step.index[2]

    if xq_out_vmem is not None and sxq_out_vmem is not None:
      xq_out, sxq_out = xq_out_vmem, sxq_out_vmem

      @pl.when(nind == 0)
      def _fill_out():
        xq_t, sx_t = quantize_rows_fp8(x_vmem[...])
        xq_out[...] = xq_t
        sxq_out[...] = jnp.broadcast_to(
            sx_t.astype(sxq_out.dtype), sxq_out.shape
        )
        sx_cache[cind] = sx_t

      return xq_out[...], sx_cache[cind]

    @pl.when(nind == 0)
    def _fill():
      xq_t, sx_t = quantize_rows_fp8(x_vmem[...])
      xq_cache[cind] = xq_t
      sx_cache[cind] = sx_t

    return xq_cache[cind], sx_cache[cind]

  def body_single_k(
      step: pltpu.PipelineStep, x_vmem, w_vmem, sw_vmem, o_vmem, *res_vmem
  ):
    xq_vmem, sxq_vmem = res_vmem if res_vmem else (None, None)
    xq, sx = _load_xq(step, x_vmem, xq_vmem, sxq_vmem)
    xw = jnp.matmul(xq, w_vmem[...], preferred_element_type=jnp.float32)
    o_vmem[...] = (xw * (sx * sw_vmem[...])).astype(o_vmem.dtype)

  def body_multi_k(
      step: pltpu.PipelineStep, x_vmem, w_vmem, sw_vmem, o_vmem, *res_vmem
  ):
    xq_vmem, sxq_vmem = res_vmem if res_vmem else (None, None)
    kind = step.index[2]

    @pl.when(kind == 0)
    def _init():
      accum_vmem[...] = jnp.zeros_like(accum_vmem)

    xq, sx = _load_xq(step, x_vmem, xq_vmem, sxq_vmem)
    xw = jnp.matmul(xq, w_vmem[...], preferred_element_type=jnp.float32)
    accum_vmem[...] += xw * (sx * sw_vmem[...])

    @pl.when(kind == grid[2] - 1)
    def _write():
      o_vmem[...] = accum_vmem[...].astype(o_vmem.dtype)

  single_k = grid[2] == 1
  dimension_semantics = (
      pltpu.PARALLEL,
      pltpu.PARALLEL if grid[1] == 1 else pltpu.ARBITRARY,
      pltpu.PARALLEL if single_k else pltpu.ARBITRARY,
  )

  if return_xq:
    out_specs = [o_spec, xq_out_spec, sxq_out_spec]
    out_refs = (o_hbm, xq_hbm, sxq_hbm)
  else:
    out_specs = o_spec
    out_refs = (o_hbm,)

  pltpu.emit_pipeline(
      body_single_k if single_k else body_multi_k,
      grid=grid,
      in_specs=[x_spec, w_spec, sw_spec],
      out_specs=out_specs,
      core_axis_name=_CORE_AXIS_NAME,
      dimension_semantics=dimension_semantics,
      _explicit_indices=True,
  )(x_hbm, w_hbm, sw_hbm, *out_refs)


@overload
def fused_fp8_matmul_pallas(
    x: jax.Array,
    w: jax.Array,
    sw: jax.Array,
    *,
    bm: int,
    bk: int,
    bn: int,
    return_xq: Literal[False] = ...,
) -> jax.Array:
  ...


@overload
def fused_fp8_matmul_pallas(
    x: jax.Array,
    w: jax.Array,
    sw: jax.Array,
    *,
    bm: int,
    bk: int,
    bn: int,
    return_xq: Literal[True],
) -> tuple[jax.Array, jax.Array, jax.Array]:
  ...


def fused_fp8_matmul_pallas(
    x: jax.Array,
    w: jax.Array,
    sw: jax.Array,
    *,
    bm: int,
    bk: int,
    bn: int,
    return_xq: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
  """Computes `(quantize_rows_fp8(x) @ w) * (sx * sw)` for `x: (m, k)` and FP8 `w: (k, n)`."""
  m, k = x.shape
  k_w, n = w.shape
  if k_w != k:
    raise ValueError(f"Contracting dims differ: x has k={k}, w has k={k_w}.")
  if sw.shape != (n,):
    raise ValueError(f"sw must have shape ({n},), got {sw.shape}.")
  if w.dtype != FP8_DTYPE:
    raise ValueError(f"w must be {FP8_DTYPE}, got {w.dtype}.")
  if m % bm:
    raise ValueError(f"m ({m}) must be divisible by bm ({bm}).")
  if k % bk:
    raise ValueError(f"k ({k}) must be divisible by bk ({bk}).")
  if n % bn:
    raise ValueError(f"n ({n}) must be divisible by bn ({bn}).")
  if return_xq and bk != k:
    raise ValueError(f"return_xq requires bk == k, got {bk=}.")

  grid_k = pl.cdiv(k, bk)
  if return_xq:
    out_type = (
        jax.ShapeDtypeStruct((m, n), x.dtype),
        jax.ShapeDtypeStruct((m, k), FP8_DTYPE),
        jax.ShapeDtypeStruct((m, SX_LANES), jnp.float32),
    )
  else:
    out_type = jax.ShapeDtypeStruct((m, n), x.dtype)

  scratch_types = [
      pltpu.VMEM((bm, bn) if grid_k > 1 else _DEAD_SCRATCH_SHAPE, jnp.float32),
      pltpu.VMEM(
          (1,) + _DEAD_SCRATCH_SHAPE if return_xq else (grid_k, bm, bk),
          FP8_DTYPE,
      ),
      pltpu.VMEM((grid_k, bm, 1), jnp.float32),
  ]

  kernel = functools.partial(
      _fused_fp8_matmul_kernel, bm=bm, bk=bk, bn=bn, return_xq=return_xq
  )
  outputs = pl.kernel(
      kernel,
      out_type=out_type,
      mesh=_tensorcore_mesh(),
      scratch_types=scratch_types,
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT_BYTES),
  )(x, w, sw)

  if return_xq:
    out, xq, sx_wide = outputs
    return out, xq, sx_wide[:, :1]
  return outputs


def _dlhs_kernel(
    g_hbm,
    w_hbm,
    sw_hbm,
    o_hbm,
    accum_vmem,
    gq_cache,
    sg_cache,
    *,
    bm: int,
    bk: int,
    bn: int,
):
  m, n = g_hbm.shape
  k, _ = w_hbm.shape
  grid = (pl.cdiv(m, bm), pl.cdiv(k, bk), pl.cdiv(n, bn))
  single_k = grid[1] == 1
  single_n = grid[2] == 1

  g_spec = pl.BlockSpec((bm, bn), lambda mi, ki, ni: (mi, ni))
  w_spec = pl.BlockSpec((bk, bn), lambda mi, ki, ni: (ki, ni))
  sw_spec = pl.BlockSpec((bn,), lambda mi, ki, ni: (ni,))
  o_spec = pl.BlockSpec((bm, bk), lambda mi, ki, ni: (mi, ki))

  def _quantize_g(g_vmem, sw_vmem):
    g = g_vmem[...]
    return quantize_rows_fp8(g * sw_vmem[...].astype(g.dtype))

  def _load_gq(step: pltpu.PipelineStep, g_vmem, sw_vmem):
    if single_k:
      return _quantize_g(g_vmem, sw_vmem)
    kind = step.index[1]
    nind = step.index[2]

    @pl.when(kind == 0)
    def _fill():
      gq_cache[nind], sg_cache[nind] = _quantize_g(g_vmem, sw_vmem)

    return gq_cache[nind], sg_cache[nind]

  def _partial(step: pltpu.PipelineStep, g_vmem, w_vmem, sw_vmem):
    gq, sg = _load_gq(step, g_vmem, sw_vmem)
    gw = jax.lax.dot_general(
        gq,
        w_vmem[...],
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return gw * sg

  def body_single_n(step: pltpu.PipelineStep, g_vmem, w_vmem, sw_vmem, o_vmem):
    o_vmem[...] = _partial(step, g_vmem, w_vmem, sw_vmem).astype(o_vmem.dtype)

  def body_multi_n(step: pltpu.PipelineStep, g_vmem, w_vmem, sw_vmem, o_vmem):
    nind = step.index[2]

    @pl.when(nind == 0)
    def _init():
      accum_vmem[...] = jnp.zeros_like(accum_vmem)

    accum_vmem[...] += _partial(step, g_vmem, w_vmem, sw_vmem)

    @pl.when(nind == grid[2] - 1)
    def _write():
      o_vmem[...] = accum_vmem[...].astype(o_vmem.dtype)

  pltpu.emit_pipeline(
      body_single_n if single_n else body_multi_n,
      grid=grid,
      in_specs=[g_spec, w_spec, sw_spec],
      out_specs=o_spec,
      core_axis_name=_CORE_AXIS_NAME,
      dimension_semantics=(
          pltpu.PARALLEL,
          pltpu.PARALLEL if single_k else pltpu.ARBITRARY,
          pltpu.PARALLEL if single_n else pltpu.ARBITRARY,
      ),
      _explicit_indices=True,
  )(g_hbm, w_hbm, sw_hbm, o_hbm)


def fused_fp8_dlhs_pallas(
    dout: jax.Array,
    rhs_q: jax.Array,
    rhs_scale: jax.Array,
    *,
    bm: int,
    bk: int,
    bn: int,
    out_dtype: jnp.dtype | None = None,
) -> jax.Array:
  """Computes `quantize_rows_fp8(dout * rhs_scale) @ rhs_q^T` of shape `(m, k)`."""
  m, n = dout.shape
  k, n_w = rhs_q.shape
  if n_w != n:
    raise ValueError(f"Contracting dims differ: dout has n={n}, rhs has {n_w}.")
  if rhs_scale.shape != (n,):
    raise ValueError(
        f"rhs_scale must have shape ({n},), got {rhs_scale.shape}."
    )
  if rhs_q.dtype != FP8_DTYPE:
    raise ValueError(f"rhs_q must be {FP8_DTYPE}, got {rhs_q.dtype}.")
  if m % bm:
    raise ValueError(f"m ({m}) must be divisible by bm ({bm}).")
  if k % bk:
    raise ValueError(f"k ({k}) must be divisible by bk ({bk}).")
  if n % bn:
    raise ValueError(f"n ({n}) must be divisible by bn ({bn}).")

  grid_k = pl.cdiv(k, bk)
  grid_n = pl.cdiv(n, bn)
  scratch_types = [
      pltpu.VMEM((bm, bk) if grid_n > 1 else _DEAD_SCRATCH_SHAPE, jnp.float32),
      pltpu.VMEM(
          (grid_n, bm, bn) if grid_k > 1 else (1,) + _DEAD_SCRATCH_SHAPE,
          FP8_DTYPE,
      ),
      pltpu.VMEM(
          (grid_n, bm, 1) if grid_k > 1 else (1,) + _DEAD_SCRATCH_SHAPE,
          jnp.float32,
      ),
  ]

  kernel = functools.partial(_dlhs_kernel, bm=bm, bk=bk, bn=bn)
  return pl.kernel(
      kernel,
      out_type=jax.ShapeDtypeStruct((m, k), out_dtype or dout.dtype),
      mesh=_tensorcore_mesh(),
      scratch_types=scratch_types,
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT_BYTES),
  )(dout, rhs_q, rhs_scale)


def _drhs_kernel(
    xq_hbm,
    sx_hbm,
    g_hbm,
    o_hbm,
    accum_vmem,
    *,
    bm: int,
    bk: int,
    bn: int,
):
  m, k = xq_hbm.shape
  _, n = g_hbm.shape
  grid = (pl.cdiv(k, bk), pl.cdiv(n, bn), pl.cdiv(m, bm))
  single_m = grid[2] == 1

  xq_spec = pl.BlockSpec((bm, bk), lambda ki, ni, mi: (mi, ki))
  sx_spec = pl.BlockSpec((bm, 1), lambda ki, ni, mi: (mi, 0))
  g_spec = pl.BlockSpec((bm, bn), lambda ki, ni, mi: (mi, ni))
  o_spec = pl.BlockSpec((bk, bn), lambda ki, ni, mi: (ki, ni))

  def _partial_product(xq_vmem, sx_vmem, g_vmem):
    g = g_vmem[...]
    gq, sg = quantize_cols_fp8(g * sx_vmem[...].astype(g.dtype))
    xg = jax.lax.dot_general(
        xq_vmem[...],
        gq,
        dimension_numbers=(((0,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return xg * sg

  def body_single_m(step: pltpu.PipelineStep, xq_vmem, sx_vmem, g_vmem, o_vmem):
    del step
    o_vmem[...] = _partial_product(xq_vmem, sx_vmem, g_vmem).astype(
        o_vmem.dtype
    )

  def body_multi_m(step: pltpu.PipelineStep, xq_vmem, sx_vmem, g_vmem, o_vmem):
    m_index = step.index[2]

    @pl.when(m_index == 0)
    def _init():
      accum_vmem[...] = jnp.zeros_like(accum_vmem)

    accum_vmem[...] += _partial_product(xq_vmem, sx_vmem, g_vmem)

    @pl.when(m_index == grid[2] - 1)
    def _write():
      o_vmem[...] = accum_vmem[...].astype(o_vmem.dtype)

  pltpu.emit_pipeline(
      body_single_m if single_m else body_multi_m,
      grid=grid,
      in_specs=[xq_spec, sx_spec, g_spec],
      out_specs=o_spec,
      core_axis_name=_CORE_AXIS_NAME,
      dimension_semantics=(
          pltpu.PARALLEL,
          pltpu.PARALLEL,
          pltpu.PARALLEL if single_m else pltpu.ARBITRARY,
      ),
      _explicit_indices=True,
  )(xq_hbm, sx_hbm, g_hbm, o_hbm)


def fused_fp8_drhs_pallas(
    lhs_q: jax.Array,
    lhs_scale: jax.Array,
    dout: jax.Array,
    *,
    bm: int,
    bk: int,
    bn: int,
    out_dtype: jnp.dtype | None = None,
) -> jax.Array:
  """Computes `lhs_q^T @ quantize_cols_fp8(dout * lhs_scale)` of shape `(k, n)`."""
  m, k = lhs_q.shape
  m_g, n = dout.shape
  if m_g != m:
    raise ValueError(f"Contracting dims differ: lhs has m={m}, dout has {m_g}.")
  if lhs_scale.shape != (m, 1):
    raise ValueError(
        f"lhs_scale must have shape ({m}, 1), got {lhs_scale.shape}."
    )
  if lhs_q.dtype != FP8_DTYPE:
    raise ValueError(f"lhs_q must be {FP8_DTYPE}, got {lhs_q.dtype}.")
  if m % bm:
    raise ValueError(f"m ({m}) must be divisible by bm ({bm}).")
  if k % bk:
    raise ValueError(f"k ({k}) must be divisible by bk ({bk}).")
  if n % bn:
    raise ValueError(f"n ({n}) must be divisible by bn ({bn}).")
  if bk % BLOCK_ALIGNMENT:
    raise ValueError(
        f"bk ({bk}) must be a multiple of {BLOCK_ALIGNMENT}."
    )

  accum_shape = (bk, bn) if m > bm else _DEAD_SCRATCH_SHAPE
  kernel = functools.partial(_drhs_kernel, bm=bm, bk=bk, bn=bn)
  return pl.kernel(
      kernel,
      out_type=jax.ShapeDtypeStruct((k, n), out_dtype or dout.dtype),
      mesh=_tensorcore_mesh(),
      scratch_types=[pltpu.VMEM(accum_shape, jnp.float32)],
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT_BYTES),
  )(lhs_q, lhs_scale, dout)
