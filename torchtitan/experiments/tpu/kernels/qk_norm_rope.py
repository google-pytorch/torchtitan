"""Pallas TPU kernels for fused per-head QK RMSNorm + rotate-half CosSinRoPE.

Unfused ``cossin_rotate(F.rms_norm(x, (D,), weight, eps), cos, sin)`` lowers in
XLA to separate reduction and elementwise fusions that materialize FP32 copies
of the ``[B, L, H, D]`` / ``[T, H, D]`` query and key activations in HBM and
trigger SparseCore layout relayouts between ``D``-minor QKV projections and
sequence-minor attention kernels. Under activation rematerialization the
forward runs twice and the backward once per layer.

These kernels run in ``[B, H, D, L]`` (head dim on sublanes, sequence on lanes)
as a single forward Pallas call, a single backward Pallas call, and a small
FP32 weight-gradient epilogue Pallas call:

* All ``D`` elements of a head reside along sublanes of ``(D, seq_tile)``, so
  the per-token ``rstd`` and backward dot-product ``m`` reduce in registers via
  8-sublane vector folds with zero cross-lane XLU shuffles.
* Each pass reads its BF16/FP32 inputs once and writes BF16/FP32 outputs once,
  keeping intermediate FP32 math in VMEM while matching the rounding order of
  ``F.rms_norm`` followed by ``CosSinRoPE`` (norm output rounded to ``x.dtype``
  before FP32 rotation).
* Weight gradients accumulate in FP32 into a
  ``(B, num_head_blocks, num_seq_blocks, D, 128)`` partial buffer so all three
  grid dimensions stay ``"parallel"``, then reduce to ``[D]`` in FP32 inside a
  small Pallas epilogue kernel so XLA's ``Bfloat16Propagation`` cannot demote
  the reduction when weights are BF16.
"""

from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
# pyformat: disable
# pylint: disable=g-bad-import-order
# With torch_tpu, torch must be imported before spmd_types to avoid a circular
# import when torch autoloads torch_tpu -> torch.distributed.fsdp -> spmd_types.
import torch  # isort: skip
import spmd_types as spmd
# pylint: enable=g-bad-import-order
# pyformat: enable
from torch.distributed.tensor import DTensor
import torch.nn.functional as F
from torch_tpu._internal import pallas

__all__ = [
    "qk_norm_rope",
    "qk_norm_rope_bwd_jax",
    "qk_norm_rope_fwd_jax",
    "supports",
]

_SUBLANES = 8
_SEQ_ALIGN = 128
_HEAD_DIM_ALIGN = 16
_MAX_HEAD_DIM = 1024
_MAX_HEADS_PER_BLOCK = 8
_MAX_SEQ_BLOCK = 1024
_MAX_DW_TILES_PER_BLOCK = 16
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


def _vmem_limit_bytes() -> int | None:
  """Returns 90% of TensorCore VMEM capacity in bytes, or None off TPU."""
  try:
    return int(0.9 * pltpu.get_tpu_info().vmem_capacity_bytes)
  except (AttributeError, RuntimeError, ValueError):
    return None


def _divisor_block(total: int, target: int, align: int = 1) -> int:
  """Returns the largest ``align``-multiple ``<= target`` dividing ``total``."""
  block = min(total, target)
  block -= block % align
  while block >= align and total % block != 0:
    block -= align
  return max(block, align)


def _sublane_sum(tile: jax.Array) -> jax.Array:
  """Sums ``(D, cols)`` over dim 0 -> ``(1, cols)`` via 8-sublane folds."""
  rows = tile.shape[0]
  folded = tile[:_SUBLANES]
  for offset in range(_SUBLANES, rows, _SUBLANES):
    folded = folded + tile[offset : offset + _SUBLANES]
  return jnp.sum(folded, axis=0, keepdims=True)


def _fwd_kernel(eps: float, x_ref, w_ref, c_ref, s_ref, o_ref) -> None:
  """Forward Pallas kernel body on ``(1, head_block, D, seq_tile)`` blocks."""
  head_block, head_dim, seq_tile = (
      x_ref.shape[1],
      x_ref.shape[2],
      x_ref.shape[3],
  )
  rotary_half = head_dim // 2
  inv_d = 1.0 / head_dim
  w = jnp.tile(w_ref[...], (1, seq_tile // _SEQ_ALIGN))
  c, s = c_ref[0], s_ref[0]

  for i in range(head_block):
    xf = x_ref[0, i].astype(jnp.float32)
    rstd = jax.lax.rsqrt(_sublane_sum(xf * xf) * inv_d + eps)
    y = ((xf * rstd) * w).astype(o_ref.dtype)
    y1 = y[:rotary_half].astype(jnp.float32)
    y2 = y[rotary_half:].astype(jnp.float32)
    o_ref[0, i, :rotary_half] = (y1 * c - y2 * s).astype(o_ref.dtype)
    o_ref[0, i, rotary_half:] = (y2 * c + y1 * s).astype(o_ref.dtype)


def _bwd_kernel(
    eps: float, x_ref, dy_ref, w_ref, c_ref, s_ref, dx_ref, dw_ref
) -> None:
  """Backward Pallas kernel body computing ``dx`` and per-tile ``dw``."""
  head_block, head_dim, seq_tile = (
      x_ref.shape[1],
      x_ref.shape[2],
      x_ref.shape[3],
  )
  rotary_half = head_dim // 2
  inv_d = 1.0 / head_dim
  w = jnp.tile(w_ref[...], (1, seq_tile // _SEQ_ALIGN))
  w1, w2 = w[:rotary_half], w[rotary_half:]
  c, s = c_ref[0], s_ref[0]
  dw_acc = jnp.zeros((head_dim, seq_tile), jnp.float32)

  for i in range(head_block):
    xf = x_ref[0, i].astype(jnp.float32)
    rstd = jax.lax.rsqrt(_sublane_sum(xf * xf) * inv_d + eps)
    n = xf * rstd
    n1, n2 = n[:rotary_half], n[rotary_half:]
    g = dy_ref[0, i]
    g1 = g[:rotary_half].astype(jnp.float32)
    g2 = g[rotary_half:].astype(jnp.float32)
    dn1 = (g1 * c + g2 * s).astype(x_ref.dtype).astype(jnp.float32)
    dn2 = (g2 * c - g1 * s).astype(x_ref.dtype).astype(jnp.float32)
    gw1, gw2 = dn1 * w1, dn2 * w2
    m = _sublane_sum(gw1 * n1 + gw2 * n2) * inv_d
    dx_ref[0, i, :rotary_half] = ((gw1 - n1 * m) * rstd).astype(dx_ref.dtype)
    dx_ref[0, i, rotary_half:] = ((gw2 - n2 * m) * rstd).astype(dx_ref.dtype)
    dw_acc = dw_acc + jnp.concatenate([dn1, dn2], axis=0) * n

  acc128 = dw_acc[:, :_SEQ_ALIGN]
  for col in range(_SEQ_ALIGN, seq_tile, _SEQ_ALIGN):
    acc128 = acc128 + dw_acc[:, col : col + _SEQ_ALIGN]
  dw_ref[0, 0, 0] = acc128


def _reduce_dw_kernel(dw_part_ref, dw_out_ref) -> None:
  """Reduces ``(tile_block, D, 128)`` partial ``dw`` tiles into FP32 VMEM."""
  head_dim = dw_part_ref.shape[1]
  d_pad = dw_out_ref.shape[1]
  acc = dw_part_ref[0]
  for tile_idx in range(1, dw_part_ref.shape[0]):
    acc = acc + dw_part_ref[tile_idx]
  if head_dim < d_pad:
    acc = jnp.concatenate(
        [acc, jnp.zeros((d_pad - head_dim, _SEQ_ALIGN), jnp.float32)], axis=0
    )
  row = _sublane_sum(jnp.transpose(acc, (1, 0)))

  @pl.when(pl.program_id(0) == 0)
  def _():
    dw_out_ref[...] = jnp.zeros_like(dw_out_ref)

  dw_out_ref[...] += jnp.broadcast_to(row, dw_out_ref.shape)


def _reduce_dw_f32(dw_part: jax.Array) -> jax.Array:
  """Reduces ``(B, nh, nl, D, 128)`` to ``(D,)`` in FP32 inside Pallas."""
  batch, num_head_blocks, num_seq_blocks, head_dim, _ = dw_part.shape
  num_tiles = batch * num_head_blocks * num_seq_blocks
  vmem_limit = _vmem_limit_bytes()
  effective_vmem = vmem_limit if vmem_limit is not None else 16 * 1024 * 1024
  dw_budget = min(2 * 1024 * 1024, effective_vmem // 8)
  max_tb = max(
      1, min(_MAX_DW_TILES_PER_BLOCK, dw_budget // (head_dim * _SEQ_ALIGN * 4))
  )
  tile_block = _divisor_block(num_tiles, max_tb, 1)
  d_pad = ((head_dim + _SEQ_ALIGN - 1) // _SEQ_ALIGN) * _SEQ_ALIGN
  dw_3d = jnp.reshape(dw_part, (num_tiles, head_dim, _SEQ_ALIGN))
  dw_pad = pl.pallas_call(
      _reduce_dw_kernel,
      grid=(num_tiles // tile_block,),
      in_specs=[
          pl.BlockSpec((tile_block, head_dim, _SEQ_ALIGN), lambda i: (i, 0, 0))
      ],
      out_specs=pl.BlockSpec((_SUBLANES, d_pad), lambda i: (0, 0)),
      out_shape=jax.ShapeDtypeStruct((_SUBLANES, d_pad), jnp.float32),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",),
          vmem_limit_bytes=vmem_limit,
      ),
      name="qk_norm_rope_bwd_dw",
  )(dw_3d)
  return dw_pad[0, :head_dim]


def _tile_sizes(
    num_heads: int, head_dim: int, seq_len: int, *dtypes: Any
) -> tuple[int, int]:
  """Returns ``(head_block, seq_tile)`` dividing ``(num_heads, seq_len)``."""
  elem_bytes = max(jnp.dtype(dt).itemsize for dt in dtypes)
  vmem_limit = _vmem_limit_bytes()
  effective_vmem = vmem_limit if vmem_limit is not None else 28 * 1024 * 1024
  # Bound both the per-head FP32 tile (4 * D * seq_tile <= min(1 MiB, vmem/16),
  # which also caps double-buffered FP32 cos/sin to <= min(2 MiB, vmem/8) and
  # per-head FP32 temporaries to ~5x that) and the multi-head x/dy/dx window
  # (<= min(2 MiB, vmem/12), i.e. <= vmem/2 across 6 double-buffered buffers).
  per_head_f32_budget = min(1024 * 1024, effective_vmem // 16)
  window_budget = min(2 * 1024 * 1024, effective_vmem // 12)
  max_lt_f32 = max(
      _SEQ_ALIGN,
      ((per_head_f32_budget // (4 * head_dim)) // _SEQ_ALIGN) * _SEQ_ALIGN,
  )
  max_lt_win = max(
      _SEQ_ALIGN,
      ((window_budget // (head_dim * elem_bytes)) // _SEQ_ALIGN) * _SEQ_ALIGN,
  )
  target_lt = min(seq_len, _MAX_SEQ_BLOCK, max_lt_f32, max_lt_win)
  seq_tile = _divisor_block(seq_len, target_lt, _SEQ_ALIGN)
  max_hb = max(
      1,
      min(
          _MAX_HEADS_PER_BLOCK,
          window_budget // (head_dim * seq_tile * elem_bytes),
      ),
  )
  head_block = _divisor_block(num_heads, max_hb, 1)
  return head_block, seq_tile


def _validate_jax_shapes(
    x: jax.Array, w: jax.Array, cos: jax.Array, sin: jax.Array
) -> tuple[int, int, int, int]:
  """Validates 4-D JAX tensor shapes and returns ``(B, seq_len, H, D)``."""
  if x.ndim != 4:
    raise ValueError(f"x must be 4-D [B, L, H, D], got {x.shape}")
  batch, seq_len, num_heads, head_dim = x.shape
  if batch <= 0 or num_heads <= 0:
    raise ValueError(f"batch and head dims must be positive, got {x.shape}")
  if (
      head_dim < _HEAD_DIM_ALIGN
      or head_dim > _MAX_HEAD_DIM
      or head_dim % _HEAD_DIM_ALIGN != 0
  ):
    raise ValueError(
        f"head_dim ({head_dim}) must be in [{_HEAD_DIM_ALIGN},"
        f" {_MAX_HEAD_DIM}] and divisible by {_HEAD_DIM_ALIGN}"
    )
  if seq_len < _SEQ_ALIGN or seq_len % _SEQ_ALIGN != 0:
    raise ValueError(
        f"seq_len ({seq_len}) must be a positive multiple of {_SEQ_ALIGN}"
    )
  if w.shape != (head_dim,):
    raise ValueError(f"weight shape {w.shape} must match ({head_dim},)")
  expected_cs = (batch, seq_len, head_dim // 2)
  if cos.shape != expected_cs or sin.shape != expected_cs:
    raise ValueError(
        f"cos/sin shapes {cos.shape}/{sin.shape} must match {expected_cs}"
    )
  return batch, seq_len, num_heads, head_dim


def qk_norm_rope_fwd_jax(
    x: jax.Array, w: jax.Array, cos: jax.Array, sin: jax.Array, eps: float
) -> jax.Array:
  """Runs the forward Pallas kernel on 4-D JAX arrays.

  Args:
    x: ``[B, L, H, D]`` queries or keys in BF16 or FP32.
    w: ``[D]`` RMSNorm weight.
    cos: ``[B, L, D/2]`` cosines.
    sin: ``[B, L, D/2]`` sines.
    eps: RMSNorm epsilon.

  Returns:
    Normalized and rotated ``[B, L, H, D]`` array in ``x.dtype``.

  Raises:
    ValueError: If input shapes or alignments violate kernel constraints.
  """
  batch, seq_len, num_heads, head_dim = _validate_jax_shapes(x, w, cos, sin)
  rotary_half = head_dim // 2
  head_block, seq_tile = _tile_sizes(num_heads, head_dim, seq_len, x.dtype)
  x_bhdl = jnp.transpose(x, (0, 2, 3, 1))
  c_bdl = jnp.transpose(cos.astype(jnp.float32), (0, 2, 1))
  s_bdl = jnp.transpose(sin.astype(jnp.float32), (0, 2, 1))
  w_2d = jnp.broadcast_to(
      w.astype(jnp.float32)[:, None], (head_dim, _SEQ_ALIGN)
  )
  x_spec = pl.BlockSpec(
      (1, head_block, head_dim, seq_tile), lambda i, j, k: (i, j, 0, k)
  )
  cs_spec = pl.BlockSpec((1, rotary_half, seq_tile), lambda i, j, k: (i, 0, k))
  w_spec = pl.BlockSpec((head_dim, _SEQ_ALIGN), lambda i, j, k: (0, 0))
  out_bhdl = pl.pallas_call(
      lambda *refs: _fwd_kernel(eps, *refs),
      grid=(batch, num_heads // head_block, seq_len // seq_tile),
      in_specs=[x_spec, w_spec, cs_spec, cs_spec],
      out_specs=x_spec,
      out_shape=jax.ShapeDtypeStruct(
          (batch, num_heads, head_dim, seq_len), x.dtype
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel"),
          vmem_limit_bytes=_vmem_limit_bytes(),
      ),
      name="qk_norm_rope_fwd",
  )(x_bhdl, w_2d, c_bdl, s_bdl)
  return jnp.transpose(out_bhdl, (0, 3, 1, 2))


def qk_norm_rope_bwd_jax(
    x: jax.Array,
    dy: jax.Array,
    w: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
  """Runs the backward Pallas kernels on 4-D JAX arrays.

  Args:
    x: ``[B, L, H, D]`` forward input.
    dy: ``[B, L, H, D]`` upstream output gradient.
    w: ``[D]`` RMSNorm weight.
    cos: ``[B, L, D/2]`` cosines.
    sin: ``[B, L, D/2]`` sines.
    eps: RMSNorm epsilon.

  Returns:
    Tuple ``(dx, dw_f32)`` where ``dx`` matches ``x`` and ``dw_f32`` is FP32.

  Raises:
    ValueError: If input shapes or alignments violate kernel constraints.
  """
  batch, seq_len, num_heads, head_dim = _validate_jax_shapes(x, w, cos, sin)
  if dy.shape != x.shape:
    raise ValueError(f"dy shape {dy.shape} must match x shape {x.shape}")
  rotary_half = head_dim // 2
  head_block, seq_tile = _tile_sizes(
      num_heads, head_dim, seq_len, x.dtype, dy.dtype
  )
  num_head_blocks = num_heads // head_block
  num_seq_blocks = seq_len // seq_tile
  x_bhdl = jnp.transpose(x, (0, 2, 3, 1))
  dy_bhdl = jnp.transpose(dy, (0, 2, 3, 1))
  c_bdl = jnp.transpose(cos.astype(jnp.float32), (0, 2, 1))
  s_bdl = jnp.transpose(sin.astype(jnp.float32), (0, 2, 1))
  w_2d = jnp.broadcast_to(
      w.astype(jnp.float32)[:, None], (head_dim, _SEQ_ALIGN)
  )
  x_spec = pl.BlockSpec(
      (1, head_block, head_dim, seq_tile), lambda i, j, k: (i, j, 0, k)
  )
  cs_spec = pl.BlockSpec((1, rotary_half, seq_tile), lambda i, j, k: (i, 0, k))
  w_spec = pl.BlockSpec((head_dim, _SEQ_ALIGN), lambda i, j, k: (0, 0))
  dw_spec = pl.BlockSpec(
      (1, 1, 1, head_dim, _SEQ_ALIGN), lambda i, j, k: (i, j, k, 0, 0)
  )
  dx_bhdl, dw_part = pl.pallas_call(
      lambda *refs: _bwd_kernel(eps, *refs),
      grid=(batch, num_head_blocks, num_seq_blocks),
      in_specs=[x_spec, x_spec, w_spec, cs_spec, cs_spec],
      out_specs=[x_spec, dw_spec],
      out_shape=[
          jax.ShapeDtypeStruct((batch, num_heads, head_dim, seq_len), x.dtype),
          jax.ShapeDtypeStruct(
              (batch, num_head_blocks, num_seq_blocks, head_dim, _SEQ_ALIGN),
              jnp.float32,
          ),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel"),
          vmem_limit_bytes=_vmem_limit_bytes(),
      ),
      name="qk_norm_rope_bwd",
  )(x_bhdl, dy_bhdl, w_2d, c_bdl, s_bdl)
  dx = jnp.transpose(dx_bhdl, (0, 3, 1, 2))
  dw = _reduce_dw_f32(dw_part)
  return dx, dw


@pallas.jax_op("tt_attn::qk_norm_rope_fwd")
def _fwd_op(
    x: jax.Array, w: jax.Array, cos: jax.Array, sin: jax.Array, eps: float
) -> jax.Array:
  return qk_norm_rope_fwd_jax(x, w, cos, sin, eps)


@pallas.jax_op("tt_attn::qk_norm_rope_bwd")
def _bwd_op(
    x: jax.Array,
    dy: jax.Array,
    w: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
  return qk_norm_rope_bwd_jax(x, dy, w, cos, sin, eps)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
  del output
  ctx.set_materialize_grads(False)
  x, w, cos, sin, eps = inputs
  ctx.save_for_backward(x, w, cos, sin)
  ctx.eps = eps


def _backward(
    ctx: Any, dy: torch.Tensor | None
) -> tuple[torch.Tensor | None, ...]:
  need_dx, need_dw, need_dcos, need_dsin, _ = ctx.needs_input_grad
  assert not (need_dcos or need_dsin)
  if dy is None or not (need_dx or need_dw):
    return None, None, None, None, None
  x, w, cos, sin = ctx.saved_tensors
  dx, dw = _bwd_op(x, dy.contiguous(), w, cos, sin, ctx.eps)
  return (
      dx if need_dx else None,
      dw if need_dw else None,
      None,
      None,
      None,
  )


_fwd_op.register_autograd(_backward, setup_context=_setup_context)


@torch._dynamo.assume_constant_result  # pylint: disable=protected-access
def _is_spmd_type_checking() -> bool:
  return spmd.is_type_checking()


def _native_qk_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
  """PyTorch reference decomposition for ``spmd_types`` propagation."""
  y = F.rms_norm(x, (x.shape[-1],), weight, eps)
  rotary_half = x.shape[-1] // 2
  cs_shape = (*x.shape[:-2], 1, rotary_half)
  cos_h = cos.float().reshape(cs_shape)
  sin_h = sin.float().reshape(cs_shape)
  yf = y.float()
  y1, y2 = yf[..., :rotary_half], yf[..., rotary_half:]
  o1 = (y1 * cos_h - y2 * sin_h).type_as(x)
  o2 = (y2 * cos_h + y1 * sin_h).type_as(x)
  return torch.cat((o1, o2), dim=-1)


def _prepare_4d(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Normalizes 3-D/4-D ``x`` and ``cos``/``sin`` to contiguous 4-D/3-D views."""
  head_dim = x.shape[-1]
  rotary_half = head_dim // 2
  if x.ndim == 3:
    num_tokens, num_heads, _ = x.shape
    batch, seq_len = 1, num_tokens
    x_4d = x.reshape(1, num_tokens, num_heads, head_dim).contiguous()
  else:
    batch, seq_len, _, _ = x.shape
    x_4d = x.contiguous()
  cos_3d = (
      cos.to(torch.float32).reshape(batch, seq_len, rotary_half).contiguous()
  )
  sin_3d = (
      sin.to(torch.float32).reshape(batch, seq_len, rotary_half).contiguous()
  )
  return x_4d, cos_3d, sin_3d


class _QkNormRopeSpmdFn(torch.autograd.Function):
  """``qk_norm_rope`` wrapper registered with ``spmd.register_decomposition``."""

  @staticmethod
  def forward(
      ctx: Any,
      x: torch.Tensor,
      weight: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      eps: float,
  ) -> torch.Tensor:
    ctx.set_materialize_grads(False)
    ctx.orig_shape = x.shape
    ctx.weight_dtype = weight.dtype
    ctx.eps = float(eps)
    with spmd.no_typecheck():
      x_4d, cos_3d, sin_3d = _prepare_4d(x, cos, sin)
      w_f32 = weight.to(torch.float32).contiguous()
      out = _fwd_op(x_4d, w_f32, cos_3d, sin_3d, ctx.eps).reshape(x.shape)
    ctx.save_for_backward(x_4d, w_f32, cos_3d, sin_3d)
    return out

  @staticmethod
  def backward(
      ctx: Any, dy: torch.Tensor | None
  ) -> tuple[torch.Tensor | None, ...]:
    need_dx, need_dw, need_dcos, need_dsin, _ = ctx.needs_input_grad
    assert not (need_dcos or need_dsin)
    if dy is None or not (need_dx or need_dw):
      return None, None, None, None, None
    x_4d, w_f32, cos_3d, sin_3d = ctx.saved_tensors
    with spmd.no_typecheck():
      dx_4d, dw_f32 = _bwd_op(
          x_4d,
          dy.reshape(x_4d.shape).contiguous(),
          w_f32,
          cos_3d,
          sin_3d,
          ctx.eps,
      )
      dx = dx_4d.reshape(ctx.orig_shape) if need_dx else None
      dw = dw_f32.to(ctx.weight_dtype) if need_dw else None
    return dx, dw, None, None, None


spmd.register_decomposition(_QkNormRopeSpmdFn, _native_qk_norm_rope)


def _has_sharded_norm_dim(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
  """Whether the last (norm/rotary) dim of any operand is sharded."""
  for tensor in (x, cos, sin):
    ps = spmd.get_partition_spec(tensor)
    if ps is not None and len(ps) > 0 and ps[-1] is not None:
      return True
  w_ps = spmd.get_partition_spec(weight)
  return w_ps is not None and len(w_ps) > 0 and w_ps[0] is not None


def supports(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
  """Returns whether ``(x, weight, cos, sin)`` satisfy Pallas kernel limits.

  Args:
    x: Query or key activation tensor (3-D ``[T, H, D]`` or 4-D ``[B, L, H,
      D]``).
    weight: 1-D ``[D]`` RMSNorm weight tensor.
    cos: Cosine cache tensor matching ``x``'s leading token dims and ``D/2``.
    sin: Sine cache tensor with the same shape, dtype, and device as ``cos``.

  Returns:
    ``True`` if ``x``, ``weight``, ``cos``, and ``sin`` are unsharded along
    ``D``
    on TPU in BF16/FP32, ``cos``/``sin`` do not require gradients, ``D`` is a
    multiple of 16 in ``[16, 1024]``, and ``L`` / ``T`` is a multiple of 128.
  """
  if any(isinstance(tensor, DTensor) for tensor in (x, weight, cos, sin)):
    return False
  if _is_spmd_type_checking() and _has_sharded_norm_dim(x, weight, cos, sin):
    return False
  if torch.is_grad_enabled() and (cos.requires_grad or sin.requires_grad):
    return False
  if x.device.type != "tpu" or any(
      tensor.device != x.device for tensor in (weight, cos, sin)
  ):
    return False
  if any(
      tensor.dtype not in _SUPPORTED_DTYPES for tensor in (x, weight, cos, sin)
  ):
    return False
  if x.ndim not in (3, 4):
    return False
  num_heads, head_dim = x.shape[-2], x.shape[-1]
  if (
      num_heads <= 0
      or head_dim < _HEAD_DIM_ALIGN
      or head_dim > _MAX_HEAD_DIM
      or head_dim % _HEAD_DIM_ALIGN != 0
  ):
    return False
  if weight.shape != (head_dim,):
    return False
  rotary_half = head_dim // 2
  if x.ndim == 4:
    batch, seq_len = x.shape[0], x.shape[1]
    if batch <= 0 or seq_len < _SEQ_ALIGN or seq_len % _SEQ_ALIGN != 0:
      return False
    valid_cs = ((batch, seq_len, rotary_half), (batch, seq_len, 1, rotary_half))
  else:
    num_tokens = x.shape[0]
    if num_tokens < _SEQ_ALIGN or num_tokens % _SEQ_ALIGN != 0:
      return False
    valid_cs = ((num_tokens, rotary_half), (num_tokens, 1, rotary_half))
  return cos.shape in valid_cs and sin.shape == cos.shape


def qk_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
  """Applies fused per-head RMSNorm and rotate-half CosSinRoPE on TPU.

  Args:
    x: ``[B, L, H, D]`` or ``[T, H, D]`` queries or keys in BF16 or FP32.
    weight: ``[D]`` RMSNorm weight in BF16 or FP32.
    cos: ``[B, L, (1,) D/2]`` or ``[T, (1,) D/2]`` cosines (first half of
      ``CosSinRoPE`` cos).
    sin: Matching sines (first half of ``CosSinRoPE`` sin).
    eps: Non-negative RMSNorm epsilon.

  Returns:
    Normalized and rotated tensor with ``x``'s shape and dtype.

  Raises:
    ValueError: If ``supports(x, weight, cos, sin)`` is false or ``eps < 0``.
  """
  if not supports(x, weight, cos, sin):
    raise ValueError(
        "Unsupported inputs for Pallas qk_norm_rope:"
        f" x={tuple(x.shape)} ({x.dtype}, {x.device}),"
        f" weight={tuple(weight.shape)} ({weight.dtype}, {weight.device}),"
        f" cos={tuple(cos.shape)} ({cos.dtype}, {cos.device}),"
        f" sin={tuple(sin.shape)} ({sin.dtype}, {sin.device})"
    )
  if eps < 0:
    raise ValueError(f"eps must be non-negative, got {eps}")
  if _is_spmd_type_checking():
    return _QkNormRopeSpmdFn.apply(x, weight, cos, sin, float(eps))
  x_4d, cos_3d, sin_3d = _prepare_4d(x, cos, sin)
  w_f32 = weight.to(torch.float32).contiguous()
  return _fwd_op(x_4d, w_f32, cos_3d, sin_3d, float(eps)).reshape(x.shape)
