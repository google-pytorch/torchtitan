"""Pure-JAX Pallas TPU kernels for fused QK-LayerNorm + RoPE + head split.

ESM-C attention prologue (HuggingFace ``EsmcAttention``)::

    q = q_norm(q_proj(x)).to(dtype)          # LayerNorm over H*D, no bias
    k = k_norm(k_proj(x)).to(dtype)
    v = v_proj(x)
    q, k, v = (t.view(B, S, H, D).transpose(1, 2) for t in (q, k, v))
    q, k = apply_rotary_pos_emb(q, k, cos, sin)   # rotate_half RoPE

Under standard XLA lowering, each layer runs a full-row LayerNorm over
``(B, S, H*D)``, a physical relayout copy into heads-first ``(B, H, S, D)``
(because ``D = 64`` is half a 128-lane tile, so the transpose is not a free
bitcast), and an elementwise RoPE chain -- plus the mirror relayout and
reductions in backward.

This module implements the fused forward and backward Pallas kernels in pure JAX
(no ``torch`` or ``torch_tpu`` dependency, so it can be unit-tested on CPU with
``interpret=True``):

* **Forward** (`qk_norm_rope_fwd`): streams one ``(block_s, H*D)`` row tile of
  ``xq``, ``xk``, ``xv`` per grid step, computes per-token mean/rstd in fp32,
  applies weight-only LayerNorm + activation-dtype rounding + ``rotate_half``
  RoPE (optionally folding a power-of-two softmax ``q_scale`` into ``q``), and
  stores ``(1, H, block_s, D)`` heads-first tiles directly.
* **Backward** (`qk_norm_rope_bwd`): reads heads-first ``(gq, gk, gv)`` tiles,
  applies the exact RoPE VJP and LayerNorm backward in VMEM, writes row-major
  ``(dxq, dxk, dxv)`` of shape ``(B, S, H*D)``, and accumulates ``(dwq, dwk)``
  sequentially across sequence tiles in fp32 per batch element before reducing
  across batch.

Lane arithmetic operates on 128-lane groups: for ``D = 64`` each group holds two
heads side by side, and ``rotate_half`` within each head uses two lane rotations
by ``D // 2`` plus a lane-mask select (wrap-around lanes are masked out).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NUM_LANES = 128
SUBLANE_ALIGN = 8
DEFAULT_BLOCK_S = 512
MAX_BLOCK_S = 1024
DEFAULT_VMEM_LIMIT_BYTES = 48 * 1024 * 1024
# Maximum valid scoped VMEM on 64 MiB TPUs (tpu7x / v5p), where 64 KiB is
# reserved by the compiler overlay (`64 MiB - 64 KiB = 65,472 KiB`).
MAX_SCOPED_VMEM_BYTES = 65472 * 1024
_KERNEL_VMEM_BUDGET_BYTES = 63 * 1024 * 1024
_BLOCK_S_CANDIDATES = (1024, 512, 256, 128, 64, 32, 16, 8)


def lane_group_width(head_dim: int) -> int:
  """Returns the VMEM lane-group width (>= 128, multiple of 128) for `head_dim`.

  Supports ``head_dim == 64`` (two heads per 128-lane group) and any positive
  multiple of ``NUM_LANES`` (one head per group).
  """
  if head_dim == 64:
    return NUM_LANES
  if head_dim >= NUM_LANES and head_dim % NUM_LANES == 0:
    return head_dim
  raise ValueError(
      f"head_dim {head_dim} must be 64 or a positive multiple of {NUM_LANES}"
  )


def pick_seq_block(seq: int, block_s: int = DEFAULT_BLOCK_S) -> int | None:
  """Returns the effective sequence tile size, or `None` if unaligned."""
  if seq <= 0 or block_s <= 0:
    return None
  bs = min(block_s, seq)
  if bs % SUBLANE_ALIGN != 0 or seq % bs != 0:
    return None
  return bs


def _estimate_bwd_vmem_bytes(
    bs: int, heads: int, head_dim: int, dtype_bytes: int = 2
) -> int:
  """Conservative backward VMEM estimate (double-buffered I/O + fp32 spills)."""
  hidden_dim = heads * head_dim
  # Mosaic pads minor-most `head_dim == 64` to 128 lanes in `(1, H, bs, D)`.
  d_pad = lane_group_width(head_dim)
  io_double_buffered = (
      10 * bs * hidden_dim * dtype_bytes
      + 6 * bs * heads * d_pad * dtype_bytes
      + 4 * bs * d_pad * dtype_bytes
  )
  fixed_io = 32 * hidden_dim * (dtype_bytes + 4)
  fp32_spills = 8 * bs * hidden_dim + 20 * bs * d_pad
  return io_double_buffered + fixed_io + fp32_spills


def default_block_s(
    seq: int, heads: int, head_dim: int, dtype_bytes: int = 2
) -> int | None:
  """Picks the largest VMEM-safe sequence tile (<= 1024) dividing `seq`."""
  if seq <= 0 or seq % SUBLANE_ALIGN != 0 or heads <= 0:
    return None
  try:
    group = lane_group_width(head_dim)
  except ValueError:
    return None
  if (heads * head_dim) % group != 0:
    return None
  if (
      seq <= MAX_BLOCK_S
      and _estimate_bwd_vmem_bytes(seq, heads, head_dim, dtype_bytes)
      <= _KERNEL_VMEM_BUDGET_BYTES
  ):
    return seq
  for bs in _BLOCK_S_CANDIDATES:
    if (
        bs <= seq
        and seq % bs == 0
        and _estimate_bwd_vmem_bytes(bs, heads, head_dim, dtype_bytes)
        <= _KERNEL_VMEM_BUDGET_BYTES
    ):
      return bs
  return None


def default_vmem_limit_bytes(
    bs: int, heads: int, head_dim: int, dtype_bytes: int = 2
) -> int:
  """Returns per-call scoped VMEM limit (`48 MiB` for small tiles, `65,472 KiB` for large)."""
  est = _estimate_bwd_vmem_bytes(bs, heads, head_dim, dtype_bytes)
  if est <= 40 * 1024 * 1024:
    return DEFAULT_VMEM_LIMIT_BYTES
  return MAX_SCOPED_VMEM_BYTES


def _resolve_seq_block(
    seq: int,
    heads: int,
    head_dim: int,
    block_s: int | None,
    dtype_bytes: int = 2,
) -> int | None:
  if block_s is None:
    return default_block_s(seq, heads, head_dim, dtype_bytes)
  return pick_seq_block(seq, block_s)


def validate_geometry(
    *,
    batch: int,
    seq: int,
    hidden_dim: int,
    heads: int,
    head_dim: int,
    cos_shape: tuple[int, ...],
    sin_shape: tuple[int, ...],
    block_s: int | None = None,
    dtype_bytes: int = 2,
) -> str | None:
  """Returns a human-readable rejection reason, or `None` when supported."""
  if batch <= 0 or seq <= 0:
    return f"batch ({batch}) and seq ({seq}) must be positive"
  if heads <= 0:
    return f"heads ({heads}) must be positive"
  if head_dim != 64 and (head_dim < NUM_LANES or head_dim % NUM_LANES != 0):
    return (
        f"head_dim {head_dim} must be 64 or a positive multiple of {NUM_LANES}"
    )
  expected_hidden = heads * head_dim
  if hidden_dim != expected_hidden:
    return f"expected projection width H*D={expected_hidden}, got {hidden_dim}"
  group = lane_group_width(head_dim)
  if hidden_dim % group != 0:
    return f"H*D={hidden_dim} is not a multiple of {group}-lane group width"
  if _resolve_seq_block(seq, heads, head_dim, block_s, dtype_bytes) is None:
    return (
        f"sequence {seq} with block_s={block_s} does not form {SUBLANE_ALIGN}-"
        "aligned tiles"
    )
  if len(cos_shape) != 3 or cos_shape != sin_shape:
    return (
        f"cos/sin shapes {cos_shape} vs {sin_shape} must match and have rank 3"
    )
  if cos_shape[0] not in (1, batch) or cos_shape[1:] != (seq, head_dim):
    return (
        f"expected cos/sin shape (1 or {batch}, {seq}, {head_dim}), got"
        f" {cos_shape}"
    )
  return None


def _first_half_head_mask(bs: int, group_width: int, head_dim: int) -> jax.Array:
  """Boolean mask `(bs, group_width)` True on the first `head_dim // 2` lanes of each head."""
  lane = jax.lax.broadcasted_iota(jnp.int32, (bs, group_width), 1)
  return jax.lax.rem(lane, head_dim) < (head_dim // 2)


def _repeat_for_group_heads(
    table: jax.Array, group_width: int, head_dim: int
) -> jax.Array:
  """Tiles `(bs, head_dim)` RoPE tables across all heads packed in one lane group."""
  reps = group_width // head_dim
  return table if reps == 1 else jnp.concatenate([table] * reps, axis=-1)


def _rotate_half_partner(
    x: jax.Array, first_half_mask: jax.Array, half_dim: int
) -> jax.Array:
  """Per head: `x[l + half_dim]` on the first half, `x[l - half_dim]` on the second.

  With a signed sine table `sin_signed = where(first_half, -sin, sin)`, forward
  RoPE is `x * cos + _rotate_half_partner(x, first_half, half) * sin_signed`,
  and its exact VJP for arbitrary `(cos, sin)` tables is
  `g * cos + _rotate_half_partner(g * sin_signed, first_half, half)`.
  """
  group_width = x.shape[-1]
  ahead = pltpu.roll(x, group_width - half_dim, 1)  # ahead[l] = x[l + half_dim]
  behind = pltpu.roll(x, half_dim, 1)  # behind[l] = x[l - half_dim]
  return jnp.where(first_half_mask, ahead, behind)


def _lane_group_row_sum(x: jax.Array, group_width: int) -> jax.Array:
  """`(rows, 1)` sum over the last axis via elementwise group adds + 1 XLU reduce."""
  acc = x[:, :group_width]
  for start in range(group_width, x.shape[-1], group_width):
    acc = acc + x[:, start : start + group_width]
  return jnp.sum(acc, axis=-1, keepdims=True)


def _layernorm_row_stats(
    x: jax.Array, eps: float, group_width: int
) -> tuple[jax.Array, jax.Array]:
  """Returns per-token `(mean, rstd)` in fp32 over the last axis of `x`."""
  inv_n = 1.0 / x.shape[-1]
  mu = _lane_group_row_sum(x, group_width) * inv_n
  d0 = x[:, :group_width] - mu
  var_acc = d0 * d0
  for start in range(group_width, x.shape[-1], group_width):
    d = x[:, start : start + group_width] - mu
    var_acc = var_acc + d * d
  var = jnp.sum(var_acc, axis=-1, keepdims=True) * inv_n
  return mu, jax.lax.rsqrt(var + eps)


def _pack_heads_to_group(
    g_ref, group_idx: int, heads_per_group: int, *, to_f32: bool = True
) -> jax.Array:
  """Concatenates the `heads_per_group` heads of group `group_idx` into `(bs, group_width)`."""
  parts = [
      g_ref[0, group_idx * heads_per_group + j] for j in range(heads_per_group)
  ]
  if to_f32:
    parts = [p.astype(jnp.float32) for p in parts]
  return parts[0] if heads_per_group == 1 else jnp.concatenate(parts, axis=-1)


def _qk_norm_rope_fwd_kernel(
    xq_ref,
    xk_ref,
    xv_ref,
    wq_ref,
    wk_ref,
    cos_ref,
    sin_ref,
    oq_ref,
    ok_ref,
    ov_ref,
    *,
    heads: int,
    head_dim: int,
    eps: float,
    q_scale: float,
):
  bs = xq_ref.shape[1]
  group_width = lane_group_width(head_dim)
  heads_per_group = group_width // head_dim
  half_dim = head_dim // 2
  n_groups = (heads * head_dim) // group_width
  first_half = _first_half_head_mask(bs, group_width, head_dim)
  cos = _repeat_for_group_heads(
      cos_ref[0].astype(jnp.float32), group_width, head_dim
  )
  sin = _repeat_for_group_heads(
      sin_ref[0].astype(jnp.float32), group_width, head_dim
  )
  sin_signed = jnp.where(first_half, -sin, sin)

  for x_ref, w_ref, o_ref, scale in (
      (xq_ref, wq_ref, oq_ref, q_scale),
      (xk_ref, wk_ref, ok_ref, 1.0),
  ):
    x = x_ref[0].astype(jnp.float32)
    mu, rstd = _layernorm_row_stats(x, eps, group_width)
    for gi in range(n_groups):
      sl = slice(gi * group_width, (gi + 1) * group_width)
      w = w_ref[:, sl].astype(jnp.float32)
      normed = ((x[:, sl] - mu) * rstd * w).astype(o_ref.dtype).astype(
          jnp.float32
      )
      out = (
          normed * cos
          + _rotate_half_partner(normed, first_half, half_dim) * sin_signed
      )
      if scale != 1.0:
        out = out * scale
      out = out.astype(o_ref.dtype)
      for j in range(heads_per_group):
        o_ref[0, gi * heads_per_group + j] = out[
            :, j * head_dim : (j + 1) * head_dim
        ]

  xv = xv_ref[0]
  for gi in range(n_groups):
    blk = xv[:, gi * group_width : (gi + 1) * group_width]
    for j in range(heads_per_group):
      ov_ref[0, gi * heads_per_group + j] = blk[
          :, j * head_dim : (j + 1) * head_dim
      ].astype(ov_ref.dtype)


def _qk_norm_rope_bwd_kernel(
    xq_ref,
    xk_ref,
    wq_ref,
    wk_ref,
    cos_ref,
    sin_ref,
    gq_ref,
    gk_ref,
    gv_ref,
    dxq_ref,
    dxk_ref,
    dxv_ref,
    dwq_ref,
    dwk_ref,
    *,
    heads: int,
    head_dim: int,
    eps: float,
    q_scale: float,
):
  s_idx = pl.program_id(1)
  bs = xq_ref.shape[1]
  hidden_dim = heads * head_dim
  group_width = lane_group_width(head_dim)
  heads_per_group = group_width // head_dim
  half_dim = head_dim // 2
  n_groups = hidden_dim // group_width
  first_half = _first_half_head_mask(bs, group_width, head_dim)
  cos = _repeat_for_group_heads(
      cos_ref[0].astype(jnp.float32), group_width, head_dim
  )
  sin = _repeat_for_group_heads(
      sin_ref[0].astype(jnp.float32), group_width, head_dim
  )
  sin_signed = jnp.where(first_half, -sin, sin)

  # Grid axis 1 is marked "arbitrary" so sequence tiles execute in order within
  # each batch index `b`, initializing `dwq`/`dwk` on tile 0 and accumulating.
  @pl.when(s_idx == 0)
  def _init_dw():
    dwq_ref[...] = jnp.zeros_like(dwq_ref)
    dwk_ref[...] = jnp.zeros_like(dwk_ref)

  for x_ref, w_ref, g_ref, dx_ref, dw_ref, scale in (
      (xq_ref, wq_ref, gq_ref, dxq_ref, dwq_ref, q_scale),
      (xk_ref, wk_ref, gk_ref, dxk_ref, dwk_ref, 1.0),
  ):
    x = x_ref[0].astype(jnp.float32)
    mu, rstd = _layernorm_row_stats(x, eps, group_width)
    acc_dxhat: jax.Array | None = None
    acc_dxhat_xhat: jax.Array | None = None
    group_terms = []
    for gi in range(n_groups):
      sl = slice(gi * group_width, (gi + 1) * group_width)
      grad_out = _pack_heads_to_group(g_ref, gi, heads_per_group, to_f32=True)
      if scale != 1.0:
        grad_out = grad_out * scale
      # Exact VJP of `n * cos + rotate_partner(n) * sin_signed` for any (cos, sin):
      # k < half receives +grad_out[k + half] * sin[k + half];
      # k >= half receives -grad_out[k - half] * sin[k - half].
      dnormed = grad_out * cos + _rotate_half_partner(
          grad_out * sin_signed, first_half, half_dim
      )
      xhat = (x[:, sl] - mu) * rstd
      dw_ref[0, :, sl] += jnp.sum(dnormed * xhat, axis=0, keepdims=True)
      dxhat = dnormed * w_ref[:, sl].astype(jnp.float32)
      dxhat_xhat = dxhat * xhat
      # Accumulate per-lane partial sums across groups, then reduce across lanes
      # once after the loop.
      if acc_dxhat is None:
        acc_dxhat, acc_dxhat_xhat = dxhat, dxhat_xhat
      else:
        acc_dxhat = acc_dxhat + dxhat
        assert acc_dxhat_xhat is not None
        acc_dxhat_xhat = acc_dxhat_xhat + dxhat_xhat
      group_terms.append((sl, dxhat, xhat))
    assert acc_dxhat is not None and acc_dxhat_xhat is not None
    inv_hidden = 1.0 / hidden_dim
    mean_dxhat = jnp.sum(acc_dxhat, axis=-1, keepdims=True) * inv_hidden
    mean_dxhat_xhat = (
        jnp.sum(acc_dxhat_xhat, axis=-1, keepdims=True) * inv_hidden
    )
    for sl, dxhat, xhat in group_terms:
      dx_ref[0, :, sl] = (
          rstd * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat)
      ).astype(dx_ref.dtype)

  for gi in range(n_groups):
    sl = slice(gi * group_width, (gi + 1) * group_width)
    dxv_ref[0, :, sl] = _pack_heads_to_group(
        gv_ref, gi, heads_per_group, to_f32=False
    ).astype(dxv_ref.dtype)


def _make_block_specs(
    hidden_dim: int,
    heads: int,
    head_dim: int,
    bs: int,
    *,
    cos_batched: bool,
):
  row = pl.BlockSpec((1, bs, hidden_dim), lambda b, s: (b, s, 0))
  weight = pl.BlockSpec((1, hidden_dim), lambda b, s: (0, 0))
  rope_index = (lambda b, s: (b, s, 0)) if cos_batched else (lambda b, s: (0, s, 0))
  rope_table = pl.BlockSpec((1, bs, head_dim), rope_index)
  heads_first = pl.BlockSpec((1, heads, bs, head_dim), lambda b, s: (b, 0, s, 0))
  return row, weight, rope_table, heads_first


def _check_common_inputs(
    xq: jax.Array,
    xk: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    heads: int,
    head_dim: int,
    block_s: int | None,
) -> tuple[int, int, int, int]:
  if xq.ndim != 3 or xk.shape != xq.shape:
    raise ValueError(
        f"Expected matching rank-3 xq/xk projections, got {xq.shape} and"
        f" {xk.shape}"
    )
  batch, seq, hidden_dim = xq.shape
  if wq.shape != (hidden_dim,) or wk.shape != (hidden_dim,):
    raise ValueError(
        f"Expected wq/wk shape ({hidden_dim},), got {wq.shape} and {wk.shape}"
    )
  dtype_bytes = xq.dtype.itemsize
  reason = validate_geometry(
      batch=batch,
      seq=seq,
      hidden_dim=hidden_dim,
      heads=heads,
      head_dim=head_dim,
      cos_shape=tuple(cos.shape),
      sin_shape=tuple(sin.shape),
      block_s=block_s,
      dtype_bytes=dtype_bytes,
  )
  if reason is not None:
    raise ValueError(f"Unsupported qk_norm_rope geometry: {reason}")
  bs = _resolve_seq_block(seq, heads, head_dim, block_s, dtype_bytes)
  assert bs is not None
  return batch, seq, hidden_dim, bs


def qk_norm_rope_fwd(
    xq: jax.Array,
    xk: jax.Array,
    xv: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    heads: int,
    eps: float,
    q_scale: float = 1.0,
    block_s: int | None = None,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Fused forward: `(B, S, H*D)` projections -> heads-first `(B, H, S, D)`.

  Args:
    xq, xk, xv: `(B, S, H*D)` query, key, and value projection outputs.
    wq, wk: `(H*D,)` bias-free LayerNorm weights for `q_norm` and `k_norm`.
    cos, sin: `(B, S, D)` or `(1, S, D)` `rotate_half` RoPE tables.
    heads: Number of attention heads `H` (shared by `q` and `k`).
    eps: LayerNorm epsilon.
    q_scale: Scale multiplied onto `q` inside the kernel (for folding a
      power-of-two attention softmax scale `D ** -0.5` at zero extra HBM pass).
    block_s: Sequence tile size per Pallas grid step, or `None` to select the
      largest VMEM-safe tile (up to 1024) dividing `S`.
    vmem_limit_bytes: Compiler VMEM limit for Mosaic TPU lowering.
    interpret: Run Pallas in CPU interpret mode for unit tests.
  """
  if xv.shape != xq.shape:
    raise ValueError(f"Expected xv shape {xq.shape}, got {xv.shape}")
  if heads <= 0 or xq.shape[-1] % heads != 0:
    raise ValueError(
        f"Projection width {xq.shape[-1]} is not divisible by heads={heads}"
    )
  head_dim = xq.shape[-1] // heads
  batch, seq, hidden_dim, bs = _check_common_inputs(
      xq, xk, wq, wk, cos, sin, heads=heads, head_dim=head_dim, block_s=block_s
  )
  if vmem_limit_bytes is None:
    vmem_limit_bytes = default_vmem_limit_bytes(
        bs, heads, head_dim, xq.dtype.itemsize
    )
  row, weight, rope_table, heads_first = _make_block_specs(
      hidden_dim, heads, head_dim, bs, cos_batched=cos.shape[0] > 1
  )
  kernel = functools.partial(
      _qk_norm_rope_fwd_kernel,
      heads=heads,
      head_dim=head_dim,
      eps=eps,
      q_scale=q_scale,
  )
  out_shape = tuple(
      jax.ShapeDtypeStruct((batch, heads, seq, head_dim), t.dtype)
      for t in (xq, xk, xv)
  )
  return pl.pallas_call(
      kernel,
      grid=(batch, seq // bs),
      in_specs=[row, row, row, weight, weight, rope_table, rope_table],
      out_specs=[heads_first, heads_first, heads_first],
      out_shape=out_shape,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel"),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      interpret=interpret,
      name="esmc_qk_norm_rope_fwd",
  )(xq, xk, xv, wq.reshape(1, hidden_dim), wk.reshape(1, hidden_dim), cos, sin)


def qk_norm_rope_bwd(
    xq: jax.Array,
    xk: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    gq: jax.Array,
    gk: jax.Array,
    gv: jax.Array,
    *,
    eps: float,
    q_scale: float = 1.0,
    block_s: int | None = None,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Fused backward returning `(dxq, dxk, dxv)` in `(B, S, H*D)` and `(dwq, dwk)`."""
  if gq.ndim != 4 or gk.shape != gq.shape or gv.shape != gq.shape:
    raise ValueError(
        f"Expected matching rank-4 heads-first gradients, got {gq.shape},"
        f" {gk.shape}, {gv.shape}"
    )
  heads, head_dim = gq.shape[1], gq.shape[3]
  batch, seq, hidden_dim, bs = _check_common_inputs(
      xq, xk, wq, wk, cos, sin, heads=heads, head_dim=head_dim, block_s=block_s
  )
  if vmem_limit_bytes is None:
    vmem_limit_bytes = default_vmem_limit_bytes(
        bs, heads, head_dim, xq.dtype.itemsize
    )
  if gq.shape != (batch, heads, seq, head_dim):
    raise ValueError(
        f"Expected gradient shape {(batch, heads, seq, head_dim)}, got"
        f" {gq.shape}"
    )
  row, weight, rope_table, heads_first = _make_block_specs(
      hidden_dim, heads, head_dim, bs, cos_batched=cos.shape[0] > 1
  )
  dw_spec = pl.BlockSpec((1, 1, hidden_dim), lambda b, s: (b, 0, 0))
  kernel = functools.partial(
      _qk_norm_rope_bwd_kernel,
      heads=heads,
      head_dim=head_dim,
      eps=eps,
      q_scale=q_scale,
  )
  out_shape = (
      jax.ShapeDtypeStruct((batch, seq, hidden_dim), xq.dtype),
      jax.ShapeDtypeStruct((batch, seq, hidden_dim), xk.dtype),
      jax.ShapeDtypeStruct((batch, seq, hidden_dim), gv.dtype),
      jax.ShapeDtypeStruct((batch, 1, hidden_dim), jnp.float32),
      jax.ShapeDtypeStruct((batch, 1, hidden_dim), jnp.float32),
  )
  # Grid dimension 0 (`batch`) is independent ("parallel"), while dimension 1
  # (`seq // bs`) MUST be "arbitrary" because `_qk_norm_rope_bwd_kernel` zeroes
  # `dwq_ref[b]` / `dwk_ref[b]` on `s_idx == 0` and accumulates `+=` across
  # subsequent sequence tiles.
  dxq, dxk, dxv, dwq, dwk = pl.pallas_call(
      kernel,
      grid=(batch, seq // bs),
      in_specs=[
          row,
          row,
          weight,
          weight,
          rope_table,
          rope_table,
          heads_first,
          heads_first,
          heads_first,
      ],
      out_specs=[row, row, row, dw_spec, dw_spec],
      out_shape=out_shape,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      interpret=interpret,
      name="esmc_qk_norm_rope_bwd",
  )(
      xq,
      xk,
      wq.reshape(1, hidden_dim),
      wk.reshape(1, hidden_dim),
      cos,
      sin,
      gq,
      gk,
      gv,
  )
  dwq = jnp.sum(dwq, axis=(0, 1)).astype(wq.dtype)
  dwk = jnp.sum(dwk, axis=(0, 1)).astype(wk.dtype)
  return dxq, dxk, dxv, dwq, dwk


def qk_norm_rope_reference(
    xq: jax.Array,
    xk: jax.Array,
    xv: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    heads: int,
    eps: float,
    q_scale: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Pure-JAX reference matching HuggingFace `EsmcAttention`'s prologue."""
  batch, seq, hidden_dim = xq.shape
  head_dim = hidden_dim // heads
  dtype = xq.dtype

  def layer_norm_no_bias(x: jax.Array, w: jax.Array) -> jax.Array:
    xf = x.astype(jnp.float32)
    mu = jnp.mean(xf, axis=-1, keepdims=True)
    var = jnp.mean((xf - mu) ** 2, axis=-1, keepdims=True)
    return (
        (xf - mu) * jax.lax.rsqrt(var + eps) * w.astype(jnp.float32)
    ).astype(dtype)

  def to_heads_first(t: jax.Array) -> jax.Array:
    return t.reshape(batch, seq, heads, head_dim).transpose(0, 2, 1, 3)

  def rotate_half(t: jax.Array) -> jax.Array:
    half = head_dim // 2
    return jnp.concatenate([-t[..., half:], t[..., :half]], axis=-1)

  c = cos[:, None].astype(jnp.float32)
  s = sin[:, None].astype(jnp.float32)
  q = to_heads_first(layer_norm_no_bias(xq, wq)).astype(jnp.float32)
  k = to_heads_first(layer_norm_no_bias(xk, wk)).astype(jnp.float32)
  v = to_heads_first(xv)
  q = q * c + rotate_half(q) * s
  if q_scale != 1.0:
    q = q * q_scale
  k = k * c + rotate_half(k) * s
  return q.astype(dtype), k.astype(xk.dtype), v
