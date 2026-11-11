"""Splash-style causal attention for DeepSeek MLA, in the layouts MLA produces.

Tokamax Splash wants q, k, v as ``[B, H, S, D]`` with one head dim for q and k,
so the stock MLA path pays for a q/k concat, a k_pe broadcast and several
transposes per layer. These kernels read MLA's projections as they are:

* ``q [B, S, H * (Dn + Dr)]``: ``wq_b``'s output; per head ``[q_nope | q_pe]``,
  q_pe not yet roped.
* ``kv [B, S, H * (Dn + Dv)]``: ``wkv_b``'s output; per head ``[k_nope | v]``.
* ``k_pe [B, S, Dr]``: roped, shared by every head, never broadcast.
* ``cos, sin [B, S, Dr]`` f32: per-token pair tables,
  ``cos = [c0, c0, c1, c1, ...]``, ``sin = [-s0, s0, -s1, s1, ...]``.

Logits are ``scale * (q_nope . k_nope + rope(q_pe) . k_pe)``, masked causally
and to the token's own document (``segment_ids``).

Supported shapes:

* dtype: bfloat16 (any float in interpret mode); cos/sin f32.
* Dn == Dv == 128.
* Dr: a multiple of 8, at most 128 (``[nope | pe]`` packs into 256 lanes).
* H: divisible by ``heads_per_step``, with ``heads_per_step * (128 + Dr)`` a
  multiple of 128.
* S: a multiple of ``block_q`` and ``block_kv``; both multiples of 128.
* segment_ids: any int32 ids. Tile skipping compares per-block [min, max]
  ranges, so it is exact for packed (contiguous) documents and conservative
  otherwise.
* Schedule (SMEM): 8 int32 per causal tile slot, ``B * #causal tiles`` slots
  (33 KiB per example at S = 32K with 1024 x 512 tiles); at most
  ``_SMEM_BUDGET_BYTES``.

Kernel design:

* Each kernel walks a schedule of the live tiles only (causal, and q and kv
  blocks that can share a document), built on device from the segment ids:
  a dynamic grid on TPU, as Tokamax's. Tiles fully below the diagonal and
  inside one document skip the element mask.
* QK^T is one 256-lane dot per head: ``[q_nope | rope(q_pe) | 0]`` against
  ``[k_nope | k_pe | 0]``, assembled in VMEM scratch.
* Softmax runs in base 2, with ``log2(e)`` folded into q.
* The forward ropes q_pe in VMEM once per q block. The backward ropes q once
  in XLA, and passes dq back through the rope in XLA.
* Backward, fused (default when it fits): one kernel over kv-major tiles;
  dK/dV accumulate per kv block, dQ in a resident ``[heads, S, 256]`` f32
  scratch, stored at each q block's last tile.
* Backward, split (long S): a dK/dV kernel over kv-major tiles, then a dQ
  kernel over q-major tiles. It recomputes P (7 instead of 5 matmuls per
  tile) and has no S-sized VMEM.

VMEM per grid step (bf16 inputs, ``hp = heads_per_step``, lane-padded,
double-buffered blocks counted twice):

* forward: ``2 * (bq * hp * (128 + Dr) * 2 + 2 * bq * 128 * 4
  + bkv * hp * 256 * 2 + bq * hp * 128 * 2) + 3 * hp * bq * 128 * 4
  + hp * bq * 256 * 2 + bkv * 256 * 2``; about 20 MiB at bq = bkv = 1024,
  hp = 4, Dr = 64.
* fused backward: ``hp * S * 256 * 4`` (dq scratch)
  ``+ 2 * S * hp * (128 + Dr) * 2`` (dq output block) plus the tiles; see
  ``_fused_bwd_vmem_bytes``. About 43 MiB at S = 4096, hp = 4, 1024 / 512.
* split backward: tiles only, independent of S.

``forward_tiles`` / ``backward_tiles`` pick tiles for a shape: the fused
backward at 4 heads x 1024 / 512 while it fits in 80% of VMEM (S <= 4K at
H = 128 on TPU7x), then at 2 heads x 1024 / 1024 (S = 8K), then the split
backward.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NUM_LANES = 128
_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)
_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453

# Scalar-prefetch order of the schedule arrays, matching every kernel's leading
# refs: example, q block, kv block, kind (0 padding, 1 unmasked, 2 masked),
# first / last step of a run (a q block in q-major order, a kv block in
# kv-major order), first step of an example, last tile of a q block.
_SCHEDULE_KEYS = ("b", "qi", "kj", "kind", "run_start", "run_end", "b_start",
                  "q_last")
# TPU7x scalar memory is 1 MiB; leave room for compiler state.
_SMEM_BUDGET_BYTES = 960 << 10


def _num_causal_tiles(s, block_q, block_kv):
  i = np.arange(s // block_q)[:, None]
  j = np.arange(s // block_kv)[None, :]
  return int(np.sum(j * block_kv <= i * block_q + block_q - 1))


def _find_bounds(row, valid):
  """First / last step of each run of equal ``row`` values among valid steps."""
  prev = jnp.concatenate([jnp.full((1,), -1, row.dtype), row[:-1]])
  nxt = jnp.concatenate([row[1:], jnp.full((1,), -1, row.dtype)])
  next_valid = jnp.concatenate([valid[1:], jnp.zeros((1,), bool)])
  start = valid & (row != prev)
  end = valid & ((row != nxt) | ~next_valid)
  return start.astype(jnp.int32), end.astype(jnp.int32)


def _schedule(segment_ids, block_q, block_kv, *, kv_major):
  """Live-tile schedule from segment ids ``[B, S]``.

  Returns ``(num_active, arrays)``: the number of live tiles (the grid's step
  count) and int32 ``[T]`` arrays keyed by ``_SCHEDULE_KEYS``, T being the
  data-independent bound ``B * #causal tiles``. Live tiles come first, in
  (b, qi, kj) order, or (b, kj, qi) when ``kv_major``; the rest are kind-0
  copies of the last live tile with every flag 0. Uses reductions, compares and
  one sort (no gathers).
  """
  b, s = segment_ids.shape
  nq, nk = s // block_q, s // block_kv
  seg = segment_ids.astype(jnp.int32)
  qmin = seg.reshape(b, nq, block_q).min(-1)
  qmax = seg.reshape(b, nq, block_q).max(-1)
  kmin = seg.reshape(b, nk, block_kv).min(-1)
  kmax = seg.reshape(b, nk, block_kv).max(-1)
  i = jnp.arange(nq, dtype=jnp.int32)[None, :, None]
  j = jnp.arange(nk, dtype=jnp.int32)[None, None, :]
  causal = j * block_kv <= i * block_q + block_q - 1
  below = j * block_kv + block_kv - 1 <= i * block_q
  overlap = ((qmin[:, :, None] <= kmax[:, None, :])
             & (kmin[:, None, :] <= qmax[:, :, None]))
  active = causal & overlap
  full = (below & (qmin == qmax)[:, :, None] & (kmin == kmax)[:, None, :]
          & (qmin[:, :, None] == kmin[:, None, :]))
  kind = jnp.where(full, 1, 2)
  last_j = jnp.max(jnp.where(active, j, -1), axis=-1, keepdims=True)
  q_last = (active & (j == last_j)).astype(jnp.int32)
  ex = jnp.arange(b, dtype=jnp.int32)[:, None, None]
  if kv_major:
    tile = (ex * nk + j) * nq + i
  else:
    tile = (ex * nq + i) * nk + j
  key = jnp.where(active, tile * 8 + kind * 2 + q_last,
                  jnp.iinfo(jnp.int32).max).reshape(-1)
  num_steps = b * _num_causal_tiles(s, block_q, block_kv)
  if num_steps * len(_SCHEDULE_KEYS) * 4 > _SMEM_BUDGET_BYTES:
    raise ValueError(
        f"B={b} x S={s} at {block_q} x {block_kv} tiles needs a "
        f"{num_steps * len(_SCHEDULE_KEYS) * 4 >> 10} KiB tile schedule; the "
        f"SMEM budget is {_SMEM_BUDGET_BYTES >> 10} KiB. Use larger tiles.")
  key = jnp.sort(key)[:num_steps]
  num_active = jnp.sum(active.astype(jnp.int32))
  valid = jnp.arange(num_steps, dtype=jnp.int32) < num_active
  key = jnp.where(valid, key, jnp.max(jnp.where(valid, key, -1)))
  low = key % 8
  tile = key // 8
  if kv_major:
    b_t, kj_t, qi_t = tile // (nk * nq), tile // nq % nk, tile % nq
    run = b_t * nk + kj_t
  else:
    b_t, qi_t, kj_t = tile // (nq * nk), tile // nk % nq, tile % nk
    run = b_t * nq + qi_t
  run_start, run_end = _find_bounds(run, valid)
  b_start, _ = _find_bounds(b_t, valid)
  arrays = dict(b=b_t, qi=qi_t, kj=kj_t, kind=jnp.where(valid, low // 2, 0),
                run_start=run_start, run_end=run_end, b_start=b_start,
                q_last=jnp.where(valid, low % 2, 0))
  return num_active, arrays


def _vmem_capacity_bytes() -> int | None:
  """Per-core VMEM capacity, or None when no TPU is attached (interpret/CPU)."""
  try:
    return pltpu.get_tpu_info().vmem_capacity_bytes
  except (AttributeError, RuntimeError, ValueError):
    return None


def _pair_swap(x):
  """x[..., 2i] <-> x[..., 2i + 1], as a permutation matmul (exact)."""
  n = x.shape[-1]
  rows = jax.lax.broadcasted_iota(jnp.int32, (n, n), 0)
  cols = jax.lax.broadcasted_iota(jnp.int32, (n, n), 1)
  perm = (cols == jnp.bitwise_xor(rows, 1)).astype(x.dtype)
  return jnp.dot(x, perm, preferred_element_type=jnp.float32)


def _rope(x, cos, sin):
  """Interleaved complex-pair rope: cos = [c, c, ...], sin = [-s, s, ...]."""
  return x * cos + _pair_swap(x) * sin


def _zero_pad_lanes(ref, used, *index):
  """Zeroes lanes ``[used:]`` of ``ref[index]``: the tail after ``[nope | pe]``."""
  width = ref.shape[-1]
  if used < width:
    rows = ref.shape[len(index)]
    ref[(*index, slice(None), slice(used, None))] = jnp.zeros(
        (rows, width - used), ref.dtype)


def _causal_doc_mask(i, j, seg_q, seg_kv, block_q, block_kv, *, kv_rows):
  """Keep-mask of a tile: causal and same document. ``kv_rows`` lays it out
  ``[block_kv, block_q]`` (backward), else ``[block_q, block_kv]``."""
  shape = (block_kv, block_q) if kv_rows else (block_q, block_kv)
  q_axis, k_axis = (1, 0) if kv_rows else (0, 1)
  q_pos = i * block_q + jax.lax.broadcasted_iota(jnp.int32, shape, q_axis)
  k_pos = j * block_kv + jax.lax.broadcasted_iota(jnp.int32, shape, k_axis)
  return jnp.logical_and(q_pos >= k_pos, seg_q == seg_kv)


def _fwd_kernel(
    b_ref, qi_ref, kj_ref, kind_ref, rs_ref, re_ref, bs_ref, ql_ref,
    q_ref, cos_ref, sin_ref, kv_ref, k_pe_ref, seg_q_ref, seg_kv_ref,
    o_ref, lse_ref,
    m_scr, l_scr, acc_scr, q_scr, k_scr,
    *, scale, block_q, block_kv, heads, dn, dr,
):
  """Forward over grid (head group, q-major live tile t)."""
  del b_ref, bs_ref, ql_ref
  t = pl.program_id(1)
  i, j, kind = qi_ref[t], kj_ref[t], kind_ref[t]
  pe = slice(NUM_LANES, NUM_LANES + dr)
  dq_head = dn + dr

  @pl.when(rs_ref[t] == 1)
  def _init():
    m_scr[...] = jnp.full_like(m_scr, _MASK_VALUE)
    l_scr[...] = jnp.zeros_like(l_scr)
    acc_scr[...] = jnp.zeros_like(acc_scr)
    c = scale * _LOG2E
    cos, sin = cos_ref[...], sin_ref[...]
    for h in range(heads):
      base = dq_head * h
      q_scr[h, :, :NUM_LANES] = (
          q_ref[:, base:base + dn].astype(jnp.float32) * c).astype(q_scr.dtype)
      q_pe = _rope(q_ref[:, base + dn:base + dq_head].astype(jnp.float32), cos, sin)
      q_scr[h, :, pe] = (q_pe * c).astype(q_scr.dtype)
      _zero_pad_lanes(q_scr, NUM_LANES + dr, h)
    _zero_pad_lanes(k_scr, NUM_LANES + dr)

  def body(masked):
    dims = (((1,), (1,)), ((), ()))
    k_scr[:, pe] = k_pe_ref[...]
    if masked:
      keep = _causal_doc_mask(i, j, seg_q_ref[...], seg_kv_ref[...], block_q,
                              block_kv, kv_rows=False)
    for h in range(heads):
      base = 2 * NUM_LANES * h
      k_scr[:, :NUM_LANES] = kv_ref[:, base:base + NUM_LANES]
      s = jax.lax.dot_general(
          q_scr[h], k_scr[...], dims, preferred_element_type=jnp.float32)
      if masked:
        s = jnp.where(keep, s, _MASK_VALUE)
      m_prev = m_scr[h]
      m_new = jnp.maximum(m_prev, jnp.max(s, axis=-1, keepdims=True))
      alpha = jnp.exp2(m_prev - m_new)
      p = jnp.exp2(s - m_new[:, :1])
      l_scr[h] = alpha * l_scr[h] + jnp.sum(p, axis=-1, keepdims=True)
      m_scr[h] = m_new
      acc_scr[h] = alpha * acc_scr[h] + jnp.dot(
          p.astype(kv_ref.dtype), kv_ref[:, base + NUM_LANES:base + 2 * NUM_LANES],
          preferred_element_type=jnp.float32)

  pl.when(kind == 1)(lambda: body(False))
  pl.when(kind == 2)(lambda: body(True))

  @pl.when(re_ref[t] == 1)
  def _finalize():
    for h in range(heads):
      l = l_scr[h]
      o_ref[:, h * NUM_LANES:(h + 1) * NUM_LANES] = (acc_scr[h] / l).astype(o_ref.dtype)
      lse_ref[h] = jnp.transpose((m_scr[h] + jnp.log2(l)) * _LN2)[:1, :]


def _tile_grads(i, j, kind_masked, q_ref, kv_ref, k_pe_ref, seg_q_ref,
                seg_kv_ref, lse_ref, d_ref, do_ref, q_view, kpk_scr, *, scale,
                block_q, block_kv, heads, dn, dr, on_head):
  """Shared backward tile body. For each head h: P, dS, then
  ``on_head(h, p_t, ds_t, do_t, qpk_ref)`` accumulates the wanted gradients.

  ``q_view(h)`` returns the VMEM ref holding head h's packed, roped q
  ``[block_q, 256]`` (filling it first if needed). Operands are read from refs
  at each use rather than held as values, which would spill. Scores are laid
  out ``[block_kv, block_q]``.
  """
  del dn
  dims = (((1,), (1,)), ((), ()))
  pe = slice(NUM_LANES, NUM_LANES + dr)
  c = scale * _LOG2E
  kpk_scr[:, pe] = k_pe_ref[...]
  if kind_masked:
    keep = _causal_doc_mask(i, j, seg_q_ref[...], seg_kv_ref[...], block_q,
                            block_kv, kv_rows=True)
  for h in range(heads):
    kb, ob = 2 * NUM_LANES * h, NUM_LANES * h
    kpk_scr[:, :NUM_LANES] = kv_ref[:, kb:kb + NUM_LANES]
    q_h = q_view(h)
    s_t = jax.lax.dot_general(kpk_scr[...], q_h[...], dims,
                              preferred_element_type=jnp.float32)
    p_t = jnp.exp2(s_t * c - lse_ref[h] * _LOG2E)
    if kind_masked:
      p_t = jnp.where(keep, p_t, 0.0)
    do_t = do_ref[:, ob:ob + NUM_LANES]
    dp_t = jax.lax.dot_general(
        kv_ref[:, kb + NUM_LANES:kb + 2 * NUM_LANES], do_t, dims,
        preferred_element_type=jnp.float32)
    ds_t = (p_t * (dp_t - d_ref[h]) * scale).astype(kpk_scr.dtype)
    on_head(h, p_t, ds_t, do_t, q_h)


def _bwd_kernel(
    b_ref, qi_ref, kj_ref, kind_ref, rs_ref, re_ref, bs_ref, ql_ref,
    q_ref, kv_ref, k_pe_ref, seg_q_ref, seg_kv_ref, lse_ref, d_ref, do_ref,
    dkv_ref, dk_pe_ref, dq_ref,
    dk_scr, dv_scr, dq_scr, qpk_scr, kpk_scr,
    *, scale, block_q, block_kv, heads, dn, dr,
):
  """Fused backward over grid (head group, kv-major live tile t)."""
  del b_ref
  t = pl.program_id(1)
  i, j, kind = qi_ref[t], kj_ref[t], kind_ref[t]
  q_rows = pl.ds(pl.multiple_of(i * block_q, block_q), block_q)
  pe = slice(NUM_LANES, NUM_LANES + dr)
  dq_head = dn + dr
  lhs_t = (((0,), (0,)), ((), ()))

  @pl.when(bs_ref[t] == 1)
  def _init_example():
    dq_scr[...] = jnp.zeros_like(dq_scr)
    _zero_pad_lanes(qpk_scr, NUM_LANES + dr)
    _zero_pad_lanes(kpk_scr, NUM_LANES + dr)

  @pl.when(rs_ref[t] == 1)
  def _init_kv_block():
    dk_scr[...] = jnp.zeros_like(dk_scr)
    dv_scr[...] = jnp.zeros_like(dv_scr)

  def q_view(h):
    qpk_scr[:, :dq_head] = q_ref[:, dq_head * h:dq_head * (h + 1)]
    return qpk_scr

  def on_head(h, p_t, ds_t, do_t, q_h):
    dv_scr[h] += jnp.dot(p_t.astype(do_t.dtype), do_t,
                         preferred_element_type=jnp.float32)
    dk_scr[h] += jnp.dot(ds_t, q_h[...], preferred_element_type=jnp.float32)
    dq_scr[h, q_rows, :] += jax.lax.dot_general(
        ds_t, kpk_scr[...], lhs_t, preferred_element_type=jnp.float32)

  args = (q_ref, kv_ref, k_pe_ref, seg_q_ref, seg_kv_ref, lse_ref, d_ref,
          do_ref, q_view, kpk_scr)
  kw = dict(scale=scale, block_q=block_q, block_kv=block_kv, heads=heads,
            dn=dn, dr=dr, on_head=on_head)
  pl.when(kind == 1)(lambda: _tile_grads(i, j, False, *args, **kw))
  pl.when(kind == 2)(lambda: _tile_grads(i, j, True, *args, **kw))

  @pl.when(re_ref[t] == 1)
  def _store_dkv():
    for h in range(heads):
      kb = 2 * NUM_LANES * h
      dkv_ref[:, kb:kb + NUM_LANES] = dk_scr[h, :, :NUM_LANES].astype(dkv_ref.dtype)
      dkv_ref[:, kb + NUM_LANES:kb + 2 * NUM_LANES] = dv_scr[h].astype(dkv_ref.dtype)
      dk_pe_ref[h] = dk_scr[h, :, pe].astype(dk_pe_ref.dtype)

  @pl.when(ql_ref[t] == 1)
  def _store_dq():
    for h in range(heads):
      qb = dq_head * h
      dq_ref[q_rows, qb:qb + dq_head] = (
          dq_scr[h, q_rows, :dq_head].astype(dq_ref.dtype))


def _bwd_dkv_kernel(
    b_ref, qi_ref, kj_ref, kind_ref, rs_ref, re_ref, bs_ref, ql_ref,
    q_ref, kv_ref, k_pe_ref, seg_q_ref, seg_kv_ref, lse_ref, d_ref, do_ref,
    dkv_ref, dk_pe_ref,
    dk_scr, dv_scr, qpk_scr, kpk_scr,
    *, scale, block_q, block_kv, heads, dn, dr,
):
  """Split backward, dK/dV half: grid (head group, kv-major live tile t)."""
  del b_ref, bs_ref, ql_ref
  t = pl.program_id(1)
  i, j, kind = qi_ref[t], kj_ref[t], kind_ref[t]
  pe = slice(NUM_LANES, NUM_LANES + dr)
  dq_head = dn + dr

  @pl.when(rs_ref[t] == 1)
  def _init():
    dk_scr[...] = jnp.zeros_like(dk_scr)
    dv_scr[...] = jnp.zeros_like(dv_scr)
    _zero_pad_lanes(qpk_scr, NUM_LANES + dr)
    _zero_pad_lanes(kpk_scr, NUM_LANES + dr)

  def q_view(h):
    qpk_scr[:, :dq_head] = q_ref[:, dq_head * h:dq_head * (h + 1)]
    return qpk_scr

  def on_head(h, p_t, ds_t, do_t, q_h):
    dv_scr[h] += jnp.dot(p_t.astype(do_t.dtype), do_t,
                         preferred_element_type=jnp.float32)
    dk_scr[h] += jnp.dot(ds_t, q_h[...], preferred_element_type=jnp.float32)

  args = (q_ref, kv_ref, k_pe_ref, seg_q_ref, seg_kv_ref, lse_ref, d_ref,
          do_ref, q_view, kpk_scr)
  kw = dict(scale=scale, block_q=block_q, block_kv=block_kv, heads=heads,
            dn=dn, dr=dr, on_head=on_head)
  pl.when(kind == 1)(lambda: _tile_grads(i, j, False, *args, **kw))
  pl.when(kind == 2)(lambda: _tile_grads(i, j, True, *args, **kw))

  @pl.when(re_ref[t] == 1)
  def _store():
    for h in range(heads):
      kb = 2 * NUM_LANES * h
      dkv_ref[:, kb:kb + NUM_LANES] = dk_scr[h, :, :NUM_LANES].astype(dkv_ref.dtype)
      dkv_ref[:, kb + NUM_LANES:kb + 2 * NUM_LANES] = dv_scr[h].astype(dkv_ref.dtype)
      dk_pe_ref[h] = dk_scr[h, :, pe].astype(dk_pe_ref.dtype)


def _bwd_dq_kernel(
    b_ref, qi_ref, kj_ref, kind_ref, rs_ref, re_ref, bs_ref, ql_ref,
    q_ref, kv_ref, k_pe_ref, seg_q_ref, seg_kv_ref, lse_ref, d_ref, do_ref,
    dq_ref,
    dq_scr, qpk_scr, kpk_scr,
    *, scale, block_q, block_kv, heads, dn, dr,
):
  """Split backward, dQ half: grid (head group, q-major live tile t)."""
  del b_ref, bs_ref, ql_ref
  t = pl.program_id(1)
  i, j, kind = qi_ref[t], kj_ref[t], kind_ref[t]
  dq_head = dn + dr
  lhs_t = (((0,), (0,)), ((), ()))

  @pl.when(rs_ref[t] == 1)
  def _init():
    dq_scr[...] = jnp.zeros_like(dq_scr)
    _zero_pad_lanes(kpk_scr, NUM_LANES + dr)
    for h in range(heads):
      qpk_scr[h, :, :dq_head] = q_ref[:, dq_head * h:dq_head * (h + 1)]
      _zero_pad_lanes(qpk_scr, NUM_LANES + dr, h)

  def on_head(h, p_t, ds_t, do_t, q_h):
    del p_t, do_t, q_h
    dq_scr[h] += jax.lax.dot_general(
        ds_t, kpk_scr[...], lhs_t, preferred_element_type=jnp.float32)

  args = (q_ref, kv_ref, k_pe_ref, seg_q_ref, seg_kv_ref, lse_ref, d_ref,
          do_ref, lambda h: qpk_scr.at[h], kpk_scr)
  kw = dict(scale=scale, block_q=block_q, block_kv=block_kv, heads=heads,
            dn=dn, dr=dr, on_head=on_head)
  pl.when(kind == 1)(lambda: _tile_grads(i, j, False, *args, **kw))
  pl.when(kind == 2)(lambda: _tile_grads(i, j, True, *args, **kw))

  @pl.when(re_ref[t] == 1)
  def _store():
    for h in range(heads):
      qb = dq_head * h
      dq_ref[:, qb:qb + dq_head] = dq_scr[h, :, :dq_head].astype(dq_ref.dtype)


def _rope_kernel(q_ref, cos_ref, sin_ref, o_ref, *, heads, dn, dr, sign):
  """Pair rope on each head's q_pe lanes of one ``[rows, heads * (dn + dr)]``
  block; ``sign = -1`` applies its transpose."""
  cos, sin = cos_ref[...], sign * sin_ref[...]
  for h in range(heads):
    base = (dn + dr) * h
    o_ref[:, base:base + dn] = q_ref[:, base:base + dn]
    o_ref[:, base + dn:base + dn + dr] = _rope(
        q_ref[:, base + dn:base + dn + dr].astype(jnp.float32), cos, sin
    ).astype(o_ref.dtype)


def _rope_packed(q, cos, sin, dr, *, sign, interpret):
  """The pair rope on every head's q_pe lanes of packed q ``[B, S, H*(128+Dr)]``.

  A Pallas kernel rather than XLA: XLA's fusion of the pair swap on this
  layout runs several times slower than the HBM roofline.
  """
  b, s, width = q.shape
  dn = NUM_LANES
  h = width // (dn + dr)
  hp = next(n for n in (4, 2, 1)
            if h % n == 0 and (n * (dn + dr)) % NUM_LANES == 0)
  rows = next(r for r in (512, 256, 128) if s % r == 0)
  return pl.pallas_call(
      functools.partial(_rope_kernel, heads=hp, dn=dn, dr=dr, sign=sign),
      grid=(b, s // rows, h // hp),
      in_specs=[
          pl.BlockSpec((None, rows, hp * (dn + dr)), lambda b_, r, g: (b_, r, g)),
          pl.BlockSpec((None, rows, dr), lambda b_, r, g: (b_, r, 0)),
          pl.BlockSpec((None, rows, dr), lambda b_, r, g: (b_, r, 0)),
      ],
      out_specs=pl.BlockSpec((None, rows, hp * (dn + dr)), lambda b_, r, g: (b_, r, g)),
      out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel")),
      interpret=interpret, name="mla_rope_q",
  )(q, cos, sin)


def _round_up(x, m):
  return -(-x // m) * m


def _fused_bwd_vmem_bytes(s, block_q, block_kv, heads, dr, itemsize):
  """VMEM of the fused backward: S-resident terms plus double-buffered tiles."""
  q_w = _round_up(heads * (NUM_LANES + dr), NUM_LANES)
  resident = (heads * s * 2 * NUM_LANES * 4      # dq scratch, f32
              + 2 * s * q_w * itemsize)          # dq output block
  tiles = 2 * (block_q * q_w * itemsize                         # q
               + 2 * block_kv * heads * 2 * NUM_LANES * itemsize  # kv, dkv
               + block_q * heads * NUM_LANES * itemsize         # do
               + heads * block_kv * NUM_LANES * 4)              # dk_pe
  scratch = (heads * block_kv * 3 * NUM_LANES * 4                # dk, dv
             + (block_q + block_kv) * 2 * NUM_LANES * itemsize)  # packed q, k
  return resident + tiles + scratch


# Used when no TPU is attached (interpret mode, or a host without a JAX TPU
# backend): TPU7x per-core VMEM.
_DEFAULT_VMEM_BYTES = 64 << 20


def _fit_block(s, block):
  """The largest of block, block / 2, ..., 128 that divides S."""
  while block > NUM_LANES and s % block:
    block //= 2
  return block


def _heads_per_step(heads, dr, most):
  return next(n for n in (4, 2, 1) if n <= most and heads % n == 0
              and (n * (NUM_LANES + dr)) % NUM_LANES == 0)


def forward_tiles(s, heads, dr):
  """``(block_q, block_kv, heads_per_step)`` for ``mla_splash_fwd``."""
  return _fit_block(s, 1024), _fit_block(s, 1024), _heads_per_step(heads, dr, 4)


def backward_tiles(s, heads, dr, itemsize=2):
  """``(block_q, block_kv, heads_per_step, split)`` for ``mla_splash_bwd``.

  The fused backward at 4 heads x 1024 / 512 when it fits, else at 2 heads x
  1024 / 1024, else the split backward at 4 heads x 1024 / 1024 (the fastest
  of each on TPU7x for S 2K-32K, H 128).
  """
  cap = _vmem_capacity_bytes() or _DEFAULT_VMEM_BYTES
  for most, bq, bkv in ((4, 1024, 512), (2, 1024, 1024)):
    hp = _heads_per_step(heads, dr, most)
    bq, bkv = _fit_block(s, bq), _fit_block(s, bkv)
    if _fused_bwd_vmem_bytes(s, bq, bkv, hp, dr, itemsize) <= 0.8 * cap:
      return bq, bkv, hp, False
  return (_fit_block(s, 1024), _fit_block(s, 1024),
          _heads_per_step(heads, dr, 4), True)


def _check(q, kv, block_q, block_kv, heads, dr):
  """Validates shapes; returns ``(B, S, H, Dn)``."""
  if dr % 8 or not 0 < dr <= NUM_LANES:
    raise ValueError(f"Dr={dr} must be a multiple of 8 in [8, {NUM_LANES}]")
  if block_q % NUM_LANES or block_kv % NUM_LANES:
    raise ValueError(
        f"block_q={block_q} and block_kv={block_kv} must be multiples of "
        f"{NUM_LANES}")
  b, s, hq = q.shape
  dn = NUM_LANES
  h = hq // (dn + dr)
  if hq != h * (dn + dr) or kv.shape != (b, s, h * 2 * NUM_LANES):
    raise ValueError(f"bad shapes: q {q.shape}, kv {kv.shape}, Dr {dr}")
  if heads < 1 or h % heads or (heads * (dn + dr)) % NUM_LANES:
    raise ValueError(
        f"heads_per_step={heads} must divide H={h} and make "
        f"heads * {dn + dr} a multiple of {NUM_LANES}")
  if s % block_q or s % block_kv:
    raise ValueError(f"S={s} must divide by block_q={block_q}, block_kv={block_kv}")
  return b, s, h, dn


def _sched_spec(fn):
  """BlockSpec index map on grid (g, t) reading the schedule refs."""
  def index_map(g, t, b_ref, qi_ref, kj_ref, *_):
    return fn(g, b_ref[t], qi_ref[t], kj_ref[t])
  return index_map


def _grid_steps(num_active, sched, interpret):
  # Interpret mode has no dynamic grid; it walks the padded schedule (kind-0
  # steps do nothing).
  return sched["b"].shape[0] if interpret else num_active


@functools.partial(
    jax.jit, static_argnames=("scale", "block_q", "block_kv", "heads_per_step",
                                "interpret"))
def mla_splash_fwd(q, kv, k_pe, cos, sin, segment_ids, *, scale,
                   block_q=1024, block_kv=1024, heads_per_step=4,
                   interpret=False):
  """Causal, document-masked MLA attention on packed q (see module docstring).

  Args:
    q: ``[B, S, H * (128 + Dr)]`` from ``wq_b``; per head ``[q_nope | q_pe]``,
      q_pe before rope.
    kv: ``[B, S, H * 256]``, per head ``[k_nope | v]``.
    k_pe: ``[B, S, Dr]`` after rope.
    cos, sin: ``[B, S, Dr]`` f32 per-token rope tables in pair layout,
      ``cos = [c0, c0, c1, c1, ...]`` and ``sin = [-s0, s0, -s1, s1, ...]``.
    segment_ids: ``[B, S]`` document ids.
    scale: Softmax scale, usually ``1 / sqrt(128 + Dr)``.
    block_q, block_kv: Q and K/V tile lengths.
    heads_per_step: Heads processed per grid step.
    interpret: Run in Pallas interpret mode (CPU).

  Returns:
    ``(o, lse)``: o ``[B, S, H * 128]`` (feeds ``wo``) and the log-sum-exp
    ``[B, H, 1, S]`` f32 for the backward.
  """
  dr = k_pe.shape[-1]
  heads = heads_per_step
  b, s, h, dn = _check(q, kv, block_q, block_kv, heads, dr)
  width = 2 * NUM_LANES
  segment_ids = segment_ids.astype(jnp.int32)
  num_active, sched = _schedule(segment_ids, block_q, block_kv, kv_major=False)

  in_specs = [
      pl.BlockSpec((None, block_q, heads * (dn + dr)), _sched_spec(lambda g, b_, i, j: (b_, i, g))),
      pl.BlockSpec((None, block_q, dr), _sched_spec(lambda g, b_, i, j: (b_, i, 0))),
      pl.BlockSpec((None, block_q, dr), _sched_spec(lambda g, b_, i, j: (b_, i, 0))),
      pl.BlockSpec((None, block_kv, heads * width), _sched_spec(lambda g, b_, i, j: (b_, j, g))),
      pl.BlockSpec((None, block_kv, dr), _sched_spec(lambda g, b_, i, j: (b_, j, 0))),
      pl.BlockSpec((None, block_q, 1), _sched_spec(lambda g, b_, i, j: (b_, i, 0))),
      pl.BlockSpec((None, 1, block_kv), _sched_spec(lambda g, b_, i, j: (b_, 0, j))),
  ]
  out_specs = [
      pl.BlockSpec((None, block_q, heads * NUM_LANES), _sched_spec(lambda g, b_, i, j: (b_, i, g))),
      pl.BlockSpec((None, heads, 1, block_q), _sched_spec(lambda g, b_, i, j: (b_, g, 0, i))),
  ]
  out_shape = [jax.ShapeDtypeStruct((b, s, h * NUM_LANES), q.dtype),
               jax.ShapeDtypeStruct((b, h, 1, s), jnp.float32)]
  operands = (q, cos, sin, kv, k_pe, segment_ids[:, :, None],
              segment_ids[:, None, :])
  tile_heads = sched["b"].shape[0] * h
  cost = pl.CostEstimate(
      flops=tile_heads * 2 * block_q * block_kv * (width + NUM_LANES),
      transcendentals=tile_heads * block_q * block_kv,
      bytes_accessed=sum(x.size * x.dtype.itemsize for x in operands)
      + sum(int(np.prod(o.shape)) * o.dtype.itemsize for o in out_shape))
  cap = _vmem_capacity_bytes()
  return pl.pallas_call(
      functools.partial(_fwd_kernel, scale=scale, block_q=block_q,
                        block_kv=block_kv, heads=heads, dn=dn, dr=dr),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(_SCHEDULE_KEYS),
          grid=(h // heads, _grid_steps(num_active, sched, interpret)),
          in_specs=in_specs, out_specs=out_specs,
          scratch_shapes=[
              pltpu.VMEM((heads, block_q, NUM_LANES), jnp.float32),
              pltpu.VMEM((heads, block_q, NUM_LANES), jnp.float32),
              pltpu.VMEM((heads, block_q, NUM_LANES), jnp.float32),
              pltpu.VMEM((heads, block_q, width), q.dtype),
              pltpu.VMEM((block_kv, width), q.dtype),
          ]),
      out_shape=out_shape,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          vmem_limit_bytes=None if cap is None else int(0.9 * cap)),
      cost_estimate=cost, interpret=interpret, name="mla_splash_fwd",
  )(*(sched[k] for k in _SCHEDULE_KEYS), *operands)


@functools.partial(
    jax.jit, static_argnames=("scale", "block_q", "block_kv", "heads_per_step",
                                "split", "interpret"))
def mla_splash_bwd(q, kv, k_pe, cos, sin, segment_ids, o, lse, do, *,
                   scale, block_q=1024, block_kv=1024, heads_per_step=2,
                   split=None, interpret=False):
  """Backward of ``mla_splash_fwd``.

  Args:
    q, kv, k_pe, cos, sin, segment_ids, scale: As for ``mla_splash_fwd``.
    o, lse: ``mla_splash_fwd``'s outputs.
    do: ``[B, S, H * 128]``, the gradient of ``o``.
    block_q, block_kv, heads_per_step, interpret: Tiling, as for the forward.
    split: Use the two-kernel backward (dK/dV, then dQ) instead of the fused
      one. ``None`` picks the fused kernel when ``_fused_bwd_vmem_bytes`` is
      at most 80% of VMEM, and the split one otherwise.

  Returns:
    ``(dq, dkv, dk_pe)``: dq in q's packed layout, already passed back through
    the rope; dkv in kv's layout; dk_pe ``[B, S, Dr]`` for the roped k_pe,
    summed over heads.
  """
  dr = k_pe.shape[-1]
  heads = heads_per_step
  b, s, h, dn = _check(q, kv, block_q, block_kv, heads, dr)
  width = 2 * NUM_LANES
  qw = heads * (dn + dr)
  segment_ids = segment_ids.astype(jnp.int32)
  d = jnp.einsum("bshd,bshd->bhs",
                 do.reshape(b, s, h, NUM_LANES).astype(jnp.float32),
                 o.reshape(b, s, h, NUM_LANES).astype(jnp.float32))[:, :, None, :]
  cap = _vmem_capacity_bytes()
  if split is None:
    split = cap is not None and _fused_bwd_vmem_bytes(
        s, block_q, block_kv, heads, dr, q.dtype.itemsize) > 0.8 * cap
  params = pltpu.CompilerParams(
      dimension_semantics=("parallel", "arbitrary"),
      vmem_limit_bytes=None if cap is None else int(0.9 * cap))
  kernel_args = dict(scale=scale, block_q=block_q, block_kv=block_kv,
                     heads=heads, dn=dn, dr=dr)
  # q is roped once here rather than in every tile; the kernels return dq
  # w.r.t. the roped q, and rope's transpose is the rope with sin negated.
  operands = (_rope_packed(q, cos, sin, dr, sign=1, interpret=interpret), kv, k_pe,
              segment_ids[:, None, :], segment_ids[:, :, None], lse, d, do)
  in_specs = [
      pl.BlockSpec((None, block_q, qw), _sched_spec(lambda g, b_, i, j: (b_, i, g))),
      pl.BlockSpec((None, block_kv, heads * width), _sched_spec(lambda g, b_, i, j: (b_, j, g))),
      pl.BlockSpec((None, block_kv, dr), _sched_spec(lambda g, b_, i, j: (b_, j, 0))),
      pl.BlockSpec((None, 1, block_q), _sched_spec(lambda g, b_, i, j: (b_, 0, i))),
      pl.BlockSpec((None, block_kv, 1), _sched_spec(lambda g, b_, i, j: (b_, j, 0))),
      pl.BlockSpec((None, heads, 1, block_q), _sched_spec(lambda g, b_, i, j: (b_, g, 0, i))),
      pl.BlockSpec((None, heads, 1, block_q), _sched_spec(lambda g, b_, i, j: (b_, g, 0, i))),
      pl.BlockSpec((None, block_q, heads * NUM_LANES), _sched_spec(lambda g, b_, i, j: (b_, i, g))),
  ]
  dkv_out_specs = [
      pl.BlockSpec((None, block_kv, heads * width), _sched_spec(lambda g, b_, i, j: (b_, j, g))),
      pl.BlockSpec((None, heads, block_kv, dr), _sched_spec(lambda g, b_, i, j: (b_, g, j, 0))),
  ]
  dkv_shapes = [jax.ShapeDtypeStruct(kv.shape, kv.dtype),
                jax.ShapeDtypeStruct((b, h, s, dr), jnp.float32)]
  dkv_scratch = [pltpu.VMEM((heads, block_kv, width), jnp.float32),
                 pltpu.VMEM((heads, block_kv, NUM_LANES), jnp.float32)]
  in_bytes = sum(x.size * x.dtype.itemsize for x in operands)
  out_bytes = (2 * kv.size + q.size) * q.dtype.itemsize + b * h * s * dr * 4

  def cost(num_steps, matmuls):
    tile_heads = num_steps * h
    return pl.CostEstimate(
        flops=tile_heads * 2 * block_q * block_kv * matmuls * NUM_LANES,
        transcendentals=tile_heads * block_q * block_kv,
        bytes_accessed=in_bytes + out_bytes)

  num_kv_major, kv_sched = _schedule(segment_ids, block_q, block_kv,
                                     kv_major=True)
  kv_grid = (h // heads, _grid_steps(num_kv_major, kv_sched, interpret))
  kv_steps = kv_sched["b"].shape[0]
  kv_prefetch = tuple(kv_sched[k] for k in _SCHEDULE_KEYS)

  if not split:
    # Matmul lanes per tile and head: QK^T 256, dV 128, dP 128, dK 256, dQ 256.
    dkv, dk_pe_h, dq = pl.pallas_call(
        functools.partial(_bwd_kernel, **kernel_args),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(_SCHEDULE_KEYS), grid=kv_grid,
            in_specs=in_specs,
            out_specs=dkv_out_specs + [
                pl.BlockSpec((None, s, qw), _sched_spec(lambda g, b_, i, j: (b_, 0, g)))],
            scratch_shapes=dkv_scratch + [
                pltpu.VMEM((heads, s, width), jnp.float32),
                pltpu.VMEM((block_q, width), q.dtype),
                pltpu.VMEM((block_kv, width), q.dtype)]),
        out_shape=dkv_shapes + [jax.ShapeDtypeStruct(q.shape, q.dtype)],
        compiler_params=params, cost_estimate=cost(kv_steps, 8),
        interpret=interpret, name="mla_splash_bwd",
    )(*kv_prefetch, *operands)
  else:
    # dK/dV: QK^T 256, dV 128, dP 128, dK 256 lanes; dQ: QK^T, dP, dQ.
    dkv, dk_pe_h = pl.pallas_call(
        functools.partial(_bwd_dkv_kernel, **kernel_args),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(_SCHEDULE_KEYS), grid=kv_grid,
            in_specs=in_specs, out_specs=dkv_out_specs,
            scratch_shapes=dkv_scratch + [
                pltpu.VMEM((block_q, width), q.dtype),
                pltpu.VMEM((block_kv, width), q.dtype)]),
        out_shape=dkv_shapes, compiler_params=params,
        cost_estimate=cost(kv_steps, 6), interpret=interpret,
        name="mla_splash_bwd_dkv",
    )(*kv_prefetch, *operands)
    num_q_major, q_sched = _schedule(segment_ids, block_q, block_kv,
                                     kv_major=False)
    dq = pl.pallas_call(
        functools.partial(_bwd_dq_kernel, **kernel_args),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(_SCHEDULE_KEYS),
            grid=(h // heads, _grid_steps(num_q_major, q_sched, interpret)),
            in_specs=in_specs,
            out_specs=pl.BlockSpec((None, block_q, qw), _sched_spec(lambda g, b_, i, j: (b_, i, g))),
            scratch_shapes=[pltpu.VMEM((heads, block_q, width), jnp.float32),
                            pltpu.VMEM((heads, block_q, width), q.dtype),
                            pltpu.VMEM((block_kv, width), q.dtype)]),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        compiler_params=params, cost_estimate=cost(q_sched["b"].shape[0], 5),
        interpret=interpret, name="mla_splash_bwd_dq",
    )(*(q_sched[k] for k in _SCHEDULE_KEYS), *operands)
  return (_rope_packed(dq, cos, sin, dr, sign=-1, interpret=interpret), dkv,
          dk_pe_h.sum(axis=1).astype(k_pe.dtype))


def mla_reference(q, kv, k_pe, cos, sin, segment_ids, *, scale):
  """Dense f32 reference for the packed-q entry. Returns ``(o, lse)``."""
  b, s, hq = q.shape
  dr = k_pe.shape[-1]
  h = hq // (NUM_LANES + dr)
  qh = q.reshape(b, s, h, NUM_LANES + dr).astype(jnp.float32)
  q_nope, q_pe = qh[..., :NUM_LANES], qh[..., NUM_LANES:]
  q_pe = q_pe * cos[:, :, None] + (
      q_pe.reshape(b, s, h, dr // 2, 2)[..., ::-1].reshape(b, s, h, dr)
  ) * sin[:, :, None]
  return _reference_unpacked(
      q_nope.reshape(b, s, h * NUM_LANES), jnp.transpose(q_pe, (0, 2, 1, 3)),
      kv, k_pe, segment_ids, scale=scale)


def _reference_unpacked(q_nope, q_pe, kv, k_pe, segment_ids, *, scale):
  """Dense f32 reference on unpacked, roped q_pe ``[B, H, S, Dr]``."""
  b, s, hdn = q_nope.shape
  h = q_pe.shape[1]
  qn = q_nope.reshape(b, s, h, NUM_LANES).astype(jnp.float32)
  qp = q_pe.astype(jnp.float32)                               # [B, H, S, Dr]
  kvh = kv.reshape(b, s, h, 2, NUM_LANES).astype(jnp.float32)
  kn, v = kvh[:, :, :, 0], kvh[:, :, :, 1]                    # [B, S, H, 128]
  kp = k_pe.astype(jnp.float32)                               # [B, S, Dr]
  logits = jnp.einsum("bqhd,bkhd->bhqk", qn, kn)
  logits += jnp.einsum("bhqd,bkd->bhqk", qp, kp)
  logits *= scale
  pos = jnp.arange(s)
  mask = (pos[:, None] >= pos[None, :])[None, None] & (
      segment_ids[:, None, :, None] == segment_ids[:, None, None, :]
  )
  logits = jnp.where(mask, logits, -jnp.inf)
  lse = jax.nn.logsumexp(logits, axis=-1)                     # [B, H, S]
  p = jnp.exp(logits - lse[..., None])
  o = jnp.einsum("bhqk,bkhd->bqhd", p, v).reshape(b, s, h * NUM_LANES)
  return o, lse[:, :, None, :]
