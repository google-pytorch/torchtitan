"""Transposed Splash attention backward for causal, packed-document GQA attention (TPU, Pallas/Mosaic).

Shape suffixes (logical dims, see .claude/CLAUDE.md):
  B batch, S sequence, H query heads, N kv heads, G = H // N query heads per kv head, D head dim (128),
  Q rows of a q block, C columns of a kv block, I q blocks, J kv blocks, T schedule steps.

Why transposed
--------------
On the 256x256 weight-stationary MXU a matmul streams LHS rows against a latched RHS (weights) tile. Tokamax's dkv
kernel computes dV = P^T dO, dK = dS^T Q and dQ = dS K. All three have a 128-wide output (D), so every latched
weight tile is half empty. This kernel computes S = Q K^T with rows = q, which makes P and dS natural [Q, C] weight
tiles, and produces dK and dV transposed:

  S     = Q    [Q, D] @ K^T [D, C]      (K^T latched, shared by the G heads)
  P     = exp2(scale * log2(e) * S - lse2) * mask   (lse2: the forward's log2-domain logsumexp)
  dP    = dO   [Q, D] @ V^T [D, C]
  dS    = P * (dP - di)               (di = rowsum(o * dO), computed in the kernel once per q block)
  dV^T += dO^T [D, Q] @ P   [Q, C]      (P latched; D = 128 streamed rows; full-width output)
  dK^T += Q^T  [D, Q] @ dS  [Q, C]      (dS latched)
  dQ   += dS   [Q, C] @ K   [C, D]

S and dP still contract over D = 128 and dQ still has a 128-wide output; those half-filled matmuls are 3/5 of the
MXU pushes and bound the kernel (README of the prototype: /data/users/pianpwk/tpu_runs/splash_tbwd_1103).

Schedule and accumulation
-------------------------
One pallas_call covers the whole batch. The grid is (N, T). The T steps walk a schedule, built on device from
the segment ids (`build_schedule`), of the active (example, q block, kv block) tiles in that lexicographic order,
padded with no-op steps to the data-independent bound B * nb * (nb + 1) / 2 (nb = S / BLOCK). Because of that order:
  - dQ of one (example, q block) accumulates in VMEM over its run of steps and is written once;
  - dK^T / dV^T of one (example, kv head) stay resident in VMEM as [D, S] f32 for the whole sequence, and so do
    K / V; there are no HBM partial sums and no reduction op;
  - each step processes all G query heads of its kv head (K^T / V^T / K loaded once per step).
A tile is skipped when its q and kv blocks share no segment id ([min, max] interval test, as Tokamax's
`_segment_block_overlap`; conservative, so ids need not be sorted). Tiles that are entirely inside one document and
strictly below the diagonal skip the element mask; every other active tile is computed whole, with the mask.

Limits
------
Causal self-attention with segment ids only (no window, no CP, no soft cap, no dropout), D = 128, G in (8, 16),
S a multiple of 512 and at most 4096, bf16 inputs. Code size: the kernel unrolls _HEAD_UNROLL heads per loop
iteration in each of its 2 tile-kind branches. An 81k-bundle variant overflowed TPU7x instruction memory and ran
3.5x slower (53k was still fine), so check the final LLO bundle count (--xla_jf_dump_to) before raising the unroll
or adding branches.
"""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

HEAD_DIM = 128
BLOCK = 512
MAX_SEQ_LEN = 4096
SUPPORTED_GROUP_SIZES = (8, 16)
NUM_LANES = 128
# Query heads unrolled per loop iteration, capped at G (so it divides every G in SUPPORTED_GROUP_SIZES). This sets the
# kernel's code size (see "Limits" above). Q235 shapes (G = 16), 16 examples:
# 16 -> 12.78 ms, 8 -> 13.04 ms.
_HEAD_UNROLL = 16
_VMEM_LIMIT_BYTES = 60 << 20

# A BlockSpec index map's result: block indices, one per array dim.
_BlockIndex = tuple[jax.Array | int, ...]

# Scalar-prefetch order of the `build_schedule` arrays, matching `_kernel`'s leading refs.
_SCHEDULE_KEYS = ("b", "qi", "kj", "kind", "q_bounds_start", "q_bounds_end", "b_bounds_start", "b_bounds_end")


def build_schedule(segment_ids_BS: jax.Array) -> dict[str, jax.Array]:
  """Tile schedule [T] for `splash_tbwd_call`, built from (traced) segment ids [B, S].

  Returns a dict of int32 [T] arrays: b, qi, kj (example, q block, kv block) and kind (0: padding step, no compute;
  1: every pair causal and same-document, no element mask; 2: masked tile), plus the accumulator bounds (as
  Tokamax's `find_bounds`, 1 on the first / last step of a run): q_bounds_start / q_bounds_end for each
  (example, q block) run (dQ) and b_bounds_start / b_bounds_end for each example (dK^T / dV^T). Active tiles come
  first in (b, qi, kj) order; the remaining steps, up to the data-independent bound of every causal tile, are
  kind-0 copies of the last active tile's indices (so they trigger no DMA) with all bounds 0. Uses only
  reductions, compares and one sort (no gathers, which XLA TPU may offload to SparseCore).
  """
  num_b, s = segment_ids_BS.shape
  num_blocks = s // BLOCK
  seg_BIQ = segment_ids_BS.astype(jnp.int32).reshape(num_b, num_blocks, BLOCK)
  min_seg_BI = seg_BIQ.min(axis=-1)
  max_seg_BI = seg_BIQ.max(axis=-1)
  qi_IJ = jnp.arange(num_blocks, dtype=jnp.int32)[:, None]
  kj_IJ = jnp.arange(num_blocks, dtype=jnp.int32)[None, :]
  # A causal tile may hold a same-document pair when the [min, max] segment-id ranges of its q block and kv block
  # overlap (as Tokamax's `_segment_block_overlap`; conservative, so ids need not be sorted).
  may_have_live_pairs_BIJ = ((max_seg_BI[:, :, None] >= min_seg_BI[:, None, :])
                             & (max_seg_BI[:, None, :] >= min_seg_BI[:, :, None]) & (kj_IJ <= qi_IJ))
  # Every pair is live when both blocks lie inside the same single document, strictly below the diagonal.
  single_doc_BI = min_seg_BI == max_seg_BI
  all_pairs_live_BIJ = ((kj_IJ < qi_IJ) & single_doc_BI[:, :, None] & single_doc_BI[:, None, :]
                        & (min_seg_BI[:, :, None] == min_seg_BI[:, None, :]))
  kind_BIJ = jnp.where(all_pairs_live_BIJ, 1, 2)

  # Compaction: sort unique keys (tile index * 8 + kind); tiles without live pairs sort last.
  tile_BIJ = (jnp.arange(num_b, dtype=jnp.int32)[:, None, None] * num_blocks + qi_IJ) * num_blocks + kj_IJ
  big = jnp.iinfo(jnp.int32).max
  key = jnp.where(may_have_live_pairs_BIJ, tile_BIJ * 8 + kind_BIJ, big).reshape(-1)
  num_steps = num_b * num_blocks * (num_blocks + 1) // 2  # every causal tile: a data-independent grid size
  key_T = jnp.sort(key)[:num_steps]
  num_active = jnp.sum(may_have_live_pairs_BIJ.astype(jnp.int32))
  valid_T = jnp.arange(num_steps, dtype=jnp.int32) < num_active
  last_key = jnp.max(jnp.where(valid_T, key_T, -1))
  key_T = jnp.where(valid_T, key_T, last_key)
  kind_T = jnp.where(valid_T, key_T % 8, 0)
  tile_T = key_T // 8
  b_T = tile_T // (num_blocks * num_blocks)
  qi_T = tile_T // num_blocks % num_blocks
  kj_T = tile_T % num_blocks

  q_bounds_start, q_bounds_end = _find_bounds(b_T * num_blocks + qi_T, valid_T)
  b_bounds_start, b_bounds_end = _find_bounds(b_T, valid_T)
  return dict(b=b_T, qi=qi_T, kj=kj_T, kind=kind_T, q_bounds_start=q_bounds_start, q_bounds_end=q_bounds_end,
              b_bounds_start=b_bounds_start, b_bounds_end=b_bounds_end)


def _find_bounds(row_T: jax.Array, valid_T: jax.Array) -> tuple[jax.Array, jax.Array]:
  """First / last step of each run of equal `row_T` values among the valid steps, as int32 [T].

  Same as Tokamax / MSL `splash_attention_mask_info.find_bounds`, except that the padding steps (valid_T False)
  repeat the last row and must not extend its run.
  """
  prev_T = jnp.concatenate([jnp.full((1,), -1, row_T.dtype), row_T[:-1]])
  next_T = jnp.concatenate([row_T[1:], jnp.full((1,), -1, row_T.dtype)])
  next_valid_T = jnp.concatenate([valid_T[1:], jnp.zeros((1,), bool)])
  start_T = valid_T & (row_T != prev_T)
  end_T = valid_T & ((row_T != next_T) | ~next_valid_T)
  return start_T.astype(jnp.int32), end_T.astype(jnp.int32)


def _kernel(
    # scalar prefetch [T] (SMEM)
    b_ref: jax.Ref, qi_ref: jax.Ref, kj_ref: jax.Ref, kind_ref: jax.Ref,
    q_bounds_start_ref: jax.Ref, q_bounds_end_ref: jax.Ref, b_bounds_start_ref: jax.Ref, b_bounds_end_ref: jax.Ref,
    # inputs (VMEM blocks)
    q_GDQ_ref: jax.Ref, do_GDQ_ref: jax.Ref, k_DS_ref: jax.Ref, k_SD_ref: jax.Ref, v_DS_ref: jax.Ref,
    lse2_GQ_ref: jax.Ref, o_GDQ_ref: jax.Ref, qseg_QL_ref: jax.Ref, kvseg_8S_ref: jax.Ref,
    # outputs (VMEM blocks)
    dq_GQD_ref: jax.Ref, dkt_DS_ref: jax.Ref, dvt_DS_ref: jax.Ref,
    # scratch (VMEM, f32)
    dq_GQD_acc: jax.Ref, dkt_DS_acc: jax.Ref, dvt_DS_acc: jax.Ref, di_GQ: jax.Ref,
    *, group_size: int, log2_scale: float, grad_scale: float,
) -> None:
  del b_ref  # used by the index maps only
  step = pl.program_id(1)
  kind = kind_ref[step]
  q_block = qi_ref[step]
  should_initialize_dq = q_bounds_start_ref[step].astype(jnp.bool_)
  should_write_dq = q_bounds_end_ref[step].astype(jnp.bool_)
  should_initialize_dkv = b_bounds_start_ref[step].astype(jnp.bool_)
  should_write_dkv = b_bounds_end_ref[step].astype(jnp.bool_)
  kv_start = pl.multiple_of(kj_ref[step] * BLOCK, BLOCK)

  @pl.when(should_initialize_dkv)
  def _():
    dkt_DS_acc[...] = jnp.zeros_like(dkt_DS_acc)
    dvt_DS_acc[...] = jnp.zeros_like(dvt_DS_acc)

  @pl.when(should_initialize_dq)
  def _():
    dq_GQD_acc[...] = jnp.zeros_like(dq_GQD_acc)
    # di = rowsum(o * dO) per head, once per q block (o / dO blocks stay resident for the block's run of steps).
    for g in range(group_size):
      od_DQ = o_GDQ_ref[g].astype(jnp.float32) * do_GDQ_ref[g].astype(jnp.float32)
      di_GQ[g:g + 1, :] = jnp.sum(od_DQ, axis=0, keepdims=True)

  def tile(masked: bool) -> None:
    """All G query heads on the whole tile."""
    kv = pl.ds(kv_start, BLOCK)
    # Shared by all heads: K^T / V^T / K slices and the element mask.
    kt_DC = k_DS_ref[:, kv]
    vt_DC = v_DS_ref[:, kv]
    k_CD = k_SD_ref[kv, :]
    if masked:
      q_pos = q_block * BLOCK + lax.broadcasted_iota(jnp.int32, (BLOCK, BLOCK), 0)
      k_pos = kv_start + lax.broadcasted_iota(jnp.int32, (BLOCK, BLOCK), 1)
      qseg_QC = jnp.tile(qseg_QL_ref[...], (1, BLOCK // NUM_LANES))
      mask_QC = (q_pos >= k_pos) & (qseg_QC == kvseg_8S_ref[0:1, kv])

    unroll = min(_HEAD_UNROLL, group_size)
    tn = (((0,), (0,)), ((), ()))  # contract dim 0 of both: A^T @ B
    nn = (((1,), (0,)), ((), ()))  # A @ B

    def heads(it: jax.Array, carry: int) -> int:
      """All five matmuls for `unroll` query heads.

      The heads share one loop body, so the scheduler interleaves one head's elementwise work with the next head's
      matmuls. The per-row lse2 / di values are transposed to [Q, 8] columns for 8 heads at a time; the lane
      broadcasts (XLU) happen inside the MXU-bound body.
      """
      for h8 in range(0, unroll, 8):
        base = it * unroll + h8
        lse2_Q8 = lse2_GQ_ref[pl.ds(base, 8), :].T
        di_Q8 = di_GQ[pl.ds(base, 8), :].T
        for j in range(8):
          g = base + j
          lse2_QC = jnp.broadcast_to(lse2_Q8[:, j:j + 1], (BLOCK, BLOCK))
          di_QC = jnp.broadcast_to(di_Q8[:, j:j + 1], (BLOCK, BLOCK))
          qt_DQ = q_GDQ_ref[g]
          dot_DQ = do_GDQ_ref[g]
          # S and dP contract dim 0 of the SEQ_MINOR q^T / dO^T tiles; Mosaic transposes them on the XLU, which is
          # idle in this MXU-bound loop (cheaper than also DMA-ing HEAD_DIM_MINOR copies).
          s_QC = lax.dot_general(qt_DQ, kt_DC, tn, preferred_element_type=jnp.float32)
          p_QC = jnp.exp2(s_QC * log2_scale - lse2_QC)
          if masked:
            p_QC = jnp.where(mask_QC, p_QC, 0.0)
          dp_QC = lax.dot_general(dot_DQ, vt_DC, tn, preferred_element_type=jnp.float32)
          ds_QC = p_QC * (dp_QC - di_QC)
          p16_QC = p_QC.astype(dot_DQ.dtype)
          ds16_QC = ds_QC.astype(dot_DQ.dtype)
          dvt_DS_acc[:, kv] += lax.dot_general(dot_DQ, p16_QC, nn, preferred_element_type=jnp.float32)
          dkt_DS_acc[:, kv] += lax.dot_general(qt_DQ, ds16_QC, nn, preferred_element_type=jnp.float32)
          dq_GQD_acc[g] += lax.dot_general(ds16_QC, k_CD, nn, preferred_element_type=jnp.float32)
      return carry

    lax.fori_loop(0, group_size // unroll, heads, 0)

  @pl.when(kind == 1)  # full tile
  def _():
    tile(masked=False)

  @pl.when(kind == 2)  # masked tile
  def _():
    tile(masked=True)

  @pl.when(should_write_dq)
  def _():
    dq_GQD_ref[...] = (dq_GQD_acc[...] * grad_scale).astype(dq_GQD_ref.dtype)

  @pl.when(should_write_dkv)
  def _():
    dkt_DS_ref[...] = (dkt_DS_acc[...] * grad_scale).astype(dkt_DS_ref.dtype)
    dvt_DS_ref[...] = dvt_DS_acc[...].astype(dvt_DS_ref.dtype)


def splash_tbwd_call(
    sched: dict[str, jax.Array], q_BHDS: jax.Array, do_BHDS: jax.Array, k_BNDS: jax.Array, k_BNSD: jax.Array,
    v_BNDS: jax.Array, lse2_BHS: jax.Array, o_BHDS: jax.Array, qseg_BSL: jax.Array, kvseg_B8S: jax.Array, *,
    scale: float, log2e: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """The pallas_call on kernel layouts (see `transposed_attention_bwd` for the model-layout entry point).

  lse2_BHS is the log2-domain logsumexp of the base-2 logits (Tokamax's `logsumexp` residual):
  lse2 = log2(sum_k 2 ** (scale * log2e * q.k)). `log2e` is the constant the forward actually multiplied by:
  log2(e) exactly, or bf16(log2(e)) = 1.4453125 (+1.8e-3) for Tokamax, which multiplies bf16 q by it. P must use
  the same value or it will not match lse2. dq and dk are still scaled by `scale` (as Tokamax's backward does):
  folding the forward's rounding of log2(e) into them as well moved dq / dk 0.18% further from a dense f32
  reference. o_BHDS is the forward output, used for di = rowsum(o * dO). Returns dq [B, H, S, D], dk^T and
  dv^T [B, N, D, S] in the input dtype.
  """
  num_b, num_h, d, s = q_BHDS.shape
  num_kv = k_BNDS.shape[1]
  g = num_h // num_kv
  num_steps = sched["b"].shape[0]

  def q_seq_minor(h: jax.Array, t: jax.Array, b: jax.Ref, qi: jax.Ref, *_: jax.Ref) -> _BlockIndex:
    return (b[t], h, 0, qi[t])

  def q_head_dim_minor(h: jax.Array, t: jax.Array, b: jax.Ref, qi: jax.Ref, *_: jax.Ref) -> _BlockIndex:
    return (b[t], h, qi[t], 0)

  def per_kv_head(h: jax.Array, t: jax.Array, b: jax.Ref, *_: jax.Ref) -> _BlockIndex:
    return (b[t], h, 0, 0)

  def q_rows(h: jax.Array, t: jax.Array, b: jax.Ref, qi: jax.Ref, *_: jax.Ref) -> _BlockIndex:
    return (b[t], h, qi[t])

  def q_segments(h: jax.Array, t: jax.Array, b: jax.Ref, qi: jax.Ref, *_: jax.Ref) -> _BlockIndex:
    return (b[t], qi[t], 0)

  def kv_segments(h: jax.Array, t: jax.Array, b: jax.Ref, *_: jax.Ref) -> _BlockIndex:
    return (b[t], 0, 0)

  in_specs = [
      pl.BlockSpec((None, g, d, BLOCK), q_seq_minor),
      pl.BlockSpec((None, g, d, BLOCK), q_seq_minor),
      pl.BlockSpec((None, None, d, s), per_kv_head),
      pl.BlockSpec((None, None, s, d), per_kv_head),
      pl.BlockSpec((None, None, d, s), per_kv_head),
      pl.BlockSpec((None, g, BLOCK), q_rows),
      pl.BlockSpec((None, g, d, BLOCK), q_seq_minor),
      pl.BlockSpec((None, BLOCK, NUM_LANES), q_segments),
      pl.BlockSpec((None, 8, s), kv_segments),
  ]
  out_specs = [
      pl.BlockSpec((None, g, BLOCK, d), q_head_dim_minor),
      pl.BlockSpec((None, None, d, s), per_kv_head),
      pl.BlockSpec((None, None, d, s), per_kv_head),
  ]
  out_shape = [
      jax.ShapeDtypeStruct((num_b, num_h, s, d), q_BHDS.dtype),
      jax.ShapeDtypeStruct((num_b, num_kv, d, s), k_BNDS.dtype),
      jax.ShapeDtypeStruct((num_b, num_kv, d, s), v_BNDS.dtype),
  ]
  scratch = [
      pltpu.VMEM((g, BLOCK, d), jnp.float32),
      pltpu.VMEM((d, s), jnp.float32),
      pltpu.VMEM((d, s), jnp.float32),
      pltpu.VMEM((g, BLOCK), jnp.float32),
  ]
  # XLA's latency-hiding scheduler treats a custom call without a cost estimate as free, and then delays async
  # collectives (FSDP reduce-scatters) until after it instead of overlapping them with it. Count every
  # scheduled step as a full tile of 5 matmuls per q head; padding steps make this an upper bound.
  operands = (q_BHDS, do_BHDS, k_BNDS, k_BNSD, v_BNDS, lse2_BHS, o_BHDS, qseg_BSL, kvseg_B8S)
  tile_heads = num_kv * num_steps * g
  cost = pl.CostEstimate(
      flops=tile_heads * 5 * 2 * BLOCK * BLOCK * d,
      transcendentals=tile_heads * BLOCK * BLOCK,
      bytes_accessed=sum(x.size * x.dtype.itemsize for x in (*operands, *out_shape)),
  )
  return pl.pallas_call(
      functools.partial(_kernel, group_size=g, log2_scale=scale * log2e, grad_scale=scale),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(_SCHEDULE_KEYS), in_specs=in_specs, out_specs=out_specs, grid=(num_kv, num_steps),
          scratch_shapes=scratch),
      out_shape=out_shape,
      compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"),
                                           vmem_limit_bytes=_VMEM_LIMIT_BYTES),
      cost_estimate=cost,
      name="splash_tbwd",
  )(*(sched[key] for key in _SCHEDULE_KEYS), *operands)


def transposed_attention_bwd(
    q_BSHD: jax.Array, k_BSND: jax.Array, v_BSND: jax.Array, out_BSHD: jax.Array, lse2_BHS: jax.Array,
    do_BSHD: jax.Array, segment_ids_BS: jax.Array, *, scale: float, log2e: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """dq, dk, dv of causal packed-document attention, all in model layouts [B, S, heads, D].

  lse2_BHS: log2-domain logsumexp from the forward (Tokamax's residual; see `splash_tbwd_call`). The layout
  changes below are transposes to sequence-minor; in the model graph, where XLA keeps q / k / v sequence-minor,
  they lower to bitcasts. Raises ValueError for shapes or dtypes outside "Limits" in the module docstring.
  """
  if q_BSHD.ndim != 4 or k_BSND.ndim != 4:
    raise ValueError(f"expected q [B, S, H, D] and k [B, S, N, D]; got {q_BSHD.shape}, {k_BSND.shape}.")
  num_b, s, num_h, d = q_BSHD.shape
  num_kv = k_BSND.shape[2]
  if (k_BSND.shape[0], k_BSND.shape[1], k_BSND.shape[3]) != (num_b, s, d):
    raise ValueError(f"k {k_BSND.shape} does not match q {q_BSHD.shape} (self-attention, equal lengths).")
  if d != HEAD_DIM:
    raise ValueError(f"splash_tbwd needs head_dim {HEAD_DIM}; got {d}.")
  if num_h % num_kv != 0 or num_h // num_kv not in SUPPORTED_GROUP_SIZES:
    raise ValueError(
        f"splash_tbwd needs {SUPPORTED_GROUP_SIZES} query heads per kv head; got {num_h} q heads, {num_kv} kv heads.")
  if s % BLOCK != 0 or s > MAX_SEQ_LEN:
    raise ValueError(f"splash_tbwd needs seq_len a multiple of {BLOCK} and <= {MAX_SEQ_LEN}; got {s}.")
  if segment_ids_BS.shape != (num_b, s):
    raise ValueError(f"segment_ids shape {segment_ids_BS.shape} != (batch, seq_len) {(num_b, s)}.")
  if q_BSHD.dtype != jnp.bfloat16:
    raise ValueError(f"splash_tbwd takes bf16 q / k / v / dO; got {q_BSHD.dtype}.")
  sched = build_schedule(segment_ids_BS)
  seg_BS = segment_ids_BS.astype(jnp.int32)
  dq_BHSD, dkt_BNDS, dvt_BNDS = splash_tbwd_call(
      sched,
      jnp.transpose(q_BSHD, (0, 2, 3, 1)),
      jnp.transpose(do_BSHD, (0, 2, 3, 1)),
      jnp.transpose(k_BSND, (0, 2, 3, 1)),
      jnp.swapaxes(k_BSND, 1, 2),
      jnp.transpose(v_BSND, (0, 2, 3, 1)),
      lse2_BHS.astype(jnp.float32),
      jnp.transpose(out_BSHD, (0, 2, 3, 1)),
      jnp.broadcast_to(seg_BS[:, :, None], (num_b, s, NUM_LANES)),
      jnp.broadcast_to(seg_BS[:, None, :], (num_b, 8, s)),
      scale=scale,
      log2e=log2e,
  )
  return (jnp.swapaxes(dq_BHSD, 1, 2), jnp.transpose(dkt_BNDS, (0, 3, 1, 2)),
          jnp.transpose(dvt_BNDS, (0, 3, 1, 2)))
