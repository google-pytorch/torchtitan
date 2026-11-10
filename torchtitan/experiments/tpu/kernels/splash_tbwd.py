"""Transposed Splash attention backward: causal, packed-document GQA attention.

TPU, Pallas/Mosaic. Shape suffixes (logical dims, see .claude/CLAUDE.md):
  B batch, S sequence, H query heads, N kv heads, G = H // N query heads per
  kv head, D head dim, Q rows of a q block, C columns of a kv block, I q
  blocks, J kv blocks, T schedule steps.

Why transposed
--------------
On the 256x256 weight-stationary MXU a matmul streams LHS rows against a
latched RHS (weights) tile. Tokamax's dkv kernel computes dV = P^T dO,
dK = dS^T Q and dQ = dS K. All three have a D-wide output, so at D = 128 every
latched weight tile is half empty. This kernel computes S = Q K^T with
rows = q, which makes P and dS natural [Q, C] weight tiles, and produces dK and
dV transposed:

  S     = Q    [Q, D] @ K^T [D, C]   (K^T latched, shared by the G heads)
  P     = exp2(scale * log2(e) * S - lse2) * mask
          (lse2: the forward's log2-domain logsumexp)
  dP    = dO   [Q, D] @ V^T [D, C]
  dS    = P * (dP - di)   (di = rowsum(o * dO), once per q block)
  dV^T += dO^T [D, Q] @ P   [Q, C]   (P latched; full-width output)
  dK^T += Q^T  [D, Q] @ dS  [Q, C]   (dS latched)
  dQ   += dS   [Q, C] @ K   [C, D]

At D = 128, S and dP still contract over 128 and dQ still has a 128-wide
output; those half-filled matmuls are 3/5 of the MXU pushes and bound the
kernel. At D = 256 every matmul fills the MXU and the transpose gains nothing;
at D = 64 the per-tile elementwise work (exp2, mask, dS), which does not shrink
with D, takes over.

Schedule and accumulation
-------------------------
One pallas_call covers the whole batch. The grid is (N, T). The T steps walk a
schedule, built on device from the segment ids (`build_schedule`), of the
active (example, q block, kv block) tiles in that lexicographic order. T is
the traced number of active tiles (a dynamic grid, as Tokamax's); the schedule
arrays are sized to the data-independent bound B * nb * (nb + 1) / 2
(nb = S / block). Because of that order:
  - dQ of one (example, q block) accumulates in VMEM over its run of steps and
    is written once;
  - dK^T / dV^T of one (example, kv head) stay resident in VMEM as [D, S] f32
    for the whole sequence, and so do K / V; there are no HBM partial sums and
    no reduction op;
  - each step processes all G query heads of its kv head (K^T / V^T / K
    loaded once per step).
A tile is skipped when its q and kv blocks share no segment id ([min, max]
interval test, as Tokamax's `_segment_block_overlap`; conservative, so ids need
not be sorted). Tiles entirely inside one document and strictly below the
diagonal skip the element mask; every other active tile is computed whole, with
the mask.

Supported shapes
----------------
`unsupported_reason` checks all of these; G = 1 also runs, but
`tokamax_splash_attention` routes it to the Tokamax backward (faster there).

- dtype: bf16 q, k, v, dO.
- G = H / N: integer, 1 <= G <= 16.
- D: multiple of 32, D >= 64; the upper bound comes from VMEM (below).
- S: multiple of 128, q and kv equal length. Tile (q block = kv block) is
  512 if it divides S, else 256, else 128. Upper bound from VMEM (below).
- B: any; the upper bound comes from SMEM (below).
- VMEM: K / V and the dK^T / dV^T accumulators stay resident for the whole
  sequence, so VMEM grows with D * S. With Dl = D rounded up to 128,
  G8 = G rounded up to 8 and kv_buffers = 1 (2 when it fits), the estimate
  (`_vmem_bytes`) must stay <= 60 MiB:

    2 * block * (6*G*D + 4*G8 + 512 + 2*G*Dl)   per-step blocks, 2 buffers
    + kv_buffers * S * (8*D + 2*Dl + 32)        per-(example, kv head) blocks
    + 4*G*block*Dl + 8*D*S + 8*G8*block         f32 scratch
    + 10 MiB + 1126 * G * max(0, D - 128)       compiler temporaries

  With 512 tiles (S a multiple of 512):
    G = 16, D = 64:  S <= 28160
    G = 16, D = 128: S <= 12800
    G = 8,  D = 256: S <= 6144
    G = 16, D = 256: S <= 1536
  Past the limit Mosaic fails at compile time with RESOURCE_EXHAUSTED.
- SMEM: the 8 int32 schedule arrays hold every causal tile, so
  8 * 4 * B * nb * (nb + 1) / 2 bytes (nb = S / block) must stay
  <= 1 MiB - 16 KiB.

Other limitations
-----------------
- Causal attention only: no sliding window, no non-causal masks.
- No context parallelism (kv length must equal q length).
- No logit soft cap, no dropout, no attention sinks.
- Packed documents only via segment ids (optional; None = one document).
- Code size: the kernel unrolls all G heads in each of its 2 tile-kind
  branches. An 81k-bundle variant overflowed TPU7x instruction memory and ran
  3.5x slower (53k was fine), so check the final LLO bundle count
  (--xla_jf_dump_to) before adding branches.
"""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Tile sizes (q block = kv block), largest first: 512 when it divides S, else
# 256, else 128. 128-wide tiles half-fill the 256-wide MXU.
BLOCK_SIZES = (512, 256, 128)
# Head dims are multiples of 32 from 64 up; VMEM bounds them from above (see
# `unsupported_reason`).
MIN_HEAD_DIM = 64
MAX_GROUP_SIZE = 16
NUM_LANES = 128
_VMEM_LIMIT_BYTES = 60 << 20
# TPU7x scalar memory is 1 MiB; the schedule arrays are prefetched into it, next
# to a few KiB of compiler state.
_SMEM_BUDGET_BYTES = (1 << 20) - (16 << 10)

# A BlockSpec index map's result: block indices, one per array dim.
_BlockIndex = tuple[jax.Array | int, ...]

# Scalar-prefetch order of the `build_schedule` arrays, matching `_kernel`'s
# leading refs.
_SCHEDULE_KEYS = (
    "b",
    "qi",
    "kj",
    "kind",
    "q_bounds_start",
    "q_bounds_end",
    "b_bounds_start",
    "b_bounds_end",
)


def build_schedule(
    segment_ids_BS: jax.Array, block: int
) -> dict[str, jax.Array]:
  """Tile schedule [T] for `splash_tbwd_call` from (traced) segment ids [B, S].

  Returns:
    num_active: int32 scalar, the number of active tiles (the grid's steps).
    b, qi, kj: int32 [T], the (example, q block, kv block) of each step.
    kind: int32 [T]; 0 padding step (no compute), 1 every pair causal and
      same-document (no element mask), 2 masked tile.
    q_bounds_start / q_bounds_end: int32 [T], 1 on the first / last step of
      each (example, q block) run (dQ accumulator), as Tokamax's `find_bounds`.
    b_bounds_start / b_bounds_end: the same for each example (dK^T / dV^T).

  Active tiles come first in (b, qi, kj) order. The remaining entries, up to
  the data-independent bound of every causal tile, are kind-0 copies of the
  last active tile's indices with all bounds 0 (the grid does not reach them).
  Uses only reductions, compares and one sort (no gathers, which XLA TPU may
  offload to SparseCore).
  """
  num_b, s = segment_ids_BS.shape
  num_blocks = s // block
  seg_BIQ = segment_ids_BS.astype(jnp.int32).reshape(num_b, num_blocks, block)
  min_seg_BI = seg_BIQ.min(axis=-1)
  max_seg_BI = seg_BIQ.max(axis=-1)
  qi_IJ = jnp.arange(num_blocks, dtype=jnp.int32)[:, None]
  kj_IJ = jnp.arange(num_blocks, dtype=jnp.int32)[None, :]
  # A causal tile may hold a same-document pair when the [min, max] segment-id
  # ranges of its q block and kv block overlap (as Tokamax's
  # `_segment_block_overlap`; conservative, so ids need not be sorted).
  may_have_live_pairs_BIJ = (
      (max_seg_BI[:, :, None] >= min_seg_BI[:, None, :])
      & (max_seg_BI[:, None, :] >= min_seg_BI[:, :, None])
      & (kj_IJ <= qi_IJ)
  )
  # Every pair is live when both blocks lie inside the same single document,
  # strictly below the diagonal.
  single_doc_BI = min_seg_BI == max_seg_BI
  all_pairs_live_BIJ = (
      (kj_IJ < qi_IJ)
      & single_doc_BI[:, :, None]
      & single_doc_BI[:, None, :]
      & (min_seg_BI[:, :, None] == min_seg_BI[:, None, :])
  )
  kind_BIJ = jnp.where(all_pairs_live_BIJ, 1, 2)

  # Compaction: sort unique keys (tile index * 8 + kind); tiles without live
  # pairs sort last.
  tile_BIJ = (
      jnp.arange(num_b, dtype=jnp.int32)[:, None, None] * num_blocks + qi_IJ
  ) * num_blocks + kj_IJ
  big = jnp.iinfo(jnp.int32).max
  key = jnp.where(
      may_have_live_pairs_BIJ, tile_BIJ * 8 + kind_BIJ, big
  ).reshape(-1)
  num_steps = _num_steps(num_b, s, block)
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
  return dict(
      num_active=num_active,
      b=b_T,
      qi=qi_T,
      kj=kj_T,
      kind=kind_T,
      q_bounds_start=q_bounds_start,
      q_bounds_end=q_bounds_end,
      b_bounds_start=b_bounds_start,
      b_bounds_end=b_bounds_end,
  )


def _find_bounds(
    row_T: jax.Array, valid_T: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """First / last step of each run of equal `row_T` among valid steps, [T].

  Same as Tokamax / MSL `splash_attention_mask_info.find_bounds`, except that
  the padding steps (valid_T False) repeat the last row and must not extend
  its run.
  """
  prev_T = jnp.concatenate([jnp.full((1,), -1, row_T.dtype), row_T[:-1]])
  next_T = jnp.concatenate([row_T[1:], jnp.full((1,), -1, row_T.dtype)])
  next_valid_T = jnp.concatenate([valid_T[1:], jnp.zeros((1,), bool)])
  start_T = valid_T & (row_T != prev_T)
  end_T = valid_T & ((row_T != next_T) | ~next_valid_T)
  return start_T.astype(jnp.int32), end_T.astype(jnp.int32)


def _kernel(
    # scalar prefetch [T] (SMEM)
    b_ref: jax.Ref,
    qi_ref: jax.Ref,
    kj_ref: jax.Ref,
    kind_ref: jax.Ref,
    q_bounds_start_ref: jax.Ref,
    q_bounds_end_ref: jax.Ref,
    b_bounds_start_ref: jax.Ref,
    b_bounds_end_ref: jax.Ref,
    # inputs (VMEM blocks)
    q_GDQ_ref: jax.Ref,
    do_GDQ_ref: jax.Ref,
    k_DS_ref: jax.Ref,
    k_SD_ref: jax.Ref,
    v_DS_ref: jax.Ref,
    lse2_GQ_ref: jax.Ref,
    o_GDQ_ref: jax.Ref,
    qseg_QL_ref: jax.Ref,
    kvseg_8S_ref: jax.Ref,
    # outputs (VMEM blocks)
    dq_GQD_ref: jax.Ref,
    dkt_DS_ref: jax.Ref,
    dvt_DS_ref: jax.Ref,
    # scratch (VMEM, f32)
    dq_GQD_acc: jax.Ref,
    dkt_DS_acc: jax.Ref,
    dvt_DS_acc: jax.Ref,
    di_GQ: jax.Ref,
    lse2_GQ: jax.Ref,
    *,
    group_size: int,
    block: int,
    log2_scale: float,
    grad_scale: float,
) -> None:
  del b_ref  # used by the index maps only
  lse2_rows = lse2_GQ if group_size % 8 else lse2_GQ_ref
  step = pl.program_id(1)
  kind = kind_ref[step]
  q_block = qi_ref[step]
  should_initialize_dq = q_bounds_start_ref[step].astype(jnp.bool_)
  should_write_dq = q_bounds_end_ref[step].astype(jnp.bool_)
  should_initialize_dkv = b_bounds_start_ref[step].astype(jnp.bool_)
  should_write_dkv = b_bounds_end_ref[step].astype(jnp.bool_)
  kv_start = pl.multiple_of(kj_ref[step] * block, block)

  @pl.when(should_initialize_dkv)
  def _():
    dkt_DS_acc[...] = jnp.zeros_like(dkt_DS_acc)
    dvt_DS_acc[...] = jnp.zeros_like(dvt_DS_acc)

  @pl.when(should_initialize_dq)
  def _():
    dq_GQD_acc[...] = jnp.zeros_like(dq_GQD_acc)
    # lse2 / di rows are read 8 heads at a time; with G not a multiple of 8,
    # lse2 is copied into scratch padded to a multiple of 8 rows (the padding
    # rows are never used).
    if group_size % 8:
      lse2_GQ[...] = jnp.zeros_like(lse2_GQ)
      di_GQ[...] = jnp.zeros_like(di_GQ)
      lse2_GQ[0:group_size, :] = lse2_GQ_ref[...]
    # di = rowsum(o * dO) per head, once per q block (o / dO blocks stay
    # resident for the block's run of steps).
    for g in range(group_size):
      od_DQ = o_GDQ_ref[g].astype(jnp.float32) * do_GDQ_ref[g].astype(
          jnp.float32
      )
      di_GQ[g : g + 1, :] = jnp.sum(od_DQ, axis=0, keepdims=True)

  def tile(masked: bool) -> None:
    """All G query heads on the whole tile."""
    kv = pl.ds(kv_start, block)
    # Shared by all heads: K^T / V^T / K slices and the element mask.
    kt_DC = k_DS_ref[:, kv]
    vt_DC = v_DS_ref[:, kv]
    k_CD = k_SD_ref[kv, :]
    if masked:
      q_pos = q_block * block + lax.broadcasted_iota(
          jnp.int32, (block, block), 0
      )
      k_pos = kv_start + lax.broadcasted_iota(jnp.int32, (block, block), 1)
      qseg_QC = jnp.tile(qseg_QL_ref[...], (1, block // NUM_LANES))
      mask_QC = (q_pos >= k_pos) & (qseg_QC == kvseg_8S_ref[0:1, kv])

    tn = (((0,), (0,)), ((), ()))  # contract dim 0 of both: A^T @ B
    nn = (((1,), (0,)), ((), ()))  # A @ B

    # All G heads are unrolled into one straight-line body, so the scheduler
    # interleaves one head's elementwise work with the next head's matmuls.
    # The per-row lse2 / di values are transposed to [Q, 8] columns for 8 heads
    # at a time; the lane broadcasts (XLU) happen inside the MXU-bound body.
    for base in range(0, group_size, 8):
      lse2_Q8 = lse2_rows[base : base + 8, :].T
      di_Q8 = di_GQ[base : base + 8, :].T
      for j in range(min(8, group_size - base)):
        g = base + j
        lse2_QC = jnp.broadcast_to(lse2_Q8[:, j : j + 1], (block, block))
        di_QC = jnp.broadcast_to(di_Q8[:, j : j + 1], (block, block))
        qt_DQ = q_GDQ_ref[g]
        dot_DQ = do_GDQ_ref[g]
        # S and dP contract dim 0 of the SEQ_MINOR q^T / dO^T tiles; Mosaic
        # transposes them on the XLU, which is idle in this MXU-bound loop
        # (cheaper than also DMA-ing HEAD_DIM_MINOR copies).
        s_QC = lax.dot_general(
            qt_DQ, kt_DC, tn, preferred_element_type=jnp.float32
        )
        p_QC = jnp.exp2(s_QC * log2_scale - lse2_QC)
        if masked:
          p_QC = jnp.where(mask_QC, p_QC, 0.0)
        dp_QC = lax.dot_general(
            dot_DQ, vt_DC, tn, preferred_element_type=jnp.float32
        )
        ds_QC = p_QC * (dp_QC - di_QC)
        p16_QC = p_QC.astype(dot_DQ.dtype)
        ds16_QC = ds_QC.astype(dot_DQ.dtype)
        dvt_DS_acc[:, kv] += lax.dot_general(
            dot_DQ, p16_QC, nn, preferred_element_type=jnp.float32
        )
        dkt_DS_acc[:, kv] += lax.dot_general(
            qt_DQ, ds16_QC, nn, preferred_element_type=jnp.float32
        )
        dq_GQD_acc[g] += lax.dot_general(
            ds16_QC, k_CD, nn, preferred_element_type=jnp.float32
        )

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
    sched: dict[str, jax.Array],
    q_BHDS: jax.Array,
    do_BHDS: jax.Array,
    k_BNDS: jax.Array,
    k_BNSD: jax.Array,
    v_BNDS: jax.Array,
    lse2_BNGS: jax.Array,
    o_BHDS: jax.Array,
    qseg_BSL: jax.Array,
    kvseg_B8S: jax.Array,
    *,
    block: int,
    single_buffer_kv: bool,
    scale: float,
    log2e: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """The pallas_call on kernel layouts.

  `transposed_attention_bwd` is the model-layout entry point.

  Args:
    lse2_BNGS: log2-domain logsumexp of the base-2 logits (Tokamax's
      `logsumexp` residual): lse2 = log2(sum_k 2 ** (scale * log2e * q.k)).
    o_BHDS: forward output, for di = rowsum(o * dO).
    log2e: the constant the forward actually multiplied by, so that P matches
      lse2: log2(e), or bf16(log2(e)) = 1.4453125 (+1.8e-3) for Tokamax, which
      multiplies bf16 q by it. dq and dk are still scaled by `scale` only (as
      Tokamax's backward): also folding log2e's rounding into them moved
      dq / dk 0.18% further from a dense f32 reference.

  Returns:
    dq [B, H, S, D], dk^T and dv^T [B, N, D, S], in the input dtype.
  """
  num_b, num_h, d, s = q_BHDS.shape
  num_kv = k_BNDS.shape[1]
  g = num_h // num_kv
  num_steps = sched["b"].shape[0]

  def q_seq_minor(
      h: jax.Array, t: jax.Array, b: jax.Ref, qi: jax.Ref, *_: jax.Ref
  ) -> _BlockIndex:
    return (b[t], h, 0, qi[t])

  def q_head_dim_minor(
      h: jax.Array, t: jax.Array, b: jax.Ref, qi: jax.Ref, *_: jax.Ref
  ) -> _BlockIndex:
    return (b[t], h, qi[t], 0)

  def per_kv_head(
      h: jax.Array, t: jax.Array, b: jax.Ref, *_: jax.Ref
  ) -> _BlockIndex:
    return (b[t], h, 0, 0)

  def q_segments(
      h: jax.Array, t: jax.Array, b: jax.Ref, qi: jax.Ref, *_: jax.Ref
  ) -> _BlockIndex:
    return (b[t], qi[t], 0)

  def kv_segments(
      h: jax.Array, t: jax.Array, b: jax.Ref, *_: jax.Ref
  ) -> _BlockIndex:
    return (b[t], 0, 0)

  # The per-(example, kv head) arrays change block only when the example or kv
  # head does, so their second buffer only prefetches the next example's K / V.
  # Dropping it (single_buffer_kv) frees VMEM for ~1.5x longer sequences, at ~1%
  # kernel time (Q235 shape: the DMA at each of the 64 example / kv head changes
  # is no longer hidden).
  resident = dict(pipeline_mode=pl.Buffered(1)) if single_buffer_kv else {}
  in_specs = [
      pl.BlockSpec((None, g, d, block), q_seq_minor),
      pl.BlockSpec((None, g, d, block), q_seq_minor),
      pl.BlockSpec((None, None, d, s), per_kv_head, **resident),
      pl.BlockSpec((None, None, s, d), per_kv_head, **resident),
      pl.BlockSpec((None, None, d, s), per_kv_head, **resident),
      pl.BlockSpec((None, None, g, block), q_seq_minor),
      pl.BlockSpec((None, g, d, block), q_seq_minor),
      pl.BlockSpec((None, block, NUM_LANES), q_segments),
      pl.BlockSpec((None, 8, s), kv_segments, **resident),
  ]
  out_specs = [
      pl.BlockSpec((None, g, block, d), q_head_dim_minor),
      pl.BlockSpec((None, None, d, s), per_kv_head, **resident),
      pl.BlockSpec((None, None, d, s), per_kv_head, **resident),
  ]
  out_shape = [
      jax.ShapeDtypeStruct((num_b, num_h, s, d), q_BHDS.dtype),
      jax.ShapeDtypeStruct((num_b, num_kv, d, s), k_BNDS.dtype),
      jax.ShapeDtypeStruct((num_b, num_kv, d, s), v_BNDS.dtype),
  ]
  scratch = [
      pltpu.VMEM((g, block, d), jnp.float32),
      pltpu.VMEM((d, s), jnp.float32),
      pltpu.VMEM((d, s), jnp.float32),
      pltpu.VMEM((_round_up(g, 8), block), jnp.float32),
      pltpu.VMEM((_round_up(g, 8), block), jnp.float32),
  ]
  # Static cost estimate, so XLA's latency-hiding scheduler overlaps async
  # collectives (FSDP reduce-scatters) with this call instead of treating it
  # as free. With T_max = num_steps (every causal tile, an upper bound on the
  # active ones):
  #   flops           = N * T_max * G * 5 matmuls * 2 * block * block * D
  #   transcendentals = N * T_max * G * block * block   (one exp2 per P entry)
  #   bytes_accessed  = every operand and output, read / written once
  operands = (
      q_BHDS,
      do_BHDS,
      k_BNDS,
      k_BNSD,
      v_BNDS,
      lse2_BNGS,
      o_BHDS,
      qseg_BSL,
      kvseg_B8S,
  )
  tile_heads = num_kv * num_steps * g
  cost = pl.CostEstimate(
      flops=tile_heads * 5 * 2 * block * block * d,
      transcendentals=tile_heads * block * block,
      bytes_accessed=sum(
          x.size * x.dtype.itemsize for x in (*operands, *out_shape)
      ),
  )
  return pl.pallas_call(
      functools.partial(
          _kernel,
          group_size=g,
          block=block,
          log2_scale=scale * log2e,
          grad_scale=scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(_SCHEDULE_KEYS),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=(num_kv, sched["num_active"]),
          scratch_shapes=scratch,
      ),
      out_shape=out_shape,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary", "arbitrary"),
          vmem_limit_bytes=_VMEM_LIMIT_BYTES,
      ),
      cost_estimate=cost,
      name="splash_tbwd",
  )(*(sched[key] for key in _SCHEDULE_KEYS), *operands)


def _round_up(x: int, multiple: int) -> int:
  return -(-x // multiple) * multiple


def block_size(s: int) -> int | None:
  """Tile size for seq_len `s`: the largest BLOCK_SIZES entry dividing it."""
  return next((block for block in BLOCK_SIZES if s % block == 0), None)


def _num_steps(num_b: int, s: int, block: int) -> int:
  """Every causal tile: a data-independent bound on the active tiles."""
  num_blocks = s // block
  return num_b * num_blocks * (num_blocks + 1) // 2


def _vmem_bytes(
    d: int, g: int, s: int, block: int, single_buffer_kv: bool
) -> int:
  """Upper estimate of one call's VMEM bytes (formula: module docstring).

  Arrays with D on lanes pad it to Dl = round_up(D, 128); [G, block] arrays
  pad G to G8 = round_up(G, 8). The temporaries term is fit to TPU7x compile
  reports with 2 MiB of margin.
  """
  d_lanes, g_rows = _round_up(d, NUM_LANES), _round_up(g, 8)
  # Blocks that change every step, double-buffered (bf16 = 2 B, f32 = 4 B).
  q_do_o = 3 * g * d * block * 2  # q^T, dO^T, o^T  [G, D, block] bf16
  lse2 = g_rows * block * 4  # [G8, block] f32
  q_segment_ids = block * NUM_LANES * 4  # [block, 128] int32
  dq = g * block * d_lanes * 2  # [G, block, Dl] bf16
  step_blocks = q_do_o + lse2 + q_segment_ids + dq
  # Blocks that change only per (example, kv head): kv_buffers copies.
  kt_vt = 2 * d * s * 2  # K^T, V^T  [D, S] bf16
  k = s * d_lanes * 2  # K  [S, Dl] bf16
  kv_segment_ids = 8 * s * 4  # [8, S] int32
  dkt_dvt = 2 * d * s * 2  # dK^T, dV^T outputs  [D, S] bf16
  per_kv_head = kt_vt + k + kv_segment_ids + dkt_dvt
  kv_buffers = 1 if single_buffer_kv else 2
  # f32 scratch: dq accumulator [G, block, Dl], dK^T / dV^T accumulators
  # [D, S], lse2 / di [G8, block].
  scratch = g * block * d_lanes * 4 + 2 * d * s * 4 + 2 * g_rows * block * 4
  temporaries = (10 << 20) + max(0, d - 128) * g * 1126
  return 2 * step_blocks + kv_buffers * per_kv_head + scratch + temporaries


def unsupported_reason(
    q_shape: tuple[int, ...], k_shape: tuple[int, ...], dtype: jnp.dtype
) -> str | None:
  """Why splash_tbwd cannot run q [B, S, H, D] / k [B, S, N, D], or None.

  Checks the "Supported shapes" in the module docstring. The caller checks the
  attention itself (causal, no window, no context parallelism).
  """
  if len(q_shape) != 4 or len(k_shape) != 4:
    return (
        f"expected q [B, S, H, D] and k [B, S, N, D]; got {q_shape}, {k_shape}"
    )
  num_b, s, num_h, d = q_shape
  num_kv = k_shape[2]
  if (k_shape[0], k_shape[1], k_shape[3]) != (num_b, s, d):
    return (
        f"k {k_shape} does not match q {q_shape} (self-attention, equal"
        " lengths)"
    )
  if dtype != jnp.bfloat16:
    return f"needs bf16 q / k / v / dO; got {dtype}"
  if num_h % num_kv != 0 or num_h // num_kv > MAX_GROUP_SIZE:
    return (
        "needs a whole number of q heads per kv head, at most"
        f" {MAX_GROUP_SIZE}; got {num_h} q / {num_kv} kv heads"
    )
  if d % 32 != 0 or d < MIN_HEAD_DIM:
    return f"needs head_dim a multiple of 32 and >= {MIN_HEAD_DIM}; got {d}"
  block = block_size(s)
  if block is None:
    return f"needs seq_len a multiple of {BLOCK_SIZES[-1]}; got {s}"
  g = num_h // num_kv
  vmem = _vmem_bytes(d, g, s, block, single_buffer_kv=True)
  if vmem > _VMEM_LIMIT_BYTES:
    return (
        f"head_dim {d}, {g} q heads per kv head and seq_len {s} need"
        f" ~{vmem >> 20} MiB of VMEM "
        f"(limit {_VMEM_LIMIT_BYTES >> 20} MiB)"
    )
  smem = len(_SCHEDULE_KEYS) * 4 * _num_steps(num_b, s, block)
  if smem > _SMEM_BUDGET_BYTES:
    return (
        f"batch {num_b} x seq_len {s} needs a {smem >> 10} KiB tile"
        " schedule in SMEM "
        f"(limit {_SMEM_BUDGET_BYTES >> 10} KiB)"
    )
  return None


def transposed_attention_bwd(
    q_BSHD: jax.Array,
    k_BSND: jax.Array,
    v_BSND: jax.Array,
    out_BSHD: jax.Array,
    lse2_BHS: jax.Array,
    do_BSHD: jax.Array,
    segment_ids_BS: jax.Array | None,
    *,
    scale: float,
    log2e: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """dq, dk, dv of causal attention, in the model layout [B, S, heads, D].

  Args:
    q_BSHD, k_BSND, v_BSND, out_BSHD, do_BSHD: forward inputs, forward output
      and output gradient.
    lse2_BHS: the forward's log2-domain logsumexp (see `splash_tbwd_call`).
    segment_ids_BS: document id per token; None means one document per example.
    scale, log2e: the constants the forward used (see `splash_tbwd_call`).

  Raises:
    ValueError: for shapes outside "Supported shapes" (module docstring).
  """
  reason = unsupported_reason(q_BSHD.shape, k_BSND.shape, q_BSHD.dtype)
  if reason is not None:
    raise ValueError(f"splash_tbwd {reason}.")
  num_b, s, num_h, d = q_BSHD.shape
  num_kv = k_BSND.shape[2]
  if segment_ids_BS is None:
    segment_ids_BS = jnp.zeros((num_b, s), jnp.int32)
  if segment_ids_BS.shape != (num_b, s):
    raise ValueError(
        f"segment_ids shape {segment_ids_BS.shape} != (batch, seq_len)"
        f" {(num_b, s)}."
    )
  block = block_size(s)
  single_buffer_kv = (
      _vmem_bytes(d, num_h // num_kv, s, block, single_buffer_kv=False)
      > _VMEM_LIMIT_BYTES
  )
  sched = build_schedule(segment_ids_BS, block)
  seg_BS = segment_ids_BS.astype(jnp.int32)
  # The transposes to sequence-minor lower to bitcasts in the model graph,
  # where XLA already keeps q / k / v sequence-minor.
  dq_BHSD, dkt_BNDS, dvt_BNDS = splash_tbwd_call(
      sched,
      jnp.transpose(q_BSHD, (0, 2, 3, 1)),
      jnp.transpose(do_BSHD, (0, 2, 3, 1)),
      jnp.transpose(k_BSND, (0, 2, 3, 1)),
      jnp.swapaxes(k_BSND, 1, 2),
      jnp.transpose(v_BSND, (0, 2, 3, 1)),
      lse2_BHS.astype(jnp.float32).reshape(num_b, num_kv, num_h // num_kv, s),
      jnp.transpose(out_BSHD, (0, 2, 3, 1)),
      jnp.broadcast_to(seg_BS[:, :, None], (num_b, s, NUM_LANES)),
      jnp.broadcast_to(seg_BS[:, None, :], (num_b, 8, s)),
      block=block,
      single_buffer_kv=single_buffer_kv,
      scale=scale,
      log2e=log2e,
  )
  return (
      jnp.swapaxes(dq_BHSD, 1, 2),
      jnp.transpose(dkt_BNDS, (0, 3, 1, 2)),
      jnp.transpose(dvt_BNDS, (0, 3, 1, 2)),
  )
