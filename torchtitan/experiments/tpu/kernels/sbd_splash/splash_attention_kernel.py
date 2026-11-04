# Copyright 2026 Google LLC

"""Implementation of Sparse Flash Attention, a.k.a. "Splash" attention."""

from collections.abc import Callable
import dataclasses
import enum
import functools
import json
import math
from typing import Any, NamedTuple

import jax
from jax import ad_checkpoint
from jax import lax
from jax import tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from torchtitan.experiments.tpu.kernels.sbd_splash import base
from torchtitan.experiments.tpu.kernels.sbd_splash import splash_attention_mask as mask_lib
from torchtitan.experiments.tpu.kernels.sbd_splash import splash_attention_mask_info as mask_info_lib


P = jax.P
MaskInfo = mask_info_lib.MaskInfo
partial = functools.partial
NUM_LANES = 128
NUM_SUBLANES = 8
# We predefine some useful dimension numbers for dot_general
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # standard matmul
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # RHS transposed

LOG2E = math.log2(math.e)
LOG2E_INV = 1 / LOG2E
# sbd fork: fixed dropout-mask generation granularity (absolute coordinates).
# Must divide every legal block size; 256 = the v6e MXU tile edge.
DROPOUT_QUANTUM = 256


def _mix_dropout_seed(seed0, seed1, h, q_quant, kv_quant):
  """Mixes (head, q_quantum, kv_quantum) into the two seed words.

  Mosaic's prng_seed accepts at most 2 seed values on this backend
  ("Setting seed with more than 2 values is not supported"), so the
  coordinate mixing happens here with distinct odd 32-bit hash constants
  (int32 wrap-around is intended). Must match mask_export.py exactly.
  """
  i32 = jnp.int32
  w0 = i32(seed0) ^ (
      i32(h) * i32(-1640531527)  # 0x9E3779B9
      + i32(q_quant) * i32(-2048144789)  # 0x85EBCA6B
      + i32(kv_quant) * i32(-1028477387)  # 0xC2B2AE35
  )
  w1 = i32(seed1) ^ (
      i32(h) * i32(668265263)  # 0x27D4EB2F
      + i32(q_quant) * i32(374761393)  # 0x165667B1
      + i32(kv_quant) * i32(-1640531535)  # 0x9E3779B1
  )
  return w0, w1


# sbd fork v019 (mask_version 6): orientation-free per-element counter hash.
# keep(w0, w1, h, q_abs, kv_abs) is a pure function of ABSOLUTE coordinates,
# so the fwd (q-major) and dkv (kv-major) bodies each generate their native
# layout -- no transpose, no chained PRNG state, no bitcast expansion, no
# band-width coupling between fwd and bwd block sizes. i32 arithmetic wraps
# mod 2^32 (two's complement == the uint32 bit pattern; must match
# mask_export.py's v6 jnp path exactly). Unsigned threshold compare via
# sign-flip + sitofp (int-vector cmp limits on Mosaic); host-validated: the
# f32 compare agrees with the exact u32 compare with 0 disagreement and the
# keep fraction sits at rate +- 3e-4 (v19_hash_host.py).
_H_C1 = -1640531535  # 0x9E3779B1
_H_C2 = -2048144777  # 0x85EBCA77
_H_C3 = -1028477379  # 0xC2B2AE3D
_H_C4 = 668265263    # 0x27D4EB2F
_H_C5 = 374761393    # 0x165667B1


def _hash_keep_bool(w0, w1, h, q_abs, kv_abs, rate):
  """Vector i32 mix -> bool keep tile in the CALLER's layout."""
  i32 = jnp.int32
  v = (q_abs * i32(_H_C1)) ^ (kv_abs * i32(_H_C2)) ^ w0
  v = v * i32(_H_C3) + (h * i32(_H_C4) + w1)
  v = v ^ lax.shift_right_logical(v, 16)
  v = v * i32(_H_C5)
  v = v ^ lax.shift_right_logical(v, 13)
  thresh_u = min(int(rate * (2**32)), 2**32 - 1)
  t_flip = thresh_u ^ 0x80000000
  t_signed = t_flip - (1 << 32) if t_flip >= (1 << 31) else t_flip
  vs = (v ^ i32(-2147483648)).astype(jnp.float32)
  return vs >= jnp.float32(t_signed)

# mypy: ignore-errors


def _not(x: jax.Array | bool) -> jax.Array | bool:
  if isinstance(x, jax.Array):
    return jnp.logical_not(x)
  return not x


def _and(x: jax.Array | bool, y: jax.Array | bool) -> jax.Array | bool:
  # sbd fork: keeps a statically-known predicate static, so `pl.when` on the
  # non-dynamic-grid path still folds away instead of emitting a branch.
  if isinstance(x, jax.Array) or isinstance(y, jax.Array):
    return jnp.logical_and(x, y)
  return x and y


class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are a mechanism to ensure that there is no cross-attention between
  segments (fraction of a sequence) that have been concatenated together into a
  sequence. Each array is a list of ids (integers). Only tokens with the same
  id are allowed to attend to each other.

  The static mask (e.g. causal) is "and-ed" with the segment id mask to form
  the actual attention mask. It is important that the latter does not have any
  all-zero rows (along dimension kv). Otherwise it would result in a invalid
  softmax (the denominator would be 0).
  This condition holds for causal self-attention because in this case segment
  ids form a block diagonal matrix so at least one element in each row is set.
  It is easy to break this condition with non-self-attention configurations.
  Attributes:
    q: segment ids along the Q sequence
    kv: segment ids along the KV sequence
  """

  q: jax.Array  # [q_seq_len]
  kv: jax.Array  # [kv_seq_len]

MaskFunctionType = Callable[..., jax.Array]


def get_kernel_name(
    is_mqa: bool, save_residuals: bool, is_segmented: bool, phase: str
) -> str:
  """Returns a unique name for all SplashAttention kernel variants."""
  assert phase in ["dq", "dkv", "fwd"]
  # Saving residuals is supported only for the fwd phase.
  assert not save_residuals or phase == "fwd"
  residuals = "_residuals" if save_residuals else "_no_residuals"
  attention_type = "mqa" if is_mqa else "mha"
  segments = "_segmented" if is_segmented else ""
  return f"splash_{attention_type}_{phase}{segments}{residuals}"


# Splash attention implementation


# We use an IntEnum to make it JSON serializable as regen metadata.
class QKVLayout(enum.IntEnum):
  HEAD_DIM_MINOR = enum.auto()  # [..., seq_len, head_dim]
  SEQ_MINOR = enum.auto()  # [..., head_dim, seq_len]


def from_head_minor(vals: tuple[Any, ...], layout: QKVLayout):
  if layout == QKVLayout.HEAD_DIM_MINOR:
    return vals
  return (*vals[:-2], vals[-1], vals[-2])


@dataclasses.dataclass(frozen=True, slots=True)
class SplashConfig:
  """Tile sizes parameterizing SplashAttention kernels.

  Those parameters have negligible effect on numerics, but affect performance
  greatly.

  Note that changing the layouts only influences the physical layout that the
  kernel will enforce. The logical interface to splash attention always takes
  the head dimension as the minormost one.
  """

  block_q: int
  block_kv: int
  block_kv_compute: int | None = None

  block_q_dkv: int | None = None
  block_kv_dkv: int | None = None
  block_kv_dkv_compute: int | None = None

  # TODO: Remove these 3 params, they're only kept for backwards compatibility.
  block_q_dq: int | None = None
  block_kv_dq: int | None = None
  use_fused_bwd_kernel: bool = True
  num_stacked_q_heads: int = 1
  q_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  k_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  v_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR

  fwd_cost_estimate: pl.CostEstimate | None = None
  bwd_cost_estimate: pl.CostEstimate | None = None

  residual_checkpoint_name: str | None = None  # whether to checkpoint outputs
  attn_logits_soft_cap: float | None = None
  fuse_reciprocal: bool = True  # whether to compute o / lse inside the kernel
  use_base2_exp: bool = True
  max_logit_const: float | None = None
  interpret: bool = False
  # The fused bwd kernel accumulates dq at every grid step. To safely avoid
  # read/write conflicts we conservatively avoid *any* in-kernel reductions.
  # This parameter allows to override this behavior and specifies the number of
  # reduction steps. For now, only 3 or all the kv steps are supported.
  dq_reduction_steps: int | None = None
  # An experimental scheduler that sometimes produces better softmax overlap.
  use_experimental_scheduler: bool = False
  # Skip the wasted causal-diagonal QK matmul. On a partial-mask (diagonal) block,
  # split the QK matmul into a qk_diag_grid x qk_diag_grid sub-grid over (kv rows,
  # q cols) and skip every sub-tile that lies entirely above the causal line
  # (kv > q), filling mask_value instead of computing it. Those entries are
  # overwritten to mask_value by _apply_mask_and_soft_cap regardless, so the result
  # is bit-exact; the elementwise/softmax ops still run on the full assembled tile.
  # PRECONDITION (enforced below): pure CausalMask + aligned SQUARE blocks with a
  # single compute tile per block (block_q == block_kv == block_kv_compute, and the
  # backward trio), and sequence length a multiple of the block. Any other config
  # raises (it does not silently corrupt); a non-causal mask raises too.
  qk_diag_skip: bool = False
  # Granularity of the diagonal skip: grid=2 -> quadrants (skip 1/4 of the diagonal
  # block, per-block waste 1/2 -> 1/4); grid=4 -> 4x4 (skip 6/16, waste -> 1/8).
  # Larger grid skips more of the triangle but uses smaller (less MXU-efficient)
  # matmuls; grid=4 is a good default at S=4096/block=2048. Must be a power of 2.
  qk_diag_grid: int = 2
  # sbd fork: dropout on the attention probabilities (post-softmax-accumulation,
  # applied only to the PV-feeding copy; online-softmax state and logsumexp stay
  # undropped). The Bernoulli keep-mask is generated in-kernel per fixed
  # DROPOUT_QUANTUM x DROPOUT_QUANTUM tile from pltpu.prng_seed hashed on the
  # ABSOLUTE quantum coordinates (seed words, head, q_quantum, kv_quantum) --
  # block-size-independent by construction, so fwd/bwd block sizes stay free.
  # rate > 0 requires the dropout_seed operand (scalar-prefetched (2,) int32).
  dropout_rate: float = 0.0
  # sbd fork: mask-generation version. v1 = per-256x256-quantum draws (full
  # fwd/bwd block decoupling). v2 = one draw per (q-quantum row, kv COMPUTE
  # BAND) -- 4x fewer seed calls at blk-1024, but requires
  # block_kv_compute == block_kv_dkv_compute (band widths must match).
  dropout_mask_version: int = 1
  # sbd fork v004 cand4: generate the dropout mask at the TOP of the compute
  # body (before the QK matmul) so the scheduler can overlap the RNG chain
  # with MXU work. Mask values unchanged (same version/keying) -- pure
  # intra-block reordering.
  dropout_hoist_gen: bool = False
  # sbd fork v005: apply-side cost knobs. Keying and realized keep pattern are
  # UNTOUCHED by all three (mask-version semantics unchanged; round-trip vs
  # mask_export stays exact).
  # fold_rescale: no per-element 1/(1-rate) multiply exists in any kernel
  # body -- fwd folds it into the epilogue l_inv (or the o write); bwd
  # prescales the do tile once (serves BOTH dv and dp: dv = p_m^T (r*do),
  # dp_s = v (r*do)^T). Stated float reassociation vs the unfolded arm.
  dropout_fold_rescale: bool = False
  # mask_as_multiplier: apply keep as a {0,1} f32 multiply instead of a
  # where-select (v1 PRNG + explicit-mask paths only).
  dropout_mask_as_multiplier: bool = False
  # sbd fork v013: per-key bias rank. When > 0, the bias operand is the
  # PADDED per-key layout [1|H, bias_per_key_rows, kv_seq] (rows identical;
  # row 0 is used; 16 rows = the bf16 tile minimum). The BlockSpec pins the
  # q-block index to 0, so traffic is O(rows * K) per block-column instead of
  # O(bq * K) -- 256x less than full-rank at blk-1024. dbias for this rank is
  # NOT supported this version (pre-named follow-on).
  bias_per_key_rows: int = 0
  # sbd fork v021 (V13-L1): rank-reduced dbias for the per-key rank --
  # dbias_pk[kv] = sum over q (and bands) of ds[kv, q], accumulated across
  # the dkv grid's q-block revisits of the q-pinned (rows, bkv) tile via
  # read-modify-write on an f32 output (aliased zero input on the dynamic
  # grid). Head-FULL per-key bias only (head-broadcast dbias needs
  # cross-head accumulation across non-consecutive grid revisits -- named
  # lead, not implemented). Unlocks trainable per-key biases.
  bias_per_key_dbias: bool = False
  # v021 cand4: reduction realization -- MXU ones-matmul form (ones(1,bq) @
  # ds^T via dot_general) instead of the cross-lane XLU row-sum. Invariant
  # identical (each ds tile reduced once over q); pick per measurement.
  bias_per_key_dbias_mxu_reduce: bool = False
  # v021 cand4c: scratch-accumulate realization -- accumulate into a VMEM
  # scratch tile (the dk/dv pattern) and write the output ONCE per (head,
  # kv-block) sweep, removing the per-step output-block read-modify-write.
  bias_per_key_dbias_scratch: bool = False
  # single_bwd_apply: mask applied ONCE in dkv (to p); dp stays unmasked and
  # ds uses the distributed form ds = dp_s * p_m - di * p (algebraically equal
  # to (masked dp_s - di) * p since masked(dp)*p == dp*masked(p)). Requires
  # fold_rescale (the reuse identity needs unscaled p_m and prescaled dp).
  dropout_single_bwd_apply: bool = False
  # sbd fork v016 (V15-L1, dkv issue-pressure): cand2 -- transpose the
  # PRE-COMPARE f32 values once and compare after, producing keep_t [kv, q]
  # directly; deletes the bool->f32 astype + the != 0.0 recheck (2 full-tile
  # ops). Mask pattern identical (same draws, same threshold semantics).
  # v4-path only; incompatible with dropout_mask_as_multiplier.
  dropout_bwd_fused_compare: bool = False
  # sbd fork v016 cand5 (hybrid of cand2+cand3, post-win refinement): draw
  # values cast to BF16 at creation (i8-range ints, exact in bf16), ONE bf16
  # transpose (half of cand2's f32 transpose bytes), ONE bf16 compare after
  # -> keep [kv, q] bool; no bool round trip. Bit-exact keep pattern.
  dropout_bwd_fused_compare_bf16: bool = False
  # sbd fork v016 cand3: carry the dkv-side keep mask as a bf16 0/1
  # MULTIPLIER: compare q-major -> astype(bf16) -> ONE bf16 transpose (half
  # the XLU bytes of the f32 .T) -> both applies become multiplies
  # (p_pv = p16 * keep16; dp = dp * keep16-promoted). Bit-exact vs the
  # select form (0*x == 0, 1*x == x exactly); halves the mask live-range
  # bytes and removes both full-tile selects. v4-path only.
  dropout_bwd_bf16_mask: bool = False
  # sbd fork v016 cand4: disable the dkv inner fori_loop unroll when PRNG
  # dropout is on (the dropout body is ~1.8x the code of featoff; full
  # unroll multiplies its live ranges -> spills). featoff path unaffected.
  dropout_dkv_no_unroll: bool = False
  # sbd fork v006 PROBE knob (never a shipping config): replace the generated
  # keep with a trivially-true tile built from iota + one runtime scalar
  # compare (no draws, no assembly) -- prices the APPLY side without
  # construction. Semantically keep-all; pair with rate ~1e-10 so
  # inv_keep == 1.0 exactly and outputs are identity vs features-off.
  dropout_cheap_construction: bool = False
  # sbd fork v006 cand3: dkv keep regen assembles the drawn bits in the
  # INTEGER domain q-major (exactly as the fwd path does), then does ONE
  # threshold + ONE astype + ONE transpose on the assembled tile -- replacing
  # the per-piece compare/astype/.T chain (16 of each at blk-1024). The draws
  # and threshold values are unchanged, so the realized keep pattern is
  # bit-identical to v1 (round-trip EXACT preserved). v1 masks only.
  dropout_int_construction: bool = False
  # sbd fork v006 PROBE (parity-broken-by-design, never ships): ONE
  # prng draw per compute tile, tiled across the block -- isolates whether
  # the construction cost is per-draw latency (cost ~ 1/16 of v1's) or a
  # fixed PRNG-regime serialization (cost unchanged). Changes the mask
  # function; keep-fraction still ~1-rate.
  dropout_probe_single_draw: bool = False
  # sbd fork v007 cand6 PROBE (cost-curve point, no export version): v5-style
  # single chain drawing HALF area ((bq/2, bkv) i32), pltpu.bitcast to i16
  # (sublane x2 expand) + xor/sitofp/cmpf threshold at 1/65536 resolution.
  # Separates volume from chain count INSIDE the one-chain regime.
  dropout_probe_half_area: bool = False
  # sbd fork v006 cand3b: dropout_mask_version=4 -- quarter-area band draws
  # with SUBLANE-EXPANDED uint8 layout: the (qt/4, band) int32 draw is
  # pltpu.bitcast to (qt, band) uint8 (hardware sublane packing defines the
  # byte->row map), thresholded with ONE uint8 compare at 1/256 rate
  # resolution. No per-lane shift/and/compare/astype chain (the cost that
  # masked v3's area reduction in v004). Round-trip exact by construction:
  # mask_export version=4 runs the identical draw+bitcast on-chip.
  # Same band-width restriction as v2/v3.

  def __post_init__(self):
    if self.block_kv_compute is None:
      object.__setattr__(self, "block_kv_compute", self.block_kv)
    if self.block_kv_dkv_compute is None:
      object.__setattr__(self, "block_kv_dkv_compute", self.block_kv_dkv)

    if self.dq_reduction_steps is not None and self.dq_reduction_steps != 3:
      raise ValueError(
          f"Invalid dq_reduction_steps: {self.dq_reduction_steps}, only 3 or"
          " None are supported."
      )
    if not self.use_fused_bwd_kernel:
      raise ValueError("Only the fused bwd kernel is supported.")

    if self.dropout_rate > 0.0 and self.dropout_mask_version in (2, 3, 4, 5):
      if self.block_kv_compute != self.block_kv_dkv_compute:
        raise ValueError(
            "dropout_mask_version=2 requires block_kv_compute =="
            f" block_kv_dkv_compute (band widths must match); got"
            f" {self.block_kv_compute} vs {self.block_kv_dkv_compute}"
        )

    if self.dropout_rate > 0.0 and self.dropout_mask_version == 5:
      if self.block_q != self.block_q_dkv:
        raise ValueError(
            "dropout_mask_version=5 keys per q-BLOCK: requires"
            f" block_q == block_q_dkv; got {self.block_q} vs"
            f" {self.block_q_dkv}"
        )

    if self.dropout_single_bwd_apply and not self.dropout_fold_rescale:
      raise ValueError(
          "dropout_single_bwd_apply requires dropout_fold_rescale (the"
          " dp*p_m distributed form needs unscaled p_m and prescaled dp)"
      )
    _v016_knobs = [
        self.dropout_bwd_fused_compare,
        self.dropout_bwd_bf16_mask,
        self.dropout_bwd_fused_compare_bf16,
    ]
    if any(_v016_knobs):
      if self.dropout_mask_version != 4:
        raise ValueError(
            "the v016 dkv knobs (fused_compare / bf16_mask /"
            " fused_compare_bf16) are implemented for mask version 4 only"
        )
      if sum(_v016_knobs) > 1:
        raise ValueError("the v016 dkv knobs are mutually exclusive")
      if self.dropout_mask_as_multiplier:
        raise ValueError(
            "v016 dkv knobs are incompatible with dropout_mask_as_multiplier"
        )
    if self.dropout_bwd_bf16_mask and not self.dropout_fold_rescale:
      raise ValueError(
          "dropout_bwd_bf16_mask requires dropout_fold_rescale (the bf16"
          " multiplier path has no exact non-fold rescale)"
      )
    if self.bias_per_key_dbias and self.bias_per_key_rows == 0:
      raise ValueError(
          "bias_per_key_dbias requires bias_per_key_rows > 0 (it is the"
          " rank-reduced gradient of the per-key bias)"
      )
    if self.dropout_mask_as_multiplier and self.dropout_mask_version != 1:
      raise ValueError(
          "dropout_mask_as_multiplier is implemented for mask version 1"
          " (and the explicit-mask operand) only"
      )
    if self.dropout_int_construction and self.dropout_mask_version != 1:
      raise ValueError(
          "dropout_int_construction is implemented for mask version 1 only"
      )

    if self.qk_diag_skip:
      # The skip fills mask_value for sub-tiles where kv > q, relying on the mask to
      # mask EXACTLY those. That holds only for aligned SQUARE blocks (kv-band > q-band
      # <=> fully above the causal line, single compute tile per block) — enforce it or
      # the skip silently corrupts. Causality is checked in _make_splash_attention.
      if not (self.block_q == self.block_kv == self.block_kv_compute):
        raise ValueError(
            "qk_diag_skip requires square forward blocks "
            "(block_q == block_kv == block_kv_compute); got "
            f"{self.block_q}/{self.block_kv}/{self.block_kv_compute}."
        )
      if self.has_backward_blocks and not (
          self.block_q_dkv == self.block_kv_dkv == self.block_kv_dkv_compute
      ):
        raise ValueError(
            "qk_diag_skip requires square backward blocks "
            "(block_q_dkv == block_kv_dkv == block_kv_dkv_compute); got "
            f"{self.block_q_dkv}/{self.block_kv_dkv}/{self.block_kv_dkv_compute}."
        )
      if self.qk_diag_grid < 2 or (self.qk_diag_grid & (self.qk_diag_grid - 1)):
        raise ValueError(
            f"qk_diag_grid must be a power of 2 >= 2; got {self.qk_diag_grid}."
        )

  @property
  def has_backward_blocks(self) -> bool:
    backward_blocks = (
        self.block_q_dkv,
        self.block_kv_dkv,
        self.block_kv_dkv_compute,
    )
    return all(b is not None for b in backward_blocks)

  @classmethod
  def get_default(cls):
    # TODO: Select better parameters based on a heuristic.
    return SplashConfig(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        block_q_dq=128,
        block_kv_dq=128,
        fuse_reciprocal=True,
    )


to_i32 = lambda x: x.astype(jnp.int32)


def _apply_mask_and_soft_cap(
    qk: jax.Array,
    mask_value: float,
    mask_ref,
    q_sequence_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    *,
    attn_logits_soft_cap: float | None,
    k_slice: pl.Slice,
    k_offset: int | jax.Array,
    bq: int,
    k_in_lanes=True,
    mask_function=None,
    has_partial_mask: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  assert mask_ref is None or q_sequence_ref is None
  assert (q_sequence_ref is None) == (mask_function is None)

  masks = []
  if has_partial_mask:
    if mask_ref is not None:
      mask = mask_ref[:, k_slice] if k_in_lanes else mask_ref[k_slice, :]
      masks.append(mask)
    elif mask_function is not None:
      # Compute the mask using the given q_sequence indices.
      # KV indices are computed on the fly. This works because we only support Q
      # sequence sharding. If we wanted to compute Q indices too, then we would
      # need to keep into account the current shard along Q sequence.

      if k_in_lanes:
        assert q_sequence_ref.shape == (bq, NUM_LANES)  # pyrefly: ignore[missing-attribute]

        k_sequence = k_offset + jax.lax.broadcasted_iota(
            jnp.int32, (bq, k_slice.size), 1
        )

        repeats, rem = divmod(k_slice.size, NUM_LANES)
        assert rem == 0
        q_sequence = jnp.tile(
            q_sequence_ref[...], (1, repeats)  # pyrefly: ignore[unsupported-operation]
        )  # [bq, k_slice.size]
      else:
        assert q_sequence_ref.shape == (NUM_SUBLANES, bq)  # pyrefly: ignore[missing-attribute]

        k_sequence = k_offset + jax.lax.broadcasted_iota(
            jnp.int32, (k_slice.size, bq), 0
        )
        q_sequence = q_sequence_ref[:1, :]  # [1, bq]  # pyrefly: ignore[unsupported-operation]
        q_sequence = jnp.broadcast_to(q_sequence, (k_slice.size, bq))

      assert q_sequence.shape == k_sequence.shape
      computed_mask = mask_function(q_sequence, k_sequence)  # pytype: disable=wrong-arg-count
      if computed_mask.dtype != jnp.dtype(jnp.bool_):
        raise ValueError(
            "Mask function must return a boolean-valued array, but got:"
            f" {computed_mask.dtype}"
        )
      masks.append(computed_mask)

  if q_segment_ids_ref is not None:
    if k_in_lanes:
      kv_ids = kv_segment_ids_ref[:1, k_slice]  # [1, k_slice]
      repeats, rem = divmod(kv_ids.shape[1], NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_kv must be a multiple of {NUM_LANES}")
      q_ids = jnp.tile(q_segment_ids_ref[:], (1, repeats))  # [bq, bkv]
    else:
      assert bq == q_segment_ids_ref.shape[-1]
      repeats, rem = divmod(bq, NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_q must be a multiple of {NUM_LANES}")
      kv_ids = jnp.tile(
          kv_segment_ids_ref[k_slice, :], (1, repeats)
      )  # [k_slice, bq]
      q_ids = q_segment_ids_ref[:1, :]  # [1, bq]
    masks.append(q_ids == kv_ids)

  def cap_logits(logits):
    if attn_logits_soft_cap is not None:
      logits = jnp.tanh(qk / attn_logits_soft_cap)
      return logits * attn_logits_soft_cap
    else:
      return logits

  if masks:
    mask = functools.reduce(jnp.logical_and, masks)
    qk = cap_logits(qk)
    if mask.ndim == 2 and qk.ndim == 3:
      mask = jnp.expand_dims(mask, axis=0)

    qk = jnp.where(mask, qk, mask_value)
  else:
    qk = cap_logits(qk)
  return qk


def flash_attention_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    bias_ref,
    dropout_mask_ref,
    dropout_seed_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    sinks_ref,
    mask_ref,
    q_sequence_ref,
    max_logit_value_ref,
    # Outputs
    o_ref,
    logsumexp_ref,
    l_linear_ref,
    max_logits_ref,
    # Scratch
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    *,
    mask_value: float,
    kv_steps: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    num_stacked_q_heads: int,
    mask_function: MaskFunctionType | None,
    bias_function=None,  # sbd fork: static f(h, q_pos, kv_pos) -> logits delta
    fuse_reciprocal: bool,  # config.fuse_reciprocal or not save_residuals
    config: SplashConfig,
):
  del mask_next_ref
  if config.dropout_rate == 0.0 and bias_function is None:
    del active_rows_ref  # only dropout/bias_function need the q-block index
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  # If the head_dim_v is not a multiple of the number of lanes, it will be
  # padded to that multiple with zeros.
  head_dim_v_repeats = pl.cdiv(head_dim_v, NUM_LANES)

  grid_idx = pl.program_id(1)
  h = pl.program_id(0)

  if block_mask_ref is not None:
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    # sbd fork: `block_mask == 0` means "visit this step, initialize and write
    # the row's output tile, compute nothing" -- the state the wrapper's
    # document-aware skipping puts a kept-but-empty row anchor in, and the
    # same convention the dkv kernel already carries. Without the gate a 0
    # would fall into the `!= 1` branch and be computed UNMASKED.
    should_run = block_mask_ref[grid_idx].astype(jnp.int32) != 0
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
    j = active_cols_ref[grid_idx].astype(jnp.int32)
    if config.dropout_rate > 0.0 or bias_function is not None:
      # sbd fork: PRNG hash / positional bias need the absolute q-block index.
      i_row = active_rows_ref[grid_idx].astype(jnp.int32)
  else:
    should_not_mask = False
    should_run = True
    j = grid_idx % kv_steps
    should_initialize = j == 0
    should_write = j == kv_steps - 1
    if config.dropout_rate > 0.0 or bias_function is not None:
      i_row = grid_idx // kv_steps

  max_logit_estimate = config.max_logit_const  # potentially None
  if max_logit_value_ref is not None:  # already ensures max_logit_const is None
    assert num_stacked_q_heads == 1
    max_logit_estimate = max_logit_value_ref[0, h]

  if config.use_base2_exp and max_logit_estimate is not None:
    max_logit_estimate *= LOG2E

  @pl.when(should_initialize)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

    sink = None
    if sinks_ref is not None:
      sink = sinks_ref[0, h].astype(m_scratch_ref.dtype)
      if config.use_base2_exp:
        sink *= LOG2E

    if sinks_ref is None and max_logit_estimate is None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    elif sinks_ref is None and max_logit_estimate is not None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    elif sinks_ref is not None and max_logit_estimate is None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, sink)  # pyrefly: ignore[bad-argument-type]
      l_scratch_ref[...] = jnp.ones_like(l_scratch_ref)
    else:  # sinks_ref is not None and max_logit_estimate is not None
      exp = jnp.exp2 if config.use_base2_exp else jnp.exp
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)  # pyrefly: ignore[bad-argument-type]
      l_scratch_ref[...] = exp(
          sink - jnp.full_like(l_scratch_ref, max_logit_estimate)  # pyrefly: ignore[bad-argument-type, unsupported-operation]
      )

  def _gen_dropout_keep(kv_compute_index, slice_k):
    if not (config.dropout_rate > 0.0 or dropout_mask_ref is not None):
      return None
    rate = config.dropout_rate
    if dropout_mask_ref is not None:
      return dropout_mask_ref[:, slice_k] != 0
    qt = DROPOUT_QUANTUM
    assert bq % qt == 0 and bkv_compute % qt == 0
    thresh = jnp.uint32(min(int(rate * (2**32)), 2**32 - 1))
    q_base_quant = i_row * (bq // qt)
    if config.dropout_mask_version == 3:
      t8 = jnp.uint32(min(int(rate * 256), 255))
      band = j * (bkv // bkv_compute) + kv_compute_index
      row_tiles = []
      for aq in range(bq // qt):
        w0, w1 = _mix_dropout_seed(
            dropout_seed_ref[0, 0], dropout_seed_ref[0, 1], h,
            q_base_quant + aq, band)
        pltpu.prng_seed(w0, w1)
        words = lax.bitcast_convert_type(
            pltpu.prng_random_bits((qt, bkv_compute // 4)), jnp.uint32)
        lanes = [((words >> (8 * l)) & jnp.uint32(0xFF)) >= t8
                 for l in range(4)]
        row_tiles.append(jnp.concatenate(
            [ln.astype(jnp.float32) for ln in lanes], axis=1))
      return jnp.concatenate(row_tiles, axis=0) != 0.0
    elif config.dropout_mask_version == 2:
      band = j * (bkv // bkv_compute) + kv_compute_index
      row_tiles = []
      for aq in range(bq // qt):
        w0, w1 = _mix_dropout_seed(
            dropout_seed_ref[0, 0], dropout_seed_ref[0, 1], h,
            q_base_quant + aq, band)
        pltpu.prng_seed(w0, w1)
        row_tiles.append(pltpu.prng_random_bits((qt, bkv_compute)))
      bits_full = jnp.concatenate(row_tiles, axis=0)
      return lax.bitcast_convert_type(bits_full, jnp.uint32) >= thresh
    else:
      kv_base_quant = j * (bkv // qt) + kv_compute_index * (bkv_compute // qt)
      row_tiles = []
      for aq in range(bq // qt):
        col_tiles = []
        for ak in range(bkv_compute // qt):
          w0, w1 = _mix_dropout_seed(
              dropout_seed_ref[0, 0], dropout_seed_ref[0, 1], h,
              q_base_quant + aq, kv_base_quant + ak)
          pltpu.prng_seed(w0, w1)
          col_tiles.append(pltpu.prng_random_bits((qt, qt)))
        row_tiles.append(jnp.concatenate(col_tiles, axis=1))
      bits_full = jnp.concatenate(row_tiles, axis=0)
      return lax.bitcast_convert_type(bits_full, jnp.uint32) >= thresh

  def body(kv_compute_index, _, has_partial_mask=False):
    slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
    keep_early = (
        _gen_dropout_keep(kv_compute_index, slice_k)
        if config.dropout_hoist_gen
        else None
    )
    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    assert m_prev.shape == (num_stacked_q_heads, bq, NUM_LANES)
    assert l_prev.shape == (num_stacked_q_heads, bq, NUM_LANES)

    q = q_ref[...] if config.q_layout == HEAD_DIM_MINOR else q_ref[...].mT
    if config.use_base2_exp:
      q *= LOG2E

    head_dim_qk = q.shape[-1]
    # Collapse the head and sequence dimensions for a larger matmul.
    q_flat = q.reshape((num_stacked_q_heads * bq, head_dim_qk))

    if config.k_layout == HEAD_DIM_MINOR:
      k = k_ref[slice_k, :]
      qk_dims = NT_DIM_NUMBERS
    else:
      k = k_ref[:, slice_k]
      qk_dims = NN_DIM_NUMBERS

    _g = config.qk_diag_grid
    if (
        config.qk_diag_skip
        and has_partial_mask
        and num_stacked_q_heads == 1
        and config.k_layout == HEAD_DIM_MINOR
        and bq % _g == 0
        and bkv_compute % _g == 0
    ):
      # Diagonal skip (forward): qk tile is [q, kv]. On an aligned square diagonal
      # block, sub-tile (q-band qi, kv-band kj) with kj > qi is fully above the causal
      # boundary (kv > q) -> masked to mask_value anyway -> skip its matmul.
      sq = bq // _g
      sk = bkv_compute // _g
      q_parts = [q_flat[i * sq:(i + 1) * sq, :] for i in range(_g)]
      k_parts = [k[j * sk:(j + 1) * sk, :] for j in range(_g)]
      rows = []
      for qi in range(_g):  # q row-band
        cols = []
        for kj in range(_g):  # kv col-band
          if kj > qi:  # fully masked -> skip matmul
            cols.append(jnp.full((sq, sk), mask_value, dtype=float32))
          else:
            cols.append(lax.dot_general(
                q_parts[qi], k_parts[kj], qk_dims, preferred_element_type=float32
            ))
        rows.append(jnp.concatenate(cols, axis=1))
      qk_flat = jnp.concatenate(rows, axis=0)
    else:
      qk_flat = lax.dot_general(
          q_flat, k, qk_dims, preferred_element_type=float32
      )
    qk = qk_flat.reshape((num_stacked_q_heads, bq, bkv_compute))

    if bias_ref is not None:
      # sbd fork (f9af38b resurrection): additive bias, pre-mask/pre-cap.
      assert num_stacked_q_heads == 1
      if config.bias_per_key_rows > 0:
        # v013: per-key row, broadcast over the query axis.
        bias = bias_ref[0:1, slice_k].astype(float32)
      else:
        bias = bias_ref[:, slice_k].astype(float32)
      if config.use_base2_exp:
        bias = bias * LOG2E
      qk = qk + bias[None]

    if bias_function is not None:
      # sbd fork: positional bias computed in-register -- zero HBM traffic.
      assert num_stacked_q_heads == 1
      q_pos = i_row * bq + lax.broadcasted_iota(
          jnp.int32, (bq, bkv_compute), 0
      )
      kv_pos = (
          j * bkv
          + kv_compute_index * bkv_compute
          + lax.broadcasted_iota(jnp.int32, (bq, bkv_compute), 1)
      )
      fbias = bias_function(h, q_pos, kv_pos).astype(float32)
      if config.use_base2_exp:
        fbias = fbias * LOG2E
      qk = qk + fbias[None]

    apply_mask_and_soft_cap = functools.partial(
        _apply_mask_and_soft_cap,
        qk,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=j * bkv + kv_compute_index * bkv_compute,
        bq=bq,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )

    qk = apply_mask_and_soft_cap()

    if max_logit_estimate is None:
      m_curr = qk.max(axis=-1)[..., None]  # pytype: disable=attribute-error
      assert m_curr.shape == (num_stacked_q_heads, bq, 1)
      m_next = jnp.maximum(m_prev, m_curr)
      assert m_next.shape == (num_stacked_q_heads, bq, NUM_LANES)
    else:
      m_next = None

    bkv_repeats, rem = divmod(bkv_compute, NUM_LANES)
    if rem != 0:
      raise NotImplementedError(
          f"{bkv_compute=} should be a multiple of {NUM_LANES}"
      )

    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    if max_logit_estimate is None:
      s_curr = exp(qk - jnp.tile(m_next, (1, 1, bkv_repeats)))  # pyrefly: ignore[bad-argument-type, unsupported-operation]
    else:
      s_curr = exp(qk - max_logit_estimate)  # pyrefly: ignore[unsupported-operation]
    assert s_curr.shape == (num_stacked_q_heads, bq, bkv_compute)

    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0, 1))
    assert l_curr.shape == (num_stacked_q_heads, bq, NUM_LANES)

    if max_logit_estimate is None:
      alpha = exp(m_prev - m_next)
      l_next = l_curr + alpha * l_prev
      m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next
    else:
      alpha = None
      l_scratch_ref[...] = l_curr + l_prev

    if config.dropout_rate > 0.0 or dropout_mask_ref is not None:
      # sbd fork: dropout applied ONLY to the PV-feeding copy of s_curr --
      # m/l scratch and logsumexp above are already final for this tile and
      # stay undropped (out = sum_j p_j * keep_j/(1-rate) * v_j exactly).
      assert num_stacked_q_heads == 1
      rate = config.dropout_rate
      inv_keep = (1.0 / (1.0 - rate)) if rate > 0.0 else 1.0
      if dropout_mask_ref is not None:
        keep = dropout_mask_ref[:, slice_k] != 0
      elif config.dropout_cheap_construction:
        # v006 probe p1: trivially-true keep, one iota + runtime scalar cmp
        # (unfoldable: compiler cannot prove the seed's sign) -- NO draws.
        _pos = lax.broadcasted_iota(jnp.int32, (bq, bkv_compute), 1)
        keep = _pos >= jnp.minimum(dropout_seed_ref[0, 0], 0)
      elif config.dropout_probe_single_draw:
        # v006 probe p3: ONE draw, tiled -- regime-presence vs per-draw cost.
        qt = DROPOUT_QUANTUM
        thresh = jnp.uint32(
            min(int(config.dropout_rate * (2**32)), 2**32 - 1)
        )
        w0, w1 = _mix_dropout_seed(
            dropout_seed_ref[0, 0], dropout_seed_ref[0, 1], h, 0, 0
        )
        pltpu.prng_seed(w0, w1)
        bits1 = pltpu.prng_random_bits((qt, qt))
        bits_full = jnp.tile(bits1, (bq // qt, bkv_compute // qt))
        keep = lax.bitcast_convert_type(bits_full, jnp.uint32) >= thresh
      else:
        qt = DROPOUT_QUANTUM
        assert bq % qt == 0 and bkv_compute % qt == 0, (
            "block sizes must be multiples of DROPOUT_QUANTUM"
        )
        thresh = jnp.uint32(min(int(rate * (2**32)), 2**32 - 1))
        q_base_quant = i_row * (bq // qt)
        if config.dropout_mask_version == 6:
          # v019 v6: orientation-free hash, q-major native -- no chain, no
          # bitcast, no band coupling (fwd/bwd block sizes decouple).
          q_abs = i_row * bq + lax.broadcasted_iota(
              jnp.int32, (bq, bkv_compute), 0
          )
          kv_abs = (
              j * bkv + kv_compute_index * bkv_compute
          ) + lax.broadcasted_iota(jnp.int32, (bq, bkv_compute), 1)
          keep = _hash_keep_bool(
              dropout_seed_ref[0, 0], dropout_seed_ref[0, 1], h,
              q_abs, kv_abs, rate,
          )
          bits_full = None
        elif config.dropout_mask_version == 5:
          # v5: ONE seed chain per (q-block, band) -- full-area draw, int32
          # threshold, zero extraction ops. Square-block restriction enforced.
          band = j * (bkv // bkv_compute) + kv_compute_index
          w0, w1 = _mix_dropout_seed(
              dropout_seed_ref[0, 0], dropout_seed_ref[0, 1], h, i_row, band,
          )
          pltpu.prng_seed(w0, w1)
          if config.dropout_probe_half_area:
            # v007 cand6 probe: half-volume single chain, i16 lanes.
            words = pltpu.prng_random_bits((bq // 2, bkv_compute))
            by16 = pltpu.bitcast(words, jnp.int16)  # (bq, bkv_c)
            _t16v = min(int(rate * 65536), 65535) ^ 0x8000
            t16f = jnp.float32(_t16v - 65536 if _t16v >= 32768 else _t16v)
            xs16 = (by16 ^ jnp.int16(-32768)).astype(jnp.float32)
            keep = xs16 >= t16f
          else:
            bits5 = pltpu.prng_random_bits((bq, bkv_compute))
            keep = lax.bitcast_convert_type(bits5, jnp.uint32) >= thresh
          bits_full = None
        elif config.dropout_mask_version == 4:
          # v4: quarter-area band draw, byte-interleaved uint8 view, ONE cmp.
          _t8v = min(int(rate * 256), 255) ^ 0x80
          t8f = jnp.float32(_t8v - 256 if _t8v >= 128 else _t8v)
          band = j * (bkv // bkv_compute) + kv_compute_index
          row_tiles = []
          for aq in range(bq // qt):
            w0, w1 = _mix_dropout_seed(
                dropout_seed_ref[0, 0], dropout_seed_ref[0, 1], h,
                q_base_quant + aq, band,
            )
            pltpu.prng_seed(w0, w1)
            words = pltpu.prng_random_bits((qt // 4, bkv_compute))
            by = pltpu.bitcast(words, jnp.int8)  # sublane-expand (qt, bkv_c)
            # unsigned >= via sign-flip in f32 (int8 cmp unsupported):
            xs = (by ^ jnp.int8(-128)).astype(jnp.float32)
            row_tiles.append(xs >= t8f)
          keep = jnp.concatenate(row_tiles, axis=0)
          bits_full = None
        elif config.dropout_mask_version == 3:
          # v3: 4 keep-columns per drawn word (int8 lanes, lane-major column
          # layout -- v3's own mask function; rate quantized to 1/256).
          t8 = jnp.uint32(min(int(rate * 256), 255))
          band = j * (bkv // bkv_compute) + kv_compute_index
          row_tiles = []
          for aq in range(bq // qt):
            w0, w1 = _mix_dropout_seed(
                dropout_seed_ref[0, 0],
                dropout_seed_ref[0, 1],
                h,
                q_base_quant + aq,
                band,
            )
            pltpu.prng_seed(w0, w1)
            words = lax.bitcast_convert_type(
                pltpu.prng_random_bits((qt, bkv_compute // 4)), jnp.uint32
            )
            lanes = [
                ((words >> (8 * l)) & jnp.uint32(0xFF)) >= t8
                for l in range(4)
            ]
            row_tiles.append(
                jnp.concatenate(
                    [ln.astype(jnp.float32) for ln in lanes], axis=1
                )
            )
          keep = jnp.concatenate(row_tiles, axis=0) != 0.0
          bits_full = None
        elif config.dropout_mask_version == 2:
          # v2: one draw per q-quantum row covering the whole compute band.
          band = j * (bkv // bkv_compute) + kv_compute_index
          row_tiles = []
          for aq in range(bq // qt):
            w0, w1 = _mix_dropout_seed(
                dropout_seed_ref[0, 0],
                dropout_seed_ref[0, 1],
                h,
                q_base_quant + aq,
                band,
            )
            pltpu.prng_seed(w0, w1)
            row_tiles.append(pltpu.prng_random_bits((qt, bkv_compute)))
          bits_full = jnp.concatenate(row_tiles, axis=0)
        else:
          kv_base_quant = j * (bkv // qt) + kv_compute_index * (
              bkv_compute // qt
          )
          row_tiles = []
          for aq in range(bq // qt):
            col_tiles = []
            for ak in range(bkv_compute // qt):
              w0, w1 = _mix_dropout_seed(
                  dropout_seed_ref[0, 0],
                  dropout_seed_ref[0, 1],
                  h,
                  q_base_quant + aq,
                  kv_base_quant + ak,
              )
              pltpu.prng_seed(w0, w1)
              col_tiles.append(pltpu.prng_random_bits((qt, qt)))
            row_tiles.append(jnp.concatenate(col_tiles, axis=1))
          bits_full = jnp.concatenate(row_tiles, axis=0)
        if bits_full is not None:
          keep = lax.bitcast_convert_type(bits_full, jnp.uint32) >= thresh
      # sbd fork v005: apply-form knobs (keep pattern identical in all forms).
      if config.dropout_mask_as_multiplier:
        s_curr = s_curr * keep.astype(s_curr.dtype)[None]
      else:
        s_curr = jnp.where(keep[None], s_curr, 0.0)
      if not config.dropout_fold_rescale:
        s_curr = s_curr * inv_keep

    s_curr_flat = s_curr.reshape((num_stacked_q_heads * bq, bkv_compute))

    if config.v_layout == HEAD_DIM_MINOR:
      v = v_ref[slice_k, :]
      sv_dims = NN_DIM_NUMBERS
    else:
      v = v_ref[:, slice_k]
      sv_dims = NT_DIM_NUMBERS

    o_curr_flat = lax.dot_general(s_curr_flat, v, sv_dims)
    o_curr = o_curr_flat.reshape((num_stacked_q_heads, bq, head_dim_v))

    if max_logit_estimate is None:
      alpha_o = jnp.tile(alpha, (1, 1, head_dim_v_repeats))  # pyrefly: ignore[bad-argument-type]
      alpha_o = alpha_o[..., : o_scratch_ref.shape[-1]]
      o_scratch_ref[...] = alpha_o * o_scratch_ref[...] + o_curr
    else:
      o_scratch_ref[...] = o_scratch_ref[...] + o_curr

  assert bkv % bkv_compute == 0
  num_iters = (
      k_ref.shape[0 if config.k_layout == HEAD_DIM_MINOR else 1] // bkv_compute
  )

  # Tried 2026-10-28 on Flint's op-point (32 q / 4 kv heads, head_dim 128,
  # sinks, window 513 + causal, packed documents; v6e), both bit-identical,
  # neither kept:
  # - Software-pipelining the grid step (QK^T of sub-step i+1 issued before
  #   sub-step i's softmax, m/l/o carried as values): no gain. With >= 2
  #   compute sub-steps the unrolled loop is one basic block and the LLO
  #   scheduler already co-issues the MXU with VALU work; the forward is bound
  #   by VALU work per tile (softmax + mask), not by missing overlap.
  # - Skipping the segment-id compare on tiles whose q and kv blocks lie in one
  #   segment (a block_mask bit set by the wrapper): -0.8% fwd+bwd.
  @pl.when(_and(should_not_mask, should_run))
  def _():
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(_and(_not(should_not_mask), should_run))
  def _():
    lax.fori_loop(
        0, num_iters, partial(body, has_partial_mask=True), None, unroll=True
    )

  @pl.when(should_write)
  def end():
    l = l_scratch_ref[...]
    m = m_scratch_ref[...]
    # sbd fork v005 (fold_rescale): the uniform 1/(1-rate) factors out of the
    # PV accumulation and rides the existing epilogue -- zero extra ops on the
    # fuse_reciprocal path (a scalar numerator on the same divide).
    _numer = 1.0
    if config.dropout_fold_rescale and config.dropout_rate > 0.0:
      _numer = 1.0 / (1.0 - config.dropout_rate)
    if fuse_reciprocal:  # allows fusing reciprocal out of the kernel
      # A row with no unmasked key (l == 0) outputs 0 rather than 0 * inf.
      l_inv = jnp.tile(
          jnp.where(l == 0.0, 0.0, _numer / l), (1, 1, head_dim_v_repeats))
      l_inv = l_inv[..., : o_scratch_ref.shape[-1]]
      o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
    elif _numer != 1.0:
      o_ref[...] = (o_scratch_ref[...] * _numer).astype(o_ref.dtype)
    else:
      o_ref[...] = o_scratch_ref[...].astype(o_ref.dtype)
    if logsumexp_ref is not None:
      assert logsumexp_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
      log = jnp.log2 if config.use_base2_exp else jnp.log
      logsumexp = m + log(l)
      logsumexp_ref[...] = logsumexp.astype(logsumexp_ref.dtype)
    if l_linear_ref is not None:
      assert l_linear_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
      l_linear_ref[...] = l.astype(l_linear_ref.dtype)
    if max_logits_ref is not None:
      assert max_logits_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
      max_logits_ref[...] = m.astype(max_logits_ref.dtype)


def _div(dividend: int, divisor: int):
  if divisor == 1:
    return dividend

  return lax.div(dividend, divisor)


def _bytes(x: jax.Array | jax.ShapeDtypeStruct | None) -> int:
  if x is None:
    return 0

  if jnp.issubdtype(x.dtype, jnp.floating):
    info = jnp.finfo
  elif jnp.issubdtype(x.dtype, jnp.integer):
    info = jnp.iinfo
  else:
    raise ValueError(f"Unsupported dtype: {x.dtype}")
  return math.ceil(math.prod(x.shape) * info(x.dtype).bits / 8)


def _splash_attention_forward(
    mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    bias: jax.Array | None,
    dropout_mask: jax.Array | None,
    dropout_seed: jax.Array | None,
    segment_ids: base.SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    save_residuals: bool,
    mask_function: MaskFunctionType | None,
    bias_function=None,
    fwd_mask_sparsity: float = 1.0,
    max_logit_value: jax.Array | None = None,
    save_max_logits: bool = True,
) -> base.SplashCustomReturnType:
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = config.block_q, config.block_kv
  bkv_compute = config.block_kv_compute
  fuse_reciprocal = config.fuse_reciprocal or not save_residuals
  bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)  # pyrefly: ignore[bad-argument-type]
  num_stacked_q_heads = config.num_stacked_q_heads

  if num_stacked_q_heads > 1 and (
      sinks is not None or max_logit_value is not None
  ):
    raise ValueError(
        "Stacked heads are not supported with sinks or max_logit_value."
    )

  if is_mqa:
    expected_kv_rank = 2
    num_kv_heads = 1
  else:
    expected_kv_rank = 3
    num_kv_heads = k.shape[0]

  if len(k.shape) != expected_kv_rank:
    raise ValueError(
        f"Expected {expected_kv_rank}-dim 'key' tensor for MQA. Instead got a"
        f" {len(k.shape)}-dim one."
    )

  if k.shape[-1] != head_dim_qk:
    raise ValueError(
        f"Expected 'key' head dimension to be: {head_dim_qk}. Instead got:"
        f" {k.shape[-1]}."
    )

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if num_q_heads % num_stacked_q_heads != 0:
    raise ValueError(
        f"{num_q_heads=} must be a multiple of {num_stacked_q_heads=}"
    )

  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if q_heads_per_kv_head % num_stacked_q_heads != 0:
    raise ValueError(
        f"{q_heads_per_kv_head=} must be a multiple of {num_stacked_q_heads=}"
    )

  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(
        f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same "
        "leading dimensions."
    )

  if bkv % bkv_compute:  # pyrefly: ignore[unsupported-operation]
    raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
  if bkv_compute % NUM_LANES:  # pyrefly: ignore[unsupported-operation]
    raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")

  if bias is not None:
    # sbd fork: [num_bias_heads, q_seq, kv_seq]; head dim 1 broadcasts.
    if num_stacked_q_heads != 1:
      raise NotImplementedError("bias with num_stacked_q_heads > 1")
    if config.bias_per_key_rows > 0:
      # v013 per-key rank: padded layout [1|H, rows, kv_seq], rows identical.
      if (bias.ndim != 3 or bias.shape[0] not in (1, num_q_heads)
          or bias.shape[1] != config.bias_per_key_rows):
        raise ValueError(
            f"per-key bias expects (1 | {num_q_heads=},"
            f" {config.bias_per_key_rows}, kv_seq); got {bias.shape}"
        )
    elif bias.ndim != 3 or bias.shape[0] not in (1, num_q_heads):
      raise ValueError(
          f"Expected bias of shape (1 | {num_q_heads=}, q_seq, kv_seq); got"
          f" {bias.shape}"
      )

  # sbd fork: dropout input validation.
  if dropout_mask is not None:
    if num_stacked_q_heads != 1:
      raise NotImplementedError("dropout with num_stacked_q_heads > 1")
    if dropout_mask.ndim != 3 or dropout_mask.shape[0] not in (1, num_q_heads):
      raise ValueError(
          f"Expected dropout_mask of shape (1 | {num_q_heads=}, q_seq,"
          f" kv_seq); got {dropout_mask.shape}"
      )
  if config.dropout_rate > 0.0 and dropout_mask is None:
    if num_stacked_q_heads != 1:
      raise NotImplementedError("dropout with num_stacked_q_heads > 1")
    if dropout_seed is None:
      raise ValueError("dropout_rate > 0 requires dropout_seed (2,) int32")
    if dropout_seed.shape != (2,):
      raise ValueError(f"Expected dropout_seed (2,); got {dropout_seed.shape}")

  kv_seq_len = k.shape[-2]
  kv_steps = kv_seq_len // bkv
  dynamic_grid = mask_info.active_rows is not None

  if segment_ids is not None:
    assert isinstance(segment_ids.q, jax.Array)  # for pytype
    assert isinstance(segment_ids.kv, jax.Array)  # for pytype
    if segment_ids.q.shape != (q_seq_len,):
      raise ValueError(
          "Invalid shape for q segment_ids: "
          f"{segment_ids.q.shape}. Expected: {(q_seq_len,)}"
      )
    if segment_ids.kv.shape != (kv_seq_len,):
      raise ValueError(
          "Invalid shape for kv segment_ids: "
          f"{segment_ids.kv.shape}. Expected: {(kv_seq_len,)}"
      )
  if config.max_logit_const is not None and max_logit_value is not None:
    raise ValueError(
        f"Only one of {config.max_logit_const=} and"
        f" {max_logit_value=} can be set."
    )
  if max_logit_value is not None:
    if max_logit_value.shape not in ((), (1,), (num_q_heads,)):
      raise ValueError(
          "max_logit_value should be a 0,1-dim jax.Array of shape (), (1,) or"
          f" ({num_q_heads=},) but got {jax.typeof(max_logit_value)}"
      )
    max_logit_value = jnp.broadcast_to(
        jnp.atleast_1d(max_logit_value), (num_q_heads,)
    )

  q_layout = config.q_layout
  k_layout = config.k_layout
  v_layout = config.v_layout

  def unravel(f):
    def index_map(h_block, grid_idx, rows_ref, cols_ref, *_):
      if dynamic_grid:
        i = to_i32(rows_ref[grid_idx])
        j = to_i32(cols_ref[grid_idx])
      else:
        i = grid_idx // kv_steps
        j = grid_idx % kv_steps
      return f(h_block, i, j)

    return index_map

  def create_kv_index_map(layout):
    def index_map(h_block, i, j):
      del i  # Unused.
      first_h_in_block = h_block * num_stacked_q_heads
      prefix = () if is_mqa else (_div(first_h_in_block, q_heads_per_kv_head),)
      return from_head_minor((*prefix, j, 0), layout)

    return index_map

  q_index_map = unravel(
      lambda h_block, i, j: from_head_minor((h_block, i, 0), q_layout)
  )
  out_index_map = unravel(lambda h_block, i, j: (h_block, i, 0))
  k_index_map = unravel(create_kv_index_map(k_layout))
  v_index_map = unravel(create_kv_index_map(v_layout))

  def mask_index_map(
      h_block, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_
  ):
    del h_block, rows_ref, cols_ref  # Unused.
    next_m = to_i32(mask_next_ref[grid_idx])  # pyrefly: ignore[unsupported-operation]
    return next_m, 0, 0

  q_segment_ids_index_map = unravel(lambda h_block, i, j: (i, 0))
  kv_segment_ids_index_map = unravel(lambda h_block, i, j: (0, j))
  # sbd fork: head-broadcast bias rides block index 0 on the head axis.
  bias_head_bcast = bias is not None and bias.shape[0] == 1
  bias_index_map = unravel(
      lambda h_block, i, j: (0 if bias_head_bcast else h_block, i, j)
  )
  dmask_head_bcast = dropout_mask is not None and dropout_mask.shape[0] == 1
  dropout_mask_index_map = unravel(
      lambda h_block, i, j: (0 if dmask_head_bcast else h_block, i, j)
  )

  # Convert the logical shape from head-minor to sequence-minor.
  in_specs = [
      pl.BlockSpec(
          from_head_minor((num_stacked_q_heads, bq, head_dim_qk), q_layout),
          q_index_map,
      ),
      pl.BlockSpec(
          from_head_minor(
              (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
              k_layout,
          ),
          k_index_map,
      ),
      pl.BlockSpec(
          from_head_minor(
              (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v), v_layout
          ),
          v_index_map,
      ),
  ]
  if bias is not None:
    if config.bias_per_key_rows > 0:
      _pk_rows = config.bias_per_key_rows
      _pk_map = unravel(
          lambda h_block, i, j: (0 if bias_head_bcast else h_block, 0, j)
      )
      in_specs.append(pl.BlockSpec((None, _pk_rows, bkv), _pk_map))
    else:
      in_specs.append(pl.BlockSpec((None, bq, bkv), bias_index_map))
  else:
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]
  if dropout_mask is not None:
    in_specs.append(pl.BlockSpec((None, bq, bkv), dropout_mask_index_map))
  else:
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]
  # sbd fork: the dropout seed rides as a regular SMEM operand, sublane-aligned
  # like sinks (NOT scalar prefetch -- a batched prefetch operand breaks the
  # vmap batching rule: "Wrong number of window_params").
  use_dropout_seed = config.dropout_rate > 0.0 and dropout_mask is None
  if use_dropout_seed:
    in_specs.append(
        pl.BlockSpec(
            (NUM_SUBLANES, 2),
            lambda *_: (0, 0),
            memory_space=pltpu.SMEM,
        )
    )
    dropout_seed = jnp.broadcast_to(
        dropout_seed.astype(jnp.int32)[None, :], (NUM_SUBLANES, 2)
    )
  else:
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]
    dropout_seed = None
  if segment_ids is not None:
    in_specs += [
        pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map),
        pl.BlockSpec((NUM_SUBLANES, bkv), kv_segment_ids_index_map),
    ]
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (q_seq_len, NUM_LANES), (0,)  # pyrefly: ignore[bad-argument-type]
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,)  # pyrefly: ignore[bad-argument-type]
    )
  else:
    in_specs += [None, None]
    q_segment_ids = kv_segment_ids = None

  if sinks is not None:
    assert sinks.shape == (num_q_heads,), f"{sinks.shape=} != {num_q_heads=}"
    # align sinks to sublanes to allow vmap and shard_map over the kernel
    in_specs += [
        pl.BlockSpec(
            (NUM_SUBLANES, num_q_heads),
            lambda h, i, j, *_: (0, 0),
            memory_space=pltpu.SMEM,
        )
    ]
    sinks = jnp.broadcast_to(
        sinks.astype(jnp.float32)[None, :], (NUM_SUBLANES, num_q_heads)
    )
  else:
    in_specs += [None]

  if mask_info.partial_mask_blocks is not None:
    in_specs.append(pl.BlockSpec((None, bq, bkv), mask_index_map))
  else:
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]

  assert mask_info.partial_mask_blocks is None or mask_info.q_sequence is None

  if mask_info.q_sequence is not None:
    q_sequence = jax.lax.broadcast_in_dim(
        mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,)
    )
    in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
  else:
    q_sequence = None
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]

  if max_logit_value is not None:
    # reshape to allow sublane selection for vmap-ping and shard_map-ping
    max_logit_value = jnp.broadcast_to(
        max_logit_value.astype(jnp.float32)[None, :],
        (NUM_SUBLANES, num_q_heads),
    )
    in_specs += [
        pl.BlockSpec(
            (NUM_SUBLANES, num_q_heads),
            lambda *_: (0, 0),
            memory_space=pltpu.SMEM,
        )
    ]
  else:
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]

  out_shapes = [
      jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim_v), q.dtype),
  ]
  out_specs = [
      pl.BlockSpec((num_stacked_q_heads, bq, head_dim_v), out_index_map),
  ]
  if save_residuals:
    logsumexp_index_map = unravel(lambda h_block, i, j, *_: (h_block, i, 0))
    # sbd fork: `max_logits` is a lane-padded (num_q_heads, q_seq_len, 128)
    # fp32 buffer -- 268 MB at the HF lane's 16k x 32-head op-point -- and a
    # custom call's outputs are opaque to XLA, so one the caller never reads
    # is allocated all the same. `fuse_reciprocal` needs it to rebuild
    # logsumexp; nothing else does, so a caller that wants only logsumexp
    # can opt out.
    need_max_logits = save_max_logits or not fuse_reciprocal

    out_shapes += [
        # logsumexp
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32)
        if fuse_reciprocal
        else None,
        # l_linear
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32)
        if not fuse_reciprocal
        else None,
        # max_logits
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32)
        if need_max_logits
        else None,
    ]
    out_specs += [
        pl.BlockSpec(
            (num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map
        )
        if fuse_reciprocal
        else None,
        pl.BlockSpec(
            (num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map
        )
        if not fuse_reciprocal
        else None,
        pl.BlockSpec(
            (num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map
        )
        if need_max_logits
        else None,
    ]
  else:
    out_shapes += [None, None, None]
    out_specs += [None, None, None]

  kernel_name = get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      is_segmented=segment_ids is not None,
      phase="fwd",
  )
  metadata = {"xprof_metadata": json.dumps(dataclasses.asdict(config))}

  def _fwd_cost_estimate(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      bias: jax.Array | None,
      dropout_mask: jax.Array | None,
      q_segment_ids: jax.Array | None,
      kv_segment_ids: jax.Array | None,
      partial_mask_blocks: jax.Array | None,
      out_shapes: list[jax.ShapeDtypeStruct],
      mask_sparsity: float,
  ) -> pl.CostEstimate:
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    kv_seq_len, head_dim_v = v.shape[-2:]

    matmul_flops = (
        2 * q_seq_len * kv_seq_len * head_dim_qk
        + 2 * q_seq_len * kv_seq_len * head_dim_v
    )

    # This is an upper bound because `mask_sparsity` is actually the mean
    # sparsity of the non-fully masked **blocks**.
    total_flops = num_q_heads * matmul_flops * mask_sparsity

    # Count expensive exp() calls
    transcendentals = num_q_heads * q_seq_len * kv_seq_len * mask_sparsity

    inputs_ = [
        q, k, v, bias, dropout_mask, q_segment_ids, kv_segment_ids,
        partial_mask_blocks,
    ]
    input_bytes = sum(map(_bytes, inputs_))
    output_bytes = sum(map(_bytes, out_shapes))
    return pl.CostEstimate(
        flops=int(total_flops),
        transcendentals=int(transcendentals),
        bytes_accessed=int(input_bytes + output_bytes),
    )

  vmem_inputs = [
      q,
      k,
      v,
      bias,
      dropout_mask,
      q_segment_ids,
      kv_segment_ids,
      mask_info.partial_mask_blocks,
  ]
  cost_estimate = config.fwd_cost_estimate or _fwd_cost_estimate(
      *vmem_inputs, out_shapes, fwd_mask_sparsity  # pyrefly: ignore[bad-argument-count, bad-argument-type]
  )

  grid_size_h = num_q_heads // num_stacked_q_heads
  if dynamic_grid:
    num_active_blocks = mask_info.num_active_blocks[0]  # pyrefly: ignore[unsupported-operation]
    grid = (grid_size_h, num_active_blocks)
    is_empty_attention_block = num_active_blocks == 0
  else:
    grid = (grid_size_h, kv_steps * (q_seq_len // bq))
    is_empty_attention_block = False

  with jax.named_scope(kernel_name):
    all_out = pl.pallas_call(
        partial(
            flash_attention_kernel,
            mask_value=mask_value,
            kv_steps=kv_steps,
            bq=bq,
            bkv=bkv,
            bkv_compute=bkv_compute,
            head_dim_v=head_dim_v,
            num_stacked_q_heads=num_stacked_q_heads,
            # note: fuse_reciprocal can only be False if save_residuals is True
            # fuse_reciprocal = (config.fuse_reciprocal or not save_residuals)
            fuse_reciprocal=fuse_reciprocal,
            config=config,
            mask_function=mask_function,
            bias_function=bias_function,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=[
                pltpu.VMEM(
                    (num_stacked_q_heads, bq, NUM_LANES), jnp.float32
                ),  # m_scratch
                pltpu.VMEM(
                    (num_stacked_q_heads, bq, NUM_LANES), jnp.float32
                ),  # l_scratch
                pltpu.VMEM(
                    (num_stacked_q_heads, bq, head_dim_v), jnp.float32
                ),  # o_scratch
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            flags={
                "XLA_TPU_FORCE_LP_LLO_SCHEDULER": (
                    config.use_experimental_scheduler
                )
            },
        ),
        out_shape=out_shapes,
        name=kernel_name,
        cost_estimate=cost_estimate,
        interpret=config.interpret,
        metadata=metadata,
    )(
        mask_info.active_rows,
        mask_info.active_cols,
        mask_info.mask_next,
        bounds_start,
        bounds_end,
        mask_info.block_mask,
        q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.mT,
        k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.mT,
        v if v_layout == QKVLayout.HEAD_DIM_MINOR else v.mT,
        bias,
        dropout_mask,
        dropout_seed,
        q_segment_ids,
        kv_segment_ids,
        sinks,
        mask_info.partial_mask_blocks,
        q_sequence,
        max_logit_value,
    )
  out, logsumexp, l_linear, max_logits = all_out

  # If there is no compute to do within an attention block, then we want to
  # initialize the output and residuals to default values. Otherwise, we will
  # read uninitialized memory. This is a common case in ring attention.
  def init_if_empty(x: jax.Array, value: float) -> jax.Array:
    if not dynamic_grid:
      return x

    return jnp.where(is_empty_attention_block, value, x)

  out = init_if_empty(out, 0.0)

  if save_residuals:
    if max_logits is not None:
      max_logits = init_if_empty(max_logits[..., 0], mask_value)

    if fuse_reciprocal:
      assert logsumexp is not None
      logsumexp = init_if_empty(logsumexp[..., 0], mask_value)
    else:
      assert l_linear is not None
      log = jnp.log2 if config.use_base2_exp else jnp.log

      l = l_linear[..., 0]
      logsumexp = max_logits + log(l)
      out = (out / l[..., None]).astype(out.dtype)
  else:
    # If we're not saving residuals, then we can't fuse the reciprocal
    # out of the kernel.
    assert fuse_reciprocal

  if config.residual_checkpoint_name is not None:
    out = ad_checkpoint.checkpoint_name(
        out, name=config.residual_checkpoint_name
    )
    if logsumexp is not None:
      logsumexp = ad_checkpoint.checkpoint_name(
          logsumexp, name=config.residual_checkpoint_name
      )
  if save_residuals:
    stats = {"logsumexp": logsumexp, "max_logits": max_logits}
    stats = jax.tree.map(jax.lax.stop_gradient, stats)
    return out, stats
  return out


@partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "save_residuals",
        "mask_value",
        "is_mqa",
        "config",
        "mask_function",
        "bias_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
    ),
)
def _splash_attention_custom(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    bias: jax.Array | None,
    dropout_mask: jax.Array | None,
    dropout_seed: jax.Array | None,
    segment_ids: base.SegmentIds | None,
    sinks: jax.Array | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    bias_function=None,
    fwd_mask_sparsity: float = 1.0,
    dkv_mask_sparsity: float = 1.0,
    max_logit_value: jax.Array | None = None,
) -> base.SplashCustomReturnType:
  # The forward function does not use the dq and dkv MaskInfos, it just forwards
  # them to the backward function as residuals. This is a way to communicate
  # arbitrary Arrays to the backward function. Since the three MaskInfos are
  # constants there is no overhead in passing them to the backward function as
  # residuals. When sharding computation MaskInfos are partitioned so both the
  # forward and the backward kernels need to work on the relevant slice. If we
  # recomputed the backward MaskInfos in the backward function from the numpy
  # mask then we would not work with the MaskInfo slice relevant to the current
  # device.
  del dkv_mask_info

  ret = _splash_attention_forward(  # pytype: disable=wrong-arg-types
      fwd_mask_info,
      q,
      k,
      v,
      bias,
      dropout_mask,
      dropout_seed,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      save_residuals=save_residuals,
      mask_function=mask_function,
      bias_function=bias_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=max_logit_value,
  )
  if save_residuals:
    out, stats = ret
    if config.use_base2_exp:  # for user, output values in natural base
      stats["logsumexp"] = stats["logsumexp"] / LOG2E
      if stats["max_logits"] is not None:
        stats["max_logits"] = stats["max_logits"] / LOG2E
    return out, stats
  else:
    return ret


def _splash_attention_fwd(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    bias: jax.Array | None,
    dropout_mask: jax.Array | None,
    dropout_seed: jax.Array | None,
    segment_ids: base.SegmentIds | None,
    sinks: jax.Array | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    bias_function=None,
    fwd_mask_sparsity: float = 1.0,
    dkv_mask_sparsity: float = 1.0,
    max_logit_value: jax.Array | None = None,
) -> tuple[tuple[jax.Array], base.SplashResidualsType]:

  # TODO: add some higher order AD check that isn't save_residuals based.
  # if save_residuals:
  #   raise NotImplementedError("Higher-order AD not supported.")

  out, stats = _splash_attention_forward(  # pytype: disable=wrong-arg-types
      fwd_mask_info,
      q,
      k,
      v,
      bias,
      dropout_mask,
      dropout_seed,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      save_residuals=True,
      mask_function=mask_function,
      bias_function=bias_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=max_logit_value,
  )
  logsumexp = stats["logsumexp"]  # save in the config base for the bwd pass
  if config.use_base2_exp:  # for user, output values in natural base
    stats["logsumexp"] = stats["logsumexp"] / LOG2E
    if stats["max_logits"] is not None:
      stats["max_logits"] = stats["max_logits"] / LOG2E
  residuals = (
      q, k, v, bias, dropout_mask, dropout_seed, segment_ids, sinks, out,
      logsumexp, dkv_mask_info,
  )
  if save_residuals:
    return (out, stats), residuals  # pyrefly: ignore[bad-return]
  else:
    return out, residuals  # pyrefly: ignore[bad-return]


# V12-L1 WARNING (v012 verifier finding): this UNFUSED dq kernel is DEAD CODE
# (the bwd uses only the fused dkv kernel; config rejects use_fused_bwd_kernel
# =False). It has NO bias parameters, and its soft-cap chain rule computes
# tanh from a bias-LESS qk_uncapped -- if a future refactor adds bias here and
# rewires the unfused path, the (1-d^2) factor will be computed from the WRONG
# argument and silently corrupt dq under soft-cap + bias. Any such refactor
# MUST add bias to qk_uncapped BEFORE the cap (mirror the fused dkv body) and
# re-run the v012 saturation-probe gate (v5_gate arm I).
def _flash_attention_dq_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    # Outputs
    dq_scratch_ref,
    dq_ref,
    *,
    mask_value: float,
    kv_steps: int,
    bq: int,
    bkv: int,
    mask_function: MaskFunctionType | None,
    config: SplashConfig,
):
  del mask_next_ref, active_rows_ref
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  grid_idx = pl.program_id(1)
  if block_mask_ref is not None:
    kv_index = active_cols_ref[grid_idx].astype(jnp.int32)
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_run = block_mask_ref[grid_idx].astype(jnp.int32) != 0  # sbd fork
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
  else:
    kv_index = grid_idx % kv_steps
    should_not_mask = False
    should_run = True
    should_initialize = kv_index == 0
    should_write = kv_index == kv_steps - 1

  @pl.when(should_initialize)
  def init():
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

  def body(has_partial_mask: bool = False):
    q = q_ref[...] if config.q_layout == HEAD_DIM_MINOR else q_ref[...].mT
    if config.use_base2_exp:
      q *= LOG2E
    # We keep k and v possibly transposed, since they are RHS of dots.
    k = k_ref[...]
    v = v_ref[...]
    logsumexp = jnp.expand_dims(logsumexp_ref[0], -1)
    do = do_ref[...]
    di = jnp.expand_dims(di_ref[0], -1)

    qk_dims = (
        NT_DIM_NUMBERS if config.k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    qk_uncapped = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)

    qk = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=pl.ds(0, bkv),  # pyrefly: ignore[bad-argument-type]
        k_offset=kv_index * bkv,
        bq=bq,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - logsumexp)  # pyrefly: ignore[unsupported-operation]
    dp_dims = (
        NT_DIM_NUMBERS if config.v_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    dp = lax.dot_general(
        do.astype(v.dtype),
        v,
        dp_dims,
        preferred_element_type=jnp.float32,
    )
    ds = (dp - di) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      ds = ds * (1 - d * d)

    dq_dims = (
        NN_DIM_NUMBERS if config.k_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    )
    dq_scratch_ref[...] += lax.dot_general(
        ds.astype(k.dtype),
        k,
        dq_dims,
        preferred_element_type=jnp.float32,
    )

  @pl.when(should_not_mask)
  def _():
    body()

  @pl.when(jnp.logical_not(should_not_mask))
  def _():
    body(has_partial_mask=True)

  @pl.when(should_write)
  def end():
    dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)


def _flash_attention_dkv_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    bias_ref,
    dropout_mask_ref,
    dropout_seed_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    # aliases
    dq_alias,
    dk_alias,
    dv_alias,
    dbias_alias,
    # Outputs
    dq_ref,
    dk_ref,
    dv_ref,
    dbias_ref,
    # Scratch
    dq_scratch_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    dbias_scratch_ref=None,
    *,
    mask_value: float,
    q_steps: int,
    bq: int,
    bkv_compute: int,
    bkv: int,
    mask_function: MaskFunctionType | None,
    bias_function=None,  # sbd fork: static f(h, q_pos, kv_pos) -> logits delta
    q_heads_per_kv_head: int,
    config: SplashConfig,
):
  del mask_next_ref, dbias_alias
  _use_prng_dropout = config.dropout_rate > 0.0 and dropout_mask_ref is None
  _need_q_block = _use_prng_dropout or bias_function is not None
  if not _need_q_block:
    del active_cols_ref  # only PRNG-dropout/bias_function need the q-block index
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  if active_rows_ref is not None:
    assert bounds_start_ref is not None
    assert bounds_end_ref is not None
    grid_idx = pl.program_id(1)
    kv_index = active_rows_ref[grid_idx].astype(jnp.int32)
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
    _pk_dbias_write = should_write  # v021 cand4c: per-q-head, un-ANDed
    # v021: per-key dbias init trigger -- first q-step of each (head,
    # kv-block) sweep, PER Q-HEAD (never group-ANDed: dbias is per q-head
    # while dk/dv are per kv-head).
    _pk_dbias_init = should_initialize
    if _need_q_block:
      # sbd fork: absolute q-block index + q-head (quantum hash / positions).
      _dropout_q_block = active_cols_ref[grid_idx].astype(jnp.int32)
      _dropout_q_head = pl.program_id(0)
  else:
    kv_index, q_head, q_index = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
    )
    grid_idx = (kv_index * q_steps) + q_index
    should_initialize = q_index == 0
    should_write = True if q_steps <= 2 else q_index == q_steps - 1
    _pk_dbias_init = should_initialize  # v021: pre-group-AND capture
    _pk_dbias_write = should_write  # v021 cand4c: pre-group-AND capture
    if _need_q_block:
      _dropout_q_block = q_index
      _dropout_q_head = q_head
    if q_heads_per_kv_head > 1:
      q_head_index_per_kv_head = lax.rem(q_head, q_heads_per_kv_head)
      should_initialize = jnp.logical_and(
          should_initialize, q_head_index_per_kv_head == 0
      )
      should_write = jnp.logical_and(
          should_write, q_head_index_per_kv_head == q_heads_per_kv_head - 1
      )

  if block_mask_ref is not None:
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_run = block_mask_ref[grid_idx].astype(jnp.int32) != 0
  else:
    should_not_mask = False
    should_run = True

  # TODO: Update docstring explaining the accumulation logic

  # Consider this situation:
  # Q_heads:   0, 1, 2, 3, 4, 5, 6, 7
  # KV_heads:  0,    1,    2,    3
  # The gradient scratch buffers should be initialized for Q_heads 0, 2, 4, 6
  # (first Q_heads to 'see' a new KV_head).
  # The gradient output buffers should be written for Q_heads 1, 3, 5, 7 (last
  # Q_heads to 'see' the current KV_head).

  @pl.when(should_initialize)
  def init():
    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
    dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

  if dbias_ref is not None and config.bias_per_key_rows > 0:
    # v021: zero the q-pinned per-key dbias accumulator at the FIRST q-step
    # of its (head, kv-block) sweep -- independent of output aliasing (which
    # the dense grid lacks and XLA may drop, R9), and per q-head (un-ANDed).
    if config.bias_per_key_dbias_scratch:
      @pl.when(_pk_dbias_init)
      def init_dbias_pk_scratch():
        dbias_scratch_ref[...] = jnp.zeros_like(dbias_scratch_ref)

      # cand4c: single write-out per (head, kv-block) sweep. UNCONDITIONAL
      # on should_run: a masked final step still owns the write (the body's
      # own should_run gating already kept masked steps out of the sum).
      @pl.when(_pk_dbias_write)
      def write_dbias_pk():
        dbias_ref[...] = dbias_scratch_ref[...].astype(dbias_ref.dtype)
    else:
      @pl.when(_pk_dbias_init)
      def init_dbias_pk():
        dbias_ref[...] = jnp.zeros_like(dbias_ref)

  def body(i, _, has_partial_mask=False):

    slice_k = pl.ds(i * bkv_compute, bkv_compute)
    q = q_ref[...]  # We keep q potentially transposed, since it's always RHS
    if config.use_base2_exp:
      scaled_q = q * LOG2E
    else:
      scaled_q = q

    def _load_kv(ref, layout):
      if layout == HEAD_DIM_MINOR:
        return ref[slice_k, :]
      return ref[:, slice_k].T

    k = _load_kv(k_ref, config.k_layout)
    v = _load_kv(v_ref, config.v_layout)
    logsumexp = logsumexp_ref[:1, :]
    do = do_ref[...]
    di = di_ref[:1, :]

    qk_dims = (
        NT_DIM_NUMBERS if config.q_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    _g = config.qk_diag_grid
    if (
        config.qk_diag_skip
        and has_partial_mask
        and bkv_compute % _g == 0
        and bq % _g == 0
    ):
      # Diagonal skip (backward dkv): qk tile is [kv, q]. On an aligned square diagonal
      # block, sub-tile (kv-band ki, q-band qj) with ki > qj is fully above the causal
      # boundary (kv > q) -> overwritten to mask_value anyway -> skip its matmul; compute
      # only ki <= qj sub-tiles; assemble the full tile for the single exp/ds/dv/dk.
      sk = bkv_compute // _g
      sq = bq // _g
      k_parts = [k[i * sk:(i + 1) * sk, :] for i in range(_g)]
      q_parts = [scaled_q[j * sq:(j + 1) * sq, :] for j in range(_g)]
      _mm = lambda kk, qq: lax.dot_general(
          kk, qq, qk_dims, preferred_element_type=jnp.float32
      )
      rows = []
      for ki in range(_g):  # kv row-band
        cols = []
        for qj in range(_g):  # q col-band
          if ki > qj:  # fully masked -> skip matmul
            cols.append(jnp.full((sk, sq), mask_value, dtype=jnp.float32))
          else:
            cols.append(_mm(k_parts[ki], q_parts[qj]))
        rows.append(jnp.concatenate(cols, axis=1))
      qk_uncapped = jnp.concatenate(rows, axis=0)
    else:
      qk_uncapped = lax.dot_general(
          k, scaled_q, qk_dims, preferred_element_type=jnp.float32
      )

    if bias_ref is not None:
      # sbd fork (f9af38b bwd resurrection): bias tile is [bq, bkv] (q-major);
      # the dkv qk tile is [kv, q] -- transpose via fp32.
      if config.bias_per_key_rows > 0:
        # v013 per-key: one row -> broadcast over the q axis (columns here).
        bias_t = bias_ref[0:1, slice_k].astype(jnp.float32).T
      else:
        bias_t = bias_ref[:, slice_k].astype(jnp.float32).T
      if config.use_base2_exp:
        bias_t = bias_t * LOG2E
      qk_uncapped = qk_uncapped + bias_t

    if bias_function is not None:
      # sbd fork: positional bias recomputed in [kv, q] orientation; the
      # function is elementwise in positions, so orientation-aligned grids
      # reproduce the fwd values exactly.
      _kvq = (bkv_compute, bq)
      q_pos_t = _dropout_q_block * bq + lax.broadcasted_iota(
          jnp.int32, _kvq, 1
      )
      kv_pos_t = (
          kv_index * bkv
          + i * bkv_compute
          + lax.broadcasted_iota(jnp.int32, _kvq, 0)
      )
      fbias_t = bias_function(_dropout_q_head, q_pos_t, kv_pos_t).astype(
          jnp.float32
      )
      if config.use_base2_exp:
        fbias_t = fbias_t * LOG2E
      qk_uncapped = qk_uncapped + fbias_t

    qk = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,  # pyrefly: ignore[bad-argument-type]
        k_offset=kv_index * bkv + i * bkv_compute,
        bq=bq,
        k_in_lanes=False,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - logsumexp)

    if config.dropout_rate > 0.0 or dropout_mask_ref is not None:
      # sbd fork: regenerate the IDENTICAL forward keep-mask. p stays unmasked
      # for the ds chain rule; dv and dp use the masked/rescaled copies.
      rate = config.dropout_rate
      inv_keep = (1.0 / (1.0 - rate)) if rate > 0.0 else 1.0
      _keep_bool_t = None  # v016 cand2: direct bool keep, [kv, q]
      _keep_mul16 = None  # v016 cand3: bf16 0/1 multiplier, [kv, q]
      if dropout_mask_ref is not None:
        _kf32 = dropout_mask_ref[:, slice_k].astype(jnp.float32).T
      elif config.dropout_cheap_construction:
        # v006 probe p1 (dkv side): trivially-true keep in [kv, q]
        # orientation, no draws/transposes/concats.
        _pos_t = lax.broadcasted_iota(jnp.int32, (bkv_compute, bq), 0)
        _kf32 = (
            _pos_t >= jnp.minimum(dropout_seed_ref[0, 0], 0)
        ).astype(jnp.float32)
      elif config.dropout_probe_single_draw:
        # v006 probe p3 (dkv side): same single draw, tiled + transposed --
        # self-consistent with the fwd probe's mask function.
        qt = DROPOUT_QUANTUM
        thresh = jnp.uint32(
            min(int(config.dropout_rate * (2**32)), 2**32 - 1)
        )
        w0, w1 = _mix_dropout_seed(
            dropout_seed_ref[0, 0], dropout_seed_ref[0, 1],
            _dropout_q_head, 0, 0,
        )
        pltpu.prng_seed(w0, w1)
        bits1 = pltpu.prng_random_bits((qt, qt))
        bits_full = jnp.tile(bits1, (bq // qt, bkv_compute // qt))
        keep_q = lax.bitcast_convert_type(bits_full, jnp.uint32) >= thresh
        _kf32 = keep_q.astype(jnp.float32).T
      else:
        qt = DROPOUT_QUANTUM
        assert bq % qt == 0 and bkv_compute % qt == 0, (
            "dkv block sizes must be multiples of DROPOUT_QUANTUM"
        )
        thresh = jnp.uint32(min(int(rate * (2**32)), 2**32 - 1))
        q_quant_base = _dropout_q_block * (bq // qt)
        if config.dropout_mask_version == 6:
          # v019 v6: orientation-free hash, kv-major NATIVE -- the identical
          # keep pattern as fwd's q-major eval (same pure function of
          # absolute coordinates), with NO transpose, NO chain, NO bitcast.
          kv_abs = (
              kv_index * bkv + i * bkv_compute
          ) + lax.broadcasted_iota(jnp.int32, (bkv_compute, bq), 0)
          q_abs = _dropout_q_block * bq + lax.broadcasted_iota(
              jnp.int32, (bkv_compute, bq), 1
          )
          _keep_bool_t = _hash_keep_bool(
              dropout_seed_ref[0, 0], dropout_seed_ref[0, 1],
              _dropout_q_head, q_abs, kv_abs, rate,
          )
          _kf32 = None
        elif config.dropout_mask_version == 5:
          # v5 regen: identical single chain keyed (q-block, band).
          band = kv_index * (bkv // bkv_compute) + i
          w0, w1 = _mix_dropout_seed(
              dropout_seed_ref[0, 0], dropout_seed_ref[0, 1],
              _dropout_q_head, _dropout_q_block, band,
          )
          pltpu.prng_seed(w0, w1)
          if config.dropout_probe_half_area:
            words = pltpu.prng_random_bits((bq // 2, bkv_compute))
            by16 = pltpu.bitcast(words, jnp.int16)
            _t16v = min(int(rate * 65536), 65535) ^ 0x8000
            t16f = jnp.float32(_t16v - 65536 if _t16v >= 32768 else _t16v)
            xs16 = (by16 ^ jnp.int16(-32768)).astype(jnp.float32)
            _kf32 = (xs16 >= t16f).astype(jnp.float32).T
          else:
            bits5 = pltpu.prng_random_bits((bq, bkv_compute))
            keep_q = lax.bitcast_convert_type(bits5, jnp.uint32) >= thresh
            _kf32 = keep_q.astype(jnp.float32).T
        elif config.dropout_mask_version == 4:
          # v4 regen: identical draw + view; assemble q-major, ONE transpose.
          _t8v = min(int(rate * 256), 255) ^ 0x80
          t8f = jnp.float32(_t8v - 256 if _t8v >= 128 else _t8v)
          band = kv_index * (bkv // bkv_compute) + i
          row_tiles = []
          for aq in range(bq // qt):
            w0, w1 = _mix_dropout_seed(
                dropout_seed_ref[0, 0], dropout_seed_ref[0, 1],
                _dropout_q_head, q_quant_base + aq, band,
            )
            pltpu.prng_seed(w0, w1)
            words = pltpu.prng_random_bits((qt // 4, bkv_compute))
            by = pltpu.bitcast(words, jnp.int8)  # sublane-expand (qt, bkv_c)
            # unsigned >= via sign-flip in f32 (int8 cmp unsupported):
            if config.dropout_bwd_fused_compare_bf16:
              # v016 cand5: bf16 pre-compare values (i8-range ints, exact)
              row_tiles.append((by ^ jnp.int8(-128)).astype(jnp.bfloat16))
              continue
            xs = (by ^ jnp.int8(-128)).astype(jnp.float32)
            if config.dropout_bwd_fused_compare:
              row_tiles.append(xs)  # v016 cand2: pre-compare f32 values
            else:
              row_tiles.append(xs >= t8f)
          if config.dropout_bwd_fused_compare_bf16:
            # v016 cand5: ONE bf16 transpose + ONE bf16 compare -> keep
            # [kv, q]; half of cand2's transpose bytes, no bool round trip.
            xs16_q = jnp.concatenate(row_tiles, axis=0)  # (bq, bkv_c) bf16
            _keep_bool_t = xs16_q.T >= jnp.bfloat16(t8f)
            _kf32 = None
          elif config.dropout_bwd_fused_compare:
            # v016 cand2: ONE f32 transpose then ONE compare -> keep [kv, q];
            # deletes the bool->f32 astype and the != 0.0 recheck. Identical
            # keep pattern (same draws, same threshold).
            xs_q = jnp.concatenate(row_tiles, axis=0)  # (bq, bkv_c) f32
            _keep_bool_t = xs_q.T >= t8f
            _kf32 = None
          elif config.dropout_bwd_bf16_mask:
            # v016 cand3: bf16 0/1 multiplier mask; the transpose moves half
            # the bytes and both applies become multiplies. Bit-exact vs the
            # select form (0*x == 0, 1*x == x exactly).
            keep_q = jnp.concatenate(row_tiles, axis=0)  # (bq, bkv_c) bool
            _keep_mul16 = keep_q.astype(jnp.bfloat16).T  # (bkv_c, bq) bf16
            _kf32 = None
          else:
            keep_q = jnp.concatenate(row_tiles, axis=0)  # (bq, bkv_c) q-major
            _kf32 = keep_q.astype(jnp.float32).T
        elif config.dropout_mask_version == 3:
          # v3 regen: same lane-major layout; band width equality enforced.
          t8 = jnp.uint32(min(int(rate * 256), 255))
          band = kv_index * (bkv // bkv_compute) + i
          col_tiles = []
          for aq in range(bq // qt):
            w0, w1 = _mix_dropout_seed(
                dropout_seed_ref[0, 0],
                dropout_seed_ref[0, 1],
                _dropout_q_head,
                q_quant_base + aq,
                band,
            )
            pltpu.prng_seed(w0, w1)
            words = lax.bitcast_convert_type(
                pltpu.prng_random_bits((qt, bkv_compute // 4)), jnp.uint32
            )
            lanes = [
                ((words >> (8 * l)) & jnp.uint32(0xFF)) >= t8
                for l in range(4)
            ]
            keep_f = jnp.concatenate(
                [ln.astype(jnp.float32) for ln in lanes], axis=1
            )
            col_tiles.append(keep_f.T)
          _kf32 = jnp.concatenate(col_tiles, axis=1)
        elif config.dropout_mask_version == 2:
          # v2 regen: band width must equal the fwd's bkv_compute -- enforced
          # in the wrapper (block_kv_compute == block_kv_dkv_compute).
          band = kv_index * (bkv // bkv_compute) + i
          col_tiles = []
          for aq in range(bq // qt):
            w0, w1 = _mix_dropout_seed(
                dropout_seed_ref[0, 0],
                dropout_seed_ref[0, 1],
                _dropout_q_head,
                q_quant_base + aq,
                band,
            )
            pltpu.prng_seed(w0, w1)
            bits = pltpu.prng_random_bits((qt, bkv_compute))  # q-major
            keep_f = (
                lax.bitcast_convert_type(bits, jnp.uint32) >= thresh
            ).astype(jnp.float32)
            col_tiles.append(keep_f.T)  # -> [kv, qt-band]
          _kf32 = jnp.concatenate(col_tiles, axis=1)
        elif config.dropout_int_construction:
          # v006 cand3: integer-domain assembly (identical draws + threshold
          # values => bit-identical keep pattern), single-shot tail ops.
          kv_quant_base = kv_index * (bkv // qt) + i * (bkv_compute // qt)
          row_tiles = []
          for aq in range(bq // qt):
            col_tiles = []
            for ak in range(bkv_compute // qt):
              w0, w1 = _mix_dropout_seed(
                  dropout_seed_ref[0, 0],
                  dropout_seed_ref[0, 1],
                  _dropout_q_head,
                  q_quant_base + aq,
                  kv_quant_base + ak,
              )
              pltpu.prng_seed(w0, w1)
              col_tiles.append(pltpu.prng_random_bits((qt, qt)))  # q-major
            row_tiles.append(jnp.concatenate(col_tiles, axis=1))
          bits_qmajor = jnp.concatenate(row_tiles, axis=0)  # [bq, bkv_c]
          keep_q = (
              lax.bitcast_convert_type(bits_qmajor, jnp.uint32) >= thresh
          )
          _kf32 = keep_q.astype(jnp.float32).T  # ONE astype + ONE transpose
        else:
          kv_quant_base = kv_index * (bkv // qt) + i * (bkv_compute // qt)
          row_tiles = []
          for ak in range(bkv_compute // qt):
            col_tiles = []
            for aq in range(bq // qt):
              w0, w1 = _mix_dropout_seed(
                  dropout_seed_ref[0, 0],
                  dropout_seed_ref[0, 1],
                  _dropout_q_head,
                  q_quant_base + aq,
                  kv_quant_base + ak,
              )
              pltpu.prng_seed(w0, w1)
              bits = pltpu.prng_random_bits((qt, qt))  # q-major, as in fwd
              keep_f = (
                  lax.bitcast_convert_type(bits, jnp.uint32) >= thresh
              ).astype(jnp.float32)
              col_tiles.append(keep_f.T)  # -> [kv, q] orientation
            row_tiles.append(jnp.concatenate(col_tiles, axis=1))
          _kf32 = jnp.concatenate(row_tiles, axis=0)
      # sbd fork v005: apply-form knobs (keep pattern identical in all forms).
      if _keep_bool_t is not None:
        # v016 cand2: bool keep produced directly at the fused compare.
        keep_t, keep_ft = _keep_bool_t, None
        p_pv = jnp.where(keep_t, p, 0.0)
      elif _keep_mul16 is not None:
        # v016 cand3: bf16 multiplier; p cast to the dv-dot input dtype first
        # (that cast happened at the dot anyway), then a 0/1 mul -- bit-exact
        # vs select-then-cast since rounding commutes with the 0/1 select.
        keep_t, keep_ft = None, _keep_mul16
        p_pv = p.astype(do.dtype) * _keep_mul16
      elif config.dropout_mask_as_multiplier:
        keep_t, keep_ft = None, _kf32
        p_pv = p * keep_ft
      else:
        keep_t, keep_ft = _kf32 != 0.0, None
        p_pv = jnp.where(keep_t, p, 0.0)
      if config.dropout_fold_rescale:
        # Prescale do once: dv = p_m^T (r*do) and dp = v (r*do)^T carry the
        # rescale with no per-element pass over the score tile.
        if rate > 0.0:
          do = do * jnp.asarray(inv_keep, do.dtype)
      else:
        p_pv = p_pv * inv_keep
    else:
      keep_t = None
      keep_ft = None
      inv_keep = 1.0
      p_pv = p

    dv = lax.dot(p_pv.astype(do.dtype), do, preferred_element_type=jnp.float32)
    dv = dv.astype(dv_scratch_ref.dtype) + dv_scratch_ref[slice_k, :]
    dv_scratch_ref[slice_k, :] = dv

    dp = lax.dot_general(
        v,
        do,
        NT_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    _has_drop = keep_t is not None or keep_ft is not None
    if _has_drop and not config.dropout_single_bwd_apply:
      if keep_ft is not None:
        dp = dp * keep_ft
      else:
        dp = jnp.where(keep_t, dp, 0.0)
      if not config.dropout_fold_rescale:
        dp = dp * inv_keep
    if _has_drop and config.dropout_single_bwd_apply:
      # ds = (masked(dp_s) - di) * p == dp_s * p_m - di * p (masked(x)*y ==
      # x*masked(y) for the finite tiles here); reuses p_pv, dp never masked.
      ds = dp * p_pv - di * p
    else:
      ds = (dp - di) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      ds = ds * (1 - d * d)

    if dbias_ref is not None:
      if config.bias_per_key_rows > 0:
        # v021: rank-reduced per-key dbias -- sum over q (lane axis of the
        # [kv, q] ds tile) into row 0 of the q-pinned (rows, bkv) f32 tile,
        # accumulated across bands and across the grid's q-block revisits
        # (consecutive for fixed (h, kv-block) under the active-pair walk;
        # aliased zeros initialize). ds already carries the soft-cap chain
        # rule; d(logits)/d(bias) = 1.
        if config.bias_per_key_dbias_mxu_reduce:
          # cand4: ones-matmul on the MXU -- (1, bq) @ contract bq -> (1, bkv)
          _ones = jnp.ones((1, ds.shape[1]), jnp.float32)
          _red2 = lax.dot_general(
              _ones, ds, (((1,), (1,)), ((), ())),
              preferred_element_type=jnp.float32,
          )  # (1, bkv_compute)
          if config.bias_per_key_dbias_scratch:
            # cand4c: accumulate in VMEM scratch (the dk/dv pattern).
            _cur = dbias_scratch_ref[0:1, slice_k]
            dbias_scratch_ref[0:1, slice_k] = _cur + _red2
          else:
            _cur = dbias_ref[0:1, slice_k]
            dbias_ref[0:1, slice_k] = _cur + _red2.astype(dbias_ref.dtype)
        else:
          _red = jnp.sum(ds, axis=1)  # (bkv_compute,)
          _cur = dbias_ref[0:1, slice_k]
          dbias_ref[0:1, slice_k] = _cur + _red[None, :].astype(
              dbias_ref.dtype
          )
      else:
        # ds is [kv, q]; dbias tile is [bq, bkv] (q-major).
        # d(logits)/d(bias)=1, and ds already includes the soft-cap chain
        # rule (bias enters pre-cap).
        dbias_ref[:, slice_k] = ds.T.astype(dbias_ref.dtype)

    dk_dims = (
        NN_DIM_NUMBERS if config.q_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    )
    dk = lax.dot_general(
        ds.astype(do.dtype), q, dk_dims, preferred_element_type=jnp.float32
    )
    dk = dk.astype(dk_scratch_ref.dtype) + dk_scratch_ref[slice_k, :]
    dk_scratch_ref[slice_k, :] = dk
    if dq_scratch_ref is not None or dq_ref is not None:
      dq = lax.dot_general(
          ds.T.astype(k.dtype),
          k,
          NN_DIM_NUMBERS,
          preferred_element_type=jnp.float32,
      )
      if dq_scratch_ref is not None:
        # Compute block size != memory block size
        dq_scratch_ref[...] += dq
      else:
        # Compute block size == memory block size
        if dq_alias is not None:
          dq_ref[...] = dq_alias[...] + dq.astype(dq_ref.dtype)
        else:
          dq_ref[...] = dq.astype(dq_ref.dtype)

  if dq_scratch_ref is not None:
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)
  elif dq_alias is not None:
    dq_ref[...] = dq_alias[...]
  else:
    dq_ref[...] = jnp.zeros_like(dq_ref)

  if dbias_ref is not None and config.bias_per_key_rows == 0:
    # Skipped blocks must leave a zero dbias tile (aliasing covers unvisited
    # tiles on the dynamic grid; this covers visited-but-masked-out steps).
    # v021: per-key dbias MUST skip this guard -- its q-pinned tile
    # ACCUMULATES across grid revisits; zeroing on a masked step would
    # destroy prior q-blocks' contributions (a masked step simply adds
    # nothing; aliased zeros cover initialization).
    @pl.when(_not(should_run))
    def init_dbias():
      dbias_ref[...] = jnp.zeros_like(dbias_ref)

  num_iters = (
      k_ref.shape[0 if config.k_layout is HEAD_DIM_MINOR else 1] // bkv_compute
  )

  # v016 cand4: with PRNG dropout the body is ~1.8x featoff's code; full
  # unroll multiplies its live ranges -> spills. featoff path keeps unroll.
  _unroll_body = not (config.dropout_dkv_no_unroll and _use_prng_dropout)

  @pl.when(jnp.logical_and(should_not_mask, should_run))
  def _():
    lax.fori_loop(0, num_iters, body, None, unroll=_unroll_body)

  @pl.when(jnp.logical_and(_not(should_not_mask), should_run))
  def _():
    lax.fori_loop(
        0, num_iters, partial(body, has_partial_mask=True), None,
        unroll=_unroll_body,
    )

  if dq_scratch_ref is not None:
    if dq_alias is not None:
      dq_ref[...] = dq_alias[...] + dq_scratch_ref[...].astype(dq_ref.dtype)
    else:
      dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)

  if dk_alias is None:
    assert dv_alias is None

    @pl.when(should_write)
    def _():
      dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)

  else:
    q_head = pl.program_id(0)
    first_q_head_in_kv_group = lax.rem(q_head, q_heads_per_kv_head) == 0

    @pl.when(jnp.logical_and(should_write, first_q_head_in_kv_group))
    def _():
      dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)

    @pl.when(jnp.logical_and(should_write, _not(first_q_head_in_kv_group)))
    def _():
      dk_ref[...] = dk_alias[...] + dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_alias[...] + dv_scratch_ref[...].astype(dv_ref.dtype)


def _splash_attention_bwd_dkv(
    q,
    k,
    v,
    bias,
    dropout_mask,
    dropout_seed,
    segment_ids,
    logsumexp,
    do,
    di,
    *,
    bq: int,
    bkv: int,
    bkv_compute: int,
    is_mqa: bool,
    mask_info: MaskInfo,
    mask_value: float,
    mask_function: MaskFunctionType | None,
    bias_function=None,
    config: SplashConfig,
    dkv_mask_sparsity: float,
):
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  kv_seq_len, head_dim_v = v.shape[-2:]
  num_kv_heads = 1 if is_mqa else k.shape[0]
  dynamic_grid = mask_info.active_rows is not None

  bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)  # pyrefly: ignore[bad-argument-type]
  if bq > q_seq_len:
    raise ValueError(f"{bq=} should not be greater than {q_seq_len=}")
  if bkv > kv_seq_len:
    raise ValueError(f"{bkv=} should not be greater than {kv_seq_len=}")
  if bkv_compute > bkv:
    raise ValueError(f"{bkv_compute=} should not be greater than {bkv=}")
  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} should be a multiple of {bkv_compute=}")

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(
        f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same "
        "leading dimensions."
    )

  kv_steps = kv_seq_len // bkv
  q_steps = q_seq_len // bq
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if dynamic_grid:

    def unravel(f):
      def index_map(h, grid_idx, rows_ref, cols_ref, *_):
        j = to_i32(rows_ref[grid_idx])
        i = to_i32(cols_ref[grid_idx])
        return f(h, i, j)

      return index_map

    grid_size = mask_info.num_active_blocks[0]  # pyrefly: ignore[unsupported-operation]
    grid = (num_q_heads, grid_size)

    def mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_):
      del h, rows_ref, cols_ref  # Unused.
      next_m = to_i32(mask_next_ref[grid_idx])  # pyrefly: ignore[unsupported-operation]
      return next_m, 0, 0

  else:
    unravel = lambda f: lambda j, h, i, *_: f(h, i, j)
    grid = (kv_steps, num_q_heads, q_steps)

    def mask_index_map(j, h, i, rows_ref, cols_ref, mask_next_ref=None, *_):
      del h, rows_ref, cols_ref  # Unused.
      grid_idx = j * q_steps + i
      next_m = to_i32(mask_next_ref[grid_idx])  # pyrefly: ignore[unsupported-operation]
      return next_m, 0, 0

  q_index_map = unravel(
      lambda h, i, j: from_head_minor((h, i, 0), config.q_layout)
  )
  o_index_map = unravel(lambda h, i, j: (h, i, 0))

  def create_kv_index_map(layout):
    def index_map(h, i, j, *_):
      del i  # Unused.
      prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
      return from_head_minor((*prefix, j, 0), layout)

    return index_map

  k_index_map = unravel(create_kv_index_map(config.k_layout))
  v_index_map = unravel(create_kv_index_map(config.v_layout))

  q_spec = pl.BlockSpec(
      from_head_minor((None, bq, head_dim_qk), config.q_layout), q_index_map
  )

  o_spec = pl.BlockSpec((None, bq, head_dim_v), o_index_map)
  k_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
          config.k_layout,
      ),
      k_index_map,
  )

  v_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v),
          config.v_layout,
      ),
      v_index_map,
  )

  # sbd fork: bwd bias supports FULL-RANK [num_q_heads, Q, K] only -- a
  # head-broadcast bias would need cross-head dbias accumulation across the
  # head-iterating grid (named lead, not implemented).
  if bias is not None:
    # v013: head-broadcast is LEGAL for per-key rank -- dbias is disabled
    # there, and the bwd only READS the bias for the qk regen (the full-rank
    # requirement exists solely for the dbias write's head-indexed aliasing).
    if bias.shape[0] != num_q_heads and config.bias_per_key_rows == 0:
      raise NotImplementedError(
          "backward bias requires full-rank [num_q_heads, q, kv] bias; "
          f"got head dim {bias.shape[0]} (broadcast dbias is a v003+ lead)"
      )
    if config.bias_per_key_rows > 0:
      _pk_hb = bias.shape[0] == 1
      bias_index_map = unravel(
          lambda h, i, j: (0 if _pk_hb else h, 0, j)
      )
      bias_spec = pl.BlockSpec(
          (None, config.bias_per_key_rows, bkv), bias_index_map
      )
    else:
      bias_index_map = unravel(lambda h, i, j: (h, i, j))
      bias_spec = pl.BlockSpec((None, bq, bkv), bias_index_map)
  else:
    bias_spec = None

  # sbd fork: dropout inputs (explicit mask tile, or SMEM seed for PRNG regen).
  if dropout_mask is not None:
    _dm_bcast = dropout_mask.shape[0] == 1
    dropout_mask_index_map = unravel(
        lambda h, i, j: (0 if _dm_bcast else h, i, j)
    )
    dropout_mask_spec = pl.BlockSpec((None, bq, bkv), dropout_mask_index_map)
  else:
    dropout_mask_spec = None
  use_dropout_seed = config.dropout_rate > 0.0 and dropout_mask is None
  if use_dropout_seed:
    if dropout_seed is None:
      raise ValueError("dropout_rate > 0 requires dropout_seed (2,) int32")
    dropout_seed_spec = pl.BlockSpec(
        (NUM_SUBLANES, 2), lambda *_: (0, 0), memory_space=pltpu.SMEM
    )
    dropout_seed = jnp.broadcast_to(
        dropout_seed.astype(jnp.int32)[None, :], (NUM_SUBLANES, 2)
    )
  else:
    dropout_seed_spec = None
    dropout_seed = None

  # On the dynamic grid, grouped-KV / MQA accumulate dk/dv across a kv
  # group's q heads through an aliased HBM buffer (`dk_alias + scratch`),
  # each q head cycling through its kv rows. Pallas TPU's 2-deep DMA pipeline
  # needs at least 3 distinct blocks in that cycle: with kv_steps <= 2 the
  # next q head prefetches `dk_alias` before the previous head's store of the
  # same block has landed (or, at one block, never re-fetches it at all).
  # There, write per-q-head outputs and reduce them outside the kernel
  # instead (tokamax 0.0.16's fix, cl/995398931).
  reduce_dkv_over_q_heads = (
      dynamic_grid and q_heads_per_kv_head != 1 and kv_steps <= 2
  )
  alias_dkv = (
      dynamic_grid and q_heads_per_kv_head != 1 and not reduce_dkv_over_q_heads
  )

  def create_dkv_index_map(h, i, j, *_):
    del i  # Unused.
    if reduce_dkv_over_q_heads:
      prefix = (h,)
    elif is_mqa:
      prefix = ()
    else:
      prefix = (_div(h, q_heads_per_kv_head),)
    return (*prefix, j, 0)

  dkv_index_map = unravel(create_dkv_index_map)
  per_q_head_or_kv_head = not is_mqa or reduce_dkv_over_q_heads

  dk_spec = pl.BlockSpec(
      (None, bkv, head_dim_qk) if per_q_head_or_kv_head else (bkv, head_dim_qk),
      dkv_index_map,
  )

  dv_spec = pl.BlockSpec(
      (None, bkv, head_dim_v) if per_q_head_or_kv_head else (bkv, head_dim_v),
      dkv_index_map,
  )
  mask_spec = pl.BlockSpec((None, bkv, bq), mask_index_map)

  q_segment_ids_index_map = unravel(lambda h, i, j: (0, i))
  if segment_ids is not None:
    kv_segment_ids_index_map = unravel(lambda h, i, j: (j, 0))

    q_segment_spec = pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map)
    kv_segment_spec = pl.BlockSpec((bkv, NUM_LANES), kv_segment_ids_index_map)
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (NUM_SUBLANES, q_seq_len), (1,)
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (kv_seq_len, NUM_LANES), (0,)
    )
  else:
    q_segment_spec = kv_segment_spec = None
    q_segment_ids = kv_segment_ids = None

  do_spec = o_spec

  logsumexp_index_map = unravel(lambda h, i, j: (h, 0, i))

  assert logsumexp.shape == di.shape == (num_q_heads, q_seq_len)
  # TODO: Remove the sublane expansion once Mosaic has all retilings
  logsumexp_shape = (num_q_heads, NUM_SUBLANES, q_seq_len)
  logsumexp = jnp.broadcast_to(jnp.expand_dims(logsumexp, -2), logsumexp_shape)
  logsumexp_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert logsumexp.ndim == len(logsumexp_spec.block_shape)  # pyrefly: ignore[bad-argument-type]

  # TODO: Remove the sublane expansion once Mosaic has all retilings
  di = jnp.broadcast_to(jnp.expand_dims(di, -2), logsumexp_shape)
  di_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert di.ndim == len(di_spec.block_shape)  # pyrefly: ignore[bad-argument-type]

  in_specs = [
      q_spec,
      k_spec,
      v_spec,
      bias_spec,
      dropout_mask_spec,
      dropout_seed_spec,
      q_segment_spec,
      kv_segment_spec,
      logsumexp_spec,
      do_spec,
      di_spec,
  ]
  if mask_info.partial_mask_blocks is not None:
    in_specs.append(mask_spec)
  else:
    in_specs.append(None)

  if mask_info.q_sequence is not None:
    in_specs.append(pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map))
    q_sequence = jax.lax.broadcast_in_dim(
        mask_info.q_sequence, (NUM_SUBLANES, q_seq_len), (1,)
    )
  else:
    q_sequence = None
    in_specs.append(None)

  dq_reduction_steps = config.dq_reduction_steps
  if not dynamic_grid and kv_steps <= 3 and dq_reduction_steps == 3:
    dq_reduction_steps = None

  dq = dq_alias_spec = None
  if dq_reduction_steps == 3:
    dq_index_map = unravel(lambda h, i, j: (j % 3, h, i, 0))
    dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
    dq_alias_spec = dq_spec
    dq_shape = jax.ShapeDtypeStruct((3, *q.shape), q.dtype)
    dq = jnp.zeros_like(dq_shape)
  else:
    dq_index_map = unravel(lambda h, i, j: (j, h, i, 0))
    dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
    # Only accumulate in fp32 if there's a small number of reduction steps.
    q_dtype = q.dtype if kv_steps <= 4 else jnp.float32
    dq_shape = jax.ShapeDtypeStruct((kv_steps, *q.shape), q_dtype)

  in_specs += [dq_alias_spec]

  if bkv == bkv_compute:
    dq_scratch = None
  else:
    dq_scratch = pltpu.VMEM((bq, head_dim_qk), jnp.float32)

  if alias_dkv:
    # in/out aliasing to accumulate within kv groups.
    in_specs += [dk_spec, dv_spec]
    dk = lax.empty(k.shape, dtype=jnp.float32)
    dv = lax.empty(v.shape, dtype=jnp.float32)
    # Keep gradients in fp32 when accumulating over head groups.
    dk_type = dv_type = jnp.float32
  else:
    in_specs += [None, None]
    dk, dv = None, None
    dk_type = jnp.float32 if reduce_dkv_over_q_heads else k.dtype
    dv_type = jnp.float32 if reduce_dkv_over_q_heads else v.dtype

  # sbd fork: dbias output (aliased zero input on the dynamic grid so
  # never-visited tiles read back as zero).
  if bias is not None:
    # v013: per-key bias has NO dbias by default (its (bq,bkv) tile writes
    # would corrupt a q-pinned rows-block); v021 adds the RANK-REDUCED dbias
    # behind bias_per_key_dbias: the q-pinned per-key spec itself becomes the
    # output spec, accumulated in f32 across grid revisits (aliased zeros).
    if config.bias_per_key_rows > 0 and config.bias_per_key_dbias:
      if bias.shape[0] != num_q_heads:
        raise NotImplementedError(
            "bias_per_key_dbias requires head-FULL per-key bias"
            " [num_q_heads, rows, kv]; head-broadcast dbias needs"
            " cross-head accumulation across non-consecutive grid"
            " revisits (named lead)"
        )
      dbias_spec = bias_spec  # the q-pinned (None, rows, bkv) per-key spec
      dbias_in = (
          jnp.zeros(bias.shape, jnp.float32) if dynamic_grid else None
      )
      in_specs += [dbias_spec if dbias_in is not None else None]
      dbias_shape = jax.ShapeDtypeStruct(bias.shape, jnp.float32)
    elif config.bias_per_key_rows > 0:
      dbias_spec = None
      dbias_in = None
      dbias_shape = None
      in_specs += [None]
    else:
      dbias_spec = bias_spec
      dbias_in = jnp.zeros_like(bias) if dynamic_grid else None
      in_specs += [dbias_spec if dbias_in is not None else None]
      dbias_shape = jax.ShapeDtypeStruct(bias.shape, bias.dtype)
  else:
    dbias_spec = None
    dbias_in = None
    in_specs += [None]
    dbias_shape = None

  out_shapes = [
      dq_shape,
      jax.ShapeDtypeStruct(
          (num_q_heads, kv_seq_len, head_dim_qk)
          if reduce_dkv_over_q_heads else k.shape, dk_type),
      jax.ShapeDtypeStruct(
          (num_q_heads, kv_seq_len, head_dim_v)
          if reduce_dkv_over_q_heads else v.shape, dv_type),
      dbias_shape,
  ]
  out_specs = [dq_spec, dk_spec, dv_spec, dbias_spec]

  kernel = functools.partial(
      _flash_attention_dkv_kernel,
      mask_value=mask_value,
      q_steps=q_steps,
      bq=bq,
      bkv_compute=bkv_compute,
      config=config,
      bkv=bkv,
      mask_function=mask_function,
      bias_function=bias_function,
      q_heads_per_kv_head=q_heads_per_kv_head,
  )

  kernel_name = get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=False,
      is_segmented=segment_ids is not None,
      phase="dkv",
  )
  metadata = {
      "xprof_metadata": json.dumps(
          dict(
              block_q_dkv=bq,
              block_kv_dkv=bkv,
              block_kv_dkv_compute=bkv_compute,
              q_layout=config.q_layout,
              k_layout=config.k_layout,
              v_layout=config.v_layout,
              use_experimental_scheduler=config.use_experimental_scheduler,
          ),
      )
  }
  args = [
      # scalar prefetch
      mask_info.active_rows,
      mask_info.active_cols,
      mask_info.mask_next,
      bounds_start,
      bounds_end,
      mask_info.block_mask,
      # inputs
      q if config.q_layout == QKVLayout.HEAD_DIM_MINOR else q.mT,
      k if config.k_layout == QKVLayout.HEAD_DIM_MINOR else k.mT,
      v if config.v_layout == QKVLayout.HEAD_DIM_MINOR else v.mT,
      bias,
      dropout_mask,
      dropout_seed,
      q_segment_ids,
      kv_segment_ids,
      logsumexp,
      do,
      di,
      mask_info.partial_mask_blocks,
      q_sequence,
  ]
  num_args = sum(1 for x in args if x is not None)
  # sbd fork: generalized alias-position loop (f9af38b pattern) -- trailing
  # aliased inputs get consecutive positions among the non-None operands, each
  # mapped to its output index.
  in_out_alias_vals = [dq, dk, dv, dbias_in]
  input_output_aliases = {}
  _alias_pos = num_args
  for _out_idx, _val in enumerate(in_out_alias_vals):
    if _val is not None:
      input_output_aliases[_alias_pos] = _out_idx
      _alias_pos += 1

  scratch_shapes = [
      dq_scratch,
      pltpu.VMEM((bkv, head_dim_qk), jnp.float32),
      pltpu.VMEM((bkv, head_dim_v), jnp.float32),
  ]
  if (
      bias is not None
      and config.bias_per_key_rows > 0
      and config.bias_per_key_dbias
      and config.bias_per_key_dbias_scratch
  ):
    # v021 cand4c: per-key dbias accumulator scratch (rows, bkv) f32.
    scratch_shapes.append(
        pltpu.VMEM((config.bias_per_key_rows, bkv), jnp.float32)
    )

  def _bwd_cost_estimate(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      q_segment_ids: jax.Array | None,
      kv_segment_ids: jax.Array | None,
      logsumexp: jax.Array,
      do: jax.Array,
      di: jax.Array,
      partial_mask_blocks: jax.Array | None,
      q_sequence: jax.Array | None,
      out_shapes: list[jax.ShapeDtypeStruct],
      mask_sparsity_factor: float,
  ) -> pl.CostEstimate:
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    kv_seq_len, head_dim_v = v.shape[-2:]

    total_matmul_flops_per_head = (
        2 * q_seq_len * kv_seq_len * head_dim_qk  # qk
        + 2 * q_seq_len * kv_seq_len * head_dim_v  # dv
        + 2 * q_seq_len * kv_seq_len * head_dim_v  # dp
        + 2 * q_seq_len * kv_seq_len * head_dim_qk  # dq
        + 2 * q_seq_len * kv_seq_len * head_dim_qk  # dk
    )

    estimated_flops = int(
        total_matmul_flops_per_head * num_q_heads * mask_sparsity_factor
    )

    exp_flops = num_q_heads * q_seq_len * kv_seq_len * mask_sparsity_factor
    if config.attn_logits_soft_cap is None:
      tanh_flops = 0
    else:
      tanh_flops = (
          2 * num_q_heads * q_seq_len * kv_seq_len * mask_sparsity_factor
      )
    estimated_transcendentals = int(exp_flops + tanh_flops)

    inputs_ = [
        q,
        k,
        v,
        q_segment_ids,
        kv_segment_ids,
        logsumexp,
        do,
        di,
        partial_mask_blocks,
        q_sequence,
    ]
    input_bytes = sum(map(_bytes, inputs_))
    output_bytes = sum(map(_bytes, out_shapes))

    estimated_bytes = input_bytes + output_bytes

    return pl.CostEstimate(
        flops=estimated_flops,
        transcendentals=estimated_transcendentals,
        bytes_accessed=estimated_bytes,
    )

  cost_estimate = config.bwd_cost_estimate or _bwd_cost_estimate(
      q,
      k,
      v,
      q_segment_ids,
      kv_segment_ids,
      logsumexp,
      do,
      di,
      mask_info.partial_mask_blocks,  # pyrefly: ignore[bad-argument-type]
      q_sequence,
      out_shapes,
      dkv_mask_sparsity,
  )

  with jax.named_scope(kernel_name):
    dq_unreduced, dk, dv, dbias = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        input_output_aliases=input_output_aliases,
        # We set all dimensions to arbitrary because:
        # 1) for heads, we are reducing over heads
        # 2) for kv_seq_len, the splash attention prefetch schedule assumes no
        #     megacore
        # 3) for q_seq_len, we are reducing over it to compute dkv
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",) * len(grid)
        ),
        name=kernel_name,
        cost_estimate=cost_estimate,
        interpret=config.interpret,
        metadata=metadata,
    )(*args, dq, dk, dv, dbias_in)
  dq = dq_unreduced.sum(axis=0)
  if reduce_dkv_over_q_heads:
    dk = dk.reshape(num_kv_heads, q_heads_per_kv_head, kv_seq_len, head_dim_qk).sum(axis=1)
    dv = dv.reshape(num_kv_heads, q_heads_per_kv_head, kv_seq_len, head_dim_v).sum(axis=1)
    if is_mqa:
      dk, dv = dk[0], dv[0]
  dq = dq.astype(q.dtype)
  dk = dk.astype(k.dtype)
  dv = dv.astype(v.dtype)
  if bias is not None and dbias is not None:
    dbias = dbias.astype(bias.dtype)
  else:
    # per-key rank: dbias disabled this version (v013); bias cotangent is
    # symbolically zero.
    dbias = None
  return dq, dk, dv, dbias


def _splash_attention_bwd(
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    bias_function,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    res: base.SplashResidualsType,
    grads: jax.Array | tuple[jax.Array, dict[str, jax.Array]],
) -> tuple[
    MaskInfo | None,  # fwd_mask_info
    MaskInfo | None,  # dvk_mask_info
    jax.Array,  # q
    jax.Array,  # k
    jax.Array,  # v
    jax.Array | None,  # bias
    jax.Array | None,  # dropout_mask
    jax.Array | None,  # dropout_seed
    base.SegmentIds | None,  # segment_ids
    jax.Array | None,  # sinks
    jax.Array | None,  # max_logit_estimate
]:
  # If `save_residuals` is True, `_splash_attention_fwd` returns `(out, stats)`,
  # so we unpack the gradients, otherwise it returns `out` and `grads` is just
  # `do`.
  if save_residuals:
    do, _ = grads
  else:
    do = grads
  del save_residuals, fwd_mask_sparsity
  if not config.has_backward_blocks:
    raise ValueError("Need to specify backward blocks.")
  bq_dkv, bkv_dkv_memory, bkv_dkv_compute = (
      config.block_q_dkv,
      config.block_kv_dkv,
      config.block_kv_dkv_compute,
  )
  (q, k, v, bias, dropout_mask, dropout_seed, segment_ids, sinks, o,
   logsumexp, dkv_mask_info) = res


  # di: [num_heads, q_seq_len]
  di = jnp.einsum("hsd,hsd->hs", o.astype(jnp.float32), do.astype(jnp.float32))  # pytype: disable=attribute-error
  dq, dk, dv, dbias = _splash_attention_bwd_dkv(
      q,
      k,
      v,
      bias,
      dropout_mask,
      dropout_seed,
      segment_ids,
      logsumexp,
      do,
      di,
      bq=bq_dkv,  # pyrefly: ignore[bad-argument-type]
      bkv=bkv_dkv_memory,  # pyrefly: ignore[bad-argument-type]
      bkv_compute=bkv_dkv_compute,  # pyrefly: ignore[bad-argument-type]
      is_mqa=is_mqa,
      mask_info=dkv_mask_info,  # pyrefly: ignore[bad-argument-type]
      mask_value=mask_value,
      mask_function=mask_function,
      bias_function=bias_function,
      config=config,
      dkv_mask_sparsity=dkv_mask_sparsity,
  )
  dsinks = None
  if sinks is not None:
    logsumexp_ = (logsumexp / LOG2E) if config.use_base2_exp else logsumexp
    sinks_exp = -jnp.exp(
        sinks[..., None, None].astype(jnp.float32)
        - logsumexp_[..., None].astype(jnp.float32)
    )
    dsinks = jnp.sum(sinks_exp.astype(o.dtype) * o * do, axis=(-1, -2))  # pyrefly: ignore[bad-argument-type]
  # Match the signature of the fwd function.
  assert dq is not None
  return (
      None,  # fwd_mask_info
      None,  # dvk_mak_info
      dq,  # q
      dk,  # k
      dv,  # v
      dbias,  # bias (v002: fused-dkv dbias; None when bias absent)
      None,  # dropout_mask
      None,  # dropout_seed
      None,  # segment_ids
      dsinks,  # sinks
      None,  # max_logit_estimate
  )


_splash_attention_custom.defvjp(_splash_attention_fwd, _splash_attention_bwd)


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "config",
        "save_residuals",
        "mask_value",
        "mask_function",
        "bias_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
    ],
)
def _splash_attention(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    bias: jax.Array | None = None,
    dropout_mask: jax.Array | None = None,
    dropout_seed: jax.Array | None = None,
    segment_ids: base.SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    config: SplashConfig | None,
    save_residuals: bool,
    mask_value: float,
    max_logit_value: jax.Array | None = None,
    mask_function: MaskFunctionType | None,
    bias_function=None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
) -> base.SplashCustomReturnType:
  return _splash_attention_custom(
      fwd_mask_info,
      dkv_mask_info,
      q,
      k,
      v,
      bias,
      dropout_mask,
      dropout_seed,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      config=config,
      max_logit_value=max_logit_value,
      mask_function=mask_function,
      bias_function=bias_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
  )


@jax.tree_util.register_pytree_node_class
class SplashAttentionKernel:

  def __init__(
      self,
      fwd_mask_info: MaskInfo,
      dkv_mask_info: MaskInfo | None,
      **kwargs,
  ):
    self.kwargs = kwargs
    self.fwd_mask_info = fwd_mask_info
    self.dkv_mask_info = dkv_mask_info

  def __call__(self, *args, **kwargs) -> base.SplashCustomReturnType:
    return _splash_attention(
        self.fwd_mask_info,
        self.dkv_mask_info,
        *args,
        **dict(self.kwargs, **kwargs),
    )

  def manual_sharding_spec(self, sharding: jax.sharding.NamedSharding):
    """Returns a value that can be used as a shard_map partition spec for the kernel."""
    if self.fwd_mask_info.block_mask is not None:
      block_mask_shape = self.fwd_mask_info.block_mask.shape
      try:
        sharding.shard_shape(block_mask_shape)
      except ValueError as exc:
        raise ValueError(
            "The sharding must divide the mask blocks evenly between devices"
        ) from exc

    if len(sharding.spec) != 1:
      raise ValueError("Only q sequence sharding is supported.")

    _resolve_spec = lambda x: sharding.spec if x is not None else None
    mask_info_specs = MaskInfo(  # pytype: disable=wrong-arg-types
        mask_next=_resolve_spec(self.fwd_mask_info.mask_next),  # pyrefly: ignore[bad-argument-type]
        active_rows=_resolve_spec(self.fwd_mask_info.active_rows),  # pyrefly: ignore[bad-argument-type]
        active_cols=_resolve_spec(self.fwd_mask_info.active_cols),  # pyrefly: ignore[bad-argument-type]
        num_active_blocks=_resolve_spec(self.fwd_mask_info.num_active_blocks),  # pyrefly: ignore[bad-argument-type]
        block_mask=_resolve_spec(self.fwd_mask_info.block_mask),  # pyrefly: ignore[bad-argument-type]
        partial_mask_blocks=jax.sharding.PartitionSpec()  # replicated  # pyrefly: ignore[bad-argument-type]
        if self.fwd_mask_info.partial_mask_blocks is not None
        else None,
        q_sequence=_resolve_spec(self.fwd_mask_info.q_sequence),  # pyrefly: ignore[bad-argument-type]
    )
    return SplashAttentionKernel(
        mask_info_specs,
        mask_info_specs if self.dkv_mask_info is not None else None,
        **self.kwargs,
    )

  def tree_flatten(self):
    return ((self.fwd_mask_info, self.dkv_mask_info), self.kwargs)

  @classmethod
  def tree_unflatten(cls, kwargs, values):
    fwd_mask_info, dkv_mask_info = values
    # NamedTuples are not preserved during pytree serialization.
    dkv_mask_info = (
        MaskInfo(*dkv_mask_info) if dkv_mask_info is not None else None
    )
    return SplashAttentionKernel(
        MaskInfo(*fwd_mask_info), dkv_mask_info, **kwargs
    )


def _make_splash_attention(
    mask: np.ndarray | mask_lib.Mask,
    *,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    q_seq_shards: int,
    bias_function=None,  # sbd fork: static f(h, q_pos, kv_pos) -> logits delta
):
  if len(mask.shape) != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")

  if isinstance(mask, np.ndarray):
    mask = mask_lib.NumpyMask(mask)

  if config is None:
    config = SplashConfig.get_default()

  if config.qk_diag_skip and not isinstance(mask, mask_lib.CausalMask):
    # The skip assumes kv > q is ALWAYS masked — a pure-causal property. Any mask
    # that admits a valid kv > q entry (bidirectional, local/sliding window, custom)
    # would be silently corrupted, so fail loud. (Square-block preconditions are
    # enforced in SplashConfig.__post_init__.)
    raise ValueError(
        "qk_diag_skip=True requires a pure CausalMask (the skip fills mask_value "
        "for all kv > q sub-tiles, assuming the mask masks exactly those); got "
        f"{type(mask).__name__}. Disable qk_diag_skip for non-causal masks."
    )

  process_fn = partial(
      mask_info_lib.process_mask,
      downcast_smem_data=downcast_smem_data,
      partial_mask_blocks_dtype=partial_mask_blocks_dtype,
      q_seq_shards=q_seq_shards,
  )

  fwd_mask_info, mask_function_fwd = process_fn(
      mask,
      (config.block_q, config.block_kv),
  )
  fwd_mask_sparsity = float(np.mean(fwd_mask_info.block_mask != 0))
  fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

  dkv_mask_info = None
  if config.has_backward_blocks:
    bq_dkv, bkv_dkv = config.block_q_dkv, config.block_kv_dkv
    dkv_mask_info, mask_function_dkv = process_fn(
        mask,
        (bq_dkv, bkv_dkv),
        is_dkv=True,
        return_dynamic_grid=config.dq_reduction_steps == 3,
    )

    assert (mask_function_fwd is None) == (mask_function_dkv is None)

    dkv_mask_sparsity = float(np.mean(dkv_mask_info.block_mask != 0))
    dkv_mask_info = tree_util.tree_map(jnp.array, dkv_mask_info)
  else:
    dkv_mask_sparsity = 1.0

  return SplashAttentionKernel(
      fwd_mask_info,
      dkv_mask_info,
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=mask_function_fwd,
      bias_function=bias_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
  )


def _make_dynamic_splash_attention(
    mask: jax.Array,
    *,
    mesh: jax.sharding.Mesh | None = None,
    mask_spec: jax.sharding.PartitionSpec | None = None,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
):
  if (mesh is not None) != (mask_spec is not None):
    raise ValueError(
        "Either both or neither of mesh and mask_spec must be specified."
    )

  if mask_spec is not None and len(mask_spec) != 1:
    raise ValueError("Only shard over the query sequence dimension.")

  if len(mask.shape) != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")

  if config is None:
    config = SplashConfig.get_default()

  # This is the only mode that supports the dynamic grid.
  config = dataclasses.replace(config, dq_reduction_steps=3)

  def process_mask_shard(mask):
    process_mask_fn = functools.partial(
        mask_info_lib._process_dynamic_mask,
        downcast_smem_data=downcast_smem_data,
        partial_mask_blocks_dtype=partial_mask_blocks_dtype,
    )

    fwd_mask_info = process_mask_fn(
        mask, (config.block_q, config.block_kv), is_dkv=False
    )

    dkv_mask_info = None
    if config.has_backward_blocks:
      dkv_mask_info = process_mask_fn(
          mask, (config.block_q_dkv, config.block_kv_dkv), is_dkv=True
      )

    return fwd_mask_info, dkv_mask_info

  kwargs = dict(
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=None,
      fwd_mask_sparsity=1.0,
      dkv_mask_sparsity=1.0,
  )

  # If the input mask is replicated we don't need to call shard_map.
  if mask_spec is None:
    fwd_mask_info, dkv_mask_info = process_mask_shard(mask)
    kernel = SplashAttentionKernel(fwd_mask_info, dkv_mask_info, **kwargs)
    return kernel

  mask_info_specs = MaskInfo(  # pytype: disable=wrong-arg-types
      mask_next=mask_spec,  # pyrefly: ignore[bad-argument-type]
      active_rows=None,
      active_cols=None,
      num_active_blocks=None,
      block_mask=mask_spec,  # pyrefly: ignore[bad-argument-type]
      partial_mask_blocks=mask_spec,  # pyrefly: ignore[bad-argument-type]
      q_sequence=None,
  )
  out_specs = (
      mask_info_specs,
      mask_info_specs if config.has_backward_blocks else None,
  )

  @partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=mask_spec,
      out_specs=out_specs,
      check_vma=False,
  )
  def process_all_shards(mask):
    return process_mask_shard(mask)

  fwd_mask_info, dkv_mask_info = process_all_shards(mask)
  kernel = SplashAttentionKernel(fwd_mask_info, dkv_mask_info, **kwargs)
  kernel_spec = SplashAttentionKernel(*out_specs, **kwargs)

  return (kernel, kernel_spec)


make_splash_mha = partial(_make_splash_attention, is_mqa=False)
make_splash_mqa = partial(_make_splash_attention, is_mqa=True)

make_splash_mha_single_device = partial(make_splash_mha, q_seq_shards=1)

make_splash_mqa_single_device = partial(make_splash_mqa, q_seq_shards=1)

make_dynamic_splash_mqa = partial(_make_dynamic_splash_attention, is_mqa=True)
make_dynamic_splash_mha = partial(_make_dynamic_splash_attention, is_mqa=False)
