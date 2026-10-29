# Copyright 2026 Google LLC

"""Splash attention with additive bias + in-kernel PRNG dropout for torch_tpu.

Wraps the vendored `sbd_splash/` kernel (a tokamax splash attention fork
adding additive bias and in-kernel PRNG dropout) as a PyTorch custom op
pair via `torch.tpu.pallas.jax_op`, following the torch_tpu
pallas signature contract: tensor args are traced inputs, absent optional
tensors are passed as None (never dummy sentinels, so no unused operand can
be dead-code-eliminated out of the lowered kernel), and every configuration
knob -- mask choice, block sizes, dropout and bias parameters -- is a
non-tensor static op argument, value-hashed into the trace cache key.
Config changes therefore always take effect on the next call, at any
shape; nothing is captured silently at trace time. Backward-only statics
ride through the forward signature so `setup_context` can hand them to the
paired backward op.

Capabilities over stock tokamax splash (which has neither feature):
  - additive attention bias: full-rank `(H, S, S)` with EXACT dbias, or the
    cheap per-key rank `(H, 16, S)` (rows identical; row 0 is the logical
    bias) with optional rank-reduced trainable dbias;
  - in-kernel PRNG dropout on the attention probabilities (fixed-quantum
    keying; fold-rescale; quarter-area mask v4 + the dkv fused-compare
    frontier config), seeded by a per-batch `(B, 2)` int32 seed-word tensor
    so changing seeds never retraces.

Coverage matches the stock splash wrapper otherwise: causal, full
(non-causal), or sliding-window masks, packed-document `segment_ids`
(exact token-level masking, plus document-aware block skipping on the
compacted grid -- see `_document_aware_mask_info`), an explicit softmax
scale, and MHA / grouped-KV / MQA head layouts (MQA routes through the
kernel's dedicated rank-reduced k/v entry).

Features off, the kernel is bit-exact with unmodified tokamax splash (the
family's standing no-regression contract). Dropout costs ~+12-13% and the
bias tiers ~+2% (per-key) / ~+12% (function-style) / ~+43% (full-rank) over
features-off at the validated op-points.

Scheduler note (measured): `use_experimental_scheduler` inverts sign across
op-points -- on a GQA fwd+bwd shape the tokamax default (False) was ~2.4%
faster, while on an MHA b4/S4096/H64 fwd+bwd shape False was an 8.06x
DETERMINISTIC regression. Sweep it per op-point x pass-set before changing
the default.

Provenance: the splash-bias-dropout kernel family (closed at ceiling
2026-08-27); every capability and price above is verifier-gated there.
"""

import math
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
import torch
import torch_tpu  # pylint: disable=unused-import
from torch.tpu import pallas  # pyrefly: ignore[missing-import]
from torchtitan.experiments.tpu.kernels.sbd_splash import (
    splash_attention_kernel as sbd_kernel,
)
from torchtitan.experiments.tpu.kernels.sbd_splash import (
    splash_attention_mask as sbd_mask,
)


_MASK_VALUE = float(jnp.finfo(jnp.float32).min)
PER_KEY_ROWS = 16  # the padded per-key bias rank (bf16 (16, 128) tile alignment)


class SbdConfig:
  """Kernel configuration, forwarded as static op arguments on every call.

  Each value is part of the op's trace-cache key, so `update(...)` takes
  effect on the next call regardless of what shapes were traced before.
  Defaults pin the family's verified frontier configuration (fold-rescale +
  mask v4 + dkv fused-compare).
  """

  block_q: int = 512
  block_kv: int = 512
  block_kv_compute: int = 512
  block_q_dkv: int = 512
  block_kv_dkv: int = 512
  block_kv_dkv_compute: int = 512
  use_experimental_scheduler: bool = False  # tokamax default; see module note
  # Natural exp (False) or exp2 with LOG2E folded into q (True, tokamax's
  # default). True multiplies q by LOG2E in q's dtype -- a second bf16
  # rounding of q that the stock splash kernel does not do. Vs a float64
  # reference at 16k (1 v6e chip, the lane's head layouts), False has stock
  # splash's output error exactly (True: 25-60% higher) and dk/dv error
  # 20-50% below stock splash's (which sums repeated K/V grads in bf16);
  # True is 2.4% faster (median fwd+bwd; forward +0.2%, backward +3.4%).
  use_base2_exp: bool = False
  # Dropout (the verified frontier config; rate 0.0 disables the paths).
  dropout_rate: float = 0.0
  dropout_fold_rescale: bool = True
  dropout_mask_version: int = 4
  dropout_bwd_fused_compare: bool = True
  # Bias rank: 0 = full-rank (H, S, S) with exact dbias; PER_KEY_ROWS = the
  # per-key rank (H, 16, S). Trainable per-key dbias via the two knobs below
  # (MXU ones-matmul reduction is the verified best realization).
  bias_per_key_rows: int = 0
  bias_per_key_dbias: bool = False
  bias_per_key_dbias_mxu_reduce: bool = True
  attn_logits_soft_cap: Optional[float] = None
  # Skip grid steps that no document in the pack can need (see
  # `_document_aware_mask_info`). Only ever active when `segment_ids` is
  # passed. Per-example block metadata serializes the batch (see
  # `_batch_map`): free at batch 1 (the lane folds its batch into one packed
  # stream); above it, ~9% on a pass whose samples are single documents, a
  # win as soon as they hold several. Off is exact too, just denser.
  document_aware_skipping: bool = True
  # The fork's `SplashConfig.dq_reduction_steps`: 0 keeps its default (one
  # unreduced dq partial per kv step, fp32 once kv_steps > 4, summed outside
  # the kernel -- `(kv_steps, H, S, D)`, the largest backward temporary at
  # long sequence); 3 accumulates dq in-kernel into three partials of q's
  # dtype AND gives the dkv pass a dynamic block list, which
  # `_document_aware_mask_info` then compacts like the forward's. Not
  # bit-identical to 0: the dq (and grouped-KV dk/dv) fp32 summation order
  # changes (~1e-6 at fp32 inputs on v6e), and the dq partials round to q's
  # dtype. Measured on Llama-3.2-1B, bs 8, seq 2048, 4x v6e: the dkv kernel
  # 178.7 -> 75.9 ms and the 45 ms dq sum gone (+19% tok/s, +15% over stock
  # splash); over 40 steps its loss stays within 2.5e-3 of 0's, inside the
  # 4.0e-3 that separates 0 from stock splash. -1 (default) picks by
  # length: 3 from four kv blocks up, else 0. At two kv blocks there is
  # almost nothing to reduce and 3's dynamic list costs ~10% of the dkv
  # pass; at four the two tie, at eight 3 is 15-25% faster (1 v6e chip,
  # block 1024).
  dq_reduction_steps: int = -1

  @classmethod
  def update(cls, **kwargs):
    for k, v in kwargs.items():
      if v is not None:
        setattr(cls, k, v)

  @classmethod
  def static_args(cls) -> tuple:
    """The static op arguments, in the order the op signatures declare."""
    return (
        cls.block_q,
        cls.block_kv,
        cls.block_kv_compute,
        cls.block_q_dkv,
        cls.block_kv_dkv,
        cls.block_kv_dkv_compute,
        cls.use_experimental_scheduler,
        cls.use_base2_exp,
        cls.attn_logits_soft_cap,
        cls.dropout_rate,
        cls.dropout_fold_rescale,
        cls.dropout_mask_version,
        cls.dropout_bwd_fused_compare,
        cls.bias_per_key_rows,
        cls.bias_per_key_dbias,
        cls.bias_per_key_dbias_mxu_reduce,
        cls.document_aware_skipping,
        cls.dq_reduction_steps,
    )


def _build_kernel(
    seq_len: int,
    causal: bool,
    scale: float,
    local_window_size: int,
    use_dropout: bool,
    use_bias: bool,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_kv_dkv_compute: int,
    use_experimental_scheduler: bool,
    use_base2_exp: bool,
    attn_logits_soft_cap: float | None,
    dropout_rate: float,
    dropout_fold_rescale: bool,
    dropout_mask_version: int,
    dropout_bwd_fused_compare: bool,
    bias_per_key_rows: int,
    bias_per_key_dbias: bool,
    bias_per_key_dbias_mxu_reduce: bool,
    dq_reduction_steps: int = 0,
):
  """Constructs the fork kernel + static mask infos for one configuration."""
  if dq_reduction_steps < 0:
    dq_reduction_steps = 3 if seq_len >= 4 * min(block_kv_dkv, seq_len) else 0
  cfg = sbd_kernel.SplashConfig(
      block_q=min(block_q, seq_len),
      block_kv=min(block_kv, seq_len),
      block_kv_compute=min(block_kv_compute, seq_len),
      block_q_dkv=min(block_q_dkv, seq_len),
      block_kv_dkv=min(block_kv_dkv, seq_len),
      block_kv_dkv_compute=min(block_kv_dkv_compute, seq_len),
      use_experimental_scheduler=use_experimental_scheduler,
      use_base2_exp=use_base2_exp,
      # See `SbdConfig.dq_reduction_steps`. 3 used to return wrong dk/dv for
      # every grouped-KV / MQA shape with a single kv block; fixed in the fork
      # (`_splash_attention_bwd_dkv`, `bounds_end == 2`).
      dq_reduction_steps=dq_reduction_steps or None,
      attn_logits_soft_cap=attn_logits_soft_cap,
      dropout_rate=dropout_rate if use_dropout else 0.0,
      dropout_fold_rescale=dropout_fold_rescale,
      dropout_mask_version=dropout_mask_version if use_dropout else 1,
      dropout_bwd_fused_compare=dropout_bwd_fused_compare and use_dropout,
      bias_per_key_rows=bias_per_key_rows if use_bias else 0,
      bias_per_key_dbias=bias_per_key_dbias and use_bias,
      bias_per_key_dbias_mxu_reduce=bias_per_key_dbias_mxu_reduce,
  )
  del scale  # applied to q by the callers; not part of the mask/config
  if local_window_size >= 0:
    # Sliding window takes precedence over `causal` (stock wrapper
    # convention): query i attends keys [i - local_window_size, i].
    mask = sbd_mask.LocalMask(
        shape=(seq_len, seq_len),
        window_size=(local_window_size, 0),
        offset=0,
        shard_count=1,
    )
  elif causal:
    mask = sbd_mask.CausalMask(shape=(seq_len, seq_len), shard_count=1)
  else:
    mask = sbd_mask.FullMask((seq_len, seq_len))
  kernel = sbd_kernel.make_splash_mha(
      mask=mask,
      config=cfg,
      q_seq_shards=1,
      save_residuals=False,
      mask_value=_MASK_VALUE,
      downcast_smem_data=False,
  )
  return cfg, kernel


# Largest `block * head_dim * itemsize` the dkv kernel tiles may span: 512
# rows of fp32 head_dim 256 (or bf16 head_dim 512). The dkv kernel holds
# double-buffered q / k / v / do tiles, the dq / dk / dv outputs and fp32
# dk / dv accumulators, all `(block, head_dim)`; at fp32 head_dim 512 with
# 512-row blocks that is 16.6 MiB, past the 16 MiB default scoped VMEM limit
# on v5e (CompileTimeScopedVmemOom on Gemma-4's global layers).
_DKV_TILE_BYTES = 512 * 256 * 4


def _dkv_blocks(block_q_dkv, block_kv_dkv, block_kv_dkv_compute, head_dim, itemsize):
  """Caps the dkv block sizes so their tiles fit `_DKV_TILE_BYTES`.

  A no-op for every layout up to fp32 head_dim 256 / bf16 head_dim 512;
  wider (or wider-typed) heads get shorter blocks. The cap is a power of two
  (never below one 128-row tile), so it divides any sequence length the
  configured power-of-two blocks divide.
  """
  cap = 128
  while 2 * cap * head_dim * itemsize <= _DKV_TILE_BYTES:
    cap *= 2
  return (
      min(block_q_dkv, cap),
      min(block_kv_dkv, cap),
      min(block_kv_dkv_compute, cap),
  )


def _batch_map(fn, args, in_axes, serial):
  """`vmap(fn, in_axes)(*args)`, or a `lax.scan` over the batch if `serial`.

  Per-example block metadata (document-aware skipping) already makes XLA
  run the batched kernel as a serial loop. Spelling that loop as a scan
  instead lets XLA place each example's operands in VMEM, which the batched
  form does not: on 1 v6e chip, 8 x 2048 single-document samples, the dkv
  kernel drops 3.25 -> 2.85 ms (Llama-3.2-1B heads) and fwd+bwd is 7-12%
  faster across the lane's head layouts. Without per-example metadata the
  vmap stays one batched kernel, which is the faster form there (an
  MHA 20 x 128 bidirectional pass: 4.7 vs 5.3 ms). `None` args (an absent
  seed / segment_ids) broadcast either way.
  """
  if serial:
    return jax.lax.scan(lambda c, xs: (c, fn(*xs)), None, tuple(args))[1]
  return jax.vmap(fn, in_axes=in_axes)(*args)


def _kernel_statics(kernel):
  return (
      kernel.fwd_mask_info,
      kernel.dkv_mask_info,
      kernel.kwargs.get("mask_value", _MASK_VALUE),
      kernel.kwargs.get("mask_function", None),
      kernel.kwargs.get("fwd_mask_sparsity", 1.0),
      kernel.kwargs.get("dkv_mask_sparsity", 1.0),
  )


def _block_doc_overlaps(seg_b, bq: int, bkv: int):
  """`(num_q_blocks, num_kv_blocks)` bool: can this tile hold a same-document pair?

  A tile can only matter if some id appears in both its q block and its kv
  block, and a shared id lies inside both blocks' `[min, max]` intervals --
  so non-overlapping intervals prove the tile is entirely cross-document.
  The converse does not hold (overlapping intervals need not share an id),
  which makes the test conservative in the safe direction: it never drops a
  tile that carries attention, and it is exact for the monotone packed
  stream `segment_ids_from_positions` produces. O(num_blocks) either way.

  Returns None when the blocks do not tile the sequence, which tells the
  caller to leave the static metadata alone.
  """
  seq_len = seg_b.shape[-1]
  num_q, q_rem = divmod(seq_len, bq)
  num_kv, kv_rem = divmod(seq_len, bkv)
  if q_rem or kv_rem:
    return None
  q_seg = seg_b.reshape(num_q, bq)
  kv_seg = seg_b.reshape(num_kv, bkv)
  q_lo, q_hi = jnp.min(q_seg, axis=-1), jnp.max(q_seg, axis=-1)
  kv_lo, kv_hi = jnp.min(kv_seg, axis=-1), jnp.max(kv_seg, axis=-1)
  return (q_lo[:, None] <= kv_hi[None, :]) & (kv_lo[None, :] <= q_hi[:, None])


def _onehot_lookup(table, idx):
  """`table[idx]` for in-range `idx`, without an XLA gather.

  On TPU7x XLA offloads large gathers to SparseCore, whose lowering fails for
  schedule-sized index arrays (from 16k tokens up); a one-hot select-reduce
  fuses into one elementwise-plus-reduce instead (tokamax 0.0.16 does the
  same in its segment pruning).
  """
  onehot = idx[..., None] == jnp.arange(table.shape[-1], dtype=idx.dtype)
  return jnp.sum(jnp.where(onehot, table, jnp.zeros_like(table)), axis=-1)


def _document_aware_mask_info(mask_info, seg_b, bq: int, bkv: int, *, is_dkv: bool):
  """Drop the grid steps this packed example cannot need; shrink the grid.

  `segment_ids` alone masks scores *inside* tiles the kernel already decided
  to visit, and that decision comes from the static causal / window block
  metadata, which is blind to documents -- so a packed stream still
  multiplies fully-masked tiles and throws the products away. At the lane's
  op-point (8 x 2048-token documents folded into one 16k stream, block 1024)
  that is 136 causal tiles per head of which 24 can hold a same-document
  pair.

  Both halves of this were measured against doing nothing, at that op-point
  (Llama-3.2-1B, 4x v6e, tok/s/device): **14,307** with segment ids alone,
  which is what the tokamax wrapper and commit a18deacfe9 do; **18,640** if
  the tiles are merely zeroed, so the kernel skips their arithmetic but still
  walks all 136 grid steps; **20,210** if the list is also compacted to 24
  steps. The interval test is 79 % of the win; the compaction is the last
  8.4 %, which is also the 8.4 % that buys parity with the stock splash kernel
  (20,102) and so the right to stop maintaining it. The whole helper costs
  0.167 ms per step across its 32 invocations, against ~340 ms saved.

  The two passes carry different metadata and so get different treatment:

  - The forward `MaskInfo` is a compacted active-block LIST (one entry per
    grid step), so its steps are filtered and restacked here and
    `num_active_blocks` -- which the kernel uses as its grid bound -- shrinks
    with them. Pallas takes a traced grid bound, so a dropped step costs
    nothing at all: no DMA, no compute, no grid iteration.
  - The dkv `MaskInfo` is, by default, a DENSE `[kv_block][q_block]` grid,
    so there the tiles are only zeroed. Both kernels read `block_mask == 0`
    as "visit, initialize, write, compute nothing", which is how the dkv
    pass already skips the statically-masked half of a causal grid; the
    arithmetic goes away, the grid iteration and its q-side DMA do not.
    With `SbdConfig.dq_reduction_steps == 3` the dkv `MaskInfo` is a list
    too (kv-major, the head axis outermost in the grid) and takes the
    compaction path below, rows and cols swapped.

  A row that loses every step keeps its first one as an anchor, because a
  row's output tile is written on its first / last visited step: on the dkv
  grid `dk`/`dv` are not aliased for MHA, so a row that vanished entirely
  would leave garbage (this is also why the builder extends the dkv grid
  over empty kv rows, and those extension entries are exactly such
  anchors). An anchor carries `block_mask == 0`, which both kernels read as
  "visit, initialize, write, compute nothing".

  Row order is preserved, so `find_bounds`' neighbour comparison still finds
  each row's first and last step; the dead tail gets row `-1` so the last
  live step always reads as a row end.
  """
  base_rows, base_cols = mask_info.active_rows, mask_info.active_cols
  base_mask, base_num = mask_info.block_mask, mask_info.num_active_blocks
  if base_mask is None:
    return mask_info
  overlaps = _block_doc_overlaps(seg_b, bq, bkv)
  if overlaps is None:
    return mask_info

  if base_rows is None or base_cols is None or base_num is None:
    # Dense grid (the dkv pass): zero the tiles in place. `grid_idx` is
    # row-major over [kv][q] there and over [q][kv] on a dense forward grid.
    dense = jnp.asarray(base_mask)
    tiles = overlaps.T if is_dkv else overlaps
    if dense.size != tiles.size:
      return mask_info
    return mask_info._replace(
        block_mask=jnp.where(tiles.reshape(-1), dense.reshape(-1), 0)
        .reshape(dense.shape)
        .astype(dense.dtype)
    )

  rows = jnp.asarray(base_rows).astype(jnp.int32)
  cols = jnp.asarray(base_cols).astype(jnp.int32)
  mask = jnp.asarray(base_mask).astype(jnp.int32)
  num_steps = rows.shape[0]
  # Entries past the static count are padding (row -1, block_mask 0); keeping
  # them out of `keep` is what lets the grid bound shrink below the static one.
  # `num_active_blocks` is already traced on this path (the kernel's grid
  # bound is dynamic), so compare it as an array rather than as a python int.
  valid = jnp.arange(num_steps) < jnp.asarray(base_num).reshape(-1)[0].astype(
      jnp.int32
  )
  # The dkv grid is transposed: its rows are kv blocks and its cols q blocks.
  q_idx, kv_idx = (cols, rows) if is_dkv else (rows, cols)
  num_q, num_kv = overlaps.shape
  # Row of `overlaps` per step, then the step's column of it: two one-hot
  # lookups rather than a 2-D gather (see `_onehot_lookup`).
  q_c, kv_c = jnp.clip(q_idx, 0, num_q - 1), jnp.clip(kv_idx, 0, num_kv - 1)
  q_rows = jnp.sum(
      jnp.where((q_c[:, None] == jnp.arange(num_q))[:, :, None], overlaps[None], False),
      axis=1, dtype=jnp.int32,
  )
  works = (_onehot_lookup(q_rows, kv_c) > 0) & (mask > 0) & valid
  # A row with no work left still has to be visited once, to initialize and
  # write its output tile (see the docstring). Rows that keep work need no
  # anchor -- and with self-attention every row does, since a block always
  # shares its own tokens' documents with itself, so this is a guard rather
  # than a cost.
  num_rows = num_kv if is_dkv else num_q
  # Which rows keep work, and each step's row's verdict: a one-hot reduce and
  # lookup rather than a scatter and a gather.
  rows_c = jnp.clip(rows, 0, num_rows - 1)
  row_eq = rows_c[:, None] == jnp.arange(num_rows)[None, :]
  row_has_work = jnp.any(row_eq & works[:, None], axis=0)
  row_start = (
      jnp.concatenate([jnp.ones((1,), dtype=bool), rows[1:] != rows[:-1]])
      & valid
  )
  anchor = row_start & ~_onehot_lookup(row_has_work, rows_c)
  keep = works | anchor
  num_keep = keep.sum().astype(jnp.int32)
  live = jnp.arange(num_steps) < num_keep
  # Stable partition, kept steps first in grid order: a stable sort keyed on
  # ~keep carries the payloads along (no argsort-then-gather).
  payload = [rows, cols, jnp.where(works, mask, 0)]
  if mask_info.mask_next is not None:
    payload.append(jnp.asarray(mask_info.mask_next).astype(jnp.int32))
  _, *payload = lax.sort(
      ((~keep).astype(jnp.int32), *payload), num_keys=1, is_stable=True
  )

  def _restack(values, dead, dtype):
    return jnp.where(live, values, dead).astype(dtype)

  repl = dict(
      active_rows=_restack(payload[0], -1, base_rows.dtype),
      active_cols=_restack(payload[1], 0, base_cols.dtype),
      block_mask=_restack(payload[2], 0, base_mask.dtype),
      num_active_blocks=num_keep[None],
  )
  if mask_info.mask_next is not None:
    repl["mask_next"] = _restack(payload[3], 0, mask_info.mask_next.dtype)
  return mask_info._replace(**repl)


def sbd_attention_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    bias: jax.Array | None,
    dropout_seed: jax.Array | None,
    segment_ids: jax.Array | None,
    causal: bool,
    scale: float,
    local_window_size: int,
    heads_first: bool,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_kv_dkv_compute: int,
    use_experimental_scheduler: bool,
    use_base2_exp: bool,
    attn_logits_soft_cap: float | None,
    dropout_rate: float,
    dropout_fold_rescale: bool,
    dropout_mask_version: int,
    dropout_bwd_fused_compare: bool,
    bias_per_key_rows: int,
    bias_per_key_dbias: bool,
    bias_per_key_dbias_mxu_reduce: bool,
    document_aware_skipping: bool,
    dq_reduction_steps: int,
) -> tuple[jax.Array, jax.Array]:
  """Forward. Layouts: q `(B, S, H, D)` torch-standard; k/v `(B, S, H_kv, D)`
  with `H_kv == H` (MHA), `H % H_kv == 0` (grouped KV), or `H_kv == 1`
  (MQA, rank-reduced kernel entry); `bias` is `(H, S, S)` full-rank or
  `(H, PER_KEY_ROWS, S)` per-key, shared across batch, or None;
  `dropout_seed` is `(B, 2)` int32 seed words, or None when dropout is off;
  `segment_ids` is `(B, S)` int32 packed-document ids (a token attends only
  within its own id, on top of the causal / window mask), or None. The
  remaining args are trace-time statics; the dkv-only ones ride along for
  `setup_context` to hand to the backward op."""
  seq_len = q.shape[2 if heads_first else 1]
  use_bias = bias is not None
  use_dropout = dropout_seed is not None
  is_mqa = k.shape[1 if heads_first else 2] == 1
  cfg, kernel = _build_kernel(
      seq_len,
      causal,
      scale,
      local_window_size,
      use_dropout,
      use_bias,
      block_q,
      block_kv,
      block_kv_compute,
      block_q_dkv,
      block_kv_dkv,
      block_kv_dkv_compute,
      use_experimental_scheduler,
      use_base2_exp,
      attn_logits_soft_cap,
      dropout_rate,
      dropout_fold_rescale,
      dropout_mask_version,
      dropout_bwd_fused_compare,
      bias_per_key_rows,
      bias_per_key_dbias,
      bias_per_key_dbias_mxu_reduce,
      dq_reduction_steps,
  )
  fwd_mi, _, mask_value, mask_fn, fwd_sp, _ = _kernel_statics(kernel)

  scale_arr = jnp.array(scale, dtype=q.dtype)
  # `heads_first` operands are already in the kernel's own (B, H, S, D)
  # layout, so the pair of transposes (one here, one back on the way out)
  # drops out -- and with it three full-size copies per call that the forward
  # would otherwise save for the backward. Measured 2026-10-01 at the lane's
  # 16k x 32-head op-point, 16 stacked layers fwd+bwd: 5.53 GiB / 308 ms
  # heads-first versus 6.87 GiB / 357 ms transposing.
  q_tr = q * scale_arr
  k_tr, v_tr = k, v
  if not heads_first:
    q_tr = jnp.transpose(q_tr, (0, 2, 1, 3))  # (B, H, S, D)
    k_tr = jnp.transpose(k, (0, 2, 1, 3))
    v_tr = jnp.transpose(v, (0, 2, 1, 3))
  if is_mqa:
    # The kernel's MQA entry takes rank-reduced k/v: (S, D) per batch.
    k_tr = k_tr[:, 0]
    v_tr = v_tr[:, 0]

  def _fwd_b(q_b, k_b, v_b, seed_b, seg_b):
    fwd_mi_b = fwd_mi
    if seg_b is not None and document_aware_skipping:
      fwd_mi_b = _document_aware_mask_info(
          fwd_mi, seg_b, cfg.block_q, cfg.block_kv, is_dkv=False
      )
    # pylint: disable=protected-access
    out_b, stats_b = sbd_kernel._splash_attention_forward(
        fwd_mi_b,
        q_b,
        k_b,
        v_b,
        bias,
        None,  # dropout_mask (explicit-mask path unused here)
        seed_b,
        _segment_ids(seg_b),
        None,  # sinks
        mask_value,
        is_mqa,
        cfg,
        True,  # save_residuals
        mask_fn,
        bias_function=None,
        fwd_mask_sparsity=fwd_sp,
        # Only logsumexp is wanted here; `max_logits` would be another
        # lane-padded (H, S, 128) fp32 buffer per call, allocated whether or
        # not anything reads it (an opaque custom call has no dead outputs).
        save_max_logits=False,
    )
    # pylint: enable=protected-access
    return out_b, stats_b["logsumexp"]

  # in_axes=None broadcasts an absent seed / segment_ids to every batch element.
  out_tr, lse = _batch_map(
      _fwd_b,
      (q_tr, k_tr, v_tr, dropout_seed, segment_ids),
      (0, 0, 0, 0 if use_dropout else None, 0 if segment_ids is not None else None),
      serial=_per_example(fwd_mi, segment_ids, document_aware_skipping),
  )
  out = out_tr if heads_first else jnp.transpose(out_tr, (0, 2, 1, 3))
  return out, lse


def sbd_attention_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    bias: jax.Array | None,
    dropout_seed: jax.Array | None,
    segment_ids: jax.Array | None,
    out: jax.Array,
    logsumexp: jax.Array,
    grad_out: jax.Array,
    causal: bool,
    scale: float,
    local_window_size: int,
    heads_first: bool,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_kv_dkv_compute: int,
    use_experimental_scheduler: bool,
    use_base2_exp: bool,
    attn_logits_soft_cap: float | None,
    dropout_rate: float,
    dropout_fold_rescale: bool,
    dropout_mask_version: int,
    dropout_bwd_fused_compare: bool,
    bias_per_key_rows: int,
    bias_per_key_dbias: bool,
    bias_per_key_dbias_mxu_reduce: bool,
    document_aware_skipping: bool,
    dq_reduction_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Backward from saved residuals -- no forward re-run. Returns
  `(dq, dk, dv, dbias)`; `dbias` is a `(0,)` placeholder when bias is None
  (the op's output arity is fixed)."""
  seq_len = q.shape[2 if heads_first else 1]
  use_bias = bias is not None
  use_dropout = dropout_seed is not None
  is_mqa = k.shape[1 if heads_first else 2] == 1
  block_q_dkv, block_kv_dkv, block_kv_dkv_compute = _dkv_blocks(
      block_q_dkv, block_kv_dkv, block_kv_dkv_compute, q.shape[-1],
      jnp.dtype(q.dtype).itemsize,
  )
  cfg, kernel = _build_kernel(
      seq_len,
      causal,
      scale,
      local_window_size,
      use_dropout,
      use_bias,
      block_q,
      block_kv,
      block_kv_compute,
      block_q_dkv,
      block_kv_dkv,
      block_kv_dkv_compute,
      use_experimental_scheduler,
      use_base2_exp,
      attn_logits_soft_cap,
      dropout_rate,
      dropout_fold_rescale,
      dropout_mask_version,
      dropout_bwd_fused_compare,
      bias_per_key_rows,
      bias_per_key_dbias,
      bias_per_key_dbias_mxu_reduce,
      dq_reduction_steps,
  )
  _, dkv_mi, mask_value, mask_fn, fwd_sp, dkv_sp = _kernel_statics(kernel)

  scale_arr = jnp.array(scale, dtype=q.dtype)
  if heads_first:
    q_tr, k_tr, v_tr = q * scale_arr, k, v
    out_tr, g_tr = out, grad_out
  else:
    q_tr = jnp.transpose(q * scale_arr, (0, 2, 1, 3))
    k_tr = jnp.transpose(k, (0, 2, 1, 3))
    v_tr = jnp.transpose(v, (0, 2, 1, 3))
    out_tr = jnp.transpose(out, (0, 2, 1, 3))
    g_tr = jnp.transpose(grad_out, (0, 2, 1, 3))
  if is_mqa:
    k_tr = k_tr[:, 0]
    v_tr = v_tr[:, 0]

  def _bwd_b(q_b, k_b, v_b, out_b, lse_b, g_b, seed_b, seg_b):
    dkv_mi_b = dkv_mi
    if seg_b is not None and document_aware_skipping:
      dkv_mi_b = _document_aware_mask_info(
          dkv_mi, seg_b, cfg.block_q_dkv, cfg.block_kv_dkv, is_dkv=True
      )
    res = (
        q_b,
        k_b,
        v_b,
        bias,
        None,  # dropout_mask
        seed_b,
        _segment_ids(seg_b),
        None,  # sinks
        out_b,
        lse_b,
        dkv_mi_b,
    )
    # pylint: disable=protected-access
    grads = sbd_kernel._splash_attention_bwd(
        False,  # save_residuals
        mask_value,
        is_mqa,
        cfg,
        mask_fn,
        None,  # bias_function
        fwd_sp,
        dkv_sp,
        res,
        g_b,
    )
    # pylint: enable=protected-access
    dq_b, dk_b, dv_b, dbias_b = grads[2], grads[3], grads[4], grads[5]
    if dbias_b is None:
      dbias_b = jnp.zeros((0,), dtype=q_b.dtype)
    return dq_b, dk_b, dv_b, dbias_b

  dq_tr, dk_tr, dv_tr, dbias_b = _batch_map(
      _bwd_b,
      (q_tr, k_tr, v_tr, out_tr, logsumexp, g_tr, dropout_seed, segment_ids),
      (0, 0, 0, 0, 0, 0, 0 if use_dropout else None,
       0 if segment_ids is not None else None),
      serial=_per_example(dkv_mi, segment_ids, document_aware_skipping),
  )

  # chain rule for q * scale; grads leave in the layout q arrived in.
  if heads_first:
    dq = dq_tr * scale_arr
    if is_mqa:
      dk = dk_tr[:, None]  # (B, S, D) -> (B, 1, S, D)
      dv = dv_tr[:, None]
    else:
      dk, dv = dk_tr, dv_tr
  else:
    dq = jnp.transpose(dq_tr, (0, 2, 1, 3)) * scale_arr
    if is_mqa:
      dk = dk_tr[:, :, None, :]  # (B, S, D) -> (B, S, 1, D)
      dv = dv_tr[:, :, None, :]
    else:
      dk = jnp.transpose(dk_tr, (0, 2, 1, 3))
      dv = jnp.transpose(dv_tr, (0, 2, 1, 3))
  # bias is shared across the batch: sum the per-batch cotangents.
  if use_bias:
    dbias = jnp.sum(dbias_b, axis=0).astype(bias.dtype)
  else:
    dbias = jnp.zeros((0,), dtype=q.dtype)
  return dq, dk, dv, dbias


def _per_example(mask_info, segment_ids, document_aware_skipping):
  """Whether `_document_aware_mask_info` gives each of several examples its
  own metadata (`_batch_map`'s `serial`). At batch 1 the vmap is the faster
  form of the single call: 8.15 vs 8.47 ms fwd+bwd, Llama-3.2-1B heads, 16k."""
  return (
      segment_ids is not None
      and segment_ids.shape[0] > 1
      and document_aware_skipping
      and mask_info.block_mask is not None
  )


def _segment_ids(seg_b):
  """One batch element's `(S,)` ids as the kernel's `SegmentIds`, or None.

  The same ids serve q and kv (self-attention over one packed stream): a
  block straddling a document boundary is visited and masked inside at token
  granularity, while the blocks no document can reach are dropped from the
  grid first by `_document_aware_mask_info`.
  """
  if seg_b is None:
    return None
  return sbd_kernel.SegmentIds(q=seg_b, kv=seg_b)


_sbd_fwd = pallas.jax_op("pallas::sbd_splash_attn_fwd", sbd_attention_fwd)
_sbd_bwd = pallas.jax_op("pallas::sbd_splash_attn_bwd", sbd_attention_bwd)

_NUM_STATIC_ARGS = 4 + len(SbdConfig.static_args())  # causal, scale, window, layout + SbdConfig


def _setup_context(ctx, inputs, output):
  q, k, v, bias, dropout_seed, segment_ids = inputs[: len(inputs) - _NUM_STATIC_ARGS]
  out, logsumexp = output
  ctx.save_for_backward(q, k, v, bias, dropout_seed, segment_ids, out, logsumexp)
  ctx.static_args = tuple(inputs[len(inputs) - _NUM_STATIC_ARGS :])


def _backward(ctx, grad_out, grad_lse):
  del grad_lse
  q, k, v, bias, dropout_seed, segment_ids, out, logsumexp = ctx.saved_tensors
  dq, dk, dv, dbias = _sbd_bwd(
      q, k, v, bias, dropout_seed, segment_ids, out, logsumexp, grad_out, *ctx.static_args
  )
  dbias_ret = dbias if bias is not None else None
  return (dq, dk, dv, dbias_ret, None, None) + (None,) * _NUM_STATIC_ARGS


_sbd_fwd.register_autograd(_backward, setup_context=_setup_context)


def _seed_words(seed: int, batch: int, device: torch.device) -> torch.Tensor:
  """Deterministic per-batch (B, 2) int32 seed words from one python int.

  splitmix64-style mixing done torch-side; the kernel further mixes per
  (head, quantum) internally, so only per-batch decorrelation matters here.
  Passing seeds as a TENSOR keeps seed changes from retracing the op.
  """
  b = torch.arange(batch, dtype=torch.int64)
  z = (torch.tensor(seed, dtype=torch.int64) + b * 0x9E3779B97F4A7C15) & (
      (1 << 64) - 1
  )
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
  z = (z ^ (z >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
  z = z ^ (z >> 31)
  lo = (z & 0xFFFFFFFF).to(torch.uint32).view(torch.int32)
  hi = ((z >> 32) & 0xFFFFFFFF).to(torch.uint32).view(torch.int32)
  return torch.stack([lo, hi], dim=1).to(device)


def sbd_splash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    dropout_seed: int | torch.Tensor = 0,
    segment_ids: Optional[torch.Tensor] = None,
    is_causal: bool = True,
    scale: Optional[float] = None,
    local_window_size: Optional[int] = None,
    heads_first: bool = False,
) -> torch.Tensor:
  """Splash attention with optional additive bias and PRNG dropout.

  Args:
    q: `(B, S, H, D)`; k/v: `(B, S, H_kv, D)` -- MHA (`H_kv == H`), grouped
      KV (`H % H_kv == 0`), and MQA (`H_kv == 1`, routed through the
      kernel's rank-reduced entry) are all supported. With
      `heads_first=True` the layout is `(B, H, S, D)` instead, in and out.
    bias: optional, shared across batch -- `(H, S, S)` full-rank (exact
      dbias) or `(H, 16, S)` per-key when `SbdConfig.bias_per_key_rows ==
      PER_KEY_ROWS` (rows identical; trainable dbias via
      `SbdConfig.bias_per_key_dbias`).
    dropout_seed: python int (expanded to per-batch seed words) or a
      precomputed `(B, 2)` int32 tensor. Only used when
      `SbdConfig.dropout_rate > 0`. Seeding is explicit and deterministic
      (no torch RNG state): pass a fresh value every step (e.g. the global
      step), and under data/tensor parallelism fold the rank in as well --
      ranks passing the same value apply identical masks at the same local
      positions. Neither form retraces on seed change (the seed reaches
      the op as a tensor input).
    segment_ids: optional `(B, S)` integer packed-document ids. A token
      attends only to keys with its own id, on top of the causal / window
      mask -- exact token-level masking; blocks are not skipped on it.
      Made contiguous and int32 here: the jax_op bridge reads an operand as
      dense regardless of its strides.
    is_causal: causal (default) or full attention. Like every `SbdConfig`
      knob, a trace-time static: each value compiles its own kernel, so
      mixing configurations across calls at the same shapes is safe.
    scale: softmax scale; defaults to `1/sqrt(D)`.
    local_window_size: sliding-window attention -- query i attends keys
      `[i - local_window_size, i]`. Takes precedence over `is_causal`
      (stock wrapper convention).
    heads_first: operands and results are `(B, H, S, D)` rather than
      torch-standard `(B, S, H, D)`. This is the kernel's OWN layout, so it
      is the cheaper call: the wrapper's transpose pair disappears, and with
      it three saved full-size copies per call. Prefer it from callers that
      already hold heads-first tensors (HF attention does).

  Returns:
    `(B, S, H, D)` attention output. Dropout follows the family convention:
    applied to the post-softmax probabilities with a single `1/(1-rate)`
    rescale; online-softmax state and logsumexp stay undropped.
  """
  # TT-UNFLATTEN-SHIM: must stay BELOW the docstring -- code above the string
  # literal would demote it from a docstring to a discarded expression and
  # leave __doc__ as None.
  from torchtitan.experiments.tpu.unflatten_shim import unflatten_tensor, flatten_tensor
  # `bias` is batch-shared (H, S, S), so it is never a token stream.
  was_flat = q.ndim == 3
  q = unflatten_tensor(q, expected_flat_ndim=3)
  k = unflatten_tensor(k, expected_flat_ndim=3)
  v = unflatten_tensor(v, expected_flat_ndim=3)

  h_axis = 1 if heads_first else 2
  s_axis = 2 if heads_first else 1
  if q.shape[h_axis] % k.shape[h_axis] != 0:
    raise ValueError(
        f"num q heads ({q.shape[h_axis]}) must be a multiple of num kv heads"
        f" ({k.shape[h_axis]})"
    )
  if SbdConfig.dropout_rate > 0.0:
    if isinstance(dropout_seed, torch.Tensor):
      seed_t = dropout_seed.to(device=q.device, dtype=torch.int32)
    else:
      seed_t = _seed_words(int(dropout_seed), q.shape[0], q.device)
  else:
    seed_t = None
  if segment_ids is not None:
    if was_flat:  # TT-UNFLATTEN-SHIM: a flat [T] stream's ids become [B, S] too.
      segment_ids = unflatten_tensor(segment_ids, expected_flat_ndim=1)
    if segment_ids.shape != (q.shape[0], q.shape[s_axis]):
      raise ValueError(
          f"segment_ids must be (B, S) = {(q.shape[0], q.shape[s_axis])}, got"
          f" {tuple(segment_ids.shape)}"
      )
    segment_ids = segment_ids.to(device=q.device, dtype=torch.int32).contiguous()
  scale_val = 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
  window_val = -1 if local_window_size is None else int(local_window_size)
  out, _ = _sbd_fwd(
      q,
      k,
      v,
      bias,
      seed_t,
      segment_ids,
      is_causal,
      scale_val,
      window_val,
      heads_first,
      *SbdConfig.static_args(),
  )
  # TT-UNFLATTEN-SHIM:
  if not was_flat:
    return out
  return flatten_tensor(out, original_flat_ndim=3)
