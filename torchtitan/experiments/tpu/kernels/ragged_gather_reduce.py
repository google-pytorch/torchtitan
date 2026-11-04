# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""TensorCore (Pallas DMA) ragged gather-reduce for static-shape MoE combine.

`ragged_gather_reduce` optionally scales rows as `bf16(f32(x) * w)` and sums
the at most `top_k` rows of each output token in FP32, returning the
`[num_out_tokens, D // 128, 128]` token layout of `ragged_gather.to_token_3d`
so a following reduce-scatter can run before the relayout back to `[T, D]`.
`ragged_gather_reduce_metadata` computes the per-dispatch routing tables once.
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

from torchtitan.experiments.tpu.kernels import ragged_gather


_RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS = 4
_RAGGED_GATHER_REDUCE_TC_LOOKAHEAD = _RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS - 2
_RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL = 64
_RAGGED_GATHER_REDUCE_TC_REDUCE_UNROLL = 32
_RAGGED_GATHER_REDUCE_TC_ALIGN = 128
_RAGGED_GATHER_REDUCE_TC_SEARCH_CHUNK = 64
_RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS = 4
_RAGGED_GATHER_REDUCE_TC_VMEM_LIMIT = 64 * 1024 * 1024

# Rows per grid step of the gather-reduce kernel.
REDUCE_BLOCK_SIZE = 1792


def ragged_gather_reduce_tc_layout(
    num_indices: int,
    top_k: int,
    block_size: int,
    num_out_tokens: int,
) -> tuple[int, int, int]:
  """Returns (max_blocks, window, slots_len) of the metadata layout."""
  align = _RAGGED_GATHER_REDUCE_TC_ALIGN
  max_blocks = pl.cdiv(num_indices + num_out_tokens, block_size - top_k)
  window = pl.cdiv(block_size, align) * align + align
  slots_len = pl.cdiv(max(num_indices, window), align) * align
  return max_blocks, window, slots_len


def _packed_stable_sort(
    key: jax.Array, max_key: int
) -> tuple[jax.Array, jax.Array]:
  """Stable sort of `arange(len(key))` by non-negative int `key` <= max_key."""
  n = key.shape[0]
  positions = jnp.arange(n, dtype=jnp.int32)
  pos_bits = max(1, (n - 1).bit_length())
  key_bits = max(1, int(max_key).bit_length())
  if pos_bits + key_bits > 32:
    sorted_key, sorted_positions = jax.lax.sort(
        (key.astype(jnp.int32), positions), num_keys=1, is_stable=True
    )
    return sorted_key, sorted_positions
  shift = jnp.uint32(pos_bits)
  packed_key = (key.astype(jnp.uint32) << shift) | positions.astype(jnp.uint32)
  sorted_packed = jax.lax.sort(packed_key, is_stable=False)
  sorted_positions = (
      sorted_packed & jnp.uint32((1 << pos_bits) - 1)
  ).astype(jnp.int32)
  return (sorted_packed >> shift).astype(jnp.int32), sorted_positions


@pallas.jax_op("tt_moe::ragged_gather_reduce_tc_meta")
def _ragged_gather_reduce_tc_meta_tpu(
    token_ids_R: jax.Array,
    valid_count_1: jax.Array,
    top_k: int,
    block_size: int,
    num_out_tokens: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Compute `(src_rows, tokens, blocks)` without emitting HloOpcode::kScan."""
  num_indices = token_ids_R.shape[0]
  max_blocks, _, slots_len = ragged_gather_reduce_tc_layout(
      num_indices, top_k, block_size, num_out_tokens=num_out_tokens
  )
  if num_out_tokens == 0 or num_indices == 0:
    empty_slots = jnp.zeros((slots_len,), jnp.int32)
    return (
        empty_slots,
        empty_slots,
        jnp.zeros(
            (1 + _RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS * max_blocks,),
            jnp.int32,
        ),
    )

  num_tokens = valid_count_1[0].astype(jnp.int32)
  positions = jnp.arange(num_indices, dtype=jnp.int32)
  valid = positions < num_tokens
  key = jnp.where(valid, token_ids_R.astype(jnp.int32), num_out_tokens).astype(
      jnp.int32
  )
  sorted_key, sorted_positions = _packed_stable_sort(key, num_out_tokens)
  pad = slots_len - num_indices
  src_rows = jnp.pad(sorted_positions, (0, pad))
  tokens = jnp.pad(sorted_key, (0, pad))

  is_run_start = jnp.concatenate(
      [jnp.ones((1,), bool), sorted_key[1:] != sorted_key[:-1]]
  )
  # Each valid token `t < num_out_tokens` appears at most `top_k` times in
  # `sorted_key`. Using a bounded window max of width >= top_k avoids
  # `jax.lax.cummax` (`HloOpcode::kScan`), which triggers an XLA
  # BFloat16Propagation assertion inside scanned layer loops (`while_body`).
  run_start = jnp.where(is_run_start, positions, jnp.int32(0))
  step = 1
  while step < top_k:
    run_start = jnp.concatenate([
        run_start[:step],
        jnp.maximum(run_start[step:], run_start[:-step]),
    ])
    step *= 2

  int_max = jnp.iinfo(jnp.int32).max
  slot_cost = jnp.where(
      sorted_key < num_out_tokens, run_start + sorted_key, int_max
  )
  chunk = _RAGGED_GATHER_REDUCE_TC_SEARCH_CHUNK
  num_chunks = pl.cdiv(num_indices, chunk)
  cost_chunks = jnp.pad(
      slot_cost,
      (0, num_chunks * chunk - num_indices),
      constant_values=int_max,
  ).reshape(num_chunks, chunk)
  threshold = jnp.arange(max_blocks + 1, dtype=jnp.int32) * (
      block_size - top_k
  )
  full_chunks = jnp.sum(
      cost_chunks[:, -1] < threshold[:, None], axis=1, dtype=jnp.int32
  )
  row = jnp.minimum(full_chunks, num_chunks - 1)
  in_chunk = jnp.sum(
      cost_chunks[row] < threshold[:, None], axis=1, dtype=jnp.int32
  )
  slot_start = row * chunk + in_chunk
  prev_key = jnp.where(
      slot_start > 0, sorted_key[jnp.maximum(slot_start - 1, 0)], -1
  )
  next_key = jnp.where(
      slot_start < num_indices,
      sorted_key[jnp.minimum(slot_start, num_indices - 1)],
      num_out_tokens,
  )
  token_start = jnp.clip(threshold - slot_start, prev_key + 1, next_key)
  table = jnp.stack(
      [
          slot_start[:-1],
          jnp.diff(slot_start),
          token_start[:-1],
          jnp.diff(token_start),
      ],
      axis=1,
  )
  num_blocks = jnp.sum(token_start[:-1] < num_out_tokens, dtype=jnp.int32)
  blocks = jnp.concatenate([num_blocks[None], table.reshape(-1)]).astype(
      jnp.int32
  )
  return src_rows, tokens, blocks


def _ragged_gather_reduce_tc_core(
    x: jax.Array,
    src_rows: jax.Array,
    tokens: jax.Array,
    blocks: jax.Array,
    num_out_tokens: int,
    top_k: int,
    block_size: int,
    num_indices: int,
) -> jax.Array:
  """Gather-reduce kernel on a `[R, D // 128, 128]` buffer -> 3D tokens."""
  _, dim0, dim1 = x.shape
  num_buffers = _RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS
  align = _RAGGED_GATHER_REDUCE_TC_ALIGN
  fields = _RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS
  _, window, slots_len = ragged_gather_reduce_tc_layout(
      num_indices, top_k, block_size, num_out_tokens=num_out_tokens
  )
  lookahead = _RAGGED_GATHER_REDUCE_TC_LOOKAHEAD
  ring = lookahead + 2

  def _ragged_gather_reduce_kernel(
      blocks_smem_ref,
      src_hbm_ref,
      tok_hbm_ref,
      x_hbm_ref,
      o_hbm_ref,
      vmem_ref,
      src_smem_ref,
      tok_smem_ref,
      recv_sem,
      send_sem,
      slots_sem,
  ):
    num_blocks = blocks_smem_ref[0]

    def _block(blk):
      entry = 1 + fields * blk
      return tuple(blocks_smem_ref[entry + f] for f in range(fields))

    def _window_start(slot_start):
      return jnp.minimum((slot_start // align) * align, slots_len - window)

    def _slots_loads(blk):
      r = blk % ring
      window_start = pl.multiple_of(_window_start(_block(blk)[0]), align)
      return [
          pltpu.make_async_copy(
              hbm_ref.at[pl.ds(window_start, window)],
              smem_ref.at[pl.ds(r * window, window)],
              slots_sem.at[r],
          )
          for hbm_ref, smem_ref in (
              (src_hbm_ref, src_smem_ref),
              (tok_hbm_ref, tok_smem_ref),
          )
      ]

    def _slot_base(blk):
      slot_start = _block(blk)[0]
      return (blk % ring) * window + slot_start - _window_start(slot_start)

    def _pow2_chunks(n, fn):
      off = 0
      for k in reversed(range(block_size.bit_length())):
        size = 1 << k
        bit = n & size

        def _fn_if_bit_set(off=off, size=size):
          fn(off, size)

        pl.when(bit != 0)(_fn_if_bit_set)
        off = off + bit

    def _wait_rows(b, n, sem):
      def _wait(off, size):
        del off
        rows = vmem_ref.at[b, pl.ds(0, size), :, :]
        pltpu.make_async_copy(rows, rows, sem.at[b]).wait()

      _pow2_chunks(n, _wait)

    def _dynamic_fori(lo, hi, unroll, body, carry):
      def _chunk(c, carry):
        for u in range(unroll):
          carry = body(lo + c * unroll + u, carry)
        return carry

      num_full = (hi - lo) // unroll
      carry = jax.lax.fori_loop(0, num_full, _chunk, carry)
      return jax.lax.fori_loop(lo + num_full * unroll, hi, body, carry)

    def _issuer(blk):
      b = blk % num_buffers
      num_tokens = _block(blk)[3]
      base = _slot_base(blk)

      def _issue(i, carry):
        pltpu.make_async_copy(
            x_hbm_ref.at[pl.ds(src_smem_ref[base + i], 1), :, :],
            vmem_ref.at[b, pl.ds(num_tokens + i, 1), :, :],
            recv_sem.at[b],
        ).start()
        return carry

      return _issue

    def _in_start(blk):
      _dynamic_fori(
          0,
          _block(blk)[1],
          _RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL,
          _issuer(blk),
          (),
      )

    def _in_wait(blk):
      _wait_rows(blk % num_buffers, _block(blk)[1], recv_sem)

    def _reducer(blk):
      b = blk % num_buffers
      _, _, token_start, num_tokens = _block(blk)

      def _zero_tokens(off, size):
        vmem_ref[b, pl.ds(off, size), :, :] = jnp.zeros(
            (size, dim0, dim1), x.dtype
        )

      _pow2_chunks(num_tokens, _zero_tokens)

      base = _slot_base(blk)

      def _reduce(i, carry):
        prev_t, total = carry
        t = tok_smem_ref[base + i] - token_start
        row = vmem_ref[b, num_tokens + i, :, :].astype(jnp.float32)
        total = jnp.where(t == prev_t, total, 0.0) + row
        vmem_ref[b, t, :, :] = total.astype(x.dtype)
        return t, total

      return _reduce

    def _out_start(blk):
      b = blk % num_buffers
      _, _, token_start, num_tokens = _block(blk)

      def _store(off, size):
        pltpu.make_async_copy(
            vmem_ref.at[b, pl.ds(off, size), :, :],
            o_hbm_ref.at[pl.ds(token_start + off, size), :, :],
            send_sem.at[b],
        ).start()

      _pow2_chunks(num_tokens, _store)

    def _out_wait(blk):
      _wait_rows(blk % num_buffers, _block(blk)[3], send_sem)

    def _slots_wait(blk):
      for load in _slots_loads(blk):
        load.wait()

    for blk in range(ring):

      @pl.when(blk < num_blocks)
      def _load_first_slots(blk=blk):
        for load in _slots_loads(blk):
          load.start()

    for blk in range(lookahead):

      @pl.when(blk < num_blocks)
      def _start_first_rows(blk=blk):
        _slots_wait(blk)
        _in_start(blk)

    @pl.loop(0, num_blocks)
    def _step(i):
      _in_wait(i)
      nxt = i + lookahead
      has_nxt = nxt < num_blocks

      @pl.when(nxt >= num_buffers)
      def _free_buffer():
        _out_wait(nxt - num_buffers)

      @pl.when(has_nxt)
      def _wait_next_slots():
        _slots_wait(nxt)

      nxt = jnp.minimum(nxt, num_blocks - 1)
      num_reduce = _block(i)[1]
      num_issue = jnp.where(has_nxt, _block(nxt)[1], 0)
      num_both = jnp.minimum(num_reduce, num_issue)
      issue = _issuer(nxt)
      reduce = _reducer(i)

      def _issue_and_reduce(j, carry):
        issue(j, ())
        return reduce(j, carry)

      unroll = _RAGGED_GATHER_REDUCE_TC_REDUCE_UNROLL
      carry = (jnp.int32(-1), jnp.zeros((dim0, dim1), jnp.float32))
      carry = _dynamic_fori(0, num_both, unroll, _issue_and_reduce, carry)
      _dynamic_fori(num_both, num_reduce, unroll, reduce, carry)
      _dynamic_fori(
          num_both,
          num_issue,
          _RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL,
          issue,
          (),
      )
      _out_start(i)

      @pl.when(i + ring < num_blocks)
      def _load_later_slots():
        for load in _slots_loads(i + ring):
          load.start()

    for k in range(num_buffers - lookahead):

      @pl.when(num_blocks - 1 - k >= 0)
      def _wait_last_stores(k=k):
        _out_wait(num_blocks - 1 - k)

  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  out_shape = jax.ShapeDtypeStruct(
      (num_out_tokens, dim0, dim1), x.dtype, manual_axis_type=manual_axis_type
  )
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  dma_sems = pltpu.SemaphoreType.DMA((num_buffers,))
  out_3d = pl.pallas_call(
      _ragged_gather_reduce_kernel,
      out_shape=out_shape,
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          hbm_spec,
          hbm_spec,
          hbm_spec,
      ],
      out_specs=hbm_spec,
      scratch_shapes=[
          pltpu.VMEM((num_buffers, block_size, dim0, dim1), x.dtype),
          pltpu.SMEM((ring * window,), jnp.int32),
          pltpu.SMEM((ring * window,), jnp.int32),
          dma_sems,
          dma_sems,
          pltpu.SemaphoreType.DMA((ring,)),
      ],
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=_RAGGED_GATHER_REDUCE_TC_VMEM_LIMIT,
      ),
      name="ragged_gather_reduce_tc",
  )(blocks, src_rows, tokens, x)
  return out_3d


_SCALE_UNFLATTEN_BLOCK = 1024


def _scale_unflatten_tc(
    x_RD: jax.Array,
    scale_R1: jax.Array | None,
    num_rows: jax.Array,
    block: int = _SCALE_UNFLATTEN_BLOCK,
) -> jax.Array:
  """`(R, D) -> (R, D // 128, 128)` over the first `num_rows` rows.

  With `scale_R1`, rows are scaled as `bf16(f32(x) * scale)` in the same pass,
  matching `(x.float() * scale).to(x.dtype)` exactly. Rows past `num_rows` are
  left unwritten; the gather-reduce kernel only reads valid rows. (XLA would
  lower the reshape to a full-size layout copy and the scale to another pass.)
  """
  rows, d = x_RD.shape
  block = min(block, rows)
  n = jnp.reshape(jnp.asarray(num_rows, jnp.int32), (1,))
  has_scale = scale_R1 is not None

  def kernel(n_ref, x_ref, *refs):
    del n_ref
    if has_scale:
      s_ref, o_ref = refs
      y = (x_ref[...].astype(jnp.float32) * s_ref[...]).astype(x_ref.dtype)
    else:
      (o_ref,) = refs
      y = x_ref[...]
    o_ref[...] = y.reshape(o_ref.shape)

  in_specs = [pl.BlockSpec((block, d), lambda i, n_ref: (i, 0))]
  operands = [x_RD]
  if scale_R1 is not None:
    in_specs.append(pl.BlockSpec((block, 1), lambda i, n_ref: (i, 0)))
    operands.append(scale_R1.astype(jnp.float32))
  manual_axis_type = getattr(jax.typeof(x_RD), "manual_axis_type", None)
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct(
          (rows, d // 128, 128), x_RD.dtype, manual_axis_type=manual_axis_type
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(pl.cdiv(n[0], block),),
          in_specs=in_specs,
          out_specs=pl.BlockSpec(
              (block, d // 128, 128), lambda i, n_ref: (i, 0, 0)
          ),
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=_RAGGED_GATHER_REDUCE_TC_VMEM_LIMIT,
      ),
      name="ragged_scale_unflatten_tc",
  )(n, *operands)


@pallas.jax_op("tt_moe::ragged_gather_reduce_tc")
def _ragged_gather_reduce_tc_tpu(
    x_RD: jax.Array,
    scale_R1: jax.Array,
    valid_count_1: jax.Array,
    src_rows: jax.Array,
    tokens: jax.Array,
    blocks: jax.Array,
    num_out_tokens: int,
    top_k: int,
    block_size: int,
    has_scale: int,
) -> jax.Array:
  """Optionally scale rows, then gather-reduce into `[num_out, D // 128, 128]`."""
  x_R3 = _scale_unflatten_tc(
      x_RD, scale_R1 if has_scale else None, valid_count_1[0]
  )
  return _ragged_gather_reduce_tc_core(
      x_R3,
      src_rows,
      tokens,
      blocks,
      num_out_tokens,
      top_k,
      block_size,
      int(x_RD.shape[0]),
  )


# ---------------------------------------------------------------------------
# Public torch API. Each op has a TPU path (the Pallas kernels above) and an
# eager reference path used on other devices and as the test oracle.
# ---------------------------------------------------------------------------


def ragged_gather_reduce_metadata(
    token_ids_R: torch.Tensor,
    valid_count_1: torch.Tensor,
    *,
    top_k: int,
    num_out_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Routing metadata `(src_rows, tokens, blocks)` for `ragged_gather_reduce`.

  Each valid row `i < valid_count` contributes to token `token_ids_R[i]`, and a
  token receives at most `top_k` rows. Compute once per dispatch and reuse in
  the combine forward and the dispatch backward.
  """
  if token_ids_R.ndim != 1 or valid_count_1.numel() != 1:
    raise ValueError(
        "ragged_gather_reduce_metadata expects token_ids_R[R] and"
        " valid_count_1[1]"
    )
  if token_ids_R.device.type == "tpu":
    return _ragged_gather_reduce_tc_meta_tpu(
        token_ids_R.to(torch.int32),
        valid_count_1.reshape(1).to(torch.int32),
        int(top_k),
        REDUCE_BLOCK_SIZE,
        int(num_out_tokens),
    )
  # The reference path only needs (src_rows, tokens); `blocks` is a TPU
  # scheduling table, so return a correctly sized placeholder.
  num_indices = int(token_ids_R.shape[0])
  max_blocks, _, slots_len = ragged_gather_reduce_tc_layout(
      num_indices, int(top_k), REDUCE_BLOCK_SIZE, num_out_tokens=num_out_tokens
  )
  positions = torch.arange(num_indices, device=token_ids_R.device)
  valid = positions < valid_count_1.reshape(1)
  key = torch.where(valid, token_ids_R.long(), int(num_out_tokens))
  perm = torch.argsort(key, stable=True)
  pad = slots_len - num_indices
  src_rows = torch.nn.functional.pad(perm.to(torch.int32), (0, pad))
  tokens = torch.nn.functional.pad(key[perm].to(torch.int32), (0, pad))
  blocks = torch.zeros(
      1 + _RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS * max_blocks,
      device=token_ids_R.device,
      dtype=torch.int32,
  )
  return src_rows, tokens, blocks


def ragged_gather_reduce(
    x_RD: torch.Tensor,
    scale_R: torch.Tensor | None,
    valid_count_1: torch.Tensor,
    metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_out_tokens: int,
    top_k: int,
) -> torch.Tensor:
  """`out[t] = sum_{i < valid_count, token(i) == t} w_i * x[i]`, in FP32.

  Args:
    x_RD: `[R, D]` rows (e.g. expert outputs).
    scale_R: optional `[R]` FP32 row weights, applied as
      `(x.float() * w).to(x.dtype)` before the sum (bit-identical to doing it
      outside, but fused into the relayout pass on TPU).
    valid_count_1: `[1]` number of leading valid rows.
    metadata: output of `ragged_gather_reduce_metadata`.
    num_out_tokens: number of output tokens.
    top_k: maximum rows per token.

  Returns:
    `[num_out_tokens, D // 128, 128]` sums in `x_RD.dtype`, in the kernels'
    token layout so a following reduce-scatter can run before the relayout.
  """
  if x_RD.ndim != 2:
    raise ValueError("ragged_gather_reduce expects x_RD[R, D]")
  d = int(x_RD.shape[1])
  ragged_gather._check_hidden(d)  # pylint: disable=protected-access
  src_rows, tokens, blocks = metadata
  if x_RD.device.type == "tpu":
    has_scale = scale_R is not None
    scale_R1 = (
        scale_R.reshape(-1, 1).to(torch.float32)
        if scale_R is not None
        else x_RD.new_zeros((1, 1), dtype=torch.float32)
    )
    return _ragged_gather_reduce_tc_tpu(
        x_RD,
        scale_R1,
        valid_count_1.reshape(1).to(torch.int32),
        src_rows.to(torch.int32),
        tokens.to(torch.int32),
        blocks.to(torch.int32),
        int(num_out_tokens),
        int(top_k),
        REDUCE_BLOCK_SIZE,
        int(has_scale),
    )
  if scale_R is not None:
    x_RD = (x_RD.float() * scale_R.reshape(-1, 1)).to(x_RD.dtype)
  num_indices = int(x_RD.shape[0])
  src = src_rows[:num_indices].long()
  tok = tokens[:num_indices].long()
  valid = tok < int(num_out_tokens)
  rows = torch.index_select(x_RD.float(), 0, torch.where(valid, src, 0))
  rows = torch.where(valid.unsqueeze(-1), rows, torch.zeros_like(rows))
  out = x_RD.new_zeros((int(num_out_tokens), d), dtype=torch.float32)
  out.index_add_(0, torch.where(valid, tok, 0), rows)
  return ragged_gather.to_token_3d(out.to(x_RD.dtype))


__all__ = [
    "REDUCE_BLOCK_SIZE",
    "ragged_gather_reduce",
    "ragged_gather_reduce_metadata",
]
