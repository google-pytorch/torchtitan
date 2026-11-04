# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications: Bridged to PyTorch/TorchTPU via @pallas.jax_op and added
# weighted gather-reduce, collision-free row-scale gather/scatter, and
# _ragged_scale_and_dscore_tc backward kernels.

"""TensorCore DMA-driven ragged gather / gather-reduce kernels for MoE Ring-of-Experts.

Adapted from MaxText's ``kernels/ragged/ragged_gather_tc.py`` and
``kernels/ragged/ragged_sort_tc.py`` (MLPerf Training v6.1 Ironwood TPU7x MoE
submissions) and bridged to PyTorch/TorchTPU via ``@pallas.jax_op``:

- ``packed_stable_sort`` packs ``(key << pos_bits) | position`` into a single
  ``uint32`` array whenever ``pos_bits + key_bits <= 32`` so an unstable single-
  array sort yields the exact stable expert order.
- ``ragged_gather_tc`` gathers ``out[i] = x[indices[start + i]]`` for
  ``i < count`` into a static ``[cap, d0, d1]`` buffer using a dynamic Pallas
  grid ``(cdiv(count, block_size),)`` so HBM reads and writes scale strictly
  with ``count`` (~``1x`` local tokens) rather than worst-case ``cap``
  (``ep_size * N``).
- ``ragged_gather_reduce_tc`` compacts valid slots per output token, optionally
  multiplies each gathered row in VMEM by its FP32 routing weight, and sums in
  FP32 in deterministic buffer order without touching rows ``>= count``.
- ``_ragged_scale_and_dscore_tc`` runs a dynamic-grid ``(cdiv(count, 1024),
d_chunks)``
  backward pass that simultaneously scales ``grad_routed_output`` by routing
  weights and computes the FP32 ``dScore`` dot product while bitwise-masking
  any uninitialized tail rows ``>= count``.
"""

import dataclasses
import functools
import math
import re

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

_RAGGED_GATHER_TC_NUM_BUFFERS = 2

_RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS = 4
_RAGGED_GATHER_REDUCE_TC_LOOKAHEAD = _RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS - 2
_RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL = 64
_RAGGED_GATHER_REDUCE_TC_REDUCE_UNROLL = 8
_RAGGED_GATHER_REDUCE_TC_ALIGN = 128
_RAGGED_GATHER_REDUCE_TC_SEARCH_CHUNK = 64
_RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS = 4


@torch._dynamo.assume_constant_result
@functools.lru_cache(maxsize=1)
def _tpu_generation() -> int | None:
  """Returns the TPU generation (e.g., 5, 6, 7), or ``None`` if unavailable."""
  get_props = getattr(
      getattr(torch, "tpu", None), "get_device_properties", None
  )
  if get_props is None:
    return None
  try:
    name = get_props().name.lower()
  except RuntimeError:
    return None
  match = re.search(r"(?:v|tpu[- ]?)(\d+)", name)
  return int(match.group(1)) if match else None


def _tpu_vmem_limit_bytes() -> int:
  """Returns the per-kernel VMEM limit in bytes for the active TPU generation."""
  gen = _tpu_generation()
  if gen is not None and gen >= 7:
    return 64 * 1024 * 1024
  if gen == 6:
    return 32 * 1024 * 1024
  return 16 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class RaggedGatherReduceMetadata:
  """Compacted slot routing and block table for ``ragged_gather_reduce_tc``."""

  src_rows: jax.Array
  tokens: jax.Array
  blocks: jax.Array
  block_size: int
  num_indices: int


def _token_vmem_bytes(hidden_dim: int, itemsize: int) -> int:
  dim0 = hidden_dim // 128
  dim1 = 128
  sublanes = 8 * max(1, 4 // itemsize)
  return (
      pl.cdiv(dim0, sublanes) * sublanes * pl.cdiv(dim1, 128) * 128 * itemsize
  )


def safe_gather_block_size(
    hidden_dim: int,
    itemsize: int,
    requested_block_size: int = 1024,
) -> int:
  """Clamps ``gather_block_size`` so 2 VMEM buffers fit within generation VMEM."""
  vmem_budget = int(0.875 * _tpu_vmem_limit_bytes())
  token_bytes = _token_vmem_bytes(hidden_dim, itemsize)
  max_fit = vmem_budget // (_RAGGED_GATHER_TC_NUM_BUFFERS * token_bytes)
  if max_fit >= 128:
    max_fit = (max_fit // 128) * 128
  return max(1, min(requested_block_size, max_fit))


def safe_reduce_block_size(
    hidden_dim: int,
    itemsize: int,
    top_k: int,
    requested_block_size: int = 896,
) -> int:
  """Clamps ``reduce_block_size`` so 4 VMEM buffers fit within generation VMEM."""
  vmem_budget = int(0.875 * _tpu_vmem_limit_bytes())
  token_bytes = _token_vmem_bytes(hidden_dim, itemsize)
  max_fit = vmem_budget // (_RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS * token_bytes)
  if max_fit >= 128:
    max_fit = (max_fit // 128) * 128
  return max(top_k + 1, min(requested_block_size, max_fit))


def _ragged_gather_reduce_tc_layout(
    num_indices: int,
    top_k: int,
    block_size: int,
    num_out_tokens: int | None = None,
) -> tuple[int, int, int]:
  if num_out_tokens is None:
    num_out_tokens = num_indices // top_k
  align = _RAGGED_GATHER_REDUCE_TC_ALIGN
  max_blocks = pl.cdiv(num_indices + num_out_tokens, block_size - top_k)
  window = pl.cdiv(block_size, align) * align + align
  slots_len = pl.cdiv(max(num_indices, window), align) * align
  return max_blocks, window, slots_len


def packed_stable_sort(
    key: jax.Array, max_key: int
) -> tuple[jax.Array, jax.Array]:
  """Stable sort of ``arange(len(key))`` by non-negative ``key <= max_key``."""
  n = key.shape[0]
  positions = jnp.arange(n, dtype=jnp.int32)
  if n <= 1:
    return key.astype(jnp.int32), positions
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
  sorted_positions = (sorted_packed & jnp.uint32((1 << pos_bits) - 1)).astype(
      jnp.int32
  )
  return (sorted_packed >> shift).astype(jnp.int32), sorted_positions


def _sorted_expert_offsets(
    sorted_experts: jax.Array, num_experts: int
) -> jax.Array:
  """Finds ``searchsorted(sorted_experts, 0..E, side='left')`` in ``O(E log M)``."""
  m = sorted_experts.shape[0]
  queries = jnp.arange(num_experts + 1, dtype=jnp.int32)
  if m == 0:
    return jnp.zeros_like(queries)
  lo = jnp.zeros_like(queries)
  hi = jnp.full_like(queries, m)
  for _ in range(max(1, m.bit_length())):
    mid = (lo + hi) >> 1
    in_bounds = mid < m
    val = sorted_experts[jnp.minimum(mid, m - 1)]
    go_right = in_bounds & (val < queries)
    lo = jnp.where(go_right, mid + 1, lo)
    hi = jnp.where(go_right, hi, mid)
  return lo


def ragged_gather_tc(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array | int,
    num_tokens: jax.Array | int,
    max_out_tokens: int,
    block_size: int = 1024,
) -> jax.Array:
  """Gathers ``out[i] = x[indices[start + i]]`` for ``i < num_tokens`` on TC."""
  if max_out_tokens <= 0:
    raise ValueError(f"max_out_tokens must be positive, got {max_out_tokens}.")
  num_indices = indices.shape[0]
  _, dim0, dim1 = x.shape
  if num_indices == 0 or (isinstance(num_tokens, int) and num_tokens == 0):
    return jnp.empty((max_out_tokens, dim0, dim1), dtype=x.dtype)

  block_size = min(block_size, max_out_tokens)
  num_buffers = _RAGGED_GATHER_TC_NUM_BUFFERS
  num_full_blocks = max_out_tokens // block_size
  rem = max_out_tokens % block_size
  num_blocks = num_full_blocks + (1 if rem > 0 else 0)

  idx_align = 128
  idx_window = pl.cdiv(block_size, idx_align) * idx_align + idx_align
  idx_pad = max((-num_indices) % idx_align, idx_window - num_indices)
  if idx_pad > 0:
    indices = jnp.pad(indices, (0, idx_pad))
  max_win_start = indices.shape[0] - idx_window

  def _ragged_gather_kernel(
      scalars_smem_ref,
      idx_hbm_ref,
      x_hbm_ref,
      o_hbm_ref,
      vmem_ref,
      idx_smem_ref,
      idx_sem,
      data_recv_sem,
      data_send_sem,
  ):
    block_idx = pl.program_id(0)
    buf_idx = block_idx % num_buffers
    start_idx = scalars_smem_ref[0]
    num_valid = scalars_smem_ref[1]

    def _run_step(step_idx, b_idx, action: str):
      base = step_idx * block_size
      row0 = start_idx + base
      win_start = jnp.minimum((row0 // idx_align) * idx_align, max_win_start)
      idx_off = row0 - win_start

      def _fetch_indices():
        copy = pltpu.make_async_copy(
            idx_hbm_ref.at[
                pl.ds(pl.multiple_of(win_start, idx_align), idx_window)
            ],
            idx_smem_ref,
            idx_sem,
        )
        copy.start()
        copy.wait()

      def _chunk(off, size: int, chunk_action: str):
        if chunk_action == "data_start":
          step_unroll = math.gcd(size, 128)

          @pl.loop(0, size // step_unroll)
          def _(chunk_i):
            base_i = off + chunk_i * step_unroll
            src_indices = [
                idx_smem_ref[idx_off + base_i + u] for u in range(step_unroll)
            ]
            for u in range(step_unroll):
              pltpu.make_async_copy(
                  x_hbm_ref.at[pl.ds(src_indices[u], 1), :, :],
                  vmem_ref.at[b_idx, pl.ds(base_i + u, 1), :, :],
                  data_recv_sem.at[b_idx],
              ).start()

        elif chunk_action == "recv_wait":
          pltpu.make_async_copy(
              o_hbm_ref.at[pl.ds(0, size), :, :],
              vmem_ref.at[b_idx, pl.ds(0, size), :, :],
              data_recv_sem.at[b_idx],
          ).wait()
        else:
          copy = pltpu.make_async_copy(
              vmem_ref.at[b_idx, pl.ds(off, size), :, :],
              o_hbm_ref.at[pl.ds(base + off, size), :, :],
              data_send_sem.at[b_idx],
          )
          if chunk_action == "send_start":
            copy.start()
          else:
            copy.wait()

      def _do_len(length: int):
        step_valid = jnp.minimum(jnp.maximum(num_valid - base, 0), length)

        def _over_valid_rows(chunk_action: str):
          @pl.when(step_valid == length)
          def _():
            _chunk(0, length, chunk_action)

          @pl.when(step_valid < length)
          def _():
            off = 0
            for k in reversed(range(length.bit_length())):
              size = 1 << k
              bit = step_valid & size

              def _chunk_if_bit_set(off=off, size=size):
                _chunk(off, size, chunk_action)

              pl.when(bit != 0)(_chunk_if_bit_set)
              off = off + bit

        if action == "data_start":

          @pl.when(step_valid > 0)
          def _():
            _fetch_indices()

          _over_valid_rows("data_start")
        elif action == "copy_out_start":
          _over_valid_rows("recv_wait")
          _over_valid_rows("send_start")
        elif action == "copy_out_wait":
          _over_valid_rows("send_wait")

      if rem == 0:
        _do_len(block_size)
      elif isinstance(step_idx, int):
        _do_len(rem if step_idx == num_blocks - 1 else block_size)
      else:
        jax.lax.cond(
            step_idx == num_blocks - 1,
            lambda: _do_len(rem),
            lambda: _do_len(block_size),
        )

    @pl.when(block_idx == 0)
    def _prologue():
      _run_step(0, 0, "data_start")

    _run_step(block_idx, buf_idx, "copy_out_start")

    num_steps = pl.num_programs(0)
    next_data_block = block_idx + 1

    @pl.when(next_data_block < num_steps)
    def _queue_next():
      next_data_buf = next_data_block % num_buffers

      @pl.when(block_idx >= 1)
      def _wait_prev_out():
        _run_step(block_idx - 1, next_data_buf, "copy_out_wait")

      _run_step(next_data_block, next_data_buf, "data_start")

    @pl.when(block_idx == num_steps - 1)
    def _epilogue():
      @pl.when(block_idx >= 1)
      def _wait_prev_out():
        prev_block = block_idx - 1
        _run_step(prev_block, prev_block % num_buffers, "copy_out_wait")

      _run_step(block_idx, buf_idx, "copy_out_wait")

  out_shape = jax.ShapeDtypeStruct((max_out_tokens, dim0, dim1), x.dtype)
  scalars = jnp.stack([
      jnp.asarray(start, dtype=jnp.int32),
      jnp.asarray(num_tokens, dtype=jnp.int32),
  ])
  num_steps = pl.cdiv(
      num_tokens if isinstance(num_tokens, int) else scalars[1], block_size
  )
  return pl.pallas_call(
      _ragged_gather_kernel,
      out_shape=out_shape,
      grid=(num_steps,),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
      scratch_shapes=[
          pltpu.VMEM((num_buffers, block_size, dim0, dim1), x.dtype),
          pltpu.SMEM((idx_window,), indices.dtype),
          pltpu.SemaphoreType.DMA(()),
          pltpu.SemaphoreType.DMA((num_buffers,)),
          pltpu.SemaphoreType.DMA((num_buffers,)),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(pltpu.ARBITRARY,),
          vmem_limit_bytes=_tpu_vmem_limit_bytes(),
      ),
      name="ragged_gather_tc",
  )(scalars, indices, x)


def ragged_gather_reduce_tc_metadata(
    indices: jax.Array,
    start: jax.Array | int,
    num_tokens: jax.Array | int,
    *,
    top_k: int,
    block_size: int = 896,
    num_out_tokens: int | None = None,
    packed_sort: bool = True,
) -> RaggedGatherReduceMetadata:
  """Derives compacted slot order and VMEM block table for ``ragged_gather_reduce_tc``."""
  num_indices = indices.shape[0]
  if num_out_tokens is None:
    num_out_tokens = num_indices // top_k
  max_blocks, _, slots_len = _ragged_gather_reduce_tc_layout(
      num_indices, top_k, block_size, num_out_tokens=num_out_tokens
  )
  if num_out_tokens == 0:
    empty_slots = jnp.zeros((slots_len,), jnp.int32)
    return RaggedGatherReduceMetadata(
        src_rows=empty_slots,
        tokens=empty_slots,
        blocks=jnp.zeros((1 + 4 * max_blocks,), jnp.int32),
        block_size=block_size,
        num_indices=num_indices,
    )

  start = jnp.asarray(start, dtype=jnp.int32)
  num_tokens = jnp.asarray(num_tokens, dtype=jnp.int32)
  positions = jnp.arange(num_indices, dtype=jnp.int32)
  rel = positions - start
  valid = (rel >= 0) & (rel < num_tokens)
  key = jnp.where(valid, indices, num_out_tokens).astype(jnp.int32)
  if packed_sort:
    sorted_key, sorted_positions = packed_stable_sort(key, num_out_tokens)
  else:
    sorted_key, sorted_positions = jax.lax.sort(
        (key, positions), num_keys=1, is_stable=True
    )
  pad = slots_len - num_indices
  src_rows = jnp.pad(sorted_positions - start, (0, pad))
  tokens = jnp.pad(sorted_key, (0, pad))

  is_run_start = jnp.concatenate(
      [jnp.ones((1,), bool), sorted_key[1:] != sorted_key[:-1]]
  )
  run_start = jax.lax.cummax(jnp.where(is_run_start, positions, 0))
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
  threshold = jnp.arange(max_blocks + 1, dtype=jnp.int32) * (block_size - top_k)
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
  blocks = jnp.concatenate([num_blocks[None], table.reshape(-1)])
  return RaggedGatherReduceMetadata(
      src_rows=src_rows.astype(jnp.int32),
      tokens=tokens.astype(jnp.int32),
      blocks=blocks.astype(jnp.int32),
      block_size=block_size,
      num_indices=num_indices,
  )


def ragged_gather_reduce_tc(
    x: jax.Array,
    metadata: RaggedGatherReduceMetadata,
    *,
    num_out_tokens: int,
    top_k: int,
    weights: jax.Array | None = None,
) -> jax.Array:
  """Reduces valid rows ``< count`` back to ``[num_out_tokens, d0, d1]`` in FP32.

  If ``weights`` (``float32[slots_len]``, indexed by compacted slot) is
  provided, each gathered row is multiplied in FP32 by its routing weight and
  rounded to ``x.dtype`` in VMEM before FP32 token accumulation, matching
  ``scale_route_weights`` + ``stable_grouped_reduce`` without reading or
  writing HBM past ``count``.
  """
  max_input_tokens, dim0, dim1 = x.shape
  block_size = metadata.block_size
  num_buffers = _RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS
  align = _RAGGED_GATHER_REDUCE_TC_ALIGN
  fields = _RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS
  _, window, slots_len = _ragged_gather_reduce_tc_layout(
      metadata.num_indices, top_k, block_size, num_out_tokens=num_out_tokens
  )
  if num_out_tokens == 0 or max_input_tokens == 0:
    return jnp.zeros((num_out_tokens, dim0, dim1), x.dtype)

  lookahead = _RAGGED_GATHER_REDUCE_TC_LOOKAHEAD
  ring = lookahead + 2
  has_weights = weights is not None

  def _ragged_gather_reduce_kernel(*args):
    if has_weights:
      (
          blocks_smem_ref,
          src_hbm_ref,
          tok_hbm_ref,
          w_hbm_ref,
          x_hbm_ref,
          o_hbm_ref,
          vmem_ref,
          src_smem_ref,
          tok_smem_ref,
          w_smem_ref,
          recv_sem,
          send_sem,
          slots_sem,
      ) = args
    else:
      (
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
      ) = args
      w_hbm_ref = None
      w_smem_ref = None

    num_blocks = blocks_smem_ref[0]

    def _block(blk):
      entry = 1 + fields * blk
      return tuple(blocks_smem_ref[entry + f] for f in range(fields))

    def _window_start(slot_start):
      return jnp.minimum((slot_start // align) * align, slots_len - window)

    def _slots_loads(blk):
      r = blk % ring
      window_start = pl.multiple_of(_window_start(_block(blk)[0]), align)
      pairs = [(src_hbm_ref, src_smem_ref), (tok_hbm_ref, tok_smem_ref)]
      if has_weights:
        pairs.append((w_hbm_ref, w_smem_ref))
      return [
          pltpu.make_async_copy(
              hbm_ref.at[pl.ds(window_start, window)],
              smem_ref.at[pl.ds(r * window, window)],
              slots_sem.at[r],
          )
          for hbm_ref, smem_ref in pairs
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

      @pl.loop(0, num_tokens)
      def _zero_token(t):
        vmem_ref[b, t, :, :] = jnp.zeros((dim0, dim1), x.dtype)

      base = _slot_base(blk)

      def _reduce(i, carry):
        prev_t, total = carry
        t = tok_smem_ref[base + i] - token_start
        row = vmem_ref[b, num_tokens + i, :, :].astype(jnp.float32)
        if has_weights:
          w = w_smem_ref[base + i]
          row = (row * w).astype(x.dtype).astype(jnp.float32)
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

  out_shape = jax.ShapeDtypeStruct((num_out_tokens, dim0, dim1), x.dtype)
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  dma_sems = pltpu.SemaphoreType.DMA((num_buffers,))
  if has_weights:
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.SMEM),
        hbm_spec,
        hbm_spec,
        hbm_spec,
        hbm_spec,
    ]
    scratch_shapes = [
        pltpu.VMEM((num_buffers, block_size, dim0, dim1), x.dtype),
        pltpu.SMEM((ring * window,), jnp.int32),
        pltpu.SMEM((ring * window,), jnp.int32),
        pltpu.SMEM((ring * window,), jnp.float32),
        dma_sems,
        dma_sems,
        pltpu.SemaphoreType.DMA((ring,)),
    ]
    call_args = (
        metadata.blocks,
        metadata.src_rows,
        metadata.tokens,
        weights.astype(jnp.float32),
        x,
    )
    kernel_name = "ragged_gather_weighted_reduce_tc"
  else:
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.SMEM),
        hbm_spec,
        hbm_spec,
        hbm_spec,
    ]
    scratch_shapes = [
        pltpu.VMEM((num_buffers, block_size, dim0, dim1), x.dtype),
        pltpu.SMEM((ring * window,), jnp.int32),
        pltpu.SMEM((ring * window,), jnp.int32),
        dma_sems,
        dma_sems,
        pltpu.SemaphoreType.DMA((ring,)),
    ]
    call_args = (metadata.blocks, metadata.src_rows, metadata.tokens, x)
    kernel_name = "ragged_gather_reduce_tc"

  return pl.pallas_call(
      _ragged_gather_reduce_kernel,
      out_shape=out_shape,
      in_specs=in_specs,
      out_specs=hbm_spec,
      scratch_shapes=scratch_shapes,
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=_tpu_vmem_limit_bytes(),
      ),
      name=kernel_name,
  )(*call_args)


def _scale_dscore_d_chunk(
    d_dim: int, block: int, itemsize: int, vmem_limit: int
) -> int | None:
  """Largest 128-multiple divisor of ``d_dim`` whose tiles fit in VMEM.

  Per grid step: three double-buffered ``(block, d_chunk)`` windows (``g``,
  ``y`` in and ``dy`` out) plus one FP32 ``(block, d_chunk)`` temporary (Mosaic
  fuses the rest); TPU7x measured 65.16 MiB at ``(1024, 4096)`` in bf16.
  """
  budget = int(0.75 * vmem_limit)
  for num_chunks in range(1, d_dim // 128 + 1):
    if d_dim % num_chunks or (d_dim // num_chunks) % 128:
      continue
    d_chunk = d_dim // num_chunks
    if block * d_chunk * (2 * 3 * itemsize + 4) <= budget:
      return d_chunk
  return None


def _ragged_scale_and_dscore_tc(
    g_gathered_RD: jax.Array,
    routed_output_RD: jax.Array,
    w_rows_cap: jax.Array,
    count: jax.Array,
    zero_pad_tail: bool,
) -> tuple[jax.Array, jax.Array]:
  """Scales ``g_gathered_RD`` by ``w_rows_cap`` and computes FP32 ``dScore`` over ``[0, count)``.

  The grid is ``(cdiv(count, 1024), d_dim // d_chunk)``: wide hidden sizes
  (e.g. 4096) split the feature dimension so the VMEM windows fit, and the
  ``dScore`` row sums accumulate across feature chunks.
  """
  cap, d_dim = g_gathered_RD.shape
  block = 1024
  itemsize = jnp.dtype(g_gathered_RD.dtype).itemsize
  vmem_limit = _tpu_vmem_limit_bytes()
  d_chunk = (
      _scale_dscore_d_chunk(d_dim, block, itemsize, vmem_limit)
      if d_dim % 128 == 0
      else None
  )
  can_use_pallas = (
      not zero_pad_tail
      and cap >= block
      and cap % block == 0
      and d_chunk is not None
  )
  if not can_use_pallas:
    iota = jnp.arange(cap, dtype=jnp.int32)
    valid = iota < count
    g_f32 = jnp.where(valid[:, None], g_gathered_RD.astype(jnp.float32), 0.0)
    y_f32 = jnp.where(valid[:, None], routed_output_RD.astype(jnp.float32), 0.0)
    w_f32 = jnp.where(valid, w_rows_cap.astype(jnp.float32), 0.0)
    grad_routed = (g_f32 * w_f32[:, None]).astype(routed_output_RD.dtype)
    dw_rows = jnp.sum(g_f32 * y_f32, axis=1)
    return grad_routed, dw_rows

  w_2d = w_rows_cap.astype(jnp.float32).reshape(cap // 128, 128)
  count_1 = jnp.reshape(jnp.asarray(count, jnp.int32), (1,))

  def _scale_dscore_kernel(c_ref, g_ref, y_ref, w_ref, dy_ref, dw_ref):
    start_row = pl.program_id(0) * block
    n_valid = c_ref[0]

    @pl.when(pl.program_id(1) == 0)
    def _init_dw():
      dw_ref[...] = jnp.zeros_like(dw_ref)

    @pl.when(start_row + block <= n_valid)
    def _full_block():
      g_f32 = g_ref[...].astype(jnp.float32)
      y_f32 = y_ref[...].astype(jnp.float32)
      w_col = jnp.broadcast_to(w_ref[...].reshape(block, 1), g_ref.shape)
      dy_ref[...] = (g_f32 * w_col).astype(dy_ref.dtype)
      dw_ref[...] += jnp.sum(g_f32 * y_f32, axis=1).reshape(8, 128)

    @pl.when(start_row < n_valid)
    def _maybe_partial():
      @pl.when(start_row + block > n_valid)
      def _partial_block():
        rows_2d = start_row + jax.lax.broadcasted_iota(
            jnp.int32, g_ref.shape, 0
        )
        valid_2d = rows_2d < n_valid
        g_f32 = jnp.where(valid_2d, g_ref[...].astype(jnp.float32), 0.0)
        y_f32 = jnp.where(valid_2d, y_ref[...].astype(jnp.float32), 0.0)
        w_col = jnp.broadcast_to(w_ref[...].reshape(block, 1), g_ref.shape)
        w_f32 = jnp.where(valid_2d, w_col, 0.0)
        dy_ref[...] = (g_f32 * w_f32).astype(dy_ref.dtype)
        dw_ref[...] += jnp.sum(g_f32 * y_f32, axis=1).reshape(8, 128)

  grad_routed_RD, dw_2d = pl.pallas_call(
      _scale_dscore_kernel,
      out_shape=[
          jax.ShapeDtypeStruct((cap, d_dim), routed_output_RD.dtype),
          jax.ShapeDtypeStruct((cap // 128, 128), jnp.float32),
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(pl.cdiv(count_1[0], block), d_dim // d_chunk),
          in_specs=[
              pl.BlockSpec((block, d_chunk), lambda i, j, c_ref: (i, j)),
              pl.BlockSpec((block, d_chunk), lambda i, j, c_ref: (i, j)),
              pl.BlockSpec((8, 128), lambda i, j, c_ref: (i, 0)),
          ],
          out_specs=[
              pl.BlockSpec((block, d_chunk), lambda i, j, c_ref: (i, j)),
              pl.BlockSpec((8, 128), lambda i, j, c_ref: (i, 0)),
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          vmem_limit_bytes=vmem_limit,
      ),
      name="ragged_scale_and_dscore_tc",
  )(count_1, g_gathered_RD, routed_output_RD, w_2d)
  return grad_routed_RD, dw_2d.reshape(cap)


# --- PyTorch <-> JAX Custom Operators (@pallas.jax_op) ---


@pallas.jax_op("tt_moe::roe_routing_metadata")
def _roe_routing_metadata_tpu(
    ids_flat_M: jax.Array,
    ep_rank_1: jax.Array,
    num_experts: int,
    ep_size: int,
    top_k: int,
    num_out_tokens: int,
    cap: int,
    reduce_block_size: int,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
  m = ids_flat_M.shape[0]
  sorted_experts_M, slot_order_M = packed_stable_sort(
      ids_flat_M, num_experts - 1
  )
  group_offsets_Ep1 = _sorted_expert_offsets(sorted_experts_M, num_experts)

  shard_idx = ep_rank_1[0].astype(jnp.int32)
  local_num_experts = num_experts // ep_size
  local_offsets_ep1 = jax.lax.dynamic_slice_in_dim(
      group_offsets_Ep1,
      shard_idx * local_num_experts,
      local_num_experts + 1,
      axis=0,
  )
  start = local_offsets_ep1[0]
  clamped_offsets_ep1 = jnp.minimum(local_offsets_ep1, start + cap)
  local_group_sizes_e = jnp.diff(clamped_offsets_ep1).astype(jnp.int32)
  count = (clamped_offsets_ep1[local_num_experts] - start).astype(jnp.int32)

  token_ids_sorted_M = (slot_order_M // top_k).astype(jnp.int32)
  iota_cap = jnp.arange(cap, dtype=jnp.int32)
  valid_cap = iota_cap < count
  if m > 0:
    safe_idx = jnp.where(valid_cap, jnp.minimum(start + iota_cap, m - 1), 0)
    window_slot_order_cap = slot_order_M[safe_idx].astype(jnp.int32)
  else:
    window_slot_order_cap = jnp.zeros((cap,), dtype=jnp.int32)

  meta = ragged_gather_reduce_tc_metadata(
      token_ids_sorted_M,
      start,
      count,
      top_k=top_k,
      block_size=reduce_block_size,
      num_out_tokens=num_out_tokens,
      packed_sort=True,
  )
  start_count_2 = jnp.stack([start, count]).astype(jnp.int32)
  return (
      local_group_sizes_e,
      token_ids_sorted_M,
      window_slot_order_cap,
      start_count_2,
      meta.src_rows,
      meta.tokens,
      meta.blocks,
  )


@pallas.jax_op("tt_moe::roe_gather_fwd")
def _roe_gather_fwd(
    x_all_GD: jax.Array,
    token_ids_sorted_M: jax.Array,
    start_count_2: jax.Array,
    src_rows: jax.Array,
    tokens: jax.Array,
    blocks: jax.Array,
    cap: int,
    top_k: int,
    gather_block_size: int,
    reduce_block_size: int,
    zero_pad_tail: bool,
) -> jax.Array:
  del src_rows, tokens, blocks, top_k, reduce_block_size
  g_tokens, d_dim = x_all_GD.shape
  x3d = x_all_GD.reshape(g_tokens, d_dim // 128, 128)
  out3d = ragged_gather_tc(
      x3d,
      token_ids_sorted_M,
      start=start_count_2[0],
      num_tokens=start_count_2[1],
      max_out_tokens=cap,
      block_size=gather_block_size,
  )
  out2d = out3d.reshape(cap, d_dim)
  if zero_pad_tail:
    valid = jnp.arange(cap, dtype=jnp.int32) < start_count_2[1]
    out2d = jnp.where(valid[:, None], out2d, jnp.zeros((), out2d.dtype))
  return out2d


@pallas.jax_op("tt_moe::roe_gather_bwd")
def _roe_gather_bwd(
    grad_routed_RD: jax.Array,
    src_rows: jax.Array,
    tokens: jax.Array,
    blocks: jax.Array,
    num_indices: int,
    num_out_tokens: int,
    top_k: int,
    reduce_block_size: int,
) -> jax.Array:
  cap, d_dim = grad_routed_RD.shape
  meta = RaggedGatherReduceMetadata(
      src_rows=src_rows,
      tokens=tokens,
      blocks=blocks,
      block_size=reduce_block_size,
      num_indices=num_indices,
  )
  grad3d = grad_routed_RD.reshape(cap, d_dim // 128, 128)
  out3d = ragged_gather_reduce_tc(
      grad3d,
      meta,
      num_out_tokens=num_out_tokens,
      top_k=top_k,
      weights=None,
  )
  return out3d.reshape(num_out_tokens, d_dim)


def _roe_gather_setup_context(ctx, inputs, output):
  del output
  (
      x_all_GD,
      token_ids_sorted_M,
      _start_count_2,
      src_rows,
      tokens,
      blocks,
      _cap,
      top_k,
      _gather_block_size,
      reduce_block_size,
      _zero_pad_tail,
  ) = inputs
  ctx.save_for_backward(src_rows, tokens, blocks)
  ctx.num_indices = token_ids_sorted_M.shape[0]
  ctx.num_out_tokens = x_all_GD.shape[0]
  ctx.top_k = top_k
  ctx.reduce_block_size = reduce_block_size


def _roe_gather_backward(ctx, grad_routed_RD):
  src_rows, tokens, blocks = ctx.saved_tensors
  grad_x_all_GD = _roe_gather_bwd(
      grad_routed_RD.contiguous(),
      src_rows,
      tokens,
      blocks,
      ctx.num_indices,
      ctx.num_out_tokens,
      ctx.top_k,
      ctx.reduce_block_size,
  )
  return (grad_x_all_GD,) + (None,) * 10


_roe_gather_fwd.register_autograd(
    _roe_gather_backward, setup_context=_roe_gather_setup_context
)


@pallas.jax_op("tt_moe::roe_unsort_fwd")
def _roe_unsort_fwd(
    routed_output_RD: jax.Array,
    scores_flat_M: jax.Array,
    token_ids_sorted_M: jax.Array,
    window_slot_order_cap: jax.Array,
    start_count_2: jax.Array,
    src_rows: jax.Array,
    tokens: jax.Array,
    blocks: jax.Array,
    num_out_tokens: int,
    top_k: int,
    gather_block_size: int,
    reduce_block_size: int,
    zero_pad_tail: bool,
) -> jax.Array:
  del gather_block_size, zero_pad_tail
  cap, d_dim = routed_output_RD.shape
  count = start_count_2[1]
  iota_cap = jnp.arange(cap, dtype=jnp.int32)
  valid_cap = iota_cap < count
  w_flat = scores_flat_M.astype(jnp.float32)
  if w_flat.shape[0] > 0:
    w_rows_cap = jnp.where(valid_cap, w_flat[window_slot_order_cap], 0.0)
  else:
    w_rows_cap = jnp.zeros((cap,), dtype=jnp.float32)

  safe_src = jnp.clip(src_rows, 0, max(cap - 1, 0))
  weights_compacted_S = w_rows_cap[safe_src]

  meta = RaggedGatherReduceMetadata(
      src_rows=src_rows,
      tokens=tokens,
      blocks=blocks,
      block_size=reduce_block_size,
      num_indices=token_ids_sorted_M.shape[0],
  )
  out3d = ragged_gather_reduce_tc(
      routed_output_RD.reshape(cap, d_dim // 128, 128),
      meta,
      num_out_tokens=num_out_tokens,
      top_k=top_k,
      weights=weights_compacted_S,
  )
  return out3d.reshape(num_out_tokens, d_dim)


@pallas.jax_op("tt_moe::roe_unsort_bwd")
def _roe_unsort_bwd(
    grad_partial_GD: jax.Array,
    routed_output_RD: jax.Array,
    scores_flat_M: jax.Array,
    token_ids_sorted_M: jax.Array,
    window_slot_order_cap: jax.Array,
    start_count_2: jax.Array,
    gather_block_size: int,
    zero_pad_tail: bool,
) -> tuple[jax.Array, jax.Array]:
  g_tokens, d_dim = grad_partial_GD.shape
  cap = routed_output_RD.shape[0]
  m = scores_flat_M.shape[0]
  start = start_count_2[0]
  count = start_count_2[1]

  g_gathered_3d = ragged_gather_tc(
      grad_partial_GD.reshape(g_tokens, d_dim // 128, 128),
      token_ids_sorted_M,
      start=start,
      num_tokens=count,
      max_out_tokens=cap,
      block_size=gather_block_size,
  )
  g_gathered_RD = g_gathered_3d.reshape(cap, d_dim)

  iota_cap = jnp.arange(cap, dtype=jnp.int32)
  valid_cap = iota_cap < count
  w_flat = scores_flat_M.astype(jnp.float32)
  if m > 0:
    w_rows_cap = jnp.where(valid_cap, w_flat[window_slot_order_cap], 0.0)
  else:
    w_rows_cap = jnp.zeros((cap,), dtype=jnp.float32)

  grad_routed_RD, dw_rows_cap = _ragged_scale_and_dscore_tc(
      g_gathered_RD,
      routed_output_RD,
      w_rows_cap,
      count,
      zero_pad_tail,
  )

  if m > 0:
    dw_valid = jnp.where(valid_cap, dw_rows_cap, 0.0)
    scatter_idx = jnp.where(valid_cap, window_slot_order_cap, m + iota_cap)
    grad_scores_padded = (
        jnp.zeros((m + cap,), dtype=jnp.float32)
        .at[scatter_idx]
        .set(dw_valid, unique_indices=True)
    )
    grad_scores_flat_M = grad_scores_padded[:m].astype(scores_flat_M.dtype)
  else:
    grad_scores_flat_M = jnp.zeros_like(scores_flat_M)

  return grad_routed_RD, grad_scores_flat_M


def _roe_unsort_setup_context(ctx, inputs, output):
  del output
  (
      routed_output_RD,
      scores_flat_M,
      token_ids_sorted_M,
      window_slot_order_cap,
      start_count_2,
      _src_rows,
      _tokens,
      _blocks,
      _num_out_tokens,
      _top_k,
      gather_block_size,
      _reduce_block_size,
      zero_pad_tail,
  ) = inputs
  ctx.save_for_backward(
      routed_output_RD,
      scores_flat_M,
      token_ids_sorted_M,
      window_slot_order_cap,
      start_count_2,
  )
  ctx.gather_block_size = gather_block_size
  ctx.zero_pad_tail = zero_pad_tail


def _roe_unsort_backward(ctx, grad_partial_GD):
  (
      routed_output_RD,
      scores_flat_M,
      token_ids_sorted_M,
      window_slot_order_cap,
      start_count_2,
  ) = ctx.saved_tensors
  grad_routed_RD, grad_scores_flat_M = _roe_unsort_bwd(
      grad_partial_GD.contiguous(),
      routed_output_RD,
      scores_flat_M,
      token_ids_sorted_M,
      window_slot_order_cap,
      start_count_2,
      ctx.gather_block_size,
      ctx.zero_pad_tail,
  )
  return (grad_routed_RD, grad_scores_flat_M) + (None,) * 11


_roe_unsort_fwd.register_autograd(
    _roe_unsort_backward, setup_context=_roe_unsort_setup_context
)


@pallas.jax_op("tt_moe::roe_row_scale_fwd")
def _roe_row_scale_fwd(
    scores_flat_M: jax.Array,
    window_slot_order_cap: jax.Array,
    start_count_2: jax.Array,
) -> jax.Array:
  cap = window_slot_order_cap.shape[0]
  m = scores_flat_M.shape[0]
  if cap == 0 or m == 0:
    return jnp.zeros((cap,), dtype=jnp.float32)
  count = start_count_2[1]
  iota_cap = jnp.arange(cap, dtype=jnp.int32)
  valid_cap = iota_cap < count
  slot_R = jnp.clip(window_slot_order_cap, 0, m - 1)
  return jnp.where(valid_cap, scores_flat_M.astype(jnp.float32)[slot_R], 0.0)


@pallas.jax_op("tt_moe::roe_row_scale_bwd")
def _roe_row_scale_bwd(
    grad_row_scale_R: jax.Array,
    window_slot_order_cap: jax.Array,
    start_count_2: jax.Array,
    num_slots: int,
) -> jax.Array:
  cap = window_slot_order_cap.shape[0]
  if cap == 0 or num_slots == 0:
    return jnp.zeros((num_slots,), dtype=jnp.float32)
  count = start_count_2[1]
  iota_cap = jnp.arange(cap, dtype=jnp.int32)
  valid_cap = iota_cap < count
  dw_valid = jnp.where(valid_cap, grad_row_scale_R.astype(jnp.float32), 0.0)
  slot_R = jnp.clip(window_slot_order_cap, 0, num_slots - 1)
  scatter_idx = jnp.where(valid_cap, slot_R, num_slots + iota_cap)
  grad_padded = (
      jnp.zeros((num_slots + cap,), dtype=jnp.float32)
      .at[scatter_idx]
      .set(dw_valid, unique_indices=True)
  )
  return grad_padded[:num_slots]


def _roe_row_scale_setup_context(ctx, inputs, output):
  del output
  scores_flat_M, window_slot_order_cap, start_count_2 = inputs
  ctx.save_for_backward(window_slot_order_cap, start_count_2)
  ctx.num_slots = scores_flat_M.shape[0]
  ctx.scores_dtype = scores_flat_M.dtype


def _roe_row_scale_backward(ctx, grad_row_scale_R):
  window_slot_order_cap, start_count_2 = ctx.saved_tensors
  grad_scores = _roe_row_scale_bwd(
      grad_row_scale_R.contiguous(),
      window_slot_order_cap,
      start_count_2,
      ctx.num_slots,
  )
  if grad_scores.dtype != ctx.scores_dtype:
    grad_scores = grad_scores.to(ctx.scores_dtype)
  return grad_scores, None, None


_roe_row_scale_fwd.register_autograd(
    _roe_row_scale_backward, setup_context=_roe_row_scale_setup_context
)


# --- Public PyTorch Entrypoints ---


def roe_compute_routing(
    ids_flat_M: torch.Tensor,
    ep_rank_1: torch.Tensor,
    *,
    num_experts: int,
    ep_size: int,
    top_k: int,
    num_out_tokens: int,
    cap: int,
    reduce_block_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
  """Computes Ring-of-Experts Pallas TC routing metadata on TPU."""
  if ids_flat_M.dtype != torch.int32:
    ids_flat_M = ids_flat_M.to(torch.int32)
  if ep_rank_1.dtype != torch.int32:
    ep_rank_1 = ep_rank_1.to(torch.int32)
  return _roe_routing_metadata_tpu(
      ids_flat_M.contiguous(),
      ep_rank_1.contiguous(),
      num_experts,
      ep_size,
      top_k,
      num_out_tokens,
      cap,
      reduce_block_size,
  )


def roe_sort_gather(
    x_all_GD: torch.Tensor,
    token_ids_sorted_M: torch.Tensor,
    start_count_2: torch.Tensor,
    src_rows: torch.Tensor,
    tokens: torch.Tensor,
    blocks: torch.Tensor,
    *,
    cap: int,
    top_k: int,
    gather_block_size: int = 1024,
    reduce_block_size: int = 896,
    zero_pad_tail: bool = False,
) -> torch.Tensor:
  """Gathers ``x_all_GD`` into ``[cap, D]`` for this rank's local experts."""
  return _roe_gather_fwd(
      x_all_GD.contiguous(),
      token_ids_sorted_M,
      start_count_2,
      src_rows,
      tokens,
      blocks,
      cap,
      top_k,
      gather_block_size,
      reduce_block_size,
      zero_pad_tail,
  )


def roe_unsort_combine(
    routed_output_RD: torch.Tensor,
    scores_flat_M: torch.Tensor,
    token_ids_sorted_M: torch.Tensor,
    window_slot_order_cap: torch.Tensor,
    start_count_2: torch.Tensor,
    src_rows: torch.Tensor,
    tokens: torch.Tensor,
    blocks: torch.Tensor,
    *,
    num_out_tokens: int,
    top_k: int,
    gather_block_size: int = 1024,
    reduce_block_size: int = 896,
    zero_pad_tail: bool = False,
) -> torch.Tensor:
  """Weighted FP32 gather-reduce of ``routed_output_RD`` back to ``[G, D]``."""
  return _roe_unsort_fwd(
      routed_output_RD.contiguous(),
      scores_flat_M.contiguous(),
      token_ids_sorted_M,
      window_slot_order_cap,
      start_count_2,
      src_rows,
      tokens,
      blocks,
      num_out_tokens,
      top_k,
      gather_block_size,
      reduce_block_size,
      zero_pad_tail,
  )


class _RoEUnsortReduce(torch.autograd.Function):
  """Unweighted gather-reduce of ``routed_output_RD`` back to ``[G, D]``.

  The transpose of ``roe_sort_gather``: forward is the TC ragged gather-reduce
  and backward is the TC ragged gather, so no routing-weight gradient is
  computed here (weights are applied on the expert activation instead).
  """

  @staticmethod
  def forward(
      ctx,
      routed_output_RD: torch.Tensor,
      token_ids_sorted_M: torch.Tensor,
      start_count_2: torch.Tensor,
      src_rows: torch.Tensor,
      tokens: torch.Tensor,
      blocks: torch.Tensor,
      num_out_tokens: int,
      top_k: int,
      gather_block_size: int,
      reduce_block_size: int,
  ) -> torch.Tensor:
    ctx.save_for_backward(
        token_ids_sorted_M, start_count_2, src_rows, tokens, blocks
    )
    ctx.cap = routed_output_RD.shape[0]
    ctx.top_k = top_k
    ctx.gather_block_size = gather_block_size
    ctx.reduce_block_size = reduce_block_size
    return _roe_gather_bwd(
        routed_output_RD.contiguous(),
        src_rows,
        tokens,
        blocks,
        token_ids_sorted_M.shape[0],
        num_out_tokens,
        top_k,
        reduce_block_size,
    )

  @staticmethod
  def backward(ctx, grad_partial_GD: torch.Tensor):
    token_ids_sorted_M, start_count_2, src_rows, tokens, blocks = (
        ctx.saved_tensors
    )
    grad_routed_RD = _roe_gather_fwd(
        grad_partial_GD.contiguous(),
        token_ids_sorted_M,
        start_count_2,
        src_rows,
        tokens,
        blocks,
        ctx.cap,
        ctx.top_k,
        ctx.gather_block_size,
        ctx.reduce_block_size,
        False,
    )
    return (grad_routed_RD,) + (None,) * 9


def roe_unsort_reduce(
    routed_output_RD: torch.Tensor,
    token_ids_sorted_M: torch.Tensor,
    start_count_2: torch.Tensor,
    src_rows: torch.Tensor,
    tokens: torch.Tensor,
    blocks: torch.Tensor,
    *,
    num_out_tokens: int,
    top_k: int,
    gather_block_size: int = 1024,
    reduce_block_size: int = 896,
) -> torch.Tensor:
  """Unweighted FP32 gather-reduce of ``routed_output_RD`` back to ``[G, D]``."""
  return _RoEUnsortReduce.apply(
      routed_output_RD,
      token_ids_sorted_M,
      start_count_2,
      src_rows,
      tokens,
      blocks,
      num_out_tokens,
      top_k,
      gather_block_size,
      reduce_block_size,
  )


def roe_gather_row_scales(
    scores_flat_M: torch.Tensor,
    window_slot_order_cap: torch.Tensor,
    start_count_2: torch.Tensor,
) -> torch.Tensor:
  """Gathers ``[cap]`` FP32 routing weights with collision-free backward scatter."""
  return _roe_row_scale_fwd(
      scores_flat_M.contiguous(),
      window_slot_order_cap.contiguous(),
      start_count_2.contiguous(),
  )


__all__ = [
    "roe_compute_routing",
    "roe_gather_row_scales",
    "roe_sort_gather",
    "roe_unsort_combine",
    "roe_unsort_reduce",
    "safe_gather_block_size",
    "safe_reduce_block_size",
]
