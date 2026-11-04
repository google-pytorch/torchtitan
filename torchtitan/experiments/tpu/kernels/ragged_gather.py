# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""TensorCore (Pallas DMA) ragged row gather for static-shape MoE transport.

`ragged_gather` copies `x[token_ids[i]]` for the first `valid_count` rows of a
static-size output with one row DMA per output row, and
`inverse_permute_scatter` is the transpose of a permutation gather as a TPU
sort. Both operate on the row-contiguous `[T, D // 128, 128]` token layout
produced by `to_token_3d`.
"""

import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas


_RAGGED_GATHER_TC_NUM_BUFFERS = 2
_GATHER_2D_SUB_ROWS = 512
_GATHER_2D_VMEM_LIMIT = 64 * 1024 * 1024

# Rows per grid step of the gather kernel.
GATHER_BLOCK_SIZE = 1536


def _ragged_gather_tc_impl(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array | int,
    num_tokens: jax.Array | int,
    max_out_tokens: int,
    block_size: int = 2048,
) -> jax.Array:
  """Executes a ragged gather on TensorCore using a Pallas kernel."""
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

      @pl.when(next_data_block >= num_buffers)
      def _wait_prev_out():
        _run_step(next_data_block - num_buffers, next_data_buf, "copy_out_wait")

      _run_step(next_data_block, next_data_buf, "data_start")

    @pl.when(block_idx == num_steps - 1)
    def _epilogue():
      for k in range(1, num_buffers):

        @pl.when(block_idx >= k)
        def _wait_prev_out(k=k):
          prev_block = block_idx - k
          _run_step(prev_block, prev_block % num_buffers, "copy_out_wait")

      _run_step(block_idx, buf_idx, "copy_out_wait")

  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  out_shape = jax.ShapeDtypeStruct(
      (max_out_tokens, dim0, dim1),
      x.dtype,
      manual_axis_type=manual_axis_type,
  )
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
      ),
      name="ragged_gather_tc",
  )(scalars, indices, x)


def _ragged_gather_tc_2d_impl(
    x: jax.Array,
    indices: jax.Array,
    num_valid: jax.Array,
    max_out_tokens: int,
    block_size: int,
) -> jax.Array:
  """Gathers `x[indices[i]]` for `i < num_valid` and zeroes `i >= num_valid` in `[R, D]`.

  Fusing the `(sub, D // 128, 128) -> (sub, D)` relayout and `i < num_valid`
  zeroing in VMEM avoids a separate full-buffer HBM relayout + select pass.
  """
  num_indices = indices.shape[0]
  _, dim0, dim1 = x.shape
  d = dim0 * dim1
  num_buffers = _RAGGED_GATHER_TC_NUM_BUFFERS
  block_size = min(block_size, max_out_tokens)
  sub = math.gcd(block_size, max_out_tokens, _GATHER_2D_SUB_ROWS)
  num_blocks = pl.cdiv(max_out_tokens, block_size)
  rem = max_out_tokens % block_size

  idx_align = 128
  idx_window = pl.cdiv(block_size, idx_align) * idx_align + idx_align
  idx_pad = max((-num_indices) % idx_align, idx_window - num_indices)
  if idx_pad > 0:
    indices = jnp.pad(indices, (0, idx_pad))
  max_win_start = indices.shape[0] - idx_window

  def _kernel(
      scalars_ref,
      idx_hbm,
      x_hbm,
      o_hbm,
      in_vmem,
      out_vmem,
      idx_smem,
      idx_sem,
      recv_sem,
      send_sem,
  ):
    blk = pl.program_id(0)
    n_valid = scalars_ref[0]

    def _with_len(step, fn):
      if rem == 0:
        fn(block_size)
      elif isinstance(step, int):
        fn(rem if step == num_blocks - 1 else block_size)
      else:
        jax.lax.cond(
            step == num_blocks - 1,
            lambda: fn(rem),
            lambda: fn(block_size),
        )

    def _valid_rows(step, length):
      return jnp.clip(n_valid - step * block_size, 0, length)

    def _start_fetch(step, slot):
      def _body(length):
        sv = _valid_rows(step, length)
        row0 = step * block_size
        win = jnp.minimum((row0 // idx_align) * idx_align, max_win_start)
        idx_off = row0 - win

        def _issue_chunk(off, size: int):
          step_unroll = math.gcd(size, 128)

          @pl.loop(0, size // step_unroll)
          def _(chunk_i):
            base_i = off + chunk_i * step_unroll
            src_indices = [
                idx_smem[idx_off + base_i + u] for u in range(step_unroll)
            ]
            for u in range(step_unroll):
              pltpu.make_async_copy(
                  x_hbm.at[pl.ds(src_indices[u], 1), :, :],
                  in_vmem.at[slot, pl.ds(base_i + u, 1), :, :],
                  recv_sem.at[slot],
              ).start()

        @pl.when(sv > 0)
        def _():
          copy = pltpu.make_async_copy(
              idx_hbm.at[pl.ds(pl.multiple_of(win, idx_align), idx_window)],
              idx_smem,
              idx_sem,
          )
          copy.start()
          copy.wait()

          @pl.when(sv == length)
          def _():
            _issue_chunk(0, length)

          @pl.when(sv < length)
          def _():
            off = 0
            for k in reversed(range(length.bit_length())):
              size = 1 << k
              bit = sv & size

              def _if_bit(off=off, size=size):
                _issue_chunk(off, size)

              pl.when(bit != 0)(_if_bit)
              off = off + bit

      _with_len(step, _body)

    def _wait_fetch(step, slot):
      def _body(length):
        sv = _valid_rows(step, length)

        @pl.when(sv == length)
        def _():
          rows = in_vmem.at[slot, pl.ds(0, length), :, :]
          pltpu.make_async_copy(rows, rows, recv_sem.at[slot]).wait()

        @pl.when(sv < length)
        def _():
          for k in reversed(range(length.bit_length())):
            size = 1 << k

            @pl.when((sv & size) != 0)
            def _(size=size):
              rows = in_vmem.at[slot, pl.ds(0, size), :, :]
              pltpu.make_async_copy(rows, rows, recv_sem.at[slot]).wait()

      _with_len(step, _body)

    def _send_copy(step, slot, length):
      return pltpu.make_async_copy(
          out_vmem.at[slot, pl.ds(0, length), :],
          o_hbm.at[pl.ds(pl.multiple_of(step * block_size, sub), length), :],
          send_sem.at[slot],
      )

    def _emit(step, slot):
      def _body(length):
        sv = _valid_rows(step, length)

        @pl.when(sv == length)
        def _full():
          @pl.loop(0, length // sub)
          def _(j):
            r0 = pl.multiple_of(j * sub, sub)
            out_vmem[slot, pl.ds(r0, sub), :] = in_vmem[
                slot, pl.ds(r0, sub), :, :
            ].reshape(sub, d)

        @pl.when(sv == 0)
        def _empty():
          prev_had_valid = (step < num_buffers) | (
              (step - num_buffers) * block_size < n_valid
          )

          @pl.when(prev_had_valid)
          def _zero_slot():
            @pl.loop(0, length // sub)
            def _(j):
              r0 = pl.multiple_of(j * sub, sub)
              out_vmem[slot, pl.ds(r0, sub), :] = jnp.zeros((sub, d), x.dtype)

        @pl.when((sv > 0) & (sv < length))
        def _partial():
          @pl.loop(0, length // sub)
          def _(j):
            r0 = pl.multiple_of(j * sub, sub)
            rows = in_vmem[slot, pl.ds(r0, sub), :, :].reshape(sub, d)
            ids = r0 + jax.lax.broadcasted_iota(jnp.int32, (sub, 1), 0)
            out_vmem[slot, pl.ds(r0, sub), :] = jnp.where(
                ids < sv, rows, jnp.zeros((), rows.dtype)
            )

        _send_copy(step, slot, length).start()

      _with_len(step, _body)

    def _wait_send(step, slot):
      _with_len(step, lambda length: _send_copy(step, slot, length).wait())

    @pl.when(blk == 0)
    def _():
      _start_fetch(0, 0)

    @pl.when(blk + 1 < num_blocks)
    def _():
      _start_fetch(blk + 1, (blk + 1) % num_buffers)

    slot = blk % num_buffers
    _wait_fetch(blk, slot)

    @pl.when(blk >= num_buffers)
    def _():
      _wait_send(blk - num_buffers, slot)

    _emit(blk, slot)

    @pl.when(blk == num_blocks - 1)
    def _():
      for k in range(min(num_buffers, num_blocks)):
        prev = blk - k

        @pl.when(prev >= 0)
        def _(prev=prev):
          _wait_send(prev, prev % num_buffers)

  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  scalars = jnp.reshape(jnp.asarray(num_valid, jnp.int32), (1,))
  return pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct(
          (max_out_tokens, d), x.dtype, manual_axis_type=manual_axis_type
      ),
      grid=(num_blocks,),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
      scratch_shapes=[
          pltpu.VMEM((num_buffers, block_size, dim0, dim1), x.dtype),
          pltpu.VMEM((num_buffers, block_size, d), x.dtype),
          pltpu.SMEM((idx_window,), indices.dtype),
          pltpu.SemaphoreType.DMA(()),
          pltpu.SemaphoreType.DMA((num_buffers,)),
          pltpu.SemaphoreType.DMA((num_buffers,)),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(pltpu.ARBITRARY,),
          vmem_limit_bytes=_GATHER_2D_VMEM_LIMIT,
      ),
      name="ragged_gather_tc",
  )(scalars, indices, x)


@pallas.jax_op("tt_moe::ragged_gather_tc")
def _ragged_gather_tc_tpu(
    x_T3: jax.Array,
    token_ids_R: jax.Array,
    valid_count_1: jax.Array,
    block_size: int,
) -> jax.Array:
  """`out[i] = x[token_ids_R[i]]` for `i < valid_count`, else 0, as `[R, D]`."""
  _, d0, d1 = x_T3.shape
  max_R = token_ids_R.shape[0]
  num_valid = valid_count_1[0].astype(jnp.int32)
  if max_R > 0 and max_R % 128 == 0:
    return _ragged_gather_tc_2d_impl(
        x_T3,
        token_ids_R.astype(jnp.int32),
        num_valid,
        max_out_tokens=max_R,
        block_size=block_size,
    )
  out_R3 = _ragged_gather_tc_impl(
      x_T3,
      token_ids_R.astype(jnp.int32),
      start=jnp.int32(0),
      num_tokens=num_valid,
      max_out_tokens=max_R,
      block_size=block_size,
  )
  valid = jnp.arange(max_R, dtype=jnp.int32) < num_valid
  return jnp.where(
      valid[:, None], out_R3.reshape(max_R, d0 * d1), jnp.zeros((), x_T3.dtype)
  )


@pallas.jax_op("tt_moe::inverse_permute_scatter")
def _inverse_permute_scatter_tpu(
    perm_N: jax.Array,
    values_R: jax.Array,
    valid_count_1: jax.Array,
) -> jax.Array:
  """`out[perm_N[j]] = values_R[j]` for `j < valid_count`, else 0."""
  n = perm_N.shape[0]
  r = values_R.shape[0]
  valid = jnp.arange(r, dtype=jnp.int32) < valid_count_1[0].astype(jnp.int32)
  v = jnp.where(valid, values_R, jnp.zeros((), values_R.dtype))
  if r < n:
    v = jnp.pad(v, (0, n - r))
  elif r > n:
    v = v[:n]
  # `perm_N` is a permutation, so a key-value sort on it places every value at
  # its destination. TPU sorts are far cheaper than scalar XLA gathers/scatters.
  _, out = jax.lax.sort(
      (perm_N.astype(jnp.int32), v), num_keys=1, is_stable=False
  )
  return out


# ---------------------------------------------------------------------------
# Public torch API. Each op has a TPU path (the Pallas kernels above) and an
# eager reference path used on other devices and as the test oracle.
# ---------------------------------------------------------------------------


def _check_hidden(d: int) -> None:
  if d % 128 != 0:
    raise ValueError(f"hidden size must be a multiple of 128, got {d}")


def to_token_3d(x_TD: torch.Tensor) -> torch.Tensor:
  """`[T, D] -> [T, D // 128, 128]`, the kernels' row-contiguous token layout.

  Apply this to the *local* tokens before an EP all-gather, so the relayout
  copy XLA inserts touches `T` rows instead of the `ep_size * T` gathered ones.
  """
  t, d = x_TD.shape
  _check_hidden(d)
  return x_TD.reshape(t, d // 128, 128)


def ragged_gather(
    x_T3: torch.Tensor,
    token_ids_R: torch.Tensor,
    valid_count_1: torch.Tensor,
) -> torch.Tensor:
  """`out[i] = x[token_ids_R[i]]` for `i < valid_count`, else 0.

  Args:
    x_T3: `[T, D // 128, 128]` tokens (see `to_token_3d`).
    token_ids_R: `[R]` source token of each output row.
    valid_count_1: `[1]` number of leading valid rows.

  Returns:
    `[R, D]` gathered rows; rows `>= valid_count` are zero.
  """
  if x_T3.ndim != 3 or token_ids_R.ndim != 1 or valid_count_1.numel() != 1:
    raise ValueError(
        "ragged_gather expects x_T3[T, D/128, 128], token_ids_R[R], and"
        " valid_count_1[1]"
    )
  if x_T3.device.type == "tpu":
    return _ragged_gather_tc_tpu(
        x_T3,
        token_ids_R.to(torch.int32),
        valid_count_1.reshape(1).to(torch.int32),
        GATHER_BLOCK_SIZE,
    )
  x_TD = x_T3.reshape(x_T3.shape[0], -1)
  max_R = int(token_ids_R.shape[0])
  valid = torch.arange(max_R, device=x_TD.device) < valid_count_1.reshape(1)
  safe_idx = torch.where(valid, token_ids_R.long(), 0)
  rows = torch.index_select(x_TD, 0, safe_idx)
  return torch.where(valid.unsqueeze(-1), rows, torch.zeros_like(rows))


def inverse_permute_scatter(
    perm_N: torch.Tensor,
    values_R: torch.Tensor,
    valid_count_1: torch.Tensor,
) -> torch.Tensor:
  """`out[perm_N[j]] = values_R[j]` for `j < valid_count`, else 0.

  Args:
    perm_N: a permutation of `arange(N)`.
    values_R: values in permuted order; rows `j >= valid_count` are ignored.
    valid_count_1: `[1]` number of leading valid rows.

  Returns:
    `[N]` values in original (unpermuted) order.
  """
  if perm_N.ndim != 1 or values_R.ndim != 1 or valid_count_1.numel() != 1:
    raise ValueError(
        "inverse_permute_scatter expects perm_N[N], values_R[R], and"
        " valid_count_1[1]"
    )
  if perm_N.device.type == "tpu":
    return _inverse_permute_scatter_tpu(
        perm_N.to(torch.int32),
        values_R,
        valid_count_1.reshape(1).to(torch.int32),
    )
  n = int(perm_N.shape[0])
  r = min(int(values_R.shape[0]), n)
  valid = torch.arange(r, device=values_R.device) < valid_count_1.reshape(1)
  out = values_R.new_zeros((n,))
  out[perm_N[:r].long()] = torch.where(
      valid, values_R[:r], torch.zeros_like(values_R[:r])
  )
  return out


__all__ = [
    "GATHER_BLOCK_SIZE",
    "inverse_permute_scatter",
    "ragged_gather",
    "to_token_3d",
]
