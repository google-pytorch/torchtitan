"""Ring-of-Experts expert GEMMs that read/write the RoE token layout directly.

The Ring-of-Experts (RoE) ragged gather and gather-reduce kernels
(``ragged_gather_tc``) move whole tokens with one DMA each, so they keep tokens
in a ``(rows, D / 128, 128)`` layout, where each token is one contiguous slab.
The grouped matmuls need ``(rows, D)`` with tokens on sublanes. Converting
between the two is an XLA relayout copy of the whole expert buffer (twice per
chunk in forward, three times in backward).

This module removes those copies by doing the relayout inside VMEM, in the
kernels that already touch the data:

- ``ragged_gather_2d``: the RoE dispatch gather, writing ``(cap, D)``.
- ``gmm_out3d``: the W2 grouped matmul, writing ``(cap, D / 128, 128)`` for
  the RoE combine gather-reduce.
- ``_accumulate_dinput_3d``: the W1/W3 dInput accumulate, writing
  ``(cap, D / 128, 128)`` for the dispatch backward gather-reduce.

The relayout uses the 32-bit sublane-strided load/store trick of
``experimental/qwen35_64c/qwen35_jax/gmmv2/relayout.py`` (bf16 only).

The PyTorch entry points ``roe_fused_w13`` and ``roe_fused_w2_combine`` fuse
the dispatch gather with the W1/W3 grouped matmul and the W2 grouped matmul
with the combine gather-reduce into one autograd op each, so the layouts of
the tensors autograd sees never change.
"""

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm as megablox_gmm
from jax.experimental.pallas.ops.tpu.megablox.gmm import tgmm as megablox_tgmm
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from torchtitan.experiments.tpu.kernels import paired_segment_matmul as psm
from torchtitan.experiments.tpu.kernels import pallas_vmem
from torchtitan.experiments.tpu.kernels import ragged_gather_tc as rgt
from torchtitan.experiments.tpu.kernels.pallas_naming import call_named

LANES = 128
# Tokens relayouted per static step: (64, 128) uint32 words = 8 vregs/load.
_RELAYOUT_ROWS = 128
_LO_MASK = 0xFFFF
_HI_MASK = 0xFFFF0000
_GATHER_IDX_ALIGN = 128
_GATHER_ISSUE_UNROLL = 16


def _u32_flat(ref, interpret: bool = False):
  """Views a bf16 ref ``[n, d0, 128]`` as uint32 ``[-1, 128]``.

  Interpret mode cannot bitcast refs; it returns ``(ref, None)`` and the
  pair helpers fall back to (slow on TPU) direct 3D slicing.
  """
  d0 = ref.shape[-2]
  if ref.dtype != jnp.bfloat16:
    raise ValueError(f"RoE relayout supports bfloat16, got {ref.dtype}")
  if d0 % 8:
    raise ValueError(f"RoE relayout needs D / 128 % 8 == 0, got {d0}")
  if interpret:
    return ref, None
  return ref.bitcast(jnp.uint32).reshape(-1, LANES), d0 // 2


def _transpose_halves(w0, w1):
  lo = (w0 & jnp.uint32(_LO_MASK)) | (w1 << 16)
  hi = (w0 >> 16) | (w1 & jnp.uint32(_HI_MASK))
  return lo, hi


def _load_pair(x_u32, rows_per_tok: int, t0: int, c: int, sub: int):
  """Returns columns ``c`` and ``c + 1`` (bf16 ``[sub, 128]`` each) of tokens.

  ``x_u32`` is ``_u32_flat`` of a ``[tokens, d0, 128]`` ref; the columns hold
  rows ``c`` and ``c + 1`` of tokens ``t0 .. t0 + sub`` with tokens on
  sublanes.
  """
  if rows_per_tok is None:
    v = x_u32[pl.ds(t0, sub), pl.ds(c, 2), :]
    return v[:, 0, :], v[:, 1, :]
  stride = 2 * rows_per_tok
  w0 = x_u32[pl.ds(t0 * rows_per_tok + c // 2, sub // 2, stride=stride), :]
  w1 = x_u32[
      pl.ds((t0 + 1) * rows_per_tok + c // 2, sub // 2, stride=stride), :
  ]
  lo, hi = _transpose_halves(w0, w1)
  return pltpu.bitcast(lo, jnp.bfloat16), pltpu.bitcast(hi, jnp.bfloat16)


def _store_pair(x_u32, rows_per_tok: int, t0: int, c: int, sub: int, lo, hi):
  """Inverse of ``_load_pair``."""
  if rows_per_tok is None:
    x_u32[pl.ds(t0, sub), pl.ds(c, 2), :] = jnp.stack([lo, hi], axis=1)
    return
  stride = 2 * rows_per_tok
  w0, w1 = _transpose_halves(
      pltpu.bitcast(lo, jnp.uint32), pltpu.bitcast(hi, jnp.uint32)
  )
  x_u32[pl.ds(t0 * rows_per_tok + c // 2, sub // 2, stride=stride), :] = w0
  x_u32[pl.ds((t0 + 1) * rows_per_tok + c // 2, sub // 2, stride=stride), :] = (
      w1
  )


def _relayout_3d_to_2d(src3d, dst2d, num_rows: int, interpret: bool) -> None:
  """Copies VMEM ``src3d[:num_rows]`` (``[n, d0, 128]``) to ``dst2d`` (``[n, D]``)."""
  x_u32, rows_per_tok = _u32_flat(src3d, interpret)
  d0 = src3d.shape[-2]
  sub = min(_RELAYOUT_ROWS, num_rows)
  assert num_rows % sub == 0, (num_rows, sub)
  for t0 in range(0, num_rows, sub):
    for c in range(0, d0, 2):
      lo, hi = _load_pair(x_u32, rows_per_tok, t0, c, sub)
      dst2d[pl.ds(t0, sub), pl.ds(c * LANES, LANES)] = lo
      dst2d[pl.ds(t0, sub), pl.ds((c + 1) * LANES, LANES)] = hi


# --- Dispatch gather with a (cap, D) output ---


def ragged_gather_2d(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array,
    num_tokens: jax.Array,
    max_out_tokens: int,
    block_size: int = 512,
    interpret: bool = False,
) -> jax.Array:
  """``out[i] = x[indices[start + i]].reshape(D)`` for ``i < num_tokens``.

  Like ``ragged_gather_tc.ragged_gather_tc`` but returns ``[max_out_tokens,
  D]``: each block of gathered ``[block, D / 128, 128]`` rows is relayouted in
  VMEM and written as whole ``[block, D]`` row blocks, so rows ``>=
  num_tokens`` of the last written block hold unspecified values and later
  blocks are not written at all.

  Args:
    x: ``[G, D / 128, 128]`` bf16 source tokens.
    indices: ``[M]`` int32 token index of each route.
    start: First route of the window.
    num_tokens: Number of routes to gather (``<= max_out_tokens``).
    max_out_tokens: Output rows (a multiple of ``block_size``).
    block_size: Rows per grid step.
    interpret: Run the Pallas kernel in interpret mode (tests).

  Returns:
    ``[max_out_tokens, D]`` bf16.
  """
  _, d0, lanes = x.shape
  if lanes != LANES:
    raise ValueError(f"x must be [G, D / 128, 128], got {x.shape}")
  d_dim = d0 * LANES
  block = min(block_size, max_out_tokens)
  if max_out_tokens % block:
    raise ValueError(
        f"max_out_tokens ({max_out_tokens}) must be a multiple of the block"
        f" size ({block})"
    )
  num_indices = indices.shape[0]
  idx_window = pl.cdiv(block, _GATHER_IDX_ALIGN) * _GATHER_IDX_ALIGN + (
      _GATHER_IDX_ALIGN
  )
  idx_pad = max((-num_indices) % _GATHER_IDX_ALIGN, idx_window - num_indices)
  if idx_pad > 0:
    indices = jnp.pad(indices, (0, idx_pad))
  max_win_start = indices.shape[0] - idx_window

  def kernel(
      scalars_ref,
      idx_hbm_ref,
      x_hbm_ref,
      o_hbm_ref,
      buf3d_ref,
      buf2d_ref,
      idx_smem_ref,
      idx_sem,
      recv_sems,
      send_sems,
  ):
    step = pl.program_id(0)
    num_steps = pl.num_programs(0)
    slot = step % 2
    start_idx = scalars_ref[0]
    count = scalars_ref[1]

    def _num_valid(s):
      return jnp.clip(count - s * block, 0, block)

    def _fetch(s, b):
      row0 = start_idx + s * block
      win = jnp.minimum(
          row0 // _GATHER_IDX_ALIGN * _GATHER_IDX_ALIGN, max_win_start
      )
      off = row0 - win
      copy = pltpu.make_async_copy(
          idx_hbm_ref.at[
              pl.ds(pl.multiple_of(win, _GATHER_IDX_ALIGN), idx_window)
          ],
          idx_smem_ref,
          idx_sem,
      )
      copy.start()
      copy.wait()
      n = _num_valid(s)
      sem = recv_sems.at[b]

      def _copy(src_row, dst_row):
        pltpu.make_async_copy(
            x_hbm_ref.at[pl.ds(src_row, 1)],
            buf3d_ref.at[b, pl.ds(dst_row, 1)],
            sem,
        ).start()

      unroll = _GATHER_ISSUE_UNROLL
      num_full = n // unroll

      @pl.loop(0, num_full)
      def _(i):
        base = i * unroll
        src_rows = [idx_smem_ref[off + base + u] for u in range(unroll)]
        for u in range(unroll):
          _copy(src_rows[u], base + u)

      @pl.loop(num_full * unroll, n)
      def _(i):
        _copy(idx_smem_ref[off + i], i)

    def _wait_recv(s, b):
      n = _num_valid(s)
      for k in reversed(range(block.bit_length())):
        size = 1 << k

        @pl.when((n & size) != 0)
        def _(size=size):
          rows = buf3d_ref.at[b, pl.ds(0, size)]
          pltpu.make_async_copy(rows, rows, recv_sems.at[b]).wait()

    def _send(s, b):
      return pltpu.make_async_copy(
          buf2d_ref.at[b],
          o_hbm_ref.at[pl.ds(pl.multiple_of(s * block, block), block)],
          send_sems.at[b],
      )

    @pl.when(step == 0)
    def _():
      _fetch(0, 0)

    _wait_recv(step, slot)

    @pl.when(step + 1 < num_steps)
    def _():
      _fetch(step + 1, 1 - slot)

    @pl.when(step >= 2)
    def _():
      _send(step - 2, slot).wait()

    _relayout_3d_to_2d(buf3d_ref.at[slot], buf2d_ref.at[slot], block, interpret)
    _send(step, slot).start()

    @pl.when(step == num_steps - 1)
    def _():
      @pl.when(step >= 1)
      def _():
        _send(step - 1, 1 - slot).wait()

      _send(step, slot).wait()

  scalars = jnp.stack([
      jnp.asarray(start, dtype=jnp.int32),
      jnp.asarray(num_tokens, dtype=jnp.int32),
  ])
  num_steps = pl.cdiv(scalars[1], block)
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((max_out_tokens, d_dim), x.dtype),
      grid=(num_steps,),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
      scratch_shapes=[
          pltpu.VMEM((2, block, d0, LANES), x.dtype),
          pltpu.VMEM((2, block, d_dim), x.dtype),
          pltpu.SMEM((idx_window,), indices.dtype),
          pltpu.SemaphoreType.DMA(()),
          pltpu.SemaphoreType.DMA((2,)),
          pltpu.SemaphoreType.DMA((2,)),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(pltpu.ARBITRARY,),
          vmem_limit_bytes=_vmem_limit(interpret),
      ),
      interpret=_interp(interpret),
      name="roe_gather_2d",
  )(scalars, indices, x)


def _vmem_limit(interpret: bool) -> int | None:
  return None if interpret else pallas_vmem.vmem_limit_bytes()


def _interp(interpret: bool):
  return pltpu.InterpretParams() if interpret else False


# --- Grouped matmuls with a (rows, N / 128, 128) output ---


def _row_mask(group_metadata, grid_id, tile_m: int, t0: int, sub: int):
  """``[sub, 128]`` mask of tile rows ``t0 .. t0 + sub`` in the tile's group."""
  group_offsets, group_ids, m_tile_ids = group_metadata
  group_id = group_ids[grid_id]
  group_start = group_offsets[group_id]
  group_end = group_offsets[group_id + 1]
  row0 = m_tile_ids[grid_id] * tile_m + t0
  rows = lax.broadcasted_iota(jnp.int32, (sub, LANES), 0) + row0
  return (rows >= group_start) & (rows < group_end)


def _first_visit(group_metadata, grid_id):
  m_tile_ids = group_metadata[2]
  prev_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
  return jnp.logical_or(
      grid_id == 0, m_tile_ids[grid_id] != m_tile_ids[prev_grid_id]
  )


def _masked_store_3d(
    out_ref,
    acc_ref,
    base_fn,
    group_metadata,
    grid_id,
    *,
    tile_m: int,
    round_acc: bool,
    interpret: bool,
) -> None:
  """``out = where(row in group, base + acc, base)`` as bf16, ``out`` 3D.

  ``base_fn(t0, c, sub, first_visit_branch)`` returns the FP32 base columns
  ``(c, c + 1)``; on a tile's first visit it is the seed (zeros or a 2D
  input), on later visits the current output.
  """
  first_visit = _first_visit(group_metadata, grid_id)
  out_u32, rows_per_tok = _u32_flat(out_ref, interpret)
  d0 = out_ref.shape[-2]
  sub = min(_RELAYOUT_ROWS, tile_m)

  def _run(is_first: bool):
    for t0 in range(0, tile_m, sub):
      mask = _row_mask(group_metadata, grid_id, tile_m, t0, sub)
      for c in range(0, d0, 2):
        if is_first:
          base_lo, base_hi = base_fn(t0, c, sub)
        else:
          prev_lo, prev_hi = _load_pair(out_u32, rows_per_tok, t0, c, sub)
          base_lo = prev_lo.astype(jnp.float32)
          base_hi = prev_hi.astype(jnp.float32)
        vals = []
        for cc, base in ((c, base_lo), (c + 1, base_hi)):
          acc = acc_ref[pl.ds(t0, sub), pl.ds(cc * LANES, LANES)]
          if round_acc:
            acc = acc.astype(jnp.bfloat16).astype(jnp.float32)
          if base is None:
            val = jnp.where(mask, acc, 0.0)
          else:
            val = jnp.where(mask, base + acc, base)
          vals.append(val.astype(jnp.bfloat16))
        _store_pair(out_u32, rows_per_tok, t0, c, sub, vals[0], vals[1])

  @pl.when(first_visit)
  def _():
    _run(True)

  @pl.when(jnp.logical_not(first_visit))
  def _():
    _run(False)


_FAST_N_CHUNK = 512


def _full_tile(group_metadata, grid_id, tile_m: int):
  """Whether the grid step's whole row tile lies inside its group."""
  group_offsets, group_ids, m_tile_ids = group_metadata
  group_id = group_ids[grid_id]
  row0 = m_tile_ids[grid_id] * tile_m
  return jnp.logical_and(
      group_offsets[group_id] <= row0,
      row0 + tile_m <= group_offsets[group_id + 1],
  )


def _fast_store_3d(out_ref, part_fn, n: int, tile_m: int, interpret: bool):
  """Unmasked store of a fully-in-group tile, ``N`` chunk by chunk.

  ``part_fn(nc, width)`` returns the bf16 ``[tile_m, width]`` output columns
  ``nc .. nc + width``. Computing one chunk and relayouting it in the same
  basic block lets the MXU work of the next chunk overlap the VPU relayout.
  """
  out_u32, rows_per_tok = _u32_flat(out_ref, interpret)
  width = min(_FAST_N_CHUNK, n)
  sub = min(_RELAYOUT_ROWS, tile_m)
  for nc in range(0, n, width):
    part = part_fn(nc, width)
    for t0 in range(0, tile_m, sub):
      for cl in range(0, width // LANES, 2):
        lo = part[t0 : t0 + sub, cl * LANES : (cl + 1) * LANES]
        hi = part[t0 : t0 + sub, (cl + 1) * LANES : (cl + 2) * LANES]
        _store_pair(out_u32, rows_per_tok, t0, nc // LANES + cl, sub, lo, hi)


def gmm_out3d(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    tile_m: int,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> jax.Array:
  """Grouped ``lhs @ rhs[g]`` written as ``[m, N / 128, 128]`` bf16.

  Rows in no group (``>= sum(group_sizes)``) of visited tiles are zero; tiles
  past ``sum(group_sizes)`` are not written.
  """
  m, k = lhs.shape
  if transpose_rhs:
    groups, n, rhs_k = rhs.shape
  else:
    groups, rhs_k, n = rhs.shape
  if rhs_k != k or m % tile_m or n % (8 * LANES):
    raise ValueError(f"Bad gmm_out3d shapes {lhs.shape} {rhs.shape} {tile_m}")
  group_metadata, num_active_tiles = psm._metadata(  # pylint: disable=protected-access
      group_sizes, m=m, tile_m=tile_m, groups=groups, visit_empty_groups=False
  )
  group_offset = jnp.array([0], dtype=jnp.int32)
  d0 = n // LANES

  def kernel(group_metadata, group_offset, lhs_ref, rhs_ref, out_ref, acc_ref):
    del group_offset
    grid_id = pl.program_id(1)
    full = _full_tile(group_metadata, grid_id, tile_m)

    def _dot(nc, width):
      if transpose_rhs:
        return lax.dot_general(
            lhs_ref[...],
            rhs_ref[pl.ds(nc, width), :],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
      return lax.dot(
          lhs_ref[...],
          rhs_ref[:, pl.ds(nc, width)],
          preferred_element_type=jnp.float32,
      )

    @pl.when(full)
    def _():
      _fast_store_3d(
          out_ref,
          lambda nc, width: _dot(nc, width).astype(jnp.bfloat16),
          n,
          tile_m,
          interpret,
      )

    @pl.when(jnp.logical_not(full))
    def _():
      acc_ref[...] = _dot(0, n)
      _masked_store_3d(
          out_ref,
          acc_ref,
          lambda t0, c, sub: (None, None),
          group_metadata,
          grid_id,
          tile_m=tile_m,
          round_acc=False,
          interpret=interpret,
      )

  def lhs_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[2][grid_id], 0

  def rhs_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[1][grid_id], 0, 0

  def out_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[2][grid_id], 0, 0

  rhs_block = (None, n, k) if transpose_rhs else (None, k, n)
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, d0, LANES), jnp.bfloat16),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[
              pl.BlockSpec((tile_m, k), lhs_indices),
              pl.BlockSpec(rhs_block, rhs_indices),
          ],
          out_specs=pl.BlockSpec((tile_m, d0, LANES), out_indices),
          grid=(1, num_active_tiles, 1),
          scratch_shapes=[pltpu.VMEM((tile_m, n), jnp.float32)],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          vmem_limit_bytes=_vmem_limit(interpret),
      ),
      interpret=_interp(interpret),
      name="moe_gmm_w2_fwd_3d",
  )(group_metadata, group_offset, lhs, rhs)


def _accumulate_dinput_3d(
    dout, rhs, existing, group_sizes, *, tile_m: int, interpret: bool = False
):
  """``existing + bf16(dout @ rhs^T)`` per group, written ``[m, K / 128, 128]``.

  Bit-identical to ``paired_segment_matmul._accumulate_dinput`` reshaped to
  3D on the rows of the groups.
  """
  m, n = dout.shape
  groups, k, _ = rhs.shape
  if k % (8 * LANES):
    raise ValueError(f"K ({k}) must be a multiple of {8 * LANES}")
  group_metadata, num_active_tiles = psm._metadata(  # pylint: disable=protected-access
      group_sizes, m=m, tile_m=tile_m, groups=groups, visit_empty_groups=False
  )
  group_offset = jnp.array([0], dtype=jnp.int32)
  d0 = k // LANES

  def kernel(
      group_metadata,
      group_offset,
      dout_ref,
      rhs_ref,
      existing_ref,
      out_ref,
      acc_ref,
  ):
    del group_offset
    grid_id = pl.program_id(1)
    full = _full_tile(group_metadata, grid_id, tile_m)

    def _dot(nc, width):
      return lax.dot_general(
          dout_ref[...],
          rhs_ref[pl.ds(nc, width), :],
          (((1,), (1,)), ((), ())),
          preferred_element_type=jnp.float32,
      )

    def _fast_part(nc, width):
      prod = _dot(nc, width).astype(jnp.bfloat16).astype(jnp.float32)
      base = existing_ref[:, pl.ds(nc, width)].astype(jnp.float32)
      return (base + prod).astype(jnp.bfloat16)

    @pl.when(full)
    def _():
      _fast_store_3d(out_ref, _fast_part, k, tile_m, interpret)

    def _seed(t0, c, sub):
      return tuple(
          existing_ref[pl.ds(t0, sub), pl.ds(cc * LANES, LANES)].astype(
              jnp.float32
          )
          for cc in (c, c + 1)
      )

    @pl.when(jnp.logical_not(full))
    def _():
      acc_ref[...] = _dot(0, k)
      _masked_store_3d(
          out_ref,
          acc_ref,
          _seed,
          group_metadata,
          grid_id,
          tile_m=tile_m,
          round_acc=True,
          interpret=interpret,
      )

  def row_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[2][grid_id], 0

  def rhs_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[1][grid_id], 0, 0

  def out_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[2][grid_id], 0, 0

  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, d0, LANES), existing.dtype),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[
              pl.BlockSpec((tile_m, n), row_indices),
              pl.BlockSpec((None, k, n), rhs_indices),
              pl.BlockSpec((tile_m, k), row_indices),
          ],
          out_specs=pl.BlockSpec((tile_m, d0, LANES), out_indices),
          grid=(1, num_active_tiles, 1),
          scratch_shapes=[pltpu.VMEM((tile_m, k), jnp.float32)],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          vmem_limit_bytes=_vmem_limit(interpret),
      ),
      interpret=_interp(interpret),
      name="moe_gmm_w13_dgrad_accum_3d",
  )(group_metadata, group_offset, dout, rhs, existing)


def _gather_reduce(
    x3d,
    src_rows,
    tokens,
    blocks,
    *,
    num_indices,
    num_out_tokens,
    top_k,
    reduce_block_size,
):
  meta = rgt.RaggedGatherReduceMetadata(
      src_rows=src_rows,
      tokens=tokens,
      blocks=blocks,
      block_size=reduce_block_size,
      num_indices=num_indices,
  )
  return rgt.ragged_gather_reduce_tc(
      x3d, meta, num_out_tokens=num_out_tokens, top_k=top_k, weights=None
  )


# --- JAX implementations of the fused ops (also used by the CPU tests) ---


def w13_fwd_impl(
    x_all3d,
    token_ids_sorted_M,
    start_count_2,
    w1,
    w3,
    group_sizes,
    *,
    cap,
    gather_block_size,
    tile_m,
    interpret=False,
):
  """Dispatch gather + paired W1/W3 grouped matmul; returns (gate, up, lhs)."""
  lhs = ragged_gather_2d(
      x_all3d,
      token_ids_sorted_M,
      start_count_2[0],
      start_count_2[1],
      cap,
      block_size=gather_block_size,
      interpret=interpret,
  )
  if interpret:
    gate, up = _paired_forward_interpret(lhs, w1, w3, group_sizes, tile_m)
  else:
    gate, up = psm._paired_forward(lhs, w1, w3, group_sizes, tile_m=tile_m)  # pylint: disable=protected-access
  return gate, up, lhs


def _paired_forward_interpret(lhs, w1, w3, group_sizes, tile_m):
  gate = megablox_gmm(
      lhs,
      w1,
      group_sizes,
      preferred_element_type=lhs.dtype,
      tiling=(tile_m, w1.shape[1], w1.shape[2]),
      interpret=True,
  )
  up = megablox_gmm(
      lhs,
      w3,
      group_sizes,
      preferred_element_type=lhs.dtype,
      tiling=(tile_m, w3.shape[1], w3.shape[2]),
      interpret=True,
  )
  return gate, up


def w13_bwd_impl(
    dgate,
    dup,
    lhs,
    w1,
    w3,
    group_sizes,
    src_rows,
    tokens,
    blocks,
    *,
    num_indices,
    num_out_tokens,
    top_k,
    reduce_block_size,
    tile_m,
    tgmm_tile_m,
    tgmm_tile_k,
    interpret=False,
):
  """Returns (dx_all3d, dw1, dw3)."""
  k, n = w1.shape[1:]
  grad_lhs = call_named(
      megablox_gmm,
      "moe_gmm_w1_dgrad",
      dgate,
      w1,
      group_sizes=group_sizes,
      preferred_element_type=dgate.dtype,
      tiling=(tile_m, n, k),
      transpose_rhs=True,
      interpret=interpret,
  )
  dlhs3d = _accumulate_dinput_3d(
      dup, w3, grad_lhs, group_sizes, tile_m=tile_m, interpret=interpret
  )
  dx_all3d = _gather_reduce(
      dlhs3d,
      src_rows,
      tokens,
      blocks,
      num_indices=num_indices,
      num_out_tokens=num_out_tokens,
      top_k=top_k,
      reduce_block_size=reduce_block_size,
  )
  if interpret:
    dw1 = megablox_tgmm(
        lhs.swapaxes(0, 1),
        dgate,
        group_sizes,
        preferred_element_type=w1.dtype,
        tiling=(tgmm_tile_m, k, n),
        interpret=True,
    )
    dw3 = megablox_tgmm(
        lhs.swapaxes(0, 1),
        dup,
        group_sizes,
        preferred_element_type=w3.dtype,
        tiling=(tgmm_tile_m, k, n),
        interpret=True,
    )
  else:
    dw1, dw3 = psm._paired_dweight(  # pylint: disable=protected-access
        lhs, dgate, dup, group_sizes, tile_m=tgmm_tile_m, tile_k=tgmm_tile_k
    )
  return dx_all3d, dw1, dw3


def w2_fwd_impl(
    hidden,
    w2,
    group_sizes,
    src_rows,
    tokens,
    blocks,
    *,
    num_indices,
    num_out_tokens,
    top_k,
    reduce_block_size,
    tile_m,
    transpose_rhs,
    interpret=False,
):
  """W2 grouped matmul (3D output) + unweighted combine gather-reduce."""
  y3d = gmm_out3d(
      hidden,
      w2,
      group_sizes,
      tile_m=tile_m,
      transpose_rhs=transpose_rhs,
      interpret=interpret,
  )
  return _gather_reduce(
      y3d,
      src_rows,
      tokens,
      blocks,
      num_indices=num_indices,
      num_out_tokens=num_out_tokens,
      top_k=top_k,
      reduce_block_size=reduce_block_size,
  )


def _tiling(tile_m, tile_k, tile_n, k_dim, n_dim):
  return (
      tile_m,
      tile_k if tile_k > 0 else k_dim,
      tile_n if tile_n > 0 else n_dim,
  )


def w2_bwd_impl(
    grad_partial3d,
    hidden,
    w2,
    group_sizes,
    token_ids_sorted_M,
    start_count_2,
    *,
    cap,
    gather_block_size,
    tile_m,
    tgmm_tile_m,
    tgmm_tile_k,
    tgmm_tile_n,
    transpose_rhs,
    interpret=False,
):
  """Returns (dhidden, dw2); mirrors ``expert_segment_matmul._grouped_mm_bwd``."""
  dout = ragged_gather_2d(
      grad_partial3d,
      token_ids_sorted_M,
      start_count_2[0],
      start_count_2[1],
      cap,
      block_size=gather_block_size,
      interpret=interpret,
  )
  lhs = hidden
  if transpose_rhs:
    _, n_dim, k_dim = w2.shape
    grad_lhs = call_named(
        megablox_gmm,
        "moe_gmm_dgrad",
        dout,
        w2,
        group_sizes=group_sizes,
        preferred_element_type=dout.dtype,
        tiling=_tiling(tile_m, 0, 0, n_dim, k_dim),
        interpret=interpret,
    )
    grad_rhs = call_named(
        megablox_tgmm,
        "moe_tgmm_wgrad",
        jnp.swapaxes(dout, 0, 1),
        lhs,
        group_sizes=group_sizes,
        preferred_element_type=w2.dtype,
        tiling=_tiling(
            tgmm_tile_m, tgmm_tile_n, tgmm_tile_k, dout.shape[1], lhs.shape[1]
        ),
        interpret=interpret,
    )
  else:
    _, k_dim, n_dim = w2.shape
    grad_lhs = call_named(
        megablox_gmm,
        "moe_gmm_dgrad",
        dout,
        w2,
        group_sizes=group_sizes,
        preferred_element_type=dout.dtype,
        tiling=_tiling(tile_m, 0, 0, n_dim, k_dim),
        transpose_rhs=True,
        interpret=interpret,
    )
    grad_rhs = call_named(
        megablox_tgmm,
        "moe_tgmm_wgrad",
        jnp.swapaxes(lhs, 0, 1),
        dout,
        group_sizes=group_sizes,
        preferred_element_type=w2.dtype,
        tiling=_tiling(
            tgmm_tile_m, tgmm_tile_k, tgmm_tile_n, lhs.shape[1], dout.shape[1]
        ),
        interpret=interpret,
    )
  return grad_lhs, grad_rhs


# --- PyTorch <-> JAX custom operators ---


@pallas.jax_op("tt_moe::roe_fused_w13_fwd")
def _roe_fused_w13_fwd(
    x_all3d: jax.Array,
    token_ids_sorted_M: jax.Array,
    start_count_2: jax.Array,
    w1: jax.Array,
    w3: jax.Array,
    group_sizes: jax.Array,
    cap: int,
    gather_block_size: int,
    tile_m: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  return w13_fwd_impl(
      x_all3d,
      token_ids_sorted_M,
      start_count_2,
      w1,
      w3,
      group_sizes,
      cap=cap,
      gather_block_size=gather_block_size,
      tile_m=tile_m,
  )


@pallas.jax_op("tt_moe::roe_fused_w13_bwd")
def _roe_fused_w13_bwd(
    dgate: jax.Array,
    dup: jax.Array,
    lhs: jax.Array,
    w1: jax.Array,
    w3: jax.Array,
    group_sizes: jax.Array,
    src_rows: jax.Array,
    tokens: jax.Array,
    blocks: jax.Array,
    num_indices: int,
    num_out_tokens: int,
    top_k: int,
    reduce_block_size: int,
    tile_m: int,
    tgmm_tile_m: int,
    tgmm_tile_k: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  return w13_bwd_impl(
      dgate,
      dup,
      lhs,
      w1,
      w3,
      group_sizes,
      src_rows,
      tokens,
      blocks,
      num_indices=num_indices,
      num_out_tokens=num_out_tokens,
      top_k=top_k,
      reduce_block_size=reduce_block_size,
      tile_m=tile_m,
      tgmm_tile_m=tgmm_tile_m,
      tgmm_tile_k=tgmm_tile_k,
  )


@pallas.jax_op("tt_moe::roe_fused_w2_fwd")
def _roe_fused_w2_fwd(
    hidden: jax.Array,
    w2: jax.Array,
    group_sizes: jax.Array,
    src_rows: jax.Array,
    tokens: jax.Array,
    blocks: jax.Array,
    num_indices: int,
    num_out_tokens: int,
    top_k: int,
    reduce_block_size: int,
    tile_m: int,
    transpose_rhs: bool,
) -> jax.Array:
  return w2_fwd_impl(
      hidden,
      w2,
      group_sizes,
      src_rows,
      tokens,
      blocks,
      num_indices=num_indices,
      num_out_tokens=num_out_tokens,
      top_k=top_k,
      reduce_block_size=reduce_block_size,
      tile_m=tile_m,
      transpose_rhs=transpose_rhs,
  )


@pallas.jax_op("tt_moe::roe_fused_w2_bwd")
def _roe_fused_w2_bwd(
    grad_partial3d: jax.Array,
    hidden: jax.Array,
    w2: jax.Array,
    group_sizes: jax.Array,
    token_ids_sorted_M: jax.Array,
    start_count_2: jax.Array,
    cap: int,
    gather_block_size: int,
    tile_m: int,
    tgmm_tile_m: int,
    tgmm_tile_k: int,
    tgmm_tile_n: int,
    transpose_rhs: bool,
) -> tuple[jax.Array, jax.Array]:
  return w2_bwd_impl(
      grad_partial3d,
      hidden,
      w2,
      group_sizes,
      token_ids_sorted_M,
      start_count_2,
      cap=cap,
      gather_block_size=gather_block_size,
      tile_m=tile_m,
      tgmm_tile_m=tgmm_tile_m,
      tgmm_tile_k=tgmm_tile_k,
      tgmm_tile_n=tgmm_tile_n,
      transpose_rhs=transpose_rhs,
  )


class _RoEFusedW13(torch.autograd.Function):
  """Dispatch gather fused with the paired W1/W3 grouped matmul."""

  @staticmethod
  def forward(
      ctx,
      x_all3d,
      w1,
      w3,
      token_ids_sorted_M,
      start_count_2,
      group_sizes,
      src_rows,
      tokens,
      blocks,
      cap,
      top_k,
      gather_block_size,
      reduce_block_size,
      tile_m,
      tgmm_tile_m,
      tgmm_tile_k,
  ):
    gate, up, lhs = _roe_fused_w13_fwd(
        x_all3d.contiguous(),
        token_ids_sorted_M,
        start_count_2,
        w1,
        w3,
        group_sizes,
        cap,
        gather_block_size,
        tile_m,
    )
    ctx.save_for_backward(lhs, w1, w3, group_sizes, src_rows, tokens, blocks)
    ctx.cfg = (
        token_ids_sorted_M.shape[0],
        x_all3d.shape[0],
        top_k,
        reduce_block_size,
        tile_m,
        tgmm_tile_m,
        tgmm_tile_k,
    )
    return gate, up

  @staticmethod
  def backward(ctx, dgate, dup):
    lhs, w1, w3, group_sizes, src_rows, tokens, blocks = ctx.saved_tensors
    dx, dw1, dw3 = _roe_fused_w13_bwd(
        dgate.contiguous(),
        dup.contiguous(),
        lhs,
        w1,
        w3,
        group_sizes,
        src_rows,
        tokens,
        blocks,
        *ctx.cfg,
    )
    return (dx, dw1, dw3) + (None,) * 13


class _RoEFusedW2Combine(torch.autograd.Function):
  """W2 grouped matmul fused with the unweighted RoE combine gather-reduce."""

  @staticmethod
  def forward(
      ctx,
      hidden,
      w2,
      group_sizes,
      token_ids_sorted_M,
      start_count_2,
      src_rows,
      tokens,
      blocks,
      num_out_tokens,
      top_k,
      gather_block_size,
      reduce_block_size,
      tile_m,
      tgmm_tiling,
      transpose_rhs,
  ):
    partial = _roe_fused_w2_fwd(
        hidden.contiguous(),
        w2,
        group_sizes,
        src_rows,
        tokens,
        blocks,
        token_ids_sorted_M.shape[0],
        num_out_tokens,
        top_k,
        reduce_block_size,
        tile_m,
        transpose_rhs,
    )
    ctx.save_for_backward(
        hidden, w2, group_sizes, token_ids_sorted_M, start_count_2
    )
    ctx.cfg = (
        hidden.shape[0],
        gather_block_size,
        tile_m,
        *tgmm_tiling,
        transpose_rhs,
    )
    return partial

  @staticmethod
  def backward(ctx, grad_partial):
    hidden, w2, group_sizes, token_ids_sorted_M, start_count_2 = (
        ctx.saved_tensors
    )
    dh, dw2 = _roe_fused_w2_bwd(
        grad_partial.contiguous(),
        hidden,
        w2,
        group_sizes,
        token_ids_sorted_M,
        start_count_2,
        *ctx.cfg,
    )
    return (dh, dw2) + (None,) * 13


def roe_fused_w13(
    x_all3d: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    token_ids_sorted_M: torch.Tensor,
    start_count_2: torch.Tensor,
    group_sizes: torch.Tensor,
    src_rows: torch.Tensor,
    tokens: torch.Tensor,
    blocks: torch.Tensor,
    *,
    cap: int,
    top_k: int,
    gather_block_size: int,
    reduce_block_size: int,
    tile_m: int,
    tgmm_tile_m: int,
    tgmm_tile_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Gathers this rank's routes from ``x_all3d`` and applies W1 and W3.

  Equivalent to ``roe_sort_gather`` followed by
  ``paired_segment_matmul_pallas_2d``, without the relayout copies.
  """
  if group_sizes.dtype != torch.int32:
    group_sizes = group_sizes.to(torch.int32)
  return _RoEFusedW13.apply(
      x_all3d,
      w1,
      w3,
      token_ids_sorted_M,
      start_count_2,
      group_sizes,
      src_rows,
      tokens,
      blocks,
      cap,
      top_k,
      gather_block_size,
      reduce_block_size,
      tile_m,
      tgmm_tile_m,
      tgmm_tile_k,
  )


def roe_fused_w2_combine(
    hidden: torch.Tensor,
    w2: torch.Tensor,
    group_sizes: torch.Tensor,
    token_ids_sorted_M: torch.Tensor,
    start_count_2: torch.Tensor,
    src_rows: torch.Tensor,
    tokens: torch.Tensor,
    blocks: torch.Tensor,
    *,
    num_out_tokens: int,
    top_k: int,
    gather_block_size: int,
    reduce_block_size: int,
    tile_m: int,
    tgmm_tiling: tuple[int, int, int],
    transpose_rhs: bool,
) -> torch.Tensor:
  """W2 then the unweighted combine reduce, to ``[G, D / 128, 128]``.

  Equivalent to ``segment_matmul_pallas_2d`` followed by
  ``roe_unsort_reduce``, without the relayout copies.
  """
  if group_sizes.dtype != torch.int32:
    group_sizes = group_sizes.to(torch.int32)
  return _RoEFusedW2Combine.apply(
      hidden,
      w2,
      group_sizes,
      token_ids_sorted_M,
      start_count_2,
      src_rows,
      tokens,
      blocks,
      num_out_tokens,
      top_k,
      gather_block_size,
      reduce_block_size,
      tile_m,
      tuple(tgmm_tiling),
      transpose_rhs,
  )


__all__ = [
    "gmm_out3d",
    "ragged_gather_2d",
    "roe_fused_w13",
    "roe_fused_w2_combine",
]
