"""Paired MegaBlox grouped matmul for expert gate/up projections.

Calling grouped matmul separately for W1 and W3 reloads the same routed-token
rows and walks the same expert metadata twice. This kernel keeps the canonical
weights and outputs separate while sharing that work in forward and dWeight.
dInput keeps two grouped matmuls, but the second adds into the first's output
in place instead of materializing a second routed-row tensor, bit-exactly.

The implementation intentionally targets the exact-row, BF16 Qwen TPU path.
It uses full N tiles and requires the total routed-row count to be a multiple
of ``tile_m``. The dWeight kernel keeps two ``(K, N)`` FP32 accumulators and
two BF16 output slabs in VMEM, so at large ``K x N`` (Qwen3-235B: 4096 x 1536
is 105 MB) it tiles K with ``tgmm_tile_k``; the K slabs are independent, each
re-reads the dOutput rows once.
"""

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.megablox import ops as gmm_ops
from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm as megablox_gmm
from torch_tpu._internal import pallas
from torchtitan.experiments.tpu.kernels.pallas_naming import call_named


# Keep the paired kernel on MegaBlox's exact metadata and boundary-mask
# contract. MegaBlox currently exposes these helpers only from its backend.
_make_group_metadata = gmm_ops.backend.make_group_metadata
_get_group_size = gmm_ops.backend._get_group_size
_get_store_mask = gmm_ops.backend._get_store_mask


def _validate_shapes(lhs, rhs1, rhs3, group_sizes, tile_m):
    if tile_m <= 0:
        raise ValueError("tile_m must be positive")
    if lhs.ndim != 2 or rhs1.ndim != 3 or rhs3.ndim != 3:
        raise ValueError("Paired grouped matmul expects 2D lhs and 3D weights")
    if group_sizes.ndim != 1 or group_sizes.dtype != jnp.int32:
        raise ValueError("group_sizes must be a 1D int32 tensor")
    if rhs1.shape != rhs3.shape:
        raise ValueError("Paired grouped matmul weights must have equal shapes")
    if rhs1.shape[0] == 0:
        raise ValueError("Paired grouped matmul requires at least one expert")
    if lhs.shape[0] == 0:
        raise ValueError("Paired grouped matmul requires at least one routed row")
    if lhs.shape[1] != rhs1.shape[1]:
        raise ValueError("Paired grouped matmul contraction dimensions must match")
    if rhs1.shape[0] != group_sizes.shape[0]:
        raise ValueError("group_sizes must have one entry per expert")
    if lhs.shape[0] % tile_m:
        raise ValueError("The total row count must be divisible by tile_m")
    if (
        lhs.dtype != jnp.bfloat16
        or rhs1.dtype != jnp.bfloat16
        or rhs3.dtype != jnp.bfloat16
    ):
        raise ValueError("Paired grouped matmul currently supports bfloat16 inputs")


def _metadata(group_sizes, *, m, tile_m, groups, visit_empty_groups):
    return _make_group_metadata(
        group_sizes=group_sizes,
        m=m,
        tm=tile_m,
        start_group=jnp.array(0, dtype=jnp.int32),
        num_nonzero_groups=groups,
        visit_empty_groups=visit_empty_groups,
    )


def _paired_forward(lhs, rhs1, rhs3, group_sizes, *, tile_m):
    """Compute the two projections from one routed-token tile load."""
    _validate_shapes(lhs, rhs1, rhs3, group_sizes, tile_m)
    m, k = lhs.shape
    groups, _, n = rhs1.shape
    group_metadata, num_active_tiles = _metadata(
        group_sizes,
        m=m,
        tile_m=tile_m,
        groups=groups,
        visit_empty_groups=False,
    )
    group_offset = jnp.array([0], dtype=jnp.int32)

    def kernel(
        group_metadata,
        group_offset,
        lhs,
        rhs1,
        rhs3,
        out1,
        out3,
        acc1,
        acc3,
    ):
        del group_offset
        grid_id = pl.program_id(1)

        acc1[...] = jnp.zeros_like(acc1)
        acc3[...] = jnp.zeros_like(acc3)
        loaded_lhs = lhs[...]
        acc1[...] += lax.dot(
            loaded_lhs,
            rhs1[...],
            preferred_element_type=jnp.float32,
        )
        acc3[...] += lax.dot(
            loaded_lhs,
            rhs3[...],
            preferred_element_type=jnp.float32,
        )

        mask = _get_store_mask(
            grid_id=grid_id,
            group_metadata=group_metadata,
            tm=tile_m,
            tn=n,
        )
        out1[...] = lax.select(
            mask[...], acc1[...], out1[...].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        out3[...] = lax.select(
            mask[...], acc3[...], out3[...].astype(jnp.float32)
        ).astype(jnp.bfloat16)

    def lhs_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        del n_i, k_i, group_offset
        return group_metadata[2][grid_id], 0

    def rhs_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        del n_i, k_i, group_offset
        return group_metadata[1][grid_id], 0, 0

    def out_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        del n_i, k_i, group_offset
        return group_metadata[2][grid_id], 0

    lhs_spec = pl.BlockSpec((tile_m, k), lhs_indices)
    rhs_spec = pl.BlockSpec((None, k, n), rhs_indices)
    out_spec = pl.BlockSpec((tile_m, n), out_indices)
    return pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
            jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[lhs_spec, rhs_spec, rhs_spec],
            out_specs=[out_spec, out_spec],
            grid=(1, num_active_tiles, 1),
            scratch_shapes=[
                pltpu.VMEM((tile_m, n), jnp.float32),
                pltpu.VMEM((tile_m, n), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
        name="moe_gmm_w13_fwd",
    )(group_metadata, group_offset, lhs, rhs1, rhs3)


def row_chunk_bounds(rows_total, tile_m, chunks):
    """Tile-aligned row boundaries for ``chunks`` pipelined gather pieces.

    The first gather is exposed and gather i+1 hides under piece i's matmul,
    which takes longer than the gather of an equal piece. So, when the sizes
    divide, the pieces ramp up 1, 2, 4, then 8 units of ``rows_total /
    (8 * chunks)``, which shortens the exposed first gather; otherwise they
    are equal.
    """
    if rows_total % (chunks * tile_m):
        raise ValueError(
            "The routed row count must split into tile-aligned chunks"
        )
    unit = rows_total // (8 * chunks)
    if chunks >= 4 and unit % tile_m == 0:
        sizes = [1, 2, 4]
        while sum(sizes) + 8 <= 8 * chunks:
            sizes.append(8)
        sizes[-1] += 8 * chunks - sum(sizes)
        sizes = [u * unit for u in sizes]
    else:
        sizes = [rows_total // chunks] * chunks
    bounds = [0]
    for size in sizes:
        bounds.append(bounds[-1] + size)
    return list(zip(bounds[:-1], bounds[1:]))


def chunk_grid(group_metadata, num_active_tiles, tile_start, tile_end):
    """The contiguous range of MegaBlox grid steps that visit m tiles in
    ``[tile_start, tile_end)``, as (first step, step count) int32 arrays.

    Steps visit m tiles in nondecreasing order, so a tile-aligned row chunk
    owns a contiguous run of the global steps, with the same expert visits it
    would get from its own metadata.
    """
    m_tile_ids = group_metadata[2]
    active = jnp.arange(m_tile_ids.shape[0]) < num_active_tiles
    first = jnp.sum(active & (m_tile_ids < tile_start)).astype(jnp.int32)
    end = jnp.sum(active & (m_tile_ids < tile_end)).astype(jnp.int32)
    return first.reshape(1), end - first


def _paired_forward_rows(
    lhs,
    rhs1,
    rhs3,
    group_metadata,
    grid,
    outs,
    *,
    tile_m,
    row_offset,
    rows_total,
):
    """``_paired_forward`` over the routed rows starting at ``row_offset``.

    ``group_metadata`` describes all routed rows; ``grid`` (``chunk_grid``)
    selects this chunk's steps. Results, and a copy of ``lhs``, land in the
    matching rows of full-size outputs: ``outs`` (h1, h3, routed rows) are
    aliased and updated in place, or, for the first chunk, fresh buffers whose
    other rows are filled by later chunks. ``row_offset`` is tile aligned, so
    every tile sees the same expert visits, and the same arithmetic, as in the
    unchunked kernel.
    """
    m, k = lhs.shape
    groups, _, n = rhs1.shape
    if row_offset % tile_m or m % tile_m:
        raise ValueError("Row chunks must be tile aligned")
    tile_offset = row_offset // tile_m
    grid_start, grid_size = grid

    def kernel(group_metadata, grid_start, lhs, rhs1, rhs3, *refs):
        out1, out3, routed, acc1, acc3 = refs[-5:]
        _, _, m_tile_ids = group_metadata
        local_id = pl.program_id(1)
        grid_id = local_id + grid_start[0]
        first_visit = jnp.logical_or(
            local_id == 0,
            m_tile_ids[grid_id] != m_tile_ids[jnp.maximum(grid_id - 1, 0)],
        )

        @pl.when(first_visit)
        def _():
            routed[...] = lhs[...]

        loaded_lhs = lhs[...]
        acc1[...] = lax.dot(
            loaded_lhs, rhs1[...], preferred_element_type=jnp.float32
        )
        acc3[...] = lax.dot(
            loaded_lhs, rhs3[...], preferred_element_type=jnp.float32
        )
        mask = _get_store_mask(
            grid_id=grid_id,
            group_metadata=group_metadata,
            tm=tile_m,
            tn=n,
        )
        out1[...] = lax.select(
            mask[...], acc1[...], out1[...].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        out3[...] = lax.select(
            mask[...], acc3[...], out3[...].astype(jnp.float32)
        ).astype(jnp.bfloat16)

    def lhs_indices(n_i, local_id, k_i, group_metadata, grid_start):
        del n_i, k_i
        return group_metadata[2][local_id + grid_start[0]] - tile_offset, 0

    def rhs_indices(n_i, local_id, k_i, group_metadata, grid_start):
        del n_i, k_i
        return group_metadata[1][local_id + grid_start[0]], 0, 0

    def out_indices(n_i, local_id, k_i, group_metadata, grid_start):
        del n_i, k_i
        return group_metadata[2][local_id + grid_start[0]], 0

    rhs_spec = pl.BlockSpec((None, k, n), rhs_indices)
    out_spec = pl.BlockSpec((tile_m, n), out_indices)
    routed_spec = pl.BlockSpec((tile_m, k), out_indices)
    in_specs = [pl.BlockSpec((tile_m, k), lhs_indices), rhs_spec, rhs_spec]
    operands = [group_metadata, grid_start, lhs, rhs1, rhs3]
    aliases = {}
    if outs is not None:
        in_specs += [pl.BlockSpec(memory_space=pl.ANY)] * 3
        operands += list(outs)
        # Operand 7 counts the three group-metadata arrays and grid_start.
        aliases = {7: 0, 8: 1, 9: 2}
    return pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct((rows_total, n), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows_total, n), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows_total, k), lhs.dtype),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=in_specs,
            out_specs=[out_spec, out_spec, routed_spec],
            grid=(1, grid_size, 1),
            scratch_shapes=[
                pltpu.VMEM((tile_m, n), jnp.float32),
                pltpu.VMEM((tile_m, n), jnp.float32),
            ],
        ),
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
        cost_estimate=pl.CostEstimate(
            flops=4 * m * k * n,
            transcendentals=0,
            bytes_accessed=(
                2 * m * k * 2 + 2 * groups * k * n * 2 + 2 * m * n * 2
            ),
        ),
        name="moe_gmm_w13_fwd",
    )(*operands)


def _gather_paired_forward(
    x, token_indices, rhs1, rhs3, group_sizes, *, tile_m, chunks
):
    """``_paired_forward(x[token_indices], ...)`` with the gather pipelined.

    The routed rows are gathered in ``chunks`` tile-aligned pieces
    (``row_chunk_bounds``). Each piece feeds a row-range kernel that also
    writes it into one full routed-row buffer, so XLA can run piece i+1's
    (SparseCore) gather under piece i's matmuls without slicing or
    concatenating routed-row tensors. Returns
    (h1, h3, x[token_indices]), bitwise equal to the unchunked computation.
    """
    _validate_shapes(x, rhs1, rhs3, group_sizes, tile_m)
    rows_total = token_indices.shape[0]
    group_metadata, num_active_tiles = _metadata(
        group_sizes,
        m=rows_total,
        tile_m=tile_m,
        groups=rhs1.shape[0],
        visit_empty_groups=False,
    )
    outs = None
    for start, end in row_chunk_bounds(rows_total, tile_m, chunks):
        outs = _paired_forward_rows(
            x[token_indices[start:end]],
            rhs1,
            rhs3,
            group_metadata,
            chunk_grid(
                group_metadata, num_active_tiles, start // tile_m, end // tile_m
            ),
            outs,
            tile_m=tile_m,
            row_offset=start,
            rows_total=rows_total,
        )
    return tuple(outs)


def _accumulate_dinput(dout, rhs, existing, group_sizes, *, tile_m):
    """Return ``existing + bf16(dout @ rhs^T)`` per expert group, in place.

    ``existing`` is aliased to the output, so the sum never holds a second
    ``(m, k)`` routed-row tensor. Each output tile is seeded from ``existing``
    the first time the grid visits it; later visits of the same tile (the next
    expert's boundary rows) keep adding to the resident block.
    """
    m, n = dout.shape
    groups, k, _ = rhs.shape
    group_metadata, num_active_tiles = _metadata(
        group_sizes,
        m=m,
        tile_m=tile_m,
        groups=groups,
        visit_empty_groups=False,
    )
    group_offset = jnp.array([0], dtype=jnp.int32)

    def kernel(group_metadata, group_offset, dout, rhs, existing, out):
        del group_offset
        _, _, m_tile_ids = group_metadata
        grid_id = pl.program_id(1)
        prev_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
        first_visit = jnp.logical_or(
            grid_id == 0, m_tile_ids[grid_id] != m_tile_ids[prev_grid_id]
        )

        @pl.when(first_visit)
        def _seed():
            out[...] = existing[...]

        acc = lax.dot_general(
            dout[...],
            rhs[...],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        # Not a no-op: acc is already FP32, and the cast to out.dtype is the
        # rounding. The path this replaces is two GMMs, each rounding its output
        # to bf16, then XLA's bf16 add (FP32 sum, one final rounding), i.e.
        # bf16(bf16(g1) + bf16(g3)). Rounding this product to bf16 first and
        # widening it back for the FP32 add below reproduces that bit for bit;
        # adding the unrounded product would keep extra mantissa bits and
        # change the result (and the training trajectory).
        acc = acc.astype(out.dtype).astype(jnp.float32)
        mask = _get_store_mask(
            grid_id=grid_id,
            group_metadata=group_metadata,
            tm=tile_m,
            tn=k,
        )
        current = out[...].astype(jnp.float32)
        out[...] = lax.select(mask[...], current + acc, current).astype(
            out.dtype
        )

    def row_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        del n_i, k_i, group_offset
        return group_metadata[2][grid_id], 0

    def rhs_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        del n_i, k_i, group_offset
        return group_metadata[1][grid_id], 0, 0

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, k), existing.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec((tile_m, n), row_indices),
                pl.BlockSpec((None, k, n), rhs_indices),
                pl.BlockSpec((tile_m, k), row_indices),
            ],
            out_specs=pl.BlockSpec((tile_m, k), row_indices),
            grid=(1, num_active_tiles, 1),
        ),
        # Operand 6 counts the three group-metadata arrays and group_offset.
        input_output_aliases={6: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
        name="moe_gmm_w3_dgrad_accum",
    )(group_metadata, group_offset, dout, rhs, existing)


def _dinput(dout1, dout3, rhs1, rhs3, group_sizes, *, tile_m):
    """dInput = dout1 @ W1^T + dout3 @ W3^T, the second term added in place.

    Both terms are rounded to BF16 before the add, exactly as when two MegaBlox
    GMMs produced them and XLA added the outputs, so the result is bit-identical.
    """
    k, n = rhs1.shape[1:]
    grad_lhs = call_named(
        megablox_gmm,
        "moe_gmm_w1_dgrad",
        dout1,
        rhs1,
        group_sizes=group_sizes,
        preferred_element_type=dout1.dtype,
        tiling=(tile_m, n, k),
        transpose_rhs=True,
    )
    return _accumulate_dinput(dout3, rhs3, grad_lhs, group_sizes, tile_m=tile_m)


def _paired_dweight(lhs, dout1, dout3, group_sizes, *, tile_m, tile_k=0):
    """Traverse each expert's routed rows once to produce two weight grads.

    This follows upstream MegaBlox TGMM's structure: use its group metadata,
    visit tiles in expert order, reset the accumulator on an expert's first
    tile, mask partial boundary tiles, and store on its final tile. The only
    material extension is carrying two dOutput operands and two FP32
    accumulators, duplicating the dot and store within one traversal so both
    gradients share the lhs load and metadata work. Empty experts are visited
    so every output slab is written.
    """
    if tile_m <= 0:
        raise ValueError("tile_m must be positive")
    if dout1.shape != dout3.shape or lhs.shape[0] != dout1.shape[0]:
        raise ValueError("Invalid paired dWeight dimensions")
    m, k = lhs.shape
    n = dout1.shape[1]
    groups = group_sizes.shape[0]
    if m % tile_m:
        raise ValueError("The total row count must be divisible by tile_m")
    tile_k = tile_k or k
    if tile_k <= 0 or k % tile_k:
        raise ValueError("tile_k must divide K")
    group_metadata, num_active_tiles = _metadata(
        group_sizes,
        m=m,
        tile_m=tile_m,
        groups=groups,
        visit_empty_groups=True,
    )
    group_offset = jnp.array([0], dtype=jnp.int32)

    def kernel(
        group_metadata,
        group_offset,
        lhs,
        dout1,
        dout3,
        out1,
        out3,
        acc1,
        acc3,
    ):
        del group_offset
        # Grid axis 1 walks the K slabs; each slab replays the full row
        # traversal below with its own accumulator, so nothing here depends
        # on the slab index except the block specs.
        grid_id = pl.program_id(2)
        group_ids = group_metadata[1]
        group = group_ids[grid_id]
        previous_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
        group_has_changed = jnp.logical_or(
            grid_id == 0, group_ids[previous_grid_id] != group
        )

        @pl.when(group_has_changed)
        def _zero_accumulators():
            acc1[...] = jnp.zeros_like(acc1)
            acc3[...] = jnp.zeros_like(acc3)

        group_has_rows = (
            _get_group_size(grid_id=grid_id, group_metadata=group_metadata) > 0
        )

        @pl.when(group_has_rows)
        def _accumulate():
            lhs_mask = _get_store_mask(
                grid_id=grid_id,
                group_metadata=group_metadata,
                tm=tile_m,
                tn=tile_k,
            )
            dout_mask = _get_store_mask(
                grid_id=grid_id,
                group_metadata=group_metadata,
                tm=tile_m,
                tn=n,
            )
            loaded_lhs = lax.select(
                lhs_mask[...],
                lhs[...],
                jnp.zeros_like(lhs),
            )
            loaded_dout1 = lax.select(
                dout_mask[...],
                dout1[...],
                jnp.zeros_like(dout1),
            )
            loaded_dout3 = lax.select(
                dout_mask[...],
                dout3[...],
                jnp.zeros_like(dout3),
            )
            # All three inputs are already BF16. Mask them without a
            # BF16->FP32->BF16 round trip, and contract the row dimensions
            # directly instead of transposing lhs only to express X.T @ dY.
            acc1[...] += lax.dot_general(
                loaded_lhs,
                loaded_dout1,
                preferred_element_type=jnp.float32,
                dimension_numbers=(((0,), (0,)), ((), ())),
            )
            acc3[...] += lax.dot_general(
                loaded_lhs,
                loaded_dout3,
                preferred_element_type=jnp.float32,
                dimension_numbers=(((0,), (0,)), ((), ())),
            )

        is_last_grid_id = grid_id == pl.num_programs(2) - 1
        next_grid_id = jnp.where(is_last_grid_id, grid_id, grid_id + 1)
        group_is_changing = jnp.logical_or(
            is_last_grid_id, group_ids[next_grid_id] != group
        )

        @pl.when(group_is_changing)
        def _store_accumulators():
            out1[...] = acc1[...].astype(jnp.bfloat16)
            out3[...] = acc3[...].astype(jnp.bfloat16)

    def lhs_indices(n_i, k_i, grid_id, group_metadata, group_offset):
        del n_i, group_offset
        return group_metadata[2][grid_id], k_i

    def dout_indices(n_i, k_i, grid_id, group_metadata, group_offset):
        del n_i, k_i, group_offset
        return group_metadata[2][grid_id], 0

    def out_indices(n_i, k_i, grid_id, group_metadata, group_offset):
        del n_i, group_offset
        return group_metadata[1][grid_id], k_i, 0

    lhs_spec = pl.BlockSpec((tile_m, tile_k), lhs_indices)
    dout_spec = pl.BlockSpec((tile_m, n), dout_indices)
    out_spec = pl.BlockSpec((None, tile_k, n), out_indices)
    return pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct((groups, k, n), jnp.bfloat16),
            jax.ShapeDtypeStruct((groups, k, n), jnp.bfloat16),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[lhs_spec, dout_spec, dout_spec],
            out_specs=[out_spec, out_spec],
            grid=(1, k // tile_k, num_active_tiles),
            scratch_shapes=[
                pltpu.VMEM((tile_k, n), jnp.float32),
                pltpu.VMEM((tile_k, n), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
        name="moe_tgmm_w13_wgrad",
    )(group_metadata, group_offset, lhs, dout1, dout3)


@pallas.jax_op("tt_experts::paired_grouped_mm")
def _paired_grouped_mm_fwd(
    lhs: jax.Array,
    rhs1: jax.Array,
    rhs3: jax.Array,
    group_sizes: jax.Array,
    tile_m: int,
    tgmm_tile_m: int,
    tgmm_tile_k: int,
) -> tuple[jax.Array, jax.Array]:
    del tgmm_tile_m, tgmm_tile_k
    return _paired_forward(lhs, rhs1, rhs3, group_sizes, tile_m=tile_m)


@pallas.jax_op("tt_experts::paired_grouped_mm_bwd")
def _paired_grouped_mm_bwd(
    dout1: jax.Array,
    dout3: jax.Array,
    lhs: jax.Array,
    rhs1: jax.Array,
    rhs3: jax.Array,
    group_sizes: jax.Array,
    tile_m: int,
    tgmm_tile_m: int,
    tgmm_tile_k: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    grad_lhs = _dinput(
        dout1, dout3, rhs1, rhs3, group_sizes, tile_m=tile_m
    )
    grad_rhs1, grad_rhs3 = _paired_dweight(
        lhs, dout1, dout3, group_sizes, tile_m=tgmm_tile_m, tile_k=tgmm_tile_k
    )
    return grad_lhs, grad_rhs1, grad_rhs3


def _setup_context(ctx, inputs, output):
    del output
    lhs, rhs1, rhs3, group_sizes, tile_m, tgmm_tile_m, tgmm_tile_k = inputs
    ctx.save_for_backward(lhs, rhs1, rhs3, group_sizes)
    ctx.tile_m = tile_m
    ctx.tgmm_tile_m = tgmm_tile_m
    ctx.tgmm_tile_k = tgmm_tile_k


def _backward(ctx, grad_output1, grad_output3):
    lhs, rhs1, rhs3, group_sizes = ctx.saved_tensors
    grad_lhs, grad_rhs1, grad_rhs3 = _paired_grouped_mm_bwd(
        grad_output1,
        grad_output3,
        lhs,
        rhs1,
        rhs3,
        group_sizes,
        ctx.tile_m,
        ctx.tgmm_tile_m,
        ctx.tgmm_tile_k,
    )
    return grad_lhs, grad_rhs1, grad_rhs3, None, None, None, None


_paired_grouped_mm_fwd.register_autograd(_backward, setup_context=_setup_context)


@pallas.jax_op("tt_experts::gather_paired_grouped_mm")
def _gather_paired_grouped_mm_fwd(
    x: jax.Array,
    token_indices: jax.Array,
    inverse: jax.Array,
    rhs1: jax.Array,
    rhs3: jax.Array,
    group_sizes: jax.Array,
    tile_m: int,
    tgmm_tile_m: int,
    tgmm_tile_k: int,
    chunks: int,
    recompute_chunks: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # recompute_chunks replaces chunks in rematerialized copies (graph pass
    # apply_recompute_gather_chunks).
    del inverse, tgmm_tile_m, tgmm_tile_k, recompute_chunks
    return _gather_paired_forward(
        x, token_indices, rhs1, rhs3, group_sizes, tile_m=tile_m, chunks=chunks
    )


def _gather_setup_context(ctx, inputs, output):
    x, _, inverse, rhs1, rhs3, group_sizes, tile_m, tgmm_tile_m, tgmm_tile_k = (
        inputs[:9]
    )
    routed = output[2]
    ctx.mark_non_differentiable(routed)
    ctx.save_for_backward(routed, inverse, rhs1, rhs3, group_sizes)
    ctx.x_shape = x.shape
    ctx.tile_m = tile_m
    ctx.tgmm_tile_m = tgmm_tile_m
    ctx.tgmm_tile_k = tgmm_tile_k


def _gather_backward(ctx, grad_output1, grad_output3, _grad_routed):
    routed, inverse, rhs1, rhs3, group_sizes = ctx.saved_tensors
    grad_routed, grad_rhs1, grad_rhs3 = _paired_grouped_mm_bwd(
        grad_output1,
        grad_output3,
        routed,
        rhs1,
        rhs3,
        group_sizes,
        ctx.tile_m,
        ctx.tgmm_tile_m,
        ctx.tgmm_tile_k,
    )
    # The same inverse-permutation gather and FP32 top-k sum as the TPU
    # dispatcher's _LocalReorder backward.
    tokens, dim = ctx.x_shape
    grad_x = (
        torch.index_select(grad_routed, 0, inverse)
        .float()
        .reshape(tokens, -1, dim)
        .sum(dim=1)
        .to(grad_routed.dtype)
    )
    return (grad_x, None, None, grad_rhs1, grad_rhs3) + (None,) * 6


_gather_paired_grouped_mm_fwd.register_autograd(
    _gather_backward, setup_context=_gather_setup_context
)


def gather_paired_segment_matmul_pallas_2d(
    x: torch.Tensor,
    token_indices: torch.Tensor,
    inverse: torch.Tensor,
    rhs1: torch.Tensor,
    rhs3: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    tile_m: int,
    tgmm_tile_m: int,
    tgmm_tile_k: int,
    chunks: int,
    recompute_chunks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``paired_segment_matmul_pallas_2d(x[token_indices], ...)``, with the
    token gather pipelined against the matmul in ``chunks`` row pieces
    (``recompute_chunks``, if nonzero, in rematerialized copies).

    ``inverse`` inverts the route permutation; backward uses it to return
    token gradients as the TPU dispatcher's local reorder does.
    """
    if group_sizes.dtype != torch.int32:
        group_sizes = group_sizes.to(torch.int32)
    h1, h3, _ = _gather_paired_grouped_mm_fwd(
        x,
        token_indices.to(torch.int32),
        inverse.to(torch.int32),
        rhs1,
        rhs3,
        group_sizes,
        tile_m,
        tgmm_tile_m,
        tgmm_tile_k,
        chunks,
        recompute_chunks or chunks,
    )
    return h1, h3


def paired_segment_matmul_pallas_2d(
    lhs: torch.Tensor,
    rhs1: torch.Tensor,
    rhs3: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    tile_m: int = 1024,
    tgmm_tile_m: int = 128,
    tgmm_tile_k: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply two equal-shaped expert projections while sharing lhs loads.

    ``group_sizes`` must sum exactly to ``lhs.shape[0]``; this kernel does not
    initialize a capacity-padding tail. ``tgmm_tile_k`` (0 = full K) tiles the
    dWeight kernel's K dimension to bound its VMEM footprint.
    """
    if rhs1.shape != rhs3.shape:
        raise ValueError("Paired grouped matmul weights must have equal shapes")
    if group_sizes.dtype != torch.int32:
        group_sizes = group_sizes.to(torch.int32)
    return _paired_grouped_mm_fwd(
        lhs, rhs1, rhs3, group_sizes, tile_m, tgmm_tile_m, tgmm_tile_k
    )


__all__ = [
    "gather_paired_segment_matmul_pallas_2d",
    "paired_segment_matmul_pallas_2d",
]
