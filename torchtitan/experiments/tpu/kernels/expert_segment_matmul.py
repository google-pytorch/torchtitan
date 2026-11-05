"""MegaBlox grouped matmul for dynamically routed TPU expert layers.

For expert ``e``, forward computes ``Y_e = X_e @ W_e``. Backward computes
``dX_e = dY_e @ W_e.T`` with another GMM and ``dW_e = X_e.T @ dY_e`` with
TGMM. dInput uses MegaBlox's logical rhs transpose so kernel-native weights do
not require a physical layout conversion. dWeight always accumulates in FP32
across token tiles and rounds once, at the final store, to the dtype of the
weight (``rhs.dtype``).

The module-level custom operators are stable Dynamo/AOTAutograd boundaries.
Tiling arguments are compile-time constants, while ``group_sizes`` remains a
runtime int32 TPU tensor, so routing changes do not trigger specialization.
"""

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.megablox import ops as gmm_ops
from jax.experimental.pallas.ops.tpu.megablox.gmm import (
    gmm as megablox_gmm,
    tgmm as megablox_tgmm,
)
from torch_tpu._internal import pallas
from torchtitan.experiments.tpu.kernels.paired_segment_matmul import (
    chunk_grid,
    row_chunk_bounds,
)
from torchtitan.experiments.tpu.kernels.pallas_naming import call_named

gmm_backend = gmm_ops.backend


def _tiling(
    tile_m: int,
    tile_k: int,
    tile_n: int,
    k_dim: int,
    n_dim: int,
) -> tuple[int, int, int]:
    """Resolve zero-valued K/N tiles to the corresponding full dimension."""
    return (
        tile_m,
        tile_k if tile_k > 0 else k_dim,
        tile_n if tile_n > 0 else n_dim,
    )


@pallas.jax_op("tt_experts::grouped_mm")
def _grouped_mm_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    tile_m: int,
    tile_k: int,
    tile_n: int,
    tgmm_tile_m: int,
    tgmm_tile_k: int,
    tgmm_tile_n: int,
    transpose_rhs: bool,
) -> jax.Array:
    del tgmm_tile_m, tgmm_tile_k, tgmm_tile_n
    if transpose_rhs:
        _, n_dim, k_dim = rhs.shape
    else:
        _, k_dim, n_dim = rhs.shape
    return call_named(
        megablox_gmm,
        "moe_gmm_fwd",
        lhs,
        rhs,
        group_sizes=group_sizes,
        preferred_element_type=lhs.dtype,
        tiling=_tiling(tile_m, tile_k, tile_n, k_dim, n_dim),
        transpose_rhs=transpose_rhs,
    )


@pallas.jax_op("tt_experts::grouped_mm_bwd")
def _grouped_mm_bwd(
    dout: jax.Array,
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    tile_m: int,
    tile_k: int,
    tile_n: int,
    tgmm_tile_m: int,
    tgmm_tile_k: int,
    tgmm_tile_n: int,
    transpose_rhs: bool,
) -> tuple[jax.Array, jax.Array]:
    # Store the dWeight in the weight's own dtype. Autograd casts a gradient to
    # the dtype of the input it belongs to, so a wider store would be rounded
    # to this dtype right after the kernel and only add an HBM write and a cast
    # round trip. MegaBlox still accumulates each tile with an FP32 dot into
    # FP32 VMEM; only the final store rounds, so the values equal an FP32 store
    # cast to this dtype.
    if transpose_rhs:
        _, n_dim, k_dim = rhs.shape
        grad_lhs = call_named(
            megablox_gmm,
            "moe_gmm_dgrad",
            dout,
            rhs,
            group_sizes=group_sizes,
            preferred_element_type=dout.dtype,
            tiling=_tiling(tile_m, tile_n, tile_k, n_dim, k_dim),
        )
        grad_rhs = call_named(
            megablox_tgmm,
            "moe_tgmm_wgrad",
            jnp.swapaxes(dout, 0, 1),
            lhs,
            group_sizes=group_sizes,
            preferred_element_type=rhs.dtype,
            tiling=_tiling(
                tgmm_tile_m,
                tgmm_tile_n,
                tgmm_tile_k,
                dout.shape[1],
                lhs.shape[1],
            ),
            interpret=False,
        )
    else:
        _, k_dim, n_dim = rhs.shape
        grad_lhs = call_named(
            megablox_gmm,
            "moe_gmm_dgrad",
            dout,
            rhs,
            group_sizes=group_sizes,
            preferred_element_type=dout.dtype,
            tiling=_tiling(tile_m, tile_n, tile_k, n_dim, k_dim),
            transpose_rhs=True,
        )
        grad_rhs = call_named(
            megablox_tgmm,
            "moe_tgmm_wgrad",
            jnp.swapaxes(lhs, 0, 1),
            dout,
            group_sizes=group_sizes,
            preferred_element_type=rhs.dtype,
            tiling=_tiling(
                tgmm_tile_m,
                tgmm_tile_k,
                tgmm_tile_n,
                lhs.shape[1],
                dout.shape[1],
            ),
            interpret=False,
        )
    return grad_lhs, grad_rhs


def _gmm_dgrad_rows(
    dout,
    rhs,
    group_metadata,
    grid,
    outs,
    *,
    tile_m,
    row_offset,
    rows_total,
    transpose_rhs,
):
    """MegaBlox full-tile dInput (``dout @ W^T`` per expert) over the routed
    rows starting at ``row_offset``, written with a copy of ``dout`` into
    full-size (aliased, after the first chunk) outputs. ``group_metadata``
    describes all routed rows and ``grid`` (``chunk_grid``) selects this
    chunk's steps."""
    m, n_dim = dout.shape
    groups = rhs.shape[0]
    k_dim = rhs.shape[2] if transpose_rhs else rhs.shape[1]
    if row_offset % tile_m or m % tile_m:
        raise ValueError("Row chunks must be tile aligned")
    tile_offset = row_offset // tile_m
    grid_start, grid_size = grid
    contract = ((1,), (0,)) if transpose_rhs else ((1,), (1,))

    def kernel(group_metadata, grid_start, dout, rhs, *refs):
        out, routed, acc = refs[-3:]
        _, _, m_tile_ids = group_metadata
        local_id = pl.program_id(1)
        grid_id = local_id + grid_start[0]
        first_visit = jnp.logical_or(
            local_id == 0,
            m_tile_ids[grid_id] != m_tile_ids[jnp.maximum(grid_id - 1, 0)],
        )

        @pl.when(first_visit)
        def _():
            routed[...] = dout[...]

        acc[...] = lax.dot_general(
            dout[...],
            rhs[...],
            (contract, ((), ())),
            preferred_element_type=jnp.float32,
        )
        mask = gmm_backend._get_store_mask(
            grid_id=grid_id, group_metadata=group_metadata, tm=tile_m, tn=k_dim
        )
        out[...] = lax.select(
            mask[...], acc[...].astype(out.dtype), out[...]
        )

    def dout_indices(n_i, local_id, k_i, group_metadata, grid_start):
        del n_i, k_i
        return group_metadata[2][local_id + grid_start[0]] - tile_offset, 0

    def rhs_indices(n_i, local_id, k_i, group_metadata, grid_start):
        del n_i, k_i
        return group_metadata[1][local_id + grid_start[0]], 0, 0

    def out_indices(n_i, local_id, k_i, group_metadata, grid_start):
        del n_i, k_i
        return group_metadata[2][local_id + grid_start[0]], 0

    in_specs = [
        pl.BlockSpec((tile_m, n_dim), dout_indices),
        pl.BlockSpec((None,) + rhs.shape[1:], rhs_indices),
    ]
    operands = [group_metadata, grid_start, dout, rhs]
    aliases = {}
    if outs is not None:
        in_specs += [pl.BlockSpec(memory_space=pl.ANY)] * 2
        operands += list(outs)
        # Operand 6 counts the three group-metadata arrays and grid_start.
        aliases = {6: 0, 7: 1}
    return pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct((rows_total, k_dim), dout.dtype),
            jax.ShapeDtypeStruct((rows_total, n_dim), dout.dtype),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=in_specs,
            out_specs=[
                pl.BlockSpec((tile_m, k_dim), out_indices),
                pl.BlockSpec((tile_m, n_dim), out_indices),
            ],
            grid=(1, grid_size, 1),
            scratch_shapes=[pltpu.VMEM((tile_m, k_dim), jnp.float32)],
        ),
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
        cost_estimate=pl.CostEstimate(
            flops=2 * m * k_dim * n_dim,
            transcendentals=0,
            bytes_accessed=(
                2 * m * n_dim * 2 + groups * k_dim * n_dim * 2 + m * k_dim * 2
            ),
        ),
        name="moe_gmm_dgrad",
    )(*operands)


@pallas.jax_op("tt_experts::gather_grouped_mm_bwd")
def _gather_grouped_mm_bwd(
    dout_tokens: jax.Array,
    token_indices: jax.Array,
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    tile_m: int,
    tgmm_tile_m: int,
    tgmm_tile_k: int,
    tgmm_tile_n: int,
    transpose_rhs: bool,
    chunks: int,
) -> tuple[jax.Array, jax.Array]:
    """``_grouped_mm_bwd`` for ``dout = dout_tokens[token_indices]``, with that
    row gather pipelined against the dInput kernel in ``chunks`` pieces."""
    rows_total = token_indices.shape[0]
    group_metadata, num_active_tiles = gmm_backend.make_group_metadata(
        group_sizes=group_sizes,
        m=rows_total,
        tm=tile_m,
        start_group=jnp.array(0, dtype=jnp.int32),
        num_nonzero_groups=rhs.shape[0],
        visit_empty_groups=False,
    )
    outs = None
    for start, end in row_chunk_bounds(rows_total, tile_m, chunks):
        outs = _gmm_dgrad_rows(
            dout_tokens[token_indices[start:end]],
            rhs,
            group_metadata,
            chunk_grid(
                group_metadata, num_active_tiles, start // tile_m, end // tile_m
            ),
            outs,
            tile_m=tile_m,
            row_offset=start,
            rows_total=rows_total,
            transpose_rhs=transpose_rhs,
        )
    grad_lhs, dout = outs
    if transpose_rhs:
        grad_rhs = call_named(
            megablox_tgmm,
            "moe_tgmm_wgrad",
            jnp.swapaxes(dout, 0, 1),
            lhs,
            group_sizes=group_sizes,
            preferred_element_type=rhs.dtype,
            tiling=_tiling(
                tgmm_tile_m,
                tgmm_tile_n,
                tgmm_tile_k,
                dout.shape[1],
                lhs.shape[1],
            ),
            interpret=False,
        )
    else:
        grad_rhs = call_named(
            megablox_tgmm,
            "moe_tgmm_wgrad",
            jnp.swapaxes(lhs, 0, 1),
            dout,
            group_sizes=group_sizes,
            preferred_element_type=rhs.dtype,
            tiling=_tiling(
                tgmm_tile_m,
                tgmm_tile_k,
                tgmm_tile_n,
                lhs.shape[1],
                dout.shape[1],
            ),
            interpret=False,
        )
    return grad_lhs, grad_rhs


class _DownProjCombine(torch.autograd.Function):
    """Expert down projection followed by the fixed-K token combine.

    Forward is ``segment_matmul_pallas_2d`` then the TPU dispatcher's stable
    grouped reduce, computed per token chunk (the same sums). Backward gathers
    the token gradient rows itself, in pipelined chunks, instead of
    materializing ``grad[token_indices]`` before the dInput GMM.
    """

    @staticmethod
    def forward(
        ctx,
        lhs,
        rhs,
        group_sizes,
        token_indices,
        top_k,
        tiling,
        tgmm_tiling,
        transpose_rhs,
        chunks,
    ):
        ctx.set_materialize_grads(False)
        tile_m, tile_k, tile_n = tiling
        routed = _grouped_mm_fwd(
            lhs,
            rhs,
            group_sizes,
            tile_m,
            tile_k,
            tile_n,
            *tgmm_tiling,
            transpose_rhs,
        )
        ctx.save_for_backward(lhs, rhs, group_sizes, token_indices)
        ctx.static_args = (tile_m, *tgmm_tiling, transpose_rhs, chunks)
        # Gather and reduce in token chunks, so the (SparseCore) gather of one
        # chunk overlaps the top-k sum of the previous one.
        permutation = torch.argsort(token_indices, stable=True)
        return torch.cat(
            [
                torch.index_select(routed, 0, rows)
                .reshape(-1, top_k, routed.shape[1])
                .float()
                .sum(dim=1)
                .to(routed.dtype)
                for rows in permutation.chunk(chunks)
            ]
        )

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return (None,) * 9
        lhs, rhs, group_sizes, token_indices = ctx.saved_tensors
        grad_lhs, grad_rhs = _gather_grouped_mm_bwd(
            grad_output,
            token_indices.to(torch.int32),
            lhs,
            rhs,
            group_sizes,
            *ctx.static_args,
        )
        return grad_lhs, grad_rhs, None, None, None, None, None, None, None


def down_proj_combine_pallas_2d(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    token_indices: torch.Tensor,
    *,
    top_k: int,
    tile_m: int,
    tgmm_tiling: tuple[int, int, int],
    transpose_rhs: bool,
    chunks: int,
) -> torch.Tensor:
    """Sum over each token's ``top_k`` routes of ``segment_matmul(lhs, rhs)``.

    Row ``r`` of ``lhs`` belongs to token ``token_indices[r]``. Uses full K/N
    tiles.
    """
    if group_sizes.dtype != torch.int32:
        group_sizes = group_sizes.to(torch.int32)
    return _DownProjCombine.apply(
        lhs,
        rhs,
        group_sizes,
        token_indices,
        top_k,
        (tile_m, 0, 0),
        tgmm_tiling,
        transpose_rhs,
        chunks,
    )


def _setup_context(ctx, inputs, output):
    del output
    lhs, rhs, group_sizes, *static_args = inputs
    ctx.save_for_backward(lhs, rhs, group_sizes)
    ctx.static_args = tuple(static_args)


def _backward(ctx, grad_output):
    lhs, rhs, group_sizes = ctx.saved_tensors
    grad_lhs, grad_rhs = _grouped_mm_bwd(
        grad_output, lhs, rhs, group_sizes, *ctx.static_args
    )
    return grad_lhs, grad_rhs, None, None, None, None, None, None, None, None


_grouped_mm_fwd.register_autograd(_backward, setup_context=_setup_context)


def segment_matmul_pallas_2d(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    tile_m: int = 1024,
    tile_k: int = 0,
    tile_n: int = 0,
    tgmm_tiling: tuple[int, int, int] = (128, 128, 128),
    transpose_rhs: bool = False,
) -> torch.Tensor:
    """Compute per-expert matmuls without moving group sizes to the host.

    Zero-valued K/N tile dimensions select the corresponding full matrix
    dimension. Distinct tiling values produce distinct compiled
    specializations. dWeight accumulation is FP32 and its final store is in
    the dtype of ``rhs``. ``transpose_rhs`` accepts weights
    in ``(E, N, K)`` order without materializing a transpose.
    """
    if group_sizes.dtype != torch.int32:
        group_sizes = group_sizes.to(torch.int32)
    return _grouped_mm_fwd(
        lhs,
        rhs,
        group_sizes,
        tile_m,
        tile_k,
        tile_n,
        *tgmm_tiling,
        transpose_rhs,
    )


__all__ = ["down_proj_combine_pallas_2d", "segment_matmul_pallas_2d"]
