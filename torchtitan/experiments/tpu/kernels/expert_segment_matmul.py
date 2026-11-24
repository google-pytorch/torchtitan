"""MegaBlox grouped matmul for dynamically routed TPU expert layers.

For expert ``e``, forward computes ``Y_e = X_e @ W_e``. Backward computes
``dX_e = dY_e @ W_e.T`` with another GMM and ``dW_e = X_e.T @ dY_e`` with
TGMM. dInput uses MegaBlox's logical rhs transpose so kernel-native weights do
not require a physical layout conversion. dWeight always accumulates in FP32
across token tiles and rounds once, at the final store, to the dtype of the
weight (``rhs.dtype``).

``segment_matmul_pallas_2d_accumulate`` is the variant for EP token chunking:
each chunk's TGMM adds its dWeight onto a running per-layer accumulator inside
the kernel (MegaBlox ``existing_out``), so the chunks' dWeights are never
stored and summed separately. ``chunk_dweight_accumulator`` starts the chain.

The module-level custom operators are stable Dynamo/AOTAutograd boundaries.
Tiling arguments are compile-time constants, while ``group_sizes`` remains a
runtime int32 TPU tensor, so routing changes do not trigger specialization.
"""

import jax
import jax.numpy as jnp
import torch
from jax.experimental.pallas.ops.tpu.megablox.gmm import (
    gmm as megablox_gmm,
    tgmm as megablox_tgmm,
)
from torch_tpu._internal import pallas
from torchtitan.experiments.tpu.kernels.pallas_naming import call_named


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
    grad_lhs = _dinput(
        dout, rhs, group_sizes, tile_m, tile_k, tile_n, transpose_rhs
    )
    grad_rhs = _dweight(
        dout,
        lhs,
        group_sizes,
        tgmm_tile_m,
        tgmm_tile_k,
        tgmm_tile_n,
        transpose_rhs,
        preferred_element_type=rhs.dtype,
    )
    return grad_lhs, grad_rhs


def _dinput(
    dout: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    tile_m: int,
    tile_k: int,
    tile_n: int,
    transpose_rhs: bool,
) -> jax.Array:
    """``dX = dY @ W.T`` per expert group, through MegaBlox GMM."""
    if transpose_rhs:
        _, n_dim, k_dim = rhs.shape
        return call_named(
            megablox_gmm,
            "moe_gmm_dgrad",
            dout,
            rhs,
            group_sizes=group_sizes,
            preferred_element_type=dout.dtype,
            tiling=_tiling(tile_m, tile_n, tile_k, n_dim, k_dim),
        )
    _, k_dim, n_dim = rhs.shape
    return call_named(
        megablox_gmm,
        "moe_gmm_dgrad",
        dout,
        rhs,
        group_sizes=group_sizes,
        preferred_element_type=dout.dtype,
        tiling=_tiling(tile_m, tile_n, tile_k, n_dim, k_dim),
        transpose_rhs=True,
    )


def _dweight(
    dout: jax.Array,
    lhs: jax.Array,
    group_sizes: jax.Array,
    tgmm_tile_m: int,
    tgmm_tile_k: int,
    tgmm_tile_n: int,
    transpose_rhs: bool,
    *,
    preferred_element_type,
    existing_out: jax.Array | None = None,
) -> jax.Array:
    """``dW = X.T @ dY`` per expert group, through MegaBlox TGMM.

    With ``existing_out``, the kernel adds each FP32 group sum onto the matching
    slab of ``existing_out`` before its single store and writes the result in
    place (MegaBlox aliases ``existing_out`` to the output). TGMM visits every
    group, empty ones included, so groups without routed rows keep their
    ``existing_out`` slab. ``preferred_element_type`` must then equal
    ``existing_out.dtype``.
    """
    kwargs = {}
    name = "moe_tgmm_wgrad"
    if existing_out is not None:
        kwargs["existing_out"] = existing_out
        name = "moe_tgmm_wgrad_accum"
    if transpose_rhs:
        return call_named(
            megablox_tgmm,
            name,
            jnp.swapaxes(dout, 0, 1),
            lhs,
            group_sizes=group_sizes,
            preferred_element_type=preferred_element_type,
            tiling=_tiling(
                tgmm_tile_m,
                tgmm_tile_n,
                tgmm_tile_k,
                dout.shape[1],
                lhs.shape[1],
            ),
            interpret=False,
            **kwargs,
        )
    return call_named(
        megablox_tgmm,
        name,
        jnp.swapaxes(lhs, 0, 1),
        dout,
        group_sizes=group_sizes,
        preferred_element_type=preferred_element_type,
        tiling=_tiling(
            tgmm_tile_m,
            tgmm_tile_k,
            tgmm_tile_n,
            lhs.shape[1],
            dout.shape[1],
        ),
        interpret=False,
        **kwargs,
    )


@pallas.jax_op("tt_experts::grouped_mm_bwd_accumulate")
def _grouped_mm_bwd_accumulate(
    dout: jax.Array,
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    dweight_acc: jax.Array | None,
    tile_m: int,
    tile_k: int,
    tile_n: int,
    tgmm_tile_m: int,
    tgmm_tile_k: int,
    tgmm_tile_n: int,
    transpose_rhs: bool,
    accumulate_fp32: bool,
) -> tuple[jax.Array, jax.Array]:
    """dInput, and dWeight added onto ``dweight_acc`` inside the TGMM kernel.

    Without ``dweight_acc`` (the first chunk to run backward), the dWeight is
    stored fresh in FP32 when ``accumulate_fp32``, else in ``rhs.dtype``.
    """
    grad_lhs = _dinput(
        dout, rhs, group_sizes, tile_m, tile_k, tile_n, transpose_rhs
    )
    if dweight_acc is None:
        acc_dtype = jnp.float32 if accumulate_fp32 else rhs.dtype
    else:
        if dweight_acc.shape != rhs.shape:
            raise ValueError(
                f"dweight_acc shape {dweight_acc.shape} must match the weight"
                f" shape {rhs.shape}"
            )
        acc_dtype = dweight_acc.dtype
    grad_rhs = _dweight(
        dout,
        lhs,
        group_sizes,
        tgmm_tile_m,
        tgmm_tile_k,
        tgmm_tile_n,
        transpose_rhs,
        preferred_element_type=acc_dtype,
        existing_out=dweight_acc,
    )
    return grad_lhs, grad_rhs


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


def _dweight_token(like: torch.Tensor) -> torch.Tensor:
    """A storage-free tensor with ``like``'s shape, dtype and device.

    Chunk dWeight accumulators pass these through forward. Only their
    gradients carry data: the running dWeight sum, in the token's dtype.
    """
    return torch.zeros((), dtype=like.dtype, device=like.device).expand(
        like.shape
    )


class _ChunkDWeightAccumulator(torch.autograd.Function):
    """Starts a chunk dWeight accumulator; see ``chunk_dweight_accumulator``."""

    @staticmethod
    def forward(ctx, weight, accumulate_fp32):
        ctx.weight_dtype = weight.dtype
        dtype = torch.float32 if accumulate_fp32 else weight.dtype
        return torch.zeros((), dtype=dtype, device=weight.device).expand(
            weight.shape
        )

    @staticmethod
    def backward(ctx, grad_acc):
        # The summed dWeight of every chunk, rounded once to the weight dtype.
        return grad_acc.to(ctx.weight_dtype), None


def chunk_dweight_accumulator(
    weight: torch.Tensor, *, accumulate_fp32: bool = True
) -> torch.Tensor:
    """Starts the in-kernel dWeight accumulator for a weight shared by chunks.

    Pass the result to the first chunk's ``*_accumulate`` grouped matmul, and
    each chunk's returned accumulator to the next chunk; the last chunk takes
    ``last_chunk=True``. In backward, which runs the chunks last to first, the
    last chunk's TGMM stores its dWeight fresh and every earlier chunk's TGMM
    adds its own onto that running sum before its single store, in FP32 when
    ``accumulate_fp32`` (else in ``weight.dtype``). This function's backward
    hands the total to ``weight``, rounded once to ``weight.dtype``.

    The accumulator is a storage-free placeholder in forward: only its
    gradient holds data.
    """
    return _ChunkDWeightAccumulator.apply(weight, accumulate_fp32)


def _accumulate_backward(ctx, grad_output, grad_acc):
    lhs, rhs, group_sizes = ctx.saved_tensors
    grad_lhs, dweight_acc = _grouped_mm_bwd_accumulate(
        grad_output,
        lhs,
        rhs,
        group_sizes,
        grad_acc,
        *ctx.static_args,
        ctx.accumulate_fp32,
    )
    return (grad_lhs, None, None, dweight_acc) + (None,) * len(ctx.static_args)


def _accumulate_forward(ctx, lhs, rhs, group_sizes, dweight_acc, static_args):
    ctx.save_for_backward(lhs, rhs, group_sizes)
    ctx.static_args = static_args
    ctx.accumulate_fp32 = dweight_acc.dtype == torch.float32
    return _grouped_mm_fwd(lhs, rhs, group_sizes, *static_args)


class _GroupedMMAccumulate(torch.autograd.Function):
    """Grouped matmul whose dWeight joins a chunk accumulator, not last."""

    @staticmethod
    def forward(ctx, lhs, rhs, group_sizes, dweight_acc, *static_args):
        out = _accumulate_forward(
            ctx, lhs, rhs, group_sizes, dweight_acc, static_args
        )
        return out, _dweight_token(dweight_acc)

    @staticmethod
    def backward(ctx, grad_output, grad_acc):
        return _accumulate_backward(ctx, grad_output, grad_acc)


class _GroupedMMAccumulateLast(torch.autograd.Function):
    """Grouped matmul for the last chunk: its backward starts the dWeight sum."""

    @staticmethod
    def forward(ctx, lhs, rhs, group_sizes, dweight_acc, *static_args):
        return _accumulate_forward(
            ctx, lhs, rhs, group_sizes, dweight_acc, static_args
        )

    @staticmethod
    def backward(ctx, grad_output):
        return _accumulate_backward(ctx, grad_output, None)


def segment_matmul_pallas_2d_accumulate(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    dweight_acc: torch.Tensor,
    *,
    last_chunk: bool,
    tile_m: int = 1024,
    tile_k: int = 0,
    tile_n: int = 0,
    tgmm_tiling: tuple[int, int, int] = (128, 128, 128),
    transpose_rhs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``segment_matmul_pallas_2d`` whose dWeight goes into ``dweight_acc``.

    ``dweight_acc`` comes from ``chunk_dweight_accumulator(rhs)`` for the first
    chunk and from the previous chunk's call after that. ``rhs`` itself gets no
    gradient from this call: its dWeight reaches it through the accumulator.
    Returns the output and the accumulator for the next chunk, or ``None`` for
    the next accumulator when ``last_chunk``. The forward is the unchanged
    ``segment_matmul_pallas_2d``.
    """
    if dweight_acc.shape != rhs.shape:
        raise ValueError(
            f"dweight_acc shape {tuple(dweight_acc.shape)} must match the weight"
            f" shape {tuple(rhs.shape)}"
        )
    if group_sizes.dtype != torch.int32:
        group_sizes = group_sizes.to(torch.int32)
    args = (
        lhs,
        rhs,
        group_sizes,
        dweight_acc,
        tile_m,
        tile_k,
        tile_n,
        *tgmm_tiling,
        transpose_rhs,
    )
    if last_chunk:
        return _GroupedMMAccumulateLast.apply(*args), None
    out, next_acc = _GroupedMMAccumulate.apply(*args)
    return out, next_acc


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


__all__ = [
    "chunk_dweight_accumulator",
    "segment_matmul_pallas_2d",
    "segment_matmul_pallas_2d_accumulate",
]
