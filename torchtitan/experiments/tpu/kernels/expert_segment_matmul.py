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
from jax.experimental.pallas.ops.tpu.megablox.gmm import (
    gmm as megablox_gmm,
    tgmm as megablox_tgmm,
)
from torch_tpu._internal import pallas
from torchtitan.experiments.tpu.kernels.paired_segment_matmul import (
    grouped_dweight,
)
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


def _grouped_dinput(dout, rhs, group_sizes, tile_m, tile_k, tile_n, transpose_rhs):
    if transpose_rhs:
        _, n_dim, k_dim = rhs.shape
    else:
        _, k_dim, n_dim = rhs.shape
    return call_named(
        megablox_gmm,
        "moe_gmm_dgrad",
        dout,
        rhs,
        group_sizes=group_sizes,
        preferred_element_type=dout.dtype,
        tiling=_tiling(tile_m, tile_n, tile_k, n_dim, k_dim),
        transpose_rhs=not transpose_rhs,
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
    grad_lhs = _grouped_dinput(
        dout, rhs, group_sizes, tile_m, tile_k, tile_n, transpose_rhs
    )
    if transpose_rhs:
        tgmm_lhs, tgmm_rhs = jnp.swapaxes(dout, 0, 1), lhs
        tiling = (tgmm_tile_m, tgmm_tile_n, tgmm_tile_k)
    else:
        tgmm_lhs, tgmm_rhs = jnp.swapaxes(lhs, 0, 1), dout
        tiling = (tgmm_tile_m, tgmm_tile_k, tgmm_tile_n)
    grad_rhs = call_named(
        megablox_tgmm,
        "moe_tgmm_wgrad",
        tgmm_lhs,
        tgmm_rhs,
        group_sizes=group_sizes,
        preferred_element_type=rhs.dtype,
        tiling=_tiling(*tiling, tgmm_lhs.shape[0], tgmm_rhs.shape[1]),
        interpret=False,
    )
    return grad_lhs, grad_rhs


@pallas.jax_op("tt_experts::grouped_mm_bwd_fp32_dweight")
def _grouped_mm_bwd_fp32_dweight(
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
    """``grouped_mm_bwd`` with the BF16-rounded dWeight stored as FP32.

    Autograd never selects this op: it would round the FP32 dWeight back to the
    BF16 weight dtype. The FP32-dWeight graph pass swaps it in where that
    rounding is immediately undone by an FP32 widening. The dWeight comes from
    the single-output form of the paired TGMM kernel, which keeps the output
    tile whole, so the TGMM tile of the output's trailing axis must be full.
    """
    grad_lhs = _grouped_dinput(
        dout, rhs, group_sizes, tile_m, tile_k, tile_n, transpose_rhs
    )
    if transpose_rhs:
        dweight_lhs, dweight_dout = dout, lhs
        tile_k, tile_n = tgmm_tile_n, tgmm_tile_k
    else:
        dweight_lhs, dweight_dout = lhs, dout
        tile_k, tile_n = tgmm_tile_k, tgmm_tile_n
    if tile_n not in (0, dweight_dout.shape[1]):
        raise ValueError("FP32 dWeight requires a full TGMM output tile")
    (grad_rhs,) = grouped_dweight(
        dweight_lhs,
        (dweight_dout,),
        group_sizes,
        tile_m=tgmm_tile_m,
        tile_k=tile_k,
        out_dtype=jnp.float32,
        name="moe_tgmm_wgrad",
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


__all__ = ["segment_matmul_pallas_2d"]
