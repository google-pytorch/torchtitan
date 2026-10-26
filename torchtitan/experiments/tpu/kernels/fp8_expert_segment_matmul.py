"""FP8 grouped matmul for dynamically routed TPU expert layers.

Same contract as `expert_segment_matmul.segment_matmul_pallas_2d`, but every
GEMM runs on FP8 operands through tokamax's Mosaic-TPU GMM/TGMM kernels. The
numerics follow MaxText's `quantization=fp8_full` recipe with the tokamax v1
GMM (`maxtext/kernels/megablox/ops.py`):

* forward ``Y = X @ W``: X quantized to e4m3 per row (over K); W to e4m3 per
  output column, one scale shared by every expert (over experts and K).
* dInput ``dX = dY @ W.T``: W's per-column scale is folded into dY, which is
  quantized to e5m2 per row (over N); the forward's raw e4m3 W is reused.
* dWeight ``dW = X.T @ dY``: X's per-row scale is folded into dY, which is
  quantized to e5m2 per column (over tokens); the forward's raw e4m3 X is
  reused.

All scales are absmax, computed in FP32 and applied by the kernels to their
FP32 accumulators. Quantization happens in XLA outside the kernels, with a
proper FP8 cast (tokamax's in-kernel quantizer rounds to integers, which is
meant for int8).
"""

import jax
import jax.numpy as jnp
import jaxtyping
import qwix

# Workaround for https://github.com/openxla/tokamax/issues/240 (as in
# tokamax_splash_attention.py): typeguard cannot parse the jaxtyping shape
# strings tokamax applies at import time, so make `jaxtyped` a pass-through
# for the duration of the import.
_real_jaxtyped = jaxtyping.jaxtyped


def _passthrough_jaxtyped(*args, **kwargs):
    del kwargs
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return lambda f: f


jaxtyping.jaxtyped = _passthrough_jaxtyped
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_kernel as tokamax_gmm  # noqa: E402  pylint: disable=g-import-not-at-top

jaxtyping.jaxtyped = _real_jaxtyped

from torchtitan.experiments.tpu.kernels import fp8_scale  # noqa: E402  pylint: disable=g-import-not-at-top

FWD_DTYPE = jnp.float8_e4m3fn
BWD_DTYPE = jnp.float8_e5m2
_PRECISION = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)

Tiling = tuple[int, int, int]


def _quantize(
    x: jax.Array, qdtype: jnp.dtype, axis: int | tuple[int, ...]
) -> tuple[jax.Array, jax.Array]:
    """Absmax-quantizes `x`, one FP32 scale per slice along reduced `axis`."""
    amax = jnp.max(jnp.abs(x), axis=axis, keepdims=True).astype(jnp.float32)
    scale = fp8_scale.scale_from_amax(amax, qdtype)
    return (x.astype(jnp.float32) / scale).astype(qdtype), scale


def _resolve(tiling: Tiling, k_dim: int, n_dim: int) -> Tiling:
    """Zero-valued K/N tiles select the full dimension."""
    tm, tk, tn = tiling
    return tm, tk if tk > 0 else k_dim, tn if tn > 0 else n_dim


def quantize_operands(
    lhs: jax.Array, rhs: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Returns ``(lhs_q [M,K], lhs_scale [M,1], rhs_q [G,K,N], rhs_scale [1,N])``."""
    lhs_q, lhs_scale = _quantize(lhs, FWD_DTYPE, axis=1)
    rhs_q, rhs_scale = _quantize(rhs, FWD_DTYPE, axis=(0, 1))
    return lhs_q, lhs_scale, rhs_q, rhs_scale.reshape(1, -1)


def fp8_gmm_fwd_quantized(
    lhs_q: jax.Array,
    lhs_scale: jax.Array,
    rhs_q: jax.Array,
    rhs_scale: jax.Array,
    group_sizes: jax.Array,
    tiling: Tiling,
    out_dtype: jnp.dtype,
    interpret: bool = False,
) -> jax.Array:
    """``lhs[M, K] @ rhs[G, K, N]`` per group, on pre-quantized e4m3 operands."""
    num_groups, k_dim, n_dim = rhs_q.shape
    # tokamax wants one scale per (group, column); the column scale is shared.
    rhs_scale = jnp.broadcast_to(rhs_scale[None], (num_groups, 1, n_dim))
    return tokamax_gmm.gmm(
        qwix.QArray(qvalue=lhs_q, scale=lhs_scale),
        qwix.QArray(qvalue=rhs_q, scale=rhs_scale),
        group_sizes,
        precision=_PRECISION,
        out_dtype=out_dtype,
        tiling=_resolve(tiling, k_dim, n_dim),
        interpret=interpret,
    )


def fp8_gmm_bwd_quantized(
    dout: jax.Array,
    lhs_q: jax.Array,
    lhs_scale: jax.Array,
    rhs_q: jax.Array,
    rhs_scale: jax.Array,
    group_sizes: jax.Array,
    dlhs_tiling: Tiling,
    drhs_tiling: Tiling,
    rhs_dtype: jnp.dtype,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Returns ``(dlhs[M, K], drhs[G, K, N])`` from the forward's FP8 operands."""
    num_groups, k_dim, n_dim = rhs_q.shape
    dout_f32 = dout.astype(jnp.float32)

    # dX = dY @ W.T contracts over N, along which W is scaled: fold W's column
    # scale into dY so the forward's raw e4m3 W can be used as is.
    dlhs_dout, dlhs_dout_scale = _quantize(dout_f32 * rhs_scale, BWD_DTYPE, 1)
    grad_lhs = tokamax_gmm.gmm(
        qwix.QArray(qvalue=dlhs_dout, scale=dlhs_dout_scale),  # scale [M, 1]
        rhs_q,
        group_sizes,
        precision=_PRECISION,
        out_dtype=dout.dtype,
        tiling=_resolve(dlhs_tiling, n_dim, k_dim),
        transpose_rhs=True,
        interpret=interpret,
    )

    # dW = X.T @ dY contracts over tokens, along which X is scaled: fold X's row
    # scale into dY, then give dY one scale per column.
    drhs_dout, drhs_dout_scale = _quantize(dout_f32 * lhs_scale, BWD_DTYPE, 0)
    grad_rhs = tokamax_gmm.tgmm(
        jnp.swapaxes(lhs_q, 0, 1),  # [K, M], raw e4m3 values
        qwix.QArray(qvalue=drhs_dout, scale=drhs_dout_scale),  # scale [1, N]
        group_sizes,
        precision=_PRECISION,
        # Store in the weight's dtype; autograd would cast to it anyway.
        out_dtype=rhs_dtype,
        tiling=_resolve(drhs_tiling, k_dim, n_dim),
        num_actual_groups=num_groups,
        interpret=interpret,
    )
    return grad_lhs, grad_rhs


def fp8_gmm_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    tiling: Tiling,
    interpret: bool = False,
) -> jax.Array:
    """``lhs[M, K] @ rhs[G, K, N]`` per group, quantizing both operands."""
    return fp8_gmm_fwd_quantized(
        *quantize_operands(lhs, rhs),
        group_sizes,
        tiling,
        lhs.dtype,
        interpret=interpret,
    )


def fp8_gmm_bwd(
    dout: jax.Array,
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    dlhs_tiling: Tiling,
    drhs_tiling: Tiling,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`fp8_gmm_bwd_quantized` on BF16 inputs, quantized as the forward does."""
    return fp8_gmm_bwd_quantized(
        dout,
        *quantize_operands(lhs, rhs),
        group_sizes,
        dlhs_tiling,
        drhs_tiling,
        rhs.dtype,
        interpret=interpret,
    )


__all__ = [
    "fp8_gmm_bwd",
    "fp8_gmm_bwd_quantized",
    "fp8_gmm_fwd",
    "fp8_gmm_fwd_quantized",
    "quantize_operands",
]
