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

Every operand is therefore quantized once per forward (and once more in the
activation-checkpoint recompute); autograd saves the FP8 values and scales
instead of the BF16 inputs, so the backward never re-reads or re-quantizes X
or W. The W1/W3 projections of a SwiGLU expert share their input, so
`fp8_paired_segment_matmul_pallas_2d` runs them as one GEMM over the
concatenated ``[W1 | W3]`` weight: X is quantized once, and dX comes out of a
single dInput GEMM instead of two GEMMs and an add.

All scales are absmax, computed in FP32 and applied by the kernels to their
FP32 accumulators. Quantization happens in XLA outside the kernels, with a
proper FP8 cast (tokamax's in-kernel quantizer rounds to integers, which is
meant for int8). FP8 residuals cross the torch boundary bitcast to int8.
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

from jax.experimental import pallas as pl  # noqa: E402  pylint: disable=g-import-not-at-top
from jax.experimental.pallas import tpu as pltpu  # noqa: E402  pylint: disable=g-import-not-at-top
import torch  # noqa: E402  pylint: disable=g-import-not-at-top,g-bad-import-order
from torch_tpu._internal import pallas  # noqa: E402  pylint: disable=g-import-not-at-top
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


# ---------------------------------------------------------------------------
# torch ops. The forward returns the FP8 residuals (bitcast to int8, which
# crosses the torch boundary everywhere) so autograd saves those, not BF16.
# ---------------------------------------------------------------------------


def _to_i8(x: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(x, jnp.int8)


def _from_i8(x: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(x, FWD_DTYPE)


def _fwd_with_residuals(lhs, rhs, group_sizes, tiling, interpret=False):
    lhs_q, lhs_scale, rhs_q, rhs_scale = quantize_operands(lhs, rhs)
    out = fp8_gmm_fwd_quantized(
        lhs_q,
        lhs_scale,
        rhs_q,
        rhs_scale,
        group_sizes,
        tiling,
        lhs.dtype,
        interpret=interpret,
    )
    return out, _to_i8(lhs_q), lhs_scale, _to_i8(rhs_q), rhs_scale


def _bwd_from_residuals(
    residuals, dout, dlhs_tiling, drhs_tiling, interpret=False
):
    lhs_q, lhs_scale, rhs_q, rhs_scale, group_sizes = residuals
    return fp8_gmm_bwd_quantized(
        dout,
        _from_i8(lhs_q),
        lhs_scale,
        _from_i8(rhs_q),
        rhs_scale,
        group_sizes,
        dlhs_tiling,
        drhs_tiling,
        dout.dtype,
        interpret=interpret,
    )


_Residuals = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]

# Every forward op takes all nine tiling ints -- forward, dInput, dWeight --
# so autograd can hand the backward ones to the backward op. torch's custom-op
# schema is inferred from annotations, which rules out varargs.


@pallas.jax_op("tt_experts::fp8_grouped_mm")
def _fp8_grouped_mm_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    fwd_tm: int,
    fwd_tk: int,
    fwd_tn: int,
    dlhs_tm: int,
    dlhs_tk: int,
    dlhs_tn: int,
    drhs_tm: int,
    drhs_tk: int,
    drhs_tn: int,
) -> _Residuals:
    del dlhs_tm, dlhs_tk, dlhs_tn, drhs_tm, drhs_tk, drhs_tn
    return _fwd_with_residuals(lhs, rhs, group_sizes, (fwd_tm, fwd_tk, fwd_tn))


@pallas.jax_op("tt_experts::fp8_paired_grouped_mm")
def _fp8_paired_grouped_mm_fwd(
    lhs: jax.Array,
    rhs_a: jax.Array,
    rhs_b: jax.Array,
    group_sizes: jax.Array,
    fwd_tm: int,
    fwd_tk: int,
    fwd_tn: int,
    dlhs_tm: int,
    dlhs_tk: int,
    dlhs_tn: int,
    drhs_tm: int,
    drhs_tk: int,
    drhs_tn: int,
) -> _Residuals:
    del dlhs_tm, dlhs_tk, dlhs_tn, drhs_tm, drhs_tk, drhs_tn
    # XLA fuses the concatenation into the FP8 cast that writes `rhs_q`.
    rhs = jnp.concatenate([rhs_a, rhs_b], axis=2)
    return _fwd_with_residuals(lhs, rhs, group_sizes, (fwd_tm, fwd_tk, fwd_tn))


@pallas.jax_op("tt_experts::fp8_grouped_mm_bwd")
def _fp8_grouped_mm_bwd(
    dout: jax.Array,
    lhs_q: jax.Array,
    lhs_scale: jax.Array,
    rhs_q: jax.Array,
    rhs_scale: jax.Array,
    group_sizes: jax.Array,
    dlhs_tm: int,
    dlhs_tk: int,
    dlhs_tn: int,
    drhs_tm: int,
    drhs_tk: int,
    drhs_tn: int,
) -> tuple[jax.Array, jax.Array]:
    return _bwd_from_residuals(
        (lhs_q, lhs_scale, rhs_q, rhs_scale, group_sizes),
        dout,
        (dlhs_tm, dlhs_tk, dlhs_tn),
        (drhs_tm, drhs_tk, drhs_tn),
    )


@pallas.jax_op("tt_experts::fp8_paired_grouped_mm_bwd")
def _fp8_paired_grouped_mm_bwd(
    dout: jax.Array,
    lhs_q: jax.Array,
    lhs_scale: jax.Array,
    rhs_q: jax.Array,
    rhs_scale: jax.Array,
    group_sizes: jax.Array,
    dlhs_tm: int,
    dlhs_tk: int,
    dlhs_tn: int,
    drhs_tm: int,
    drhs_tk: int,
    drhs_tn: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    grad_lhs, grad_rhs = _bwd_from_residuals(
        (lhs_q, lhs_scale, rhs_q, rhs_scale, group_sizes),
        dout,
        (dlhs_tm, dlhs_tk, dlhs_tn),
        (drhs_tm, drhs_tk, drhs_tn),
    )
    grad_a, grad_b = jnp.split(grad_rhs, 2, axis=2)
    return grad_lhs, grad_a, grad_b


def _register_autograd(fwd_op, bwd_op, num_weights: int) -> None:
    """Autograd that saves the forward's FP8 residuals instead of its inputs."""

    def setup_context(ctx, inputs, output):
        group_sizes = inputs[1 + num_weights]
        _, lhs_q, lhs_scale, rhs_q, rhs_scale = output
        ctx.mark_non_differentiable(lhs_q, lhs_scale, rhs_q, rhs_scale)
        ctx.save_for_backward(lhs_q, lhs_scale, rhs_q, rhs_scale, group_sizes)
        ctx.num_static = len(inputs) - (2 + num_weights)
        # Drop the forward tiling; the backward op takes dInput and dWeight's.
        ctx.bwd_tilings = tuple(inputs[2 + num_weights + 3 :])

    def backward(ctx, grad_out, *unused_residual_grads):
        del unused_residual_grads
        grads = bwd_op(grad_out, *ctx.saved_tensors, *ctx.bwd_tilings)
        # (lhs, *weights) grads, then group_sizes and the static ints.
        return tuple(grads) + (None,) * (1 + ctx.num_static)

    fwd_op.register_autograd(backward, setup_context=setup_context)


_register_autograd(_fp8_grouped_mm_fwd, _fp8_grouped_mm_bwd, num_weights=1)
_register_autograd(
    _fp8_paired_grouped_mm_fwd, _fp8_paired_grouped_mm_bwd, num_weights=2
)


# ---------------------------------------------------------------------------
# SwiGLU into W2: the op takes the paired W1/W3 output ``[gate | up]`` and
# runs ``silu(gate) * up`` -> per-row e4m3 quantization -> W2 GEMM. With
# ``fused_quant`` a Pallas kernel produces the FP8 hidden and its scales in one
# pass over ``[gate | up]``; the BF16 hidden never reaches HBM. The backward
# returns ``d[gate | up]`` in the paired op's layout, so no split or concat
# sits between the two ops.
# ---------------------------------------------------------------------------

_SWIGLU_QUANT_TILE_M = 256


def _swiglu_f32(gate_up: jax.Array) -> jax.Array:
    gate, up = jnp.split(gate_up.astype(jnp.float32), 2, axis=-1)
    return jax.nn.silu(gate) * up


def _swiglu_quant_kernel(gate_up_ref, q_ref, scale_ref):
    hidden = _swiglu_f32(gate_up_ref[...])
    amax = jnp.max(jnp.abs(hidden), axis=1, keepdims=True)
    scale = fp8_scale.scale_from_amax(amax, FWD_DTYPE)
    q_ref[...] = (hidden / scale).astype(FWD_DTYPE)
    scale_ref[...] = scale


def swiglu_quantize(
    gate_up: jax.Array, fused: bool, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    """``silu(gate) * up`` as ``(e4m3 [M, F], FP32 row scale [M, 1])``."""
    if not fused:
        return _quantize(_swiglu_f32(gate_up), FWD_DTYPE, axis=1)
    rows, two_f = gate_up.shape
    tile_m = min(_SWIGLU_QUANT_TILE_M, rows)
    return pl.pallas_call(
        _swiglu_quant_kernel,
        grid=(pl.cdiv(rows, tile_m),),
        in_specs=[pl.BlockSpec((tile_m, two_f), lambda i: (i, 0))],
        out_specs=[
            pl.BlockSpec((tile_m, two_f // 2), lambda i: (i, 0)),
            pl.BlockSpec((tile_m, 1), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, two_f // 2), FWD_DTYPE),
            jax.ShapeDtypeStruct((rows, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(gate_up)


def _swiglu_grad(gate_up, d_hidden):
    """``d[gate | up]`` of ``silu(gate) * up`` (BF16 out, FP32 math)."""
    num_hidden = gate_up.shape[-1] // 2
    gate = gate_up[:, :num_hidden].astype(jnp.float32)
    up = gate_up[:, num_hidden:].astype(jnp.float32)
    d_hidden = d_hidden.astype(jnp.float32)
    sig = jax.nn.sigmoid(gate)
    d_gate = d_hidden * up * sig * (1.0 + gate * (1.0 - sig))
    d_up = d_hidden * gate * sig
    return d_gate.astype(gate_up.dtype), d_up.astype(gate_up.dtype)


def _swiglu_grad_kernel(gate_up_ref, d_hidden_ref, d_gate_up_ref):
    num_hidden = d_hidden_ref.shape[1]
    d_gate, d_up = _swiglu_grad(gate_up_ref[...], d_hidden_ref[...])
    d_gate_up_ref[:, :num_hidden] = d_gate
    d_gate_up_ref[:, num_hidden:] = d_up


def swiglu_grad(
    gate_up: jax.Array, d_hidden: jax.Array, fused: bool, interpret=False
) -> jax.Array:
    """``d[gate | up]``, in one Pallas pass when ``fused``."""
    if not fused:
        return jnp.concatenate(_swiglu_grad(gate_up, d_hidden), axis=-1)
    rows, two_f = gate_up.shape
    tile_m = min(_SWIGLU_QUANT_TILE_M, rows)
    return pl.pallas_call(
        _swiglu_grad_kernel,
        grid=(pl.cdiv(rows, tile_m),),
        in_specs=[
            pl.BlockSpec((tile_m, two_f), lambda i: (i, 0)),
            pl.BlockSpec((tile_m, two_f // 2), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((tile_m, two_f), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((rows, two_f), gate_up.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(gate_up, d_hidden)


def swiglu_w2_fwd(gate_up, rhs, group_sizes, tiling, fused, interpret=False):
    """Returns ``(out[M, N], h_q (int8 bits), h_scale, rhs_q (int8), rhs_scale)``."""
    h_q, h_scale = swiglu_quantize(gate_up, fused, interpret=interpret)
    rhs_q, rhs_scale = _quantize(rhs, FWD_DTYPE, axis=(0, 1))
    rhs_scale = rhs_scale.reshape(1, -1)
    out = fp8_gmm_fwd_quantized(
        h_q,
        h_scale,
        rhs_q,
        rhs_scale,
        group_sizes,
        tiling,
        gate_up.dtype,
        interpret=interpret,
    )
    return out, _to_i8(h_q), h_scale, _to_i8(rhs_q), rhs_scale


def swiglu_w2_bwd(
    dout,
    gate_up,
    h_q,
    h_scale,
    rhs_q,
    rhs_scale,
    group_sizes,
    dlhs_tiling,
    drhs_tiling,
    fused,
    interpret=False,
):
    """Returns ``(d_gate_up[M, 2F], d_rhs)``; straight-through over the cast."""
    d_hidden, grad_rhs = fp8_gmm_bwd_quantized(
        dout,
        _from_i8(h_q),
        h_scale,
        _from_i8(rhs_q),
        rhs_scale,
        group_sizes,
        dlhs_tiling,
        drhs_tiling,
        dout.dtype,
        interpret=interpret,
    )
    return swiglu_grad(gate_up, d_hidden, fused, interpret=interpret), grad_rhs


@pallas.jax_op("tt_experts::fp8_swiglu_grouped_mm")
def _fp8_swiglu_fwd(
    gate_up: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    fused_quant: int,
    fwd_tm: int,
    fwd_tk: int,
    fwd_tn: int,
    dlhs_tm: int,
    dlhs_tk: int,
    dlhs_tn: int,
    drhs_tm: int,
    drhs_tk: int,
    drhs_tn: int,
) -> _Residuals:
    del dlhs_tm, dlhs_tk, dlhs_tn, drhs_tm, drhs_tk, drhs_tn
    return swiglu_w2_fwd(
        gate_up, rhs, group_sizes, (fwd_tm, fwd_tk, fwd_tn), bool(fused_quant)
    )


@pallas.jax_op("tt_experts::fp8_swiglu_grouped_mm_bwd")
def _fp8_swiglu_bwd(
    dout: jax.Array,
    gate_up: jax.Array,
    h_q: jax.Array,
    h_scale: jax.Array,
    rhs_q: jax.Array,
    rhs_scale: jax.Array,
    group_sizes: jax.Array,
    fused_quant: int,
    dlhs_tm: int,
    dlhs_tk: int,
    dlhs_tn: int,
    drhs_tm: int,
    drhs_tk: int,
    drhs_tn: int,
) -> tuple[jax.Array, jax.Array]:
    return swiglu_w2_bwd(
        dout,
        gate_up,
        h_q,
        h_scale,
        rhs_q,
        rhs_scale,
        group_sizes,
        (dlhs_tm, dlhs_tk, dlhs_tn),
        (drhs_tm, drhs_tk, drhs_tn),
        bool(fused_quant),
    )


def _swiglu_setup_context(ctx, inputs, output):
    gate_up, _, group_sizes, fused_quant, *tilings = inputs
    _, h_q, h_scale, rhs_q, rhs_scale = output
    ctx.mark_non_differentiable(h_q, h_scale, rhs_q, rhs_scale)
    ctx.save_for_backward(gate_up, h_q, h_scale, rhs_q, rhs_scale, group_sizes)
    ctx.num_static = 1 + len(tilings)
    ctx.bwd_static = (fused_quant, *tilings[3:])


def _swiglu_backward(ctx, grad_out, *unused_residual_grads):
    del unused_residual_grads
    d_gate_up, grad_rhs = _fp8_swiglu_bwd(
        grad_out, *ctx.saved_tensors, *ctx.bwd_static
    )
    # gate_up, rhs, group_sizes, fused_quant + tiling ints.
    return (d_gate_up, grad_rhs, None) + (None,) * ctx.num_static


_fp8_swiglu_fwd.register_autograd(
    _swiglu_backward, setup_context=_swiglu_setup_context
)


def _static(fwd: Tiling, dlhs: Tiling, drhs: Tiling) -> tuple[int, ...]:
    static = (*fwd, *dlhs, *drhs)
    assert len(static) == 9, static
    return static


def _int32(group_sizes: torch.Tensor) -> torch.Tensor:
    return (
        group_sizes
        if group_sizes.dtype == torch.int32
        else group_sizes.to(torch.int32)
    )


def fp8_segment_matmul_pallas_2d(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    fwd_tiling: Tiling = (512, 0, 0),
    dlhs_tiling: Tiling = (512, 0, 0),
    drhs_tiling: Tiling = (512, 0, 0),
) -> torch.Tensor:
    """FP8 per-expert matmuls; see the module docstring for the numerics.

    Each tiling is ``(tile_m, tile_k, tile_n)`` in the kernel's own operand
    order, with zero K/N selecting the full dimension: forward ``(M, K, N)``,
    dInput ``(M, N, K)``, dWeight ``(M, K, N)``.
    """
    out, *_ = _fp8_grouped_mm_fwd(
        lhs,
        rhs,
        _int32(group_sizes),
        *_static(fwd_tiling, dlhs_tiling, drhs_tiling),
    )
    return out


def fp8_paired_segment_matmul_pallas_2d(
    lhs: torch.Tensor,
    rhs_a: torch.Tensor,
    rhs_b: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    fwd_tiling: Tiling = (512, 0, 0),
    dlhs_tiling: Tiling = (512, 0, 0),
    drhs_tiling: Tiling = (512, 0, 0),
    split: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """``(lhs @ rhs_a, lhs @ rhs_b)`` per expert as one FP8 GEMM over ``[a | b]``.

    Tilings are as in `fp8_segment_matmul_pallas_2d`, for N = 2 * rhs_a's N.
    With ``split=False`` returns the ``[M, 2N]`` concatenation instead.
    """
    out, *_ = _fp8_paired_grouped_mm_fwd(
        lhs,
        rhs_a,
        rhs_b,
        _int32(group_sizes),
        *_static(fwd_tiling, dlhs_tiling, drhs_tiling),
    )
    return out.chunk(2, dim=-1) if split else out


def fp8_swiglu_segment_matmul_pallas_2d(
    gate_up: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    fused_quant: bool = True,
    fwd_tiling: Tiling = (512, 0, 0),
    dlhs_tiling: Tiling = (512, 0, 0),
    drhs_tiling: Tiling = (512, 0, 0),
) -> torch.Tensor:
    """``(silu(gate) * up) @ rhs`` per expert, with ``gate_up = [gate | up]``.

    The hidden is quantized per row to e4m3 (by a fused Pallas kernel when
    ``fused_quant``); tilings are as in `fp8_segment_matmul_pallas_2d`.
    """
    out, *_ = _fp8_swiglu_fwd(
        gate_up,
        rhs,
        _int32(group_sizes),
        int(fused_quant),
        *_static(fwd_tiling, dlhs_tiling, drhs_tiling),
    )
    return out


__all__ = [
    "fp8_gmm_bwd",
    "fp8_gmm_bwd_quantized",
    "fp8_gmm_fwd",
    "fp8_gmm_fwd_quantized",
    "fp8_paired_segment_matmul_pallas_2d",
    "fp8_segment_matmul_pallas_2d",
    "fp8_swiglu_segment_matmul_pallas_2d",
    "quantize_operands",
]
