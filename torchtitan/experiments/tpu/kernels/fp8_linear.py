"""TPU-native FP8 linear layer.

torchao's `Float8Linear` targets CUDA `_scaled_mm`: it quantizes every
operand once per contraction axis in separate reduce and convert passes, and
forces the operand layouts cuBLAS needs. On TPU those layouts become real
transposes, and the extra passes cost more than FP8 saves on Qwen3's attention
projections. Here the GEMMs are plain XLA `dot_general`s inside a `jax_op`,
and every row-wise quantization is one Pallas pass (read BF16, write FP8 plus
an FP32 row scale) instead of XLA's separate absmax and convert passes.

Numerics (absmax, FP32 scales applied to FP32 accumulators), following
MaxText's `fp8_full`:

* forward ``y = x @ w.T``: x in e4m3, one scale per row (over K); w in e4m3,
  one scale per output channel (over K).
* dInput ``dx = dy @ w``: w's channel scale is folded into dy, which is
  quantized to e5m2 per row (over N); the forward's e4m3 w is reused.
* dWeight ``dw = dy.T @ x``: BF16 by default. Contracting over tokens needs
  dy scaled per column, i.e. a second full pass over dy plus a transpose,
  which costs more than the FP8 GEMM saves at these shapes. With
  ``fp8_grad_weight`` it is FP8 as in MaxText: x's row scale is folded into
  dy, which is quantized to e5m2 per column; the forward's e4m3 x is reused.
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from torchtitan.experiments.tpu.kernels import fp8_scale

FWD_DTYPE = jnp.float8_e4m3fn
BWD_DTYPE = jnp.float8_e5m2
_ROW_TILE = 256


def _quantize(
    x: jax.Array, qdtype: jnp.dtype, axis: int
) -> tuple[jax.Array, jax.Array]:
    """Absmax-quantizes `x`, one FP32 scale per slice along reduced `axis`."""
    amax = jnp.max(jnp.abs(x), axis=axis, keepdims=True).astype(jnp.float32)
    scale = fp8_scale.scale_from_amax(amax, qdtype)
    return (x.astype(jnp.float32) / scale).astype(qdtype), scale


def _row_quant_kernel(x_ref, mult_ref, q_ref, scale_ref):
    v = x_ref[...].astype(jnp.float32) * mult_ref[...]
    amax = jnp.max(jnp.abs(v), axis=1, keepdims=True)
    scale = fp8_scale.scale_from_amax(amax, q_ref.dtype)
    q_ref[...] = (v / scale).astype(q_ref.dtype)
    scale_ref[...] = scale


def row_quantize(
    x: jax.Array,
    qdtype: jnp.dtype,
    col_mult: jax.Array | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """``x * col_mult`` as ``(qdtype [M, K], FP32 row scale [M, 1])``, one pass."""
    rows, cols = x.shape
    if col_mult is None:
        col_mult = jnp.ones((1, cols), jnp.float32)
    tile_m = min(_ROW_TILE, rows)
    return pl.pallas_call(
        _row_quant_kernel,
        grid=(pl.cdiv(rows, tile_m),),
        in_specs=[
            pl.BlockSpec((tile_m, cols), lambda i: (i, 0)),
            pl.BlockSpec((1, cols), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((tile_m, cols), lambda i: (i, 0)),
            pl.BlockSpec((tile_m, 1), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, cols), qdtype),
            jax.ShapeDtypeStruct((rows, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="fp8_row_quantize",
    )(x, col_mult.astype(jnp.float32).reshape(1, cols))


def _dot(lhs, rhs, contract_lhs: int, contract_rhs: int) -> jax.Array:
    return jax.lax.dot_general(
        lhs,
        rhs,
        (((contract_lhs,), (contract_rhs,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


def fp8_linear_fwd(
    x: jax.Array, w: jax.Array, interpret: bool = False
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Returns ``(y [...,N], x_q [...,K], x_scale [...,1], w_q [N,K], w_scale [N,1])``.

    The GEMM keeps x's leading dims so XLA can lay y out for its consumer (a
    flat [M, N] result forced a relayout before the attention head reshape).
    """
    lead, k = x.shape[:-1], x.shape[-1]
    x_q, x_scale = row_quantize(
        x.reshape(-1, k), FWD_DTYPE, interpret=interpret
    )
    x_q, x_scale = x_q.reshape(*lead, k), x_scale.reshape(*lead, 1)
    w_q, w_scale = _quantize(w, FWD_DTYPE, axis=1)  # small; XLA is fine
    y = _dot(x_q, w_q, x.ndim - 1, 1) * x_scale * w_scale.reshape(-1)
    return y.astype(x.dtype), x_q, x_scale, w_q, w_scale


def _flat(a: jax.Array) -> jax.Array:
    return a.reshape(-1, a.shape[-1])


def fp8_linear_dx(
    dy: jax.Array, w_q: jax.Array, w_scale: jax.Array, interpret: bool = False
) -> jax.Array:
    """``dy @ w`` with w's channel scale folded into the e5m2 dy."""
    dy_q, dy_scale = row_quantize(
        _flat(dy),
        BWD_DTYPE,
        col_mult=w_scale.reshape(1, -1),
        interpret=interpret,
    )
    dx = (_dot(dy_q, w_q, 1, 0) * dy_scale).astype(dy.dtype)
    return dx.reshape(*dy.shape[:-1], w_q.shape[1])


def fp8_linear_bwd(
    dy: jax.Array,
    x_q: jax.Array,
    x_scale: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """``(dx, dw)``, both FP8 (MaxText `fp8_full`)."""
    dx = fp8_linear_dx(dy, w_q, w_scale, interpret=interpret)
    # dw contracts over tokens, along which x is scaled.
    dw_q, dw_scale = _quantize(
        _flat(dy).astype(jnp.float32) * _flat(x_scale), BWD_DTYPE, 0
    )
    dw = _dot(dw_q, _flat(x_q), 0, 0) * dw_scale.reshape(-1, 1)
    return dx, dw.astype(dy.dtype)


def fp8_linear_bwd_hp_dw(
    dy: jax.Array,
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """``(dx, dw)`` with FP8 dx and BF16 dw."""
    dx = fp8_linear_dx(dy, w_q, w_scale, interpret=interpret)
    dw = _dot(_flat(dy), _flat(x), 0, 0).astype(dy.dtype)
    return dx, dw


# FP8 residuals cross the torch boundary bitcast to int8.
def _to_i8(x: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(x, jnp.int8)


def _from_i8(x: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(x, FWD_DTYPE)


@pallas.jax_op("tt_fp8::linear")
def _fp8_linear_op(
    x: jax.Array, w: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    y, x_q, x_scale, w_q, w_scale = fp8_linear_fwd(x, w)
    return y, _to_i8(x_q), x_scale, _to_i8(w_q), w_scale


@pallas.jax_op("tt_fp8::linear_bwd")
def _fp8_linear_bwd_op(
    dy: jax.Array,
    x_q: jax.Array,
    x_scale: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return fp8_linear_bwd(dy, _from_i8(x_q), x_scale, _from_i8(w_q), w_scale)


@pallas.jax_op("tt_fp8::linear_hp_dw")
def _fp8_linear_hp_dw_op(
    x: jax.Array, w: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    y, _, _, w_q, w_scale = fp8_linear_fwd(x, w)
    return y, _to_i8(w_q), w_scale


@pallas.jax_op("tt_fp8::linear_hp_dw_bwd")
def _fp8_linear_hp_dw_bwd_op(
    dy: jax.Array, x: jax.Array, w_q: jax.Array, w_scale: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return fp8_linear_bwd_hp_dw(dy, x, _from_i8(w_q), w_scale)


def _setup_context(ctx, inputs, output):
    del inputs
    _, x_q, x_scale, w_q, w_scale = output
    ctx.mark_non_differentiable(x_q, x_scale, w_q, w_scale)
    ctx.save_for_backward(x_q, x_scale, w_q, w_scale)


def _backward(ctx, grad_out, *unused_residual_grads):
    del unused_residual_grads
    return tuple(_fp8_linear_bwd_op(grad_out.contiguous(), *ctx.saved_tensors))


def _hp_dw_setup_context(ctx, inputs, output):
    _, w_q, w_scale = output
    ctx.mark_non_differentiable(w_q, w_scale)
    ctx.save_for_backward(inputs[0], w_q, w_scale)


def _hp_dw_backward(ctx, grad_out, *unused_residual_grads):
    del unused_residual_grads
    return tuple(
        _fp8_linear_hp_dw_bwd_op(grad_out.contiguous(), *ctx.saved_tensors)
    )


_fp8_linear_op.register_autograd(_backward, setup_context=_setup_context)
_fp8_linear_hp_dw_op.register_autograd(
    _hp_dw_backward, setup_context=_hp_dw_setup_context
)


@pallas.jax_op("tt_fp8::linear_fwd_only")
def _fp8_linear_fwd_only_op(x: jax.Array, w: jax.Array) -> jax.Array:
    return fp8_linear_fwd(x, w)[0]


def _fwd_only_setup_context(ctx, inputs, output):
    del output
    ctx.save_for_backward(*inputs)


def _fwd_only_backward(ctx, grad_out):
    # Plain BF16 GEMMs: XLA consumes dy in whatever layout it arrives in, while
    # quantizing dy needs a row-major copy plus an extra pass.
    x, w = ctx.saved_tensors
    dx = grad_out @ w
    dw = grad_out.reshape(-1, grad_out.shape[-1]).t() @ x.reshape(
        -1, x.shape[-1]
    )
    return dx, dw


_fp8_linear_fwd_only_op.register_autograd(
    _fwd_only_backward, setup_context=_fwd_only_setup_context
)


def fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    fp8_grad_weight: bool = False,
    fp8_grad_input: bool = True,
) -> torch.Tensor:
    """``x @ weight.T`` in FP8 (no bias); see the module docstring."""
    if not fp8_grad_input:
        return _fp8_linear_fwd_only_op(x, weight)
    op = _fp8_linear_op if fp8_grad_weight else _fp8_linear_hp_dw_op
    return op(x, weight)[0]


__all__ = [
    "fp8_linear",
    "fp8_linear_bwd",
    "fp8_linear_bwd_hp_dw",
    "fp8_linear_fwd",
    "row_quantize",
]
