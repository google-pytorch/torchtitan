"""TPU-native FP8 linear layer.

Every row-wise quantization in the FP8 linear is one Pallas pass: read BF16,
write FP8 plus an FP32 row scale. XLA instead runs separate absmax and convert
passes. Scales are absmax (`fp8_scale.scale_from_amax`).
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from torchtitan.experiments.tpu.kernels import fp8_scale

_ROW_TILE = 256


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


__all__ = [
    "row_quantize",
]
