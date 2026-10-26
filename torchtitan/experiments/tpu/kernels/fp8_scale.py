"""FP8 scale computation shared by the TPU FP8 kernels."""

import jax
import jax.numpy as jnp

# Smallest normal FP32. TPUs flush subnormals to zero, so a slice whose
# `amax / fmax` underflows would otherwise get a zero scale and quantize to
# inf/NaN. Flooring the scale at this value keeps `x / scale <= fmax` (since
# `|x| <= amax < fmax * MIN_SCALE`) and leaves every other scale unchanged.
MIN_SCALE = float(jnp.finfo(jnp.float32).tiny)


def scale_from_amax(amax: jax.Array, qdtype: jnp.dtype) -> jax.Array:
    """FP32 absmax scale ``amax / fmax``, floored at the smallest normal FP32."""
    return jnp.maximum(amax / float(jnp.finfo(qdtype).max), MIN_SCALE)
