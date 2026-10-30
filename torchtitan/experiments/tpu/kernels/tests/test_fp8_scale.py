"""Tests for fp8_scale."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from torchtitan.experiments.tpu.kernels import fp8_scale

_E4M3 = jnp.float8_e4m3fn
_E5M2 = jnp.float8_e5m2


class ScaleFromAmaxTest(absltest.TestCase):

    def test_normal_scales_are_amax_over_fmax(self):
        amax = jnp.array([[1.0], [448.0], [3e-5]], jnp.float32)
        for qdtype in (_E4M3, _E5M2):
            np.testing.assert_array_equal(
                fp8_scale.scale_from_amax(amax, qdtype),
                amax / float(jnp.finfo(qdtype).max),
            )

    def test_zero_and_subnormal_scales_are_floored(self):
        # 1e-40 / 448 is subnormal in FP32; TPUs flush it to zero.
        amax = jnp.array([[0.0], [1e-40], [1e-36]], jnp.float32)
        scale = fp8_scale.scale_from_amax(amax, _E4M3)
        np.testing.assert_array_equal(
            scale, jnp.full_like(amax, fp8_scale.MIN_SCALE)
        )

    def test_floored_scale_keeps_values_in_range(self):
        # |x| <= amax < fmax * MIN_SCALE, so x / scale stays finite.
        x = jnp.array([[1e-38, -1e-38, 0.0]], jnp.float32)
        amax = jnp.max(jnp.abs(x), axis=1, keepdims=True)
        scale = fp8_scale.scale_from_amax(amax, _E4M3)
        q = (x / scale).astype(_E4M3).astype(jnp.float32)
        self.assertTrue(bool(jnp.isfinite(q).all()))
        self.assertLessEqual(
            float(jnp.abs(q).max()), float(jnp.finfo(_E4M3).max)
        )


if __name__ == "__main__":
    absltest.main()
