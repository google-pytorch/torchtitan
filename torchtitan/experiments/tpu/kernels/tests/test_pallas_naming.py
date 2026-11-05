"""Tests for pallas_naming."""

from absl.testing import absltest
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from torchtitan.experiments.tpu.kernels.pallas_naming import call_named


def _copy_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...]


def _copy(x):
    return pl.pallas_call(
        _copy_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=pltpu.InterpretParams(),
    )(x)


_jitted_copy = jax.jit(_copy)


def _lowered_text(fn, *args):
    return jax.jit(fn).lower(*args).as_text(debug_info=True)


class CallNamedTest(absltest.TestCase):

    def setUp(self):
        super().setUp()
        self.x = jnp.zeros((8, 128), jnp.float32)

    def test_scopes_kernel(self):
        text = _lowered_text(
            lambda x: call_named(_copy, "my_named_kernel", x), self.x
        )
        self.assertIn("my_named_kernel", text)

    def test_traces_jitted_fn_inline(self):
        jaxpr = jax.make_jaxpr(
            lambda x: call_named(_jitted_copy, "my_named_kernel", x)
        )(self.x)
        primitives = [eqn.primitive.name for eqn in jaxpr.eqns]
        self.assertNotIn("jit", primitives)
        self.assertNotIn("pjit", primitives)

    def test_scope_ends_after_call(self):
        call_named(_copy, "my_named_kernel", self.x)
        self.assertNotIn("my_named_kernel", _lowered_text(_copy, self.x))


if __name__ == "__main__":
    absltest.main()
