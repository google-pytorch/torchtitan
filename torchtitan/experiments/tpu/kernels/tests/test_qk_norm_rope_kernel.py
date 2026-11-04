"""CPU Pallas interpret-mode and geometry tests for `qk_norm_rope_kernel`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from torchtitan.experiments.tpu.kernels import qk_norm_rope_kernel as kernel


class QkNormRopeKernelTest(parameterized.TestCase):

  def test_geometry_validation_accepts_and_rejects_expected_cases(self):
    self.assertIsNone(
        kernel.validate_geometry(
            batch=2,
            seq=2048,
            hidden_dim=18 * 64,
            heads=18,
            head_dim=64,
            cos_shape=(1, 2048, 64),
            sin_shape=(1, 2048, 64),
            block_s=512,
        )
    )
    self.assertIsNone(
        kernel.validate_geometry(
            batch=2,
            seq=32,
            hidden_dim=2 * 128,
            heads=2,
            head_dim=128,
            cos_shape=(2, 32, 128),
            sin_shape=(2, 32, 128),
            block_s=16,
        )
    )
    # Sub-64 head_dim (e.g. 32) is rejected.
    self.assertIn(
        "head_dim",
        kernel.validate_geometry(
            batch=1,
            seq=16,
            hidden_dim=4 * 32,
            heads=4,
            head_dim=32,
            cos_shape=(1, 16, 32),
            sin_shape=(1, 16, 32),
            block_s=16,
        )
        or "",
    )
    # Odd head count at D=64 violates 128-lane group alignment.
    self.assertIn(
        "group width",
        kernel.validate_geometry(
            batch=1,
            seq=16,
            hidden_dim=3 * 64,
            heads=3,
            head_dim=64,
            cos_shape=(1, 16, 64),
            sin_shape=(1, 16, 64),
            block_s=16,
        )
        or "",
    )
    # Projection width mismatch.
    self.assertIn(
        "projection width",
        kernel.validate_geometry(
            batch=1,
            seq=16,
            hidden_dim=256,
            heads=2,
            head_dim=64,
            cos_shape=(1, 16, 64),
            sin_shape=(1, 16, 64),
            block_s=16,
        )
        or "",
    )
    # Invalid cos/sin batch broadcast dimension (2 != 1 and 2 != batch=4).
    self.assertIn(
        "cos/sin shape",
        kernel.validate_geometry(
            batch=4,
            seq=16,
            hidden_dim=2 * 64,
            heads=2,
            head_dim=64,
            cos_shape=(2, 16, 64),
            sin_shape=(2, 16, 64),
            block_s=16,
        )
        or "",
    )
    # Non-8-sublane-aligned block_s.
    self.assertIn(
        "aligned tiles",
        kernel.validate_geometry(
            batch=1,
            seq=24,
            hidden_dim=2 * 64,
            heads=2,
            head_dim=64,
            cos_shape=(1, 24, 64),
            sin_shape=(1, 24, 64),
            block_s=12,
        )
        or "",
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="d64_multi_tile_batched_qscale",
          batch=2,
          seq=32,
          block_s=16,
          heads=4,
          head_dim=64,
          cos_batch=2,
          q_scale=0.125,
          asymmetric_tables=False,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="d64_four_tiles_unbatched_asymmetric_tables",
          batch=2,
          seq=64,
          block_s=16,
          heads=4,
          head_dim=64,
          cos_batch=1,
          q_scale=1.0,
          asymmetric_tables=True,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="d128_multi_tile_bf16",
          batch=2,
          seq=32,
          block_s=16,
          heads=2,
          head_dim=128,
          cos_batch=1,
          q_scale=0.125,
          asymmetric_tables=True,
          dtype=jnp.bfloat16,
      ),
  )
  def test_interpret_forward_and_backward_match_reference(
      self,
      batch: int,
      seq: int,
      block_s: int,
      heads: int,
      head_dim: int,
      cos_batch: int,
      q_scale: float,
      asymmetric_tables: bool,
      dtype: jnp.dtype,
  ):
    hidden_dim = heads * head_dim
    eps = 1e-5
    keys = jax.random.split(jax.random.PRNGKey(42), 10)
    xq = (jax.random.normal(keys[0], (batch, seq, hidden_dim)) * 1.5 + 0.2).astype(dtype)
    xk = (jax.random.normal(keys[1], (batch, seq, hidden_dim)) * 1.2 - 0.1).astype(dtype)
    xv = jax.random.normal(keys[2], (batch, seq, hidden_dim)).astype(dtype)
    wq = (1.0 + 0.1 * jax.random.normal(keys[3], (hidden_dim,))).astype(dtype)
    wk = (1.0 + 0.1 * jax.random.normal(keys[4], (hidden_dim,))).astype(dtype)

    if asymmetric_tables:
      cos = jax.random.uniform(
          keys[5], (cos_batch, seq, head_dim), minval=-1.0, maxval=1.0
      ).astype(dtype)
      sin = jax.random.uniform(
          keys[6], (cos_batch, seq, head_dim), minval=-1.0, maxval=1.0
      ).astype(dtype)
    else:
      angles = jax.random.uniform(
          keys[5], (cos_batch, seq, head_dim // 2), minval=-3.0, maxval=3.0
      )
      emb = jnp.concatenate([angles, angles], axis=-1)
      cos = jnp.cos(emb).astype(dtype)
      sin = jnp.sin(emb).astype(dtype)

    gq = jax.random.normal(keys[7], (batch, heads, seq, head_dim)).astype(dtype)
    gk = jax.random.normal(keys[8], (batch, heads, seq, head_dim)).astype(dtype)
    gv = jax.random.normal(keys[9], (batch, heads, seq, head_dim)).astype(dtype)

    q_got, k_got, v_got = kernel.qk_norm_rope_fwd(
        xq,
        xk,
        xv,
        wq,
        wk,
        cos,
        sin,
        heads=heads,
        eps=eps,
        q_scale=q_scale,
        block_s=block_s,
        interpret=True,
    )
    dxq_got, dxk_got, dxv_got, dwq_got, dwk_got = kernel.qk_norm_rope_bwd(
        xq,
        xk,
        wq,
        wk,
        cos,
        sin,
        gq,
        gk,
        gv,
        eps=eps,
        q_scale=q_scale,
        block_s=block_s,
        interpret=True,
    )

    def ref_fn(xq_, xk_, xv_, wq_, wk_):
      return kernel.qk_norm_rope_reference(
          xq_.astype(jnp.float32),
          xk_.astype(jnp.float32),
          xv_.astype(jnp.float32),
          wq_.astype(jnp.float32),
          wk_.astype(jnp.float32),
          cos.astype(jnp.float32),
          sin.astype(jnp.float32),
          heads=heads,
          eps=eps,
          q_scale=q_scale,
      )

    (q_ref, k_ref, v_ref), vjp_fn = jax.vjp(ref_fn, xq, xk, xv, wq, wk)
    dxq_ref, dxk_ref, dxv_ref, dwq_ref, dwk_ref = vjp_fn(
        (gq.astype(jnp.float32), gk.astype(jnp.float32), gv.astype(jnp.float32))
    )

    tol = 1.5e-2 if dtype == jnp.bfloat16 else 1e-5
    for name, got, ref in (
        ("q", q_got, q_ref),
        ("k", k_got, k_ref),
        ("v", v_got, v_ref),
        ("dxq", dxq_got, dxq_ref),
        ("dxk", dxk_got, dxk_ref),
        ("dxv", dxv_got, dxv_ref),
        ("dwq", dwq_got, dwq_ref),
        ("dwk", dwk_got, dwk_ref),
    ):
      got_f = np.asarray(got, dtype=np.float32)
      ref_f = np.asarray(ref, dtype=np.float32)
      rel_err = np.linalg.norm(got_f - ref_f) / max(
          np.linalg.norm(ref_f), 1e-12
      )
      self.assertLessEqual(rel_err, tol, msg=f"Mismatch in {name}: {rel_err}")


if __name__ == "__main__":
  absltest.main()
