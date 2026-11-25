"""CPU (Pallas interpret mode) tests for the RoE fused-layout expert kernels."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from torchtitan.experiments.tpu.kernels import roe_fused_gmm as rfg


def _bf16(rng, shape, scale=1.0):
  return jnp.asarray(rng.standard_normal(shape) * scale, dtype=jnp.bfloat16)


def _group_rows(group_sizes):
  offsets = np.concatenate([[0], np.cumsum(group_sizes)])
  return offsets, int(offsets[-1])


class RoEFusedGmmInterpretTest(absltest.TestCase):

  def test_ragged_gather_2d_matches_gather_reshape(self):
    rng = np.random.default_rng(0)
    g, d0 = 96, 8
    x = _bf16(rng, (g, d0, 128))
    m = 700
    indices = jnp.asarray(rng.integers(0, g, size=m), dtype=jnp.int32)
    cap = 512
    for start, count, block in (
        (37, 300, 128),
        (0, 512, 256),
        (5, 1, 128),
        (130, 129, 128),
    ):
      out = rfg.ragged_gather_2d(
          x,
          indices,
          jnp.int32(start),
          jnp.int32(count),
          cap,
          block_size=block,
          interpret=True,
      )
      self.assertEqual(out.shape, (cap, d0 * 128))
      want = x[indices[start : start + count]].reshape(count, d0 * 128)
      np.testing.assert_array_equal(
          np.asarray(out[:count]).view(np.uint16),
          np.asarray(want).view(np.uint16),
          err_msg=f"start={start} count={count} block={block}",
      )

  def _check_gmm_out3d(
      self, transpose_rhs, group_sizes=(0, 100, 37, 0, 200, 50)
  ):
    rng = np.random.default_rng(1)
    tile_m, m, k, n = 128, 512, 256, 1024
    group_sizes = np.array(group_sizes, dtype=np.int32)
    offsets, total = _group_rows(group_sizes)
    lhs = _bf16(rng, (m, k))
    rhs = _bf16(rng, (len(group_sizes), k, n), 0.05)
    rhs_in = jnp.swapaxes(rhs, 1, 2) if transpose_rhs else rhs
    out = rfg.gmm_out3d(
        lhs,
        rhs_in,
        jnp.asarray(group_sizes),
        tile_m=tile_m,
        transpose_rhs=transpose_rhs,
        interpret=True,
    )
    self.assertEqual(out.shape, (m, n // 128, 128))
    out2d = np.asarray(out.reshape(m, n)[:total].astype(jnp.float32))
    want = np.zeros((total, n), np.float32)
    for gi in range(len(group_sizes)):
      lo, hi = offsets[gi], offsets[gi + 1]
      if hi > lo:
        want[lo:hi] = np.asarray(
            jnp.dot(lhs[lo:hi], rhs[gi], preferred_element_type=jnp.float32)
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
    np.testing.assert_allclose(out2d, want, rtol=1e-2, atol=1e-2)

  def test_gmm_out3d(self):
    self._check_gmm_out3d(False)

  def test_gmm_out3d_transposed_rhs(self):
    self._check_gmm_out3d(True)

  def test_gmm_out3d_full_tiles(self):
    self._check_gmm_out3d(False, (300, 0, 200))
    self._check_gmm_out3d(True, (300, 0, 200))

  def test_accumulate_dinput_3d(self):
    rng = np.random.default_rng(2)
    tile_m, m, k, n = 128, 384, 1024, 256
    group_sizes = np.array([60, 0, 130, 90], dtype=np.int32)
    self._check_dinput(group_sizes, tile_m, m, k, n, rng)
    self._check_dinput(
        np.array([300, 0, 70], dtype=np.int32), tile_m, m, k, n, rng
    )

  def _check_dinput(self, group_sizes, tile_m, m, k, n, rng):
    offsets, total = _group_rows(group_sizes)
    dout = _bf16(rng, (m, n))
    rhs = _bf16(rng, (len(group_sizes), k, n), 0.05)
    existing = _bf16(rng, (m, k))
    out = rfg._accumulate_dinput_3d(  # pylint: disable=protected-access
        dout,
        rhs,
        existing,
        jnp.asarray(group_sizes),
        tile_m=tile_m,
        interpret=True,
    )
    out2d = np.asarray(out.reshape(m, k)[:total].astype(jnp.float32))
    want = np.zeros((total, k), np.float32)
    for gi in range(len(group_sizes)):
      lo, hi = offsets[gi], offsets[gi + 1]
      if hi > lo:
        prod = (
            jax.lax.dot_general(
                dout[lo:hi],
                rhs[gi],
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        want[lo:hi] = np.asarray(
            (existing[lo:hi].astype(jnp.float32) + prod)
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
    np.testing.assert_allclose(out2d, want, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
  absltest.main()
