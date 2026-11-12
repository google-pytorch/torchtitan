"""Interpret-mode parity tests for the MLA Splash kernels (``mla_splash.py``)."""

import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from torchtitan.experiments.tpu.kernels import mla_splash

_B, _S = 2, 512

# (documents per example, block_q, block_kv, heads, heads_per_step, Dr):
# partial and full documents, square and rectangular tiles, 1/2/4 heads per
# step, and the Dr range (DeepSeek's 64, plus 32 and the 128 maximum).
_CASES = (
    (((100, 300, 112), (512,)), 128, 128, 4, 2, 64),
    (((512,), (256, 256)), 128, 256, 4, 4, 64),
    (((512,), (512,)), 256, 128, 4, 2, 64),
    (((200, 312), (512,)), 128, 128, 4, 4, 32),
    (((512,), (64, 448)), 256, 256, 2, 1, 128),
)


def _inputs(docs, dtype, heads=4, dr=64):
  keys = jax.random.split(jax.random.PRNGKey(3), 4)
  q = jax.random.normal(keys[0], (_B, _S, heads * (128 + dr)), dtype)
  kv = jax.random.normal(keys[1], (_B, _S, heads * 256), dtype)
  k_pe = jax.random.normal(keys[2], (_B, _S, dr), dtype)
  do = jax.random.normal(keys[3], (_B, _S, heads * 128), dtype)
  pos = jnp.stack([jnp.concatenate([jnp.arange(n) for n in d]) for d in docs])
  inv_freq = 10000.0 ** (-jnp.arange(dr // 2) / (dr // 2))
  theta = pos[..., None].astype(jnp.float32) * inv_freq
  cos = jnp.repeat(jnp.cos(theta), 2, -1)
  sin = jnp.stack([-jnp.sin(theta), jnp.sin(theta)], -1).reshape(_B, _S, dr)
  seg = jnp.stack([
      jnp.repeat(jnp.arange(len(d)), jnp.array(d), total_repeat_length=_S)
      for d in docs
  ]).astype(jnp.int32)
  return q, kv, k_pe, cos, sin, seg, do


def _rel_err(x, ref):
  x = x.astype(jnp.float32)
  return float(jnp.linalg.norm(x - ref) / jnp.linalg.norm(ref))


class MlaSplashTest(parameterized.TestCase):

  @parameterized.product(
      case=_CASES,
      dtype_tol=((jnp.float32, 1e-4), (jnp.bfloat16, 3e-2)),
      split=(False, True),
  )
  def test_matches_reference(self, case, dtype_tol, split):
    docs, block_q, block_kv, heads, heads_per_step, dr = case
    dtype, tol = dtype_tol
    q, kv, k_pe, cos, sin, seg, do = _inputs(docs, dtype, heads, dr)
    scale = 1.0 / math.sqrt(128 + dr)
    tiles = dict(block_q=block_q, block_kv=block_kv,
                 heads_per_step=heads_per_step, interpret=True)

    o, lse = mla_splash.mla_splash_fwd(
        q, kv, k_pe, cos, sin, seg, scale=scale, **tiles)
    grads = mla_splash.mla_splash_bwd(
        q, kv, k_pe, cos, sin, seg, o, lse, do, scale=scale, split=split,
        **tiles)

    o_ref, lse_ref = mla_splash.mla_reference(
        q, kv, k_pe, cos, sin, seg, scale=scale)

    def loss(q_, kv_, k_pe_):
      o_, _ = mla_splash.mla_reference(
          q_, kv_, k_pe_, cos, sin, seg, scale=scale)
      return jnp.sum(o_ * do.astype(jnp.float32))

    grads_ref = jax.grad(loss, argnums=(0, 1, 2))(
        q.astype(jnp.float32), kv.astype(jnp.float32),
        k_pe.astype(jnp.float32))

    self.assertLess(_rel_err(o, o_ref), tol)
    self.assertLess(_rel_err(lse, lse_ref), tol)
    for name, g, g_ref in zip(("dq", "dkv", "dk_pe"), grads, grads_ref):
      self.assertLess(_rel_err(g, g_ref), tol, msg=name)

  @parameterized.parameters(
      dict(kv_major=False, block_q=128, block_kv=128),
      dict(kv_major=True, block_q=128, block_kv=128),
      dict(kv_major=False, block_q=256, block_kv=128),
      dict(kv_major=True, block_q=128, block_kv=256),
  )
  def test_schedule_lists_live_tiles_in_order(self, kv_major, block_q,
                                              block_kv):
    docs = ((100, 300, 112), (512,), (64, 64, 384), (200, 312))
    seg = jnp.stack([
        jnp.repeat(jnp.arange(len(d)), jnp.array(d), total_repeat_length=_S)
        for d in docs
    ]).astype(jnp.int32)
    num_active, sched = mla_splash._schedule(
        seg, block_q, block_kv, kv_major=kv_major)
    sched = {k: [int(x) for x in v] for k, v in sched.items()}
    num_active = int(num_active)

    # Brute force: a tile is live iff some causal pair in it shares a
    # document; it needs no mask iff every pair is causal and same-document.
    expected = []
    for b, row in enumerate(seg.tolist()):
      for i in range(_S // block_q):
        for j in range(_S // block_kv):
          pairs = [(row[q] == row[k], q >= k)
                   for q in range(i * block_q, (i + 1) * block_q)
                   for k in range(j * block_kv, (j + 1) * block_kv)]
          if any(same and causal for same, causal in pairs):
            kind = 1 if all(same and causal for same, causal in pairs) else 2
            expected.append((b, i, j, kind))
    expected.sort(key=(lambda x: (x[0], x[2], x[1])) if kv_major else None)
    got = list(zip(*(sched[k][:num_active] for k in ("b", "qi", "kj", "kind"))))
    self.assertEqual(got, expected)
    self.assertTrue(all(k == 0 for k in sched["kind"][num_active:]))

    # Run flags: a run is a q block (q-major) or a kv block (kv-major).
    run = [(b, j if kv_major else i) for b, i, j, _ in expected]
    starts = [int(t == 0 or run[t] != run[t - 1]) for t in range(num_active)]
    ends = [int(t == num_active - 1 or run[t] != run[t + 1])
            for t in range(num_active)]
    self.assertEqual(sched["run_start"][:num_active], starts)
    self.assertEqual(sched["run_end"][:num_active], ends)
    # q_last marks each (example, q block)'s live tile with the largest j.
    last = {}
    for t, (b, i, j, _) in enumerate(expected):
      if (b, i) not in last or j > expected[last[(b, i)]][2]:
        last[(b, i)] = t
    self.assertEqual(
        sched["q_last"][:num_active],
        [int(last[(b, i)] == t) for t, (b, i, _, _) in enumerate(expected)])

  def test_tile_policy(self):
    # 64 MiB VMEM (no TPU attached): fused 4 x 1024/512 up to S 4K, fused
    # 2 x 1024/1024 at 8K, split beyond.
    self.assertEqual(mla_splash.backward_tiles(4096, 128, 64),
                     (1024, 512, 4, False))
    self.assertEqual(mla_splash.backward_tiles(8192, 128, 64),
                     (1024, 1024, 2, False))
    self.assertEqual(mla_splash.backward_tiles(16384, 128, 64),
                     (1024, 1024, 4, True))
    # S not a multiple of 1024; H not a multiple of 4; Dr = 128 allows 1 head.
    self.assertEqual(mla_splash.forward_tiles(1536, 128, 64), (512, 512, 4))
    self.assertEqual(mla_splash.forward_tiles(4096, 6, 64), (1024, 1024, 2))
    self.assertEqual(mla_splash.forward_tiles(4096, 3, 128), (1024, 1024, 1))

  def test_rejects_unsupported_shapes(self):
    q, kv, k_pe, cos, sin, seg, _ = _inputs(((512,), (512,)), jnp.float32)
    with self.assertRaisesRegex(ValueError, "heads_per_step"):
      mla_splash.mla_splash_fwd(q, kv, k_pe, cos, sin, seg, scale=1.0,
                                block_q=128, block_kv=128, heads_per_step=3,
                                interpret=True)
    with self.assertRaisesRegex(ValueError, "multiples of 128"):
      mla_splash.mla_splash_fwd(q, kv, k_pe, cos, sin, seg, scale=1.0,
                                block_q=96, block_kv=128, heads_per_step=2,
                                interpret=True)
    with self.assertRaisesRegex(ValueError, "must divide by"):
      mla_splash.mla_splash_fwd(q, kv, k_pe, cos, sin, seg, scale=1.0,
                                block_q=384, block_kv=128, heads_per_step=2,
                                interpret=True)


if __name__ == "__main__":
  absltest.main()
