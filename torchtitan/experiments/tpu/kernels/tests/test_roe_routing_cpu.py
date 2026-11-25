# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests of the Ring-of-Experts routing metadata (pure JAX, no Pallas)."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from torchtitan.experiments.tpu.kernels import ragged_gather_tc as rg

_FIELDS = rg._RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS


def _reduce_with_tables(x, src_rows, tokens, blocks, num_out_tokens, d):
  """Emulates ragged_gather_reduce_tc's block table in numpy (FP32 sums)."""
  out = np.zeros((num_out_tokens, d), np.float32)
  num_blocks = int(blocks[0])
  covered = np.zeros((num_out_tokens,), np.int32)
  for b in range(num_blocks):
    slot_start, num_slots, token_start, num_tokens = (
        int(v) for v in blocks[1 + _FIELDS * b : 1 + _FIELDS * (b + 1)]
    )
    covered[token_start : token_start + num_tokens] += 1
    for i in range(slot_start, slot_start + num_slots):
      t = int(tokens[i])
      assert token_start <= t < token_start + num_tokens
      out[t] += x[int(src_rows[i])]
  assert (covered == 1).all(), "every output token is written exactly once"
  return out


def _make_ids(rng, g_tokens, num_experts, top_k):
  return np.stack(
      [rng.choice(num_experts, top_k, replace=False) for _ in range(g_tokens)]
  ).astype(np.int32)


class RoeWindowRoutingTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(cap_factor=1.5, rank=0),
      dict(cap_factor=1.5, rank=3),
      dict(cap_factor=0.6, rank=1),  # capped: count < routed slots
  )
  def test_window_matches_full(self, cap_factor, rank):
    num_experts, ep_size, top_k, g_tokens = 32, 4, 4, 256
    block_size, d = 48, 8
    rng = np.random.default_rng(rank)
    ids = jnp.asarray(_make_ids(rng, g_tokens, num_experts, top_k).reshape(-1))
    ep_rank = jnp.asarray([rank], jnp.int32)
    expected_local = g_tokens * top_k // ep_size
    cap = int(np.ceil(cap_factor * expected_local / 8) * 8)
    kw = dict(
        num_experts=num_experts,
        ep_size=ep_size,
        top_k=top_k,
        num_out_tokens=g_tokens,
        cap=cap,
        reduce_block_size=block_size,
    )

    # Full-M routing, written out (the jax_op body with window_routing=False).
    sorted_e, slot_order = rg.packed_stable_sort(ids, num_experts - 1)
    offs = rg._sorted_expert_offsets(sorted_e, num_experts)
    e_loc = num_experts // ep_size
    loc = offs[rank * e_loc : (rank + 1) * e_loc + 1]
    start = loc[0]
    clamped = jnp.minimum(loc, start + cap)
    count = int(clamped[-1] - start)
    tok_full = slot_order // top_k
    meta_full = rg.ragged_gather_reduce_tc_metadata(
        tok_full,
        start,
        count,
        top_k=top_k,
        block_size=block_size,
        num_out_tokens=g_tokens,
    )

    sizes, tok_w, slot_w, sc2, src_w, toks_w, blocks_w = (
        rg._roe_window_routing_metadata(ids, ep_rank, **kw)
    )
    self.assertEqual(int(sc2[0]), 0)
    self.assertEqual(int(sc2[1]), count)
    np.testing.assert_array_equal(
        np.asarray(sizes), np.diff(np.asarray(clamped))
    )
    self.assertEqual(tok_w.shape, (cap,))
    s = int(start)
    np.testing.assert_array_equal(
        np.asarray(tok_w[:count]), np.asarray(tok_full[s : s + count])
    )
    np.testing.assert_array_equal(
        np.asarray(slot_w[:count]), np.asarray(slot_order[s : s + count])
    )
    # Compacted slots: identical (src row relative to the window, token).
    np.testing.assert_array_equal(
        np.asarray(src_w[:count]), np.asarray(meta_full.src_rows[:count])
    )
    np.testing.assert_array_equal(
        np.asarray(toks_w[:count]), np.asarray(meta_full.tokens[:count])
    )
    nb = int(blocks_w[0])
    self.assertEqual(nb, int(meta_full.blocks[0]))
    np.testing.assert_array_equal(
        np.asarray(blocks_w[: 1 + _FIELDS * nb]),
        np.asarray(meta_full.blocks[: 1 + _FIELDS * nb]),
    )

    # End to end: the reduce tables give the reference combine.
    x = rng.standard_normal((cap, d)).astype(np.float32)
    out = _reduce_with_tables(
        x,
        np.asarray(src_w),
        np.asarray(toks_w),
        np.asarray(blocks_w),
        g_tokens,
        d,
    )
    ref = np.zeros((g_tokens, d), np.float32)
    for i in range(count):
      ref[int(tok_w[i])] += x[i]
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)

  @parameterized.parameters(
      dict(cap_factor=1.5, rank=0, weighted=False),
      dict(cap_factor=1.5, rank=2, weighted=True),
      dict(cap_factor=0.6, rank=1, weighted=False),
  )
  def test_run_reduce_kernel_matches(self, cap_factor, rank, weighted):
    num_experts, ep_size, top_k, g_tokens = 32, 4, 4, 256
    block_size, d = 48, 256
    rng = np.random.default_rng(10 + rank)
    ids = jnp.asarray(_make_ids(rng, g_tokens, num_experts, top_k).reshape(-1))
    ep_rank = jnp.asarray([rank], jnp.int32)
    cap = int(np.ceil(cap_factor * g_tokens * top_k / ep_size / 8) * 8)
    kw = dict(
        num_experts=num_experts,
        ep_size=ep_size,
        top_k=top_k,
        num_out_tokens=g_tokens,
        cap=cap,
        reduce_block_size=block_size,
    )
    base = rg._roe_window_routing_metadata(ids, ep_rank, **kw)
    runs = rg._roe_window_routing_metadata(ids, ep_rank, run_reduce=True, **kw)
    for a, b in zip(base[:6], runs[:6]):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    count = int(base[3][1])
    src_rows, tokens = base[4], base[5]
    x = jnp.asarray(
        rng.standard_normal((cap, d // 128, 128)).astype(np.float32)
    ).astype(jnp.bfloat16)
    weights = None
    if weighted:
      weights = jnp.asarray(rng.random(src_rows.shape[0]).astype(np.float32))

    def _run(blocks):
      meta = rg.RaggedGatherReduceMetadata(
          src_rows=src_rows,
          tokens=tokens,
          blocks=blocks,
          block_size=block_size,
          num_indices=cap,
      )
      return rg.ragged_gather_reduce_tc(
          x, meta, num_out_tokens=g_tokens, top_k=top_k, weights=weights
      )

    rg._INTERPRET = jax.default_backend() == "cpu"
    try:
      out_base = np.asarray(_run(base[6]).astype(jnp.float32))
      out_runs = np.asarray(_run(runs[6]).astype(jnp.float32))
    finally:
      rg._INTERPRET = False
    np.testing.assert_array_equal(out_runs, out_base)
    # Reference.
    xf = np.asarray(x.astype(jnp.float32)).reshape(cap, d)
    ref = np.zeros((g_tokens, d), np.float32)
    toks = np.asarray(base[1])
    for i in range(count):
      ref[int(toks[i])] += xf[i]
    if not weighted:
      np.testing.assert_allclose(
          out_base.reshape(g_tokens, d), ref, rtol=2e-2, atol=2e-2
      )


if __name__ == "__main__":
  absltest.main()
