"""CPU tests for `experiments/tpu/attention_balance.py`."""

import os
import socket
import tempfile

from absl.testing import absltest
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchtitan.experiments.tpu import attention_balance as ab

_SEQ = 256
_ROWS = 4
_STEPS = 3


def _fake_batch(rng, rank, step):
  """One rank's flat batch of random documents, tagged by (rank, step)."""
  lengths = rng.integers(1, 3 * _SEQ, size=_ROWS * _SEQ // 4)
  positions = np.concatenate([np.arange(n) for n in lengths])[: _ROWS * _SEQ]
  inputs = np.arange(_ROWS * _SEQ, dtype=np.int32) + (rank * 100 + step) * 10**4
  labels = inputs.astype(np.int64)
  labels[::7] = -100
  return {
      "input": torch.from_numpy(inputs),
      "labels": torch.from_numpy(labels),
      "positions": torch.from_numpy(positions),
      "padding_mask": torch.zeros(_ROWS * _SEQ, dtype=torch.bool),
      "num_valid_tokens": int((labels != -100).sum()),
  }


class RowCostsTest(absltest.TestCase):

  def test_documents_split_at_row_start_and_zeros(self):
    # Row 0 continues a document from the previous row: lengths 2, 3, 3.
    pos = np.array([3, 4, 0, 1, 2, 0, 1, 2])
    np.testing.assert_allclose(ab.row_costs(pos, 8), [4.0 + 9.0 + 9.0])

  def test_one_cost_per_row(self):
    pos = np.array([0, 1, 2, 3, 0, 1, 0, 1])
    np.testing.assert_allclose(ab.row_costs(pos, 4), [16.0, 8.0])


class LptAssignTest(absltest.TestCase):

  def test_equal_rows_and_deterministic(self):
    costs = np.random.default_rng(0).exponential(size=8 * 16)
    a = ab.lpt_assign(costs, 8, 16)
    np.testing.assert_array_equal(a, ab.lpt_assign(costs.copy(), 8, 16))
    np.testing.assert_array_equal(np.bincount(a, minlength=8), [16] * 8)

  def test_balances_better_than_rank_order(self):
    costs = np.random.default_rng(1).exponential(size=64 * 16)
    before = costs.reshape(64, 16).sum(1)
    after = np.bincount(ab.lpt_assign(costs, 64, 16), weights=costs, minlength=64)
    self.assertLess(after.max() / after.mean(), before.max() / before.mean())
    self.assertLess(after.max() / after.mean(), 1.1)

  def test_reproduces_c4_simulation(self):
    path = os.environ.get("TT_ATTN_BALANCE_SIM_DOC_LENS")
    if not path:
      self.skipTest("set TT_ATTN_BALANCE_SIM_DOC_LENS to a doc_lens.npy of C4 tokens")
    lens = np.random.default_rng(0).permutation(np.load(path))
    seq, world, rows = 4096, 128, 16
    ends = np.cumsum(lens)
    n_rows = int(ends[-1] // seq)
    doc = np.searchsorted(ends, np.arange(n_rows * seq), side="right")
    positions = np.arange(n_rows * seq) - np.concatenate([[0], ends[:-1]])[doc]
    costs = ab.row_costs(positions, seq)
    before, after = [], []
    for s in range(n_rows // (world * rows)):
      c = costs[s * world * rows:(s + 1) * world * rows]
      b = c.reshape(world, rows).sum(1)
      a = np.bincount(ab.lpt_assign(c, world, rows), weights=c, minlength=world)
      before.append(b.max() / b.mean())
      after.append(a.max() / a.mean())
    self.assertGreater(np.mean(before), 1.3)
    self.assertLess(np.mean(after), 1.06)


def _free_port():
  with socket.socket() as s:
    s.bind(("localhost", 0))
    return s.getsockname()[1]


def _exchange_worker(rank, world, port, out_dir):
  dist.init_process_group(
      "gloo", init_method=f"tcp://localhost:{port}", rank=rank, world_size=world
  )
  rng = np.random.default_rng(100 + rank)
  it = ab.balance_rows(
      iter([_fake_batch(rng, rank, s) for s in range(_STEPS)]),
      seq_len=_SEQ,
      num_tokens_per_rank=_ROWS * _SEQ,
      dp_world_size=world,
  )
  torch.save([next(it) for _ in range(_STEPS)], os.path.join(out_dir, f"{rank}.pt"))
  it.close()
  dist.destroy_process_group()


class RowExchangeTest(absltest.TestCase):

  def test_rows_conserved_and_balanced(self):
    world = 4
    with tempfile.TemporaryDirectory() as d:
      mp.spawn(_exchange_worker, args=(world, _free_port(), d), nprocs=world)
      outs = [torch.load(os.path.join(d, f"{r}.pt"), weights_only=True) for r in range(world)]
    rngs = [np.random.default_rng(100 + r) for r in range(world)]
    for step in range(_STEPS):
      ins = [_fake_batch(rngs[r], r, step) for r in range(world)]
      for o, i in zip((outs[r][step] for r in range(world)), ins):
        for k in ("input", "labels", "positions", "padding_mask"):
          self.assertEqual(o[k].dtype, i[k].dtype)
          self.assertEqual(o[k].shape, i[k].shape)
        self.assertEqual(o["num_valid_tokens"], int((o["labels"] != -100).sum()))

      def rows(batches):
        keys = ("input", "labels", "positions", "padding_mask")
        return sorted(
            tuple(torch.cat([b[k].reshape(_ROWS, _SEQ)[j].to(torch.int64) for k in keys]).tolist())
            for b in batches for j in range(_ROWS)
        )

      step_outs = [outs[r][step] for r in range(world)]
      self.assertEqual(rows(ins), rows(step_outs))  # same global batch, rows intact
      before = [ab.row_costs(b["positions"].numpy(), _SEQ).sum() for b in ins]
      after = [ab.row_costs(b["positions"].numpy(), _SEQ).sum() for b in step_outs]
      self.assertLessEqual(max(after) / np.mean(after), max(before) / np.mean(before))


if __name__ == "__main__":
  absltest.main()
