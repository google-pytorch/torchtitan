"""Balance packed rows' attention cost across data-parallel ranks.

With per-example Splash tile skipping (Tokamax 0.0.16+, called once per
example), a rank's attention time depends on how its rows happen to be packed: a row
holding one 4096-token document costs several times a row of short documents.
FSDP's per-layer collectives then wait for the slowest rank, every layer.

Each step, `AttentionBalancedIterator` re-deals the step's rows across DP ranks
so per-rank attention cost is even. All ranks all-gather per-row costs, compute
the same deterministic longest-processing-time (LPT) assignment with exactly
`rows_per_rank` rows per rank, and exchange whole rows over a CPU (gloo) group.
The global batch is unchanged and cross-entropy divides by the global
valid-token count, so its gradient is unchanged up to summation order. Terms
computed per rank are not invariant: a microbatch-wise MoE load-balance loss,
per-rank token dropping (capacity factor) and per-rank metrics see different
rows. A batch-wise aux loss (as in the Q235 recipes) is invariant.

Row cost is the sum of squared document lengths, derived from `positions` the
same way the model derives segment ids: a document starts at the row start or
wherever `positions == 0`.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from torchtitan.components.loss import IGNORE_INDEX

__all__ = ["AttentionBalancedIterator", "balance_rows", "lpt_assign", "row_costs"]

_ROW_KEYS = ("input", "labels", "positions", "padding_mask")


def row_costs(positions: np.ndarray, seq_len: int) -> np.ndarray:
  """Sum of squared document lengths for each row of a flat positions stream."""
  rows = positions.reshape(-1, seq_len)
  costs = np.empty(rows.shape[0])
  for i, row in enumerate(rows):
    starts = np.flatnonzero(row == 0)
    if starts.size == 0 or starts[0] != 0:
      starts = np.concatenate([[0], starts])
    lengths = np.diff(np.append(starts, seq_len)).astype(np.float64)
    costs[i] = float(np.dot(lengths, lengths))
  return costs


def lpt_assign(costs: np.ndarray, num_ranks: int, rows_per_rank: int) -> np.ndarray:
  """Heaviest row first, each to the least-loaded rank that still has room.

  `costs` holds `num_ranks * rows_per_rank` rows in global (rank-major) order.
  Ties break by row id and then rank id, so every rank computes the same
  assignment. Returns the destination rank of each row.
  """
  n = costs.shape[0]
  if n != num_ranks * rows_per_rank:
    raise ValueError(f"{n} rows cannot fill {num_ranks} x {rows_per_rank}")
  heap = [(0.0, rank) for rank in range(num_ranks)]
  filled = [0] * num_ranks
  dest = np.empty(n, dtype=np.int64)
  for row in np.lexsort((np.arange(n), -costs)):
    load, rank = heapq.heappop(heap)
    dest[row] = rank
    filled[rank] += 1
    if filled[rank] < rows_per_rank:
      heapq.heappush(heap, (load + float(costs[row]), rank))
  return dest


class AttentionBalancedIterator:
  """Wraps a per-rank iterator of flat `TrainerBatch` dicts.

  Each step consumes exactly one parent batch per rank, so `get_state` /
  `set_state` delegate to the parent and resuming replays the same steps.
  """

  def __init__(
      self,
      parent: Iterator[dict[str, Any]],
      *,
      seq_len: int,
      rows_per_rank: int,
      group: Any,
  ) -> None:
    self._parent = parent
    self._seq_len = seq_len
    self._rows = rows_per_rank
    self._group = group
    # Rows are interchangeable, so ranks are indexed by their position in
    # `group`, which need not match the DP rank the dataloader shards by.
    self._rank = dist.get_rank(group)
    self._world = dist.get_world_size(group)

  def __iter__(self) -> AttentionBalancedIterator:
    return self

  def get_state(self) -> Any:
    return self._parent.get_state()  # pytype: disable=attribute-error

  def set_state(self, state: Any) -> None:
    self._parent.set_state(state)  # pytype: disable=attribute-error

  def close(self) -> None:
    close = getattr(self._parent, "close", None)
    if close is not None:
      close()
    dist.destroy_process_group(self._group)

  def __next__(self) -> dict[str, Any]:
    batch = next(self._parent)
    extra = set(batch) - set(_ROW_KEYS) - {"num_valid_tokens"}
    if extra:
      raise ValueError(f"attention balancing does not know batch keys {sorted(extra)}")
    tensors = {k: torch.as_tensor(batch[k]) for k in _ROW_KEYS}
    rows = torch.stack(
        [tensors[k].reshape(-1, self._seq_len).to(torch.int64) for k in _ROW_KEYS],
        dim=1,
    )  # [rows_per_rank, len(_ROW_KEYS), seq_len]
    assert rows.shape[0] == self._rows, (rows.shape, self._rows)

    local = torch.from_numpy(row_costs(tensors["positions"].numpy(), self._seq_len))
    gathered = [torch.empty_like(local) for _ in range(self._world)]
    dist.all_gather(gathered, local, group=self._group)
    dest = lpt_assign(torch.cat(gathered).numpy(), self._world, self._rows)

    # Send rows in local order grouped by destination; receive them grouped by
    # source rank in the senders' local order. Every rank thus sees the same
    # deterministic row order for a given step.
    mine = dest[self._rank * self._rows:(self._rank + 1) * self._rows]
    send = rows[torch.from_numpy(np.argsort(mine, kind="stable"))]
    send_counts = np.bincount(mine, minlength=self._world)
    recv_counts = (dest.reshape(self._world, self._rows) == self._rank).sum(1)
    row_elems = len(_ROW_KEYS) * self._seq_len
    recv = torch.empty((self._rows, len(_ROW_KEYS), self._seq_len), dtype=torch.int64)
    dist.all_to_all_single(
        recv.view(-1),
        send.reshape(-1),
        output_split_sizes=(recv_counts * row_elems).tolist(),
        input_split_sizes=(send_counts * row_elems).tolist(),
        group=self._group,
    )

    out: dict[str, Any] = {
        k: recv[:, i].reshape(-1).to(tensors[k].dtype) for i, k in enumerate(_ROW_KEYS)
    }
    out["num_valid_tokens"] = int((out["labels"] != IGNORE_INDEX).sum())
    return out


def balance_rows(
    parent: Iterator[dict[str, Any]],
    *,
    seq_len: int,
    num_tokens_per_rank: int,
    dp_world_size: int,
) -> Iterator[dict[str, Any]]:
  """Wraps `parent` with cross-rank balancing when there is more than one rank."""
  if dp_world_size == 1:
    return parent
  if num_tokens_per_rank % seq_len:
    raise ValueError("attention_balance needs whole rows in each local batch")
  if dp_world_size != dist.get_world_size():
    # The row exchange spans the whole world; with other parallel dims the DP
    # group would need its own gloo subgroup.
    raise ValueError("attention_balance assumes the DP group is the whole world")
  return AttentionBalancedIterator(
      parent,
      seq_len=seq_len,
      rows_per_rank=num_tokens_per_rank // seq_len,
      group=dist.new_group(backend="gloo"),
  )
