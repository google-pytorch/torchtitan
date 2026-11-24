"""CPU emulation of a context-parallel group for TPU CP tests.

Every rank of a CP group runs on its own thread (``Ranks``), and
``emulated_cp_group`` routes the CP attentions' and GatedDeltaNet's collectives
through those threads: the functional collectives of the traced
(``torch.compile``) path and ``spmd.redistribute`` of the eager path. Shared by
``tests/test_cp_attention.py`` and ``qwen3_5/tests/test_cp_gated_delta.py``.
"""

import concurrent.futures
import contextlib
import threading
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
# pyformat: disable
# Import torch before spmd_types to avoid circular import with torch_tpu.
import torch
import spmd_types as spmd
# pyformat: enable
import torch.distributed._functional_collectives as funcol
from torchtitan.experiments.tpu.distributed import cp_attention
from torchtitan.experiments.tpu.distributed.cp_attention import reorder_sequence


class Ranks:
  """Runs one function per CP rank on its own thread; emulates collectives."""

  def __init__(self, world_size: int):
    self.world_size = world_size
    self._barrier = threading.Barrier(world_size, timeout=120)
    self._slots = [None] * world_size
    self._local = threading.local()

  @property
  def rank(self) -> int:
    return self._local.rank

  def group(self):
    return SimpleNamespace(size=lambda: self.world_size, rank=lambda: self.rank)

  def exchange(self, value):
    """Returns every rank's ``value`` in rank order (an all-gather)."""
    self._slots[self.rank] = value
    self._barrier.wait()
    gathered = list(self._slots)
    self._barrier.wait()
    return gathered

  def run(self, fn):
    def run_rank(rank):
      self._local.rank = rank
      try:
        return fn(rank)
      except BaseException:
        self._barrier.abort()  # Unblock the other ranks.
        raise

    with concurrent.futures.ThreadPoolExecutor(self.world_size) as pool:
      return list(pool.map(run_rank, range(self.world_size)))


def _all_to_all(ranks: Ranks, x: torch.Tensor) -> torch.Tensor:
  """Equal-split all-to-all along dim 0: chunk i goes to rank i."""
  chunks = ranks.exchange(x.chunk(ranks.world_size))
  return torch.cat([c[ranks.rank] for c in chunks])


class _AllToAll(torch.autograd.Function):
  """``funcol.all_to_all_single`` (differentiable, as on torch >= 2.11)."""

  @staticmethod
  def forward(ctx, x, ranks):
    ctx.ranks = ranks
    return _all_to_all(ranks, x)

  @staticmethod
  def backward(ctx, grad):
    return _all_to_all(ctx.ranks, grad.contiguous()), None


class _Redistribute(torch.autograd.Function):
  """``spmd.redistribute`` from S(src_dim) to S(dst_dim), or to R if None."""

  @staticmethod
  def forward(ctx, x, ranks, src_dim, dst_dim):
    ctx.ranks, ctx.src_dim, ctx.dst_dim = ranks, src_dim, dst_dim
    full = torch.cat(ranks.exchange(x), dim=src_dim)
    if dst_dim is None:
      return full
    return full.chunk(ranks.world_size, dim=dst_dim)[ranks.rank].contiguous()

  @staticmethod
  def backward(ctx, grad):
    ranks = ctx.ranks
    if ctx.dst_dim is None:  # All-gather: the gradient is a reduce-scatter.
      full = torch.stack(ranks.exchange(grad)).sum(0)
    else:
      full = torch.cat(ranks.exchange(grad), dim=ctx.dst_dim)
    shard = full.chunk(ranks.world_size, dim=ctx.src_dim)[ranks.rank]
    return shard.contiguous(), None, None, None


@contextlib.contextmanager
def emulated_cp_group(ranks: Ranks, traced: bool):
  """Routes the CP attentions' collectives through ``ranks``."""

  def all_to_all_single(x, output_split_sizes, input_split_sizes, group):
    del group
    assert output_split_sizes is None and input_split_sizes is None
    return _AllToAll.apply(x, ranks)

  def all_gather_tensor(x, gather_dim, group):
    del group
    return torch.cat(ranks.exchange(x), dim=gather_dim)

  def reduce_scatter_tensor(x, reduce_op, scatter_dim, group):
    del group
    assert reduce_op == "sum"
    full = torch.stack(ranks.exchange(x)).sum(0)
    return full.chunk(ranks.world_size, dim=scatter_dim)[
        ranks.rank
    ].contiguous()

  def redistribute(x, group, *, src, dst, backward_options=None):
    del group, backward_options
    dst_dim = dst.dim if isinstance(dst, spmd.Shard) else None
    return _Redistribute.apply(x, ranks, src.dim, dst_dim)

  with contextlib.ExitStack() as stack:
    stack.enter_context(
        mock.patch.object(
            cp_attention, "spmd_mesh_group", lambda axis: ranks.group()
        )
    )
    # Functional collectives also run eagerly (e.g. cp_reduce_scatter).
    for name, fn in (
        ("all_to_all_single", all_to_all_single),
        ("all_gather_tensor", all_gather_tensor),
        ("reduce_scatter_tensor", reduce_scatter_tensor),
    ):
      stack.enter_context(mock.patch.object(funcol, name, fn))
    if traced:
      # The branch the CP attentions take under torch.compile.
      stack.enter_context(
          mock.patch.object(cp_attention, "_tracing", lambda: True)
      )
    else:
      stack.enter_context(mock.patch.object(spmd, "redistribute", redistribute))
    yield


def shard(x: torch.Tensor, strategy: str, cp: int, rank: int, seq_len: int):
  """Rank ``rank``'s shard of a folded ``[B * S, ...]`` tensor, as cp_shard lays it out."""
  if strategy == "ulysses":
    return x.chunk(cp)[rank]
  rest = list(x.shape[1:])
  bsz = x.shape[0] // seq_len
  x = reorder_sequence(x.reshape([bsz, seq_len] + rest), cp, seq_dim=1)
  x = x.reshape([bsz, cp, seq_len // cp] + rest).movedim(1, 0)
  return x[rank].reshape([bsz * seq_len // cp] + rest)


def unshard(shards, strategy: str, cp: int, seq_len: int) -> torch.Tensor:
  """Inverse of ``shard`` over all ranks."""
  x = torch.cat(shards)
  if strategy == "ulysses":
    return x
  rest = list(x.shape[1:])
  bsz = x.shape[0] // seq_len
  x = x.reshape([cp, bsz, seq_len // cp] + rest).movedim(0, 1)
  x = x.reshape([bsz, seq_len] + rest)
  x = reorder_sequence(x, cp, seq_dim=1, to_contiguous=True)
  return x.reshape([bsz * seq_len] + rest)


def packed_positions(doc_lengths_per_row) -> torch.Tensor:
  """Folded ``[B * S]`` positions that restart at 0 for every packed document."""
  return torch.cat(
      [torch.arange(n) for row in doc_lengths_per_row for n in row]
  )


def reset_cp_state(test: absltest.TestCase) -> None:
  """Clears the CP rank cache and the rank-miss warnings."""
  for clear in (
      cp_attention._CP_RANK_CACHE.clear,
      cp_attention._CP_RANK_MISS_WARNED.clear,
  ):
    clear()
    test.addCleanup(clear)


# Two sequences of 32 tokens, with documents crossing the head-tail chunks.
SEQ_LEN = 32
DOCS = ((5, 11, 9, 7), (16, 3, 13))


__all__ = [
    "DOCS",
    "Ranks",
    "SEQ_LEN",
    "emulated_cp_group",
    "packed_positions",
    "reset_cp_state",
    "shard",
    "unshard",
]
