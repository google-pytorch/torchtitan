"""Head-tail sequence order for load-balanced context parallelism (CP).

With causal attention, splitting a sequence into ``cp`` contiguous shards gives
the last rank the most work. Head-tail load balancing (MaxText's
``reorder_sequence``) cuts the sequence into ``2 * cp`` chunks and gives rank
``r`` chunks ``r`` and ``2 * cp - 1 - r``, so every rank does the same amount of
causal attention work. This module is the single definition of that order: the
CP inputs (``cp_attention.reorder_sequence``), the Splash masks and the
reference attention all derive from it.
"""

import numpy as np


def headtail_chunk_order(cp_size: int) -> list[int]:
  """Order of the ``2 * cp_size`` chunks: ``[0, 2cp-1, 1, 2cp-2, ...]``."""
  return [c for r in range(cp_size) for c in (r, 2 * cp_size - 1 - r)]


def headtail_positions(seq_len: int, cp_size: int) -> np.ndarray:
  """Positions ``0..seq_len-1`` rearranged into head-tail order.

  Entry ``i`` is the original position of the token at index ``i`` of the
  head-tail sequence; rank ``r`` holds indices
  ``[r * seq_len // cp_size, (r + 1) * seq_len // cp_size)``.
  """
  if seq_len % (2 * cp_size):
    raise ValueError(
        f"Sequence length ({seq_len}) must be divisible by 2 * cp_size"
        f" ({2 * cp_size})."
    )
  chunks = np.arange(seq_len, dtype=np.int32).reshape(2 * cp_size, -1)
  return chunks[headtail_chunk_order(cp_size)].reshape(-1)
