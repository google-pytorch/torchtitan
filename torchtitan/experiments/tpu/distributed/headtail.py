"""Head-tail sequence order and Splash helpers for load-balanced CP.

With causal attention, splitting a sequence into ``cp`` contiguous shards gives
the last rank the most work. Head-tail load balancing (MaxText's
``reorder_sequence``) cuts the sequence into ``2 * cp`` chunks and gives rank
``r`` chunks ``r`` and ``2 * cp - 1 - r``, so every rank does the same amount of
causal attention work. This module is the single definition of that order and
the shared CP mask / segment-id / MaskInfo-slicing helpers used by both the
stock JAX Splash kernel and the Tokamax Splash kernel.
"""

from typing import Any
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


def reorder_sequence_jax(x: Any, cp_size: int) -> Any:
  """Reorder a 1D JAX array of length S into head-tail load-balanced order."""
  s = x.shape[0]
  chunks = x.reshape(2 * cp_size, s // (2 * cp_size))
  order = np.array(headtail_chunk_order(cp_size), dtype=np.int32)
  return chunks[order, :].reshape(s)


def make_headtail_splash_mask(
    splash_attention_mask: Any,
    mask_shape: tuple[int, int],
    is_causal: bool,
    local_window_size: int | None,
    cp_size: int,
) -> Any:
  """Single-head Splash mask; with CP, its Q rows are in head-tail order.

  KV all-gather CP shards Q in head-tail order, so the stock causal / local
  mask gets its ``q_sequence`` permuted to match. The stock masks compare and
  hash ``q_sequence``, so kernel caching still works.
  """
  if local_window_size is not None:
    # Sliding-window causal: each q attends to the previous local_window_size
    # keys plus itself (window_size=(left, right), right=0 keeps it causal).
    mask = splash_attention_mask.LocalMask(
        shape=mask_shape, window_size=(local_window_size, 0), offset=0
    )
  elif is_causal:
    mask = splash_attention_mask.CausalMask(shape=mask_shape)
  else:
    # Every key is visible, so the Q order does not matter.
    return splash_attention_mask.FullMask(mask_shape)
  if cp_size > 1:
    # Assumes Splash reads mask.q_sequence lazily (__getitem__, __eq__/__hash__,
    # process_mask); guarded by HeadTailMaskTest and test_cp_shard_parity.
    mask.q_sequence = headtail_positions(mask_shape[0], cp_size)
  return mask


def make_cp_segment_ids(
    segment_ids_cls: Any,
    segment_ids_b: Any,
    cp_size: int,
    q_seq_len: int,
    cp_rank: Any = None,
) -> Any:
  """Build Splash ``SegmentIds(q, kv)`` for one batch element under head-tail CP."""
  if segment_ids_b is None:
    return None
  if cp_size == 1:
    return segment_ids_cls(q=segment_ids_b, kv=segment_ids_b)
  import jax  # pylint: disable=g-import-not-at-top

  q_seg = reorder_sequence_jax(segment_ids_b, cp_size)
  q_seg = jax.lax.dynamic_slice_in_dim(
      q_seg, cp_rank[0] * q_seq_len, q_seq_len, axis=0
  )
  return segment_ids_cls(q=q_seg, kv=segment_ids_b)


def slice_cp_mask_info(
    mask_info: Any,
    cp_size: int,
    cp_rank: Any,
    field_axes: tuple[tuple[str, int], ...],
) -> Any:
  """Slice a Splash/Tokamax ``MaskInfo`` for ``q_seq_shards=cp_size`` to ``cp_rank``."""
  if cp_size <= 1 or mask_info is None:
    return mask_info
  if cp_rank is None:
    raise ValueError("cp_rank must be provided when cp_size > 1.")
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  r = cp_rank[0]

  def _slice(x, axis: int):
    if x is None:
      return None
    arr = jnp.asarray(x)
    shard_len = arr.shape[axis] // cp_size
    return jax.lax.dynamic_slice_in_dim(
        arr, r * shard_len, shard_len, axis=axis
    )

  replace_fn = getattr(
      mask_info, "_replace", getattr(mask_info, "replace", None)
  )
  return replace_fn(**{
      name: _slice(getattr(mask_info, name), axis) for name, axis in field_axes
  })
