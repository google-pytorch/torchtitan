"""Splash Attention integration for torch_tpu via Pallas.

Wraps JAX's splash_attention_kernel for use with PyTorch tensors on TPU.
Uses torch_tpu._internal.pallas.custom_jax_kernel() to bridge JAX -> PyTorch.
"""

import functools
import hashlib
import math
import os
import typing
from typing import Any, Optional

# JAX imports
import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import QKVLayout
import jax.numpy as jnp
import numpy as np
import torch
from torch import nn
from torch.distributed.tensor import DTensor
# torch_tpu Pallas bridge
from torch_tpu._internal import pallas
from torchtitan.experiments.tpu.headtail import (
    make_cp_segment_ids,
    make_headtail_splash_mask,
    slice_cp_mask_info,
)
from torchtitan.models.common.attention import segment_ids_from_positions
from torchtitan.tools.logging import logger

custom_jax_kernel = pallas.custom_jax_kernel

_DEFAULT_MASK_VALUE = splash_attention_kernel.DEFAULT_MASK_VALUE
_warned_cp_block_skipping = False


def _document_aware_mask_info(mask_info, segment_ids_b, block_q, block_kv, is_dkv):
  """Extend block sparsity to skip tiles that are entirely cross-document.

  `segment_ids` alone masks scores *inside* tiles the kernel already chose to
  visit; that choice comes from `block_mask`, built from the static `CausalMask`
  and blind to documents -- so packed sequences still multiply fully-masked
  tiles and discard them.

  `segment_ids` is non-decreasing, so a block's documents are the interval
  [min, max] over it and a tile has a same-document pair iff two intervals
  overlap: exact, and O(num_blocks). Only `block_mask` and `data_next` change,
  the latter mirroring JAX's `_process_dynamic_mask`.
  """
  base_block_mask = mask_info.block_mask
  base_data_next = mask_info.data_next
  if base_block_mask is None or base_data_next is None:
    return mask_info

  seq_len = segment_ids_b.shape[-1]
  num_q_blocks, q_rem = divmod(seq_len, block_q)
  num_kv_blocks, kv_rem = divmod(seq_len, block_kv)
  if q_rem or kv_rem:
    return mask_info
  if base_block_mask.shape[-2:] != (num_q_blocks, num_kv_blocks):
    # Sharded or otherwise unexpected layout: leave the static metadata alone.
    return mask_info

  q_seg = segment_ids_b.reshape(num_q_blocks, block_q)
  kv_seg = segment_ids_b.reshape(num_kv_blocks, block_kv)
  q_lo, q_hi = jnp.min(q_seg, axis=-1), jnp.max(q_seg, axis=-1)
  kv_lo, kv_hi = jnp.min(kv_seg, axis=-1), jnp.max(kv_seg, axis=-1)
  overlaps = (q_lo[:, None] <= kv_hi[None, :]) & (kv_lo[None, :] <= q_hi[:, None])

  block_mask = jnp.where(overlaps[None], base_block_mask, 0).astype(
      base_block_mask.dtype
  )
  if is_dkv:
    data_next = jnp.arange(num_q_blocks, dtype=jnp.int32)[None, :, None]
  else:
    data_next = jnp.arange(num_kv_blocks, dtype=jnp.int32)[None, None, :]
  data_next = jnp.where(
      block_mask == 0, 0, jnp.broadcast_to(data_next, block_mask.shape)
  ).astype(base_data_next.dtype)
  return mask_info._replace(block_mask=block_mask, data_next=data_next)


def _maybe_document_aware_mask_info(
    mask_info,
    segment_ids_b,
    block_q,
    block_kv,
    is_dkv,
    use_document_aware_block_skipping,
):
  """Choose document-aware block skipping or batch-native masking.

  Document-aware block skipping: specialize block metadata for each packed
  example and omit blocks that cannot contain same-document token pairs.
  Batch-native masking: reuse the static causal block metadata for every
  example and rely on runtime segment IDs for document boundaries. Both modes
  pass ``segment_ids_b`` to Splash and enforce the same token-level causal and
  packed-document mask.

  Document-aware block skipping can save work for packs with many short
  documents, but its example-dependent metadata makes JAX/XLA lower the
  surrounding ``vmap`` as a serial batch loop on TPU today. Batch-native
  masking avoids that loop and is faster for the measured Qwen3-30B C4
  training distribution, but its extra block work may lose on sparser packs.
  Keep document-aware block skipping as the kernel API default; recipes should
  select batch-native masking only for a validated distribution and shape.
  """
  if not use_document_aware_block_skipping:
    return mask_info
  return _document_aware_mask_info(
      mask_info, segment_ids_b, block_q, block_kv, is_dkv
  )


_MASK_INFO_SLICE_AXES = (
    ("data_next", 1),
    ("mask_next", 1),
    ("block_mask", 1),
    ("q_sequence", 0),
)

_make_mask = functools.partial(make_headtail_splash_mask, splash_attention_mask)


def _slice_mask_info(mask_info, cp_size: int, cp_rank: jax.Array | None):
  """Slice a stock JAX MaskInfo crafted for q_seq_shards=cp_size to cp_rank's shard."""
  return slice_cp_mask_info(mask_info, cp_size, cp_rank, _MASK_INFO_SLICE_AXES)


def _make_splash_attention_fn(
    seq_len: int,
    n_heads: int,
    n_kv_heads: int,
    is_causal: bool = True,
    local_window_size: int | None = None,
    block_q: int = 512,
    block_kv: int = 512,
    block_kv_compute: int = 512,
    block_q_dkv: int = 512,
    block_kv_dkv: int = 512,
    block_kv_dkv_compute: int = 512,
    block_q_dq: int | None = None,
    block_kv_dq: int | None = None,
    use_fused_bwd_kernel: bool = True,
    q_layout: str = "HEAD_DIM_MINOR",
    k_layout: str = "HEAD_DIM_MINOR",
    v_layout: str = "HEAD_DIM_MINOR",
    use_vmap_bwd: bool = False,
    use_document_aware_block_skipping: bool = True,
    kv_seq_len: int | None = None,
):
  """Create JAX splash attention forward and backward functions.

  Returns:
    (splash_fn, splash_bwd_fn): forward function taking (q, k, v) returning
      (out, logsumexp), and backward function taking (q, k, v, out, logsumexp,
      grad_out) returning (grad_q, grad_k, grad_v) WITHOUT re-running the
      forward (uses saved residuals to call the backward kernel directly).
  """
  q_seq_len = seq_len
  kv_seq_len = kv_seq_len if kv_seq_len is not None else q_seq_len
  if kv_seq_len % q_seq_len != 0:
    raise ValueError(
        f"kv_seq_len ({kv_seq_len}) must be a multiple of q_seq_len"
        f" ({q_seq_len})."
    )
  cp_size = kv_seq_len // q_seq_len
  global _warned_cp_block_skipping
  if (
      cp_size > 1
      and use_document_aware_block_skipping
      and not _warned_cp_block_skipping
  ):
    _warned_cp_block_skipping = True
    logger.warning(
        "Splash attention: use_document_aware_block_skipping=True is not"
        " supported with head-tail CP (cp_size=%d); falling back to"
        " batch-native masking (packed documents are still masked exactly via"
        " segment_ids). Set sa_use_document_aware_block_skipping=False to"
        " acknowledge.",
        cp_size,
    )

  def _get_layout(layout_str):
    if layout_str == "HEAD_DIM_MINOR":
      return QKVLayout.HEAD_DIM_MINOR
    elif layout_str == "SEQ_MINOR":
      return QKVLayout.SEQ_MINOR
    else:
      raise ValueError(f"Unknown layout: {layout_str}")

  block_sizes = splash_attention_kernel.BlockSizes(
      block_q=min(block_q, q_seq_len),
      block_kv=min(block_kv, kv_seq_len),
      block_kv_compute=min(block_kv_compute, kv_seq_len),
      block_q_dkv=min(block_q_dkv, q_seq_len),
      block_kv_dkv=min(block_kv_dkv, kv_seq_len),
      block_kv_dkv_compute=min(block_kv_dkv_compute, kv_seq_len),
      block_q_dq=min(block_q_dq, q_seq_len)
      if not use_fused_bwd_kernel and block_q_dq is not None
      else None,
      block_kv_dq=min(block_kv_dq, kv_seq_len)
      if not use_fused_bwd_kernel and block_kv_dq is not None
      else None,
      use_fused_bwd_kernel=use_fused_bwd_kernel,
      q_layout=_get_layout(q_layout),
      k_layout=_get_layout(k_layout),
      v_layout=_get_layout(v_layout),
  )

  mask_shape = (kv_seq_len, kv_seq_len)
  single_mask = _make_mask(mask_shape, is_causal, local_window_size, cp_size)

  multi_head_mask = splash_attention_mask.MultiHeadMask(
      masks=(single_mask,) * n_heads
  )

  is_mqa = n_kv_heads == 1

  with jax.default_device(jax.devices("cpu")[0]):
    if is_mqa:
      splash_kernel = splash_attention_kernel.make_splash_mqa(
          mask=multi_head_mask,
          head_shards=1,
          q_seq_shards=cp_size,
          block_sizes=block_sizes,
      )
    else:
      splash_kernel = splash_attention_kernel.make_splash_mha(
          mask=multi_head_mask,
          head_shards=1,
          q_seq_shards=cp_size,
          block_sizes=block_sizes,
      )

  # Static mask infos and kernel kwargs from creation time.
  # These are the same for every forward/backward call with the same config.
  fwd_mask_info = jax.tree_util.tree_map(
      np.asarray, splash_kernel.fwd_mask_info
  )
  dq_mask_info = jax.tree_util.tree_map(np.asarray, splash_kernel.dq_mask_info)
  dkv_mask_info = jax.tree_util.tree_map(
      np.asarray, splash_kernel.dkv_mask_info
  )
  _mask_value = splash_kernel.kwargs.get("mask_value", _DEFAULT_MASK_VALUE)
  _mask_function = splash_kernel.kwargs.get("mask_function", None)
  _attn_logits_soft_cap = splash_kernel.kwargs.get("attn_logits_soft_cap", None)
  _residual_checkpoint_name = splash_kernel.kwargs.get(
      "residual_checkpoint_name", None
  )
  _interpret = splash_kernel.kwargs.get("interpret", False)

  def _segment_ids(segment_ids_b, cp_rank=None):
    return make_cp_segment_ids(
        splash_attention_kernel.SegmentIds,
        segment_ids_b,
        cp_size,
        q_seq_len,
        cp_rank,
    )

  @jax.jit
  def splash_fn(q, k, v, segment_ids=None, cp_rank=None):
    """Forward: returns (out, logsumexp) for the whole batch."""
    head_dim = q.shape[-1]
    scale = jnp.array(1.0 / math.sqrt(head_dim), dtype=q.dtype)
    q_scaled = q * scale
    fwd_mask_info_sliced = _slice_mask_info(fwd_mask_info, cp_size, cp_rank)

    def _single_fwd_with_lse(q_b, k_b, v_b, segment_ids_b):
      """Forward for one batch element; returns (out_b, logsumexp_b)."""
      # q_b is already scaled (q * 1/sqrt(head_dim)) from the caller.
      seg_ids = None
      fwd_mask_info_b = fwd_mask_info_sliced
      if segment_ids_b is not None:
        seg_ids = _segment_ids(segment_ids_b, cp_rank)
        # TODO: Support document-aware block skipping for cp_size > 1 by slicing
        # segment_ids into q_seg and kv_seg before computing block bounds.
        if cp_size == 1:
          fwd_mask_info_b = _maybe_document_aware_mask_info(
              fwd_mask_info_sliced,
              segment_ids_b,
              block_sizes.block_q,
              block_sizes.block_kv,
              is_dkv=False,
              use_document_aware_block_skipping=use_document_aware_block_skipping,
          )

      out_b, (logsumexp_b,) = splash_attention_kernel._splash_attention_forward(
          fwd_mask_info_b,
          q_b,
          k_b,
          v_b,
          segment_ids=seg_ids,
          sinks=None,
          mask_value=_mask_value,
          is_mqa=is_mqa,
          block_sizes=block_sizes,
          residual_checkpoint_name=_residual_checkpoint_name,
          save_residuals=True,
          mask_function=_mask_function,
          attn_logits_soft_cap=_attn_logits_soft_cap,
          interpret=_interpret,
      )
      return out_b, logsumexp_b

    if segment_ids is None:
      out, logsumexp = jax.vmap(_single_fwd_with_lse, in_axes=(0, 0, 0, None))(
          q_scaled, k, v, None
      )
    else:
      out, logsumexp = jax.vmap(_single_fwd_with_lse)(
          q_scaled, k, v, segment_ids
      )
    return out, logsumexp

  @jax.jit
  def splash_bwd_fn(q, k, v, out, logsumexp, g, segment_ids=None, cp_rank=None):
    """Backward: uses saved (out, logsumexp) — no extra forward pass.

    Supports both `jax.vmap` and `jax.lax.scan` over the batch dimension.
    By default, it uses `jax.lax.scan` to be memory safe, sizing the XLA
    backward program for one element at a time to avoid O(batch) growth in
    program size. `jax.vmap` can be enabled for better throughput but may
    hit memory limits for large batches.

    Args:
      q: Query tensor.
      k: Key tensor.
      v: Value tensor.
      out: Output of the forward pass.
      logsumexp: Logsumexp values from the forward pass.
      g: Gradient of the loss with respect to the output (`out`).
      segment_ids: Segment IDs.
      cp_rank: Optional context-parallel rank tensor of shape (1,).
    """
    head_dim = q.shape[-1]
    scale = jnp.array(1.0 / math.sqrt(head_dim), dtype=q.dtype)
    q_scaled = q * scale
    dq_mask_info_sliced = _slice_mask_info(dq_mask_info, cp_size, cp_rank)
    dkv_mask_info_sliced = _slice_mask_info(dkv_mask_info, cp_size, cp_rank)

    def _single_bwd(
        q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b, segment_ids_b
    ):
      """Backward for one batch element using precomputed residuals."""
      seg_ids = None
      dq_mask_info_b = dq_mask_info_sliced
      dkv_mask_info_b = dkv_mask_info_sliced
      if segment_ids_b is not None:
        seg_ids = _segment_ids(segment_ids_b, cp_rank)
        # TODO: Support document-aware block skipping for cp_size > 1 in bwd.
        if cp_size == 1:
          dkv_mask_info_b = _maybe_document_aware_mask_info(
              dkv_mask_info_sliced,
              segment_ids_b,
              block_sizes.block_q_dkv,
              block_sizes.block_kv_dkv,
              is_dkv=True,
              use_document_aware_block_skipping=use_document_aware_block_skipping,
          )
          if dq_mask_info_sliced is not None:
            dq_mask_info_b = _maybe_document_aware_mask_info(
                dq_mask_info_sliced,
                segment_ids_b,
                block_sizes.block_q_dq,
                block_sizes.block_kv_dq,
                is_dkv=False,
                use_document_aware_block_skipping=use_document_aware_block_skipping,
            )

      res = (
          q_scaled_b,
          k_b,
          v_b,
          seg_ids,
          None,  # sinks
          out_b,
          logsumexp_b,
          dq_mask_info_b,
          dkv_mask_info_b,
      )
      all_grads = splash_attention_kernel._splash_attention_bwd(
          False,  # save_residuals (nondiff)
          _mask_value,  # mask_value (nondiff)
          is_mqa,  # is_mqa (nondiff)
          block_sizes,  # block_sizes (nondiff)
          _residual_checkpoint_name,  # residual_checkpoint_name (nondiff)
          _mask_function,  # mask_function (nondiff)
          _attn_logits_soft_cap,  # attn_logits_soft_cap (nondiff)
          _interpret,  # interpret (nondiff)
          res,  # residuals
          g_b,  # do (output gradient)
      )
      # all_grads = (dfwd_mask_info=None, ddq_mask_info=None, ddkv_mask_info=None,
      #              dq_scaled, dk, dv, dsegment_ids=None, dsinks)
      return all_grads[3], all_grads[4], all_grads[5]  # dq_scaled, dk, dv

    def _scan_bwd_body(carry, args):
      """Single-element backward body for jax.lax.scan (sequential over batch)."""
      q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b = args[:6]
      segment_ids_b = args[6] if len(args) > 6 else None
      dq_scaled_b, dk_b, dv_b = _single_bwd(
          q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b, segment_ids_b
      )
      return carry, (dq_scaled_b, dk_b, dv_b)

    if use_vmap_bwd:
      if segment_ids is None:
        dq_scaled, dk, dv = jax.vmap(
            _single_bwd, in_axes=(0, 0, 0, 0, 0, 0, None)
        )(q_scaled, k, v, out, logsumexp, g, None)
      else:
        dq_scaled, dk, dv = jax.vmap(_single_bwd)(
            q_scaled, k, v, out, logsumexp, g, segment_ids
        )
    else:
      scan_xs = (q_scaled, k, v, out, logsumexp, g)
      if segment_ids is not None:
        scan_xs += (segment_ids,)

      _, (dq_scaled, dk, dv) = jax.lax.scan(
          _scan_bwd_body,
          None,
          scan_xs,
      )

    # Chain rule: q_scaled = q * scale  =>  dq = dq_scaled * scale
    dq = dq_scaled * scale
    return dq, dk, dv

  return splash_fn, splash_bwd_fn


_splash_op_name_cache = {}


@torch._dynamo.assume_constant_result
def _get_splash_op_names(**splash_kwargs):
  """Create and cache PyTorch forward/backward functions for splash attention."""
  cache_key = tuple(sorted(splash_kwargs.items()))
  if cache_key in _splash_op_name_cache:
    return _splash_op_name_cache[cache_key]

  splash_fn, splash_bwd_fn = _make_splash_attention_fn(**splash_kwargs)
  torch_fwd_fn = custom_jax_kernel(splash_fn, name="splash_attention")
  torch_bwd_fn = custom_jax_kernel(splash_bwd_fn, name="splash_attention_bwd")

  # Truncate SHA-256 to 16 hex characters (64 bits) for op name readability.
  identifier = hashlib.sha256(str(cache_key).encode("utf-8")).hexdigest()[:16]
  op_fwd_name = f"splash_attn_fwd_{identifier}"
  op_bwd_name = f"splash_attn_bwd_{identifier}"
  torch_fwd_fn = torch.library.custom_op(
      f"pallas::{op_fwd_name}",
      torch_fwd_fn,
      mutates_args=(),
      schema=(
          "(Tensor q, Tensor k, Tensor v, Tensor? segment_ids, Tensor?"
          " cp_rank=None) -> (Tensor, Tensor)"
      ),
  )
  torch_fwd_fn.register_fake(  # pyrefly: ignore[missing-attribute]
      # lse: (batch, num_q_heads, q_seq_len, ??)
      lambda q, k, v, segment_ids=None, cp_rank=None: (
          torch.empty_like(q),
          torch.empty(
              q.shape[0],
              q.shape[1],
              q.shape[2],
              dtype=torch.float32,
              device=q.device,
          ),
      )
  )

  torch_bwd_fn = torch.library.custom_op(
      f"pallas::{op_bwd_name}",
      torch_bwd_fn,
      mutates_args=(),
      schema=(
          "(Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, Tensor"
          " grad_out, Tensor? segment_ids, Tensor? cp_rank=None) -> (Tensor,"
          " Tensor, Tensor)"
      ),
  )
  torch_bwd_fn.register_fake(  # pyrefly: ignore[missing-attribute]
      lambda q, k, v, out, lse, grad_out, segment_ids=None, cp_rank=None: (
          torch.empty_like(q),
          torch.empty_like(k),
          torch.empty_like(v),
      )
  )

  def setup_context(ctx, inputs, output):
    q, k, v, segment_ids, cp_rank = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, segment_ids, cp_rank)

  def backward(ctx, grad_out, grad_lse):
    del grad_lse
    q, k, v, out, lse, segment_ids, cp_rank = ctx.saved_tensors
    grad_q, grad_k, grad_v = torch_bwd_fn(
        q, k, v, out, lse, grad_out, segment_ids, cp_rank
    )
    return grad_q, grad_k, grad_v, None, None

  torch_fwd_fn.register_autograd(backward, setup_context=setup_context)

  # Note: We return string names instead of the functions themselves.
  # Passing callables through activation checkpointing in Dynamo can trigger
  # safety guards that check for mutation. Since "function guards" are not yet
  # fully supported in PyTorch, this can cause failures. Using strings avoids
  # this issue as they are easily guarded by Dynamo.
  _splash_op_name_cache[cache_key] = (op_fwd_name, op_bwd_name)
  return op_fwd_name, op_bwd_name


def splash_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    segment_ids: Optional[torch.Tensor] = None,
    cp_rank: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    enable_gqa: bool = False,  # Kept for API compatibility
    block_q: int = 512,
    block_kv: int = 512,
    block_dkv: int = 512,
    block_kv_compute: int = 512,
    block_q_dkv: int = 512,
    block_kv_dkv: int = 512,
    block_kv_dkv_compute: int = 512,
    block_q_dq: Optional[int] = None,
    block_kv_dq: Optional[int] = None,
    use_fused_bwd_kernel: bool = True,
    use_vmap_bwd: bool = False,
    use_document_aware_block_skipping: bool = True,
    q_layout: str = "HEAD_DIM_MINOR",
    k_layout: str = "HEAD_DIM_MINOR",
    v_layout: str = "HEAD_DIM_MINOR",
) -> torch.Tensor:
  """Replacement for F.scaled_dot_product_attention using splash attention.

  Supports backward pass: gradients flow through q, k, v via the JAX splash
  backward kernel. The forward saves (out, logsumexp) so the backward can call
  the pallas backward kernel directly — no extra forward pass at backward time.

  Args:
    q: Query tensor.
    k: Key tensor.
    v: Value tensor.
    segment_ids: Optional segment IDs for document packing.
    cp_rank: Optional context-parallel rank tensor of shape (1,) when q is
      sequence-sharded (q_seq_len < kv_seq_len).
    scale: Scale factor for attention logits.
    is_causal: Whether to apply causal mask.
    enable_gqa: Kept for API compatibility.
    block_q: Block size for Q.
    block_kv: Block size for KV.
    block_dkv: Block size for the dk/dv backward kernel tiles. Smaller values
      reduce the XLA program size (enabling larger batch sizes) at a small
      efficiency cost. Default 512 matches block_kv; use 256 for large batches.
    block_kv_compute: Block size for KV compute.
    block_q_dkv: Block size for Q in DKV.
    block_kv_dkv: Block size for KV in DKV.
    block_kv_dkv_compute: Block size for KV compute in DKV.
    block_q_dq: Block size for Q in DQ.
    block_kv_dq: Block size for KV in DQ.
    use_fused_bwd_kernel: Whether to use fused backward kernel.
    use_vmap_bwd: Whether to vectorize backward over the batch. When false,
      backward uses a memory-safe sequential scan. When true, batch-native
      masking also permits XLA to lower backward as a batched custom call.
    use_document_aware_block_skipping: Select document-aware block skipping when
      true or batch-native masking when false. Both modes retain exact
      segment-ID masking. This option is independent of ``use_vmap_bwd``:
      batch-native masking still benefits the always-vmapped forward when
      backward uses scan; enabling both makes backward batch-native as well.
      Only document-aware block skipping prunes whole blocks using per-example
      metadata.
    q_layout: Layout for Q.
    k_layout: Layout for K.
    v_layout: Layout for V.
  """

  block_q = int(os.environ.get("SPLASH_BLOCK_Q", block_q))
  block_kv = int(os.environ.get("SPLASH_BLOCK_KV", block_kv))
  block_dkv = int(os.environ.get("SPLASH_BLOCK_DKV", block_dkv))

  batch, n_heads, seq_len, head_dim = q.shape
  n_kv_heads = k.shape[1]
  kv_seq_len = k.shape[2]

  if enable_gqa != (n_heads != n_kv_heads):
    raise ValueError(
        f"enable_gqa ({enable_gqa}) does not match inputs: "
        f"n_heads={n_heads}, n_kv_heads={n_kv_heads}"
    )

  splash_args = dict(
      seq_len=seq_len,
      n_heads=n_heads,
      n_kv_heads=n_kv_heads,
      is_causal=is_causal,
      local_window_size=local_window_size,
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute,
      block_q_dkv=block_q_dkv,
      block_kv_dkv=block_kv_dkv,
      block_kv_dkv_compute=block_kv_dkv_compute,
      block_q_dq=block_q_dq,
      block_kv_dq=block_kv_dq,
      use_fused_bwd_kernel=use_fused_bwd_kernel,
      use_vmap_bwd=use_vmap_bwd,
      use_document_aware_block_skipping=use_document_aware_block_skipping,
      q_layout=q_layout,
      k_layout=k_layout,
      v_layout=v_layout,
  )
  if kv_seq_len != seq_len:
    splash_args["kv_seq_len"] = kv_seq_len
  torch_fwd_name, torch_bwd_name = _get_splash_op_names(**splash_args)
  torch_fwd_fn = getattr(torch.ops.pallas, torch_fwd_name)
  torch_bwd_fn = getattr(torch.ops.pallas, torch_bwd_name)

  if scale is not None:
    default_scale = 1.0 / math.sqrt(head_dim)
    if abs(scale - default_scale) > 1e-6:
      scale_ratio = scale / default_scale
      q = q * scale_ratio  # plain PyTorch op — autograd handles gradient here

  input_dtype = q.dtype
  if n_kv_heads == 1:
    # For MQA (1 kv head) the splash attentionn kernel expects the kv
    # head dimension to be squeezed.
    k = torch.squeeze(k, dim=1)
    v = torch.squeeze(v, dim=1)
  result, _ = torch_fwd_fn(q, k, v, segment_ids, cp_rank)
  result = typing.cast(torch.Tensor, result)

  if result.dtype != input_dtype:
    result = result.to(input_dtype)
  return result


def splash_block_size_for(
    seq_len: int, max_block_size: int = 512, min_block_size: int = 128
) -> int:
  """Largest power-of-two block size <= ``max_block_size`` dividing ``seq_len``.

  The splash kernel needs every block size to divide the sequence length, so a
  fixed 512 block fails for lengths like 768 that bucketed or packed inputs
  produce. Returns 0 if no block >= ``min_block_size`` divides ``seq_len``.
  """
  block = max_block_size
  while block >= min_block_size:
    if seq_len % block == 0:
      return block
    block //= 2
  return 0


_SPLASH_BLOCK_KWARGS = (
    "block_q",
    "block_kv",
    "block_dkv",
    "block_kv_compute",
    "block_q_dkv",
    "block_kv_dkv",
    "block_kv_dkv_compute",
)


class SplashAttention(nn.Module):
  """Inner attention using custom Splash Attention Pallas kernel.

  Accepts (B, S, H, D) inputs, or a folded (T, H, D) token stream treated as
  B=1. With ``auto_block_size=True`` the block sizes are picked per call from
  the sequence length (``splash_block_size_for``), for callers whose sequence
  lengths vary (e.g. bucketed RL batches); if no block size fits, the call
  falls back to ``original_module``.
  """

  def __init__(
      self,
      is_causal: bool = True,
      local_window_size: Optional[int] = None,
      original_module: Optional[nn.Module] = None,
      auto_block_size: bool = False,
      **sa_kwargs,
  ) -> None:
    super().__init__()
    self.is_causal = is_causal
    self.local_window_size = local_window_size
    self.original_module = original_module
    self.auto_block_size = auto_block_size
    self.sa_kwargs = sa_kwargs

  def forward(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      *,
      scale: Optional[float] = None,
      enable_gqa: bool = False,
      is_causal: Optional[bool] = None,
      attention_masks: Any = None,
      positions: Optional[torch.Tensor] = None,
      segment_ids: Optional[torch.Tensor] = None,
      cp_rank: Optional[torch.Tensor] = None,
      **kwargs,
  ) -> torch.Tensor:
    was_3d = q.dim() == 3
    seq_len = q.shape[0] if was_3d else q.shape[1]
    causal = is_causal if is_causal is not None else self.is_causal
    block = splash_block_size_for(seq_len) if self.auto_block_size else None
    # Fall back to the original module on CPU, for sequences too short for the
    # Pallas kernel, or when no block size divides the sequence length.
    if (
        q.device.type == "cpu" or seq_len < 128 or block == 0
    ) and self.original_module is not None:
      cp_kwargs = {}
      if cp_rank is not None:
        cp_kwargs["cp_rank"] = cp_rank
      return self.original_module(
          q,
          k,
          v,
          scale=scale,
          enable_gqa=enable_gqa,
          is_causal=causal,
          attention_masks=attention_masks,
          positions=positions,
          segment_ids=segment_ids,
          **cp_kwargs,
          **kwargs,
      )

    if kwargs.get("out_transform") is not None:
      raise NotImplementedError(
          "SplashAttention does not support `out_transform`."
      )

    q_is_dt = isinstance(q, DTensor)
    q_mesh = q_placements = None
    if q_is_dt:
      q_mesh, q_placements = q.device_mesh, q.placements
      q = q.to_local()

    if isinstance(k, DTensor):
      k = k.to_local()
    if isinstance(v, DTensor):
      v = v.to_local()
    if isinstance(positions, DTensor):
      positions = positions.to_local()
    if isinstance(segment_ids, DTensor):
      segment_ids = segment_ids.to_local()

    # `segment_ids` is computed once per step in `Decoder.forward` and threaded
    # down. The fallback is for callers that pass only `positions` --
    # correct, but
    # it runs once per layer, which profiling showed to be 25% of device time.
    if segment_ids is None and positions is not None:
      segment_ids = segment_ids_from_positions(positions)

    if was_3d:
      q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    if segment_ids is not None and segment_ids.dim() == 1:
      segment_ids = segment_ids.unsqueeze(0)

    # TorchTitan passes tensors as (B, S, H, D). However, the splash_sdpa
    # Pallas kernel expects (B, H, S, D). We transpose them here to match the
    # kernel's expected layout.
    q_trans = q.transpose(1, 2).contiguous()
    k_trans = k.transpose(1, 2).contiguous()
    v_trans = v.transpose(1, 2).contiguous()

    extra_sa_kwargs = dict(self.sa_kwargs)
    if block:
      extra_sa_kwargs.update({name: block for name in _SPLASH_BLOCK_KWARGS})
    if cp_rank is not None:
      extra_sa_kwargs["cp_rank"] = cp_rank

    out = splash_sdpa(
        q_trans,
        k_trans,
        v_trans,
        segment_ids=segment_ids,
        scale=scale,
        is_causal=causal,
        local_window_size=self.local_window_size,
        enable_gqa=enable_gqa,
        **extra_sa_kwargs,
    )

    # Transpose back to (B, S, H, D)
    out = out.transpose(1, 2)
    if was_3d:
      out = out.squeeze(0)
    if q_is_dt:
      out = DTensor.from_local(out, q_mesh, q_placements)
    return out
