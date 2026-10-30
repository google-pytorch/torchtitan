"""Splash Attention integration for torch_tpu via Pallas and Tokamax.

Wraps Tokamax's splash_attention_kernel for use with PyTorch tensors on TPU.
Uses torch_tpu._internal.pallas.pallas.jax_op to bridge JAX -> PyTorch.
"""

import functools
import math
from typing import Optional

# JAX imports
import jax
import jax.numpy as jnp
# Workaround for https://github.com/openxla/tokamax/issues/240 (same as
# linear_softmax_cross_entropy_loss.py): tokamax applies @jaxtyping.jaxtyped
# at import time, and typeguard's AST instrumentation cannot parse jaxtyping
# shape-string annotations ("*B T H d"). Swap in a pass-through decorator for
# the duration of the tokamax import.
import jaxtyping
import numpy as np
from torchtitan.experiments.tpu.headtail import (
    make_cp_segment_ids,
    make_headtail_splash_mask,
    slice_cp_mask_info,
)

_real_jaxtyped = jaxtyping.jaxtyped


def _dummy_jaxtyped(*args, **kwargs):
  def decorator(f):
    return f

  if len(args) == 1 and callable(args[0]):
    return args[0]
  return decorator


jaxtyping.jaxtyped = _dummy_jaxtyped
try:
  from tokamax._src.ops.splash_attention import splash_attention_kernel  # noqa: E402  pylint: disable=g-import-not-at-top
  from tokamax._src.ops.splash_attention import splash_attention_mask  # noqa: E402  pylint: disable=g-import-not-at-top
except ImportError:
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel  # noqa: E402  pylint: disable=g-import-not-at-top
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask  # noqa: E402  pylint: disable=g-import-not-at-top

jaxtyping.jaxtyped = _real_jaxtyped

import torch  # noqa: E402  pylint: disable=g-import-not-at-top,g-bad-import-order
# torch_tpu Pallas bridge
from torch_tpu._internal import pallas


# Taken from tokamax._src.ops.splash_attention.base.DEFAULT_MASK_VALUE.
_DEFAULT_MASK_VALUE = -0.7 * float(
    np.finfo(np.dtype("float32")).max
)


_MASK_INFO_SLICE_AXES = (
    ("mask_next", 0),
    ("active_rows", 0),
    ("active_cols", 0),
    ("num_active_blocks", 0),
    ("block_mask", 0),
    ("q_sequence", 0),
)

_make_mask = functools.partial(make_headtail_splash_mask, splash_attention_mask)


def _slice_mask_info(mask_info, cp_size: int, cp_rank: jax.Array | None):
  """Slice a Tokamax MaskInfo crafted for q_seq_shards=cp_size to cp_rank's shard."""
  return slice_cp_mask_info(mask_info, cp_size, cp_rank, _MASK_INFO_SLICE_AXES)


class TokamaxConfig:
  """Global configuration container for hardware-static hyperparameters."""
  block_q: int = 512
  block_kv: int = 512
  block_kv_compute: int = 512
  block_q_dkv: int = 512
  block_kv_dkv: int = 512
  block_kv_dkv_compute: int = 512
  block_q_dq: Optional[int] = None
  block_kv_dq: Optional[int] = None
  use_fused_bwd_kernel: bool = True
  use_vmap_bwd: bool = False
  q_layout: str = "HEAD_DIM_MINOR"
  k_layout: str = "HEAD_DIM_MINOR"
  v_layout: str = "HEAD_DIM_MINOR"
  num_stacked_q_heads: int = 1

  @classmethod
  def update(cls, **kwargs):
    for k, v in kwargs.items():
      if v is not None:
        setattr(cls, k, v)


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
    kv_seq_len: int | None = None,
    num_stacked_q_heads: int = 1,
):
  """Create JAX splash attention forward and backward functions using Tokamax.

  Returns:
    (splash_fn, splash_bwd_fn): forward function taking (q, k, v) returning
      (out, logsumexp), and backward function taking (q, k, v, out, logsumexp,
      grad_out) returning (grad_q, grad_k, grad_v) WITHOUT re-running the
      forward.
  """
  del n_heads
  q_seq_len = seq_len
  kv_seq_len = kv_seq_len if kv_seq_len is not None else q_seq_len
  if kv_seq_len % q_seq_len != 0:
    raise ValueError(
        f"kv_seq_len ({kv_seq_len}) must be a multiple of q_seq_len"
        f" ({q_seq_len})."
    )
  cp_size = kv_seq_len // q_seq_len

  def _get_layout(layout_str):
    if layout_str == "HEAD_DIM_MINOR":
      return splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR
    elif layout_str == "SEQ_MINOR":
      return splash_attention_kernel.QKVLayout.SEQ_MINOR
    else:
      raise ValueError(f"Unknown layout: {layout_str}")

  config = splash_attention_kernel.SplashConfig(
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
      num_stacked_q_heads=num_stacked_q_heads,
  )
  if cp_size > 1 and (
      getattr(config, "qk_diag_skip", False)
      or getattr(config, "sv_diag_skip", False)
  ):
    raise NotImplementedError(
        "Tokamax *_diag_skip assumes natural Q order; unsupported with"
        " head-tail CP."
    )

  mask_shape = (kv_seq_len, kv_seq_len)
  single_mask = _make_mask(mask_shape, is_causal, local_window_size, cp_size)

  is_mqa = n_kv_heads == 1

  if is_mqa:
    splash_kernel = splash_attention_kernel.make_splash_mqa_single_device(
        mask=single_mask,
        config=config,
        q_seq_shards=cp_size,
    )
  else:
    splash_kernel = splash_attention_kernel.make_splash_mha_single_device(
        mask=single_mask,
        config=config,
        q_seq_shards=cp_size,
    )

  # Static mask infos and kernel kwargs from creation time.
  # Packed documents are masked only by the runtime segment IDs below, over the
  # static causal/local block metadata (the batch-native masking mode of
  # splash_attention.py). TODO: document-aware block skipping. Tokamax's
  # MaskInfo is a compacted active-block list (active_rows/active_cols), not
  # the block_mask grid that splash_attention._document_aware_mask_info
  # rewrites, so skipping cross-document blocks needs a per-example recompaction.
  fwd_mask_info = splash_kernel.fwd_mask_info
  dkv_mask_info = splash_kernel.dkv_mask_info
  _mask_value = splash_kernel.kwargs.get("mask_value", _DEFAULT_MASK_VALUE)
  _mask_function = splash_kernel.kwargs.get("mask_function", None)
  _fwd_mask_sparsity = splash_kernel.kwargs.get("fwd_mask_sparsity", 1.0)
  _dkv_mask_sparsity = splash_kernel.kwargs.get("dkv_mask_sparsity", 1.0)

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
      # pylint: disable=protected-access
      out_b, stats_b = splash_attention_kernel._splash_attention_forward(
          fwd_mask_info_sliced,
          q_b,
          k_b,
          v_b,
          segment_ids=_segment_ids(segment_ids_b, cp_rank),
          sinks=None,
          mask_value=_mask_value,
          is_mqa=is_mqa,
          config=config,
          save_residuals=True,
          mask_function=_mask_function,
          fwd_mask_sparsity=_fwd_mask_sparsity,
      )
      # pylint: enable=protected-access
      return out_b, stats_b["logsumexp"]

    # in_axes=None broadcasts a None segment_ids to every batch element.
    seg_axis = None if segment_ids is None else 0
    out, logsumexp = jax.vmap(
        _single_fwd_with_lse, in_axes=(0, 0, 0, seg_axis)
    )(q_scaled, k, v, segment_ids)
    return out, logsumexp

  @jax.jit
  def splash_bwd_fn(q, k, v, out, logsumexp, g, segment_ids=None, cp_rank=None):
    """Backward: uses saved (out, logsumexp) — no extra forward pass."""
    head_dim = q.shape[-1]
    scale = jnp.array(1.0 / math.sqrt(head_dim), dtype=q.dtype)
    q_scaled = q * scale
    dkv_mask_info_sliced = _slice_mask_info(dkv_mask_info, cp_size, cp_rank)

    def _single_bwd(
        q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b, segment_ids_b
    ):
      """Backward for one batch element using precomputed residuals."""
      res = (
          q_scaled_b,
          k_b,
          v_b,
          _segment_ids(segment_ids_b, cp_rank),
          None,  # sinks
          out_b,
          logsumexp_b,
          dkv_mask_info_sliced,
          None,  # prng_key (added in upstream tokamax)
      )
      # pylint: disable=protected-access
      all_grads = splash_attention_kernel._splash_attention_bwd(
          False,  # save_residuals (nondiff)
          _mask_value,  # mask_value (nondiff)
          is_mqa,  # is_mqa (nondiff)
          config,  # config (nondiff)
          _mask_function,  # mask_function (nondiff)
          _fwd_mask_sparsity,  # fwd_mask_sparsity (nondiff)
          _dkv_mask_sparsity,  # dkv_mask_sparsity (nondiff)
          res,  # residuals
          g_b,  # do (output gradient)
      )
      # pylint: enable=protected-access
      return all_grads[2], all_grads[3], all_grads[4]  # dq_scaled, dk, dv

    def _scan_bwd_body(carry, args):
      """Single-element backward body for jax.lax.scan (sequential over batch)."""
      q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b = args[:6]
      segment_ids_b = args[6] if len(args) > 6 else None
      dq_scaled_b, dk_b, dv_b = _single_bwd(
          q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b, segment_ids_b
      )
      return carry, (dq_scaled_b, dk_b, dv_b)

    if use_vmap_bwd:
      seg_axis = None if segment_ids is None else 0
      dq_scaled, dk, dv = jax.vmap(
          _single_bwd, in_axes=(0, 0, 0, 0, 0, 0, seg_axis)
      )(q_scaled, k, v, out, logsumexp, g, segment_ids)
    else:
      scan_xs = (q_scaled, k, v, out, logsumexp, g)
      if segment_ids is not None:
        scan_xs += (segment_ids,)
      _, (dq_scaled, dk, dv) = jax.lax.scan(_scan_bwd_body, None, scan_xs)
    # Chain rule: q_scaled = q * scale  =>  dq = dq_scaled * scale
    dq = dq_scaled * scale
    return dq, dk, dv

  return splash_fn, splash_bwd_fn


def jax_attention_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: jax.Array | None,
    cp_rank: jax.Array | None,
    is_causal: bool,
    local_window_size: int,  # Sentinel int: -1 means None
    scale: float,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_kv_dkv_compute: int,
    block_q_dq: int | None,
    block_kv_dq: int | None,
    use_fused_bwd_kernel: bool,
    use_vmap_bwd: bool,
    q_layout: str,
    k_layout: str,
    v_layout: str,
    num_stacked_q_heads: int,
) -> tuple[jax.Array, jax.Array]:
  """Forward attention in JAX using Pallas.

  Accepts raw PyTorch layouts: (batch, seq_len, heads, head_dim), and
  optional int32 ``segment_ids`` of shape (batch, kv_seq_len) for packed
  documents. The tiling/layout fields arrive as arguments (see
  _kernel_config_args) so they are part of the jax_op compile-cache key.
  """
  seq_len = q.shape[1]
  kv_seq_len = k.shape[1]
  n_heads = q.shape[2]
  n_kv_heads = k.shape[2]

  local_window_size_opt = local_window_size if local_window_size != -1 else None

  # 1. Transpose/Squeeze inside JAX so JIT fuses operations
  q_tr = jnp.transpose(q, (0, 2, 1, 3))

  if n_kv_heads == 1:
    k_tr = jnp.squeeze(k, axis=2)
    v_tr = jnp.squeeze(v, axis=2)
  else:
    k_tr = jnp.transpose(k, (0, 2, 1, 3))
    v_tr = jnp.transpose(v, (0, 2, 1, 3))

  splash_fn, _ = _make_splash_attention_fn(
      seq_len=seq_len,
      kv_seq_len=kv_seq_len,
      n_heads=n_heads,
      n_kv_heads=n_kv_heads,
      is_causal=is_causal,
      local_window_size=local_window_size_opt,
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
      q_layout=q_layout,
      k_layout=k_layout,
      v_layout=v_layout,
      num_stacked_q_heads=num_stacked_q_heads,
  )

  # Apply scaling inside the compiled JAX pipeline
  default_scale = 1.0 / math.sqrt(q.shape[-1])
  if abs(scale - default_scale) > 1e-6:
    q_tr = q_tr * (scale / default_scale)

  out_tr, logsumexp = splash_fn(q_tr, k_tr, v_tr, segment_ids, cp_rank)

  # Transpose back: (B, H, S, d) -> (B, S, H, d)
  out = jnp.transpose(out_tr, (0, 2, 1, 3))
  return out, logsumexp


def jax_attention_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    logsumexp: jax.Array,
    grad_out: jax.Array,
    segment_ids: jax.Array | None,
    cp_rank: jax.Array | None,
    is_causal: bool,
    local_window_size: int,  # Sentinel int: -1 means None
    scale: float,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_kv_dkv_compute: int,
    block_q_dq: int | None,
    block_kv_dq: int | None,
    use_fused_bwd_kernel: bool,
    use_vmap_bwd: bool,
    q_layout: str,
    k_layout: str,
    v_layout: str,
    num_stacked_q_heads: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Backward attention in JAX calling direct Pallas kernel backward."""
  seq_len = q.shape[1]
  kv_seq_len = k.shape[1]
  n_heads = q.shape[2]
  n_kv_heads = k.shape[2]

  local_window_size_opt = local_window_size if local_window_size != -1 else None

  # 1. Transpose/Squeeze inputs inside JAX
  q_tr = jnp.transpose(q, (0, 2, 1, 3))
  out_tr = jnp.transpose(out, (0, 2, 1, 3))
  grad_out_tr = jnp.transpose(grad_out, (0, 2, 1, 3))

  if n_kv_heads == 1:
    k_tr = jnp.squeeze(k, axis=2)
    v_tr = jnp.squeeze(v, axis=2)
  else:
    k_tr = jnp.transpose(k, (0, 2, 1, 3))
    v_tr = jnp.transpose(v, (0, 2, 1, 3))

  _, splash_bwd_fn = _make_splash_attention_fn(
      seq_len=seq_len,
      kv_seq_len=kv_seq_len,
      n_heads=n_heads,
      n_kv_heads=n_kv_heads,
      is_causal=is_causal,
      local_window_size=local_window_size_opt,
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
      q_layout=q_layout,
      k_layout=k_layout,
      v_layout=v_layout,
      num_stacked_q_heads=num_stacked_q_heads,
  )

  default_scale = 1.0 / math.sqrt(q.shape[-1])
  if abs(scale - default_scale) > 1e-6:
    q_tr = q_tr * (scale / default_scale)

  dq_tr, dk_tr, dv_tr = splash_bwd_fn(
      q_tr, k_tr, v_tr, out_tr, logsumexp, grad_out_tr, segment_ids, cp_rank
  )

  # Chain rule for the q pre-scaling above: splash_bwd_fn returns the
  # gradient wrt the SCALED q, so a non-default scale needs the same
  # ratio applied to dq (without it, dq is off by default_scale/scale).
  if abs(scale - default_scale) > 1e-6:
    dq_tr = dq_tr * (scale / default_scale)

  # Transpose/Expand back to original PyTorch layouts
  dq = jnp.transpose(dq_tr, (0, 2, 1, 3))

  if n_kv_heads == 1:
    dk = jnp.expand_dims(dk_tr, axis=2)
    dv = jnp.expand_dims(dv_tr, axis=2)
  else:
    dk = jnp.transpose(dk_tr, (0, 2, 1, 3))
    dv = jnp.transpose(dv_tr, (0, 2, 1, 3))

  return dq, dk, dv


# Register standard top-level global Custom Operators exactly like call_kernel.py!
splash_op: torch._library.custom_ops.CustomOpDef = pallas.jax_op(
    "pallas::tokamax_splash_attn_fwd", jax_attention_fwd
)

splash_op_backward: torch._library.custom_ops.CustomOpDef = pallas.jax_op(
    "pallas::tokamax_splash_attn_bwd", jax_attention_bwd
)


_KERNEL_CONFIG_FIELDS = (
    "block_q",
    "block_kv",
    "block_kv_compute",
    "block_q_dkv",
    "block_kv_dkv",
    "block_kv_dkv_compute",
    "block_q_dq",
    "block_kv_dq",
    "use_fused_bwd_kernel",
    "use_vmap_bwd",
    "q_layout",
    "k_layout",
    "v_layout",
    "num_stacked_q_heads",
)


def _kernel_config_args() -> tuple:
  """TokamaxConfig values, in jax_attention_fwd/bwd argument order.

  Passed as op arguments rather than read inside the JAX functions: jax_op
  caches compiled kernels by argument values, so a config read inside would be
  invisible to the cache and a changed config could reuse a stale kernel.
  """
  return tuple(getattr(TokamaxConfig, name) for name in _KERNEL_CONFIG_FIELDS)


def setup_context(ctx, inputs, output, **kwargs):
  del kwargs
  # inputs: q, k, v, segment_ids, cp_rank, is_causal, local_window_size,
  # scale, then the _KERNEL_CONFIG_FIELDS values.
  (
      q,
      k,
      v,
      segment_ids,
      cp_rank,
      is_causal,
      local_window_size,
      scale,
  ) = inputs[:8]
  out, logsumexp = output
  ctx.save_for_backward(q, k, v, out, logsumexp, segment_ids, cp_rank)

  ctx.is_causal = is_causal
  ctx.local_window_size = local_window_size
  ctx.scale = scale
  ctx.kernel_config = inputs[8:]


def backward(ctx, grad_out, grad_lse):
  del grad_lse  # Unused
  q, k, v, out, logsumexp, segment_ids, cp_rank = ctx.saved_tensors

  # Call our clean JAX backward global top-level custom operator!
  dq, dk, dv = splash_op_backward(
      q,
      k,
      v,
      out,
      logsumexp,
      grad_out,
      segment_ids,
      cp_rank,
      ctx.is_causal,
      ctx.local_window_size,
      ctx.scale,
      *ctx.kernel_config,
  )
  # One gradient per forward input; segment_ids, cp_rank, and the scalars get None.
  return (dq, dk, dv) + (None,) * (5 + len(ctx.kernel_config))


# Register the autograd!
splash_op.register_autograd(backward, setup_context=setup_context)


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
    enable_gqa: bool = False,
    block_q: Optional[int] = None,
    block_kv: Optional[int] = None,
    block_dkv: Optional[int] = None,
    block_kv_compute: Optional[int] = None,
    block_q_dkv: Optional[int] = None,
    block_kv_dkv: Optional[int] = None,
    block_kv_dkv_compute: Optional[int] = None,
    block_q_dq: Optional[int] = None,
    block_kv_dq: Optional[int] = None,
    use_fused_bwd_kernel: Optional[bool] = None,
    use_vmap_bwd: Optional[bool] = None,
    q_layout: Optional[str] = None,
    k_layout: Optional[str] = None,
    v_layout: Optional[str] = None,
) -> torch.Tensor:
  """Replacement for F.scaled_dot_product_attention using Tokamax splash attention.

  ``segment_ids`` is an optional (batch, kv_seq_len) int32 tensor of per-token
  document ids for packed sequences; tokens attend only within their own
  document, on top of the causal/local mask.
  """
  # These are accepted for signature compatibility with the JAX splash wrapper
  # but are NOT the channel that reaches the kernel: the kernel is configured
  # from the process-global TokamaxConfig, which the patch populates once at
  # setup.
  # Passing a value here that disagrees with TokamaxConfig would silently do
  # nothing, so refuse it rather than discard it -- a caller who tiles through
  # this path is wrong about how the kernel is configured.
  _passed = {
      "block_q": block_q, "block_kv": block_kv, "block_dkv": block_dkv,
      "block_kv_compute": block_kv_compute, "block_q_dkv": block_q_dkv,
      "block_kv_dkv": block_kv_dkv, "block_kv_dkv_compute": block_kv_dkv_compute,
      "block_q_dq": block_q_dq, "block_kv_dq": block_kv_dq,
      "use_fused_bwd_kernel": use_fused_bwd_kernel, "use_vmap_bwd": use_vmap_bwd,
      "q_layout": q_layout, "k_layout": k_layout, "v_layout": v_layout,
  }
  _conflicts = {
      k: (v, getattr(TokamaxConfig, k))
      for k, v in _passed.items()
      if v is not None and hasattr(TokamaxConfig, k) and v != getattr(TokamaxConfig, k)
  }
  if _conflicts:
    raise ValueError(
        "splash_sdpa received per-call tiling that disagrees with the active"
        " TokamaxConfig; this path cannot honor per-call tiles, so the values"
        " would be silently ignored. Set them on the SplashAttentionKernelConfig"
        f" instead. Conflicts (passed, active): {_conflicts}"
    )
  # Validate enable_gqa rather than ignore it. The stock splash_sdpa in
  # splash_attention.py enforces this same invariant, and the two functions are
  # meant to be interchangeable via a config flag -- so silently accepting a
  # wrong value here would make the two kernels disagree on identical input:
  # stock raises, tokamax runs on. That check catches a real mistake (passing
  # enable_gqa=False for a GQA shape); dropping it when the flag is flipped
  # would lose the assertion precisely when the kernel changes underneath you.
  n_heads, n_kv_heads = q.shape[2], k.shape[2]
  if enable_gqa != (n_heads != n_kv_heads):
    raise ValueError(
        f"enable_gqa ({enable_gqa}) does not match inputs: "
        f"n_heads={n_heads}, n_kv_heads={n_kv_heads}"
    )

  head_dim = q.shape[-1]
  if scale is None:
    scale = 1.0 / math.sqrt(head_dim)

  local_window_size_val = local_window_size if local_window_size is not None else -1

  out, _ = splash_op(
      q,
      k,
      v,
      segment_ids,
      cp_rank,
      is_causal,
      local_window_size_val,
      scale,
      *_kernel_config_args(),
  )
  return out
