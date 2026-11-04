"""PyTorch / TorchTPU bridge for the fused QK-LayerNorm + RoPE Pallas kernel.

The pure-JAX Pallas forward and backward kernels live in
`qk_norm_rope_kernel.py`. This module keeps module import completely free of
`jax` and `torch_tpu` so CPU-only environments and contract tests can import
`supported()` without a TPU runtime installed; the `pallas.jax_op` custom ops
and autograd formula are registered lazily on first TPU invocation.
"""

import functools
import torch

NUM_LANES = 128
SUBLANE_ALIGN = 8
DEFAULT_BLOCK_S = 512


def lane_group_width(head_dim: int) -> int:
  """Returns the 128-lane VMEM group width for `head_dim` (64 or k * 128)."""
  if head_dim == 64:
    return NUM_LANES
  if head_dim >= NUM_LANES and head_dim % NUM_LANES == 0:
    return head_dim
  raise ValueError(
      f"head_dim {head_dim} must be 64 or a positive multiple of {NUM_LANES}"
  )


def pick_seq_block(seq: int, block_s: int = DEFAULT_BLOCK_S) -> int | None:
  """Returns the effective sequence tile size, or `None` if unaligned."""
  if seq <= 0 or block_s <= 0:
    return None
  bs = min(block_s, seq)
  if bs % SUBLANE_ALIGN != 0 or seq % bs != 0:
    return None
  return bs


def supported(
    xq: torch.Tensor,
    heads: int,
    head_dim: int,
    *,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    block_s: int = DEFAULT_BLOCK_S,
) -> str | None:
  """Returns why `fused_qk_norm_rope` cannot run on these inputs, or `None`."""
  if xq.device.type != "tpu":
    return f"tensors are on {xq.device.type}, not tpu"
  if xq.ndim != 3:
    return f"expected (B, S, H*D) projections, got rank {xq.ndim}"
  batch, seq, hidden_dim = xq.shape
  if batch <= 0 or seq <= 0:
    return f"batch ({batch}) and seq ({seq}) must be positive"
  if heads <= 0:
    return f"heads ({heads}) must be positive"
  if head_dim != 64 and (head_dim < NUM_LANES or head_dim % NUM_LANES != 0):
    return (
        f"head_dim {head_dim} must be 64 or a positive multiple of {NUM_LANES}"
    )
  expected_hidden = heads * head_dim
  if hidden_dim != expected_hidden:
    return f"expected projection width H*D={expected_hidden}, got {hidden_dim}"
  group = lane_group_width(head_dim)
  if hidden_dim % group != 0:
    return f"H*D={hidden_dim} is not a whole number of {group}-lane groups"
  if pick_seq_block(seq, block_s) is None:
    return (
        f"sequence {seq} with block_s={block_s} is not a {SUBLANE_ALIGN}-"
        "aligned tile multiple"
    )
  if cos is not None or sin is not None:
    if cos is None or sin is None:
      return "both cos and sin must be provided together"
    if cos.device != xq.device or sin.device != xq.device:
      return "cos/sin must be on the same TPU device as xq"
    if cos.ndim != 3 or tuple(cos.shape) != tuple(sin.shape):
      return (
          f"cos/sin shapes {tuple(cos.shape)} vs {tuple(sin.shape)} must match"
          " and have rank 3"
      )
    if cos.shape[0] not in (1, batch) or tuple(cos.shape[1:]) != (
        seq,
        head_dim,
    ):
      return (
          f"expected cos/sin shape (1 or {batch}, {seq}, {head_dim}), got"
          f" {tuple(cos.shape)}"
      )
  return None


@functools.cache
def _get_pallas_ops():
  """Lazily imports `torch_tpu` + JAX Pallas kernels and registers autograd."""
  import jax  # pylint: disable=g-import-not-at-top
  from torch_tpu._internal import pallas  # pylint: disable=g-import-not-at-top
  from torchtitan.experiments.tpu.kernels import qk_norm_rope_kernel as kernel  # pylint: disable=g-import-not-at-top

  def _fwd_jax(
      xq: jax.Array,
      xk: jax.Array,
      xv: jax.Array,
      wq: jax.Array,
      wk: jax.Array,
      cos: jax.Array,
      sin: jax.Array,
      heads: int,
      eps: float,
      q_scale: float,
      block_s: int,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    return kernel.qk_norm_rope_fwd(
        xq,
        xk,
        xv,
        wq,
        wk,
        cos,
        sin,
        heads=heads,
        eps=eps,
        q_scale=q_scale,
        block_s=block_s,
    )

  def _bwd_jax(
      xq: jax.Array,
      xk: jax.Array,
      wq: jax.Array,
      wk: jax.Array,
      cos: jax.Array,
      sin: jax.Array,
      gq: jax.Array,
      gk: jax.Array,
      gv: jax.Array,
      eps: float,
      q_scale: float,
      block_s: int,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return kernel.qk_norm_rope_bwd(
        xq,
        xk,
        wq,
        wk,
        cos,
        sin,
        gq,
        gk,
        gv,
        eps=eps,
        q_scale=q_scale,
        block_s=block_s,
    )

  fwd_op = pallas.jax_op("pallas::qk_norm_rope_fwd", _fwd_jax)
  bwd_op = pallas.jax_op("pallas::qk_norm_rope_bwd", _bwd_jax)

  def _setup_context(ctx, inputs, output):
    del output
    xq, xk, _, wq, wk, cos, sin, _, eps, q_scale, block_s = inputs
    ctx.save_for_backward(xq, xk, wq, wk, cos, sin)
    ctx.eps = eps
    ctx.q_scale = q_scale
    ctx.block_s = block_s

  def _backward(ctx, gq, gk, gv):
    xq, xk, wq, wk, cos, sin = ctx.saved_tensors
    dxq, dxk, dxv, dwq, dwk = bwd_op(
        xq,
        xk,
        wq,
        wk,
        cos,
        sin,
        gq.contiguous(),
        gk.contiguous(),
        gv.contiguous(),
        ctx.eps,
        ctx.q_scale,
        ctx.block_s,
    )
    return dxq, dxk, dxv, dwq, dwk, None, None, None, None, None, None

  fwd_op.register_autograd(_backward, setup_context=_setup_context)
  return fwd_op, bwd_op


def fused_qk_norm_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    heads: int,
    eps: float,
    q_scale: float = 1.0,
    block_s: int = DEFAULT_BLOCK_S,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Runs the fused QK-LayerNorm + `rotate_half` RoPE + heads-first prologue.

  Args:
    xq, xk, xv: `(B, S, H*D)` query, key, and value projection outputs on TPU.
    wq, wk: `(H*D,)` bias-free LayerNorm weights of `q_norm` and `k_norm`.
    cos, sin: `(B, S, D)` or `(1, S, D)` `rotate_half` RoPE tables matching
      `xq.dtype`.
    heads: Number of query/key/value attention heads `H`.
    eps: Shared LayerNorm epsilon.
    q_scale: Multiplier folded into `q` (exact in bf16 for powers of two such as
      `64 ** -0.5 = 0.125`, allowing the attention kernel to run with scale 1).
    block_s: Sequence tile size per Pallas grid step.

  Returns:
    Heads-first `(B, H, S, D)` `(q, k, v)` tensors.
  """
  if heads <= 0 or xq.ndim != 3 or xq.shape[-1] % heads != 0:
    raise ValueError(
        f"Expected rank-3 xq with last dim divisible by heads={heads}, got"
        f" shape={tuple(xq.shape)}"
    )
  head_dim = xq.shape[-1] // heads
  reason = supported(xq, heads, head_dim, cos=cos, sin=sin, block_s=block_s)
  if reason is not None:
    raise ValueError(f"fused_qk_norm_rope unsupported inputs: {reason}")
  if tuple(xk.shape) != tuple(xq.shape) or tuple(xv.shape) != tuple(xq.shape):
    raise ValueError(
        f"Expected matching xq/xk/xv shapes, got {tuple(xq.shape)},"
        f" {tuple(xk.shape)}, {tuple(xv.shape)}"
    )
  hidden_dim = xq.shape[-1]
  if tuple(wq.shape) != (hidden_dim,) or tuple(wk.shape) != (hidden_dim,):
    raise ValueError(
        f"Expected wq/wk shape ({hidden_dim},), got {tuple(wq.shape)} and"
        f" {tuple(wk.shape)}"
    )
  bs = pick_seq_block(xq.shape[1], block_s)
  assert bs is not None
  fwd_op, _ = _get_pallas_ops()
  return fwd_op(
      xq.contiguous(),
      xk.contiguous(),
      xv.contiguous(),
      wq.contiguous(),
      wk.contiguous(),
      cos.contiguous(),
      sin.contiguous(),
      int(heads),
      float(eps),
      float(q_scale),
      int(bs),
  )
