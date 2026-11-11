# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch op for the packed-q MLA Splash kernels (``mla_splash.py``).

``mla_attention(q, kv, k_pe, cos, sin, segment_ids, scale=...)`` takes the
tensors DeepSeek MLA produces before any split, rope, concat or transpose:

* ``q [B, S, H * (128 + Dr)]``: ``wq_b``'s output, q_pe not yet roped.
* ``kv [B, S, H * 256]``: ``wkv_b``'s output, per head ``[k_nope | v]``.
* ``k_pe [B, S, Dr]``: roped (shared by every head, so roping it is cheap).
* ``cos, sin [B, S, Dr]`` f32: per-token pair tables, see ``rope_pair_tables``.

and returns ``o [B, S, H * 128]``, which feeds ``wo`` directly. The backward
returns dq in q's layout (through the rope), dkv in kv's layout and dk_pe for
the roped k_pe. Tiles, and fused vs split backward, follow the shape
(``mla_splash.forward_tiles`` / ``backward_tiles``).
"""

import jax
import torch
from torch_tpu._internal import pallas

from torchtitan.experiments.tpu.kernels import mla_splash


def _jax_mla_fwd(q: jax.Array, kv: jax.Array, k_pe: jax.Array, cos: jax.Array,
                 sin: jax.Array, segment_ids: jax.Array, scale: float,
                 block_q: int, block_kv: int,
                 heads_per_step: int) -> tuple[jax.Array, jax.Array]:
  return mla_splash.mla_splash_fwd(
      q, kv, k_pe, cos, sin, segment_ids, scale=scale, block_q=block_q,
      block_kv=block_kv, heads_per_step=heads_per_step)


def _jax_mla_bwd(q: jax.Array, kv: jax.Array, k_pe: jax.Array, cos: jax.Array,
                 sin: jax.Array, segment_ids: jax.Array, o: jax.Array,
                 lse: jax.Array, do: jax.Array, scale: float, block_q: int,
                 block_kv: int, heads_per_step: int,
                 split: bool) -> tuple[jax.Array, jax.Array, jax.Array]:
  return mla_splash.mla_splash_bwd(
      q, kv, k_pe, cos, sin, segment_ids, o, lse, do, scale=scale,
      block_q=block_q, block_kv=block_kv, heads_per_step=heads_per_step,
      split=split)


mla_attn_op: torch._library.custom_ops.CustomOpDef = pallas.jax_op(
    "pallas::mla_splash_attn_fwd", _jax_mla_fwd
)
mla_attn_op_backward: torch._library.custom_ops.CustomOpDef = pallas.jax_op(
    "pallas::mla_splash_attn_bwd", _jax_mla_bwd
)

def _setup_context(ctx, inputs, output, **kwargs):
  del kwargs
  q, kv, k_pe, cos, sin, segment_ids, scale = inputs[:7]
  o, lse = output
  ctx.save_for_backward(q, kv, k_pe, cos, sin, segment_ids, o, lse)
  ctx.scale = scale


def _backward(ctx, grad_o, grad_lse):
  del grad_lse
  q, kv, k_pe, cos, sin, segment_ids, o, lse = ctx.saved_tensors
  tiles = mla_splash.backward_tiles(
      q.shape[1], o.shape[-1] // 128, k_pe.shape[-1])
  dq, dkv, dk_pe = mla_attn_op_backward(
      q, kv, k_pe, cos, sin, segment_ids, o, lse, grad_o.contiguous(),
      ctx.scale, *tiles)
  return (dq, dkv, dk_pe) + (None,) * 7


mla_attn_op.register_autograd(_backward, setup_context=_setup_context)


def rope_pair_tables(
    cis: torch.Tensor, batch: int, seq: int
) -> tuple[torch.Tensor, torch.Tensor]:
  """``(cos, sin)`` pair tables ``[B, S, Dr]`` f32 from complex ``cis [B*S, Dr/2]``.

  ``cos = [c0, c0, c1, c1, ...]`` and ``sin = [-s0, s0, -s1, s1, ...]``, so the
  interleaved complex-pair rope is ``x * cos + pair_swap(x) * sin``.
  """
  c, s = cis.real.float(), cis.imag.float()
  cos = torch.stack([c, c], dim=-1).flatten(-2)
  sin = torch.stack([-s, s], dim=-1).flatten(-2)
  return cos.view(batch, seq, -1), sin.view(batch, seq, -1)


def mla_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
  """MLA attention on unsplit projections; see the module docstring."""
  tiles = mla_splash.forward_tiles(
      q.shape[1], kv.shape[-1] // 256, k_pe.shape[-1])
  o, _ = mla_attn_op(q, kv, k_pe, cos, sin, segment_ids, scale, *tiles)
  return o


__all__ = ["mla_attention", "rope_pair_tables"]
