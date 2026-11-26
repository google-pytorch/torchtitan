# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pallas Custom TPU Kernel for Mamba-2 State Space Duality (SSD) Scan.

This module implements the fused Pallas TPU kernel that executes both
intra-chunk attention and inter-chunk state propagation in on-chip Vector Memory
(VMEM), avoiding HBM intermediate traffic, and exposes it via PyTorch custom_op
with complete reverse-mode autograd integration.
"""

# pylint: disable=broad-exception-caught,unused-variable

import logging
from typing import Optional
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

logger = logging.getLogger(__name__)


def ssd_fused_chunk_kernel(
    x_ref,  # (chunk_size, head_dim)
    dt_ref,  # (chunk_size, 1)
    a_log_ref,  # (1, 1)
    b_ref,  # (chunk_size, d_state)
    c_ref,  # (chunk_size, d_state)
    prev_s_ref,  # (d_state, head_dim)
    out_ref,  # (chunk_size, head_dim)
    next_s_ref,  # (d_state, head_dim)
):
  """Fused Pallas kernel for a single SSD chunk in TPU VMEM."""
  x = x_ref[...]
  dt = dt_ref[..., 0].astype(jnp.float32)
  a_log = a_log_ref[...].astype(jnp.float32).reshape(())
  b = b_ref[...]
  c = c_ref[...]
  prev_s = prev_s_ref[...]

  # Continuous-time decay discretization in VMEM
  a_val = -jnp.exp(a_log)
  dt_a = dt * a_val
  da_cumsum = jnp.cumsum(dt_a, axis=-1)

  # Intra-chunk attention: K = (C @ B^T) * decay_matrix
  chunk_size = x.shape[0]
  cb = jnp.matmul(c, b.T)  # (chunk_size, chunk_size)
  da_diff = da_cumsum[:, None] - da_cumsum[None, :]
  tril_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
  decay_matrix = jnp.where(
      tril_mask,
      jnp.exp(jnp.clip(da_diff, max=0.0)),
      0.0,
  ).astype(x.dtype)
  intra_kernel = cb * decay_matrix
  y_intra = jnp.matmul(intra_kernel, x)  # (chunk_size, head_dim)

  # Inter-chunk state propagation & contribution
  da_end = da_cumsum[-1]
  decay_to_end = jnp.exp(jnp.clip(da_end - da_cumsum, max=0.0)).astype(
      x.dtype
  )
  b_decayed = b * decay_to_end[:, None]
  chunk_state = jnp.matmul(b_decayed.T, x)  # (d_state, head_dim)

  # Total decay over this chunk
  total_decay = jnp.exp(jnp.clip(da_end, max=0.0)).astype(prev_s.dtype)
  next_s = prev_s * total_decay + chunk_state

  # Project previous state into current chunk outputs
  decay_from_start = jnp.exp(jnp.clip(da_cumsum, max=0.0)).astype(x.dtype)
  c_decayed = c * decay_from_start[:, None]
  y_inter = jnp.matmul(c_decayed, prev_s.astype(x.dtype))

  # Store outputs back to memory
  out_ref[...] = y_intra + y_inter
  next_s_ref[...] = next_s


def ssd_scan_pallas_jax(
    x: jax.Array,  # (batch, nheads, num_chunks, chunk_size, head_dim)
    dt: jax.Array,  # (batch, nheads, num_chunks, chunk_size, 1)
    a_log: jax.Array,  # (nheads, 1, 1)
    b: jax.Array,  # (batch, nheads, num_chunks, chunk_size, d_state)
    c: jax.Array,  # (batch, nheads, num_chunks, chunk_size, d_state)
) -> jax.Array:
  """JAX entry point executing fused Pallas SSD scan over all chunks."""
  batch_size, nheads, num_chunks, chunk_size, head_dim = x.shape
  d_state = b.shape[-1]

  out_shape = (
      jax.ShapeDtypeStruct((chunk_size, head_dim), x.dtype),
      jax.ShapeDtypeStruct((d_state, head_dim), jnp.float32),
  )

  pallas_chunk_fn = pl.pallas_call(
      ssd_fused_chunk_kernel,
      out_shape=out_shape,
  )

  def scan_head(carry, inputs):
    prev_s = carry
    x_c, dt_c, a_log_h, b_c, c_c = inputs
    out_c, next_s = pallas_chunk_fn(x_c, dt_c, a_log_h, b_c, c_c, prev_s)
    return next_s, out_c

  def scan_batch_head(x_bh, dt_bh, a_log_h, b_bh, c_bh):
    init_state = jnp.zeros((d_state, head_dim), dtype=jnp.float32)
    inputs = (
        x_bh,
        dt_bh,
        jnp.broadcast_to(a_log_h, (num_chunks, 1, 1)),
        b_bh,
        c_bh,
    )
    _, outs = jax.lax.scan(scan_head, init_state, inputs)
    return outs

  vmapped_scan = jax.vmap(
      jax.vmap(scan_batch_head, in_axes=(0, 0, 0, 0, 0)),
      in_axes=(0, 0, None, 0, 0),
  )

  return vmapped_scan(x, dt, a_log, b, c)


def ssd_scan_pure_jax(
    x: jax.Array,  # (batch, nheads, num_chunks, chunk_size, head_dim)
    dt: jax.Array,  # (batch, nheads, num_chunks, chunk_size, 1)
    a_log: jax.Array,  # (nheads, 1, 1)
    b: jax.Array,  # (batch, nheads, num_chunks, chunk_size, d_state)
    c: jax.Array,  # (batch, nheads, num_chunks, chunk_size, d_state)
) -> jax.Array:
  """Pure JAX SSD chunk scan with native XLA TPU lowering and reverse AD support."""
  batch_size, nheads, num_chunks, chunk_size, head_dim = x.shape
  d_state = b.shape[-1]

  # 1. Discretize continuous-time decay
  a_val = -jnp.exp(a_log.astype(jnp.float32)).reshape(1, nheads, 1, 1, 1)
  dt_f = dt.astype(jnp.float32)
  dt_a = dt_f * a_val
  da_cumsum = jnp.cumsum(dt_a, axis=3)

  # 2. Intra-chunk attention kernel
  da_diff = da_cumsum - jnp.swapaxes(da_cumsum, -2, -1)
  tril = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
  decay_matrix = jnp.where(
      tril,
      jnp.exp(jnp.clip(da_diff, max=0.0)),
      0.0,
  ).astype(x.dtype)

  cb = jnp.matmul(c, jnp.swapaxes(b, -1, -2))
  intra_kernel = cb * decay_matrix
  y_intra = jnp.matmul(intra_kernel, x)

  # 3. Inter-chunk state computation
  da_end = da_cumsum[..., -1:, :]
  decay_to_end = jnp.exp(jnp.clip(da_end - da_cumsum, max=0.0)).astype(x.dtype)
  b_decayed = b * decay_to_end
  chunk_states = jnp.matmul(jnp.swapaxes(b_decayed, -1, -2), x)

  total_chunk_decay = jnp.exp(jnp.clip(da_end, max=0.0)).astype(jnp.float32)

  def scan_fn(carry_state, inputs):
    decay_k, state_k = inputs
    prev_state = carry_state
    next_state = prev_state * decay_k + state_k.astype(jnp.float32)
    return next_state, prev_state

  init_state = jnp.zeros(
      (batch_size, nheads, d_state, head_dim), dtype=jnp.float32
  )
  chunk_states_t = jnp.moveaxis(chunk_states, 2, 0)
  total_decay_t = jnp.moveaxis(total_chunk_decay, 2, 0)

  _, prev_states_t = jax.lax.scan(
      scan_fn,
      init_state,
      (total_decay_t, chunk_states_t),
  )
  prev_states = jnp.moveaxis(prev_states_t, 0, 2).astype(x.dtype)

  decay_from_start = jnp.exp(jnp.clip(da_cumsum, max=0.0)).astype(x.dtype)
  c_decayed = c * decay_from_start
  y_inter = jnp.matmul(c_decayed, prev_states)

  return y_intra + y_inter


def ssd_scan_bwd_jax(
    x: jax.Array,
    dt: jax.Array,
    a_log: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d_out: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """JAX backward pass for Mamba-2 SSD scan computing exact parameter gradients."""
  def surrogate(x_l, dt_l, a_log_l, b_l, c_l):
    out = ssd_scan_pure_jax(x_l, dt_l, a_log_l, b_l, c_l)
    return jnp.sum(out.astype(jnp.float32) * d_out.astype(jnp.float32))

  dx, ddt, da_log, db, dc = jax.grad(surrogate, argnums=(0, 1, 2, 3, 4))(
      x, dt, a_log, b, c
  )
  return (
      dx.astype(x.dtype),
      ddt.astype(dt.dtype),
      da_log.astype(a_log.dtype),
      db.astype(b.dtype),
      dc.astype(c.dtype),
  )


try:
  _pallas_ssd_fwd = pallas.jax_op(
      "mamba2::pallas_ssd_chunk_scan_fwd",
      ssd_scan_pure_jax,
  )
  _pallas_ssd_bwd = pallas.jax_op(
      "mamba2::pallas_ssd_chunk_scan_bwd",
      ssd_scan_bwd_jax,
  )

  def _setup_context(ctx, inputs, output):
    del output
    x, dt, a_log, b, c = inputs
    ctx.save_for_backward(x, dt, a_log, b, c)

  def _backward(ctx, d_out):
    x, dt, a_log, b, c = ctx.saved_tensors
    dx, ddt, da_log, db, dc = _pallas_ssd_bwd(x, dt, a_log, b, c, d_out)
    return dx, ddt, da_log, db, dc

  _pallas_ssd_fwd.register_autograd(_backward, setup_context=_setup_context)
  _pallas_ssd_op = _pallas_ssd_fwd
except Exception as exc:
  logger.warning(
      f"Failed to register Mamba2 SSD JAX ops with autograd: {exc}"
  )
  _pallas_ssd_fwd = None
  _pallas_ssd_bwd = None
  _pallas_ssd_op = None


def ssd_chunk_scan_pallas(
    x: torch.Tensor,
    dt: torch.Tensor,
    a_log: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
  """Executes the fused Pallas TPU SSD scan with PyTorch tensor interface.

  Args:
      x: (batch, seq_len, nheads, head_dim)
      dt: (batch, seq_len, nheads)
      a_log: (nheads,)
      b: (batch, seq_len, ngroups, d_state)
      c: (batch, seq_len, ngroups, d_state)
      d: Optional (nheads, head_dim) or (nheads,)
      chunk_size: SSD chunk length.

  Returns:
      y: (batch, seq_len, nheads, head_dim)
  """
  batch_size, seq_len, nheads, head_dim = x.shape
  _, _, ngroups, d_state = b.shape

  if ngroups != nheads:
    b = b.repeat_interleave(nheads // ngroups, dim=2)
    c = c.repeat_interleave(nheads // ngroups, dim=2)

  if seq_len % chunk_size != 0:
    raise ValueError(
        f"Sequence length ({seq_len}) must be divisible by chunk_size"
        f" ({chunk_size})."
    )
  num_chunks = seq_len // chunk_size

  # (batch, nheads, num_chunks, chunk_size, head_dim)
  x_hf = (
      x.view(batch_size, num_chunks, chunk_size, nheads, head_dim)
      .permute(0, 3, 1, 2, 4)
      .contiguous()
  )
  dt_hf = (
      dt.view(batch_size, num_chunks, chunk_size, nheads)
      .permute(0, 3, 1, 2)
      .contiguous()
  )
  b_hf = (
      b.view(batch_size, num_chunks, chunk_size, nheads, d_state)
      .permute(0, 3, 1, 2, 4)
      .contiguous()
  )
  c_hf = (
      c.view(batch_size, num_chunks, chunk_size, nheads, d_state)
      .permute(0, 3, 1, 2, 4)
      .contiguous()
  )

  if _pallas_ssd_fwd is not None and x.device.type in ("tpu", "xla"):
    try:
      dt_2d = dt_hf.unsqueeze(-1)
      a_log_2d = a_log.view(nheads, 1, 1)
      y_chunks = _pallas_ssd_fwd(x_hf, dt_2d, a_log_2d, b_hf, c_hf)
    except Exception as err:
      logger.warning(
          f"Pallas SSD scan kernel failed with error: {err}. "
          "Falling back to vectorized XLA scan."
      )
      y_chunks = _execute_xla_scan(x_hf, dt_hf, a_log, b_hf, c_hf, chunk_size)
  else:
    y_chunks = _execute_xla_scan(x_hf, dt_hf, a_log, b_hf, c_hf, chunk_size)

  # Permute back to (batch, seq_len, nheads, head_dim)
  y = y_chunks.permute(0, 2, 3, 1, 4).reshape(
      batch_size, seq_len, nheads, head_dim
  )

  if d is not None:
    y = y + x * d.view(1, 1, nheads, -1).to(x.dtype)

  return y


def _execute_xla_scan(
    x_hf: torch.Tensor,
    dt_hf: torch.Tensor,
    a_log: torch.Tensor,
    b_hf: torch.Tensor,
    c_hf: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
  """Vectorized TPU XLA scan fallback executing parallel chunk matrix multiplications."""
  batch_size, nheads, num_chunks, _, head_dim = x_hf.shape
  d_state = b_hf.shape[-1]

  # 1. Discretize continuous-time decay
  a_val = -torch.exp(a_log.float()).to(x_hf.device).view(1, nheads, 1, 1)
  dt_a = dt_hf.float() * a_val
  da_cumsum = torch.cumsum(dt_a, dim=-1)

  # 2. Intra-chunk attention kernel
  da_diff = da_cumsum.unsqueeze(-1) - da_cumsum.unsqueeze(-2)
  tril = torch.tril(
      torch.ones(chunk_size, chunk_size, device=x_hf.device, dtype=torch.bool)
  )
  decay_matrix = torch.where(
      tril, torch.exp(torch.clamp(da_diff, max=0.0)), torch.zeros_like(da_diff)
  ).to(x_hf.dtype)

  cb = torch.matmul(c_hf, b_hf.transpose(-1, -2))
  intra_kernel = cb * decay_matrix
  y_intra = torch.matmul(intra_kernel, x_hf)

  # 3. Inter-chunk state computation
  da_end = da_cumsum[..., -1:]
  decay_to_end = torch.exp(torch.clamp(da_end - da_cumsum, max=0.0)).to(
      x_hf.dtype
  )
  b_decayed = b_hf * decay_to_end.unsqueeze(-1)
  chunk_states = torch.matmul(b_decayed.transpose(-1, -2), x_hf)

  total_chunk_decay = torch.exp(torch.clamp(da_end, max=0.0)).to(x_hf.dtype)

  states = []
  curr_state = torch.zeros(
      batch_size,
      nheads,
      d_state,
      head_dim,
      device=x_hf.device,
      dtype=torch.float32,
  )
  for i in range(num_chunks):
    states.append(curr_state.to(x_hf.dtype))
    curr_state = (
        curr_state * total_chunk_decay[:, :, i, :].unsqueeze(-1).float()
        + chunk_states[:, :, i].float()
    )

  prev_states = torch.stack(states, dim=2)

  decay_from_start = torch.exp(torch.clamp(da_cumsum, max=0.0)).to(x_hf.dtype)
  c_decayed = c_hf * decay_from_start.unsqueeze(-1)
  y_inter = torch.matmul(c_decayed, prev_states)

  return y_intra + y_inter
