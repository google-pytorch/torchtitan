# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Count-bounded Pallas SwiGLU (and optional per-row scale) for MoE experts.

Under static-shape expert parallelism (Ring-of-Experts or ragged all-to-all),
each rank's expert buffer ``[R, F]`` is sized for worst-case or capacity-bounded
routing, while only the leading ``count = sum(num_tokens_per_expert_E)`` rows
contain valid routed tokens. Running ``silu(gate) * up * row_scale`` eagerly
over all ``R`` rows wastes HBM bandwidth and VPU cycles on the unused tail
(e.g. ``7/8`` of ``R`` at ``EP = 8`` dropless). This kernel executes over a
dynamic Pallas grid ``(cdiv(aligned_count, block), F // f_chunk)`` and zeroes
any padding rows inside the boundary ``row_align`` tile so downstream grouped
matmuls never observe uninitialized HBM.
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from torch_tpu._internal import pallas
from torchtitan.experiments.tpu.kernels import pallas_vmem


def _choose_swiglu_tiling(
    num_rows: int,
    f_dim: int,
    itemsize: int,
    vmem_limit: int,
    *,
    is_bwd: bool,
) -> tuple[int, int] | None:
  """Selects ``(row_block, f_chunk)`` fitting within ``0.75 * vmem_limit``."""
  if f_dim < 128 or f_dim % 128 != 0:
    return None
  row_block = None
  for candidate in (1024, 512, 256, 128):
    if num_rows >= candidate and num_rows % candidate == 0:
      row_block = candidate
      break
  if row_block is None:
    return None

  budget = int(0.75 * vmem_limit)
  bytes_per_elem = (2 * 5 * itemsize + 16) if is_bwd else (2 * 3 * itemsize + 8)
  for num_chunks in range(1, f_dim // 128 + 1):
    if f_dim % num_chunks != 0 or (f_dim // num_chunks) % 128 != 0:
      continue
    f_chunk = f_dim // num_chunks
    if row_block * f_chunk * bytes_per_elem <= budget:
      return row_block, f_chunk
  return None


def _fast_sigmoid(x: jax.Array) -> jax.Array:
  """Hardware-tanh sigmoid matching XLA HLO ``logistic``."""
  return 0.5 + 0.5 * jnp.tanh(0.5 * x)


def _swiglu_scaled_fwd_jax(
    gate_RF: jax.Array,
    up_RF: jax.Array,
    row_scale_R: jax.Array,
    group_sizes_E: jax.Array,
    row_align: int,
) -> jax.Array:
  num_rows, f_dim = gate_RF.shape
  vmem_limit = pallas_vmem.vmem_limit_bytes()
  tiling = _choose_swiglu_tiling(
      num_rows,
      f_dim,
      jnp.dtype(gate_RF.dtype).itemsize,
      vmem_limit,
      is_bwd=False,
  )
  raw_count = jnp.sum(group_sizes_E, dtype=jnp.int32)
  count = jnp.clip(raw_count, 0, num_rows)
  if tiling is None:
    valid = (jnp.arange(num_rows, dtype=jnp.int32) < count)[:, None]
    g = jnp.where(valid, gate_RF.astype(jnp.float32), 0.0)
    u = jnp.where(valid, up_RF.astype(jnp.float32), 0.0)
    act = (g * _fast_sigmoid(g)).astype(gate_RF.dtype).astype(jnp.float32)
    unscaled = (act * u).astype(gate_RF.dtype).astype(jnp.float32)
    w = jnp.where(valid, row_scale_R.astype(jnp.float32)[:, None], 0.0)
    return (unscaled * w).astype(gate_RF.dtype)

  block, f_chunk = tiling
  aligned = jnp.minimum(pl.cdiv(count, row_align) * row_align, num_rows)
  scalars_2 = jnp.stack([count, aligned])
  w_col_in = row_scale_R.astype(jnp.float32).reshape(num_rows, 1)

  def _kernel(s_ref, g_ref, u_ref, w_ref, o_ref):
    start_row = pl.program_id(0) * block
    n_valid = s_ref[0]

    @pl.when(start_row + block <= n_valid)
    def _full():
      g = g_ref[...].astype(jnp.float32)
      u = u_ref[...].astype(jnp.float32)
      act = (g * _fast_sigmoid(g)).astype(o_ref.dtype).astype(jnp.float32)
      unscaled = (act * u).astype(o_ref.dtype).astype(jnp.float32)
      w_col = jnp.broadcast_to(w_ref[...], g_ref.shape)
      o_ref[...] = (unscaled * w_col).astype(o_ref.dtype)

    @pl.when(start_row + block > n_valid)
    def _tail():
      @pl.when(start_row < n_valid)
      def _partial():
        rows_2d = start_row + jax.lax.broadcasted_iota(
            jnp.int32, g_ref.shape, 0
        )
        valid_2d = rows_2d < n_valid
        g = jnp.where(valid_2d, g_ref[...].astype(jnp.float32), 0.0)
        u = jnp.where(valid_2d, u_ref[...].astype(jnp.float32), 0.0)
        act = jnp.where(
            valid_2d,
            (g * _fast_sigmoid(g)).astype(o_ref.dtype).astype(jnp.float32),
            0.0,
        )
        unscaled = (act * u).astype(o_ref.dtype).astype(jnp.float32)
        w_col = jnp.broadcast_to(w_ref[...], g_ref.shape)
        w_m = jnp.where(valid_2d, w_col, 0.0)
        o_ref[...] = (unscaled * w_m).astype(o_ref.dtype)

      @pl.when(start_row >= n_valid)
      def _zero():
        o_ref[...] = jnp.zeros_like(o_ref)

  return pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct((num_rows, f_dim), gate_RF.dtype),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(pl.cdiv(scalars_2[1], block), f_dim // f_chunk),
          in_specs=[
              pl.BlockSpec((block, f_chunk), lambda i, j, s_ref: (i, j)),
              pl.BlockSpec((block, f_chunk), lambda i, j, s_ref: (i, j)),
              pl.BlockSpec((block, 1), lambda i, j, s_ref: (i, 0)),
          ],
          out_specs=pl.BlockSpec((block, f_chunk), lambda i, j, s_ref: (i, j)),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel"),
          vmem_limit_bytes=vmem_limit,
      ),
      name="expert_swiglu_scaled_fwd",
  )(scalars_2, gate_RF, up_RF, w_col_in)


def _swiglu_scaled_bwd_jax(
    grad_hidden_RF: jax.Array,
    gate_RF: jax.Array,
    up_RF: jax.Array,
    row_scale_R: jax.Array,
    group_sizes_E: jax.Array,
    row_align: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  num_rows, f_dim = gate_RF.shape
  vmem_limit = pallas_vmem.vmem_limit_bytes()
  tiling = _choose_swiglu_tiling(
      num_rows,
      f_dim,
      jnp.dtype(gate_RF.dtype).itemsize,
      vmem_limit,
      is_bwd=True,
  )
  raw_count = jnp.sum(group_sizes_E, dtype=jnp.int32)
  count = jnp.clip(raw_count, 0, num_rows)
  if tiling is None:
    valid_1d = jnp.arange(num_rows, dtype=jnp.int32) < count
    valid = valid_1d[:, None]
    dy = jnp.where(valid, grad_hidden_RF.astype(jnp.float32), 0.0)
    g = jnp.where(valid, gate_RF.astype(jnp.float32), 0.0)
    u = jnp.where(valid, up_RF.astype(jnp.float32), 0.0)
    sig = _fast_sigmoid(g)
    act = jnp.where(
        valid, (g * sig).astype(gate_RF.dtype).astype(jnp.float32), 0.0
    )
    unscaled = (act * u).astype(gate_RF.dtype).astype(jnp.float32)
    dw = jnp.sum(dy * unscaled, axis=1)
    w = jnp.where(valid, row_scale_R.astype(jnp.float32)[:, None], 0.0)
    d_unscaled = (dy * w).astype(gate_RF.dtype).astype(jnp.float32)
    du = (d_unscaled * act).astype(up_RF.dtype)
    d_act = (d_unscaled * u).astype(gate_RF.dtype).astype(jnp.float32)
    dsilu = sig * (1.0 + g * (1.0 - sig))
    dg = jnp.where(valid, (d_act * dsilu).astype(gate_RF.dtype), 0)
    return dg, du, dw

  block, f_chunk = tiling
  aligned = jnp.minimum(pl.cdiv(count, row_align) * row_align, num_rows)
  scalars_2 = jnp.stack([count, aligned])
  w_col_in = row_scale_R.astype(jnp.float32).reshape(num_rows, 1)
  zero_dw = jnp.zeros((num_rows, 1), dtype=jnp.float32)

  def _kernel(
      s_ref, dy_ref, g_ref, u_ref, w_ref, z_dw_ref, dg_ref, du_ref, dw_ref
  ):
    del z_dw_ref
    start_row = pl.program_id(0) * block
    n_valid = s_ref[0]

    @pl.when(pl.program_id(1) == 0)
    def _init_dw():
      dw_ref[...] = jnp.zeros_like(dw_ref)

    @pl.when(start_row + block <= n_valid)
    def _full():
      dy_f = dy_ref[...].astype(jnp.float32)
      g_f = g_ref[...].astype(jnp.float32)
      u_f = u_ref[...].astype(jnp.float32)
      sig = _fast_sigmoid(g_f)
      act = (g_f * sig).astype(dg_ref.dtype).astype(jnp.float32)
      unscaled = (act * u_f).astype(dg_ref.dtype).astype(jnp.float32)
      dw_ref[...] += jnp.sum(dy_f * unscaled, axis=1, keepdims=True)
      w_col = jnp.broadcast_to(w_ref[...], dy_ref.shape)
      d_unscaled = (dy_f * w_col).astype(dg_ref.dtype).astype(jnp.float32)
      du_ref[...] = (d_unscaled * act).astype(du_ref.dtype)
      d_act = (d_unscaled * u_f).astype(dg_ref.dtype).astype(jnp.float32)
      dsilu = sig * (1.0 + g_f * (1.0 - sig))
      dg_ref[...] = (d_act * dsilu).astype(dg_ref.dtype)

    @pl.when(start_row + block > n_valid)
    def _tail():
      @pl.when(start_row < n_valid)
      def _partial():
        rows_2d = start_row + jax.lax.broadcasted_iota(
            jnp.int32, dy_ref.shape, 0
        )
        valid_2d = rows_2d < n_valid
        dy_f = jnp.where(valid_2d, dy_ref[...].astype(jnp.float32), 0.0)
        g_f = jnp.where(valid_2d, g_ref[...].astype(jnp.float32), 0.0)
        u_f = jnp.where(valid_2d, u_ref[...].astype(jnp.float32), 0.0)
        sig = _fast_sigmoid(g_f)
        act = jnp.where(
            valid_2d,
            (g_f * sig).astype(dg_ref.dtype).astype(jnp.float32),
            0.0,
        )
        unscaled = (act * u_f).astype(dg_ref.dtype).astype(jnp.float32)
        dw_ref[...] += jnp.sum(dy_f * unscaled, axis=1, keepdims=True)
        w_col = jnp.broadcast_to(w_ref[...], dy_ref.shape)
        w_m = jnp.where(valid_2d, w_col, 0.0)
        d_unscaled = (dy_f * w_m).astype(dg_ref.dtype).astype(jnp.float32)
        du_ref[...] = (d_unscaled * act).astype(du_ref.dtype)
        d_act = (d_unscaled * u_f).astype(dg_ref.dtype).astype(jnp.float32)
        dsilu = sig * (1.0 + g_f * (1.0 - sig))
        dg_ref[...] = jnp.where(
            valid_2d,
            (d_act * dsilu).astype(dg_ref.dtype),
            jnp.zeros((), dg_ref.dtype),
        )

      @pl.when(start_row >= n_valid)
      def _zero():
        dg_ref[...] = jnp.zeros_like(dg_ref)
        du_ref[...] = jnp.zeros_like(du_ref)

  dg, du, dw_col = pl.pallas_call(
      _kernel,
      out_shape=[
          jax.ShapeDtypeStruct((num_rows, f_dim), gate_RF.dtype),
          jax.ShapeDtypeStruct((num_rows, f_dim), up_RF.dtype),
          jax.ShapeDtypeStruct((num_rows, 1), jnp.float32),
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(pl.cdiv(scalars_2[1], block), f_dim // f_chunk),
          in_specs=[
              pl.BlockSpec((block, f_chunk), lambda i, j, s_ref: (i, j)),
              pl.BlockSpec((block, f_chunk), lambda i, j, s_ref: (i, j)),
              pl.BlockSpec((block, f_chunk), lambda i, j, s_ref: (i, j)),
              pl.BlockSpec((block, 1), lambda i, j, s_ref: (i, 0)),
              pl.BlockSpec((block, 1), lambda i, j, s_ref: (i, 0)),
          ],
          out_specs=[
              pl.BlockSpec((block, f_chunk), lambda i, j, s_ref: (i, j)),
              pl.BlockSpec((block, f_chunk), lambda i, j, s_ref: (i, j)),
              pl.BlockSpec((block, 1), lambda i, j, s_ref: (i, 0)),
          ],
      ),
      input_output_aliases={5: 2},
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          vmem_limit_bytes=vmem_limit,
      ),
      name="expert_swiglu_scaled_bwd",
  )(scalars_2, grad_hidden_RF, gate_RF, up_RF, w_col_in, zero_dw)
  return dg, du, dw_col.reshape(num_rows)


@pallas.jax_op("tt_moe::expert_swiglu_scaled_fwd")
def _expert_swiglu_scaled_fwd(
    gate_RF: jax.Array,
    up_RF: jax.Array,
    row_scale_R: jax.Array,
    group_sizes_E: jax.Array,
    row_align: int,
) -> jax.Array:
  return _swiglu_scaled_fwd_jax(
      gate_RF, up_RF, row_scale_R, group_sizes_E, row_align
  )


@pallas.jax_op("tt_moe::expert_swiglu_scaled_bwd")
def _expert_swiglu_scaled_bwd(
    grad_hidden_RF: jax.Array,
    gate_RF: jax.Array,
    up_RF: jax.Array,
    row_scale_R: jax.Array,
    group_sizes_E: jax.Array,
    row_align: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  return _swiglu_scaled_bwd_jax(
      grad_hidden_RF, gate_RF, up_RF, row_scale_R, group_sizes_E, row_align
  )


def _scaled_setup_context(ctx, inputs, output):
  del output
  gate_RF, up_RF, row_scale_R, group_sizes_E, row_align = inputs
  ctx.save_for_backward(gate_RF, up_RF, row_scale_R, group_sizes_E)
  ctx.row_align = row_align
  ctx.scale_dtype = row_scale_R.dtype


def _scaled_backward(ctx, grad_hidden_RF):
  gate_RF, up_RF, row_scale_R, group_sizes_E = ctx.saved_tensors
  dg, du, dw = _expert_swiglu_scaled_bwd(
      grad_hidden_RF.contiguous(),
      gate_RF,
      up_RF,
      row_scale_R,
      group_sizes_E,
      ctx.row_align,
  )
  if dw.dtype != ctx.scale_dtype:
    dw = dw.to(ctx.scale_dtype)
  return dg, du, dw, None, None


_expert_swiglu_scaled_fwd.register_autograd(
    _scaled_backward, setup_context=_scaled_setup_context
)


def expert_swiglu(
    gate_RF: torch.Tensor,
    up_RF: torch.Tensor,
    num_tokens_per_expert_E: torch.Tensor,
    *,
    row_scale_R: torch.Tensor | None = None,
    row_align: int = 1024,
) -> torch.Tensor:
  """Computes ``silu(gate_RF) * up_RF`` (times ``row_scale_R``) over valid rows."""
  num_rows, f_dim = gate_RF.shape
  if (
      gate_RF.device.type == "tpu"
      and gate_RF.dtype == torch.bfloat16
      and up_RF.dtype == torch.bfloat16
      and row_scale_R is not None
      and num_rows >= 128
      and num_rows % 128 == 0
      and f_dim >= 128
      and f_dim % 128 == 0
  ):
    gs_i32 = (
        num_tokens_per_expert_E
        if num_tokens_per_expert_E.dtype == torch.int32
        else num_tokens_per_expert_E.to(torch.int32)
    )
    scale_f32 = (
        row_scale_R
        if row_scale_R.dtype == torch.float32
        else row_scale_R.float()
    )
    return _expert_swiglu_scaled_fwd(
        gate_RF.contiguous(),
        up_RF.contiguous(),
        scale_f32.contiguous(),
        gs_i32.contiguous(),
        row_align,
    )

  hidden_RF = F.silu(gate_RF) * up_RF
  if row_scale_R is not None:
    hidden_RF = (hidden_RF.float() * row_scale_R.reshape(-1, 1)).to(
        hidden_RF.dtype
    )
  return hidden_RF


__all__ = ["expert_swiglu"]
