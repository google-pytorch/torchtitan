# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conv1D + SiLU forward and unrolled shift-multiply backward primitives."""

import contextlib
import contextvars
import math
import os
from typing import Any, Iterator, Optional, Tuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from .runtime_utils import target_platform

# gdn_conv1d_input_in_qkv_dtype (see the comment above _conv1d_silu_fwd_narrow_input at the end of this file).
# conv1d_silu_fwd / conv1d_silu_bwd take it as an argument; left at None they read this trace-time setting, which
# gdn_decoupled_conv1d (gdn_bwd/api.py) scopes around its primal, forward and backward rules from its own static
# argument. A scoped setting rather than an argument on every internal helper because the backward rule is traced
# when the cotangent is pulled back, after the model's forward has returned, so only the custom_vjp's static
# arguments reach it; and the rules reach these functions through several helpers (jax_compute_gdn_states, the CP
# forward) that do not otherwise need it.
_CONV_INPUT_IN_QKV_DTYPE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "gdn_conv1d_input_in_qkv_dtype", default=False
)


@contextlib.contextmanager
def conv_input_in_qkv_dtype_scope(enabled: bool) -> Iterator[None]:
    """Scopes the default of conv1d_silu_fwd / conv1d_silu_bwd's `conv_input_in_qkv_dtype` while tracing."""
    token = _CONV_INPUT_IN_QKV_DTYPE.set(bool(enabled))
    try:
        yield
    finally:
        _CONV_INPUT_IN_QKV_DTYPE.reset(token)


def _resolve_conv_input_in_qkv_dtype(
    conv_input_in_qkv_dtype: Optional[bool],
) -> bool:
    if conv_input_in_qkv_dtype is None:
        return _CONV_INPUT_IN_QKV_DTYPE.get()
    return bool(conv_input_in_qkv_dtype)


def _build_conv_tap_masks(
    segment_ids: jax.Array,
    kernel_size: int,
    conv_halo_seg: Optional[jax.Array] = None,
) -> Tuple[jax.Array, list[jax.Array]]:
    """Builds (valid_mask, tap_masks) for causal Conv1D with sequence packing."""
    _, seq_len = segment_ids.shape
    halo_len = kernel_size - 1
    seg_pos = jnp.maximum(segment_ids.astype(jnp.int32), 0)
    valid_mask = (seg_pos > 0)[:, :, None].astype(jnp.float32)

    if conv_halo_seg is not None:
        halo_pos = jnp.maximum(
            conv_halo_seg.astype(jnp.int32).reshape(seg_pos.shape[0], halo_len), 0
        )
        full_seg = jnp.concatenate([halo_pos, seg_pos], axis=1)
    else:
        full_seg = jnp.pad(seg_pos, ((0, 0), (halo_len, 0)))

    tap_masks = [
        ((seg_pos > 0) & (full_seg[:, k : k + seq_len] == seg_pos))[:, :, None].astype(
            jnp.float32
        )
        for k in range(kernel_size)
    ]
    return valid_mask, tap_masks


def conv1d_silu_fwd(
    qkv: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    kernel_size: int,
    conv_state: Optional[jax.Array] = None,
    segment_ids: Optional[jax.Array] = None,
    conv_halo_seg: Optional[jax.Array] = None,
    output_dtype: Optional[jax.typing.DTypeLike] = None,
    conv_input_in_qkv_dtype: Optional[bool] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Forward Conv1D + SiLU returning (conv_out, qkv_conv).

    `conv_input_in_qkv_dtype` (None: the conv_input_in_qkv_dtype_scope() setting,
    default False) keeps the halo-extended input
    in qkv's dtype and casts each tap to float32 where it is read. Same values;
    see _conv1d_silu_fwd_narrow_input.
    """
    if _resolve_conv_input_in_qkv_dtype(conv_input_in_qkv_dtype):
        return _conv1d_silu_fwd_narrow_input(
            qkv,
            conv_weight,
            conv_bias,
            kernel_size,
            conv_state,
            segment_ids,
            conv_halo_seg,
            output_dtype,
        )
    _, seq_len, _ = qkv.shape
    if conv_weight.ndim == 3:
        conv_weight_3d = conv_weight.astype(jnp.float32)
    else:
        conv_weight_3d = conv_weight[:, None, :].astype(jnp.float32)

    if conv_state is not None:
        conv_input = jnp.concatenate(
            [conv_state.astype(jnp.float32), qkv.astype(jnp.float32)], axis=1
        )
    else:
        conv_input = jnp.pad(
            qkv.astype(jnp.float32), ((0, 0), (kernel_size - 1, 0), (0, 0))
        )
    if segment_ids is not None:
        valid_mask, tap_masks = _build_conv_tap_masks(
            segment_ids, kernel_size, conv_halo_seg
        )
        conv_out = sum(
            (
                (conv_input[:, k : k + seq_len, :] * tap_masks[k])
                * conv_weight_3d[k, 0, :]
                for k in range(kernel_size)
            ),
            jnp.zeros(qkv.shape, dtype=jnp.float32),
        )
        if conv_bias is not None:
            conv_out = conv_out + conv_bias.astype(jnp.float32)
        conv_out = conv_out * valid_mask
    else:
        conv_out = sum(
            (
                conv_input[:, k : k + seq_len, :] * conv_weight_3d[k, 0, :]
                for k in range(kernel_size)
            ),
            jnp.zeros(qkv.shape, dtype=jnp.float32),
        )
        if conv_bias is not None:
            conv_out = conv_out + conv_bias.astype(jnp.float32)
    qkv_conv = jax.nn.silu(conv_out)
    return conv_out, qkv_conv.astype(output_dtype or qkv.dtype)


def conv1d_silu_bwd(
    qkv: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    dy: jax.Array,
    kernel_size: int,
    conv_out: Optional[jax.Array] = None,
    conv_state: Optional[jax.Array] = None,
    return_d_conv_state: bool = False,
    segment_ids: Optional[jax.Array] = None,
    conv_halo_seg: Optional[jax.Array] = None,
    conv_input_in_qkv_dtype: Optional[bool] = None,
):
    """Dedicated Conv1D + SiLU backward pass using JAX primitives.

    `conv_input_in_qkv_dtype` (None: the conv_input_in_qkv_dtype_scope() setting,
    default False) keeps the halo-extended input
    in qkv's dtype and casts each tap to float32 where it is read. Same values;
    see _conv1d_silu_bwd_narrow_input.
    """
    if _resolve_conv_input_in_qkv_dtype(conv_input_in_qkv_dtype):
        return _conv1d_silu_bwd_narrow_input(
            qkv,
            conv_weight,
            conv_bias,
            dy,
            kernel_size,
            conv_out,
            conv_state,
            return_d_conv_state,
            segment_ids,
            conv_halo_seg,
        )
    _, seq_len, _ = qkv.shape
    if conv_weight.ndim == 3:
        conv_weight_3d = conv_weight.astype(jnp.float32)
    else:
        conv_weight_3d = conv_weight[:, None, :].astype(jnp.float32)

    if conv_state is not None:
        conv_input = jnp.concatenate(
            [conv_state.astype(jnp.float32), qkv.astype(jnp.float32)], axis=1
        )
    else:
        conv_input = jnp.pad(
            qkv.astype(jnp.float32), ((0, 0), (kernel_size - 1, 0), (0, 0))
        )

    valid_mask = None
    tap_masks = None
    if segment_ids is not None:
        valid_mask, tap_masks = _build_conv_tap_masks(
            segment_ids, kernel_size, conv_halo_seg
        )

    if conv_out is None:
        # 1. Forward pass: z = conv1d(x) + b
        if segment_ids is not None and tap_masks is not None and valid_mask is not None:
            computed_out = sum(
                (
                    (conv_input[:, k : k + seq_len, :] * tap_masks[k])
                    * conv_weight_3d[k, 0, :]
                    for k in range(kernel_size)
                ),
                jnp.zeros(qkv.shape, dtype=jnp.float32),
            )
            if conv_bias is not None:
                computed_out = computed_out + conv_bias.astype(jnp.float32)
            computed_out = computed_out * valid_mask
        else:
            computed_out = sum(
                (
                    conv_input[:, k : k + seq_len, :] * conv_weight_3d[k, 0, :]
                    for k in range(kernel_size)
                ),
                jnp.zeros(qkv.shape, dtype=jnp.float32),
            )
            if conv_bias is not None:
                computed_out = computed_out + conv_bias.astype(jnp.float32)
        z = computed_out.astype(jnp.float32)
    else:
        z = conv_out.astype(jnp.float32)

    # 2. Adjoint: dz = dy * SiLU'(z)
    sig_z = jax.nn.sigmoid(z)
    silu_prime = sig_z * (1.0 + z * (1.0 - sig_z))
    dz = dy.astype(jnp.float32) * silu_prime
    if valid_mask is not None:
        dz = dz * valid_mask

    # 3. Parameter gradients:
    # Bias gradient: db = sum(dz)
    if conv_bias is not None:
        db = jnp.sum(dz, axis=(0, 1)).astype(conv_bias.dtype)
        if conv_bias.ndim != 1:
            db = db.reshape(conv_bias.shape)
    else:
        db = None

    # Weight gradient: dw[k] = sum_{b,t} dz[b,t] * conv_input[b, t+k]
    dw_rows = []
    for k in range(kernel_size):
        x_k = conv_input[:, k : k + seq_len, :]
        dz_k = dz * tap_masks[k] if tap_masks is not None else dz
        dw_rows.append(jnp.sum(dz_k * x_k, axis=(0, 1)))
    dw = jnp.stack(dw_rows, axis=0)
    if conv_weight.ndim == 3:
        dw = dw[:, None, :].astype(conv_weight.dtype)
    else:
        dw = dw.astype(conv_weight.dtype)

    # 4. Input gradient: dx = transposed convolution of dz with reversed w
    dz_pad = jnp.pad(dz, ((0, 0), (0, kernel_size - 1), (0, 0)))
    w_rev = conv_weight_3d[::-1]
    if tap_masks is not None:
        tap_masks_pad = [
            jnp.pad(tm, ((0, 0), (0, kernel_size - 1), (0, 0))) for tm in tap_masks
        ]
        dx = sum(
            (
                (
                    dz_pad[:, j : j + seq_len, :]
                    * tap_masks_pad[kernel_size - 1 - j][:, j : j + seq_len, :]
                )
                * w_rev[j, 0, :]
                for j in range(kernel_size)
            ),
            jnp.zeros(qkv.shape, dtype=jnp.float32),
        )
    else:
        dx = sum(
            (
                dz_pad[:, k : k + seq_len, :] * w_rev[k, 0, :]
                for k in range(kernel_size)
            ),
            jnp.zeros(qkv.shape, dtype=jnp.float32),
        )
    dx = dx.astype(qkv.dtype)

    if return_d_conv_state:
        d_cs_list = []
        for m in range(kernel_size - 1):
            val = jnp.zeros((qkv.shape[0], qkv.shape[2]), dtype=jnp.float32)
            for t in range(m + 1):
                k_idx = m - t
                dz_t = (
                    dz[:, t, :] * tap_masks[k_idx][:, t, :]
                    if tap_masks is not None
                    else dz[:, t, :]
                )
                val = val + dz_t * conv_weight_3d[k_idx, 0, :]
            d_cs_list.append(val)
        d_conv_state = jnp.stack(d_cs_list, axis=1).astype(qkv.dtype)
        return dx, dw, db, d_conv_state

    return dx, dw, db


# --------------------------------------------------------------------------------------------------------------------
# gdn_conv1d_input_in_qkv_dtype: the halo-extended conv input in qkv's dtype.
#
# The default path above builds conv_input = [conv_state | qkv] (or qkv left-padded by kernel_size - 1 zeros) in
# float32 and slices the kernel_size taps out of it. That materializes a float32 copy of the whole input, one buffer
# of batch x (sequence + kernel_size - 1) x channels per GDN layer in the backward pass, written once and read
# kernel_size times.
#
# The functions below build conv_input in qkv's dtype (bf16 for bf16 qkv and conv_state; the promoted dtype if they
# differ) and cast each tap slice to float32 where it is read, so XLA fuses the cast into the consumer and the
# float32 copy is never written. Every product and sum is formed in the same order, on the same float32 values, as in
# the default path (bf16 -> float32 is exact), so op by op every output is bit-identical; only which tensors are
# materialized in HBM changes. Compiled (measured on CPU), the forward outputs and dx / d_conv_state / db stay
# bit-identical, while the weight gradient dw, a float32 reduction over batch x sequence, can come out in a different
# summation order when XLA fuses the tap's cast into the reduction (CPU: a few ulp, only when the pre-activation is
# recomputed without packing or a conv state). No jax.lax.optimization_barrier is used: the narrow buffer alone is
# what removes the copy.
# Tests: tests/unit/gdn_conv1d_input_in_qkv_dtype_test.py.


def _conv_input_narrow(qkv, conv_state, kernel_size):
    """Halo-extended conv input in the narrowest exact dtype (bf16 for bf16 qkv and conv_state)."""
    if conv_state is not None:
        cdt = jnp.promote_types(qkv.dtype, conv_state.dtype)
        return jnp.concatenate([conv_state.astype(cdt), qkv.astype(cdt)], axis=1)
    return jnp.pad(qkv, ((0, 0), (kernel_size - 1, 0), (0, 0)))


def _conv_weight_3d_f32(conv_weight):
    return (
        conv_weight.astype(jnp.float32)
        if conv_weight.ndim == 3
        else conv_weight[:, None, :].astype(jnp.float32)
    )


def _preact_from_narrow_input(
    conv_input, w3d, conv_bias, kernel_size, seq_len, tap_masks, valid_mask
):
    """f32 pre-activation z with the default path's op order: sum over taps k of (x_k * mask_k) * w_k, + b, * valid."""
    if tap_masks is not None:
        z = sum(
            (conv_input[:, k : k + seq_len, :].astype(jnp.float32) * tap_masks[k])
            * w3d[k, 0, :]
            for k in range(kernel_size)
        )
        if conv_bias is not None:
            z = z + conv_bias.astype(jnp.float32)
        return z * valid_mask
    z = sum(
        conv_input[:, k : k + seq_len, :].astype(jnp.float32) * w3d[k, 0, :]
        for k in range(kernel_size)
    )
    if conv_bias is not None:
        z = z + conv_bias.astype(jnp.float32)
    return z


def _conv1d_silu_fwd_narrow_input(
    qkv,
    conv_weight,
    conv_bias,
    kernel_size,
    conv_state,
    segment_ids,
    conv_halo_seg,
    output_dtype,
):
    """conv1d_silu_fwd reading the conv input in qkv's dtype.

    Returns (f32 conv_out, qkv_conv) like the default path.
    """
    _, seq_len, _ = qkv.shape
    w3d = _conv_weight_3d_f32(conv_weight)
    conv_input = _conv_input_narrow(qkv, conv_state, kernel_size)
    valid_mask = tap_masks = None
    if segment_ids is not None:
        valid_mask, tap_masks = _build_conv_tap_masks(
            segment_ids, kernel_size, conv_halo_seg
        )
    conv_out = _preact_from_narrow_input(
        conv_input, w3d, conv_bias, kernel_size, seq_len, tap_masks, valid_mask
    )
    qkv_conv = jax.nn.silu(conv_out)
    return conv_out, qkv_conv.astype(output_dtype or qkv.dtype)


def _conv1d_silu_bwd_narrow_input(
    qkv,
    conv_weight,
    conv_bias,
    dy,
    kernel_size,
    conv_out,
    conv_state,
    return_d_conv_state,
    segment_ids,
    conv_halo_seg,
):
    """conv1d_silu_bwd reading the conv input in qkv's dtype; the same steps 1-4 as the default path."""
    _, seq_len, _ = qkv.shape
    w3d = _conv_weight_3d_f32(conv_weight)
    conv_input = _conv_input_narrow(qkv, conv_state, kernel_size)
    valid_mask = tap_masks = None
    if segment_ids is not None:
        valid_mask, tap_masks = _build_conv_tap_masks(
            segment_ids, kernel_size, conv_halo_seg
        )
    if conv_out is None:
        conv_out = _preact_from_narrow_input(
            conv_input, w3d, conv_bias, kernel_size, seq_len, tap_masks, valid_mask
        )
    z = conv_out.astype(jnp.float32)

    # dz = dy * SiLU'(z), as in the default path.
    sig_z = jax.nn.sigmoid(z)
    silu_prime = sig_z * (1.0 + z * (1.0 - sig_z))
    dz = dy.astype(jnp.float32) * silu_prime
    if valid_mask is not None:
        dz = dz * valid_mask

    if conv_bias is not None:
        db = jnp.sum(dz, axis=(0, 1)).astype(conv_bias.dtype)
        if conv_bias.ndim != 1:
            db = db.reshape(conv_bias.shape)
    else:
        db = None

    # dw[k] = sum_{b,t} dz[b,t] * conv_input[b, t+k], each tap cast to float32 where it is read.
    dw_rows = []
    for k in range(kernel_size):
        x_k = conv_input[:, k : k + seq_len, :].astype(jnp.float32)
        dz_k = dz * tap_masks[k] if tap_masks is not None else dz
        dw_rows.append(jnp.sum(dz_k * x_k, axis=(0, 1)))
    dw = jnp.stack(dw_rows, axis=0)
    dw = (
        dw[:, None, :].astype(conv_weight.dtype)
        if conv_weight.ndim == 3
        else dw.astype(conv_weight.dtype)
    )

    # dx: transposed convolution of dz with the reversed weights (does not read conv_input).
    dz_pad = jnp.pad(dz, ((0, 0), (0, kernel_size - 1), (0, 0)))
    w_rev = w3d[::-1]
    if tap_masks is not None:
        tap_masks_pad = [
            jnp.pad(tm, ((0, 0), (0, kernel_size - 1), (0, 0))) for tm in tap_masks
        ]
        dx = sum(
            (
                (
                    dz_pad[:, j : j + seq_len, :]
                    * tap_masks_pad[kernel_size - 1 - j][:, j : j + seq_len, :]
                )
                * w_rev[j, 0, :]
                for j in range(kernel_size)
            ),
            jnp.zeros(qkv.shape, dtype=jnp.float32),
        )
    else:
        dx = sum(
            (
                dz_pad[:, k : k + seq_len, :] * w_rev[k, 0, :]
                for k in range(kernel_size)
            ),
            jnp.zeros(qkv.shape, dtype=jnp.float32),
        )
    dx = dx.astype(qkv.dtype)

    if return_d_conv_state:
        d_cs_list = []
        for m in range(kernel_size - 1):
            val = jnp.zeros((qkv.shape[0], qkv.shape[2]), dtype=jnp.float32)
            for t in range(m + 1):
                k_idx = m - t
                dz_t = (
                    dz[:, t, :] * tap_masks[k_idx][:, t, :]
                    if tap_masks is not None
                    else dz[:, t, :]
                )
                val = val + dz_t * w3d[k_idx, 0, :]
            d_cs_list.append(val)
        d_conv_state = jnp.stack(d_cs_list, axis=1).astype(qkv.dtype)
        return dx, dw, db, d_conv_state
    return dx, dw, db


# --------------------------------------------------------------------------------------------------------------------
# Zero-Halo Wide-Channel 2D VMEM-Fused Pallas Conv1D + SiLU Forward & Backward (`kernel_size=4`)
# --------------------------------------------------------------------------------------------------------------------

_CONV_MIN_BLOCK_C = 128
_CONV_MAX_PALLAS_SEQ_LEN = 16384
_CONV_VMEM_BUDGET_BYTES = 32 * 1024 * 1024
_CONV_FWD_VREG_ELEMS = 65536
_CONV_BWD_VREG_ELEMS = 65536


def _conv_manual_axis_type(x: jax.Array) -> Any:
    return getattr(jax.typeof(x), "manual_axis_type", None)


def _can_use_conv1d_silu_pallas(
    qkv: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    kernel_size: int,
    conv_state: Optional[jax.Array],
    interpret: bool | pltpu.InterpretParams,
) -> bool:
    """Returns True when wide-channel 2D Pallas `conv1d_silu` is supported."""
    if kernel_size != 4 or qkv.ndim != 3:
        return False
    batch, seq_len, channels = qkv.shape
    max_pallas_seq = (
        131072
        if os.environ.get("TT_GDN_PALLAS_CONV_64K", "0") == "1"
        and _choose_conv_seq_tile(seq_len, channels) is not None
        else _CONV_MAX_PALLAS_SEQ_LEN
    )
    if (
        batch < 1
        or seq_len < 8
        or seq_len % 8 != 0
        or seq_len > max_pallas_seq
        or channels < _CONV_MIN_BLOCK_C
        or channels % _CONV_MIN_BLOCK_C != 0
    ):
        return False
    if conv_weight.size != 4 * channels:
        return False
    if conv_bias is not None and conv_bias.size != channels:
        return False
    if conv_state is not None and conv_state.shape != (batch, 3, channels):
        return False
    if not bool(interpret) and target_platform(qkv) == "cpu":
        return False
    return True


def _choose_conv_block_c(
    seq_len: int,
    channels: int,
    io_bytes_per_elem: int,
) -> int:
    """Picks wide channel tile (`512` / `256` / `128`) fitting double-buffered VMEM."""
    max_c = max(
        _CONV_MIN_BLOCK_C,
        _CONV_VMEM_BUDGET_BYTES // max(1, 2 * seq_len * io_bytes_per_elem),
    )
    if max_c >= 512:
        target_c = 512
    elif max_c >= 256:
        target_c = 256
    else:
        target_c = 128
    return math.gcd(channels, target_c)


def _build_pallas_seg_buf(
    segment_ids: jax.Array,
    conv_halo_seg: Optional[jax.Array],
) -> jax.Array:
    """Builds compact `(B, S + 16, 128)` `int32` `[5 zeros, halo(3), seg_pos(S), 8 zeros]`."""
    batch, seq_len = segment_ids.shape
    seg_pos = jnp.maximum(segment_ids.astype(jnp.int32), 0)
    z5 = jnp.zeros((batch, 5), dtype=jnp.int32)
    z8 = jnp.zeros((batch, 8), dtype=jnp.int32)
    if conv_halo_seg is not None:
        halo_pos = jnp.maximum(conv_halo_seg.astype(jnp.int32).reshape(batch, 3), 0)
        seg_1d = jnp.concatenate([z5, halo_pos, seg_pos, z8], axis=1)
    else:
        seg_1d = jnp.pad(seg_pos, ((0, 0), (8, 8)))
    return jnp.broadcast_to(
        seg_1d[:, :, None], (batch, seq_len + 16, _CONV_MIN_BLOCK_C)
    )


# Wave 14: sequence-tiled conv1d_silu kernels. The full-sequence kernels above
# stream `[S, block_c]` column strips (512 B - 1 KiB strided row bursts); these
# grid over `(B, S // block_s, C // block_c)` with wide contiguous rows and
# 8-row halo blocks (radius-3 taps), so DMAs are multi-KiB contiguous bursts.
# Ghostfish (B=1, C=20480, segments): bwd 8K 1.809 -> 1.544 ms, 4K 0.858 ->
# 0.800 ms with 512 x 4096 tiles; the forward is ~5% slower tiled, so it
# keeps the column-strip kernel unless CONV_SEQ_TILED_FWD is set.
CONV_SEQ_TILED = True
CONV_SEQ_TILED_FWD = False
CONV_TILED_BLOCK_S = 512
CONV_TILED_BLOCK_C = 4096


def _choose_conv_seq_tile(
    seq_len: int, channels: int
) -> Optional[Tuple[int, int, int]]:
    """`(block_s, block_c, seq_chunk)` for the sequence-tiled kernels, or None."""
    if not CONV_SEQ_TILED:
        return None
    block_c = math.gcd(channels, CONV_TILED_BLOCK_C)
    block_s = math.gcd(seq_len, CONV_TILED_BLOCK_S)
    if (
        block_c < _CONV_MIN_BLOCK_C
        or block_c % _CONV_MIN_BLOCK_C != 0
        or block_s < 16
        or block_s % 8 != 0
        or seq_len // block_s < 2
    ):
        return None
    seq_chunk = max(8, (_CONV_BWD_VREG_ELEMS // block_c) // 8 * 8)
    seq_chunk = math.gcd(block_s, seq_chunk)
    if seq_chunk < 8 or seq_chunk % 8 != 0:
        return None
    return block_s, block_c, seq_chunk


def _conv_wb_3d(conv_weight, conv_bias, channels):
    w_2d = conv_weight.reshape(4, channels).astype(jnp.float32)
    if conv_bias is not None:
        wb_rows = jnp.concatenate(
            [w_2d, conv_bias.reshape(1, channels).astype(jnp.float32)], axis=0
        )
        return jnp.broadcast_to(wb_rows[:, None, :], (5, 8, channels)), 5
    return jnp.broadcast_to(w_2d[:, None, :], (4, 8, channels)), 4


def _conv_seg_masks(s_win_128, lo, n, c_tiles, with_future):
    """Tap masks for `n` rows whose seg rows sit at `s_win_128[lo : lo + n]`."""
    s_cur = s_win_128[lo : lo + n]
    s_0 = pltpu.roll(s_win_128, 3, axis=0)[lo : lo + n]
    s_1 = pltpu.roll(s_win_128, 2, axis=0)[lo : lo + n]
    s_2 = pltpu.roll(s_win_128, 1, axis=0)[lo : lo + n]
    valid = s_cur > 0
    masks = [
        valid,
        valid & (s_0 == s_cur),
        valid & (s_1 == s_cur),
        valid & (s_2 == s_cur),
    ]
    if with_future:
        hi = lo + 8
        s_p1 = pltpu.roll(s_win_128, 7, axis=0)[hi : hi + n]
        s_p2 = pltpu.roll(s_win_128, 6, axis=0)[hi : hi + n]
        s_p3 = pltpu.roll(s_win_128, 5, axis=0)[hi : hi + n]
        masks += [
            valid & (s_p1 == s_cur),
            valid & (s_p2 == s_cur),
            valid & (s_p3 == s_cur),
        ]
    if c_tiles > 1:
        masks = [jnp.tile(m, (1, c_tiles)) for m in masks]
    return masks


def _conv1d_silu_fwd_pallas_seqtiled(
    qkv,
    conv_weight,
    conv_bias,
    conv_state,
    segment_ids,
    conv_halo_seg,
    out_dt,
    tile,
    interpret,
):
    """Sequence-tiled causal Conv1D + SiLU forward (see `CONV_SEQ_TILED`)."""
    batch, seq_len, channels = qkv.shape
    block_s, block_c, seq_chunk = tile
    n_s = seq_len // block_s
    rows8 = block_s // 8
    c_tiles = block_c // _CONV_MIN_BLOCK_C
    mat = _conv_manual_axis_type(qkv)
    has_bias = conv_bias is not None
    has_state = conv_state is not None
    has_segments = segment_ids is not None
    wb_3d, wb_rows_n = _conv_wb_3d(conv_weight, conv_bias, channels)

    in_args = [qkv, qkv, wb_3d]
    in_specs = [
        pl.BlockSpec((1, block_s, block_c), lambda b, s, c: (b, s, c)),
        pl.BlockSpec(
            (1, 8, block_c),
            lambda b, s, c: (b, jnp.maximum(s * rows8 - 1, 0), c),
        ),
        pl.BlockSpec((wb_rows_n, 8, block_c), lambda b, s, c: (0, 0, c)),
    ]
    if has_state:
        in_args.append(jnp.pad(conv_state, ((0, 0), (5, 0), (0, 0))))
        in_specs.append(pl.BlockSpec((1, 8, block_c), lambda b, s, c: (b, 0, c)))
    if has_segments:
        seg_buf = _build_pallas_seg_buf(segment_ids, conv_halo_seg)
        in_args.extend([seg_buf[:, :seq_len, :], seg_buf[:, 8 : seq_len + 8, :]])
        in_specs.extend(
            [
                pl.BlockSpec(
                    (1, 8, _CONV_MIN_BLOCK_C), lambda b, s, c: (b, s * rows8, 0)
                ),
                pl.BlockSpec(
                    (1, block_s, _CONV_MIN_BLOCK_C), lambda b, s, c: (b, s, 0)
                ),
            ]
        )

    def _kernel(*refs):
        it = iter(refs)
        x_ref = next(it)
        xprev_ref = next(it)
        wb_ref = next(it)
        state_ref = next(it) if has_state else None
        seg_prev8_ref = next(it) if has_segments else None
        seg_main_ref = next(it) if has_segments else None
        out_ref = next(it)
        s_idx = pl.program_id(1)
        w8 = [wb_ref[k] for k in range(4)]
        b_8 = wb_ref[4] if has_bias else None
        if state_ref is not None:
            first8 = state_ref[0, :, :].astype(jnp.float32)
        else:
            first8 = jnp.zeros((8, block_c), jnp.float32)
        is_first = (s_idx == 0).astype(jnp.float32)
        xprev8 = xprev_ref[0, :, :].astype(jnp.float32)
        halo8 = xprev8 + (first8 - xprev8) * is_first
        for c0 in range(0, block_s, seq_chunk):
            sub_n = seq_chunk // 8
            x_cur = x_ref[0, pl.ds(c0, seq_chunk), :].astype(jnp.float32)
            prefix_8 = (
                halo8 if c0 == 0 else x_ref[0, pl.ds(c0 - 8, 8), :].astype(jnp.float32)
            )
            x_ext = jnp.concatenate([prefix_8, x_cur], axis=0)
            xs = [
                pltpu.roll(x_ext, 3 - k, axis=0)[8 : seq_chunk + 8] for k in range(3)
            ] + [x_cur]
            ws = [jnp.tile(w, (sub_n, 1)) for w in w8]
            if seg_prev8_ref is not None and seg_main_ref is not None:
                s_prev8 = (
                    seg_prev8_ref[0, :, :]
                    if c0 == 0
                    else seg_main_ref[0, pl.ds(c0 - 8, 8), :]
                )
                s_cur = seg_main_ref[0, pl.ds(c0, seq_chunk), :]
                s_win = jnp.concatenate([s_prev8, s_cur], axis=0)
                valid, m0, m1, m2 = _conv_seg_masks(s_win, 8, seq_chunk, c_tiles, False)
                z = (
                    jnp.where(m0, xs[0], 0.0) * ws[0]
                    + jnp.where(m1, xs[1], 0.0) * ws[1]
                    + jnp.where(m2, xs[2], 0.0) * ws[2]
                    + jnp.where(valid, xs[3], 0.0) * ws[3]
                )
                if b_8 is not None:
                    z = z + jnp.tile(b_8, (sub_n, 1))
                z = jnp.where(valid, z, 0.0)
            else:
                z = xs[0] * ws[0] + xs[1] * ws[1] + xs[2] * ws[2] + xs[3] * ws[3]
                if b_8 is not None:
                    z = z + jnp.tile(b_8, (sub_n, 1))
            out_ref[0, pl.ds(c0, seq_chunk), :] = jax.nn.silu(z).astype(out_ref.dtype)

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(
            (batch, seq_len, channels), out_dt, manual_axis_type=mat
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch, n_s, channels // block_c),
            in_specs=in_specs,
            out_specs=pl.BlockSpec((1, block_s, block_c), lambda b, s, c: (b, s, c)),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel"),
        ),
        interpret=interpret,
        name="conv1d_silu_fwd",
    )(*in_args)


def _conv1d_silu_bwd_pallas_seqtiled(
    qkv,
    conv_weight,
    conv_bias,
    dy,
    conv_state,
    return_d_conv_state,
    segment_ids,
    conv_halo_seg,
    tile,
    interpret,
):
    """Sequence-tiled causal Conv1D + SiLU backward (see `CONV_SEQ_TILED`)."""
    batch, seq_len, channels = qkv.shape
    block_s, block_c, seq_chunk = tile
    n_s = seq_len // block_s
    rows8 = block_s // 8
    last_row8 = seq_len // 8 - 1
    c_tiles = block_c // _CONV_MIN_BLOCK_C
    chunk_starts = list(range(0, block_s, seq_chunk))[::-1]
    mat = _conv_manual_axis_type(qkv)
    has_bias = conv_bias is not None
    has_state = conv_state is not None
    has_segments = segment_ids is not None
    wb_3d, wb_rows_n = _conv_wb_3d(conv_weight, conv_bias, channels)

    main = pl.BlockSpec((1, block_s, block_c), lambda b, s, c: (b, s, c))
    prev8 = pl.BlockSpec(
        (1, 8, block_c), lambda b, s, c: (b, jnp.maximum(s * rows8 - 1, 0), c)
    )
    next8 = pl.BlockSpec(
        (1, 8, block_c),
        lambda b, s, c: (b, jnp.minimum((s + 1) * rows8, last_row8), c),
    )
    in_args = [qkv, qkv, qkv, dy, dy, wb_3d]
    in_specs = [
        main,
        prev8,
        next8,
        main,
        next8,
        pl.BlockSpec((wb_rows_n, 8, block_c), lambda b, s, c: (0, 0, c)),
    ]
    if has_state:
        in_args.append(jnp.pad(conv_state, ((0, 0), (5, 0), (0, 0))))
        in_specs.append(pl.BlockSpec((1, 8, block_c), lambda b, s, c: (b, 0, c)))
    if has_segments:
        seg_buf = _build_pallas_seg_buf(segment_ids, conv_halo_seg)
        in_args.extend(
            [
                seg_buf[:, :seq_len, :],
                seg_buf[:, 8 : seq_len + 8, :],
                seg_buf[:, 16 : seq_len + 16, :],
            ]
        )
        in_specs.extend(
            [
                pl.BlockSpec(
                    (1, 8, _CONV_MIN_BLOCK_C), lambda b, s, c: (b, s * rows8, 0)
                ),
                pl.BlockSpec(
                    (1, block_s, _CONV_MIN_BLOCK_C), lambda b, s, c: (b, s, 0)
                ),
                pl.BlockSpec(
                    (1, 8, _CONV_MIN_BLOCK_C),
                    lambda b, s, c: (b, (s + 1) * rows8 - 1, 0),
                ),
            ]
        )

    out_shapes = [
        jax.ShapeDtypeStruct(
            (batch, seq_len, channels), qkv.dtype, manual_axis_type=mat
        ),
        jax.ShapeDtypeStruct(
            (batch, n_s, 4, 8, channels), jnp.float32, manual_axis_type=mat
        ),
    ]
    out_specs = [
        main,
        pl.BlockSpec((1, 1, 4, 8, block_c), lambda b, s, c: (b, s, 0, 0, c)),
    ]
    if has_bias:
        out_shapes.append(
            jax.ShapeDtypeStruct(
                (batch, n_s, 8, channels), jnp.float32, manual_axis_type=mat
            )
        )
        out_specs.append(pl.BlockSpec((1, 1, 8, block_c), lambda b, s, c: (b, s, 0, c)))
    if return_d_conv_state:
        out_shapes.append(
            jax.ShapeDtypeStruct(
                (batch, n_s, 8, channels), qkv.dtype, manual_axis_type=mat
            )
        )
        out_specs.append(pl.BlockSpec((1, 1, 8, block_c), lambda b, s, c: (b, s, 0, c)))

    def _kernel(*refs):
        it = iter(refs)
        x_ref = next(it)
        xprev_ref = next(it)
        xnext_ref = next(it)
        dy_ref = next(it)
        dynext_ref = next(it)
        wb_ref = next(it)
        state_ref = next(it) if has_state else None
        seg_prev8_ref = next(it) if has_segments else None
        seg_main_ref = next(it) if has_segments else None
        seg_next8_ref = next(it) if has_segments else None
        dx_ref = next(it)
        dw_ref = next(it)
        db_ref = next(it) if has_bias else None
        d_cs_ref = next(it) if return_d_conv_state else None
        s_idx = pl.program_id(1)
        w8 = [wb_ref[k] for k in range(4)]
        b_8 = wb_ref[4] if has_bias else None

        if state_ref is not None:
            first8 = state_ref[0, :, :].astype(jnp.float32)
        else:
            first8 = jnp.zeros((8, block_c), jnp.float32)
        is_first = (s_idx == 0).astype(jnp.float32)
        xprev8 = xprev_ref[0, :, :].astype(jnp.float32)
        halo8 = xprev8 + (first8 - xprev8) * is_first

        # dz of the 8 rows after this tile (0 past the sequence end).
        x16 = jnp.concatenate(
            [
                x_ref[0, pl.ds(block_s - 8, 8), :].astype(jnp.float32),
                xnext_ref[0, :, :].astype(jnp.float32),
            ],
            axis=0,
        )
        xn = [pltpu.roll(x16, 3 - k, axis=0)[8:16] for k in range(3)]
        xn.append(x16[8:16])
        dyn = dynext_ref[0, :, :].astype(jnp.float32)
        if seg_main_ref is not None and seg_next8_ref is not None:
            s_win_n = jnp.concatenate(
                [
                    seg_main_ref[0, pl.ds(block_s - 8, 8), :],
                    seg_next8_ref[0, :, :],
                ],
                axis=0,
            )
            vn, n0, n1, n2 = _conv_seg_masks(s_win_n, 8, 8, c_tiles, False)
            zn = (
                jnp.where(n0, xn[0], 0.0) * w8[0]
                + jnp.where(n1, xn[1], 0.0) * w8[1]
                + jnp.where(n2, xn[2], 0.0) * w8[2]
                + jnp.where(vn, xn[3], 0.0) * w8[3]
            )
            if b_8 is not None:
                zn = zn + b_8
            zn = jnp.where(vn, zn, 0.0)
        else:
            vn = None
            zn = xn[0] * w8[0] + xn[1] * w8[1] + xn[2] * w8[2] + xn[3] * w8[3]
            if b_8 is not None:
                zn = zn + b_8
        sig_n = jax.nn.sigmoid(zn)
        dzn = dyn * (sig_n * (1.0 + zn * (1.0 - sig_n)))
        if vn is not None:
            dzn = jnp.where(vn, dzn, 0.0)
        dz_suf8 = dzn * (s_idx != n_s - 1).astype(jnp.float32)

        zero8 = jnp.zeros((8, block_c), jnp.float32)
        dw_acc = [zero8] * 4
        db_acc = zero8 if has_bias else None
        for c0 in chunk_starts:
            sub_n = seq_chunk // 8
            s_c = pl.ds(c0, seq_chunk)
            x_cur = x_ref[0, s_c, :].astype(jnp.float32)
            dy_cur = dy_ref[0, s_c, :].astype(jnp.float32)
            prefix_8 = (
                halo8 if c0 == 0 else x_ref[0, pl.ds(c0 - 8, 8), :].astype(jnp.float32)
            )
            x_ext = jnp.concatenate([prefix_8, x_cur], axis=0)
            xs = [
                pltpu.roll(x_ext, 3 - k, axis=0)[8 : seq_chunk + 8] for k in range(3)
            ] + [x_cur]
            ws = [jnp.tile(w, (sub_n, 1)) for w in w8]
            dz_ext_tail = dz_suf8
            if (
                seg_prev8_ref is not None
                and seg_main_ref is not None
                and seg_next8_ref is not None
            ):
                s_prev8 = (
                    seg_prev8_ref[0, :, :]
                    if c0 == 0
                    else seg_main_ref[0, pl.ds(c0 - 8, 8), :]
                )
                s_cur = seg_main_ref[0, s_c, :]
                s_next8 = (
                    seg_next8_ref[0, :, :]
                    if c0 + seq_chunk == block_s
                    else seg_main_ref[0, pl.ds(c0 + seq_chunk, 8), :]
                )
                s_win = jnp.concatenate([s_prev8, s_cur, s_next8], axis=0)
                valid, b0, b1, b2, f1, f2, f3 = _conv_seg_masks(
                    s_win, 8, seq_chunk, c_tiles, True
                )
                z = (
                    jnp.where(b0, xs[0], 0.0) * ws[0]
                    + jnp.where(b1, xs[1], 0.0) * ws[1]
                    + jnp.where(b2, xs[2], 0.0) * ws[2]
                    + jnp.where(valid, xs[3], 0.0) * ws[3]
                )
                if b_8 is not None:
                    z = z + jnp.tile(b_8, (sub_n, 1))
                z = jnp.where(valid, z, 0.0)
                sig_z = jax.nn.sigmoid(z)
                dz = jnp.where(valid, dy_cur * (sig_z * (1.0 + z * (1.0 - sig_z))), 0.0)
                dzs = [
                    jnp.where(b0, dz, 0.0),
                    jnp.where(b1, dz, 0.0),
                    jnp.where(b2, dz, 0.0),
                    dz,
                ]
                dz_ext = jnp.concatenate([dz, dz_ext_tail], axis=0)
                u2 = jnp.where(
                    f1, pltpu.roll(dz_ext, 7, axis=0)[8 : seq_chunk + 8], 0.0
                )
                u1 = jnp.where(
                    f2, pltpu.roll(dz_ext, 6, axis=0)[8 : seq_chunk + 8], 0.0
                )
                u0 = jnp.where(
                    f3, pltpu.roll(dz_ext, 5, axis=0)[8 : seq_chunk + 8], 0.0
                )
            else:
                z = xs[0] * ws[0] + xs[1] * ws[1] + xs[2] * ws[2] + xs[3] * ws[3]
                if b_8 is not None:
                    z = z + jnp.tile(b_8, (sub_n, 1))
                sig_z = jax.nn.sigmoid(z)
                dz = dy_cur * (sig_z * (1.0 + z * (1.0 - sig_z)))
                dzs = [dz, dz, dz, dz]
                dz_ext = jnp.concatenate([dz, dz_ext_tail], axis=0)
                u2 = pltpu.roll(dz_ext, 7, axis=0)[8 : seq_chunk + 8]
                u1 = pltpu.roll(dz_ext, 6, axis=0)[8 : seq_chunk + 8]
                u0 = pltpu.roll(dz_ext, 5, axis=0)[8 : seq_chunk + 8]
            dx_cur = dz * ws[3] + u2 * ws[2] + u1 * ws[1] + u0 * ws[0]
            dx_ref[0, s_c, :] = dx_cur.astype(dx_ref.dtype)
            dz_suf8 = dz[:8, :]
            for k in range(4):
                dw_acc[k] = dw_acc[k] + jnp.sum(
                    (dzs[k] * xs[k]).reshape(sub_n, 8, block_c), axis=0
                )
            if db_acc is not None:
                db_acc = db_acc + jnp.sum(dz.reshape(sub_n, 8, block_c), axis=0)
            if c0 == 0 and d_cs_ref is not None:
                row8 = lax.broadcasted_iota(jnp.int32, (8, block_c), 0)
                r1 = (
                    jnp.where(row8 >= 1, pltpu.roll(dzs[1][:8, :], 1, axis=0), 0.0)
                    * w8[1]
                )
                r2 = (
                    jnp.where(row8 >= 2, pltpu.roll(dzs[2][:8, :], 2, axis=0), 0.0)
                    * w8[2]
                )
                d_cs_ref[0, 0, :, :] = (dzs[0][:8, :] * w8[0] + r1 + r2).astype(
                    d_cs_ref.dtype
                )
        for k in range(4):
            dw_ref[0, 0, k, :, :] = dw_acc[k]
        if db_ref is not None:
            db_ref[0, 0, :, :] = db_acc

    outs = pl.pallas_call(
        _kernel,
        out_shape=tuple(out_shapes),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch, n_s, channels // block_c),
            in_specs=in_specs,
            out_specs=tuple(out_specs),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel"),
        ),
        interpret=interpret,
        name="conv1d_silu_bwd",
    )(*in_args)
    it = iter(outs)
    dx = next(it)
    dw_2d = jnp.sum(next(it), axis=(0, 1, 3))
    dw = (dw_2d[:, None, :] if conv_weight.ndim == 3 else dw_2d).astype(
        conv_weight.dtype
    )
    db = None
    if has_bias:
        db = jnp.sum(next(it), axis=(0, 1, 2)).astype(conv_bias.dtype)
        if conv_bias.ndim != 1:
            db = db.reshape(conv_bias.shape)
    if return_d_conv_state:
        d_conv_state = next(it)[:, 0, :3, :]
        return dx, dw, db, d_conv_state
    return dx, dw, db


def conv1d_silu_fwd_pallas(
    qkv: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    kernel_size: int,
    conv_state: Optional[jax.Array] = None,
    segment_ids: Optional[jax.Array] = None,
    conv_halo_seg: Optional[jax.Array] = None,
    output_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array:
    """Wide-channel 2D VMEM-fused causal Conv1D + SiLU forward (`[B, S, C]`).

    Uses 2D `grid = (B, C // block_c)` (`block_c` up to `512`,
    `dimension_semantics=
    ("parallel", "parallel")`) so `seg_ref` `(b, 0, 0)` is DMA'd once per batch
    item
    while `4x` wider channel tiles stream `1 KiB` contiguous row bursts into VMEM.
    """
    if not _can_use_conv1d_silu_pallas(
        qkv, conv_weight, conv_bias, kernel_size, conv_state, interpret
    ):
        _, qkv_conv = conv1d_silu_fwd(
            qkv=qkv,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            kernel_size=kernel_size,
            conv_state=conv_state,
            segment_ids=segment_ids,
            conv_halo_seg=conv_halo_seg,
            output_dtype=output_dtype,
        )
        return qkv_conv

    batch, seq_len, channels = qkv.shape
    out_dt = jnp.dtype(output_dtype or qkv.dtype)
    use_seq_tiled_fwd = CONV_SEQ_TILED_FWD or (
        seq_len > _CONV_MAX_PALLAS_SEQ_LEN
        and os.environ.get("TT_GDN_PALLAS_CONV_FWD_64K", "1") == "1"
    )
    seq_tile = _choose_conv_seq_tile(seq_len, channels) if use_seq_tiled_fwd else None
    if seq_tile is not None:
        return _conv1d_silu_fwd_pallas_seqtiled(
            qkv,
            conv_weight,
            conv_bias,
            conv_state,
            segment_ids,
            conv_halo_seg,
            out_dt,
            seq_tile,
            interpret,
        )
    if seq_len > _CONV_MAX_PALLAS_SEQ_LEN:
        _, qkv_conv = conv1d_silu_fwd(
            qkv=qkv,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            kernel_size=kernel_size,
            conv_state=conv_state,
            segment_ids=segment_ids,
            conv_halo_seg=conv_halo_seg,
            output_dtype=output_dtype,
        )
        return qkv_conv
    io_bytes = jnp.dtype(qkv.dtype).itemsize + out_dt.itemsize
    block_c = _choose_conv_block_c(seq_len, channels, io_bytes)
    c_tiles = block_c // _CONV_MIN_BLOCK_C
    seq_chunk = max(64, _CONV_FWD_VREG_ELEMS // block_c)
    mat = _conv_manual_axis_type(qkv)
    has_bias = conv_bias is not None
    has_state = conv_state is not None
    has_segments = segment_ids is not None

    w_2d = conv_weight.reshape(4, channels).astype(jnp.float32)
    if has_bias:
        assert conv_bias is not None
        wb_rows = jnp.concatenate(
            [w_2d, conv_bias.reshape(1, channels).astype(jnp.float32)], axis=0
        )
        wb_3d = jnp.broadcast_to(wb_rows[:, None, :], (5, 8, channels))
        wb_rows_n = 5
    else:
        wb_3d = jnp.broadcast_to(w_2d[:, None, :], (4, 8, channels))
        wb_rows_n = 4

    in_args: list[jax.Array] = [qkv, wb_3d]
    in_specs: list[pl.BlockSpec] = [
        pl.BlockSpec((1, seq_len, block_c), lambda b, c: (b, 0, c)),
        pl.BlockSpec((wb_rows_n, 8, block_c), lambda b, c: (0, 0, c)),
    ]
    if has_state:
        assert conv_state is not None
        state_pad = jnp.pad(conv_state, ((0, 0), (5, 0), (0, 0)))
        in_args.append(state_pad)
        in_specs.append(pl.BlockSpec((1, 8, block_c), lambda b, c: (b, 0, c)))
    if has_segments:
        assert segment_ids is not None
        seg_buf = _build_pallas_seg_buf(segment_ids, conv_halo_seg)
        in_args.append(seg_buf)
        in_specs.append(
            pl.BlockSpec((1, seq_len + 16, _CONV_MIN_BLOCK_C), lambda b, c: (b, 0, 0))
        )

    def _fwd_kernel(*refs):
        idx = 0
        qkv_ref = refs[idx]
        idx += 1
        wb_ref = refs[idx]
        idx += 1
        state_ref = None
        if has_state:
            state_ref = refs[idx]
            idx += 1
        seg_ref = None
        if has_segments:
            seg_ref = refs[idx]
            idx += 1
        out_ref = refs[idx]

        w0_8 = wb_ref[0]
        w1_8 = wb_ref[1]
        w2_8 = wb_ref[2]
        w3_8 = wb_ref[3]
        b_8 = wb_ref[4] if has_bias else None

        for c0 in range(0, seq_len, seq_chunk):
            c_len = min(seq_chunk, seq_len - c0)
            sub_n = c_len // 8
            s_c = pl.ds(c0, c_len)
            x_cur = qkv_ref[0, s_c, :].astype(jnp.float32)
            if c0 == 0:
                if state_ref is not None:
                    prefix_8 = state_ref[0, :, :].astype(jnp.float32)
                else:
                    prefix_8 = jnp.zeros((8, block_c), dtype=jnp.float32)
            else:
                prefix_8 = qkv_ref[0, pl.ds(c0 - 8, 8), :].astype(jnp.float32)
            x_ext = jnp.concatenate([prefix_8, x_cur], axis=0)
            x_0 = pltpu.roll(x_ext, 3, axis=0)[8 : c_len + 8]
            x_1 = pltpu.roll(x_ext, 2, axis=0)[8 : c_len + 8]
            x_2 = pltpu.roll(x_ext, 1, axis=0)[8 : c_len + 8]
            x_3 = x_cur

            w0 = jnp.tile(w0_8, (sub_n, 1))
            w1 = jnp.tile(w1_8, (sub_n, 1))
            w2 = jnp.tile(w2_8, (sub_n, 1))
            w3 = jnp.tile(w3_8, (sub_n, 1))

            if seg_ref is not None:
                s_ext_128 = seg_ref[0, pl.ds(c0, c_len + 8), :]
                s_cur_128 = s_ext_128[8 : c_len + 8]
                s_0_128 = pltpu.roll(s_ext_128, 3, axis=0)[8 : c_len + 8]
                s_1_128 = pltpu.roll(s_ext_128, 2, axis=0)[8 : c_len + 8]
                s_2_128 = pltpu.roll(s_ext_128, 1, axis=0)[8 : c_len + 8]
                valid_128 = s_cur_128 > 0
                b0_128 = valid_128 & (s_0_128 == s_cur_128)
                b1_128 = valid_128 & (s_1_128 == s_cur_128)
                b2_128 = valid_128 & (s_2_128 == s_cur_128)
                if c_tiles > 1:
                    valid_b = jnp.tile(valid_128, (1, c_tiles))
                    b0 = jnp.tile(b0_128, (1, c_tiles))
                    b1 = jnp.tile(b1_128, (1, c_tiles))
                    b2 = jnp.tile(b2_128, (1, c_tiles))
                else:
                    valid_b, b0, b1, b2 = valid_128, b0_128, b1_128, b2_128
                z = (
                    jnp.where(b0, x_0, 0.0) * w0
                    + jnp.where(b1, x_1, 0.0) * w1
                    + jnp.where(b2, x_2, 0.0) * w2
                    + jnp.where(valid_b, x_3, 0.0) * w3
                )
                if b_8 is not None:
                    z = jnp.where(valid_b, z + jnp.tile(b_8, (sub_n, 1)), 0.0)
                else:
                    z = jnp.where(valid_b, z, 0.0)
            else:
                z = x_0 * w0 + x_1 * w1 + x_2 * w2 + x_3 * w3
                if b_8 is not None:
                    z = z + jnp.tile(b_8, (sub_n, 1))

            out_ref[0, s_c, :] = jax.nn.silu(z).astype(out_ref.dtype)

    return pl.pallas_call(
        _fwd_kernel,
        out_shape=jax.ShapeDtypeStruct(
            (batch, seq_len, channels), out_dt, manual_axis_type=mat
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch, channels // block_c),
            in_specs=in_specs,
            out_specs=pl.BlockSpec((1, seq_len, block_c), lambda b, c: (b, 0, c)),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        interpret=interpret,
        name="conv1d_silu_fwd",
    )(*in_args)


def conv1d_silu_bwd_pallas(
    qkv: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    dy: jax.Array,
    kernel_size: int,
    conv_out: Optional[jax.Array] = None,
    conv_state: Optional[jax.Array] = None,
    return_d_conv_state: bool = False,
    segment_ids: Optional[jax.Array] = None,
    conv_halo_seg: Optional[jax.Array] = None,
    *,
    interpret: bool | pltpu.InterpretParams = False,
):
    """Wide-channel 2D single-pass VMEM-fused causal Conv1D + SiLU backward (`conv_out=None`).

    Uses 2D `grid = (B, C // block_c)` (`block_c = 256` @ `8K`, `512` @ `4K`,
    `dimension_semantics=("parallel", "parallel")`) so `seg_ref` `(b, 0, 0)` is
    DMA'd
    once per batch item, and sweeps sequence chunks in reverse in 1 single VPU
    pass
    (`z -> dz -> dx & dw`, `dz_suf8` in VREGs, zero `dz_vmem_ref` scratch).
    """
    if conv_out is not None or not _can_use_conv1d_silu_pallas(
        qkv, conv_weight, conv_bias, kernel_size, conv_state, interpret
    ):
        return conv1d_silu_bwd(
            qkv=qkv,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            dy=dy,
            kernel_size=kernel_size,
            conv_out=conv_out,
            conv_state=conv_state,
            return_d_conv_state=return_d_conv_state,
            segment_ids=segment_ids,
            conv_halo_seg=conv_halo_seg,
        )

    batch, seq_len, channels = qkv.shape
    seq_tile = _choose_conv_seq_tile(seq_len, channels)
    if seq_tile is not None:
        return _conv1d_silu_bwd_pallas_seqtiled(
            qkv,
            conv_weight,
            conv_bias,
            dy,
            conv_state,
            return_d_conv_state,
            segment_ids,
            conv_halo_seg,
            seq_tile,
            interpret,
        )
    io_bytes = (
        jnp.dtype(qkv.dtype).itemsize
        + jnp.dtype(dy.dtype).itemsize
        + jnp.dtype(qkv.dtype).itemsize
    )
    block_c = _choose_conv_block_c(seq_len, channels, io_bytes)
    c_tiles = block_c // _CONV_MIN_BLOCK_C
    seq_chunk = max(64, _CONV_BWD_VREG_ELEMS // block_c)
    chunk_starts = list(range(0, seq_len, seq_chunk))[::-1]
    mat = _conv_manual_axis_type(qkv)
    has_bias = conv_bias is not None
    has_state = conv_state is not None
    has_segments = segment_ids is not None

    w_2d = conv_weight.reshape(4, channels).astype(jnp.float32)
    if has_bias:
        assert conv_bias is not None
        wb_rows = jnp.concatenate(
            [w_2d, conv_bias.reshape(1, channels).astype(jnp.float32)], axis=0
        )
        wb_3d = jnp.broadcast_to(wb_rows[:, None, :], (5, 8, channels))
        wb_rows_n = 5
    else:
        wb_3d = jnp.broadcast_to(w_2d[:, None, :], (4, 8, channels))
        wb_rows_n = 4

    in_args: list[jax.Array] = [qkv, dy, wb_3d]
    in_specs: list[pl.BlockSpec] = [
        pl.BlockSpec((1, seq_len, block_c), lambda b, c: (b, 0, c)),
        pl.BlockSpec((1, seq_len, block_c), lambda b, c: (b, 0, c)),
        pl.BlockSpec((wb_rows_n, 8, block_c), lambda b, c: (0, 0, c)),
    ]
    if has_state:
        assert conv_state is not None
        state_pad = jnp.pad(conv_state, ((0, 0), (5, 0), (0, 0)))
        in_args.append(state_pad)
        in_specs.append(pl.BlockSpec((1, 8, block_c), lambda b, c: (b, 0, c)))
    if has_segments:
        assert segment_ids is not None
        seg_buf = _build_pallas_seg_buf(segment_ids, conv_halo_seg)
        in_args.append(seg_buf)
        in_specs.append(
            pl.BlockSpec((1, seq_len + 16, _CONV_MIN_BLOCK_C), lambda b, c: (b, 0, 0))
        )

    out_shapes_list: list[jax.ShapeDtypeStruct] = [
        jax.ShapeDtypeStruct(
            (batch, seq_len, channels), qkv.dtype, manual_axis_type=mat
        ),
        jax.ShapeDtypeStruct(
            (batch, 4, 8, channels), jnp.float32, manual_axis_type=mat
        ),
    ]
    out_specs_list: list[pl.BlockSpec] = [
        pl.BlockSpec((1, seq_len, block_c), lambda b, c: (b, 0, c)),
        pl.BlockSpec((1, 4, 8, block_c), lambda b, c: (b, 0, 0, c)),
    ]
    if has_bias:
        out_shapes_list.append(
            jax.ShapeDtypeStruct(
                (batch, 8, channels), jnp.float32, manual_axis_type=mat
            )
        )
        out_specs_list.append(pl.BlockSpec((1, 8, block_c), lambda b, c: (b, 0, c)))
    if return_d_conv_state:
        out_shapes_list.append(
            jax.ShapeDtypeStruct((batch, 8, channels), qkv.dtype, manual_axis_type=mat)
        )
        out_specs_list.append(pl.BlockSpec((1, 8, block_c), lambda b, c: (b, 0, c)))

    def _bwd_kernel(*refs):
        idx = 0
        qkv_ref = refs[idx]
        idx += 1
        dy_ref = refs[idx]
        idx += 1
        wb_ref = refs[idx]
        idx += 1
        state_ref = None
        if has_state:
            state_ref = refs[idx]
            idx += 1
        seg_ref = None
        if has_segments:
            seg_ref = refs[idx]
            idx += 1

        dx_ref = refs[idx]
        idx += 1
        dw_ref = refs[idx]
        idx += 1
        db_ref = None
        if has_bias:
            db_ref = refs[idx]
            idx += 1
        d_cs_ref = None
        if return_d_conv_state:
            d_cs_ref = refs[idx]

        w0_8 = wb_ref[0]
        w1_8 = wb_ref[1]
        w2_8 = wb_ref[2]
        w3_8 = wb_ref[3]
        b_8 = wb_ref[4] if has_bias else None

        dz_suf8 = jnp.zeros((8, block_c), dtype=jnp.float32)
        dw0_acc = jnp.zeros((8, block_c), dtype=jnp.float32)
        dw1_acc = jnp.zeros((8, block_c), dtype=jnp.float32)
        dw2_acc = jnp.zeros((8, block_c), dtype=jnp.float32)
        dw3_acc = jnp.zeros((8, block_c), dtype=jnp.float32)
        db_acc = jnp.zeros((8, block_c), dtype=jnp.float32) if has_bias else None

        for c0 in chunk_starts:
            c_len = min(seq_chunk, seq_len - c0)
            sub_n = c_len // 8
            s_c = pl.ds(c0, c_len)
            x_cur = qkv_ref[0, s_c, :].astype(jnp.float32)
            dy_cur = dy_ref[0, s_c, :].astype(jnp.float32)
            if c0 == 0:
                if state_ref is not None:
                    prefix_8 = state_ref[0, :, :].astype(jnp.float32)
                else:
                    prefix_8 = jnp.zeros((8, block_c), dtype=jnp.float32)
            else:
                prefix_8 = qkv_ref[0, pl.ds(c0 - 8, 8), :].astype(jnp.float32)
            x_ext = jnp.concatenate([prefix_8, x_cur], axis=0)
            x_0 = pltpu.roll(x_ext, 3, axis=0)[8 : c_len + 8]
            x_1 = pltpu.roll(x_ext, 2, axis=0)[8 : c_len + 8]
            x_2 = pltpu.roll(x_ext, 1, axis=0)[8 : c_len + 8]
            x_3 = x_cur

            w0 = jnp.tile(w0_8, (sub_n, 1))
            w1 = jnp.tile(w1_8, (sub_n, 1))
            w2 = jnp.tile(w2_8, (sub_n, 1))
            w3 = jnp.tile(w3_8, (sub_n, 1))

            if seg_ref is not None:
                s_win_128 = seg_ref[0, pl.ds(c0, c_len + 16), :]
                s_cur_128 = s_win_128[8 : c_len + 8]
                s_0_128 = pltpu.roll(s_win_128, 3, axis=0)[8 : c_len + 8]
                s_1_128 = pltpu.roll(s_win_128, 2, axis=0)[8 : c_len + 8]
                s_2_128 = pltpu.roll(s_win_128, 1, axis=0)[8 : c_len + 8]
                s_p1_128 = pltpu.roll(s_win_128, 7, axis=0)[16 : c_len + 16]
                s_p2_128 = pltpu.roll(s_win_128, 6, axis=0)[16 : c_len + 16]
                s_p3_128 = pltpu.roll(s_win_128, 5, axis=0)[16 : c_len + 16]

                valid_128 = s_cur_128 > 0
                b0_128 = valid_128 & (s_0_128 == s_cur_128)
                b1_128 = valid_128 & (s_1_128 == s_cur_128)
                b2_128 = valid_128 & (s_2_128 == s_cur_128)
                f1_128 = valid_128 & (s_p1_128 == s_cur_128)
                f2_128 = valid_128 & (s_p2_128 == s_cur_128)
                f3_128 = valid_128 & (s_p3_128 == s_cur_128)
                if c_tiles > 1:
                    valid_b = jnp.tile(valid_128, (1, c_tiles))
                    b0 = jnp.tile(b0_128, (1, c_tiles))
                    b1 = jnp.tile(b1_128, (1, c_tiles))
                    b2 = jnp.tile(b2_128, (1, c_tiles))
                    f1 = jnp.tile(f1_128, (1, c_tiles))
                    f2 = jnp.tile(f2_128, (1, c_tiles))
                    f3 = jnp.tile(f3_128, (1, c_tiles))
                else:
                    valid_b, b0, b1, b2 = valid_128, b0_128, b1_128, b2_128
                    f1, f2, f3 = f1_128, f2_128, f3_128

                z = (
                    jnp.where(b0, x_0, 0.0) * w0
                    + jnp.where(b1, x_1, 0.0) * w1
                    + jnp.where(b2, x_2, 0.0) * w2
                    + jnp.where(valid_b, x_3, 0.0) * w3
                )
                if b_8 is not None:
                    z = jnp.where(valid_b, z + jnp.tile(b_8, (sub_n, 1)), 0.0)
                else:
                    z = jnp.where(valid_b, z, 0.0)
                sig_z = jax.nn.sigmoid(z)
                dz = jnp.where(
                    valid_b, dy_cur * (sig_z * (1.0 + z * (1.0 - sig_z))), 0.0
                )
                dz_0 = jnp.where(b0, dz, 0.0)
                dz_1 = jnp.where(b1, dz, 0.0)
                dz_2 = jnp.where(b2, dz, 0.0)
                dz_3 = dz

                dz_ext = jnp.concatenate([dz, dz_suf8], axis=0)
                d2 = pltpu.roll(dz_ext, 7, axis=0)[8 : c_len + 8]
                d1 = pltpu.roll(dz_ext, 6, axis=0)[8 : c_len + 8]
                d0 = pltpu.roll(dz_ext, 5, axis=0)[8 : c_len + 8]
                u3_sh = dz
                u2_sh = jnp.where(f1, d2, 0.0)
                u1_sh = jnp.where(f2, d1, 0.0)
                u0_sh = jnp.where(f3, d0, 0.0)
            else:
                z = x_0 * w0 + x_1 * w1 + x_2 * w2 + x_3 * w3
                if b_8 is not None:
                    z = z + jnp.tile(b_8, (sub_n, 1))
                sig_z = jax.nn.sigmoid(z)
                dz = dy_cur * (sig_z * (1.0 + z * (1.0 - sig_z)))
                dz_0 = dz
                dz_1 = dz
                dz_2 = dz
                dz_3 = dz

                dz_ext = jnp.concatenate([dz, dz_suf8], axis=0)
                u3_sh = dz
                u2_sh = pltpu.roll(dz_ext, 7, axis=0)[8 : c_len + 8]
                u1_sh = pltpu.roll(dz_ext, 6, axis=0)[8 : c_len + 8]
                u0_sh = pltpu.roll(dz_ext, 5, axis=0)[8 : c_len + 8]

            dx_cur = u3_sh * w3 + u2_sh * w2 + u1_sh * w1 + u0_sh * w0
            dx_ref[0, s_c, :] = dx_cur.astype(dx_ref.dtype)
            dz_suf8 = dz[:8, :]

            dw0_acc = dw0_acc + jnp.sum((dz_0 * x_0).reshape(sub_n, 8, block_c), axis=0)
            dw1_acc = dw1_acc + jnp.sum((dz_1 * x_1).reshape(sub_n, 8, block_c), axis=0)
            dw2_acc = dw2_acc + jnp.sum((dz_2 * x_2).reshape(sub_n, 8, block_c), axis=0)
            dw3_acc = dw3_acc + jnp.sum((dz_3 * x_3).reshape(sub_n, 8, block_c), axis=0)
            if db_acc is not None:
                db_acc = db_acc + jnp.sum(dz.reshape(sub_n, 8, block_c), axis=0)

            if c0 == 0 and d_cs_ref is not None:
                row8 = lax.broadcasted_iota(jnp.int32, (8, block_c), 0)
                u0_8 = dz_0[:8, :]
                u1_8 = dz_1[:8, :]
                u2_8 = dz_2[:8, :]
                r1 = jnp.where(row8 >= 1, pltpu.roll(u1_8, 1, axis=0), 0.0) * w1_8
                r2 = jnp.where(row8 >= 2, pltpu.roll(u2_8, 2, axis=0), 0.0) * w2_8
                d_cs_8 = u0_8 * w0_8 + r1 + r2
                d_cs_ref[0, :, :] = d_cs_8.astype(d_cs_ref.dtype)

        dw_ref[0, 0, :, :] = dw0_acc
        dw_ref[0, 1, :, :] = dw1_acc
        dw_ref[0, 2, :, :] = dw2_acc
        dw_ref[0, 3, :, :] = dw3_acc
        if db_ref is not None and db_acc is not None:
            db_ref[0, :, :] = db_acc

    pallas_outs = pl.pallas_call(
        _bwd_kernel,
        out_shape=tuple(out_shapes_list),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch, channels // block_c),
            in_specs=in_specs,
            out_specs=tuple(out_specs_list),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        interpret=interpret,
        name="conv1d_silu_bwd",
    )(*in_args)

    out_idx = 0
    dx = pallas_outs[out_idx]
    out_idx += 1
    dw_partials = pallas_outs[out_idx]
    out_idx += 1
    dw_2d = jnp.sum(dw_partials, axis=(0, 2))
    if conv_weight.ndim == 3:
        dw = dw_2d[:, None, :].astype(conv_weight.dtype)
    else:
        dw = dw_2d.astype(conv_weight.dtype)

    if has_bias:
        assert conv_bias is not None
        db_partials = pallas_outs[out_idx]
        out_idx += 1
        db = jnp.sum(db_partials, axis=(0, 1)).astype(conv_bias.dtype)
        if conv_bias.ndim != 1:
            db = db.reshape(conv_bias.shape)
    else:
        db = None

    if return_d_conv_state:
        d_cs_pad = pallas_outs[out_idx]
        d_conv_state = d_cs_pad[:, :3, :]
        return dx, dw, db, d_conv_state
    return dx, dw, db
