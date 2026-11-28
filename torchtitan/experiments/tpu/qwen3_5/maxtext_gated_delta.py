"""Bridge from TorchTitan Qwen3.5 GatedDeltaNet to MaxText's fused Pallas GDN kernel."""

from collections.abc import Sequence
import functools
import os

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from torchtitan.experiments.tpu.kernels._maxtext_gdn import gdn_decoupled_conv1d


def _input_spec(*tensors: torch.Tensor) -> str:
    return ";".join(
        ",".join(str(d) for d in t.shape) + ":" + str(t.dtype).split(".")[-1]
        for t in tensors
    )


def _parse_input_spec(spec: str) -> list[jax.ShapeDtypeStruct]:
    structs = []
    for item in spec.split(";"):
        shape, dtype = item.split(":")
        structs.append(
            jax.ShapeDtypeStruct(
                tuple(int(d) for d in shape.split(",") if d), jnp.dtype(dtype)
            )
        )
    return structs


def _rebuild_vjp(vjp_struct, residuals: Sequence[jax.Array]):
    leaves = jax.tree_util.tree_leaves(vjp_struct)
    expected = [(tuple(l.shape), l.dtype) for l in leaves]
    actual = [(tuple(r.shape), r.dtype) for r in residuals]
    if expected != actual:
        raise ValueError(
            "MaxText GDN residuals do not match VJP structure:"
            f" expected {expected}, got {actual}"
        )
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(vjp_struct), list(residuals)
    )


def _maxtext_gdn_jax(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    segment_ids: jax.Array | None,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    bwd_natural_layout: str,
    keep_bf16_states: bool,
) -> jax.Array:
    gdn_chunk_sz = int(os.environ.get("TT_MAXTEXT_GDN_CHUNK_SIZE", "64"))
    recompute_tinv = (
        gdn_chunk_sz > 64 or os.environ.get("TT_MAXTEXT_GDN_RECOMPUTE_TINV", "0") == "1"
    )
    fwd_hgs = int(os.environ.get("TT_MAXTEXT_GDN_HEAD_GROUP", "0"))
    out, _ = gdn_decoupled_conv1d(
        qkv,
        b,
        a,
        conv_weight,
        None,
        a_log,
        dt_bias,
        None,
        None,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel_size,
        gdn_chunk_sz,
        True,
        jnp.bfloat16,
        None,
        segment_ids,
        conv_input_in_qkv_dtype=True,
        pass1_states_and_bf16_qkv=True,
        bwd_natural_layout=bwd_natural_layout,
        conv_natural_layout=True,
        fwd_head_group_size=fwd_hgs,
        keep_bf16_states=keep_bf16_states,
        recompute_t_inv_bwd=recompute_tinv,
    )
    return out


def _maxtext_gdn_vjp(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    *,
    segment_ids: jax.Array | None,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    bwd_natural_layout: str,
    keep_bf16_states: bool,
):
    return jax.vjp(
        functools.partial(
            _maxtext_gdn_jax,
            segment_ids=segment_ids,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size,
            bwd_natural_layout=bwd_natural_layout,
            keep_bf16_states=keep_bf16_states,
        ),
        qkv,
        b,
        a,
        conv_weight,
        a_log,
        dt_bias,
    )


@pallas.jax_op("tt_qwen35::maxtext_gdn_seg_fwd_res")
def _maxtext_gdn_seg_fwd_res(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    segment_ids: jax.Array,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    bwd_natural_layout: str,
    keep_bf16_states: bool,
) -> list[jax.Array]:
    out, vjp = _maxtext_gdn_vjp(
        qkv,
        b,
        a,
        conv_weight,
        a_log,
        dt_bias,
        segment_ids=segment_ids,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        bwd_natural_layout=bwd_natural_layout,
        keep_bf16_states=keep_bf16_states,
    )
    return [out, *jax.tree_util.tree_leaves(vjp)]


@pallas.jax_op("tt_qwen35::maxtext_gdn_seg_bwd_res")
def _maxtext_gdn_seg_bwd_res(
    residuals: Sequence[jax.Array],
    grad_out: jax.Array,
    input_spec: str,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    bwd_natural_layout: str,
    keep_bf16_states: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    structs = _parse_input_spec(input_spec)

    def vjp_of(qkv, b, a, conv_weight, a_log, dt_bias, segment_ids):
        return _maxtext_gdn_vjp(
            qkv,
            b,
            a,
            conv_weight,
            a_log,
            dt_bias,
            segment_ids=segment_ids,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size,
            bwd_natural_layout=bwd_natural_layout,
            keep_bf16_states=keep_bf16_states,
        )

    _, vjp_struct = jax.eval_shape(vjp_of, *structs)
    vjp = _rebuild_vjp(vjp_struct, residuals)
    return vjp(grad_out.astype(structs[0].dtype))


def _setup_ctx_seg(ctx, inputs, output):
    (
        qkv,
        b,
        a,
        conv_weight,
        a_log,
        dt_bias,
        segment_ids,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel_size,
        bwd_natural_layout,
        keep_bf16_states,
    ) = inputs
    residuals = output[1:]
    ctx.mark_non_differentiable(*residuals)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*residuals)
    ctx.input_spec = _input_spec(qkv, b, a, conv_weight, a_log, dt_bias, segment_ids)
    ctx.static_args = (
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel_size,
        bwd_natural_layout,
        keep_bf16_states,
    )


def _bwd_seg(ctx, grad_outputs):
    grad_out = grad_outputs[0]
    if grad_out is None:
        return (None,) * 14
    grads = _maxtext_gdn_seg_bwd_res(
        list(ctx.saved_tensors),
        grad_out.contiguous(),
        ctx.input_spec,
        *ctx.static_args,
    )
    return (*grads, *(None,) * 8)


_maxtext_gdn_seg_fwd_res.register_autograd(_bwd_seg, setup_context=_setup_ctx_seg)


@pallas.jax_op("tt_qwen35::maxtext_gdn_seg_fwd_only")
def _maxtext_gdn_seg_fwd_only(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    segment_ids: jax.Array,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    bwd_natural_layout: str,
    keep_bf16_states: bool,
) -> jax.Array:
    return _maxtext_gdn_jax(
        qkv,
        b,
        a,
        conv_weight,
        a_log,
        dt_bias,
        segment_ids,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel_size,
        bwd_natural_layout,
        keep_bf16_states,
    )


def maxtext_conv_gated_delta(
    query_BSC: torch.Tensor | None,
    key_BSC: torch.Tensor | None,
    value_BSC: torch.Tensor | None,
    a_BSH: torch.Tensor,
    b_BSH: torch.Tensor,
    conv_q_weight_C1W: torch.Tensor,
    conv_k_weight_C1W: torch.Tensor,
    conv_v_weight_C1W: torch.Tensor,
    A_log_H: torch.Tensor,
    dt_bias_H: torch.Tensor,
    segment_ids_BS: torch.Tensor | None,
    *,
    key_head_dim: int,
    value_head_dim: int,
    qkv_BSC: torch.Tensor | None = None,
) -> torch.Tensor:
    """Runs MaxText's fused Conv1D + GDN forward & 1-pass Pallas backward on [B, S, ...]."""
    conv_kernel_size = conv_q_weight_C1W.shape[-1]
    if qkv_BSC is not None:
        orig_dtype = qkv_BSC.dtype
        num_k_heads = conv_q_weight_C1W.shape[0] // key_head_dim
        num_v_heads = conv_v_weight_C1W.shape[0] // value_head_dim
        qkv_BSD = qkv_BSC.to(torch.bfloat16).contiguous()
        bs_shape = qkv_BSC.shape[:2]
        dev = qkv_BSC.device
    else:
        assert query_BSC is not None and key_BSC is not None and value_BSC is not None
        orig_dtype = query_BSC.dtype
        c_q = query_BSC.shape[-1]
        c_v = value_BSC.shape[-1]
        num_k_heads = c_q // key_head_dim
        num_v_heads = c_v // value_head_dim
        qkv_BSD = (
            torch.cat([query_BSC, key_BSC, value_BSC], dim=-1)
            .to(torch.bfloat16)
            .contiguous()
        )
        bs_shape = query_BSC.shape[:2]
        dev = query_BSC.device
    b_in = b_BSH.to(torch.bfloat16).contiguous()
    a_in = a_BSH.to(torch.bfloat16).contiguous()
    conv_w = (
        torch.cat([conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W], dim=0)
        .permute(2, 1, 0)
        .to(torch.float32)
        .contiguous()
    )
    a_log_in = A_log_H.to(torch.float32).contiguous()
    dt_bias_in = dt_bias_H.to(torch.float32).contiguous()
    if segment_ids_BS is None:
        seg_in = torch.ones(bs_shape, dtype=torch.int32, device=dev)
    else:
        # segment_ids_from_positions starts at 0; MaxText uses >= 1 for active docs and 0 for padding.
        seg_in = (segment_ids_BS.to(torch.int32) + 1).contiguous()

    bwd_layout = os.environ.get("TT_MAXTEXT_GDN_BWD_LAYOUT", "all")
    keep_bf16 = os.environ.get("TT_MAXTEXT_GDN_BF16_STATES", "0") == "1"
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (qkv_BSD, b_in, a_in, conv_w, a_log_in, dt_bias_in)
    )
    if not needs_grad:
        out = _maxtext_gdn_seg_fwd_only(
            qkv_BSD,
            b_in,
            a_in,
            conv_w,
            a_log_in,
            dt_bias_in,
            seg_in,
            num_k_heads,
            num_v_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size,
            bwd_layout,
            keep_bf16,
        )
    else:
        out = _maxtext_gdn_seg_fwd_res(
            qkv_BSD,
            b_in,
            a_in,
            conv_w,
            a_log_in,
            dt_bias_in,
            seg_in,
            num_k_heads,
            num_v_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size,
            bwd_layout,
            keep_bf16,
        )[0]
    return out.to(orig_dtype)
