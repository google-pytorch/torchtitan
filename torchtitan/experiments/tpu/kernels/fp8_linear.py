"""PyTorch / TorchTPU bridge for fused FP8 linear Pallas kernels."""

import functools

import torch

BLOCK_ALIGNMENT = 128


def supported(x: torch.Tensor, weight: torch.Tensor) -> str | None:
  """Returns why `fp8_linear` cannot run on `(x, weight)`, or `None`."""
  if x.device.type != "tpu":
    return f"tensors are on {x.device.type}, not tpu"
  if weight.ndim != 2:
    return f"expected a 2-D (out, in) weight, got rank {weight.ndim}"
  if x.ndim < 2:
    return f"expected (..., in) activations, got rank {x.ndim}"
  out_features, in_features = weight.shape
  if x.shape[-1] != in_features:
    return f"activation width {x.shape[-1]} != weight in_features {in_features}"
  rows = x.numel() // in_features
  for name, dim in (("rows", rows), ("in", in_features), ("out", out_features)):
    if dim <= 0 or dim % BLOCK_ALIGNMENT != 0:
      return f"{name} dimension {dim} is not a multiple of {BLOCK_ALIGNMENT}"
  if x.dtype not in (torch.bfloat16, torch.float32):
    return f"activations must be bfloat16 or float32, got {x.dtype}"
  return None


@functools.cache
def _get_pallas_ops():
  """Lazily imports `torch_tpu` + `fp8_linear_kernel` and registers autograd."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from torch_tpu._internal import pallas  # pylint: disable=g-import-not-at-top
  from torchtitan.experiments.tpu.kernels import fp8_linear_kernel as kernel  # pylint: disable=g-import-not-at-top

  def to_i8(x: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(x, jnp.int8)

  def from_i8(x: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(x, kernel.FP8_DTYPE)

  def _fwd_jax(
      x: jax.Array, weight: jax.Array
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    m, k = x.shape
    n = weight.shape[0]
    w_q, w_scale = kernel.quantize_cols_fp8(weight.T)
    tiles = kernel.get_fwd_tiles(m, k, n, lhs_dtype=x.dtype)
    run = functools.partial(
        kernel.fused_fp8_matmul_pallas,
        x,
        w_q,
        w_scale,
        bm=tiles.block_m,
        bk=tiles.block_k,
        bn=tiles.block_n,
    )
    if tiles.block_k == k:
      out, x_q, x_scale = run(return_xq=True)
    else:
      out = run(return_xq=False)
      x_q, x_scale = kernel.quantize_rows_fp8(x)
    return out, to_i8(x_q), x_scale, to_i8(w_q), w_scale

  def _bwd_jax(
      dout: jax.Array,
      x_q: jax.Array,
      x_scale: jax.Array,
      w_q: jax.Array,
      w_scale: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    x_q = from_i8(x_q)
    w_q = from_i8(w_q)
    m, n = dout.shape
    k = w_q.shape[0]
    dlhs_tiles = kernel.get_dlhs_tiles(m, k, n, lhs_dtype=dout.dtype)
    dx = kernel.fused_fp8_dlhs_pallas(
        dout,
        w_q,
        w_scale,
        bm=dlhs_tiles.block_m,
        bk=dlhs_tiles.block_k,
        bn=dlhs_tiles.block_n,
    )
    drhs_tiles = kernel.get_drhs_tiles(m, k, n, lhs_dtype=dout.dtype)
    dw = kernel.fused_fp8_drhs_pallas(
        x_q,
        x_scale,
        dout,
        bm=drhs_tiles.block_m,
        bk=drhs_tiles.block_k,
        bn=drhs_tiles.block_n,
        out_dtype=dout.dtype,
    )
    return dx, dw.T

  fwd_op = pallas.jax_op("pallas::fp8_linear_fwd", _fwd_jax)
  bwd_op = pallas.jax_op("pallas::fp8_linear_bwd", _bwd_jax)

  def _setup_context(ctx, inputs, output):
    del inputs
    _, x_q, x_scale, w_q, w_scale = output
    ctx.mark_non_differentiable(x_q, x_scale, w_q, w_scale)
    ctx.save_for_backward(x_q, x_scale, w_q, w_scale)

  def _backward(ctx, grad_out, *unused_residual_grads):
    del unused_residual_grads
    dx, dw = bwd_op(grad_out.contiguous(), *ctx.saved_tensors)
    return dx, dw

  fwd_op.register_autograd(_backward, setup_context=_setup_context)
  return fwd_op, bwd_op


def fp8_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
  """Computes `x @ weight.T` in FP8 for `(..., in)` activations and `(out, in)` weight."""
  reason = supported(x, weight)
  if reason is not None:
    raise ValueError(f"fp8_linear unsupported inputs: {reason}")
  fwd_op, _ = _get_pallas_ops()
  lead = x.shape[:-1]
  out, *_ = fwd_op(
      x.reshape(-1, x.shape[-1]).contiguous(), weight.to(x.dtype).contiguous()
  )
  return out.reshape(*lead, weight.shape[0])
