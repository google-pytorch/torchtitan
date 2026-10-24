"""Chunked gated delta rule on TPU with Tokamax's Pallas KDA kernels.

GatedDeltaNet's recurrence is Kimi Delta Attention (KDA) with one decay per
head instead of one per key channel: per token, ``S = exp(g) S``, then the delta
update ``S += beta k (v - k^T S)^T`` and ``o = scale * q^T S`` with
L2-normalized ``q`` and ``k``. ``tokamax_gated_delta`` broadcasts the per-head
log decay over the key channels and runs Tokamax's Mosaic TPU KDA forward and
backward kernels, which keep each 64-token chunk in VMEM, unlike the XLA
``chunked_gated_delta``, whose ``[chunk, chunk]`` intermediates round-trip
through HBM.

The JAX functions are bridged to PyTorch with ``torch_tpu``'s ``jax_op``. The
forward op also returns the backward kernel's residuals (the normalized
``q``/``k``, cumulative decays, chunk states, ...: the leaves of the
``jax.vjp`` closure), and the backward op runs only the backward kernels on
them instead of re-running the forward kernel. Under full rematerialization
the residuals live from the recomputed forward to the backward of the same
layer.

Packed documents: given ``segment_ids``, the recurrence restarts at every
document through Tokamax's own ``segment_ids`` / ``max_num_segments`` API. The
kernels then lay every document out from a 64-token chunk boundary (up to
``seq_len + 63 * max_num_segments`` tokens per row) instead of masking within
chunks, so ``max_num_segments`` bounds both the documents per row and the
extra work; ``tokamax_segment_ids`` merges documents beyond it into the last.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import functools

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from torchtitan.config.override import derive, override
from torchtitan.experiments.tpu.qwen3_5.gated_delta import TpuGatedDeltaKernel

try:
  from tokamax._src.ops.kda import api as kda_api
  # Registers the Mosaic TPU backend with `kda_api` (`api.py` imports it behind
  # a `try`, so without this dependency the call fails over to XLA).
  from tokamax._src.ops.kda import pallas_mosaic_tpu  # pylint: disable=unused-import
except (ImportError, SyntaxError) as e:
  # The Tokamax wheels up to 0.1.0 have no KDA op, and under typeguard >= 3 they
  # do not import at all (SyntaxError, openxla/tokamax#240).
  raise ImportError(
      "TokamaxGatedDeltaKernel needs a Tokamax build with the Kimi Delta"
      " Attention op (tokamax._src.ops.kda): github.com/openxla/tokamax HEAD."
  ) from e

# Default lower bound on the per-token log decay. The Mosaic kernel factors the
# decay of each 16-token sub-block around the sub-block's midpoint (its
# "safe gate"), so intermediates grow like ``exp(-8 * g)``; Tokamax validates
# it for log decays in ``[-5, 0)`` (KDA's own ``lower_bound`` range). Faster
# decays are clamped: a decay of ``exp(-5)`` = 0.7% per token already discards
# the state within two tokens.
DEFAULT_MIN_LOG_DECAY = -5.0

# Default packed-document budget: one document per this many tokens of a row
# (`TokamaxGatedDeltaKernel.Config.max_num_segments`). Tokamax pads each
# document to a 64-token chunk boundary, so this bounds the extra kernel tokens
# at 63 / 512 = 12%.
TOKENS_PER_SEGMENT = 512


def _heads_first(x: jax.Array) -> jax.Array:
  """``[B, T, H, ...]`` -> ``[H, B, T, ...]`` (Tokamax KDA's layout)."""
  return jnp.moveaxis(x, 2, 0)


def _gated_delta(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    min_log_decay: float,
    segment_ids: jax.Array | None = None,
    max_num_segments: int | None = None,
) -> jax.Array:
  """Gated delta rule via Mosaic KDA; ``[B, T, H, D]`` in and out.

  ``segment_ids`` (``[B, T]``, 1-based per row) restarts the recurrence at
  every document, with at most ``max_num_segments`` documents per row.
  """
  q_h, k_h, v_h = _heads_first(q), _heads_first(k), _heads_first(v)
  log_decay = jnp.maximum(_heads_first(g).astype(jnp.float32), min_log_decay)
  gate = jnp.broadcast_to(log_decay[..., None], q_h.shape)
  out, _ = kda_api.kimi_delta_attention(
      q_h,
      k_h,
      v_h,
      gate,
      _heads_first(beta).astype(q.dtype),
      use_qk_l2norm=True,
      segment_ids=segment_ids,
      max_num_segments=max_num_segments,
      implementation="mosaic",
  )
  return jnp.moveaxis(out, 0, 2)


def _gated_delta_vjp(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    min_log_decay: float,
    segment_ids: jax.Array | None = None,
    max_num_segments: int | None = None,
):
  """Returns ``(out, vjp)``; ``vjp`` is a pytree of the backward's residuals."""
  return jax.vjp(
      functools.partial(
          _gated_delta,
          min_log_decay=min_log_decay,
          segment_ids=segment_ids,
          max_num_segments=max_num_segments,
      ),
      q,
      k,
      v,
      g,
      beta,
  )


def _input_spec(*tensors: torch.Tensor) -> str:
  """Encodes input shapes and dtypes as one static ``jax_op`` argument."""
  return ";".join(
      ",".join(str(d) for d in t.shape) + ":" + str(t.dtype).split(".")[-1]
      for t in tensors
  )


def _parse_input_spec(spec: str) -> list[jax.ShapeDtypeStruct]:
  """Inverse of `_input_spec`: the inputs' shapes and dtypes."""
  structs = []
  for item in spec.split(";"):
    shape, dtype = item.split(":")
    structs.append(
        jax.ShapeDtypeStruct(
            tuple(int(d) for d in shape.split(",") if d), jnp.dtype(dtype)
        )
    )
  return structs


@pallas.jax_op("tt_qwen35::gated_delta_fwd_res")
def _gated_delta_fwd_res(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    min_log_decay: float,
) -> list[jax.Array]:
  """Returns ``[out, *residuals]``."""
  out, vjp = _gated_delta_vjp(q, k, v, g, beta, min_log_decay)
  return [out, *jax.tree_util.tree_leaves(vjp)]


@pallas.jax_op("tt_qwen35::gated_delta_bwd_res")
def _gated_delta_bwd_res(
    residuals: Sequence[jax.Array],
    grad_out: jax.Array,
    input_spec: str,
    min_log_decay: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Runs the backward kernels on ``_gated_delta_fwd_res``'s residuals."""
  structs = _parse_input_spec(input_spec)
  # The structure (and static backward jaxpr) of the forward's VJP closure,
  # traced abstractly from the same input shapes; its leaves are `residuals`.
  _, vjp_struct = jax.eval_shape(
      functools.partial(_gated_delta_vjp, min_log_decay=min_log_decay),
      *structs,
  )
  vjp = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(vjp_struct), list(residuals)
  )
  return vjp(grad_out.astype(structs[0].dtype))


def _setup_context_res(ctx, inputs, output):
  *tensors, min_log_decay = inputs
  residuals = output[1:]
  ctx.mark_non_differentiable(*residuals)
  ctx.set_materialize_grads(False)
  ctx.save_for_backward(*residuals)
  ctx.input_spec = _input_spec(*tensors)
  ctx.min_log_decay = min_log_decay


def _backward_res(ctx, grad_outputs):
  grad_out = grad_outputs[0]
  if grad_out is None:
    return (None,) * 6
  grads = _gated_delta_bwd_res(
      list(ctx.saved_tensors), grad_out, ctx.input_spec, ctx.min_log_decay
  )
  return (*grads, None)


_gated_delta_fwd_res.register_autograd(
    _backward_res, setup_context=_setup_context_res
)


@pallas.jax_op("tt_qwen35::gated_delta_segments_fwd_res")
def _gated_delta_segments_fwd_res(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    segment_ids: jax.Array,
    min_log_decay: float,
    max_num_segments: int,
) -> list[jax.Array]:
  """``_gated_delta_fwd_res`` restarting at every document of ``segment_ids``."""
  out, vjp = _gated_delta_vjp(
      q,
      k,
      v,
      g,
      beta,
      min_log_decay,
      segment_ids=segment_ids,
      max_num_segments=max_num_segments,
  )
  return [out, *jax.tree_util.tree_leaves(vjp)]


@pallas.jax_op("tt_qwen35::gated_delta_segments_bwd_res")
def _gated_delta_segments_bwd_res(
    residuals: Sequence[jax.Array],
    grad_out: jax.Array,
    input_spec: str,
    min_log_decay: float,
    max_num_segments: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Runs the backward kernels on ``_gated_delta_segments_fwd_res``'s residuals."""
  structs = _parse_input_spec(input_spec)

  def vjp_of(q, k, v, g, beta, segment_ids):
    return _gated_delta_vjp(
        q,
        k,
        v,
        g,
        beta,
        min_log_decay,
        segment_ids=segment_ids,
        max_num_segments=max_num_segments,
    )

  _, vjp_struct = jax.eval_shape(vjp_of, *structs)
  vjp = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(vjp_struct), list(residuals)
  )
  return vjp(grad_out.astype(structs[0].dtype))


def _setup_context_segments_res(ctx, inputs, output):
  *tensors, min_log_decay, max_num_segments = inputs
  residuals = output[1:]
  ctx.mark_non_differentiable(*residuals)
  ctx.set_materialize_grads(False)
  ctx.save_for_backward(*residuals)
  ctx.input_spec = _input_spec(*tensors)
  ctx.min_log_decay = min_log_decay
  ctx.max_num_segments = max_num_segments


def _backward_segments_res(ctx, grad_outputs):
  grad_out = grad_outputs[0]
  if grad_out is None:
    return (None,) * 8
  grads = _gated_delta_segments_bwd_res(
      list(ctx.saved_tensors),
      grad_out,
      ctx.input_spec,
      ctx.min_log_decay,
      ctx.max_num_segments,
  )
  return (*grads, None, None, None)


_gated_delta_segments_fwd_res.register_autograd(
    _backward_segments_res, setup_context=_setup_context_segments_res
)


def tokamax_segment_ids(
    segment_ids_BS: torch.Tensor, max_num_segments: int
) -> torch.Tensor:
  """Converts ``segment_ids_from_positions`` ids to Tokamax's ``segment_ids``.

  Args:
    segment_ids_BS: ``[B, S]`` document index of each token, non-decreasing
      along each row in steps of one (``segment_ids_from_positions``, possibly
      counting on from earlier rows).
    max_num_segments: Documents per row that Tokamax keeps apart.

  Returns:
    ``[B, S]`` int32 ids that start at 1 in every row (Tokamax reserves 0 for
    padding, whose output it zeroes). Documents beyond ``max_num_segments``
    share the last id, so they run as one sequence, as without segment ids,
    instead of being dropped.
  """
  ids = segment_ids_BS - segment_ids_BS[:, :1] + 1
  return ids.clamp(max=max_num_segments).to(torch.int32)


def tokamax_gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    min_log_decay: float = DEFAULT_MIN_LOG_DECAY,
    segment_ids: torch.Tensor | None = None,
    max_num_segments: int | None = None,
) -> torch.Tensor:
  """Gated delta rule with Tokamax's Mosaic TPU KDA kernels.

  Args:
    q: ``(bs, seq_len, n_heads, key_head_dim)`` un-normalized queries.
    k: ``(bs, seq_len, n_heads, key_head_dim)`` un-normalized keys.
    v: ``(bs, seq_len, n_heads, value_head_dim)`` values.
    g: ``(bs, seq_len, n_heads)`` log-space decay (negative).
    beta: ``(bs, seq_len, n_heads)`` update gate in ``(0, 1)``.
    min_log_decay: ``g`` is clamped to at least this value (see
      ``DEFAULT_MIN_LOG_DECAY``).
    segment_ids: Optional ``(bs, seq_len)`` int32 document ids in Tokamax's
      convention (see ``tokamax_segment_ids``); the recurrence restarts at every
      document.
    max_num_segments: Upper bound on the documents per row; required with
      ``segment_ids``.

  Returns:
    ``(bs, seq_len, n_heads, value_head_dim)`` output in ``q.dtype``. Applies
    Q/K L2 normalization and ``1 / sqrt(key_head_dim)`` query scaling, as
    ``gated_delta.chunked_gated_delta`` does. Without ``segment_ids``,
    ``seq_len`` must be a multiple of 64 (the kernel's chunk size).
  """
  if segment_ids is None:
    return _gated_delta_fwd_res(q, k, v, g, beta, float(min_log_decay))[0]
  if max_num_segments is None:
    raise ValueError("tokamax_gated_delta: segment_ids need max_num_segments.")
  return _gated_delta_segments_fwd_res(
      q,
      k,
      v,
      g,
      beta,
      segment_ids,
      float(min_log_decay),
      int(max_num_segments),
  )[0]


class TokamaxGatedDeltaKernel(TpuGatedDeltaKernel):
  """``TpuGatedDeltaKernel`` with Tokamax's Mosaic KDA kernels on TPU.

  The kernels keep each 64-token chunk in VMEM, so ``chunk_size`` does not
  apply. With packed documents (``segment_ids_BS``), the recurrence restarts at
  each of the first ``max_num_segments`` documents of every row.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TpuGatedDeltaKernel.Config):
    # Per-token log decays below this are clamped to it (see
    # `DEFAULT_MIN_LOG_DECAY`).
    min_log_decay: float = DEFAULT_MIN_LOG_DECAY
    # Packed documents per row kept apart (Tokamax's `max_num_segments`);
    # later documents of a row share one recurrence. Tokamax lays every
    # document out from a 64-token chunk boundary, so the kernels run up to
    # `seq_len + 63 * max_num_segments` tokens per row. None: one document per
    # `TOKENS_PER_SEGMENT` tokens of the row (at most 12% more kernel tokens;
    # 16 per 8K row, which covers every row of `c4_test` packed at 8K, at
    # most 13 documents). `dataloader.max_num_documents` (plus one for a
    # padded row's tail) bounds the documents per row.
    max_num_segments: int | None = None

  def __init__(self, config: Config):
    super().__init__(config)
    self.min_log_decay = config.min_log_decay
    self.max_num_segments = config.max_num_segments

  def recurrence(
      self,
      q_BSHK: torch.Tensor,
      k_BSHK: torch.Tensor,
      v_BSHV: torch.Tensor,
      g_BSH: torch.Tensor,
      beta_BSH: torch.Tensor,
      segment_ids_BS: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if segment_ids_BS is None:
      return tokamax_gated_delta(
          q_BSHK,
          k_BSHK,
          v_BSHV,
          g_BSH,
          beta_BSH,
          min_log_decay=self.min_log_decay,
      )
    max_num_segments = self.max_num_segments or max(
        1, q_BSHK.shape[1] // TOKENS_PER_SEGMENT
    )
    return tokamax_gated_delta(
        q_BSHK,
        k_BSHK,
        v_BSHV,
        g_BSH,
        beta_BSH,
        min_log_decay=self.min_log_decay,
        segment_ids=tokamax_segment_ids(segment_ids_BS, max_num_segments),
        max_num_segments=max_num_segments,
    )


@override(
    target=TpuGatedDeltaKernel.Config,
    exact=True,
    description="Run the Qwen3.5 GatedDeltaNet with Tokamax's TPU kernels.",
)
def tokamax_gated_delta_kernel(
    cfg: TpuGatedDeltaKernel.Config,
) -> TokamaxGatedDeltaKernel.Config:
  return derive(cfg, TokamaxGatedDeltaKernel.Config)
