"""TPU no-drop overflow tail for capacity-bounded Ring-of-Experts EP.

``components/ep_overflow.py`` describes the scheme. Chunk 0 (sorted local
slots ``[0, cap)``) runs on the regular Ring-of-Experts kernels; these ops add
sorted slots ``[cap, count_total)`` in windows of ``cap`` slots inside one
``lax.while_loop`` per direction, the PyTorch counterpart of the JAX recipe's
forked ``gmmv2`` overflow loop (``qwen35_jax/moe_kernels.py``,
``chunk_num_tokens`` / ``chunk_group_sizes``). Without overflow both loops
run zero iterations.

- Forward (``tt_moe::roe_overflow_fwd``): takes chunk 0's partial combine as
  the loop carry (donated with ``TT_EP_OVERFLOW_DONATE=1``) and
  scatter-adds each window's ``W2((act(x W1) * (x W3)) * w)`` rows into it.
- Backward (``tt_moe::roe_overflow_bwd``): recomputes each window, takes the
  VJP of the window with respect to its rows, routing weights, and the expert
  weights, and scatter-adds the row and weight gradients. It saves nothing
  per window.

On TPU the expert matmuls are the Pallas megablox grouped matmuls (``gmm``
with its ``tgmm`` custom VJP), which visit only the tiles of the window's
groups. torch_tpu cannot lower XLA ``ragged_dot`` (``mhlo.ragged_dot`` is
illegal there); the ``ragged_dot`` path is the CPU/reference fallback. Rows
past ``sum(group_sizes)`` are masked with ``where`` after every matmul, so
whatever a kernel leaves there never reaches the outputs or gradients.

Environment knobs (read when the op is traced):

- ``TT_EP_OVERFLOW_IMPL=while|fori`` (default ``while``): ``while`` runs
  a ``lax.while_loop`` with a data-dependent trip count (zero iterations
  without overflow). ``fori`` is the static fallback in case the dynamic loop
  fails to compile: a ``lax.fori_loop`` over the worst-case number of windows,
  each skipped with ``lax.cond`` when it starts past the routed count.
- ``TT_EP_OVERFLOW_MAX_WINDOWS`` (``fori`` only, default: worst case,
  which guarantees no drops): cap on the static window count. Routes past
  ``cap * (1 + max_windows)`` are then dropped.
- ``TT_EP_OVERFLOW_TILES="tm,tk,tn"`` (default ``0,0,0`` = auto: the
  largest of 512/256/128 that fits): megablox tile sizes. A tile that does
  not divide its dimensions falls back to auto.
- ``TT_EP_OVERFLOW_DONATE=1`` (default ``0``): donate chunk 0's partial
  combine to the forward op as the loop carry. jax_op registers donated
  tensors as plain (non-mutated) inputs, so donation is safe only when nothing
  else reads the partial; inside one compiled XLA program the carry copy is
  usually elided anyway.
"""

import math
import os

import jax
from jax import lax
from jax.experimental.pallas.ops.tpu.megablox import ops as megablox_ops
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

_F32 = jnp.float32


def _act(x: jax.Array, activation: str) -> jax.Array:
  if activation == "gelu_tanh":
    return jax.nn.gelu(x, approximate=True)
  return jax.nn.silu(x)


def chunk_group_sizes(
    group_sizes_e: jax.Array, start: jax.Array, num_tokens: jax.Array
) -> jax.Array:
  """Per-local-expert sizes of sorted slots ``[start, start + num_tokens)``."""
  ends = jnp.cumsum(group_sizes_e)
  begins = ends - group_sizes_e
  stop = start + num_tokens
  return (jnp.clip(ends, start, stop) - jnp.clip(begins, start, stop)).astype(
      jnp.int32
  )


def _window_tables(slot_pad, group_sizes_e, count, start, *, cap, top_k):
  """Slots, validity, tokens, and group sizes of one ``cap``-slot window."""
  slots = lax.dynamic_slice_in_dim(slot_pad, start, cap, axis=0)
  iota = jnp.arange(cap, dtype=jnp.int32)
  valid = start + iota < count
  n = jnp.clip(count - start, 0, cap).astype(jnp.int32)
  gs = chunk_group_sizes(group_sizes_e, start, n)
  tok = jnp.where(valid, slots // top_k, 0)
  return slots, valid, tok, gs


def _tile(dim: int, prefs=(512, 256, 128)) -> int:
  for t in prefs:
    if dim % t == 0:
      return t
  return dim


def _tiling(m: int, k: int, n: int, tiles: tuple[int, int, int]):
  """Megablox ``(tm, tk, tn)`` for an ``[m, k] @ [g, k, n]`` grouped matmul.

  The custom VJP reuses the tiling for the transposed dInput matmul, where K
  and N swap roles, so ``tk`` and ``tn`` must divide both. A requested tile
  that does not fit falls back to the automatic choice.
  """
  tm_req, tk_req, tn_req = tiles
  kn = math.gcd(k, n)
  tm = tm_req if tm_req > 0 and m % tm_req == 0 else _tile(m)
  tk = tk_req if tk_req > 0 and kn % tk_req == 0 else _tile(kn)
  tn = tn_req if tn_req > 0 and kn % tn_req == 0 else _tile(kn)
  return tm, tk, tn


def _grouped_mm(lhs, rhs, gs, *, use_pallas, interpret, tiles):
  """Grouped ``lhs @ rhs[group]`` in f32; rows past ``sum(gs)`` undefined."""
  if not use_pallas:
    return lax.ragged_dot(lhs, rhs, gs, preferred_element_type=_F32)
  m, k = lhs.shape
  n = rhs.shape[2]
  return megablox_ops.gmm(
      lhs,
      rhs,
      gs,
      _F32,
      _tiling(m, k, n, tiles),
      None,
      None,
      False,
      interpret,
  )


def _window_core(rows, w_rows, w1, w3, w2, gs, valid, opts):
  """Expert output rows ``[cap, D]`` (f32) of a window, zero where invalid."""
  mm = lambda a, b: _grouped_mm(
      a,
      b,
      gs,
      use_pallas=opts["use_pallas"],
      interpret=opts["interpret"],
      tiles=opts["tiles"],
  )
  keep = lambda t: jnp.where(valid[:, None], t, 0.0)
  gate = keep(mm(rows, w1))
  up = keep(mm(rows, w3))
  h = (_act(gate, opts["activation"]) * up * w_rows[:, None]).astype(rows.dtype)
  return keep(mm(h, w2))


def _gather_rows(x_all, tok, valid, cap):
  rows = jnp.take(x_all, tok, axis=0).reshape(cap, -1)
  return jnp.where(valid[:, None], rows, jnp.zeros((), rows.dtype))


def _run_windows(body, init, count, *, cap, fori, max_windows):
  """Folds ``body(start, carry)`` over windows ``start = cap, 2 cap, ...``.

  ``fori=False``: a ``lax.while_loop`` while ``start < count`` (dynamic trip
  count). ``fori=True``: a ``lax.fori_loop`` over ``max_windows`` windows, each
  a ``lax.cond`` that skips windows starting at or past ``count``.
  """
  if not fori:
    _, out = lax.while_loop(
        lambda c: c[0] < count,
        lambda c: (c[0] + cap, body(c[0], c[1])),
        (jnp.int32(cap), init),
    )
    return out

  def fbody(i, carry):
    start = (i + 1) * cap
    return lax.cond(start < count, lambda c: body(start, c), lambda c: c, carry)

  return lax.fori_loop(0, max_windows, fbody, init)


def _opts(activation, use_pallas, interpret, tiles):
  return dict(
      activation=activation,
      use_pallas=use_pallas,
      interpret=interpret,
      tiles=tuple(tiles),
  )


def overflow_fwd_impl(
    partial,
    x_all,
    scores_flat_M,
    w1,
    w3,
    w2,
    slot_order_M,
    group_sizes_e,
    count,
    *,
    cap: int,
    top_k: int,
    activation: str,
    use_pallas: bool = False,
    interpret: bool = False,
    fori: bool = False,
    max_windows: int = 0,
    tiles: tuple[int, int, int] = (0, 0, 0),
):
  """``partial`` plus the expert outputs of sorted slots ``[cap, count)``."""
  opts = _opts(activation, use_pallas, interpret, tiles)
  num_rows = partial.shape[0]
  slot_pad = jnp.pad(slot_order_M.astype(jnp.int32), (0, cap))
  count = count.reshape(()).astype(jnp.int32)
  scores_f32 = scores_flat_M.astype(_F32)

  def body(start, acc):
    slots, valid, tok, gs = _window_tables(
        slot_pad, group_sizes_e, count, start, cap=cap, top_k=top_k
    )
    rows = _gather_rows(x_all, tok, valid, cap)
    w_rows = jnp.where(valid, scores_f32[jnp.where(valid, slots, 0)], 0.0)
    y = _window_core(rows, w_rows, w1, w3, w2, gs, valid, opts)
    y = jnp.where(valid[:, None], y, 0.0).astype(acc.dtype)
    dst = jnp.where(valid, tok, num_rows)  # Out of range: dropped.
    return acc.at[dst].add(y.reshape((cap,) + acc.shape[1:]), mode="drop")

  return _run_windows(
      body, partial, count, cap=cap, fori=fori, max_windows=max_windows
  )


def overflow_bwd_impl(
    grad_out,
    x_all,
    scores_flat_M,
    w1,
    w3,
    w2,
    slot_order_M,
    group_sizes_e,
    count,
    *,
    cap: int,
    top_k: int,
    activation: str,
    use_pallas: bool = False,
    interpret: bool = False,
    fori: bool = False,
    max_windows: int = 0,
    tiles: tuple[int, int, int] = (0, 0, 0),
):
  """``(d x_all, d scores_flat_M, d w1, d w3, d w2)`` of the overflow tail."""
  opts = _opts(activation, use_pallas, interpret, tiles)
  num_rows = x_all.shape[0]
  m = scores_flat_M.shape[0]
  slot_pad = jnp.pad(slot_order_M.astype(jnp.int32), (0, cap))
  count = count.reshape(()).astype(jnp.int32)
  scores_f32 = scores_flat_M.astype(_F32)

  def body(start, carry):
    dx, ds, dw1, dw3, dw2 = carry
    slots, valid, tok, gs = _window_tables(
        slot_pad, group_sizes_e, count, start, cap=cap, top_k=top_k
    )
    rows = _gather_rows(x_all, tok, valid, cap)
    w_rows = jnp.where(valid, scores_f32[jnp.where(valid, slots, 0)], 0.0)
    gy = jnp.take(grad_out, tok, axis=0).reshape(cap, -1).astype(_F32)
    gy = jnp.where(valid[:, None], gy, 0.0)
    _, vjp = jax.vjp(
        lambda r, w, a, b, c: _window_core(r, w, a, b, c, gs, valid, opts),
        rows,
        w_rows,
        w1,
        w3,
        w2,
    )
    d_rows, d_w_rows, d1, d3, d2 = vjp(gy)
    d_rows = jnp.where(valid[:, None], d_rows, 0.0).astype(dx.dtype)
    dst = jnp.where(valid, tok, num_rows)
    dx = dx.at[dst].add(d_rows.reshape((cap,) + dx.shape[1:]), mode="drop")
    ds = ds.at[jnp.where(valid, slots, m)].add(
        jnp.where(valid, d_w_rows, 0.0), mode="drop"
    )
    return (
        dx,
        ds,
        dw1 + d1.astype(dw1.dtype),
        dw3 + d3.astype(dw3.dtype),
        dw2 + d2.astype(dw2.dtype),
    )

  init = (
      jnp.zeros_like(x_all),
      jnp.zeros((m,), _F32),
      jnp.zeros_like(w1),
      jnp.zeros_like(w3),
      jnp.zeros_like(w2),
  )
  dx, ds, dw1, dw3, dw2 = _run_windows(
      body, init, count, cap=cap, fori=fori, max_windows=max_windows
  )
  return dx, ds.astype(scores_flat_M.dtype), dw1, dw3, dw2


def _roe_overflow_fwd_jax(
    partial: jax.Array,
    x_all: jax.Array,
    scores_flat_M: jax.Array,
    w1: jax.Array,
    w3: jax.Array,
    w2: jax.Array,
    slot_order_M: jax.Array,
    group_sizes_e: jax.Array,
    count_1: jax.Array,
    cap: int,
    top_k: int,
    gelu_tanh: bool,
    use_pallas: bool,
    fori: bool,
    max_windows: int,
    tile_m: int,
    tile_k: int,
    tile_n: int,
) -> jax.Array:
  return overflow_fwd_impl(
      partial,
      x_all,
      scores_flat_M,
      w1,
      w3,
      w2,
      slot_order_M,
      group_sizes_e,
      count_1,
      cap=cap,
      top_k=top_k,
      activation="gelu_tanh" if gelu_tanh else "silu",
      use_pallas=use_pallas,
      fori=fori,
      max_windows=max_windows,
      tiles=(tile_m, tile_k, tile_n),
  )


def _roe_overflow_bwd_jax(
    grad_out: jax.Array,
    x_all: jax.Array,
    scores_flat_M: jax.Array,
    w1: jax.Array,
    w3: jax.Array,
    w2: jax.Array,
    slot_order_M: jax.Array,
    group_sizes_e: jax.Array,
    count_1: jax.Array,
    cap: int,
    top_k: int,
    gelu_tanh: bool,
    use_pallas: bool,
    fori: bool,
    max_windows: int,
    tile_m: int,
    tile_k: int,
    tile_n: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  return overflow_bwd_impl(
      grad_out,
      x_all,
      scores_flat_M,
      w1,
      w3,
      w2,
      slot_order_M,
      group_sizes_e,
      count_1,
      cap=cap,
      top_k=top_k,
      activation="gelu_tanh" if gelu_tanh else "silu",
      use_pallas=use_pallas,
      fori=fori,
      max_windows=max_windows,
      tiles=(tile_m, tile_k, tile_n),
  )


# The undecorated functions stay importable so tests can register them under
# the schema jax_op derives from their annotations (Tensor/int/bool).
_roe_overflow_fwd = pallas.jax_op(
    "tt_moe::roe_overflow_fwd", _roe_overflow_fwd_jax
)
_roe_overflow_fwd_donated = pallas.jax_op(
    "tt_moe::roe_overflow_fwd_donated",
    _roe_overflow_fwd_jax,
    donate_argnums=(0,),
)
_roe_overflow_bwd = pallas.jax_op(
    "tt_moe::roe_overflow_bwd", _roe_overflow_bwd_jax
)


def overflow_env_config(num_slots: int, cap: int) -> dict:
  """The ``TT_EP_OVERFLOW_*`` settings (see the module docstring)."""
  impl = os.environ.get("TT_EP_OVERFLOW_IMPL", "while").strip().lower()
  if impl not in ("while", "fori"):
    raise ValueError(
        f"TT_EP_OVERFLOW_IMPL must be 'while' or 'fori', got {impl!r}"
    )
  worst_windows = max(-(-(num_slots - cap) // cap), 0) if cap > 0 else 0
  max_windows = int(os.environ.get("TT_EP_OVERFLOW_MAX_WINDOWS", "0"))
  if max_windows <= 0 or max_windows > worst_windows:
    max_windows = worst_windows
  tiles_str = os.environ.get("TT_EP_OVERFLOW_TILES", "0,0,0")
  tiles = tuple(int(t) for t in tiles_str.split(","))
  if len(tiles) != 3 or any(t < 0 or (t and t % 128) for t in tiles):
    raise ValueError(
        "TT_EP_OVERFLOW_TILES must be 3 comma-separated multiples of"
        f" 128 (0 = auto), got {tiles_str!r}"
    )
  donate = os.environ.get("TT_EP_OVERFLOW_DONATE", "0") == "1"
  return dict(
      fori=impl == "fori",
      max_windows=max_windows,
      tiles=tiles,
      donate=donate,
  )


class _RoEOverflowTail(torch.autograd.Function):
  """Overflow tail; backward recomputes every overflow window."""

  @staticmethod
  def forward(
      ctx,
      partial,
      x_all,
      scores_flat_M,
      w1,
      w3,
      w2,
      slot_order_M,
      group_sizes_e,
      count_1,
      cap,
      top_k,
      activation,
  ):
    env = overflow_env_config(slot_order_M.shape[0], cap)
    static = (
        cap,
        top_k,
        activation == "gelu_tanh",
        partial.device.type == "tpu",
        env["fori"],
        env["max_windows"],
        *env["tiles"],
    )
    fwd = _roe_overflow_fwd_donated if env["donate"] else _roe_overflow_fwd
    out = fwd(
        partial,
        x_all.contiguous(),
        scores_flat_M.contiguous(),
        w1,
        w3,
        w2,
        slot_order_M,
        group_sizes_e,
        count_1,
        *static,
    )
    ctx.save_for_backward(
        x_all, scores_flat_M, w1, w3, w2, slot_order_M, group_sizes_e, count_1
    )
    ctx.cfg = static
    return out

  @staticmethod
  def backward(ctx, grad_out):
    dx, ds, dw1, dw3, dw2 = _roe_overflow_bwd(
        grad_out.contiguous(), *ctx.saved_tensors, *ctx.cfg
    )
    return (grad_out, dx, ds, dw1, dw3, dw2) + (None,) * 6


def roe_overflow_tail(partial, routing, weights, activation="silu"):
  """TPU overflow tail; see ``components/ep_overflow.overflow_tail``."""
  w1, w3, w2 = weights
  return _RoEOverflowTail.apply(
      partial,
      routing.x_all_G,
      routing.scores_flat_M,
      w1,
      w3,
      w2,
      routing.slot_order_M,
      routing.group_sizes_e,
      routing.count_total.reshape(1).to(torch.int32),
      routing.cap,
      routing.top_k,
      activation,
  )
