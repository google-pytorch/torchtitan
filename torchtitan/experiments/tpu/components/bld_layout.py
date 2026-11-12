"""Run dense decoder blocks on [B, L, ...] views on TPU.

TT-UNFLATTEN-SHIM companion. Upstream d771abb0 folded the [B, L] token dims
into one T dim, so every activation inside a transformer block became [T, D] /
[T, H, K]. XLA's TPU layout assignment then picks T-minor layouts for the FP32
RMSNorm outputs and for q/k around RoPE. On qwen3-8b/v6e-8 that added ~260 ms
of transposes and ~90 ms of slower dW matmuls per two steps (-9% TPS); the
matmul count and the recompute are unchanged.

``apply_bld_layout`` fixes this after the model is built, by swapping the class
of eligible modules for a subclass that views [T, D] as [B, L, D] using the
unflatten sequence length:

* ``Llama3TransformerBlock`` / ``Qwen3TransformerBlock`` (dense or MoE): view
  the input as [B, L, D], run the block's own ``forward``, view back. The
  block's external [T, D] contract is unchanged.
* ``GQAttention`` with ``QKVLinear``, ``FusedQKVLinear`` or
  ``TpuPackedQKVLinear``: accept [B, L, D]; q/k/v are [B, L, H, K],
  the RoPE cache is fetched by the configured RoPE module and viewed as
  [B, L, 1, W], and the rotation is that module's own ``apply_rotary_emb`` --
  so ``real_rope`` and ``bf16_rope`` keep working. Exact ``CosSinRoPE`` uses a
  rotate-half that does not concatenate the FP32 q/k halves (same FP32
  products and sums; bit-exact on CPU).
* MoE (any ``feed_forward``-less block's ``moe``): flatten [B, L, D] to [T, D]
  around the unchanged MoE ``forward``.

Modules whose ``forward`` is not the known upstream implementation (MLA, custom
blocks, custom QKV projections, RoPE subclasses overriding ``forward``) are left
alone, and so is any block containing them. Outside the TPU trainer (no
unflatten sequence length) every patched module runs its upstream [T] path.

The math is unchanged, but the matmuls and reductions see different shapes, so
results match the [T] path to reduction-order rounding, not bit for bit.

Because this swaps built modules rather than claiming config nodes, it composes
with every ``@override``, including the RoPE ones, which a block-level override
cannot (nested overrides are rejected).

REMOVAL: keep while [T] graphs get slower than [B, L] ones due to XLA layout
heuristics. Investigate if we can control or update these heuristics and
re-measure (bld_block_layout on vs off) before deleting.
"""

import torch
import torch.nn.functional as F
import torch_remat as remat
from torch import nn

from torchtitan.experiments.tpu.components.qkv import TpuPackedQKVLinear
from torchtitan.experiments.tpu.components.unflatten_shim import UnflattenShim
from torchtitan.experiments.tpu.components.unflatten_shim import get_unflatten_seq_len
from torchtitan.models.common import CosSinRoPE
from torchtitan.models.common import FusedQKVLinear
from torchtitan.models.common import GQAttention
from torchtitan.models.common import QKVLinear
from torchtitan.models.common.rope import RoPE
from torchtitan.models.llama3.model import Llama3TransformerBlock
from torchtitan.models.qwen3.model import Qwen3TransformerBlock
from torchtitan.tools.logging import logger


def _bld_seq_len(x: torch.Tensor) -> int | None:
  """The L of a [T, D] activation's [B, L] split, or None for the [T] path."""
  seq_len = get_unflatten_seq_len()
  if x.ndim != 2 or seq_len is None or x.shape[0] % seq_len != 0:
    return None
  return seq_len


def cossin_rotate(
    x: torch.Tensor, cos_h: torch.Tensor, sin_h: torch.Tensor
) -> torch.Tensor:
  """``CosSinRoPE`` rotate-half on half-width FP32 cos/sin, without ``cat`` on x.

  ``CosSinRoPE`` computes ``x * cat(c, c) + cat(-x2, x1) * cat(s, s)`` in FP32,
  i.e. per half ``x1*c + (-x2)*s`` and ``x2*c + x1*s``. Computing the halves
  directly performs the same FP32 operations; only the activation-dtype
  outputs are concatenated.
  """
  h = x.shape[-1] // 2
  xf = x.float()
  x1, x2 = xf[..., :h], xf[..., h:]
  o1 = x1 * cos_h + (-x2) * sin_h
  o2 = x2 * cos_h + x1 * sin_h
  return torch.cat((o1.type_as(x), o2.type_as(x)), dim=-1)


class _BLDGQAttention(GQAttention):
  """``GQAttention`` that also accepts, and then returns, [B, L, D]."""

  def _bld_qkv(self, x: torch.Tensor):
    b, l, _ = x.shape
    qkv = self.qkv_linear
    hd = qkv.head_dim
    # isinstance, not `type(...) is`: SimpleFSDP later swaps parametrized
    # modules to generated subclasses (e.g. `SimpleFSDPQKVLinear_7`).
    # `_attention_eligible` already checked each one keeps the upstream forward.
    if isinstance(qkv, QKVLinear):
      return tuple(
          proj(x).view(b, l, -1, hd) for proj in (qkv.wq, qkv.wk, qkv.wv)
      )
    if isinstance(qkv, TpuPackedQKVLinear):
      # Same single-split projection as `TpuPackedQKVLinear.forward`.
      packed = qkv.wqkv_packed
      sizes = (qkv.q_rows, qkv.kv_rows, qkv.kv_rows)
      weights = torch.split(packed.weight, sizes, dim=0)
      biases = (
          (None,) * 3
          if packed.bias is None
          else torch.split(packed.bias, sizes, dim=0)
      )
      return tuple(
          F.linear(x, w, bias).view(b, l, -1, hd)
          for w, bias in zip(weights, biases)
      )
    assert isinstance(qkv, FusedQKVLinear), type(qkv)
    # Same split as `FusedQKVLinear.forward`, with [B, L] leading dims:
    # [B, L, n_kv_heads * R * hd] -> [B, L, n_kv_heads, R, hd] -> q/k/v.
    fused = qkv.wqkv(x).view(b, l, -1, qkv.r_dim, hd)
    xq, xk, xv = torch.split(fused, [qkv.heads_per_kv, 1, 1], dim=-2)
    return (
        xq.reshape(b, l, -1, hd),
        xk.reshape(b, l, -1, hd),
        xv.reshape(b, l, -1, hd),
    )

  def _bld_rope(self, xq, xk, positions):
    b, l, _, head_dim = xq.shape
    # Upstream [T, 1, W] cache (gathered by positions) -> [B, L, 1, W].
    cache = self.rope._reshape_cache(xq.reshape(b * l, -1, head_dim), positions)
    cache = cache.view(b, l, *cache.shape[1:])
    if type(self.rope).apply_rotary_emb is CosSinRoPE.apply_rotary_emb:
      cos_h = cache[..., : head_dim // 2]
      sin_h = cache[..., head_dim : head_dim + head_dim // 2]
      return cossin_rotate(xq, cos_h, sin_h), cossin_rotate(xk, cos_h, sin_h)
    return self.rope.apply_rotary_emb(xq, xk, cache)

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if x_TD.ndim != 3:
      return super().forward(x_TD, attention_masks, positions, segment_ids)
    b, l, _ = x_TD.shape
    xq, xk, xv = remat.region(
        self._bld_qkv,
        self.remat_region_name("qkv"),
        recompute=self.remat_should_recompute("qkv"),
    )(x_TD)
    remat.recompute_needs_tensor(xq, xk, xv)

    if self.q_norm is not None or self.k_norm is not None:
      assert self.q_norm is not None and self.k_norm is not None
      xq = self.q_norm(xq)
      xk = self.k_norm(xk)

    if self.rope is not None:
      xq, xk = self._bld_rope(xq, xk, positions)

    # The TT-UNFLATTEN-SHIM passes 4-D q/k/v through (and unflattens the 1-D
    # positions/segment_ids). Other inner attentions take upstream [T, H, K].
    if not isinstance(self.inner_attention, UnflattenShim):
      xq, xk, xv = (t.reshape(b * l, *t.shape[2:]) for t in (xq, xk, xv))
    out = remat.region(
        self.inner_attention,
        self.remat_region_name("inner_attention"),
        recompute=self.remat_should_recompute("inner_attention"),
    )(
        xq,
        xk,
        xv,
        attention_masks=attention_masks,
        scale=self.scaling,
        enable_gqa=self.enable_gqa,
        positions=positions,
        segment_ids=segment_ids,
    )
    remat.recompute_needs_tensor(out)
    out = remat.region(
        self.wo,
        self.remat_region_name("wo"),
        recompute=self.remat_should_recompute("wo"),
    )(out.reshape(b, l, -1))
    remat.recompute_needs_tensor(out)
    return out


class _BLDBlock(nn.Module):
  """Mixin: compute a [T, D]-in/[T, D]-out block on a [B, L, D] view."""

  def forward(self, x, attention_masks, positions=None, segment_ids=None):
    seq_len = _bld_seq_len(x)
    if seq_len is None:
      return super().forward(x, attention_masks, positions, segment_ids)
    num_tokens, dim = x.shape
    out = super().forward(
        x.view(num_tokens // seq_len, seq_len, dim),
        attention_masks,
        positions,
        segment_ids,
    )
    return out.reshape(num_tokens, dim)


class _BLDMoE(nn.Module):
  """Mixin: run a [T, D] MoE on the flattened tokens of a [B, L, D] input."""

  def forward(self, x, *args, **kwargs):
    if x.ndim != 3:
      return super().forward(x, *args, **kwargs)
    b, l, d = x.shape
    out = super().forward(x.reshape(b * l, d), *args, **kwargs)
    return out.reshape(b, l, -1)


_BLOCK_FORWARDS = (Llama3TransformerBlock.forward, Qwen3TransformerBlock.forward)
_QKV_FORWARDS = (
    QKVLinear.forward,
    FusedQKVLinear.forward,
    TpuPackedQKVLinear.forward,
)
_SUBCLASSES: dict[tuple[type, type], type] = {}


def _swap_class(module: nn.Module, mixin: type) -> None:
  cls = type(module)
  if issubclass(cls, mixin):
    return
  key = (mixin, cls)
  if key not in _SUBCLASSES:
    new_cls = type(f"TpuBLD{cls.__name__}_{len(_SUBCLASSES)}", (mixin, cls), {})
    # Make the generated class importable by name, as SimpleFSDP does for its
    # own wrappers, so pickle / the graph trainer's GraphPickler resolve it.
    new_cls.__module__ = __name__
    globals()[new_cls.__name__] = new_cls
    _SUBCLASSES[key] = new_cls
  module.__class__ = _SUBCLASSES[key]


def _class_forward(module: nn.Module):
  """The module's class ``forward``, or None if an instance attribute shadows it.

  Wrappers such as the graph trainer's `annotate_module_fqns` or
  `apply_spmd_safe` bind ``module.forward`` on the instance; a later class swap
  would not take effect, so such modules must be patched before they run.
  """
  if "forward" in vars(module):
    return None
  return type(module).forward


def _attention_eligible(attn: nn.Module) -> bool:
  # Class-forward identity rather than exact types: SimpleFSDP swaps
  # parametrized modules to generated subclasses that keep the forward.
  return (
      isinstance(attn, GQAttention)
      and _class_forward(attn) in (GQAttention.forward, _BLDGQAttention.forward)
      and _class_forward(attn.qkv_linear) in _QKV_FORWARDS
      and (attn.rope is None or type(attn.rope).forward is RoPE.forward)
  )


def bld_skip_reason(parallelism) -> str | None:
  """Why the [B, L] layout must stay off for this recipe, or None."""
  if (
      parallelism.context_parallel_degree > 1
      or parallelism.tensor_parallel_degree > 1
  ):
    # CP/TP shard the token dim, so a local [T] is not B whole sequences.
    return "context or tensor parallelism shards the token dim"
  return None


def apply_bld_layout(
    model: nn.Module, *, skip_reason: str | None = None
) -> int:
  """Patch eligible dense blocks in ``model`` in place; returns how many.

  Args:
    model: The built model.
    skip_reason: From ``bld_skip_reason``; if set, patch nothing and log it.
  """
  if skip_reason is not None:
    logger.info(f"TPU [B, L] block layout skipped: {skip_reason}")
    return 0
  patched = 0
  for module in list(model.modules()):
    if not isinstance(module, (Llama3TransformerBlock, Qwen3TransformerBlock)):
      continue
    if isinstance(module, _BLDBlock):
      patched += 1
      continue
    if _class_forward(module) not in _BLOCK_FORWARDS:
      continue
    if not _attention_eligible(module.attention):
      continue
    if getattr(module, "moe_enabled", False):
      if _class_forward(module.moe) is None:
        continue
      _swap_class(module.moe, _BLDMoE)
    if type(module.attention) is GQAttention:
      module.attention.__class__ = _BLDGQAttention
    else:
      _swap_class(module.attention, _BLDGQAttention)
    _swap_class(module, _BLDBlock)
    patched += 1
  logger.info(f"TPU [B, L] block layout applied to {patched} transformer blocks")
  return patched


__all__ = ["apply_bld_layout", "bld_skip_reason", "cossin_rotate"]
