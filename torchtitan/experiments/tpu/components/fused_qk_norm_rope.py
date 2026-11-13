"""Patch attention modules to run fused Pallas QK RMSNorm + CosSinRoPE on TPU.

Dimension legend (for shape-suffixed tensor names in this module):
  b: local batch size
  l: sequence length per example
  h: number of local query or key/value attention heads
  k: query/key head dimension (head_dim)
"""

from collections import Counter
import sys
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torchtitan.experiments.tpu.components import bld_layout
from torchtitan.experiments.tpu.kernels import qk_norm_rope as qk_norm_rope_kernel
from torchtitan.models.common import CosSinRoPE
from torchtitan.models.common import GQAttention
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.rope import RoPE
from torchtitan.tools.logging import logger

__all__ = [
    "apply_fused_qk_norm_rope",
]

_SUBCLASSES: dict[tuple[type[Any], type[Any]], type[Any]] = {}


def _class_forward(module: nn.Module) -> Any:
  """Returns ``type(module).forward``, or None if shadowed on the instance."""
  if "forward" in vars(module):
    return None
  return type(module).forward


def _swap_class(module: nn.Module, mixin: type[Any]) -> None:
  """Swaps ``module.__class__`` to a pickleable ``(mixin, cls)`` subclass."""
  cls = type(module)
  if issubclass(cls, mixin):
    return
  key = (mixin, cls)
  if key not in _SUBCLASSES:
    cls_name = (
        f"{mixin.__name__}_{cls.__module__}_{cls.__qualname__}".replace(
            ".", "_"
        )
    )
    new_cls = type(cls_name, (mixin, cls), {})
    new_cls.__module__ = __name__
    globals()[new_cls.__name__] = new_cls
    _SUBCLASSES[key] = new_cls
  module.__class__ = _SUBCLASSES[key]


def _reshape_rope_cache(
    rope: nn.Module,
    xq: torch.Tensor,
    positions: torch.Tensor | None,
) -> torch.Tensor:
  """Returns RoPE cache aligned to 3-D ``[T, ...]`` or 4-D ``[B, L, ...]``."""
  if xq.ndim == 4:
    batch, seq_len, _, head_dim = xq.shape
    cache = rope._reshape_cache(  # pylint: disable=protected-access
        xq.reshape(batch * seq_len, -1, head_dim), positions
    )
    return cache.view(batch, seq_len, *cache.shape[1:])
  return rope._reshape_cache(xq, positions)  # pylint: disable=protected-access


def _try_fused_qk_norm_rope(
    rope: nn.Module,
    q_norm: nn.RMSNorm,
    k_norm: nn.RMSNorm,
    xq: torch.Tensor,
    xk: torch.Tensor,
    positions: torch.Tensor | None,
    *,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
  """Runs fused QK-norm + CosSinRoPE, or returns None when unsupported."""
  if xq.ndim not in (3, 4) or xk.ndim != xq.ndim:
    return None
  head_dim = xq.shape[-1]
  if xk.shape[-1] != head_dim or head_dim < 2:
    return None
  if cache is None:
    cache = _reshape_rope_cache(rope, xq, positions)
  if cache.shape[-1] != 2 * head_dim:
    return None
  # Read each weight once (SimpleFSDP exposes weight as an all-gathering
  # @property).
  if q_weight is None:
    q_weight = q_norm.weight
  if k_weight is None:
    k_weight = k_norm.weight
  cos_h = cache[..., : head_dim // 2]
  sin_h = cache[..., head_dim : head_dim + head_dim // 2]
  if not (
      qk_norm_rope_kernel.supports(xq, q_weight, cos_h, sin_h)
      and qk_norm_rope_kernel.supports(xk, k_weight, cos_h, sin_h)
  ):
    return None
  return (
      qk_norm_rope_kernel.qk_norm_rope(
          xq, q_weight, cos_h, sin_h, q_norm.eps
      ),
      qk_norm_rope_kernel.qk_norm_rope(
          xk, k_weight, cos_h, sin_h, k_norm.eps
      ),
  )


def _apply_qk_norm_and_rope_with_fallback(
    rope: nn.Module,
    q_norm: nn.RMSNorm,
    k_norm: nn.RMSNorm,
    xq: torch.Tensor,
    xk: torch.Tensor,
    positions: torch.Tensor | None,
    *,
    bld: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Runs fused QK-norm + RoPE, or unfused fallback with one weight read each."""
  cache: torch.Tensor | None = None
  q_weight: torch.Tensor | None = None
  k_weight: torch.Tensor | None = None
  if (
      xq.ndim in (3, 4)
      and xk.ndim == xq.ndim
      and xq.shape[-1] == xk.shape[-1]
      and xq.shape[-1] >= 2
  ):
    cache = _reshape_rope_cache(rope, xq, positions)
    if cache.shape[-1] == 2 * xq.shape[-1]:
      q_weight = q_norm.weight
      k_weight = k_norm.weight
      fused = _try_fused_qk_norm_rope(
          rope,
          q_norm,
          k_norm,
          xq,
          xk,
          positions,
          q_weight=q_weight,
          k_weight=k_weight,
          cache=cache,
      )
      if fused is not None:
        return fused

  if q_weight is None:
    q_weight = q_norm.weight
  if k_weight is None:
    k_weight = k_norm.weight
  xq = F.rms_norm(xq, q_norm.normalized_shape, q_weight, q_norm.eps)
  xk = F.rms_norm(xk, k_norm.normalized_shape, k_weight, k_norm.eps)
  if cache is None:
    cache = _reshape_rope_cache(rope, xq, positions)
  if (
      bld
      and type(rope).apply_rotary_emb is CosSinRoPE.apply_rotary_emb
      and cache.shape[-1] == 2 * xq.shape[-1]
  ):
    head_dim = xq.shape[-1]
    cos_h = cache[..., : head_dim // 2]
    sin_h = cache[..., head_dim : head_dim + head_dim // 2]
    return (
        bld_layout.cossin_rotate(xq, cos_h, sin_h),
        bld_layout.cossin_rotate(xk, cos_h, sin_h),
    )
  return rope.apply_rotary_emb(xq, xk, cache)


def _fused_qk_norm_rope_eligible(attn: nn.Module) -> bool:
  """Returns whether ``attn`` uses standard ``RMSNorm`` Q/K norms + ``CosSinRoPE``."""
  norms = (getattr(attn, "q_norm", None), getattr(attn, "k_norm", None))
  rope = getattr(attn, "rope", None)
  return (
      rope is not None
      and _class_forward(rope) is RoPE.forward
      and "apply_rotary_emb" not in vars(rope)
      and type(rope).apply_rotary_emb is CosSinRoPE.apply_rotary_emb
      and all(
          isinstance(norm, nn.RMSNorm)
          and _class_forward(norm) is nn.RMSNorm.forward
          # LayerStack (`use_scan_layers=True`) deletes `"weight"` from
          # `norm._parameters` before `apply_patches` runs and restores it via
          # `functional_call` during execution; check `elementwise_affine` too.
          and getattr(norm, "elementwise_affine", False)
          and getattr(norm, "weight", True) is not None
          and norm.eps is not None
          for norm in norms
      )
  )


class _FusedBLDGQAttentionMixin(nn.Module):
  """Mixin fusing QK-norm + CosSinRoPE on ``_BLDGQAttention`` via its hook."""

  def _apply_qk_norm_and_rope(
      self,
      xq_blhk: torch.Tensor,
      xk_blhk: torch.Tensor,
      positions: torch.Tensor | None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return _apply_qk_norm_and_rope_with_fallback(
        self.rope,
        self.q_norm,
        self.k_norm,
        xq_blhk,
        xk_blhk,
        positions,
        bld=True,
    )


class _PassthroughRMSNormMixin(nn.Module):
  """Defers Q/K RMSNorm to ``_FusedQkNormRoPEMixin`` without copying forward."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x


class _FusedQkNormRoPEMixin(nn.Module):
  """Runs deferred Q/K RMSNorm + CosSinRoPE in one fused Pallas kernel."""

  _fused_qk_norms: tuple[nn.RMSNorm, nn.RMSNorm]

  def forward(
      self,
      query: torch.Tensor,
      key: torch.Tensor | None = None,
      positions: torch.Tensor | None = None,
      *,
      inverse: bool = False,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if key is None:
      raise RuntimeError(
          "Fused QK-norm + RoPE requires both query and key tensors; got"
          " key=None after deferring q_norm/k_norm."
      )
    q_norm, k_norm = self._fused_qk_norms
    if inverse:
      query = F.rms_norm(
          query, q_norm.normalized_shape, q_norm.weight, q_norm.eps
      )
      key = F.rms_norm(key, k_norm.normalized_shape, k_norm.weight, k_norm.eps)
      return super().forward(query, key, positions, inverse=inverse)
    return _apply_qk_norm_and_rope_with_fallback(
        self, q_norm, k_norm, query, key, positions, bld=False
    )


def _is_patched(attn: nn.Module) -> bool:
  """Returns whether ``attn`` is already patched for fused QK-norm + RoPE."""
  if isinstance(attn, _FusedBLDGQAttentionMixin):
    return True
  rope = getattr(attn, "rope", None)
  return isinstance(rope, _FusedQkNormRoPEMixin) and getattr(
      rope, "_fused_qk_norms", None
  ) == (getattr(attn, "q_norm", None), getattr(attn, "k_norm", None))


def apply_fused_qk_norm_rope(model: nn.Module) -> int:
  """Enables fused Pallas QK-norm + CosSinRoPE on eligible attention modules.

  Args:
    model: Built model whose ``GQAttention`` / ``Gemma4Attention`` submodules
      are inspected and patched in place.

  Returns:
    Number of eligible attention modules patched (or already patched).
  """
  gemma4_mod = sys.modules.get(
      "torchtitan.experiments.tpu.examples.gemma4.model"
  )
  gemma4_attn_cls = (
      getattr(gemma4_mod, "Gemma4Attention", None)
      if gemma4_mod is not None
      else None
  )
  bld_attn_fwd = bld_layout._BLDGQAttention.forward  # pylint: disable=protected-access

  modules = list(model.modules())
  rope_counts = Counter(
      id(mod.rope)
      for mod in modules
      if isinstance(mod, BaseAttention)
      and getattr(mod, "rope", None) is not None
  )

  patched = 0
  for attn in modules:
    if not isinstance(attn, GQAttention):
      continue
    if _is_patched(attn):
      patched += 1
      continue
    if not _fused_qk_norm_rope_eligible(attn):
      continue
    fwd = _class_forward(attn)
    if fwd is bld_attn_fwd:
      _swap_class(attn, _FusedBLDGQAttentionMixin)
      patched += 1
    elif fwd is GQAttention.forward or (
        gemma4_attn_cls is not None
        and isinstance(attn, gemma4_attn_cls)
        and fwd is gemma4_attn_cls.forward
    ):
      if rope_counts[id(attn.rope)] > 1:
        continue
      _swap_class(attn.q_norm, _PassthroughRMSNormMixin)
      _swap_class(attn.k_norm, _PassthroughRMSNormMixin)
      _swap_class(attn.rope, _FusedQkNormRoPEMixin)
      # Store as a plain tuple (not a submodule) so state_dict() and
      # named_parameters() stay identical while LayerStack functional_call
      # still injects weights onto attn.q_norm / attn.k_norm.
      attn.rope._fused_qk_norms = (attn.q_norm, attn.k_norm)  # pylint: disable=protected-access
      patched += 1

  if patched == 0:
    logger.warning(
        "tpu_config.fused_qk_norm_rope=True was requested, but 0 eligible"
        " attention modules with RMSNorm q_norm/k_norm and exact CosSinRoPE"
        " were found."
    )
  else:
    logger.info(
        "TPU fused QK-norm + RoPE enabled on %d attention modules", patched
    )
  return patched
