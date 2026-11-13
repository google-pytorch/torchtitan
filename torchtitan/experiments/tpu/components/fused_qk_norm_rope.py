"""Patch attention modules to run fused Pallas QK RMSNorm + CosSinRoPE on TPU.

Dimension legend (for shape-suffixed tensor names in this module):
  b: local batch size
  l: sequence length per example
  t: folded token count (b * l, or t / tp under sequence parallelism)
  d: model hidden dimension
  h: number of local query or key/value attention heads
  k: query/key head dimension (head_dim)
  v: value head dimension
"""

import importlib
import sys
from typing import Any

import torch
from torch import nn
import torch_remat as remat
from torchtitan.experiments.tpu.components import bld_layout
from torchtitan.models.common import CosSinRoPE
from torchtitan.models.common import GQAttention
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.rope import RoPE
from torchtitan.tools.logging import logger

__all__ = [
    "apply_fused_qk_norm_rope",
]

_SUBCLASSES: dict[tuple[type[Any], type[Any]], type[Any]] = {}
_UNSET = object()
_qk_norm_rope_mod: Any = _UNSET


def _get_qk_norm_rope_module() -> Any:
  """Lazily imports ``kernels.qk_norm_rope`` on first use."""
  global _qk_norm_rope_mod
  if _qk_norm_rope_mod is _UNSET:
    try:
      _qk_norm_rope_mod = importlib.import_module(
          "torchtitan.experiments.tpu.kernels.qk_norm_rope"
      )
    except ImportError as exc:
      logger.warning(
          "Pallas qk_norm_rope unavailable (%s); falling back to XLA.", exc
      )
      _qk_norm_rope_mod = None
  return _qk_norm_rope_mod


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
    new_cls = type(
        f"TpuFusedQkNormRoPE{cls.__name__}_{len(_SUBCLASSES)}",
        (mixin, cls),
        {},
    )
    new_cls.__module__ = __name__
    globals()[new_cls.__name__] = new_cls
    _SUBCLASSES[key] = new_cls
  module.__class__ = _SUBCLASSES[key]


def _try_fused_qk_norm_rope(
    attn: nn.Module,
    xq: torch.Tensor,
    xk: torch.Tensor,
    positions: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
  """Runs fused QK-norm + CosSinRoPE, or returns None when unsupported."""
  kernel_mod = _get_qk_norm_rope_module()
  if kernel_mod is None:
    return None
  head_dim = xq.shape[-1]
  if xq.ndim == 4:
    batch, seq_len, _, _ = xq.shape
    cache = attn.rope._reshape_cache(  # pylint: disable=protected-access
        xq.reshape(batch * seq_len, -1, head_dim), positions
    )
    cache = cache.view(batch, seq_len, *cache.shape[1:])
  else:
    cache = attn.rope._reshape_cache(xq, positions)  # pylint: disable=protected-access
  if cache.shape[-1] != 2 * head_dim:
    return None
  cos_h = cache[..., : head_dim // 2]
  sin_h = cache[..., head_dim : head_dim + head_dim // 2]
  if not (
      kernel_mod.supports(xq, attn.q_norm.weight, cos_h, sin_h)
      and kernel_mod.supports(xk, attn.k_norm.weight, cos_h, sin_h)
  ):
    return None
  return (
      kernel_mod.qk_norm_rope(
          xq, attn.q_norm.weight, cos_h, sin_h, attn.q_norm.eps
      ),
      kernel_mod.qk_norm_rope(
          xk, attn.k_norm.weight, cos_h, sin_h, attn.k_norm.eps
      ),
  )


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


class _FusedGQAttentionMixin(nn.Module):
  """Mixin fusing QK-norm + CosSinRoPE on ``GQAttention`` / ``_BLDGQAttention``."""

  def _apply_qk_norm_and_rope(
      self,
      xq_blhk: torch.Tensor,
      xk_blhk: torch.Tensor,
      positions: torch.Tensor | None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    fused = _try_fused_qk_norm_rope(self, xq_blhk, xk_blhk, positions)
    if fused is not None:
      return fused
    return super()._apply_qk_norm_and_rope(xq_blhk, xk_blhk, positions)

  def forward(
      self,
      x: torch.Tensor,
      attention_masks: AttentionMasksType | None,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if x.ndim != 2:
      return super().forward(x, attention_masks, positions, segment_ids)
    xq_thk, xk_thk, xv_thv = remat.region(
        self.qkv_linear,
        self.remat_region_name("qkv"),
        recompute=self.remat_should_recompute("qkv"),
    )(x)
    remat.recompute_needs_tensor(xq_thk, xk_thk, xv_thv)
    fused = _try_fused_qk_norm_rope(self, xq_thk, xk_thk, positions)
    if fused is not None:
      xq_thk, xk_thk = fused
    else:
      assert self.q_norm is not None and self.k_norm is not None
      assert self.rope is not None
      xq_thk = self.q_norm(xq_thk)
      xk_thk = self.k_norm(xk_thk)
      xq_thk, xk_thk = self.rope(xq_thk, xk_thk, positions)
    out_thv = remat.region(
        self.inner_attention,
        self.remat_region_name("inner_attention"),
        recompute=self.remat_should_recompute("inner_attention"),
    )(
        xq_thk,
        xk_thk,
        xv_thv,
        attention_masks=attention_masks,
        scale=self.scaling,
        enable_gqa=self.enable_gqa,
        positions=positions,
        segment_ids=segment_ids,
    )
    remat.recompute_needs_tensor(out_thv)
    out_td = out_thv.contiguous().view(out_thv.shape[0], -1)
    out_td = remat.region(
        self.wo,
        self.remat_region_name("wo"),
        recompute=self.remat_should_recompute("wo"),
    )(out_td)
    remat.recompute_needs_tensor(out_td)
    return out_td


class _FusedGemma4AttentionMixin(nn.Module):
  """Mixin fusing QK-norm + CosSinRoPE inside ``Gemma4Attention``."""

  def forward(
      self,
      x_td: torch.Tensor,
      attention_masks: AttentionMasksType | None,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    xq_thk, xk_thk, xv_thv = self.qkv_linear(x_td)
    fused = _try_fused_qk_norm_rope(self, xq_thk, xk_thk, positions)
    if fused is not None:
      xq_thk, xk_thk = fused
      xv_thv = self.v_norm(xv_thv)
    else:
      assert self.q_norm is not None and self.k_norm is not None
      assert self.rope is not None
      xq_thk = self.q_norm(xq_thk)
      xk_thk = self.k_norm(xk_thk)
      xv_thv = self.v_norm(xv_thv)
      xq_thk, xk_thk = self.rope(xq_thk, xk_thk, positions)

    if isinstance(attention_masks, dict):
      mask = (
          attention_masks["sliding_window_mask"]
          if self.is_sliding
          else attention_masks["basic_mask"]
      )
    else:
      mask = attention_masks

    out_thv = self.inner_attention(
        xq_thk,
        xk_thk,
        xv_thv,
        attention_masks=mask,
        scale=self.scaling,
        enable_gqa=self.enable_gqa,
        positions=positions,
        segment_ids=segment_ids,
    ).contiguous()
    return self.wo(out_thv.view(out_thv.shape[0], -1))


def apply_fused_qk_norm_rope(model: nn.Module) -> int:
  """Enables fused Pallas QK-norm + CosSinRoPE on eligible attention modules.

  Args:
    model: Built model whose ``GQAttention`` / ``Gemma4Attention`` submodules
      are inspected and patched in place.

  Returns:
    Number of eligible attention modules patched (or already patched).
  """
  if _get_qk_norm_rope_module() is None:
    return 0

  gemma4_mod = sys.modules.get(
      "torchtitan.experiments.tpu.examples.gemma4.model"
  )
  gemma4_attn_cls = (
      getattr(gemma4_mod, "Gemma4Attention", None)
      if gemma4_mod is not None
      else None
  )
  bld_attn_fwd = bld_layout._BLDGQAttention.forward  # pylint: disable=protected-access

  patched = 0
  for attn in list(model.modules()):
    if not isinstance(attn, GQAttention):
      continue
    if not _fused_qk_norm_rope_eligible(attn):
      continue
    if isinstance(attn, (_FusedGQAttentionMixin, _FusedGemma4AttentionMixin)):
      patched += 1
      continue
    fwd = _class_forward(attn)
    if fwd in (GQAttention.forward, bld_attn_fwd):
      _swap_class(attn, _FusedGQAttentionMixin)
      patched += 1
    elif (
        gemma4_attn_cls is not None
        and isinstance(attn, gemma4_attn_cls)
        and fwd is gemma4_attn_cls.forward
    ):
      _swap_class(attn, _FusedGemma4AttentionMixin)
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
