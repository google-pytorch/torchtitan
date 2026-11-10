"""Sliding-window-aware SDPA inner attention for Muse Glimmer on TPU.

Upstream Muse Glimmer only implements its per-layer sliding windows for the
``flex`` (window-keyed ``BlockMask`` dict) and ``varlen`` (FA3 kernel arg)
backends. Neither runs on TPU, and upstream's
``ScaledDotProductInnerAttention`` silently ignores the window, so every
sliding-window layer would run as full causal attention.

``SlidingWindowScaledDotProductInnerAttention`` carries the layer's window and
applies it as an explicit causal sliding-window mask to SDPA. It also exposes
``sliding_window_size`` for ``workarounds.use_splash_attention_patch``, which
matches ``ScaledDotProduct*`` inner attentions and uses that attribute to pick
Splash's ``LocalMask``.

Window semantics (the two conventions differ by one):

* Muse Glimmer ``window_size=W``: a query attends to itself and ``W - 1``
  previous tokens, i.e. ``0 <= q - k < W`` (``get_sliding_window_mask_mod``).
* Splash ``local_window_size=L``: ``LocalMask(window_size=(L, 0))`` attends to
  ``L`` previous tokens plus itself, i.e. ``0 <= q - k <= L``.

So ``sliding_window_size == window_size - 1``.

Packed documents: the dataloader packs several documents per row and marks
each document start with ``positions == 0``. Upstream masks documents only
through flex ``attention_masks``, and its ``Attention.forward`` never passes
``positions`` to the inner attention, so on TPU nothing would stop attention
across documents. ``TpuMuseGlimmerAttention`` forwards ``positions``; the SDPA
here ANDs its causal/window mask with a same-document mask, and the Splash /
Tokamax patches turn ``positions`` into ``segment_ids``.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.attention import ScaledDotProductInnerAttention
from torchtitan.models.common.attention import segment_ids_from_positions
from torchtitan.models.muse_glimmer.model import Attention


def sliding_window_causal_mask(
    seq_len: int, window_size: int | None, device: torch.device
) -> torch.Tensor:
  """Returns a ``[seq_len, seq_len]`` bool mask with ``0 <= q - k < window``.

  ``window_size=None`` gives the plain causal mask.
  """
  q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
  kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)
  distance = q_idx - kv_idx
  mask = distance >= 0
  if window_size is not None:
    mask &= distance < window_size
  return mask


def packed_sliding_window_causal_mask(
    seq_len: int,
    window_size: int | None,
    device: torch.device,
    *,
    segment_ids_BL: torch.Tensor | None = None,
) -> torch.Tensor:
  """Causal (optionally windowed) mask restricted to each packed document.

  Args:
    seq_len: Row length ``L``.
    window_size: Muse Glimmer window, or None for a global layer.
    device: Device for the mask.
    segment_ids_BL: Per-token document index ``[B, L]``. Only equality within a
      row matters, so IDs computed on the folded ``[T]`` stream (which keep
      counting across rows) are fine.

  Returns:
    ``[L, L]`` without ``segment_ids_BL``, else ``[B, 1, L, L]`` (broadcast over
    heads). Every query keeps itself, so no row is fully masked.
  """
  mask = sliding_window_causal_mask(seq_len, window_size, device)
  if segment_ids_BL is None:
    return mask
  same_document_BLL = segment_ids_BL.unsqueeze(2) == segment_ids_BL.unsqueeze(1)
  return (mask.unsqueeze(0) & same_document_BLL).unsqueeze(1)


def _unflatten_to_4d(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
  """TT-UNFLATTEN-SHIM: reshapes folded ``[T, H, D]`` to ``[B, L, H, D]``.

  Mirrors ``ScaledDotProductInnerAttention.forward``: the sequence length comes
  from ``set_unflatten_seq_len``; without it the whole token stream is treated
  as one sequence.
  """
  if q.ndim != 3:
    return q, k, v, False
  seq_len = None
  try:
    from torchtitan.experiments.tpu.components.unflatten_shim import get_unflatten_seq_len  # pylint: disable=g-import-not-at-top

    seq_len = get_unflatten_seq_len()
  except ImportError:
    pass
  num_tokens = q.shape[0]
  if seq_len is None or num_tokens % seq_len != 0:
    seq_len = num_tokens
  batch_size = num_tokens // seq_len
  return (
      q.reshape(batch_size, seq_len, *q.shape[1:]),
      k.reshape(batch_size, seq_len, *k.shape[1:]),
      v.reshape(batch_size, seq_len, *v.shape[1:]),
      True,
  )


class SlidingWindowScaledDotProductInnerAttention(
    ScaledDotProductInnerAttention
):
  """SDPA inner attention with an optional causal sliding window.

  ``window_size=None`` is a global (full causal) layer. Given ``positions`` (or
  ``segment_ids``), attention is also restricted to the query's own packed
  document. A global layer without either defers to the upstream
  implementation unchanged.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(ScaledDotProductInnerAttention.Config):
    # Muse Glimmer convention: attend to self + (window_size - 1) prior tokens.
    window_size: int | None = None

  def __init__(self, config: Config) -> None:
    super().__init__(config)
    if config.window_size is not None and config.window_size < 1:
      raise ValueError(
          f"window_size must be >= 1 or None, got {config.window_size}"
      )
    self.window_size = config.window_size

  @property
  def sliding_window_size(self) -> int | None:
    """Window in Splash ``LocalMask`` convention (see module docstring)."""
    if self.window_size is None:
      return None
    return self.window_size - 1

  def forward(
      self,
      q_BLHK: torch.Tensor,
      k_BLHK: torch.Tensor,
      v_BLHV: torch.Tensor,
      *,
      attention_masks: AttentionMasksType | None = None,
      scale: float | None = None,
      enable_gqa: bool = False,
      is_causal: bool = True,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
      **kwargs,
  ) -> torch.Tensor:
    if segment_ids is None and positions is not None:
      segment_ids = segment_ids_from_positions(positions)
    if self.window_size is None and segment_ids is None:
      return super().forward(
          q_BLHK,
          k_BLHK,
          v_BLHV,
          attention_masks=attention_masks,
          scale=scale,
          enable_gqa=enable_gqa,
          is_causal=is_causal,
          **kwargs,
      )
    if attention_masks is not None:
      raise ValueError(
          "SlidingWindowScaledDotProductInnerAttention does not support"
          " runtime attention_masks."
      )
    if not is_causal:
      raise ValueError(
          "Sliding-window and packed-document attention are always causal."
      )

    q_BLHK, k_BLHK, v_BLHV, was_3d = _unflatten_to_4d(q_BLHK, k_BLHK, v_BLHV)
    batch_size, seq_len = q_BLHK.shape[:2]
    q_BHLK, k_BHLK, v_BHLV = (
        q_BLHK.transpose(1, 2),
        k_BLHK.transpose(1, 2),
        v_BLHV.transpose(1, 2),
    )
    mask = packed_sliding_window_causal_mask(
        seq_len,
        self.window_size,
        q_BHLK.device,
        segment_ids_BL=(
            None
            if segment_ids is None
            else segment_ids.reshape(batch_size, seq_len)
        ),
    )
    with sdpa_kernel(self.sdpa_backends, set_priority=True):
      out_BHLV = F.scaled_dot_product_attention(
          q_BHLK,
          k_BHLK,
          v_BHLV,
          attn_mask=mask,
          scale=scale,
          enable_gqa=enable_gqa,
      )
    out_BLHV = out_BHLV.transpose(1, 2)
    if was_3d:
      return out_BLHV.reshape(-1, *out_BLHV.shape[2:])
    return out_BLHV


class TpuMuseGlimmerAttention(Attention):
  """Muse Glimmer ``Attention`` that passes ``positions`` to inner attention.

  Identical to upstream ``Attention.forward`` except that the inner attention
  also receives ``positions``, from which it derives packed-document
  boundaries. ``attention_masks`` is only built for flex/varlen backends, which
  the TPU flavors never use, so upstream's window-keyed mask dict lookup is not
  needed here.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Attention.Config):
    pass

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks: AttentionMasksType | None,
      positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    num_tokens = x_TD.shape[0]
    xq, xk, xv = self.qkv_linear(x_TD)

    # QK normalization before RoPE; the query is also scaled by a tuned
    # constant.
    if self.q_norm is not None or self.k_norm is not None:
      assert self.q_norm is not None and self.k_norm is not None
      xq = self.q_norm(xq) * self.scale_query_by
      xk = self.k_norm(xk)

    # iRoPE: RoPE is skipped on NoPE layers.
    if self.rope is not None:
      xq, xk = self.rope(xq, xk, positions)

    output = self.inner_attention(
        xq,
        xk,
        xv,
        attention_masks=attention_masks,
        scale=self.scaling,
        enable_gqa=self.enable_gqa,
        positions=positions,
    ).contiguous()
    output = output.view(num_tokens, -1)

    if self.o_gate is not None:
      output = output * torch.sigmoid(self.o_gate(x_TD))

    return self.wo(output)
