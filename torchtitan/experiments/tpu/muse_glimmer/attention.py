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

The SDPA inner attention and its mask helpers live in ``tpu/masked_sdpa.py``
(shared with Qwen3.5 and DeepSeek-V3); they are re-exported here.
"""

from dataclasses import dataclass

import torch
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.muse_glimmer.model import Attention

# Re-exported for existing importers (muse_glimmer/__init__.py, its tests).
from torchtitan.experiments.tpu.masked_sdpa import packed_sliding_window_causal_mask  # pylint: disable=unused-import,g-bad-import-order
from torchtitan.experiments.tpu.masked_sdpa import SlidingWindowScaledDotProductInnerAttention  # pylint: disable=unused-import,g-bad-import-order
from torchtitan.experiments.tpu.masked_sdpa import sliding_window_causal_mask  # pylint: disable=unused-import,g-bad-import-order


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
