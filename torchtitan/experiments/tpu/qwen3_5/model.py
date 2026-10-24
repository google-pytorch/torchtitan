"""Qwen3.5 attention and transformer block modules for TPU."""

from dataclasses import dataclass

import torch

# Before `torchtitan.models.qwen3_5`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu.qwen3_5 import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.qwen3_5.model import Qwen35Attention, Qwen35AttentionMaskDict, Qwen35TransformerBlock


class TpuQwen35Attention(Qwen35Attention):
  """``Qwen35Attention`` that passes ``positions`` to the inner attention.

  Identical to upstream ``Qwen35Attention.forward`` except that the inner
  attention also receives ``positions``, from which the document-masked SDPA
  and the Splash / Tokamax attention patches derive packed-document
  boundaries.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Qwen35Attention.Config):
    pass

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks: AttentionMasksType | None,
      positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    num_tokens = x_TD.shape[0]

    # wq is 2x wider: produces query + gate
    xq_gate_THC = self.wq(x_TD).view(num_tokens, -1, self.head_dim * 2)
    xq_THK, gate_THV = xq_gate_THC.chunk(2, dim=-1)
    xk_THK = self.wk(x_TD).view(num_tokens, -1, self.head_dim)
    xv_THV = self.wv(x_TD).view(num_tokens, -1, self.head_dim)

    # QK norm (before RoPE)
    xq_THK = self.q_norm(xq_THK)
    xk_THK = self.k_norm(xk_THK)

    # Partial RoPE: only first rotary_dim elements get positional encoding
    assert self.rotary_dim <= self.head_dim
    xq_THR, xq_THP = (
        xq_THK[..., : self.rotary_dim],
        xq_THK[..., self.rotary_dim :],
    )
    xk_THR, xk_THP = (
        xk_THK[..., : self.rotary_dim],
        xk_THK[..., self.rotary_dim :],
    )
    xq_THR, xk_THR = self.rope(xq_THR, xk_THR, positions)
    xq_THK = torch.cat([xq_THR, xq_THP], dim=-1)
    xk_THK = torch.cat([xk_THR, xk_THP], dim=-1)

    out_THV = self.inner_attention(
        xq_THK,
        xk_THK,
        xv_THV,
        attention_masks=attention_masks,
        scale=self.scaling,
        enable_gqa=self.enable_gqa,
        # TPU: pass `positions`; SDPA / Splash mask packed documents with it.
        positions=positions,
    ).contiguous()

    # Output gating
    out_THV = out_THV * torch.sigmoid(gate_THV)
    out_TD = out_THV.view(num_tokens, -1)
    return self.wo(out_TD)


class TpuQwen35TransformerBlock(Qwen35TransformerBlock):
  """``Qwen35TransformerBlock`` that also passes ``positions`` to GatedDeltaNet.

  Identical to upstream ``Qwen35TransformerBlock.forward`` except that the
  GatedDeltaNet (``gated_delta.TpuGatedDeltaNet``) also receives ``positions``,
  from which it derives packed-document boundaries.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Qwen35TransformerBlock.Config):
    pass

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks: Qwen35AttentionMaskDict | None,
      positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    layer_mask = (
        attention_masks[self.attn_mask_key]
        if attention_masks is not None
        else None
    )
    h_TD = self.attention_norm(x_TD)
    # TPU: both layer types get `positions` (upstream: full attention only).
    h_TD = self.attn(h_TD, layer_mask, positions)
    x_TD = x_TD + h_TD

    h_TD = self.ffn_norm(x_TD)
    if self.moe_enabled:
      x_TD = x_TD + self.moe(h_TD)
    else:
      x_TD = x_TD + self.feed_forward(h_TD)
    return x_TD
