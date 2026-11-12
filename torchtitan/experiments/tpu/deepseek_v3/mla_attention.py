# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek MLA attention on the packed-q MLA Splash kernel (TPU).

Same parameters, state dict and math as the upstream ``Attention``; only the
attention core changes. Upstream splits q into nope/pe, ropes q_pe, concats
q and k (broadcasting k_pe to every head), splits kv, and the Splash path then
transposes q, k and v to ``[B, H, S, D]`` (about 22 ms of copies per 236B layer
at 32K tokens per rank). Here ``wq_b``'s and ``wkv_b``'s outputs go to the
kernel as they are; q_pe is roped inside the kernel and only the tiny shared
k_pe is roped outside.

Legend: T tokens (B * S), B batch, S sequence, H heads, D model dim.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.config import derive
from torchtitan.experiments.tpu.components.unflatten_shim import (
    get_unflatten_seq_len,
)
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.deepseek_v3.model import Attention
from torchtitan.protocols.model import ModelConfigConverter


class TpuMLAAttention(Attention):
  """``Attention`` whose core is ``kernels.mla_attention.mla_attention``."""

  @dataclass(kw_only=True, slots=True)
  class Config(Attention.Config):
    pass

  def __init__(self, config: Config):
    super().__init__(config)
    if self.q_lora_rank == 0:
      raise ValueError("TpuMLAAttention needs q_lora_rank > 0 (wq_a / wq_b)")
    if self.qk_nope_head_dim != 128 or self.v_head_dim != 128:
      raise ValueError(
          "TpuMLAAttention needs qk_nope_head_dim == v_head_dim == 128, got "
          f"{self.qk_nope_head_dim} / {self.v_head_dim}")

  def forward(
      self,
      x: torch.Tensor,
      attention_masks: AttentionMasksType,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
  ):
    from torchtitan.experiments.tpu.kernels.mla_attention import (  # pylint: disable=g-import-not-at-top
        mla_attention,
        rope_pair_tables,
    )

    if attention_masks is not None:
      raise ValueError("TpuMLAAttention takes segment_ids, not attention_masks")
    if positions is None or segment_ids is None:
      raise ValueError("TpuMLAAttention needs per-token positions and segment_ids")
    num_tokens = x.shape[0]
    seq = get_unflatten_seq_len() or segment_ids.shape[-1]
    batch = num_tokens // seq

    q_TQ = self.wq_b(self.q_norm(self.wq_a(x)))       # Q = H * (128 + Dr)
    kv_a_TL = self.wkv_a(x)                          # L = kv_lora_rank + Dr
    c_kv_TC, k_pe_TR = torch.split(
        kv_a_TL, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

    cis_T1R = self.rope._reshape_cache(k_pe_TR.unsqueeze(1), positions)
    k_pe_TR = self.rope.apply_rotary_emb(
        k_pe_TR.unsqueeze(1), None, cis_T1R).squeeze(1)
    cos_BSR, sin_BSR = rope_pair_tables(cis_T1R.squeeze(1), batch, seq)

    kv_TK = self.wkv_b(self.kv_norm(c_kv_TC))        # K = H * 256
    o_BSV = mla_attention(
        q_TQ.view(batch, seq, -1).contiguous(),
        kv_TK.view(batch, seq, -1).contiguous(),
        k_pe_TR.view(batch, seq, -1).contiguous(),
        cos_BSR.contiguous(),
        sin_BSR.contiguous(),
        segment_ids.reshape(batch, seq).to(torch.int32).contiguous(),
        scale=self.softmax_scale,
    )
    return self.wo(o_BSV.view(num_tokens, -1))


class TpuMLAAttentionConverter(ModelConfigConverter):
  """Select ``TpuMLAAttention`` for every DeepSeek ``Attention`` block."""

  @dataclass(kw_only=True, slots=True)
  class Config(ModelConfigConverter.Config):
    pass

  def __init__(self, config: Config):
    pass

  def convert(self, model_config):
    for _, config, parent, field_name in list(
        model_config.traverse(Attention.Config)
    ):
      if type(config) is not Attention.Config:
        continue
      replacement = derive(config, TpuMLAAttention.Config)
      if parent is None:
        model_config = replacement
      elif isinstance(parent, list):
        parent[field_name] = replacement
      else:
        setattr(parent, field_name, replacement)
    return model_config


__all__ = ["TpuMLAAttention", "TpuMLAAttentionConverter"]
