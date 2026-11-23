# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchtitan.models.common.attention import InnerAttention

AttentionMasksType = Any


def segment_ids_from_positions(positions: torch.Tensor) -> torch.Tensor:
  """Compute SplashAttention `segment_ids` from dataloader `positions`.

  `positions` restarts at 0 at each document boundary:
      [0, 1, 2, 3, 0, 1, 2, 3, 4, ...]
  and this returns a 0-indexed document id per token:
      [0, 0, 0, 0, 1, 1, 1, 1, 1, ...]
  """
  return (torch.cumsum((positions == 0).int(), dim=-1) - 1).int()


@dataclass(frozen=True, slots=True, kw_only=True)
class TpuAttentionMetadata:
  """Attention metadata for TPU Splash/SDPA inner attention."""

  segment_ids: torch.Tensor | None = None
  positions: torch.Tensor | None = None


class ScaledDotProductInnerAttention(InnerAttention):
  """Inner attention using ``F.scaled_dot_product_attention`` for TPU/CPU.

  Upstream removed ``ScaledDotProductInnerAttention`` in #5296 in favor of
  Flex/Varlen; the TPU lane keeps it as the base module that SplashAttention
  patches in ``workarounds.py`` and as a CPU/TPU fallback.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(InnerAttention.Config):

    @property
    def is_full_attention(self) -> bool:
      return True

    def build_attention_metadata(
        self,
        positions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
    ) -> TpuAttentionMetadata:
      del padding_mask, max_num_documents, max_context_length
      return TpuAttentionMetadata(
          segment_ids=segment_ids_from_positions(positions),
          positions=positions,
      )

  sdpa_backends: list[SDPBackend] = []

  def __init__(self, config: Config | None = None) -> None:
    if config is None:
      config = ScaledDotProductInnerAttention.Config()
    super().__init__()
    if not self.sdpa_backends:
      self.sdpa_backends = [
          SDPBackend.CUDNN_ATTENTION,
          SDPBackend.MATH,
      ]

    try:
      tpu_mod = torch.get_device_module("tpu")
      is_tpu = tpu_mod is not None and tpu_mod.is_available()
    except (RuntimeError, ValueError, ImportError):
      is_tpu = False

    if is_tpu:
      self.sdpa_backends = [
          SDPBackend.OVERRIDEABLE,
          SDPBackend.MATH,
      ]

  def forward(
      self,
      q_BLHK: torch.Tensor,
      k_BLHK: torch.Tensor,
      v_BLHV: torch.Tensor,
      *,
      attention_metadata: Any = None,
      attention_masks: AttentionMasksType | None = None,
      scale: float | None = None,
      enable_gqa: bool = False,
      is_causal: bool = True,
      **kwargs,
  ) -> torch.Tensor:
    del attention_metadata, kwargs
    if attention_masks is not None:
      raise ValueError(
          "ScaledDotProductInnerAttention does not support attention_masks; it "
          "only supports causal/non-causal attention via is_causal."
      )
    was_3d = q_BLHK.ndim == 3
    if was_3d:
      seq_len = None
      try:
        from torchtitan.experiments.tpu.components.unflatten_shim import (
            get_unflatten_seq_len,
        )

        seq_len = get_unflatten_seq_len()
      except ImportError:
        pass
      num_tokens = q_BLHK.shape[0]
      if seq_len is None:
        seq_len = num_tokens
      elif num_tokens % seq_len != 0:
        raise ValueError(
            f"TT-UNFLATTEN-SHIM: num_tokens ({num_tokens}) is not cleanly "
            f"divisible by seq_len ({seq_len})."
        )
      batch = num_tokens // seq_len
      q_BLHK = q_BLHK.view(batch, seq_len, *q_BLHK.shape[1:])
      k_BLHK = k_BLHK.view(batch, seq_len, *k_BLHK.shape[1:])
      v_BLHV = v_BLHV.view(batch, seq_len, *v_BLHV.shape[1:])

    q_BHLK = q_BLHK.transpose(1, 2)
    k_BHLK = k_BLHK.transpose(1, 2)
    v_BHLV = v_BLHV.transpose(1, 2)

    with sdpa_kernel(self.sdpa_backends):
      output_BHLV = F.scaled_dot_product_attention(
          q_BHLK,
          k_BHLK,
          v_BHLV,
          is_causal=is_causal,
          scale=scale,
          enable_gqa=enable_gqa,
      )
    output_BLHV = output_BHLV.transpose(1, 2)
    if was_3d:
      return output_BLHV.reshape(-1, *output_BLHV.shape[2:])
    return output_BLHV
