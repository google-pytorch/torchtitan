# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in activation-dtype cos/sin RoPE for TPU.

The stock :class:`CosSinRoPE` deliberately evaluates the rotation in FP32.
This override instead converts the FP32 cosine/sine cache to each activation's
dtype before applying the rotation. For Qwen training with BF16 activations,
that matches MaxText's numerical policy and avoids FP32 rotary arithmetic.

The override is exact so RoPE subclasses with different cache contracts are
left untouched. It preserves cache construction, state, and checkpoint keys.
"""

from dataclasses import dataclass

import torch

from torchtitan.config import derive, override
from torchtitan.models.common import CosSinRoPE


class TpuActivationDtypeCosSinRoPE(CosSinRoPE):
  """Apply RoPE without materializing FP32 query/key intermediates.

  ``CosSinRoPE`` upcasts the complete query and key activations to FP32, does
  the rotate-half multiplies and adds in FP32, and casts the results back to
  their input dtypes. This override keeps the cache itself in FP32, but casts
  only its broadcast cos/sin slices to the corresponding activation dtype and
  performs the visible rotation there. On TPU this removes large FP32
  elementwise intermediates and shortens both the forward and recomputed
  backward graphs; the attention custom calls themselves are unchanged.

  With FP32 activations the behavior is unchanged. With BF16 activations this
  deliberately rounds cos/sin before the rotation and is not bit-exact with
  the base class. It is intended for an explicit BF16 numerical policy such as
  MaxText parity, and should not silently replace FP32 RoPE for exact
  checkpoint continuation or before the desired convergence gate passes.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(CosSinRoPE.Config):
    pass

  # Only the rotation differs; cache lookup stays in the inherited
  # `RoPE.forward` / `_reshape_cache`. Keeping the override in
  # `apply_rotary_emb` lets callers that fetch the cache themselves (e.g. the
  # TPU [B, L] block layout in `experiments/tpu/components/bld_layout.py`) keep this
  # numerical policy.
  @staticmethod
  def apply_rotary_emb(
      query: torch.Tensor,
      key: torch.Tensor | None,
      rope_cache: torch.Tensor,
      *,
      inverse: bool = False,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if inverse:
      raise NotImplementedError("CosSinRoPE does not support inverse rotation.")
    head_dim = query.shape[-1]
    cos = rope_cache[..., :head_dim]
    sin = rope_cache[..., head_dim:]

    query_cos = cos.to(query.dtype)
    query_sin = sin.to(query.dtype)
    query_out = query * query_cos + CosSinRoPE._rotate_half(query) * query_sin
    if key is None:
      return query_out
    key_cos = cos.to(key.dtype)
    key_sin = sin.to(key.dtype)
    key_out = key * key_cos + CosSinRoPE._rotate_half(key) * key_sin
    return query_out, key_out


@override(
    # Upstream d771abb0 made `override()` keyword-only and removed the separate
    # name argument: an override is now identified by the decorated factory's
    # import path, which is what `override.imports` lists.
    target=CosSinRoPE.Config,
    exact=True,
    description="Evaluate cos/sin RoPE in the activation dtype on TPU.",
)
def tpu_activation_dtype_cossin_rope(
    config: CosSinRoPE.Config,
) -> TpuActivationDtypeCosSinRoPE.Config:
  return derive(config, TpuActivationDtypeCosSinRoPE.Config)


__all__ = [
    "TpuActivationDtypeCosSinRoPE",
    "tpu_activation_dtype_cossin_rope",
]
