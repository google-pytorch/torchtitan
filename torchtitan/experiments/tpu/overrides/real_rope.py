# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real-arithmetic complex RoPE for TPU, so the rotation can be traced in `scan`.

`ComplexRoPE.apply_rotary_emb` calls `view_as_complex` on `query`, which
requires grad. A higher-order op such as `scan` re-traces its body through
functionalization, which rewrites that to `view_as_complex_copy` -- and only
the non-copy variant has a derivative:

  RuntimeError: view_as_complex_copy does not support automatic
  differentiation for outputs with complex dtype.

The override is exact, so `CosSinRoPE` and its own TPU override
(`bf16_rope.py`) are left untouched. Remove it once `view_as_complex_copy` has
an autograd formula upstream; `test_real_rope.py` fails when that lands.
"""

from dataclasses import dataclass

import torch

from torchtitan.config import derive, override
from torchtitan.models.common import ComplexRoPE


class TpuRealArithmeticComplexRoPE(ComplexRoPE):
  """The `ComplexRoPE` rotation, with the complex multiply written out.

  The base class multiplies as complex numbers, once per tensor::

      x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
      out = torch.view_as_real(x_ * rope_cache).flatten(3).type_as(x)

  This does the same multiply on the real and imaginary halves directly --
  ``(a + bi)(c + di) = (ac - bd) + (ad + bc)i`` -- so no complex view of a
  tensor that requires grad ever appears. (``view_as_real`` on the cache is
  fine: a buffer never requires grad.)

  `apply_rotary_emb` is the only method that differs. The cache, its complex
  dtype, and every state and checkpoint key are inherited unchanged, and the
  output is bit-identical to the base class rather than merely close.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(ComplexRoPE.Config):
    pass

  @staticmethod
  def apply_rotary_emb(
      query: torch.Tensor,
      key: torch.Tensor | None,
      rope_cache: torch.Tensor,
      *,
      inverse: bool = False,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply complex RoPE using adjacent-dim pairs, in real arithmetic."""
    if inverse:
      rope_cache = rope_cache.conj()
    cache = torch.view_as_real(rope_cache)
    cos, sin = cache[..., 0:1], cache[..., 1:2]

    def rotate(x: torch.Tensor) -> torch.Tensor:
      pairs = x.float().reshape(*x.shape[:-1], -1, 2)
      real, imag = pairs[..., 0:1], pairs[..., 1:2]
      rotated = torch.cat(
          [real * cos - imag * sin, real * sin + imag * cos], dim=-1
      )
      return rotated.flatten(-2).type_as(x)

    if key is None:
      return rotate(query)
    return rotate(query), rotate(key)


@override(
    # Upstream d771abb0 made `override()` keyword-only and removed the separate
    # name argument: an override is now identified by the decorated factory's
    # import path, which is what `override.imports` lists.
    target=ComplexRoPE.Config,
    exact=True,
    description=(
        "Express complex RoPE in real arithmetic so it can be traced inside a"
        " higher-order op such as scan."
    ),
)
def tpu_real_arithmetic_complex_rope(
    config: ComplexRoPE.Config,
) -> TpuRealArithmeticComplexRoPE.Config:
  return derive(config, TpuRealArithmeticComplexRoPE.Config)


__all__ = [
    "TpuRealArithmeticComplexRoPE",
    "tpu_real_arithmetic_complex_rope",
]
