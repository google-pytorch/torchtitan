# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from einops import rearrange
import torch
from torch import nn, Tensor
from torchtitan.models.flux.layers import apply_rope, EmbedND, rope


# --- Original (Reference) Implementation ---
def rope_ref(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope_ref(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND_ref(nn.Module):

    def __init__(self, dim: int, theta: int, axes_dim: tuple):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [
                rope_ref(ids[..., i], self.axes_dim[i], self.theta)
                for i in range(n_axes)
            ],
            dim=-3,
        )
        return emb.unsqueeze(2)


class TestFluxRopeEquivalence(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.bsz = 2
        self.seqlen = 16
        self.n_heads = 4
        self.head_dim = 64
        self.theta = 10000
        self.axes_dim = (16, 24, 24)  # sum to 64
        self.ids = torch.randint(0, 1000, (self.bsz, self.seqlen, 3), dtype=torch.long)
        self.xq = torch.randn(
            self.bsz, self.seqlen, self.n_heads, self.head_dim, dtype=torch.bfloat16
        )
        self.xk = torch.randn(
            self.bsz, self.seqlen, self.n_heads, self.head_dim, dtype=torch.bfloat16
        )

    def test_rope_equivalence(self):
        """Test that the imported torchtitan layers match the reference implementation."""
        embed_ref = EmbedND_ref(self.head_dim, self.theta, self.axes_dim)
        config = EmbedND.Config(
            dim=self.head_dim, theta=self.theta, axes_dim=self.axes_dim
        )
        embed_opt = EmbedND(config)

        freqs_ref = embed_ref(self.ids)
        freqs_opt = embed_opt(self.ids)

        xq_out_ref, xk_out_ref = apply_rope_ref(self.xq, self.xk, freqs_ref)
        xq_out_opt, xk_out_opt = apply_rope(self.xq, self.xk, freqs_opt)

        torch.testing.assert_close(xq_out_ref, xq_out_opt, atol=0.0, rtol=0.0)
        torch.testing.assert_close(xk_out_ref, xk_out_opt, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
