# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from einops import rearrange
import torch
from torch import nn, Tensor
from torchtitan.models.flux import _make_single_block_config
from torchtitan.models.flux.model.layers import (
    apply_rope,
    EmbedND,
    SingleStreamBlock,
)


class SingleStreamBlock_ref(nn.Module):
    """Original SingleStreamBlock with fused linear1 layer and torch.split."""

    def __init__(self, optimized_block):
        super().__init__()
        self.hidden_size = optimized_block.hidden_size
        self.num_heads = optimized_block.num_heads
        self.mlp_hidden_dim = optimized_block.mlp_hidden_dim
        self.pre_norm = optimized_block.pre_norm
        self.norm = optimized_block.norm
        self.mlp_act = optimized_block.mlp_act
        self.modulation = optimized_block.modulation
        self.inner_attention = optimized_block.inner_attention
        self.linear2 = optimized_block.linear2

        # Merge lin_qkv and lin_mlp into linear1
        hs = self.hidden_size
        mlp_dim = self.mlp_hidden_dim

        self.linear1 = nn.Linear(hs, hs * 3 + mlp_dim, bias=True)

        # Concatenate weights and biases from optimized_block.lin_qkv and lin_mlp
        with torch.no_grad():
            weight = torch.cat(
                [optimized_block.lin_qkv.weight, optimized_block.lin_mlp.weight],
                dim=0,
            )
            bias = torch.cat(
                [optimized_block.lin_qkv.bias, optimized_block.lin_mlp.bias],
                dim=0,
            )
            self.linear1.weight.copy_(weight)
            self.linear1.bias.copy_(bias)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift

        # Original: fused linear1 + split
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        q, k = apply_rope(q, k, pe)
        attn = self.inner_attention(q, k, v)
        attn = rearrange(attn, "B L H D -> B L (H D)")

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output

# TODO(jerryx99): b/540005818 re-enable once fixed
@unittest.skip("Skipping Flux QKV split test due to numerical tolerance issues")
class TestFluxSingleStreamUnbind(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.bsz = 2
        self.seqlen = 32
        self.hidden_size = 128
        self.num_heads = 4
        self.head_dim = self.hidden_size // self.num_heads
        self.theta = 10000
        self.axes_dim = (16, 8, 8)  # sum to 32 (head_dim)

        self.ids = torch.randint(
            0, 1000, (self.bsz, self.seqlen, 3), dtype=torch.long
        )
        embed = EmbedND(
            EmbedND.Config(
                dim=self.head_dim, theta=self.theta, axes_dim=self.axes_dim
            )
        )
        self.pe = embed(self.ids)

        self.x = torch.randn(self.bsz, self.seqlen, self.hidden_size)
        self.vec = torch.randn(self.bsz, self.hidden_size)

        self.single_cfg = _make_single_block_config(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=4.0,
        )

    def test_split_linear_weights_standalone(self):
        """Directly test that splitting a linear layer matches separate linear layers."""
        hs = self.hidden_size
        mlp_dim = int(hs * 4.0)

        linear1 = nn.Linear(hs, hs * 3 + mlp_dim, bias=True)

        lin_qkv = nn.Linear(hs, hs * 3, bias=True)
        lin_mlp = nn.Linear(hs, mlp_dim, bias=True)

        with torch.no_grad():
            lin_qkv.weight.copy_(linear1.weight[: hs * 3, :])
            lin_qkv.bias.copy_(linear1.bias[: hs * 3])

            lin_mlp.weight.copy_(linear1.weight[hs * 3 :, :])
            lin_mlp.bias.copy_(linear1.bias[hs * 3 :])

        x_mod = torch.randn(self.bsz, self.seqlen, hs)

        # Reference: linear1 + split
        qkv_ref, mlp_ref = torch.split(
            linear1(x_mod), [3 * hs, mlp_dim], dim=-1
        )

        # Optimized: separate linear layers
        qkv_opt = lin_qkv(x_mod)
        mlp_opt = lin_mlp(x_mod)

        self.assertEqual((qkv_ref - qkv_opt).abs().max().item(), 0.0)
        self.assertEqual((mlp_ref - mlp_opt).abs().max().item(), 0.0)

    def test_single_stream_block_splitting_equivalence(self):
        """Test SingleStreamBlock equivalence between fused linear1+split and separated lin_qkv+lin_mlp."""
        block_opt = SingleStreamBlock(self.single_cfg)
        block_orig = SingleStreamBlock_ref(block_opt)

        out_opt = block_opt(self.x, self.vec, self.pe)
        out_orig = block_orig(self.x, self.vec, self.pe)

        self.assertLess((out_opt - out_orig).abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
