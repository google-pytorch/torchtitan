# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from einops import rearrange
import torch
from torch import Tensor
from torchtitan.models.flux.flavors import (
    _make_double_block_config,
    _make_single_block_config,
)
from torchtitan.models.flux.model.layers import (
    apply_rope,
    DoubleStreamBlock,
    EmbedND,
    SelfAttention,
    SingleStreamBlock,
)


# --- Original (Reference) Implementation ---
def self_attention_forward_ref(module, x: Tensor, pe: Tensor) -> Tensor:
    qkv = module.qkv(x)
    q, k, v = rearrange(
        qkv, "B L (K H D) -> K B L H D", K=3, H=module.num_heads
    )
    q, k = module.norm(q, k, v)
    q, k = apply_rope(q, k, pe)
    x = module.inner_attention(q, k, v, is_causal=False)
    x = rearrange(x, "B L H D -> B L (H D)")
    x = module.proj(x)
    return x


def double_stream_block_forward_ref(
    module, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor
) -> tuple[Tensor, Tensor]:
    img_mod1, img_mod2 = module.img_mod(vec)
    txt_mod1, txt_mod2 = module.txt_mod(vec)

    # prepare image for attention
    img_modulated = module.img_norm1(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    img_qkv = module.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = rearrange(
        img_qkv, "B L (K H D) -> K B L H D", K=3, H=module.num_heads
    )
    img_q, img_k = module.img_attn.norm(img_q, img_k, img_v)

    # prepare txt for attention
    txt_modulated = module.txt_norm1(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_qkv = module.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = rearrange(
        txt_qkv, "B L (K H D) -> K B L H D", K=3, H=module.num_heads
    )
    txt_q, txt_k = module.txt_attn.norm(txt_q, txt_k, txt_v)

    # run actual attention
    q = torch.cat((txt_q, img_q), dim=1)
    k = torch.cat((txt_k, img_k), dim=1)
    v = torch.cat((txt_v, img_v), dim=1)

    q, k = apply_rope(q, k, pe)
    attn = module.inner_attention(q, k, v, is_causal=False)
    attn = rearrange(attn, "B L H D -> B L (H D)")

    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

    # calculate the img blocks
    img = img + img_mod1.gate * module.img_attn.proj(img_attn)
    img = img + img_mod2.gate * module.img_mlp(
        (1 + img_mod2.scale) * module.img_norm2(img) + img_mod2.shift
    )

    # calculate the txt blocks
    txt = txt + txt_mod1.gate * module.txt_attn.proj(txt_attn)
    txt = txt + txt_mod2.gate * module.txt_mlp(
        (1 + txt_mod2.scale) * module.txt_norm2(txt) + txt_mod2.shift
    )
    return img, txt


def single_stream_block_forward_ref(
    module, x: Tensor, vec: Tensor, pe: Tensor
) -> Tensor:
    mod, _ = module.modulation(vec)
    x_mod = (1 + mod.scale) * module.pre_norm(x) + mod.shift
    qkv = module.lin_qkv(x_mod)
    mlp = module.lin_mlp(x_mod)

    q, k, v = rearrange(
        qkv, "B L (K H D) -> K B L H D", K=3, H=module.num_heads
    )
    q, k = module.norm(q, k, v)

    # compute attention
    q, k = apply_rope(q, k, pe)
    attn = module.inner_attention(q, k, v, is_causal=False)
    attn = rearrange(attn, "B L H D -> B L (H D)")

    # compute activation in mlp stream, cat again and run second linear layer
    output = module.linear2(torch.cat((attn, module.mlp_act(mlp)), 2))
    return x + mod.gate * output


class TestFluxRearrangeEquivalence(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.bsz = 2
        self.img_len = 16
        self.txt_len = 16
        self.seqlen = self.img_len + self.txt_len
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
        self.img = torch.randn(self.bsz, self.img_len, self.hidden_size)
        self.txt = torch.randn(self.bsz, self.txt_len, self.hidden_size)
        self.vec = torch.randn(self.bsz, self.hidden_size)

        self.double_cfg = _make_double_block_config(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
        )
        self.single_cfg = _make_single_block_config(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=4.0,
        )

    def test_rearrange_vs_unbind_standalone(self):
        """Directly test the rearrange vs view+unbind tensor manipulation."""
        qkv = torch.randn(
            self.bsz, self.seqlen, 3 * self.num_heads * self.head_dim
        )

        q_ref, k_ref, v_ref = rearrange(
            qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
        )

        B, L, _ = qkv.shape
        qkv_view = qkv.view(B, L, 3, self.num_heads, -1)
        q_opt, k_opt, v_opt = torch.unbind(qkv_view, dim=2)

        torch.testing.assert_close(q_ref, q_opt, atol=0.0, rtol=0.0)
        torch.testing.assert_close(k_ref, k_opt, atol=0.0, rtol=0.0)
        torch.testing.assert_close(v_ref, v_opt, atol=0.0, rtol=0.0)

    def test_self_attention_equivalence(self):
        """Test SelfAttention equivalence between reference rearrange and optimized unbind implementations."""
        attn_opt = self.double_cfg.img_attn.build()

        out_ref = self_attention_forward_ref(attn_opt, self.x, self.pe)
        out_opt = attn_opt(self.x, self.pe)

        torch.testing.assert_close(out_ref, out_opt, atol=0.0, rtol=0.0)

    def test_double_stream_block_equivalence(self):
        """Test DoubleStreamBlock equivalence between reference rearrange and optimized unbind implementations."""
        block_opt = DoubleStreamBlock(self.double_cfg)

        img_ref, txt_ref = double_stream_block_forward_ref(
            block_opt, self.img, self.txt, self.vec, self.pe
        )
        img_opt, txt_opt = block_opt(self.img, self.txt, self.vec, self.pe)

        torch.testing.assert_close(img_ref, img_opt, atol=0.0, rtol=0.0)
        torch.testing.assert_close(txt_ref, txt_opt, atol=0.0, rtol=0.0)

    def test_single_stream_block_equivalence(self):
        """Test SingleStreamBlock equivalence between reference rearrange and optimized unbind implementations."""
        block_opt = SingleStreamBlock(self.single_cfg)

        out_ref = single_stream_block_forward_ref(
            block_opt, self.x, self.vec, self.pe
        )
        out_opt = block_opt(self.x, self.vec, self.pe)

        torch.testing.assert_close(out_ref, out_opt, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
