# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# imported from black-forest-labs/FLUX
# Shape suffixes: B = batch, L = per-sample sequence, T = packed tokens,
# D = feature dimension, H = heads, K = query/key head dimension, V = value dimension.
import math
from dataclasses import dataclass, field

import spmd_types as spmd
import torch
from einops import rearrange
from torch import nn, Tensor

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.base.module import Module, Sequential

from torchtitan.models.common.activation import SiLU
from torchtitan.models.common.attention import (
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import GELU, LayerNorm, RMSNorm

DP = MeshAxisName.DP
CP = MeshAxisName.CP


@spmd.local_map(
    in_types=(spmd.PartitionSpec("dp", "cp", None), None),
    out_types=(
        spmd.PartitionSpec("dp", "cp", None),
        spmd.PartitionSpec("dp", "cp", None),
    ),
)
def local_split_text_image(
    combined: torch.Tensor,
    text_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return combined[:, :text_seq_len], combined[:, text_seq_len:]


@spmd.local_map(
    in_types=(
        spmd.PartitionSpec(DP, CP, None, None),
        spmd.PartitionSpec(DP, None, None, None),
        spmd.PartitionSpec(DP, None, None, None),
    ),
    out_types=(
        spmd.PartitionSpec((DP, CP), None, None),
        spmd.PartitionSpec(DP, None, None),
        spmd.PartitionSpec(DP, None, None),
    ),
)
def _pack_attention_inputs(
    q_BLHK: torch.Tensor, k_BLHK: torch.Tensor, v_BLHV: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return q_BLHK.flatten(0, 1), k_BLHK.flatten(0, 1), v_BLHV.flatten(0, 1)


def _prepare_flux_attention_inputs(
    q_BLHK: torch.Tensor, k_BLHK: torch.Tensor, v_BLHV: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather K/V along sequence before packing to preserve sample boundaries."""
    # TODO: Use ContextParallelTransform once Flux supports more CP variants.
    cp_group = spmd_mesh_group(CP)
    if cp_group is not None:
        k_BLHK, v_BLHV = (
            spmd.redistribute(
                x_BLHD,
                cp_group,
                src=spmd.S(1),
                dst=spmd.R,
                backward_options={"op_dtype": x_BLHD.dtype},
            )
            for x_BLHD in (k_BLHK, v_BLHV)
        )
    return _pack_attention_inputs(q_BLHK, k_BLHK, v_BLHV)


@spmd.local_map(
    in_types=(spmd.PartitionSpec((DP, CP), None, None), None),
    out_types=spmd.PartitionSpec(DP, CP, None, None),
)
def _unpack_flux_attention_output(
    out_THV: torch.Tensor, batch_size: int
) -> torch.Tensor:
    return out_THV.unflatten(0, (batch_size, -1))


def _varlen_attention(
    inner_attention: VarlenInnerAttention,
    q_BLHK: Tensor,
    k_BLHK: Tensor,
    v_BLHV: Tensor,
    *,
    attention_metadata: VarlenAttentionMetadata,
) -> Tensor:
    q_THK, k_THK, v_THV = _prepare_flux_attention_inputs(q_BLHK, k_BLHK, v_BLHV)
    out_THV = inner_attention(
        q_THK, k_THK, v_THV, attention_metadata=attention_metadata
    )
    return _unpack_flux_attention_output(out_THV, q_BLHK.shape[0])


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    # Return cos and sin separately to avoid tiny dimensions
    cos = torch.cos(out).float()
    sin = torch.sin(out).float()
    return torch.stack([cos, sin], dim=-1) # Just pass cos and sin together

def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    # freqs_cis has shape (..., d/2, 2), where [..., 0] is cos and [..., 1] is sin
    cos = freqs_cis[..., 0]
    sin = freqs_cis[..., 1]

    # xq shape: (..., d). We reshape to (..., d/2, 2)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_0, xq_1 = xq_[..., 0], xq_[..., 1]
    xk_0, xk_1 = xk_[..., 0], xk_[..., 1]

    xq_out_0 = xq_0 * cos - xq_1 * sin
    xq_out_1 = xq_0 * sin + xq_1 * cos

    xk_out_0 = xk_0 * cos - xk_1 * sin
    xk_out_1 = xk_0 * sin + xk_1 * cos

    xq_out = torch.stack([xq_out_0, xq_out_1], dim=-1).reshape(*xq.shape).type_as(xq)
    xk_out = torch.stack([xk_out_0, xk_out_1], dim=-1).reshape(*xk.shape).type_as(xk)
    return xq_out, xk_out


class EmbedND(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        theta: int
        axes_dim: tuple

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.theta = config.theta
        self.axes_dim = config.axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-2,
        )

        return emb.unsqueeze(2)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Creates sinusoidal timestep embeddings.

    Args:
        t (Tensor): A 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): The dimension of the output.
        max_period (int, optional): Controls the minimum frequency of the embeddings. Default is 10000.
        time_factor (float, optional): Scaling factor for time. Default is 1000.0.

    Returns:
        Tensor: An (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    with torch.device(t.device):
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_layer: Linear.Config
        out_layer: Linear.Config
        in_dim: int
        hidden_dim: int

    def __init__(self, config: Config):
        super().__init__()
        self.in_layer = config.in_layer.build()
        self.silu = SiLU.Config().build()
        self.out_layer = config.out_layer.build()

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class QKNorm(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        query_norm: RMSNorm.Config
        key_norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.query_norm = config.query_norm.build()
        self.key_norm = config.key_norm.build()

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        qkv: Linear.Config
        proj: Linear.Config
        norm: QKNorm.Config
        num_heads: int = 8
        qkv_bias: bool = False
        inner_attention: VarlenInnerAttention.Config = field(
            default_factory=VarlenInnerAttention.Config
        )

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.qkv = config.qkv.build()
        self.norm = config.norm.build()
        self.proj = config.proj.build()
        self.inner_attention = config.inner_attention.build()

    def forward(
        self, x: Tensor, pe: Tensor, *, attention_metadata: VarlenAttentionMetadata
    ) -> Tensor:
        qkv = self.qkv(x)
        B, L, _ = qkv.shape
        qkv = qkv.view(B, L, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, dim=2)
        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        x = _varlen_attention(
            self.inner_attention, q, k, v, attention_metadata=attention_metadata
        )
        x = rearrange(x, "B L H D -> B L (H D)")
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        lin: Linear.Config
        double: bool

    def __init__(self, config: Config):
        super().__init__()
        self.is_double = config.double
        self.multiplier = 6 if config.double else 3
        self.lin = config.lin.build()

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int
        img_mod: Modulation.Config
        txt_mod: Modulation.Config
        img_attn: SelfAttention.Config
        txt_attn: SelfAttention.Config
        img_mlp_in: Linear.Config
        img_mlp_out: Linear.Config
        txt_mlp_in: Linear.Config
        txt_mlp_out: Linear.Config
        mlp_ratio: float = 4.0
        qkv_bias: bool = False
        inner_attention: VarlenInnerAttention.Config = field(
            default_factory=VarlenInnerAttention.Config
        )

    def __init__(self, config: Config):
        super().__init__()

        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        ln_cfg = LayerNorm.Config(
            normalized_shape=config.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        gelu_cfg = GELU.Config(approximate="tanh")

        self.img_mod = config.img_mod.build()
        self.img_norm1 = ln_cfg.build()
        self.img_attn = config.img_attn.build()

        self.img_norm2 = ln_cfg.build()
        self.img_mlp = Sequential(
            config.img_mlp_in.build(),
            gelu_cfg.build(),
            config.img_mlp_out.build(),
        )

        self.txt_mod = config.txt_mod.build()
        self.txt_norm1 = ln_cfg.build()
        self.txt_attn = config.txt_attn.build()

        self.txt_norm2 = ln_cfg.build()
        self.txt_mlp = Sequential(
            config.txt_mlp_in.build(),
            gelu_cfg.build(),
            config.txt_mlp_out.build(),
        )

        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        *,
        attention_metadata: VarlenAttentionMetadata,
    ) -> tuple[Tensor, Tensor]:
        @spmd.local_map(
            in_types=(
                spmd.PartitionSpec("dp", "cp", None, None),
                spmd.PartitionSpec("dp", "cp", None, None),
            ),
            out_types=spmd.PartitionSpec("dp", "cp", None, None),
        )
        def _local_concat_text_image_attention_states(
            text: torch.Tensor,
            image: torch.Tensor,
        ) -> torch.Tensor:
            return torch.cat((text, image), dim=1)

        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        assert txt_mod2 is not None
        assert img_mod2 is not None

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        B, L, _ = img_qkv.shape
        img_qkv = img_qkv.view(B, L, 3, self.num_heads, -1)
        img_q, img_k, img_v = torch.unbind(img_qkv, dim=2)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        B, L, _ = txt_qkv.shape
        txt_qkv = txt_qkv.view(B, L, 3, self.num_heads, -1)
        txt_q, txt_k, txt_v = torch.unbind(txt_qkv, dim=2)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = _local_concat_text_image_attention_states(txt_q, img_q)
        k = _local_concat_text_image_attention_states(txt_k, img_k)
        v = _local_concat_text_image_attention_states(txt_v, img_v)

        q, k = apply_rope(q, k, pe)
        attn = _varlen_attention(
            self.inner_attention, q, k, v, attention_metadata=attention_metadata
        )
        attn = rearrange(attn, "B L H D -> B L (H D)")

        txt_attn, img_attn = local_split_text_image(attn, txt.shape[1])

        # calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )
        return img, txt


class SingleStreamBlock(Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int
        lin_qkv: Linear.Config
        lin_mlp: Linear.Config
        linear2: Linear.Config
        modulation: Modulation.Config
        norm: QKNorm.Config
        mlp_ratio: float = 4.0
        qk_scale: float | None = None
        inner_attention: VarlenInnerAttention.Config = field(
            default_factory=VarlenInnerAttention.Config
        )

    def __init__(self, config: Config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = config.qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(config.hidden_size * config.mlp_ratio)
        # qkv and mlp_in
        self.lin_qkv = config.lin_qkv.build()
        self.lin_mlp = config.lin_mlp.build()
        # proj and mlp_out
        self.linear2 = config.linear2.build()

        self.norm = config.norm.build()

        self.hidden_size = config.hidden_size
        self.pre_norm = LayerNorm.Config(
            normalized_shape=config.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        ).build()

        self.mlp_act = GELU.Config(approximate="tanh").build()
        self.modulation = config.modulation.build()
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        *,
        attention_metadata: VarlenAttentionMetadata,
    ) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        # TODO(b/517661974): Add support for loading split linear layer weights.
        qkv = self.lin_qkv(x_mod)
        mlp = self.lin_mlp(x_mod)

        B, L, _ = qkv.shape
        qkv = qkv.view(B, L, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, dim=2)
        q, k = self.norm(q, k, v)

        # compute attention
        q, k = apply_rope(q, k, pe)
        attn = _varlen_attention(
            self.inner_attention, q, k, v, attention_metadata=attention_metadata
        )
        attn = rearrange(attn, "B L H D -> B L (H D)")

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        linear: Linear.Config
        adaln_linear: Linear.Config
        hidden_size: int
        patch_size: int
        out_channels: int

    def __init__(self, config: Config):
        super().__init__()
        self.norm_final = LayerNorm.Config(
            normalized_shape=config.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        ).build()
        self.linear = config.linear.build()
        self.adaLN_modulation = Sequential(
            nn.SiLU(),
            config.adaln_linear.build(),
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
