# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Alternative FLOP-counting conventions for TPU MFU reporting."""

import torch.nn as nn

from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import MoE


def get_maxtext_decoder_nparams_and_flops(
    model_config: Decoder.Config,
    model: nn.Module,
    seq_len: int,
) -> tuple[int, int]:
    """Return parameter count and MaxText-style FLOPs for a standard decoder.

    This is an independent port of MaxText's standard decoder calculation at
    commit 263b8c18e32ed1da4dbb4eb1476c5e3d87b4d9eb. It counts dense matmuls,
    routed MoE matmuls, and causally masked quadratic attention, then uses the
    conventional 3x multiplier for forward plus backward. It rejects decoder
    variants with unsupported attention or shared-expert blocks rather than
    silently applying a formula that may not match MaxText.
    """
    if seq_len <= 0:
        raise ValueError("Sequence length must be positive")
    if not model_config.layers:
        raise ValueError("Model must have at least one decoder layer")

    nparams = sum(parameter.numel() for parameter in model.parameters())

    dim = model_config.dim
    # MaxText counts the LM-head projection but not the embedding lookup.
    forward_weight_flops = 2 * dim * model_config.vocab_size
    forward_attention_flops = 0

    for layer in model_config.layers:
        attention = layer.attention
        if not isinstance(attention, GQAttention.Config):
            raise ValueError("MaxText decoder FLOPs require grouped-query attention")
        n_heads = attention.n_heads
        n_kv_heads = attention.n_kv_heads or n_heads
        head_dim = attention.head_dim or dim // n_heads

        # Q/K/V and output projections.
        forward_weight_flops += 2 * dim * (
            (n_heads + 2 * n_kv_heads) * head_dim + n_heads * head_dim
        )
        # MaxText halves the usual non-causal attention FLOPs for causal masks.
        forward_attention_flops += 2 * seq_len * n_heads * head_dim

        moe = layer.moe
        if moe is None:
            feed_forward = layer.feed_forward
            if not isinstance(feed_forward, FeedForward.Config):
                raise ValueError("MaxText decoder FLOPs require a SwiGLU FFN")
            # SwiGLU has two input projections and one output projection.
            forward_weight_flops += 6 * dim * feed_forward.w1.out_features
            continue

        if not isinstance(moe, MoE.Config) or moe.shared_experts is not None:
            raise ValueError(
                "MaxText decoder FLOPs support routed experts without shared experts"
            )
        # Count the router and only the top-k activated expert matmuls.
        # Upstream d771abb0 moved the grouped experts behind `RoutedExperts`.
        forward_weight_flops += (
            2 * dim * moe.num_experts
            + 6
            * dim
            * moe.routed_experts.inner_experts.hidden_dim
            * moe.router.top_k
        )

    # MaxText's training convention is one forward plus two backward FLOPs.
    return nparams, 3 * (forward_weight_flops + forward_attention_flops)
