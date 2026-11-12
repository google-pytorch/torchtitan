# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import torchtitan.models.common as common_models
import torchtitan.models.deepseek_v3 as deepseek_v3_models
from torchtitan.experiments.tpu.deepseek_v3.infra.parallelize import parallelize_deepseekv3
from torchtitan.protocols.model_spec import ModelSpec

__all__ = [
    "parallelize_deepseekv3",
    "deepseekv3_configs",
    "model_registry",
]


def _testmodel(
    attn_backend: str = "sdpa",
) -> deepseek_v3_models.DeepSeekV3Model.Config:
  dim = 128
  head_dim = 16
  n_layers = 3
  vocab_size = 2048
  # pytype: disable=wrong-keyword-args
  layers = deepseek_v3_models._build_dsv3_layers(
      n_layers=n_layers,
      n_dense_layers=n_layers,
      dim=dim,
      n_heads=8,
      q_lora_rank=0,
      kv_lora_rank=512,
      qk_nope_head_dim=128,
      qk_rope_head_dim=head_dim,
      v_head_dim=128,
      mscale=0.70,
      dense_hidden_dim=256,
      moe_hidden_dim=128,
      num_experts=8,
      num_shared_experts=2,
      router_top_k=2,
      router_score_func="softmax",
      attn_backend=attn_backend,
      moe_comm_backend="standard",
      non_blocking_capacity_factor=None,
      rope=common_models.ComplexRoPE.Config(
          dim=head_dim,
          max_context_length=128,
          theta=10000.0,
          scaling="yarn",
          rope_factor=40.0,
          beta_fast=32.0,
          beta_slow=1.0,
          original_seq_len=128,
      ),
  )
  # pytype: enable=wrong-keyword-args
  return deepseek_v3_models.DeepSeekV3Model.Config(
      vocab_size=vocab_size,
      dim=dim,
      tok_embeddings=common_models.Embedding.Config(
          num_embeddings=vocab_size,
          embedding_dim=dim,
          param_init=deepseek_v3_models._EMBEDDING_INIT,
      ),
      norm=common_models.RMSNorm.Config(
          normalized_shape=dim, param_init=deepseek_v3_models._NORM_INIT
      ),
      lm_head=common_models.Linear.Config(
          in_features=dim,
          out_features=vocab_size,
          param_init=deepseek_v3_models._output_linear_init(dim),
      ),
      layers=layers,
  )


def _testmodel_moe(
    attn_backend: str = "sdpa",
) -> deepseek_v3_models.DeepSeekV3Model.Config:
  dim = 128
  head_dim = 16
  n_layers = 3
  vocab_size = 2048
  # pytype: disable=wrong-keyword-args
  layers = deepseek_v3_models._build_dsv3_layers(
      n_layers=n_layers,
      n_dense_layers=1,
      dim=dim,
      n_heads=8,
      q_lora_rank=0,
      kv_lora_rank=512,
      qk_nope_head_dim=128,
      qk_rope_head_dim=head_dim,
      v_head_dim=128,
      mscale=0.70,
      dense_hidden_dim=256,
      moe_hidden_dim=128,
      num_experts=8,
      num_shared_experts=2,
      router_top_k=2,
      router_score_func="softmax",
      attn_backend=attn_backend,
      moe_comm_backend="standard",
      non_blocking_capacity_factor=None,
      rope=common_models.ComplexRoPE.Config(
          dim=head_dim,
          max_context_length=128,
          theta=10000.0,
          scaling="yarn",
          rope_factor=40.0,
          beta_fast=32.0,
          beta_slow=1.0,
          original_seq_len=128,
      ),
  )
  # pytype: enable=wrong-keyword-args
  return deepseek_v3_models.DeepSeekV3Model.Config(
      vocab_size=vocab_size,
      dim=dim,
      tok_embeddings=common_models.Embedding.Config(
          num_embeddings=vocab_size,
          embedding_dim=dim,
          param_init=deepseek_v3_models._EMBEDDING_INIT,
      ),
      norm=common_models.RMSNorm.Config(
          normalized_shape=dim, param_init=deepseek_v3_models._NORM_INIT
      ),
      lm_head=common_models.Linear.Config(
          in_features=dim,
          out_features=vocab_size,
          param_init=deepseek_v3_models._output_linear_init(dim),
      ),
      layers=layers,
  )


def _upstream_flavor(flavor: str, attn_backend: str):
  """Builds an upstream deepseek-v3 flavor at its full context length.

  Upstream d771abb0 changed
  `deepseek_v3_models.deepseekv3_configs[flavor]` from a callable into a
  `(builder, max_context_length)` pair, and made `seq_len` a required
  keyword-only argument of the builder. Passing the flavor's own
  `max_context_length` reproduces what the old callable produced.
  """
  build_config, max_context_length = deepseek_v3_models.deepseekv3_configs[
      flavor
  ]
  return build_config(
      attn_backend=attn_backend,
      moe_comm_backend="standard",
      seq_len=max_context_length,
  )


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "sdpa",
    converters: list | None = None,
) -> ModelSpec:
  """Upstream's DeepSeek-V3 spec, built for TPU.

  Differs from ``torchtitan.models.deepseek_v3.model_registry`` in two ways:
  attention defaults to ``"sdpa"``, whose ``ScaledDotProductInnerAttention``
  the Splash / Tokamax patch replaces (TPU has no flex attention lowering),
  and the parallelizer is the TPU one, which adds the SimpleFSDP path.

  Args:
    flavor: An upstream flavor (``debugmodel``, ``16B``, ``236B``, ``671B``).
    seq_len: Context length; defaults to the flavor's maximum.
    attn_backend: Upstream attention backend name.
    converters: Model config converters (e.g. the TPU MoE converters).

  Returns:
    The model spec.
  """
  spec = deepseek_v3_models.model_registry(
      flavor,
      seq_len=seq_len,
      attn_backend=attn_backend,
      converters=converters,
  )
  return dataclasses.replace(spec, parallelize_fn=parallelize_deepseekv3)


deepseekv3_configs = {
    "testmodel": _testmodel(),
    "testmodel_moe": _testmodel_moe(),
    "debugmodel": _upstream_flavor("debugmodel", "sdpa"),
    "debugmodel_flex_attn": _upstream_flavor("debugmodel", "flex"),
    "16B": _upstream_flavor("16B", "flex"),
    "236B": _upstream_flavor("236B", "flex"),
    "671B": _upstream_flavor("671B", "flex"),
}
