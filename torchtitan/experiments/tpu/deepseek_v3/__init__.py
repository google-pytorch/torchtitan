# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan.models.common as common_models
from torchtitan.models.common.embedding import VocabParallelEmbedding as Embedding
import torchtitan.models.deepseek_v3.flavors as deepseek_v3_models
from torchtitan.experiments.tpu.components.attention import (
    make_tpu_model_registry,
    use_tpu_sdpa,
)

model_registry = make_tpu_model_registry(deepseek_v3_models.build_model_config)
def parallelize_deepseekv3(model, **kwargs):
  return model.parallelize(**kwargs)

__all__ = [
    "parallelize_deepseekv3",
    "deepseekv3_configs",
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
      router_score_func=common_models.Softmax.Config(),
      attn_backend="flex" if attn_backend == "sdpa" else attn_backend,
      enable_sp=False,
      enable_ep=False,
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
      max_context_length=128,
      vocab_size=vocab_size,
      dim=dim,
      tok_embeddings=Embedding.Config(
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
      router_score_func=common_models.Softmax.Config(),
      attn_backend="flex" if attn_backend == "sdpa" else attn_backend,
      enable_sp=False,
      enable_ep=False,
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
      max_context_length=128,
      vocab_size=vocab_size,
      dim=dim,
      tok_embeddings=Embedding.Config(
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

  Upstream d771abb0 changed the flavor table entries (now
  `flavors.MODEL_FLAVORS`, #4913) from a callable into a
  `(builder, max_context_length)` pair, and made `seq_len` a required
  keyword-only argument of the builder. Passing the flavor's own
  `max_context_length` reproduces what the old callable produced.
  """
  from torchtitan.config.parallelism import ParallelismConfig

  build_config, max_context_length = deepseek_v3_models.MODEL_FLAVORS[flavor]
  cfg = build_config(
      attn_backend="flex" if attn_backend == "sdpa" else attn_backend,
      seq_len=max_context_length,
      parallelism=ParallelismConfig(),
  )
  return use_tpu_sdpa(cfg) if attn_backend == "sdpa" else cfg


deepseekv3_configs = {
    "testmodel": _testmodel(),
    "testmodel_moe": _testmodel_moe(),
    "debugmodel": _upstream_flavor("debugmodel", "sdpa"),
    "debugmodel_flex_attn": _upstream_flavor("debugmodel", "flex"),
    "16B": _upstream_flavor("16B", "flex"),
    "236B": _upstream_flavor("236B", "flex"),
    "671B": _upstream_flavor("671B", "flex"),
}
