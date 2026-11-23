import torchtitan.models.common as common_models
from torchtitan.models.common.embedding import VocabParallelEmbedding as Embedding
from torchtitan.experiments.tpu.components.attention import (
    make_tpu_model_registry,
    use_tpu_sdpa,
)
import torchtitan.models.llama3.flavors as llama3_models

model_registry = make_tpu_model_registry(llama3_models.build_model_config)

from .infra.parallelize import parallelize_llama

__all__ = [
    "parallelize_llama",
    "llama3_configs",
]


def _testmodel(attn_backend: str = "flex") -> llama3_models.Llama3Model.Config:
  dim = 64
  n_heads = 8
  n_layers = 1
  return llama3_models.Llama3Model.Config(
      max_context_length=512,
      dim=dim,
      vocab_size=128,
      tok_embeddings=Embedding.Config(
          num_embeddings=128,
          embedding_dim=dim,
          param_init=llama3_models._EMBEDDING_INIT,
      ),
      norm=common_models.RMSNorm.Config(
          normalized_shape=dim,
          param_init=llama3_models._NORM_INIT,
      ),
      lm_head=common_models.Linear.Config(
          in_features=dim,
          out_features=128,
          param_init=llama3_models._LINEAR_INIT,
      ),
      layers=llama3_models._build_llama3_layers(
          n_layers=n_layers,
          dim=dim,
          n_heads=n_heads,
          n_kv_heads=n_heads,
          hidden_dim=4 * dim,
          attn_backend="flex" if attn_backend == "sdpa" else attn_backend,
          enable_sp=False,
          rope=common_models.ComplexRoPE.Config(
              dim=dim // n_heads,
              max_context_length=512,
          ),
      ),
  )


def _upstream_flavor(
    flavor: str, attn_backend: str
) -> llama3_models.Llama3Model.Config:
  """Builds an upstream llama3 flavor at that flavor's full context length.

  Upstream d771abb0 changed the flavor table entries from a
  `(attn_backend) -> Config` callable into a `(builder, max_context_length)`
  pair, and made `seq_len` a required keyword-only argument of the builder;
  #4913 moved the table to `flavors.MODEL_FLAVORS`. Passing the flavor's own
  `max_context_length` reproduces what the old callable produced.
  """
  from torchtitan.config.parallelism import ParallelismConfig

  build_config, max_context_length = llama3_models.MODEL_FLAVORS[flavor]
  cfg = build_config(
      attn_backend="flex" if attn_backend == "sdpa" else attn_backend,
      seq_len=max_context_length,
      parallelism=ParallelismConfig(),
  )
  return use_tpu_sdpa(cfg) if attn_backend == "sdpa" else cfg


llama3_configs = {
    "testmodel": _testmodel("sdpa"),
    "debugmodel": _upstream_flavor("debugmodel", "sdpa"),
    "debugmodel_flex_attn": _upstream_flavor("debugmodel", "flex"),
    "1B": _upstream_flavor("1B", "sdpa"),
    "8B": _upstream_flavor("8B", "sdpa"),
    "70B": _upstream_flavor("70B", "sdpa"),
    "405B": _upstream_flavor("405B", "sdpa"),
}
