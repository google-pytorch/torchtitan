import torchtitan.models.common as common_models
import torchtitan.models.qwen3 as qwen3_models

from .infra.parallelize import parallelize_qwen3

from torchtitan.models.qwen3.model import Qwen3Model

from .model import MaxTextQwen3Config

__all__ = [
    "MaxTextQwen3Config",
    "parallelize_qwen3",
    "qwen3_configs",
    "model_registry",
    "TESTMODEL_MAX_CONTEXT_LENGTH",
]

# Context length of `_testmodel_moe`, named so the `ModelSpec.max_context_length`
# that upstream d771abb0 made required can reuse the exact value its RoPE config
# is built with (matches upstream's `("testmodel", 128)`).
TESTMODEL_MAX_CONTEXT_LENGTH = 128


def _testmodel_moe(
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
) -> Qwen3Model.Config:
    dim = 128
    head_dim = 16
    n_layers = 3
    vocab_size = 2048
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=qwen3_models._qwen3_norm(dim),
        tok_embeddings=common_models.Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=qwen3_models._EMBEDDING_INIT
        ),
        lm_head=common_models.Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=qwen3_models._LINEAR_INIT,
        ),

        # pytype: disable=wrong-keyword-args
        layers=qwen3_models._build_qwen3_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=8,
            n_kv_heads=8,
            head_dim=head_dim,
            moe_hidden_dim=128,
            num_experts=8,
            top_k=2,
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
            rope=common_models.CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=TESTMODEL_MAX_CONTEXT_LENGTH,
                theta=1000000.0,
            ),
        ),
        # pytype: enable=wrong-keyword-args
    )



def _upstream_flavor(flavor: str, attn_backend: str):
  """Builds an upstream qwen3 flavor at that flavor's full context length.

  Upstream d771abb0 changed `qwen3_models.qwen3_configs[flavor]` from a
  `(attn_backend) -> Config` callable into a `(builder, max_context_length)`
  pair, and made `seq_len` a required keyword-only argument of the builder.
  Passing the flavor's own `max_context_length` reproduces what the old
  callable produced.
  """
  build_config, max_context_length = qwen3_models.qwen3_configs[flavor]
  return build_config(attn_backend, seq_len=max_context_length)


# Model configs by flavor.
qwen3_configs = {
    "testmodel": _upstream_flavor("testmodel", "sdpa"),
    "testmodel_moe": _testmodel_moe("sdpa"),
    "debugmodel": _upstream_flavor("debugmodel", "sdpa"),
    "debugmodel_moe": _upstream_flavor("debugmodel_moe", "sdpa"),
    "0.6B": _upstream_flavor("0.6B", "sdpa"),
    "1.7B": _upstream_flavor("1.7B", "sdpa"),
    "4B": _upstream_flavor("4B", "sdpa"),
    "8B": _upstream_flavor("8B", "sdpa"),
    "32B": _upstream_flavor("32B", "sdpa"),
    "30B-A3B": _upstream_flavor("30B-A3B", "sdpa"),
}


# The TPU lane used to special-case 8B here to build separate Q/K/V projections;
# it now takes upstream's fused QKV (packed by TpuPackedQKVConverter in the
# recipes), so the upstream registry is used as is.
model_registry = qwen3_models.model_registry

from .config_registry import *
