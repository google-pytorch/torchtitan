"""Flint (RecursiveSI/flint, 39.8B total / 2.5B active MoE) for the TPU lane.

Architecture constants are from the HF checkpoint's ``config.json``
(``model_type: flint``, transformers 5.21 ``FlintConfig``).
"""

from collections.abc import Callable, Sequence
import copy
from functools import partial

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.config.override import derive
from torchtitan.experiments.tpu.qwen3.infra.parallelize import parallelize_qwen3
from torchtitan.models.common import CosSinRoPE, Embedding, Linear
from torchtitan.models.common.config_utils import (
    make_ffn_config,
    make_gqa_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.qwen3.model import Qwen3TransformerBlock
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import FlintAttention, FlintModel, FlintSinkAttention
from .sharding import set_flint_sharding_config
from .state_dict_adapter import FlintStateDictAdapter

__all__ = [
    "FlintAttention",
    "FlintModel",
    "FlintSinkAttention",
    "FlintStateDictAdapter",
    "flint_layer_types",
    "model_registry",
]

_EPS = 1e-6
_INIT_STD = 0.02
_LINEAR_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.normal_, std=_INIT_STD)
}
_NORM_INIT: dict[str, Callable] = {"weight": nn.init.ones_}
_EMBEDDING_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.normal_, std=_INIT_STD)
}
# HF FlintPreTrainedModel._init_weights: normal(0, initializer_range).
_SINKS_INIT: dict[str, Callable] = {
    "sinks": partial(nn.init.normal_, std=_INIT_STD)
}


def _residual_output_init(n_layers: int) -> dict[str, Callable]:
  # Megatron-style scaled init for residual output projections (as Qwen3).
  return {"weight": partial(nn.init.normal_, std=_INIT_STD / (2 * n_layers) ** 0.5)}


def _norm(dim: int) -> RMSNorm.Config:
  return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def flint_layer_types(n_layers: int) -> list[str]:
  """HF FlintConfig default: full attention on layers 0 and 1, every
  ``i % 4 == 1`` layer and the last layer; sliding window elsewhere."""
  return [
      "full_attention"
      if i < 2 or i % 4 == 1 or i == n_layers - 1
      else "sliding_attention"
      for i in range(n_layers)
  ]


def _build_flint_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    shared_hidden_dim: int,
    num_experts: int,
    top_k: int,
    sliding_window: int,
    rope_theta: float,
    max_context_length: int,
    layer_types: Sequence[str] | None = None,
    mlp_only_layers: Sequence[int] = (0, 1),
    load_balance_coeff: float | None = 1e-3,
    moe_comm_backend: str = "standard",
) -> list[Qwen3TransformerBlock.Config]:
  layer_types = list(layer_types or flint_layer_types(n_layers))
  if len(layer_types) != n_layers:
    raise ValueError(
        f"layer_types has {len(layer_types)} entries for {n_layers} layers"
    )
  output_init = _residual_output_init(n_layers)
  experts_init = {
      "w1_EFD": _LINEAR_INIT["weight"],
      "w2_EDF": output_init["weight"],
      "w3_EFD": _LINEAR_INIT["weight"],
  }
  layers = []
  for layer_id, layer_type in enumerate(layer_types):
    sliding = layer_type == "sliding_attention"
    if not sliding and layer_type != "full_attention":
      raise ValueError(f"unknown layer type {layer_type!r}")
    gqa = make_gqa_config(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        wqkv_param_init=_LINEAR_INIT,
        wo_param_init=output_init,
        inner_attention=FlintSinkAttention.Config(
            window_size=sliding_window if sliding else None
        ),
        # Only the sliding layers are rotary; full attention is NoPE.
        rope=CosSinRoPE.Config(
            dim=head_dim,
            max_context_length=max_context_length,
            theta=rope_theta,
        )
        if sliding
        else None,
        qk_norm=_norm(head_dim),
    )
    attention = derive(gqa, FlintAttention.Config, param_init=_SINKS_INIT)
    if layer_id in mlp_only_layers:
      ffn = make_ffn_config(
          dim=dim,
          hidden_dim=dense_hidden_dim,
          w1_param_init=_LINEAR_INIT,
          w2w3_param_init=_LINEAR_INIT,
      )
      moe = None
    else:
      ffn = None
      moe = make_moe_config(
          num_experts=num_experts,
          load_balance_coeff=load_balance_coeff,
          router=make_router_config(
              dim=dim,
              num_experts=num_experts,
              gate_param_init=_LINEAR_INIT,
              top_k=top_k,
              score_func="sigmoid",
              route_norm=True,
          ),
          routed_experts=make_routed_experts_config(
              dim=dim,
              hidden_dim=moe_hidden_dim,
              num_experts=num_experts,
              top_k=top_k,
              param_init=experts_init,
              comm_backend=moe_comm_backend,
          ),
          # Flint's shared expert is always added in full (no sigmoid gate).
          shared_experts=make_ffn_config(
              dim=dim,
              hidden_dim=shared_hidden_dim,
              w1_param_init=_LINEAR_INIT,
              w2w3_param_init=output_init,
          ),
      )
    layers.append(
        Qwen3TransformerBlock.Config(
            attention_norm=_norm(dim),
            ffn_norm=_norm(dim),
            attention=attention,
            feed_forward=ffn,
            moe=moe,
        )
    )
  return layers


def _model_config(
    *, dim: int, vocab_size: int, layers: list
) -> FlintModel.Config:
  return FlintModel.Config(
      dim=dim,
      vocab_size=vocab_size,
      norm=_norm(dim),
      enable_weight_tying=False,
      tok_embeddings=Embedding.Config(
          num_embeddings=vocab_size,
          embedding_dim=dim,
          param_init=_EMBEDDING_INIT,
      ),
      lm_head=Linear.Config(
          in_features=dim, out_features=vocab_size, param_init=_LINEAR_INIT
      ),
      layers=layers,
  )


def _flavor(flavor: str, *, max_context_length: int, moe_comm_backend: str):
  common = dict(
      max_context_length=max_context_length, moe_comm_backend=moe_comm_backend
  )
  if flavor == "testmodel":
    # One of each layer kind: dense/full, MoE/sliding, MoE/full, MoE/sliding.
    return _model_config(
        dim=128,
        vocab_size=2048,
        layers=_build_flint_layers(
            n_layers=4,
            dim=128,
            n_heads=8,
            n_kv_heads=2,
            head_dim=32,
            dense_hidden_dim=256,
            moe_hidden_dim=64,
            shared_hidden_dim=128,
            num_experts=8,
            top_k=2,
            sliding_window=8,
            rope_theta=10000.0,
            layer_types=[
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
            mlp_only_layers=(0,),
            **common,
        ),
    )
  if flavor == "debugmodel":
    return _model_config(
        dim=256,
        vocab_size=2048,
        layers=_build_flint_layers(
            n_layers=8,
            dim=256,
            n_heads=8,
            n_kv_heads=2,
            head_dim=64,
            dense_hidden_dim=768,
            moe_hidden_dim=128,
            shared_hidden_dim=256,
            num_experts=8,
            top_k=2,
            sliding_window=129,
            rope_theta=10000.0,
            **common,
        ),
    )
  if flavor == "40B":
    return _model_config(
        dim=2048,
        vocab_size=150144,
        layers=_build_flint_layers(
            n_layers=49,
            dim=2048,
            n_heads=32,
            n_kv_heads=4,
            head_dim=128,
            dense_hidden_dim=6144,
            moe_hidden_dim=512,
            shared_hidden_dim=1024,
            num_experts=256,
            top_k=6,
            sliding_window=513,
            rope_theta=10000.0,
            **common,
        ),
    )
  raise ValueError(f"unknown Flint flavor {flavor!r}")


_DEFAULT_MAX_CONTEXT = {"testmodel": 128, "debugmodel": 2048, "40B": 4096}
FLAVORS = tuple(_DEFAULT_MAX_CONTEXT)


def build_flavor_config(
    flavor: str,
    *,
    max_context_length: int | None = None,
    num_layers: int | None = None,
    moe_comm_backend: str = "standard",
    enable_sp: bool = True,
    enable_ep: bool = False,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> FlintModel.Config:
  """Flavor config; ``num_layers`` truncates to the first N layers (proxy runs)."""
  config = copy.deepcopy(
      _flavor(
          flavor,
          max_context_length=max_context_length
          or _DEFAULT_MAX_CONTEXT[flavor],
          moe_comm_backend=moe_comm_backend,
      )
  )
  if num_layers is not None:
    if not 0 < num_layers <= len(config.layers):
      raise ValueError(f"num_layers must be in [1, {len(config.layers)}]")
    config.layers = config.layers[:num_layers]
  set_flint_sharding_config(config, enable_sp=enable_sp, enable_ep=enable_ep)
  if converters is not None:
    validate_converter_order(converters)
    for c in converters:
      config = c.build().convert(config)
  return config


def model_registry(
    flavor: str,
    *,
    max_context_length: int | None = None,
    num_layers: int | None = None,
    moe_comm_backend: str = "standard",
    enable_sp: bool = True,
    enable_ep: bool = False,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
  config = build_flavor_config(
      flavor,
      max_context_length=max_context_length,
      num_layers=num_layers,
      moe_comm_backend=moe_comm_backend,
      enable_sp=enable_sp,
      enable_ep=enable_ep,
      converters=converters,
  )
  return ModelSpec(
      name="flint",
      flavor=flavor,
      model=config,
      max_context_length=max_context_length or _DEFAULT_MAX_CONTEXT[flavor],
      parallelize_fn=parallelize_qwen3,
      pipelining_fn=None,
      post_optimizer_build_fn=register_moe_load_balancing_hook,
      state_dict_adapter=FlintStateDictAdapter,
  )
