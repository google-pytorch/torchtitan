"""DeepSeek-V3 on TPU.

Wraps upstream ``torchtitan.models.deepseek_v3`` (model, MTP, state-dict
adapter) for the TPU lane:

* Every layer's inner attention (MTP layers included) becomes the lane's
  document-masked SDPA (``tpu/masked_sdpa.py``,
  ``SlidingWindowScaledDotProductInnerAttention`` with ``window_size=None``).
  Upstream MLA already passes ``positions`` / ``segment_ids`` to its inner
  attention, but the stock ``ScaledDotProductInnerAttention`` drops them, so
  packed rows would attend across documents; ``flex`` (upstream's default)
  does not run on TPU. The SDPA takes MLA's unequal head dims (q/k
  ``qk_nope + qk_rope``, v ``v_head_dim``) as is.
* ``parallelize_fn`` is ``infra.parallelize.parallelize_deepseek_v3`` (AC ->
  per-block compile with ``dynamic=False`` -> SimpleFSDP or FSDP2); pipeline
  parallelism is not supported on TPU, so ``pipelining_fn`` is None.

Flavors: the small test flavors, upstream's ``debugmodel`` / ``16B`` /
``236B`` / ``671B``, ``671B-31L``, the 31-layer proxy of 671B (3 dense + 28
MoE layers, same widths) that matches MaxText's half-depth DeepSeek-V3 run on
one 4x4x4 v7x slice, and ``671B-2L-16E``, the smallest real-width proxy that
trains on 4 v6e chips (671B's widths and routing, 1 dense + 1 MoE layer, 16
instead of 256 routed experts), for per-rung measurements.
"""

import dataclasses

import torchtitan.models.common as common_models
import torchtitan.models.deepseek_v3 as deepseek_v3_models
from torchtitan.experiments.tpu.masked_sdpa import SlidingWindowScaledDotProductInnerAttention
from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model_spec import ModelSpec

from .infra.parallelize import parallelize_deepseek_v3
from .infra.parallelize import parallelize_deepseek_v3_graph_trainer

__all__ = [
    "PROXY_31L_N_LAYERS",
    "REDUCED_671B_FLAVOR",
    "deepseekv3_configs",
    "graph_trainer_model_registry",
    "model_registry",
    "parallelize_deepseek_v3",
    "parallelize_deepseekv3",
    "use_tpu_attention",
]

# The 671B proxy keeps upstream 671B's 3 dense layers and 28 of its 58 MoE
# layers.
PROXY_31L_N_LAYERS = 31

# Upstream flavors this module wraps, plus the proxy built from 671B.
_UPSTREAM_FLAVORS = ("debugmodel", "16B", "236B", "671B")
_PROXY_FLAVORS = {"671B-31L": ("671B", PROXY_31L_N_LAYERS)}

# The per-rung measurement proxy: 671B widths on every axis except depth and
# the routed-expert count, sized for 4 v6e chips (3.37B parameters, ~13.5 GB of
# fp32 parameters + gradients + AdamW state per chip under FSDP over 4). A
# third (MoE) layer (4.31B) still trains per-block compiled but leaves the
# GraphTrainer joint program 2.2 GB short on the SDPA path (measured).
REDUCED_671B_FLAVOR = "671B-2L-16E"
_REDUCED_671B_SHAPE = dict(n_dense_layers=1, n_moe_layers=1, num_experts=16)


def use_tpu_attention(
    config: deepseek_v3_models.DeepSeekV3Model.Config,
) -> deepseek_v3_models.DeepSeekV3Model.Config:
  """Switches every layer (MTP layers too) to the document-masked SDPA.

  Modifies ``config`` in place and returns it.

  Args:
    config: A DeepSeek-V3 model config.

  Returns:
    The same config.
  """
  for layer in [*config.layers, *config.mtp_layers]:
    layer.attention.inner_attention = (
        SlidingWindowScaledDotProductInnerAttention.Config()
    )
  return config


def _truncate_layers(
    config: deepseek_v3_models.DeepSeekV3Model.Config, n_layers: int
) -> deepseek_v3_models.DeepSeekV3Model.Config:
  """Keeps the first ``n_layers`` decoder layers (dense ones come first).

  The MTP layers are rebuilt from the new last layer, as upstream builds them
  from its last layer, so their depth-scaled init follows the proxy's depth.
  """
  num_mtp_layers = len(config.mtp_layers)
  config.layers = config.layers[:n_layers]
  # pylint: disable-next=protected-access
  config.mtp_layers = deepseek_v3_models._build_mtp_layers(
      config.layers[-1], dim=config.dim, num_mtp_layers=num_mtp_layers
  )
  return config


def _testmodel() -> deepseek_v3_models.DeepSeekV3Model.Config:
  return _small_testmodel(n_dense_layers=3)


def _testmodel_moe() -> deepseek_v3_models.DeepSeekV3Model.Config:
  return _small_testmodel(n_dense_layers=1)


def _small_testmodel(
    n_dense_layers: int,
) -> deepseek_v3_models.DeepSeekV3Model.Config:
  """3-layer dim-128 model used by the CPU-vs-device component tests."""
  dim = 128
  head_dim = 16
  n_layers = 3
  vocab_size = 2048
  # pylint: disable-next=protected-access
  layers = deepseek_v3_models._build_dsv3_layers(
      n_layers=n_layers,
      n_dense_layers=n_dense_layers,
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
      attn_backend="sdpa",
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
  # pylint: disable=protected-access
  config = deepseek_v3_models.DeepSeekV3Model.Config(
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
  # pylint: enable=protected-access
  return use_tpu_attention(config)


def _upstream_flavor(
    flavor: str, num_mtp_layers: int = 0
) -> deepseek_v3_models.DeepSeekV3Model.Config:
  """Builds an upstream flavor (or the proxy) at its full context length.

  Upstream d771abb0 changed ``deepseekv3_configs[flavor]`` into a
  ``(builder, max_context_length)`` pair with a required ``seq_len``.
  """
  base_flavor, n_layers = _PROXY_FLAVORS.get(flavor, (flavor, None))
  build_config, max_context_length = deepseek_v3_models.deepseekv3_configs[
      base_flavor
  ]
  config = build_config(
      attn_backend="sdpa",
      moe_comm_backend="standard",
      num_mtp_layers=num_mtp_layers,
      seq_len=max_context_length,
  )
  if n_layers is not None:
    config = _truncate_layers(config, n_layers)
  return use_tpu_attention(config)


def _reduced_671b(
    *,
    n_dense_layers: int,
    n_moe_layers: int,
    num_experts: int,
    seq_len: int,
) -> deepseek_v3_models.DeepSeekV3Model.Config:
  """671B with fewer layers and routed experts, everything else unchanged.

  Upstream's ``_671b`` has no knob for the expert count, so its layer
  arguments are repeated here (``TestDeepSeekV3Flavors`` checks them against
  ``transformers.DeepseekV3Config``); embeddings, final norm and head come from
  upstream's 671B config. Group-limited routing keeps 8 groups / top-4 groups /
  top-8 experts, so with 16 experts every group holds 2 and the top-8 are the
  experts of the 4 chosen groups.
  """
  config = _upstream_flavor("671B")
  rope = config.layers[0].attention.rope
  # pylint: disable-next=protected-access
  config.layers = deepseek_v3_models._build_dsv3_layers(
      n_layers=n_dense_layers + n_moe_layers,
      n_dense_layers=n_dense_layers,
      dim=7168,
      n_heads=128,
      q_lora_rank=1536,
      kv_lora_rank=512,
      qk_nope_head_dim=128,
      qk_rope_head_dim=64,
      v_head_dim=128,
      mscale=1.0,
      dense_hidden_dim=18432,
      moe_hidden_dim=2048,
      num_experts=num_experts,
      num_shared_experts=1,
      router_top_k=8,
      router_score_func="sigmoid",
      router_num_expert_groups=8,
      router_num_limited_groups=4,
      router_route_scale=2.5,
      router_route_norm=True,
      aux_loss_coeff=1e-3,
      attn_backend="sdpa",
      moe_comm_backend="standard",
      non_blocking_capacity_factor=None,
      rope=dataclasses.replace(rope, max_context_length=seq_len),
  )
  config.mtp_layers = []
  return use_tpu_attention(config)


deepseekv3_configs = {
    "testmodel": _testmodel(),
    "testmodel_moe": _testmodel_moe(),
    **{
        flavor: _upstream_flavor(flavor)
        for flavor in (*_UPSTREAM_FLAVORS, *_PROXY_FLAVORS)
    },
    REDUCED_671B_FLAVOR: _reduced_671b(**_REDUCED_671B_SHAPE, seq_len=4096),
}


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
    num_mtp_layers: int = 0,
) -> ModelSpec:
  """Returns a TPU ``ModelSpec`` for a DeepSeek-V3 flavor.

  Args:
    flavor: One of upstream's flavors or ``671B-31L``.
    seq_len: Context length; defaults to the flavor's maximum.
    converters: Optional model-config converters, applied before the TPU
      attention swap.
    num_mtp_layers: Multi-token-prediction depth (0 disables MTP).

  Returns:
    The model spec, using ``parallelize_deepseek_v3``.
  """
  if flavor == REDUCED_671B_FLAVOR:
    if num_mtp_layers:
      raise ValueError(f"{flavor} has no MTP layers")
    model_spec = model_registry("671B", seq_len=seq_len)
    config = _reduced_671b(
        **_REDUCED_671B_SHAPE, seq_len=model_spec.max_context_length
    )
    # As upstream's model_registry: converters see the final layer list.
    if converters is not None:
      validate_converter_order(converters)
      for converter in converters:
        config = converter.build().convert(config)
    return dataclasses.replace(model_spec, flavor=flavor, model=config)
  base_flavor, n_layers = _PROXY_FLAVORS.get(flavor, (flavor, None))
  if base_flavor not in _UPSTREAM_FLAVORS:
    raise ValueError(
        f"Unsupported DeepSeek-V3 flavor on TPU: {flavor!r}. Supported:"
        f" {_UPSTREAM_FLAVORS + tuple(_PROXY_FLAVORS) + (REDUCED_671B_FLAVOR,)}"
    )
  model_spec = deepseek_v3_models.model_registry(
      base_flavor,
      seq_len=seq_len,
      attn_backend="sdpa",
      moe_comm_backend="standard",
      converters=converters,
      num_mtp_layers=num_mtp_layers,
  )
  if n_layers is not None:
    _truncate_layers(model_spec.model, n_layers)
  use_tpu_attention(model_spec.model)
  return dataclasses.replace(
      model_spec,
      flavor=flavor,
      parallelize_fn=parallelize_deepseek_v3,
      # PP is not supported on TPU; parallelize_deepseek_v3 rejects it.
      pipelining_fn=None,
  )


def graph_trainer_model_registry(flavor: str, **kwargs) -> ModelSpec:
  """Returns the TPU GraphTrainer ``ModelSpec`` for a DeepSeek-V3 flavor.

  ``model_registry``'s spec (TPU attention, MoE balancers, state-dict adapter,
  optimizer hooks) with ``parallelize_deepseek_v3_graph_trainer``. The model
  class stays ``DeepSeekV3Model``: every parameterized module has a
  ``param_init``, so ``init_states`` needs no SimpleFSDP-parametrization
  wrapper (skill ``references/model.md``).

  Args:
    flavor: As for ``model_registry``.
    **kwargs: Forwarded to ``model_registry``.

  Returns:
    The GraphTrainer model spec.
  """
  return dataclasses.replace(
      model_registry(flavor, **kwargs),
      parallelize_fn=parallelize_deepseek_v3_graph_trainer,
      pipelining_fn=None,
  )


from .config_registry import *  # pylint: disable=wildcard-import,g-import-not-at-top,g-bad-import-order
