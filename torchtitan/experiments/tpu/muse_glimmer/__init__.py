"""Muse Glimmer on TPU (text-only).

Wraps upstream ``torchtitan.models.muse_glimmer`` for the TPU lane:

* Swaps every layer's inner attention for
  ``SlidingWindowScaledDotProductInnerAttention`` so the per-layer sliding
  windows are honored on SDPA and picked up by the Splash attention patch.
  Upstream only implements the windows for flex/varlen, neither of which runs
  on TPU.
* Swaps every layer's ``Attention`` for ``TpuMuseGlimmerAttention``, which
  passes ``positions`` to the inner attention so packed documents are masked.
* Replaces upstream's ``parallelize_fn`` with
  ``infra.parallelize.parallelize_muse_glimmer`` (adds SimpleFSDP and the TPU
  compile modes).
"""

import dataclasses

import torchtitan.models.muse_glimmer as upstream_muse_glimmer
from torchtitan.models.muse_glimmer.model import MuseGlimmerModel
from torchtitan.models.muse_glimmer.state_dict_adapter import MuseGlimmerStateDictAdapter
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .attention import SlidingWindowScaledDotProductInnerAttention
from .attention import TpuMuseGlimmerAttention
from .infra.parallelize import parallelize_muse_glimmer

__all__ = [
    "SlidingWindowScaledDotProductInnerAttention",
    "TESTMODEL_MAX_CONTEXT_LENGTH",
    "TESTMODEL_WINDOW_SIZE",
    "TpuMuseGlimmerAttention",
    "model_registry",
    "muse_glimmer_configs",
    "parallelize_muse_glimmer",
    "use_tpu_attention",
]

# The small `testmodel` flavor used by the parity unit tests. The window is
# smaller than the tests' sequence lengths so the sliding window is exercised.
TESTMODEL_MAX_CONTEXT_LENGTH = 128
TESTMODEL_WINDOW_SIZE = 4

# Text-only upstream flavors supported on TPU.
_UPSTREAM_TEXT_FLAVORS = ("debugmodel", "30B")


def use_tpu_attention(config: MuseGlimmerModel.Config) -> MuseGlimmerModel.Config:
  """Switches each layer to the TPU attention modules.

  The outer attention becomes ``TpuMuseGlimmerAttention`` (same fields, passes
  ``positions`` through for packed-document masking) and the inner attention
  the window-aware SDPA one. Modifies ``config`` in place and returns it. The
  window is read from the outer ``Attention.Config.window_size`` (None for
  global layers), so the per-layer SWA/global pattern built by upstream is
  preserved.

  Args:
    config: A Muse Glimmer model config.

  Returns:
    The same config, with TPU inner attention on every layer.
  """
  for layer in config.layers:
    attention = TpuMuseGlimmerAttention.Config(**{
        field.name: getattr(layer.attention, field.name)
        for field in dataclasses.fields(layer.attention)
    })
    attention.inner_attention = SlidingWindowScaledDotProductInnerAttention.Config(
        window_size=attention.window_size
    )
    layer.attention = attention
  return config


def _testmodel() -> MuseGlimmerModel.Config:
  """4-layer model: layers 0-2 are sliding-window + RoPE, layer 3 global + NoPE.

  Uses 8 KV heads (vs. 2 in production) so TP up to 8 can be tested.
  """
  # pylint: disable-next=protected-access
  config = upstream_muse_glimmer._muse_glimmer_config(
      dim=128,
      n_layers=4,
      n_heads=16,
      n_kv_heads=8,
      head_dim=16,
      vocab_size=2048,
      max_context_length=TESTMODEL_MAX_CONTEXT_LENGTH,
      window_pattern=[TESTMODEL_WINDOW_SIZE] * 3 + [0],
      output_multiplier=1.0,
      attn_backend="sdpa",
  )
  return use_tpu_attention(config)


def _upstream_flavor(flavor: str) -> MuseGlimmerModel.Config:
  """Builds an upstream flavor at its full context length, with TPU attention."""
  build_config, max_context_length = upstream_muse_glimmer.muse_glimmer_configs[
      flavor
  ]
  return use_tpu_attention(
      build_config(attn_backend="sdpa", seq_len=max_context_length)
  )


muse_glimmer_configs = {
    "testmodel": _testmodel(),
    **{flavor: _upstream_flavor(flavor) for flavor in _UPSTREAM_TEXT_FLAVORS},
}


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
  """Returns a TPU ``ModelSpec`` for a text-only Muse Glimmer flavor.

  Args:
    flavor: ``testmodel`` or one of the upstream text flavors.
    seq_len: Context length; defaults to the flavor's maximum.
    converters: Optional model-config converters (applied before the TPU
      attention swap).

  Returns:
    The model spec, using ``parallelize_muse_glimmer``.
  """
  if flavor == "testmodel":
    if converters:
      raise ValueError("converters are not supported for testmodel")
    if seq_len is not None and seq_len > TESTMODEL_MAX_CONTEXT_LENGTH:
      raise ValueError(
          f"Requested seq_len {seq_len} exceeds max context length "
          f"{TESTMODEL_MAX_CONTEXT_LENGTH} for flavor {flavor}"
      )
    return ModelSpec(
        name="muse_glimmer",
        flavor=flavor,
        model=_testmodel(),
        max_context_length=seq_len or TESTMODEL_MAX_CONTEXT_LENGTH,
        parallelize_fn=parallelize_muse_glimmer,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=MuseGlimmerStateDictAdapter,
    )

  if flavor not in _UPSTREAM_TEXT_FLAVORS:
    raise ValueError(
        f"Unsupported Muse Glimmer flavor on TPU: {flavor!r}. Supported:"
        f" {('testmodel',) + _UPSTREAM_TEXT_FLAVORS}"
    )
  model_spec = upstream_muse_glimmer.model_registry(
      flavor,
      seq_len=seq_len,
      attn_backend="sdpa",
      converters=converters,
  )
  use_tpu_attention(model_spec.model)
  model_spec.parallelize_fn = parallelize_muse_glimmer
  # PP is not supported on TPU yet; parallelize_muse_glimmer rejects it.
  model_spec.pipelining_fn = None
  return model_spec


from .config_registry import *  # pylint: disable=wildcard-import,g-import-not-at-top,g-bad-import-order
