"""Qwen3.8 (text-only) on TPU.

Qwen3.8 checkpoints use the Qwen3.5 architecture: upstream
``torchtitan.models.qwen3_8`` is a version-specific registry over
``torchtitan.models.qwen3_5``. Likewise, this package is a registry over the
upstream ``qwen3_8`` flavors that applies the Qwen3.5 TPU lane
(``experiments/tpu/qwen3_5``, ``to_tpu_model_spec``): TPU GatedDeltaNet and
document-masked SDPA modules, no vision encoder, SimpleFSDP-aware
parallelization. It imports the Qwen3.5 TPU package because it is the same
model, not a different one.
"""

# Before `torchtitan.models.qwen3_5`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu.qwen3_5 import attn_gym_stub  # pylint: disable=unused-import,g-bad-import-order
from torchtitan.experiments.tpu.qwen3_5 import to_tpu_model_spec
import torchtitan.models.qwen3_8 as upstream_qwen3_8
from torchtitan.models.qwen3_8 import qwen3_8_configs
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

__all__ = [
    "model_registry",
    "qwen3_8_configs",
]


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "sdpa",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
  """Returns a TPU ``ModelSpec`` for a text-only Qwen3.8 flavor.

  Args:
    flavor: An upstream Qwen3.8 flavor (``debugmodel``, ``debugmodel_moe``,
      ``27B``, ``2.4T-A95B``).
    seq_len: Context length; defaults to the flavor's maximum.
    attn_backend: Only ``"sdpa"`` (see ``experiments/tpu/qwen3_5``).
    converters: Optional model-config converters.

  Returns:
    The model spec, named ``qwen3_8``.
  """
  if attn_backend != "sdpa":
    raise ValueError(
        "Qwen3.8 on TPU supports only attn_backend='sdpa', got "
        f"{attn_backend!r}"
    )
  model_spec = upstream_qwen3_8.model_registry(
      flavor, seq_len=seq_len, attn_backend="sdpa", converters=converters
  )
  return to_tpu_model_spec(model_spec)


from .config_registry import *  # pylint: disable=wildcard-import,g-import-not-at-top,g-bad-import-order
