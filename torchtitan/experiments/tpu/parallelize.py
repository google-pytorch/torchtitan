# Copyright 2026 Google LLC

"""Routes ``model.parallelize(...)`` through the TPU parallelization steps.

Upstream models own their parallelization lifecycle (``BaseModel.parallelize``)
and the training engine calls it right after it builds the model. The TPU
recipes build upstream model classes, so the TPU steps around that lifecycle
are installed on the built model instance here: the model patches
(``workarounds.apply_patches``) and, for the regular trainer, the per-family
wrappers that add ``spmd_safe``, SimpleFSDP and the TPU compile modes.
"""

from collections.abc import Callable
import types
from typing import Any

from torchtitan.protocols.model import BaseModel


def tpu_parallelize_fn(model: BaseModel) -> Callable[..., Any] | None:
  """Returns the regular-trainer TPU parallelize wrapper for ``model``, if any.

  The wrappers call back into the model class's ``parallelize``. Gemma4 owns
  its TPU lifecycle in ``Gemma4Model.parallelize``; DeepSeek-V3, Flux and the
  HF backend use their own ``parallelize`` unchanged.
  """
  # pylint: disable=g-import-not-at-top
  from torchtitan.models.llama3 import Llama3Model
  from torchtitan.models.muse_glimmer import MuseGlimmerModel
  from torchtitan.models.qwen3 import Qwen3Model
  from torchtitan.models.qwen3_5.model import Qwen35Model

  if isinstance(model, Llama3Model):
    from torchtitan.experiments.tpu.llama3.infra.parallelize import parallelize_llama

    return parallelize_llama
  if isinstance(model, Qwen3Model):
    from torchtitan.experiments.tpu.qwen3.infra.parallelize import parallelize_qwen3

    return parallelize_qwen3
  if isinstance(model, Qwen35Model):
    from torchtitan.experiments.tpu.qwen3_5.infra.parallelize import parallelize_qwen3_5

    return parallelize_qwen3_5
  if isinstance(model, MuseGlimmerModel):
    from torchtitan.experiments.tpu.muse_glimmer.infra.parallelize import parallelize_muse_glimmer

    return parallelize_muse_glimmer
  # pylint: enable=g-import-not-at-top
  return None


def _takes_tpu_compile_mode(model: BaseModel) -> bool:
  # By name: importing the Gemma4 example package here would register it.
  return any(cls.__name__ == "Gemma4Model" for cls in type(model).__mro__)


def install_tpu_parallelize(
    model: BaseModel,
    config: Any,
    *,
    tpu_compile_mode: str | None = None,
) -> None:
  """Makes the engine's ``model.parallelize(...)`` call run the TPU steps.

  Installs a one-shot override on the instance. When the engine calls it, it
  applies the TPU model patches (right before parallelization, outside the
  meta-device context the model is built in) and then parallelizes.

  Args:
    model: The freshly built model.
    config: The trainer config, for ``workarounds.apply_patches``.
    tpu_compile_mode: The regular TPU trainer's ``compile.mode``. When set, the
      model's family wrapper (``tpu_parallelize_fn``) replaces the model's own
      ``parallelize`` and receives the mode, as does ``Gemma4Model``. ``None``
      (TpuGraphTrainer, whose models own their TPU lifecycle, and non-TPU
      configs) only applies the patches.
  """
  from torchtitan.experiments.tpu import workarounds  # pylint: disable=g-import-not-at-top

  tpu_fn = tpu_parallelize_fn(model) if tpu_compile_mode is not None else None

  def tpu_parallelize(self_model: BaseModel, **kwargs: Any):
    # Drop the override first: the TPU wrappers delegate to
    # `model.parallelize(...)`, which must reach the model class's method.
    del self_model.parallelize
    workarounds.apply_patches(self_model, config)
    if tpu_fn is not None:
      return tpu_fn(self_model, tpu_compile_mode=tpu_compile_mode, **kwargs)
    if tpu_compile_mode is not None and _takes_tpu_compile_mode(self_model):
      kwargs["tpu_compile_mode"] = tpu_compile_mode
    return self_model.parallelize(**kwargs)

  model.parallelize = types.MethodType(tpu_parallelize, model)
