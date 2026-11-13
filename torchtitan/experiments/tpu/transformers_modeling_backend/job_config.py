"""Job config for the HF backend on the TPU lane."""

import dataclasses

from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig


@dataclasses.dataclass(kw_only=True, slots=True)
class TPUTransformersBackendConfig(TPUTrainerConfig):
  """A TPU job config that also carries the HuggingFace model id.

  The HF backend reads the id off the *job* config -- ``HFTransformerModel.
  Config.update_from_config`` does ``getattr(config, "hf_model", "")`` -- so it
  has to be a top-level field rather than one of the per-model sub-configs.
  ``TPUTrainerConfig`` is a slotted dataclass, so it cannot simply be set at
  runtime; subclassing is what adds the slot.

  Subclassing ``TPUTrainerConfig`` (rather than the backend's
  ``SFTTrainer.Config``) is what lets the TPU entry point accept this config
  unchanged: ``gmain`` reaches for ``config.tpu_config``, and ``train.py``
  gates its TPU behavior on ``isinstance(config, TPUTrainerConfig)``.
  """

  hf_model: str = ""
  # A JSON object applied to the loaded HF config after the flavor's
  # overrides, e.g. ``{"attn_pdrop": 0, "resid_pdrop": 0}`` to train a
  # checkpoint without its dropout. Read by the shared
  # backend's ``update_from_config`` through the same ``getattr(config, ...)``
  # path as ``hf_model``.
  hf_config_overrides: str = ""
  # ESM-C only: run QK-LayerNorm + head split + RoPE + heads-first as one
  # fused Pallas kernel each way (kernels/qk_norm_rope.py) instead of XLA's
  # norm / relayout-copy / RoPE passes. Read by the model config's
  # ``update_from_config`` and carried as ``tpu_fused_qk_norm_rope``.
  esmc_fused_qk_norm_rope: bool = False
  # ESM-C only: SwiGLU that saves sigmoid(gate) so the backward does not
  # recompute it in the fused down-projection dX epilogue (+1 activation of
  # (tokens, intermediate_size) per layer). Carried as
  # ``tpu_swiglu_saved_sigmoid``.
  esmc_swiglu_saved_sigmoid: bool = False
  hf_float8_recipe: str = ""
  hf_float8_modules: str = ""
