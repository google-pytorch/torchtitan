# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dict adapters for the HF transformers backend.

Two complementary pieces live here:

- ``HFTransformerStateDictAdapter`` -- the ``StateDictAdapter`` plugged into the
  model class. HFTransformerModel wraps an HF ForCausalLM as ``self.model``, so the
  only difference between TorchTitan FQNs and HF safetensors keys is a ``model.``
  prefix (plus tied-embedding handling). Used by the checkpoint system.

- ``hf_to_titan_moe_state_dict`` / ``titan_to_hf_moe_state_dict`` -- functions
  that convert MoE expert weights between HF and Titan layouts after the Titan
  MoE replacement (packed ``gate_up_proj`` <-> structured W13, plus router and
  shared-expert key renames). Non-MoE keys pass through. Used by the
  numerical-equivalence and round-trip tests.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import Any, TYPE_CHECKING

import torch
from safetensors import safe_open

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

if TYPE_CHECKING:
    from .model import HFTransformerModel

logger = logging.getLogger(__name__)

# Mapping from titan key -> original HF key for keys where the reverse
# regex would produce a different result than the original (e.g. a
# ``router.weight`` that must map back exactly rather than to ``gate.weight``).
_TITAN_TO_ORIGINAL_HF_KEY: dict[str, str] = {}

_TITAN_W13_WEIGHT = "routed_experts.w13.weight"
_TITAN_W2_WEIGHT = "routed_experts.w2.weight"


class HFTransformerStateDictAdapter(StateDictAdapter):
    """State dict adapter for HFTransformerModel.

    Since HFTransformerModel wraps an HF ForCausalLM as self.model, the only
    difference between TorchTitan FQNs and HF safetensors keys is a "model."
    prefix. No weight reshaping or renaming is needed.

    Handles weight tying: when tie_word_embeddings is True, some HF checkpoints
    omit lm_head.weight from safetensors (it shares storage with embed_tokens).

    Each key is resolved against the keys the checkpoint actually has, so a
    text-only model can also load the decoder of a multimodal checkpoint, which
    stores it under `model.language_model.*` (see `_checkpoint_key`).
    """

    def __init__(
        self,
        model_config: HFTransformerModel.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self._tie_word_embeddings = getattr(model_config, "tie_word_embeddings", False)
        self._checkpoint_keys = (
            _checkpoint_keys(hf_assets_path)
            if hf_assets_path and os.path.isdir(hf_assets_path)
            else None
        )
        # HF key of the model -> key in the checkpoint, recorded by to_hf so
        # that from_hf can invert it.
        self._checkpoint_key_of: dict[str, str] = {}

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        # When weights are tied, lm_head.weight may not exist in the
        # safetensors file. Exclude it so DCP doesn't fail on missing key.
        if (
            self._tie_word_embeddings
            and "lm_head.weight" in hf_state_dict
            and _embedding_key(hf_state_dict) is not None
        ):
            del hf_state_dict["lm_head.weight"]
        self._checkpoint_key_of = {k: self._checkpoint_key(k) for k in hf_state_dict}
        num_mapped = sum(k != v for k, v in self._checkpoint_key_of.items())
        if num_mapped:
            logger.info(
                f"Mapped {num_mapped} keys to '{_COMPOSITE_TEXT_MODEL_PREFIX}' keys of "
                f"the multimodal HF checkpoint at {self.hf_assets_path}."
            )
        return {self._checkpoint_key_of[k]: v for k, v in hf_state_dict.items()}

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        model_key_of = {v: k for k, v in self._checkpoint_key_of.items()}
        hf_state_dict = {model_key_of.get(k, k): v for k, v in hf_state_dict.items()}
        # Reconstruct lm_head.weight from embed_tokens if it was excluded
        embedding_key = _embedding_key(hf_state_dict)
        if "lm_head.weight" not in hf_state_dict and embedding_key is not None:
            hf_state_dict["lm_head.weight"] = hf_state_dict[embedding_key]
        return {"model." + k: v for k, v in hf_state_dict.items()}

    def _checkpoint_key(self, key: str) -> str:
        """Returns the checkpoint key that holds the model's HF key `key`.

        A key the checkpoint has is used as is, so text-only models load
        text-only checkpoints and multimodal models load multimodal ones
        unchanged. A text-only model's decoder key (`model.*`) missing from a
        multimodal checkpoint is read from `model.language_model.*`, where
        multimodal checkpoints (e.g. Gemma-3/4 ForConditionalGeneration) store
        the same decoder next to their vision/audio towers.
        """
        if self._checkpoint_keys is None or key in self._checkpoint_keys:
            return key
        if key.startswith("model.") and not key.startswith(
            _COMPOSITE_TEXT_MODEL_PREFIX
        ):
            composite_key = _COMPOSITE_TEXT_MODEL_PREFIX + key.removeprefix("model.")
            if composite_key in self._checkpoint_keys:
                return composite_key
        return key


_COMPOSITE_TEXT_MODEL_PREFIX = "model.language_model."

_EMBEDDING_KEYS = (
    "model.embed_tokens.weight",
    _COMPOSITE_TEXT_MODEL_PREFIX + "embed_tokens.weight",
)


def _embedding_key(hf_state_dict: dict[str, Any]) -> str | None:
    """Returns the input embedding's key, for text-only or multimodal models."""
    return next((k for k in _EMBEDDING_KEYS if k in hf_state_dict), None)


def _checkpoint_keys(hf_assets_path: str) -> set[str]:
    """Returns the tensor names of the safetensors checkpoint in a folder."""
    index_path = os.path.join(hf_assets_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return set(json.load(f)["weight_map"])
    keys = set()
    for path in glob.glob(os.path.join(hf_assets_path, "*.safetensors")):
        with safe_open(path, framework="pt") as f:
            keys.update(f.keys())
    return keys


def _build_hf_to_titan_patterns() -> list[tuple[str, str, bool]]:
    return [
        (r"^(.*\.)gate\.weight$", r"\1router.gate.weight", False),
        (r"^(.*\.)router\.weight$", r"\1router.gate.weight", False),
        (r"^(.*\.)router\.proj\.weight$", r"\1router.gate.weight", False),
        (r"^(.*\.)gate\.e_score_correction_bias$", r"\1expert_bias", False),
        (r"^(.*\.)experts\.down_proj$", rf"\1{_TITAN_W2_WEIGHT}", False),
        # Shared experts use FeedForward (w1/w2/w3 attribute names, not params)
        (r"^(.*\.shared_experts)\.gate_proj\.weight$", r"\1.w1.weight", False),
        (r"^(.*\.shared_experts)\.up_proj\.weight$", r"\1.w3.weight", False),
        (r"^(.*\.shared_experts)\.down_proj\.weight$", r"\1.w2.weight", False),
        (
            r"^(.*\.)shared_expert\.gate_proj\.weight$",
            r"\1shared_experts.w1.weight",
            False,
        ),
        (
            r"^(.*\.)shared_expert\.up_proj\.weight$",
            r"\1shared_experts.w3.weight",
            False,
        ),
        (
            r"^(.*\.)shared_expert\.down_proj\.weight$",
            r"\1shared_experts.w2.weight",
            False,
        ),
        (r"^(.*\.)shared_expert_gate\.weight$", r"\1shared_experts.gate.weight", False),
    ]


def _build_titan_to_hf_patterns() -> list[tuple[str, str, bool]]:
    return [
        (r"^(.*\.)router\.gate\.weight$", r"\1gate.weight", False),
        (r"^(.*\.)expert_bias$", r"\1gate.e_score_correction_bias", False),
        (
            rf"^(.*\.){re.escape(_TITAN_W2_WEIGHT)}$",
            r"\1experts.down_proj",
            False,
        ),
        (
            r"^(.*\.)shared_experts\.w1\.weight$",
            r"\1shared_experts.gate_proj.weight",
            False,
        ),
        (
            r"^(.*\.)shared_experts\.w3\.weight$",
            r"\1shared_experts.up_proj.weight",
            False,
        ),
        (
            r"^(.*\.)shared_experts\.w2\.weight$",
            r"\1shared_experts.down_proj.weight",
            False,
        ),
        (
            r"^(.*\.)shared_experts\.gate\.weight$",
            r"\1shared_expert_gate.weight",
            False,
        ),
    ]


def hf_to_titan_moe_state_dict(
    hf_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert HF MoE state dict keys/values to titan format.

    Handles packed ``gate_up_proj`` -> structured W13, router and shared
    expert key renames, and expert bias mapping. Non-MoE keys pass through.

    Args:
        hf_state_dict: State dict with HF-format keys.

    Returns:
        State dict with titan-format keys and structured expert weights.
    """
    patterns = _build_hf_to_titan_patterns()
    titan_state_dict = {}

    for key, value in hf_state_dict.items():
        # HF stores W13 as [E, 2F, D]; Titan preserves gate/up as a semantic
        # axis so TP can shard F without separating the two projections.
        # Match ``.experts.`` with a leading dot so the routed experts are not
        # confused with a shared expert (e.g. ``shared_experts.down_proj``).
        if key.endswith(".experts.gate_up_proj"):
            prefix = key[: -len("experts.gate_up_proj")]
            titan_state_dict[f"{prefix}{_TITAN_W13_WEIGHT}"] = value.unflatten(
                1, (2, -1)
            )
            continue

        # Handle expert down_proj rename.
        if key.endswith(".experts.down_proj"):
            prefix = key[: -len("experts.down_proj")]
            titan_state_dict[f"{prefix}{_TITAN_W2_WEIGHT}"] = value
            continue

        # Try regex patterns
        converted = False
        for pattern, replacement, _ in patterns:
            new_key = re.sub(pattern, replacement, key)
            if new_key != key:
                titan_state_dict[new_key] = value
                converted = True
                _TITAN_TO_ORIGINAL_HF_KEY[new_key] = key
                break

        if not converted:
            titan_state_dict[key] = value

    return titan_state_dict


def titan_to_hf_moe_state_dict(
    titan_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert titan MoE state dict keys/values to HF format.

    Inverse of ``hf_to_titan_moe_state_dict``. Packs structured W13 back into
    ``gate_up_proj`` and reverses all key renames.

    Args:
        titan_state_dict: State dict with titan-format keys.

    Returns:
        State dict with HF-format keys and packed expert weights.
    """
    patterns = _build_titan_to_hf_patterns()
    hf_state_dict = {}

    for key, value in titan_state_dict.items():
        if key.endswith(_TITAN_W13_WEIGHT):
            prefix = key[: -len(_TITAN_W13_WEIGHT)]
            hf_state_dict[f"{prefix}experts.gate_up_proj"] = value.flatten(1, 2)
            continue

        if key in _TITAN_TO_ORIGINAL_HF_KEY:
            hf_state_dict[_TITAN_TO_ORIGINAL_HF_KEY[key]] = value
            continue

        converted = False
        for pattern, replacement, _ in patterns:
            new_key = re.sub(pattern, replacement, key)
            if new_key != key:
                hf_state_dict[new_key] = value
                converted = True
                break

        if not converted:
            hf_state_dict[key] = value

    return hf_state_dict
