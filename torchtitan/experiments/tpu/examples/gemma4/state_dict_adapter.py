import json
import os
import re
import struct
from typing import Any

import torch

from torchtitan.experiments.tpu.examples.gemma4.model import Gemma4Model
from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

# Non-text towers of the multimodal checkpoint, with any leading "model." removed.
_IGNORED_HF_PREFIXES = (
    "vision_tower.",
    "embed_vision.",
    "audio_tower.",
    "embed_audio.",
    "multi_modal_projector.",
)


def _normalize_hf_key(key: str) -> str | None:
    """Normalize text-only HF keys (``model.layers.*``) to ``model.language_model.layers.*``.

    Returns None for keys of the non-text towers, which are ignored.
    """
    if key.startswith("model.language_model."):
        return key
    bare_key = key.removeprefix("model.")
    if bare_key.startswith(_IGNORED_HF_PREFIXES):
        return None
    if key.startswith("model."):
        return "model.language_model." + bare_key
    return key


def _single_file_safetensors_keys(hf_assets_path: str) -> set[str] | None:
    """Tensor names in a single-file ``model.safetensors``, or None if absent.

    Reads only the header (8-byte little-endian length + JSON), so this is
    cheap even for multi-GB checkpoints.
    """
    path = os.path.join(hf_assets_path, "model.safetensors")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    return {k for k in header if k != "__metadata__"}


class Gemma4StateDictAdapter(StateDictAdapter):
    """Converts state dicts between TorchTitan Gemma4Model and HuggingFace Gemma4 format."""

    def __init__(
        self, model_config: Gemma4Model.Config, hf_assets_path: str | None
    ) -> None:
        super().__init__(model_config, hf_assets_path)
        for idx, layer in enumerate(model_config.layers):
            if isinstance(layer.attention.qkv_linear, FusedQKVLinear.Config):
                raise NotImplementedError(
                    f"Gemma4StateDictAdapter does not support FusedQKVLinear "
                    f"(layer {idx} has use_fused_qkv=True)."
                )

        # ``to_hf`` builds the placeholder that ``dcp.load`` fills from disk, and
        # DCP is strict: every placeholder key must exist in the checkpoint. Two
        # things about Gemma 4 HF checkpoints therefore have to be known up
        # front, from the safetensors index (sharded) or the single-file header:
        #  * They are tied (tie_word_embeddings=True) and omit lm_head.weight,
        #    while the TPU recipes build an untied model. ``to_hf`` must then not
        #    request lm_head.weight and ``from_hf`` fills it from embed_tokens.
        #  * Multimodal Gemma4ForConditionalGeneration checkpoints (the released
        #    31B / 26B-A4B) use ``model.language_model.*``; text-only
        #    Gemma4ForCausalLM checkpoints use ``model.*``. ``from_hf`` accepts
        #    both; ``to_hf`` has to emit the layout that is on disk.
        # Unknown (no hf_assets_path / no safetensors there) keeps the defaults:
        # emit lm_head.weight and the multimodal layout.
        hf_keys: set[str] | None = None
        if self.fqn_to_index_mapping is not None:
            hf_keys = set(self.fqn_to_index_mapping)
        elif hf_assets_path:
            hf_keys = _single_file_safetensors_keys(hf_assets_path)
        self._hf_has_lm_head: bool | None = (
            "lm_head.weight" in hf_keys if hf_keys is not None else None
        )
        self._hf_text_only_layout: bool = bool(hf_keys) and not any(
            k.startswith("model.language_model.") for k in hf_keys
        )

        self.from_hf_map = {
            "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention module
            "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.language_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.language_model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.language_model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            # Transformer layer norms & scalar
            "model.language_model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.language_model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_layernorm.weight",
            "model.language_model.layers.{}.pre_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.language_model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.post_feedforward_layernorm.weight",
            "model.language_model.layers.{}.layer_scalar": "layers.{}.layer_scalar",
            # Dense MLP
            "model.language_model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.language_model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.language_model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # MoE norms, router, and down_proj
            "model.language_model.layers.{}.pre_feedforward_layernorm_2.weight": "layers.{}.moe.pre_feedforward_layernorm_2.weight",
            "model.language_model.layers.{}.post_feedforward_layernorm_1.weight": "layers.{}.post_feedforward_layernorm_1.weight",
            "model.language_model.layers.{}.post_feedforward_layernorm_2.weight": "layers.{}.post_feedforward_layernorm_2.weight",
            "model.language_model.layers.{}.router.proj.weight": "layers.{}.moe.router.gate.weight",
            "model.language_model.layers.{}.router.scale": "layers.{}.moe.router.scale",
            "model.language_model.layers.{}.router.per_expert_scale": "layers.{}.moe.router.per_expert_scale",
            "model.language_model.layers.{}.experts.down_proj": "layers.{}.moe.routed_experts.inner_experts.w2_EDF",
            # Final norm & output head
            "model.language_model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from a TorchTitan Gemma4 state dict to a HuggingFace state dict."""
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict: dict[str, Any] = {}
        pending_w1: dict[str, torch.Tensor] = {}
        pending_w3: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            if "layers" in key:
                match = re.search(r"\d+", key)
                if match is None:
                    continue
                layer_num = match.group(0)
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)

                if (
                    abstract_key
                    == "layers.{}.moe.routed_experts.inner_experts.w1_EFD"
                ):
                    pending_w1[layer_num] = value
                    if layer_num in pending_w3:
                        hf_key = f"model.language_model.layers.{layer_num}.experts.gate_up_proj"
                        hf_state_dict[hf_key] = torch.cat(
                            [pending_w1.pop(layer_num), pending_w3.pop(layer_num)],
                            dim=1,
                        )
                    continue

                if (
                    abstract_key
                    == "layers.{}.moe.routed_experts.inner_experts.w3_EFD"
                ):
                    pending_w3[layer_num] = value
                    if layer_num in pending_w1:
                        hf_key = f"model.language_model.layers.{layer_num}.experts.gate_up_proj"
                        hf_state_dict[hf_key] = torch.cat(
                            [pending_w1.pop(layer_num), pending_w3.pop(layer_num)],
                            dim=1,
                        )
                    continue

                if abstract_key in to_hf_map:
                    new_key = to_hf_map[abstract_key].format(layer_num)
                    hf_state_dict[new_key] = value
            else:
                if key == "lm_head.weight" and (
                    self.model_config.enable_weight_tying
                    or self._hf_has_lm_head is False
                ):
                    if self.fqn_to_index_mapping:
                        self.fqn_to_index_mapping.pop("lm_head.weight", None)
                    continue
                if key in to_hf_map:
                    hf_state_dict[to_hf_map[key]] = value

        if self._hf_text_only_layout:
            hf_state_dict = {
                k.replace("model.language_model.", "model.", 1): v
                for k, v in hf_state_dict.items()
            }
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from a HuggingFace Gemma4 state dict to a TorchTitan state dict."""
        state_dict: dict[str, Any] = {}

        embed_weight = (
            hf_state_dict.get("model.language_model.embed_tokens.weight")
            if "model.language_model.embed_tokens.weight" in hf_state_dict
            else hf_state_dict.get("model.embed_tokens.weight")
        )
        # Gemma 4 HF checkpoints tie embeddings and have no lm_head.weight key,
        # while TPU recipes set enable_weight_tying=False. Copy embed_tokens into
        # lm_head whenever the checkpoint has no lm_head.weight key.
        if "lm_head.weight" not in hf_state_dict and embed_weight is not None:
            state_dict["lm_head.weight"] = embed_weight

        for raw_key, value in hf_state_dict.items():
            key = _normalize_hf_key(raw_key)
            if key is None:
                continue

            if "layers" in key:
                match = re.search(r"\d+", key)
                if match is None:
                    continue
                layer_num = match.group(0)
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)

                if (
                    abstract_key
                    == "model.language_model.layers.{}.experts.gate_up_proj"
                ):
                    w1_EFD, w3_EFD = torch.chunk(value, 2, dim=1)
                    state_dict[
                        f"layers.{layer_num}.moe.routed_experts.inner_experts.w1_EFD"
                    ] = w1_EFD.contiguous()
                    state_dict[
                        f"layers.{layer_num}.moe.routed_experts.inner_experts.w3_EFD"
                    ] = w3_EFD.contiguous()
                    continue

                if abstract_key in self.from_hf_map:
                    new_key = self.from_hf_map[abstract_key].format(layer_num)
                    state_dict[new_key] = value
            else:
                if key in self.from_hf_map:
                    state_dict[self.from_hf_map[key]] = value

        return state_dict
