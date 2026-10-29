# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for HFTransformerStateDictAdapter key mapping.

Run with:
    python -m pytest \
      torchtitan/experiments/transformers_modeling_backend/tests/test_state_dict_adapter.py
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from torchtitan.experiments.transformers_modeling_backend.state_dict_adapter import (
    HFTransformerStateDictAdapter,
)

# TorchTitan FQNs of a tied-embedding HFTransformerModel wrapping a text-only
# ForCausalLM as `model`.
_TEXT_MODEL_KEYS = (
    "model.model.embed_tokens.weight",
    "model.model.layers.0.mlp.up_proj.weight",
    "model.model.norm.weight",
    "model.lm_head.weight",
)

# TorchTitan FQNs of a tied-embedding HFTransformerModel wrapping a multimodal
# ForConditionalGeneration as `model`.
_MULTIMODAL_MODEL_KEYS = (
    "model.model.language_model.embed_tokens.weight",
    "model.model.language_model.layers.0.mlp.up_proj.weight",
    "model.model.language_model.norm.weight",
    "model.model.vision_tower.encoder.weight",
    "model.lm_head.weight",
)

_TEXT_CHECKPOINT_KEYS = (
    "model.embed_tokens.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.norm.weight",
)

_MULTIMODAL_CHECKPOINT_KEYS = (
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.mlp.up_proj.weight",
    "model.language_model.norm.weight",
    "model.vision_tower.encoder.weight",
)


def _write_checkpoint(folder: str, keys: tuple[str, ...], *, with_index: bool) -> None:
    # Sharded checkpoints come with an index; single-file ones usually don't.
    filename = "model-00001-of-00001.safetensors" if with_index else "model.safetensors"
    save_file({k: torch.zeros(2) for k in keys}, os.path.join(folder, filename))
    if with_index:
        with open(os.path.join(folder, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {k: filename for k in keys}}, f)


def _make_adapter(folder: str) -> HFTransformerStateDictAdapter:
    return HFTransformerStateDictAdapter(
        SimpleNamespace(tie_word_embeddings=True), folder  # pyrefly: ignore
    )


class TestHFTransformerStateDictAdapter(unittest.TestCase):
    def _assert_round_trip(
        self,
        model_keys: tuple[str, ...],
        checkpoint_keys: tuple[str, ...],
        expected_hf_keys: set[str],
        *,
        with_index: bool = False,
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _write_checkpoint(folder, checkpoint_keys, with_index=with_index)
            adapter = _make_adapter(folder)
            state_dict = {k: torch.zeros(2) for k in model_keys}

            hf_state_dict = adapter.to_hf(state_dict)
            self.assertEqual(set(hf_state_dict), expected_hf_keys)
            self.assertEqual(set(adapter.from_hf(hf_state_dict)), set(model_keys))

    def test_text_model_from_text_checkpoint(self):
        self._assert_round_trip(
            _TEXT_MODEL_KEYS, _TEXT_CHECKPOINT_KEYS, set(_TEXT_CHECKPOINT_KEYS)
        )

    def test_text_model_from_multimodal_checkpoint(self):
        # The decoder is read from the checkpoint's language model; the vision
        # tower is not requested.
        expected = {k for k in _MULTIMODAL_CHECKPOINT_KEYS if "vision_tower" not in k}
        for with_index in (False, True):
            with self.subTest(with_index=with_index):
                self._assert_round_trip(
                    _TEXT_MODEL_KEYS,
                    _MULTIMODAL_CHECKPOINT_KEYS,
                    expected,
                    with_index=with_index,
                )

    def test_multimodal_model_from_multimodal_checkpoint(self):
        # Keys already match, so nothing is renamed and the tower is loaded.
        self._assert_round_trip(
            _MULTIMODAL_MODEL_KEYS,
            _MULTIMODAL_CHECKPOINT_KEYS,
            set(_MULTIMODAL_CHECKPOINT_KEYS),
        )


if __name__ == "__main__":
    unittest.main()
