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

"""TPU flux configs, selected with `--module=torchtitan.experiments.tpu.flux`."""

import dataclasses

from torchtitan.models.flux.config_registry import flux_debugmodel
from torchtitan.models.flux.config_registry import model_registry
from torchtitan.models.flux.trainer import FluxTrainer

_TEST_ASSETS = "tests/assets/"


def flux_testmodel() -> FluxTrainer.Config:
  """Tiny flux config that trains fully offline, for tests.

  Like upstream's `flux_debugmodel_test`, it uses the local test tokenizer and
  randomly initialized encoders. Unlike it, it pairs the `flux-test` model with
  the `t5-micro` / `clip-micro` encoders (64-dim outputs) instead of `flux-debug`
  with full-size T5-XXL / CLIP-L configs, which do not fit test memory limits.

  Returns:
    The trainer config.
  """
  config = flux_debugmodel()
  config.model_spec = model_registry("flux-test")
  config.hf_assets_path = f"{_TEST_ASSETS}tokenizer"
  config.tokenizer.test_mode = True
  config.tokenizer.t5_tokenizer_path = f"{_TEST_ASSETS}tokenizer"
  config.tokenizer.clip_tokenizer_path = f"{_TEST_ASSETS}tokenizer"
  config.encoder.random_init = True
  config.encoder.t5_encoder = f"{_TEST_ASSETS}flux_test_encoders/t5-micro"
  config.encoder.clip_encoder = f"{_TEST_ASSETS}flux_test_encoders/clip-micro"
  # Point both datasets at the local cc12m_test shards explicitly, so the
  # config does not depend on the upstream default (repo-relative) paths.
  config.dataloader.dataset_path = f"{_TEST_ASSETS}cc12m_test"
  validation = config.validator.dataloader.dataset
  config.validator.dataloader.dataset = dataclasses.replace(
      validation,
      dataset=dataclasses.replace(
          validation.dataset,
          source=dataclasses.replace(
              validation.dataset.source, path=f"{_TEST_ASSETS}cc12m_test"
          ),
      ),
  )
  return config
