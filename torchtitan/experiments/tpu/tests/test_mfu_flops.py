# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu.examples.gemma4 import gemma4_configs
from torchtitan.experiments.tpu.mfu_flops import (
    get_maxtext_decoder_nparams_and_flops,
)
from torchtitan.experiments.tpu.qwen3 import (
    MaxTextQwen3Config,
    config_registry as tpu_qwen3_config_registry,
    model_registry as tpu_qwen3_model_registry,
    qwen3_configs,
)


def _build(model_config):
  with torch.device("meta"):
    return model_config.build()


class MaxTextQwen3FlopsTest(absltest.TestCase):

  def test_qwen3_8b_matches_maxtext(self) -> None:
    model_config = qwen3_configs["8B"]
    self.assertEqual(
        get_maxtext_decoder_nparams_and_flops(
            model_config, _build(model_config), seq_len=2048
        )[1],
        47_220_523_008,
    )

  def test_qwen3_30b_a3b_matches_maxtext(self) -> None:
    model_config = qwen3_configs["30B-A3B"]
    self.assertEqual(
        get_maxtext_decoder_nparams_and_flops(
            model_config, _build(model_config), seq_len=2048
        )[1],
        20_665_860_096,
    )


class FlopsConventionTest(absltest.TestCase):

  def test_qwen3_model_config_reports_torchtitan_flops(self) -> None:
    model_config = tpu_qwen3_model_registry("30B-A3B").model
    self.assertEqual(
        model_config.get_nparams_and_flops(
            _build(model_config), seq_len=2048
        )[1],
        23_083_044_864,
    )

  def test_maxtext_qwen3_config_reports_maxtext_flops(self) -> None:
    model_config = MaxTextQwen3Config.from_qwen3(qwen3_configs["30B-A3B"])
    self.assertEqual(
        model_config.get_nparams_and_flops(
            _build(model_config), seq_len=2048
        )[1],
        20_665_860_096,
    )

  def test_only_maxtext_recipes_report_maxtext_flops(self) -> None:
    """MaxText FLOPs belong to the ``*maxtext*`` recipes and no others."""
    recipes = {
        name: fn
        for name, fn in inspect.getmembers(
            tpu_qwen3_config_registry, inspect.isfunction
        )
        if fn.__module__ == tpu_qwen3_config_registry.__name__
        and not name.startswith("_")
        and not inspect.signature(fn).parameters
    }
    self.assertTrue(any("maxtext" in name for name in recipes))
    for name, recipe in recipes.items():
      with self.subTest(recipe=name):
        self.assertEqual(
            isinstance(recipe().model_spec.model, MaxTextQwen3Config),
            "maxtext" in name,
        )


class MaxTextGemma4FlopsTest(absltest.TestCase):

  def test_gemma4_31b_matches_maxtext(self) -> None:
    model_config = gemma4_configs["31B"]
    nparams, flops = model_config.get_nparams_and_flops(
        _build(model_config), seq_len=2048
    )
    self.assertGreater(nparams, 30_000_000_000)
    self.assertEqual(flops, 188_705_931_264)

  def test_gemma4_26b_a4b_matches_maxtext(self) -> None:
    model_config = gemma4_configs["26B-A4B"]
    nparams, flops = model_config.get_nparams_and_flops(
        _build(model_config), seq_len=2048
    )
    self.assertGreater(nparams, 25_000_000_000)
    self.assertEqual(flops, 24_063_442_944)


if __name__ == "__main__":
  absltest.main()
