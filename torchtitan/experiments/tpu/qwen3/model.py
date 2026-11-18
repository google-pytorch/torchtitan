# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass

import torch.nn as nn

from torchtitan.experiments.tpu.profiling.mfu_flops import (
    get_maxtext_decoder_nparams_and_flops,
)
from torchtitan.models.qwen3.model import Qwen3Model


@dataclass(kw_only=True, slots=True)
class MaxTextQwen3Config(Qwen3Model.Config):
    """Qwen3 config that builds the upstream model but uses MaxText FLOPs.

    TPU Qwen3 configs report upstream torchtitan FLOPs by default. Only the
    ``*maxtext*`` recipes wrap their model config in this class, so their FLOPs
    and MFU are comparable with MaxText's. The TPU GraphTrainer config
    conversion carries it over (``graph_trainer.qwen3.with_maxtext_flops``).
    """

    @classmethod
    def from_qwen3(cls, config: Qwen3Model.Config) -> "MaxTextQwen3Config":
        if isinstance(config, cls):
            return config
        return cls(
            **{
                field.name: getattr(config, field.name)
                for field in dataclasses.fields(Qwen3Model.Config)
                if field.init
            }
        )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
        return get_maxtext_decoder_nparams_and_flops(self, model, seq_len)
