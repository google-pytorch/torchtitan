# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass

import torch.nn as nn

from torchtitan.experiments.graph_trainer.qwen3 import (
    model_registry as graph_trainer_model_registry,
)
from torchtitan.experiments.graph_trainer.qwen3.parallelize import annotate_qwen3
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.tpu.graph_trainer.parallelize import (
    parallelize_tpu_graph_trainer,
)
from torchtitan.experiments.tpu.mfu_flops import (
    get_maxtext_decoder_nparams_and_flops,
)
from torchtitan.experiments.tpu.qwen3.infra.parallelize import (
    configure_qwen_simple_fsdp_shard_dims,
)
from torchtitan.protocols.model_spec import ModelSpec


class MaxTextTpuGraphTrainerQwen3Model(GraphTrainerQwen3Model):
    """GraphTrainer Qwen3 that reports MaxText FLOPs.

    The GraphTrainer counterpart of ``MaxTextQwen3Config``. The TPU GraphTrainer
    config conversion switches to it when the recipe's model config is a
    ``MaxTextQwen3Config``; see ``with_maxtext_flops``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GraphTrainerQwen3Model.Config):
        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_maxtext_decoder_nparams_and_flops(self, model, seq_len)


def with_maxtext_flops(
    config: GraphTrainerQwen3Model.Config,
) -> MaxTextTpuGraphTrainerQwen3Model.Config:
    """Return ``config`` unchanged except that it reports MaxText FLOPs."""
    return MaxTextTpuGraphTrainerQwen3Model.Config(
        **{
            field.name: getattr(config, field.name)
            for field in dataclasses.fields(config)
            if field.init
        }
    )


def parallelize_qwen3(model: GraphTrainerQwen3Model, **kwargs) -> nn.Module:
    """Qwen3 on the TPU graph trainer: Qwen's uneven-expert shard axes, then the shared setup."""
    return parallelize_tpu_graph_trainer(
        model,
        annotate=annotate_qwen3,
        configure_shard_dims=configure_qwen_simple_fsdp_shard_dims,
        **kwargs,
    )


def model_registry(flavor: str) -> ModelSpec:
    return dataclasses.replace(
        graph_trainer_model_registry(flavor),
        parallelize_fn=parallelize_qwen3,
    )
