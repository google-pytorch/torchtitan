# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass

import torch.nn as nn

from torchtitan.config import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.qwen3 import (
    model_registry as graph_trainer_model_registry,
)
from torchtitan.experiments.graph_trainer.qwen3.parallelize import (
    parallelize_qwen3 as graph_trainer_parallelize_qwen3,
)
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.tpu.graph_trainer.simple_fsdp import (
    install_tpu_replicate_computation,
)
from torchtitan.experiments.tpu.mfu_flops import (
    get_maxtext_decoder_nparams_and_flops,
)
from torchtitan.experiments.tpu.qwen3_sharding import (
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


def parallelize_qwen3(
    model: GraphTrainerQwen3Model,
    *,
    training: TrainingConfig,
    parallel_dims: ParallelDims,
    **kwargs,
) -> nn.Module:
    """Apply the TPU uneven-FSDP policy before setup."""
    configure_qwen_simple_fsdp_shard_dims(model, parallel_dims)
    model = graph_trainer_parallelize_qwen3(
        model,
        training=training,
        parallel_dims=parallel_dims,
        **kwargs,
    )
    # Creates process groups, so every rank runs it here, right after sharding.
    for holder in model.modules():
        if hasattr(holder, "build_split_meshes"):
            holder.build_split_meshes()
    install_tpu_replicate_computation(model)
    return model


def model_registry(flavor: str) -> ModelSpec:
    return dataclasses.replace(
        graph_trainer_model_registry(flavor),
        parallelize_fn=parallelize_qwen3,
    )


def to_graph_trainer_model_spec(model_spec: ModelSpec) -> ModelSpec:
    """Convert an already-specialized Qwen spec to its TPU graph counterpart.

    Recipes use this after applying TPU converters to the regular model config.
    It mirrors upstream ``to_graph_trainer_config``: preserve that configured
    model and the recipe-owned hooks, but replace its model type and execution
    callables with the GraphTrainer versions.
    """
    graph_spec = model_registry(model_spec.flavor)
    graph_model_cls = type(graph_spec.model)
    graph_model = graph_model_cls(
        **{
            field.name: getattr(model_spec.model, field.name)
            for field in dataclasses.fields(model_spec.model)
        }
    )
    return dataclasses.replace(
        model_spec,
        model=graph_model,
        parallelize_fn=graph_spec.parallelize_fn,
        pipelining_fn=graph_spec.pipelining_fn,
    )
