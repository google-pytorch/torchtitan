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
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.tpu.graph_trainer.simple_fsdp import (
    install_tpu_replicate_computation,
)
from torchtitan.experiments.tpu.profiling.mfu_flops import (
    get_maxtext_decoder_nparams_and_flops,
)
from torchtitan.experiments.tpu.qwen3_sharding import (
    configure_qwen_simple_fsdp_shard_dims,
)
from torchtitan.protocols.model import BaseModel


graph_trainer_parallelize_qwen3 = GraphTrainerQwen3Model.parallelize


class TpuGraphTrainerQwen3Model(GraphTrainerQwen3Model):
    """GraphTrainer Qwen3 with TPU uneven-FSDP policy and replicate_computation."""

    @dataclass(kw_only=True, slots=True)
    class Config(GraphTrainerQwen3Model.Config):
        pass

    def parallelize(
        self,
        *,
        training: TrainingConfig,
        parallel_dims: ParallelDims | None = None,
        parallelism_context: ParallelDims | None = None,
        **kwargs,
    ) -> nn.Module:
        if parallel_dims is None:
            parallel_dims = parallelism_context
        assert parallel_dims is not None
        return parallelize_qwen3(
            self,
            training=training,
            parallel_dims=parallel_dims,
            **kwargs,
        )


class MaxTextTpuGraphTrainerQwen3Model(TpuGraphTrainerQwen3Model):
    """GraphTrainer Qwen3 that reports MaxText FLOPs.

    The GraphTrainer counterpart of ``MaxTextQwen3Config``. The TPU GraphTrainer
    config conversion switches to it when the recipe's model config is a
    ``MaxTextQwen3Config``; see ``with_maxtext_flops``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TpuGraphTrainerQwen3Model.Config):
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
    parallel_dims: ParallelDims | None = None,
    parallelism_context: ParallelDims | None = None,
    **kwargs,
) -> nn.Module:
    """Apply the TPU uneven-FSDP policy before setup."""
    if parallel_dims is None:
        parallel_dims = parallelism_context
    assert parallel_dims is not None
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


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
) -> TpuGraphTrainerQwen3Model.Config:
    base = graph_trainer_model_registry(
        flavor,
        seq_len=seq_len,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
    )
    return TpuGraphTrainerQwen3Model.Config(
        **{
            field.name: getattr(base, field.name)
            for field in dataclasses.fields(base)
        }
    )


model_registry._model_config_cls = TpuGraphTrainerQwen3Model.Config  # type: ignore[attr-defined]


def to_graph_trainer_model_config(
    model_config: BaseModel.Config,
) -> TpuGraphTrainerQwen3Model.Config:
    """Convert an already-specialized Qwen model config to its TPU graph counterpart.

    Recipes use this after applying TPU converters to the regular model config.
    It mirrors upstream ``to_graph_trainer_config``: preserve that configured
    model and the recipe-owned hooks, but replace its model config type with the
    TPU GraphTrainer version.
    """
    return TpuGraphTrainerQwen3Model.Config(
        **{
            field.name: getattr(model_config, field.name)
            for field in dataclasses.fields(model_config)
        }
    )


# Backwards-compatible alias for callers during transition
to_graph_trainer_model_spec = to_graph_trainer_model_config
