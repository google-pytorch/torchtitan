# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass

import torch.nn as nn

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.ep_eager_chunk import (
    maybe_apply_ep_overlap_eager_chunking,
)
from torchtitan.experiments.graph_trainer.qwen3 import (
    model_registry as graph_trainer_model_registry,
)
from torchtitan.experiments.graph_trainer.qwen3.parallelize import (
    annotate_qwen3,
    parallelize_qwen3 as graph_trainer_parallelize_qwen3,
)
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.tpu.graph_trainer.simple_fsdp import (
    install_tpu_replicate_computation,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_fsdp import apply_tpu_fsdp
from torchtitan.models.common.moe import configure_moe_aux_loss_reduction
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


def _parallelize_qwen3_explicit_fsdp(
    model: GraphTrainerQwen3Model,
    *,
    training: TrainingConfig,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> nn.Module:
    """graph_trainer's ``parallelize_qwen3`` with explicit TPU FSDP in place of SimpleFSDP.

    Same steps in the same order as the upstream function, which applies
    SimpleFSDP unconditionally and so cannot be reused with another sharding.
    """
    if (
        training.num_tokens_per_microbatch_per_dp_rank % parallel_dims.seq_len_divisor
        != 0
    ):
        raise ValueError(
            f"Token count {training.num_tokens_per_microbatch_per_dp_rank} must be "
            f"divisible by {parallel_dims.seq_len_divisor} (TP degree x 2 CP degree)"
        )
    annotate_qwen3(model)
    if (
        parallelism.spmd_backend == "spmd_types"
        or parallel_dims.tp_enabled
        or parallel_dims.ep_enabled
    ):
        model.parallelize(parallel_dims)
    configure_moe_aux_loss_reduction(model, parallel_dims)
    model = apply_tpu_fsdp(model, parallel_dims=parallel_dims, training=training)
    maybe_apply_ep_overlap_eager_chunking(model, compile_config)
    return apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )


def parallelize_qwen3(
    model: GraphTrainerQwen3Model,
    *,
    training: TrainingConfig,
    parallel_dims: ParallelDims,
    fsdp_implementation: str = "simple_fsdp",
    **kwargs,
) -> nn.Module:
    """Apply the TPU uneven-FSDP policy before setup.

    ``fsdp_implementation`` is ``TPUConfig.fsdp_implementation``: ``simple_fsdp``
    delegates to graph_trainer's parallelize, ``explicit`` shards with
    ``tpu_fsdp`` instead.
    """
    configure_qwen_simple_fsdp_shard_dims(model, parallel_dims)
    if fsdp_implementation == "explicit":
        model = _parallelize_qwen3_explicit_fsdp(
            model, training=training, parallel_dims=parallel_dims, **kwargs
        )
    elif fsdp_implementation == "simple_fsdp":
        model = graph_trainer_parallelize_qwen3(
            model,
            training=training,
            parallel_dims=parallel_dims,
            **kwargs,
        )
    else:
        raise ValueError(f"unknown fsdp_implementation {fsdp_implementation!r}")
    # Creates process groups, so every rank runs it here, right after sharding.
    for holder in model.modules():
        if hasattr(holder, "build_split_meshes"):
            holder.build_split_meshes()
    if fsdp_implementation == "simple_fsdp":
        install_tpu_replicate_computation(model)
    return model


def model_registry(flavor: str) -> ModelSpec:
    return dataclasses.replace(
        graph_trainer_model_registry(flavor),
        parallelize_fn=parallelize_qwen3,
    )
