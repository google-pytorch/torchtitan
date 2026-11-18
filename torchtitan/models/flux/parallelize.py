# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    DataParallelMeshDims,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    enable_fsdp_symm_mem,
    resolve_fsdp_mesh,
)
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from torchtitan.models.flux.model.hf_embedder import FluxEmbedder
from torchtitan.models.flux.model.model import FluxModel
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel as simple_fsdp_data_parallel,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    MixedPrecisionPolicy as SimpleFSDPMixedPrecisionPolicy,
)
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.distributed.spmd_utils import apply_spmd_safe
from torchtitan.tools.logging import logger


def parallelize_flux(
    model: FluxModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    if (
        isinstance(training, tpu_job_config.TPUTrainerConfig)
        and training.splash_attention_kernel.use_splash_attention_kernel
    ):
        workarounds.use_splash_attention_patch(model)

    if ac_config is not None:
        apply_ac(model, ac_config)

    model.parallelize(parallel_dims)
    annotate_replicated_parameters(model, parallel_dims)

    if compile_config.enable and "model" in compile_config.components:
        apply_compile(model, compile_config)

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)

    use_simple_fsdp = getattr(parallelism, "use_simple_fsdp", False)
    if use_simple_fsdp:
        dp_mode = (
            "hybrid_shard" if parallel_dims.dp_replicate_enabled else "fully_shard"
        )
        mp_policy = SimpleFSDPMixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )
        model = simple_fsdp_data_parallel(
            model,
            dp_mesh,
            mode=dp_mode,
            mp_policy=mp_policy,
        )
        logger.info(f"Applied SimpleFSDP ({dp_mode}) to the model")
    else:
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            cpu_offload=training.enable_cpu_offload,
            enable_symm_mem=parallelism.enable_fsdp_symm_mem,
            dp_mesh_dims=dp_mesh_dims,
        )

    logger.info("Applied fully_shard to the model")

    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    enable_symm_mem: bool = False,
    dp_mesh_dims: DataParallelMeshDims | None = None,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        cpu_offload (bool): Whether to offload model parameters to CPU. Defaults to False.
        enable_symm_mem (bool): Whether to enable symmetric-memory FSDP communication.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if dp_mesh_dims is not None:
        fsdp_config["dp_mesh_dims"] = dp_mesh_dims
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    linear_layers = [
        model.img_in,
        model.time_in,
        model.vector_in,
        model.txt_in,
    ]
    for layer in linear_layers:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(layer, **fsdp_config)

    # pyrefly: ignore [not-iterable]
    for block in model.double_blocks:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(
            block,
            **fsdp_config,
        )

    # pyrefly: ignore [not-iterable]
    for block in model.single_blocks:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(
            block,
            **fsdp_config,
        )

    # apply FSDP to last layer. Set reshard_after_forward=False for last layer to avoid gather right after reshard
    # pyrefly: ignore [no-matching-overload]
    fully_shard(model.final_layer, **fsdp_config, reshard_after_forward=False)

    # Wrap all the rest of model
    fully_shard(model, **fsdp_config)

    if enable_symm_mem:
        enable_fsdp_symm_mem(model)

    # Disable FSDP's automatic gradient division for all FSDP modules
    disable_fsdp_gradient_division(model)


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    """
    Apply torch.compile to each DoubleStreamBlock and SingleStreamBlock, which
    makes compilation efficient due to repeated structure.
    """
    # pyrefly: ignore [not-iterable]
    for block in model.double_blocks:
        # pyrefly: ignore [missing-attribute]
        apply_spmd_safe(block)
        block.compile(backend=compile_config.backend, fullgraph=True)

    # pyrefly: ignore [not-iterable]
    for block in model.single_blocks:
        # pyrefly: ignore [missing-attribute]
        apply_spmd_safe(block)
        block.compile(backend=compile_config.backend, fullgraph=True)

    logger.info(
        "Compiling each DoubleStreamBlock and SingleStreamBlock with torch.compile"
    )


def _is_splash_attention_fwd(func) -> bool:
    """True for the Pallas splash-attention forward ops.

    Covers all splash variants registered under the `pallas` namespace: splash_attn_fwd_<hash> (splash_attention.py),
    tokamax_splash_attn_fwd (tokamax_splash_attention.py) and sbd_splash_attn_fwd (splash_bias_dropout.py).
    """
    return getattr(func, "namespace", None) == "pallas" and "splash_attn_fwd" in getattr(
        func, "_opname", ""
    )


def _full_ac_save_splash_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
    """Full activation checkpointing, except the splash-attention forward is saved.

    Saving the splash forward's outputs (attention output and the residuals its backward needs) avoids re-running
    the kernel in backward. Since torch 2.12, vanilla AC tags every op for recompute at dispatch time, which
    recomputes the splash forward in backward and roughly doubles its cost; torch 2.11 effectively saved it.
    """
    if _is_splash_attention_fwd(func):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _checkpoint_block(block: nn.Module, ac_config) -> nn.Module:
    return ptd_checkpoint_wrapper(
        block,
        preserve_rng_state=ac_config.preserve_rng_state,
        context_fn=lambda: create_selective_checkpoint_contexts(_full_ac_save_splash_policy),
    )


def apply_ac(model: nn.Module, ac_config: ActivationCheckpointingConfig):
    """Apply full activation checkpointing to the Flux double/single blocks."""

    # pyrefly: ignore [missing-attribute]
    for layer_id, block in model.double_blocks.named_children():
        # TODO(b/514461185): Set preserve_rng_state=True once bug is fixed.
        block = _checkpoint_block(block, ac_config)
        # pyrefly: ignore [missing-attribute]
        model.double_blocks.register_module(layer_id, block)

    # pyrefly: ignore [missing-attribute]
    for layer_id, block in model.single_blocks.named_children():
        # TODO(b/514461185): Set preserve_rng_state=True once bug is fixed.
        block = _checkpoint_block(block, ac_config)
        # pyrefly: ignore [missing-attribute]
        model.single_blocks.register_module(layer_id, block)

    logger.info("Applied full activation checkpointing to the model")


def parallelize_encoders(
    t5_model: FluxEmbedder,
    clip_model: FluxEmbedder,
    parallel_dims: ParallelDims,
    *,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    enable_symm_mem: bool = False,
):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
    )

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    use_simple_fsdp = getattr(parallelism, "use_simple_fsdp", False)
    if use_simple_fsdp:
        dp_mode = (
            "hybrid_shard" if parallel_dims.dp_replicate_enabled else "fully_shard"
        )
        mp_policy = SimpleFSDPMixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )
        t5_model = simple_fsdp_data_parallel(
            t5_model,
            dp_mesh,
            mode=dp_mode,
            mp_policy=mp_policy,
        )
        logger.info(f"Applied SimpleFSDP ({dp_mode}) to the T5 encoder model")
    else:
        fsdp_config: dict[str, Any] = {
            "mesh": dp_mesh,
            "mp_policy": mp_policy,
        }
        if dp_mesh_dims is not None:
            fsdp_config["dp_mesh_dims"] = dp_mesh_dims
        if training.enable_cpu_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy()

        # NOTE: only apply FSDP to the T5 encoder, not the CLIP text encoder.
        # CLIP Text encoder has low computation / communication ratio, so it's not necessary to apply FSDP to it.
        hf_module = t5_model.hf_module
        assert isinstance(hf_module, nn.Module)
        annotate_replicated_parameters(hf_module, parallel_dims)
        # pyrefly: ignore [missing-attribute, not-iterable]
        for block in hf_module.encoder.block:
            fully_shard(block, **fsdp_config)
        fully_shard(hf_module, **fsdp_config)

        if enable_symm_mem:
            enable_fsdp_symm_mem(hf_module)

        # Disable FSDP's automatic gradient division for all FSDP modules
        disable_fsdp_gradient_division(hf_module)

        logger.info("Applied fully_shard to the T5 encoder model")

    logger.info("Applied fully_shard to the T5 encoder model")

    return t5_model, clip_model
