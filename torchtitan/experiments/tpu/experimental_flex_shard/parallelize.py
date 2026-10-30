"""Setup-time replacement of SimpleFSDP with initialized FlexShard storage."""

import types
from functools import wraps

import torch
from torch import nn

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.tpu.experimental_flex_shard.dependency import load_executor
from torchtitan.tools.logging import logger


def install_initialized_model(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
) -> nn.Module:
    """Shard initialized local parameters before the optimizer is built."""
    executor = load_executor()
    mesh = parallel_dims.get_mesh("fsdp")
    if mesh.ndim != 1:
        raise ValueError("Traceable FlexShard requires a one-dimensional FSDP mesh")
    executor.data_parallel(
        model,
        mesh,
        mode="fully_shard",
        mp_policy=executor.MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        ),
    )
    logger.info("Installed traceable FlexShard on %d FSDP ranks", mesh.size())
    return model


def defer_installation(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
) -> nn.Module:
    """Install after Trainer's to_empty/init_weights, before optimizer setup.

    SimpleFSDP supports meta parameters; the portable storage executor does not.
    The root init hook is setup-only. Forward uses pure parameter properties,
    not parameter-swapping hooks or a deferred first-forward initialization.
    """
    original_init = model.init_states

    @wraps(original_init)
    def initialize_and_shard(self, *, buffer_device=None):
        if hasattr(self, "_graph_flex_shard_buckets"):
            raise ValueError("Do not reinitialize a model after FlexShard installation")
        original_init(buffer_device=buffer_device)
        install_initialized_model(self, parallel_dims=parallel_dims, training=training)

    model.init_states = types.MethodType(initialize_and_shard, model)
    return model


def clip_grad_norm(model: nn.Module, max_norm: float) -> torch.Tensor:
    """Reduce shard norms globally and count replicated parameters once."""
    buckets = model._graph_flex_shard_buckets
    assert buckets
    from torch.distributed.tensor import DTensor, Replicate, Shard

    norm_gradients = []
    for bucket in buckets:
        for info, owners in zip(bucket.infos, bucket.owners, strict=True):
            parameter = owners[0].module._parameters[owners[0].param_name]
            if parameter.grad is None:
                continue
            placement = Replicate() if bucket.replicated else Shard(info.placement.dim)
            norm_gradients.append(
                DTensor.from_local(
                    parameter.grad, bucket.storage._mesh, (placement,), run_check=False
                )
            )
    # Norm-only DTensors reuse the exact stock _NormPartial reduction order.
    # Parameters, forward collectives, and graph-produced gradients stay local.
    if not norm_gradients:
        return torch.zeros((), device=next(model.parameters()).device)
    norm = torch.nn.utils.get_total_norm(norm_gradients, 2).full_tensor()
    torch.nn.utils.clip_grads_with_norm_(list(model.parameters()), max_norm, norm)
    return norm
