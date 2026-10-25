"""The TPU graph trainer's one parallelize function.

Every model on the TPU graph trainer is sharded the same way: explicit TPU
FSDP over the data-parallel mesh (:mod:`tpu_fsdp`), plain local shards and a
hand-written gather / reduce-scatter pair, with DTensor kept only as a view
for initialization and checkpoints. What differs between models is how their
FX nodes are annotated and which parameters need a non-default shard axis, so
those two come in as callables and the rest is shared.

Upstream graph_trainer's per-model ``parallelize_*`` functions apply
SimpleFSDP unconditionally and so cannot be reused; this keeps their step
order (annotate, model-parallel ``Module.parallelize``, MoE aux-loss meshes,
FSDP, EP eager chunking, compile) and adds the TPU shard-axis planning in
front, which has to see the stacks before anything shards them.
"""

from collections.abc import Callable

from torch import nn

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.ep_eager_chunk import (
    maybe_apply_ep_overlap_eager_chunking,
)
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import (
    configure_simple_fsdp_shard_dims,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_fsdp import apply_tpu_fsdp
from torchtitan.models.common.moe import configure_moe_aux_loss_reduction


__all__ = ["parallelize_tpu_graph_trainer"]


def parallelize_tpu_graph_trainer(
    model: nn.Module,
    *,
    annotate: Callable[[nn.Module], None],
    configure_shard_dims: Callable[
        [nn.Module, ParallelDims], object
    ] = configure_simple_fsdp_shard_dims,
    training: TrainingConfig,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> nn.Module:
    """Shard, annotate and compile ``model`` for the TPU graph trainer.

    Args:
      model: the graph-trainer model, on meta.
      annotate: the model's FX annotation (``annotate_qwen3``, ...).
      configure_shard_dims: picks per-parameter FSDP shard axes before sharding
        (``_fsdp_shard_dim`` hooks, stack flattening). The default handles layer
        stacks and vocab-sized weights for any decoder; a model with more
        uneven parameters passes its own, which must include the default's work.
      training, parallel_dims, parallelism, compile_config, ac_config,
        dump_folder: the trainer's ``parallelize_fn`` keywords.
    """
    del ac_config  # The graph trainer's compile owns activation checkpointing.
    if (
        training.num_tokens_per_microbatch_per_dp_rank % parallel_dims.seq_len_divisor
        != 0
    ):
        raise ValueError(
            f"Token count {training.num_tokens_per_microbatch_per_dp_rank} must be "
            f"divisible by {parallel_dims.seq_len_divisor} (TP degree x 2 CP degree)"
        )
    configure_shard_dims(model, parallel_dims)
    annotate(model)
    if (
        parallelism.spmd_backend == "spmd_types"
        or parallel_dims.tp_enabled
        or parallel_dims.ep_enabled
    ):
        model.parallelize(parallel_dims)
    configure_moe_aux_loss_reduction(model, parallel_dims)
    model = apply_tpu_fsdp(model, parallel_dims=parallel_dims, training=training)
    # Creates process groups, so every rank runs it here, right after sharding.
    for holder in model.modules():
        if hasattr(holder, "build_split_meshes"):
            holder.build_split_meshes()
    maybe_apply_ep_overlap_eager_chunking(model, compile_config)
    return apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )
