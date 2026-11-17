"""Dense Qwen3 recipe selecting the experiment without changing shared defaults."""

from dataclasses import replace
from functools import partial

from torch import nn

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer, default_adamw
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    apply_simple_fsdp,
)
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.tpu.experimental_flex_shard.data import (
    FixedTokenDataLoader,
    IntegerTokenizer,
)
from torchtitan.experiments.tpu.experimental_flex_shard.trainer import (
    TraceableCompileConfig,
    TraceableFlexShardTrainer,
)
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig
from torchtitan.models.qwen3 import model_registry


def parallelize_simple_fsdp(model, *, parallel_dims, training, **kwargs):
    """Unchanged SimpleFSDP control for the same dense-only recipe."""
    annotate_module_fqns(model)
    return apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)


def qwen3_traceable_flex_shard() -> TraceableFlexShardTrainer.Config:
    """Ten-step actual-Qwen tracing recipe on synthetic single documents."""
    model_spec = model_registry(
        "testmodel", seq_len=16, attn_backend="sdpa", fuse_qkv=False
    )
    # The stock graph-model subclass only disables parameter materialization
    # during init_states; it does not replace the Qwen architecture or forward.
    import dataclasses

    model_spec = replace(
        model_spec,
        model=GraphTrainerQwen3Model.Config(
            **{
                field.name: getattr(model_spec.model, field.name)
                for field in dataclasses.fields(model_spec.model)
            }
        ),
        parallelize_fn=parallelize_simple_fsdp,
    )
    model_spec.model.enable_weight_tying = False
    model_spec.model.tok_embeddings.param_init = {
        "weight": partial(nn.init.normal_, std=0.02)
    }
    config = TraceableFlexShardTrainer.Config(
        model_spec=model_spec,
        use_traceable_flex_shard=True,
        training=TrainingConfig(
            steps=10,
            max_context_length=16,
            num_tokens_per_microbatch_per_dp_rank=16,
            dtype="float32",
            mixed_precision_param="float32",
            mixed_precision_reduce="float32",
            max_norm=1.0,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            data_parallel_replicate_degree=1,
            tensor_parallel_degree=1,
            expert_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
            spmd_backend="partial_dtensor",
            enable_sequence_parallel=False,
        ),
        compile=TraceableCompileConfig(
            enable=True,
            mode="aot_fx_trace",
            components=["model"],
            enable_passes=True,
            memory_policy="none",
            optimizer_compile="eager",
        ),
        debug=DebugConfig(seed=42, deterministic=True),
        loss=CrossEntropyLoss.Config(),
        tokenizer=IntegerTokenizer.Config(),
        dataloader=FixedTokenDataLoader.Config(),
        optimizer=default_adamw(lr=8e-4, implementation="foreach"),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0,
            decay_ratio=0.0,
            min_lr_factor=1.0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        tpu_config=TPUConfig(bld_block_layout=False, enable_amp=False),
    )
    config.checkpoint.enable = False
    config.validator.enable = False
    return config
