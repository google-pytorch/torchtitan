"""Experiment-owned trainer; the shared SimpleFSDP path stays unchanged."""

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

from torchtitan.experiments.graph_trainer.common_utils import annotate_module_fqns
from torchtitan.experiments.tpu.experimental_flex_shard.parallelize import (
    clip_grad_norm,
    defer_installation,
)
from torchtitan.experiments.tpu.graph_trainer.config import TpuGraphTrainerCompileConfig
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer


@dataclass(kw_only=True, slots=True)
class TraceableCompileConfig(TpuGraphTrainerCompileConfig):
    # "none" keeps the joint graph unchanged before terminal TPU compilation.
    memory_policy: Literal["none", "default", "full"] = "none"


def validate_config(config):
    """Reject unsupported training semantics instead of silently ignoring them."""
    parallelism = config.parallelism
    degrees = (
        parallelism.data_parallel_replicate_degree,
        parallelism.tensor_parallel_degree,
        parallelism.context_parallel_degree,
        parallelism.pipeline_parallel_degree,
        parallelism.expert_parallel_degree,
    )
    if any(degree != 1 for degree in degrees):
        raise ValueError(
            "Traceable FlexShard supports full-world 1D FSDP only; no HSDP/TP/CP/PP/EP"
        )
    if parallelism.data_parallel_shard_degree != -1:
        raise ValueError("Set data_parallel_shard_degree=-1 for full-world FSDP")
    if (
        parallelism.spmd_backend != "partial_dtensor"
        or parallelism.enable_sequence_parallel
    ):
        raise ValueError("Use partial_dtensor without sequence parallelism")
    if config.training.enable_cpu_offload:
        raise ValueError("Traceable FlexShard does not support parameter offload")
    if config.training.num_tokens_per_train_step != -1:
        raise ValueError(
            "Use one microbatch per train step; gradient accumulation is not supported"
        )
    if config.checkpoint.enable or config.validator.enable:
        raise ValueError(
            "Distributed checkpoint and validator integration are not supported"
        )
    compile_config = config.compile
    if (
        compile_config.mode != "aot_fx_trace"
        or compile_config.optimizer_compile != "eager"
    ):
        raise ValueError("Use aot_fx_trace and eager optimizer execution")
    if (
        compile_config.use_scan_layers
        or compile_config.graph_split.enabled
        or compile_config.compile_grad_norm
        or compile_config.precompile_artifact_dir
        or compile_config.fsdp_barrier_policy != "none"
        or compile_config.fsdp_allreduce_slice_parameter_patterns
        or compile_config.enable_inplace_graph_gradient_accumulation
    ):
        raise ValueError(
            "Scan, graph splitting, DTensor norm, precompile, "
            "and specialized FSDP passes are not supported"
        )
    if compile_config.memory_policy not in ("none", "default", "full"):
        raise ValueError("Use no-remat, default, or full-recompute policy; no offload")
    if config.tpu_config.bld_block_layout or config.tpu_config.enable_amp:
        raise ValueError(
            "Use the unchanged model layout and explicit mixed-precision "
            "policy, not AMP"
        )
    if (
        config.splash_attention_kernel.use_splash_attention_kernel
        or config.splash_attention_kernel.use_tokamax_splash_attention_kernel
        or config.loss_kernel.use_loss_kernel
        or config.lora.use_lora
        or config.override.imports
        or compile_config.enable_autoparallel
        or compile_config.ep_overlap.enabled
        or parallelism.fsdp_reshard_after_forward != "default"
    ):
        raise ValueError(
            "Kernel conversions, LoRA, overrides, autoparallel, "
            "and explicit reshard policies are not supported"
        )
    assert config.model_spec is not None
    if config.model_spec.name != "qwen3" or any(
        layer.moe is not None for layer in config.model_spec.model.layers
    ):
        raise ValueError("The current experiment validates dense Qwen3 only")


class TraceableFlexShardTrainer(TpuGraphTrainer):
    """Use portable FlexShard storage with TorchTitan's existing FX trainer.

    Opting out preserves the TPU GraphTrainer's existing parallelize function.
    Opting in changes setup and shard-aware clipping, not the model or tracer.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TpuGraphTrainer.Config):
        use_traceable_flex_shard: bool = False
        compile: TraceableCompileConfig = field(default_factory=TraceableCompileConfig)

    def __init__(self, config: Config):
        if config.use_traceable_flex_shard:
            validate_config(config)
            assert config.model_spec is not None

            def parallelize(model, *, parallel_dims, training, ac_config, **kwargs):
                # GraphTrainer's FX memory policy is the sole checkpoint owner.
                # Eager ActivationCheckpointing config is deliberately not applied.
                from torchtitan.tools.logging import logger

                logger.warning(
                    "Eager activation_checkpoint policy %s is not applied; "
                    "use compile.memory_policy for this GraphTrainer experiment",
                    type(ac_config).__qualname__,
                )
                annotate_module_fqns(model)
                return defer_installation(
                    model, parallel_dims=parallel_dims, training=training
                )

            config.model_spec = dataclasses.replace(
                config.model_spec, parallelize_fn=parallelize
            )
        super().__init__(config)

    def _clip_grad_norm(self):
        if not self.config.use_traceable_flex_shard:
            return super()._clip_grad_norm()
        assert len(self.model_parts) == 1
        return clip_grad_norm(self.model_parts[0], self.config.training.max_norm)
