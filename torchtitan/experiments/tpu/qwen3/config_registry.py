# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import os
from typing import Literal, cast
from torch import nn
from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import (
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.config.transform import apply_transforms, ContextParallelTransform
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.cp_attention import tpu_cp_inner_attention, tpu_cp_load_balancer
from torchtitan.experiments.tpu.float8_converter import TpuFloat8LinearConverter
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.graph_trainer.config import (
    TpuGraphTrainerCompileConfig,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.graph_trainer.qwen3 import (
    model_registry as graph_trainer_model_registry,
    to_graph_trainer_model_spec,
    with_maxtext_flops,
)
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.qkv import TpuPackedQKVConverter
import torchtitan.experiments.tpu.qwen3 as tpu_qwen3
from torchtitan.experiments.tpu.qwen3 import model_registry
from torchtitan.experiments.tpu.qwen3.infra.parallelize import parallelize_qwen3 as tpu_parallelize_qwen3
from torchtitan.experiments.tpu.qwen3.model import MaxTextQwen3Config
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRoutedExperts,
    TpuRoutedExpertsConverter,
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    LossKernelConfig,
    SplashAttentionKernelConfig,
    TPUConfig,
    TpuTrainerCompileConfig,
    TPUTrainerConfig,
)
from torchtitan.experiments.tpu.datasets import c4_test_dataloader, TPUGrainDataLoader
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.config_utils import make_aux_loss_config
from torchtitan.models.common.moe import MoE
from torchtitan.models.qwen3.config_registry import qwen3_30b_a3b_global_aux
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.profiler import Profiler


# Upstream d771abb0 identifies an override by the import path of its decorated
# `@override` factory, so each target is `module.function`, not just `module`.
_FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"
)
_BF16_ROPE_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.bf16_rope"
    ".tpu_activation_dtype_cossin_rope"
)
_DISCARDED_TOKEN_COUNTER_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.discarded_token_counter"
    ".tpu_discarded_token_counter"
)


# Shared by both Qwen3-235B lines, the scan loop and the graph splitter.
_Q235_SHARED_FLAGS = tpu_flags.merge_flags(
    # Select TPU7x's throughput-oriented operating point for this compute-bound
    # recipe. This changes device frequency, not graph numerics.
    tpu_flags.TPU7X_THROUGHPUT_P_STATE,
    {
        # Raise the scoped VMEM budget to 128 MiB so the compiler can retain
        # more Q235 kernel intermediates in VMEM instead of spilling them.
        "xla_tpu_scoped_vmem_limit_kib": 131072,
        # Keep activation host copies off SparseCore, where they would contend
        # with the FSDP collectives. In the isolated A/B this moved MFU only
        # from 18.425% to 18.448% and did not itself create D2H overlap.
        "xla_tpu_enable_offloading_copy_to_sparsecore": False,
        # Prove the true copy/overwrite ordering across donated aliases. This
        # removed the defensive activation snapshot, moved all 93 measured D2H
        # waits to zero, and improved MFU from 18.528% to 19.490% in the A/B.
        "xla_tpu_copy_insertion_use_region_analysis": True,
        # Use the tested search budget for the large Q235 layer regions. This
        # value was evaluated together with region analysis, not in isolation.
        "xla_tpu_copy_insertion_use_region_analysis_limit": 1000,
    },
)

# What the graph-splitter line adds on top.
_Q235_COMPILED_OFFLOAD_FLAGS = tpu_flags.merge_flags(
    _Q235_SHARED_FLAGS,
    # Aggregate SparseCore collectives and enable layer-level latency hiding so
    # FSDP communication can be scheduled alongside independent layer compute.
    tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
    # Use both SparseCores for reduce-scatter in the by-layer FSDP schedule.
    tpu_flags.DUAL_SPARSECORE_REDUCE_SCATTER,
    {
        # Preserve eligible all-gathers as asynchronous all-gathers instead of
        # rewriting them to all-reduce-plus-slice. The native non-leading-axis
        # FSDP path relies on this to avoid the slower rewrite.
        "xla_tpu_prefer_async_allgather_to_allreduce": True,
    },
)

# What the scan-layers line adds on top.
_Q235_SCAN_FLAGS = tpu_flags.merge_flags(
    _Q235_SHARED_FLAGS,
    # Each layer's expert-gradient reduce-scatters land at the end of the scan
    # body with little compute left to hide them.
    tpu_flags.HIERARCHICAL_REDUCE_SCATTER,
    tpu_flags.SELECTIVE_ICI_RS_PIPELINING,
    tpu_flags.ICI_AG_PIPELINING,
    tpu_flags.SPARSECORE_ASYNC_ALL_GATHER,
    # As the splitter does: it halves the expert-weight all-gather wait.
    tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
    {
        # Raise the backward layer scheduler compute threshold so expert
        # gradient reduce-scatters overlap with the expert matmul (otherwise
        # the smaller SwiGLU backward fusion gets chosen for overlap).
        "xla_tpu_layer_scheduler_min_compute_latency_for_backward_pass": (
            25000000
        ),
        # SELECTIVE_ICI_RS_PIPELINING stops at 1024 KiB per shard, which keeps
        # the 24 MiB expert-gradient shards inside their iteration, waiting at
        # the end of the body. Pipeline them too, for about 3 GiB more HBM.
        "xla_tpu_ici_rs_pipelining_threshold_kib": 32768,
    },
)


def _use_maxtext_flops(config: TPUTrainerConfig) -> None:
  """Report MaxText FLOPs and MFU instead of upstream torchtitan's.

  Only the ``*maxtext*`` recipes call this, so their MFU is comparable with
  MaxText's published numbers. MaxText halves attention FLOPs for the causal
  mask; torchtitan does not discount causal sparsity.
  """
  assert config.model_spec is not None
  model = config.model_spec.model
  if isinstance(model, GraphTrainerQwen3Model.Config):
    config.model_spec.model = with_maxtext_flops(model)
  else:
    config.model_spec.model = MaxTextQwen3Config.from_qwen3(model)


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Select the fused loss and its pre-construction LM-head override together."""
  config.loss = PallasCrossEntropyLoss.Config()
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)


def _use_global_batch_aux_loss(config: TPUTrainerConfig) -> None:
  """Use Qwen's global auxiliary load-balancing loss for every MoE layer."""
  assert config.model_spec is not None
  for _, moe, _, _ in config.model_spec.model.traverse(MoE.Config):
    moe.load_balance_coeff = None
    moe.router.aux_loss = make_aux_loss_config(
        type="batch_wise", coeff=1e-3, top_k=moe.router.top_k
    )


def qwen3_debugmodel() -> TPUTrainerConfig:
  """Qwen3 debug model configuration for TPU."""

  model_spec = model_registry("debugmodel", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_qwen3

  model_config = cast(Qwen3Model.Config, model_spec.model)
  # Disable weight tying to mimic the configuration of larger Qwen3 models
  # (e.g. 8B, 14B, 32B). Note that this is different from the non-TPU
  # "debugmodel" in the config registry.
  model_config.enable_weight_tying = False
  model_config.tok_embeddings.param_init = {
      "weight": partial(nn.init.normal_, std=1.0)
  }
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * 2048,
          max_context_length=2048,
          steps=10,
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          pipeline_parallel_schedule="Interleaved1F1B"
      ),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(),
      validator=Validator.Config(
          freq=5,
          steps=10,
      ),
  )


def graph_trainer_qwen3_debugmodel() -> TpuGraphTrainer.Config:
  """Qwen3 debug model using TPU GraphTrainer."""
  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      qwen3_debugmodel(), graph_trainer_model_registry
  )
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
  )
  return config


def qwen3_moe_testmodel() -> TPUTrainerConfig:
  """Qwen3 MoE test model configuration for TPU."""
  cfg = qwen3_debugmodel()
  config = tpu_qwen3.qwen3_configs["testmodel_moe"]
  # Upstream d771abb0 made ModelSpec.max_context_length required.
  # `testmodel_moe` is built by `_testmodel_moe`, whose RoPE config pins
  # `TESTMODEL_MAX_CONTEXT_LENGTH`; `training.max_context_length` below is set
  # to the same value.
  cfg.model_spec = ModelSpec(
      name="qwen3_tpu",
      flavor="testmodel_moe",
      model=config,
      max_context_length=tpu_qwen3.TESTMODEL_MAX_CONTEXT_LENGTH,
      parallelize_fn=tpu_parallelize_qwen3,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=Qwen3StateDictAdapter,
  )
  cfg.model_spec.model.enable_weight_tying = False

  # Override with previous debug_model train_config values
  # Upstream d771abb0 replaced local_batch_size/global_batch_size/seq_len with
  # token budgets. Set the context length first; the microbatch budget is four
  # sequences at that new context.
  cfg.training.max_context_length = 128
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 4 * 128

  return cfg


def qwen3_1_7b() -> TPUTrainerConfig:
  """Qwen3 1.7B model configuration for TPU."""
  model_spec = model_registry("1.7B", attn_backend="sdpa")
  model_config = cast(Qwen3Model.Config, model_spec.model)
  model_config.enable_weight_tying = False
  model_config.tok_embeddings.param_init = {
      "weight": partial(nn.init.normal_, std=1.0)
  }
  model_spec.parallelize_fn = tpu_parallelize_qwen3

  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=600,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          max_norm=1.0,
          num_tokens_per_microbatch_per_dp_rank=1 * 2048,
          max_context_length=2048,
          steps=15,
          mixed_precision_reduce="bfloat16",
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=False,
      ),
      activation_checkpoint=FullAC.Config(),
      tpu_config=TPUConfig(
          eager_mode="DEFER_AND_FUSE",
          use_jax_profiler=False,
      ),
      compile=TpuTrainerCompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
          materialize_forward_before_backward=True,
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_block_q=512,
          sa_block_kv=512,
          sa_block_dkv=512,
          sa_block_kv_compute=512,
          sa_block_q_dkv=512,
          sa_block_kv_dkv=512,
          sa_block_kv_dkv_compute=512,
          sa_block_q_dq=512,
          sa_block_kv_dq=512,
      ),
      loss_kernel=LossKernelConfig(
          use_loss_kernel=False,
      ),
      validator=Validator.Config(
          enable=False,
          freq=5,
          steps=10,
      ),
  )


def reg_trainer_qwen3_8b() -> TPUTrainerConfig:
  """Qwen3 8B model configuration for TPU (regular trainer)."""
  model_spec = model_registry("8B", attn_backend="sdpa")
  model_config = cast(Qwen3Model.Config, model_spec.model)
  model_config.enable_weight_tying = False
  model_config.tok_embeddings.param_init = {
      "weight": partial(nn.init.normal_, std=1.0)
  }
  model_spec.parallelize_fn = tpu_parallelize_qwen3

  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=600,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          max_norm=1.0,
          num_tokens_per_microbatch_per_dp_rank=6 * 2048,
          max_context_length=2048,
          steps=15,
          mixed_precision_reduce="float32",
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          enable_tpu_topo_mesh=True,
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=False,
      ),
      activation_checkpoint=FullAC.Config(),
      tpu_config=TPUConfig(
          eager_mode="DEFER_AND_FUSE",
          use_jax_profiler=True,
      ),
      compile=TpuTrainerCompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
          materialize_forward_before_backward=True,
          compile_grad_norm=True,
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_block_q=1024,
          sa_block_kv=1024,
          sa_block_dkv=1024,
          sa_block_kv_compute=1024,
          sa_block_q_dkv=1024,
          sa_block_kv_dkv=1024,
          sa_block_kv_dkv_compute=1024,
          sa_block_q_dq=1024,
          sa_block_kv_dq=1024,
      ),
      loss_kernel=LossKernelConfig(
          use_loss_kernel=True,
      ),
      validator=Validator.Config(
          enable=False,
          freq=5,
          steps=10,
      ),
  )


def _qwen3_cp(
    config: TPUTrainerConfig,
    *,
    strategy: Literal["ulysses", "all_gather"],
    cp_degree: int,
) -> TPUTrainerConfig:
  """Enables TPU context parallelism, following pattern set by upstream torchtitan_recipes/muse_glimmer.py for CUDA.

  Sets the CP degree, the SPMD backend and the load balancer, then applies
  core ContextParallelTransform with the TPU CP inner attention for the
  model's attention backend.
  """
  assert config.model_spec is not None
  config.parallelism.context_parallel_degree = cp_degree
  config.parallelism.spmd_backend = "spmd_types"
  config.parallelism.context_parallel_load_balancer = tpu_cp_load_balancer(
      strategy
  )
  model = cast(Qwen3Model.Config, config.model_spec.model)
  inner_attention = model.first_full_attention_backend
  assert inner_attention is not None
  cp_inner_attention = tpu_cp_inner_attention(inner_attention._owner, strategy)
  return apply_transforms(
      config, [ContextParallelTransform(inner_attention=cp_inner_attention)]
  )


def qwen3_debugmodel_allgather_cp2() -> TPUTrainerConfig:
  """Qwen3 debug model with load-balanced KV all-gather CP degree 2.

  The debug model runs without the Splash kernel, so KV all-gather CP opts in
  to its plain-PyTorch reference attention, which is fine at this size.
  """
  config = _qwen3_cp(qwen3_debugmodel(), strategy="all_gather", cp_degree=2)
  assert config.model_spec is not None
  for _, attn, _, _ in config.model_spec.model.traverse(BaseAttention.Config):
    attn.inner_attention.allow_reference_attention = True
  return config


def qwen3_debugmodel_ulysses_cp2() -> TPUTrainerConfig:
  """Qwen3 debug model with Ulysses (all-to-all) CP degree 2."""
  return _qwen3_cp(qwen3_debugmodel(), strategy="ulysses", cp_degree=2)


def _qwen3_1_7b_32k() -> TPUTrainerConfig:
  """Qwen3 1.7B at 32k context, one sequence per data-parallel rank."""
  config = qwen3_1_7b()
  config.training.max_context_length = 32768
  config.training.num_tokens_per_microbatch_per_dp_rank = 32768
  return config


def qwen3_1_7b_allgather_cp4() -> TPUTrainerConfig:
  """Qwen3 1.7B at 32k context with load-balanced KV all-gather CP degree 4."""
  return _qwen3_cp(_qwen3_1_7b_32k(), strategy="all_gather", cp_degree=4)


def qwen3_1_7b_ulysses_cp4() -> TPUTrainerConfig:
  """Qwen3 1.7B at 32k context with Ulysses (all-to-all) CP degree 4."""
  return _qwen3_cp(_qwen3_1_7b_32k(), strategy="ulysses", cp_degree=4)


def _qwen3_dense_model_spec(flavor: str) -> ModelSpec:
  model_spec = model_registry(flavor, attn_backend="sdpa")
  model_config = cast(Qwen3Model.Config, model_spec.model)
  model_config.enable_weight_tying = False
  model_config.tok_embeddings.param_init = {
      "weight": partial(nn.init.normal_, std=1.0)
  }
  model_spec.parallelize_fn = tpu_parallelize_qwen3
  return model_spec


def qwen3_8b() -> TpuGraphTrainer.Config:
  """Qwen3 8B model configuration for TPU."""
  model_spec = _qwen3_dense_model_spec("8B")

  config = TPUTrainerConfig(
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      # Keep the established dense optimizer path until native fused AdamW has
      # a dedicated long-convergence validation for this recipe.
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=600,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          max_norm=1.0,
          num_tokens_per_microbatch_per_dp_rank=6 * 2048,
          max_context_length=2048,
          steps=15,
          mixed_precision_reduce="float32",
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          enable_tpu_topo_mesh=True,
          # The current best-performing benchmark recipe uses Zero-2. Keep it
          # explicit while Zero-3 memory/overlap policy is tuned separately.
          fsdp_reshard_after_forward="never",
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=False,
      ),
      # GraphTrainer does not apply this eager AC config; its graph
      # rematerialization policy is selected on the final compile config below.
      activation_checkpoint=SelectiveAC.Config(),
      tpu_config=TPUConfig(),
      # Disable Shardy HLO sharding v3 (enabled by default in XLA cl/979222288 /
      # libtpu 0.0.48; see cl/979992733), which segfaults in
      # mlir::sdy::buildTensorFactorShardings on GraphTrainer.
      xla_flags=tpu_flags.merge_flags(
          xla_enable_hlo_sharding_v3=False,
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          use_tokamax_splash_attention_kernel=True,
          # Tokamax masks packed documents with segment IDs and has no
          # document-aware block skipping.
          sa_use_document_aware_block_skipping=False,
          sa_q_layout="SEQ_MINOR",
          sa_k_layout="SEQ_MINOR",
          sa_v_layout="HEAD_DIM_MINOR",
          sa_block_q=1024,
          sa_block_kv=1024,
          sa_block_dkv=1024,
          sa_block_kv_compute=512,
          sa_block_q_dkv=2048,
          sa_block_kv_dkv=2048,
          sa_block_kv_dkv_compute=1024,
          sa_block_q_dq=1024,
          sa_block_kv_dq=1024,
      ),
      validator=Validator.Config(
          enable=False,
          freq=5,
          steps=10,
      ),
  )
  _enable_pallas_cross_entropy(config)
  graph_config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      config, graph_trainer_model_registry
  )
  graph_config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
      memory_policy="default",
      optimizer_compile="in_graph",
  )
  return graph_config


def qwen3_8b_zero3() -> TpuGraphTrainer.Config:
  """Qwen3 8B GraphTrainer recipe retaining sharded parameters after forward."""
  cfg = qwen3_8b()
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  cfg.compile.memory_policy = "full"
  cfg.compile.enable_remat_barriers = True
  cfg.compile.fsdp_barrier_policy = "interleave"
  cfg.splash_attention_kernel.use_vmap_bwd = True
  cfg.splash_attention_kernel.sa_use_document_aware_block_skipping = False
  return cfg


def qwen3_8b_maxtext_parity() -> TpuGraphTrainer.Config:
  """Qwen3 8B recipe matching currently available MaxText policies.

  MaxText keeps parameters sharded across its FSDP axis and rematerializes the
  layer body, which is operationally closest to Zero-3. Relative to the tuned
  Zero-3 recipe, this also rotates Q/K and reduces FSDP gradients in BF16.
  Layer scan is intentionally omitted, so this matches numerical and kernel
  policy but not the loop representation.
  """
  cfg = qwen3_8b_zero3()
  cfg.training.mixed_precision_reduce = "bfloat16"
  cfg.override.imports.append(_BF16_ROPE_OVERRIDE)
  _use_maxtext_flops(cfg)
  return cfg


# The Qwen3-8B tpu7x-4x4x4 launch flags, carried by the 32B config.
_QWEN3_DENSE_4X4X4_FLAGS = tpu_flags.merge_flags(
    tpu_flags.TPU7X_THROUGHPUT_P_STATE,
    {"xla_tpu_scoped_vmem_limit_kib": 65472},
    tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
    tpu_flags.HIERARCHICAL_REDUCE_SCATTER,
)


def qwen3_32b() -> TpuGraphTrainer.Config:
  """Qwen3 32B recipe for a tpu7x-4x4x4 slice.

  Uses the tuned Qwen3-8B Zero-3 policy: Zero-2 would keep all ~61 GiB of
  gathered BF16 weights resident.
  """
  cfg = qwen3_8b_zero3()
  cfg.model_spec = to_graph_trainer_model_spec(_qwen3_dense_model_spec("32B"))
  # Packed QKV removes the fused projection's interleaved-output relayouts:
  # -3.5% step time and -19.7 GiB peak HBM at local batch 8.
  cfg.model_spec.model = (
      TpuPackedQKVConverter.Config().build().convert(cfg.model_spec.model)
  )
  # Upstream d771abb0 replaced local_batch_size/global_batch_size/seq_len with
  # token budgets. Set the context length first; the budgets are 8 local and
  # 1024 global sequences at that context.
  cfg.training.max_context_length = 4096
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 8 * 4096
  cfg.training.num_tokens_per_train_step = 1024 * 4096
  # Prefetch beats interleave by 0.3-0.6% here for ~4 GiB more peak HBM.
  cfg.compile.fsdp_barrier_policy = "prefetch"
  cfg.libtpu_init_flags = tpu_flags.merge_flags(
      cfg.libtpu_init_flags, _QWEN3_DENSE_4X4X4_FLAGS
  )
  return cfg


def qwen3_32b_4x8x8() -> TpuGraphTrainer.Config:
  """Qwen3 32B recipe for a tpu7x-4x8x8 slice (512 devices).

  At world 512 the w1/w3 gradient shards are 25600 / 512 = 50 rows. That is
  not a multiple of 8, so their reduce-scatters run on the TensorCore instead
  of SparseCore (~2.5 s/step). All-reduce plus a local slice keeps them on
  SparseCore: 14.33 -> 12.34 s, for +49 GiB peak HBM.
  """
  cfg = qwen3_32b()
  # 4096 global sequences (upstream d771abb0 token budget).
  cfg.training.num_tokens_per_train_step = 4096 * 4096
  # Before the w1/w3 rewrite, interleave beat prefetch here by 2.3% step time
  # and 56 GiB peak HBM.
  cfg.compile.fsdp_barrier_policy = "interleave"
  cfg.compile.fsdp_allreduce_slice_parameter_patterns = (
      "layers.*.feed_forward.w1.weight",
      "layers.*.feed_forward.w3.weight",
  )
  cfg.compile.fsdp_allreduce_slice_max_input_bytes = 25600 * 5120 * 4
  return cfg


def qwen3_32b_maxtext_parity() -> TpuGraphTrainer.Config:
  """Qwen3 32B recipe with MaxText's BF16 reductions and BF16 RoPE."""
  cfg = qwen3_32b()
  cfg.training.mixed_precision_reduce = "bfloat16"
  cfg.override.imports.append(_BF16_ROPE_OVERRIDE)
  _use_maxtext_flops(cfg)
  return cfg


def qwen3_moe_30b() -> TpuGraphTrainer.Config:
  """TPU specialization of the global-aux Qwen3 30B recipe."""
  cfg = TPUTrainerConfig.derive_from(qwen3_30b_a3b_global_aux())
  cfg.optimizer.implementation = "foreach"
  cfg.loss = CrossEntropyLoss.Config()
  cfg.model_spec = model_registry(
      "30B-A3B-global-aux",
      attn_backend="sdpa",
      converters=[
          TpuTokenDispatcherConverter.Config(),
          TpuGroupedExpertsConverter.Config(),
          TpuRoutedExpertsConverter.Config(),
      ],
  )
  cfg.model_spec.parallelize_fn = tpu_parallelize_qwen3
  # Real C4 as on main, through the TPU loader so `--dataloader.dataset_path`
  # can point at a local mirror. Upstream's GrainDataLoader shuffles by
  # default; main's HuggingFaceTextDataLoader did not. Keep main's data order.
  cfg.dataloader = TPUGrainDataLoader.Config(dataset="c4", shuffle=False)
  # Upstream d771abb0 replaced local_batch_size/global_batch_size/seq_len with
  # token budgets. `qwen3_30b_a3b_global_aux` leaves max_context_length at its
  # 2048 default.
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 6 * 2048
  cfg.training.num_tokens_per_train_step = 768 * 2048
  cfg.parallelism = ParallelismConfig(
      use_simple_fsdp=True,
      data_parallel_shard_degree=-1,
      expert_parallel_degree=128,
  )
  cfg.tpu_config = TPUConfig()
  # Disable Shardy HLO sharding v3 (enabled by default in XLA cl/979222288 /
  # libtpu 0.0.48; see cl/979992733), which segfaults in
  # mlir::sdy::buildTensorFactorShardings on GraphTrainer.
  cfg.xla_flags = tpu_flags.merge_flags(
      cfg.xla_flags,
      xla_enable_hlo_sharding_v3=False,
  )
  cfg.splash_attention_kernel = SplashAttentionKernelConfig(
      use_splash_attention_kernel=True,
      sa_block_q=1024,
      sa_block_kv=1024,
      sa_block_dkv=1024,
      sa_block_kv_compute=1024,
      sa_block_q_dkv=1024,
      sa_block_kv_dkv=1024,
      sa_block_kv_dkv_compute=1024,
      sa_block_q_dq=1024,
      sa_block_kv_dq=1024,
  )
  cfg.loss_kernel = LossKernelConfig(use_loss_kernel=False)
  cfg.validator = Validator.Config(enable=False)
  graph_config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      cfg, graph_trainer_model_registry
  )
  graph_config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
      memory_policy="full",
  )
  return graph_config


def _qwen3_moe_30b_fsdp_only_full_ac() -> TpuGraphTrainer.Config:
  """Batch-6 FSDP128 baseline with local dropless expert dispatch."""
  cfg = qwen3_moe_30b()
  _enable_pallas_cross_entropy(cfg)
  cfg.parallelism = ParallelismConfig(
      use_simple_fsdp=True,
      enable_tpu_topo_mesh=True,
      data_parallel_shard_degree=-1,
      expert_parallel_degree=1,
      # The current best-performing benchmark recipe uses Zero-2. Keep it
      # explicit while Zero-3 memory/overlap policy is tuned separately.
      fsdp_reshard_after_forward="never",
  )
  assert cfg.model_spec is not None
  cfg.model_spec.model = (
      TpuPackedQKVConverter.Config().build().convert(cfg.model_spec.model)
  )
  for _, experts, _, _ in cfg.model_spec.model.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    experts.use_pallas_paired_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 256
    experts.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)
    experts.use_kernel_native_weight_layout = True
  _scale_routes_before_down_proj(cfg)
  # Upstream d771abb0 replaced local_batch_size/global_batch_size/seq_len with
  # token budgets.
  cfg.training.max_context_length = 2048
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 6 * 2048
  cfg.training.num_tokens_per_train_step = 768 * 2048
  cfg.training.steps = 8
  cfg.tpu_config = TPUConfig()
  cfg.optimizer.implementation = "fused"
  cfg.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      backend="tpu",
      components=["model"],
      memory_policy="tpu_selective",
      save_regions=("moe.routing", "attention.inner_attention"),
      compile_grad_norm=True,
      enable_remat_barriers=True,
      fsdp_allreduce_slice_parameter_patterns=(
          "layers.*.attention.qkv_linear.wqkv_packed.weight",
          "layers.*.attention.wo.weight",
      ),
      # The validated QKV and output-projection gradients are 40 and 32 MiB.
      # Larger reductions can lose to native reduce-scatter because all-reduce
      # sends roughly twice the ring traffic and materializes the full result.
      fsdp_allreduce_slice_max_input_bytes=40 * 1024**2,
  )
  cfg.splash_attention_kernel = SplashAttentionKernelConfig(
      use_splash_attention_kernel=True,
      use_tokamax_splash_attention_kernel=True,
      # Batch-native masking lets the always-vmapped forward avoid a serial
      # batch loop; use_vmap_bwd applies the same optimization to backward.
      # Runtime segment IDs still preserve exact packed-document boundaries.
      # At local batch 6, the combined change reduced XProf peak HBM from
      # 14.924 to 14.575 GiB (-0.349 GiB); no vmap memory penalty was observed.
      use_vmap_bwd=True,
      sa_use_document_aware_block_skipping=False,
      sa_q_layout="SEQ_MINOR",
      sa_k_layout="SEQ_MINOR",
      sa_block_q=1024,
      sa_block_kv=1024,
      sa_block_dkv=1024,
      # Matched Qwen3-30B/TPU7x profiling showed a 26.3-ms (1.60%) step win and
      # 0.199-GiB lower peak HBM. The attention calls themselves took 2.5 ms
      # longer, but module span fell 24.6 ms and exposed collectives fell
      # 7.4 ms: this is a whole-program resource/scheduling improvement.
      sa_block_kv_compute=512,
      sa_block_q_dkv=2048,
      sa_block_kv_dkv=2048,
      sa_block_kv_dkv_compute=1024,
      sa_block_q_dq=1024,
      sa_block_kv_dq=1024,
  )
  cfg.loss_kernel = LossKernelConfig(use_loss_kernel=False)
  cfg.metrics.log_freq = 1
  cfg.checkpoint = CheckpointManager.Config(enable=False)
  cfg.profiler = Profiler.Config(enable_profiling=False)
  cfg.validator = Validator.Config(enable=False)
  return cfg


def qwen3_moe_30b_fsdp_only_full_ac_profile() -> TpuGraphTrainer.Config:
  """Profile the batch-6 FSDP128 baseline on rank zero."""
  cfg = _qwen3_moe_30b_fsdp_only_full_ac()
  cfg.profiler = Profiler.Config(
      enable_profiling=int(os.environ.get("RANK", "0")) == 0,
      profile_freq=5,
      profiler_warmup=2,
      profiler_active=1,
  )
  return cfg


def qwen3_moe_30b_fsdp_only_full_ac_benchmark() -> TpuGraphTrainer.Config:
  """Profiler-free bracket for the batch-6 FSDP128 comparison."""
  cfg = _qwen3_moe_30b_fsdp_only_full_ac()
  cfg.training.steps = 30
  cfg.metrics.log_freq = 5
  cfg.tpu_config.enable_mfu_benchmark = True
  cfg.tpu_config.mfu_benchmark_start_step = 10
  return cfg


def qwen3_moe_30b_fsdp_only_attention_parity_benchmark() -> TpuGraphTrainer.Config:
  """Q30 benchmark adding MaxText's BF16 RoPE numerical policy.

  The base FSDP recipe already enables the correctness-preserving batch-native
  masking and Qwen3-30B/TPU7x tile choices. This separate recipe changes only
  RoPE arithmetic to BF16 for MaxText parity. That change is not numerically
  equivalent to TorchTitan's FP32 RoPE and must remain opt-in unless longer
  convergence validation justifies promoting it.
  """
  cfg = qwen3_moe_30b_fsdp_only_full_ac_benchmark()
  # Match MaxText's BF16 rotary arithmetic. This is an explicit numerical
  # policy choice, not a numerics-preserving optimization; keep it out of the
  # default recipe pending a longer convergence comparison.
  cfg.override.imports.append(_BF16_ROPE_OVERRIDE)
  return cfg


def qwen3_moe_30b_fsdp_only_maxtext_benchmark() -> TpuGraphTrainer.Config:
  """Q30 benchmark with MaxText's numerics and FLOP accounting.

  The attention-parity benchmark (BF16 RoPE) plus
  BF16 gradient reduction, matching the other ``*maxtext*`` recipes, and
  MaxText FLOPs, so its MFU is comparable with MaxText's published Q30 numbers.
  """
  cfg = qwen3_moe_30b_fsdp_only_attention_parity_benchmark()
  cfg.training.mixed_precision_reduce = "bfloat16"
  _use_maxtext_flops(cfg)
  return cfg


def _scale_routes_before_down_proj(cfg: TPUTrainerConfig) -> None:
  """Apply routing scores before W2 so remat does not replay the W2 forward."""
  assert cfg.model_spec is not None
  routed = list(cfg.model_spec.model.traverse(TpuRoutedExperts.Config))
  if not routed:
    raise ValueError("Scaling routes before W2 needs TpuRoutedExperts")
  for _, config, _, _ in routed:
    config.scale_before_down_proj = True


def _qwen3_moe_235b_fsdp_only() -> TpuGraphTrainer.Config:
  """Build the common Qwen3 235B-A22B FSDP configuration."""
  cfg = qwen3_moe_30b_fsdp_only_full_ac_benchmark()
  # Q30's rewrite is tuned for its full-world FP32 graph. Q235's explicit
  # FSDP-128, HSDP, and BF16 recipes violate that pass's current validator.
  cfg.compile.fsdp_allreduce_slice_parameter_patterns = ()
  cfg.compile.fsdp_allreduce_slice_max_input_bytes = 0
  # Likewise, the 30B tpu_selective saves were measured only on 30B, so
  # Q235 keeps plain full rematerialization unless a recipe picks another
  # policy.
  cfg.compile.memory_policy = "full"
  cfg.compile.save_regions = ()
  cfg.model_spec = to_graph_trainer_model_spec(
      model_registry(
          "235B-A22B",
          attn_backend="sdpa",
          converters=[
              TpuTokenDispatcherConverter.Config(),
              TpuGroupedExpertsConverter.Config(),
              TpuRoutedExpertsConverter.Config(),
          ],
      )
  )
  model_config = cast(Qwen3Model.Config, cfg.model_spec.model)
  # Replacing model_spec also replaces nested MoE policy inherited from the
  # 30B base recipe, so apply the intended Q235 routing policy afterwards.
  _use_global_batch_aux_loss(cfg)
  cfg.splash_attention_kernel.use_tokamax_splash_attention_kernel = True
  _scale_routes_before_down_proj(cfg)
  # The TPU Qwen parallelizer shards the uneven vocabulary and expert banks on
  # their divisible hidden axes when the FSDP mesh is wider than dim 0.
  for _, experts, _, _ in model_config.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    experts.use_pallas_paired_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 256
    # A dWeight is tiled along K, the weight's input dimension: D = 4096 for
    # the paired W1/W3, F = 1536 for W2. Untiled, the paired dWeight needs
    # 105 MB at 4096 x 1536 against Mosaic's 64 MB. Its N tile must stay full.
    #
    # The paired dWeight prefers a 2048 K tile, which W2 cannot take. A 2048
    # tile on a 1536 axis does not clamp: it allocates a f32[1,2048,4096]
    # window, 33.55 MB, doubled by the two buffering levels to the whole 64 MB
    # budget, and the module OOMs before the first step (measured on 64 v7x).
    # So the two kernels are tiled independently.
    experts.pallas_grouped_mm_tgmm_tiling = (512, 2048, 0)
    # A K tile that does not divide F leaves a masked remainder tile.
    experts.pallas_grouped_mm_w2_tgmm_tiling = (256, 768, 0)
    # With W2 sharded on its trailing axis (FSDP wider than the expert count),
    # the transposed-RHS dWeight is stored [E, D, F]: tile D and keep F whole.
    experts.pallas_grouped_mm_w2_transposed_tgmm_tiling = (256, 0, 2048)
    experts.use_kernel_native_weight_layout = True
  # Upstream d771abb0 replaced local_batch_size/global_batch_size/seq_len with
  # token budgets.
  cfg.training.max_context_length = 4096
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 16 * 4096
  cfg.training.num_tokens_per_train_step = 8192 * 4096
  cfg.training.steps = 15
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  cfg.compile.graph_split.enabled = True
  cfg.compile.graph_split.forward_layers_per_executable = 1
  cfg.compile.graph_split.forward_isolate_prefix = True
  cfg.compile.graph_split.backward_partition_mode = "fsdp_pipeline"
  cfg.compile.graph_split.backward_layers_per_executable = 1
  # This measured Q235 split formed one remat region per backward executable
  # and reduced the batch-16 allocator deficit by 9.30 GiB. Those executable
  # boundaries already provide the layer-local lifetime ordering that remat
  # barriers add to a multi-layer graph, so do not add redundant dependencies.
  cfg.compile.enable_remat_barriers = False
  cfg.compile.fsdp_barrier_policy = "none"
  cfg.compile.optimizer_compile = "eager"
  cfg.hf_assets_path = "./assets/hf/Qwen3-235B-A22B"
  cfg.metrics.log_freq = 1
  return cfg


def _qwen3_moe_235b_fsdp_activation_offload() -> TpuGraphTrainer.Config:
  """Add the common Q235 activation-offload execution policy."""
  cfg = _qwen3_moe_235b_fsdp_only()
  # Offload the layer boundaries, and keep the Splash output and LSE of the
  # last 36 layers in HBM so backward skips their forward recompute (~16 ms
  # each). Split-boundary donation is what frees the HBM for these saves.
  # Each saved layer costs ~1 GiB at the HBM peak. Measured at world 512: N=16,
  # 24, 32, 36, 40 cut step time 1.0, 1.4, 1.9, 2.1, 2.3%; N=36 leaves ~6 GiB
  # free and N=40 ~2 GiB. Lower N first if something else needs HBM (longer
  # context, bigger batch).
  cfg.compile.memory_policy = "tpu_selective_and_offload"
  cfg.compile.save_last_n_layers = {
      "attention.inner_attention": 36
  }
  cfg.compile.graph_split.forward_layers_per_executable = 1
  # Current ``by_layer`` is the direct successor of the ``sequential`` mode
  # used by the measured schedule: one complete backward layer per
  # independently compiled child.
  cfg.compile.graph_split.backward_partition_mode = "by_layer"
  cfg.compile.graph_split.backward_layers_per_executable = 1
  # Issue each H2D pair in the child immediately before its consumer. This
  # removed 1.31 s of exposed H2D in the measured BF16 arm at a 0.96-GiB HBM
  # cost from carrying the completed activation across one child boundary.
  cfg.compile.graph_split.cpu_offload_prefetch_executables = 1
  cfg.libtpu_init_flags = tpu_flags.merge_flags(
      cfg.libtpu_init_flags,
      _Q235_COMPILED_OFFLOAD_FLAGS,
  )
  return cfg


def _without_splash_saves(cfg: TpuGraphTrainer.Config) -> None:
  """Fall back to plain offload; the Splash saves were measured only at w512."""
  cfg.compile.memory_policy = "full_and_offload"
  cfg.compile.save_last_n_layers = {}


def qwen3_moe_235b_fsdp_activation_offload_maxtext() -> TpuGraphTrainer.Config:
  """MaxText-numerics variant of the Q235 activation-offload recipe."""
  cfg = _qwen3_moe_235b_fsdp_activation_offload()
  _without_splash_saves(cfg)
  cfg.training.mixed_precision_reduce = "bfloat16"
  if _BF16_ROPE_OVERRIDE not in cfg.override.imports:
    cfg.override.imports.append(_BF16_ROPE_OVERRIDE)
  _use_maxtext_flops(cfg)
  return cfg


def _configure_qwen3_moe_235b_fsdp_128(
    cfg: TpuGraphTrainer.Config,
    *,
    replicate_degree: int,
) -> None:
  """Configure an FSDP-128 mesh and its matching global batch size."""
  cfg.parallelism.data_parallel_replicate_degree = replicate_degree
  cfg.parallelism.data_parallel_shard_degree = 128
  # Upstream d771abb0 replaced local_batch_size/global_batch_size/seq_len with
  # token budgets. Every caller passes a config descended from
  # `_qwen3_moe_235b_fsdp_only`, which has already set `max_context_length`, so
  # scale the 2048-sequence global batch by the context length in effect.
  cfg.training.num_tokens_per_train_step = (
      2048 * replicate_degree * cfg.training.max_context_length
  )


def qwen3_moe_235b_hsdp_activation_offload_maxtext() -> TpuGraphTrainer.Config:
  """MaxText-numerics Q235 offload recipe using 4x128 HSDP.

  Matching the FSDP shard degree to the 128-expert dimension keeps expert
  weights on their natural shard axis and avoids hidden-axis reconstruction.
  """
  cfg = qwen3_moe_235b_fsdp_activation_offload_maxtext()
  _configure_qwen3_moe_235b_fsdp_128(cfg, replicate_degree=4)
  return cfg


def qwen3_moe_235b_fsdp_activation_offload_4x4x4() -> TpuGraphTrainer.Config:
  """FP32-reduction Q235 offload recipe for a 4x4x4 TPU7x slice.

  Pure FSDP over 128 ranks divides both the 128 experts and the 151,936-token
  vocabulary evenly. Keep local batch 16 and use one microbatch per step.
  """
  cfg = _qwen3_moe_235b_fsdp_activation_offload()
  _without_splash_saves(cfg)
  _configure_qwen3_moe_235b_fsdp_128(cfg, replicate_degree=1)
  # This topology's large FSDP reductions benefit from libtpu's hierarchical
  # n-D lowering. Keep the unvalidated HSDP and MaxText-numerics variants on
  # their existing policy.
  cfg.libtpu_init_flags = tpu_flags.merge_flags(
      cfg.libtpu_init_flags,
      tpu_flags.HIERARCHICAL_REDUCE_SCATTER,
  )
  return cfg


def qwen3_moe_235b_fsdp_activation_offload_maxtext_4x4x4(
) -> TpuGraphTrainer.Config:
  """MaxText-numerics Q235 offload recipe for a 4x4x4 TPU7x slice."""
  cfg = qwen3_moe_235b_fsdp_activation_offload_maxtext()
  _configure_qwen3_moe_235b_fsdp_128(cfg, replicate_degree=1)
  return cfg


def qwen3_moe_235b_fsdp_smoke() -> TpuGraphTrainer.Config:
  """Four-layer FSDP512 smoke for the production 235B execution path."""
  cfg = _qwen3_moe_235b_fsdp_only()
  assert cfg.model_spec is not None
  model_config = cast(Qwen3Model.Config, cfg.model_spec.model)
  model_config.layers = model_config.layers[:4]
  # Upstream d771abb0 replaced local_batch_size/global_batch_size/seq_len with
  # token budgets. Set the context length first; the microbatch budget below is
  # one sequence at the new, shorter context.
  cfg.training.max_context_length = 128
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 1 * 128
  # `num_tokens_per_train_step` has to be restated, not inherited: under the
  # old fields `_qwen3_moe_235b_fsdp_only`'s `global_batch_size = 8192` was a
  # sequence count that re-scaled automatically with this shorter context, but
  # a token budget is absolute. 8192 * 128 keeps the pre-import behaviour.
  cfg.training.num_tokens_per_train_step = 8192 * 128
  cfg.training.steps = 2
  cfg.tpu_config.enable_mfu_benchmark = False
  return cfg


def qwen3_moe_235b_scan() -> TpuGraphTrainer.Config:
  """Qwen3 235B-A22B on the graph trainer with its 94 layers scanned.

  Shaped after MaxText's published Ironwood recipe
  (`tpu-recipes/training/ironwood/qwen3-235b-a22b/4k-bf16-tpu7x-4x8x8`): 256
  chips, sequence 4096, batch 16 per device, activation offloading on, no
  expert parallelism. Batch and global batch are the two things to override on
  a smaller slice -- `num_tokens_per_train_step` must equal ranks x
  `num_tokens_per_microbatch_per_dp_rank` or config validation fails with a
  message that reads like a scan error.
  """
  cfg = qwen3_moe_30b_fsdp_only_full_ac_benchmark()
  cfg.model_spec = to_graph_trainer_model_spec(
      model_registry(
          "235B-A22B",
          attn_backend="sdpa",
          converters=[
              TpuTokenDispatcherConverter.Config(),
              TpuGroupedExpertsConverter.Config(),
              TpuRoutedExpertsConverter.Config(),
          ],
      )
  )
  # Store the QKV weights as [all Q][all K][all V] rather than grouped per KV
  # head, so attention reads Q, K and V without per-layer copies.
  cfg.model_spec.model = (
      TpuPackedQKVConverter.Config().build().convert(cfg.model_spec.model)
  )
  model_config = cast(Qwen3Model.Config, cfg.model_spec.model)
  _use_global_batch_aux_loss(cfg)
  _scale_routes_before_down_proj(cfg)
  cfg.splash_attention_kernel.use_tokamax_splash_attention_kernel = True
  # Clearing the coeff does not stop `MoE.forward` accumulating into a counter
  # nothing reads, and a scan body cannot write to a tensor it closed over.
  # TODO(#22): delete once `MoE.forward` guards that write.
  cfg.override.imports.append(_DISCARDED_TOKEN_COUNTER_OVERRIDE)
  for _, experts, _, _ in model_config.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    # The splitter's expert kernels: W1/W3 paired, with its K-tiled dWeight.
    experts.use_pallas_paired_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 256
    # A dWeight is tiled along K, the weight's input dimension: D for the
    # paired W1/W3, the smaller F for W2. The K tile that measures best for
    # W1/W3 does not fit W2, so tile them independently.
    experts.pallas_grouped_mm_tgmm_tiling = (512, 2048, 0)
    # A K tile that does not divide F leaves a masked remainder tile.
    experts.pallas_grouped_mm_w2_tgmm_tiling = (256, 768, 0)
    experts.use_kernel_native_weight_layout = True
  # Upstream d771abb0 replaced local_batch_size/global_batch_size/seq_len with
  # token budgets.
  cfg.training.max_context_length = 4096
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 16 * 4096
  cfg.training.num_tokens_per_train_step = 8192 * 4096
  cfg.training.steps = 15
  # The 30B chain leaves this at the real `c4`, whose only builder config is
  # `default`; asking for `en` fails before the first step.
  cfg.dataloader = c4_test_dataloader()
  # 94 layers of 128 experts do not fit without Zero-3 plus offloading, which
  # is also what MaxText does here (`remat_policy=custom`,
  # `decoder_layer_input=offload`).
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  cfg.compile.use_scan_layers = True
  cfg.compile.memory_policy = "full_and_scan_offload"
  # The 30B recipe's save regions apply only to tpu_selective.
  cfg.compile.save_regions = ()
  # The 30B recipe's reduce-scatter -> all-reduce pass picks weights by
  # per-layer name (layers.*...). Scan stacks the layers into one tensor, so no
  # name matches and the pass would raise; turn it off.
  cfg.compile.fsdp_allreduce_slice_parameter_patterns = ()
  # The routing metrics stash tensors on the module, which a scan body cannot
  # do (https://github.com/google-pytorch/torchtitan/issues/28).
  cfg.metrics.enable_moe_routing = False
  cfg.metrics.log_freq = 1
  # Left to the command line these get forgotten, and a scan step measured
  # without them runs ~10% slow.
  cfg.libtpu_init_flags = tpu_flags.merge_flags(
      cfg.libtpu_init_flags,
      _Q235_SCAN_FLAGS,
  )
  return cfg


def qwen3_moe_235b_scan_torchao_float8() -> TpuGraphTrainer.Config:
  """`qwen3_moe_235b_scan` with torchao rowwise float8 dense linears.

  A reference for the upstream float8 path on Qwen3, not a speed recipe. The
  attention projections (wqkv/wo) become torchao `Float8Linear`s through
  `TpuFloat8LinearConverter`, the converter `llama3_70b_float8` uses, so their
  matmuls go through TorchTPU's `torch._scaled_mm`. The router gate and lm_head
  stay BF16. So do the routed experts: they run in `TpuGroupedExperts`' Pallas
  kernels, which torchao's converters do not target.
  """
  cfg = qwen3_moe_235b_scan()
  assert cfg.model_spec is not None
  # Convert only the attention projections (wqkv/wo); the routed experts are
  # grouped matmuls, not linears. filter_fqns entries match as substrings of a
  # linear's FQN and keep it in BF16:
  # * router.gate: its logits pick each token's top-k experts, so FP8 error can
  #   flip routing, and it is tiny (dim x num_experts). The converter also
  #   rejects router gates that are not filtered out.
  # * lm_head: its error lands directly on the vocab logits and the loss.
  converter = TpuFloat8LinearConverter.Config(
      recipe_name="rowwise", filter_fqns=["router.gate", "lm_head"]
  ).build()
  cfg.model_spec.model = converter.convert(cfg.model_spec.model)
  return cfg


def qwen3_moe_235b_scan_proxy() -> TpuGraphTrainer.Config:
  """`qwen3_moe_235b_scan` cut to 12 of its 94 layers, so it fits a 32-chip slice.

  Every layer is a real 235B layer -- dim 4096, 128 experts, top-k 8 -- so the
  scan body is the shape the full model would scan over. Only the trip count
  changes, which is the one thing scan is meant not to care about.
  """
  cfg = qwen3_moe_235b_scan()
  assert cfg.model_spec is not None
  model_config = cast(Qwen3Model.Config, cfg.model_spec.model)
  model_config.layers = model_config.layers[:12]
  # Upstream d771abb0 replaced local_batch_size/global_batch_size/seq_len with
  # token budgets. `qwen3_moe_235b_scan` leaves max_context_length at 4096.
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 1 * 4096
  cfg.training.num_tokens_per_train_step = 32 * 4096
  return cfg


def _with_fp8_experts(cfg: TpuGraphTrainer.Config) -> TpuGraphTrainer.Config:
  assert cfg.model_spec is not None
  for _, experts, _, _ in cfg.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.use_fp8_grouped_mm = True
    # In-kernel W1/W3 lhs quantization needs v7x, which this recipe targets.
    experts.fp8_w13_fused_lhs_quant = True
    # Each rank quantizes its FSDP shard of the expert weights and the scan
    # body all-gathers FP8 instead of BF16.
    experts.fp8_gather_quantized_weights = True
  # The FP8 fused SwiGLU + W2 op cannot apply the routing scores before W2, so
  # keep the score multiply after the experts, with its backward in one FP32
  # fusion.
  for _, routed, _, _ in cfg.model_spec.model.traverse(TpuRoutedExperts.Config):
    routed.scale_before_down_proj = False
  for _, dispatcher, _, _ in cfg.model_spec.model.traverse(
      TpuAllToAllTokenDispatcher.Config
  ):
    dispatcher.fuse_route_score_backward = True
  return cfg


def qwen3_moe_235b_scan_fp8() -> TpuGraphTrainer.Config:
  """`qwen3_moe_235b_scan` with FP8 expert GEMMs.

  The expert W1/W3/W2 GEMMs and their backward run on FP8 operands through
  tokamax's Mosaic GMM/TGMM (e4m3 forward, e5m2 gradients, dynamic row-wise
  scales; see `kernels/fp8_expert_segment_matmul.py`). Attention, router and
  lm_head stay BF16.
  """
  return _with_fp8_experts(qwen3_moe_235b_scan())


def qwen3_moe_235b_scan_maxtext() -> TpuGraphTrainer.Config:
  """`qwen3_moe_235b_scan` with MaxText's numerics instead of torchtitan's.

  Two changes, both bf16 where torchtitan keeps f32: the gradient reduce and
  the rotary arithmetic. Neither is bit-preserving, so the base recipe stays the
  accuracy reference and this one is the speed comparison against MaxText's
  published Ironwood number.
  """
  cfg = qwen3_moe_235b_scan()
  cfg.training.mixed_precision_reduce = "bfloat16"
  cfg.override.imports.append(_BF16_ROPE_OVERRIDE)
  _use_maxtext_flops(cfg)
  return cfg


def qwen3_moe_30b_proxy() -> TpuGraphTrainer.Config:
  """Proxy model for Qwen3 30B MoE (4 layers instead of 48) for testing/iterations."""
  cfg = qwen3_moe_30b()
  assert cfg.model_spec is not None
  model_config = cast(Qwen3Model.Config, cfg.model_spec.model)
  # Slice layers to only keep 4 layers to save memory and compile time
  model_config.layers = model_config.layers[:4]
  return cfg
