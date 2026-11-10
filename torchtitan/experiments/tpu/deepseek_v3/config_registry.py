# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

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
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.components.datasets import (
    c4_test_dataloader,
    TPUGrainDataLoader,
)
from torchtitan.experiments.tpu.components.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.components.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuRoutedExpertsConverter,
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.deepseek_v3 import model_registry
from torchtitan.experiments.tpu.graph_trainer.config import (
    TpuGraphTrainerCompileConfig,
)
from torchtitan.experiments.tpu.graph_trainer.deepseek_v3 import (
    model_registry as graph_trainer_model_registry,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    LossKernelConfig,
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
)
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3.model import DeepSeekV3Model
from torchtitan.tools.profiler import Profiler


_FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"
)

# The Qwen3-235B offload recipe's libtpu flags
# (qwen3/config_registry.py: _Q235_COMPILED_OFFLOAD_FLAGS), carried over as the
# starting point. Each was measured on Q235, none yet on DeepSeek-V3.
_DSV3_236B_OFFLOAD_FLAGS = tpu_flags.merge_flags(
    tpu_flags.TPU7X_THROUGHPUT_P_STATE,
    tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
    tpu_flags.DUAL_SPARSECORE_REDUCE_SCATTER,
    {
        "xla_tpu_scoped_vmem_limit_kib": 131072,
        # Keep activation host copies off SparseCore, where the FSDP
        # collectives run.
        "xla_tpu_enable_offloading_copy_to_sparsecore": False,
        # Prove copy/overwrite ordering across donated aliases instead of
        # snapshotting activations defensively.
        "xla_tpu_copy_insertion_use_region_analysis": True,
        "xla_tpu_copy_insertion_use_region_analysis_limit": 1000,
        # The native non-leading-axis FSDP gathers (expert banks, wkv_a and
        # the router gate are on dim 1 here) need async all-gathers kept as
        # all-gathers rather than rewritten to all-reduce plus slice.
        "xla_tpu_prefer_async_allgather_to_allreduce": True,
    },
)


def _tpu_moe_converters() -> list:
  return [
      TpuTokenDispatcherConverter.Config(),
      TpuGroupedExpertsConverter.Config(),
      TpuRoutedExpertsConverter.Config(),
  ]


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Select the fused loss and its pre-construction LM-head override together."""
  config.loss = PallasCrossEntropyLoss.Config()
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)


def _disable_expert_bias_balancer(config: TPUTrainerConfig) -> None:
  """Turn off the expert-bias balancer, keeping DeepSeek's aux loss.

  The balancer (``load_balance_coeff``) updates ``expert_bias`` in place from
  a counter that ``MoE.forward`` accumulates; neither has been exercised on the
  TPU graph path. DeepSeek-V3's own microbatch-wise aux loss (coeff 1e-3,
  computed locally with no collective) stays on. This is a numerics change
  against upstream: recipes that call this are labelled as such.
  """
  assert config.model_spec is not None
  for _, moe, _, _ in config.model_spec.model.traverse(MoE.Config):
    moe.load_balance_coeff = None


def deepseek_v3_debugmodel() -> TPUTrainerConfig:
  """DeepSeek-v3 debug model configuration for TPU."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry("debugmodel", attn_backend="sdpa"),
      dataloader=c4_test_dataloader(),
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
      parallelism=ParallelismConfig(
          expert_parallel_degree=1,
      ),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(),
  )


def graph_trainer_deepseek_v3_debugmodel() -> TpuGraphTrainer.Config:
  """DeepSeek-v3 debug model using TPU GraphTrainer."""
  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      deepseek_v3_debugmodel(), graph_trainer_model_registry
  )
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
  )
  return config


def _with_tpu_kernels(config: TPUTrainerConfig) -> None:
  """TPU MoE modules, SimpleFSDP, and Tokamax Splash for MLA attention.

  Tokamax Splash takes MLA's unequal head dims (q/k 192, v 128) and turns the
  per-document positions into segment ids, so packed rows stop attending
  across documents as they do under the plain ``sdpa`` backend. Every query
  head has its own K/V head here, so there is nothing to stack.
  """
  assert config.model_spec is not None
  config.model_spec = model_registry(
      config.model_spec.flavor,
      seq_len=config.training.max_context_length,
      attn_backend="sdpa",
      converters=_tpu_moe_converters(),
  )
  _disable_expert_bias_balancer(config)
  config.parallelism = ParallelismConfig(
      use_simple_fsdp=True,
      data_parallel_shard_degree=-1,
      expert_parallel_degree=1,
  )
  config.splash_attention_kernel = SplashAttentionKernelConfig(
      use_splash_attention_kernel=True,
      use_tokamax_splash_attention_kernel=True,
      # Tokamax has no document-aware block skipping; segment ids still mask
      # packed documents exactly.
      sa_use_document_aware_block_skipping=False,
  )
  # Shardy HLO sharding v3 (on by default since libtpu 0.0.48) segfaults in
  # mlir::sdy::buildTensorFactorShardings on GraphTrainer; see qwen3_moe_30b.
  config.xla_flags = tpu_flags.merge_flags(
      config.xla_flags,
      xla_enable_hlo_sharding_v3=False,
  )


def deepseek_v3_debugmodel_tokamax() -> TPUTrainerConfig:
  """Debug model on the TPU stack: TPU MoE, SimpleFSDP, Tokamax Splash.

  The eager / per-block compile reference for the GraphTrainer recipe below.
  Numerics differ from ``deepseek_v3_debugmodel`` by design: document-masked
  attention instead of causal-only, and no expert-bias balancer
  (``_disable_expert_bias_balancer``).
  """
  config = deepseek_v3_debugmodel()
  _with_tpu_kernels(config)
  return config


def graph_trainer_deepseek_v3_debugmodel_tokamax() -> TpuGraphTrainer.Config:
  """``deepseek_v3_debugmodel_tokamax`` on the TPU GraphTrainer.

  With the 236B recipe's execution policy: one graph-split executable per
  layer and layer-boundary host offload. Compiled as one program instead, XLA
  fails in the vmapped Tokamax forward ("reshape should have supported layout
  before reaching the emitter", libtpu 0.0.47); split, each layer compiles.
  """
  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      deepseek_v3_debugmodel_tokamax(), graph_trainer_model_registry
  )
  config.parallelism.fsdp_reshard_after_forward = "always"
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
      memory_policy="full_and_offload",
  )
  split = config.compile.graph_split
  split.enabled = True
  split.forward_layers_per_executable = 1
  split.forward_isolate_prefix = True
  split.backward_partition_mode = "by_layer"
  split.backward_layers_per_executable = 1
  split.cpu_offload_prefetch_executables = 1
  return config


def _deepseek_v3_236b_fsdp_only() -> TpuGraphTrainer.Config:
  """DeepSeek-V3 236B (DeepSeek-V2 shape), FSDP-only on the GraphTrainer.

  Patterned on ``_qwen3_moe_235b_fsdp_only``: one graph-split executable per
  layer, full rematerialization, fused cross entropy, Pallas grouped matmuls.
  Starts from random weights. Numerics class: semantic against upstream (no
  expert-bias balancer).
  """
  base = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      # The DeepSeek-V2 tokenizer: its 102,400-token vocabulary is the 236B
      # flavor's. DeepSeek-V3's (129,280) would index past the embedding.
      hf_assets_path="./assets/hf/DeepSeek-V2",
      metrics=MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry("236B", seq_len=4096),
      dataloader=TPUGrainDataLoader.Config(dataset="c4", shuffle=False),
      optimizer=default_adamw(lr=2.2e-4, implementation="fused"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="cosine",
          min_lr_factor=0.1,
      ),
      training=TrainingConfig(
          max_context_length=4096,
          # Half of Q235's 16 per rank: MLA's 128 full-width K/V heads make
          # the attention recompute larger. Not yet tuned.
          num_tokens_per_microbatch_per_dp_rank=8 * 4096,
          steps=15,
      ),
      checkpoint=CheckpointManager.Config(enable=False),
      profiler=Profiler.Config(enable_profiling=False),
      validator=Validator.Config(enable=False),
      tpu_config=TPUConfig(),
      loss_kernel=LossKernelConfig(use_loss_kernel=False),
  )
  _with_tpu_kernels(base)
  _enable_pallas_cross_entropy(base)
  base.parallelism = ParallelismConfig(
      use_simple_fsdp=True,
      enable_tpu_topo_mesh=True,
      data_parallel_shard_degree=-1,
      expert_parallel_degree=1,
      fsdp_reshard_after_forward="always",
  )
  model_config = cast(DeepSeekV3Model.Config, base.model_spec.model)
  for _, experts, _, _ in model_config.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    # Not the paired W1/W3 kernel Q235 uses: its forward holds one whole
    # expert weight of each, double-buffered, in VMEM. At D = 5120 that is
    # 2 x 2 x 15 MiB, and the kernel needs 70.5 of TPU7x's 64 MiB (measured).
    # The unpaired kernel holds one weight at a time.
    experts.use_pallas_paired_grouped_mm = False
    experts.pallas_grouped_mm_tile_m = 256
    experts.use_kernel_native_weight_layout = True
    # Same expert hidden size as Q235 (F = 1536), wider model (D = 5120 vs
    # 4096). A dWeight is tiled along K, its weight's input dimension: D for
    # the paired W1/W3, F for W2. Q235's 2048 K tile would leave a masked
    # remainder on D = 5120, so these use 1024, which divides it. Unmeasured.
    experts.pallas_grouped_mm_tgmm_tiling = (512, 1024, 0)
    experts.pallas_grouped_mm_w2_tgmm_tiling = (256, 768, 0)
    # 160 experts do not divide 128+ ranks, so W2 is sharded on its trailing D
    # axis and stored [E, D, F]: tile D, keep F whole.
    experts.pallas_grouped_mm_w2_transposed_tgmm_tiling = (256, 0, 1024)

  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      base, graph_trainer_model_registry
  )
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      backend="tpu",
      components=["model"],
      memory_policy="full",
      compile_grad_norm=True,
      enable_remat_barriers=False,
      fsdp_barrier_policy="none",
      optimizer_compile="eager",
  )
  config.compile.graph_split.enabled = True
  config.compile.graph_split.forward_layers_per_executable = 1
  config.compile.graph_split.forward_isolate_prefix = True
  config.compile.graph_split.backward_partition_mode = "fsdp_pipeline"
  config.compile.graph_split.backward_layers_per_executable = 1
  return config


def _deepseek_v3_236b_fsdp_activation_offload() -> TpuGraphTrainer.Config:
  """Adds full rematerialization with layer-boundary host offload."""
  config = _deepseek_v3_236b_fsdp_only()
  config.compile.memory_policy = "full_and_offload"
  config.compile.graph_split.backward_partition_mode = "by_layer"
  # Issue each host-to-device copy in the executable right before its
  # consumer, as the Q235 offload recipe does.
  config.compile.graph_split.cpu_offload_prefetch_executables = 1
  config.compile.graph_split.expert_all_gathers_under_splash = True
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags,
      _DSV3_236B_OFFLOAD_FLAGS,
  )
  return config


def deepseek_v3_236b_fsdp_activation_offload_4x4x4() -> TpuGraphTrainer.Config:
  """DeepSeek-V3 236B, FSDP over the 128 ranks of a TPU7x 4x4x4 slice.

  Shard axes at 128 ranks: the 102,400-token vocabulary divides; the 160-expert
  banks, ``wkv_a`` (576 rows) and the router gate (160 rows) move to an axis
  that divides (see ``configure_deepseek_v3_simple_fsdp_shard_dims``).
  """
  config = _deepseek_v3_236b_fsdp_activation_offload()
  config.parallelism.data_parallel_shard_degree = 128
  # One microbatch per step: 128 ranks x 8 sequences.
  config.training.num_tokens_per_train_step = (
      128 * config.training.num_tokens_per_microbatch_per_dp_rank
  )
  # As in the Q235 4x4x4 recipe: hierarchical n-D reduce-scatter lowering, and
  # each layer's reduce-scatters run in the next executable.
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags,
      tpu_flags.HIERARCHICAL_REDUCE_SCATTER,
  )
  config.compile.graph_split.backward_partition_mode = "fsdp_pipeline"
  config.compile.graph_split.pipeline_kinds = ("rs",)
  return config


def deepseek_v3_236b_fsdp_smoke() -> TpuGraphTrainer.Config:
  """The 236B offload recipe cut to 4 layers, for a few chips.

  Full-width layers (1 dense, 3 MoE), so the Pallas tiles, Splash shapes and
  shard axes are the production ones; short sequences, the test tokenizer and
  local C4 sample so it runs without assets.
  """
  config = _deepseek_v3_236b_fsdp_activation_offload()
  model_config = cast(DeepSeekV3Model.Config, config.model_spec.model)
  model_config.layers = model_config.layers[:4]
  config.hf_assets_path = "./tests/assets/tokenizer"
  config.dataloader = c4_test_dataloader()
  config.training.max_context_length = 1024
  config.training.num_tokens_per_microbatch_per_dp_rank = 1 * 1024
  config.training.num_tokens_per_train_step = -1
  config.training.steps = 10
  return config
