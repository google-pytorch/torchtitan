"""TPU configuration registry for Qwen3.5 (Dense and MoE) models."""

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import (
    CommConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.datasets import (
    TPUGrainDataLoader,
    c4_test_dataloader,
)
from torchtitan.experiments.tpu.graph_trainer.config import (
    GraphSplitterConfig,
    TpuGraphTrainerCompileConfig,
    TpuGraphTrainerMemoryPolicy,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.qwen3_5 import model_registry
from torchtitan.experiments.tpu.qwen3_5.gated_delta import TpuGatedDeltaNet
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
    TpuTrainerCompileConfig,
)

_TPU_MOE_CONVERTERS = [
    TpuTokenDispatcherConverter.Config(),
    TpuGroupedExpertsConverter.Config(),
]

FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"
)
# Applied via override resolution so recipes can pass `min_log_decay` /
# `max_num_segments` kwargs (`tokamax_gated_delta` imports KDA lazily).
TOKAMAX_GATED_DELTA_OVERRIDE = (
    "torchtitan.experiments.tpu.qwen3_5.tokamax_gated_delta."
    "tokamax_gated_delta_kernel"
)


def _enable_pallas_cross_entropy(
    cfg: TPUTrainerConfig,
    *,
    b_block_size: int = 1024,
    h_block_size: int = 512,
    v_block_size: int = 2048,
) -> None:
  """Selects the Pallas fused linear cross-entropy loss and LM-head wrapper."""
  cfg.loss = PallasCrossEntropyLoss.Config(
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
  )
  if FUSED_LOSS_OVERRIDE not in cfg.override.imports:
    cfg.override.imports.append(FUSED_LOSS_OVERRIDE)


def _use_tpu_moe_kernels(cfg: TPUTrainerConfig) -> None:
  """Selects the Qwen3-30B-A3B TPU recipe's MoE kernels.

  Pallas paired grouped matmuls with kernel-native expert weights, and dScore
  in one fused FP32 reduction instead of saving an FP32 copy of every expert
  output. These are per-model-config settings, so a recipe that replaces
  `model_spec` must apply them again.

  Args:
    cfg: The recipe to update in place.
  """
  assert cfg.model_spec is not None
  experts = list(cfg.model_spec.model.traverse(TpuGroupedExperts.Config))
  dispatchers = list(
      cfg.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config)
  )
  if not experts or not dispatchers:
    raise ValueError(
        "The TPU MoE kernels need TpuGroupedExperts and a TPU token "
        "dispatcher (_TPU_MOE_CONVERTERS)"
    )
  for _, experts_config, _, _ in experts:
    experts_config.use_pallas_grouped_mm = True
    experts_config.use_pallas_paired_grouped_mm = True
    experts_config.pallas_grouped_mm_tile_m = 256
    experts_config.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)
    experts_config.use_kernel_native_weight_layout = True
  for _, dispatcher, _, _ in dispatchers:
    dispatcher.fuse_route_score_backward = True


def _no_gated_delta_document_resets(cfg: TPUTrainerConfig) -> None:
  """Runs every GatedDeltaNet over whole rows, without packed-document resets.

  Pretraining on packed text does not need the causal convolution and the
  recurrence to restart at each document (upstream does not without
  `cu_seqlens`), and the Tokamax kernels' restarts lay every document out
  from a 64-token chunk boundary. Full attention still masks documents.

  Args:
    cfg: The recipe to update in place.
  """
  assert cfg.model_spec is not None
  delta_nets = list(cfg.model_spec.model.traverse(TpuGatedDeltaNet.Config))
  if not delta_nets:
    raise ValueError("Expected TpuGatedDeltaNet layers in cfg.model_spec.model")
  for _, delta_net, _, _ in delta_nets:
    delta_net.reset_at_documents = False


def _use_tokamax_gated_delta(
    cfg: TPUTrainerConfig, **override_kwargs: float | int | None
) -> None:
  """Runs every GatedDeltaNet with Tokamax's Mosaic TPU KDA kernels."""
  has_existing = any(
      (entry if isinstance(entry, str) else entry[0])
      == TOKAMAX_GATED_DELTA_OVERRIDE
      for entry in cfg.override.imports
  )
  if has_existing and not override_kwargs:
    return
  cfg.override.imports = [
      entry
      for entry in cfg.override.imports
      if (entry if isinstance(entry, str) else entry[0])
      != TOKAMAX_GATED_DELTA_OVERRIDE
  ]
  cfg.override.imports.append(
      (TOKAMAX_GATED_DELTA_OVERRIDE, override_kwargs)
      if override_kwargs
      else TOKAMAX_GATED_DELTA_OVERRIDE
  )


def _use_graph_split(
    base_cfg: TPUTrainerConfig,
    *,
    layers_per_executable: int,
    memory_policy: TpuGraphTrainerMemoryPolicy = "full",
    save_regions: tuple[str, ...] = (),
) -> TpuGraphTrainer.Config:
  """Converts `base_cfg` to per-layer executables on `TpuGraphTrainer`.

  As one unrolled step graph, the 40-layer model neither compiles within an
  hour on 8 host CPUs nor fits in HBM on one tpu7x-2x2x1 host: XLA keeps most
  layers' unsharded expert gradients (3 x 512 MiB bf16 weights, 3 x 1 GiB fp32
  gradients per layer) alive until late in the step. Per-layer forward and
  backward executables bound both: each backward child reduce-scatters its own
  layers' gradients, and repeated layers hit the compilation cache. As in the
  Qwen3-235B recipes, `fsdp_pipeline` issues each child's all-gather one child
  early and its reduce-scatter one child late, and AdamW runs outside the step
  graph as torch_tpu's fused AdamW (graph splitting does not support
  `optimizer_compile='in_graph'`).

  Args:
    base_cfg: The base TPU recipe to convert.
    layers_per_executable: Transformer layers per forward and backward child. On
      the 40-layer model, 2 layers per executable cut the 84 children to 44; in
      the A/B that chose it (32K tokens per rank, one tpu7x-2x2x1 host) that had
      24% shorter step time (+31% throughput) than 1.
    memory_policy: GraphTrainer activation checkpointing policy (`"full"` or
      `"tpu_selective"`).
    save_regions: Region names kept alive across forward/backward executables
      when `memory_policy == "tpu_selective"`.

  Returns:
    A `TpuGraphTrainer.Config` configured for ZeRO-3 graph splitting.
  """
  # Imported lazily because `qwen3_5.__init__` re-exports `config_registry`
  # while `graph_trainer.qwen3_5` imports `qwen3_5`.
  # pylint: disable=g-import-not-at-top
  from torchtitan.experiments.tpu.graph_trainer.qwen3_5 import (
      model_registry as graph_trainer_model_registry,
  )
  # pylint: enable=g-import-not-at-top

  cfg = TpuGraphTrainer.Config.from_tpu_trainer_config(
      base_cfg, graph_trainer_model_registry
  )
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  cfg.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
      mode="aot_fx_trace",
      memory_policy=memory_policy,
      save_regions=save_regions,
      optimizer_compile="eager",
      compile_grad_norm=True,
      enable_remat_barriers=False,
      fsdp_barrier_policy="none",
      graph_split=GraphSplitterConfig(
          enabled=True,
          forward_layers_per_executable=layers_per_executable,
          forward_isolate_prefix=True,
          backward_partition_mode="fsdp_pipeline",
          backward_layers_per_executable=layers_per_executable,
      ),
  )
  # torch_tpu's fused AdamW updates every parameter in one program, computing
  # in FP32 and storing in each parameter buffer's dtype.
  cfg.optimizer.implementation = "fused"
  return cfg


def qwen35_debugmodel() -> TPUTrainerConfig:
  """Debug config for hybrid Qwen3.5 (dense) on TPU."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry("debugmodel", seq_len=256),
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * 256,
          max_context_length=256,
          steps=10,
      ),
      dataloader=c4_test_dataloader(),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
      ),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(preserve_rng_state=False),
      compile=CompileConfig(enable=False, backend="tpu"),
      validator=Validator.Config(enable=False),
      tpu_config=TPUConfig(enable_amp=False),
  )


def qwen35_moe_debugmodel() -> TPUTrainerConfig:
  """Debug config for 4-layer hybrid Qwen3.5 MoE on TPU."""
  cfg = qwen35_debugmodel()
  cfg.model_spec = model_registry(
      "debugmodel_moe", seq_len=256, converters=_TPU_MOE_CONVERTERS
  )
  return cfg


def qwen35_moe_debugmodel_fused_ce() -> TPUTrainerConfig:
  """Debug config for 4-layer hybrid Qwen3.5 MoE with Pallas fused CE loss."""
  cfg = qwen35_moe_debugmodel()
  cfg.training.max_context_length = 256
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 1024
  _enable_pallas_cross_entropy(
      cfg, b_block_size=1024, h_block_size=128, v_block_size=512
  )
  return cfg


def qwen35_moe_debugmodel_tpu_moe_kernels() -> TPUTrainerConfig:
  """Debug config for 4-layer hybrid Qwen3.5 MoE with the TPU MoE kernels.

  The routed experts of `qwen35_moe_35b` (`_use_tpu_moe_kernels`): Pallas
  paired grouped matmuls on kernel-native expert weights, and the fused
  route-score backward.
  """
  cfg = qwen35_moe_debugmodel()
  _use_tpu_moe_kernels(cfg)
  return cfg


def qwen35_moe_35b() -> TPUTrainerConfig:
  """Qwen3.5-35B-A3B (40 hybrid layers, 256 experts) on the regular trainer.

  As in the Qwen3 TPU recipes, FP32 parameters, gradients and AdamW state
  (16 bytes per parameter) with bf16 compute and FP32 gradient reduction.
  That is 65 GiB per rank on one tpu7x-2x2x1 host (4 chips, 8 ranks), which
  then runs out of HBM in the second step: use two or more hosts, or
  `--training.dtype=bfloat16` (8 bytes per parameter) on one. SimpleFSDP
  resharding after forward (ZeRO-3), full activation checkpointing, Pallas
  paired grouped matmuls for the experts, Splash attention, Pallas fused
  linear cross-entropy, and torch_tpu's DEFER_AND_FUSE eager mode with a
  compiled gradient norm. 32K tokens per rank (16 sequences of 2048).
  """
  cfg = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./assets/hf/Qwen3.5-35B-A3B",
      # Logging syncs the host on the loss; every 5 steps lets the host
      # dispatch the next step while the current one runs.
      metrics=MetricsProcessor.Config(log_freq=5),
      model_spec=model_registry("35B-A3B", converters=_TPU_MOE_CONVERTERS),
      optimizer=default_adamw(lr=3e-4, implementation="foreach"),
      # A 2-step warmup to the peak LR diverged the randomly initialized
      # 40-layer model (loss 12.77 -> 6.45 -> 8.48 -> NaN at step 4 on
      # tpu7x-2x2x1). Match the upstream Qwen3-30B-A3B schedule (LR 3e-4, 200
      # warmup steps); short benchmark runs clamp the warmup to
      # `training.steps`, as the Qwen3 TPU benchmark recipes do.
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=200,
          decay_ratio=0.8,
          decay_type="cosine",
          min_lr_factor=0.1,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=16 * 2048,
          max_context_length=2048,
          steps=10,
          mixed_precision_reduce="float32",
      ),
      dataloader=TPUGrainDataLoader.Config(dataset="c4"),
      parallelism=ParallelismConfig(
          expert_parallel_degree=1,
          fsdp_reshard_after_forward="always",
          use_simple_fsdp=True,
      ),
      comm=CommConfig(train_timeout_seconds=600),
      activation_checkpoint=FullAC.Config(preserve_rng_state=False),
      compile=TpuTrainerCompileConfig(
          enable=False, backend="tpu", compile_grad_norm=True
      ),
      validator=Validator.Config(enable=False),
      # Full attention in Qwen3.5 35B-A3B has head_dim=256 (2x Qwen3's 128):
      # Splash's backward overflows the default 32 MiB of scoped VMEM with
      # 2048 blocks, so use 1024.
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_block_q=1024,
          sa_block_kv=1024,
          sa_block_dkv=1024,
          sa_block_kv_compute=512,
          sa_block_q_dkv=1024,
          sa_block_kv_dkv=1024,
          sa_block_kv_dkv_compute=512,
          sa_block_q_dq=1024,
          sa_block_kv_dq=1024,
      ),
      # The largest scoped VMEM limit TPU7x accepts (65536 is rejected); the
      # default is 32 MiB.
      libtpu_init_flags=tpu_flags.merge_flags(
          {"xla_tpu_scoped_vmem_limit_kib": 65472}
      ),
      tpu_config=TPUConfig(
          enable_amp=False,
          eager_mode="DEFER_AND_FUSE",
      ),
  )
  _use_tpu_moe_kernels(cfg)
  _enable_pallas_cross_entropy(cfg)
  return cfg


def qwen35_moe_35b_split_tokamax_benchmark() -> TpuGraphTrainer.Config:
  """Qwen3.5-35B-A3B benchmark: ZeRO-3, graph split, Tokamax GatedDeltaNet.

  Runs on any TPU7x slice from one tpu7x-2x2x1 host (4 chips, 8 ranks) up to
  256 FSDP ranks (256 routed experts sharded on dim 0): bf16 parameters and
  AdamW state, ZeRO-3 with selective rematerialization, GraphTrainer with 2
  layers per forward and backward executable, Tokamax's Mosaic KDA kernels for
  the 30 GatedDeltaNet layers (without packed-document resets), Pallas paired
  grouped matmuls for the experts, Splash attention, and the Pallas fused
  linear cross-entropy. 32K tokens per rank (16 sequences of 2048).
  """
  base_cfg = qwen35_moe_35b()
  # bf16 parameters and AdamW state (8 bytes per parameter) instead of
  # qwen35_moe_35b's FP32 master weights, so that the benchmark fits one
  # tpu7x-2x2x1 host: with FP32 master weights its second step runs out of
  # HBM there (0.6 GiB short at 32K tokens per rank, 13 GiB at 64K).
  base_cfg.training.dtype = "bfloat16"
  base_cfg.dataloader = c4_test_dataloader()
  base_cfg.hf_assets_path = "./tests/assets/tokenizer"
  base_cfg.parallelism = ParallelismConfig(
      data_parallel_replicate_degree=1,
      data_parallel_shard_degree=-1,
      tensor_parallel_degree=1,
      pipeline_parallel_degree=1,
      context_parallel_degree=1,
      expert_parallel_degree=1,
      use_simple_fsdp=True,
      enable_tpu_topo_mesh=True,
  )
  # Match the Qwen3 GraphTrainer recipes: disable HLO sharding v3 (Shardy) on
  # partitioned child executables.
  base_cfg.xla_flags = tpu_flags.merge_flags(
      xla_enable_hlo_sharding_v3=False,
  )
  base_cfg.tpu_config = TPUConfig(
      enable_amp=False,
      enable_mfu_benchmark=False,
  )
  _use_tokamax_gated_delta(base_cfg)
  _no_gated_delta_document_resets(base_cfg)
  # Save each layer's MoE router top-k / sort permutation and the 10 full
  # attention layers' Splash output + LSE (~5.98 GiB logical at 64K tokens/rank)
  # so backward skips re-running router top-k/argsort and Splash forward.
  cfg = _use_graph_split(
      base_cfg,
      layers_per_executable=2,
      memory_policy="tpu_selective",
      save_regions=("moe.routing", "attention.inner_attention"),
  )
  # Throughput DVFS operating point (p_state=7) and SparseCore collective
  # offload (async AG, dual-SC RS, latency-hiding scheduler), merged onto
  # qwen35_moe_35b's 65472 KiB scoped VMEM limit.
  cfg.libtpu_init_flags = tpu_flags.merge_flags(
      cfg.libtpu_init_flags,
      tpu_flags.TPU7X_THROUGHPUT_P_STATE,
      tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
      tpu_flags.SPARSECORE_ASYNC_ALL_GATHER,
      tpu_flags.DUAL_SPARSECORE_REDUCE_SCATTER,
  )
  return cfg
