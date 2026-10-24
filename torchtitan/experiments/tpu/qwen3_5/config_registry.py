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
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.qwen3_5 import model_registry
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
)

_TPU_MOE_CONVERTERS = [
    TpuTokenDispatcherConverter.Config(),
    TpuGroupedExpertsConverter.Config(),
]

FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"
)
# The override is imported only when the recipe is built: the module needs a
# Tokamax build with the KDA op.
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
  for _, experts, _, _ in cfg.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.use_pallas_grouped_mm = True
    experts.use_pallas_paired_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 256
    experts.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)
    experts.use_kernel_native_weight_layout = True
  for _, dispatcher, _, _ in cfg.model_spec.model.traverse(
      TpuAllToAllTokenDispatcher.Config
  ):
    dispatcher.fuse_route_score_backward = True


def _use_tokamax_gated_delta(cfg: TPUTrainerConfig) -> None:
  """Runs every GatedDeltaNet with Tokamax's Mosaic TPU KDA kernels."""
  if TOKAMAX_GATED_DELTA_OVERRIDE not in cfg.override.imports:
    cfg.override.imports.append(TOKAMAX_GATED_DELTA_OVERRIDE)


def _use_graph_split(
    cfg: TPUTrainerConfig, *, layers_per_executable: int
) -> None:
  """Runs the ZeRO-3 step as per-layer executables (GraphTrainer split).

  As one unrolled step graph, the 40-layer model neither compiles within an
  hour on 8 host CPUs nor fits in HBM on one tpu7x-2x2x1 host: XLA keeps most
  layers' unsharded expert gradients (3 x 512 MiB bf16 per layer) alive until
  late in the step. Per-layer forward and backward executables bound both:
  each backward child reduce-scatters its own layers' gradients, and repeated
  layers hit the compilation cache. As in the Qwen3-235B recipes,
  `fsdp_pipeline` issues each child's all-gather one child early and its
  reduce-scatter one child late, and AdamW runs outside the step graph as
  torch_tpu's fused AdamW (graph splitting does not support
  `optimizer_in_graph`).

  Args:
    cfg: The recipe to update in place.
    layers_per_executable: Transformer layers per forward and backward child. On
      the 40-layer model at 32K tokens per rank, 2 layers per executable cut the
      84-child launch overhead to 44 children: 3.51 vs 4.65 s/step.
  """
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  cfg.tpu_config.graph_trainer_memory_policy = "full"
  cfg.tpu_config.graph_trainer_save_regions = ()
  cfg.tpu_config.enable_graph_trainer_remat_schedule = False
  cfg.tpu_config.graph_trainer_fsdp_schedule = "none"
  cfg.tpu_config.use_graph_split = True
  graph_split = cfg.tpu_config.graph_split
  graph_split.forward_layers_per_executable = layers_per_executable
  graph_split.forward_isolate_prefix = True
  graph_split.backward_partition_mode = "fsdp_pipeline"
  graph_split.backward_layers_per_executable = layers_per_executable
  cfg.tpu_config.optimizer_in_graph = False
  cfg.tpu_config.compile_optimizer = False
  cfg.tpu_config.compile_grad_norm = True
  # torch_tpu's fused AdamW updates every parameter in one program, computing
  # in FP32 and storing in each buffer's (here BF16) dtype.
  cfg.optimizer.implementation = "fused"


def _use_expert_parallel(cfg: TPUTrainerConfig, *, ep_degree: int) -> None:
  """Shards the routed experts over `ep_degree` ranks (Expert Parallelism).

  `TpuQwen35Model.Config.update_from_config` switches the token dispatcher to
  `TpuQwen35TokenDispatcher` (static-shape all-to-all with a per-rank token
  capacity) and the expert matmuls to the unpaired Pallas kernels.

  Args:
    cfg: The recipe to update in place.
    ep_degree: Expert-parallel degree; it must divide the number of experts.
  """
  cfg.parallelism.expert_parallel_degree = ep_degree
  # `CompiledGradNorm` needs a single mesh; the eager norm reduces over the
  # dense (`fsdp`) and expert (`efsdp`, `ep`) meshes.
  cfg.tpu_config.compile_grad_norm = False
  if cfg.tpu_config.use_graph_split:
    cfg.tpu_config.graph_trainer_memory_policy = "full"
    cfg.tpu_config.graph_trainer_save_regions = ()
    # Under `fsdp_pipeline`, backward child `L` holds the all-gather of layer
    # `L - 1` and the reduce-scatter of layer `L + 1`, so the 3:1
    # GatedDeltaNet:Attention cycle yields 6 distinct backward children, each
    # with the EP all-to-alls (~9 min to compile each on 8 CPUs). `by_layer`
    # children with one layer each are self-contained: 3 distinct backward
    # children.
    graph_split = cfg.tpu_config.graph_split
    graph_split.forward_layers_per_executable = 1
    graph_split.backward_layers_per_executable = 1
    graph_split.backward_partition_mode = "by_layer"


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
  cfg.training.dtype = "bfloat16"
  cfg.training.mixed_precision_param = "bfloat16"
  cfg.training.mixed_precision_reduce = "bfloat16"
  cfg.training.max_context_length = 256
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 1024
  _enable_pallas_cross_entropy(
      cfg, b_block_size=1024, h_block_size=128, v_block_size=512
  )
  return cfg


def qwen35_moe_35b() -> TPUTrainerConfig:
  """Qwen3.5-35B-A3B (40 hybrid layers, 256 experts) on the regular trainer.

  Fits one tpu7x-2x2x1 host (4 chips, 8 ranks) and up: bf16 parameters,
  gradients and AdamW state (8 bytes per parameter), SimpleFSDP resharding
  after forward (ZeRO-3), full activation checkpointing, Pallas paired grouped
  matmuls for the experts, Splash attention, Pallas fused linear cross-entropy,
  and torch_tpu's DEFER_AND_FUSE eager mode with a compiled gradient norm.
  32K tokens per rank (16 sequences of 2048).
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
          dtype="bfloat16",
          mixed_precision_param="bfloat16",
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
      compile=CompileConfig(enable=False, backend="tpu"),
      validator=Validator.Config(enable=False),
      # Full attention in Qwen3.5 35B-A3B has head_dim=256 (2x Qwen3's 128), so
      # 1024 block sizes keep scoped VMEM well below the 33 MiB hardware limit.
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
      libtpu_init_flags={
          "xla_tpu_scoped_vmem_limit_kib": "65472",
      },
      tpu_config=TPUConfig(
          enable_amp=False,
          eager_mode="DEFER_AND_FUSE",
          compile_grad_norm=True,
      ),
  )
  _use_tpu_moe_kernels(cfg)
  _enable_pallas_cross_entropy(cfg)
  return cfg


def qwen35_moe_35b_split_tokamax_benchmark() -> TPUTrainerConfig:
  """Qwen3.5-35B-A3B benchmark: ZeRO-3, graph split, Tokamax GatedDeltaNet.

  Runs on any TPU7x slice from one tpu7x-2x2x1 host (4 chips, 8 ranks) up:
  bf16 parameters and AdamW state, ZeRO-3 with selective rematerialization,
  GraphTrainer with 2 layers per forward and backward executable, Tokamax's
  Mosaic KDA kernels for the 30 GatedDeltaNet layers, Pallas paired grouped
  matmuls for the experts, Splash attention, and the Pallas fused linear
  cross-entropy. 32K tokens per rank (16 sequences of 2048).
  """
  cfg = qwen35_moe_35b()
  cfg.dataloader = c4_test_dataloader()
  cfg.hf_assets_path = "./tests/assets/tokenizer"
  cfg.parallelism = ParallelismConfig(
      data_parallel_replicate_degree=1,
      data_parallel_shard_degree=-1,
      tensor_parallel_degree=1,
      pipeline_parallel_degree=1,
      context_parallel_degree=1,
      expert_parallel_degree=1,
      use_simple_fsdp=True,
      enable_tpu_topo_mesh=True,
  )
  # Log every 5 steps, as the Qwen3 TPU benchmark recipes do: logging syncs the
  # host on the loss, and syncing every step keeps the host from dispatching
  # step N+1 while step N runs.
  cfg.metrics.log_freq = 5
  cfg.xla_flags = tpu_flags.merge_flags(
      xla_enable_hlo_sharding_v3=False,
  )
  cfg.compile = CompileConfig(
      enable=True,
      backend="aot_eager",
  )
  cfg.tpu_config = TPUConfig(
      enable_amp=False,
      compile_mode="aot_fx_trace",
      use_graph_trainer=True,
      enable_mfu_benchmark=False,
  )
  _use_graph_split(cfg, layers_per_executable=2)
  # Save each layer's MoE router top-k / sort permutation and the 10 full
  # attention layers' Splash output + LSE (~5.98 GiB logical at 64K tokens/rank)
  # so backward skips re-running router top-k/argsort and Splash forward.
  cfg.tpu_config.graph_trainer_memory_policy = "tpu_selective"
  cfg.tpu_config.graph_trainer_save_regions = (
      "moe.routing",
      "attention.inner_attention",
  )
  _use_tokamax_gated_delta(cfg)
  cfg.libtpu_init_flags = tpu_flags.merge_flags(
      cfg.libtpu_init_flags,
      tpu_flags.TPU7X_THROUGHPUT_P_STATE,
      tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
      tpu_flags.SPARSECORE_ASYNC_ALL_GATHER,
      tpu_flags.DUAL_SPARSECORE_REDUCE_SCATTER,
      {
          "xla_tpu_scoped_vmem_limit_kib": 65472,
      },
  )
  return cfg


def qwen35_moe_35b_split_tokamax_ep_benchmark() -> TPUTrainerConfig:
  """`qwen35_moe_35b_split_tokamax_benchmark` with Expert Parallelism (EP=4)."""
  cfg = qwen35_moe_35b_split_tokamax_benchmark()
  _use_expert_parallel(cfg, ep_degree=4)
  return cfg
