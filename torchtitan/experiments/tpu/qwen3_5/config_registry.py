"""TPU configuration registry for Qwen3.5 (Dense and MoE) models."""

import dataclasses
from typing import Literal, TypeVar

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
from torchtitan.config.transform import ContextParallelTransform, apply_transforms
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.components.datasets import (
    TPUGrainDataLoader,
    c4_test_dataloader,
)
from torchtitan.experiments.tpu.distributed.cp_attention import (
    tpu_cp_inner_attention,
    tpu_cp_load_balancer,
)
from torchtitan.experiments.tpu.graph_trainer.config import (
    GraphSplitterConfig,
    TpuGraphTrainerCompileConfig,
    TpuGraphTrainerMemoryPolicy,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.components import expert_parallel
from torchtitan.experiments.tpu.components.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.components.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.qwen3_5 import model_registry
from torchtitan.experiments.tpu.qwen3_5.cp_gated_delta import GatedDeltaNetCPStrategy, GatedDeltaNetContextParallelTransform, ReduceDtype
from torchtitan.experiments.tpu.qwen3_5.gated_delta import TpuGatedDeltaNet
from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
    TpuTrainerCompileConfig,
)
from torchtitan.models.common.attention import BaseAttention

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


def _use_tpu_moe_kernels(cfg: TPUTrainerConfig, *, tile_m: int = 256) -> None:
  """Selects the Qwen3-30B-A3B TPU recipe's MoE kernels.

  Pallas paired grouped matmuls with kernel-native expert weights, and dScore
  in one fused FP32 reduction instead of saving an FP32 copy of every expert
  output. These are per-model-config settings, so a recipe that replaces
  `model_spec` must apply them again.

  Args:
    cfg: The recipe to update in place.
    tile_m: Forward and dWeight row tile size for the Pallas grouped matmuls.
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
    experts_config.pallas_grouped_mm_tile_m = tile_m
    experts_config.pallas_grouped_mm_tgmm_tiling = (tile_m, 0, 0)
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


def _use_expert_parallel(
    cfg: TpuGraphTrainer.Config,
    *,
    ep_degree: int,
    num_token_chunks: int | None = None,
) -> None:
  """Turns on expert parallelism (EP) in a split Qwen3.5 MoE recipe.

  `TpuQwen35Model.Config.update_from_config` passes
  `parallelism.expert_parallel_degree` and the `tpu_config.ep_*` fields set
  here to `use_expert_parallel`: `ep_transport="auto"` (Ring-of-Experts while
  `ep_degree <= top_k`, the ragged all-to-all above that), routing weights
  applied before the down projection, and `num_token_chunks` sequence chunks
  per MoE layer with at most two in flight. Chunking divides each chunk's
  dropless `[ep_degree * T / C * top_k, D]` expert buffer by C and lets one
  chunk's token exchange overlap another chunk's expert matmuls.

  It also changes the rest of the recipe:

  - The gradient norm is eager: the compiled one needs a single mesh, and the
    experts live on the `(efsdp, ep)` mesh. `DEFER_AND_FUSE` fuses the eager
    norm across parameters.
  - One layer per forward and backward executable, with
    `backward_partition_mode="by_layer"`: `fsdp_pipeline` treats every
    backward all-gather and reduce-scatter as an FSDP collective, and the EP
    token exchange isn't one.
  - The topology-aware device mesh stays on (inherited from the base
    recipe): `ParallelDims.build_mesh` registers each mesh axis's logical
    layout with `spmd_types`, so the `efsdp` and `ep` axes pass its
    orthogonality check when the mesh reorders the devices.
    `--parallelism.no-enable_tpu_topo_mesh` turns it off.
  - SparseCore offload for the EP and expert-FSDP collectives, with gathers
    and copies kept on the TensorCore (the `tpu_flags` bundles below).

  `--parallelism.expert_parallel_degree` on the command line changes the EP
  degree but not the chunk count: pass `--tpu_config.ep_token_chunks` with it.

  Args:
    cfg: The recipe to update in place.
    ep_degree: The expert-parallel degree. It must divide the number of routed
      experts and the number of FSDP ranks.
    num_token_chunks: Sequence chunks per MoE layer, or `None` for
      `default_num_token_chunks(ep_degree)` (`ep_degree` clamped to 4..16).
  """
  cfg.parallelism.expert_parallel_degree = ep_degree
  cfg.compile.compile_grad_norm = False
  if not cfg.tpu_config.eager_mode:
    cfg.tpu_config.eager_mode = "DEFER_AND_FUSE"
  cfg.tpu_config.ep_transport = "auto"
  cfg.tpu_config.ep_scale_before_down_proj = True
  cfg.tpu_config.ep_token_chunks = (
      expert_parallel.default_num_token_chunks(ep_degree)
      if num_token_chunks is None
      else num_token_chunks
  )
  graph_split = cfg.compile.graph_split
  graph_split.forward_layers_per_executable = 1
  graph_split.backward_layers_per_executable = 1
  graph_split.backward_partition_mode = "by_layer"
  cfg.libtpu_init_flags = tpu_flags.merge_flags(
      cfg.libtpu_init_flags,
      tpu_flags.SPARSECORE_COLLECTIVE_OFFLOAD,
      tpu_flags.SPARSECORE_REDUCE_SCATTER_V2,
      tpu_flags.LAYER_SCHEDULER_FOR_DEPENDENT_COLLECTIVES,
      tpu_flags.EP_GATHER_COPY_ON_TENSORCORE,
  )


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


def qwen35_moe_debugmodel_pallas_gmm() -> TPUTrainerConfig:
  """`qwen35_moe_debugmodel` with unpaired Pallas grouped matmuls.

  Canonical weight layout and 256-row tiles, and nothing else from
  `_use_tpu_moe_kernels`. EP tests that apply routing weights before W2
  (`--tpu_config.ep_scale_before_down_proj`) use it, since only the Pallas
  grouped matmuls take per-row scales.
  """
  cfg = qwen35_moe_debugmodel()
  assert cfg.model_spec is not None
  for _, experts, _, _ in cfg.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.use_pallas_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 256
    experts.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)
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


def qwen35_moe_debugmodel_pallas_experts() -> TPUTrainerConfig:
  """Debug Qwen3.5 MoE with paired Pallas grouped matmuls on standard weights.

  `qwen35_moe_debugmodel_pallas_gmm` with Pallas paired W1+W3 (and unpaired
  W2) grouped matmuls, with the expert weights in the standard `(E, out, in)`
  layout, so a CPU reference that runs `torch._grouped_mm` on the same
  parameters can check loss, gradients and parameters
  (`test_ep_parity_distributed`).
  """
  cfg = qwen35_moe_debugmodel_pallas_gmm()
  assert cfg.model_spec is not None
  for _, experts, _, _ in cfg.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.use_pallas_paired_grouped_mm = True
  return cfg


def graph_trainer_qwen35_moe_debugmodel() -> TpuGraphTrainer.Config:
  """`qwen35_moe_debugmodel` on TpuGraphTrainer, split as the 35B recipes are.

  `_use_graph_split` with one layer per forward and backward executable and
  `tpu_selective` rematerialization saving the MoE routing (the 35B benchmark
  also saves the Splash attention output, which the debug model's SDPA does
  not produce). Runs on any TPU: no Tokamax GatedDeltaNet.
  """
  return _use_graph_split(
      qwen35_moe_debugmodel(),
      layers_per_executable=1,
      memory_policy="tpu_selective",
      save_regions=("moe.routing",),
  )


def graph_trainer_qwen35_moe_debugmodel_tpu_moe_kernels() -> (
    TpuGraphTrainer.Config
):
  """`graph_trainer_qwen35_moe_debugmodel` with the 35B/397B TPU MoE kernels.

  Paired Pallas grouped matmuls on kernel-native expert weights and the fused
  route-score backward (`_use_tpu_moe_kernels`), as the split benchmark
  recipes run them, for GraphTrainer EP-vs-EP=1 tests.
  """
  cfg = graph_trainer_qwen35_moe_debugmodel()
  _use_tpu_moe_kernels(cfg)
  return cfg


def _debugmodel_moe_8l16e(cfg: TPUTrainerConfig) -> None:
  """Gives `cfg` the debug MoE model at 8 layers, 16 experts, top-4.

  Two cycles of period 4 (3 GatedDeltaNet + 1 full attention), so a period-4
  scan loops twice, and 16 experts split over EP = 2, 4 or 8 on 8 chips with
  `efsdp = 8 / EP`. Everything else is `debugmodel_moe`'s.
  """
  # pylint: disable-next=g-import-not-at-top
  import torchtitan.models.qwen3_5 as upstream_qwen3_5
  from torchtitan.experiments.tpu.qwen3_5 import to_tpu_model_spec  # pylint: disable=g-import-not-at-top

  spec = upstream_qwen3_5.model_registry(
      "debugmodel_moe", seq_len=256, attn_backend="sdpa"
  )
  rope = next(l.attention.rope for l in spec.model.layers if l.attention)
  # pylint: disable-next=protected-access
  spec.model.layers = upstream_qwen3_5._build_qwen35_moe_layers(
      rope=rope, attn_backend="sdpa", n_layers=8, dim=256, n_heads=4,
      n_kv_heads=2, head_dim=64, rotary_dim=16, moe_hidden_dim=256,
      num_experts=16, top_k=4, shared_expert_hidden_dim=256, n_key_heads=2,
      n_value_heads=4, key_head_dim=128, value_head_dim=128,
  )
  model = spec.model
  for converter in _TPU_MOE_CONVERTERS:
    model = converter.build().convert(model)
  cfg.model_spec = to_tpu_model_spec(dataclasses.replace(spec, model=model))


def graph_trainer_qwen35_moe_debugmodel_ep() -> TpuGraphTrainer.Config:
  """Split (one layer per executable) debug MoE, 8 layers, 16 experts, EP = 2.

  The scan + EP proxy's reference lane. `auto` transport: Ring-of-Experts while
  EP <= top_k = 4, ragged all-to-all at EP = 8. Two token chunks, the most a
  scan body allows (`check_scan_expert_parallel`).
  """
  base = qwen35_moe_debugmodel()
  _debugmodel_moe_8l16e(base)
  # Paired Pallas grouped matmuls: EP's routing weights before W2 need them.
  _use_tpu_moe_kernels(base)
  cfg = _use_graph_split(
      base,
      layers_per_executable=1,
      memory_policy="tpu_selective",
      save_regions=("moe.routing",),
  )
  _use_expert_parallel(cfg, ep_degree=2, num_token_chunks=2)
  return cfg


def graph_trainer_qwen35_moe_debugmodel_scan_ep() -> TpuGraphTrainer.Config:
  """`graph_trainer_qwen35_moe_debugmodel_ep` as a period-4 scan (2 cycles)."""
  cfg = graph_trainer_qwen35_moe_debugmodel_ep()
  cfg.compile.graph_split = GraphSplitterConfig(enabled=False)
  cfg.compile.use_scan_layers = True
  cfg.compile.scan_period = 4
  cfg.compile.memory_policy = "full"
  cfg.compile.save_regions = ()
  cfg.override.imports.append(
      "torchtitan.experiments.tpu.overrides.discarded_token_counter"
      ".tpu_discarded_token_counter"
  )
  return cfg


def qwen35_moe_35b(flavor: str = "35B-A3B") -> TPUTrainerConfig:
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

  Args:
    flavor: Model flavor. The 397B-A17B recipe reuses this one with
      `"397B-A17B"` and then overrides the batch, dtypes and parallelism.
  """
  cfg = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path=f"./assets/hf/Qwen3.5-{flavor}",
      # Logging syncs the host on the loss; every 5 steps lets the host
      # dispatch the next step while the current one runs.
      metrics=MetricsProcessor.Config(log_freq=5),
      model_spec=model_registry(flavor, converters=_TPU_MOE_CONVERTERS),
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


def qwen35_moe_35b_split_tokamax_benchmark(
    flavor: str = "35B-A3B",
) -> TpuGraphTrainer.Config:
  """Qwen3.5-35B-A3B benchmark: ZeRO-3, graph split, Tokamax GatedDeltaNet.

  Runs on any TPU7x slice from tpu7x-2x2x2 (2 hosts, 8 chips, 16 ranks) up to
  256 FSDP ranks (256 routed experts sharded on dim 0; pass
  `--training.dtype=bfloat16` to fit one tpu7x-2x2x1 host): FP32 master
  weights, gradients and AdamW state with bf16 compute, ZeRO-3 with selective
  rematerialization, GraphTrainer with 2 layers per forward and backward
  executable, Tokamax's Mosaic KDA kernels for the 30 GatedDeltaNet layers
  (without packed-document resets), Pallas paired grouped matmuls for the
  experts, Splash attention, and the Pallas fused linear cross-entropy. 32K
  tokens per rank (16 sequences of 2048).

  Args:
    flavor: Model flavor. The 397B-A17B EP recipe reuses this one with
      `"397B-A17B"` (60 layers, 512 experts) and then overrides the batch,
      dtypes and parallelism.
  """
  base_cfg = qwen35_moe_35b(flavor)
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
  # One hidden block (H=2048) keeps every x tile resident in the fused LCE
  # backward; v=640 divides the 248320 vocabulary (388 blocks). Needs bf16
  # activations: on TPU7x the backward fits the kernel's VMEM limit (90% of 64
  # MiB) only single-buffered (~51.9 MiB; fp32 hidden states would need ~69.9
  # MiB), which `linear_softmax_cross_entropy_loss` selects. On one TPU7x
  # core at 64K tokens (measured with 8 x 8192 tokens/rank): 564 vs 633 ms
  # forward + backward with (1024, 512, 2048).
  _enable_pallas_cross_entropy(
      base_cfg, b_block_size=1024, h_block_size=2048, v_block_size=640
  )
  # Save each layer's MoE router top-k / sort permutation and the 10 full
  # attention layers' Splash output + LSE (~5.98 GiB logical at 64K tokens/rank)
  # so backward skips re-running router top-k/argsort and Splash forward.
  cfg = _use_graph_split(
      base_cfg,
      layers_per_executable=2,
      memory_policy="tpu_selective",
      save_regions=("moe.routing", "attention.inner_attention"),
  )
  # 512-row expert tiles, measured at EP=1 with 64K tokens per rank (8 x 8192,
  # ~2K routed rows per expert on average): the routed-expert MLP's forward +
  # backward takes 22.9 vs 26.8 ms per layer on one TPU7x core with the
  # Qwen3-30B tiles (256). The 32K-token default and the EP recipes inherit
  # this tile.
  _use_tpu_moe_kernels(cfg, tile_m=512)
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


def qwen35_moe_35b_scan_tokamax_benchmark() -> TpuGraphTrainer.Config:
  """`qwen35_moe_35b_split_tokamax_benchmark` on scan instead of the graph split.

  Each scan step runs one 4-layer cycle (3 GatedDeltaNet + 1 full attention).
  Gradients reduce in float32, as in `qwen35_moe_35b`.
  """
  cfg = qwen35_moe_35b_split_tokamax_benchmark()
  cfg.compile.graph_split = GraphSplitterConfig(enabled=False)
  cfg.compile.use_scan_layers = True
  cfg.compile.scan_period = 4
  # The scan body recomputes everything in backward (`scan_body_remat_window`):
  # `tpu_selective` rejects scan, and save regions do not reach the body.
  cfg.compile.memory_policy = "full"
  cfg.compile.save_regions = ()
  # MoE.forward adds to a per-layer counter that nothing reads (the TPU
  # registry turns load balancing off); a scan body cannot write it.
  cfg.override.imports.append(
      "torchtitan.experiments.tpu.overrides.discarded_token_counter"
      ".tpu_discarded_token_counter"
  )
  return cfg


def qwen35_moe_35b_split_tokamax_ep_benchmark() -> TpuGraphTrainer.Config:
  """`qwen35_moe_35b_split_tokamax_benchmark` with expert parallelism (EP = 4).

  `_use_expert_parallel` at EP = 4: Ring-of-Experts (EP <= top_k = 8) in 4
  token chunks per MoE layer. Runs on tpu7x-2x2x2 (16 ranks) and larger
  slices. Pass `--parallelism.expert_parallel_degree` together with
  `--tpu_config.ep_token_chunks` for another EP degree.
  """
  cfg = qwen35_moe_35b_split_tokamax_benchmark()
  _use_expert_parallel(cfg, ep_degree=4)
  return cfg


def qwen35_moe_397b_split_tokamax_ep_benchmark() -> TpuGraphTrainer.Config:
  """Qwen3.5-397B-A17B 8K-context benchmark with expert parallelism (EP = 8).

  `qwen35_moe_35b_split_tokamax_benchmark` for the 60-layer, 512-expert,
  top-10 397B-A17B model on tpu7x-2x4x4 (32 chips, 64 ranks) and larger
  slices, with `_use_expert_parallel` at EP = 8 (Ring-of-Experts):

  - BF16 parameters, gradients and AdamW state (8 bytes per parameter, about
    46 GiB per rank on 64 ranks, where FP32 would need about 92 GiB) and a
    BF16 gradient reduce-scatter.
  - 24,576 tokens per rank (3 sequences of 8192), in 12 token chunks of 2,048
    per MoE layer.
  - The default fused cross-entropy tiles (the 35B recipe's assume one
    H = 2048 hidden block) and K-tiled expert dWeight matmuls that fit VMEM at
    D = 4096, F = 1024.

  `--parallelism.expert_parallel_degree=16 --tpu_config.ep_token_chunks=16`
  switches to the ragged all-to-all (EP > top_k), and
  `--parallelism.expert_parallel_degree=1` runs the same split and
  rematerialization without EP.
  """
  cfg = qwen35_moe_35b_split_tokamax_benchmark("397B-A17B")
  cfg.training.dtype = "bfloat16"
  cfg.training.mixed_precision_param = "bfloat16"
  cfg.training.mixed_precision_reduce = "bfloat16"
  cfg.training.max_context_length = 8192
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 3 * 8192
  _enable_pallas_cross_entropy(cfg)
  for _, experts, _, _ in cfg.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    # Untiled, the paired W1 + W3 dWeight needs two FP32 (K, N) accumulators
    # plus double-buffered BF16 slabs, 86 MiB of VMEM at D = 4096, F = 1024:
    # tile K at 1024. W2's K (F = 1024) needs no tiling; tile D at 2048 when W2
    # is transposed.
    tile_m = experts.pallas_grouped_mm_tile_m
    experts.pallas_grouped_mm_tgmm_tiling = (tile_m, 1024, 0)
    experts.pallas_grouped_mm_w2_tgmm_tiling = (tile_m, 0, 0)
    experts.pallas_grouped_mm_w2_transposed_tgmm_tiling = (tile_m, 0, 2048)
  _use_expert_parallel(cfg, ep_degree=8, num_token_chunks=12)
  return cfg


# A TPUTrainerConfig or a subclass (e.g. TpuGraphTrainer.Config).
_TPUConfigT = TypeVar("_TPUConfigT", bound=TPUTrainerConfig)


def _qwen35_cp(
    config: _TPUConfigT,
    *,
    strategy: Literal["ulysses", "all_gather"],
    cp_degree: int,
    gated_delta_cp_strategy: GatedDeltaNetCPStrategy = "all_to_all",
    gated_delta_cp_reduce_dtype: ReduceDtype = "float32",
) -> _TPUConfigT:
  """Enables TPU context parallelism for hybrid Qwen3.5.

  As ``qwen3/config_registry.py::_qwen3_cp``, sets the CP degree and the load
  balancer of ``strategy`` and applies core ``ContextParallelTransform`` with
  the TPU CP inner attention to the full-attention layers. Also applies
  ``GatedDeltaNetContextParallelTransform``, which runs the GatedDeltaNet
  layers head-parallel over the same token layout (``cp_gated_delta``):
  ``n_key_heads`` and ``n_value_heads`` must be divisible by ``cp_degree``.

  Args:
    config: The recipe to enable CP on.
    strategy: The full-attention CP strategy, ``"all_gather"`` (load-balanced KV
      all-gather) or ``"ulysses"``.
    cp_degree: The context-parallel degree.
    gated_delta_cp_strategy: ``GatedDeltaNetContextParallelTransform``'s
      ``cp_strategy``.
    gated_delta_cp_reduce_dtype: Its ``cp_reduce_dtype``.

  Returns:
    A validated copy of ``config`` with CP enabled.
  """
  assert config.model_spec is not None
  load_balancer = tpu_cp_load_balancer(strategy)
  config.parallelism.context_parallel_degree = cp_degree
  config.parallelism.context_parallel_load_balancer = load_balancer
  inner_attention = config.model_spec.model.first_full_attention_backend
  assert inner_attention is not None
  return apply_transforms(
      config,
      [
          ContextParallelTransform(
              inner_attention=tpu_cp_inner_attention(
                  inner_attention._owner, strategy
              )
          ),
          GatedDeltaNetContextParallelTransform(
              load_balancer=load_balancer,
              cp_strategy=gated_delta_cp_strategy,
              cp_reduce_dtype=gated_delta_cp_reduce_dtype,
          ),
      ],
  )


def _allow_reference_attention(cfg: TPUTrainerConfig) -> None:
  """Lets KV all-gather CP run its plain-PyTorch attention (no Splash)."""
  assert cfg.model_spec is not None
  for _, attn, _, _ in cfg.model_spec.model.traverse(BaseAttention.Config):
    attn.inner_attention.allow_reference_attention = True


def qwen35_debugmodel_allgather_cp2() -> TPUTrainerConfig:
  """Qwen3.5 debug model with KV all-gather CP degree 2 (head-parallel GDN).

  The debug model runs without the Splash kernel, so KV all-gather CP opts in
  to its plain-PyTorch reference attention, which is fine at this size.
  """
  cfg = _qwen35_cp(qwen35_debugmodel(), strategy="all_gather", cp_degree=2)
  _allow_reference_attention(cfg)
  return cfg


def qwen35_debugmodel_ulysses_cp2() -> TPUTrainerConfig:
  """Qwen3.5 debug model with Ulysses CP degree 2 (head-parallel GDN)."""
  return _qwen35_cp(qwen35_debugmodel(), strategy="ulysses", cp_degree=2)


def qwen35_debugmodel_ulysses_cp2_agrs() -> TPUTrainerConfig:
  """``qwen35_debugmodel_ulysses_cp2``, ``allgather_reducescatter`` GDN CP."""
  return _qwen35_cp(
      qwen35_debugmodel(),
      strategy="ulysses",
      cp_degree=2,
      gated_delta_cp_strategy="allgather_reducescatter",
  )


def qwen35_moe_debugmodel_allgather_cp2() -> TPUTrainerConfig:
  """Qwen3.5 MoE debug model with KV all-gather CP degree 2."""
  cfg = _qwen35_cp(qwen35_moe_debugmodel(), strategy="all_gather", cp_degree=2)
  _allow_reference_attention(cfg)
  return cfg


def qwen35_moe_debugmodel_ulysses_cp2() -> TPUTrainerConfig:
  """Qwen3.5 MoE debug model with Ulysses CP degree 2."""
  return _qwen35_cp(qwen35_moe_debugmodel(), strategy="ulysses", cp_degree=2)


def _split_tokamax_benchmark_for_cp(cp: int) -> TpuGraphTrainer.Config:
  """``qwen35_moe_35b_split_tokamax_benchmark`` sized for context parallelism.

  32K-token sequences and 64K tokens per device (a CP group shares one
  data-parallel rank's tokens, so the per-rank budget is scaled by ``cp``),
  where CP pays off; Splash's document-aware block skipping is off, as KV
  all-gather CP cannot use it and at long context it costs more than it saves.
  Longer contexts set ``--training.max_context_length`` (and, from 64K,
  ``--splash_attention_kernel.no-sa_use_fused_bwd_kernel``).
  """
  cfg = qwen35_moe_35b_split_tokamax_benchmark()
  cfg.training.max_context_length = 32768
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 65536 * cp
  cfg.splash_attention_kernel.sa_use_document_aware_block_skipping = False
  return cfg


def qwen35_moe_35b_split_tokamax_benchmark_allgather_cp2() -> (
    TpuGraphTrainer.Config
):
  """``qwen35_moe_35b_split_tokamax_benchmark`` with KV all-gather CP 2."""
  return _qwen35_cp(
      _split_tokamax_benchmark_for_cp(2),
      strategy="all_gather",
      cp_degree=2,
  )


def qwen35_moe_35b_split_tokamax_benchmark_ulysses_cp2() -> (
    TpuGraphTrainer.Config
):
  """``qwen35_moe_35b_split_tokamax_benchmark`` with Ulysses CP degree 2."""
  return _qwen35_cp(
      _split_tokamax_benchmark_for_cp(2),
      strategy="ulysses",
      cp_degree=2,
  )


def qwen35_moe_35b_split_tokamax_benchmark_ulysses_cp2_agrs() -> (
    TpuGraphTrainer.Config
):
  """Ulysses CP 2 with the ``allgather_reducescatter`` GatedDeltaNet strategy.

  Reduces in bfloat16. Faster than ``all_to_all`` at long context on CP2, for
  more HBM (see ``CPTpuGatedDeltaNet``).
  """
  return _qwen35_cp(
      _split_tokamax_benchmark_for_cp(2),
      strategy="ulysses",
      cp_degree=2,
      gated_delta_cp_strategy="allgather_reducescatter",
      gated_delta_cp_reduce_dtype="bfloat16",
  )
