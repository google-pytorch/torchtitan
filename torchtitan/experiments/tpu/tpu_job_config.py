"""TPU job config."""

import dataclasses
from typing import Literal

import torchtitan.config
from torchtitan.experiments.graph_trainer.configs import GraphTrainerMemoryPolicy
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.cp_attention import validate_tpu_context_parallel
import torchtitan.trainer


GraphTrainerFSDPSchedule = Literal["none", "prefetch", "interleave"]
BackwardPartitionMode = Literal[
    "none",
    "by_layer",
    "fsdp_pipeline",
]
# These policies are registered by TPU-only modules, so keep their CLI/config
# type out of the shared GraphTrainer policy Literal.
TpuGraphTrainerMemoryPolicy = Literal[
    GraphTrainerMemoryPolicy,
    "full_and_offload",
    "full_and_scan_offload",
    "tpu_selective",
]


@dataclasses.dataclass
class GraphSplitterConfig:
  # Maximum number of complete transformer layers assigned to one forward
  # executable. -1 keeps forward as one executable.
  forward_layers_per_executable: int = -1
  # Assign non-layer work before the first transformer layer to its own partition.
  forward_isolate_prefix: bool = False
  # Backward subdivision applied after the top-level forward/backward split.
  # ``by_layer`` groups adjacent layer backward computations and co-locates any
  # attributable FSDP collective chains with their layer. ``fsdp_pipeline``
  # instead moves each layer's AG earlier and RS later to form overlap stages.
  backward_partition_mode: BackwardPartitionMode = "none"
  # Maximum number of complete transformer layers assigned to one backward
  # executable. Grouping layers widens the compiler scheduling
  # window without changing activation-checkpoint boundaries: every layer keeps
  # its saved input and can be recomputed immediately before its own backward.
  # Multi-layer rematerialization is a separate memory-policy tradeoff that can
  # reduce saved/offloaded boundaries only by extending intermediate lifetimes
  # or repeating recompute, so it is intentionally outside this splitter.
  backward_layers_per_executable: int = 1
  # Reuse dead internal split-boundary buffers through XLA input donation.
  # This should default to True: last-use donation is preferable and is safe
  # unless reusable-buffer identification is wrong. It remains False until
  # TorchTPU includes donation in its compilation cache key (torch_tpu#3614).
  # Original graph inputs and forward-saved tensors are never candidates.
  donate_internal_buffers: bool = False
  # Number of complete backward child executables by which to advance each H2D
  # reload pair relative to its first consumer. Zero keeps the pair in the
  # consumer child; one places it in the immediately preceding child and carries
  # only the completed device tensor across the boundary.
  cpu_offload_prefetch_executables: int = 0


@dataclasses.dataclass
class TPUConfig:
  use_simple_fsdp: bool = False
  # How the graph trainer shards parameters over the data-parallel mesh.
  # ``simple_fsdp``: graph_trainer's DTensor parametrization. ``explicit``:
  # parameters are plain local shards and the gather / reduce-scatter pair is
  # emitted directly (torchtitan/experiments/tpu/graph_trainer/tpu_fsdp.py);
  # DTensor appears only as a view for initialization and checkpoints.
  fsdp_implementation: Literal["simple_fsdp", "explicit"] = "simple_fsdp"
  compile_mode: str = "layer"
  apply_rope_complex_workaround: bool = False
  # Compute dense Llama3/Qwen3 transformer blocks on [B, L, ...] views instead
  # of upstream's folded [T, ...] tensors, which make XLA pick slow TPU layouts
  # (see experiments/tpu/bld_layout.py). Numerics match to rounding order.
  bld_block_layout: bool = True
  use_jax_profiler: bool = False
  # For GraphTrainer, partition the transformed joint FX graph into separately
  # compiled forward and backward modules. All finer splitting is nested under
  # this boundary. Other TPU trainers still use this flag for their older eager
  # forward/backward synchronization path; this PR does not replace those paths.
  # GraphTrainer splitting requires graph_trainer_fsdp_schedule="none",
  # optimizer_in_graph=False, and no serialized precompiled FX artifact;
  # TorchTPU's normal XLA/Tier-2 cache remains supported.
  use_graph_split: bool = False
  graph_split: GraphSplitterConfig = dataclasses.field(
      default_factory=GraphSplitterConfig
  )
  # Compile AdamW step into its own separate FX graph.
  compile_optimizer: bool = False
  # Compile the DTensor-aware global gradient norm as a separate static graph.
  compile_grad_norm: bool = False
  # Lower explicitly selected FSDP gradient reduce-scatters as all-reduce plus
  # a rank-local slice. The patterns match parameter FQNs recorded by
  # GraphTrainer; an empty tuple is a no-op. This is a TPU latency workaround,
  # not a generally cheaper collective, so recipes must also provide the
  # largest validated full-gradient payload.
  fsdp_allreduce_slice_parameter_patterns: tuple[str, ...] = ()
  fsdp_allreduce_slice_max_input_bytes: int = 0
  # Run via TpuGraphTrainer (joint fwd+loss+bwd aot_fx_trace and FX-graph SAC passes).
  use_graph_trainer: bool = False
  # Whether to compile the optimizer step into the joint fwd+loss+bwd graph.
  optimizer_in_graph: bool = False
  # GraphTrainer rematerializes its joint forward/backward FX graph instead of
  # tracing eager activation-checkpoint wrappers.
  graph_trainer_memory_policy: TpuGraphTrainerMemoryPolicy = "default"
  # CPU-offload controls propagated into GraphTrainer's compile config.
  graph_trainer_cpu_offload_budget_gb: float = 100.0
  # ``tpu_selective`` only: full recompute, except these save regions, whose
  # values are saved for backward in every layer: "moe.routing" and
  # "attention.inner_attention" (see ``experiments/tpu/remat_regions.py``).
  graph_trainer_save_regions: tuple[str, ...] = ()
  # ``tpu_selective`` only: save regions to save only in their last N
  # transformer layers, which backward reaches first. List each region here or
  # in graph_trainer_save_regions, not both. Example:
  # {"attention.inner_attention": 4}, or on the CLI
  # --tpu_config.graph_trainer_save_last_n_layers attention.inner_attention 4
  graph_trainer_save_last_n_layers: dict[str, int] = dataclasses.field(
      default_factory=dict
  )
  # Run the decoder layers as a single scan instead of a Python loop, so the
  # compiler sees one loop body rather than n_layers copies of it. Each layer's
  # weights become a row of one stacked tensor, which is what gives the scan
  # something to iterate over.
  #
  # Model checkpoints are unaffected: they keep the per-layer keys either way.
  # Optimizer checkpoints are not, because the parameter names change, so a run
  # cannot resume across a flip of this flag.
  # https://github.com/google-pytorch/torchtitan/issues/21
  use_scan_layers: bool = False
  # Keep rematerialized layer workspaces local in the XLA schedule.
  enable_graph_trainer_remat_schedule: bool = False
  # TPU SimpleFSDP scheduling for GraphTrainer rematerialization. ``prefetch``
  # preserves recomputed all-gathers and places them in preceding dense compute;
  # ``interleave`` also fences each next all-gather on prior reduce-scatter
  # completion, bounding memory more tightly at the cost of AG/RS concurrency.
  # TODO: Move this and the other graph_trainer_* options into a dedicated
  # GraphTrainer TPU config once that nested config exists.
  graph_trainer_fsdp_schedule: GraphTrainerFSDPSchedule = "none"
  # Release layer L's recomputed parameters at backward layer L+N. N=1 is the
  # lowest-memory useful prefetch and the best measured real-model policy;
  # larger values expose more AGs for overlap but increase parameter lifetime.
  graph_trainer_fsdp_prefetch_distance: int = 1
  # Emit one aggregate MFU measurement after the configured start log step
  # through the end of training. Both boundaries must be metric log steps.
  enable_mfu_benchmark: bool = False
  mfu_benchmark_start_step: int = 5
  # Disable Automatic Mixed Precision (AMP), so all training is done in uniform
  # precision.
  enable_amp: bool = True
  log_freq: int = 10
  eager_mode: Literal[
      "",
      "DEFER_AND_FUSE",
      "INTERNAL_DEFER_ALL",
      "DEFER_NEVER",
      "DEFER_NEVER_AND_LAUNCH_BLOCKING",
  ] = ""


@dataclasses.dataclass
class ConformerConfig:
  use_ctc_loss: bool = False
  use_jax_ctc_loss: bool = False
  # Target sequence length for synthetic data generation when CTC loss is enabled.
  # Input sequence length must be >= target sequence length.
  ctc_loss_target_len: int = 30


@dataclasses.dataclass
class Qwen3Config:
  use_gmm_kernel: bool = False
  use_fill_indices_kernel: bool = False


@dataclasses.dataclass
class FluxConfig:
  precomputed_dataset_path: str | None = None
  """Path to the precomputed embeddings"""
  random_dataset: bool = False
  """Whether to train on random instead of real data"""
  img_size: int | None = None
  """Overrides the image size (and derived max_context_length) of the config.

  Upstream nests img_size under the tyro-suppressed dataloader.dataset, so it
  cannot be set from the command line otherwise.
  """


@dataclasses.dataclass
class LoRAConfig:
  # Whether to enable LoRA.
  use_lora: bool = False
  lora_rank: int = 16
  lora_alpha: float = 16.0
  lora_dtype: str = "bfloat16"
  # Whether to force LoRA parameters to use DDP (replicated) instead of FSDP
  # (sharded).
  force_lora_parameter_ddp: bool = True


@dataclasses.dataclass
class SplashAttentionKernelConfig:
  use_splash_attention_kernel: bool = False
  use_tokamax_splash_attention_kernel: bool = False
  # Forward is always vmapped. Enabling this also vectorizes backward instead
  # of using its memory-safe sequential scan. With batch-native masking below,
  # this permits a batched backward custom call. Vectorized intermediates can
  # scale with local batch size, so recipes should validate peak memory.
  use_vmap_bwd: bool = False
  # Independently select document-aware block skipping when True or
  # batch-native masking when False. Both retain exact segment-ID masking.
  # False remains useful with a scanned backward because forward still benefits;
  # combine False with use_vmap_bwd=True to make backward batch-native too.
  sa_use_document_aware_block_skipping: bool = True
  # Splash attention block sizes for performance optimization.
  sa_block_q: int = 512
  sa_block_kv: int = 512
  sa_block_dkv: int = 512
  sa_block_kv_compute: int = 512
  sa_block_q_dkv: int = 512
  sa_block_kv_dkv: int = 512
  sa_block_kv_dkv_compute: int = 512
  sa_block_q_dq: int = 512
  sa_block_kv_dq: int = 512
  sa_use_fused_bwd_kernel: bool = True
  sa_q_layout: str = "HEAD_DIM_MINOR"
  sa_k_layout: str = "HEAD_DIM_MINOR"
  sa_v_layout: str = "HEAD_DIM_MINOR"


@dataclasses.dataclass
class LossKernelConfig:
  use_loss_kernel: bool = False
  # Linear softmax cross entropy loss block sizes for performance optimization.
  loss_b_block_size: int = 1024
  loss_h_block_size: int = 512
  loss_v_block_size: int = 2048
  enable_pallas_loss_kernel: bool = True


# Shared base for every TPU-targeted lane (torch_tpu, jax, torchax). Holds the
# lane-agnostic, TPU-flavored config blocks (splash attention, Pallas loss
# kernel, LoRA, per-model toggles). The truly lane-specific runtime knobs
# (``tpu_config`` for torch_tpu, ``jax_config`` for jax, ``torchax_config`` for
# torchax) live on the subclasses below.
@dataclasses.dataclass(kw_only=True, slots=True)
class BaseTPUTrainerConfig(torchtitan.trainer.Trainer.Config):
  xla_flags: tpu_flags.FlagMap = dataclasses.field(
      default_factory=tpu_flags.FlagMap
  )
  libtpu_init_flags: tpu_flags.FlagMap = dataclasses.field(
      default_factory=tpu_flags.FlagMap
  )
  extra_xla_flags: tpu_flags.FlagMap = dataclasses.field(
      default_factory=tpu_flags.FlagMap
  )
  extra_libtpu_init_flags: tpu_flags.FlagMap = dataclasses.field(
      default_factory=tpu_flags.FlagMap
  )
  conformer: ConformerConfig = dataclasses.field(
      default_factory=ConformerConfig
  )
  qwen3: Qwen3Config = dataclasses.field(default_factory=Qwen3Config)
  flux: FluxConfig = dataclasses.field(default_factory=FluxConfig)
  lora: LoRAConfig = dataclasses.field(default_factory=LoRAConfig)
  splash_attention_kernel: SplashAttentionKernelConfig = dataclasses.field(
      default_factory=SplashAttentionKernelConfig
  )
  loss_kernel: LossKernelConfig = dataclasses.field(
      default_factory=LossKernelConfig
  )

  def __post_init__(self):
    # Upstream d771abb0 validates CUDA-graph capturability in
    # `Trainer.Config.__post_init__`, and rejects any MoE token dispatcher it
    # does not recognize -- which is all of the TPU ones. There is no CUDA
    # graph to capture in this lane, so opt out before the check runs. Must
    # come before `super()`, which is what calls `_validate_cuda_graphs`.
    self.training.disable_cuda_graphs = True
    # NOTE: explicit ``super(...)`` (rather than zero-arg ``super()``) is
    # required because ``@dataclass(slots=True)`` replaces the class object
    # after decoration, leaving the ``__class__`` cell that zero-arg
    # ``super()`` relies on pointing at a stale class. See bpo-46404.
    super(BaseTPUTrainerConfig, self).__post_init__()
    if (
        self.model_spec is not None
        and self.parallelism.context_parallel_degree > 1
    ):
      # CP recipes apply core ContextParallelTransform with a TPU CP inner
      # attention; core validates the CP backend, this adds the TPU rules.
      validate_tpu_context_parallel(
          self.model_spec.model,
          self.parallelism,
          training=self.training,
          splash_attention_kernel=self.splash_attention_kernel,
      )
    # This prevents creating folder in
    # torchtitan.distributed.utils.init_distributed
    # that is crashing when running on TAP.
    self.comm.trace_buf_size = 0

  @classmethod
  def derive_from(cls, src, **overrides):
    """Copy-construct a ``cls`` instance from a sibling ``BaseTPUTrainerConfig``
    subclass, then apply ``overrides``.

    Used by the jax / torchax config registries to delegate to the tpu lane:
    call the tpu function, then ``derive_from`` the resulting
    ``TPUTrainerConfig`` onto ``JaxJobConfig`` / ``TorchaxJobConfig``. Fields
    that exist only on the source (e.g. ``tpu_config``) are silently dropped;
    fields that exist only on the destination (``jax_config`` etc.) get their
    default unless supplied via ``overrides``. The model_spec.flavor and most
    other state flow through unchanged — call sites typically only need to
    mutate the lane-specific bits after construction.
    """
    dst_names = {f.name for f in dataclasses.fields(cls)}
    src_names = {f.name for f in dataclasses.fields(src)}
    shared = (dst_names & src_names) - set(overrides)
    return cls(
        **{name: getattr(src, name) for name in shared},
        **overrides,
    )


@dataclasses.dataclass(kw_only=True, slots=True)
class TPUTrainerConfig(BaseTPUTrainerConfig):
  tpu_config: TPUConfig = dataclasses.field(default_factory=TPUConfig)

  def __post_init__(self):
    super(TPUTrainerConfig, self).__post_init__()
    if self.tpu_config.compile_grad_norm:
      if self.tpu_config.optimizer_in_graph:
        raise ValueError(
            "compile_grad_norm cannot be combined with optimizer_in_graph"
        )
      # PP needs an additional reduction across stages, which each own only a
      # subset of the parameters. EP mixes dense and expert gradients on
      # different meshes, so upstream computes their norms separately before
      # combining the scalar results. Both can be supported by tracing those
      # extra reductions; the current compiled path handles one gradient list.
      if (
          self.parallelism.expert_parallel_degree > 1
          or self.parallelism.pipeline_parallel_degree > 1
      ):
        raise ValueError("compile_grad_norm currently requires EP1 and PP1")
    # Only the graph trainer builds its model through
    # `to_tpu_graph_trainer_config`, which is what applies the scan rewrite.
    # Anywhere else the flag would be accepted and then do nothing.
    if self.tpu_config.use_scan_layers and not self.tpu_config.use_graph_trainer:
      raise ValueError(
          "--tpu-config.use-scan-layers needs "
          "--tpu-config.use-graph-trainer; no other lane applies it."
      )
    if self.tpu_config.fsdp_implementation != "simple_fsdp":
      # Only the graph trainer's parallelize wrappers take the choice (the graph
      # trainer always shards with a TPU-side FSDP; the eager lanes do not).
      if not self.tpu_config.use_graph_trainer:
        raise ValueError(
            f"--tpu-config.fsdp-implementation={self.tpu_config.fsdp_implementation} "
            "needs --tpu-config.use-graph-trainer"
        )
      if self.parallelism.tensor_parallel_degree > 1:
        raise ValueError(
            "--tpu-config.fsdp-implementation=explicit does not support "
            "tensor parallelism yet"
        )
    # Set the default location of dump_folder. Allow the runtime CLI
    # configuration (i.e. --job.dump_folder=...) to override this parameter.
    if not self.dump_folder:
      self.dump_folder = "/tmp/outputs"
