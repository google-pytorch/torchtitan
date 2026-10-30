"""TPU job config."""

import dataclasses
from typing import Literal

import torchtitan.config
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.cp_attention import validate_tpu_context_parallel
import torchtitan.trainer


@dataclasses.dataclass
class TPUConfig:
  """TorchTPU runtime controls shared by the regular and graph trainers."""

  use_simple_fsdp: bool = False
  apply_rope_complex_workaround: bool = False
  # Compute dense Llama3/Qwen3 transformer blocks on [B, L, ...] views instead
  # of upstream's folded [T, ...] tensors, which make XLA pick slow TPU layouts
  # (see experiments/tpu/bld_layout.py). Numerics match to rounding order.
  bld_block_layout: bool = True
  use_jax_profiler: bool = False
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


TpuTrainerCompileMode = Literal["layer", "fullgraph", "whole", "block"]
TpuTrainerOptimizerCompileMode = Literal["eager", "separate_graph"]


@dataclasses.dataclass(kw_only=True, slots=True)
class TpuTrainerCompileConfig(torchtitan.config.CompileConfig):
  """Generic compile policy plus regular-Trainer TorchTPU controls."""

  # TPU model families support different subsets: Llama/Qwen use
  # layer/fullgraph, Conformer uses layer/whole, and redacted uses block/whole.
  mode: TpuTrainerCompileMode = "layer"
  # TorchTPU builds lazy XLA graphs even though TPUTrainer does not capture a
  # joint FX graph. Materializing loss before backward ends the forward XLA
  # graph while preserving autograd, so backward forms a second XLA graph.
  # This is one runtime boundary, not GraphTrainer's multi-executable splitter.
  materialize_forward_before_backward: bool = False
  # How optimizer.step executes: normally through TorchTPU eager/deferred
  # execution, or as a separately captured and compiled FX graph.
  optimizer_compile: TpuTrainerOptimizerCompileMode = "eager"
  # Compile the DTensor-aware global gradient norm as a separate static graph.
  compile_grad_norm: bool = False


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
  # Tokamax only. Query heads sharing a KV head that the forward processes as
  # one tile.
  sa_num_stacked_q_heads: int = 1
  # Tokamax only. Run the forward once per example instead of vmapping it.
  # Tokamax skips tiles with no same-document pair, but a vmapped call must keep
  # every tile that any example in the batch needs.
  use_batch_loop_fwd: bool = False
  # Tokamax only. The same for the backward; takes precedence over use_vmap_bwd.
  use_batch_loop_bwd: bool = False
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
    dst_fields = {f.name: f for f in dataclasses.fields(cls)}
    dst_names = set(dst_fields)
    src_names = {f.name for f in dataclasses.fields(src)}
    shared = (dst_names & src_names) - set(overrides)
    values = {name: getattr(src, name) for name in shared}

    # Sibling TPU lanes share Trainer's generic compile policy, but TorchTPU's
    # concrete config adds execution controls that JAX and TorchAX must not
    # inherit merely because they reuse a TorchTPU recipe. Reconstruct the
    # destination's declared/default type and carry only the generic policy.
    if "compile" in values:
      compile_factory = dst_fields["compile"].default_factory
      if compile_factory is dataclasses.MISSING:
        raise TypeError(f"{cls.__name__}.compile must have a default factory")
      dst_compile = compile_factory()
      src_compile = values["compile"]
      if type(src_compile) is not type(dst_compile):
        values["compile"] = type(dst_compile)(
            enable=src_compile.enable,
            components=list(src_compile.components),
            backend=src_compile.backend,
        )
    return cls(
        **values,
        **overrides,
    )


@dataclasses.dataclass(kw_only=True, slots=True)
class TPUTrainerConfig(BaseTPUTrainerConfig):
  compile: TpuTrainerCompileConfig = dataclasses.field(
      default_factory=TpuTrainerCompileConfig
  )
  tpu_config: TPUConfig = dataclasses.field(default_factory=TPUConfig)

  def __post_init__(self):
    super(TPUTrainerConfig, self).__post_init__()
    if type(self.compile) is torchtitan.config.CompileConfig:
      # ``derive_from`` may receive an upstream Trainer.Config. Preserve its
      # generic compile policy while restoring the TPU-specific config type.
      self.compile = TpuTrainerCompileConfig(
          enable=self.compile.enable,
          components=list(self.compile.components),
          backend=self.compile.backend,
      )
    if self.compile.compile_grad_norm:
      # PP needs an additional reduction across stages, which each own only a
      # subset of the parameters. EP mixes dense and expert gradients on
      # different meshes, so upstream computes their norms separately before
      # combining the scalar results. Both can be supported by tracing those
      # extra reductions; the current compiled path handles one gradient list.
      if (
          self.parallelism.expert_parallel_degree > 1
          or self.parallelism.pipeline_parallel_degree > 1
      ):
        raise ValueError(
            "compile.compile_grad_norm currently requires EP1 and PP1"
        )
    # Set the default location of dump_folder. Allow the runtime CLI
    # configuration (i.e. --job.dump_folder=...) to override this parameter.
    if not self.dump_folder:
      self.dump_folder = "/tmp/outputs"
