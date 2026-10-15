"""TPU training recipes for Muse Glimmer (text-only)."""

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.graph_trainer.config import TpuGraphTrainerCompileConfig
from torchtitan.experiments.tpu.graph_trainer.muse_glimmer import model_registry as graph_trainer_model_registry
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.tpu_job_config import LossKernelConfig
from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig
from torchtitan.experiments.tpu.tpu_job_config import TpuTrainerCompileConfig
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

__all__ = [
    "graph_trainer_muse_glimmer_debugmodel",
    "muse_glimmer_debugmodel",
    "muse_glimmer_debugmodel_fused_ce",
    "muse_glimmer_30b_full_logits",
    "muse_glimmer_30b_fused_ce",
    "reg_trainer_muse_glimmer_30b_fused_ce",
]

_DEBUGMODEL_CONTEXT_LENGTH = 2048

# Upstream muse_glimmer_30b's default context length.
_30B_CONTEXT_LENGTH = 16384

# Overrides are identified by the import path of the decorated factory.
_FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss."
    "fused_loss_soft_capped_lm_head"
)

# The dense tpu7x-4x4x4 launch flags, the same set the Qwen3-8B/32B 4x4x4
# recipes use.
_MUSE_GLIMMER_4X4X4_FLAGS = tpu_flags.merge_flags(
    tpu_flags.TPU7X_THROUGHPUT_P_STATE,
    {"xla_tpu_scoped_vmem_limit_kib": 65472},
    tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
    tpu_flags.HIERARCHICAL_REDUCE_SCATTER,
)


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Select the fused loss and its pre-construction LM-head override together.

  Same idiom as the qwen3 registry, with two Muse specifics: the override
  targets ``SoftCappedLinear.Config`` (Muse's lm_head type), and the head's
  soft-cap / output multiplier are copied into the loss config, because the
  wrapper only hands ``(hidden, weight.T)`` to the loss and the transform is
  applied inside the fused kernel.
  """
  lm_head = config.model_spec.model.lm_head
  config.loss = PallasCrossEntropyLoss.Config(
      soft_cap=lm_head.output_soft_cap_temp,
      output_multiplier=lm_head.output_multiplier,
  )
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)


def muse_glimmer_debugmodel() -> TPUTrainerConfig:
  """Muse Glimmer debug model (8 layers, dim 256) on TPU with SimpleFSDP.

  Unlike upstream ``muse_glimmer_debugmodel`` this uses plain
  ``CrossEntropyLoss``: ``TPUTrainer`` rebuilds ``loss_fn`` from the config via
  ``loss.build_cross_entropy_loss``, which drops ``ChunkedLossWrapper``. The
  soft-cap lives in ``SoftCappedLinear`` (the ``lm_head``), so it is still
  applied on the full-logits path.

  The legacy Pallas fused loss (``--loss_kernel.use_loss_kernel``) is not
  supported: it bypasses ``SoftCappedLinear.forward`` and would drop the output
  multiplier and soft-cap, so ``workarounds.use_output_projection_patch``
  rejects it. Use ``muse_glimmer_debugmodel_fused_ce`` instead, which applies
  the soft-cap inside the fused kernel.
  """
  # Imported here to avoid a circular import: the package __init__ imports this
  # module at the end.
  from torchtitan.experiments.tpu.muse_glimmer import model_registry  # pylint: disable=g-import-not-at-top

  model_spec = model_registry("debugmodel", seq_len=_DEBUGMODEL_CONTEXT_LENGTH)
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
          num_tokens_per_microbatch_per_dp_rank=8 * _DEBUGMODEL_CONTEXT_LENGTH,
          max_context_length=_DEBUGMODEL_CONTEXT_LENGTH,
          steps=10,
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(use_simple_fsdp=True),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(),
  )


def muse_glimmer_debugmodel_fused_ce() -> TPUTrainerConfig:
  """``muse_glimmer_debugmodel`` with the fused Pallas cross entropy.

  The ``SoftCappedLinear`` lm_head is replaced by ``FusedLossLinearWrapper`` and
  ``PallasCrossEntropyLoss`` re-applies the head's ``cap * tanh(z * mul / cap)``
  to each logits tile in-kernel (cap 20, multiplier 1.0 for the debug flavor),
  so the full fp32 logits are never materialized.
  """
  config = muse_glimmer_debugmodel()
  _enable_pallas_cross_entropy(config)
  return config


def graph_trainer_muse_glimmer_debugmodel() -> TpuGraphTrainer.Config:
  """Muse Glimmer debug model using TPU GraphTrainer."""
  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      muse_glimmer_debugmodel(), graph_trainer_model_registry
  )
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
  )
  return config


def _muse_glimmer_30b_base() -> TPUTrainerConfig:
  """Model, data, optimizer and kernels shared by every 30B recipe.

  Model and optimizer settings follow upstream ``muse_glimmer_30b``: context
  16384, one sequence per device, AdamW lr 3e-4 with 200 warmup steps. The
  loss is plain ``CrossEntropyLoss`` over the full logits; the recipes below
  pick the trainer and, optionally, the fused loss.
  """
  from torchtitan.experiments.tpu.muse_glimmer import model_registry  # pylint: disable=g-import-not-at-top

  model_spec = model_registry("30B", seq_len=_30B_CONTEXT_LENGTH)
  config = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      # Test tokenizer, like the other TPU recipes; pass the Muse Glimmer
      # tokenizer via --hf_assets_path for meaningful loss values.
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      # foreach for the regular trainer; the GraphTrainer recipes use fused.
      optimizer=default_adamw(lr=3e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
      training=TrainingConfig(
          max_norm=1.0,
          num_tokens_per_microbatch_per_dp_rank=1 * _30B_CONTEXT_LENGTH,
          max_context_length=_30B_CONTEXT_LENGTH,
          steps=30,
          mixed_precision_reduce="float32",
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          enable_tpu_topo_mesh=True,
          data_parallel_shard_degree=-1,
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(),
      # Disable Shardy HLO sharding v3 (default since libtpu 0.0.48), matching
      # the other TPU recipes.
      xla_flags=tpu_flags.merge_flags(
          xla_enable_hlo_sharding_v3=False,
      ),
      compile=TpuTrainerCompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      # JAX Pallas Splash (kernels/splash_attention.py): LocalMask for the 39
      # sliding-window layers, causal for the 13 global ones. Tokamax Splash
      # (0.0.14) measured ~5% slower: its wrapper has no document-aware block
      # skipping, so the global layers compute cross-document blocks of the
      # packed rows.
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
      loss_kernel=LossKernelConfig(use_loss_kernel=False),
      validator=Validator.Config(enable=False),
  )
  # Merged into the default, which keeps the diagnostic coordinator off.
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags, _MUSE_GLIMMER_4X4X4_FLAGS
  )
  return config


def muse_glimmer_30b_full_logits() -> TpuGraphTrainer.Config:
  """Muse Glimmer 30B GraphTrainer recipe for a tpu7x-4x4x4 slice.

  SimpleFSDP Zero-3 over all devices with the prefetch FSDP schedule, full
  graph rematerialization except each layer's Splash output and LSE and its
  FFN down-projection (``w2``) output, JAX Pallas Splash attention and plain
  cross entropy over the full logits. The 4x4x4 libtpu flags ride on the
  config.

  Full logits vs fused cross entropy, measured on tpu7x-4x4x4 with fp32
  weights and optimizer state (the default ``training.dtype``) and bf16
  compute, with full recompute (with the two saves the fused loss is ~3 %
  slower):

  * Why full logits here: with 16384 tokens per device the logits are not
    what sets peak HBM (46.18 GiB full vs 46.12 GiB fused, both measured
    with plain ``full`` rematerialization), so the fused
    kernel saves nothing, while XLA's plain LM-head matmul + cross entropy
    runs ~2% faster than the vocab-tiled Pallas kernel (2,913 vs 2,851 TPS,
    43.8% vs 42.9% MFU).
  * When to use the fused loss: once the logits dominate memory. They scale
    with tokens per microbatch, so at 2 x 16384 tokens per device peak HBM is
    87.62 GiB with full logits vs 51.84 GiB fused. Use it for larger local
    batches or longer sequences, on the regular trainer (which runs out of
    HBM at step 1 with full logits), or whenever HBM headroom is needed
    elsewhere.
  """
  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      _muse_glimmer_30b_base(), graph_trainer_model_registry
  )
  # Zero-3: Zero-2 would keep all ~60 GiB of gathered BF16 weights resident.
  config.parallelism.fsdp_reshard_after_forward = "always"
  config.tpu_config = TPUConfig()
  # GraphTrainer ignores the eager AC config; its graph rematerialization
  # policy is selected here.
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
      optimizer_compile="in_graph",
      memory_policy="tpu_selective",
      save_regions=("attention.inner_attention", "feed_forward.w2"),
      enable_remat_barriers=True,
      fsdp_barrier_policy="prefetch",
  )
  # Fused AdamW traces into the optimizer-in-graph step and matches foreach
  # (2,924 vs 2,913 TPS, same HBM and loss).
  config.optimizer.implementation = "fused"
  return config


def muse_glimmer_30b_fused_ce() -> TpuGraphTrainer.Config:
  """``muse_glimmer_30b_full_logits`` with the soft-capped fused loss.

  The LM-head matmul, soft cap and cross entropy run as one Pallas kernel, so
  the [tokens, 202048] logits are never materialized. ~3% slower than
  full logits at 16384 tokens per device, but needed once the logits dominate
  memory; see ``muse_glimmer_30b_full_logits`` for when to use which.
  """
  config = muse_glimmer_30b_full_logits()
  _enable_pallas_cross_entropy(config)
  return config


def reg_trainer_muse_glimmer_30b_fused_ce() -> TPUTrainerConfig:
  """Muse Glimmer 30B on the regular ``TPUTrainer`` with the fused loss.

  The per-layer compile, selective AC and graph-split setup tuned on
  tpu7x-2x2x2, with the same model, data and flags as
  ``muse_glimmer_30b_full_logits``.
  The fused loss is required: with full logits this trainer runs out of HBM
  at step 1 on tpu7x-4x4x4.
  """
  config = _muse_glimmer_30b_base()
  config.tpu_config = TPUConfig(eager_mode="DEFER_AND_FUSE")
  config.compile = TpuTrainerCompileConfig(
      enable=True,
      backend="tpu",
      components=["model"],
      materialize_forward_before_backward=True,
      compile_grad_norm=True,
  )
  _enable_pallas_cross_entropy(config)
  return config
