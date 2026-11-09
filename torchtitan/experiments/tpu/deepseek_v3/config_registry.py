"""TPU training recipes for DeepSeek-V3."""

import dataclasses

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.graph_trainer.config import TpuGraphTrainerCompileConfig
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.loss import PallasMTPCrossEntropyLoss
from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig
from torchtitan.experiments.tpu.tpu_job_config import TpuTrainerCompileConfig
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.models.deepseek_v3.mtp import MTPLoss

__all__ = [
    "deepseek_v3_671b_2l_16e",
    "deepseek_v3_671b_2l_16e_fused_ce",
    "deepseek_v3_671b_2l_16e_tpu_moe",
    "deepseek_v3_671b_2l_16e_tpu_moe_pallas",
    "deepseek_v3_671b_2l_16e_tpu_moe_pallas_ep4",
    "deepseek_v3_671b_2l_16e_target_proxy",
    "deepseek_v3_671b_2l_16e_target_proxy_mtp",
    "deepseek_v3_671b_31l_4x4x4",
    "deepseek_v3_671b_31l_4x4x4_mtp",
    "deepseek_v3_debugmodel",
    "deepseek_v3_debugmodel_fused_ce",
    "deepseek_v3_debugmodel_fused_ce_mtp",
    "deepseek_v3_debugmodel_mtp",
    "deepseek_v3_debugmodel_target_mtp",
    "deepseek_v3_debugmodel_tpu_moe",
    "deepseek_v3_debugmodel_tpu_moe_pallas",
    "graph_trainer_deepseek_v3_671b_2l_16e",
    "graph_trainer_deepseek_v3_671b_2l_16e_fused_ce",
    "graph_trainer_deepseek_v3_671b_2l_16e_tpu_moe",
    "graph_trainer_deepseek_v3_671b_2l_16e_tpu_moe_pallas",
    "graph_trainer_deepseek_v3_debugmodel",
    "graph_trainer_deepseek_v3_debugmodel_mtp",
    "graph_trainer_deepseek_v3_debugmodel_target_mtp",
    "graph_trainer_deepseek_v3_debugmodel_tpu_moe",
]

_DEBUGMODEL_CONTEXT_LENGTH = 2048
# The target's sequence length (MaxText Lineage and the 671B-31L proxy).
_PROXY_CONTEXT_LENGTH = 4096
# Overrides are identified by the import path of the decorated factory.
_FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"
)
# Total MTP loss weight: MaxText's ``mtp_loss_scaling_factor`` (base.yml 0.1;
# the Lineage MLPerf config ``deepseek3-671b-lineage.yml`` sets
# ``mtp_num_layers: 1`` and does not override it). Upstream ``MTPLoss``
# defaults to 0.3 (DeepSeek-V3's weight for its first 10T tokens).
_MTP_LOSS_SCALE = 0.1


def _enable_pallas_cross_entropy(config):
  """Select the fused loss and its pre-construction LM-head override together.

  The override swaps the plain ``Linear`` lm_head for
  ``FusedLossLinearWrapper`` (same weight and checkpoint key), which hands
  ``(hidden, weight.T)`` to the fused kernel, so the ``[tokens, vocab]`` fp32
  logits are never materialized (vocab 129280 at 671B). DeepSeek-V3 has no
  logit soft-cap or multiplier, so the loss keeps its defaults.
  """
  config.loss = PallasCrossEntropyLoss.Config()
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)
  return config


def _use_mtp_loss(config):
  """Switch the recipe's loss to its multi-token-prediction form.

  Plain cross entropy becomes upstream's ``MTPLoss``; the fused Pallas loss
  becomes ``PallasMTPCrossEntropyLoss`` with the same kernel settings. Both
  weigh the MTP depths by ``_MTP_LOSS_SCALE``. The model must already have
  its MTP layers (``model_registry(..., num_mtp_layers=...)``).
  """
  if not config.model_spec.model.mtp_layers:
    raise ValueError("The recipe's model has no MTP layers")
  if isinstance(config.loss, PallasCrossEntropyLoss.Config):
    config.loss = PallasMTPCrossEntropyLoss.Config(
        **{
            field.name: getattr(config.loss, field.name)
            for field in dataclasses.fields(config.loss)
        },
        mtp_scale=_MTP_LOSS_SCALE,
    )
  else:
    config.loss = MTPLoss.Config(mtp_scale=_MTP_LOSS_SCALE)
  return config


def _enable_tpu_moe(
    config, *, pallas_grouped_mm: bool, expert_parallel_degree: int = 1
):
  """Rebuild the recipe's model with the lane's MoE converters (rung 5).

  Same flavor and context length, built through ``model_registry(...,
  converters=tpu_moe_converters())`` so the converters see the final layer
  list (the reduced and 31-layer proxies included), then the Pallas switches.
  """
  # pylint: disable-next=g-import-not-at-top
  from torchtitan.experiments.tpu.deepseek_v3 import model_registry, tpu_moe_converters, use_tpu_moe_kernels

  spec = config.model_spec
  config.model_spec = dataclasses.replace(
      model_registry(
          spec.flavor,
          seq_len=spec.max_context_length,
          converters=tpu_moe_converters(),
          num_mtp_layers=len(spec.model.mtp_layers),
      ),
      parallelize_fn=spec.parallelize_fn,
      pipelining_fn=spec.pipelining_fn,
  )
  use_tpu_moe_kernels(
      config.model_spec.model,
      pallas_grouped_mm=pallas_grouped_mm,
      # Under EP the routing scores stay on the token's rank, so they are
      # applied after the return all-to-all, not before W2.
      scale_before_down_proj=pallas_grouped_mm and expert_parallel_degree == 1,
  )
  config.parallelism.expert_parallel_degree = expert_parallel_degree
  return config


def deepseek_v3_debugmodel() -> TPUTrainerConfig:
  """DeepSeek-V3 debug model (6 layers: 1 dense + 5 MoE, MLA) on TPU.

  Upstream ``debugmodel`` with the document-masked SDPA and SimpleFSDP,
  eager by default (``--compile.enable`` for per-block compile). Plain
  ``CrossEntropyLoss``: ``TPUTrainer`` rebuilds ``loss_fn`` from the config via
  ``loss.build_cross_entropy_loss``, which drops ``ChunkedLossWrapper``. MoE
  balancing is upstream's: the auxiliary-loss-free expert bias
  (``load_balance_coeff``) plus the microbatch-wise aux loss, EP 1.
  """
  # Imported here to avoid a circular import: the package __init__ imports this
  # module at the end.
  from torchtitan.experiments.tpu.deepseek_v3 import model_registry  # pylint: disable=g-import-not-at-top

  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry(
          "debugmodel", seq_len=_DEBUGMODEL_CONTEXT_LENGTH
      ),
      dataloader=c4_test_dataloader(),
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
      parallelism=ParallelismConfig(
          expert_parallel_degree=1,
          use_simple_fsdp=True,
      ),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(),
      # Off by default; --compile.enable turns on per-block compile on TorchTPU.
      compile=TpuTrainerCompileConfig(enable=False, backend="tpu"),
  )


def deepseek_v3_debugmodel_mtp() -> TPUTrainerConfig:
  """``deepseek_v3_debugmodel`` with one MTP layer and upstream's ``MTPLoss``.

  The MTP block (a copy of the last, MoE, layer plus ``enorm`` / ``hnorm`` /
  ``eh_proj`` / ``mtp_norm``) gets the decoder layers' AC and per-block
  compile; MTP loss weight 0.1 (MaxText's).
  """
  # pylint: disable-next=g-import-not-at-top
  from torchtitan.experiments.tpu.deepseek_v3 import model_registry

  config = deepseek_v3_debugmodel()
  config.model_spec = model_registry(
      "debugmodel", seq_len=_DEBUGMODEL_CONTEXT_LENGTH, num_mtp_layers=1
  )
  return _use_mtp_loss(config)


def deepseek_v3_671b_2l_16e() -> TPUTrainerConfig:
  """Per-rung measurement proxy on 4 v6e chips: ``671B-2L-16E``, per-block compile.

  671B's widths (dim 7168, MLA 128 heads with q_lora 1536 / kv_lora 512 / q-k
  head 192 / v head 128, vocab 129280, dense FFN 18432, expert FFN 2048, one
  shared expert, sigmoid group-limited top-8 routing) with 1 dense + 1 MoE
  layer and 16 routed experts. One 4096-token sequence per rank per step, as
  the 64-chip target runs. Same balancers, loss and optimizer as the debug
  recipe; peak LR 2.2e-4 (DeepSeek-V3's) after 10 warmup steps. The test tokenizer only emits ids <
  2048 of the 129280-row vocabulary, which changes no shape.
  """
  # pylint: disable-next=g-import-not-at-top
  from torchtitan.experiments.tpu.deepseek_v3 import REDUCED_671B_FLAVOR, model_registry

  config = deepseek_v3_debugmodel()
  config.model_spec = model_registry(
      REDUCED_671B_FLAVOR, seq_len=_PROXY_CONTEXT_LENGTH
  )
  config.optimizer = default_adamw(lr=2.2e-4, implementation="foreach")
  # Warmup 2 (the debug recipe's) spiked the loss from 12.3 to 15.0 at steps
  # 2-3 at this width (measured on a 3-layer variant, per-block compile). With
  # 10 warmup steps: 12.26 -> 3.26 over 20 steps, the largest rise (+0.64)
  # inside warmup and none past it beyond the largest earlier drop.
  config.lr_scheduler = LRSchedulersContainer.Config(
      warmup_steps=10,
      decay_ratio=0.8,
      decay_type="linear",
      min_lr_factor=0.0,
  )
  config.training = TrainingConfig(
      num_tokens_per_microbatch_per_dp_rank=_PROXY_CONTEXT_LENGTH,
      max_context_length=_PROXY_CONTEXT_LENGTH,
      steps=20,
  )
  config.compile = TpuTrainerCompileConfig(enable=True, backend="tpu")
  return config


def _graph_trainer(config: TPUTrainerConfig) -> TpuGraphTrainer.Config:
  # pylint: disable-next=g-import-not-at-top
  from torchtitan.experiments.tpu.deepseek_v3 import graph_trainer_model_registry

  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      config, graph_trainer_model_registry
  )
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
  )
  return config


def graph_trainer_deepseek_v3_debugmodel() -> TpuGraphTrainer.Config:
  """``deepseek_v3_debugmodel`` on the TPU GraphTrainer (rung 2).

  Forward + loss + backward as one XLA program; the optimizer stays eager.
  The expert-bias balancer keeps working because ``tpu_backend_compile_pass``
  writes ``tokens_per_expert_E`` back after each execution.
  """
  return _graph_trainer(deepseek_v3_debugmodel())


def graph_trainer_deepseek_v3_671b_2l_16e() -> TpuGraphTrainer.Config:
  """``deepseek_v3_671b_2l_16e`` on the TPU GraphTrainer (rung 2).

  Full recompute with remat barriers and interleaved FSDP all-gathers: on the
  document-masked SDPA path each layer materializes [128, 4096, 4096] fp32
  attention scores, and the joint program needs 22.07 GB of temporaries under
  the default policy against ~18.9 GB free (measured), so the per-block
  recipe's selective AC does not carry over.
  """
  config = _graph_trainer(deepseek_v3_671b_2l_16e())
  config.compile.memory_policy = "full"
  config.compile.enable_remat_barriers = True
  config.compile.fsdp_barrier_policy = "interleave"
  config.parallelism.fsdp_reshard_after_forward = "always"
  return config


def deepseek_v3_debugmodel_fused_ce() -> TPUTrainerConfig:
  """``deepseek_v3_debugmodel`` with the fused Pallas linear + cross entropy."""
  return _enable_pallas_cross_entropy(deepseek_v3_debugmodel())


def deepseek_v3_debugmodel_fused_ce_mtp() -> TPUTrainerConfig:
  """``deepseek_v3_debugmodel_mtp`` with the fused loss for every depth."""
  return _use_mtp_loss(
      _enable_pallas_cross_entropy(deepseek_v3_debugmodel_mtp())
  )


def graph_trainer_deepseek_v3_debugmodel_mtp() -> TpuGraphTrainer.Config:
  """``deepseek_v3_debugmodel_mtp`` on the TPU GraphTrainer."""
  return _graph_trainer(deepseek_v3_debugmodel_mtp())


def deepseek_v3_671b_2l_16e_fused_ce() -> TPUTrainerConfig:
  """``deepseek_v3_671b_2l_16e`` with the fused Pallas linear + cross entropy."""
  return _enable_pallas_cross_entropy(deepseek_v3_671b_2l_16e())


def graph_trainer_deepseek_v3_671b_2l_16e_fused_ce() -> TpuGraphTrainer.Config:
  """``graph_trainer_deepseek_v3_671b_2l_16e`` with the fused linear + CE."""
  return _enable_pallas_cross_entropy(graph_trainer_deepseek_v3_671b_2l_16e())


def deepseek_v3_debugmodel_tpu_moe() -> TPUTrainerConfig:
  """``deepseek_v3_debugmodel`` with the TPU dispatcher (rung 5), native experts."""
  return _enable_tpu_moe(deepseek_v3_debugmodel(), pallas_grouped_mm=False)


def deepseek_v3_debugmodel_tpu_moe_pallas() -> TPUTrainerConfig:
  """``deepseek_v3_debugmodel_tpu_moe`` with the Pallas grouped matmul."""
  return _enable_tpu_moe(deepseek_v3_debugmodel(), pallas_grouped_mm=True)


def graph_trainer_deepseek_v3_debugmodel_tpu_moe() -> TpuGraphTrainer.Config:
  """``graph_trainer_deepseek_v3_debugmodel`` with the TPU dispatcher."""
  return _graph_trainer(deepseek_v3_debugmodel_tpu_moe())


def deepseek_v3_671b_2l_16e_tpu_moe() -> TPUTrainerConfig:
  """Rung-5 proxy recipe: fused CE + the TPU dispatcher, native experts.

  Builds on ``deepseek_v3_671b_2l_16e_fused_ce`` (Splash stays a command-line
  flag, as for rungs 3-4).
  """
  return _enable_tpu_moe(
      deepseek_v3_671b_2l_16e_fused_ce(), pallas_grouped_mm=False
  )


def deepseek_v3_671b_2l_16e_tpu_moe_pallas() -> TPUTrainerConfig:
  """``deepseek_v3_671b_2l_16e_tpu_moe`` with the Pallas grouped matmul."""
  return _enable_tpu_moe(
      deepseek_v3_671b_2l_16e_fused_ce(), pallas_grouped_mm=True
  )


def graph_trainer_deepseek_v3_671b_2l_16e_tpu_moe() -> TpuGraphTrainer.Config:
  """``graph_trainer_deepseek_v3_671b_2l_16e_fused_ce`` with the TPU dispatcher."""
  return _enable_tpu_moe(
      graph_trainer_deepseek_v3_671b_2l_16e_fused_ce(), pallas_grouped_mm=False
  )


def graph_trainer_deepseek_v3_671b_2l_16e_tpu_moe_pallas() -> (
    TpuGraphTrainer.Config
):
  """``graph_trainer_deepseek_v3_671b_2l_16e_tpu_moe`` with Pallas experts."""
  return _enable_tpu_moe(
      graph_trainer_deepseek_v3_671b_2l_16e_fused_ce(), pallas_grouped_mm=True
  )


def deepseek_v3_671b_2l_16e_tpu_moe_pallas_ep4() -> TPUTrainerConfig:
  """``deepseek_v3_671b_2l_16e_tpu_moe_pallas`` at EP 4 (4 experts per chip).

  The static-shape ragged all-to-all dispatcher; scores applied after the
  return exchange (no score-before-W2 under EP).
  """
  return _enable_tpu_moe(
      deepseek_v3_671b_2l_16e_fused_ce(),
      pallas_grouped_mm=True,
      expert_parallel_degree=4,
  )


# The 64-chip target: one tpu7x 4x4x4 slice, 128 ranks (one per TensorCore).
_TARGET_EP_DEGREE = 16


def _target_recipe(
    flavor: str, expert_parallel_degree: int, num_mtp_layers: int = 0
) -> TPUTrainerConfig:
  """The 64-chip target recipe's settings on any flavor (see below)."""
  config = _enable_pallas_cross_entropy(deepseek_v3_debugmodel())
  # pylint: disable-next=g-import-not-at-top
  from torchtitan.experiments.tpu.deepseek_v3 import model_registry, tpu_moe_converters, use_tpu_moe_kernels

  config.model_spec = model_registry(
      flavor,
      seq_len=_PROXY_CONTEXT_LENGTH,
      converters=tpu_moe_converters(),
      num_mtp_layers=num_mtp_layers,
  )
  use_tpu_moe_kernels(
      config.model_spec.model,
      pallas_grouped_mm=True,
      scale_before_down_proj=False,
  )
  config.optimizer = default_adamw(lr=2.2e-4, implementation="foreach")
  config.lr_scheduler = LRSchedulersContainer.Config(
      warmup_steps=10,
      decay_ratio=0.8,
      decay_type="linear",
      min_lr_factor=0.0,
  )
  config.training = TrainingConfig(
      num_tokens_per_microbatch_per_dp_rank=_PROXY_CONTEXT_LENGTH,
      max_context_length=_PROXY_CONTEXT_LENGTH,
      steps=30,
  )
  config.parallelism = ParallelismConfig(
      use_simple_fsdp=True,
      data_parallel_shard_degree=-1,
      expert_parallel_degree=expert_parallel_degree,
  )
  config.activation_checkpoint = FullAC.Config()
  config.splash_attention_kernel = SplashAttentionKernelConfig(
      use_splash_attention_kernel=True
  )
  config.compile = TpuTrainerCompileConfig(enable=True, backend="tpu")
  config.checkpoint = CheckpointManager.Config(enable=False)
  config.tpu_config = TPUConfig(
      enable_mfu_benchmark=True, mfu_benchmark_start_step=10
  )
  config.metrics = MetricsProcessor.Config(log_freq=1)
  if num_mtp_layers:
    _use_mtp_loss(config)
  return config


def deepseek_v3_671b_31l_4x4x4() -> TPUTrainerConfig:
  """``671B-31L`` (3 dense + 28 MoE, 256 experts top-8) on tpu7x-4x4x4.

  Per-block compile, SimpleFSDP over all 128 ranks (one per TensorCore) for
  the dense weights, EP 16 (16 experts per rank; expert banks FSDP-sharded
  over the 8-way EDP mesh), the static-shape ragged EP dispatcher, Pallas
  grouped matmul (scores after the return exchange), Splash with segment ids,
  fused linear + CE, full AC. Global batch 128 x 4096 (one 4096-token sequence
  per rank), MaxText Lineage's. bf16 experts, dropless, MTP 0 (the TPU
  trainer has no MTP loss yet); MaxText's run is fp8 experts, capacity factor
  3.0, MTP 1, so compare tokens/s/device, not MFU.

  EP degree: the dropless receive buffer is ``EP x 32768`` rows of 7168 bf16
  (7.5 GB at EP 16, 30 GB at EP 64, 60 GB at EP 128), so EP 64 / 128 do not
  fit a 96 GiB rank next to ~42 GB of FP32 state; lower EP gathers more
  expert weights per layer over the EDP mesh. ``--parallelism.
  expert_parallel_degree=32`` is the next step up if EP 16's peak HBM leaves
  room. Per-rank estimate in the onboarding record.
  """
  return _target_recipe("671B-31L", _TARGET_EP_DEGREE)


def deepseek_v3_671b_2l_16e_target_proxy() -> TPUTrainerConfig:
  """``deepseek_v3_671b_31l_4x4x4``'s exact settings on ``671B-2L-16E``, EP 4.

  The pre-dispatch gate's reduced proxy: same dispatcher, kernels, AC, loss,
  attention, compile and token budget per rank, on 4 v6e chips.
  """
  return _target_recipe("671B-2L-16E", 4)


def deepseek_v3_671b_31l_4x4x4_mtp() -> TPUTrainerConfig:
  """``deepseek_v3_671b_31l_4x4x4`` plus one MTP layer (MaxText Lineage's MTP 1).

  The MTP block is a 29th MoE block (ragged EP dispatcher, Pallas experts) with
  the decoder layers' full AC and per-block compile; its head shares the LM
  head and runs through the fused loss (``PallasMTPCrossEntropyLoss``, MTP
  weight 0.1 = MaxText's ``mtp_loss_scaling_factor``). 337.4B parameters
  (MaxText reports 337.418B for its 31-layer MTP-1 proxy).
  """
  return _target_recipe("671B-31L", _TARGET_EP_DEGREE, num_mtp_layers=1)


def deepseek_v3_671b_2l_16e_target_proxy_mtp() -> TPUTrainerConfig:
  """``deepseek_v3_671b_2l_16e_target_proxy`` plus one MTP layer (MoE, 16 experts)."""
  return _target_recipe("671B-2L-16E", 4, num_mtp_layers=1)


def deepseek_v3_debugmodel_target_mtp() -> TPUTrainerConfig:
  """``deepseek_v3_671b_31l_4x4x4_mtp``'s settings on ``debugmodel``, EP 4.

  The MTP recipe's switches together (ragged EP dispatcher, Pallas experts,
  Splash, fused loss for both depths, full AC) at end-to-end-test size.
  """
  return _target_recipe("debugmodel", 4, num_mtp_layers=1)


def graph_trainer_deepseek_v3_debugmodel_target_mtp() -> TpuGraphTrainer.Config:
  """``deepseek_v3_debugmodel_target_mtp`` on the TPU GraphTrainer."""
  return _graph_trainer(deepseek_v3_debugmodel_target_mtp())

