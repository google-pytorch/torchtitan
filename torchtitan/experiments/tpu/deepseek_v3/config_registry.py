"""TPU training recipes for DeepSeek-V3."""

import dataclasses

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.graph_trainer.config import TpuGraphTrainerCompileConfig
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.tpu_job_config import TpuTrainerCompileConfig
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

__all__ = [
    "deepseek_v3_671b_2l_16e",
    "deepseek_v3_671b_2l_16e_fused_ce",
    "deepseek_v3_671b_2l_16e_tpu_moe",
    "deepseek_v3_671b_2l_16e_tpu_moe_pallas",
    "deepseek_v3_debugmodel",
    "deepseek_v3_debugmodel_fused_ce",
    "deepseek_v3_debugmodel_tpu_moe",
    "deepseek_v3_debugmodel_tpu_moe_pallas",
    "graph_trainer_deepseek_v3_671b_2l_16e",
    "graph_trainer_deepseek_v3_671b_2l_16e_fused_ce",
    "graph_trainer_deepseek_v3_671b_2l_16e_tpu_moe",
    "graph_trainer_deepseek_v3_671b_2l_16e_tpu_moe_pallas",
    "graph_trainer_deepseek_v3_debugmodel",
    "graph_trainer_deepseek_v3_debugmodel_tpu_moe",
]

_DEBUGMODEL_CONTEXT_LENGTH = 2048
# The target's sequence length (MaxText Lineage and the 671B-31L proxy).
_PROXY_CONTEXT_LENGTH = 4096
# Overrides are identified by the import path of the decorated factory.
_FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"
)


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


def _enable_tpu_moe(config, *, pallas_grouped_mm: bool):
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
      config.model_spec.model, pallas_grouped_mm=pallas_grouped_mm
  )
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
