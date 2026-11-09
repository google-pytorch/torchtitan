"""TPU training recipes for DeepSeek-V3."""

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
from torchtitan.experiments.tpu.tpu_job_config import TpuTrainerCompileConfig
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

__all__ = [
    "deepseek_v3_671b_2l_16e",
    "deepseek_v3_debugmodel",
    "graph_trainer_deepseek_v3_671b_2l_16e",
    "graph_trainer_deepseek_v3_debugmodel",
]

_DEBUGMODEL_CONTEXT_LENGTH = 2048
# The target's sequence length (MaxText Lineage and the 671B-31L proxy).
_PROXY_CONTEXT_LENGTH = 4096


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
