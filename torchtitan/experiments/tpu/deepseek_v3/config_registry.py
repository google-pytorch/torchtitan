"""TPU training recipes for DeepSeek-V3."""

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.graph_trainer.deepseek_v3 import model_registry as graph_trainer_model_registry
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.graph_trainer.config import TpuGraphTrainerCompileConfig
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer
from torchtitan.experiments.tpu.tpu_job_config import TpuTrainerCompileConfig
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

__all__ = [
    "deepseek_v3_debugmodel",
    "graph_trainer_deepseek_v3_debugmodel",
]

_DEBUGMODEL_CONTEXT_LENGTH = 2048


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


def graph_trainer_deepseek_v3_debugmodel() -> TpuGraphTrainer.Config:
  """DeepSeek-v3 debug model using TPU GraphTrainer (not yet onboarded: rung 2)."""
  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      deepseek_v3_debugmodel(), graph_trainer_model_registry
  )
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
  )
  return config
