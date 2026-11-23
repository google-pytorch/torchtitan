# Copyright 2026 Google LLC
"""TPU job configurations for Mamba-3 models."""

# pylint: disable=invalid-name,unused-argument

from torchtitan.components import checkpointer as checkpoint
from torchtitan.components import loss
from torchtitan.observability import metrics
from torchtitan.components import optim as optimizer
from torchtitan.components.optim import lr_scheduler
from torchtitan.config import configs
from torchtitan.distributed import activation_remat
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.experiments.tpu.components.datasets import c4_test_dataloader
from torchtitan.components.optim import Optim
from . import model
from . import parallelize


def _2_7b(ssm_backend: str = "pallas", **kwargs) -> model.Mamba3Config:
  return model.Mamba3Config(
      vocab_size=128256,
      d_model=2560,
      n_layers=64,
      n_heads=80,
      d_state=128,
      expand=2,
      chunk_size=256,
      ssm_backend=ssm_backend,
      **kwargs,
  )


mamba3_configs = {
    "2.7B": _2_7b,
}

_DEFAULT_MAX_CONTEXT_LENGTH = 262144


def model_registry(
    flavor: str,
    ssm_backend: str = "pallas",
    seq_len: int | None = None,
    **kwargs,
) -> model.Mamba3Config:
  del seq_len
  return mamba3_configs[flavor](ssm_backend=ssm_backend, **kwargs)


def mamba3_2_7b_256k() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=1),
      model=model_registry("2.7B"),
      dataloader=c4_test_dataloader(),
      optim=Optim.Config(
          optimizer=optimizer.default_adamw(lr=3e-4, implementation="foreach"),
          lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
              warmup_steps=100,
              decay_ratio=0.8,
              decay_type="linear",
              min_lr_factor=0.0,
          ),
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * 262144,
          max_context_length=262144,
          steps=1000,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpointer=None,
      activation_checkpoint=activation_remat.FullActivationRematConfig(),
  )
