# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""TPU job configurations for Mamba-2 models."""

# pylint: disable=invalid-name,unused-argument

# Upstream d771abb0 split `components/checkpoint.py` into the
# `components/checkpointer/` package and moved `components/lr_scheduler.py`
# under `components/optimizer/`. Alias both back to their historical local
# names so the `checkpoint.CheckpointManager` / `lr_scheduler.*` call sites
# below are unchanged.
from torchtitan.components import checkpointer as checkpoint
from torchtitan.components import loss
from torchtitan.components import metrics
from torchtitan.components import optimizer
from torchtitan.components.optimizer import lr_scheduler
from torchtitan.config import configs
from torchtitan.distributed import activation_checkpoint
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.experiments.tpu.components.datasets import c4_test_dataloader
from torchtitan.protocols import model_spec
from . import model
from . import parallelize


def _debugmodel(ssm_backend: str = "pallas", **kwargs) -> model.Mamba2Config:
  return model.Mamba2Config(
      vocab_size=2048,
      d_model=128,
      n_layers=2,
      n_heads=4,
      d_state=16,
      d_conv=4,
      expand=2,
      ssm_backend=ssm_backend,
  )


def _130m(ssm_backend: str = "pallas", **kwargs) -> model.Mamba2Config:
  return model.Mamba2Config(
      vocab_size=128256,
      d_model=768,
      n_layers=24,
      n_heads=24,
      d_state=16,
      d_conv=4,
      expand=2,
      ssm_backend=ssm_backend,
  )


def _370m(ssm_backend: str = "pallas", **kwargs) -> model.Mamba2Config:
  return model.Mamba2Config(
      vocab_size=128256,
      d_model=1024,
      n_layers=48,
      n_heads=32,
      d_state=16,
      d_conv=4,
      expand=2,
      ssm_backend=ssm_backend,
  )


def _780m(ssm_backend: str = "pallas", **kwargs) -> model.Mamba2Config:
  return model.Mamba2Config(
      vocab_size=128256,
      d_model=1536,
      n_layers=48,
      n_heads=48,
      d_state=16,
      d_conv=4,
      expand=2,
      ssm_backend=ssm_backend,
  )


def _1_3b(ssm_backend: str = "pallas", **kwargs) -> model.Mamba2Config:
  return model.Mamba2Config(
      vocab_size=128256,
      d_model=2048,
      n_layers=48,
      n_heads=64,
      d_state=16,
      d_conv=4,
      expand=2,
      ssm_backend=ssm_backend,
  )


def _2_7b(ssm_backend: str = "pallas", **kwargs) -> model.Mamba2Config:
  return model.Mamba2Config(
      vocab_size=128256,
      d_model=2560,
      n_layers=64,
      n_heads=80,
      d_state=16,
      d_conv=4,
      expand=2,
      ssm_backend=ssm_backend,
  )


mamba2_configs = {
    "debugmodel": _debugmodel,
    "130M": _130m,
    "130m_pallas": _130m,
    "370M": _370m,
    "780M": _780m,
    "1.3B": _1_3b,
    "2.7B": _2_7b,
}


# Mamba-2 is a state-space model: it has no positional embedding and no
# attention window, so no flavor has an architectural context limit and
# `Mamba2Config` carries no context field to read one from. The longest context
# any recipe in this file trains at is `mamba2_2_7b_256k`'s 262144, so declare
# that for every flavor; callers wanting a different bound pass `seq_len`.
_DEFAULT_MAX_CONTEXT_LENGTH = 262144


def model_registry(
    flavor: str,
    ssm_backend: str = "pallas",
    seq_len: int | None = None,
    **kwargs,
) -> model_spec.ModelSpec:
  config = mamba2_configs[flavor](ssm_backend=ssm_backend, **kwargs)
  # Upstream d771abb0 made ModelSpec.max_context_length required.
  return model_spec.ModelSpec(
      name="mamba2",
      flavor=flavor,
      model=config,
      max_context_length=seq_len or _DEFAULT_MAX_CONTEXT_LENGTH,
      parallelize_fn=parallelize.parallelize_mamba2,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def mamba2_debug() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry("debugmodel"),
      dataloader=c4_test_dataloader(),
      optimizer=optimizer.default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * 512,
          max_context_length=512,
          steps=10,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=10, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.SelectiveAC.Config(),
  )


def mamba2_130m() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=10),
      model_spec=model_registry("130M"),
      dataloader=c4_test_dataloader(),
      optimizer=optimizer.default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=16 * 2048,
          max_context_length=2048,
          steps=10000,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=1000, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.SelectiveAC.Config(),
  )


def mamba2_370m() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=10),
      model_spec=model_registry("370M"),
      dataloader=c4_test_dataloader(),
      optimizer=optimizer.default_adamw(lr=6e-4, implementation="foreach"),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * 2048,
          max_context_length=2048,
          steps=10000,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=1000, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.SelectiveAC.Config(),
  )


def mamba2_780m() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=10),
      model_spec=model_registry("780M"),
      dataloader=c4_test_dataloader(),
      optimizer=optimizer.default_adamw(lr=5e-4, implementation="foreach"),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=4 * 2048,
          max_context_length=2048,
          steps=10000,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=1000, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.SelectiveAC.Config(),
  )


def mamba2_1_3b() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=10),
      model_spec=model_registry("1.3B"),
      dataloader=c4_test_dataloader(),
      optimizer=optimizer.default_adamw(lr=4e-4, implementation="foreach"),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=2 * 2048,
          max_context_length=2048,
          steps=10000,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=1000, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.SelectiveAC.Config(),
  )


def mamba2_2_7b() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=10),
      model_spec=model_registry("2.7B"),
      dataloader=c4_test_dataloader(),
      optimizer=optimizer.default_adamw(lr=3e-4, implementation="foreach"),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=2 * 2048,
          max_context_length=2048,
          steps=10000,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=1000, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.SelectiveAC.Config(),
  )


def mamba2_2_7b_256k() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry("2.7B"),
      dataloader=c4_test_dataloader(),
      optimizer=optimizer.default_adamw(lr=3e-4, implementation="foreach"),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=100,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * 262144,
          max_context_length=262144,
          steps=1000,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=100, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.FullAC.Config(),
  )
