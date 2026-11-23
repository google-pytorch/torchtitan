from torchtitan.config.parallelism import ParallelismConfig
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.components.optim.lr_scheduler import LRSchedulersContainer
from torchtitan.experiments.tpu.components.optim import default_adamw
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.config.configs import (
    TrainingConfig,
)
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
)
from torchtitan.experiments.tpu.graph_trainer.config import (
    TpuGraphTrainerCompileConfig,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.components.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.deepseek_v3 import model_registry
from torchtitan.components.optim import Optim


def deepseek_v3_debugmodel() -> TPUTrainerConfig:
  """DeepSeek-v3 debug model configuration for TPU."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=MetricsProcessor.Config(log_freq=1),
      model=model_registry("debugmodel", attn_backend="sdpa"),
      dataloader=c4_test_dataloader(),
      optim=Optim.Config(
          optimizer=default_adamw(lr=8e-4, implementation="foreach"),
          lr_scheduler=LRSchedulersContainer.Config(
              warmup_steps=2,
              decay_ratio=0.8,
              decay_type="linear",
              min_lr_factor=0.0,
          ),
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * 2048,
          steps=10,
      ),
      parallelism=ParallelismConfig(
          expert_parallel_degree=1,
      ),
      checkpointer=None,
      activation_remat=DefaultActivationRematConfig(),
  )


def graph_trainer_deepseek_v3_debugmodel() -> TpuGraphTrainer.Config:
  """DeepSeek-v3 debug model using TPU GraphTrainer."""
  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      deepseek_v3_debugmodel(), GraphTrainerDeepSeekV3Model.Config
  )
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
  )
  return config
