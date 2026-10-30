# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.config.configs import (
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.tpu_job_config import (
    SplashAttentionKernelConfig,
    TPUConfig,
    TpuTrainerCompileConfig,
    TPUTrainerConfig,
)
# Imported from the defining submodules rather than the
# `torchtitan.components.data` package: strict-deps checking cannot tell a
# package attribute from an unlisted submodule, so the package-level form
# is reported as an ambiguous import.
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import FirstFitPackingConfig
from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.hf_datasets.text_datasets import ChatProcessor
from torchtitan.experiments.graph_trainer.llama3 import (
    model_registry as graph_trainer_model_registry,
)
from torchtitan.experiments.tpu.graph_trainer.config import (
    TpuGraphTrainerCompileConfig,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.llama3.infra.parallelize import parallelize_llama as tpu_parallelize_llama
from torchtitan.models.llama3.config_registry import model_registry


# Llama3's `ComplexRoPE` calls `view_as_complex`, which cannot be traced inside
# a higher-order op such as `scan`; see `overrides/real_rope.py`. The rewrite is
# bit-identical and perf-neutral, so every recipe carries it.
# Upstream d771abb0 identifies an override by the import path of its decorated
# `@override` factory, so each target is `module.function`, not just `module`.
_REAL_ROPE_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.real_rope"
    ".tpu_real_arithmetic_complex_rope"
)
_FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"
)

_LLAMA3_SCAN_FLAGS = tpu_flags.merge_flags(
    tpu_flags.TPU7X_THROUGHPUT_P_STATE,
    {
        "xla_tpu_scoped_vmem_limit_kib": 131072,
        "xla_tpu_enable_offloading_copy_to_sparsecore": False,
        "xla_tpu_copy_insertion_use_region_analysis": True,
        "xla_tpu_copy_insertion_use_region_analysis_limit": 1000,
    },
    tpu_flags.HIERARCHICAL_REDUCE_SCATTER,
    tpu_flags.SELECTIVE_ICI_RS_PIPELINING,
    tpu_flags.ICI_AG_PIPELINING,
    tpu_flags.SPARSECORE_ASYNC_ALL_GATHER,
    tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
    tpu_flags.CONVOLUTION_UNFUSING,
)


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Select the fused loss and its pre-construction LM-head override together."""
  config.loss = PallasCrossEntropyLoss.Config()
  config.override.imports.append(_FUSED_LOSS_OVERRIDE)


def _on_scan_lane(config: TPUTrainerConfig) -> TpuGraphTrainer.Config:
  """Move a recipe onto the graph trainer with scanned layers and host offload.

  Offload is not optional above 8B: without it the scan bodies keep every
  layer's input live and 405B does not compile.
  """
  graph_config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      config, graph_trainer_model_registry
  )
  graph_config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
      memory_policy="full_and_scan_offload",
      use_scan_layers=True,
  )
  # bf16 Adam moments halve the optimizer state, which is 12.7 GiB at 405B and
  # the difference between sequence 8192 fitting and not.
  graph_config.optimizer.implementation = "fused_opt_states_bf16"
  # Disable Shardy HLO sharding v3 (enabled by default in XLA cl/979222288 /
  # libtpu 0.0.48; see cl/979992733), which segfaults in
  # mlir::sdy::buildTensorFactorShardings on GraphTrainer.
  graph_config.xla_flags["xla_enable_hlo_sharding_v3"] = "false"
  graph_config.libtpu_init_flags = tpu_flags.merge_flags(
      graph_config.libtpu_init_flags,
      _LLAMA3_SCAN_FLAGS,
  )
  return graph_config


def llama3_debugmodel() -> TPUTrainerConfig:
  """Llama3 debug model configuration for TPU."""
  model_spec = model_registry("debugmodel", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  config = TPUTrainerConfig(
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
          num_tokens_per_microbatch_per_dp_rank=8 * 2048,
          max_context_length=2048,
          steps=10,
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          pipeline_parallel_schedule="Interleaved1F1B"
      ),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(),
      validator=Validator.Config(
          freq=5,
          steps=10,
      ),
  )
  config.override.imports.append(_REAL_ROPE_OVERRIDE)
  return config


def graph_trainer_llama3_debugmodel() -> TpuGraphTrainer.Config:
  """Llama3 debug model using TPU GraphTrainer."""
  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      llama3_debugmodel(), graph_trainer_model_registry
  )
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
  )
  return config


def llama3_1b() -> TPUTrainerConfig:
  """Llama3 1B model configuration for TPU."""
  model_spec = model_registry("1B", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  config = TPUTrainerConfig(
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
          num_tokens_per_microbatch_per_dp_rank=4 * 128,
          max_context_length=128,
          steps=3,
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          pipeline_parallel_schedule="Interleaved1F1B"
      ),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(),
      validator=Validator.Config(
          freq=5,
          steps=10,
      ),
  )
  config.override.imports.append(_REAL_ROPE_OVERRIDE)
  return config


def llama3_8b() -> TPUTrainerConfig:
  """Llama3 8B model configuration for TPU."""
  model_spec = model_registry("8B", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  config = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./assets/hf/Llama-3.1-8B",
      model_spec=model_spec,
      optimizer=default_adamw(lr=3e-4, eps=1e-8, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=200,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=1 * 8192,
          max_context_length=8192,
          steps=1000,
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=5),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          enable_tpu_topo_mesh=True,
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=True,
      ),
      activation_checkpoint=FullAC.Config(),
      validator=Validator.Config(
          enable=False,
          freq=500,
          steps=1200,
          dataloader=c4_test_dataloader(),
      ),
      tpu_config=TPUConfig(
          eager_mode="DEFER_AND_FUSE",
      ),
      compile=TpuTrainerCompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_q_layout="SEQ_MINOR",
          sa_k_layout="SEQ_MINOR",
          sa_block_q=2048,
          sa_block_kv=2048,
          sa_block_dkv=2048,
          sa_block_kv_compute=2048,
          sa_block_q_dkv=2048,
          sa_block_kv_dkv=2048,
          sa_block_kv_dkv_compute=2048,
          sa_block_q_dq=2048,
          sa_block_kv_dq=2048,
      ),
  )
  config.override.imports.append(_REAL_ROPE_OVERRIDE)
  _enable_pallas_cross_entropy(config)
  return config


def llama3_70b() -> TPUTrainerConfig:
  """Llama3 70B model configuration for TPU."""
  model_spec = model_registry("70B", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  config = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=1.5e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=200,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=1 * 8192,
          max_context_length=8192,
          steps=1000,
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=5),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          enable_tpu_topo_mesh=True,
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=True,
      ),
      activation_checkpoint=FullAC.Config(),
      validator=Validator.Config(
          enable=False,
          freq=500,
          steps=1200,
          dataloader=c4_test_dataloader(),
      ),
      tpu_config=TPUConfig(
          eager_mode="DEFER_AND_FUSE",
      ),
      compile=TpuTrainerCompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_q_layout="SEQ_MINOR",
          sa_k_layout="SEQ_MINOR",
          sa_block_q=2048,
          sa_block_kv=2048,
          sa_block_dkv=2048,
          sa_block_kv_compute=2048,
          sa_block_q_dkv=2048,
          sa_block_kv_dkv=2048,
          sa_block_kv_dkv_compute=2048,
          sa_block_q_dq=2048,
          sa_block_kv_dq=2048,
      ),
  )
  config.override.imports.append(_REAL_ROPE_OVERRIDE)
  _enable_pallas_cross_entropy(config)
  return config


def llama3_405b() -> TPUTrainerConfig:
  """Llama3 405B model configuration for TPU."""
  model_spec = model_registry("405B", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  config = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=8e-5, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=600,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=1 * 8192,
          max_context_length=8192,
          steps=1000,
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=5),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          enable_tpu_topo_mesh=True,
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=True,
      ),
      activation_checkpoint=FullAC.Config(),
      validator=Validator.Config(
          enable=False,
          freq=500,
          steps=1200,
          dataloader=c4_test_dataloader(),
      ),
      tpu_config=TPUConfig(
          eager_mode="DEFER_AND_FUSE",
      ),
      compile=TpuTrainerCompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_q_layout="SEQ_MINOR",
          sa_k_layout="SEQ_MINOR",
          sa_block_q=2048,
          sa_block_kv=2048,
          sa_block_dkv=2048,
          sa_block_kv_compute=2048,
          sa_block_q_dkv=2048,
          sa_block_kv_dkv=2048,
          sa_block_kv_dkv_compute=2048,
          sa_block_q_dq=2048,
          sa_block_kv_dq=2048,
      ),
  )
  config.override.imports.append(_REAL_ROPE_OVERRIDE)
  _enable_pallas_cross_entropy(config)
  return config


def sft_llama3_8b_metamath() -> TPUTrainerConfig:
  """Llama3 8B SFT on MetaMathQA for TPU."""

  def process_sample(sample):
    # Required by ChatDataset: maps this dataset's columns onto chat messages.
    # `response` holds the full chain-of-thought and the final answer, so it
    # passes through whole -- splitting reasoning into its own key risks a chat
    # template that does not reference that key silently dropping it.
    return [
        {"role": "user", "content": sample["query"]},
        {"role": "assistant", "content": sample["response"]},
    ]

  model_spec = model_registry("8B", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  config = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./assets/hf/Llama-3.1-8B",
      model_spec=model_spec,
      optimizer=default_adamw(lr=2e-5, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=25,
          decay_ratio=0.9,
          decay_type="cosine",
          min_lr_factor=0.1,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=2 * 8192,
          max_context_length=8192,
          steps=500,
      ),
      dataloader=GrainDataLoader.Config(
          dataset=FirstFitPackingConfig(
              dataset=SingleDatasetConfig(
                  source=HuggingFaceRandomAccessSource.Config(
                      path="meta-math/MetaMathQA",
                      split="train",
                  ),
                  processor=ChatProcessor.Config(messages_fn=process_sample),
                  post_filters=(lambda sample: sample is not None,),
              ),
          ),
      ),
      metrics=MetricsProcessor.Config(
          log_freq=5,
          enable_tensorboard=True,
          save_tb_folder="tb_logs/sft_llama3_8b_metamath",
      ),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          enable_tpu_topo_mesh=True,
      ),
      checkpoint=CheckpointManager.Config(
          # `enable` is on only so the initial HF weight load happens;
          # `load_only` then suppresses every save, including the
          # unconditional last-step one.
          enable=True,
          load_only=True,
          initial_load_in_hf=True,
          initial_load_model_only=True,
      ),
      activation_checkpoint=FullAC.Config(),
      tpu_config=TPUConfig(
          eager_mode="DEFER_AND_FUSE",
      ),
      compile=TpuTrainerCompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_q_layout="SEQ_MINOR",
          sa_k_layout="SEQ_MINOR",
          sa_block_q=2048,
          sa_block_kv=2048,
          sa_block_dkv=2048,
          sa_block_kv_compute=2048,
          sa_block_q_dkv=2048,
          sa_block_kv_dkv=2048,
          sa_block_kv_dkv_compute=2048,
          sa_block_q_dq=2048,
          sa_block_kv_dq=2048,
      ),
  )
  config.override.imports.append(_REAL_ROPE_OVERRIDE)
  _enable_pallas_cross_entropy(config)
  return config


def llama3_8b_scan() -> TpuGraphTrainer.Config:
  """Llama3 8B on the scan lane."""
  return _on_scan_lane(llama3_8b())


def llama3_70b_scan() -> TpuGraphTrainer.Config:
  """Llama3 70B on the scan lane."""
  return _on_scan_lane(llama3_70b())


def llama3_405b_scan() -> TpuGraphTrainer.Config:
  """Llama3 405B on the scan lane. Layerwise does not compile at this size."""
  return _on_scan_lane(llama3_405b())


def _with_float8(
    config: TPUTrainerConfig, flavor: str, recipe_name: str = "rowwise"
) -> TPUTrainerConfig:
  """Swaps `config`'s model for one with torchao Float8Linear layers.

  Uses `TpuFloat8LinearConverter`, which builds a TPU-compatible torchao config
  (see `float8_converter.py` for how it differs from the CUDA converter).
  """

  # Imported here so that only float8 configs require torchao.
  from torchtitan.experiments.tpu.float8_converter import TpuFloat8LinearConverter  # pylint: disable=g-import-not-at-top

  model_spec = model_registry(
      flavor,
      attn_backend="sdpa",
      converters=[
          TpuFloat8LinearConverter.Config(
              recipe_name=recipe_name,
              filter_fqns=["lm_head", "output"],
          )
      ],
  )
  # model_registry installs the GPU parallelize_fn; restore the TPU one.
  model_spec.parallelize_fn = config.model_spec.parallelize_fn
  config.model_spec = model_spec
  return config


def llama3_70b_float8() -> TPUTrainerConfig:
  """`llama3_70b` with torchao rowwise float8 linears."""
  return _with_float8(llama3_70b(), "70B")
