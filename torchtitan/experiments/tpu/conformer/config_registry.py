import os
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import (
    default_adamw,
    OptimizersContainer,
)
from torchtitan.components.validate import Validator
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.protocols.model_spec import ModelSpec
# NOTE: these are named directly rather than reached via
# `TPUTrainerConfig().<field>.__class__`. Upstream d771abb0 made
# `model_spec` a required field, so the no-arg `TPUTrainerConfig()` that
# idiom relied on now raises.
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.tpu.tpu_job_config import (
    ConformerConfig,
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
)
import torchtitan.experiments.tpu.conformer as tpu_conformer
from torchtitan.experiments.tpu.conformer.infra.parallelize import parallelize_conformer
from torchtitan.distributed.activation_checkpoint import FullAC


def _conformer_model_spec(flavor: str) -> ModelSpec:
  model_config = tpu_conformer.conformer_args[flavor]
  # Upstream d771abb0 made ModelSpec.max_context_length required. The conformer
  # config already declares its own context bound as `max_seq_len` (see
  # `ConformerModelArgs`, which also rewrites it from
  # `training.max_context_length` in `update_from_config`), so use that.
  return ModelSpec(
      name="conformer_tpu",
      flavor=flavor,
      model=model_config,
      max_context_length=model_config.max_seq_len,
      parallelize_fn=parallelize_conformer,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def conformer_test() -> TPUTrainerConfig:
  """Conformer test model configuration for TPU."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="tests/assets/tokenizer",
      dump_folder=os.environ.get("TEST_TMPDIR", ".") + "/outputs",
      model_spec=_conformer_model_spec("test"),
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
      dataloader=c4_test_dataloader(),
      tokenizer=HuggingFaceTokenizer.Config(),
      validator=Validator.Config(enable=True),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=768 * 512,
          max_context_length=512,
          steps=10,
          dtype="float32",
          mixed_precision_param="bfloat16",
      ),
      parallelism=ParallelismConfig(
          data_parallel_shard_degree=-1,
          data_parallel_replicate_degree=1,
      ),
      activation_checkpoint=FullAC.Config(),
      compile=CompileConfig(
          enable=False,
          backend="tpu",
          components=["model"],
      ),
      tpu_config=TPUConfig(
          use_simple_fsdp=True,
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
      ),
      conformer=ConformerConfig(
          use_ctc_loss=True,
      ),
  )


def conformer_debugmodel() -> TPUTrainerConfig:
  """Conformer debug model configuration for TPU."""
  config = conformer_test()
  config.model_spec = _conformer_model_spec("debugmodel")
  config.training = config.training.__class__(
      num_tokens_per_microbatch_per_dp_rank=4 * 128,
      max_context_length=128,
      steps=3,
      dtype="float32",
      mixed_precision_param="bfloat16",
  )
  return config
