from torchtitan.experiments.tpu.components.optim import default_adamw
import os
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optim.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optim.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.experiments.tpu.components.datasets import c4_test_dataloader
from torchtitan.components.tokenizer import HuggingFaceTokenizer
# NOTE: these are named directly rather than reached via
# `TPUTrainerConfig().<field>.__class__`. Upstream d771abb0 made
# `model_spec` a required field, so the no-arg `TPUTrainerConfig()` that
# idiom relied on now raises.
from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.tpu.tpu_job_config import (
    ConformerConfig,
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
    TpuTrainerCompileConfig,
)
import torchtitan.experiments.tpu.conformer as tpu_conformer
from torchtitan.experiments.tpu.conformer.infra.parallelize import parallelize_conformer
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.components.optim import Optim


def _conformer_model_config(flavor: str):
  return tpu_conformer.conformer_args[flavor]


def conformer_test() -> TPUTrainerConfig:
  """Conformer test model configuration for TPU."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="tests/assets/tokenizer",
      dump_folder=os.environ.get("TEST_TMPDIR", ".") + "/outputs",
      model=_conformer_model_config("test"),
      optim=Optim.Config(
          optimizer=default_adamw(lr=8e-4, implementation="foreach"),
          lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
      ),
      dataloader=c4_test_dataloader(),
      tokenizer=HuggingFaceTokenizer.Config(),
      validator=Validator.Config(),
precision=PrecisionConfig(param_dtype="bfloat16"),
            training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=768 * 512,
          steps=10,
          dtype="float32",
      ),
      parallelism=ParallelismConfig(
          data_parallel_shard_degree=-1,
          data_parallel_replicate_degree=1,
      ),
      activation_remat=FullActivationRematConfig(),
      compile=TpuTrainerCompileConfig(
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
  config.model = _conformer_model_config("debugmodel")
  config.training = config.training.__class__(
      num_tokens_per_microbatch_per_dp_rank=4 * 128,
      steps=3,
      dtype="float32",
      mixed_precision_param="bfloat16",
  )
  return config
