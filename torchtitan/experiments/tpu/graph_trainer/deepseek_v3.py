"""DeepSeek-V3 on the TPU GraphTrainer path.

Wraps upstream ``torchtitan.experiments.graph_trainer.deepseek_v3`` with the TPU
SimpleFSDP policy, as ``graph_trainer/qwen3.py`` does for Qwen3:

* Before upstream's SimpleFSDP setup, uneven weights move to an axis that
  divides the FSDP degree (``configure_deepseek_v3_simple_fsdp_shard_dims``).
* After it, the split meshes are built and the TPU collectives for non-leading
  shards are installed.
"""

import dataclasses

from torch import nn
from torchtitan.config import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    model_registry as graph_trainer_model_registry,
)
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
)
from torchtitan.experiments.graph_trainer.deepseek_v3.parallelize import (
    parallelize_deepseekv3 as graph_trainer_parallelize_deepseekv3,
)
from torchtitan.experiments.tpu.deepseek_v3.infra.parallelize import (
    configure_deepseek_v3_simple_fsdp_shard_dims,
)
from torchtitan.experiments.tpu.graph_trainer.simple_fsdp import (
    install_tpu_replicate_computation,
)
from torchtitan.protocols.model_spec import ModelSpec


def parallelize_deepseekv3(
    model: GraphTrainerDeepSeekV3Model,
    *,
    training: TrainingConfig,
    parallel_dims: ParallelDims,
    **kwargs,
) -> nn.Module:
  """Applies the TPU uneven-FSDP policy around upstream's setup."""
  if parallel_dims.tp_enabled or parallel_dims.pp_enabled:
    # Upstream's GraphTrainer TP goes through Module.parallelize (spmd_types),
    # which this fork does not use (see graph_trainer/muse_glimmer.py).
    raise NotImplementedError(
        "TPU GraphTrainer DeepSeek-V3 supports SimpleFSDP/HSDP only; TP and"
        " PP are not supported."
    )
  configure_deepseek_v3_simple_fsdp_shard_dims(model, parallel_dims)
  model = graph_trainer_parallelize_deepseekv3(
      model,
      training=training,
      parallel_dims=parallel_dims,
      **kwargs,
  )
  # Creates process groups, so every rank runs it here, right after sharding.
  for holder in model.modules():
    if hasattr(holder, "build_split_meshes"):
      holder.build_split_meshes()
  install_tpu_replicate_computation(model)
  return model


def model_registry(flavor: str) -> ModelSpec:
  """Upstream's GraphTrainer DeepSeek-V3 spec with the TPU parallelizer.

  ``TpuGraphTrainer.Config.from_tpu_trainer_config`` takes only the model
  class, ``parallelize_fn`` and ``pipelining_fn`` from this spec; the field
  values (TPU converters, ``sdpa`` attention, recipe edits) come from the
  recipe's own model config.
  """
  return dataclasses.replace(
      graph_trainer_model_registry(flavor),
      parallelize_fn=parallelize_deepseekv3,
  )

