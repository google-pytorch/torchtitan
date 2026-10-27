"""Parallelization utilities for Conformer on TPU."""

import enum
import torch
from torch.distributed import fsdp
from torch.distributed._composable import replicate
import torch.nn as nn
import torchtitan.config
import torchtitan.distributed
import torchtitan.trainer
from torchtitan.experiments.graph_trainer import simple_fsdp
from torchtitan.experiments.tpu import tpu_job_config
import torchtitan.experiments.tpu.utils as tpu_utils
from torchtitan.tools import logging


class ParallelStrategy(enum.Enum):
  # FSDP
  FSDP2 = enum.auto()
  SIMPLEFSDP = enum.auto()
  # DDP
  DDP = enum.auto()
  DDP_W_FSDP2 = enum.auto()  # FSDP2 with replicate mode
  DDP_W_SIMPLEFSDP = enum.auto()  # SimpleFSDP with replicate mode
  # No parallelism
  NONE = enum.auto()


def _determine_parallel_strategy(
    parallel_dims: torchtitan.distributed.ParallelDims,
    train_config: torchtitan.trainer.Trainer.Config | tpu_job_config.TPUTrainerConfig,
) -> ParallelStrategy:
  """Determine parallel strategy based on config."""
  fsdp_enabled = parallel_dims.fsdp_enabled
  ddp_enabled = parallel_dims.dp_replicate_enabled

  use_simple = False
  amp = True

  if isinstance(train_config, tpu_job_config.TPUTrainerConfig):
    use_simple = train_config.tpu_config.use_simple_fsdp
    amp = train_config.tpu_config.enable_amp

  if fsdp_enabled:
    return ParallelStrategy.SIMPLEFSDP if use_simple else ParallelStrategy.FSDP2

  if ddp_enabled:
    if use_simple:
      return ParallelStrategy.DDP_W_SIMPLEFSDP
    if amp:
      return ParallelStrategy.DDP_W_FSDP2
    return ParallelStrategy.DDP

  return ParallelStrategy.NONE


def parallelize_conformer(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    train_config: torchtitan.trainer.Trainer.Config | tpu_job_config.TPUTrainerConfig,
) -> nn.Module:
  """Apply parallelism and activation checkpointing to Conformer."""

  if parallel_dims.tp_enabled:
    raise RuntimeError("Tensor Parallelism is not supported for Conformer yet")

  # Apply activation checkpointing if enabled
  if train_config.activation_checkpoint is not None:
    apply_ac(model)

  strategy = _determine_parallel_strategy(parallel_dims, train_config)
  logging.logger.info(f"Using parallel strategy: {strategy}")

  if strategy == ParallelStrategy.NONE:
    return model

  # Resolve DeviceMesh and mode based on strategy
  if strategy in (ParallelStrategy.FSDP2, ParallelStrategy.SIMPLEFSDP):
    if parallel_dims.dp_replicate_enabled:
      dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
      reshard = not parallel_dims.pp_enabled
      dp_mode = "hybrid_shard"
    else:
      dp_mesh = parallel_dims.get_mesh(["fsdp"])
      reshard = not parallel_dims.pp_enabled
      dp_mode = "fully_shard"
  elif strategy in (
      ParallelStrategy.DDP,
      ParallelStrategy.DDP_W_FSDP2,
      ParallelStrategy.DDP_W_SIMPLEFSDP,
  ):
    dp_mesh = parallel_dims.get_mesh("dp_replicate")
    reshard = False
    dp_mode = "replicate"
  else:
    dp_mesh = None
    reshard = False
    dp_mode = None

  fsdp_config = {}
  if dp_mesh:
    fsdp_config["mesh"] = dp_mesh

  enable_amp = True
  if isinstance(train_config, tpu_job_config.TPUTrainerConfig):
    enable_amp = train_config.tpu_config.enable_amp

  if enable_amp:
    fsdp_config["mp_policy"] = fsdp.MixedPrecisionPolicy(
        param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            train_config.training.mixed_precision_param
        ],
        reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            train_config.training.mixed_precision_reduce
        ],
    )

  if hasattr(model, "model"):
    encoder = model.model
  else:
    encoder = model

  # Apply compilation if enabled
  model_compile_enabled = (
      train_config.compile.enable and "model" in train_config.compile.components
  )
  compile_mode = "layer"
  if isinstance(train_config, tpu_job_config.TPUTrainerConfig):
    compile_mode = train_config.compile.mode

  if model_compile_enabled:
    model = apply_compile(model, train_config)

  if strategy == ParallelStrategy.DDP:
    logging.logger.info("Applying Native DDP (replicate) to the model")
    replicate.replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)
    return model

  if strategy in (
      ParallelStrategy.SIMPLEFSDP,
      ParallelStrategy.DDP_W_SIMPLEFSDP,
  ):
    mp_policy = None
    if enable_amp:
      mp_policy = simple_fsdp.MixedPrecisionPolicy(
          param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
              train_config.training.mixed_precision_param
          ],
          reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
              train_config.training.mixed_precision_reduce
          ],
      )
    model = simple_fsdp.data_parallel(
        model,
        dp_mesh,
        mode=dp_mode,
        mp_policy=mp_policy,
    )
    logging.logger.info(f"Applied Simple FSDP (dp mode={dp_mode}) to the model")
    return model

  # Shard each individual Conformer layer (only if not compiling whole model)
  if not (model_compile_enabled and compile_mode == "whole"):
    if hasattr(encoder, "layers"):
      layers = encoder.layers
    elif hasattr(encoder, "conformer_layers"):
      layers = encoder.conformer_layers
    else:
      layers = None

    if layers is not None:
      logging.logger.info(f"Sharding Conformer layers (FSDP, mode={dp_mode}).")
      for layer in layers.children():
        fsdp.fully_shard(layer, **fsdp_config, reshard_after_forward=reshard)
    else:
      logging.logger.warning("No layers found to shard.")

    # Shard the full model (if it's a separate module)
    if encoder is not model:
      fsdp.fully_shard(encoder, **fsdp_config, reshard_after_forward=reshard)

  # Shard the outer wrapper (which is the compiled model if whole compiled)
  fsdp.fully_shard(model, **fsdp_config)

  logging.logger.info("Applied FSDP to the model")

  return model


def apply_compile(
    model: nn.Module, train_config: torchtitan.trainer.Trainer.Config
) -> nn.Module:
  """Apply torch.compile to layers or whole model."""
  compile_mode = "layer"
  if isinstance(train_config, tpu_job_config.TPUTrainerConfig):
    compile_mode = train_config.compile.mode

  if hasattr(model, "model"):
    encoder = model.model
  else:
    encoder = model

  if hasattr(encoder, "layers"):
    layers = encoder.layers
  elif hasattr(encoder, "conformer_layers"):
    layers = encoder.conformer_layers
  else:
    layers = None

  if compile_mode == "whole":
    if hasattr(model, "model"):
      logging.logger.info("Applying torch.compile to Whole Model (via model.model).")
      model.model = torch.compile(
          encoder,
          backend=train_config.compile.backend,
          fullgraph=True,
          dynamic=False,
      )
      return model
    else:
      logging.logger.info("Applying torch.compile to Whole Model.")
      return torch.compile(
          model,
          backend=train_config.compile.backend,
          fullgraph=True,
          dynamic=False,
      )
  elif compile_mode == "layer" and layers is not None:
    logging.logger.info("Applying torch.compile to Conformer layers.")
    for layer_id, layer in layers.named_children():
      compiled = torch.compile(
          layer,
          backend=train_config.compile.backend,
          fullgraph=True,
          dynamic=False,
      )
      layers.register_module(layer_id, compiled)
    return model
  else:
    logging.logger.warning(
        f"Compile mode {compile_mode} not supported or no layers found."
    )
    return model


def apply_ac(model: nn.Module) -> None:
  """Apply activation checkpointing."""
  logging.logger.info("Applying activation checkpointing to Conformer layers.")

  if hasattr(model, "model"):
    encoder = model.model
  else:
    encoder = model

  if hasattr(encoder, "layers"):
    layers = encoder.layers
  elif hasattr(encoder, "conformer_layers"):
    layers = encoder.conformer_layers
  else:
    layers = None

  if layers is not None:
    for layer_id, layer in layers.named_children():
      wrapped_layer = tpu_utils.ptd_checkpoint_wrapper_with_early_stop(
          layer, preserve_rng_state=False
      )
      layers.register_module(layer_id, wrapped_layer)
  else:
    logging.logger.warning("No layers found to apply AC.")
