# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer compatible with TorchTpu."""

import contextlib
import dataclasses
import json
import logging
from typing import Any

import os
import spmd_types as spmd
import torch
import torch.distributed as dist
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.config.configurable import Configurable
from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu.components.grad_norm import CompiledGradNorm
import torchtitan.experiments.tpu.profiling.profiler_workaround as profiler_workaround
import torchtitan.experiments.tpu.deepseek_v3   # trigger model registration
import torchtitan.experiments.tpu.flux  # trigger model registration
import torchtitan.experiments.tpu.examples.gemma4  # trigger model registration
from torchtitan.experiments.tpu.graph_trainer.scan_guards import check_scan_was_applied
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
import torchtitan.experiments.tpu.llama3  # trigger model registration
import torchtitan.experiments.tpu.muse_glimmer  # trigger model registration
import torchtitan.experiments.tpu.qwen3   # trigger model registration
import torchtitan.experiments.tpu.qwen3_5  # trigger model registration
import torchtitan.experiments.tpu.tpu_job_config as tpu_job_config_module
from torchtitan.experiments.tpu.utils import with_split_optimizers_by_mesh
from torchtitan.observability.logging import init_logger
from torchtitan.protocols.model_spec import ModelSpec
import torchtitan.train
import torchtitan.trainer
from torchtitan.training_engine import TrainingEngine

logger = logging.getLogger(__name__)


class TpuTrainingEngine(TrainingEngine):
  """Training engine with TPU-specific optimizations.

  Adds the following optional behaviors controlled by ``compile`` policy:

  compile.materialize_forward_before_backward:
    Inserts a `synchronize(loss, wait=False)` call between the forward+loss
    computation and the backward pass. This splits the monolithic XLA
    computation graph into two smaller graphs (forward and backward),
    significantly reducing first-step XLA compilation time.
    The torch_tpu synchronize() materializes the forward graph without
    breaking autograd, so loss.backward() still propagates gradients
    correctly through the full model.

  compile.compile_grad_norm:
    Compiles the DTensor-aware global gradient norm as a standalone static XLA
    graph. In-place clipping remains on PyTorch's eager path.

  compile.optimizer_compile='separate_graph':
    Runs ``optimizers.step()`` as one separately compiled XLA graph.
  """

  def __init__(
      self,
      config: Configurable.Config,
      *,
      model_config: Any,
      max_num_documents: int | None,
      output_dir: str,
  ) -> None:
    super().__init__(
        config,
        model_config=model_config,
        max_num_documents=max_num_documents,
        output_dir=output_dir,
    )
    self._is_tpu_config = isinstance(
        config, tpu_job_config_module.TPUTrainerConfig
    )
    self._compiled_grad_norm: CompiledGradNorm | None = None
    self._materialize_forward_before_backward = False
    if self._is_tpu_config:
      if not config.tpu_config.enable_amp:
        logger.info("AMP is disabled, using uniform precision training.")
      self._compiled_grad_norm = (
          CompiledGradNorm() if config.compile.compile_grad_norm else None
      )
      self._materialize_forward_before_backward = (
          config.compile.materialize_forward_before_backward
          and not self.parallel_dims.pp_enabled
      )

  def _initialize_model(
      self,
      model_spec: ModelSpec,
      *,
      compile_config: Any,
      create_seed_checkpoint: bool = False,
  ) -> None:
    super()._initialize_model(
        model_spec,
        compile_config=compile_config,
        create_seed_checkpoint=create_seed_checkpoint,
    )
    if self._is_tpu_config:
      # The base engine builds `loss_fn` from `config.loss` (and, under
      # pipeline parallelism, binds it into the PP schedule); the TPU recipes
      # keep their legacy loss-kernel selection on top of that.
      from torchtitan.experiments.tpu.components.loss import build_cross_entropy_loss  # pylint: disable=g-import-not-at-top

      self.loss_fn = build_cross_entropy_loss(self.config)

  def _initialize_optimizer(self, model_spec: ModelSpec) -> None:
    super()._initialize_optimizer(model_spec)
    if (
        self._is_tpu_config
        and self.config.compile.optimizer_compile == "separate_graph"
    ):
      from torchtitan.experiments.tpu.compiled_optimizers.separate_graph import apply_compiled_optimizer  # pylint: disable=g-import-not-at-top

      apply_compiled_optimizer(self)

  def _clip_grad_norm(self) -> torch.Tensor:
    if self._compiled_grad_norm is None:
      return super()._clip_grad_norm()
    parameters = [p for m in self.model_parts for p in m.parameters()]
    return self._compiled_grad_norm.clip_grad_norm_(
        parameters,
        self.config.training.max_norm,
    )

  def forward_backward_microbatch(
      self,
      *,
      microbatch_group: list[TrainingMicrobatch],
      global_valid_tokens: torch.Tensor,
      accumulation_index: int = 0,
  ) -> torch.Tensor:
    from torchtitan.experiments.tpu.components.unflatten_shim import set_unflatten_seq_len  # pylint: disable=g-import-not-at-top

    with set_unflatten_seq_len(self.config.training.max_context_length):
      return super().forward_backward_microbatch(
          microbatch_group=microbatch_group,
          global_valid_tokens=global_valid_tokens,
          accumulation_index=accumulation_index,
      )

  def _non_pp_forward_backward_body(
      self,
      *,
      inputs: torch.Tensor | tuple[torch.Tensor, ...],
      labels: torch.Tensor | tuple[torch.Tensor, ...],
      model_kwargs: dict[str, Any],
      loss_kwargs: dict[str, Any],
  ) -> torch.Tensor:
    if not self._materialize_forward_before_backward:
      return super()._non_pp_forward_backward_body(
          inputs=inputs,
          labels=labels,
          model_kwargs=model_kwargs,
          loss_kwargs=loss_kwargs,
      )

    # Materialize forward/backward separately for the regular TPUTrainer:
    # The forward pass and loss computation are materialized as one XLA graph,
    # and the backward pass is compiled separately. This reduces peak
    # compilation time vs. one monolithic forward+backward graph.
    from torch_tpu._internal.sync import synchronize  # pylint: disable=g-import-not-at-top

    assert len(self.model_parts) == 1
    with self.train_context():
      pred = self.model_parts[0](inputs, **model_kwargs)
      loss, self.loss_metrics = self.loss_fn(
          pred,
          labels,  # pyrefly: ignore[bad-argument-type]
          **loss_kwargs,
      )
      del pred

      # Materialize the forward graph without waiting for execution to
      # complete. This splits the deferred-op graph so the backward pass
      # compiles as a separate (smaller) XLA program. Autograd state is
      # unaffected — loss.backward() can still propagate gradients.
      synchronize(loss, wait=False)

      with spmd.no_typecheck():
        loss.backward()
    return loss


class TPUTrainer(torchtitan.trainer.Trainer):
  """Trainer subclass with TPU-specific optimizations.

  The training loop is upstream's; the TPU behaviors live in
  :class:`TpuTrainingEngine`.
  """

  engine_cls = TpuTrainingEngine
  engine: TpuTrainingEngine

  def __init__(self, job_config: Configurable.Config):
    if isinstance(job_config, tpu_job_config_module.TPUTrainerConfig):
      # The engine builds `loss_fn` from `job_config.loss` (and, under
      # pipeline parallelism, binds it into the PP schedule), so fill
      # `global_vocab_size` from the final model config first.
      from torchtitan.experiments.tpu.components.loss import set_global_vocab_size  # pylint: disable=g-import-not-at-top

      if job_config.model_spec is not None:
        set_global_vocab_size(job_config.loss, job_config.model_spec.model)
    # Under EP, split optimizer param groups by device mesh; the engine runs
    # this hook before it builds the LR schedulers.
    model_spec = getattr(job_config, "model_spec", None)
    if model_spec is not None:
      # The eager (non-scan) step runs the SwiGLU gate/up projections as two
      # GEMMs, over a [2F, D] w13 that FSDP shards on dim 0 unless tensor
      # parallelism needs upstream's stacked [2, F, D]; see
      # TpuPackedFeedForward and TpuTwoProjFeedForward.
      from torchtitan.experiments.tpu.components.ffn import TpuTwoProjFeedForwardConverter  # pylint: disable=g-import-not-at-top

      parallelism = getattr(job_config, "parallelism", None)
      tp_degree = (
          parallelism.tensor_parallel_degree if parallelism is not None else 1
      )
      job_config.model_spec = dataclasses.replace(
          model_spec,
          model=TpuTwoProjFeedForwardConverter.Config(packed=tp_degree <= 1)
          .build()
          .convert(model_spec.model),
          post_optimizer_build_fn=with_split_optimizers_by_mesh(
              model_spec.post_optimizer_build_fn
          ),
      )
    super().__init__(job_config)


def _apply_flux_img_size(
    train_config: tpu_job_config_module.TPUTrainerConfig,
) -> None:
  """Applies `--flux.img_size` to the dataset processors and context length."""
  img_size = train_config.flux.img_size
  if img_size is None:
    return
  import dataclasses  # pylint: disable=g-import-not-at-top
  from torchtitan.models.flux import utils as flux_utils  # pylint: disable=g-import-not-at-top

  def _with_img_size(dataset):
    if hasattr(dataset, "processor"):
      return dataclasses.replace(
          dataset,
          processor=dataclasses.replace(dataset.processor, img_size=img_size),
      )
    if hasattr(dataset, "dataset"):  # e.g. FluxValidationDatasetConfig
      return dataclasses.replace(dataset, dataset=_with_img_size(dataset.dataset))
    return dataset

  dataloader = train_config.dataloader
  train_config.dataloader = dataclasses.replace(
      dataloader, dataset=_with_img_size(dataloader.dataset)
  )
  validator = getattr(train_config, "validator", None)
  if validator is not None and getattr(validator, "dataloader", None):
    validator.dataloader = dataclasses.replace(
        validator.dataloader,
        dataset=_with_img_size(validator.dataloader.dataset),
    )
  latent = img_size // flux_utils.IMAGE_LATENT_SIZE_RATIO
  train_config.training.max_context_length = (
      latent // flux_utils.PATCH_HEIGHT
  ) * (latent // flux_utils.PATCH_WIDTH) + (
      train_config.tokenizer.max_t5_encoding_len
  )
  logger.info(
      f"--flux.img_size={img_size}: max_context_length="
      f"{train_config.training.max_context_length}"
  )


def _fit_rope_to_context(model_config: Any, max_context_length: int) -> None:
  """Sizes every RoPE cache to `max_context_length` (pre-d771abb0 behaviour).

  Upstream d771abb0 dropped the resize from `Decoder.Config.update_from_config`
  and expects callers to pass `seq_len=` to `model_registry`; the TPU configs do
  not, so caches were built for the flavor's full context (e.g. 131072 for
  llama3), which ran llama3-70b on tpu7x-2x2x4 out of program HBM.
  """
  import dataclasses  # pylint: disable=g-import-not-at-top
  from torchtitan.models.common.rope import RoPE  # pylint: disable=g-import-not-at-top

  if model_config is None or max_context_length <= 0:
    return
  for _, rope, parent, attr in model_config.traverse(RoPE.Config):
    if parent is None or rope.max_context_length <= max_context_length:
      continue
    new_rope = dataclasses.replace(rope, max_context_length=max_context_length)
    if isinstance(parent, list):
      parent[attr] = new_rope
    else:
      setattr(parent, attr, new_rope)


def start_trainer(
    train_config: tpu_job_config_module.TPUTrainerConfig
    | TpuGraphTrainer.Config,
):
  """Starts the training process."""
  rank = int(os.getenv("RANK", "0"))
  if rank == 0:
    init_logger()

  from torchtitan.experiments.tpu.distributed import cloud_distributed_utils  # pylint: disable=g-import-not-at-top
  from torchtitan.experiments.tpu import workarounds  # pylint: disable=g-import-not-at-top

  tpu_job_config_module.validate_compile_components(train_config.compile)
  workarounds.raise_dynamo_recompile_limits(train_config.model_spec.model)
  profiler_workaround.init(train_config)

  # Reject a replicate degree that would straddle DCN, before ParallelDims is
  # built from this config. No-op on a single slice.
  cloud_distributed_utils.maybe_validate_multislice_parallelism(
      train_config.parallelism
  )
  if cloud_distributed_utils.is_multi_slice():
    # The MegaScale PJRT client cannot serve pinned host memory.
    workarounds.disable_pinned_host_memory()

  trainer: torchtitan.trainer.Trainer | None = None
  model_spec = train_config.model_spec
  assert model_spec is not None
  model_name = model_spec.name
  if model_name != "flux":
    _fit_rope_to_context(
        model_spec.model, train_config.training.max_context_length
    )
  is_graph_trainer = isinstance(train_config, TpuGraphTrainer.Config)
  if model_name == "flux":
    # FluxTrainer owns a separate train_step and bypasses the TPU integration.
    if train_config.compile.compile_grad_norm:
      raise ValueError("compile.compile_grad_norm does not support FluxTrainer")
    from torchtitan.models.flux.trainer import FluxTrainer
    _apply_flux_img_size(train_config)
    trainer_cls = FluxTrainer
  elif is_graph_trainer:
    trainer_cls = TpuGraphTrainer
  else:
    if isinstance(train_config, tpu_job_config_module.TPUTrainerConfig):
      if train_config.compile.mode == "fullgraph":
        logger.warning(
            "Running fullgraph compile mode (`%s`) on regular TPUTrainer. "
            "Note: this wraps forward+backward together via `torch.compile(model, fullgraph=True, backend='tpu')`, "
            "but does NOT capture the optimizer step right inside the graph.",
            train_config.compile.mode,
        )
    trainer_cls = TPUTrainer

  # Synchronize all ranks before trainer initialization. On TPU (XLA), the
  # device-setup graph for rank 0 may compile faster than for other ranks,
  # letting rank 0 race past set_determinism's broadcast_object_list into
  # train_step. There rank 0 waits at an XLA all_reduce while the other ranks
  # are still compiling their broadcast graph — a distributed deadlock. This
  # barrier forces all ranks to flush their pending device-init XLA ops
  # simultaneously before any trainer code runs.
  if dist.is_initialized():
    dist.barrier()

  # TODO: b/498659628 - Remove once hanging is fixed.
  if train_config.debug.seed is None:
    train_config.debug.seed = 42
    logger.info(
        f"Setting default seed {train_config.debug.seed} to avoid hang on TPU"
    )

  # Apply TPU-specific workarounds before model parallelization
  # Moved from deprecated args.py files
  orig_parallelize = model_spec.parallelize_fn

  def patched_parallelize(model, *args, **kwargs):
    from torchtitan.experiments.tpu import workarounds

    workarounds.apply_patches(model, train_config)
    if isinstance(train_config, tpu_job_config_module.TPUTrainerConfig):
      if model_spec.name in (
          "llama3",
          "muse_glimmer",
          "qwen3",
          "qwen3_5",
          "gemma4",
      ):
        kwargs["tpu_compile_mode"] = train_config.compile.mode
    return orig_parallelize(model, *args, **kwargs)

  if not is_graph_trainer:
    model_spec.parallelize_fn = patched_parallelize

  try:
    with contextlib.ExitStack() as stack:
      torchtitan.train._maybe_enable_min_cost_redistribute(  # pylint: disable=protected-access
          stack,
          train_config.parallelism,
      )
      # TODO: Make TPUTrainerConfig owned by TPUTrainer so all TPU trainer
      # variants can use the upstream-native ``config.build()`` construction
      # path.
      trainer = trainer_cls(train_config)
      # Asked of the built model rather than wrapped around `parallelize_fn`:
      # graph-trainer recipes already carry their graph-specific model spec and
      # parallelize_fn, so the regular trainer wrapper above is not installed.
      for part in trainer.engine.model_parts:
        check_scan_was_applied(
            part,
            use_scan_layers=(
                train_config.compile.use_scan_layers
                if is_graph_trainer
                else False
            ),
        )
      train_config.maybe_log()

      if train_config.create_seed_checkpoint:
        assert int(os.environ["WORLD_SIZE"]) == 1, (
            "Must create seed checkpoint using a single device, to disable"
            " sharding."
        )
        assert (
            train_config.checkpointer is not None
        ), "Must configure checkpointer when creating a seed checkpoint."
        trainer.engine.save_checkpoint(last_step=True)
        logger.info("Created seed checkpoint")
      else:
        if rank == 0:
          logger.info(
              "Training with config: %s",
              json.dumps(train_config.to_dict(), indent=2),
          )
        # TT-UNFLATTEN-SHIM:
        from torchtitan.experiments.tpu.components.unflatten_shim import set_unflatten_seq_len
        # Upstream d771abb0 renamed training.seq_len ->
        # training.max_context_length.
        with set_unflatten_seq_len(train_config.training.max_context_length):
          trainer.train()
  except Exception:
    if trainer:
      trainer.close()
    raise
  else:
    trainer.close()
    if torch.distributed.is_initialized():
      torch.distributed.barrier()
      torch.distributed.destroy_process_group()
    logger.info("Process group destroyed")


if __name__ == "__main__":
  gmain.handle_main(start_trainer)
