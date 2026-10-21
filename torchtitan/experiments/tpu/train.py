# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer compatible with TorchTpu."""

import argparse
import contextlib
import functools
import json
import tempfile
from typing import Any, List, Optional, Sequence, Tuple, cast

from absl import flags
from absl.flags import argparse_flags
import os
import torch
import torch.distributed as dist
from torchtitan.config.configurable import Configurable
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu.grad_norm import CompiledGradNorm
import torchtitan.experiments.tpu.profiler_workaround as profiler_workaround
import torchtitan.experiments.tpu.deepseek_v3   # trigger model registration
import torchtitan.experiments.tpu.flux  # trigger model registration
from torchtitan.experiments.tpu.graph_trainer.config import (
    to_tpu_graph_trainer_config,
)
from torchtitan.experiments.tpu.graph_trainer.scan_guards import check_scan_was_applied
import torchtitan.experiments.tpu.llama3  # trigger model registration
import torchtitan.experiments.tpu.muse_glimmer  # trigger model registration
import torchtitan.experiments.tpu.qwen3   # trigger model registration
import torchtitan.experiments.tpu.qwen3_5  # trigger model registration
import torchtitan.experiments.tpu.tpu_job_config as tpu_job_config_module
from torchtitan.tools.logging import init_logger, logger
import torchtitan.train

from absl import app


class TPUTrainer(torchtitan.train.Trainer):
  """Trainer subclass with TPU-specific optimizations.

  Adds the following optional behaviors controlled by tpu_config flags:

  use_graph_split:
    Inserts a `synchronize(loss, wait=False)` call between the forward+loss
    computation and the backward pass. This splits the monolithic XLA
    computation graph into two smaller graphs (forward and backward),
    significantly reducing first-step XLA compilation time.
    The torch_tpu synchronize() materializes the forward graph without
    breaking autograd, so loss.backward() still propagates gradients
    correctly through the full model.

  compile_grad_norm:
    Compiles the DTensor-aware global gradient norm as a standalone static XLA
    graph. In-place clipping remains on PyTorch's eager path.
  """

  def __init__(self, job_config: Configurable.Config):
    super().__init__(job_config)

    self._compiled_grad_norm: Optional[CompiledGradNorm] = None
    if isinstance(job_config, tpu_job_config_module.TPUTrainerConfig):
      from torchtitan.experiments.tpu.loss import build_cross_entropy_loss
      self.loss_fn = build_cross_entropy_loss(job_config)

      tpu_config = job_config.tpu_config
      if not tpu_config.enable_amp:
        logger.info("AMP is disabled, using uniform precision training.")
      self._compiled_grad_norm = (
          CompiledGradNorm() if tpu_config.compile_grad_norm else None
      )
      if tpu_config.compile_optimizer:
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

  def forward_backward_step(
      self,
      *,
      input_dict: dict[str, Any] | list[dict[str, Any]],
      global_valid_tokens: torch.Tensor,
  ) -> torch.Tensor:
    from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len  # pylint: disable=g-import-not-at-top
    from torchtitan.protocols.model import BaseModel  # pylint: disable=g-import-not-at-top

    config = self.config
    with set_unflatten_seq_len(config.training.max_context_length):
      use_graph_split = (
          isinstance(config, tpu_job_config_module.TPUTrainerConfig)
          and config.tpu_config.use_graph_split
          and not self.parallel_dims.pp_enabled
      )
      if not use_graph_split:
        return super().forward_backward_step(
            input_dict=input_dict,
            global_valid_tokens=global_valid_tokens,
        )

      # Graph-split forward/backward for TPU:
      # The forward pass and loss computation are materialized as one XLA graph,
      # and the backward pass is compiled separately. This reduces peak
      # compilation time vs. one monolithic forward+backward graph.
      from torch_tpu._internal.sync import synchronize  # pylint: disable=g-import-not-at-top

      assert len(self.model_parts) == 1
      assert isinstance(input_dict, dict)
      inputs, labels, extra_kwargs = cast(
          BaseModel, self.model_parts[0]
      ).preprocess_inputs(
          input_dict,
          parallel_dims=self.parallel_dims,
          parallelism=self.config.parallelism,
          max_num_documents=self.dataloader.max_num_documents,
          max_context_length=self.config.training.max_context_length,
      )
      self.ntokens_seen += (
          labels[0].numel() if isinstance(labels, tuple) else labels.numel()
      )

      with self.train_context():
        pred = self.model_parts[0](inputs, **extra_kwargs)
        loss_kwargs = {}
        if "positions" in extra_kwargs:
          loss_kwargs["positions"] = extra_kwargs["positions"]
        loss, _ = self.loss_fn(pred, labels, global_valid_tokens, **loss_kwargs)
        del pred

        # Materialize the forward graph without waiting for execution to
        # complete. This splits the deferred-op graph so the backward pass
        # compiles as a separate (smaller) XLA program. Autograd state is
        # unaffected — loss.backward() can still propagate gradients.
        synchronize(loss, wait=False)

        loss.backward()

      return loss


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


def start_trainer(train_config: tpu_job_config_module.TPUTrainerConfig):
  """Starts the training process."""
  rank = int(os.getenv("RANK", "0"))
  if rank == 0:
    init_logger()

  from torchtitan.experiments.tpu import cloud_distributed_utils  # pylint: disable=g-import-not-at-top
  from torchtitan.experiments.tpu import workarounds  # pylint: disable=g-import-not-at-top

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

  trainer: Optional[torchtitan.trainer.Trainer] = None
  model_spec = train_config.model_spec
  assert model_spec is not None
  model_name = model_spec.name
  if model_name != "flux":
    _fit_rope_to_context(
        model_spec.model, train_config.training.max_context_length
    )
  use_graph_trainer = model_name != "flux" and (
      train_config.tpu_config.use_graph_trainer
      or train_config.tpu_config.compile_mode == "aot_fx_trace"
  )
  if model_name == "flux":
    # FluxTrainer owns a separate train_step and bypasses the TPU integration.
    if train_config.tpu_config.compile_grad_norm:
      raise ValueError("compile_grad_norm does not support FluxTrainer")
    from torchtitan.models.flux.trainer import FluxTrainer
    _apply_flux_img_size(train_config)
    trainer_cls = FluxTrainer
  # Note: TpuGraphTrainer is designed primarily for joint full-graph tracing
  # ("aot_fx_trace" / "fullgraph"). We keep `--tpu-config.use-graph-trainer`
  # as an independent switch for experimentation, but enabling `use_graph_trainer`
  # without fullgraph compilation (e.g. alongside `layer` compile mode) can cause
  # severe HBM fragmentation and stack reservation collisions (e.g. 78.46G memory
  # reservation bottlenecks across multiple disjoint overlay modules).
  elif use_graph_trainer:
    if (
        train_config.tpu_config.use_graph_trainer
        and train_config.tpu_config.compile_mode
        not in ("aot_fx_trace", "fullgraph")
    ):
      logger.warning(
          "TpuGraphTrainer enabled without fullgraph compile mode"
          f" (current mode: {train_config.tpu_config.compile_mode}). This may"
          " lead to severe HBM fragmentation and stack reservation OOMs."
      )
    from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
      TpuGraphTrainer,
      TpuGraphTrainerConfig,
    )

    if model_name == "qwen3":
      from torchtitan.experiments.tpu.graph_trainer.qwen3 import (
          model_registry as gt_registry,
      )
    elif model_name == "llama3":
      from torchtitan.experiments.graph_trainer.llama3 import model_registry as gt_registry
    elif model_name == "deepseek_v3":
      from torchtitan.experiments.graph_trainer.deepseek_v3 import model_registry as gt_registry
    elif model_name == "muse_glimmer":
      from torchtitan.experiments.tpu.graph_trainer.muse_glimmer import (
          model_registry as gt_registry,
      )
    else:
      raise ValueError(
          f"Model {model_name} is not currently supported with GraphTrainer."
      )

    gt_cfg = to_tpu_graph_trainer_config(
        train_config, gt_registry, target_cls=TpuGraphTrainer.Config
    )
    if gt_cfg.model_spec is not None:
      _fit_rope_to_context(
          gt_cfg.model_spec.model, gt_cfg.training.max_context_length
      )
      if train_config.tpu_config.bld_block_layout:
        # The graph trainer's parallelize_fn binds instance-level forward
        # wrappers (`annotate_module_fqns`, `apply_spmd_safe`), after which a
        # class swap no longer takes effect; the post-init
        # `workarounds.apply_patches` call is too late. See bld_layout.py.
        gt_parallelize = gt_cfg.model_spec.parallelize_fn

        def _bld_then_parallelize(model, *args, **kwargs):
          from torchtitan.experiments.tpu.bld_layout import apply_bld_layout, bld_skip_reason

          apply_bld_layout(
              model,
              skip_reason=bld_skip_reason(train_config.parallelism),
          )
          return gt_parallelize(model, *args, **kwargs)

        gt_cfg.model_spec.parallelize_fn = _bld_then_parallelize

    splash_obj = cast(Any, getattr(train_config, "splash_attention_kernel", None))
    moe_obj = cast(Any, getattr(train_config, "moe", None))
    tpu_extra = TpuGraphTrainerConfig(
        device="tpu",
        attention_backend=getattr(
            splash_obj,
            "sa_attention_backend",
            "splash"
            if getattr(splash_obj, "use_splash_attention_kernel", False)
            else "sdpa-no-scale",
        )
        if splash_obj is not None
        else "sdpa-no-scale",
        moe_enabled=getattr(moe_obj, "enabled", False)
        if moe_obj is not None
        else False,
        moe_padded_dispatcher=getattr(moe_obj, "padded_dispatcher", False)
        if moe_obj is not None
        else False,
        moe_capacity_factor=getattr(moe_obj, "capacity_factor", 1.25)
        if moe_obj is not None
        else 1.25,
        ep_degree=getattr(
            train_config.parallelism,
            "expert_parallel_degree",
            getattr(train_config.parallelism, "ep", 1),
        ),
        optimizer_in_graph=train_config.tpu_config.optimizer_in_graph,
        optimizer_compile=train_config.tpu_config.compile_optimizer,
        compile_mode=train_config.tpu_config.compile_mode,
        enable_mfu_benchmark=train_config.tpu_config.enable_mfu_benchmark,
        mfu_benchmark_start_step=train_config.tpu_config.mfu_benchmark_start_step,
        mfu_benchmark_final_step=train_config.training.steps,
        job_config=train_config,
    )

    def trainer_cls(cfg):
      return TpuGraphTrainer(gt_cfg, tpu_extra=tpu_extra)
  else:
    if isinstance(train_config, tpu_job_config_module.TPUTrainerConfig):
      if train_config.tpu_config.compile_mode == "fullgraph":
        logger.warning(
            "Running fullgraph compile mode (`%s`) on regular Trainer (`--tpu-config.use-graph-trainer=False`). "
            "Note: this wraps forward+backward together via `torch.compile(model, fullgraph=True, backend='tpu')`, "
            "but does NOT capture the optimizer step right inside the graph.",
            train_config.tpu_config.compile_mode,
        )
      if getattr(train_config.compile, "optimizer_in_graph", False) or getattr(
          train_config.tpu_config, "optimizer_in_graph", False
      ):
        logger.warning(
            "`optimizer_in_graph` (`--compile.optimizer-in-graph` or `--tpu-config.optimizer-in-graph`) "
            "is set while running on regular Trainer (`use_graph_trainer=False`). "
            "Regular Trainer does not support optimizer-in-graph (capturing `optimizer.step()` inside the graph). "
            "To enable joint fwd+bwd+optimizer compilation, you must use `--tpu-config.use-graph-trainer`."
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
      if model_spec.name in ("llama3", "muse_glimmer", "qwen3", "qwen3_5"):
        if (
            train_config.tpu_config.use_graph_trainer
            or train_config.tpu_config.compile_mode == "aot_fx_trace"
        ):
          kwargs.pop("tpu_compile_mode", None)
        else:
          kwargs["tpu_compile_mode"] = train_config.tpu_config.compile_mode
    return orig_parallelize(model, *args, **kwargs)

  model_spec.parallelize_fn = patched_parallelize

  try:
    with contextlib.ExitStack() as stack:
      torchtitan.train._maybe_enable_min_cost_redistribute(  # pylint: disable=protected-access
          stack,
          train_config.parallelism,
      )
      trainer = trainer_cls(train_config)
      # Asked of the built model rather than wrapped around `parallelize_fn`:
      # the graph trainer -- the only lane scan runs on -- builds its own config
      # and its own parallelize_fn, so a wrapper on the one patched above is
      # never called.
      for part in trainer.model_parts:
        check_scan_was_applied(
            part, use_scan_layers=train_config.tpu_config.use_scan_layers
        )
      train_config.maybe_log()

      if train_config.checkpoint.create_seed_checkpoint:
        assert int(os.environ["WORLD_SIZE"]) == 1, (
            "Must create seed checkpoint using a single device, to disable"
            " sharding."
        )
        assert (
            train_config.checkpoint.enable
        ), "Must enable checkpointing when creating a seed checkpoint."
        trainer.checkpointer.save(curr_step=0, last_step=True)
        logger.info("Created seed checkpoint")
      else:
        if rank == 0:
          logger.info(
              "Training with config: %s",
              json.dumps(train_config.to_dict(), indent=2),
          )
        # TT-UNFLATTEN-SHIM:
        from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
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
