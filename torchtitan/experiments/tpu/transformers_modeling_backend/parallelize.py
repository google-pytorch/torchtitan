"""Parallelize an HF backend model for TorchTPU.

Everything except data parallelism is the backend's own
`parallelize_hf_transformers` (TP, activation checkpointing, per-layer
compile), called with ``skip_dp=True``. This module supplies only the data
parallel wrapping, and supplies SimpleFSDP rather than FSDP2.

That choice is not stylistic. FSDP2 injects its all-gather from a
``forward_pre_hook``, so Dynamo hits a collective inside the hook's own frame,
graph-breaks there, and bails out of the wrapping ``__call__`` entirely --
leaving no compile region and dropping the whole layer body to eager dispatch.
SimpleFSDP registers a parametrization instead, which puts the all-gather on
the ``.weight`` access path, so the break lands *between* the all-gather and
the layer body and the body stays intact. Measured on this lane at 3B:
SimpleFSDP ~30 HLO modules per step, FSDP2 414 with torch.compile disabled
outright, and OOM at the same batch size.

This mirrors `experiments/tpu/llama3/infra/parallelize.py`, which does the
same swap for the torchtitan-native llama3.
"""

import dataclasses

import torch
from torch import nn

import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel as simple_fsdp_data_parallel,
    disable_active_parametrization,
    MixedPrecisionPolicy as SimpleFSDPMixedPrecisionPolicy,
)
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.experiments.transformers_modeling_backend.parallelize import (
    parallelize_hf_transformers,
)
from torchtitan.tools.logging import logger

from .model import substitute_tpu_experts



def _defer_parametrization_during_init(model: nn.Module) -> None:
  """Let the HF backend tie its weights after SimpleFSDP has wrapped it.

  The trainer parallelizes first and calls ``init_weights`` second, so
  ``HFTransformerModel.init_states`` runs ``model.tie_weights()`` when
  SimpleFSDP's parametrization is already installed. Reading ``.weight`` then
  yields the computed plain tensor rather than the underlying ``nn.Parameter``,
  and HF assigns it straight onto ``lm_head``::

      TypeError: cannot assign 'torch.tpu.BFloat16Tensor' as parameter
      'weight' (torch.nn.Parameter or None expected)

  torchtitan's own SimpleFSDP models solve this by overriding ``init_states``
  to run under ``disable_active_parametrization`` (see
  ``experiments/graph_trainer/qwen3/model.py``). Subclassing is awkward here
  because the backend's ``Config`` is also a ``PretrainedConfig``, so the same
  guard is installed on the instance instead. Any model without tied
  embeddings never reaches the failing branch, so this is a no-op for them.
  """
  original_init_states = model.init_states

  def init_states(*args, **kwargs):
    with disable_active_parametrization():
      return original_init_states(*args, **kwargs)

  model.init_states = init_states


def parallelize_hf_transformers_tpu(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config,
    dump_folder: str,
    tpu_compile_mode: str = "layer",
    **kwargs,
) -> nn.Module:
  """Apply TP/AC/compile via the backend, then SimpleFSDP data parallelism."""
  # TODO b/499075866: re-enable CPU offload for TPU torchtitan run.
  if training.enable_cpu_offload:
    logger.warning(
        "CPU offload is not supported for TPU torchtitan run, setting it to"
        " False."
    )
    training.enable_cpu_offload = False

  compile_mode = tpu_compile_mode
  effective_compile_config = compile_config
  if compile_mode in ("fullgraph", "aot_fx_trace") and compile_config.enable:
    # These modes compile the whole model below, so the backend must not also
    # compile per layer.
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  if effective_compile_config.enable:
    for layer in model.layers.values():
      apply_spmd_safe(layer)
    # HF backend configs have no layers list, so the trainer could not size
    # Dynamo's recompile limits from the config; size them from the layers that
    # are compiled here.
    from torchtitan.experiments.tpu import workarounds  # pylint: disable=g-import-not-at-top

    workarounds.raise_dynamo_recompile_limits_for_num_layers(len(model.layers))

  # The shared parallelize builds and swaps the Titan MoE (phase 2) from the
  # configs probed at construction; rewrite those first so it builds the TPU
  # grouped-expert implementation.
  substitute_tpu_experts(model)

  parallelize_hf_transformers(
      model,
      parallel_dims=parallel_dims,
      training=training,
      parallelism=parallelism,
      compile_config=effective_compile_config,
      ac_config=ac_config,
      dump_folder=dump_folder,
      # With SimpleFSDP this lane wraps data parallelism itself (below); with it
      # off, the shared parallelize applies its own fully_shard in this same
      # call -- Module.parallelize() may run only once per instance.
      skip_dp=parallelism.use_simple_fsdp,
  )

  if parallelism.use_simple_fsdp:
    # The shard axis is named "fsdp" on some bases and "dp_shard" on others
    # (this one activates dp_shard). Ask for whichever this ParallelDims
    # actually exposes rather than hardcoding one and failing at mesh lookup.
    _shard_axis = None
    for _cand in ("fsdp", "dp_shard"):
      try:
        parallel_dims.get_mesh([_cand])
        _shard_axis = _cand
        break
      except ValueError as e:
        # get_mesh raises ValueError for an axis this ParallelDims does not
        # activate; say which one was skipped and why, so a base that renames
        # the axis again does not fail three lines down with no trail.
        logger.warning(
            "SimpleFSDP: mesh axis %r unavailable (%s); trying the next", _cand, e
        )
    if _shard_axis is None:
      raise ValueError(
          "Neither 'fsdp' nor 'dp_shard' is an activated mesh axis; "
          "cannot place SimpleFSDP."
      )
    dp_mesh_names = (
        ["dp_replicate", _shard_axis]
        if parallel_dims.dp_replicate_enabled
        else [_shard_axis]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
    dp_mode = (
        "hybrid_shard" if parallel_dims.dp_replicate_enabled else "fully_shard"
    )
    mp_policy = SimpleFSDPMixedPrecisionPolicy(
        param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            training.mixed_precision_param
        ],
        reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            training.mixed_precision_reduce
        ],
    )
    model = simple_fsdp_data_parallel(
        model, dp_mesh, mode=dp_mode, mp_policy=mp_policy
    )
    _defer_parametrization_during_init(model)
    logger.info("Applied SimpleFSDP (%s) to the HF backend model", dp_mode)

  if compile_mode == "fullgraph":
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")

  return model
