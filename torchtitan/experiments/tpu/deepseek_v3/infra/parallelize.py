"""Parallelization for DeepSeek-V3 on TPU.

Mirrors upstream ``models/deepseek_v3/parallelize.py`` on the
``partial_dtensor`` path -- TP (``Module.parallelize``) -> AC -> per-block
compile -> FSDP -- with three TPU differences:

* Per-block compile uses ``dynamic=False``. DeepSeek-V3's dense and MoE layers
  are the same ``DeepSeekV3TransformerBlock`` class with different shapes
  (dense FFN vs routed + shared experts); with ``dynamic=None`` Dynamo's
  automatic dynamic shapes can mark the differing dims symbolic on the second
  layer type, which the TPU backend does not support (gotchas.md).
* ``parallelism.use_simple_fsdp`` selects SimpleFSDP (the lane default), with
  the shared shard-dim planner for uneven vocab / expert dims.
* ``tpu_compile_mode`` (``layer`` / ``fullgraph`` / ``aot_fx_trace``) as in the
  other TPU models; dense layers get ``spmd_safe`` under compile (MoE layers do
  not, as in TPU Qwen3).

CP and PP are rejected until they are validated on TPU. EP (no TP) follows
upstream / the TPU Qwen3 wrapper: ``Module.parallelize`` installs the EP
sharding, then SimpleFSDP wraps the expert banks on the EDP mesh.

``parallelize_deepseek_v3_graph_trainer`` is the GraphTrainer counterpart
(whole-step joint graph): the same checks and shard-dim planner, then upstream
GraphTrainer's DeepSeek-V3 setup (FQN annotation -> SimpleFSDP -> GraphTrainer
compile).
"""

import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.distributed.compile import _maybe_regional_inductor_backend
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.activation_checkpoint import tpu_selective_ac_config
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.experiments.graph_trainer.simple_fsdp import MixedPrecisionPolicy
from torchtitan.experiments.graph_trainer.deepseek_v3.parallelize import parallelize_deepseekv3 as graph_trainer_parallelize_deepseekv3
from torchtitan.experiments.tpu.graph_trainer.simple_fsdp import install_tpu_replicate_computation
from torchtitan.experiments.tpu.qwen3_sharding import configure_qwen_simple_fsdp_shard_dims
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3.mtp import apply_fsdp_to_mtp_decoder
from torchtitan.tools.logging import logger


def _validate_supported(
    parallel_dims: torchtitan.distributed.ParallelDims,
    parallelism: torchtitan.config.ParallelismConfig,
) -> None:
  if parallel_dims.cp_enabled:
    raise NotImplementedError(
        "Context parallelism is not supported for DeepSeek-V3 on TPU yet."
    )
  if parallel_dims.pp_enabled:
    raise NotImplementedError(
        "Pipeline parallelism is not supported for DeepSeek-V3 on TPU yet."
    )
  if parallel_dims.ep_enabled and parallel_dims.tp_enabled:
    raise NotImplementedError(
        "EP with TP is not supported for DeepSeek-V3 on TPU (the TPU combine"
        " has no SP-local layout)."
    )
  if parallelism.spmd_backend != "partial_dtensor":
    raise NotImplementedError(
        "DeepSeek-V3 on TPU supports only"
        " parallelism.spmd_backend=partial_dtensor."
    )


def _is_moe_layer(layer: nn.Module) -> bool:
  return any(isinstance(m, MoE) for m in layer.modules())


def _apply_compile_static(
    model: nn.Module, compile_config: torchtitan.config.CompileConfig
) -> None:
  """``torch.compile`` each decoder layer with ``fullgraph=True, dynamic=False``.

  Same settings as upstream ``apply_compile`` otherwise (scalar outputs for
  MoE dispatch, no forward side effects replayed under AC recompute).
  """
  torch._dynamo.config.capture_scalar_outputs = True
  torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = (
      True  # pyrefly: ignore [bad-assignment]
  )
  backend = _maybe_regional_inductor_backend(model, compile_config.backend)
  # pyrefly: ignore [missing-attribute]
  for _, block in model.layers.named_children():
    block.compile(backend=backend, fullgraph=True, dynamic=False)
  logger.info(
      "Compiling each DeepSeekV3TransformerBlock with torch.compile"
      " (dynamic=False)"
  )


def _shard_expert_banks_on_edp_mesh(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
) -> None:
  """SimpleFSDP for the EP-sharded expert banks, on the EDP mesh.

  ``apply_simple_fsdp`` does this itself, but finds the banks by walking
  ``model.layers`` for ``TransformerBlock`` instances; activation checkpointing
  has already replaced each block with a ``CheckpointWrapper``, so under AC it
  finds none and then tries to shard the EP DTensors over the dense FSDP mesh
  ("Cannot concatenate overlapping meshes", measured). Same placement as
  ``apply_simple_fsdp``'s EP branch (dim 1 only when EDP x EP exceeds the
  expert count); ``data_parallel`` then skips these modules in the model-wide
  pass.
  """
  if parallel_dims.dp_replicate_enabled:
    edp_mesh = parallel_dims.get_mesh(["dp_replicate", "efsdp"])
    dp_mode = (
        "hybrid_shard"
        if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled
        else "replicate"
    )
  else:
    edp_mesh = parallel_dims.get_mesh(["efsdp"])
    dp_mode = "fully_shard"
  mp_policy = MixedPrecisionPolicy(
      param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
          training.mixed_precision_param
      ],
      reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
          training.mixed_precision_reduce
      ],
  )
  for moe in [m for m in model.modules() if isinstance(m, MoE)]:
    inner_experts = moe.routed_experts.inner_experts
    shard_dim = (
        1
        if edp_mesh["efsdp"].size() * parallel_dims.ep
        > inner_experts.num_experts
        else 0
    )
    moe.routed_experts.inner_experts = data_parallel(
        inner_experts,
        edp_mesh,
        dp_mode,
        mp_policy=mp_policy,
        shard_dim=shard_dim,
    )


def parallelize_deepseek_v3(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config: "torchtitan.distributed.activation_checkpoint.ActivationCheckpointingConfig | None",
    dump_folder: str,
    skip_dp: bool = False,
    tpu_compile_mode: str = "layer",
    **kwargs,
) -> nn.Module:
  """Parallelizes a DeepSeek-V3 model for TPU.

  Args:
    model: The ``DeepSeekV3Model`` (may be on meta device).
    parallel_dims: Parallel dimensions for this run.
    training: Training config (mixed-precision dtypes, CPU offload).
    parallelism: Parallelism config; ``use_simple_fsdp`` selects SimpleFSDP
      instead of FSDP2.
    compile_config: ``torch.compile`` config.
    ac_config: Activation checkpointing config, or None.
    dump_folder: Dump folder for AC debugging artifacts.
    skip_dp: Skip data parallelism (e.g. for inference).
    tpu_compile_mode: One of ``layer``, ``fullgraph`` or ``aot_fx_trace``.
    **kwargs: Ignored; accepted for signature compatibility with train.py.

  Returns:
    The parallelized model.
  """
  del kwargs  # Unused.
  _validate_supported(parallel_dims, parallelism)

  compile_enabled = compile_config.enable and "model" in compile_config.components
  if compile_enabled and hasattr(model, "layers"):
    for layer in model.layers.values():
      if not _is_moe_layer(layer):
        apply_spmd_safe(layer)
  per_block_compile = compile_enabled and tpu_compile_mode not in (
      "fullgraph",
      "aot_fx_trace",
  )

  # Under partial_dtensor, Module.parallelize is needed for TP and EP (as
  # upstream's parallelize_deepseekv3): it applies the sharding configs set by
  # DeepSeekV3Model.Config.update_from_config, which for EP shard the expert
  # banks on the "ep" mesh and wire the token dispatcher's EP mesh.
  if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
    model.parallelize(parallel_dims)

  if parallel_dims.ep_enabled:
    # Core SelectiveAC saves all_to_all_single outputs (the EP dispatcher's
    # count / offset exchanges); torch-tpu's split compiler rejects the
    # backward wait on a saved collective ("Expected call_function for
    # wait_tensor producer, got placeholder", measured), so recompute them.
    ac_config = tpu_selective_ac_config(ac_config)
  if ac_config is not None:
    ac_config.build(dump_folder=dump_folder).apply(model)

  if per_block_compile:
    _apply_compile_static(model, compile_config)

  if not skip_dp and (
      parallel_dims.dp_shard_enabled or parallel_dims.dp_replicate_enabled
  ):
    if parallelism.use_simple_fsdp:
      # Shared planner: dim 1 for uneven vocab weights (and, once the TPU
      # grouped-expert converter is on, uneven expert banks).
      configure_qwen_simple_fsdp_shard_dims(model, parallel_dims)
      if parallel_dims.ep_enabled:
        _shard_expert_banks_on_edp_mesh(model, parallel_dims, training)
      model = apply_simple_fsdp(
          model, parallel_dims=parallel_dims, training=training
      )
      # As the TPU Qwen3 wrapper: split meshes (creates process groups, so
      # every rank runs it right after sharding), then native TPU collectives
      # for non-leading FSDP shards.
      for holder in model.modules():
        if hasattr(holder, "build_split_meshes"):
          holder.build_split_meshes()
      install_tpu_replicate_computation(model)
    else:
      edp_mesh = None
      if parallel_dims.ep_enabled:
        edp_mesh = parallel_dims.get_optional_mesh(
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
      dp_mesh_names = (
          ["dp_replicate", "fsdp"]
          if parallel_dims.dp_replicate_enabled
          else ["fsdp"]
      )
      apply_fsdp_to_mtp_decoder(
          # pyrefly: ignore [bad-argument-type]
          model,
          parallel_dims.get_mesh(dp_mesh_names),
          param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
              training.mixed_precision_param
          ],
          reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
              training.mixed_precision_reduce
          ],
          pp_enabled=parallel_dims.pp_enabled,
          cpu_offload=training.enable_cpu_offload,
          reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
          ep_degree=parallel_dims.ep,
          edp_mesh=edp_mesh,
          dp_mesh_dims=None,
          edp_mesh_dims=None,
          enable_symm_mem=parallelism.enable_fsdp_symm_mem,
      )

  if tpu_compile_mode == "fullgraph" and compile_enabled:
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")
  return model


def parallelize_deepseek_v3_graph_trainer(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    parallelism: torchtitan.config.ParallelismConfig,
    **kwargs,
) -> nn.Module:
  """Parallelizes a DeepSeek-V3 model for the TPU GraphTrainer.

  SimpleFSDP / HSDP only. TP is rejected too: upstream's GraphTrainer TP path
  goes through ``Module.parallelize`` with ``spmd_types``, which the TPU lane
  does not use.

  Args:
    model: The ``DeepSeekV3Model`` (may be on meta device).
    parallel_dims: Parallel dimensions for this run.
    parallelism: Parallelism config.
    **kwargs: Forwarded to upstream GraphTrainer's ``parallelize_deepseekv3``.

  Returns:
    The parallelized model.
  """
  _validate_supported(parallel_dims, parallelism)
  if parallel_dims.tp_enabled:
    raise NotImplementedError(
        "Tensor parallelism is not supported for DeepSeek-V3 on the TPU"
        " GraphTrainer; use SimpleFSDP / HSDP."
    )
  configure_qwen_simple_fsdp_shard_dims(model, parallel_dims)
  model = graph_trainer_parallelize_deepseekv3(
      model,
      parallel_dims=parallel_dims,
      parallelism=parallelism,
      **kwargs,
  )
  # As the TPU Qwen3 GraphTrainer wrapper (graph_trainer/qwen3.py).
  for holder in model.modules():
    if hasattr(holder, "build_split_meshes"):
      holder.build_split_meshes()
  install_tpu_replicate_computation(model)
  return model
