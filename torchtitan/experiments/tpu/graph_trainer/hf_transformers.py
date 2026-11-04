"""HuggingFace transformers backend models on the TPU GraphTrainer path.

Runs `experiments/tpu/transformers_modeling_backend` (e.g. the ESM-C
`hf_esmc_mlm_uniref50_graph` recipe) under `TpuGraphTrainer`, which traces
forward + loss + backward (and optionally the AdamW step) into one FX graph and
lowers it as a single XLA program. Compared with per-layer `torch.compile`
(`--tpu_config.compile_mode=layer`), saved activations stay inside that single
program so XLA does not pin them to the default layout at per-layer
forward/backward program boundaries.

What stays identical to the TPU HF backend lane:
* The HuggingFace model, `preprocess_inputs`, `splash_tpu` attention, loss,
  dataloader, optimizer, and SimpleFSDP data parallelism.

What changes:
* Per-layer `torch.compile` and eager activation checkpointing are disabled
  because `TpuGraphTrainer` compiles the full step and applies its own
  graph-level memory policy (`--tpu_config.graph_trainer_memory_policy`).
* After `convert_hf_to_module` and SimpleFSDP wrapping complete, every
  submodule's `forward` is annotated with its GraphTrainer-canonical FQN
  (`tok_embeddings`, `layers.<N>`, `norm`, `lm_head`) so graph passes can
  attribute FX nodes to transformer blocks.

Currently supports dense models under SimpleFSDP / HSDP; TP, CP, EP, PP, and
MoE models are rejected explicitly.
"""

from collections.abc import Iterable
import dataclasses
from typing import Any

from torch import nn
from torch.fx.traceback import annotate_fn
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import _MODULE_FQN
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer
from torchtitan.experiments.tpu.transformers_modeling_backend.job_config import TPUTransformersBackendConfig
from torchtitan.experiments.tpu.transformers_modeling_backend.parallelize import parallelize_hf_transformers_tpu
from torchtitan.protocols.model_spec import ModelSpec

_ORIG_FORWARD_ATTR = "_graph_trainer_orig_forward"


def _iter_layer_modules(model: nn.Module) -> Iterable[nn.Module]:
  try:
    layers = model.layers
  except AttributeError:
    return ()
  if layers is None:
    return ()
  if isinstance(layers, nn.ModuleDict):
    return layers.values()
  return layers


def annotate_hf_module_fqns(model: nn.Module) -> None:
  """Idempotently annotates submodules with GraphTrainer-canonical FQNs.

  `TPUHFTransformerModel` wraps the HuggingFace model at `self.model`, so
  `model.named_modules()` prefixes every path with `model.` (e.g.
  `model.esmc.layers.0`), hiding `tok_embeddings`, `layers.<N>`, `norm`, and
  `lm_head` from GraphTrainer passes. This helper canonicalizes those subtrees
  and records each submodule's unwrapped `forward` on `_graph_trainer_orig_forward`
  so repeated calls replace rather than stack `annotate_fn` wrappers.
  """
  by_id: dict[int, tuple[nn.Module, str]] = {}
  for fqn, sub in model.named_modules():
    if fqn:
      by_id[id(sub)] = (sub, fqn)

  def _override_subtree(root: nn.Module | None, prefix: str) -> None:
    if root is None or isinstance(root, nn.Identity):
      return
    for sub_fqn, sub in root.named_modules():
      canonical = f"{prefix}.{sub_fqn}" if sub_fqn else prefix
      by_id[id(sub)] = (sub, canonical)

  for attr in ("tok_embeddings", "norm"):
    _override_subtree(getattr(model, attr, None), attr)

  try:
    layers = model.layers
  except AttributeError:
    layers = None
  if isinstance(layers, nn.ModuleDict):
    for key, block in layers.items():
      _override_subtree(block, f"layers.{key}")
  elif layers is not None:
    for idx, block in enumerate(layers):
      _override_subtree(block, f"layers.{idx}")

  _override_subtree(getattr(model, "lm_head", None), "lm_head")

  for sub, fqn in by_id.values():
    orig_forward = getattr(sub, _ORIG_FORWARD_ATTR, sub.forward)
    setattr(sub, _ORIG_FORWARD_ATTR, orig_forward)
    sub.forward = annotate_fn({_MODULE_FQN: fqn})(orig_forward)


# Backward-compatible private alias.
_annotate_hf_module_fqns = annotate_hf_module_fqns


@dataclasses.dataclass(kw_only=True, slots=True)
class TpuGraphTrainerHFConfig(
    TpuGraphTrainer.Config, TPUTransformersBackendConfig
):
  """`TpuGraphTrainer.Config` combined with `TPUTransformersBackendConfig`."""


def parallelize_hf_transformers_graph(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: Any,
    ac_config: Any = None,
    dump_folder: str,
    **kwargs,
) -> nn.Module:
  """Applies TPU SimpleFSDP wrapping, then annotates canonical module FQNs."""
  del ac_config, kwargs  # GraphTrainer remats via its graph memory policy.
  if (
      parallel_dims.tp_enabled
      or parallel_dims.pp_enabled
      or parallel_dims.cp_enabled
      or parallel_dims.ep_enabled
  ):
    raise NotImplementedError(
        "TPU GraphTrainer HF backend supports SimpleFSDP/HSDP only; TP, CP,"
        " EP and PP are not supported."
    )
  if not parallelism.use_simple_fsdp:
    raise ValueError(
        "TPU GraphTrainer HF backend requires --parallelism.use_simple_fsdp"
    )
  if getattr(getattr(model, "config", None), "is_moe", False) or any(
      getattr(layer, "moe_enabled", False)
      for layer in _iter_layer_modules(model)
  ):
    raise NotImplementedError(
        "TPU GraphTrainer HF backend currently supports dense models only; MoE"
        " layers are not yet supported."
    )

  # Run `parallelize_hf_transformers_tpu` BEFORE `annotate_hf_module_fqns`:
  # `convert_hf_to_module` swaps `nn.Embedding.__class__ = Embedding`, and an
  # instance-bound `sub.forward` installed beforehand would shadow the swapped
  # class's `Embedding.forward`. SimpleFSDP installs parameter property
  # descriptors without wrapping `forward`, so annotating afterward wraps the
  # final module classes cleanly.
  model = parallelize_hf_transformers_tpu(
      model,
      parallel_dims=parallel_dims,
      training=training,
      parallelism=parallelism,
      compile_config=dataclasses.replace(compile_config, enable=False),
      ac_config=None,
      dump_folder=dump_folder,
      tpu_compile_mode="aot_fx_trace",
  )
  annotate_hf_module_fqns(model)
  return model


def make_model_registry(recipe_spec: ModelSpec):
  """Returns a GraphTrainer `flavor -> ModelSpec` callable for an HF recipe."""

  def model_registry(flavor: str) -> ModelSpec:
    del flavor
    return dataclasses.replace(
        recipe_spec,
        parallelize_fn=parallelize_hf_transformers_graph,
        pipelining_fn=None,
    )

  return model_registry
