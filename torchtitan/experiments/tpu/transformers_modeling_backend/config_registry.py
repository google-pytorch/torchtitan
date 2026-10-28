"""TPU configs for the HuggingFace transformers modeling backend.

These are the GPU registry's configs with three things changed: the config
type (so the TPU entry point accepts it), the parallelize function (SimpleFSDP
instead of FSDP2), and `attn_implementation` (sdpa rather than flex, which has
no TPU path). Model, inputs, loss and trainer are the backend's and the TPU
lane's existing ones.

    python -m torchtitan.experiments.tpu.train \
        --module=torchtitan.experiments.tpu.transformers_modeling_backend \
        --config=hf_debugmodel
"""


from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config.configs import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.tpu.activation_checkpoint import (
    AttentionOnlyAC,
    AttentionSavingSelectiveAC,
)
from . import attention as _tpu_attention  # noqa: F401  (registers splash_tpu
                                            #  before the backend builds its
                                            #  flavor Configs at import time)
from torchtitan.experiments.transformers_modeling_backend import (
    TitanModelConfig,
    TitanMoeModelConfig,
    flavors as _shared_flavors,
    model_registry,
)
from torchtitan.experiments.transformers_modeling_backend.model import HFTransformerModel
from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig, TPUConfig
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.hf_datasets.text_datasets import DATASETS

from .job_config import TPUTransformersBackendConfig
from .model import as_tpu_model_config
from .parallelize import parallelize_hf_transformers_tpu

# Like the backend's "full_moe", but takes its architecture from the
# checkpoint instead of pinning one. "full_moe" sets dim/n_layers/n_heads/
# num_experts/... explicitly, and _initialize_attributes records every
# explicitly-set field into _titan_injected_model_args, which
# update_from_config then applies OVER the loaded HF config -- so --hf-model
# selects the weights' name while the model is built at full_moe's fixed
# 2048-dim / 48-layer / 128-expert shape. Leaving the HF-mapped fields at
# their None defaults is what "full" already does: nothing is injected, so
# AutoConfig's values stand. The MoE-ness still comes from the flavor
# (TitanMoeModelConfig gates the MoE replacement), which genuinely cannot be
# inferred from --hf-model since that is applied after the config is built.
# experts_implementation="native": the HF experts kernel is irrelevant here --
# moe_replacement swaps the whole MoE block for the Titan MoE (with the TPU
# grouped-expert kernel, see model.py) -- and the default "grouped_mm" refuses
# every family without a settable implementation (GraniteMoe, OLMoE, ...).
# Defined here rather than in the shared flavors table: it is this lane's way
# of running arbitrary MoE checkpoints.
_TPU_FLAVORS = {
    "full_moe_auto": HFTransformerModel.Config(
        model_config=TitanMoeModelConfig(experts_implementation="native")
    ),
    # A two-layer model shaped for the kernels: head_dim 128 (the backend's
    # "debugmodel" has 16-wide heads, which the splash kernel does not take)
    # and grouped-query attention, so a parity test exercises the same code
    # paths as the full model at a few MB of state. --hf-model still picks
    # the family (Qwen3, Llama, ...); these override its size.
    "debugmodel_kernels": HFTransformerModel.Config(
        model_config=TitanModelConfig(dim=512, n_layers=2, n_heads=4, n_kv_heads=2)
    ),
}


def _model_registry(flavor: str, *, seq_len: int):
  """The backend's model_registry, extended with this lane's own flavors."""
  if flavor in _TPU_FLAVORS and flavor not in _shared_flavors:
    spec = model_registry(next(iter(_shared_flavors)), seq_len=seq_len)
    spec.flavor = flavor
    spec.model = _TPU_FLAVORS[flavor]
    return spec
  return model_registry(flavor, seq_len=seq_len)

def _fused_ce_loss() -> PallasCrossEntropyLoss.Config:
  """The fused-CE loss config with this lane's tuned Mosaic tiles.

  The kernel's own defaults (b=1024, h=512, v=2048) come from the muse_glimmer
  recipe, not from this op-point. `b_block_size` must divide the token count
  (loss.py rejects a mismatch rather than falling back to XLA silently).

  b is 2048 here: swept on Llama-3.2-1B at 16,384 tokens per rank, 512 / 1024
  / 2048 gave 20,342 / 21,716 / 22,161 tok/s/device and 4096 will not compile
  (RESOURCE_EXHAUSTED). h and v stay at the recipe's values -- both directions
  on each were worse (256: 21,562; 1024: 21,535; v=1024: 20,892; v=4096:
  20,639, which also perturbs the first-step loss). Sweep from the CLI with
  ``--loss.b_block_size=...`` (the job config's ``loss`` section is the single
  source; ``build_cross_entropy_loss`` reads it after CLI parsing).
  """
  return PallasCrossEntropyLoss.Config(
      b_block_size=2048, h_block_size=512, v_block_size=2048
  )


# Small enough to compile and run a step on a single v6e chip.
_DEFAULT_HF_MODEL = "Qwen/Qwen3-0.6B"


def _is_hf_moe(hf_model: str) -> bool:
  """True when the HF config declares routed experts.

  The backend gates its MoE replacement on the *flavor* being a
  TitanMoeModelConfig, not on the model. Ask the model instead, so selecting a
  MoE checkpoint is enough -- which is the same principle the phase-1 probe
  uses once the gate is open.
  """
  if not hf_model:
    return False
  try:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(hf_model)
  except Exception:
    return False
  for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
    for holder in (cfg, getattr(cfg, "text_config", None)):
      v = getattr(holder, attr, None) if holder is not None else None
      if isinstance(v, int) and v > 1:
        return True
  return False


def _training_config(*, local_batch_size: int, seq_len: int, steps: int) -> TrainingConfig:
  """Express the (local_batch_size, seq_len) recipe in the token budget."""
  training = TrainingConfig(steps=steps)
  training.num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len
  training.max_context_length = seq_len
  return training


def _set_attn_implementation(model_config, attn: str) -> None:
  """Point HF at an attention implementation this lane can actually run.

  The shared backend configures flex attention unconditionally, which has no
  TPU lowering, and its _configure_hf_attention takes no argument. Set the
  same three fields it sets: HF selects on the underscore field,
  update_from_config re-applies _titan_injected_model_args over the loaded HF
  config, and the public name is kept in step so the two do not disagree.
  splash_tpu is registered with AttentionInterface by this package's attention
  module at import.
  """
  model_config._titan_injected_model_args["attn_implementation"] = attn
  model_config.attn_implementation = attn
  model_config._attn_implementation = attn


def _set_attn_mask_type(model_config, mask_type: str) -> None:
  """Select causal vs document-causal masking for the packed token stream.

  The Grain packer folds the batch into one token-major stream per rank and
  resets ``positions`` at every document boundary. ``block_causal`` keeps each
  sample's attention inside its own document (flex builds a document BlockMask
  on GPU; the splash adapter turns the same positions into segment ids on
  TPU). The backend's ``full`` flavor defaults to plain ``causal``, under which
  a 16k-token stream is ONE causal attention problem -- 8x the work of eight
  2k sequences and cross-sample attention. Same injection path as
  _set_attn_implementation so update_from_config re-applies it.
  """
  model_config._titan_injected_model_args["attn_mask_type"] = mask_type
  model_config.attn_mask_type = mask_type


def _c4_test_loader() -> GrainDataLoader.Config:
  """This base replaced the HF text dataloader with Grain (upstream #4088)."""
  return GrainDataLoader.Config(
      dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
      shuffle=False,
  )


def _tpu_hf_config(
    flavor: str,
    hf_model: str,
    *,
    local_batch_size: int = 1,
    seq_len: int = 2048,
    steps: int = 10,
    compile: bool = False,
    attn: str = "sdpa",
    fused_loss: bool = False,
    attn_mask_type: str = "block_causal",
    save_attention: bool = False,
    use_pallas_grouped_mm: bool = False,
) -> TPUTransformersBackendConfig:
  """Build a TPU job config around one of the backend's model flavors."""
  model_spec = _model_registry(flavor, seq_len=seq_len)
  # The TPU subclass carries everything the lane changes about the model
  # object (structural part resolution, attached head/norm, fused-loss
  # return, ...); the shared backend only has the hooks. Re-classing a copy
  # also leaves the shared flavors table untouched.
  model_spec.model = as_tpu_model_config(model_spec.model, fused_loss=fused_loss)
  # Sets TpuGroupedExperts.Config.use_pallas_grouped_mm on the swapped MoE
  # experts instead of upstream's torch._grouped_mm (see
  # model._use_pallas_grouped_mm).
  model_spec.model.use_pallas_grouped_mm = use_pallas_grouped_mm
  model_spec.parallelize_fn = parallelize_hf_transformers_tpu
  _set_attn_mask_type(model_spec.model, attn_mask_type)
  # flex attention has no TPU lowering. "sdpa" is the portable default;
  # "splash_tpu" routes attention through the TPU splash kernel via HF's
  # AttentionInterface (see attention.py), which works for any architecture
  # rather than only those whose modules the lane's patch can pattern-match.
  # Go through _configure_hf_attention rather than setting the attribute:
  # it also updates _titan_injected_model_args, which update_from_config
  # re-applies afterwards and would otherwise overwrite a bare assignment.
  _set_attn_implementation(model_spec.model, attn)
  return TPUTransformersBackendConfig(
      hf_model=hf_model,
      loss=_fused_ce_loss() if fused_loss else CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      # This base states the recipe as a token budget rather than
      # (local_batch_size, seq_len) -- see _set_batch below.
      training=_training_config(
          local_batch_size=local_batch_size, seq_len=seq_len, steps=steps
      ),
      # SimpleFSDP is the data-parallel implementation on this lane; see
      # parallelize.py for why it is not FSDP2. The shared backend requires
      # spmd_types, which this fork does not default to.
      parallelism=ParallelismConfig(use_simple_fsdp=True, spmd_backend="spmd_types"),
      # "layer" compiles each decoder block on its own. Compiling is what makes
      # the SimpleFSDP choice pay off: its all-gather sits on the .weight
      # access path, so the graph break lands before the layer body and the
      # body stays one compile region.
      tpu_config=TPUConfig(compile_mode="layer"),
      # Kernel tuning for the splash_tpu attention implementation, pushed to
      # the HF adapter by workarounds.apply_patches; CLI overrides via
      # --splash_attention_kernel.sa_*. 1024 forward/dkv blocks are the
      # measured optimum on the packed stream (v004: 512 worse, 2048 will not
      # compile); the dq blocks are unused under the fused backward.
      splash_attention_kernel=SplashAttentionKernelConfig(
          sa_block_q=1024, sa_block_kv=1024, sa_block_kv_compute=1024,
          sa_block_q_dkv=1024, sa_block_kv_dkv=1024, sa_block_kv_dkv_compute=1024,
      ),
      compile=CompileConfig(
          enable=compile,
          backend="tpu",
          components=["model"],
      ),
      dataloader=_c4_test_loader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      # ``save_attention`` swaps in the policy that also keeps the Pallas
      # attention output, as upstream's default set already does for every
      # attention implementation it knows about (SDPA variants, flex).
      activation_checkpoint=(
          AttentionSavingSelectiveAC.Config() if save_attention
          else SelectiveAC.Config()
      ),
  )


def hf_debugmodel() -> TPUTransformersBackendConfig:
  """Two-layer, narrow model -- the smoke test for the TPU path."""
  return _tpu_hf_config("debugmodel", _DEFAULT_HF_MODEL, seq_len=512)


def hf_debugmodel_max() -> TPUTransformersBackendConfig:
  """``hf_full_max``'s feature set on the two-layer kernel-shaped debug model.

  Splash attention, fused CE, selective AC keeping the attention output and
  per-layer compile, at parity-test scale. The fused-CE token tile must
  divide the token count: pass ``--loss.b_block_size`` for small batches.
  """
  return _tpu_hf_config("debugmodel_kernels", _DEFAULT_HF_MODEL, compile=True,
                        attn="splash_tpu", fused_loss=True, save_attention=True)


def hf_debugmodel_kernels_sdpa() -> TPUTransformersBackendConfig:
  """The kernel-shaped debug model on the portable path (sdpa, plain CE,
  per-layer compile): the control for hf_debugmodel_max in parity tests."""
  return _tpu_hf_config("debugmodel_kernels", _DEFAULT_HF_MODEL, compile=True)


def hf_debugmodel_kernels_splash() -> TPUTransformersBackendConfig:
  """The kernel-shaped debug model with splash attention and plain CE."""
  return _tpu_hf_config("debugmodel_kernels", _DEFAULT_HF_MODEL, compile=True,
                        attn="splash_tpu")


def hf_full_max() -> TPUTransformersBackendConfig:
  """The lane's config: splash attention, fused CE, per-layer compile.

  Selective AC also keeps the Pallas attention output instead of recomputing
  it. Variants are CLI flags rather than configs: ``--compile.no-enable`` for
  eager (the splash adapter reports fallbacks only outside Dynamo tracing),
  ``--hf_model`` / ``--hf_config_overrides`` for the checkpoint. A head the
  fused CE cannot reproduce (head bias, a transform block, a logit transform
  it does not take) is refused; use ``hf_full_max_plain_ce`` for those.
  """
  return _tpu_hf_config("full", _DEFAULT_HF_MODEL, compile=True,
                        attn="splash_tpu", fused_loss=True, save_attention=True)


def hf_full_max_plain_ce() -> TPUTransformersBackendConfig:
  """``hf_full_max`` with the model's own head and plain cross entropy."""
  return _tpu_hf_config("full", _DEFAULT_HF_MODEL, compile=True,
                        attn="splash_tpu", save_attention=True)


def hf_moe_auto() -> TPUTransformersBackendConfig:
  """A MoE checkpoint, built at its own architecture, on TPU grouped experts.

  The backend gates its MoE replacement on the flavor being a
  TitanMoeModelConfig, and ``--hf_model`` is applied after the config is
  built, so a MoE checkpoint needs this config. "full_moe_auto" leaves the
  HF-mapped fields at None so AutoConfig supplies them (the backend's
  "full_moe" flavor sets them explicitly and would build the same synthetic
  model for every checkpoint). No expert parallelism: experts stay replicated
  and SimpleFSDP shards them.

  Runs with per-layer compile and use_pallas_grouped_mm. hf_moe_auto_compile
  drops use_pallas_grouped_mm, hf_moe_auto_eager drops both.
  """
  return _tpu_hf_config("full_moe_auto", _DEFAULT_HF_MODEL, attn="splash_tpu",
                        compile=True, use_pallas_grouped_mm=True)


def hf_moe_auto_max() -> TPUTransformersBackendConfig:
  """``hf_moe_auto`` plus what ``hf_full_max`` adds: the fused CE (no full fp32
  logits -- 18 GiB at 32k tokens for a 150k vocabulary) and selective AC that
  keeps the attention output."""
  return _tpu_hf_config("full_moe_auto", _DEFAULT_HF_MODEL, attn="splash_tpu",
                        compile=True, use_pallas_grouped_mm=True, fused_loss=True,
                        save_attention=True)


def hf_moe_auto_max_attnac() -> TPUTransformersBackendConfig:
  """TUNING: ``hf_moe_auto_max`` with full AC that keeps the attention output."""
  config = hf_moe_auto_max()
  config.activation_checkpoint = AttentionOnlyAC.Config()
  return config


def hf_moe_auto_eager() -> TPUTransformersBackendConfig:
  """hf_moe_auto without compile or use_pallas_grouped_mm (debugging baseline)."""
  return _tpu_hf_config("full_moe_auto", _DEFAULT_HF_MODEL, attn="splash_tpu")


def hf_moe_auto_compile() -> TPUTransformersBackendConfig:
  """hf_moe_auto without use_pallas_grouped_mm (compile only; A/B reference)."""
  return _tpu_hf_config("full_moe_auto", _DEFAULT_HF_MODEL, attn="splash_tpu",
                        compile=True)


def as_cpu_reference(config: TPUTransformersBackendConfig) -> TPUTransformersBackendConfig:
  """The single-device CPU counterpart of a lane config, for parity tests.

  Everything the lane chooses because it runs on TPU is turned back into the
  portable equivalent: the fused Pallas cross entropy becomes the model's
  logits plus plain cross entropy, compile is off. Model, data, optimizer and
  schedule are untouched, so a CPU run of this config on the same batches is
  the reference the TPU run is compared against
  (``base_distributed_device_test._run_cpu_verification``).

  The attention implementation is deliberately left at ``splash_tpu``: on CPU
  the adapter falls back to SDPA with the mask the kernel applies (documents
  from positions, padding, causal, window), so both sides compute the same
  attention. Switching to plain ``sdpa`` would not -- HF's own masks do not
  infer documents from reset positions for every architecture, so the
  reference could attend across packed documents while the kernel does not.
  The mask derivation itself is checked against HF's mask functions by
  ``tests/test_attention_masks.py``.
  """
  model_config = config.model_spec.model
  if getattr(model_config, "fused_loss", False):
    model_config.fused_loss = False
    config.loss = CrossEntropyLoss.Config()
  config.compile.enable = False
  return config
