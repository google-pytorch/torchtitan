"""The HF transformers backend model as the TPU lane runs it.

``HFTransformerModel`` in the shared backend assumes the Llama nesting
(``model.model.layers``, ``model.lm_head``) and a causal LM head. This lane
trains the wider HF zoo -- BERT-style encoders, GPT-2, OPT, T5, composite
configs -- so everything that generalizes the wrapper lives here, as a
subclass, and the shared backend keeps only small hooks:

- parts are resolved structurally (``_structure``): the block stack, token
  embedding, final norm, output head and rotary module are found by shape,
  not by attribute path;
- an architecture without a final norm gets an Identity, one without a
  vocabulary head gets an lm_head tied to its token embedding;
- ``build`` keeps HF's __init__-computed buffers through meta-device
  construction and ``to_empty`` (``_meta_init``);
- the learned position table is sized for ``max_seq_len`` and a sequence it
  cannot address is refused (the TPU gather clamps silently);
- an attention implementation the architecture refuses falls back to sdpa /
  eager instead of failing in __init__;
- the fused Pallas cross-entropy gets the trunk's hidden states and the head
  weight instead of logits (``fused_loss`` on the Config);
- Titan modules swapped in by ``moe_replacement`` are re-initialized after
  the trainer's ``to_empty`` destroyed their CPU-initialized weights, and the
  TPU grouped-expert kernel is substituted for the portable experts
  (``substitute_tpu_experts``, called from the lane's parallelize).

The lane's registry swaps a flavor's ``HFTransformerModel.Config`` for this
class's ``Config`` (``as_tpu_model_config``) so ``Config.build`` constructs a
``TPUHFTransformerModel``.
"""

import contextlib
import copy
import json
from dataclasses import dataclass

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import AttentionInterface, PreTrainedModel

from torchtitan.experiments.transformers_modeling_backend.model import (
    _ATTN_IMPLEMENTATION as _FLEX_IMPLEMENTATION,
    HFTransformerModel,
    SliceableModuleDict,
)
from torchtitan.tools.logging import logger

from . import _meta_init, _structure
# Registers ``attn_implementation="splash_tpu"`` with HF's AttentionInterface.
# Imported here, not only by the config registry: a worker that receives the
# Config by pickle (the test harness, any custom entry point) imports this
# module and nothing else from the lane, and an unregistered name would be
# quietly built as sdpa.
from . import attention as _attention  # noqa: F401  pylint: disable=unused-import


class _Attrs:
  """Attribute view over a (nested) config dict, for ``_structure`` helpers."""

  def __init__(self, d):
    self._d = d

  def __getattr__(self, name):
    v = self._d.get(name)
    return _Attrs(v) if isinstance(v, dict) else v


def _parse_config_overrides(spec: str) -> dict:
  """Parse a JSON object of HF config overrides, e.g. ``{"attn_pdrop": 0}``.

  JSON rather than ``key=value`` pairs: lists and nested values survive, and a
  typo (``false`` vs ``False``, an unquoted string) is an error instead of a
  silently kept string.
  """
  if not spec.strip():
    return {}
  try:
    out = json.loads(spec)
  except json.JSONDecodeError as e:
    raise ValueError(f"hf_config_overrides is not valid JSON: {spec!r} ({e})") from e
  if not isinstance(out, dict):
    raise ValueError(f"hf_config_overrides must be a JSON object, got {spec!r}")
  return out


class TPUHFTransformerModel(HFTransformerModel):
  """HFTransformerModel generalized past the Llama nesting, for the TPU lane."""

  @dataclass(kw_only=True, slots=True)
  class Config(HFTransformerModel.Config):
    """Adds no fields, so a flavor Config can be re-classed in place; the
    fused-loss switch rides on the instance as ``fused_loss``."""

    # Composite checkpoints whose text tower has no causal-LM / AutoModel
    # entry, by the parent's model_type.
    _TEXT_TOWERS = {
        "blip": "BlipTextModel",
        "clap": "ClapTextModel",
        "kosmos-2": "Kosmos2TextModel",
        "siglip": "SiglipTextModel",
    }

    # Zero-argument super() is unusable in these methods: @dataclass(slots=True)
    # rebuilds the class, so the cell it would resolve against is the
    # discarded one. Every base call below names the base explicitly.

    def build(self, **kwargs):
      clone = self._replace()
      # HF modules compute real buffers in __init__ (causal masks, rope
      # tables, embedding normalizers) and a plain meta build throws all of
      # them away. See _meta_init.py.
      with _meta_init.params_meta_buffers_real():
        instance = self._owner(config=clone, **kwargs)
      if self.param_init is not None:
        instance._param_init = self.param_init
      return instance

    def update_from_config(self, *, config=None, **kwargs):
      hf_model_id = getattr(config, "hf_model", "")
      raw, _ = PretrainedConfig.get_config_dict(hf_model_id)

      # The base derives head_dim as dim // num_attention_heads when the HF
      # config has none. A composite config (SegFormer's per-stage lists; a
      # vision tower) states neither at the top level, which makes that a
      # TypeError. Such a config carries no head_dim of its own, so a value
      # set here survives the base's attribute copy and short-circuits the
      # derivation. Text-bearing composites (CLIP) are unaffected: the base
      # swaps in their text_config, which has both sizes.
      if (
          "head_dim" not in raw
          and "text_config" not in raw
          and not (raw.get("hidden_size") and raw.get("num_attention_heads"))
      ):
        self.head_dim = _structure.derive_head_dim(_Attrs(raw))

      # Splash tuning from the job config's ``splash_attention_kernel``
      # section, carried on the model config: HF hands each attention layer to
      # the adapter as ``module``, which reads it from ``module.config``.
      # Injected like the attention fields so it reaches the loaded HF config.
      from torchtitan.experiments.tpu.workarounds import extract_splash_kwargs  # pylint: disable=g-import-not-at-top

      self._titan_injected_model_args["splash_kernel_kwargs"] = extract_splash_kwargs(
          getattr(config, "splash_attention_kernel", None)
      )
      HFTransformerModel.Config.update_from_config(self, config=config, **kwargs)
      self._complete_from_hf_config(hf_model_id)


      # is_moe was decided from the *flavor* in __init__, before --hf-model
      # had been applied, so a MoE checkpoint selected on the command line
      # arrives with a dense flavor and the MoE replacement never runs. The
      # checkpoint is final here: say so instead of running HF's MoE unreplaced.
      # Titan's GroupedExperts hardcodes SiLU-gated experts and a plain top-k
      # router. A MoE whose experts use another activation (Gemma-4: GeGLU,
      # gelu_pytorch_tanh, plus a router with RMSNorm / scale / per-expert
      # scale) would be swapped for a different model -- the backend's own
      # MODEL_COMPATIBILITY.md lists Gemma-4 as FAIL. Keep HF's MoE for those.
      activation = getattr(self, "hidden_activation", None) or getattr(self, "hidden_act", "silu")
      titan_moe_ok = activation in ("silu", "swish")
      if getattr(self, "is_moe", False) and not titan_moe_ok:
        raise ValueError(
            f"{hf_model_id}'s experts use {activation!r}; Titan's MoE swap supports "
            "SiLU-gated experts only and would train a different model. Use a dense "
            "config (e.g. hf_full_max), which keeps HF's own MoE block."
        )
      if not getattr(self, "is_moe", False):
        for holder in (raw, raw.get("text_config") or {}):
          experts = next(
              (holder.get(k) for k in ("num_experts", "n_routed_experts", "num_local_experts")
               if isinstance(holder.get(k), int) and holder.get(k) > 1),
              None,
          )
          if experts and titan_moe_ok:
            logger.warning(
                f"{hf_model_id} declares {experts} experts but the selected flavor "
                "is dense, so the MoE replacement will not run and HF's own MoE "
                "executes unreplaced. Select a *_moe config for this model."
            )
            break
          if experts:
            logger.info(
                f"{hf_model_id}: {experts} {activation} experts run on HF's own MoE "
                "block (Titan's swap supports SiLU experts only)."
            )
            break

      # The base forces initializer_range = 1.0 (its Llama-like init patch
      # reads the token-embedding std from it). HF's stock _init_weights uses
      # the same field as the std for every Linear/Conv1D/Embedding, and the
      # stock path runs whenever the patch does not match (GPT-2, BERT, XLM-R
      # are not named <Prefix>DecoderLayer): those models were built at
      # std=1.0 instead of 0.02, which lands cross entropy near 120 where
      # ln(vocab) is 10.8, with nothing failing. Keep the checkpoint's value;
      # the patch states its own 1.0 directly.
      self.initializer_range = raw.get("initializer_range", 0.02)

      # Per-run HF config overrides from the job config (a JSON object,
      # e.g. zeroing a checkpoint's attention dropout because the splash
      # kernel has no dropout path), applied last so they win over both the
      # checkpoint and the flavor. An unknown field is an error, not a no-op.
      for key, value in _parse_config_overrides(
          getattr(config, "hf_config_overrides", "")
      ).items():
        if not hasattr(self, key):
          raise ValueError(
              f"hf_config_overrides: {hf_model_id} config has no field {key!r}"
          )
        setattr(self, key, value)
      self._mirror_logit_transform_into_fused_loss(config, hf_model_id)
      return self

    def _mirror_logit_transform_into_fused_loss(self, job_config, hf_model_id: str) -> None:
      """Hand the head's logit transform to the fused CE, which re-applies it.

      The fused loss receives (trunk hidden, head weight) and computes
      ``soft_cap * tanh(output_multiplier * (h @ W) / soft_cap)`` in-kernel,
      forward and backward. Gemma's final soft-cap, Cohere's logit_scale,
      Granite's logits_scaling divisor and Falcon-H1's lm_head_multiplier
      are all that form, so instead of refusing those models the transform is
      copied into the loss config. The trainer runs this hook before it builds
      the loss, which is what makes the hand-off possible. A value the user set
      explicitly is never overridden: a mismatch is an error.
      """
      if not getattr(self, "fused_loss", False):
        return
      from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss

      loss_config = getattr(job_config, "loss", None)
      if not isinstance(loss_config, PallasCrossEntropyLoss.Config):
        return
      cap = getattr(self, "final_logit_softcapping", None)
      multiplier = 1.0
      for field, invert in (("logit_scale", False), ("logits_scaling", True), ("lm_head_multiplier", False)):
        value = getattr(self, field, None)
        if value is not None and value != 1.0:
          multiplier *= (1.0 / value) if invert else value
      if cap is None and multiplier == 1.0:
        return
      if loss_config.soft_cap is not None and loss_config.soft_cap != cap:
        raise ValueError(
            f"{hf_model_id} soft-caps its logits at {cap} but --loss.soft_cap="
            f"{loss_config.soft_cap}; leave it unset to take the model's value"
        )
      if loss_config.output_multiplier not in (1.0, multiplier):
        raise ValueError(
            f"{hf_model_id} scales its logits by {multiplier} but "
            f"--loss.output_multiplier={loss_config.output_multiplier}"
        )
      loss_config.soft_cap = cap
      loss_config.output_multiplier = multiplier
      self.fused_loss_logit_transform = (cap, multiplier)
      logger.info(
          f"{hf_model_id}: fused CE re-applies the head's logit transform "
          f"(soft_cap={cap}, output_multiplier={multiplier})"
      )

    def _complete_from_hf_config(self, hf_model_id: str) -> None:
      """Carry over what the base's attribute copy misses from the HF config.

      The base copies ``vars(hf_config)`` minus underscore keys and the
      architecture name from the causal-LM mapping. Four things fall through
      that net, each measured on the 2026-09-30 200-model sweep:

      - a composite checkpoint whose text tower has no causal-LM class
        (Qwen2-VL's ``qwen2_vl_text``, CLIP's ``clip_text_model``): take the
        bare model class from the AutoModel mapping; the attached lm_head
        supplies the vocabulary projection;
      - HF config *properties* (ModernBERT's ``sliding_window``, Gemma-4's
        ``layer_types``) exist on the HF config class, not in ``vars``;
      - private fields the modeling code reads (OPT's
        ``_remove_final_layer_norm``);
      - an HF ``attribute_map`` entry that inverts a backend alias
        (DistilBERT: ``hidden_size -> dim`` vs the backend's
        ``dim -> hidden_size``) recurses on every read.
      """
      import inspect

      from transformers import AutoConfig

      from torchtitan.experiments.transformers_modeling_backend.model import (
          _REMOTE_CONFIG_DENYLIST,
      )

      raw, _ = PretrainedConfig.get_config_dict(hf_model_id)
      top_config = AutoConfig.from_pretrained(
          hf_model_id,
          trust_remote_code=raw.get("model_type", "") not in _REMOTE_CONFIG_DENYLIST,
      )
      hf_config = getattr(top_config, "text_config", None) or top_config  # the base builds the text tower
      # model_type is a class attribute on HF configs, so the base's vars()
      # copy never carries it; the resolution below needs it.
      if not getattr(self, "model_type", None):
        self.model_type = hf_config.model_type

      # Break alias cycles between HF's attribute_map and the backend's.
      tt_to_hf = getattr(self, "_tt_to_hf_attribute_map", {})
      for hf_name, real in list(getattr(self, "attribute_map", {}).items()):
        if tt_to_hf.get(real) == hf_name:
          del self.attribute_map[hf_name]
          if real in self.__dict__ and hf_name not in self.__dict__:
            self.__dict__[hf_name] = self.__dict__[real]

      # Class properties and private fields of the HF config class.
      for name, member in inspect.getmembers(type(hf_config)):
        if isinstance(member, property) and not hasattr(type(self), name):
          try:
            setattr(self, name, getattr(hf_config, name))
          except Exception:  # a property that needs state we do not carry
            pass
      skip = {"_attn_implementation", "_attn_implementation_internal", "_name_or_path",
              "_commit_hash", "_experts_implementation", "_output_attentions"}
      for key, value in vars(hf_config).items():
        if key.startswith("_") and key not in skip and not hasattr(self, key):
          setattr(self, key, value)

      # A text tower with no causal-LM class: build the bare text model. The
      # AutoModel mapping knows some (Qwen2-VL's Qwen2VLTextModel); the rest
      # are only reachable as a <Prefix>TextModel in the parent's module, and
      # are named explicitly rather than picked from the module's exports.
      if not getattr(self, "architectures", None):
        from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

        cls_name = MODEL_MAPPING_NAMES.get(hf_config.model_type)
        if cls_name is None and top_config is not hf_config:
          cls_name = self._TEXT_TOWERS.get(top_config.model_type)
          if cls_name is None:
            raise ValueError(
                f"{hf_model_id}: no causal-LM or text-tower class known for "
                f"model_type {top_config.model_type!r}; add it to "
                "TPUHFTransformerModel.Config._TEXT_TOWERS"
            )
        if cls_name:
          self.architectures = [cls_name]

  def __init__(self, config: Config):
    # Set before the base __init__ runs: its layer conversion goes through
    # the ``layers`` property, and ``norm`` / ``lm_head`` consult these.
    object.__setattr__(self, "_added_norm", None)
    object.__setattr__(self, "_added_lm_head", None)
    object.__setattr__(self, "_context_length", None)
    super().__init__(config)

    self._position_capacity = _structure.position_capacity(self.model)
    # T5 positions with relative bias and DistilBERT with its own embedding,
    # so neither takes position_ids; passing it is a TypeError rather than
    # something they ignore. Ask the signature once.
    self._forward_kwargs = _structure.accepted_kwargs(self.model.forward)
    # Set on the Config by the registry's fused_loss=True configs.
    self._fused_loss = bool(getattr(config, "fused_loss", False))
    # Set on the Config by the registry's MoE configs (see
    # _use_pallas_grouped_mm).
    object.__setattr__(
        self, "_use_pallas_grouped_mm", bool(getattr(config, "use_pallas_grouped_mm", False))
    )
    # Set by update_from_config when the head's logit transform was handed to
    # the fused CE (see _mirror_logit_transform_into_fused_loss).
    self._mirrored_logit_transform = getattr(config, "fused_loss_logit_transform", None)

    # BERT, DistilBERT and XLM-R normalize inside every block and end without
    # a trailing norm. Callers still expect the part to exist, so stand in an
    # Identity, which is what "no final norm" computes.
    if _structure.find_norm(self.model)[0] is None:
      self._added_norm = nn.Identity()

    # A bare encoder has no output projection, and a classification head maps
    # to num_labels rather than to the vocabulary -- neither can produce token
    # logits. Attach a head tied to the token embedding, as any fine-tuning
    # harness does to put a token objective on such a model. This adds
    # parameters the checkpoint does not carry; it is for training and
    # measurement, not for restoring a published head.
    head, _ = _structure.find_head(self.model)
    head_out = _structure.head_out_features(head)
    vocab = getattr(config, "vocab_size", None)
    if head is not None and head_out is not None and vocab and head_out != vocab:
      logger.info(
          f"{type(self.model).__name__}'s head maps to {head_out} labels, not "
          f"the {vocab}-token vocabulary; attaching an lm_head and training the "
          "backbone on a token objective instead."
      )
      head = None
    if head is None:
      emb, _ = _structure.find_embedding(self.model)
      if isinstance(emb, nn.Embedding):
        self._added_lm_head = nn.Linear(emb.embedding_dim, emb.num_embeddings, bias=False)
        self._added_lm_head.weight = emb.weight
        logger.info(
            f"{type(self.model).__name__} has no usable output projection; "
            f"attached an lm_head tied to the token embedding "
            f"({emb.embedding_dim} -> {emb.num_embeddings})."
        )

    if self._fused_loss:
      blocker = self._fused_loss_blocker()
      if blocker is not None:
        raise ValueError(
            f"fused_loss is not supported for {type(self.model).__name__}: "
            f"{blocker}. The fused CE computes hidden @ weight and nothing else, "
            "so it would train a different model. Use the plain cross-entropy "
            "config (hf_full_max_plain_ce)."
        )

  # Post-head logit transforms in HF causal-LM heads, as config fields, with the
  # value that makes each one a no-op: Gemma-2/3/4 soft-cap, Cohere's scale,
  # Granite's divisor, Falcon-H1's multiplier.
  _LOGIT_TRANSFORMS = {
      "final_logit_softcapping": None,
      "logit_scale": 1.0,
      "logits_scaling": 1.0,
      "lm_head_multiplier": 1.0,
  }

  def _fused_loss_blocker(self) -> str | None:
    """Why the fused CE cannot stand in for this model's head, or None.

    The fused loss receives (trunk hidden states, head weight) and computes
    exactly hidden @ weight -> cross entropy. That equals the model's logits
    only for a bias-free linear head applied directly to the trunk output.
    Anything else -- a head behind its own dense/norm block (BERT's
    cls.predictions, DistilBERT's vocab_transform, ModernBERT's head), a head
    bias (phi-2), or a logit transform after the head -- would be silently
    dropped.
    """
    if self._added_lm_head is not None:
      return None  # the lane's own head: tied, bias-free, applied to the trunk
    head, path = _structure.find_head(self.model)
    if head is None:
      return "it has no output head"
    if tuple(path or ()) != ("lm_head",):
      return (
          f"its output head ({'.'.join(path or ('?',))}) is not a top-level "
          "lm_head, so layers between the trunk and the head would be skipped"
      )
    if not isinstance(head, nn.Linear):
      return f"its lm_head is a {type(head).__name__}, not a Linear"
    if head.bias is not None:
      return "its lm_head has a bias"
    if getattr(self, "_mirrored_logit_transform", None) is not None:
      return None  # the fused CE re-applies the head's transform in-kernel
    configs = [self.model.config]
    text = getattr(self.model.config, "text_config", None)
    if text is not None:
      configs.append(text)
    for cfg in configs:
      for field, neutral in self._LOGIT_TRANSFORMS.items():
        value = getattr(cfg, field, neutral)
        if value is not None and value != neutral:
          return f"its head applies {field}={value} to the logits"
    return None

  # ---- construction -------------------------------------------------------

  def _build_hf_model(self, model_cls, config):
    # Size the learned position table to hold max_seq_len before building.
    # The config bridges max_seq_len -> max_position_embeddings, but a family
    # that reserves slots at the front addresses fewer rows than it has:
    # XLM-R starts position ids at padding_idx + 1, so a 2048-row table tops
    # out at 2046 and the last two positions of every batch read out of range
    # -- silently on TPU, where the gather clamps.
    wanted = getattr(config, "max_seq_len", None)
    # Remembered before the growth below: max_seq_len aliases
    # max_position_embeddings, so after growing the table it reads the table
    # size, not the context length forward bounds positions by.
    self._context_length = wanted or None
    if wanted:
      try:
        with torch.device("meta"):
          probe = model_cls(config=config)
        cap = _structure.position_capacity(probe)
        if cap is not None and cap < wanted:
          config.max_position_embeddings = config.max_position_embeddings + (wanted - cap)
          logger.info(
              f"Grew the position table by {wanted - cap} to address {wanted} "
              f"positions ({type(probe).__name__} reserves slots at the front "
              "of its table)."
          )
      except Exception as e:
        logger.warning(f"Could not size the position table: {e}")
    return self._build_with_supported_attention(model_cls, config)

  @staticmethod
  def _set_attn_everywhere(config, attn: str) -> None:
    """Set the attention implementation on a config and its sub-configs.

    A model that builds a nested backbone from its own sub-config
    (grounding-dino's SwinBackbone) never sees a change made only at the top.
    """
    config._attn_implementation = attn
    config.attn_implementation = attn
    for name in dir(config):
      if not name.endswith("_config"):
        continue
      sub = getattr(config, name, None)
      if sub is not None and hasattr(sub, "_attn_implementation"):
        sub._attn_implementation = attn
        sub.attn_implementation = attn

  @classmethod
  def _build_with_supported_attention(cls, model_cls, config):
    """Build the model, backing off to an attention path it accepts.

    Registering an implementation with AttentionInterface only reaches
    architectures that dispatch through it. The BERT family indexes a fixed
    class dict instead -- BERT_SELF_ATTENTION_CLASSES[config._attn_implementation]
    -- so a name it does not know is a KeyError in __init__, and Electra and
    the T5 family raise ValueError for the same reason. Attention that is
    slower is not attention that is wrong.
    """
    wanted = getattr(config, "_attn_implementation", None)
    if wanted == "splash_tpu" and wanted not in AttentionInterface._global_mapping:
      # Not an architecture limit -- the adapter was never imported in this
      # process. Falling back here would train on sdpa under a config that
      # says splash, which is the silent no-op this lane refuses to allow.
      raise RuntimeError(
          f"attn_implementation={wanted!r} is not registered with HF's "
          "AttentionInterface in this process (import "
          "torchtitan.experiments.tpu.transformers_modeling_backend.attention)."
      )
    attempts = [wanted] + [a for a in ("sdpa", "eager") if a != wanted]
    last = None
    for attempt in attempts:
      if attempt != wanted:
        cls._set_attn_everywhere(config, attempt)
      try:
        model = model_cls(config=config)
      except (ValueError, KeyError) as e:
        text = str(e)
        if attempt is not None and attempt not in text and (
            "attention implementation" not in text
        ):
          raise
        last = e
        continue
      if attempt != wanted:
        logger.warning(
            f"{model_cls.__name__} does not accept attn_implementation="
            f"{wanted!r}; built with {attempt!r}."
        )
      return model
    raise last

  # ---- parts, resolved structurally ---------------------------------------

  def _part(self, finder, what):
    found, _ = finder(self.model)
    if found is None:
      raise AttributeError(f"Could not locate {what} on {type(self.model).__name__}")
    return found

  def _assign(self, finder, what, value):
    _, path = finder(self.model)
    if path is None:
      raise AttributeError(f"Could not locate {what} to assign")
    _structure.set_at(self.model, path, value)

  @property
  def tok_embeddings(self):
    """The token embedding table, wherever this architecture keeps it."""
    return self._part(_structure.find_embedding, "tok_embeddings")

  @tok_embeddings.setter
  def tok_embeddings(self, value):
    self._assign(_structure.find_embedding, "tok_embeddings", value)

  @property
  def layers(self):
    """The decoder/encoder block stack."""
    return self._part(_structure.find_layers, "the block stack")

  @layers.setter
  def layers(self, value):
    self._assign(_structure.find_layers, "layers", value)

  @property
  def norm(self):
    """The final norm, or an Identity for an architecture without one."""
    if self._added_norm is not None:
      return self._added_norm
    return self._part(_structure.find_norm, "norm")

  @norm.setter
  def norm(self, value):
    self._assign(_structure.find_norm, "norm", value)

  @property
  def lm_head(self):
    """The output projection, native to the architecture or attached for it."""
    if self._added_lm_head is not None:
      return self._added_lm_head
    return self._part(_structure.find_head, "lm_head")

  @lm_head.setter
  def lm_head(self, value):
    self._assign(_structure.find_head, "lm_head", value)

  @property
  def rotary_emb(self):
    """The rotary embedding module; AttributeError for a learned-position model
    (GPT-2, OPT, BERT), which is what ``hasattr`` callers expect."""
    return self._part(_structure.find_rotary, "rotary_emb")

  @rotary_emb.setter
  def rotary_emb(self, value):
    self._assign(_structure.find_rotary, "rotary_emb", value)

  def named_children(self):
    """A flattened view of the parts torchtitan expects.

    Parts an architecture does not have are skipped rather than yielded as
    None (GPT-2 and OPT position with learned embeddings and have no
    rotary_emb), and anything outside the five buckets is yielded too:
    nn.Module._apply walks children(), so a module omitted here is one
    .to_empty() never materializes -- it stays on meta and dies at the first
    forward with "Cannot copy out of meta tensor". GPT-2 has wpe, OPT has
    embed_positions.
    """
    seen: set[int] = set()
    for name in ("tok_embeddings", "layers", "norm", "lm_head", "rotary_emb"):
      try:
        child = getattr(self, name)
      except AttributeError:
        continue
      if child is None or id(child) in seen:
        continue
      seen.add(id(child))
      yield name, child

    # Mark the backbone covered before walking: its children are yielded
    # below, so yielding the backbone itself as well would hand the same
    # subtree to the caller twice -- and Module.parallelize() refuses to visit
    # a module more than once.
    backbone, _ = _structure.find_backbone(self.model)
    if backbone is not None:
      seen.add(id(backbone))
    for holder in (backbone, self.model):
      if holder is None:
        continue
      for child_name, child in holder.named_children():
        if child is None or id(child) in seen:
          continue
        # Skip a module that *contains* something already yielded. BERT keeps
        # its head at cls.predictions.decoder and its token table at
        # embeddings.word_embeddings, so yielding `cls` or `embeddings` here
        # would hand the caller a second route to a part already given.
        if any(id(d) in seen for d in child.modules()):
          continue
        seen.add(id(child))
        yield child_name, child

  # ---- masks and forward --------------------------------------------------

  def get_attention_masks(self, positions: torch.Tensor):
    # A flex BlockMask is only consumable by the flex attention path, and
    # flex has no TPU lowering -- building one here is what puts an
    # untranslatable gather/concatenate into the graph. This lane runs sdpa
    # or the splash kernel, which build their own causal masking, so only
    # produce a BlockMask when flex is actually the implementation.
    if getattr(self.model.config, "_attn_implementation", "") != _FLEX_IMPLEMENTATION:
      return None
    return super().get_attention_masks(positions)

  def _check_position_capacity(self, seq_len: int) -> None:
    """Refuse a sequence the learned position table cannot address.

    An out-of-range embedding gather raises on CPU but is silently clamped on
    TPU, so overrunning the table costs no error and shows up only as numerics
    wrong by an order of magnitude. RoPE models report no capacity and are not
    checked.
    """
    cap = self._position_capacity
    if cap is not None and seq_len > cap:
      raise ValueError(
          f"{type(self.model).__name__} positions at most {cap} tokens but was "
          f"given seq_len={seq_len}. Lower the config's seq_len to {cap} or "
          "less; running past the table reads out-of-range entries, which TPU "
          "returns silently as garbage."
      )

  def forward(self, *args, **kwargs):
    positions = kwargs.pop("positions", None)
    attention_masks = kwargs.pop("attention_masks", None)
    model_args = (args[0].unsqueeze(0), *args[1:])

    if positions is not None:
      # Per-document positions (reset at packed-sample boundaries) drive RoPE;
      # the mask handles cross-sample masking (segment ids on splash).
      kwargs["position_ids"] = positions.unsqueeze(0)
    else:
      local_seq_len = args[0].shape[0]
      kwargs["position_ids"] = torch.arange(local_seq_len, device=args[0].device).unsqueeze(0)

    if attention_masks is not None:
      kwargs["attention_mask"] = attention_masks

    # Document segment ids for the splash adapter, derived once per forward
    # and threaded to every layer through HF's attention kwargs; each layer
    # deriving its own repeats the same work per layer.
    if (
        positions is not None
        and getattr(self.model.config, "_attn_implementation", None) == "splash_tpu"
        and getattr(self.model.config, "attn_mask_type", "causal") == "block_causal"
    ):
      kwargs["tpu_segment_ids"] = _attention.document_segment_ids(kwargs["position_ids"])

    # The trainer folds the batch into one token-major stream, so the stream
    # is BS x seq_len long while the position table only has to cover one
    # sample: with per-document ``positions`` (reset at every sample boundary)
    # the largest index is bounded by the context length, not the stream.
    # Without them the arange above runs the full stream.
    if positions is not None and self._context_length:
      self._check_position_capacity(min(model_args[0].shape[-1], self._context_length))
    else:
      self._check_position_capacity(model_args[0].shape[-1])

    # An encoder-decoder (T5, BART, Marian, mBART) will not run without
    # something to feed its decoder. Teacher-force the same sequence, which
    # is the token objective this lane trains.
    if getattr(self.model.config, "is_encoder_decoder", False):
      kwargs.setdefault("decoder_input_ids", model_args[0])

    # T5 positions with relative bias and DistilBERT with its own embedding,
    # so neither accepts position_ids -- and neither ignores it.
    if self._forward_kwargs is not None:
      for k in [k for k in kwargs if k not in self._forward_kwargs]:
        kwargs.pop(k)

    # Run the whole model and take hidden states from whatever it returns.
    # Calling a resolved backbone directly only works when the backbone
    # accepts input_ids -- BERT's is its encoder, which takes embeddings --
    # and the wrapper's output type varies by head (BaseModelOutput,
    # MaskedLMOutput, ...). When the head we train through is one this lane
    # attached (or the fused loss wants the trunk), call base_model: HF's own
    # handle on the trunk, which takes input_ids on every family checked;
    # output_hidden_states no longer reaches the task-head forwards in
    # transformers 5.x.
    use_trunk = self._added_lm_head is not None or self._fused_loss
    base = getattr(self.model, "base_model", None) if use_trunk else None
    if base is not None and base is not self.model:
      output = base(*model_args, **kwargs)
    else:
      output = self.model(*model_args, **kwargs)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
      states = getattr(output, "hidden_states", None)
      hidden = states[-1] if states else None
    if hidden is None and isinstance(output, (tuple, list)) and output:
      first = output[0]
      if hasattr(first, "dim") and first.dim() == 3:
        hidden = first
    if hidden is None:
      logits = getattr(output, "logits", None)
      if logits is None:
        raise AttributeError(
            f"{type(self.model).__name__} returned neither hidden states nor "
            "logits to train on"
        )
      if self._fused_loss:
        # The fused CE needs the pre-projection hidden states; an encoder-
        # decoder with its own head (T5, mT5) returns logits only, and T5
        # rescales before its head, so the pair cannot be recovered here.
        raise ValueError(
            f"fused_loss is not supported for {type(self.model).__name__}: it "
            "returns logits rather than hidden states. Use the plain "
            "cross-entropy config (hf_full_max_plain_ce)."
        )
      return logits.squeeze(0)
    hidden_states = hidden.squeeze(0)

    if self._fused_loss:
      # Hand the loss the operands instead of the logits. Materializing them
      # costs T*vocab (Llama-3.2-1B at 16k tokens: 16384 x 128256 bf16 = 4 GB
      # a step); the fused kernel (PallasCrossEntropyLoss) does projection,
      # softmax and CE without ever writing that tensor.
      weight = getattr(self.lm_head, "weight", None)
      if weight is None:
        raise AttributeError(f"fused loss needs a weight on {type(self.lm_head).__name__}")
      return hidden_states, weight.t()

    if self._skip_lm_head:
      return hidden_states
    return self.lm_head(hidden_states)

  # ---- materialization ----------------------------------------------------

  def to_empty(self, *, device, recurse: bool = True):
    """Materialize parameters, but keep buffer values.

    torch's to_empty() runs empty_like over buffers too, discarding the causal
    masks / rope tables / normalizers that __init__ computed.
    """
    return _meta_init.to_empty_keep_buffers(self, device)

  def init_states(self, *, buffer_device: torch.device | None = None) -> None:
    # The base's Llama-like patched _init_weights reads the token-embedding
    # std from config.initializer_range, which the base forces to 1.0
    # (TorchTitan's init). This lane keeps the checkpoint's value instead
    # (update_from_config) so HF's *stock* init -- what runs when the patch
    # does not match: GPT-2, BERT, XLM-R -- uses 0.02 rather than 1.0. When
    # the patch is what runs, give it TorchTitan's 1.0 for the embedding only:
    # overriding initializer_range itself would also reach model-specific
    # _init_weights overrides (Flint's experts, router and attention sinks),
    # which then start at std 1.0 -- a router saturated from step 0.
    patched = getattr(PreTrainedModel._init_weights, "__name__", "") == "_init_weights_patched"
    if patched:
      self.model.config.embedding_init_std = 1.0
    # Sentinel: every parameter starts as NaN, so one the initializer never
    # wrote is detectable afterwards instead of training on to_empty garbage.
    # The base's Llama-like patch is installed on PreTrainedModel itself and
    # dispatches on Llama's attribute names, so a model that merely shares
    # the class-name pattern (OPT: out_proj / fc1 / fc2, with biases) gets
    # part of its weights skipped -- measured as loss nan from step 1.
    with torch.no_grad():
      for p in self.model.parameters():
        p.fill_(float("nan"))
    try:
      super().init_states(buffer_device=buffer_device)
    finally:
      self.model.config.__dict__.pop("embedding_init_std", None)
    self._init_untouched_parameters()

    # Titan MoE modules swapped in by moe_replacement stay on meta until the
    # trainer's to_empty() (it parallelizes FIRST and materializes SECOND), so
    # those expert weights arrive here as fresh, uninitialized storage. HF's _init_weights dispatches on module type and a stacked
    # [E, *, *] expert weight is not an nn.Linear, so nothing rewrites them
    # and the first MoE forward reads nan (measured: finite=0.0000 for
    # w1/w2/w3 with healthy activations and routing entering the same call).
    self._init_swapped_native_modules()

    # to_empty() hands every parameter fresh storage, which breaks the tie
    # made in __init__. HF's tie_weights only knows about its own head.
    if self._added_lm_head is not None:
      self._added_lm_head.weight = self.tok_embeddings.weight

    self._reassert_device()

  def _init_untouched_parameters(self) -> None:
    """Initialize parameters the model's own init never wrote (still NaN).

    Default HF semantics for what the Llama-like patch skips: weights at the
    config's init std, biases and anything one-dimensional at zero. Logged so
    a model relying on this is visible rather than silently different.
    """
    std = float(getattr(self.model.config, "init_std", None)
                or getattr(self.model.config, "initializer_range", None) or 0.02)
    touched_up = []
    with torch.no_grad():
      for name, p in self.model.named_parameters():
        if not torch.isnan(p).any():
          continue
        if p.dim() >= 2:
          nn.init.trunc_normal_(p, mean=0.0, std=std, a=-3 * std, b=3 * std)
        else:
          p.zero_()
        touched_up.append(name)
    if touched_up:
      logger.warning(
          "%d parameter(s) were left uninitialized by %s's init and were set to "
          "HF defaults (std %.3g / zero): %s%s",
          len(touched_up), type(self.model).__name__, std,
          ", ".join(touched_up[:6]), " ..." if len(touched_up) > 6 else "",
      )

  def _init_swapped_native_modules(self) -> None:
    """Re-initialize Titan modules swapped into the HF tree after to_empty().

    Only modules that own a ``param_init`` table are touched -- exactly the
    set built by ``moe_replacement``. ``init_states`` is called on the
    outermost such module; it recurses to its own children.
    """
    roots = [m for m in self.model.modules() if getattr(m, "_param_init", None) is not None]
    if not roots:
      return
    inner = set()
    for r in roots:
      inner.update(id(c) for c in r.modules() if c is not r)
    roots = [r for r in roots if id(r) not in inner]

    # Must run with SimpleFSDP's parametrization disabled, otherwise reads of
    # .w1_EFD return the computed tensor and the initializer writes into a
    # temporary. Only take the guard if parametrization is active:
    # disable_active_parametrization() is not reentrant (its finally sets the
    # flag back to True unconditionally), and the lane's parallelize already
    # wraps this call in it.
    ctx = contextlib.nullcontext()
    try:
      from torchtitan.experiments.graph_trainer import simple_fsdp as _sfsdp

      if getattr(_sfsdp, "_active_parametrization", False):
        ctx = _sfsdp.disable_active_parametrization()
    except Exception:  # SimpleFSDP not in use
      pass
    with ctx:
      for r in roots:
        r.init_states(buffer_device=self._reinit_buffer_device())
    logger.info("Re-initialized %d swapped native module(s) after materialization.", len(roots))

  def _reinit_buffer_device(self) -> torch.device | None:
    for param in self.model.parameters():
      if param.device.type != "meta":
        return param.device
    return None

  def _reassert_device(self) -> None:
    """Pull back anything init left on the wrong device.

    Some HF initializers build a tensor without a device and assign it over
    the materialized parameter -- Pegasus and Marian do this for sinusoidal
    positions (``torch.empty(n_pos, dim)`` lands on CPU). Upstream HF bug;
    guard until it is fixed.
    """
    device = next((p.device for p in self.model.parameters() if p.device.type != "cpu"), None)
    if device is None:
      return
    moved = []
    for name, mod in self.model.named_modules():
      for attr, val in list(mod._parameters.items()):
        if val is not None and val.device != device:
          mod._parameters[attr] = nn.Parameter(val.data.to(device), requires_grad=val.requires_grad)
          moved.append(f"{name}.{attr}")
      for attr, val in list(mod._buffers.items()):
        if val is not None and val.device != device:
          mod._buffers[attr] = val.to(device)
          moved.append(f"{name}.{attr}")
    if moved:
      logger.warning(
          f"init left {len(moved)} tensor(s) off {device}; moved back (first: "
          f"{moved[0]}). This is an upstream initializer that builds a tensor "
          "without a device."
      )


def as_tpu_model_config(
    model_config: HFTransformerModel.Config, *, fused_loss: bool = False
) -> "TPUHFTransformerModel.Config":
  """A copy of a flavor's Config that builds a TPUHFTransformerModel.

  The flavor Configs are module-level instances built by the backend's own
  ``__init__`` (``model_config=...``), which ``dataclasses.replace`` cannot
  re-run; the subclass adds no slots, so re-classing a shallow copy is the
  same object layout with the TPU behaviour attached -- and the shared
  ``flavors`` entry is left untouched.
  """
  cfg = copy.copy(model_config)
  cfg.__class__ = TPUHFTransformerModel.Config
  cfg.fused_loss = fused_loss
  # The shallow copy shares the flavor's override dict, which the registry
  # writes into (attn_implementation): without its own copy, selecting splash
  # on one config would change what every other config built from the same
  # flavor replays in update_from_config.
  overrides = getattr(cfg, "_titan_injected_model_args", None)
  if overrides is not None:
    cfg._titan_injected_model_args = dict(overrides)
  return cfg


def substitute_tpu_experts(model: nn.Module) -> None:
  """Swap the TPU grouped experts and token dispatcher into each probed MoE config.

  ``moe_replacement`` works in two phases: at construction it probes every HF
  MoE block into a Titan MoE config stored on the layer as
  ``_native_moe_config``; at parallelize time it builds and swaps the module.
  Registered ModelConfigConverters never see those configs (they run over the
  model spec before construction), so this rewrites the stored configs in
  between -- the lane's parallelize calls it before handing the model to the
  shared parallelize. Expert parallelism is deliberately not involved:
  experts stay replicated and SimpleFSDP shards them, as the qwen3 TPU lane
  does.

  The dispatcher matters more than the experts. The standard one combines with
  ``scatter_add`` over a ``(rows, dim)``-expanded index, whose backward is an
  element-wise ``gather`` of rows x dim values; XLA runs that on TPU at a few
  percent of HBM bandwidth (Qwen3-30B-A3B on v6e-32: 9.2 ms per call, 83% of
  the step). The TPU dispatcher's VJPs gather whole rows (``index_select``),
  as the native qwen3 and gemma4 MoE recipes already use.
  """
  from torchtitan.tools import utils  # pylint: disable=g-import-not-at-top

  if utils.device_type != "tpu":
    return
  from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExpertsConverter
  from torchtitan.experiments.tpu.token_dispatcher import TpuTokenDispatcherConverter

  converter = TpuGroupedExpertsConverter(TpuGroupedExpertsConverter.Config())
  dispatcher = TpuTokenDispatcherConverter(TpuTokenDispatcherConverter.Config())
  use_pallas = bool(getattr(model, "_use_pallas_grouped_mm", False))
  n = 0
  for layer in model.layers.values():
    cfg = getattr(layer, "_native_moe_config", None)
    if cfg is not None:
      cfg = dispatcher.convert(converter.convert(cfg))
      if use_pallas:
        _use_pallas_grouped_mm(cfg)
      layer._native_moe_config = cfg
      n += 1
  if n:
    logger.info(
        "Substituted the TPU grouped experts and token dispatcher in %d MoE layer(s)%s.",
        n, " (use_pallas_grouped_mm)" if use_pallas else "",
    )


def _use_pallas_grouped_mm(moe_config) -> None:
  """Set use_pallas_grouped_mm on every TpuGroupedExperts in a converted MoE config.

  Otherwise the experts run torch._grouped_mm (XLA ragged dot). Uses the
  unpaired kernels with bounded K/N tiles, which fit scoped VMEM at any expert
  width. The qwen3 lane's paired W1/W3 kernel is faster but holds full
  (dim, hidden) weight slabs in VMEM. Where those fit, enable it in the loop
  below with (it needs an untiled N in the dWeight tiling):

    experts.use_pallas_paired_grouped_mm = True
    experts.pallas_grouped_mm_tgmm_tiling = (256, tile, 0)
  """
  from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts

  for _, experts, _, _ in moe_config.traverse(TpuGroupedExperts.Config):
    # 0 selects the full dimension, which is small enough below 512.
    tile = 512 if min(experts.dim, experts.hidden_dim) >= 512 else 0
    experts.use_pallas_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 256
    experts.pallas_grouped_mm_tile_k = tile
    experts.pallas_grouped_mm_tile_n = tile
    experts.pallas_grouped_mm_tgmm_tiling = (256, tile, tile)
    experts.use_kernel_native_weight_layout = True