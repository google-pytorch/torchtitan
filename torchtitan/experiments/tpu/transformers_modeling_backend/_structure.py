"""Locate the parts of a HuggingFace model that torchtitan needs by name.

`HFTransformerModel` exposes `tok_embeddings` / `layers` / `norm` / `lm_head`
so the rest of torchtitan can find them. Those properties used to hardcode the
Llama nesting (`model.model.layers`), so anything shaped differently raised
`AttributeError` before the first step -- OPT keeps its blocks at
`model.decoder.layers`, GPT-2 at `transformer.h`.

This module resolves each part in two passes: a table of known nestings, then
a structural fallback that looks for the thing itself (the longest ModuleList
of same-class blocks is the decoder stack; the sole nn.Embedding under the
backbone is the token embedding). The fallback is what lets an architecture
nobody enumerated still work.
"""

from torch import nn

# Ordered attribute chains, most common first. Each entry is tried with
# getattr from the wrapped HF model.
_LAYER_PATHS = (
    ("model", "layers"),            # Llama, Qwen, Mistral, Gemma
    ("model", "decoder", "layers"), # OPT, BART decoder
    ("transformer", "h"),           # GPT-2, Falcon
    ("decoder", "layers"),
    ("model", "encoder", "layers"),
    ("esmc", "layers"),             # ESMC (ESM Cambrian)
    ("esm", "encoder", "layer"),    # ESM
    ("encoder", "layer"),           # BERT family
    ("layers",),
)
_EMBED_PATHS = (
    ("model", "embed_tokens"),
    ("model", "decoder", "embed_tokens"),
    ("transformer", "wte"),
    ("embeddings", "word_embeddings"),
    ("model", "embeddings", "word_embeddings"),
    ("esmc", "embed_tokens"),       # ESMC (ESM Cambrian)
    ("esm", "embeddings", "word_embeddings"),
    ("shared",),
)
_NORM_PATHS = (
    ("model", "norm"),
    ("model", "decoder", "final_layer_norm"),
    ("transformer", "ln_f"),
    ("model", "final_layernorm"),
    ("model", "final_norm"),      # ModernBERT
    ("vit", "layernorm"),         # ViT
    ("esmc", "norm"),             # ESMC (ESM Cambrian)
    ("esm", "encoder", "emb_layer_norm_after"),  # ESM-1b / ESM-2
    ("encoder", "emb_layer_norm_after"),
    ("encoder", "final_layer_norm"),
    ("norm",),
)
_HEAD_PATHS = (
    ("lm_head",),
    ("cls", "predictions", "decoder"),
    ("score",),
    ("qa_outputs",),
    ("classifier",),
)


def _walk(root, path):
    obj = root
    for part in path:
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _resolve(root, paths):
    for path in paths:
        found = _walk(root, path)
        if found is not None:
            return found, path
    return None, None


def _path_of(root, target):
    """The dotted path of an existing submodule, by identity."""
    if target is None:
        return None
    for name, mod in root.named_modules():
        if mod is target:
            return tuple(name.split(".")) if name else ()
    return None


def _ask_hf(root, getter):
    """Ask the HF model itself where a part lives.

    `get_input_embeddings` / `get_output_embeddings` are implemented by every
    PreTrainedModel, so they cover architectures no local table enumerates --
    DistilBERT calls its head `vocab_projector`, ModernBERT calls it
    `decoder`. Returning None is a real answer: a bare encoder has no head.
    """
    fn = getattr(root, getter, None)
    if fn is None:
        return None, None
    try:
        found = fn()
    except Exception:
        return None, None
    if found is None:
        return None, None
    return found, _path_of(root, found)


def find_layers(root):
    """The decoder/encoder block stack.

    Falls back to the longest ModuleList/ModuleDict whose entries share a
    class -- that is what a transformer block stack looks like structurally,
    whatever it is called.
    """
    found, path = _resolve(root, _LAYER_PATHS)
    if found is not None:
        return found, path

    best, best_path = None, None
    for name, mod in root.named_modules():
        if not isinstance(mod, (nn.ModuleList, nn.ModuleDict)):
            continue
        entries = list(mod.values()) if isinstance(mod, nn.ModuleDict) else list(mod)
        if len(entries) < 2:
            continue
        if len({type(e) for e in entries}) != 1:
            continue
        if best is None or len(entries) > len(best):
            best, best_path = mod, tuple(name.split("."))
    return best, best_path


def find_embedding(root):
    found, path = _ask_hf(root, "get_input_embeddings")
    if found is not None and path is not None:
        return found, path
    found, path = _resolve(root, _EMBED_PATHS)
    if found is not None:
        return found, path
    # Structural fallback: the widest nn.Embedding is the token table.
    best, best_path = None, None
    for name, mod in root.named_modules():
        if isinstance(mod, nn.Embedding):
            if best is None or mod.num_embeddings > best.num_embeddings:
                best, best_path = mod, tuple(name.split("."))
    return best, best_path


def find_norm(root):
    """The final norm, or (None, None) for a model that has none.

    BERT, DistilBERT and XLM-R normalize inside every block and end without a
    trailing norm, so None is the correct answer for them rather than a gap
    in the table. Callers treat the final norm as optional.
    """
    return _resolve(root, _NORM_PATHS)


def find_head(root):
    """The output projection, or (None, None) for a model that has none.

    A bare encoder (sentence-transformers' MiniLM is a plain BertModel) and a
    vision classifier both legitimately answer None here; callers treat the
    head as optional rather than assuming every architecture has one.
    """
    found, path = _ask_hf(root, "get_output_embeddings")
    if found is not None and path is not None:
        return found, path
    return _resolve(root, _HEAD_PATHS)


def set_at(root, path, value):
    """Assign back through a path returned by one of the finders."""
    if not path:
        raise AttributeError("no resolved path to assign through")
    obj = root
    for part in path[:-1]:
        obj = getattr(obj, part)
    setattr(obj, path[-1], value)


def find_backbone(root):
    """The module that owns the block stack (model / transformer / decoder).

    Derived from wherever the stack was found, so it needs no separate table.
    """
    _, path = find_layers(root)
    if not path or len(path) < 2:
        return None, None
    return _walk(root, path[:-1]), path[:-1]


def find_rotary(root):
    """Rotary embedding module, or (None, None) for learned-position models."""
    backbone, bpath = find_backbone(root)
    for holder, hpath in ((backbone, bpath), (root, ())):
        if holder is not None and hasattr(holder, "rotary_emb"):
            return getattr(holder, "rotary_emb"), tuple(hpath) + ("rotary_emb",)
    return None, None


# Learned absolute position tables, by the name each family gives them.
# A model that positions with RoPE has none of these and is unbounded here.
_POS_EMBED_PATHS = (
    ("transformer", "wpe"),                             # GPT-2
    ("model", "decoder", "embed_positions"),            # OPT
    ("embeddings", "position_embeddings"),              # BERT/DistilBERT
    ("model", "embeddings", "position_embeddings"),
    ("bert", "embeddings", "position_embeddings"),
    ("roberta", "embeddings", "position_embeddings"),   # RoBERTa, XLM-R
    ("esm", "embeddings", "position_embeddings"),       # ESM-1b / ESM-2
    ("distilbert", "embeddings", "position_embeddings"),
    ("electra", "embeddings", "position_embeddings"),
)


def find_position_embedding(root):
    """The learned absolute position table, or (None, None) under RoPE.

    Structural fallback: an nn.Embedding whose name says it holds positions.
    GPT-2's `wpe` is the one that does not spell the word out.
    """
    found, path = _resolve(root, _POS_EMBED_PATHS)
    # ViT-style models keep positions in a plain Parameter under the same
    # attribute name an encoder gives its nn.Embedding (Dinov2's
    # embeddings.position_embeddings is a Parameter). Only a real embedding
    # table has a capacity to report.
    if isinstance(found, nn.Embedding):
        return found, path
    for name, mod in root.named_modules():
        if not isinstance(mod, nn.Embedding):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if "pos" in leaf.lower() or leaf == "wpe":
            return mod, tuple(name.split("."))
    return None, None


def position_capacity(root):
    """Longest sequence the model can position, or None if unbounded (RoPE).

    This is the table size minus whatever the family reserves at the front:
    RoBERTa/XLM-R start position ids at `padding_idx + 1`, and OPT's
    positional embedding carries an explicit `offset`. Reading the module
    rather than `config.max_position_embeddings` is what makes the number
    exact -- XLM-R reports 514 there but can only address 512.
    """
    pos, _ = find_position_embedding(root)
    if pos is None:
        return None
    reserved = getattr(pos, "offset", 0)
    if not isinstance(reserved, int):
        reserved = 0
    if getattr(pos, "padding_idx", None) is not None:
        reserved = max(reserved, pos.padding_idx + 1)
    return pos.num_embeddings - reserved


def head_out_features(head):
    """How many classes the head projects to, or None if it cannot be read.

    A head is not always a Linear: RoBERTa wraps dense + out_proj in a
    RobertaClassificationHead, so the width lives on the last Linear inside
    it rather than on the module itself.
    """
    if head is None:
        return None
    direct = getattr(head, "out_features", None)
    if direct is not None:
        return direct
    last = None
    for mod in head.modules():
        if isinstance(mod, nn.Linear):
            last = mod
    return last.out_features if last is not None else None


def accepted_kwargs(fn):
    """The keyword names a forward accepts, or None if it takes **kwargs.

    None means 'pass everything' -- a forward with **kwargs swallows what it
    does not use, so filtering against it would drop valid arguments.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return {
        name
        for name, p in params.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


def moe_attr_name(layer):
    """The attribute holding this layer's MoE block, or None if it has none.

    Models differ: most use ``mlp``, Mixtral and GraniteMoe use
    ``block_sparse_moe``. Fall back to whichever child module looks like a MoE
    block rather than enumerating every family -- the caller only needs the
    attribute found; the probe that follows reads the geometry off the module.
    """
    if hasattr(layer, "mlp"):
        return "mlp"
    for name in ("block_sparse_moe", "feed_forward", "moe", "mlp_moe"):
        if hasattr(layer, name):
            return name
    for name, child in layer.named_children():
        cls = type(child).__name__.lower()
        if "moe" in cls or "sparse" in cls or "experts" in cls:
            return name
    return None


def derive_dim(config, field, default=1):
    """Read a structural size the top-level config does not state.

    Composite configs keep it in a tower sub-config; staged encoders keep a
    per-stage list, of which the last stage is the one the head sees.
    """
    for name in ("text_config", "vision_config", "audio_config"):
        sub = getattr(config, name, None)
        v = getattr(sub, field, None) if sub is not None else None
        if isinstance(v, int) and v > 0:
            return v
    v = getattr(config, field, None)
    if isinstance(v, (list, tuple)) and v:
        return v[-1]
    if isinstance(v, int) and v > 0:
        return v
    return default


def derive_head_dim(config):
    """head_dim for an architecture that does not state it at the top level.

    Callers read config.head_dim directly, so leaving it unset raises
    AttributeError rather than degrading. A composite config (CLIP, BLIP,
    CLAP) keeps real dims in text_config / vision_config, and a staged
    encoder (SegFormer) keeps per-stage lists. Prefer the text tower.
    """
    for name in ("text_config", "vision_config", "audio_config"):
        sub = getattr(config, name, None)
        if sub is None:
            continue
        h = getattr(sub, "hidden_size", None)
        n = getattr(sub, "num_attention_heads", None)
        if h and n:
            return h // n
    h = getattr(config, "hidden_sizes", None)
    n = getattr(config, "num_attention_heads", None)
    if isinstance(h, (list, tuple)) and h:
        n_last = n[-1] if isinstance(n, (list, tuple)) and n else n
        if n_last:
            return h[-1] // n_last
    return 64
