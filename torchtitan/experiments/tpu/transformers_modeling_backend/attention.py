"""Register TPU attention kernels with HuggingFace's attention interface.

The TPU lane's splash-attention patch works by walking the model for modules
whose class name contains "ScaledDotProductAttention" -- torchtitan's own
attention modules. HF models have no such module: every attention layer calls
`ALL_ATTENTION_FUNCTIONS[config._attn_implementation]` inline, so the patch
finds nothing and raises.

That interface is the generic hook. Registering a kernel here reaches every HF
architecture that uses the standard attention path, without naming any of them.

One kernel serves every call: ``sbd_splash_attention``, a tokamax splash
fork that adds an in-kernel PRNG dropout path and, since 2026-10-01,
document-aware block skipping on its compacted grid -- so dropout models no
longer fall back to SDPA, and dropout-free ones give nothing up for it.
Measured that day on Llama-3.2-1B, 4x v6e (8 x 2048 tokens folded into one
16k stream, block 1024: 24 of 136 causal blocks per layer can hold a
same-document pair): 20,063 tok/s/device against 20,102 for the stock
JAX-experimental splash it replaced, and 14,307 for the same kernel before
the skipping. The stock kernel is gone from this path; its dropout-free
block skipping was the only thing it did better, and it cannot do dropout at
all.

The translation of HF's contract: ``is_causal`` passed through (encoders run
bidirectional), document boundaries and a 2D padding mask as segment ids,
grouped-KV / MQA heads straight through (no K/V expansion), a sliding
window, and heads-first operands handed over without a transpose. What the
kernel cannot express (an additive float mask, a sequence that is not a
whole number of blocks) falls back to SDPA and *says so*, once per distinct
reason. A silent fallback is how a kernel that never fired gets reported as
a speedup.

HF builds a mask only for implementations registered with its
``AttentionMaskInterface``; ``splash_tpu`` registers FA2's mask function, so
a 2D padding mask arrives here as-is (or as None when nothing is padded) and
the module's ``is_causal`` plus the packed ``position_ids`` carry the rest.
"""

import logging
from typing import Any

import torch
from torchtitan.models.common.attention import segment_ids_from_positions
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask
from transformers.modeling_utils import AttentionInterface

from torchtitan.tools.logging import logger

# Kernel tuning (block sizes, backward variant, layouts) comes from the job
# config's ``splash_attention_kernel`` section. ``update_from_config`` copies it
# onto the model config as ``splash_kernel_kwargs``; HF passes each attention
# layer in as ``module``, so the adapter reads it from ``module.config`` rather
# than from process-global state. Unset means ``SbdConfig``'s defaults.
# Which of those ``sa_*`` fields the kernel has, and what it calls them. The
# rest of ``workarounds.SPLASH_KEYS`` -- the dq-kernel blocks, the backward-variant
# switches, the operand layouts -- has no counterpart here and is reported
# once as ignored.
_SBD_KERNEL_KWARGS = {
    "block_q": "block_q",
    "block_kv": "block_kv",
    "block_kv_compute": "block_kv_compute",
    "block_q_dkv": "block_q_dkv",
    "block_kv_dkv": "block_kv_dkv",
    "block_kv_dkv_compute": "block_kv_dkv_compute",
    "use_document_aware_block_skipping": "document_aware_skipping",
}
# Keys in ``_reported`` that are informational (the kernel ran); everything
# else in that set is a fallback reason. Tests read this.
INFO_REPORTS = frozenset({"splash_tpu fired", "splash_tpu args", "splash_tpu ignored"})
# The kernel needs both sequence lengths to be whole blocks, so every
# configured block is stepped down to the largest size that divides them
# (Mosaic's floor is 128) instead of gating on one number and launching with
# another -- a 512-gate / 1024-launch pair once let a 512-multiple that was
# not a 1024-multiple through the gate and into a kernel shape error.
_MIN_BLOCK = 128
# `sbd_splash.splash_attention_kernel.DROPOUT_QUANTUM`: the PRNG emits its mask
# in tiles of this many tokens and asserts on a block that is not a multiple.
_DROPOUT_QUANTUM = 256
_reported: set[str] = set()
# Resolved on first dropout call; see `_seed_salt`.
_SEED_SALT: int | None = None


def _tracing() -> bool:
    """True while Dynamo is tracing.

    Diagnostics cannot run inside a compiled region: Dynamo refuses to trace
    the logging call, and marking the helper `disable()`d turns the refusal
    into a hard "Skip calling torch.compiler.disable()'d function" error
    rather than a graph break. Guarding the call sites lets the branch fold
    away during tracing while staying live in eager, where the fallback
    decision is the one worth hearing about.
    """
    try:
        return torch.compiler.is_compiling()
    except Exception:
        return False


def _report(reason: str) -> None:
    """Report once. Call sites guard with `_tracing()`; see that helper."""
    if reason not in _reported:
        _reported.add(reason)
        logger.warning("splash_tpu: falling back to SDPA (%s)", reason)


def _log_once(key: str, message: str, *args) -> None:
    if key not in _reported:
        _reported.add(key)
        logger.info(message, *args)


def _seed_salt() -> int:
    """A per-rank constant to fold into the dropout seed (0 off a mesh).

    Resolved once: the rank cannot change within a process, and a python int
    folds into a captured graph as a constant.
    """
    global _SEED_SALT
    if _SEED_SALT is None:
        try:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except Exception:  # pragma: no cover - no mesh is the common local case
            rank = 0
        _SEED_SALT = (rank * 0x9E3779B1) & 0x7FFFFFFF
    return _SEED_SALT


def _pick_block(q_len: int, kv_len: int, preferred: int) -> int | None:
    """Largest block <= ``preferred`` that tiles both lengths, or None."""
    block = preferred
    while block >= _MIN_BLOCK:
        if q_len % block == 0 and kv_len % block == 0:
            return block
        block //= 2
    return None


def _kernel_kwargs(module, q_len: int, kv_len: int) -> dict[str, Any]:
    """The model's configured tuning with every block clamped to the sequence."""
    out = dict(getattr(getattr(module, "config", None), "splash_kernel_kwargs", None) or {})
    for key, value in out.items():
        if key.startswith("block_") and isinstance(value, int):
            out[key] = _pick_block(q_len, kv_len, value)
    return out


def _padding_mask(attention_mask) -> torch.Tensor | None:
    """The 2D per-token keep-mask behind ``attention_mask``, or None.

    HF's 2D mask is ``(B, S)`` int/bool with 1/True for real tokens and
    0/False for padding; that is exactly one segment-id bit. Anything else --
    a 4D additive float mask, a BlockMask -- carries structure this adapter
    does not decode (a float bias is the one thing the kernel truly cannot
    take; see ``splash_bias_dropout`` for that), so the caller falls back.
    The dtype test is deliberately shape/dtype-only: inspecting values would
    be a device sync in eager and a graph break under compile.
    """
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        return None
    if attention_mask.dtype.is_floating_point:
        return None
    return attention_mask != 0


def _unsupported(module, query, key, attention_mask, dropout, is_causal, window) -> str | None:
    """Why splash cannot take this call, or None if it can."""
    if attention_mask is not None and _padding_mask(attention_mask) is None:
        return (
            f"attention_mask of shape {tuple(attention_mask.shape)} "
            f"{attention_mask.dtype} is not a 2D padding mask; splash takes "
            "boolean structure only (causal / window / segment ids)"
        )
    q_len, kv_len = query.shape[-2], key.shape[-2]
    if _pick_block(q_len, kv_len, _MIN_BLOCK) is None:
        return f"sequence {q_len}x{kv_len} is not a multiple of {_MIN_BLOCK}"
    if query.device.type != "tpu":
        return f"tensors are on {query.device.type}, not tpu"
    if window is not None and not is_causal:
        # splash_sdpa's window is a causal LocalMask((w, 0)); a bidirectional
        # sliding window (ModernBERT's local layers) needs a symmetric one.
        return f"bidirectional sliding window {window}; the kernel's window is causal-only"
    if dropout:
        # The in-kernel PRNG generates its mask a fixed quantum at a time, and
        # asserts on a block that is not a multiple of it -- a trace-time
        # AssertionError from inside the kernel, where this gate owes the
        # caller a named fallback instead. Short encoder sequences reach it
        # (seq 128 gives a 128 block), and only with dropout: the quantum is
        # not consulted otherwise.
        problem = _dropout_block_problem(module, q_len, kv_len)
        if problem is not None:
            return problem
    return None


def _dropout_block_problem(module, q_len: int, kv_len: int) -> str | None:
    """Why the blocks this call would actually use cannot run dropout, or None.

    Checks the EFFECTIVE blocks -- the configured ``sa_*`` values clamped to
    the sequence exactly as ``_kernel_kwargs`` will hand them over (or the
    kernel's own default, clamped the same way the wrapper does) -- not just
    whether the sequence tiles the quantum: a configured block of 128 at
    sequence 256 tiles fine and still trips the kernel's assert. The PRNG
    quantum applies to the q block and the kv compute block of both the
    forward and the dkv pass (the memory kv blocks are multiples of those).
    """
    from torchtitan.experiments.tpu.kernels.splash_bias_dropout import SbdConfig

    effective = _kernel_kwargs(module, q_len, kv_len)
    for key in ("block_q", "block_kv_compute", "block_q_dkv", "block_kv_dkv_compute"):
        block = effective.get(key)
        if key not in effective:
            block = min(getattr(SbdConfig, key), q_len)
        if block is None or block % _DROPOUT_QUANTUM:
            return (
                f"{key}={block} at sequence {q_len}x{kv_len} is not a multiple of "
                f"the kernel's {_DROPOUT_QUANTUM}-token dropout quantum; zero the "
                "model's attention dropout or configure blocks that are"
            )
    return None


def _window(module, kwargs) -> int | None:
    """The layer's sliding window in HF's convention, or None.

    HF passes it to the attention function as ``sliding_window`` (Mistral and
    Mixtral set nothing on the module); others also keep it on the module.
    Only a genuine positive int is used: whatever happens to sit under that
    name on a wrapped module is how a bogus mask extent reaches the kernel.
    """
    window = kwargs.get("sliding_window")
    if window is None:
        window = getattr(module, "sliding_window", None)
    return window if isinstance(window, int) and window > 0 else None


def document_segment_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Per-document segment ids (numbered from 1) from packed ``position_ids``.

    The model computes this once per forward and passes it to every layer as
    ``tpu_segment_ids``: deriving it inside each attention call repeats the
    same work in every layer.
    """
    return segment_ids_from_positions(position_ids) + 1


def _segment_ids(module, attention_mask, kwargs) -> torch.Tensor | None:
    """Fold document boundaries and padding into one segment-id tensor.

    Documents come from the model's ``tpu_segment_ids`` (or, failing that,
    ``position_ids``, which restart at 0 per packed sample) and are numbered
    from 1; padding tokens get segment 0, the MaxText convention
    (``tfds_data_processing_c4_mlperf`` / ``tunix_adapter``). Real tokens then
    never share a segment with padding, padding attends only to padding (so
    no all-masked row goes NaN), and the loss ignores those rows anyway.
    Returns None when neither source applies, which lets the kernel run its
    plain causal / full mask with no segment pass at all.
    """
    seg = None
    if _document_masked(module):
        seg = kwargs.get("tpu_segment_ids")
        if seg is None:
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                raise ValueError(
                    "splash_tpu: document-causal attention (attn_mask_type="
                    "'block_causal') needs position_ids or tpu_segment_ids; "
                    "without them packed documents would attend across "
                    "boundaries"
                )
            seg = document_segment_ids(position_ids)
    keep = _padding_mask(attention_mask)
    if keep is not None:
        if seg is None:
            seg = torch.ones_like(keep, dtype=torch.int32)
        seg = torch.where(keep, seg, torch.zeros_like(seg))
    return seg


def _eager_attention(module, query, key, value, mask, is_causal, scaling, dropout,
                     softcap=None, sinks=None):
    """Plain attention with HF's soft-cap and/or attention sinks, for the
    fallback (SDPA has neither).

    Same arithmetic as HF's eager paths: scale, ``softcap * tanh(x / softcap)``,
    mask, softmax in fp32 -- over the scores plus one per-head sink logit
    column when ``sinks`` is given (gpt-oss, Flint), whose probability is then
    dropped -- dropout, @ v. ``mask`` is what ``_fallback_mask`` built
    (boolean or additive, broadcast over heads) or None.
    """
    heads, kv_heads = query.shape[1], key.shape[1]
    if kv_heads != heads:
        key = key.repeat_interleave(heads // kv_heads, dim=1)
        value = value.repeat_interleave(heads // kv_heads, dim=1)
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)
    if mask is None and is_causal:
        q_len, kv_len = query.shape[-2], key.shape[-2]
        mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device).tril(kv_len - q_len)
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    scores = scores.float()
    if sinks is not None:
        sink = sinks.float().reshape(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
        probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    else:
        probs = torch.softmax(scores, dim=-1)
    probs = probs.to(query.dtype)
    if dropout and module.training:
        probs = torch.nn.functional.dropout(probs, p=dropout)
    return torch.matmul(probs, value).transpose(1, 2).contiguous()


def _fallback_mask(module, query, key, attention_mask, kwargs, is_causal, window):
    """The dense equivalent of the kernel's mask, for the SDPA fallback.

    Returns ``(mask, is_causal)`` for ``sdpa_attention_forward``. Documents and
    padding come from the same ``_segment_ids`` the kernel uses; causality and
    the sliding window are folded in, so the fallback computes what the kernel
    would have. An incoming 4D mask (the reason for falling back, e.g. an
    additive float mask) is kept and narrowed, never replaced. With no
    document, padding or window structure the call is passed through
    unchanged. Cost: one (B, S, S) boolean per call -- 256 MiB at a folded
    16k stream -- acceptable on a path that is already the slow one.
    """
    q_len, kv_len = query.shape[-2], key.shape[-2]
    two_d = attention_mask if (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2) else None
    seg = _segment_ids(module, two_d, kwargs) if q_len == kv_len else None
    if seg is None and window is None:
        return (None if two_d is not None else attention_mask), is_causal
    q_pos = torch.arange(q_len, device=query.device)[:, None] + (kv_len - q_len)
    k_pos = torch.arange(kv_len, device=query.device)[None, :]
    allowed = torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device)
    if is_causal:
        allowed = allowed & (k_pos <= q_pos)
    # HF's conventions: a causal window of W keeps the W keys ending at the
    # query (``sliding_window_overlay``: q - k < W); a bidirectional one keeps
    # |q - k| <= W (``sliding_window_bidirectional_overlay``).
    if window is not None and is_causal:
        allowed = allowed & (q_pos - k_pos < window)
    elif window is not None:
        allowed = allowed & ((q_pos - k_pos).abs() <= window)
    allowed = allowed[None]
    if seg is not None:
        allowed = allowed & (seg[:, :, None] == seg[:, None, :])
    allowed = allowed[:, None]  # (B, 1, S, S): broadcast over heads
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
        if attention_mask.dtype == torch.bool:
            return attention_mask & allowed, False
        return attention_mask.masked_fill(~allowed, torch.finfo(attention_mask.dtype).min), False
    return allowed, False


def _document_masked(module) -> bool:
    """True when the flavor asked for document-causal attention.

    Mirrors ``HFTransformerModel.get_attention_masks``: with
    ``attn_mask_type == "block_causal"`` the flex path builds a document
    BlockMask so packed samples do not attend across boundaries. Splash cannot
    take that BlockMask, so this adapter expresses the same mask as
    ``segment_ids`` derived from the per-document ``position_ids`` HF threads
    through ``**kwargs``.
    """
    cfg = getattr(module, "config", None)
    return getattr(cfg, "attn_mask_type", "causal") == "block_causal"


def splash_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask=None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
):
    """HF attention implementation backed by the TPU splash-attention kernel."""
    # Same default as HF's own sdpa_attention_forward: a module that does not
    # say is the decoder-style causal one.
    if is_causal is None:
        is_causal = getattr(module, "is_causal", True)
    # Attention-logit soft-cap (Gemma-2: attn_logit_softcapping=50), which HF
    # hands every attention implementation as ``softcap``. Neither HF's SDPA
    # integration nor a kernel that ignores **kwargs applies it, so it must be
    # taken here or it is silently dropped.
    softcap = kwargs.pop("softcap", None)
    if softcap is not None and softcap <= 0:
        softcap = None
    # Attention sinks (gpt-oss, Flint): one learned logit per head in the
    # softmax denominator, handed over as ``s_aux``. The kernel adds it as an
    # extra column (and returns its gradient); the eager fallback below keeps
    # it too, so no path trains without the sinks.
    sinks = kwargs.pop("s_aux", None)
    window = _window(module, kwargs)
    reason = _unsupported(module, query, key, attention_mask, dropout, is_causal, window)
    if reason is not None:
        if not _tracing():
            _report(reason)
        # Fall back to the SAME attention, not just the same kernel interface:
        # HF hands this adapter no document mask (packing lives in
        # position_ids), so a bare SDPA call would let packed documents attend
        # across boundaries and ignore the sliding window.
        fb_mask, fb_causal = _fallback_mask(
            module, query, key, attention_mask, kwargs, is_causal, window
        )
        kwargs.pop("tpu_segment_ids", None)
        if softcap is not None or sinks is not None:
            return _eager_attention(
                module, query, key, value, fb_mask, fb_causal, scaling, dropout,
                softcap=softcap, sinks=sinks,
            ), None
        return sdpa_attention_forward(
            module, query, key, value, fb_mask,
            dropout=dropout, scaling=scaling, is_causal=fb_causal, **kwargs,
        )


    # The trainer hands the model one token-major stream per rank (batch
    # folded into the sequence: 8 x 2048 arrives as [1, 16384]). Without
    # segment ids the kernel runs ONE causal problem over the whole stream:
    # every sample attends to every earlier sample (not the semantics the
    # flex/GPU path has under block_causal) and the attention work grows
    # quadratically -- measured 2026-09-19 on Llama-3.2-1B, 4x v6e: splash
    # went from 151 ms to 618 ms per step and the step from 0.59 s to 1.15 s.
    # ``position_ids`` restart at 0 at every document boundary; turning them
    # into segment ids restores per-document masking AND lets the kernel skip
    # every block whose q and kv segments do not overlap.
    segment_ids = _segment_ids(module, attention_mask, kwargs)
    kernel_kwargs = _kernel_kwargs(module, query.shape[-2], key.shape[-2])

    if not _tracing():
        _log_once(
            "splash_tpu args",
            "splash_tpu: first call q=%s k=%s v=%s dtype=%s scaling=%s window=%s "
            "is_causal=%s dropout=%s kernel=%s",
            tuple(query.shape), tuple(key.shape), tuple(value.shape),
            query.dtype, scaling, window, is_causal, dropout, kernel_kwargs,
        )
    seg_desc = (
        "none" if segment_ids is None
        else "document+padding" if attention_mask is not None
        else "document"
    )

    out = _sbd_forward(query, key, value, segment_ids, is_causal, scaling, window,
                       dropout, kernel_kwargs, seg_desc, softcap=softcap, sinks=sinks)
    return out, None


def _sbd_forward(query, key, value, segment_ids, is_causal, scaling, window,
                 dropout, kernel_kwargs, seg_desc, softcap=None, sinks=None):
    """The kernel call: document-aware skipping, exact segment masking at
    token granularity, in-kernel PRNG dropout."""
    from torchtitan.experiments.tpu.kernels.splash_bias_dropout import (
        SbdConfig,
        sbd_splash_attention,
    )

    ignored = sorted(k for k in kernel_kwargs if k not in _SBD_KERNEL_KWARGS)
    if ignored and not _tracing():
        _log_once("splash_tpu ignored", "splash_tpu: the kernel has no %s; ignored", ignored)
    blocks = {
        _SBD_KERNEL_KWARGS[k]: v
        for k, v in kernel_kwargs.items()
        if k in _SBD_KERNEL_KWARGS
    }
    # The seed is a (B, 2) int32 tensor drawn from torch's RNG on the device:
    # fresh per call and per layer, reproducible under torch.manual_seed, and
    # a tensor operand so a new seed never retraces the op. Verified
    # 2026-10-01 on v6e that the draw survives graph capture -- three
    # successive calls on identical inputs differ under eager AND under
    # `torch.compile(backend="tpu", fullgraph=True)`, while two calls after
    # the same `manual_seed` agree bit-for-bit. A frozen draw would be
    # invisible otherwise: dropout would still *look* applied, with one mask
    # reused for the whole run.
    dropout_seed = torch.randint(
        -(2**31), 2**31 - 1, (query.shape[0], 2), dtype=torch.int32, device=query.device
    )
    # torchtitan seeds every rank identically (weights and data order depend
    # on it), so without a rank-dependent salt every data-parallel rank would
    # drop the same (position, head) entries -- the kernel's docstring asks
    # the caller to fold the rank in for exactly this reason. The salt is a
    # rank constant, so it costs nothing in a captured graph.
    salt = _seed_salt()
    if salt:
        dropout_seed = dropout_seed ^ salt
    SbdConfig.update(dropout_rate=float(dropout), **blocks)
    # Set explicitly, None included: ``update`` skips None values, which would
    # leak one model's soft-cap into the next call that has none. The kernel
    # applies it as cap * tanh(logits / cap) on the scaled logits -- HF's form.
    SbdConfig.attn_logits_soft_cap = softcap
    # HF hands (B, H, S, D) and wants it back; `heads_first` is the kernel's
    # own layout, so nothing is transposed on either side (the torch-standard
    # entry would cost three full-size copies per call, saved until the
    # backward). Grouped-KV and MQA go in as-is, no K/V expansion.
    out = sbd_splash_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        dropout_seed=dropout_seed,
        segment_ids=segment_ids,
        is_causal=is_causal,
        scale=scaling,
        # The kernel's LocalMask((w, 0)) keeps q - k <= w, i.e. w + 1 keys; HF's
        # window of W keeps W (its FA2 path passes W - 1 the same way).
        local_window_size=None if window is None else window - 1,
        heads_first=True,
        sinks=sinks,
    )
    if not _tracing():
        _log_once(
            "splash_tpu fired",
            "splash_tpu: kernel active (heads=%d kv_heads=%d seq=%d head_dim=%d "
            "causal=%s dropout=%s segment_ids=%s sinks=%s)",
            query.shape[1], key.shape[1], query.shape[-2], query.shape[-1], is_causal,
            dropout, seg_desc, sinks is not None,
        )
    # HF's contract is (B, S, H, D) out (its own sdpa path transposes too),
    # while heads-first hands back (B, H, S, D) -- the same single transpose
    # the splash path above does, with three input copies saved.
    return out.transpose(1, 2).contiguous()


def register() -> None:
    """Make `attn_implementation="splash_tpu"` selectable on any HF model."""
    AttentionInterface._global_mapping["splash_tpu"] = splash_attention_forward
    # HF's masking code early-exits -- and drops the caller's mask -- for any
    # implementation missing from its mask interface. FA2's mask function is
    # the right contract here: it returns the 2D (B, S) padding mask as given,
    # or None when nothing is padded, which is exactly what ``_padding_mask``
    # turns into segment ids. Causal / window / document structure stays this
    # adapter's job (from ``is_causal``, ``sliding_window`` and positions).
    AttentionMaskInterface._global_mapping["splash_tpu"] = flash_attention_mask


register()
