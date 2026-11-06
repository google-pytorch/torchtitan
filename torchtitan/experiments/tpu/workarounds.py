import inspect
import logging
from typing import Any, Optional
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torchtitan.tools.logging import logger


SPLASH_KEYS = {
    "block_q", "block_kv", "block_dkv", "block_kv_compute",
    "block_q_dkv", "block_kv_dkv", "block_kv_dkv_compute",
    "block_q_dq", "block_kv_dq", "use_fused_bwd_kernel",
    "use_vmap_bwd", "use_document_aware_block_skipping",
    "q_layout", "k_layout", "v_layout"
}


def extract_splash_kwargs(sa_config: Any) -> dict[str, Any]:
  """Safe prefix-mapping helper to map sa_config attributes to standard names."""
  if sa_config is None:
    return {}
  kwargs = {}
  for k in SPLASH_KEYS:
    val = getattr(sa_config, f"sa_{k}", getattr(sa_config, k, None))
    if val is not None:
      kwargs[k] = val
  return kwargs


def use_splash_attention_patch(model: nn.Module, sa_config: Any = None) -> None:
  """Patches splash attention into the model."""
  from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa as standard_splash_sdpa  # pylint: disable=g-import-not-at-top
  import types

  # Extract parameters internally using our safe prefix mapper
  sa_kwargs = extract_splash_kwargs(sa_config)

  def _splash_forward(
      self,
      q,
      k,
      v,
      *,
      scale=None,
      enable_gqa=False,
      is_causal=True,
      **extra_args,
  ):
    """Replace ScaledDotProductAttention.forward with splash_sdpa.

    Reads ``sliding_window_size`` from the patched module if set; otherwise
    behaves as before (causal or full mask depending on ``is_causal``). Most
    torchtitan wrappers don't expose this attribute → defaults to None.
    """
    if extra_args.get("attention_masks") is not None:
      raise ValueError(
          "TPU SplashAttention Pallas kernel does not support runtime"
          " attention_masks."
      )

    # TorchTitan passes tensors as (B, S, H, D). However, the splash_sdpa Pallas kernel
    # expects (B, H, S, D). We transpose them here to match the kernel's expected layout.
    # We also prevent torch.compile failures by ensuring tensors are contiguous for the
    # custom Pallas kernel, which doesn't handle arbitrary strides. This is a
    # no-op in eager mode if the tensors are already contiguous.
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    local_window_size = getattr(self, "sliding_window_size", None)


    out = standard_splash_sdpa(
        q,
        k,
        v,
        scale=scale,
        is_causal=is_causal,
        local_window_size=local_window_size,
        enable_gqa=enable_gqa,
        **sa_kwargs,
    )
    # Transpose back to (B, S, H, D) so the rest of the TorchTitan model receives its expected format
    return out.transpose(1, 2)

  def _splash_sdpa_native(self, *, query, key, value, scale=None, **kwargs):
    """Replaces a module's ``_sdpa_native_torch`` method with splash_sdpa.

    For attention modules that compute SDPA through an overridable
    ``_sdpa_native_torch(query=, key=, value=, scale=)`` method on (B, S, H, D)
    inputs, and tile k/v heads for GQA (``_maybe_tile_qkv_for_gqa``) before
    invoking it. This wrapper applies the same GQA tiling, transposes inputs to
    (B, H, S, D) for splash_sdpa, and transposes the output back.

    For mixed local/global attention models, reads
    ``self.sliding_window_size`` per-layer:
      * None  → full causal (CausalMask).
      * int N → causal sliding-window (LocalMask with window_size=(N, 0)).
                The splash kernel's per-window cache means each unique window
                size compiles its own kernel; mixing windowed and non-windowed
                layers in one model is supported.
    """
    query, key, value = self._maybe_tile_qkv_for_gqa(query, key, value)
    # Transpose from (B, S, H, D) to (B, H, S, D).
    query = query.transpose(-3, -2)
    key = key.transpose(-3, -2)
    value = value.transpose(-3, -2)

    # Prevent torch.compile failures by ensuring tensors are contiguous for the
    # custom Pallas kernel, which doesn't handle arbitrary strides. This is a
    # no-op in eager mode if the tensors are already contiguous.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # Per-layer mask selection: `sliding_window_size` is None for global and
    # an int for local layers. Models without local layers don't set the
    # attribute at all; default to None (full causal).
    local_window_size = getattr(self, "sliding_window_size", None)

    # splash_sdpa expects enable_gqa=False as heads are already tiled.
    out = standard_splash_sdpa(
        query,
        key,
        value,
        scale=scale,
        is_causal=True,
        local_window_size=local_window_size,
        enable_gqa=False,
        **sa_kwargs,
    )
    # Transpose back to the caller's (B, S, H, D) layout
    return out.transpose(-3, -2).contiguous()

  # Patch all attention modules in the model, noting whether each layer has a
  # local sliding-window — useful for mixed local/global models so the kernel
  # cache shows the expected mix of CausalMask and LocalMask compiled ops.
  from torchtitan.experiments.tpu.kernels.splash_attention import SplashAttention  # pylint: disable=g-import-not-at-top

  replacements = []
  _patched = 0
  _windowed = 0
  _causal = 0

  # Identify attention modules to replace
  # (ScaledDotProductAttention & FlexAttention).
  for parent_module in model.modules():
    for child_name, child_module in parent_module.named_children():
      cls_name = type(child_module).__name__

      # NOTE: matches both ScaledDotProductAttention and torchtitan's
      # ScaledDotProductInnerAttention (renamed upstream in d771abb0); the
      # latter does not contain the old name as a substring, so the old check
      # silently matched nothing.
      if "ScaledDotProduct" in cls_name:
        if hasattr(child_module, "_sdpa_native_torch"):
          # In-place method replacement
          child_module._sdpa_native_torch = types.MethodType(
              _splash_sdpa_native, child_module
          )
          _patched += 1
          if getattr(child_module, "sliding_window_size", None) is not None:
            _windowed += 1
          else:
            _causal += 1
        else:
          replacements.append((parent_module, child_name, child_module))
      elif "FlexAttention" in cls_name:
        replacements.append((parent_module, child_name, child_module))

  # Apply replacements
  for parent_module, child_name, child_module in replacements:
    # SplashAttention needs the window size to pick its mask: LocalMask for a
    # sliding-window layer, CausalMask otherwise. The child is the inner
    # attention being replaced (ScaledDotProductAttention / FlexAttention), the
    # parent the attention module that owns it (e.g. GQAttention). Models set
    # sliding_window_size on either one, so check the child then the parent.
    local_window_size = getattr(child_module, "sliding_window_size", None)
    if local_window_size is None:
      local_window_size = getattr(parent_module, "sliding_window_size", None)

    # Ulysses CP inner attentions keep their all-to-all and take the kernel as
    # their local attention; see experiments/tpu/cp_attention.py.
    is_cp = hasattr(child_module, "set_local_attention")
    splash_attn = SplashAttention(
        is_causal=True,  # Default for LMs
        local_window_size=local_window_size,
        original_module=child_module.local_forward if is_cp else child_module,
        **sa_kwargs,
    )
    # TT-UNFLATTEN-SHIM:
    from torchtitan.experiments.tpu.unflatten_shim import UnflattenShim
    splash_attn = UnflattenShim(splash_attn)

    if is_cp:
      child_module.set_local_attention(splash_attn)
    else:
      setattr(parent_module, child_name, splash_attn)
    _patched += 1
    if local_window_size is not None:
      _windowed += 1
    else:
      _causal += 1

  if _patched == 0:
    all_classes = sorted(list({type(m).__name__ for _, m in model.named_modules()}))
    raise ValueError(
        "Splash attention patch was requested but NO matching attention"
        " modules (ScaledDotProduct*, FlexAttention) were found in the"
        f" model. Found module types: {all_classes}"
    )
  else:
    logger.info(
        "Splash attention ENABLED: replaced %d attention modules "
        "(causal=%d, sliding-window=%d)",
        _patched,
        _causal,
        _windowed,
    )
  if any(
      "KVAllGatherCP" in type(child).__name__ for _, _, child in replacements
  ) and sa_kwargs.get("use_document_aware_block_skipping", True):
    logger.warning(
        "Splash attention: use_document_aware_block_skipping=True is not"
        " supported with KV all-gather CP; falling back to batch-native masking"
        " (packed documents are still masked exactly via segment_ids). Set"
        " sa_use_document_aware_block_skipping=False to acknowledge."
    )


def use_tokamax_splash_attention_patch(model: nn.Module, sa_config: Any) -> None:
  """Patches zero-overhead Tokamax layout-fused splash attention into the model."""
  from torchtitan.experiments.tpu.kernels.tokamax_splash_attention import TokamaxConfig  # pylint: disable=g-import-not-at-top
  from torchtitan.experiments.tpu.kernels.tokamax_splash_attention import splash_sdpa as tokamax_splash_sdpa  # pylint: disable=g-import-not-at-top
  from torchtitan.models.common.attention import segment_ids_from_positions  # pylint: disable=g-import-not-at-top
  import types

  # Update core JAX trace compilation configs internally from config registry
  # Every SplashAttentionKernelConfig field is sa_-prefixed with no unprefixed
  # alias, so an unprefixed getattr returns None for all of them and
  # TokamaxConfig.update -- which skips None -- silently keeps its own defaults,
  # discarding the recipe's entire tuned block config. Read the sa_ names and
  # validate that something resolved. Listing the fields explicitly also makes
  # this call the whitelist: it cannot forward a key TokamaxConfig lacks.
  def _sa(name):
    return getattr(sa_config, f"sa_{name}", getattr(sa_config, name, None))

  _tokamax_values = dict(
      block_q=_sa("block_q"),
      block_kv=_sa("block_kv"),
      block_kv_compute=_sa("block_kv_compute"),
      block_q_dkv=_sa("block_q_dkv"),
      block_kv_dkv=_sa("block_kv_dkv"),
      block_kv_dkv_compute=_sa("block_kv_dkv_compute"),
      block_q_dq=_sa("block_q_dq"),
      block_kv_dq=_sa("block_kv_dq"),
      use_fused_bwd_kernel=_sa("use_fused_bwd_kernel"),
      use_vmap_bwd=_sa("use_vmap_bwd"),
      num_stacked_q_heads=_sa("num_stacked_q_heads"),
      use_batch_loop_fwd=_sa("use_batch_loop_fwd"),
      q_layout=_sa("q_layout"),
      k_layout=_sa("k_layout"),
      v_layout=_sa("v_layout"),
  )
  # Guard the bug above, without rejecting a caller that deliberately passes an empty
  # config to mean "just use the kernel's defaults" (some parity tests do exactly
  # that). The two are distinguishable: a real SplashAttentionKernelConfig exposes
  # sa_-prefixed attributes, so if those exist and still nothing resolved, the reader
  # is broken; if none exist, the caller genuinely wants defaults - log it so the
  # fallback is visible rather than silent.
  _is_splash_config = any(a.startswith("sa_") for a in dir(sa_config))
  if _is_splash_config and all(v is None for v in _tokamax_values.values()):
    raise ValueError(
        "Tokamax splash attention was requested but no splash_attention_kernel "
        f"values resolved from {type(sa_config).__name__}, which does expose "
        "sa_-prefixed fields. The kernel would silently run at library defaults "
        "instead of the recipe's tiles."
    )
  if not _is_splash_config:
    logger.info(
        "Tokamax splash: %s exposes no sa_-prefixed fields; running at the "
        "kernel's default tiling.", type(sa_config).__name__,
    )
  TokamaxConfig.update(**_tokamax_values)
  logger.info(
      "Tokamax splash: applied %s",
      {k: v for k, v in _tokamax_values.items() if v is not None},
  )

  def _get_kwargs(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}

  # SPLASH_KEYS is the union over splash implementations; tokamax's splash_sdpa
  # accepts a strict subset of it (e.g. it has no use_document_aware_block_skipping),
  # so filtering against the static set passes kwargs the callee rejects with
  # "Unexpected keyword arguments". Derive the filter from the callee's actual
  # signature instead, so it tracks the kernel rather than drifting from it.
  _tokamax_sdpa_params = frozenset(
      inspect.signature(tokamax_splash_sdpa).parameters
  )

  def _filter_splash_kwargs(kwargs_dict):
    # PURE: this runs inside the traced forward (scan body). A logger call here
    # graph-breaks Dynamo with "logging.Logger method not supported for
    # non-export cases" - log once at patch time instead, below.
    return {k: v for k, v in kwargs_dict.items() if k in _tokamax_sdpa_params}

  # Extract parameters dynamically using our shared prefix mapper helper!
  sa_kwargs = extract_splash_kwargs(sa_config)
  _dropped_sa = sorted(set(sa_kwargs) - _tokamax_sdpa_params)
  if _dropped_sa:
    logger.info(
        "Tokamax splash: %d/%d config kwargs forwarded; dropping %s"
        " (not in tokamax splash_sdpa's signature)",
        len(sa_kwargs) - len(_dropped_sa), len(sa_kwargs), _dropped_sa,
    )

  def _splash_forward(
      self,
      q,
      k,
      v,
      *,
      scale=None,
      enable_gqa=False,
      is_causal=True,
      **extra_args,
  ):
    # TT-UNFLATTEN-SHIM:
    from torchtitan.experiments.tpu.unflatten_shim import unflatten_tensor, flatten_tensor
    was_flat = q.ndim == 3
    q = unflatten_tensor(q, expected_flat_ndim=3)
    k = unflatten_tensor(k, expected_flat_ndim=3)
    v = unflatten_tensor(v, expected_flat_ndim=3)
    for kw, val in extra_args.items():
      if isinstance(val, torch.Tensor) and kw in ("positions", "segment_ids"):
        extra_args[kw] = unflatten_tensor(val, expected_flat_ndim=1)

    if extra_args.get("attention_masks") is not None:
      raise ValueError(
          "TPU SplashAttention Pallas kernel does not support runtime"
          " attention_masks."
      )

    local_window_size = getattr(self, "sliding_window_size", None)

    # Same fallback as SplashAttention.forward: Decoder normally computes
    # segment_ids once per step, but a caller passing only `positions` must
    # still get packed-document masking rather than silently losing it.
    if (
        extra_args.get("segment_ids") is None
        and extra_args.get("positions") is not None
    ):
      extra_args["segment_ids"] = segment_ids_from_positions(
          extra_args["positions"]
      )

    # Tokamax JAX-side transpositions path (Direct pass-through of contiguous layouts)
    # The jax_op bridge reads operands by shape alone, so every tensor operand,
    # segment_ids included, must be dense.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if extra_args.get("segment_ids") is not None:
      extra_args["segment_ids"] = extra_args["segment_ids"].contiguous()
    combined_args = {**extra_args, **sa_kwargs}
    forward_kwargs = _filter_splash_kwargs(combined_args)
    out = tokamax_splash_sdpa(
        q,
        k,
        v,
        scale=scale,
        is_causal=is_causal,
        local_window_size=local_window_size,
        enable_gqa=enable_gqa,
        **forward_kwargs,
    )
    # TT-UNFLATTEN-SHIM:
    if not was_flat:
      return out
    return flatten_tensor(out, original_flat_ndim=3)

  def _splash_sdpa_native(self, *, query, key, value, scale=None, **kwargs):
    # K/V stay at num_kv_heads: the kernel groups q head h onto kv head
    # h // (num_heads // num_kv_heads), the same mapping as GQA tiling,
    # without materializing num_heads copies of K/V.
    local_window_size = getattr(self, "sliding_window_size", None)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    combined_args = {**kwargs, **sa_kwargs}
    forward_kwargs = _filter_splash_kwargs(combined_args)
    out = tokamax_splash_sdpa(
        query,
        key,
        value,
        scale=scale,
        is_causal=True,
        local_window_size=local_window_size,
        enable_gqa=query.shape[2] != key.shape[2],
        **forward_kwargs,
    )
    return out

  _patched = 0
  _patched_forward = 0
  _windowed = 0
  _causal = 0
  for _, module in model.named_modules():
    cls_name = type(module).__name__
    # NOTE: matches both ScaledDotProductAttention and torchtitan's
    # ScaledDotProductInnerAttention (renamed upstream in d771abb0); the
    # latter does not contain the old name as a substring, so the old check
    # silently matched nothing.
    if "ScaledDotProduct" in cls_name:
      if hasattr(module, "_sdpa_native_torch"):
        module._sdpa_native_torch = types.MethodType(
            _splash_sdpa_native, module
        )
      elif hasattr(module, "set_local_attention"):  # Ulysses CP
        module.set_local_attention(types.MethodType(_splash_forward, module))
        _patched_forward += 1
      else:
        module.forward = types.MethodType(_splash_forward, module)
        _patched_forward += 1
      _patched += 1
      if getattr(module, "sliding_window_size", None) is not None:
        _windowed += 1
      else:
        _causal += 1

  if _patched == 0:
    all_classes = sorted(list({type(m).__name__ for _, m in model.named_modules()}))
    raise ValueError(
        "Tokamax splash attention patch was requested but NO matching "
        f"attention modules (ScaledDotProduct*) were found in the model. Found module types: {all_classes}"
    )
  else:
    logger.info(
        "Tokamax splash attention ENABLED: patched %d attention modules "
        "(causal=%d, sliding-window=%d)",
        _patched, _causal, _windowed,
    )

  # Only modules patched via forward receive segment_ids, so only they are
  # affected. `_sdpa_native_torch` modules run with the field at its default and
  # must not warn.
  if _patched_forward and sa_kwargs.get("use_document_aware_block_skipping"):
    logger.warning(
        "Tokamax splash: use_document_aware_block_skipping=True is not"
        " supported; packed documents are still masked exactly via segment_ids,"
        " but cross-document blocks are computed and discarded. Set"
        " sa_use_document_aware_block_skipping=False to acknowledge, or use the"
        " stock splash kernel for block skipping."
    )


class PallasLossLinear(nn.Module):
  """Wrapper for model.output to return (x, weight) for Pallas loss.

  Holding the Linear module as a child names the parameter
  ``lm_head.original_linear.weight`` which is not mapped by model state dict
  adapters and breaks loading from checkpoint with a KeyError. We add hooks to
  fix at state-dict layer to strip the ``original_linear`` infix on save and
  restore on load, making the wrapper invisible to checkpointing paths but
  preserving module tree hierarchy.
  """

  _INFIX = "original_linear."

  def __init__(self, original_linear: nn.Linear):
    super().__init__()
    self.original_linear = original_linear
    self._register_state_dict_hook(PallasLossLinear._strip_infix)
    self._register_load_state_dict_pre_hook(self._restore_infix)

  @staticmethod
  def _strip_infix(module, state_dict, prefix, local_metadata):
    """Rename ``<prefix>original_linear.X`` -> ``<prefix>X`` on the way out."""
    for key in [k for k in state_dict if k.startswith(prefix + module._INFIX)]:
      state_dict[prefix + key[len(prefix + module._INFIX):]] = state_dict.pop(key)
    return state_dict

  def _restore_infix(self, state_dict, prefix, *args, **kwargs):
    """Rename ``<prefix>X`` -> ``<prefix>original_linear.X`` on the way in."""
    for key in [
        k
        for k in state_dict
        if k.startswith(prefix) and not k.startswith(prefix + self._INFIX)
    ]:
      state_dict[prefix + self._INFIX + key[len(prefix):]] = state_dict.pop(key)

  @property
  def weight(self) -> torch.Tensor:
    # Dynamically retrieve the weight to avoid caching a stale, unsharded
    # reference before FSDP sharding.
    return self.original_linear.weight

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x, self.weight.t()


def _check_lm_head_is_plain_projection(lm_head: nn.Linear) -> None:
  """Raises if the fused loss would skip work done in ``lm_head.forward``.

  ``PallasLossLinear`` returns ``(x, weight.T)`` and never calls the head's
  ``forward``, so the fused loss computes plain linear cross-entropy. That is
  only correct for a head whose ``forward`` is ``nn.Linear``'s (or one that
  already returns those operands). E.g. Muse Glimmer's ``SoftCappedLinear``
  applies an output multiplier and tanh soft-cap in ``forward``; wrapping it
  would silently change the loss and gradients.

  Args:
    lm_head: The output projection about to be wrapped.

  Raises:
    ValueError: If ``lm_head`` overrides ``forward`` with extra computation.
  """
  if getattr(lm_head, "returns_loss_operands", False):
    return  # E.g. FusedLossLinearWrapper: already returns (x, weight.T).
  if type(lm_head).forward is not nn.Linear.forward:
    raise ValueError(
        "--loss_kernel.use_loss_kernel cannot be used with"
        f" {type(lm_head).__name__} as lm_head: it overrides forward (e.g. an"
        " output multiplier or logit soft-cap), which the fused Pallas loss"
        " would silently skip. Disable the loss kernel for this model."
    )


def use_output_projection_patch(
    model: nn.Module,
) -> None:
  """Replaces model.output with an identity-like module that returns (x, weight)."""
  if hasattr(model, "output") and isinstance(model.output, nn.Linear):
    model.output = PallasLossLinear(model.output)
    logger.info("Patched model.output to return (x, weight) for Pallas loss")
  elif hasattr(model, "output_transform") and isinstance(model.output_transform, nn.Linear):
    # Some models name their output projection layer output_transform
    model.output_transform = PallasLossLinear(model.output_transform)
    logger.info(
        "Patched model.output_transform to return (x, weight) for Pallas loss"
    )
  elif hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
    # Original torchtitan models (e.g. Llama) use lm_head for output projection
    _check_lm_head_is_plain_projection(model.lm_head)
    model.lm_head = PallasLossLinear(model.lm_head)
    logger.info("Patched model.lm_head to return (x, weight) for Pallas loss")
  else:
    logger.warning("Failed to patch output projection layer. Model output projection layer not found (output, output_transform, or lm_head) or is not nn.Linear.")


def use_gmm_kernel_patch(
    model: nn.Module,
) -> None:
  """Enables the GMM kernel for GroupedExperts modules in the model.

  This workload-specific monkey patch directly intercepts and overrides the
  `_run_experts_grouped_mm` runner inside Torchtitan's MoE module. This allows
  us to synchronously route execution to our locally compiled JAX/Pallas
  kernel (`gmm.grouped_matrix_multiply`).
  """
  logger.info("Applying GMM Kernel patch to Torchtitan's MoE module.")
  from torchtitan.experiments.tpu.kernels import gmm  # pylint: disable=g-import-not-at-top
  import torchtitan.models.common.moe as torchtitan_moe  # pylint: disable=g-import-not-at-top

  # Explicitly intercept and rewrite the Torchtitan inner executor
  # for `_run_experts_grouped_mm`. This is significantly more robust than
  # hijacking PyTorch's ATen dispatch because it guarantees we call the custom
  # Pallas kernel synchronously without torch._grouped_mm interfering.
  def _custom_run_experts_grouped_mm(
      w1: torch.Tensor,
      w2: torch.Tensor,
      w3: torch.Tensor,
      x: torch.Tensor,
      num_tokens_per_expert: torch.Tensor,
  ) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    h = F.silu(
        gmm.grouped_matrix_multiply(
            x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets
        )
    )
    h = h * gmm.grouped_matrix_multiply(
        x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )
    out = gmm.grouped_matrix_multiply(
        h, w2.bfloat16().transpose(-2, -1), offs=offsets
    ).type_as(x)
    return out

  # Bypass the Triton-based padding dependency that Torchtitan injects for
  # non-EP setups. Pallas naturally handles the unpadded jagged offsets.
  def bypassed_padding_wrapper(fn):
    logger.info(
        "indices_padding_wrapper bypassed since Pallas kernel handles padding."
    )
    return fn

  torchtitan_moe.indices_padding_wrapper = bypassed_padding_wrapper  # pyrefly: ignore[missing-attribute]
  torchtitan_moe._run_experts_grouped_mm = _custom_run_experts_grouped_mm


def use_fill_indices_patch(
    model: nn.Module,
) -> None:
  """Enables the fill indices kernel for MoE modules in the model.

  This workload-specific monkey patch replaces the default implementations in
  torchtitan.models.moe.kernels with our TPU-specific JAX implementation.
  """
  # from torchtitan.experiments.tpu.kernels import fill_indices as tpu_fill_indices
  # from torchtitan.models.moe import kernels as moe_kernels

  # # Patch both because generate_permute_indices defaults to use_cpu=True
  # moe_kernels.fill_indices_wrapper = tpu_fill_indices.fill_indices
  # moe_kernels.fill_indices_cpu = tpu_fill_indices.fill_indices
  # logger.info("Patched MoE kernels to use TPU fill_indices")
  # TODO Evaluate if we can remove this since fill_indices no longer exists.
  pass


def use_cpu_safe_histc_patch() -> None:
  """Monkey patches torch.histc to support int dtypes on CPU."""
  original_histc = torch.histc

  def _cpu_safe_histc(input, bins=100, min=0, max=0, *, out=None):
    if input.device.type == "cpu":
      return original_histc(input.float(), bins, min, max, out=out).to(input.dtype)
    return original_histc(input, bins, min, max, out=out)

  torch.histc = _cpu_safe_histc
  logger.info("Monkey-patched torch.histc to support int dtypes on CPU")


def _check_fused_loss_logit_transform(model: nn.Module, job_config: Any) -> None:
  """Raises if the fused loss would apply a different logit transform.

  ``FusedLossLinearWrapper`` replaces the LM head at config time and records the
  replaced head's ``soft_cap`` / ``output_multiplier``, but never applies them:
  ``PallasCrossEntropyLoss`` does, from its own copy of the two values. If the
  copies diverge (e.g. a recipe builds the loss for one Muse Glimmer flavor and
  then swaps in another), training would silently use the wrong logits.

  Args:
    model: The built model (or one pipeline stage of it).
    job_config: The TPU job config whose loss will consume the head's output.

  Raises:
    ValueError: If the head and the loss disagree, or if the head is a fused
      loss wrapper but no fused loss is configured.
  """
  lm_head = getattr(model, "lm_head", None)
  # FusedLossLinearWrapper marks itself with `returns_loss_operands`.
  if not getattr(lm_head, "returns_loss_operands", False):
    return
  # Matched by fields rather than by type: importing loss.py here would create
  # a models -> workarounds -> loss -> models dependency cycle.
  loss_config = getattr(job_config, "loss", None)
  if hasattr(loss_config, "soft_cap") and hasattr(
      loss_config, "output_multiplier"
  ):
    loss = (loss_config.soft_cap, loss_config.output_multiplier)
  elif getattr(
      getattr(job_config, "loss_kernel", None), "use_loss_kernel", False
  ):
    # loss.build_cross_entropy_loss's legacy path: an untransformed fused loss.
    loss = (None, 1.0)
  else:
    raise ValueError(
        f"lm_head is a {type(lm_head).__name__} but the loss is"
        f" {type(loss_config).__name__}; the wrapper only returns"
        " (hidden, weight) operands for PallasCrossEntropyLoss."
    )
  head = (lm_head.soft_cap, lm_head.output_multiplier)
  if head != loss:
    raise ValueError(
        "Fused-loss logit transform mismatch: lm_head was replaced with"
        f" (soft_cap, output_multiplier)={head}, but the fused loss is"
        f" configured with {loss}."
    )


def disable_pinned_host_memory() -> None:
  """Makes the data pipeline allocate pageable batches instead of pinned ones.

  Multi-slice jobs run on the MegaScale PJRT client, which implements no
  ``HostMemoryAllocator`` (see ``PjrtBackend::GetHostAllocator`` in
  torch_tpu's ``csrc/pjrt/pjrt_state.cc``). The collator's page-locked staging
  buffers go through torch_tpu's ``TpuPinnedAllocator``, which needs one, and
  the failure arrives as an absl ``FATAL`` from inside the PJRT C API --
  ``HostMemoryAllocator not implemented for client`` -- so it aborts the
  process rather than raising something catchable. The allocation therefore has
  to be avoided up front.

  ``TextCollator`` reads ``HAS_PIN_MEMORY`` as a module global on every call, so
  rebinding it before the first batch is built is sufficient. Costs the async
  host-to-device copy: a copy out of pageable memory is synchronous whatever
  ``non_blocking`` says.
  """
  from torchtitan.components.data import collators  # pylint: disable=g-import-not-at-top

  if not collators.HAS_PIN_MEMORY:
    return
  collators.HAS_PIN_MEMORY = False
  logger.info(
      "Multi-slice: allocating pageable batches. The MegaScale PJRT client"
      " implements no host memory allocator, so a pinned allocation would"
      " abort the process; this gives up the async host-to-device copy."
  )


def apply_patches(model: nn.Module, job_config: Any) -> None:
  """Applies all TPU-specific patches and workarounds to the model."""
  _check_fused_loss_logit_transform(model, job_config)

  # Enable CPU safe histc workaround.
  use_cpu_safe_histc_patch()

  if job_config.parallelism.context_parallel_degree > 1:
    from torchtitan.experiments.tpu.cp_attention import install_tpu_cp_patches  # pylint: disable=g-import-not-at-top

    # Runtime patch, so install it here (in every process, right before
    # parallelize) rather than in the config.
    install_tpu_cp_patches()

  # Splash Attention Patch
  if hasattr(job_config, "splash_attention_kernel"):
    sa_config = job_config.splash_attention_kernel
    if sa_config.use_tokamax_splash_attention_kernel:
      use_tokamax_splash_attention_patch(model, sa_config)
    elif sa_config.use_splash_attention_kernel:
      use_splash_attention_patch(model, sa_config)

  # Pallas Loss Patch (Output Projection)
  if (
      hasattr(job_config, "loss_kernel")
      and job_config.loss_kernel.use_loss_kernel
  ):
    use_output_projection_patch(model)

  # Qwen3 GMM MoE Patches
  if hasattr(job_config, "qwen3") and job_config.qwen3.use_gmm_kernel:
    use_gmm_kernel_patch(model)
  if hasattr(job_config, "qwen3") and job_config.qwen3.use_fill_indices_kernel:
    use_fill_indices_patch(model)

  # TT-UNFLATTEN-SHIM: run dense blocks on [B, L, ...] views for TPU layouts.
  tpu_config = getattr(job_config, "tpu_config", None)
  if tpu_config is not None and tpu_config.bld_block_layout:
    from torchtitan.experiments.tpu.bld_layout import apply_bld_layout, bld_skip_reason  # pylint: disable=g-import-not-at-top

    apply_bld_layout(
        model,
        skip_reason=bld_skip_reason(job_config.parallelism),
    )


def raise_dynamo_recompile_limits(model_config) -> None:
  """Size Dynamo's recompile limits from ``model_config.layers``.

  Called by both TPU trainers before the first compiled forward. HF
  transformers backend configs have no ``layers`` list; their TPU parallelize
  calls ``raise_dynamo_recompile_limits_for_num_layers`` with the built model's
  layer count instead.
  """
  raise_dynamo_recompile_limits_for_num_layers(
      len(getattr(model_config, "layers", ()) or ())
  )


def raise_dynamo_recompile_limits_for_num_layers(num_layers: int) -> None:
  """Size Dynamo's recompile limits for per-layer compile under SimpleFSDP.

  Parametrization / AC wrapping gives every decoder layer its own module
  class, so Dynamo recompiles the shared checkpoint-wrapper ``forward`` once
  per layer (a type guard miss, not a real recompile), and once more per layer
  for validation, which runs under no_grad. torch's default recompile_limit of
  8 exactly fits the 8-layer debug models and fails hard (fullgraph) at 52/80
  layers.
  """
  import torch._dynamo.config as dynamo_config  # pylint: disable=g-import-not-at-top

  wanted = max(64, 2 * num_layers + 16)
  dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, wanted)
  dynamo_config.accumulated_recompile_limit = max(
      dynamo_config.accumulated_recompile_limit, 8 * wanted
  )
