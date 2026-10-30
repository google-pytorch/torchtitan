"""Flint (RecursiveSI/flint) on the common TorchTitan decoder blocks.

Flint is a Qwen3-MoE-style decoder with four deviations, each handled here or
in the flavor builders (``__init__.py``):

* Per-head attention sinks (HF ``sinks``, checkpoint ``softmax_offset``): one
  learned logit per head joins the softmax and attends to nothing.
* Sliding-window attention (window 513, HF convention ``0 <= q - k < 513``) on
  35 of 49 layers; RoPE only on the sliding layers, the full-attention layers
  are NoPE.
* A sigmoid router without expert bias whose top-k scores are renormalized.
* An ungated SwiGLU shared expert added to every MoE layer; layers 0 and 1 are
  dense SwiGLU.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_remat as remat
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.attention import (
    AttentionMasksType,
    GQAttention,
    InnerAttention,
    segment_ids_from_positions,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.config_utils import (
    moe_load_balance_aux_loss_configs,
    update_moe_aux_loss_configs,
)
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module


def _unflatten_to_4d(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
  """TT-UNFLATTEN-SHIM: reshapes folded ``[T, H, D]`` to ``[B, L, H, D]``.

  The row length comes from ``set_unflatten_seq_len``; without it the whole
  token stream is one sequence (``ScaledDotProductInnerAttention`` does the
  same).
  """
  if q.ndim != 3:
    return q, k, v, False
  seq_len = None
  try:
    from torchtitan.experiments.tpu.unflatten_shim import get_unflatten_seq_len  # pylint: disable=g-import-not-at-top

    seq_len = get_unflatten_seq_len()
  except ImportError:
    pass
  num_tokens = q.shape[0]
  if seq_len is None:
    seq_len = num_tokens
  elif num_tokens % seq_len != 0:
    raise ValueError(
        f"TT-UNFLATTEN-SHIM: num_tokens ({num_tokens}) is not divisible by"
        f" seq_len ({seq_len})."
    )
  batch_size = num_tokens // seq_len
  return (
      q.reshape(batch_size, seq_len, *q.shape[1:]),
      k.reshape(batch_size, seq_len, *k.shape[1:]),
      v.reshape(batch_size, seq_len, *v.shape[1:]),
      True,
  )


class FlintSinkAttention(InnerAttention):
  """Causal attention with an optional sliding window and attention sinks.

  The reference path for Flint: scores and softmax in float32, as HF's eager
  Flint attention computes them (QK-normed logits reach tens). Each head's
  sink logit is an extra softmax column with no value. Packed documents are
  masked from ``segment_ids`` (or ``positions``), so a token never attends
  across a document boundary.

  The class name deliberately avoids ``ScaledDotProduct``: the generic TPU
  splash patches match that substring and would drop the sinks.

  Window convention: ``window_size=W`` keeps ``0 <= q - k < W`` (HF's
  ``sliding_window``). ``sliding_window_size`` exposes the splash ``LocalMask``
  convention (``W - 1`` previous keys).
  """

  @dataclass(kw_only=True, slots=True)
  class Config(InnerAttention.Config):
    window_size: int | None = None
    # Run the sbd Splash Pallas kernel, which takes the sinks as an extra
    # softmax column and returns their gradient (TPU only).
    use_splash_kernel: bool = False

  def __init__(self, config: Config) -> None:
    super().__init__()
    if config.window_size is not None and config.window_size < 1:
      raise ValueError(
          f"window_size must be >= 1 or None, got {config.window_size}"
      )
    self.window_size = config.window_size
    self.use_splash_kernel = config.use_splash_kernel

  @property
  def sliding_window_size(self) -> int | None:
    return None if self.window_size is None else self.window_size - 1

  def forward(
      self,
      q_BLHK: torch.Tensor,
      k_BLHK: torch.Tensor,
      v_BLHV: torch.Tensor,
      *,
      sinks: torch.Tensor | None = None,
      attention_masks: AttentionMasksType | None = None,
      scale: float | None = None,
      enable_gqa: bool = False,
      is_causal: bool = True,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
      **kwargs,
  ) -> torch.Tensor:
    if attention_masks is not None:
      raise ValueError("FlintSinkAttention does not take attention_masks.")
    if not is_causal:
      raise ValueError("Flint attention is always causal.")
    if segment_ids is None and positions is not None:
      segment_ids = segment_ids_from_positions(positions)

    q_BLHK, k_BLHK, v_BLHV, was_3d = _unflatten_to_4d(q_BLHK, k_BLHK, v_BLHV)
    batch_size, seq_len, n_heads, head_dim = q_BLHK.shape
    n_kv_heads = k_BLHK.shape[2]
    if scale is None:
      scale = head_dim**-0.5

    if self.use_splash_kernel:
      from torchtitan.experiments.tpu.kernels.splash_bias_dropout import sbd_splash_attention  # pylint: disable=g-import-not-at-top

      out_BLHV = sbd_splash_attention(
          q_BLHK.contiguous(),
          k_BLHK.contiguous(),
          v_BLHV.contiguous(),
          segment_ids=None
          if segment_ids is None
          else segment_ids.reshape(batch_size, seq_len),
          is_causal=True,
          scale=scale,
          local_window_size=self.sliding_window_size,
          sinks=None if sinks is None else sinks.float(),
      )
      if was_3d:
        return out_BLHV.reshape(-1, *out_BLHV.shape[2:])
      return out_BLHV

    # [B, H, L, D]; KV heads repeated for GQA.
    q = q_BLHK.transpose(1, 2)
    k = k_BLHK.transpose(1, 2)
    v = v_BLHV.transpose(1, 2)
    if n_kv_heads != n_heads:
      k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)
      v = v.repeat_interleave(n_heads // n_kv_heads, dim=1)

    scores = torch.matmul(q.float(), k.float().transpose(2, 3)) * scale

    pos = torch.arange(seq_len, device=q.device)
    distance = pos[:, None] - pos[None, :]
    keep = distance >= 0
    if self.window_size is not None:
      keep = keep & (distance < self.window_size)
    keep = keep[None, None]  # [1, 1, L, L]
    if segment_ids is not None:
      seg = segment_ids.reshape(batch_size, seq_len)
      keep = keep & (seg[:, None, :, None] == seg[:, None, None, :])
    scores = scores.masked_fill(~keep, float("-inf"))

    if sinks is not None:
      sink_col = sinks.float().reshape(1, n_heads, 1, 1)
      sink_col = sink_col.expand(batch_size, n_heads, seq_len, 1)
      probs = F.softmax(torch.cat([scores, sink_col], dim=-1), dim=-1)[
          ..., :-1
      ]
    else:
      probs = F.softmax(scores, dim=-1)
    out = torch.matmul(probs.to(v.dtype), v)  # [B, H, L, D]
    out_BLHV = out.transpose(1, 2)
    if was_3d:
      return out_BLHV.reshape(-1, *out_BLHV.shape[2:])
    return out_BLHV


class FlintAttention(GQAttention):
  """GQA attention with QK-norm, per-layer RoPE/NoPE and attention sinks.

  Identical to ``GQAttention.forward`` except that the inner attention also
  receives this layer's ``sinks``. ``rope=None`` (full-attention layers) skips
  rotary embeddings.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(GQAttention.Config):
    pass

  def __init__(self, config: Config) -> None:
    super().__init__(config)
    self.sinks = nn.Parameter(torch.empty(self.n_heads))
    # Read by sinks-aware attention kernels.
    self.sliding_window_size = getattr(
        self.inner_attention, "sliding_window_size", None
    )

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks: AttentionMasksType | None,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    xq_THK, xk_THK, xv_THV = remat.region(
        self.qkv_linear,
        self.remat_region_name("qkv"),
        recompute=self.remat_should_recompute("qkv"),
    )(x_TD)
    remat.recompute_needs_tensor(xq_THK, xk_THK, xv_THV)

    assert self.q_norm is not None and self.k_norm is not None
    xq_THK = self.q_norm(xq_THK)
    xk_THK = self.k_norm(xk_THK)

    if self.rope is not None:
      xq_THK, xk_THK = self.rope(xq_THK, xk_THK, positions)

    sinks = self.sinks
    if isinstance(sinks, DTensor):
      sinks = sinks.to_local(grad_placements=sinks.placements)

    out_THV = remat.region(
        self.inner_attention,
        self.remat_region_name("inner_attention"),
        recompute=self.remat_should_recompute("inner_attention"),
    )(
        xq_THK,
        xk_THK,
        xv_THV,
        sinks=sinks,
        attention_masks=attention_masks,
        scale=self.scaling,
        enable_gqa=self.enable_gqa,
        positions=positions,
        segment_ids=segment_ids,
    )
    remat.recompute_needs_tensor(out_THV)
    out_THV = out_THV.contiguous()
    out_TD = out_THV.view(out_THV.shape[0], -1)
    out_TD = remat.region(
        self.wo,
        self.remat_region_name("wo"),
        recompute=self.remat_should_recompute("wo"),
    )(out_TD)
    remat.recompute_needs_tensor(out_TD)
    return out_TD


class FlintModel(Decoder):
  """Flint text decoder."""

  @dataclass(kw_only=True, slots=True)
  class Config(Decoder.Config):
    dim: int = 2048
    vocab_size: int = 150144

    def update_from_config(self, *, config, **kwargs) -> None:
      Decoder.Config.update_from_config(self, config=config, **kwargs)
      parallelism = config.parallelism
      if parallelism.tensor_parallel_degree > 1:
        raise NotImplementedError(
            "Flint does not support tensor parallelism: the per-head sinks"
            " are not sharded with the heads (HF Flint replicates attention"
            " for the same reason)."
        )
      if parallelism.context_parallel_degree > 1:
        raise NotImplementedError(
            "Flint does not support context parallelism yet."
        )
      if moe_load_balance_aux_loss_configs(self.layers):
        missing = [
            name
            for name in ("local_global_batch_size", "global_batch_size")
            if kwargs.get(name) is None
        ]
        if missing:
          raise ValueError(
              "MoELoadBalanceAuxLoss is configured, so update_from_config"
              f" needs {' and '.join(missing)} (in samples, as Trainer passes"
              " them)."
          )
        update_moe_aux_loss_configs(
            self.layers,
            pp_enabled=parallelism.pipeline_parallel_degree > 1,
            local_global_batch_size=kwargs["local_global_batch_size"],
            global_batch_size=kwargs["global_batch_size"],
            seq_len=config.training.max_context_length,
        )

      from torchtitan.experiments.tpu.examples.flint.sharding import set_flint_sharding_config  # pylint: disable=g-import-not-at-top

      set_flint_sharding_config(
          self,
          enable_sp=parallelism.enable_sequence_parallel,
          enable_ep=parallelism.expert_parallel_degree > 1,
      )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
      """``6 * active params`` plus window-aware attention FLOPs.

      Same convention as the HF transformers backend, so the two lanes report
      comparable MFU: routed experts count at ``top_k / num_experts``, the
      embedding table does not count, the LM head does, and a sliding layer
      attends to at most its window.
      """
      nparams, active_nparams = get_nparams_and_active_nparams(model)
      attention_flops = 0
      for layer in self.layers:
        attention = layer.attention
        window = getattr(attention.inner_attention, "window_size", None)
        attention_flops += quadratic_attention_flops_per_token(
            num_heads=attention.n_heads,
            qk_head_dim=attention.head_dim,
            v_head_dim=attention.head_dim,
            seq_len=seq_len,
            sliding_window_size=window,
        )
      return nparams, 6 * active_nparams + attention_flops


__all__ = ["FlintAttention", "FlintModel", "FlintSinkAttention"]
