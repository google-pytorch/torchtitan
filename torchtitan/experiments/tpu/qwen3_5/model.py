"""Qwen3.5 attention, transformer block and model modules for TPU."""

from dataclasses import dataclass
from typing import Any

# pyformat: disable
# Import torch before spmd_types to avoid circular import with torch_tpu.
import torch
import spmd_types as spmd
# pyformat: enable

# Before `torchtitan.models.qwen3_5`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu.qwen3_5 import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import annotate_input_spmd_types, set_current_spmd_mesh, spmd_local_context
from torchtitan.experiments.tpu.components.expert_parallel import use_expert_parallel
from torchtitan.models.common.attention import AttentionMasksType, FlexInnerAttention, VarlenInnerAttention
from torchtitan.models.common.decoder_sharding import CP, decoder_input_sharding, DP, SpmdType, TP
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding
from torchtitan.models.qwen3_5.model import Qwen35Attention, Qwen35AttentionMaskDict, Qwen35Model, Qwen35TransformerBlock
from torchtitan.models.qwen3_5.sharding import annotate_deltanet_cu_seqlens


class TpuQwen35Attention(Qwen35Attention):
  """``Qwen35Attention`` that passes ``positions`` to the inner attention.

  Identical to upstream ``Qwen35Attention.forward`` except that the inner
  attention also receives ``positions``, from which the document-masked SDPA
  and the Splash / Tokamax attention patches derive packed-document
  boundaries.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Qwen35Attention.Config):
    pass

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks: AttentionMasksType | None,
      positions: torch.Tensor | None = None,
      *,
      full_segment_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    num_tokens = x_TD.shape[0]

    # wq is 2x wider: produces query + gate
    xq_gate_THC = self.wq(x_TD).view(num_tokens, -1, self.head_dim * 2)
    xq_THK, gate_THV = xq_gate_THC.chunk(2, dim=-1)
    xk_THK = self.wk(x_TD).view(num_tokens, -1, self.head_dim)
    xv_THV = self.wv(x_TD).view(num_tokens, -1, self.head_dim)

    # QK norm (before RoPE)
    xq_THK = self.q_norm(xq_THK)
    xk_THK = self.k_norm(xk_THK)

    # Partial RoPE: only first rotary_dim elements get positional encoding
    assert self.rotary_dim <= self.head_dim
    xq_THR, xq_THP = (
        xq_THK[..., : self.rotary_dim],
        xq_THK[..., self.rotary_dim :],
    )
    xk_THR, xk_THP = (
        xk_THK[..., : self.rotary_dim],
        xk_THK[..., self.rotary_dim :],
    )
    xq_THR, xk_THR = self.rope(xq_THR, xk_THR, positions)
    xq_THK = torch.cat([xq_THR, xq_THP], dim=-1)
    xk_THK = torch.cat([xk_THR, xk_THP], dim=-1)

    out_THV = self.inner_attention(
        xq_THK,
        xk_THK,
        xv_THV,
        attention_masks=attention_masks,
        scale=self.scaling,
        enable_gqa=self.enable_gqa,
        # TPU: pass `positions`; SDPA / Splash mask packed documents with it.
        positions=positions,
        # Context parallelism: the whole sequence's segment_ids, computed
        # once per batch by cp_shard, so the CP inner attention neither
        # all-gathers positions nor recomputes them in every layer.
        **(
            {}
            if full_segment_ids is None
            else {"segment_ids": full_segment_ids}
        ),
    ).contiguous()

    # Output gating
    out_THV = out_THV * torch.sigmoid(gate_THV)
    out_TD = out_THV.view(num_tokens, -1)
    return self.wo(out_TD)


class TpuQwen35TransformerBlock(Qwen35TransformerBlock):
  """``Qwen35TransformerBlock`` that also passes ``positions`` to GatedDeltaNet.

  Identical to upstream ``Qwen35TransformerBlock.forward`` except that the
  GatedDeltaNet (``gated_delta.TpuGatedDeltaNet``) also receives ``positions``,
  from which it derives packed-document boundaries. Under context parallelism
  the attention also gets ``TpuQwen35Model.forward``'s ``cp_full_segment_ids``
  and the GatedDeltaNet its ``cp_full_positions``.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Qwen35TransformerBlock.Config):
    pass

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks: Qwen35AttentionMaskDict | None,
      positions: torch.Tensor | None = None,
      *,
      full_positions: torch.Tensor | None = None,
      full_segment_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    layer_mask = (
        attention_masks[self.attn_mask_key]
        if attention_masks is not None
        else None
    )
    h_TD = self.attention_norm(x_TD)
    # TPU: both layer types get `positions` (upstream: full attention only),
    # and their whole-sequence document boundaries under CP only.
    if self.full_attn:
      cp_kwargs = {"full_segment_ids": full_segment_ids}
    else:
      cp_kwargs = {"full_positions": full_positions}
    h_TD = self.attn(
        h_TD,
        layer_mask,
        positions,
        **{k: v for k, v in cp_kwargs.items() if v is not None},
    )
    x_TD = x_TD + h_TD

    h_TD = self.ffn_norm(x_TD)
    if self.moe_enabled:
      x_TD = x_TD + self.moe(h_TD)
    else:
      x_TD = x_TD + self.feed_forward(h_TD)
    return x_TD


class TpuQwen35Model(Qwen35Model):
  """``Qwen35Model`` that wires TPU EP execution and CP input sharding.

  Its config applies TPU expert parallelism (``use_expert_parallel``). Under
  context parallelism, ``preprocess_inputs`` delegates input sharding to the
  CP inner attention's ``cp_shard``, as core ``Decoder.preprocess_inputs``
  does, instead of upstream's own ``prepare_context_parallel_input``. The TPU
  KV all-gather ``cp_shard`` lays every sequence out head-tail (upstream would
  head-tail the folded ``[B * S]`` token stream), caches the CP-rank tensor and
  precomputes the full-sequence ``segment_ids``; the CP GatedDeltaNet
  (``cp_gated_delta``) relies on the same layout.
  ``preprocess_inputs`` also passes the unsharded positions as
  ``cp_full_positions``, so the CP attention and GatedDeltaNet layers get the
  whole sequence's document boundaries without per-layer all-gathers.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Qwen35Model.Config):

    def update_from_config(self, *, config, **kwargs) -> None:
      Qwen35Model.Config.update_from_config(self, config=config, **kwargs)
      tpu_cfg = getattr(config, "tpu_config", None)
      use_expert_parallel(
          self,
          ep_degree=config.parallelism.expert_parallel_degree,
          ep_transport=getattr(tpu_cfg, "ep_transport", None),
          scale_before_down_proj=getattr(
              tpu_cfg, "ep_scale_before_down_proj", None
          ),
          ep_capacity_factor=getattr(tpu_cfg, "ep_capacity_factor", None),
          num_token_chunks=getattr(tpu_cfg, "ep_token_chunks", None),
          max_inflight_chunks=getattr(tpu_cfg, "ep_max_inflight_chunks", None),
          accumulate_chunk_weight_grads_fp32=getattr(
              tpu_cfg, "ep_chunk_dw_fp32", None
          ),
          paired_grouped_mm=getattr(tpu_cfg, "ep_paired_grouped_mm", None),
          roe_window_routing=getattr(tpu_cfg, "ep_roe_window_routing", None),
          roe_save_routing_metadata=getattr(
              tpu_cfg, "ep_roe_save_routing_metadata", None
          ),
          roe_reduce_block_size=getattr(
              tpu_cfg, "ep_roe_reduce_block_size", None
          ),
          roe_fused_w13_gather=getattr(
              tpu_cfg, "ep_roe_fused_w13_gather", None
          ),
      )

  def forward(  # pyrefly: ignore [bad-override]
      self,
      tokens: torch.Tensor,
      *,
      attention_masks: Any = None,
      positions: torch.Tensor | None = None,
      cp_full_positions: torch.Tensor | None = None,
      cp_full_segment_ids: torch.Tensor | None = None,
      **kwargs,
  ):
    """``Qwen35Model.forward`` plus the CP document-boundary inputs.

    Args:
      tokens: Token ids (or embeddings when ``tok_embeddings`` is ``None``).
      attention_masks: As upstream.
      positions: As upstream; under CP, this rank's shard.
      cp_full_positions: Under context parallelism, the positions of every token
        of the unsharded sequence, in sequence order (the CP-sharded
        ``positions`` before ``cp_shard``). The CP attention and GatedDeltaNet
        layers derive the whole sequence's document boundaries from them instead
        of all-gathering ``positions`` in every layer. Used by GatedDeltaNet.
      cp_full_segment_ids: Under context parallelism, the whole sequence's
        segment_ids in the CP attention's layout (computed once by
        ``cp_shard``). Used by the full-attention layers.
      **kwargs: Upstream's multimodal arguments.

    Returns:
      As upstream.
    """
    if cp_full_positions is None and cp_full_segment_ids is None:
      return super().forward(
          tokens,
          attention_masks=attention_masks,
          positions=positions,
          **kwargs,
      )
    unsupported = sorted(
        k for k, v in kwargs.items() if v is not None and k != "special_tokens"
    )
    if unsupported:
      raise NotImplementedError(
          "cp_full_positions / cp_full_segment_ids support text-only batches;"
          f" got {unsupported}."
      )
    # Text-only `Qwen35Model.forward` that also passes the CP inputs.
    with spmd_local_context("dp"):
      x = (
          self.tok_embeddings(tokens)
          if self.tok_embeddings is not None
          else tokens
      )
    if spmd.is_type_checking():
      spmd.assert_type(
          x,
          {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
          spmd.PartitionSpec(("dp", "cp"), None),
      )
    for layer in self.layers.values():
      x = layer(
          x,
          attention_masks,
          positions,
          full_positions=cp_full_positions,
          full_segment_ids=cp_full_segment_ids,
      )
    x = self.norm(x) if self.norm is not None else x
    if self._skip_lm_head:
      return x
    return self.lm_head(x) if self.lm_head is not None else x

  def preprocess_inputs(
      self,
      input_dict: dict[str, torch.Tensor],
      *,
      parallel_dims: ParallelDims,
      parallelism: ParallelismConfig,
      max_num_documents: int | None = None,
      max_context_length: int | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not parallel_dims.cp_enabled:
      return super().preprocess_inputs(
          input_dict,
          parallel_dims=parallel_dims,
          parallelism=parallelism,
          max_num_documents=max_num_documents,
          max_context_length=max_context_length,
      )
    batch: dict[str, Any] = dict(input_dict)
    padding_mask = batch.pop("padding_mask", None)
    if batch.pop("mrope_positions", None) is not None:
      # TPU: CP shards text tokens only; vision tokens would need scattering
      # into the sequence before sharding.
      raise NotImplementedError(
          "Qwen3.5 context parallelism on TPU is text-only (no"
          " mrope_positions)."
      )
    positions = batch.get("positions")
    if positions is None:
      # Not an assert (stripped under `python -O`): RoPE on each rank's token
      # shard, the KV all-gather cp_shard and the GatedDeltaNet document
      # resets all need the global positions.
      raise ValueError(
          "Qwen3.5 context parallelism on TPU needs RoPE positions in the"
          " batch."
      )
    inner = self.config.first_full_attention_backend
    if isinstance(
        inner, (FlexInnerAttention.Config, VarlenInnerAttention.Config)
    ):
      batch["attention_masks"] = self.get_attention_masks(
          positions=positions,
          padding_mask=padding_mask,
          max_num_documents=max_num_documents,
          max_context_length=max_context_length,
      )
    input_sharding = {**decoder_input_sharding(), **multimodal_input_sharding()}
    # TPU: the CP inner attention shards the inputs (see the class docstring).
    batch = self._cp_shard_inputs(
        batch, input_sharding, parallel_dims, parallelism
    )
    # Unsharded positions and the segment_ids cp_shard computed from them:
    # identical on every CP rank. Explicit inputs, so compiled blocks see them.
    batch["cp_full_positions"] = positions
    full_segment_ids = getattr(batch["positions"], "_cp_full_segment_ids", None)
    if full_segment_ids is not None:
      batch["cp_full_segment_ids"] = full_segment_ids
    for key in ("cp_full_positions", "cp_full_segment_ids"):
      if key in batch:
        # Rows (dim 0) are this data-parallel rank's; KV all-gather
        # segment_ids are (B, S).
        trailing = (None,) * (batch[key].ndim - 1)
        input_sharding[key] = SpmdType(
            {DP: spmd.V, CP: spmd.R, TP: spmd.R},
            partition_spec=spmd.PartitionSpec(DP, *trailing),
        )
    batch = annotate_input_spmd_types(parallel_dims, batch, input_sharding)
    attention_masks = batch.get("attention_masks")
    if attention_masks is not None:
      with set_current_spmd_mesh(parallel_dims.spmd_dense_mesh()):
        annotate_deltanet_cu_seqlens(attention_masks)
    inputs = batch.pop("input")
    labels = batch.pop("labels")
    return inputs, labels, batch
