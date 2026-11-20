"""Qwen3.5 attention, transformer block and model modules for TPU."""

from dataclasses import dataclass
from typing import Any

import torch

# Before `torchtitan.models.qwen3_5`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu.qwen3_5 import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import annotate_input_spmd_types, set_current_spmd_mesh
from torchtitan.experiments.tpu.components.expert_parallel import use_expert_parallel
from torchtitan.models.common.attention import AttentionMasksType, FlexInnerAttention, VarlenInnerAttention
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
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
    ).contiguous()

    # Output gating
    out_THV = out_THV * torch.sigmoid(gate_THV)
    out_TD = out_THV.view(num_tokens, -1)
    return self.wo(out_TD)


class TpuQwen35TransformerBlock(Qwen35TransformerBlock):
  """``Qwen35TransformerBlock`` that also passes ``positions`` to GatedDeltaNet.

  Identical to upstream ``Qwen35TransformerBlock.forward`` except that the
  GatedDeltaNet (``gated_delta.TpuGatedDeltaNet``) also receives ``positions``,
  from which it derives packed-document boundaries.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Qwen35TransformerBlock.Config):
    pass

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks: Qwen35AttentionMaskDict | None,
      positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    layer_mask = (
        attention_masks[self.attn_mask_key]
        if attention_masks is not None
        else None
    )
    h_TD = self.attention_norm(x_TD)
    # TPU: both layer types get `positions` (upstream: full attention only).
    h_TD = self.attn(h_TD, layer_mask, positions)
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
      )

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
    batch = annotate_input_spmd_types(parallel_dims, batch, input_sharding)
    attention_masks = batch.get("attention_masks")
    if attention_masks is not None:
      with set_current_spmd_mesh(parallel_dims.spmd_dense_mesh()):
        annotate_deltanet_cu_seqlens(attention_masks)
    inputs = batch.pop("input")
    labels = batch.pop("labels")
    return inputs, labels, batch
