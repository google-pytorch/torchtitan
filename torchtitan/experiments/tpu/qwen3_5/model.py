"""Qwen3.5 attention, transformer block, and model modules for TPU."""

from dataclasses import dataclass

import torch
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts

# Before `torchtitan.models.qwen3_5`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu.qwen3_5 import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.experiments.tpu.qwen3_5 import token_dispatcher
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.moe import MoE
from torchtitan.models.qwen3_5.model import Qwen35Attention, Qwen35AttentionMaskDict, Qwen35Model, Qwen35TransformerBlock


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
  """``Qwen35Model`` whose config and parallelization wire TPU EP execution.

  Under expert parallelism, ``Config.update_from_config`` switches the MoE
  layers to ``TpuQwen35RoutedExperts`` and ``TpuQwen35TokenDispatcher``, keeps
  paired Pallas grouped matmuls enabled only when ``allow_ep_paired_grouped_mm``
  is set, and scales ``num_token_chunks`` with ``expert_parallel_degree`` when
  ``auto_scale_token_chunks`` is enabled. ``parallelize`` upgrades unconfigured
  TPU expert layers to unpaired Pallas grouped matmuls so dynamic or empty
  expert segments under EP never hit ``torch._grouped_mm``.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Qwen35Model.Config):

    def update_from_config(self, *, config, **kwargs) -> None:
      # Zero-argument super() does not work in slots dataclasses on Python
      # 3.12 (the class is re-created); call the parent explicitly.
      Qwen35Model.Config.update_from_config(self, config=config, **kwargs)
      ep_deg = config.parallelism.expert_parallel_degree
      if ep_deg > 1:
        # Only model-config changes belong here. `compile_grad_norm` (rejected
        # under EP > 1 by `TPUTrainerConfig.__post_init__` at parse time) and
        # `tpu_config.eager_mode` (applied by `gmain` before the trainer is
        # built) must be set in the recipe or on the command line.
        graph_split = getattr(
            getattr(config, "compile", None), "graph_split", None
        )
        if (
            graph_split is not None
            and graph_split.enabled
            and graph_split.backward_partition_mode != "by_layer"
        ):
          raise ValueError(
              "Expert parallelism needs"
              " --compile.graph_split.backward_partition_mode=by_layer: the EP"
              " activation collectives cannot be moved by"
              f" {graph_split.backward_partition_mode!r}. Use a *_ep_* recipe"
              " or pass the flag."
          )
        # `TpuGroupedExperts.parallelize` requires `EP == 1` exact-row dispatch
        # for the paired (w1 + w3) Pallas kernel unless the experts opt in to
        # the EP-aware paired kernel; otherwise fall back to unpaired Pallas.
        for _, experts, _, _ in self.traverse(TpuGroupedExperts.Config):
          if not experts.allow_ep_paired_grouped_mm:
            experts.use_pallas_paired_grouped_mm = False
        routed_cfgs, _ = token_dispatcher.use_ep_modules(self)
        for routed in routed_cfgs:
          if routed.auto_scale_token_chunks:
            routed.num_token_chunks = token_dispatcher.default_num_token_chunks(
                ep_deg
            )

  def _init_self_buffers(
      self, *, buffer_device: torch.device | None = None
  ) -> None:
    super()._init_self_buffers(buffer_device=buffer_device)
    ep_mesh = getattr(self, "_ep_mesh", None)
    if ep_mesh is not None and "_ep_local_rank" in self._buffers:
      target_dev = (
          buffer_device
          if buffer_device is not None
          else self._ep_local_rank.device
      )
      self._ep_local_rank = torch.tensor(
          [int(ep_mesh.get_local_rank())], dtype=torch.long
      ).to(device=target_dev)

  def parallelize(self, parallel_dims) -> None:
    ep_mesh = parallel_dims.get_optional_mesh("ep")
    if ep_mesh is not None and getattr(ep_mesh, "device_type", None) == "tpu":
      # Under EP > 1 on TPU, recipes without `_use_tpu_moe_kernels` leave
      # `use_pallas_grouped_mm=False`, falling back to `torch._grouped_mm`
      # (`chlo.ragged_dot`), whose backward reads uninitialized HBM on dynamic
      # or empty expert segments (torch_tpu#2668). Switch tile-aligned expert
      # layers to unpaired Pallas grouped matmuls while preserving any explicit
      # tiling already configured by `_use_tpu_moe_kernels`.
      for module in self.modules():
        if (
            isinstance(module, TpuGroupedExperts)
            and not module.use_pallas_grouped_mm
            and hasattr(module, "w1_EFD")
            and module.w1_EFD.shape[-2] % 128 == 0
            and module.w1_EFD.shape[-1] % 128 == 0
        ):
          module.use_pallas_grouped_mm = True
          module.pallas_grouped_mm_tile_m = 256
          module.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)
    super().parallelize(parallel_dims)
    if ep_mesh is not None and callable(
        getattr(ep_mesh, "get_local_rank", None)
    ):
      self._ep_mesh = ep_mesh
      param = next(self.parameters(), None)
      param_dev = param.device if param is not None else torch.device("cpu")
      rank_buf = (
          torch.empty((1,), dtype=torch.long, device="meta")
          if param_dev.type == "meta"
          else torch.tensor(
              [int(ep_mesh.get_local_rank())], dtype=torch.long
          ).to(device=param_dev)
      )
      self.register_buffer("_ep_local_rank", rank_buf, persistent=False)
      for module in self.modules():
        if isinstance(module, MoE):
          inner_experts = module.routed_experts.inner_experts
          dispatcher = module.routed_experts.token_dispatcher
          if isinstance(dispatcher, token_dispatcher.TpuQwen35TokenDispatcher):
            dispatcher.bind_ep_rank_holder(self)
            if not getattr(inner_experts, "use_pallas_grouped_mm", False):
              dispatcher.roe_zero_pad_tail = True
        else:
          dispatcher = getattr(module, "token_dispatcher", None)
          if isinstance(dispatcher, token_dispatcher.TpuQwen35TokenDispatcher):
            dispatcher.bind_ep_rank_holder(self)
