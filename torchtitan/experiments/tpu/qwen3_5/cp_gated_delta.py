"""Context parallelism for the Qwen3.5 GatedDeltaNet on TPU.

Under context parallelism (``parallelism.context_parallel_degree > 1``) each
rank holds ``S / cp`` tokens of every sequence. Qwen3.5's full-attention layers
run the CP inner attentions of ``experiments/tpu/distributed/cp_attention.py``
(Ulysses or load-balanced KV all-gather). Its GatedDeltaNet layers cannot run on
a token shard: the causal convolution and the delta-rule recurrence carry state
through the whole sequence, in order. ``CPTpuInnerGatedDeltaNet`` runs them
head-parallel instead, as Ulysses does for attention (and Megatron-LM does for
Mamba):

1. All-to-all the projected ``q`` / ``k`` / ``v`` channels and the ``a`` / ``b``
   gate inputs from token-sharded (``T / cp`` tokens, every head) to
   head-sharded (``T`` tokens, ``1 / cp`` of the heads).
2. Put the tokens back in sequence order: under KV all-gather CP (the
   ``"headtail"`` load balancer) rank ``r`` holds chunks ``r`` and
   ``2 * cp - 1 - r`` of every sequence; under Ulysses CP the folded token
   stream is split into ``cp`` contiguous pieces, which the all-to-all already
   concatenates in order.
3. Run the causal convolution and the recurrence over whole sequences on the
   rank's heads, with the rank's slice of the depthwise convolution weights,
   ``A_log`` and ``dt_bias``. The slice is picked with ``index_select`` on the
   cached CP-rank tensor, so every rank runs the same compiled graph.
4. Restore the CP token order and all-to-all the output back to token-sharded.

The result is the single-device GatedDeltaNet (no approximation of the
recurrence); each rank computes ``1 / cp`` of it, and the layer works with both
full-attention CP strategies. ``n_key_heads`` and ``n_value_heads`` must be
divisible by ``cp`` (Qwen3.5-35B-A3B: 16 and 32). Per token, the all-to-alls
move the ``q`` / ``k`` / ``v`` / ``a`` / ``b`` inputs and the output once each
way (twice with activation checkpointing, which reruns them in backward).

Recipes enable it with ``GatedDeltaNetContextParallelTransform`` next to core
``ContextParallelTransform`` (see ``config_registry._qwen35_cp``);
``validate_gated_delta_context_parallel`` rejects CP configs whose
GatedDeltaNet layers are not converted, and
``revert_gated_delta_context_parallel`` (run by
``revert_tpu_context_parallel``) converts them back for ``cp == 1`` reruns.
"""

from dataclasses import dataclass, fields
from typing import Any, Literal

# pyformat: disable
# Import torch before spmd_types to avoid circular import with torch_tpu.
import torch
import spmd_types as spmd
import torch.nn.functional as F
# pyformat: enable
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.config.transform import convert_config_type, ModelConfigTransform
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.experiments.tpu.components.unflatten_shim import get_unflatten_seq_len
from torchtitan.experiments.tpu.distributed.cp_attention import (
    cp_all_gather,
    cp_all_gather_with_grad,
    cp_all_to_all,
    init_cp_rank_buffer,
    module_cp_rank,
    register_cp_rank_buffer,
    cp_reduce_scatter,
    register_tpu_cp_reverter,
    register_tpu_cp_validator,
    reorder_sequence,
)

# Before `torchtitan.models.qwen3_5`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu.qwen3_5 import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.experiments.tpu.qwen3_5.gated_delta import TpuGatedDeltaNet, TpuInnerGatedDeltaNet
from torchtitan.models.qwen3_5.gdn import GatedDeltaNet
from torchtitan.protocols.module import Module

__all__ = [
    "CPTpuGatedDeltaNet",
    "CPTpuInnerGatedDeltaNet",
    "GatedDeltaNetContextParallelTransform",
    "revert_gated_delta_context_parallel",
    "validate_gated_delta_context_parallel",
]

_TOKEN_DIM = 0
_CHANNEL_DIM = 1

LoadBalancer = Literal["headtail"] | None
GatedDeltaNetCPStrategy = Literal["all_to_all", "allgather_reducescatter"]
_CP_STRATEGIES = ("all_to_all", "allgather_reducescatter")
ReduceDtype = Literal["float32", "bfloat16"]


def _rows(num_tokens: int) -> tuple[int, int]:
  """``(B, S)`` of a folded ``[B * S]`` token stream in head-tail layout.

  ``S`` comes from ``set_unflatten_seq_len``, as in KV all-gather
  ``cp_shard``, which reorders per sequence of length ``S``; without it the
  stream is one sequence.
  """
  seq_len = get_unflatten_seq_len()
  if seq_len is None:
    seq_len = num_tokens
  elif num_tokens % seq_len:
    # TpuInnerGatedDeltaNet unflattens to [B, S] rows and rejects this too.
    raise ValueError(
        f"GatedDeltaNet CP: {num_tokens} tokens are not a multiple of the"
        f" set_unflatten_seq_len sequence length {seq_len}."
    )
  return num_tokens // seq_len, seq_len


def _headtail_to_sequence_order(x: torch.Tensor, cp: int) -> torch.Tensor:
  """Puts ``[T, ...]`` gathered in head-tail layout back in sequence order.

  Rank ``r`` held ``[B, S / cp]`` (chunks ``r`` and ``2 * cp - 1 - r`` of each
  sequence), so the rank-major concatenation is ``[cp, B, S / cp]``.
  """
  bsz, seq_len = _rows(x.shape[0])
  rest = list(x.shape[1:])
  x = x.reshape([cp, bsz, seq_len // cp] + rest).movedim(0, 1)
  x = reorder_sequence(
      x.reshape([bsz, seq_len] + rest), cp_size=cp, to_contiguous=True
  )
  return x.reshape([bsz * seq_len] + rest)


def _sequence_to_headtail_order(x: torch.Tensor, cp: int) -> torch.Tensor:
  """Inverse of ``_headtail_to_sequence_order``."""
  bsz, seq_len = _rows(x.shape[0])
  rest = list(x.shape[1:])
  x = reorder_sequence(x.reshape([bsz, seq_len] + rest), cp_size=cp)
  x = x.reshape([bsz, cp, seq_len // cp] + rest).movedim(1, 0)
  return x.reshape([bsz * seq_len] + rest)


def _local_cols(x: torch.Tensor, rank: torch.Tensor, cp: int) -> torch.Tensor:
  """Rank ``rank``'s contiguous ``1 / cp`` slice of dim 1 (``out_proj``)."""
  local = x.shape[1] // cp
  return (
      x.reshape(x.shape[0], cp, local)
      .index_select(1, rank)
      .reshape(x.shape[0], local)
  )


def _check_heads_divisible(n_key_heads: int, n_value_heads: int, cp: int):
  if n_key_heads % cp or n_value_heads % cp:
    raise ValueError(
        f"GatedDeltaNet CP shards heads: n_key_heads ({n_key_heads}) and"
        f" n_value_heads ({n_value_heads}) must be divisible by cp ({cp})."
    )


def _local_heads(x: torch.Tensor, rank: torch.Tensor, cp: int) -> torch.Tensor:
  """Rank ``rank``'s contiguous ``1 / cp`` slice of dim 0 (per-head params)."""
  local = x.shape[0] // cp
  rest = list(x.shape[1:])
  return (
      x.reshape([cp, local] + rest)
      .index_select(0, rank)
      .reshape([local] + rest)
  )


# The input projections, in ``GatedDeltaNet.forward`` order.
_INPUT_PROJECTIONS = (
    "in_proj_q",
    "in_proj_k",
    "in_proj_v",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
)


def _is_plain_linear(module: torch.nn.Module) -> bool:
  """Whether ``module`` is an ``nn.Linear`` that computes ``F.linear``.

  Head-sliced projections read ``weight``/``bias`` directly, which would skip
  a subclass's own ``forward`` (e.g. a Float8 linear).
  """
  return (
      isinstance(module, torch.nn.Linear)
      and type(module).forward is torch.nn.Linear.forward
  )


def _gather_positions(
    positions: torch.Tensor | None,
    full_positions: torch.Tensor | None,
    group: Any,
) -> torch.Tensor | None:
  """All-gathers ``positions`` unless ``full_positions`` already holds them.

  Document resets need every token's position (one per token). Returns
  ``None`` when there is nothing to gather.
  """
  if positions is None or full_positions is not None:
    return None
  return cp_all_gather(positions.reshape(-1), group)


def _sequence_positions(
    gathered_positions: torch.Tensor | None,
    full_positions: torch.Tensor | None,
    cp: int,
    headtail: bool,
) -> torch.Tensor | None:
  """Every token's position in sequence order, or ``None`` (no resets)."""
  if full_positions is not None:
    return full_positions.reshape(-1)
  if headtail and gathered_positions is not None:
    return _headtail_to_sequence_order(gathered_positions, cp)
  return gathered_positions


class CPTpuInnerGatedDeltaNet(TpuInnerGatedDeltaNet):
  """Head-parallel ``TpuInnerGatedDeltaNet`` for context parallelism.

  See the module docstring. Requires an active CP mesh axis: a ``cp == 1``
  reference run of a CP recipe reverts it to ``TpuInnerGatedDeltaNet`` with
  ``revert_tpu_context_parallel``.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TpuInnerGatedDeltaNet.Config):
    load_balancer: LoadBalancer = None
    """How ``cp_shard`` lays the tokens out over the CP group: ``"headtail"``
    (KV all-gather CP) or ``None`` (contiguous, Ulysses CP). Must equal
    ``parallelism.context_parallel_load_balancer``."""

  def __init__(self, config: Config):
    super().__init__(config)
    if config.load_balancer not in ("headtail", None):
      raise ValueError(
          "CPTpuInnerGatedDeltaNet supports load_balancer 'headtail' or None,"
          f" got {config.load_balancer!r}."
      )
    self.load_balancer = config.load_balancer

  def parallelize(self, parallel_dims: Any) -> None:
    super().parallelize(parallel_dims)
    register_cp_rank_buffer(self, parallel_dims)

  def _init_self_buffers(
      self, *, buffer_device: torch.device | None = None
  ) -> None:
    if hasattr(super(), "_init_self_buffers"):
      super()._init_self_buffers(buffer_device=buffer_device)  # pyrefly: ignore[missing-attribute]
    init_cp_rank_buffer(self, buffer_device)

  def forward(
      self,
      query_TC: torch.Tensor,
      key_TC: torch.Tensor,
      value_TC: torch.Tensor,
      a_TH: torch.Tensor,
      b_TH: torch.Tensor,
      conv_q_weight_C1W: torch.Tensor,
      conv_k_weight_C1W: torch.Tensor,
      conv_v_weight_C1W: torch.Tensor,
      A_log_H: torch.Tensor,
      dt_bias_H: torch.Tensor,
      cu_seqlens: torch.Tensor | None,
      *,
      key_head_dim: int,
      value_head_dim: int,
      positions: torch.Tensor | None = None,
      full_positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Head-parallel GatedDeltaNet over the CP group (see the module docstring).

    Args:
      query_TC: Token-sharded queries, ``(T / cp, C)``.
      key_TC: Token-sharded keys.
      value_TC: Token-sharded values.
      a_TH: Token-sharded decay gate input.
      b_TH: Token-sharded beta gate input.
      conv_q_weight_C1W: Full (all heads) depthwise conv weight for ``q``.
      conv_k_weight_C1W: Full depthwise conv weight for ``k``.
      conv_v_weight_C1W: Full depthwise conv weight for ``v``.
      A_log_H: Full ``A_log``.
      dt_bias_H: Full ``dt_bias``.
      cu_seqlens: Unused (rows come from ``set_unflatten_seq_len``); may be
        ``None``.
      key_head_dim: Key head dim.
      value_head_dim: Value head dim.
      positions: This rank's token-sharded positions. With document resets and
        no ``full_positions``, they are all-gathered once per call.
      full_positions: The unsharded positions of every token in sequence order
        (``TpuQwen35Model.forward``'s ``cp_full_positions``). When given, they
        replace the per-layer all-gather of ``positions``.

    Returns:
      The token-sharded ``(T / cp, H, V)`` output.
    """
    group = spmd_mesh_group(MeshAxisName.CP)
    if group is None:
      # Running the single-device module here would silently compute the
      # causal convolution and recurrence on each rank's token shard. As the
      # CP inner attentions do, require the CP axis; a cp == 1 rerun of a CP
      # recipe reverts the layer first (revert_tpu_context_parallel).
      raise RuntimeError(
          "CPTpuInnerGatedDeltaNet requires an active CP mesh axis of size > 1"
          " (spmd_mesh_group('cp') is None). For a cp == 1 run of a CP recipe,"
          " call revert_tpu_context_parallel on the model config."
      )
    cp = group.size()
    _check_heads_divisible(
        query_TC.shape[-1] // key_head_dim,
        value_TC.shape[-1] // value_head_dim,
        cp,
    )
    headtail = self.load_balancer == "headtail"
    rank = module_cp_rank(self, query_TC.device, group)

    # (T / cp, C) -> (T, C / cp): every token of the rank's heads. One
    # all-to-all per input: packing them into one measured slower on TPU7x
    # (XLA overlaps the later transfers with the earlier inputs' conv).
    query_TC, key_TC, value_TC, a_TH, b_TH = (
        cp_all_to_all(x, group, src_dim=_TOKEN_DIM, dst_dim=_CHANNEL_DIM)
        for x in (query_TC, key_TC, value_TC, a_TH, b_TH)
    )
    gathered_positions = _gather_positions(positions, full_positions, group)
    type_check = spmd.is_type_checking()
    if type_check:
      out_type = (
          spmd.get_local_type(value_TC),
          spmd.get_partition_spec(value_TC),
      )
    # The kernels below carry no spmd_types annotations.
    with spmd.no_typecheck():
      if headtail:
        query_TC, key_TC, value_TC, a_TH, b_TH = (
            _headtail_to_sequence_order(x, cp)
            for x in (query_TC, key_TC, value_TC, a_TH, b_TH)
        )
      sequence_positions = _sequence_positions(
          gathered_positions, full_positions, cp, headtail
      )
      out_THV = self.local_forward(
          query_TC,
          key_TC,
          value_TC,
          a_TH,
          b_TH,
          _local_heads(conv_q_weight_C1W, rank, cp),
          _local_heads(conv_k_weight_C1W, rank, cp),
          _local_heads(conv_v_weight_C1W, rank, cp),
          _local_heads(A_log_H, rank, cp),
          _local_heads(dt_bias_H, rank, cp),
          cu_seqlens,
          key_head_dim=key_head_dim,
          value_head_dim=value_head_dim,
          positions=sequence_positions,
      )
      out_TC = out_THV.reshape(out_THV.shape[0], -1)
      if headtail:
        out_TC = _sequence_to_headtail_order(out_TC, cp)
    if type_check and not spmd.has_local_type(out_TC):
      spmd.assert_type(out_TC, *out_type)
    # (T, C / cp) -> (T / cp, C), then (T / cp, H, V).
    out_TC = cp_all_to_all(
        out_TC, group, src_dim=_CHANNEL_DIM, dst_dim=_TOKEN_DIM
    )
    return out_TC.unflatten(-1, (-1, value_head_dim))

  def local_forward(self, *args, **kwargs) -> torch.Tensor:
    """The single-device ``TpuInnerGatedDeltaNet`` on this rank's heads.

    Takes ``TpuInnerGatedDeltaNet.forward``'s arguments for whole sequences
    (sequence order) and this rank's slices of the per-head parameters.
    ``CPTpuGatedDeltaNet``'s all-gather / reduce-scatter strategy calls it
    directly.
    """
    return super().forward(*args, **kwargs)


class CPTpuGatedDeltaNet(TpuGatedDeltaNet):
  """``TpuGatedDeltaNet`` with a selectable context-parallel strategy.

  ``cp_strategy="all_to_all"`` is ``TpuGatedDeltaNet`` with
  ``CPTpuInnerGatedDeltaNet``: five all-to-alls move the projected ``q`` /
  ``k`` / ``v`` / ``a`` / ``b`` to head-sharded and one moves the output back.

  ``cp_strategy="allgather_reducescatter"`` moves the layer input instead:

  1. All-gather ``x`` (``T / cp`` tokens -> ``T`` tokens, ``D`` channels;
     reduce-scatter of its gradient in backward) and put it in sequence order.
  2. Project it with this rank's rows of the six input projections (its
     ``1 / cp`` of the heads), run the convolution, the recurrence and the
     gated norm over whole sequences on those heads.
  3. Multiply by this rank's columns of ``out_proj``, giving a partial
     ``(T, D)`` output, and reduce-scatter it back to ``T / cp`` tokens (summed
     in ``cp_reduce_dtype``; all-gather of the gradient in backward).

  Per rank and direction, the all-to-alls move ``(cp - 1) / cp**2 * T * C``
  values (``C`` = the projected channels in, ``C_v`` out), while the all-gather
  and the reduce-scatter move ``(cp - 1) / cp * T * D`` each. For
  Qwen3.5-35B-A3B (``D = 2048``, ``C = 8256`` in, ``4096`` out) and the default
  float32 ``cp_reduce_dtype`` that is about the same bytes at ``cp = 2`` and
  twice at ``cp = 4`` (0.66x / 1.33x with bfloat16), in two collectives instead
  of six; backward also keeps the all-gathered ``(T, D)`` input in HBM.
  Benchmark it against ``all_to_all`` before enabling it.

  The result equals the single-device layer up to rounding: each rank's
  partial ``out_proj`` product (and input-gradient partial) is rounded to the
  activation dtype before the ``cp_reduce_dtype`` sum. Works with either
  full-attention CP strategy (load balancer ``"headtail"`` or ``None``) and
  every ``cp`` dividing ``n_key_heads`` and ``n_value_heads``. The input
  projections and ``out_proj`` must be plain ``nn.Linear`` (their weights are
  sliced).
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TpuGatedDeltaNet.Config):
    cp_strategy: GatedDeltaNetCPStrategy = "all_to_all"
    """How the GatedDeltaNet layers exchange data under CP (class docstring)."""

    cp_reduce_dtype: ReduceDtype = "float32"
    """dtype of the ``allgather_reducescatter`` reductions (the forward
    ``out_proj`` reduce-scatter and the backward input-gradient
    reduce-scatter)."""

  def __init__(self, config: Config):
    super().__init__(config)
    if config.cp_strategy not in _CP_STRATEGIES:
      raise ValueError(
          f"cp_strategy must be one of {_CP_STRATEGIES}, got"
          f" {config.cp_strategy!r}."
      )
    self.cp_strategy = config.cp_strategy
    self.cp_reduce_dtype = TORCH_DTYPE_MAP[config.cp_reduce_dtype]

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks=None,
      positions: torch.Tensor | None = None,
      *,
      full_positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """``TpuGatedDeltaNet.forward`` with the configured CP strategy."""
    group = spmd_mesh_group(MeshAxisName.CP)
    if self.cp_strategy == "all_to_all" or group is None:
      return super().forward(
          x_TD, attention_masks, positions, full_positions=full_positions
      )
    if not self.reset_at_documents:
      positions = full_positions = None
    return self._forward_allgather_reducescatter(
        x_TD, group, positions, full_positions
    )

  def _forward_allgather_reducescatter(
      self,
      x_TD: torch.Tensor,
      group: Any,
      positions: torch.Tensor | None,
      full_positions: torch.Tensor | None,
  ) -> torch.Tensor:
    """The ``allgather_reducescatter`` strategy (see the class docstring)."""
    inner = self.inner_gated_delta_net
    if not isinstance(inner, CPTpuInnerGatedDeltaNet):
      raise TypeError(
          "cp_strategy='allgather_reducescatter' needs a"
          f" CPTpuInnerGatedDeltaNet, got {type(inner).__qualname__}."
      )
    for name in (*_INPUT_PROJECTIONS, "out_proj"):
      proj = getattr(self, name)
      if not _is_plain_linear(proj):
        raise ValueError(
            f"cp_strategy='allgather_reducescatter' slices {name}'s weight, so"
            f" it needs a plain nn.Linear; got {type(proj).__qualname__}."
        )
    cp = group.size()
    _check_heads_divisible(
        self.in_proj_q.out_features // self.key_head_dim,
        self.in_proj_v.out_features // self.value_head_dim,
        cp,
    )
    headtail = inner.load_balancer == "headtail"
    rank = module_cp_rank(inner, x_TD.device, group)
    gathered_positions = _gather_positions(positions, full_positions, group)
    type_check = spmd.is_type_checking()
    if type_check:
      x_type = (spmd.get_local_type(x_TD), spmd.get_partition_spec(x_TD))
    # (T / cp, D) -> (T, D).
    x_full_TD = cp_all_gather_with_grad(
        x_TD, group, dim=_TOKEN_DIM, reduce_dtype=self.cp_reduce_dtype
    )
    # The kernels below carry no spmd_types annotations.
    with spmd.no_typecheck():
      if headtail:
        x_full_TD = _headtail_to_sequence_order(x_full_TD, cp)
      sequence_positions = _sequence_positions(
          gathered_positions, full_positions, cp, headtail
      )
      num_tokens = x_full_TD.shape[0]
      # This rank's heads of the six input projections.
      query_TC, key_TC, value_TC, gate_TC, a_TH, b_TH = (
          F.linear(
              x_full_TD,
              _local_heads(proj.weight, rank, cp),
              None if proj.bias is None else _local_heads(proj.bias, rank, cp),
          )
          for proj in (getattr(self, name) for name in _INPUT_PROJECTIONS)
      )
      output_THV = inner.local_forward(
          query_TC,
          key_TC,
          value_TC,
          a_TH,
          b_TH,
          _local_heads(self.conv_q.weight, rank, cp),
          _local_heads(self.conv_k.weight, rank, cp),
          _local_heads(self.conv_v.weight, rank, cp),
          _local_heads(self.A_log, rank, cp),
          _local_heads(self.dt_bias, rank, cp),
          None,  # cu_seqlens: rows come from set_unflatten_seq_len.
          key_head_dim=self.key_head_dim,
          value_head_dim=self.value_head_dim,
          positions=sequence_positions,
      )
      gate_THV = gate_TC.view(num_tokens, -1, self.value_head_dim)
      output_THV = self.norm(output_THV, gate_THV)
      partial_TD = F.linear(
          output_THV.reshape(num_tokens, -1),
          _local_cols(self.out_proj.weight, rank, cp),
      )
      if headtail:
        partial_TD = _sequence_to_headtail_order(partial_TD, cp)
      # Sum the heads' partial outputs, (T, D) -> (T / cp, D).
      out_TD = cp_reduce_scatter(
          partial_TD, group, dim=_TOKEN_DIM, reduce_dtype=self.cp_reduce_dtype
      )
      if self.out_proj.bias is not None:
        out_TD = out_TD + self.out_proj.bias
    if type_check and not spmd.has_local_type(out_TD):
      spmd.assert_type(out_TD, *x_type)
    return out_TD


@dataclass(kw_only=True, slots=True)
class GatedDeltaNetContextParallelTransform(ModelConfigTransform):
  """Run every TPU GatedDeltaNet head-parallel under context parallelism.

  The GatedDeltaNet counterpart of core ``ContextParallelTransform`` (which
  only rewrites attention): replaces each ``TpuInnerGatedDeltaNet`` config
  with a ``CPTpuInnerGatedDeltaNet`` config, keeping its fields (the kernel
  included).
  """

  load_balancer: LoadBalancer
  """``parallelism.context_parallel_load_balancer`` of the recipe."""

  cp_strategy: GatedDeltaNetCPStrategy = "all_to_all"
  """``CPTpuGatedDeltaNet.Config.cp_strategy``. With a strategy other than
  ``"all_to_all"`` every ``TpuGatedDeltaNet`` also becomes a
  ``CPTpuGatedDeltaNet``."""

  cp_reduce_dtype: ReduceDtype = "float32"
  """``CPTpuGatedDeltaNet.Config.cp_reduce_dtype``."""

  def transform(self, model: Module.Config) -> Module.Config:
    if self.cp_strategy not in _CP_STRATEGIES:
      raise ValueError(
          f"cp_strategy must be one of {_CP_STRATEGIES}, got"
          f" {self.cp_strategy!r}."
      )
    for _, delta_net, parent, attr in list(
        model.traverse(TpuGatedDeltaNet.Config)
    ):
      if isinstance(delta_net, CPTpuGatedDeltaNet.Config):
        converted = delta_net
      elif self.cp_strategy == "all_to_all":
        continue
      else:
        converted = convert_config_type(delta_net, CPTpuGatedDeltaNet)
        _replace_child(parent, attr, converted)
      converted.cp_strategy = self.cp_strategy
      converted.cp_reduce_dtype = self.cp_reduce_dtype
    for _, inner, parent, attr in list(
        model.traverse(TpuInnerGatedDeltaNet.Config)
    ):
      converted = convert_config_type(inner, CPTpuInnerGatedDeltaNet)
      converted.load_balancer = self.load_balancer
      _replace_child(parent, attr, converted)
    return model


def _replace_child(parent: Any, attr: Any, child: Any) -> None:
  """Replaces ``parent[attr]`` (a list) or ``parent.attr`` with ``child``."""
  if isinstance(parent, list):
    parent[attr] = child
  else:
    setattr(parent, attr, child)


@register_tpu_cp_reverter
def revert_gated_delta_context_parallel(model: Any) -> None:
  """Swaps the CP GatedDeltaNet configs back to their single-device versions.

  Registered with ``revert_tpu_context_parallel``, for ``cp == 1`` reruns of
  CP recipes (``CPTpuInnerGatedDeltaNet`` requires an active CP mesh axis):
  every ``CPTpuInnerGatedDeltaNet`` becomes a ``TpuInnerGatedDeltaNet`` and
  every ``CPTpuGatedDeltaNet`` a ``TpuGatedDeltaNet``.

  Args:
    model: The model config, modified in place.
  """
  for cp_cls, single_cls in (
      (CPTpuGatedDeltaNet, TpuGatedDeltaNet),
      (CPTpuInnerGatedDeltaNet, TpuInnerGatedDeltaNet),
  ):
    for _, cp_config, parent, attr in list(model.traverse(cp_cls.Config)):
      single = single_cls.Config(**{
          f.name: getattr(cp_config, f.name)
          for f in fields(single_cls.Config)
          if f.init
      })
      _replace_child(parent, attr, single)


@register_tpu_cp_validator
def validate_gated_delta_context_parallel(model: Any, parallelism: Any) -> None:
  """Rejects CP configs whose GatedDeltaNet layers would run on token shards.

  Registered with ``validate_tpu_context_parallel``. With CP on, every
  GatedDeltaNet must run ``CPTpuInnerGatedDeltaNet`` with the recipe's load
  balancer and head counts divisible by ``cp``.

  Args:
    model: The model config.
    parallelism: The parallelism config.
  """
  cp = parallelism.context_parallel_degree
  if cp <= 1:
    return
  delta_nets = list(model.traverse(GatedDeltaNet.Config))
  if delta_nets and parallelism.tensor_parallel_degree > 1:
    raise NotImplementedError(
        "GatedDeltaNet context parallelism does not support tensor parallelism."
    )
  if delta_nets and parallelism.expert_parallel_degree > 1:
    raise NotImplementedError(
        "Qwen3.5 context parallelism with expert parallelism is not validated"
        " yet; use expert_parallel_degree=1 with context parallelism."
    )
  if delta_nets and getattr(model, "vision_encoder", None) is not None:
    raise NotImplementedError(
        "Qwen3.5 context parallelism on TPU is text-only; the model config has"
        " a vision encoder."
    )
  for fqn, delta_net, _, _ in delta_nets:
    inner = delta_net.inner_gated_delta_net
    owner = getattr(inner, "_owner", None)
    if owner is None or not issubclass(owner, CPTpuInnerGatedDeltaNet):
      raise ValueError(
          f"{fqn}.inner_gated_delta_net ({type(inner).__qualname__}) is not"
          f" context-parallel: with context_parallel_degree={cp} it would run"
          " its causal convolution and recurrence on each rank's token shard."
          " Apply GatedDeltaNetContextParallelTransform (see"
          " qwen3_5/config_registry.py::_qwen35_cp)."
      )
    if inner.load_balancer != parallelism.context_parallel_load_balancer:
      raise ValueError(
          f"{fqn}.inner_gated_delta_net expects the"
          f" {inner.load_balancer!r} token layout, but"
          " context_parallel_load_balancer is"
          f" {parallelism.context_parallel_load_balancer!r}."
      )
    n_key_heads = delta_net.in_proj_q.out_features // delta_net.key_head_dim
    n_value_heads = delta_net.in_proj_v.out_features // delta_net.value_head_dim
    for name, count in (
        ("n_key_heads", n_key_heads),
        ("n_value_heads", n_value_heads),
    ):
      if count % cp:
        raise ValueError(
            f"{fqn}: GatedDeltaNet CP shards heads, so {name} ({count}) must"
            f" be divisible by context_parallel_degree ({cp})."
        )
