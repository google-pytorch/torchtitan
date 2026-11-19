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
# pyformat: enable
from torchtitan.config.transform import convert_config_type, ModelConfigTransform
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.experiments.tpu.components.unflatten_shim import get_unflatten_seq_len
from torchtitan.experiments.tpu.distributed.cp_attention import (
    cp_all_gather,
    cp_all_to_all,
    cp_input_layout,
    cp_rank_tensor,
    register_tpu_cp_reverter,
    register_tpu_cp_validator,
    reorder_sequence,
)

# Before `torchtitan.models.qwen3_5`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu.qwen3_5 import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.experiments.tpu.qwen3_5.gated_delta import TpuInnerGatedDeltaNet
from torchtitan.models.qwen3_5.gdn import GatedDeltaNet
from torchtitan.protocols.module import Module

__all__ = [
    "CPTpuInnerGatedDeltaNet",
    "GatedDeltaNetContextParallelTransform",
    "revert_gated_delta_context_parallel",
    "validate_gated_delta_context_parallel",
]

_TOKEN_DIM = 0
_CHANNEL_DIM = 1

LoadBalancer = Literal["headtail"] | None


def _rows(num_tokens: int) -> tuple[int, int]:
  """``(B, S)`` of a folded ``[B * S]`` token stream in head-tail layout.

  ``S`` comes from the shim, as in KV all-gather ``cp_shard``, which
  reorders per sequence of length ``S`` and treats a stream that ``S`` does
  not divide as one sequence. Raises if that disagrees with the layout
  ``cp_shard`` recorded for this token count (e.g. a different
  ``set_unflatten_seq_len`` at sharding and at forward time), which would
  otherwise silently scramble the token order.
  """
  seq_len = get_unflatten_seq_len()
  if seq_len is None or num_tokens % seq_len:
    # One sequence, as the CP input sharding treats a shorter stream.
    seq_len = num_tokens
  rows = (num_tokens // seq_len, seq_len)
  recorded = cp_input_layout(num_tokens)
  if recorded is not None and recorded != rows:
    raise RuntimeError(
        f"GatedDeltaNet CP: cp_shard laid {num_tokens} tokens out as"
        f" (rows, row_len)={recorded}, but set_unflatten_seq_len gives {rows}"
        " at forward time; shard and run the model under the same"
        " set_unflatten_seq_len."
    )
  return rows


def _check_contiguous_layout(num_tokens: int) -> None:
  """Raises unless ``cp_shard`` split one contiguous token stream (Ulysses).

  The all-to-all concatenates the ranks' shards in rank order, which is
  sequence order only if each rank holds a contiguous piece of the whole
  folded stream (sequence dim 0), not ``[B, S / cp]`` of a ``[B, S]`` input.
  """
  recorded = cp_input_layout(num_tokens)
  if recorded is not None and recorded[0] != 1:
    raise RuntimeError(
        "GatedDeltaNet Ulysses CP needs a folded [B * S] token stream sharded"
        f" on dim 0, but cp_shard sharded {recorded[0]} rows of"
        f" {recorded[1]} tokens."
    )


def _headtail_to_sequence_order(x: torch.Tensor, cp: int) -> torch.Tensor:
  """``[T, ...]`` gathered from the CP ranks in head-tail layout -> sequence order.

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


def _local_heads(x: torch.Tensor, rank: torch.Tensor, cp: int) -> torch.Tensor:
  """Rank ``rank``'s contiguous ``1 / cp`` slice of dim 0 (per-head parameters)."""
  local = x.shape[0] // cp
  rest = list(x.shape[1:])
  return (
      x.reshape([cp, local] + rest)
      .index_select(0, rank)
      .reshape([local] + rest)
  )


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
      cu_seqlens: torch.Tensor,
      *,
      key_head_dim: int,
      value_head_dim: int,
      positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
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
    n_key_heads = query_TC.shape[-1] // key_head_dim
    n_value_heads = value_TC.shape[-1] // value_head_dim
    if n_key_heads % cp or n_value_heads % cp:
      raise ValueError(
          f"GatedDeltaNet CP shards heads: n_key_heads ({n_key_heads}) and"
          f" n_value_heads ({n_value_heads}) must be divisible by cp ({cp})."
      )
    headtail = self.load_balancer == "headtail"
    rank = cp_rank_tensor(query_TC.device, group)

    # (T / cp, C) -> (T, C / cp): every token of the rank's heads. One
    # all-to-all per input: packing them into one measured slower on TPU7x
    # (XLA overlaps the later transfers with the earlier inputs' conv).
    query_TC, key_TC, value_TC, a_TH, b_TH = (
        cp_all_to_all(x, group, src_dim=_TOKEN_DIM, dst_dim=_CHANNEL_DIM)
        for x in (query_TC, key_TC, value_TC, a_TH, b_TH)
    )
    if not headtail:
      _check_contiguous_layout(query_TC.shape[0])
    full_positions = None
    if positions is not None:
      # Document resets need every token's position (4 bytes per token).
      full_positions = cp_all_gather(positions.reshape(-1), group)
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
        if full_positions is not None:
          full_positions = _headtail_to_sequence_order(full_positions, cp)
      out_THV = super().forward(
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
          positions=full_positions,
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

  def transform(self, model: Module.Config) -> Module.Config:
    for _, inner, parent, attr in model.traverse(TpuInnerGatedDeltaNet.Config):
      converted = convert_config_type(inner, CPTpuInnerGatedDeltaNet)
      converted.load_balancer = self.load_balancer
      if isinstance(parent, list):
        parent[attr] = converted
      else:
        setattr(parent, attr, converted)
    return model


@register_tpu_cp_reverter
def revert_gated_delta_context_parallel(model: Any) -> None:
  """Swaps every ``CPTpuInnerGatedDeltaNet`` config back to ``TpuInnerGatedDeltaNet``.

  Registered with ``revert_tpu_context_parallel``, for ``cp == 1`` reruns of
  CP recipes (``CPTpuInnerGatedDeltaNet`` requires an active CP mesh axis).

  Args:
    model: The model config, modified in place.
  """
  for _, inner, parent, attr in model.traverse(CPTpuInnerGatedDeltaNet.Config):
    single = TpuInnerGatedDeltaNet.Config(**{
        f.name: getattr(inner, f.name)
        for f in fields(TpuInnerGatedDeltaNet.Config)
    })
    if isinstance(parent, list):
      parent[attr] = single
    else:
      setattr(parent, attr, single)


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
  for fqn, delta_net, _, _ in model.traverse(GatedDeltaNet.Config):
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
    if parallelism.tensor_parallel_degree > 1:
      raise NotImplementedError(
          "GatedDeltaNet context parallelism does not support tensor"
          " parallelism."
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
