"""Dropless static-shape expert parallelism (EP > 1) for TPU MoE models.

Upstream's ``AllToAllTokenDispatcher`` copies the per-rank split sizes to the
host and calls a variable-size ``all_to_all_single``. TorchTPU rejects that
even eagerly: "asymmetric output buffer sizes across ranks are not supported on
TPU" (measured, DeepSeek-V3 debugmodel at EP 4), and a host sync of split
sizes cannot be compiled either.

``TpuRaggedEPTokenDispatcher`` keeps every shape static and every size on the
device:

1. Each rank sorts its ``N = T * K`` routes by global expert (the TPU local
   reorder). Three equal-split all-to-alls of ``int32[ep, E / ep]`` tables
   exchange the per-(rank, local expert) counts, the receivers' write offsets
   and the senders' row offsets. No rank index enters the program: a rank
   constant made each rank compile a different per-block program and the
   ragged collective never completed (measured hang, EP 4).
2. From those every rank has ``E``-entry offset and size vectors (one update
   per destination rank and local expert) and
   ``torch.tpu.ops.ragged_all_to_all`` writes each slice straight into the
   destination's expert-major order inside a zero-initialized
   ``[ep * N, D]`` buffer. No token is dropped whatever the routing skew:
   ``ep * N`` is the most rows a rank can receive.
3. The grouped matmul runs on that buffer with the exact per-local-expert
   counts, so only the received rows do work.
4. ``combine`` sends the expert outputs back with the mirrored ragged
   all-to-all into the local ``[N, D]`` expert-sorted order, then applies the
   EP=1 fixed-``top_k`` FP32 reduction (no element-wise scatter-add combine).

The backward of each ragged all-to-all is the mirrored ragged all-to-all, with
its offsets precomputed in the forward. The buffer is ``ep`` times the routed
activations (``ep * N * D``), so large EP degrees cost HBM; size EP with it.

Bounded receive buffer (opt-in, ``receive_capacity_factor``): the buffer is
``ceil(factor * N)`` rows rounded up to ``receive_capacity_multiple`` (the
grouped-matmul row tile) instead of ``ep * N``. Still dropless: receivers clip
each slice to the capacity (so an overflowing exchange never writes out of
bounds), record an overflow flag and the received row count in
``receive_stats``, and ``ReceiveOverflowReplay`` (driven by ``TPUTrainer``)
reads the flags once per step before the optimizer and, if any rank of any
layer overflowed, restores the model buffers and re-runs the whole step with a
larger buffer (``receive_replay_capacity_factor``, then the worst-case
``ep * N`` program). The optimizer only ever sees a step in which no slice was
clipped. Without the trainer's replay (``receive_stats`` unset) a bounded
dispatcher refuses to run rather than drop tokens.

Adapted from the Qwen3.5 TPU EP dispatcher (``qwen3_5/token_dispatcher.py`` on
branch ``qwen3_5-35b-a3b-tpu-moe-ep``, commit 65a061601), made model-agnostic
and TPU-only (no host fallback).
"""

from collections.abc import Iterator
import contextlib
from dataclasses import dataclass
import math

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.autograd.function import once_differentiable

from torchtitan.config import derive
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.experiments.tpu.remat_regions import annotate_region
from torchtitan.experiments.tpu.remat_regions import MOE_ROUTING
from torchtitan.experiments.tpu.token_dispatcher import scale_route_weights
from torchtitan.experiments.tpu.token_dispatcher import stable_grouped_reduce
from torchtitan.experiments.tpu.token_dispatcher import TpuAllToAllTokenDispatcher
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.common.token_dispatcher import LocalDispatchMetadata
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.tools.logging import logger


# Pallas grouped-matmul row tile of the DeepSeek-V3 / Qwen3 TPU recipes.
DEFAULT_RECEIVE_CAPACITY_MULTIPLE = 512


def _ragged_all_to_all(
    operand: torch.Tensor,
    output_rows: int,
    input_offsets: torch.Tensor,
    send_sizes: torch.Tensor,
    output_offsets: torch.Tensor,
    recv_sizes: torch.Tensor,
    group_name: str,
) -> torch.Tensor:
  """``torch.tpu.ops.ragged_all_to_all`` into a fresh zero buffer.

  The operand is detached: the op's own autograd kernel issues eager
  collectives on the offset vectors, which the GraphTrainer's ``make_fx``
  tracer cannot capture. ``_RaggedAllToAll`` supplies the backward instead.
  """
  src = operand.detach().contiguous()
  output = src.new_zeros((output_rows, *src.shape[1:]))
  if src.device.type != "tpu":
    raise NotImplementedError(
        "TpuRaggedEPTokenDispatcher runs on TPU only (device"
        f" {src.device.type})"
    )
  return torch.tpu.ops.ragged_all_to_all(
      src,
      output,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      group_name,
  )


class _RaggedAllToAll(torch.autograd.Function):
  """Differentiable ragged all-to-all; the backward is the mirrored exchange.

  Takes the four dispatch vectors plus the two combine offset vectors (the
  combine sizes are the dispatch sizes swapped), each once: Dynamo rejects an
  ``autograd.Function`` given the same tensor twice. ``reverse`` selects the
  combine direction for the forward (and dispatch for the backward).
  """

  @staticmethod
  def forward(
      ctx,
      operand,
      output_rows,
      disp_input_offsets,
      disp_send_sizes,
      disp_output_offsets,
      disp_recv_sizes,
      comb_input_offsets,
      comb_output_offsets,
      reverse,
      group_name,
  ):
    ctx.set_materialize_grads(False)
    ctx.operand_rows = operand.shape[0]
    ctx.group_name = group_name
    ctx.reverse = reverse
    ctx.save_for_backward(
        disp_input_offsets,
        disp_send_sizes,
        disp_output_offsets,
        disp_recv_sizes,
        comb_input_offsets,
        comb_output_offsets,
    )
    return _ragged_all_to_all(
        operand,
        output_rows,
        *_directed(
            disp_input_offsets,
            disp_send_sizes,
            disp_output_offsets,
            disp_recv_sizes,
            comb_input_offsets,
            comb_output_offsets,
            reverse,
        ),
        group_name,
    )

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_out):
    if grad_out is None or not ctx.needs_input_grad[0]:
      return (None,) * 10
    grad_operand = _ragged_all_to_all(
        grad_out,
        ctx.operand_rows,
        *_directed(*ctx.saved_tensors, not ctx.reverse),
        ctx.group_name,
    )
    return (grad_operand,) + (None,) * 9


def _directed(
    disp_input_offsets,
    disp_send_sizes,
    disp_output_offsets,
    disp_recv_sizes,
    comb_input_offsets,
    comb_output_offsets,
    reverse,
):
  """(input offsets, send sizes, output offsets, recv sizes) for a direction."""
  if reverse:
    return (
        comb_input_offsets,
        disp_recv_sizes,
        comb_output_offsets,
        disp_send_sizes,
    )
  return (disp_input_offsets, disp_send_sizes, disp_output_offsets, disp_recv_sizes)


def _exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
  """Row-major exclusive prefix sum over all elements, int32, same shape."""
  flat = x.reshape(-1)
  return (flat.cumsum(0, dtype=torch.int32) - flat).view(x.shape)


def _exchange(x_Pe: torch.Tensor, pg) -> torch.Tensor:
  """Equal-split all-to-all of an int32 ``[ep, e]`` table over the EP group.

  Row ``p`` goes to EP rank ``p``; row ``p`` of the result came from rank
  ``p``. Equal splits keep the buffers symmetric (TorchTPU rejects
  asymmetric ones) and the result needs no rank index, so every rank compiles
  the same program.
  """
  out = funcol.all_to_all_single(x_Pe.contiguous(), None, None, group=pg)
  if isinstance(out, funcol.AsyncCollectiveTensor):
    out = out.wait()
  return out


@dataclass(frozen=True, kw_only=True)
class RaggedEPDispatchMetadata:
  """What ``combine`` needs to send the expert outputs back."""

  token_indices_experts_sorted_N: torch.Tensor
  topk_scores_experts_sorted_N: torch.Tensor
  routed_rows: int
  disp_input_offsets: torch.Tensor
  disp_send_sizes: torch.Tensor
  disp_output_offsets: torch.Tensor
  disp_recv_sizes: torch.Tensor
  comb_input_offsets: torch.Tensor
  comb_output_offsets: torch.Tensor


class TpuRaggedEPTokenDispatcher(TpuAllToAllTokenDispatcher):
  """``TpuAllToAllTokenDispatcher`` with a static-shape ragged EP transport.

  EP = 1 (no EP mesh) is exactly ``TpuAllToAllTokenDispatcher``.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TpuAllToAllTokenDispatcher.Config):
    receive_capacity_factor: float | None = None
    """Receive buffer rows as a multiple of the average a rank receives
    (``N = tokens x top_k``), rounded up to ``receive_capacity_multiple`` and
    capped at the most rows a rank can receive (``ep x tokens x min(top_k,
    local experts)``). ``None`` (default): the worst-case ``ep x N`` buffer,
    no overflow check. Needs ``TPUTrainer`` (overflow replay)."""

    receive_replay_capacity_factor: float | None = None
    """Buffer factor for the first replay of an overflowed step; ``None``
    replays straight on the worst-case program. A replay that overflows
    again falls through to the worst-case program."""

    receive_capacity_multiple: int = DEFAULT_RECEIVE_CAPACITY_MULTIPLE
    """Row multiple of a bounded buffer (the grouped-matmul row tile)."""

  # ``receive_level`` values: the configured factor, the replay factor, the
  # worst-case program (``ReceiveOverflowReplay`` steps through them).
  LEVELS = ("bounded", "replay", "worst_case")

  def __init__(self, config: Config):
    super().__init__(config)
    for name in ("receive_capacity_factor", "receive_replay_capacity_factor"):
      factor = getattr(config, name)
      if factor is not None and not factor > 0:
        raise ValueError(f"{name} must be > 0, got {factor}")
    if config.receive_capacity_multiple < 1:
      raise ValueError(
          "receive_capacity_multiple must be >= 1, got"
          f" {config.receive_capacity_multiple}"
      )
    if (
        config.receive_replay_capacity_factor is not None
        and config.receive_capacity_factor is None
    ):
      raise ValueError(
          "receive_replay_capacity_factor needs receive_capacity_factor"
      )
    if (
        config.receive_replay_capacity_factor is not None
        and config.receive_replay_capacity_factor
        <= config.receive_capacity_factor
    ):
      raise ValueError(
          "receive_replay_capacity_factor must exceed receive_capacity_factor"
      )
    self.receive_capacity_factor = config.receive_capacity_factor
    self.receive_replay_capacity_factor = config.receive_replay_capacity_factor
    self.receive_capacity_multiple = config.receive_capacity_multiple
    # Set by ReceiveOverflowReplay (host side). ``receive_level`` indexes
    # LEVELS; Dynamo guards on it, so each level is its own compiled program.
    self.receive_level = 0
    # int32[3] per step: overflow flag, most rows received in one call,
    # average rows per rank (N). Running max over the step's calls.
    self.receive_stats: torch.Tensor | None = None

  @property
  def bounded(self) -> bool:
    """Whether the receive buffer is bounded (EP > 1 and a factor is set)."""
    return self.ep_mesh is not None and self.receive_capacity_factor is not None

  def receive_capacity(self, routed_rows: int, ep_size: int) -> int | None:
    """Receive buffer rows at the current level; ``None`` = worst-case path."""
    if self.receive_capacity_factor is None:
      return None
    factor = (
        self.receive_capacity_factor,
        self.receive_replay_capacity_factor,
        None,
    )[self.receive_level]
    if factor is None:
      return None
    rows = max(routed_rows, 1)
    multiple = self.receive_capacity_multiple
    capacity = -(-math.ceil(factor * rows) // multiple) * multiple
    # Exact bound: a token's top_k routes go to distinct experts, so it sends
    # at most min(top_k, local experts) rows to one rank. Below ``ep * N``
    # only when a rank holds fewer experts than top_k (the 16-expert proxies
    # at EP 4: 2 * N; the 256-expert target at EP 16 holds 16 >= 8).
    local_experts = self.num_experts // ep_size
    tokens = -(-rows // self.top_k)
    return min(capacity, ep_size * tokens * min(self.top_k, local_experts))

  def dispatch(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      topk_expert_ids_TK: torch.Tensor,
      num_local_tokens_per_expert_E: torch.Tensor,
  ):
    if self.ep_mesh is None:
      return super().dispatch(
          x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
      )
    ep_size = self.ep_mesh.size()
    if self.num_experts % ep_size:
      raise ValueError(
          f"{self.num_experts} experts do not divide over EP {ep_size}"
      )
    e = self.num_experts // ep_size
    (
        routed_input_ND,
        token_indices_experts_sorted_N,
        topk_scores_experts_sorted_N,
    ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)
    num_routed = routed_input_ND.shape[0]
    capacity = self.receive_capacity(num_routed, ep_size)
    if capacity is None:
      # The most rows any rank can receive: every rank routing all of its
      # rows to this rank's experts. Dropless by construction.
      received_rows = ep_size * max(num_routed, 1)
    else:
      if self.receive_stats is None:
        raise RuntimeError(
            "TpuRaggedEPTokenDispatcher with receive_capacity_factor needs the"
            " overflow replay of TPUTrainer (ReceiveOverflowReplay); this"
            " trainer does not install it, and a bounded buffer without it"
            " would drop tokens"
        )
      received_rows = capacity

    with maybe_set_sparse_mesh():
      pg = self.ep_mesh.get_group()
      with torch.no_grad():
        with annotate_region(MOE_ROUTING):
          # What this rank sends: per destination rank, per local expert.
          send_sizes_De = num_local_tokens_per_expert_E.to(torch.int32).view(
              ep_size, e
          )
          # What each source rank sends to this rank's experts.
          recv_sizes_Se = _exchange(send_sizes_De, pg)
        send_offsets_De = _exclusive_cumsum(send_sizes_De)
        # This rank's receive buffer is expert-major:
        # [local_e 0 (src 0..ep-1), ..., local_e e-1 (src 0..ep-1)].
        write_offsets_Se = _exclusive_cumsum(recv_sizes_Se.t()).t()
        disp_input_offsets = send_offsets_De.reshape(-1)
        if capacity is None:
          disp_send_sizes = send_sizes_De.reshape(-1)
          # Where this rank's slices land in each destination's buffer.
          disp_output_offsets = _exchange(write_offsets_Se, pg).reshape(-1)
        else:
          # Clip every slice to the buffer (equal to recv_sizes unless this
          # rank overflows; an overflowed step is replayed, never applied) and
          # send the clipped sizes back with the write offsets, so senders
          # never write past the capacity.
          received_total = recv_sizes_Se.sum(dtype=torch.int32)
          recv_sizes_Se = torch.minimum(
              recv_sizes_Se, (capacity - write_offsets_Se).clamp(min=0)
          )
          write_offsets_Se = write_offsets_Se.clamp(max=capacity - 1)
          offsets_and_sizes = _exchange(
              torch.cat([write_offsets_Se, recv_sizes_Se], dim=1), pg
          )
          disp_output_offsets = offsets_and_sizes[:, :e].reshape(-1)
          disp_send_sizes = offsets_and_sizes[:, e:].reshape(-1)
          stats = torch.stack([
              (received_total > capacity).to(torch.int32),
              received_total,
              torch.full_like(received_total, num_routed),
          ])
          self.receive_stats.copy_(torch.maximum(self.receive_stats, stats))
        disp_recv_sizes = recv_sizes_Se.reshape(-1)
        comb_input_offsets = write_offsets_Se.reshape(-1)
        # Where each source keeps its rows for this rank's experts.
        comb_output_offsets = _exchange(send_offsets_De, pg).reshape(-1)
        num_tokens_per_local_expert_e = recv_sizes_Se.sum(
            dim=0, dtype=torch.int32
        )

      routed_input_RD = _RaggedAllToAll.apply(
          routed_input_ND,
          received_rows,
          disp_input_offsets,
          disp_send_sizes,
          disp_output_offsets,
          disp_recv_sizes,
          comb_input_offsets,
          comb_output_offsets,
          False,
          pg.group_name,
      )

    metadata = RaggedEPDispatchMetadata(
        token_indices_experts_sorted_N=token_indices_experts_sorted_N,
        topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
        routed_rows=num_routed,
        disp_input_offsets=disp_input_offsets,
        disp_send_sizes=disp_send_sizes,
        disp_output_offsets=disp_output_offsets,
        disp_recv_sizes=disp_recv_sizes,
        comb_input_offsets=comb_input_offsets,
        comb_output_offsets=comb_output_offsets,
    )
    return routed_input_RD, num_tokens_per_local_expert_e, metadata

  def combine(self, routed_output_RD, metadata, x_TD):
    if self.ep_mesh is None:
      return super().combine(routed_output_RD, metadata, x_TD)
    assert isinstance(metadata, RaggedEPDispatchMetadata)
    with maybe_set_sparse_mesh():
      routed_output_ND = _RaggedAllToAll.apply(
          routed_output_RD,
          metadata.routed_rows,
          metadata.disp_input_offsets,
          metadata.disp_send_sizes,
          metadata.disp_output_offsets,
          metadata.disp_recv_sizes,
          metadata.comb_input_offsets,
          metadata.comb_output_offsets,
          True,
          self.ep_mesh.get_group().group_name,
      )
    if self.fuse_route_score_backward:
      routed_output_ND = scale_route_weights(
          routed_output_ND, metadata.topk_scores_experts_sorted_N
      )
    else:
      routed_output_ND = (
          routed_output_ND.float()
          * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
      ).to(routed_output_ND.dtype)
    return stable_grouped_reduce(
        routed_output_ND, metadata.token_indices_experts_sorted_N, self.top_k
    )


class TpuRaggedEPTokenDispatcherConverter(ModelConfigConverter):
  """Select ``TpuRaggedEPTokenDispatcher`` for the standard / TPU dispatcher."""

  @dataclass(kw_only=True, slots=True)
  class Config(ModelConfigConverter.Config):
    pass

  def __init__(self, config: Config):
    pass

  def convert(self, model_config):
    matches = list(model_config.traverse(AllToAllTokenDispatcher.Config))
    for _, config, parent, field_name in matches:
      if type(config) not in (
          AllToAllTokenDispatcher.Config,
          TpuAllToAllTokenDispatcher.Config,
      ):
        continue
      replacement = derive(config, TpuRaggedEPTokenDispatcher.Config)
      if parent is None:
        model_config = replacement
      elif isinstance(parent, list):
        parent[field_name] = replacement
      else:
        setattr(parent, field_name, replacement)
    return model_config


class ReceiveBufferOverflow(Exception):
  """A bounded receive buffer overflowed this step; replay it larger."""


def _bounded_dispatchers(
    model_parts: list[torch.nn.Module],
) -> list[TpuRaggedEPTokenDispatcher]:
  seen = {}
  for part in model_parts:
    for module in part.modules():
      dispatcher = getattr(module, "token_dispatcher", None)
      if isinstance(dispatcher, TpuRaggedEPTokenDispatcher) and dispatcher.bounded:
        seen[id(dispatcher)] = dispatcher
  return list(seen.values())


class ReceiveOverflowReplay:
  """Host side of the bounded receive buffer: exact overflow detect + replay.

  Per step (``TPUTrainer.train_step``): ``begin_step`` snapshots every model
  buffer (the MoE ``tokens_per_expert_E`` counters and aux-loss accumulators
  the forward mutates) and zeroes ``receive_stats``; the step runs at level 0
  (the configured factor); ``check`` (called before the optimizer, from
  ``TPUTrainer._clip_grad_norm``) reduces the overflow flags over all layers
  and ranks into one host read and raises ``ReceiveBufferOverflow`` if any
  slice was clipped. The trainer then ``restore``s the buffers and re-runs the
  same batches one level up (replay factor, then the worst-case program, which
  cannot overflow and is not checked). The optimizer, the balancer pre-hook,
  the LR scheduler and the gradients only see the attempt that did not clip.
  """

  def __init__(
      self,
      model_parts: list[torch.nn.Module],
      dispatchers: list[TpuRaggedEPTokenDispatcher],
      device: torch.device,
  ):
    self.dispatchers = dispatchers
    for dispatcher in dispatchers:
      dispatcher.receive_stats = torch.zeros(3, dtype=torch.int32, device=device)
      dispatcher.receive_level = 0
    self._buffers = [b for part in model_parts for b in part.buffers()]
    self._snapshot: list[torch.Tensor] | None = None
    self.level = 0
    self.replays = 0
    # Most rows one rank received in one MoE call this step / average rows
    # (from the step's first attempt); and the level the step was applied at.
    self.max_load: float | None = None
    self.applied_level = 0

  @classmethod
  def maybe_create(
      cls, model_parts: list[torch.nn.Module], device: torch.device
  ) -> "ReceiveOverflowReplay | None":
    dispatchers = _bounded_dispatchers(model_parts)
    if not dispatchers:
      return None
    return cls(model_parts, dispatchers, device)

  def levels(self) -> Iterator[int]:
    """Levels to try this step: bounded, replay (if configured), worst case."""
    yield 0
    if any(d.receive_replay_capacity_factor is not None for d in self.dispatchers):
      yield 1
    yield 2

  @contextlib.contextmanager
  def at_level(self, level: int):
    previous = self.level
    self.level = level
    for dispatcher in self.dispatchers:
      dispatcher.receive_level = level
    try:
      yield
    finally:
      self.level = previous
      for dispatcher in self.dispatchers:
        dispatcher.receive_level = previous

  @torch.no_grad()
  def begin_step(self) -> None:
    if self._snapshot is None:
      self._snapshot = [torch.empty_like(b) for b in self._buffers]
    for snapshot, buffer in zip(self._snapshot, self._buffers, strict=True):
      snapshot.copy_(buffer)
    for dispatcher in self.dispatchers:
      dispatcher.receive_stats.zero_()
    self.max_load = None
    self.applied_level = 0

  @torch.no_grad()
  def restore(self) -> None:
    assert self._snapshot is not None
    for snapshot, buffer in zip(self._snapshot, self._buffers, strict=True):
      buffer.copy_(snapshot)
    for dispatcher in self.dispatchers:
      dispatcher.receive_stats.zero_()

  @torch.no_grad()
  def check(self, step: int) -> None:
    """Raises ``ReceiveBufferOverflow`` if any rank clipped a slice this attempt.

    One int32[3] max all-reduce over the world and one host read per attempt
    (all ranks must agree to replay: the step's collectives span them all).
    The worst-case level writes no stats and is not checked.
    """
    if self.level == 2:
      self.applied_level = 2
      return
    stats = torch.stack([d.receive_stats for d in self.dispatchers]).amax(0)
    if dist.is_initialized():
      dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    overflow, max_rows, avg_rows = stats.tolist()
    if self.max_load is None:
      self.max_load = max_rows / max(avg_rows, 1)
    if not overflow:
      self.applied_level = self.level
      return
    self.replays += 1
    capacity = self.dispatchers[0].receive_capacity(
        avg_rows, self.dispatchers[0].ep_mesh.size()
    )
    logger.warning(
        f"EP receive buffer overflow at step {step}, level"
        f" {TpuRaggedEPTokenDispatcher.LEVELS[self.level]}: {max_rows} rows"
        f" received ({max_rows / max(avg_rows, 1):.3f}x the average"
        f" {avg_rows}) > capacity {capacity}; replaying the step"
        f" (replays so far: {self.replays})"
    )
    raise ReceiveBufferOverflow(step)

  def metrics(self) -> dict[str, float]:
    metrics = {
        "ep_receive/replays": float(self.replays),
        "ep_receive/applied_level": float(self.applied_level),
    }
    if self.max_load is not None:
      metrics["ep_receive/max_load"] = self.max_load
    return metrics


__all__ = [
    "RaggedEPDispatchMetadata",
    "ReceiveBufferOverflow",
    "ReceiveOverflowReplay",
    "TpuRaggedEPTokenDispatcher",
    "TpuRaggedEPTokenDispatcherConverter",
]
