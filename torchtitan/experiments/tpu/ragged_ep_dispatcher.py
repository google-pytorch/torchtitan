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

Adapted from the Qwen3.5 TPU EP dispatcher (``qwen3_5/token_dispatcher.py`` on
branch ``qwen3_5-35b-a3b-tpu-moe-ep``, commit 65a061601), made model-agnostic
and TPU-only (no host fallback).
"""

from dataclasses import dataclass

import torch
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
    pass

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
    # The most rows any rank can receive: every rank routing all of its rows
    # to this rank's experts. Dropless by construction.
    received_rows = ep_size * max(num_routed, 1)

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
        disp_send_sizes = send_sizes_De.reshape(-1)
        # Where this rank's slices land in each destination's buffer.
        disp_output_offsets = _exchange(write_offsets_Se, pg).reshape(-1)
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


__all__ = [
    "RaggedEPDispatchMetadata",
    "TpuRaggedEPTokenDispatcher",
    "TpuRaggedEPTokenDispatcherConverter",
]
