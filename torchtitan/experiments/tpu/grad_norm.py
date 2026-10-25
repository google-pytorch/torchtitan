# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compiled gradient-norm helpers for TPU training."""

import functools
from collections.abc import Callable
from typing import cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.tpu.graph_trainer import tpu_fsdp
from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
    tpu_backend_compile_pass,
)
from torchtitan.tools.logging import logger


GradSignature = tuple[
    tuple[
        torch.Size,
        torch.dtype,
        DeviceMesh | None,
        tuple[Placement, ...] | None,
    ],
    ...,
]


def _grad_signature(
    grads: list[torch.Tensor], specs: list[tpu_fsdp.ShardSpec | None]
) -> GradSignature:
  signature = []
  for grad, spec in zip(grads, specs, strict=True):
    if isinstance(grad, DTensor):
      signature.append((grad.shape, grad.dtype, grad.device_mesh, grad.placements))
    elif spec is not None:
      # Explicit TPU FSDP: the shard's mesh and placements live on its spec.
      signature.append((grad.shape, grad.dtype, spec.mesh, spec.placements))
    else:
      signature.append((grad.shape, grad.dtype, None, None))
  return tuple(signature)


def global_grad_norm(
    grads: list[torch.Tensor],
    specs: list[tpu_fsdp.ShardSpec | None] | None = None,
) -> torch.Tensor:
  """Compute the global L2 norm of ``grads`` over the whole mesh.

  DTensor gradients (SimpleFSDP) carry their own placements, and
  ``get_total_norm`` leaves a partial scalar whose ``full_tensor`` is the
  reduction. Plain shards under explicit TPU FSDP carry theirs in ``specs``
  (``None`` for a parameter that is not sharded), and ``tpu_fsdp`` reduces
  them over the shard axis explicitly.
  """
  if specs is not None and any(spec is not None for spec in specs):
    if any(isinstance(grad, DTensor) for grad in grads):
      raise ValueError("gradients mix DTensors and explicitly sharded tensors")
    # Same arithmetic as the SimpleFSDP path below, on DTensor views.
    grads = tpu_fsdp.grads_as_dtensors(grads, specs)
  total_norm = torch.nn.utils.get_total_norm(
      grads,
      norm_type=2.0,
      error_if_nonfinite=False,
      foreach=True,
  )
  if isinstance(total_norm, DTensor):
    # get_total_norm leaves an internal partial scalar; full_tensor lowers its
    # global norm reduction here without gathering any gradient tensor.
    return total_norm.full_tensor()
  return total_norm


class CompiledGradNorm:
  """Compile and cache the global norm calculation for a trainer's gradients."""

  def __init__(self) -> None:
    self._runner: Callable[[list[torch.Tensor]], torch.Tensor] | None = None
    self._signature: GradSignature | None = None

  @torch.no_grad()
  def clip_grad_norm_(
      self,
      parameters: list[torch.Tensor],
      max_norm: float,
      spec_by_id: dict[int, tpu_fsdp.ShardSpec] | None = None,
  ) -> torch.Tensor:
    """``spec_by_id`` (``tpu_fsdp.param_specs_for``) marks explicitly sharded
    parameters; absent or empty, every gradient is a DTensor or replicated."""
    with_grad = [parameter for parameter in parameters if parameter.grad is not None]
    grads = cast(list[torch.Tensor], [parameter.grad for parameter in with_grad])
    spec_by_id = spec_by_id or {}
    specs = [spec_by_id.get(id(parameter)) for parameter in with_grad]
    signature = _grad_signature(grads, specs)
    if self._runner is None:
      if not grads:
        raise ValueError("compile_grad_norm requires at least one gradient")
      # Gradients do not exist until backward, so their first materialized
      # shapes and placements are the earliest reliable trace inputs.
      logger.info(
          "[compiled_grad_norm] compiling global norm for %d gradients",
          len(grads),
      )
      # The specs are trace-time constants (group names, placements), so they
      # are bound here and only the gradients are graph inputs.
      traced = minimal_fx_tracer(
          functools.partial(global_grad_norm, specs=specs)
      )(grads)
      # This is a standalone graph; is_fwd avoids backward-only AC marking and
      # selects the forward debug-dump bucket.
      traced.gm = tpu_backend_compile_pass(
          traced.gm,
          traced.example_inputs,
          is_fwd=True,
      )
      self._runner = run_traced(traced, _validate_runtime=True)
      self._signature = signature
    elif signature != self._signature:
      raise ValueError(
          "gradient metadata changed after compiling the gradient norm: "
          f"trace-time {self._signature}, runtime {signature}"
      )
    # Do not retrace a collective independently on ranks that might diverge.
    total_norm = self._runner(grads)

    # Compile the collective-heavy reduction, but leave clipping as one
    # bandwidth-bound eager foreach multiply. Capturing that mutation would
    # return every gradient and, without safe output donation, add a
    # model-sized copy-back or require rebinding every Parameter.grad.
    torch.nn.utils.clip_grads_with_norm_(
        parameters,
        max_norm,
        total_norm,
        foreach=True,
    )
    return total_norm


@torch.no_grad()
def clip_grad_norm_eager_(
    parameters: list[torch.Tensor],
    max_norm: float,
    spec_by_id: dict[int, tpu_fsdp.ShardSpec],
) -> torch.Tensor:
  """Eager clip for explicitly sharded parameters (no compiled norm).

  The base trainer's clip sees plain shards and would use their local norms;
  this routes them through the same DTensor-view norm the compiled path uses.
  """
  with_grad = [p for p in parameters if p.grad is not None]
  grads = cast(list[torch.Tensor], [p.grad for p in with_grad])
  total_norm = global_grad_norm(grads, [spec_by_id.get(id(p)) for p in with_grad])
  torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach=True)
  return total_norm
