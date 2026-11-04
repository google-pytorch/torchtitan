# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compiled gradient-norm helpers for TPU training."""

from collections.abc import Callable
from typing import cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
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


def _grad_signature(grads: list[torch.Tensor]) -> GradSignature:
  return tuple(
      (grad.shape, grad.dtype, grad.device_mesh, grad.placements)
      if isinstance(grad, DTensor)
      else (grad.shape, grad.dtype, None, None)
      for grad in grads
  )


def _group_by_mesh_and_placements(
    tensors: list[torch.Tensor],
    grads: list[torch.Tensor],
) -> list[list[torch.Tensor]]:
  """Group tensors by their corresponding gradient's (DeviceMesh, placements)."""
  groups: dict[
      tuple[DeviceMesh | None, tuple[Placement, ...] | None],
      list[torch.Tensor],
  ] = {}
  for tensor, grad in zip(tensors, grads, strict=True):
    key = (
        (grad.device_mesh, grad.placements)
        if isinstance(grad, DTensor)
        else (None, None)
    )
    groups.setdefault(key, []).append(tensor)
  return list(groups.values())


def global_grad_norm(grads: list[torch.Tensor]) -> torch.Tensor:
  """Compute the global L2 norm while preserving DTensor placement semantics."""
  grad_groups = _group_by_mesh_and_placements(grads, grads)
  group_norms: list[torch.Tensor] = []
  for group in grad_groups:
    norm = torch.nn.utils.get_total_norm(
        group,
        norm_type=2.0,
        error_if_nonfinite=False,
        foreach=True,
    )
    if isinstance(norm, DTensor):
      # get_total_norm leaves an internal partial scalar; full_tensor lowers its
      # global norm reduction here without gathering any gradient tensor.
      norm = norm.full_tensor()
    group_norms.append(norm)
  if len(group_norms) == 1:
    return group_norms[0]
  return torch.linalg.vector_norm(torch.stack(group_norms), 2.0)


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
  ) -> torch.Tensor:
    params_with_grads = [
        parameter for parameter in parameters if parameter.grad is not None
    ]
    grads = cast(
        list[torch.Tensor],
        [parameter.grad for parameter in params_with_grads],
    )
    signature = _grad_signature(grads)
    if self._runner is None:
      if not grads:
        raise ValueError("compile_grad_norm requires at least one gradient")
      # Gradients do not exist until backward, so their first materialized
      # shapes and placements are the earliest reliable trace inputs.
      logger.info(
          "[compiled_grad_norm] compiling global norm for %d gradients",
          len(grads),
      )
      traced = minimal_fx_tracer(global_grad_norm)(grads)
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
    # bandwidth-bound eager foreach multiply per mesh group. Capturing that
    # mutation would return every gradient and, without safe output donation,
    # add a model-sized copy-back or require rebinding every Parameter.grad.
    for param_group in _group_by_mesh_and_placements(params_with_grads, grads):
      torch.nn.utils.clip_grads_with_norm_(
          param_group,
          max_norm,
          total_norm,
          foreach=True,
      )
    return total_norm


@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | list[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
  """Clip the gradient norm of an iterable of parameters across one or more meshes."""
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  else:
    parameters = list(parameters)
  params_with_grads = [p for p in parameters if p.grad is not None]
  if not params_with_grads:
    return torch.tensor(0.0)
  grads = [cast(torch.Tensor, p.grad) for p in params_with_grads]
  grad_groups = _group_by_mesh_and_placements(grads, grads)
  param_groups = _group_by_mesh_and_placements(params_with_grads, grads)
  if len(grad_groups) <= 1:
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach=foreach
    )
    # Return a plain global norm, as the multi-mesh path below does.
    if isinstance(total_norm, DTensor):
      total_norm = total_norm.full_tensor()
    torch.nn.utils.clip_grads_with_norm_(
        params_with_grads, max_norm, total_norm, foreach=foreach
    )
    return total_norm

  group_norms: list[torch.Tensor] = []
  for g_group in grad_groups:
    g_norm = torch.nn.utils.get_total_norm(
        g_group, norm_type, error_if_nonfinite, foreach=foreach
    )
    if isinstance(g_norm, DTensor):
      g_norm = g_norm.full_tensor()
    group_norms.append(g_norm)
  total_norm = torch.linalg.vector_norm(torch.stack(group_norms), norm_type)
  for p_group in param_groups:
    torch.nn.utils.clip_grads_with_norm_(
        p_group, max_norm, total_norm, foreach=foreach
    )
  return total_norm

