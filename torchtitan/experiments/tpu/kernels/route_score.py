# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU route-score gradient reduction."""

import jax
import jax.numpy as jnp
import torch
import torch_tpu  # pylint: disable=unused-import
from torch.tpu import pallas  # pyrefly: ignore[missing-import]


@pallas.jax_op("tt_moe::route_score_grad")
def _route_score_grad_tpu(
    grad_output_ND: jax.Array,
    routed_output_ND: jax.Array,
) -> jax.Array:
  return jnp.sum(
      grad_output_ND.astype(jnp.float32)
      * routed_output_ND.astype(jnp.float32),
      axis=1,
  )


def route_score_grad(
    grad_output_ND: torch.Tensor,
    routed_output_ND: torch.Tensor,
) -> torch.Tensor:
  """Return FP32 dScore[N] without exposing an FP32 [N, D] tensor.

  Both inputs remain BF16 at the operation boundary. Multiplication and the
  hidden-dimension reduction are FP32, matching the native expression's
  precision while allowing XLA to keep the full-size product inside a fusion.
  """
  if grad_output_ND.ndim != 2 or routed_output_ND.ndim != 2:
    raise ValueError("route-score gradient inputs must be rank-2 tensors")
  if grad_output_ND.shape != routed_output_ND.shape:
    raise ValueError(
        "route-score gradient inputs must have the same shape: "
        f"{grad_output_ND.shape} != {routed_output_ND.shape}"
    )
  if (
      grad_output_ND.dtype != torch.bfloat16
      or routed_output_ND.dtype != torch.bfloat16
  ):
    raise ValueError("route-score gradient inputs must both be bfloat16")
  if grad_output_ND.device != routed_output_ND.device:
    raise ValueError("route-score gradient inputs must be on the same device")

  if grad_output_ND.device.type == "cpu":
    return (grad_output_ND.float() * routed_output_ND.float()).sum(dim=1)
  if grad_output_ND.device.type != "tpu":
    raise ValueError(
        "route-score gradient supports only CPU reference and TPU execution"
    )
  return _route_score_grad_tpu(grad_output_ND, routed_output_ND)


__all__ = ["route_score_grad"]
