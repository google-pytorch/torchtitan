# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixed-K FP32 sum of each token's routed rows."""

import torch


def route_sum(
    src_ND: torch.Tensor, rows_N: torch.Tensor, top_k: int
) -> torch.Tensor:
  """Return ``src_ND[rows_N].reshape(T, top_k, D).float().sum(1)``.

  ``rows_N`` is token-major: entry ``t * top_k + k`` is the source row of
  route ``k`` of token ``t``. The result has ``src_ND``'s dtype.

  On TPU, the sum over the middle axis of the token-major ``(T, K, D)``
  gather runs at about half of HBM bandwidth, because each token's ``K`` rows
  share a tile. Gathering route-major instead, ``(K, T, D)``, turns the sum
  into adds of whole ``(T, D)`` slabs. The adds pair routes ``k`` and
  ``k + K/2``, then halve again: that is the order of XLA's TPU reduce over
  the middle axis for ``K`` in 2, 4, 8, so the TPU result is bitwise
  unchanged. Other ``K`` keep the token-major sum.
  """
  num_rows, dim = rows_N.shape[0], src_ND.shape[1]
  if top_k not in (2, 4, 8):
    return (
        torch.index_select(src_ND, 0, rows_N)
        .reshape(-1, top_k, dim)
        .float()
        .sum(dim=1)
        .to(src_ND.dtype)
    )
  rows_KT = rows_N.reshape(-1, top_k).t().reshape(-1)
  routes = torch.index_select(src_ND, 0, rows_KT).reshape(
      top_k, num_rows // top_k, dim
  )
  partial = [route.float() for route in routes.unbind(0)]
  while len(partial) > 1:
    half = len(partial) // 2
    partial = [partial[i] + partial[i + half] for i in range(half)]
  return partial[0].to(src_ND.dtype)
