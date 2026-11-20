# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expert parallelism helpers for TPU.

Provides `use_expert_parallel` to configure a model's MoE layers for TPU EP.
"""

import torchtitan.config
from torchtitan.experiments.tpu.components.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.components.token_dispatcher import TpuAllToAllTokenDispatcher, TpuRoutedExperts
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher


def use_expert_parallel(
    model_config,
    *,
    ep_degree: int,
    ep_transport: str = "ragged_all_to_all",
) -> None:
  """Configures a MoE model config for expert parallelism on TPU.

  A no-op when ``ep_degree`` is 1. Under EP > 1 it:

  - turns plain ``RoutedExperts`` configs into ``TpuRoutedExperts``, whose
    ``parallelize`` binds the EP rank buffer that the ragged transport needs;
  - turns the all-to-all dispatchers into ``TpuAllToAllTokenDispatcher`` with
    ``ep_transport`` (the dropless, static-shape ragged all-to-all by default);
  - turns off the paired (w1 + w3) Pallas grouped matmul, which
    ``TpuGroupedExperts.parallelize`` allows only at EP = 1.

  Args:
    model_config: The model configuration to update in place.
    ep_degree: The expert-parallel degree.
    ep_transport: The EP transport to set on the dispatchers.
  """
  if ep_degree > 1:
    # The paired kernel now zeroes output tiles on their first visit (#182),
    # which may make it EP-safe, but it stays off under EP as in #6 until it
    # has its own EP tests.

    # TpuRoutedExperts binds the EP rank buffer in `parallelize`.
    for _, experts, parent, field_name in list(
        model_config.traverse(RoutedExperts.Config)
    ):
      if type(experts) is RoutedExperts.Config:  # pylint: disable=unidiomatic-typecheck
        ep_experts = torchtitan.config.derive(
            experts,
            TpuRoutedExperts.Config,
        )
        if isinstance(parent, list):
          parent[field_name] = ep_experts
        else:
          setattr(parent, field_name, ep_experts)

    # Unpaired Pallas grouped matmuls under EP.
    for _, group_experts, _, _ in model_config.traverse(TpuGroupedExperts.Config):
      group_experts.use_pallas_paired_grouped_mm = False

    for _, dispatcher, parent, field_name in list(
        model_config.traverse(AllToAllTokenDispatcher.Config)
    ):
      if type(dispatcher) in (  # pylint: disable=unidiomatic-typecheck
          AllToAllTokenDispatcher.Config,
          TpuAllToAllTokenDispatcher.Config,
      ):
        ep_dispatcher = torchtitan.config.derive(
            dispatcher,
            TpuAllToAllTokenDispatcher.Config,
            ep_transport=ep_transport,
        )
        if isinstance(parent, list):
          parent[field_name] = ep_dispatcher
        else:
          setattr(parent, field_name, ep_dispatcher)
