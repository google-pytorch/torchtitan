# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 sharding helpers shared by regular Trainer and GraphTrainer."""

import torch.nn as nn

import torchtitan.distributed
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import (
    configure_simple_fsdp_shard_dims,
)
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.tools.logging import logger


def configure_qwen_simple_fsdp_shard_dims(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
) -> None:
  """Use divisible hidden axes for uneven Qwen parameters under SimpleFSDP."""
  fsdp_degree = parallel_dims.dp_shard * parallel_dims.cp
  if fsdp_degree <= 1:
    return

  hidden_axis_modules = configure_simple_fsdp_shard_dims(model, parallel_dims)
  stacks = any(hasattr(m, "plan_shard_dims") for m in model.modules())

  # With EP, apply_simple_fsdp separately wraps each expert bank on the EDP
  # mesh and owns its shard-axis selection.
  if not parallel_dims.ep_enabled:
    qwen_expert_modules = [
        (module_fqn, module)
        for module_fqn, module in model.named_modules()
        if isinstance(module, TpuGroupedExperts)
    ]
    for module_fqn, module in qwen_expert_modules:
      if module.num_experts % fsdp_degree == 0:
        continue

      parameters = list(module.named_parameters(recurse=False))
      if not parameters:
        if stacks:
          continue  # Stacking took them; the holder above shards them instead.
        raise ValueError(f"{module_fqn} has no direct parameters to FSDP-shard")
      # W2's dim 1 (F=1536) gives a 3-row stripe at 512 ranks, which no TPU
      # tile fits. Its trailing D axis gives 8 rows, but the gather then stores
      # W2 as [E, D, F], so the grouped matmul reads it as a transposed RHS
      # instead of paying a full relayout copy.
      # TODO: remove the trailing-W2/transposed-RHS path if GraphSplitter
      # aligns with Scan's flattened dim-0 expert sharding, or if FlexShard
      # provides tile-aligned expert shards.
      use_trailing_w2 = (
          module.use_kernel_native_weight_layout
          and any(
              name == "w2_EFD"
              and parameter.ndim == 3
              and parameter.shape[2] % fsdp_degree == 0
              for name, parameter in parameters
          )
      )
      shard_dims: dict[str, int] = {}
      incompatible: list[str] = []
      for name, parameter in parameters:
        if parameter.ndim != 3 or parameter.shape[0] != module.num_experts:
          incompatible.append(f"{name}{tuple(parameter.shape)}")
          continue
        if use_trailing_w2 and name == "w2_EFD":
          shard_dims[name] = 2
        elif parameter.shape[1] % fsdp_degree == 0:
          shard_dims[name] = 1
        else:
          incompatible.append(f"{name}{tuple(parameter.shape)}")
      if incompatible:
        raise ValueError(
            f"Cannot FSDP-shard {module_fqn} over {fsdp_degree} ranks; "
            f"incompatible expert parameters: "
            f"{incompatible}"
        )
      # See the documented apply_simple_fsdp hook above.
      if use_trailing_w2:
        setattr(module, "_fsdp_shard_dim", shard_dims)
        module._use_transposed_w2_rhs = True
      else:
        setattr(module, "_fsdp_shard_dim", 1)
      hidden_axis_modules.append(module_fqn)

  if hidden_axis_modules:
    logger.info(
        "Using non-leading SimpleFSDP dimensions for %d uneven Qwen modules",
        len(hidden_axis_modules),
    )


__all__ = ["configure_qwen_simple_fsdp_shard_dims"]
