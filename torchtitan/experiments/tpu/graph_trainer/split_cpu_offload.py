# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-offload support for independently compiled TPU graph children."""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HostDonorSpec:
  shape: tuple[int, ...]
  stride: tuple[int, ...]
  dtype: torch.dtype


def initialize_pinned_host_buffer(source: torch.Tensor) -> torch.Tensor:
  """Create one pinned-host tensor through the normal AO D2H operation."""
  host = torch.ops.ao.offload.default(source)
  return torch.ops.ao.wait_tensor.default(host, source, source)


def is_ao_transfer(
    node: torch.fx.Node,
    op_name: str | None = None,
) -> bool:
  """Return whether ``node`` is an activation-offload placement op."""
  if not (
      node.op == "call_function"
      and getattr(node.target, "namespace", None) == "ao"
  ):
    return False
  return op_name is None or str(node.target).startswith(f"ao.{op_name}.")


def _first_downstream_partition(
    node: torch.fx.Node,
    partition_map: dict[torch.fx.Node, int],
) -> int:
  pending = deque(node.users)
  visited: set[torch.fx.Node] = set()
  partitions = []
  while pending:
    user = pending.popleft()
    if user in visited:
      continue
    visited.add(user)
    if not is_ao_transfer(user) and user in partition_map:
      partitions.append(partition_map[user])
      continue
    pending.extend(user.users)
  return min(partitions, default=-1)


def _preceding_partition(
    partition: int,
    partition_map: dict[torch.fx.Node, int],
    distance: int,
) -> int:
  partitions = sorted(set(partition_map.values()))
  index = partitions.index(partition)
  return partitions[max(0, index - distance)]


def _ao_transfer_partition(
    producer: torch.fx.Node,
    wait: torch.fx.Node,
    partition_map: dict[torch.fx.Node, int],
    backward_prefetch_executables: int,
) -> int:
  """Choose one executable for an asynchronous AO copy/completion pair.

  Forward D2H belongs to the partition containing its final device-storage
  consumer. Backward H2D starts the configured number of complete child
  executables before the partition containing its first consumer. In both
  cases only the completed tensor may cross a child boundary; XLA's async
  start/done state remains local to the executable that created it.
  """
  if is_ao_transfer(producer, "offload"):
    if len(wait.args) > 2 and isinstance(wait.args[2], torch.fx.Node):
      partition = partition_map.get(wait.args[2], -1)
      if partition >= 0:
        return partition
    return _first_downstream_partition(wait, partition_map)

  consumer_partition = _first_downstream_partition(wait, partition_map)
  if consumer_partition < 0:
    return consumer_partition
  return _preceding_partition(
      consumer_partition,
      partition_map,
      backward_prefetch_executables,
  )


def place_ao_transfer_pairs(
    gm: torch.fx.GraphModule,
    partition_map: dict[torch.fx.Node, int],
    *,
    backward_prefetch_executables: int = 0,
) -> None:
  """Place ``(offload, wait)`` and ``(reload, wait)`` pairs in child executables.

  Each pair is assigned atomically so an executable-local async descriptor
  never crosses a child boundary. A forward D2H pair belongs to the source
  storage's final-consumer child. When its inputs are already available, its
  start moves before that child's ordinary nodes to maximize overlap; the wait
  remains ordered after the final storage use. A backward H2D pair belongs to
  the consumer child or ``backward_prefetch_executables`` complete children
  earlier: zero means the consumer child and one its immediate predecessor.
  """
  if backward_prefetch_executables < 0:
    raise ValueError("backward_prefetch_executables must be nonnegative")

  producers = [
      node
      for node in gm.graph.nodes
      if is_ao_transfer(node, "offload") or is_ao_transfer(node, "reload")
  ]
  paired_waits: set[torch.fx.Node] = set()
  for producer in producers:
    waits = [
        user
        for user in producer.users
        if is_ao_transfer(user, "wait_tensor")
        and user.args
        and user.args[0] is producer
    ]
    if len(waits) != 1:
      raise ValueError(
          f"AO transfer {producer.name!r} must have exactly one completion; "
          f"found {len(waits)}"
      )
    wait = waits[0]
    if wait in paired_waits:
      raise ValueError(f"AO completion {wait.name!r} has multiple producers")
    partition = _ao_transfer_partition(
        producer,
        wait,
        partition_map,
        backward_prefetch_executables,
    )
    if partition < 0:
      raise ValueError(
          f"Cannot assign AO transfer pair {producer.name!r}/{wait.name!r} "
          "to a split partition"
      )
    partition_map[producer] = partition
    partition_map[wait] = partition
    paired_waits.add(wait)

  for node in gm.graph.nodes:
    if not is_ao_transfer(node, "wait_tensor"):
      continue
    if node not in paired_waits:
      raise ValueError(
          f"AO completion {node.name!r} has no matching transfer producer"
      )

  nodes = list(gm.graph.nodes)
  node_order = {node: index for index, node in enumerate(nodes)}
  first_node_by_partition: dict[int, torch.fx.Node] = {}
  for node in nodes:
    if node.op in ("placeholder", "output") or is_ao_transfer(node):
      continue
    partition = partition_map.get(node)
    if partition is not None and partition not in first_node_by_partition:
      first_node_by_partition[partition] = node

  moved = False
  for node in nodes:
    if not is_ao_transfer(node, "offload"):
      continue
    anchor = first_node_by_partition.get(partition_map[node])
    if anchor is None or node_order[node] >= node_order[anchor]:
      continue
    if any(
        input_node.op != "placeholder"
        and node_order[input_node] >= node_order[anchor]
        for input_node in node.all_input_nodes
    ):
      continue
    anchor.prepend(node)
    moved = True

  if moved:
    gm.graph.lint()
    gm.recompile()


def _as_tuple(value: Any) -> tuple[Any, ...]:
  if isinstance(value, tuple):
    return value
  if isinstance(value, list):
    return tuple(value)
  return (value,)


def thread_host_offload_donors(
    forward: torch.fx.GraphModule,
) -> tuple[
    tuple[HostDonorSpec, ...],
    tuple[int, ...],
    tuple[torch.Tensor, ...],
]:
  """Add graph inputs for persistent host-offload buffers.

  Process the complete forward graph before optional layer partitioning to
  identify D2H completion nodes inserted by the offload pass and add one
  same-shaped host-buffer placeholder for each. If the graph is subsequently
  split, ``split_module`` automatically wires those inputs to the owning child
  modules from this dataflow.

  The placeholders represent cross-step state, so the returned specs and output
  positions let ``SplitForwardBackwardModule`` initialize the host buffers once
  and feed each step's completed D2H outputs back as the next step's donors.
  """
  output = next(
      node for node in forward.graph.nodes if node.op == "output"
  )
  outputs = _as_tuple(output.args[0])
  first_op = next(
      node for node in forward.graph.nodes if node.op != "placeholder"
  )
  specs = []
  output_positions = []
  examples = []
  total_bytes = 0
  wired_outputs: set[torch.fx.Node] = set()

  for output_position, offload_wait in enumerate(outputs):
    if not isinstance(offload_wait, torch.fx.Node):
      continue
    if not (
        is_ao_transfer(offload_wait, "wait_tensor")
        and not _is_backward_node(offload_wait)
        and offload_wait not in wired_outputs
    ):
      continue
    value_meta = offload_wait.meta.get("val")
    if not (
        isinstance(value_meta, torch.Tensor)
        and value_meta.device.type == "cpu"
    ):
      continue

    donor_index = len(specs)
    with forward.graph.inserting_before(first_op):
      donor = forward.graph.placeholder(f"host_donor_{donor_index}")
    donor.meta = dict(offload_wait.meta)
    donor.meta["tpu_force_donate"] = True
    wait_args = list(offload_wait.args)
    while len(wait_args) < 3:
      wait_args.append(None)
    # TPU-only dataflow hook: TorchTPU ignores wait_tensor's keepalive value,
    # while split_module uses this edge to thread the donated host input into
    # the child. PyTorch's non-TPU implementation instead frees keepalive.
    wait_args[1] = donor
    offload_wait.args = tuple(wait_args)

    specs.append(
        HostDonorSpec(
            shape=tuple(int(dim) for dim in value_meta.shape),
            stride=tuple(int(dim) for dim in value_meta.stride()),
            dtype=value_meta.dtype,
        )
    )
    output_positions.append(output_position)
    examples.append(value_meta)
    total_bytes += value_meta.numel() * value_meta.element_size()
    wired_outputs.add(offload_wait)

  forward.graph.lint()
  forward.recompile()
  logger.info(
      "Threaded %d pinned-host outputs as recurrent donated inputs (%.2f GiB)",
      len(specs),
      total_bytes / (1024**3),
  )
  return tuple(specs), tuple(output_positions), tuple(examples)
