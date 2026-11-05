# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recognize collectives and their FSDP roles in TPU GraphTrainer FX graphs.

Collective classification comes from TorchInductor's bucketing helpers, plus
TorchTPU's dim-d all-gather and reduce-scatter, single ops with no wait node.
Graph transformations that move or group collectives share these helpers.
"""

import operator
from typing import Literal

import torch
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor,
    is_all_reduce_tensor,
    is_all_to_all_tensor,
    is_reduce_scatter_tensor,
    is_wait_tensor,
)

from torchtitan.experiments.graph_trainer.cpu_offload import _is_view

FsdpRole = Literal["ag", "rs"]


def is_getitem(node: torch.fx.Node) -> bool:
  return node.op == "call_function" and (
      node.target == operator.getitem
      or getattr(node.target, "__name__", "") == "getitem"
  )


def op_name(node: torch.fx.Node) -> str | None:
  if node.op != "call_function" or not isinstance(
      node.target, torch._ops.OpOverload
  ):
    return None
  return node.target.name()


def is_gather(node: torch.fx.Node) -> bool:
  return is_all_gather_into_tensor(node) or op_name(node) == "tpu::_all_gather_dim"


def is_reduce_scatter(node: torch.fx.Node) -> bool:
  return (
      is_reduce_scatter_tensor(node)
      or op_name(node) == "tpu::_reduce_scatter_dim"
  )


def is_collective(node: torch.fx.Node) -> bool:
  return (
      is_gather(node)
      or is_reduce_scatter(node)
      or is_all_reduce_tensor(node)
      or is_all_to_all_tensor(node)
  )


def collective_producer(node: torch.fx.Node) -> torch.fx.Node | None:
  """The collective whose result ``node`` (a getitem or wait) reads, if any."""
  if not node.args or not isinstance(node.args[0], torch.fx.Node):
    return None
  producer = node.args[0]
  while is_getitem(producer):
    if not producer.args or not isinstance(producer.args[0], torch.fx.Node):
      return None
    producer = producer.args[0]
  if is_collective(producer):
    return producer
  return None


def _is_movement(node: torch.fx.Node) -> bool:
  """A one-input view or copy (incl. dtype casts): no compute to schedule."""
  return (
      node.op == "call_function"
      and len(node.all_input_nodes) == 1
      and (
          _is_view(node)
          or node.target
          in (torch.ops.aten._to_copy.default, torch.ops.aten.clone.default)
      )
  )


def collect_collective_chains(
    gm: torch.fx.GraphModule,
) -> dict[torch.fx.Node, set[torch.fx.Node]]:
  """Collect indivisible collective and FSDP dtype-conversion chains.

  For each all-gather launch, repeatedly follow tensor inputs through upstream
  ``call_function`` nodes toward its parameter placeholder. Retain only nodes
  used exclusively by that launch chain. This keeps layout preparation with its
  all-gather without maintaining an op allowlist or absorbing shared compute.
  A BF16 FSDP reduction has this shape (the first cast is omitted when the
  computed gradient is already BF16):

    compute partition                 RS partition
    dWeight -> [to(BF16)]  |  -> RS(BF16) -> wait -> to(FP32) -> grad output

  The post-RS cast retains its module-layer metadata. If it is treated as layer
  compute, that metadata assigns it to the earlier compute partition even
  though its input is produced by the later RS partition. The FX graph is
  acyclic, but the resulting executable graph is not:

    compute -> RS -> compute

  Keep the casts and views after a reduction with the launch/wait chain to
  preserve a single forward dependency between the two executables. Before a
  reduction, the chain also takes the views and widening casts of the computed
  gradient: a view must not cross an executable boundary (TorchTPU
  materializes a strided input with an eager copy), and the narrower tensor is
  cheaper to carry. A narrowing cast, like ``to(BF16)`` above, stays with the
  compute.
  """
  chains = {node: {node} for node in gm.graph.nodes if is_collective(node)}
  for node in gm.graph.nodes:
    if is_getitem(node) or is_wait_tensor(node):
      producer = collective_producer(node)
      if producer is not None:
        chains[producer].add(node)
  for producer, chain in chains.items():
    if not (is_reduce_scatter(producer) or is_all_reduce_tensor(producer)):
      continue
    pending = list(chain)
    while pending:
      for user in pending.pop().users:
        if user not in chain and _is_movement(user):
          chain.add(user)
          pending.append(user)
    node = producer.args[0]
    while (
        _is_movement(node)
        and set(node.users) <= chain
        and node.meta["val"].element_size()
        >= node.all_input_nodes[0].meta["val"].element_size()
    ):
      chain.add(node)
      node = node.all_input_nodes[0]
  for producer, chain in chains.items():
    if not is_gather(producer):
      continue
    candidates: set[torch.fx.Node] = set()
    pending = list(producer.all_input_nodes)
    while pending:
      node = pending.pop()
      if (
          node in chain
          or node in candidates
          or node.op != "call_function"
          or is_collective(node)
          or is_wait_tensor(node)
      ):
        continue
      candidates.add(node)
      pending.extend(node.all_input_nodes)
    while candidates:
      shared = {
          node
          for node in candidates
          if any(
              user not in candidates and user not in chain
              for user in node.users
          )
      }
      if not shared:
        break
      candidates.difference_update(shared)
    chain.update(candidates)
  return chains


def chain_boundary(
    chain: set[torch.fx.Node],
) -> tuple[set[torch.fx.Node], set[torch.fx.Node]]:
  """The nodes outside ``chain`` that it reads and that read it."""
  inputs = {n for c in chain for n in c.all_input_nodes if n not in chain}
  users = {n for c in chain for n in c.users if n not in chain}
  return inputs, users


def classify_fsdp_roles(
    collective_chains: dict[torch.fx.Node, set[torch.fx.Node]],
) -> dict[torch.fx.Node, FsdpRole | None]:
  """Classify each collective chain by its role in a backward graph.

  Chains are classified by their role, whatever op implements them: one fed
  only by graph inputs is a parameter gather (``"ag"``); one read only by
  gradient outputs, directly or through another such chain, is a gradient
  reduction (``"rs"``). Other collectives, such as MoE token exchange or
  tensor/context-parallel activation collectives, depend on compute and map to
  ``None``.
  """
  member = {
      node: launch
      for launch, chain in collective_chains.items()
      for node in chain
  }
  roles: dict[torch.fx.Node, FsdpRole | None] = {}
  for launch in reversed(list(collective_chains)):
    inputs, users = chain_boundary(collective_chains[launch])
    if all(n.op in ("placeholder", "get_attr") for n in inputs):
      roles[launch] = "ag"
    elif users and all(
        n.op == "output" or roles.get(member.get(n)) == "rs" for n in users
    ):
      roles[launch] = "rs"
    else:
      roles[launch] = None
  return roles


__all__ = [
    "FsdpRole",
    "chain_boundary",
    "classify_fsdp_roles",
    "collect_collective_chains",
    "collective_producer",
    "is_collective",
    "is_gather",
    "is_getitem",
    "is_reduce_scatter",
    "op_name",
]
