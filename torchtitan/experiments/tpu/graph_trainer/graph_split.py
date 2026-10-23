# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Split a GraphTrainer joint FX graph into independently compiled modules.

``compile.graph_split.enabled`` divides the mandatory joint graph into forward
and backward GraphModules connected by explicit residual tensors. The forward
graph can then be divided into sequential layer ranges. The backward graph can remain
whole, group adjacent layer computations with their FSDP collectives, or form
staggered all-gather/compute/reduce-scatter pipeline stages. These subdivisions
are therefore always nested under the forward/backward split.

Each rematerialization region is one transformer layer. Combining regions in
one executable changes only the compiler scheduling scope; it preserves the
per-layer activation-checkpoint boundaries and just-in-time recompute order.
Multi-layer rematerialization instead trades fewer saved or offloaded boundaries
for longer-lived intermediates or repeated recompute and is outside this pass.

The result wrapper retains both the executable GraphModule and the node-to-
partition metadata used to validate that layers and collective chains remain
indivisible.
"""

import operator
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor,
    is_all_reduce_tensor,
    is_all_to_all_tensor,
    is_reduce_scatter_tensor,
    is_wait_tensor,
)
from torch.fx.passes.split_module import split_module

from torchtitan.experiments.graph_trainer.cpu_offload import _is_view
from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    _collect_bucketed_fsdp_comms,
)
from torchtitan.experiments.graph_trainer.graph_pp.partition import (
    _forward_mutations_to_materialize,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    is_mutation_node,
)
from torchtitan.experiments.tpu.graph_trainer.passes.expert_all_gather_deadline import (
    apply_expert_all_gather_deadline,
)
from torchtitan.experiments.tpu.graph_trainer.split_cpu_offload import (
  HostDonorSpec,
  initialize_pinned_host_buffer,
  is_ao_transfer,
  place_ao_transfer_pairs,
  thread_host_offload_donors,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
    tpu_backend_compile_partitioned_pass,
    tpu_backend_compile_pass,
)
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
  from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
  from torchtitan.experiments.tpu.graph_trainer.config import (
      GraphSplitterConfig,
  )
  from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
      TpuGraphTrainer,
  )


def _synchronize_tpu(values: list[torch.Tensor], *, wait: bool) -> None:
  """Synchronize lazily so host-only graph-split tests need no TorchTPU."""
  from torch_tpu._internal.sync import synchronize  # pytype: disable=import-error

  synchronize(values, wait=wait)


def _enqueue_forward_graph_outputs(values: list[torch.Tensor]) -> None:
  values = [value for value in values if value.device.type == "tpu"]
  if values:
    _synchronize_tpu(values, wait=False)


def _as_tuple(value: Any) -> tuple[Any, ...]:
  if isinstance(value, tuple):
    return value
  if isinstance(value, list):
    return tuple(value)
  return (value,)


class SplitForwardBackwardModule(nn.Module):

  def __init__(
      self,
      forward_graph: torch.fx.GraphModule,
      backward_graph: torch.fx.GraphModule,
      *,
      forward_output_positions: tuple[int, ...],
      backward_output_positions: tuple[int, ...],
      constant_outputs: dict[int, Any],
      num_direct_forward_outputs: int,
      num_residual_outputs: int,
      num_outputs: int,
      host_donor_specs: tuple[HostDonorSpec, ...] = (),
      host_donor_output_positions: tuple[int, ...] = (),
  ) -> None:
    super().__init__()
    self.forward_graph = forward_graph
    self.backward_graph = backward_graph
    self.forward_output_positions = forward_output_positions
    self.backward_output_positions = backward_output_positions
    self.constant_outputs = constant_outputs
    self.num_direct_forward_outputs = num_direct_forward_outputs
    self.num_residual_outputs = num_residual_outputs
    self.num_outputs = num_outputs
    self.host_donor_specs: tuple[HostDonorSpec, ...] = host_donor_specs
    self.host_donor_output_positions: tuple[int, ...] = (
        host_donor_output_positions
    )
    self._host_donors: tuple[torch.Tensor, ...] | None = None

  def _initialize_host_donors(
      self,
      flat_program_inputs: tuple[Any, ...],
  ) -> tuple[torch.Tensor, ...]:
    """Create one persistent pinned-host tensor per offloaded activation.

    ``host_donor_specs`` describes the slots; ``flat_program_inputs`` is scanned
    only to select their TPU device. An ordinary pinned CPU tensor is not a TPU
    tensor backed by PJRT pinned-host memory. TorchTPU does not currently expose
    a direct allocator for the latter, so each slot is created through AO D2H.
    Materializing one at a time bounds temporary device storage to one
    activation.
    """
    children = tuple(self.forward_graph.children())
    directly_compiled = hasattr(self.forward_graph, "_tpu_executable")
    children_compiled = bool(children) and all(
        any(hasattr(module, "_tpu_executable") for module in child.modules())
        for child in children
    )
    if not directly_compiled and not children_compiled:
      raise RuntimeError(
          "Host-donor threading requires a compiled TPU forward program"
      )
    device = next(
        (
            value.device
            for value in flat_program_inputs
            if isinstance(value, torch.Tensor) and value.device.type == "tpu"
        ),
        None,
    )
    if device is None:
      raise RuntimeError("Cannot initialize host donors without a TPU input")
    donors = []
    for spec in self.host_donor_specs:
      source = torch.empty_strided(
          spec.shape,
          spec.stride,
          dtype=spec.dtype,
          device=device,
      )
      donor = initialize_pinned_host_buffer(source)
      # Bound temporary HBM usage while constructing the persistent host pool.
      _synchronize_tpu([donor], wait=True)
      donors.append(donor)
      del source
    logger.info(
        "Initialized %d persistent pinned-host activation donor buffers",
        len(donors),
    )
    return tuple(donors)

  def forward(self, *flat_inputs: Any) -> tuple[Any, ...]:
    if self.host_donor_specs and self._host_donors is None:
      self._host_donors = self._initialize_host_donors(flat_inputs)
    donor_inputs = self._host_donors or ()
    forward_outputs = _as_tuple(
        self.forward_graph(*flat_inputs, *donor_inputs)
    )
    direct_forward_outputs = forward_outputs[
        : self.num_direct_forward_outputs
    ]
    residual_end = (
        self.num_direct_forward_outputs + self.num_residual_outputs
    )
    residuals = forward_outputs[
        self.num_direct_forward_outputs : residual_end
    ]
    # A compiled child may return lazy outputs without enqueueing its executable.
    # Residual dependencies cover only the forward children that produce those
    # residuals; direct outputs such as loss may come from other children. Enqueue
    # every tensor output before invoking backward so all forward children precede
    # backward collectives. wait=False preserves asynchronous host execution.
    _enqueue_forward_graph_outputs(list(forward_outputs))
    backward_outputs = _as_tuple(
        self.backward_graph(*flat_inputs, *residuals)
    )
    if self.host_donor_output_positions:
      self._host_donors = tuple(
          forward_outputs[position]
          for position in self.host_donor_output_positions
      )

    outputs = [None] * self.num_outputs
    positioned_outputs = chain(
        zip(
            self.forward_output_positions,
            direct_forward_outputs,
            strict=True,
        ),
        zip(
            self.backward_output_positions,
            backward_outputs,
            strict=True,
        ),
        self.constant_outputs.items(),
    )
    for position, value in positioned_outputs:
      outputs[position] = value
    return tuple(outputs)


@dataclass
class SplitForwardBackwardGraph:
  module: SplitForwardBackwardModule
  forward_graph: torch.fx.GraphModule
  backward_graph: torch.fx.GraphModule
  forward_example_inputs: tuple[Any, ...]
  backward_example_inputs: tuple[Any, ...]
  residual_nodes: tuple[torch.fx.Node, ...]


@dataclass
class PartitionedGraph:
  module: torch.fx.GraphModule
  partition_map: dict[torch.fx.Node, int]
  layer_to_partition: dict[int, int]
  collective_layer_to_partition: dict[str, dict[int, int]]


@dataclass(frozen=True)
class GraphSplitter:
  """Validate, partition, and compile one GraphTrainer trace."""

  enabled: bool
  config: "GraphSplitterConfig"
  n_layers: int

  @classmethod
  def from_trainer_config(
      cls,
      config: "TpuGraphTrainer.Config",
      *,
      optimizer_in_graph: bool,
  ) -> "GraphSplitter":
    split_config = config.compile.graph_split
    enabled = split_config.enabled
    if split_config.forward_layers_per_executable == 0 or (
        split_config.forward_layers_per_executable < -1
    ):
      raise ValueError(
          "forward_layers_per_executable must be -1 or positive"
      )
    if split_config.backward_partition_mode not in (
        "none",
        "by_layer",
        "fsdp_pipeline",
    ):
      raise ValueError(
          "backward_partition_mode must be none, by_layer, or fsdp_pipeline"
      )
    if split_config.backward_layers_per_executable < 1:
      raise ValueError(
          "backward_layers_per_executable must be at least 1"
      )
    if split_config.backward_partition_mode != "none" and not enabled:
      raise ValueError(
          "backward_partition_mode requires compile.graph_split.enabled"
      )
    if split_config.cpu_offload_prefetch_executables < 0:
      raise ValueError(
          "cpu_offload_prefetch_executables must be nonnegative"
      )
    if split_config.cpu_offload_prefetch_executables and (
        not enabled or split_config.backward_partition_mode == "none"
    ):
      raise ValueError(
          "cpu_offload_prefetch_executables requires a split backward graph"
      )
    if enabled and config.compile.fsdp_barrier_policy != "none":
      raise ValueError(
          "Graph splitting does not currently compose with "
          "compile.fsdp_barrier_policy because the whole-graph collective "
          "placement pass is not partition-boundary aware"
      )
    # Optimizer-in-graph needs a third explicit phase rather than treating its
    # state reads, mutations, and parameter updates as backward.
    if enabled and optimizer_in_graph:
      raise ValueError(
          "TPU GraphTrainer graph split does not support "
          "optimizer_compile='in_graph'"
      )
    # This artifact format stores one transformed FX graph. TorchTPU's XLA and
    # Tier-2 executable caches remain supported for every compiled child.
    if enabled and config.compile.precompile_artifact_dir:
      raise ValueError(
          "TPU GraphTrainer graph split does not support serialized "
          "precompiled FX artifacts"
      )
    return cls(
        enabled=enabled,
        config=split_config,
        n_layers=len(config.model_spec.model.layers) if enabled else 0,
    )

  def split_and_compile(self, traced: "TracedResult") -> torch.fx.GraphModule:
    """Split one transformed joint graph and compile its child executables."""
    if not self.enabled:
      raise RuntimeError("GraphSplitter is disabled")

    if self.config.expert_all_gather_deadline:
      apply_expert_all_gather_deadline(traced.gm)
    split = split_forward_backward_graph(traced.gm, traced.example_inputs)

    # Host-slot recurrence belongs to the top-level forward signature, whether
    # that program is compiled whole or subdivided into layer children.
    donor_specs, donor_output_positions, donor_examples = (
        thread_host_offload_donors(split.forward_graph)
    )
    split.module.host_donor_specs = donor_specs
    split.module.host_donor_output_positions = donor_output_positions
    forward_example_inputs = (*split.forward_example_inputs, *donor_examples)
    donor_input_positions = tuple(
        range(len(split.forward_example_inputs), len(forward_example_inputs))
    )

    if self.config.forward_layers_per_executable > 0:
      forward = split_forward_graph_by_layer(
          split.forward_graph,
          self.config.forward_layers_per_executable,
          isolate_prefix=self.config.forward_isolate_prefix,
      )
      split.forward_graph = forward.module
      split.module.forward_graph = forward.module
      tpu_backend_compile_partitioned_pass(
          forward.module,
          forward_example_inputs,
          donate_internal_buffers=self.config.donate_internal_buffers,
      )
    else:
      tpu_backend_compile_pass(
          split.forward_graph,
          forward_example_inputs,
          donated_input_positions=donor_input_positions,
      )

    mode = self.config.backward_partition_mode
    if mode in ("by_layer", "fsdp_pipeline"):
      backward = split_backward_graph_by_layer(
          split.backward_graph,
          n_layers=self.n_layers,
          layers_per_executable=self.config.backward_layers_per_executable,
          pipeline_collectives=mode == "fsdp_pipeline",
          cpu_offload_prefetch_executables=(
              self.config.cpu_offload_prefetch_executables
          ),
      )
      split.backward_graph = backward.module
      split.module.backward_graph = backward.module
      tpu_backend_compile_partitioned_pass(
          backward.module,
          split.backward_example_inputs,
          is_fwd=False,
          donate_internal_buffers=self.config.donate_internal_buffers,
      )
    else:
      tpu_backend_compile_pass(
          split.backward_graph,
          split.backward_example_inputs,
          is_fwd=False,
      )
    return split.module


def _is_getitem(node: torch.fx.Node) -> bool:
  return node.op == "call_function" and (
      node.target == operator.getitem
      or getattr(node.target, "__name__", "") == "getitem"
  )


def _collective_producer(node: torch.fx.Node) -> torch.fx.Node | None:
  if not node.args or not isinstance(node.args[0], torch.fx.Node):
    return None
  producer = node.args[0]
  while _is_getitem(producer):
    if not producer.args or not isinstance(producer.args[0], torch.fx.Node):
      return None
    producer = producer.args[0]
  if _is_collective(producer):
    return producer
  return None


def _is_collective(node: torch.fx.Node) -> bool:
  return (
      is_all_gather_into_tensor(node)
      or is_reduce_scatter_tensor(node)
      or is_all_reduce_tensor(node)
      or is_all_to_all_tensor(node)
  )


def _is_spmd_region_marker(node: torch.fx.Node, marker: str) -> bool:
  return (
      node.op == "call_function"
      and str(getattr(node.target, "overloadpacket", node.target))
      == f"torch_tpu.{marker}"
  )


def _collect_collective_chains(
    gm: torch.fx.GraphModule,
) -> dict[torch.fx.Node, set[torch.fx.Node]]:
  """Collect indivisible collective and FSDP dtype-conversion chains.

  Collective classification comes from TorchInductor's bucketing helpers. For
  each all-gather launch, repeatedly follow tensor inputs through upstream
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

  Keep the post-RS cast with the launch/wait chain to preserve a single
  forward dependency between the two executables.
  """
  chains = {
      node: {node}
      for node in gm.graph.nodes
      if _is_collective(node)
  }
  for node in gm.graph.nodes:
    if _is_getitem(node) or is_wait_tensor(node):
      producer = _collective_producer(node)
      if producer is not None:
        chains[producer].add(node)
        if is_reduce_scatter_tensor(producer):
          chains[producer].update(
              user
              for user in node.users
              if user.op == "call_function"
              and user.target is torch.ops.aten._to_copy.default
              and len(user.all_input_nodes) == 1
          )
  for producer, chain in chains.items():
    if not is_all_gather_into_tensor(producer):
      continue
    candidates: set[torch.fx.Node] = set()
    pending = list(producer.all_input_nodes)
    while pending:
      node = pending.pop()
      if (
          node in chain
          or node in candidates
          or node.op != "call_function"
          or _is_collective(node)
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


def _body_nodes(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
  return [
      node
      for node in gm.graph.nodes
      if node.op not in ("placeholder", "output")
  ]


def _group_nodes_by_layer(
    nodes: list[torch.fx.Node],
    *,
    excluded: set[torch.fx.Node] | None = None,
) -> tuple[dict[torch.fx.Node, int], dict[int, set[torch.fx.Node]]]:
  excluded = excluded or set()
  node_layer_ids = {node: _get_layer_id(node) for node in nodes}
  layer_nodes: dict[int, set[torch.fx.Node]] = defaultdict(set)
  for node, layer_id in node_layer_ids.items():
    if layer_id >= 0 and node not in excluded:
      layer_nodes[layer_id].add(node)
  return node_layer_ids, dict(layer_nodes)


def _assign_partition(
    partition_map: dict[torch.fx.Node, int],
    nodes: set[torch.fx.Node],
    partition: int,
    owner: str,
) -> None:
  conflicts = {
      node.name: partition_map[node]
      for node in nodes
      if node in partition_map and partition_map[node] != partition
  }
  if conflicts:
    raise ValueError(f"{owner} has conflicting ownership: {conflicts}")
  partition_map.update(dict.fromkeys(nodes, partition))


def _order_child_boundaries(parent: torch.fx.GraphModule) -> None:
  """Order each child's inputs by last read and outputs by first availability.

  ``split_module`` numbers a child's inputs and outputs in discovery order, and
  only ``parent`` calls the child, so the order carries no meaning. It does
  matter to donation: XLA aliases donated inputs to equal-size outputs pairwise
  in that order, and an output cannot be written before its donor's last read.
  Sorting both sides lets the earliest-freed input back the earliest-ready
  output, e.g. a host-to-device reload that depends on nothing else.

  TODO: drop this ordering once TorchTPU can pass explicit input->output
  aliases to XLA; today it only marks donatable inputs (``jax.buffer_donor``)
  and XLA picks the pairs. To check the pairs XLA picked, dump HLO with
  ``XLA_FLAGS=--xla_dump_to=<dir>`` and read each module's
  ``input_output_alias``.
  """
  calls = parent.graph.find_nodes(op="call_module")
  # Each child is rewritten together with its one call site.
  assert len({call.target for call in calls}) == len(calls), (
      "a split child is called more than once"
  )
  for call in calls:
    child = parent.get_submodule(call.target)
    nodes = list(child.graph.nodes)
    position = {node: index for index, node in enumerate(nodes)}
    depth: dict[torch.fx.Node, int] = {}
    for node in nodes:
      depth[node] = 1 + max(map(depth.get, node.all_input_nodes), default=-1)

    inputs = [node for node in nodes if node.op == "placeholder"]
    assert all(node.op == "placeholder" for node in nodes[: len(inputs)])
    last_read = {
        node: max((position[user] for user in node.users), default=-1)
        for node in inputs
    }
    inputs_by_read = sorted(inputs, key=last_read.__getitem__)
    first_op = nodes[len(inputs)]
    for node in inputs_by_read:
      first_op.prepend(node)
    call.args = tuple(call.args[inputs.index(node)] for node in inputs_by_read)

    output = child.graph.output_node()
    values = output.args[0]
    if isinstance(values, (tuple, list)):
      order = sorted(
          range(len(values)),
          key=lambda i: (depth[values[i]], position[values[i]]),
      )
      output.args = (tuple(values[i] for i in order),)
      new_index = {old: new for new, old in enumerate(order)}
      assert all(_is_getitem(user) for user in call.users)
      for user in list(call.users):
        user.args = (call, new_index[user.args[1]])
    child.recompile()
  parent.recompile()


def _materialize_partitioned_graph(
    gm: torch.fx.GraphModule,
    partition_map: dict[torch.fx.Node, int],
    layer_nodes: dict[int, set[torch.fx.Node]],
    collective_layer_to_partition: dict[str, dict[int, int]] | None = None,
) -> PartitionedGraph:
  actual_partitions = sorted(set(partition_map.values()))
  normalized = {
      partition: index for index, partition in enumerate(actual_partitions)
  }
  partition_map = {
      node: normalized[partition] for node, partition in partition_map.items()
  }

  layer_to_partition = {}
  for layer_id, nodes in layer_nodes.items():
    partitions = {partition_map[node] for node in nodes}
    if len(partitions) != 1:
      raise ValueError(
          f"Layer {layer_id} spans partitions {sorted(partitions)}"
      )
    layer_to_partition[layer_id] = next(iter(partitions))

  collective_layer_to_partition = {
      kind: {
          layer_id: normalized[partition]
          for layer_id, partition in layer_partitions.items()
      }
      for kind, layer_partitions in (
          collective_layer_to_partition or {}
      ).items()
  }
  chunked_gm = split_module(gm, None, lambda node: partition_map[node])
  _order_child_boundaries(chunked_gm)
  chunked_gm.graph.lint()
  chunked_gm.recompile()
  return PartitionedGraph(
      module=chunked_gm,
      partition_map=partition_map,
      layer_to_partition=layer_to_partition,
      collective_layer_to_partition=collective_layer_to_partition,
  )


def _preserves_numel(view: torch.fx.Node) -> bool:
  source = view.args[0] if view.args else None
  if not isinstance(source, torch.fx.Node):
    return False
  out, src = view.meta.get("val"), source.meta.get("val")
  return (
      isinstance(out, torch.Tensor)
      and isinstance(src, torch.Tensor)
      and out.numel() == src.numel()
  )


def split_forward_graph_by_layer(
    gm: torch.fx.GraphModule,
    layers_per_executable: int,
    *,
    isolate_prefix: bool = False,
) -> PartitionedGraph:
  """Split forward into ascending, complete transformer-layer ranges."""
  if layers_per_executable < 1:
    raise ValueError(
        "layers_per_executable must be at least 1, got "
        f"{layers_per_executable}"
    )

  body_nodes = _body_nodes(gm)
  ao_nodes = {node for node in body_nodes if is_ao_transfer(node)}
  view_nodes = {node for node in body_nodes if _is_view(node)}
  node_layer_ids, layer_nodes = _group_nodes_by_layer(
      body_nodes, excluded=ao_nodes | view_nodes
  )
  layer_ids = sorted(layer_nodes)
  if not layer_ids:
    raise ValueError("Cannot split forward: no layer metadata was found")

  first_layer_index = next(
      index
      for index, node in enumerate(body_nodes)
      if node_layer_ids[node] >= 0
  )
  has_prefix = isolate_prefix and any(
      node.op == "call_function" and not is_ao_transfer(node)
      for node in body_nodes[:first_layer_index]
  )
  partition_offset = int(has_prefix)
  layer_to_target = {
      layer_id: partition_offset + index // layers_per_executable
      for index, layer_id in enumerate(layer_ids)
  }

  partition_map: dict[torch.fx.Node, int] = {}
  current_partition = 0
  target_partition = 0
  current_layer_id = -1
  collective_wait_counts = Counter(
      producer
      for node in body_nodes
      if is_wait_tensor(node)
      and (producer := _collective_producer(node)) is not None
  )
  pending_collectives: dict[torch.fx.Node, int] = {}
  spmd_safe_depth = 0
  for node in body_nodes:
    if node in ao_nodes:
      continue
    layer_id = node_layer_ids[node]
    starts_layer = layer_id >= 0 and layer_id != current_layer_id
    if starts_layer:
      current_layer_id = layer_id
      target_partition = max(target_partition, layer_to_target[layer_id])
      if not pending_collectives and spmd_safe_depth == 0:
        current_partition = target_partition

    collective_producer = _collective_producer(node)
    if is_wait_tensor(node):
      if collective_producer not in partition_map:
        raise ValueError(
            f"Collective producer for {node.name!r} is not partitioned"
        )
      partition_map[node] = partition_map[collective_producer]
      remaining_waits = pending_collectives[collective_producer] - 1
      if remaining_waits:
        pending_collectives[collective_producer] = remaining_waits
      else:
        del pending_collectives[collective_producer]
    elif _is_getitem(node) and collective_producer is not None:
      if collective_producer not in partition_map:
        raise ValueError(
            f"Collective producer for {node.name!r} is not partitioned"
        )
      partition_map[node] = partition_map[collective_producer]
    else:
      partition_map[node] = current_partition
      if _is_collective(node):
        pending_collectives[node] = max(1, collective_wait_counts[node])

    if _is_spmd_region_marker(node, "enter_spmd_safe_region"):
      spmd_safe_depth += 1
    elif _is_spmd_region_marker(node, "exit_spmd_safe_region"):
      spmd_safe_depth -= 1
      if spmd_safe_depth < 0:
        raise ValueError("Unbalanced TPU SPMD safe-region exit")

  if pending_collectives:
    raise ValueError(
        f"Forward graph has {len(pending_collectives)} unwaited collectives"
    )
  if spmd_safe_depth:
    raise ValueError("Unbalanced TPU SPMD safe-region entry")

  # A view left with its producer layer becomes a second executable output
  # beside its base, which XLA must materialize as a copy. Place each
  # size-preserving view with its earliest consumer so only the base crosses
  # the boundary. Slices stay put: moving one would pass its whole base.
  moved_views: Counter = Counter()
  for node in reversed(body_nodes):
    if node not in view_nodes or not _preserves_numel(node):
      continue
    user_partitions = [partition_map.get(user) for user in node.users]
    if user_partitions and None not in user_partitions:
      target = max(partition_map[node], min(user_partitions))
      if target != partition_map[node]:
        partition_map[node] = target
        moved_views[str(node.target)] += 1
  if moved_views:
    logger.info(
        "Placed %d forward views with their consumers: %s",
        sum(moved_views.values()),
        dict(moved_views),
    )

  place_ao_transfer_pairs(gm, partition_map)

  chunked = _materialize_partitioned_graph(
      gm, partition_map, layer_nodes
  )
  target_count = (
      len(layer_ids) + layers_per_executable - 1
  ) // layers_per_executable + partition_offset
  actual_count = len(set(chunked.partition_map.values()))
  if actual_count != target_count:
    logger.warning(
        "Targeted %d TPU forward executables, produced %d after preserving "
        "layer, collective, and SPMD-safe boundaries",
        target_count,
        actual_count,
    )
  logger.info(
      "Split TPU forward graph into %d executables (prefix=%s): layers=%s",
      actual_count,
      has_prefix,
      chunked.layer_to_partition,
  )
  return chunked


def _collect_backward_fsdp_owners(
    gm: torch.fx.GraphModule,
    *,
    n_layers: int,
    collective_chains: dict[torch.fx.Node, set[torch.fx.Node]],
) -> dict[torch.fx.Node, tuple[str, int]]:
  _, comms = _collect_bucketed_fsdp_comms(gm, n_layers=n_layers)
  owners = {
      comm.launch: (comm.kind, comm.logical_index)
      for comm in comms
      if comm.direction == "bwd"
      and comm.kind in ("ag", "rs")
      and comm.logical_index is not None
      and comm.logical_index < n_layers
  }
  missing_chains = [
      launch.name for launch in owners if launch not in collective_chains
  ]
  if missing_chains:
    raise ValueError(
        "FSDP backward collectives have incomplete chains: "
        f"{sorted(missing_chains)}"
    )
  return owners


def _validate_backward_partitions(
    partition_map: dict[torch.fx.Node, int],
    collective_chains: dict[torch.fx.Node, set[torch.fx.Node]],
) -> None:
  for producer, chain in collective_chains.items():
    partitions = {partition_map[node] for node in chain}
    if len(partitions) != 1:
      raise ValueError(
          f"Collective {producer.name!r} spans backward partitions "
          f"{sorted(partitions)}"
      )
  for node, partition in partition_map.items():
    for input_node in node.all_input_nodes:
      input_partition = partition_map.get(input_node)
      if input_partition is not None and input_partition > partition:
        raise ValueError(
            "Backward split has a reverse dependency: "
            f"{input_node.name} ({input_partition}) -> "
            f"{node.name} ({partition})"
        )


def split_backward_graph_by_layer(
    gm: torch.fx.GraphModule,
    *,
    n_layers: int,
    layers_per_executable: int = 1,
    pipeline_collectives: bool = False,
    cpu_offload_prefetch_executables: int = 0,
) -> PartitionedGraph:
  """Split backward by layer, optionally pipelining complete FSDP AG/RS pairs.

  The default keeps attributable FSDP collective chains with their layer and
  also supports backward graphs with no FSDP collectives. Pipelining requires
  both AG and RS ownership for every layer so it can safely form fill, overlap,
  and drain stages. CPU-offload reload pairs may be placed a configured number
  of complete child executables before their first consumer; zero keeps the
  pair in the consumer child.
  """
  if n_layers < 1:
    raise ValueError(f"n_layers must be positive, got {n_layers}")
  if layers_per_executable < 1:
    raise ValueError(
        "layers_per_executable must be at least 1, got "
        f"{layers_per_executable}"
    )

  body_nodes = _body_nodes(gm)
  ao_nodes = {node for node in body_nodes if is_ao_transfer(node)}
  collective_chains = _collect_collective_chains(gm)
  collective_nodes: set[torch.fx.Node] = set().union(
      *collective_chains.values()
  )
  # A recomputed view keeps its producer layer's FQN but is consumed by the
  # layer being differentiated; place views with their first consumer instead.
  view_nodes = {node for node in body_nodes if _is_view(node)}
  _, layer_nodes = _group_nodes_by_layer(
      body_nodes, excluded=collective_nodes | ao_nodes | view_nodes
  )
  expected_layers = set(range(n_layers))
  if set(layer_nodes) != expected_layers:
    missing = sorted(expected_layers - layer_nodes.keys())
    unexpected = sorted(layer_nodes.keys() - expected_layers)
    raise ValueError(
        "Backward split requires compute metadata for every layer; "
        f"missing layers {missing}, unexpected layers {unexpected}"
    )

  layer_ids = sorted(layer_nodes, reverse=True)
  layer_indices = {
      layer_id: index for index, layer_id in enumerate(layer_ids)
  }
  compute_offset = 2 if pipeline_collectives else 1
  raw_layer_to_partition = {
      layer_id: compute_offset + index // layers_per_executable
      for layer_id, index in layer_indices.items()
  }
  owners = _collect_backward_fsdp_owners(
      gm,
      n_layers=n_layers,
      collective_chains=collective_chains,
  )

  if pipeline_collectives:
    owned_collectives = set(owners.values())
    missing_comms = {}
    for layer_id in layer_ids:
      missing = tuple(
          kind
          for kind in ("ag", "rs")
          if (kind, layer_id) not in owned_collectives
      )
      if missing:
        missing_comms[layer_id] = missing
    if missing_comms:
      raise ValueError(
          "FSDP backward pipeline requires AG and RS ownership for every "
          f"layer; missing {missing_comms}"
      )

  def collective_partition(kind: str, layer_id: int) -> int:
    if not pipeline_collectives:
      return raw_layer_to_partition[layer_id]
    index = layer_indices[layer_id]
    if kind == "ag":
      return 1 if index == 0 else 2 + (index - 1) // layers_per_executable
    if index == len(layer_ids) - 1:
      return 3 + (len(layer_ids) - 1) // layers_per_executable
    return 2 + (index + 1) // layers_per_executable

  partition_map: dict[torch.fx.Node, int] = {}
  raw_collective_partitions: dict[str, dict[int, int]] = {
      "ag": {},
      "rs": {},
  }
  for launch, (kind, layer_id) in owners.items():
    partition = collective_partition(kind, layer_id)
    _assign_partition(
        partition_map,
        collective_chains[launch],
        partition,
        f"FSDP {kind} layer {layer_id}",
    )
    raw_collective_partitions[kind][layer_id] = partition

  for layer_id, nodes in layer_nodes.items():
    _assign_partition(
        partition_map,
        nodes,
        raw_layer_to_partition[layer_id],
        f"layer {layer_id}",
    )
  for node in reversed(body_nodes):
    if node in view_nodes and node not in partition_map:
      user_partitions = [
          partition_map[user] for user in node.users if user in partition_map
      ]
      if user_partitions:
        partition_map[node] = min(user_partitions)
  for node in body_nodes:
    if node not in partition_map and node not in ao_nodes:
      partition_map[node] = max(
          (
              partition_map[input_node]
              for input_node in node.all_input_nodes
              if input_node in partition_map
          ),
          default=0,
      )

  place_ao_transfer_pairs(
      gm,
      partition_map,
      backward_prefetch_executables=cpu_offload_prefetch_executables,
  )
  _validate_backward_partitions(partition_map, collective_chains)
  chunked = _materialize_partitioned_graph(
      gm,
      partition_map,
      layer_nodes,
      raw_collective_partitions,
  )
  logger.info(
      "Split TPU backward into %d executables with at most %d layers each "
      "(pipeline_collectives=%s): compute=%s collectives=%s",
      len(set(chunked.partition_map.values())),
      layers_per_executable,
      pipeline_collectives,
      chunked.layer_to_partition,
      chunked.collective_layer_to_partition,
  )
  return chunked


def split_forward_backward_graph(
    gm: torch.fx.GraphModule,
    example_inputs: tuple[Any, ...],
) -> SplitForwardBackwardGraph:
  """Partition a make_fx joint graph using ``autograd_backward`` metadata."""
  nodes = list(gm.graph.nodes)
  placeholders = [node for node in nodes if node.op == "placeholder"]
  if len(placeholders) != len(example_inputs):
    raise ValueError(
        "Graph input count does not match example inputs: "
        f"{len(placeholders)} != {len(example_inputs)}"
    )

  output_node = next(node for node in nodes if node.op == "output")
  original_outputs = _as_tuple(output_node.args[0])
  backward_node_set: set[torch.fx.Node] = set()
  for node in nodes:
    if node.op == "output":
      continue
    if _is_backward_node(node) or any(
        input_node in backward_node_set for input_node in node.all_input_nodes
    ):
      backward_node_set.add(node)
  backward_nodes = [node for node in nodes if node in backward_node_set]
  if not backward_nodes:
    raise ValueError("Cannot split a graph with no autograd_backward nodes")

  residual_nodes: list[torch.fx.Node] = []
  residual_set: set[torch.fx.Node] = set()
  backward_recomputed_nodes: set[torch.fx.Node] = set()
  missing = object()

  def make_available_to_backward(node: torch.fx.Node) -> None:
    if node in backward_node_set or node.op in ("placeholder", "get_attr"):
      return
    value = node.meta.get("val", missing)
    if value is missing:
      raise ValueError(
          f"Graph split boundary node {node.name!r} has no example value"
      )
    if isinstance(value, torch.Tensor):
      if node not in residual_set:
        residual_set.add(node)
        residual_nodes.append(node)
      return
    if node in backward_recomputed_nodes:
      return
    backward_recomputed_nodes.add(node)
    for input_node in node.all_input_nodes:
      make_available_to_backward(input_node)

  for node in backward_nodes:
    for input_node in node.all_input_nodes:
      make_available_to_backward(input_node)

  forward_output_positions: list[int] = []
  backward_output_positions: list[int] = []
  constant_outputs: dict[int, Any] = {}
  forward_output_nodes: list[torch.fx.Node] = []
  backward_output_nodes: list[torch.fx.Node] = []
  for position, output in enumerate(original_outputs):
    if not isinstance(output, torch.fx.Node):
      constant_outputs[position] = output
    elif (
        "val" in output.meta
        and not isinstance(output.meta["val"], torch.Tensor)
    ):
      constant_outputs[position] = output.meta["val"]
    elif output in backward_node_set:
      backward_output_positions.append(position)
      backward_output_nodes.append(output)
    else:
      forward_output_positions.append(position)
      forward_output_nodes.append(output)

  # Reuse GraphPP's forward-mutation analysis. ``mutation_saved_values`` are
  # forward-created tensor bases whose mutated contents backward must receive
  # as residuals. ``forward_mutation_outputs`` are otherwise-dead mutation
  # results exposed only to keep state updates (for example, MoE token-count
  # buffer writes) live in the forward graph; the wrapper ignores their values.
  mutation_saved_values, forward_mutation_outputs = (
      _forward_mutations_to_materialize(
          gm,
          fwd_outputs=forward_output_nodes,
          bwd_outputs=backward_output_nodes,
          backward_only_names=set(),
      )
  )
  for node in mutation_saved_values:
    make_available_to_backward(node)
  forward_mutation_outputs = [
      node
      for node in forward_mutation_outputs
      if node not in forward_output_nodes and node not in residual_set
  ]
  forward_roots = [
      *forward_output_nodes,
      *residual_nodes,
      *forward_mutation_outputs,
  ]
  forward_needed: set[torch.fx.Node] = set(placeholders)

  def add_forward_dependencies(node: torch.fx.Node) -> None:
    if node in forward_needed:
      return
    if node in backward_node_set:
      raise ValueError(
          f"Forward output {node.name!r} depends on a backward graph node"
      )
    forward_needed.add(node)
    for input_node in node.all_input_nodes:
      add_forward_dependencies(input_node)

  for root in forward_roots:
    add_forward_dependencies(root)

  discarded_forward_mutations = [
      node.name
      for node in nodes
      if node not in backward_node_set
      and node not in forward_needed
      and is_mutation_node(node)
  ]
  if discarded_forward_mutations:
    raise ValueError(
        "Graph split would discard forward mutation nodes: "
        f"{discarded_forward_mutations}"
    )

  forward_fx = torch.fx.Graph()
  forward_env: dict[torch.fx.Node, torch.fx.Node] = {}
  for node in nodes:
    if node.op == "output" or node not in forward_needed:
      continue
    forward_env[node] = forward_fx.node_copy(node, forward_env.__getitem__)
  forward_fx.output(
      tuple(forward_env[node] for node in forward_roots)
  )
  forward_gm = torch.fx.GraphModule(gm, forward_fx)
  forward_gm.graph.lint()
  forward_gm.recompile()

  backward_fx = torch.fx.Graph()
  backward_env: dict[torch.fx.Node, torch.fx.Node] = {}
  for node in placeholders:
    backward_env[node] = backward_fx.node_copy(node, lambda _: None)
  backward_get_attrs = {
      input_node
      for copied_node in [*backward_recomputed_nodes, *backward_nodes]
      for input_node in copied_node.all_input_nodes
      if input_node.op == "get_attr"
  }
  for node in nodes:
    if node in backward_get_attrs:
      backward_env[node] = backward_fx.node_copy(node, backward_env.__getitem__)
  for residual in residual_nodes:
    placeholder = backward_fx.placeholder(f"residual_{residual.name}")
    placeholder.meta = dict(residual.meta)
    backward_env[residual] = placeholder
  for node in nodes:
    if node in backward_recomputed_nodes or node in backward_node_set:
      backward_env[node] = backward_fx.node_copy(node, backward_env.__getitem__)
  backward_fx.output(tuple(backward_env[node] for node in backward_output_nodes))
  backward_gm = torch.fx.GraphModule(gm, backward_fx)
  backward_gm.graph.lint()
  backward_gm.recompile()

  residual_examples = tuple(node.meta["val"] for node in residual_nodes)
  logger.info(
      "Split TPU graph into %d forward and %d backward nodes with %d tensor "
      "residuals and %d side-effect outputs",
      len(list(forward_gm.graph.nodes)),
      len(list(backward_gm.graph.nodes)),
      len(residual_nodes),
      len(forward_mutation_outputs),
  )
  module = SplitForwardBackwardModule(
      forward_gm,
      backward_gm,
      forward_output_positions=tuple(forward_output_positions),
      backward_output_positions=tuple(backward_output_positions),
      constant_outputs=constant_outputs,
      num_direct_forward_outputs=len(forward_output_nodes),
      num_residual_outputs=len(residual_nodes),
      num_outputs=len(original_outputs),
  )
  return SplitForwardBackwardGraph(
      module=module,
      forward_graph=forward_gm,
      backward_graph=backward_gm,
      forward_example_inputs=example_inputs,
      backward_example_inputs=(*example_inputs, *residual_examples),
      residual_nodes=tuple(residual_nodes),
  )


__all__ = [
    "GraphSplitter",
    "PartitionedGraph",
    "SplitForwardBackwardGraph",
    "SplitForwardBackwardModule",
    "split_backward_graph_by_layer",
    "split_forward_backward_graph",
    "split_forward_graph_by_layer",
]
