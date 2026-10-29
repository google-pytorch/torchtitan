from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch import nn
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    annotate_module_fqns,
)
from torchtitan.experiments.graph_trainer.cpu_offload import (
    _is_view,
    apply_cpu_offload_pass,
    tag_all_offloadable_activations,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _full_memory_policy_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.experiments.tpu.graph_trainer import graph_split as graph_split_lib
from torchtitan.experiments.tpu.graph_trainer.graph_split import (
    split_backward_graph_by_layer,
    split_forward_graph_by_layer,
    split_forward_backward_graph,
)
from torchtitan.experiments.tpu.graph_trainer.split_cpu_offload import (
    thread_host_offload_donors,
)


class _ToyTransformer(nn.Module):

  def __init__(self, n_layers: int) -> None:
    super().__init__()
    self.layers = nn.ModuleList(
        nn.Linear(8, 8, bias=False) for _ in range(n_layers)
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    inputs = torch.sin(inputs)
    for layer in self.layers:
      inputs = torch.relu(layer(inputs))
    return inputs


class _BufferMutatingToyTransformer(_ToyTransformer):

  def __init__(self, n_layers: int) -> None:
    super().__init__(n_layers)
    self.register_buffer("counters", torch.zeros(n_layers))

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    for layer_id, layer in enumerate(self.layers):
      with torch.no_grad():
        self.counters[layer_id].add_(inputs.detach().sum())
      inputs = torch.relu(layer(inputs))
    return inputs


class _OffloadToyTransformer(nn.Module):

  def __init__(self, n_layers: int) -> None:
    super().__init__()
    self.layers = nn.ModuleList(
        nn.Sequential(nn.Linear(8, 8, bias=False), nn.ReLU())
        for _ in range(n_layers)
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      inputs = layer(inputs)
    return inputs


class _ViewOutputLayer(nn.Module):
  """Ends in a view, like the TPU [B, L] block wrapper's ``.view(T, D)``."""

  def __init__(self) -> None:
    super().__init__()
    self.linear = nn.Linear(8, 8, bias=False)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    return torch.relu(self.linear(inputs.view(4, -1, 8))).view(-1, 8)


class GraphSplitTest(absltest.TestCase):

  def _build_toy_model(self, n_layers: int) -> _ToyTransformer:
    """Build the shared prefix-plus-transformer-layer test model."""
    model = _ToyTransformer(n_layers)
    annotate_module_fqns(model)
    return model

  def _trace_toy_model(
      self,
      *,
      model: nn.Module,
      inputs: torch.Tensor,
      gradient_epilogue: bool = False,
      use_shape_scalar: bool = False,
  ):
    """Trace loss and explicit gradients for the shared test model."""

    def step(inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
      loss = model(inputs).float().sum()
      gradients = torch.autograd.grad(
          loss, (*tuple(model.parameters()), inputs)
      )
      if gradient_epilogue:
        gradients = tuple(
            torch.ops.aten.alias.default(gradient) for gradient in gradients
        )
      if use_shape_scalar:
        gradients = tuple(gradient * inputs.shape[0] for gradient in gradients)
      return (loss, *gradients)

    return minimal_fx_tracer(step, module=model)(inputs)

  def _apply_passes_and_split(
      self,
      traced,
      *,
      fsdp_reshard_after_forward: str,
  ):
    """Apply full remat, then split the real joint trace at fwd/bwd."""
    _full_memory_policy_pass(
        traced.gm,
        config=SimpleNamespace(
            compile=SimpleNamespace(full_recompute_save_ops=""),
            parallelism=SimpleNamespace(
                fsdp_reshard_after_forward=fsdp_reshard_after_forward,
                pipeline_parallel_degree=1,
            ),
        ),
    )
    selective_activation_remat_pass(traced.gm)
    return split_forward_backward_graph(traced.gm, traced.example_inputs)

  def _trace_simple_fsdp(
      self,
      *,
      n_layers: int,
      reduce_dtype: torch.dtype = torch.float32,
      fsdp_reshard_after_forward: str = "always",
  ):
    """Trace the shared model through real SimpleFSDP with fake collectives."""

    self.assertFalse(dist.is_initialized())
    dist.init_process_group(
        "fake", store=FakeStore(), rank=0, world_size=2
    )
    try:
      mesh = DeviceMesh("cpu", [0, 1])
      with torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True):
        model = self._build_toy_model(n_layers)
        model = data_parallel(
            model,
            device_mesh=mesh,
            mode="fully_shard",
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=reduce_dtype,
            ),
        )
        inputs = torch.randn(
            2, 8, dtype=torch.bfloat16, requires_grad=True
        )

        traced = self._trace_toy_model(model=model, inputs=inputs)
    finally:
      dist.destroy_process_group()

    return self._apply_passes_and_split(
        traced,
        fsdp_reshard_after_forward=fsdp_reshard_after_forward,
    )

  def _assert_collective_chains_are_atomic(
      self,
      gm: torch.fx.GraphModule,
      partition_map: dict[torch.fx.Node, int],
  ) -> None:
    chains = graph_split_lib._collect_collective_chains(gm)
    self.assertNotEmpty(chains)
    for launch, nodes in chains.items():
      self.assertEqual(
          {partition_map[node] for node in nodes},
          {partition_map[launch]},
          f"collective chain rooted at {launch.name} was split",
      )

  def test_forward_backward_split_preserves_values_mutations_and_ordering(self):
    """Preserve values and forward mutations while ordering forward first.

    The traced result is ``(loss, parameter gradients..., input gradient)``.
    Splitting must preserve every value, keep autograd work in backward, and
    enqueue all forward children before any backward collective can run.
    """
    model = self._build_toy_model(n_layers=2)
    inputs = torch.randn(6, 8, requires_grad=True)
    traced = self._trace_toy_model(model=model, inputs=inputs)
    expected = run_traced(traced, module=model)(inputs)

    split = self._apply_passes_and_split(
        traced,
        fsdp_reshard_after_forward="never",
    )
    self.assertTrue(
        all(
            not node.meta.get("autograd_backward", False)
            for node in split.forward_graph.graph.nodes
        )
    )
    self.assertNotEmpty(
        [
            node
            for node in split.backward_graph.graph.nodes
            if node.meta.get("autograd_backward", False)
        ]
    )
    backward = split_backward_graph_by_layer(
        split.backward_graph,
        n_layers=2,
    )
    self.assertEqual(
        backward.collective_layer_to_partition,
        {"ag": {}, "rs": {}},
    )
    with self.assertRaisesRegex(
        ValueError, "requires AG and RS ownership for every layer"
    ):
      split_backward_graph_by_layer(
          split.backward_graph,
          n_layers=2,
          pipeline_collectives=True,
      )
    split.backward_graph = backward.module
    split.module.backward_graph = backward.module
    traced.gm = split.module
    with mock.patch.object(
        graph_split_lib, "_enqueue_forward_graph_outputs"
    ) as enqueue:
      actual = run_traced(traced, module=model)(inputs)

    self.assertLen(actual, len(expected))
    output_names = (
        "loss",
        *(f"parameter gradient {i}" for i, _ in enumerate(model.parameters())),
        "input gradient",
    )
    for name, expected_value, actual_value in zip(
        output_names, expected, actual, strict=True
    ):
      with self.subTest(output=name):
        torch.testing.assert_close(actual_value, expected_value)
    self.assertLen(output_names, len(actual))
    enqueue.assert_called_once()
    enqueued_outputs = enqueue.call_args.args[0]
    self.assertLen(
        enqueued_outputs,
        split.module.num_direct_forward_outputs + len(split.residual_nodes),
    )
    self.assertIs(enqueued_outputs[0], actual[0])

    # Side-effect-only writes have no users in the joint graph. They must still
    # become ignored forward outputs so each layer's persistent buffer updates.
    mutating_model = _BufferMutatingToyTransformer(n_layers=2)
    annotate_module_fqns(mutating_model)
    mutating_inputs = torch.randn(6, 8, requires_grad=True)
    mutating_trace = self._trace_toy_model(
        model=mutating_model,
        inputs=mutating_inputs,
    )
    joint_mutations = [
        node
        for node in mutating_trace.gm.graph.nodes
        if graph_split_lib.is_mutation_node(node)
    ]
    self.assertLen(joint_mutations, 2)
    self.assertTrue(all(not node.users for node in joint_mutations))
    initial_counters = mutating_model.counters.clone()
    run_traced(mutating_trace, module=mutating_model)(mutating_inputs)
    expected_counters = mutating_model.counters.clone()
    mutating_model.counters.copy_(initial_counters)

    mutating_split = self._apply_passes_and_split(
        mutating_trace,
        fsdp_reshard_after_forward="never",
    )
    self.assertLen(
        [
            node
            for node in mutating_split.forward_graph.graph.nodes
            if graph_split_lib.is_mutation_node(node)
        ],
        2,
    )
    mutating_trace.gm = mutating_split.module
    run_traced(mutating_trace, module=mutating_model)(mutating_inputs)
    torch.testing.assert_close(mutating_model.counters, expected_counters)

  def test_forward_backward_boundary_edge_cases(self):
    """Keep gradient epilogues and dynamic scalar uses on the backward side."""
    model = self._build_toy_model(n_layers=2)
    with self.subTest("untagged gradient epilogue stays in backward"):
      inputs = torch.randn(2, 8, requires_grad=True)
      traced = self._trace_toy_model(
          model=model,
          inputs=inputs,
          gradient_epilogue=True,
      )
      split = self._apply_passes_and_split(
          traced,
          fsdp_reshard_after_forward="never",
      )
      aliases = [
          node
          for node in traced.gm.graph.nodes
          if node.op == "call_function"
          and node.target is torch.ops.aten.alias.default
      ]
      backward_names = {
          node.name for node in split.backward_graph.graph.nodes
      }
      self.assertNotEmpty(aliases)
      self.assertTrue(
          all(alias.name in backward_names for alias in aliases),
      )

    with self.subTest("dynamic scalar is recomputed instead of saved"):
      inputs = torch.randn(2, 8, requires_grad=True)
      torch._dynamo.mark_dynamic(inputs, 0)
      traced = self._trace_toy_model(
          model=model,
          inputs=inputs,
          use_shape_scalar=True,
      )
      split = self._apply_passes_and_split(
          traced,
          fsdp_reshard_after_forward="never",
      )
      size_nodes = [
          node
          for node in traced.gm.graph.nodes
          if node.target is torch.ops.aten.sym_size.int
      ]
      self.assertLen(size_nodes, 1)
      self.assertNotIn(size_nodes[0], split.residual_nodes)
      self.assertIn(
          size_nodes[0].name,
          {node.name for node in split.backward_graph.graph.nodes},
      )

    with self.subTest("unclassified forward mutation fails closed"):
      late_mutation_model = self._build_toy_model(n_layers=2)
      late_mutation_model.register_buffer("counter", torch.zeros(()))
      inputs = torch.randn(2, 8, requires_grad=True)

      def step(inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        loss = late_mutation_model(inputs).sum()
        with torch.no_grad():
          late_mutation_model.counter.add_(inputs.detach().sum())
        return (loss, *torch.autograd.grad(loss, inputs))

      traced = minimal_fx_tracer(step, module=late_mutation_model)(inputs)
      with self.assertRaisesRegex(
          ValueError, "would discard forward mutation nodes"
      ):
        split_forward_backward_graph(traced.gm, traced.example_inputs)

  def _trace_cpu_offload_graph(self):
    model = _OffloadToyTransformer(n_layers=3)
    annotate_module_fqns(model)
    inputs = torch.randn(512, 8, requires_grad=True)
    traced = self._trace_toy_model(model=model, inputs=inputs)
    layer_0_output = next(
        node
        for node in traced.gm.graph.nodes
        if node.target is torch.ops.aten.relu.default
        and _get_layer_id(node) == 0
    )
    layer_0_output.meta["recompute"] = CheckpointPolicy.MUST_CPU_OFFLOAD
    apply_cpu_offload_pass(
        traced.gm,
        prefetch_lookahead=0,
        defer_n_layers=0,
    )
    return traced

  def test_cpu_offload_ops_are_compiled_inside_layer_children(self):
    """Keep each AO copy and wait together instead of in the eager parent."""
    traced = self._trace_cpu_offload_graph()
    split = split_forward_backward_graph(
        traced.gm, traced.example_inputs
    )
    donor_specs, donor_output_positions, donor_examples = (
        thread_host_offload_donors(split.forward_graph)
    )
    forward = split_forward_graph_by_layer(split.forward_graph, 1)
    backward = split_backward_graph_by_layer(
        split.backward_graph,
        n_layers=3,
    )

    ao_targets = {
        torch.ops.ao.offload.default,
        torch.ops.ao.reload.default,
        torch.ops.ao.wait_tensor.default,
    }

    def targets(module: torch.fx.GraphModule) -> set[object]:
      return {
          node.target
          for node in module.graph.nodes
          if node.op == "call_function"
      }

    self.assertTrue(
        all(target not in ao_targets for target in targets(forward.module))
    )
    self.assertTrue(
        all(target not in ao_targets for target in targets(backward.module))
    )
    forward_child_targets = set().union(
        *(targets(child) for child in forward.module.children())
    )
    backward_child_targets = set().union(
        *(targets(child) for child in backward.module.children())
    )
    self.assertIn(torch.ops.ao.offload.default, forward_child_targets)
    self.assertNotIn(torch.ops.ao.reload.default, forward_child_targets)
    self.assertIn(torch.ops.ao.reload.default, backward_child_targets)
    self.assertNotIn(torch.ops.ao.offload.default, backward_child_targets)
    for parent in (forward.module, backward.module):
      for child in parent.children():
        child_nodes = set(child.graph.nodes)
        for node in child_nodes:
          if node.target not in {
              torch.ops.ao.offload.default,
              torch.ops.ao.reload.default,
          }:
            continue
          self.assertTrue(
              any(
                  user.target == torch.ops.ao.wait_tensor.default
                  and user in child_nodes
                  for user in node.users
              )
          )

    self.assertLen(donor_specs, 1)
    self.assertLen(donor_output_positions, 1)
    self.assertLen(donor_examples, 1)
    split.module.forward_graph = forward.module
    split.module.host_donor_specs = donor_specs
    with self.assertRaisesRegex(RuntimeError, "compiled TPU forward program"):
      split.module._initialize_host_donors(split.forward_example_inputs)
    forward.module._tpu_executable = object()
    with self.assertRaisesRegex(RuntimeError, "without a TPU input"):
      split.module._initialize_host_donors(split.forward_example_inputs)
    del forward.module._tpu_executable
    for child in forward.module.children():
      wrapper = torch.nn.Module()
      wrapper._tpu_executable = object()
      child.add_module("compiled_child", wrapper)
    with self.assertRaisesRegex(RuntimeError, "without a TPU input"):
      split.module._initialize_host_donors(split.forward_example_inputs)
    parent_donors = [
        node
        for node in forward.module.graph.nodes
        if node.op == "placeholder"
        and node.meta.get("tpu_force_donate", False)
    ]
    self.assertLen(parent_donors, 1)
    child_donors = [
        node
        for child in forward.module.children()
        for node in child.graph.nodes
        if node.op == "placeholder"
        and node.meta.get("tpu_force_donate", False)
    ]
    self.assertLen(child_donors, 1)
    donor_waits = [
        node
        for child in forward.module.children()
        for node in child.graph.nodes
        if node.target == torch.ops.ao.wait_tensor.default
        and len(node.args) > 1
        and node.args[1] in child_donors
    ]
    self.assertLen(donor_waits, 1)

  def test_view_layer_boundaries_offload_and_split_by_layer(self):
    """Saved view boundaries offload their bases and stay with their consumers."""
    model = _OffloadToyTransformer(n_layers=3)
    model.layers = nn.ModuleList(_ViewOutputLayer() for _ in range(3))
    annotate_module_fqns(model)
    traced = self._trace_toy_model(
        model=model, inputs=torch.randn(512, 8, requires_grad=True)
    )
    _full_memory_policy_pass(
        traced.gm,
        config=SimpleNamespace(
            compile=SimpleNamespace(full_recompute_save_ops=""),
            parallelism=SimpleNamespace(
                fsdp_reshard_after_forward="always",
                pipeline_parallel_degree=1,
            ),
        ),
    )
    tag_all_offloadable_activations(traced.gm)
    apply_cpu_offload_pass(traced.gm, prefetch_lookahead=0, defer_n_layers=0)
    selective_activation_remat_pass(traced.gm)
    split = split_forward_backward_graph(traced.gm, traced.example_inputs)
    split_backward_graph_by_layer(split.backward_graph, n_layers=3)

    offloads = [
        node
        for node in split.forward_graph.graph.nodes
        if node.target is torch.ops.ao.offload.default
    ]
    self.assertLen(offloads, 2)

    forward = split_forward_graph_by_layer(split.forward_graph, 1)
    for child in forward.module.children():
      returned: list[torch.fx.Node] = []
      torch.fx.node.map_arg(child.graph.output_node().args, returned.append)
      self.assertFalse(
          [node.name for node in returned if _is_view(node)], child.code
      )

  def test_cpu_reload_prefetch_distance_counts_executables(self):
    """Zero uses the consumer child; one uses its immediate predecessor."""
    for distance in (0, 1):
      with self.subTest(distance=distance):
        traced = self._trace_cpu_offload_graph()
        split = split_forward_backward_graph(
            traced.gm, traced.example_inputs
        )
        backward = split_backward_graph_by_layer(
            split.backward_graph,
            n_layers=3,
            cpu_offload_prefetch_executables=distance,
        )

        reload_0 = next(
            node
            for node in split.backward_graph.graph.nodes
            if node.target == torch.ops.ao.reload.default
            and _get_layer_id(node) == 0
        )
        wait_0 = next(
            user
            for user in reload_0.users
            if user.target == torch.ops.ao.wait_tensor.default
        )
        # Layer 0's output is consumed while recomputing layer 1. In backward
        # order, layer 2's child immediately precedes that consumer child.
        expected_partition = backward.layer_to_partition[1 + distance]
        self.assertEqual(backward.partition_map[reload_0], expected_partition)
        self.assertEqual(backward.partition_map[wait_0], expected_partition)

  def test_unsplit_forward_still_donates_persistent_host_slot(self):
    """The default forward setting must not fall back to per-step mappings."""
    traced = self._trace_cpu_offload_graph()
    splitter = graph_split_lib.GraphSplitter(
        enabled=True,
        config=SimpleNamespace(
            forward_layers_per_executable=-1,
            forward_isolate_prefix=False,
            backward_partition_mode="none",
            backward_layers_per_executable=1,
            cpu_offload_prefetch_executables=0,
            donate_internal_buffers=False,
        ),
        n_layers=3,
    )

    with (
        mock.patch.object(
            graph_split_lib, "tpu_backend_compile_pass"
        ) as compile_pass,
        mock.patch.object(
            graph_split_lib, "tpu_backend_compile_partitioned_pass"
        ) as compile_partitioned,
    ):
      module = splitter.split_and_compile(traced)

    compile_partitioned.assert_not_called()
    self.assertEqual(compile_pass.call_count, 2)
    forward_call = compile_pass.call_args_list[0]
    original_input_count = len(traced.example_inputs)
    self.assertEqual(
        forward_call.kwargs["donated_input_positions"],
        (original_input_count,),
    )
    self.assertLen(forward_call.args[1], original_input_count + 1)
    self.assertLen(module.host_donor_specs, 1)
    self.assertLen(module.host_donor_output_positions, 1)
    self.assertTrue(
        any(
            node.op == "placeholder"
            and node.meta.get("tpu_force_donate", False)
            for node in module.forward_graph.graph.nodes
        )
    )

  def test_partition_layouts_reuse_one_simple_fsdp_trace(self):
    """Check forward, by-layer, and pipelined four-layer layouts.

    Forward visits ``L0 -> L1 -> L2 -> L3``; backward visits the layers in
    reverse. ``AGn``, layer ``Ln`` compute, ``RSn``, and every collective's
    launch/getitem/wait chain must retain the ownership shown below.
    """
    split = self._trace_simple_fsdp(n_layers=4)
    forward = split.forward_graph
    backward = split.backward_graph
    self.assertFalse(
        any("fsdp_bucket" in node.meta for node in backward.graph.nodes)
    )

    forward_scenarios = (
        # [prefix,L0,L1] [L2,L3]
        (2, False, {0: 0, 1: 0, 2: 1, 3: 1}),
        # [prefix] [L0] [L1] [L2] [L3]
        (1, True, {0: 1, 1: 2, 2: 3, 3: 4}),
    )
    for layers_per_executable, isolate_prefix, expected in forward_scenarios:
      with self.subTest(
          direction="forward",
          layers_per_executable=layers_per_executable,
          isolate_prefix=isolate_prefix,
      ):
        chunked = split_forward_graph_by_layer(
            forward,
            layers_per_executable,
            isolate_prefix=isolate_prefix,
        )
        self.assertEqual(chunked.layer_to_partition, expected)

    by_layer_one = (1, 2, 3, 4)
    by_layer_three = (1, 1, 1, 2)
    backward_scenarios = (
        # [AG3] [AG2,C3] [AG1,RS3,C2] ... [RS0]
        (
            "staggered-one-layer",
            1,
            True,
            (2, 3, 4, 5),
            (1, 2, 3, 4),
            (3, 4, 5, 6),
        ),
        # Coalesce pairs of compute stages while preserving pipeline fill/drain.
        (
            "staggered-two-layers",
            2,
            True,
            (2, 2, 3, 3),
            (1, 2, 2, 3),
            (2, 3, 3, 4),
        ),
        # By-layer mode keeps each AG/compute/RS chain in one executable.
        (
            "by-layer-one-layer",
            1,
            False,
            by_layer_one,
            by_layer_one,
            by_layer_one,
        ),
        # Width three accepts a partial final group for L0.
        (
            "by-layer-three-layers",
            3,
            False,
            by_layer_three,
            by_layer_three,
            by_layer_three,
        ),
    )
    for name, width, pipeline, compute, ag, rs in backward_scenarios:
      with self.subTest(name):
        chunked = split_backward_graph_by_layer(
            backward,
            n_layers=4,
            layers_per_executable=width,
            pipeline_collectives=pipeline,
        )
        layer_ids = (3, 2, 1, 0)
        self.assertEqual(
            chunked.layer_to_partition,
            dict(zip(layer_ids, compute, strict=True)),
        )
        self.assertEqual(
            chunked.collective_layer_to_partition["ag"],
            dict(zip(layer_ids, ag, strict=True)),
        )
        self.assertEqual(
            chunked.collective_layer_to_partition["rs"],
            dict(zip(layer_ids, rs, strict=True)),
        )
        self._assert_collective_chains_are_atomic(
            backward, chunked.partition_map
        )

    # A real Zero2-like trace has RS but no backward AG ownership.
    zero2_backward = self._trace_simple_fsdp(
        n_layers=2,
        fsdp_reshard_after_forward="never",
    ).backward_graph
    zero2_like = split_backward_graph_by_layer(
        zero2_backward,
        n_layers=2,
    )
    self.assertEmpty(zero2_like.collective_layer_to_partition["ag"])
    self.assertEqual(
        set(zero2_like.collective_layer_to_partition["rs"]),
        {0, 1},
    )
    self.assertEqual(
        zero2_like.collective_layer_to_partition["rs"],
        zero2_like.layer_to_partition,
    )
    self._assert_collective_chains_are_atomic(
        zero2_backward, zero2_like.partition_map
    )
    with self.assertRaisesRegex(
        ValueError, "requires AG and RS ownership for every layer"
    ):
      split_backward_graph_by_layer(
          zero2_backward,
          n_layers=2,
          pipeline_collectives=True,
      )

    # Missing layer ownership must fail instead of silently cutting by position.
    for node in backward.graph.nodes:
      node.meta.pop("custom", None)
    with self.assertRaisesRegex(ValueError, "compute metadata for every layer"):
      split_backward_graph_by_layer(backward, n_layers=4)

  def test_backward_fsdp_pipeline_keeps_bf16_reduce_cast_with_wait(self):
    """Keep ``dW -> RS(BF16) -> wait -> cast(FP32)`` in one executable."""
    gm = self._trace_simple_fsdp(
        n_layers=2,
        reduce_dtype=torch.bfloat16,
    ).backward_graph

    chunked = split_backward_graph_by_layer(
        gm,
        n_layers=2,
        pipeline_collectives=True,
    )

    reduce_casts = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.aten._to_copy.default
        and len(node.all_input_nodes) == 1
        and graph_split_lib.is_wait_tensor(node.all_input_nodes[0])
    ]
    self.assertLen(reduce_casts, 2)
    for cast in reduce_casts:
      wait = cast.all_input_nodes[0]
      self.assertEqual(
          chunked.partition_map[cast], chunked.partition_map[wait]
      )

    # Also verify that when `_movedim_reduce_scatter_tensor` inserts a post-RS
    # view (`reshape` / `movedim`) between `wait_tensor` and the post-RS
    # `_to_copy` cast, both the view and the cast stay with `wait_tensor`.
    gm_with_post_rs_view = self._trace_simple_fsdp(
        n_layers=2,
        reduce_dtype=torch.bfloat16,
    ).backward_graph
    inserted_views = []
    for node in list(gm_with_post_rs_view.graph.nodes):
      if (
          node.op == "call_function"
          and node.target is torch.ops.aten._to_copy.default
          and len(node.all_input_nodes) == 1
          and graph_split_lib.is_wait_tensor(node.all_input_nodes[0])
      ):
        wait = node.all_input_nodes[0]
        with gm_with_post_rs_view.graph.inserting_before(node):
          view = gm_with_post_rs_view.graph.call_function(
              torch.ops.aten.view.default, (wait, list(wait.meta["val"].shape))
          )
          view.meta["val"] = wait.meta["val"]
        node.replace_input_with(wait, view)
        inserted_views.append((wait, view, node))
    gm_with_post_rs_view.recompile()

    chunked_view = split_backward_graph_by_layer(
        gm_with_post_rs_view,
        n_layers=2,
        pipeline_collectives=True,
    )
    self.assertLen(inserted_views, 2)
    for wait, view, cast in inserted_views:
      self.assertEqual(
          chunked_view.partition_map[view], chunked_view.partition_map[wait]
      )
      self.assertEqual(
          chunked_view.partition_map[cast], chunked_view.partition_map[wait]
      )

  def test_reduce_scatter_chain_stops_at_resizing_views_and_compute(self):
    """Only casts and size-preserving views after an RS wait join its chain."""
    gm = self._trace_simple_fsdp(
        n_layers=2,
        reduce_dtype=torch.bfloat16,
    ).backward_graph
    graph = gm.graph
    reduce_casts = [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.aten._to_copy.default
        and len(node.all_input_nodes) == 1
        and graph_split_lib.is_wait_tensor(node.all_input_nodes[0])
    ]
    self.assertLen(reduce_casts, 2)
    waits = [cast.all_input_nodes[0] for cast in reduce_casts]

    def insert_before(cast, target, args, shape):
      """Makes ``target(<cast's input>, *args)`` the cast's input."""
      source = cast.all_input_nodes[0]
      with graph.inserting_before(cast):
        node = graph.call_function(target, (source, *args))
      node.meta["val"] = torch.empty(
          shape, dtype=source.meta["val"].dtype, device="meta"
      )
      cast.replace_input_with(source, node)
      return node

    # Layer 0: wait -> view -> expand (more elements) -> cast.
    shape = list(waits[0].meta["val"].shape)
    view = insert_before(
        reduce_casts[0], torch.ops.aten.view.default, (shape,), shape
    )
    expand = insert_before(
        reduce_casts[0],
        torch.ops.aten.expand.default,
        ([2, *shape],),
        [2, *shape],
    )
    # Layer 1: wait -> mul (compute) -> cast.
    mul = insert_before(
        reduce_casts[1],
        torch.ops.aten.mul.Tensor,
        (2.0,),
        list(waits[1].meta["val"].shape),
    )
    gm.recompile()

    chains = graph_split_lib._collect_collective_chains(gm)
    chain0, chain1 = (
        next(chain for chain in chains.values() if wait in chain)
        for wait in waits
    )
    self.assertIn(view, chain0)
    self.assertNotIn(expand, chain0)
    self.assertNotIn(reduce_casts[0], chain0)
    self.assertNotIn(mul, chain1)
    self.assertNotIn(reduce_casts[1], chain1)


if __name__ == "__main__":
  absltest.main()
