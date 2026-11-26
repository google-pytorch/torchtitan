"""Scan + expert parallelism on a fake mesh: placement, plan, refusals (CPU).

The 8-chip parity run (scan + EP vs the splitter at the same EP) is the
end-to-end check; these pin the pieces it depends on without a TPU.
"""

import contextlib
import types
import unittest
import unittest.mock

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.experiments.tpu.graph_trainer.layer_stack import (
    LayerStack,
    LayerStackHolder,
)
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import (
    plan_stacked_gathers,
)
from torchtitan.experiments.tpu.graph_trainer.scan_guards import (
    check_scan_expert_parallel,
)
from torchtitan.protocols.sharding import ShardingConfig

E, X, Y, L = 16, 4, 8, 3
EXPERT = "moe__experts__w1"
DENSE = "attention__wq__weight"


@contextlib.contextmanager
def _fake_pg(world):
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=world)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _pdims(world, ep, dp_replicate=1):
    import torchtitan.distributed.parallel_dims as parallel_dims_module

    parallel_dims_module.device_type = "cpu"
    return ParallelDims(
        dp_replicate=dp_replicate,
        dp_shard=world // dp_replicate,
        cp=1,
        tp=1,
        pp=1,
        ep=ep,
        world_size=world,
    )


def _placed(pdims, device="meta"):
    """Stash, dense SimpleFSDP, place: the order `parallelize_fn` runs them."""
    holder = LayerStackHolder()
    if device == "meta":
        w = torch.empty(L, E, X, Y, device="meta")
    else:
        w = torch.arange(L * E * X * Y, dtype=torch.float32).view(L, E, X, Y)
    holder.register_parameter(EXPERT, nn.Parameter(w))
    holder.register_parameter(
        DENSE, nn.Parameter(torch.empty(L, 64, 16, device="meta"))
    )
    holder.stash_expert_stacks({EXPERT: 0})
    mode = "hybrid_shard" if pdims.dp_replicate_enabled else "fully_shard"
    names = ["dp_replicate", "dp_shard"] if pdims.dp_replicate_enabled else ["dp_shard"]
    data_parallel(holder, pdims.get_mesh(names), mode, shard_dim={DENSE: 1})
    holder.place_expert_stacks(pdims)
    return holder


class TestPlacement(unittest.TestCase):
    def test_expert_stack_is_the_splitters_layer_layout_one_axis_in(self):
        with _fake_pg(8):
            holder = _placed(_pdims(8, ep=2))
            w = holder._parameters[EXPERT]
            self.assertEqual(w.device_mesh.mesh_dim_names, ("efsdp", "ep"))
            self.assertEqual(
                w.placements, (_StridedShard(1, split_factor=2), Shard(1))
            )
            self.assertEqual(tuple(w.to_local().shape), (L, E // 8, X, Y))
            # Dense stacks are untouched by EP.
            d = holder._parameters[DENSE]
            self.assertEqual(d.device_mesh.mesh_dim_names, ("dp_shard",))

    def test_each_rank_holds_its_piece_of_its_ep_block(self):
        # (efsdp=i, ep=j) holds piece i of EP block j, so gathering efsdp gives
        # EP block j: the experts j's dispatch routes to.
        with _fake_pg(8):
            pdims = _pdims(8, ep=2)
            with unittest.mock.patch.object(
                DeviceMesh, "get_coordinate", return_value=[1, 1]
            ):
                holder = _placed(pdims, device="cpu")
            local = holder._parameters[EXPERT].to_local()
            full = torch.arange(L * E * X * Y, dtype=torch.float32).view(L, E, X, Y)
            block, piece = E // 2, E // 8
            torch.testing.assert_close(
                local, full[:, 1 * block + 1 * piece : 1 * block + 2 * piece]
            )

    def test_ep_equal_to_world_replicates_efsdp(self):
        with _fake_pg(8):
            w = _placed(_pdims(8, ep=8))._parameters[EXPERT]
            self.assertEqual(w.placements, (Replicate(), Shard(1)))
            self.assertEqual(tuple(w.to_local().shape), (L, E // 8, X, Y))

    def test_row_view_shares_storage_and_is_the_layer_layout(self):
        with _fake_pg(8):
            holder = _placed(_pdims(8, ep=2), device="cpu")
            stack = holder._parameters[EXPERT]
            row = holder.layer_view(EXPERT, stack, 2)
            self.assertIsInstance(row, DTensor)
            self.assertEqual(tuple(row.shape), (E, X, Y))
            self.assertEqual(
                row.placements, (_StridedShard(0, split_factor=2), Shard(0))
            )
            with torch.no_grad():  # as init_states writes it
                row.to_local().fill_(7.0)
            self.assertTrue(torch.all(stack.to_local()[2] == 7.0))


class TestPlan(unittest.TestCase):
    def test_body_gathers_efsdp_only(self):
        with _fake_pg(8):
            pdims = _pdims(8, ep=2)
            plan = plan_stacked_gathers(_placed(pdims))
            spec = plan[EXPERT]
            self.assertTrue(spec.sharded)
            self.assertEqual(spec.gather_dim, 0)  # the expert axis of one row
            self.assertEqual(dist.get_world_size(spec.group), 4)  # efsdp
            self.assertIsNone(spec.replicate_group)
            self.assertEqual(dist.get_world_size(plan[DENSE].group), 8)

    def test_hsdp_adds_the_replicate_all_reduce(self):
        with _fake_pg(16):
            plan = plan_stacked_gathers(_placed(_pdims(16, ep=2, dp_replicate=2)))
            spec = plan[EXPERT]
            self.assertEqual(dist.get_world_size(spec.group), 4)  # efsdp
            self.assertEqual(dist.get_world_size(spec.replicate_group), 2)

    def test_efsdp_of_one_gathers_nothing(self):
        with _fake_pg(8):
            spec = plan_stacked_gathers(_placed(_pdims(8, ep=8)))[EXPERT]
            self.assertFalse(spec.sharded)

    def test_an_unplaced_expert_stack_is_refused(self):
        holder = LayerStackHolder()
        holder.register_parameter(
            EXPERT, nn.Parameter(torch.empty(L, E, X, Y, device="meta"))
        )
        holder.stash_expert_stacks({EXPERT: 0})
        with self.assertRaisesRegex(RuntimeError, "never placed"):
            plan_stacked_gathers(holder)


class _Experts(nn.Module):
    def __init__(self, layouts):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(E, X, Y, device="meta"))
        self._sharding_config = ShardingConfig(state_shardings=layouts)


class _Layer(nn.Module):
    def __init__(self, layouts):
        super().__init__()
        self.experts = _Experts(layouts)
        self.norm = nn.Parameter(torch.empty(X, device="meta"))


def _stack(layouts):
    layers = nn.ModuleDict({str(i): _Layer(layouts) for i in range(L)})
    stack = LayerStack(layers, LayerStackHolder())
    stack.unregister_and_allocate()
    return stack


_EP = spmd.SpmdType({"efsdp": spmd.R, "ep": spmd.S(0)})


class TestExpertDims(unittest.TestCase):
    def test_read_from_the_declared_placement_not_the_name(self):
        self.assertEqual(
            _stack({"w1": _EP}).expert_parallel_dims(), {"experts__w1": 0}
        )

    def test_a_renamed_weight_takes_the_modules_one_ep_placement(self):
        # TpuGroupedExperts renames to kernel-native names only at parallelize.
        self.assertEqual(
            _stack({"w1_canonical": _EP}).expert_parallel_dims(),
            {"experts__w1": 0},
        )

    def test_ambiguous_ep_placements_raise(self):
        other = spmd.SpmdType({"efsdp": spmd.R, "ep": spmd.S(1)})
        with self.assertRaisesRegex(ValueError, "cannot place its stack"):
            _stack({"a": _EP, "b": other}).expert_parallel_dims()

    def test_no_ep_placement_means_no_expert_stack(self):
        dense = spmd.SpmdType({"tp": spmd.S(0)})
        self.assertEqual(_stack({"w1": dense}).expert_parallel_dims(), {})


def _config(ep=2, **tpu):
    fields = dict(
        ep_transport=None,
        ep_token_chunks=None,
        ep_max_inflight_chunks=None,
    )
    fields.update(tpu)
    model = types.SimpleNamespace(traverse=lambda cls: [])
    return types.SimpleNamespace(
        parallelism=types.SimpleNamespace(expert_parallel_degree=ep),
        tpu_config=types.SimpleNamespace(**fields),
        model_spec=types.SimpleNamespace(model=model),
    )


class TestRefusals(unittest.TestCase):
    def test_static_ep_passes(self):
        check_scan_expert_parallel(_config(ep_transport="auto", ep_token_chunks=2))
        check_scan_expert_parallel(_config(ep=1, ep_transport="all_to_all"))

    def test_host_syncing_transport(self):
        with self.assertRaisesRegex(ValueError, "all_to_all"):
            check_scan_expert_parallel(_config(ep_transport="all_to_all"))

    def test_ragged_transport(self):
        with self.assertRaisesRegex(ValueError, "ragged"):
            check_scan_expert_parallel(_config(ep_transport="ragged_all_to_all"))

    def test_auto_past_top_k_picks_ragged(self):
        cfg = _config(ep=8)
        dispatcher = types.SimpleNamespace(top_k=4, ep_transport=None)
        cfg.model_spec.model.traverse = lambda cls: (
            [(None, dispatcher, None, None)] if cls.__name__ == "Config"
            and cls.__qualname__.startswith("LocalTokenDispatcher") else []
        )
        with self.assertRaisesRegex(ValueError, "ragged"):
            check_scan_expert_parallel(cfg)
        dispatcher.top_k = 8
        check_scan_expert_parallel(cfg)

    def test_grow_buffer(self):
        with self.assertRaisesRegex(ValueError, "grow"):
            check_scan_expert_parallel(_config(ep_capacity_overflow="grow"))

    def test_more_chunks_than_in_flight(self):
        with self.assertRaisesRegex(ValueError, "EP token chunks"):
            check_scan_expert_parallel(_config(ep_token_chunks=4))


if __name__ == "__main__":
    unittest.main()
