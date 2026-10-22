"""One shard axis per stack, not one per holder.

Sharding every stack on dim 1 breaks once the mesh is wider than dim 1 of some
stack: ``data_parallel`` quietly replicates that parameter, and the run OOMs
with nothing pointing at the cause. Qwen3-235B hits this at 129 ranks, where
the expert stacks' dim 1 is ``num_experts = 128``.

A fake process group is enough: the failure is about mesh size, not devices.
"""

import contextlib
import unittest
import unittest.mock

import torch
import torch.distributed as dist
from absl.testing import absltest
from torch import nn
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.experiments.tpu.graph_trainer.layer_stack import (
    _shard_dim,
    LayerStackHolder,
)
from torchtitan.experiments.tpu.graph_trainer import scan_fsdp
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import plan_stacked_gathers


N_LAYERS = 94

# Qwen3-235B-A22B's stacked parameters, layer axis in front.
Q235_STACKS = {
    "moe__experts__w1_EDF": (N_LAYERS, 128, 4096, 1536),
    "moe__experts__w2_EFD": (N_LAYERS, 128, 1536, 4096),
    "moe__experts__w3_EDF": (N_LAYERS, 128, 4096, 1536),
    "moe__router__gate__weight": (N_LAYERS, 128, 4096),
    "attention__qkv_linear__wqkv": (N_LAYERS, 9216, 4096),
    "attention_norm__weight": (N_LAYERS, 4096),
    "attention__q_norm__weight": (N_LAYERS, 128),
}

# Axis 1 is num_experts = 128 in these; everything else divides on axis 1.
SPLIT = [
    n for n in Q235_STACKS if n.startswith(("moe__experts__", "moe__router__"))
]


def _pick(shape, degree):
    return _shard_dim(torch.empty(shape, device="meta"), degree)


@contextlib.contextmanager
def _fake_pg(world):
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=world)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _holder(shapes=Q235_STACKS, device="meta"):
    """The stacks, as `unregister_and_allocate` leaves them (zeros if real)."""
    holder = LayerStackHolder()
    for name, shape in shapes.items():
        holder.register_parameter(
            name, nn.Parameter(torch.zeros(shape, device=device))
        )
    return holder


def _shard(holder, mesh, style="fully_shard"):
    """`configure_qwen_simple_fsdp_shard_dims` + `apply_simple_fsdp`, in order."""
    degree = mesh.size(mesh.ndim - 1)
    data_parallel(holder, mesh, style, shard_dim=holder.plan_shard_dims(degree))
    holder.shard_split_stacks()
    return holder


@contextlib.contextmanager
def _sharded_holder(world, shapes=Q235_STACKS, shard_dim=None):
    """The stacks, wrapped by SimpleFSDP on a `world`-wide fake mesh."""
    with _fake_pg(world):
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("fsdp",))
        holder = _holder(shapes)
        if shard_dim is None:
            _shard(holder, mesh)
        else:
            data_parallel(holder, mesh, "fully_shard", shard_dim=shard_dim)
        yield holder


class TestShardDimPicker(unittest.TestCase):

    def test_dim_one_is_kept_whenever_it_divides(self):
        # Every published 235B number was taken at 128 ranks on dim 1. Moving a
        # stack off dim 1 there would make the comparison not one.
        for name, shape in Q235_STACKS.items():
            if shape[1] % 128 == 0:
                self.assertEqual(_pick(shape, 128), 1, name)

    def test_every_expert_stack_takes_the_same_axis_at_512_ranks(self):
        # Both hidden axes of every expert stack divide 512, and the unrolled
        # path puts all three on the same one (its dim 1, this dim 2). Taking
        # the widest instead would send w2_EFD to dim 3 by itself.
        for name in Q235_STACKS:
            if name.startswith("moe__experts__"):
                self.assertEqual(_pick(Q235_STACKS[name], 512), 2, name)

    def test_an_expert_stack_moves_off_num_experts_past_128_ranks(self):
        # Dim 1 is the 128 experts. By 2048 only w2_EFD's minor axis still
        # divides, so it is the one stack that has to leave dim 2.
        self.assertEqual(_pick(Q235_STACKS["moe__experts__w1_EDF"], 2048), 2)
        self.assertEqual(_pick(Q235_STACKS["moe__experts__w2_EFD"], 2048), 3)

    def test_a_gather_off_dim_zero_is_transposed_onto_dim_zero(self):
        # `funcol`'s fast path needs every axis before the gather axis to be
        # size 1, and a layer slice starts with `num_experts`. Off dim 0 it
        # would fall back to `cat(chunk(t, world_size))`; the transpose has to
        # both avoid that and reassemble on the axis the caller asked for.
        shard = torch.arange(3 * 4 * 6).reshape(3, 4, 6).float()
        ranks = 4

        def fake_all_gather(x, gather_dim, group):
            self.assertEqual(gather_dim, 0, "gathered off dim 0")
            return torch.cat([x] * ranks, dim=0)

        with unittest.mock.patch.object(
            scan_fsdp.funcol, "all_gather_tensor", fake_all_gather
        ):
            out = scan_fsdp._gather_on(shard, 2, "group")
        self.assertTrue(torch.equal(out, torch.cat([shard] * ranks, dim=2)))

    def test_subtile_two_d_gather_and_scatter_fold_minor_axis_to_align_tiles(
        self,
    ):
        # For 2-D weight shards `[rows, cols]`, when `rows % 8 != 0` (e.g.
        # Llama 405B w1/w3 at 26 rows/rank, Qwen3 wqkv at 9 rows/rank, or
        # Llama 70B wk/wv at 4 cols/rank on gather_dim=1), `_movedim_*` folds
        # `k = 8 // gcd(rows, 8)` from `cols` into `rows` so XLA sees an
        # 8-row-aligned collective.
        ranks = 4
        with _fake_pg(ranks):
            mesh = init_device_mesh("cpu", (ranks,), mesh_dim_names=("fsdp",))
            group_name = mesh.get_group(0).group_name
            for shape, gather_dim, expected_folded in (
                ((26, 1024), 0, (104, 256)),
                ((9, 1024), 0, (72, 128)),
                ((1024, 4), 1, (8, 512)),
            ):
                shard = (
                    torch.arange(shape[0] * shape[1], dtype=torch.float32)
                    .reshape(shape)
                    .requires_grad_(True)
                )
                seen_gather_shapes = []
                seen_scatter_shapes = []

                def fake_all_gather(x, gather_dim, group, tag=""):
                    self.assertEqual(gather_dim, 0)
                    seen_gather_shapes.append(tuple(x.shape))
                    return torch.cat([x] * ranks, dim=0)

                def fake_reduce_scatter(
                    x, reduceOp, scatter_dim, group, tag=""
                ):
                    self.assertEqual(scatter_dim, 0)
                    seen_scatter_shapes.append(tuple(x.shape))
                    return x.chunk(ranks, dim=0)[0] * ranks

                spec = scan_fsdp.ScanFSDPSpec(
                    group_name=group_name,
                    param_dtype=None,
                    reduce_dtype=None,
                    sharded=True,
                    gather_dim=gather_dim,
                )
                with (
                    unittest.mock.patch.object(
                        scan_fsdp, "_orig_all_gather_single", fake_all_gather
                    ),
                    unittest.mock.patch.object(
                        scan_fsdp,
                        "_orig_reduce_scatter_single",
                        fake_reduce_scatter,
                    ),
                ):
                    whole = scan_fsdp.gather_slice(shard, spec)
                    expected_whole = torch.cat([shard] * ranks, dim=gather_dim)
                    self.assertEqual(tuple(whole.shape), tuple(expected_whole.shape))
                    self.assertTrue(torch.equal(whole, expected_whole))
                    grad_out = torch.randn_like(whole)
                    whole.backward(grad_out)

                self.assertEqual(seen_gather_shapes, [expected_folded])
                self.assertEqual(
                    seen_scatter_shapes,
                    [(expected_folded[0] * ranks, expected_folded[1])],
                )
                expected_grad = (
                    grad_out.chunk(ranks, dim=gather_dim)[0] * ranks
                )
                self.assertTrue(torch.equal(shard.grad, expected_grad))

    def test_fold_survives_strides_that_change_between_trace_and_replay(self):
        # TorchTPU traces against fake strides and replays with its own. The
        # fused loss's `lm_head` gradient is contiguous in the first and
        # transposed in the second (128 or 256 ranks put it on Shard(0) with
        # 1002 or 501 rows/rank, so it folds), and a `.contiguous()` traced as
        # a no-op left `view` to fail at replay. Trace contiguous, replay
        # transposed.
        ranks = 4
        rows, cols = 26, 1024  # per rank; 26 rows -> k = 4
        with _fake_pg(ranks):
            mesh = init_device_mesh("cpu", (ranks,), mesh_dim_names=("fsdp",))
            group_name = mesh.get_group(0).group_name

            def fake_all_gather(x, gather_dim, group, tag=""):
                return torch.cat([x] * ranks, dim=0)

            def fake_reduce_scatter(x, reduceOp, scatter_dim, group, tag=""):
                return x.chunk(ranks, dim=0)[0] * ranks

            cases = (
                (
                    lambda x: scan_fsdp._movedim_all_gather_tensor(x, 0, group_name),
                    (rows, cols),
                    lambda x: torch.cat([x] * ranks, dim=0),
                ),
                (
                    lambda x: scan_fsdp._movedim_reduce_scatter_tensor(
                        x, "sum", 0, group_name
                    ),
                    (rows * ranks, cols),
                    lambda x: x.chunk(ranks, dim=0)[0] * ranks,
                ),
            )
            with (
                unittest.mock.patch.object(
                    scan_fsdp, "_orig_all_gather_single", fake_all_gather
                ),
                unittest.mock.patch.object(
                    scan_fsdp, "_orig_reduce_scatter_single", fake_reduce_scatter
                ),
            ):
                for collective, shape, expected in cases:
                    traced = make_fx(collective)(torch.zeros(shape))
                    transposed = torch.randn(shape[::-1]).t()
                    self.assertFalse(transposed.is_contiguous())
                    out = torch.fx.Interpreter(traced).run(transposed)
                    self.assertTrue(torch.equal(out, expected(transposed)))

    def test_the_layer_axis_is_never_chosen(self):
        for name, shape in Q235_STACKS.items():
            for degree in (128, 1024, 2048, 4096):
                self.assertNotEqual(_pick(shape, degree), 0, f"{name} @ {degree}")
        # And a stack with no other axis is refused, not sharded on dim 0.
        with self.assertRaises(ValueError):
            _pick((N_LAYERS,), 128)

    def test_configure_simple_fsdp_shard_dims_allows_narrow_hidden_axis_for_replication(
        self,
    ):
        # Llama 3 8B at 8192 ranks: `[128256, 4096]` has `128256 % 8192 != 0`
        # and `4096 < 8192`. Setting `_fsdp_shard_dim = 1` lets `data_parallel`
        # downgrade `Shard(1)` to `Replicate()`.
        model = nn.Module()
        model.tok_embeddings = nn.Embedding(128256, 4096, device="meta")
        model.lm_head = nn.Linear(4096, 128256, bias=False, device="meta")
        pdims = unittest.mock.MagicMock(dp_shard=8192, cp=1)
        with self.assertLogs(scan_fsdp.logger, level="WARNING") as cm:
            moved = scan_fsdp.configure_simple_fsdp_shard_dims(model, pdims)
        self.assertTrue(
            any(
                "tok_embeddings.weight" in msg and "replicate" in msg
                for msg in cm.output
            )
        )
        self.assertEqual(moved, ["tok_embeddings", "lm_head"])
        self.assertEqual(model.tok_embeddings._fsdp_shard_dim, 1)
        self.assertEqual(model.lm_head._fsdp_shard_dim, 1)
        with _fake_pg(8192):
            mesh = init_device_mesh("cpu", (8192,), mesh_dim_names=("fsdp",))
            data_parallel(
                model.tok_embeddings,
                mesh,
                "fully_shard",
                shard_dim=model.tok_embeddings._fsdp_shard_dim,
            )
            self.assertEqual(
                model.tok_embeddings._parameters["weight"].placements,
                (Replicate(),),
            )
        # Still refuses when `shape[1] >= fsdp_degree` and neither axis divides.
        bad = nn.Module()
        bad.tok_embeddings = nn.Embedding(128256, 6144, device="meta")
        with self.assertRaises(ValueError):
            scan_fsdp.configure_simple_fsdp_shard_dims(
                bad, unittest.mock.MagicMock(dp_shard=4096, cp=1)
            )


class TestQ235Placements(unittest.TestCase):
    """What ``data_parallel`` and the scan body's plan end up with."""

    def test_a_dict_places_exactly_like_the_int_it_repeats(self):
        # Every other `data_parallel` caller passes an int, so the dict has to
        # be a superset of that path and not a second one beside it.
        def placements(shard_dim):
            with _sharded_holder(128, shard_dim=shard_dim) as holder:
                return {n: p.placements for n, p in holder._parameters.items()}

        self.assertEqual(
            placements(1), placements({name: 1 for name in Q235_STACKS})
        )

    def test_an_already_distributed_parameter_is_refused(self):
        # A DTensor here means EP or TP ran first, and `ReplicateComputation`
        # would re-wrap it with the module-level axis, losing the per-stack one.
        with _fake_pg(8):
            mesh = init_device_mesh("cpu", (8,), mesh_dim_names=("fsdp",))
            holder = LayerStackHolder()
            holder.register_parameter(
                "w",
                nn.Parameter(
                    distribute_tensor(
                        torch.empty(4, 16, device="meta"), mesh, (Shard(1),)
                    )
                ),
            )
            with self.assertRaises(NotImplementedError):
                data_parallel(holder, mesh, "fully_shard", shard_dim={"w": 1})

    def test_hsdp_plans_off_the_shard_axis_not_the_whole_mesh(self):
        # `dp_replicate=16, dp_shard=128` is how 235B reaches 8k chips. Only the
        # shard axis splits the tensor, so only its width may pick the dim.
        with _fake_pg(2048):
            mesh = init_device_mesh(
                "cpu", (16, 128), mesh_dim_names=("dp_replicate", "fsdp")
            )
            holder = _shard(_holder(), mesh, "hybrid_shard")
            for name, spec in plan_stacked_gathers(holder).items():
                self.assertTrue(spec.sharded, name)
                self.assertEqual(spec.gather_dim, 0, name)  # dim 1 still divides


class TestSplitStacks(unittest.TestCase):
    """Past ``num_experts`` ranks, ``[L, E, X, ...]`` is sharded on axes 1 and 2."""

    def test_nothing_is_split_while_num_experts_divides(self):
        with _sharded_holder(128) as holder:
            self.assertEqual(holder.split_meshes, {})
            for name, shape in Q235_STACKS.items():
                self.assertEqual(tuple(holder._parameters[name].shape), shape, name)

    def test_split_stacks_keep_their_logical_shape(self):
        for world in (256, 2048):
            with self.subTest(world=world), _sharded_holder(world) as holder:
                ranks_per_expert = world // 128
                plan = plan_stacked_gathers(holder)
                for name in SPLIT:
                    layers, a, b, *rest = Q235_STACKS[name]
                    param = holder._parameters[name]
                    self.assertEqual(tuple(param.shape), Q235_STACKS[name], name)
                    self.assertEqual(param.placements, (Shard(1), Shard(2)), name)
                    self.assertEqual(
                        tuple(param.to_local().shape),
                        (layers, 1, b // ranks_per_expert, *rest),
                        name,
                    )
                    # One flat gather over the whole FSDP group, no transpose.
                    self.assertEqual(plan[name].gather_dim, 0, name)
                    self.assertEqual(plan[name].split_shape, (a, b, *rest), name)
                for name in set(Q235_STACKS) - set(SPLIT):
                    self.assertEqual(len(holder._parameters[name].placements), 1)
                self.assertEqual(len(holder.split_meshes), 1)

    def test_replicated_stacks_split_when_no_single_axis_divides(self):
        # At 8192 ranks no single axis of Q235_STACKS[SPLIT] is >= 8192, so
        # `data_parallel` downgrades them to `Replicate()`; `shard_split_stacks`
        # must still split them onto `(128, 64)`.
        split_shapes = {k: Q235_STACKS[k] for k in SPLIT}
        with _sharded_holder(8192, shapes=split_shapes) as holder:
            plan = plan_stacked_gathers(holder)
            for name, shape in split_shapes.items():
                layers, a, b, *rest = shape
                param = holder._parameters[name]
                self.assertEqual(param.placements, (Shard(1), Shard(2)), name)
                self.assertEqual(
                    tuple(param.to_local().shape),
                    (layers, 1, b // 64, *rest),
                    name,
                )
                self.assertTrue(plan[name].sharded, name)
                self.assertEqual(plan[name].split_shape, (a, b, *rest), name)

    def test_ddp_replicate_mode_is_not_split(self):
        with _fake_pg(256):
            mesh = init_device_mesh("cpu", (256,), mesh_dim_names=("dp_replicate",))
            holder = _shard(_holder(), mesh, "replicate")
            self.assertEqual(holder.split_meshes, {})
            for name in SPLIT:
                self.assertEqual(
                    holder._parameters[name].placements, (Replicate(),), name
                )

    def test_a_shard_spanning_two_experts_is_not_split(self):
        # 192 ranks: neither 128 % 192 nor 192 % 128, so a rank's rows would
        # straddle experts and have no DTensor spelling in [E, X, ...].
        with _sharded_holder(192) as holder:
            self.assertEqual(holder.split_meshes, {})

    def test_hsdp_keeps_the_replicate_axis_in_front(self):
        with _fake_pg(512):
            mesh = init_device_mesh(
                "cpu", (2, 256), mesh_dim_names=("dp_replicate", "fsdp")
            )
            holder = _shard(_holder(), mesh, "hybrid_shard")
            plan = plan_stacked_gathers(holder)
            for name in SPLIT:
                param = holder._parameters[name]
                self.assertEqual(
                    param.placements, (Replicate(), Shard(1), Shard(2)), name
                )
                # A layer, as init and the state dict see it.
                self.assertEqual(
                    param[0].placements, (Replicate(), Shard(0), Shard(1))
                )
                self.assertIsNotNone(plan[name].replicate_group_name, name)

    def test_layer_writes_land_in_the_stack(self):
        # What init_states and load_state_dict do to a layer has to show up in
        # the stack. [2 layers, 2 entries, 4, 3] over 4 ranks: 2 ranks per
        # entry, so rank 0 holds rows 0-1 of entry 0 of each layer.
        with _fake_pg(4):
            mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("fsdp",))
            holder = _shard(_holder({"w1": (2, 2, 4, 3)}), mesh).to_empty(
                device="cpu"
            )
            stack = holder._parameters["w1"]
            self.assertEqual(tuple(stack.to_local().shape), (2, 1, 2, 3))
            with torch.no_grad():
                stack.zero_()
                nn.init.constant_(stack[0], 7.0)  # a DTensor op, like init_states
            torch.testing.assert_close(
                stack.to_local()[0], torch.full((1, 2, 3), 7.0)
            )
            self.assertEqual(stack.to_local()[1].abs().sum().item(), 0.0)

    def test_unstacked_grouped_experts_split_onto_2d_mesh_past_num_experts(
        self,
    ):
        from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
        from torchtitan.experiments.tpu.graph_trainer.tpu_sharding import (
            configure_tpu_fsdp_shard_dims,
            shard_split_fsdp_modules,
            SplitStackReplicateComputation,
        )
        for world in (2048, 8192):
            with self.subTest(world=world), _fake_pg(world):
                mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("fsdp",))
                model = nn.Module()
                layer = nn.Module()
                with torch.device("meta"):
                    layer.experts = TpuGroupedExperts.Config(
                        dim=4096,
                        hidden_dim=1536,
                        num_experts=128,
                        use_pallas_grouped_mm=True,
                        use_kernel_native_weight_layout=True,
                    ).build()
                model.layers = nn.ModuleDict({"0": layer})
                pdims = unittest.mock.MagicMock(
                    dp_shard=world, cp=1, ep_enabled=False
                )
                moved = configure_tpu_fsdp_shard_dims(
                    model, pdims, include_unstacked_experts=True
                )
                self.assertEqual(moved, ["layers.0.experts"])
                self.assertEqual(layer.experts._fsdp_shard_dim, 1)
                data_parallel(
                    layer.experts,
                    mesh,
                    "fully_shard",
                    shard_dim=layer.experts._fsdp_shard_dim,
                )
                shard_split_fsdp_modules(model)
                ranks_per_expert = world // 128
                for name, expected_b in (
                    ("w1_EDF", 4096),
                    ("w2_EFD", 1536),
                    ("w3_EDF", 4096),
                ):
                    param = layer.experts._parameters[name]
                    self.assertEqual(param.placements, (Shard(0), Shard(1)), name)
                    self.assertEqual(
                        tuple(param.to_local().shape),
                        (1, expected_b // ranks_per_expert, 5632 - expected_b),
                        name,
                    )
                comp = layer.experts._simple_fsdp_parametrization_list[0]
                self.assertIsInstance(comp, SplitStackReplicateComputation)
                self.assertEqual(comp.gather_dim, 0)

    def test_unstacked_decoder_linear_moves_to_dim_one_when_leading_axis_is_uneven(
        self,
    ):
        from torchtitan.experiments.tpu.graph_trainer.tpu_sharding import (
            configure_tpu_fsdp_shard_dims,
        )

        model = nn.Module()
        layer = nn.Module()
        layer.w1 = nn.Linear(4096, 13824, bias=False, device="meta")
        layer.w2 = nn.Linear(13824, 4096, bias=False, device="meta")
        model.layers = nn.ModuleDict({"0": layer})
        pdims = unittest.mock.MagicMock(dp_shard=4096, cp=1, ep_enabled=False)
        moved = configure_tpu_fsdp_shard_dims(model, pdims)
        self.assertEqual(moved, ["layers.0.w1"])
        self.assertEqual(layer.w1._fsdp_shard_dim, 1)
        self.assertFalse(hasattr(layer.w2, "_fsdp_shard_dim"))


if __name__ == "__main__":
    absltest.main()

