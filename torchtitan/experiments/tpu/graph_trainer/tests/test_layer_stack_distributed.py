"""Stacking must not change what the model computes, on the real device.

Structural claims -- FQNs, views, placements, checkpoints -- hold wherever the
tensors live, so ``test_layer_stack`` asserts them on CPU. Numerics do not. The
stacked layers run as one batched matmul per op instead of ``n_layers`` separate
ones: a different kernel, a different accumulation order. So the parity check is
repeated here on ``--device``, across ``--num_devices`` ranks, with the stacks
actually sharded.
"""

import functools

from absl import logging
from absl.testing import absltest
import torch
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)

from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu.graph_trainer.layer_stack import scan_plan, STACK_ATTR
from torchtitan.experiments.tpu.graph_trainer.tpu_fsdp import (
    shard_module,
    SPECS_ATTR,
    unshard,
)
from torchtitan.experiments.tpu.graph_trainer.tests.test_layer_stack import (
    _build,
    _moe_config,
    SEQ_LEN,
    TOKENS,
)


def _models(device, world_size):
    """A plain model and a stacked one, built identically on ``device``."""
    # The FSDP mesh is placed on this device type, so it has to name the
    # device the weights are actually on.
    import torchtitan.distributed.parallel_dims as parallel_dims  # pylint: disable=g-import-not-at-top

    parallel_dims.device_type = device.type
    plain = _build(stacked=False, device=device, world_size=world_size)
    stacked = _build(stacked=True, device=device, world_size=world_size)
    return plain, stacked


def forward_matches_unrolled(device, rank, world_size):
    """One batched matmul per op must give the same bits as n separate ones."""
    del rank
    plain, stacked = _models(device, world_size)
    tokens = torch.randint(0, plain.config.vocab_size, (TOKENS,), device=device)
    torch.testing.assert_close(stacked(tokens), plain(tokens), rtol=0, atol=0)


def flattened_forward_matches_unrolled(device, rank, world_size):
    """A flattened stack must compute what the unrolled experts do.

    With 2 experts and at least 4 ranks, ``flatten_for`` stores the expert and
    router stacks as ``[L, E*X, ...]`` and the body views each gathered layer
    back to ``[E, X, ...]``. Getting that view wrong mixes rows across experts
    and still runs, so compare against the plain model bit for bit.
    """
    del rank
    if world_size < 4 or world_size % 2:
        logging.warning("skipped: 2 experts flatten only on an even mesh of >= 4")
        return
    import torchtitan.distributed.parallel_dims as parallel_dims  # pylint: disable=g-import-not-at-top

    parallel_dims.device_type = device.type
    plain = _build(
        stacked=False,
        config=_moe_config(num_experts=2),
        device=device,
        world_size=world_size,
    )
    stacked = _build(
        stacked=True,
        flatten=True,
        config=_moe_config(num_experts=2),
        device=device,
        world_size=world_size,
    )
    holder = getattr(stacked, STACK_ATTR)
    assert holder.layer_shapes, "nothing was flattened, so this checks nothing"

    # The two arms initialize on different meshes, so their random weights
    # differ; copy the plain one's. `distribute_tensor` places each whole
    # tensor on the stacked view's own mesh -- the split mesh, for a flattened
    # layer -- and `src_data_rank=None` slices locally, since every rank already
    # holds the whole tensor.
    want = {
        k: v.full_tensor() if isinstance(v, DTensor) else v
        for k, v in plain.state_dict().items()
    }
    stacked.load_state_dict({
        k: distribute_tensor(want[k], v.device_mesh, v.placements, src_data_rank=None)
        if isinstance(v, DTensor)
        else want[k]
        for k, v in stacked.state_dict().items()
    })

    tokens = torch.randint(0, plain.config.vocab_size, (TOKENS,), device=device)
    # `TOKENS` packs `TOKENS // SEQ_LEN` sequences; `_moe_config`'s RoPE cache
    # only covers `SEQ_LEN` positions.
    positions = torch.arange(SEQ_LEN, device=device).repeat(TOKENS // SEQ_LEN)
    # One forward at a time, so only one model's gathered weights are live.
    expected = plain(tokens, positions).detach()
    got = stacked(tokens, positions).detach()
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def stacks_are_sharded_off_the_layer_axis(device, rank, world_size):
    """Sharding dim 0 would split one layer's weight across ranks.

    Worth repeating on a real mesh: with ``n_layers`` below the rank count,
    which is the usual case, a dim-0 shard silently falls back to ``Replicate``.
    """
    del rank
    _, stacked = _models(device, world_size)
    holder = getattr(stacked, STACK_ATTR)
    specs = getattr(holder, SPECS_ATTR)
    for name, _ in holder.named_parameters(recurse=False):
        placements = specs[name].placements
        assert placements == (Shard(1),), f"{name} is {placements}"


def _shard_stacks(holder, mesh, mode, *, shard_dim, param_dtype=None, reduce_dtype=None):
    """Shard a bare holder's stacks and return the scan body's per-row plan."""
    shard_module(
        holder, mesh, mode, shard_dim=shard_dim, param_dtype=param_dtype, reduce_dtype=reduce_dtype
    )
    specs = getattr(holder, SPECS_ATTR)
    return specs, scan_plan(specs)


def a_replicated_stack_still_sums_its_gradient(device, rank, world_size):
    """A stack too narrow to shard must still have its gradient summed.

    ``shard_module`` downgrades ``Shard`` to ``Replicate`` when dim 1 is below
    the mesh size -- a norm weight on a wide mesh. There is then no gather for
    the body to differentiate through, and the body has to ask for the sum
    itself. Skip it and every rank steps on its own microbatch's gradient,
    silently.
    """
    if world_size == 1:
        logging.warning("skipped: one rank, nothing to downgrade or sum")
        return
    mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("dp_shard",))
    holder = nn.Module()
    # Dim 1 is 1, so there is no way to split it across the ranks.
    holder.register_parameter(
        "narrow", nn.Parameter(torch.zeros(2, 1, 4, device=device))
    )
    specs, plan = _shard_stacks(holder, mesh, "fully_shard", shard_dim=1)

    param = holder._parameters["narrow"]
    assert specs["narrow"].placements == (Replicate(),), specs["narrow"].placements
    replica = param.detach().clone().requires_grad_(True)

    # Scaling by the rank makes an unsummed gradient visibly this rank's own.
    weight = unshard(replica[0], plan["narrow"])
    (weight.sum() * (rank + 1)).backward()

    summed = float(world_size * (world_size + 1) // 2)
    torch.testing.assert_close(
        replica.grad[0], torch.full_like(replica.grad[0], summed)
    )


def a_sharded_stack_gathers_and_reduces_like_unrolled(
    device, rank, world_size, shard_dim=1
):
    """The body's gather must compute what DTensor's redistribute does.

    ``_AllGatherReduceScatter`` hand-writes the pair ``redistribute`` runs, so
    the two can drift apart: a gather on the wrong axis, a reduce that never
    runs, a reduce in the wrong dtype. On one rank both collectives are the
    identity and every version passes, so this only means anything on a real
    mesh.
    """
    if world_size == 1:
        logging.warning("skipped: one rank, both collectives are the identity")
        return
    mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("dp_shard",))
    holder = nn.Module()
    # Two layers, and two axes the mesh splits evenly.
    holder.register_parameter(
        "stack",
        nn.Parameter(torch.randn(2, 4 * world_size, 2 * world_size, device=device)),
    )
    specs, plan = _shard_stacks(
        holder,
        mesh,
        "fully_shard",
        shard_dim=shard_dim,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    param = holder._parameters["stack"]
    assert specs["stack"].placements == (Shard(shard_dim),), specs["stack"].placements

    # Unrolled: the weight DTensor's redistribute hands layer 0.
    want_shard = param.detach().clone().requires_grad_(True)
    want = (
        specs["stack"].to_dtensor(want_shard)
        .redistribute(
            placements=[Replicate()],
            forward_dtype=torch.bfloat16,
            backward_dtype=torch.float32,
        )
        .to_local(grad_placements=[Partial()])[0]
    )

    # Scanned: what the body computes for the same row.
    got_shard = param.detach().clone().requires_grad_(True)
    got = funcol.wait_tensor(unshard(got_shard[0], plan["stack"]))

    assert got.dtype == torch.bfloat16, got.dtype
    torch.testing.assert_close(got, want, rtol=0, atol=0)

    # Scaling by the rank makes an unsummed gradient visibly this rank's own.
    for gathered in (want, got):
        (gathered.float().sum() * (rank + 1)).backward()
    torch.testing.assert_close(got_shard.grad, want_shard.grad, rtol=0, atol=0)


def a_doubly_replicated_stack_sums_on_both_axes(device, rank, world_size):
    """A stack too narrow to shard is `Partial` on both axes, not one.

    `shard_module` downgrades the shard placement to `Replicate` when dim 1 is
    below the shard degree -- a norm weight on any wide mesh, so this is the
    common case and not a corner. Under `hybrid_shard` that leaves the weight
    replicated twice over and a gradient owing a sum to each axis. Sum only the
    shard axis and each replicate peer keeps its own microbatch's gradient, so
    the two halves of the mesh hold different weights and nothing says so.
    """
    if world_size < 4 or world_size % 2:
        logging.warning("skipped: hybrid_shard needs an even mesh of at least 4")
        return
    shard_degree = world_size // 2
    mesh = init_device_mesh(
        device.type,
        (2, shard_degree),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    holder = nn.Module()
    # Dim 1 is 1, so there is no way to split it across the shard axis.
    holder.register_parameter(
        "narrow", nn.Parameter(torch.randn(2, 1, 4, device=device))
    )
    specs, plan = _shard_stacks(holder, mesh, "hybrid_shard", shard_dim=1)

    param = holder._parameters["narrow"]
    assert specs["narrow"].placements == (Replicate(), Replicate()), specs["narrow"].placements

    # Unrolled: replicated already, so DTensor's only contribution is asking
    # for the sum on both axes on the way back.
    want_local = param.detach().clone().requires_grad_(True)
    want = DTensor.from_local(
        want_local, mesh, (Replicate(), Replicate())
    ).to_local(grad_placements=[Partial(), Partial()])[0]

    # Scanned: what the body computes for the same row.
    got_local = param.detach().clone().requires_grad_(True)
    got = funcol.wait_tensor(unshard(got_local[0], plan["narrow"]))
    torch.testing.assert_close(got, want, rtol=0, atol=0)

    # Scaling by the rank makes an unsummed gradient visibly this rank's own.
    for gathered in (want, got):
        (gathered.sum() * (rank + 1)).backward()
    torch.testing.assert_close(got_local.grad, want_local.grad, rtol=0, atol=0)


def a_hybrid_sharded_stack_sums_over_both_axes(device, rank, world_size):
    """Under ``hybrid_shard`` the gradient owes a sum on two axes, not one.

    The shard axis contributes the reduce-scatter the 1-D path already does.
    The replicate axis contributes an all-reduce on top, because every rank in
    it computed the same weight's gradient from a different microbatch. Drop
    that second sum and the run still trains and still reports a falling loss,
    on gradients that are `1/dp_replicate` of the truth.
    """
    if world_size < 4 or world_size % 2:
        logging.warning("skipped: hybrid_shard needs an even mesh of at least 4")
        return
    shard_degree = world_size // 2
    mesh = init_device_mesh(
        device.type,
        (2, shard_degree),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    holder = nn.Module()
    # Dim 1 divides by the shard axis but not by the whole mesh, which is the
    # case a plan that reads `mesh.size()` instead of the shard degree refuses.
    holder.register_parameter(
        "wide", nn.Parameter(torch.randn(2, shard_degree, 4, device=device))
    )
    specs, plan = _shard_stacks(holder, mesh, "hybrid_shard", shard_dim=1)

    param = holder._parameters["wide"]
    assert specs["wide"].placements == (Replicate(), Shard(1)), specs["wide"].placements

    # Unrolled: the weight DTensor's redistribute hands layer 0.
    want_shard = param.detach().clone().requires_grad_(True)
    want = (
        specs["wide"].to_dtensor(want_shard)
        .redistribute(placements=[Replicate(), Replicate()])
        .to_local(grad_placements=[Partial(), Partial()])[0]
    )

    # Scanned: what the body computes for the same row.
    got_shard = param.detach().clone().requires_grad_(True)
    got = funcol.wait_tensor(unshard(got_shard[0], plan["wide"]))
    torch.testing.assert_close(got, want, rtol=0, atol=0)

    # Scaling by the rank makes an unsummed gradient visibly this rank's own.
    for gathered in (want, got):
        (gathered.sum() * (rank + 1)).backward()
    torch.testing.assert_close(got_shard.grad, want_shard.grad, rtol=0, atol=0)

def an_unevenly_sharded_stack_is_refused(device, rank, world_size):
    """A stack the mesh cannot split evenly must fail while the plan is built.

    ``shard_module`` leaves the last rank a short shard, and the body's
    ``all_gather`` needs equal ones. Unrolled it still trains, because the
    module-attribute gather pads around its collectives. So only the scan path
    has to refuse, and it should do that here rather than at a collective in
    the loop.
    """
    del rank
    if world_size == 1:
        logging.warning("skipped: one rank, every size divides by one")
        return
    mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("dp_shard",))
    holder = nn.Module()
    # Above the mesh size, so it is sharded rather than downgraded, and one row
    # too many to split.
    holder.register_parameter(
        "uneven", nn.Parameter(torch.zeros(2, world_size + 1, 4, device=device))
    )
    shard_module(holder, mesh, "fully_shard", shard_dim=1, param_dtype=None, reduce_dtype=None)
    spec = getattr(holder, SPECS_ATTR)["uneven"]
    assert spec.placements == (Shard(1),), spec.placements
    try:
        scan_plan({"uneven": spec})
    except ValueError as err:
        assert "evenly" in str(err), err
    else:
        raise AssertionError("an unevenly sharded stack was planned")


def an_unevenly_sharded_hybrid_stack_is_refused(device, rank, world_size):
    """The 1-D refusal has to survive the second mesh axis.

    Only the last axis shards the tensor, so only its degree decides whether the
    shards come out equal. A plan that asks the whole mesh instead gets the
    answer wrong in both directions.
    """
    del rank
    if world_size < 4 or world_size % 2:
        logging.warning("skipped: hybrid_shard needs an even mesh of at least 4")
        return
    shard_degree = world_size // 2
    mesh = init_device_mesh(
        device.type,
        (2, shard_degree),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    holder = nn.Module()
    # One row too many for the shard axis to split.
    holder.register_parameter(
        "uneven", nn.Parameter(torch.zeros(2, shard_degree + 1, 4, device=device))
    )
    shard_module(holder, mesh, "hybrid_shard", shard_dim=1, param_dtype=None, reduce_dtype=None)
    spec = getattr(holder, SPECS_ATTR)["uneven"]
    assert spec.placements == (Replicate(), Shard(1)), spec.placements
    try:
        scan_plan({"uneven": spec})
    except ValueError as err:
        # The degree it names has to be the shard axis, not the whole mesh:
        # asking the mesh refuses sizes that shard fine.
        assert f"{shard_degree} ranks" in str(err), err
    else:
        raise AssertionError("an unevenly sharded hybrid stack was planned")

class TestLayerStackDistributed(
    base_distributed_device_test.BaseDistributedDeviceTest
):
    """Each check runs inside the spawned workers, so it asserts there too.

    The launcher returns nothing, so a failure arrives as the child's re-raised
    exception, not as a diff printed here.
    """

    def _on_devices(self, worker):
        distributed.run_distributed(
            self.num_devices, self.accelerator_device_type, worker
        )

    def test_forward_matches_unrolled(self):
        self._on_devices(forward_matches_unrolled)

    def test_flattened_forward_matches_unrolled(self):
        self._on_devices(flattened_forward_matches_unrolled)

    def test_stacks_are_sharded_off_the_layer_axis(self):
        self._on_devices(stacks_are_sharded_off_the_layer_axis)

    def test_a_replicated_stack_still_sums_its_gradient(self):
        self._on_devices(a_replicated_stack_still_sums_its_gradient)

    def test_a_sharded_stack_gathers_and_reduces_like_unrolled(self):
        self._on_devices(a_sharded_stack_gathers_and_reduces_like_unrolled)

    def test_a_stack_sharded_off_dim_one_gathers_the_same_way(self):
        # What an expert stack falls back to once the mesh outgrows num_experts.
        self._on_devices(
            functools.partial(
                a_sharded_stack_gathers_and_reduces_like_unrolled, shard_dim=2
            )
        )

    def test_an_unevenly_sharded_stack_is_refused(self):
        self._on_devices(an_unevenly_sharded_stack_is_refused)

    def test_a_hybrid_sharded_stack_sums_over_both_axes(self):
        self._on_devices(a_hybrid_sharded_stack_sums_over_both_axes)

    def test_a_doubly_replicated_stack_sums_on_both_axes(self):
        self._on_devices(a_doubly_replicated_stack_sums_on_both_axes)

    def test_an_unevenly_sharded_hybrid_stack_is_refused(self):
        self._on_devices(an_unevenly_sharded_hybrid_stack_is_refused)


if __name__ == "__main__":
    absltest.main()
