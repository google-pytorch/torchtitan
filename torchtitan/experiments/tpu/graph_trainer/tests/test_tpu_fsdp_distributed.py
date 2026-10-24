"""Explicit TPU FSDP must compute what SimpleFSDP's DTensor path computes.

Every check compares against DTensor's own ``redistribute`` (the collective
pair SimpleFSDP ran) or against a SimpleFSDP-sharded model built the same way,
on ``--device`` across ``--num_devices`` ranks. On one rank both collectives
are the identity and every version passes, so these only mean anything on a
real mesh; each worker skips itself with a warning when the mesh is too small.
"""

import functools

from absl import logging
from absl.testing import absltest
import torch
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu.graph_trainer import tpu_fsdp
from torchtitan.experiments.tpu.graph_trainer.layer_stack import STACK_ATTR
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import gather_slice
from torchtitan.experiments.tpu.graph_trainer.tests import test_layer_stack
from torchtitan.experiments.tpu.graph_trainer.tpu_fsdp import (
    dtensor_params,
    shard_module,
    SPECS_ATTR,
)


def _wait(t: torch.Tensor) -> torch.Tensor:
    return funcol.wait_tensor(t) if isinstance(t, funcol.AsyncCollectiveTensor) else t


def _randn(*shape, device, seed=0) -> torch.Tensor:
    """A random tensor that is the same on every rank.

    Spawned workers seed their default generator independently, and
    ``torch.manual_seed`` would also initialize the TPU in every process, so
    draw on a private CPU generator and move the result.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen).to(device)


def _mesh(device, world_size, *, hybrid):
    if hybrid:
        return init_device_mesh(
            device.type, (2, world_size // 2), mesh_dim_names=("dp_replicate", "dp_shard")
        )
    return init_device_mesh(device.type, (world_size,), mesh_dim_names=("dp_shard",))


def _skip_small(world_size, *, hybrid) -> bool:
    if world_size == 1 or (hybrid and (world_size < 4 or world_size % 2)):
        logging.warning("skipped: needs 2 ranks, or an even mesh of 4 for HSDP")
        return True
    return False


def shard_matches_dtensor(
    device, rank, world_size, *, hybrid, shard_dim, rows_per_rank, param_dtype
):
    """The gathered weight and the shard's gradient match DTensor's redistribute.

    ``rows_per_rank`` sizes the shard axis; a non-integer value makes the
    split uneven, which exercises the padding around the collectives that
    ``redistribute`` does internally.
    """
    if _skip_small(world_size, hybrid=hybrid):
        return
    mesh = _mesh(device, world_size, hybrid=hybrid)
    degree = mesh.size(mesh.ndim - 1)
    size = int(rows_per_rank * degree)
    shape = [6, 8]
    shape[shard_dim] = size
    module = nn.Module()
    module.register_parameter("weight", nn.Parameter(_randn(*shape, device=device)))
    whole = module.weight.detach().clone()
    shard_module(
        module,
        mesh,
        "hybrid_shard" if hybrid else "fully_shard",
        shard_dim=shard_dim,
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
    )
    spec = getattr(module, SPECS_ATTR)["weight"]
    want_placements = ((Replicate(),) if hybrid else ()) + (Shard(shard_dim),)
    assert spec.placements == want_placements, spec.placements
    shard = module._parameters["weight"]
    assert not isinstance(shard, DTensor)

    # Reference: the same shard through DTensor, exactly as SimpleFSDP ran it.
    want_shard = shard.detach().clone().requires_grad_(True)
    want = (
        DTensor.from_local(
            want_shard,
            mesh,
            want_placements,
            run_check=False,
            shape=torch.Size(shape),
            stride=torch.empty(shape, device="meta").stride(),
        )
        .redistribute(
            placements=[Replicate()] * mesh.ndim,
            forward_dtype=param_dtype,
            backward_dtype=torch.float32,
        )
        .to_local(grad_placements=[Partial()] * mesh.ndim)
    )
    got = _wait(module.weight)
    assert got.dtype == (param_dtype or torch.float32), got.dtype
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    torch.testing.assert_close(got.float(), whole.to(got.dtype).float(), rtol=0, atol=0)

    # Scaling by the rank makes an unsummed gradient visibly this rank's own.
    for gathered in (want, got):
        (gathered.float().sum() * (rank + 1)).backward()
    torch.testing.assert_close(shard.grad, want_shard.grad, rtol=0, atol=0)
    assert shard.grad.dtype == torch.float32, shard.grad.dtype


def narrow_weight_is_replicated_and_summed(device, rank, world_size, *, hybrid):
    """A weight too narrow to shard is replicated, and its gradient still summed.

    SimpleFSDP downgrades ``Shard`` to ``Replicate`` when the axis is below the
    shard degree -- a norm weight on any wide mesh. Under HSDP the gradient
    then owes a sum on both axes.
    """
    if _skip_small(world_size, hybrid=hybrid):
        return
    mesh = _mesh(device, world_size, hybrid=hybrid)
    module = nn.Module()
    module.register_parameter("w", nn.Parameter(_randn(1, 4, device=device)))
    shard_module(
        module,
        mesh,
        "hybrid_shard" if hybrid else "fully_shard",
        shard_dim=0,
        param_dtype=None,
        reduce_dtype=None,
    )
    spec = getattr(module, SPECS_ATTR)["w"]
    assert spec.placements == (Replicate(),) * mesh.ndim, spec.placements
    param = module._parameters["w"]

    want_local = param.detach().clone().requires_grad_(True)
    want = DTensor.from_local(
        want_local, mesh, (Replicate(),) * mesh.ndim, run_check=False
    ).to_local(grad_placements=[Partial()] * mesh.ndim)
    got = module.w
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    for gathered in (want, got):
        (gathered.sum() * (rank + 1)).backward()
    torch.testing.assert_close(param.grad, want_local.grad, rtol=0, atol=0)


def dtensor_window_shares_storage(device, rank, world_size):
    """Inside ``dtensor_params`` the attribute is a DTensor view onto the shard."""
    del rank
    if world_size == 1:
        logging.warning("skipped: one rank, nothing is sharded")
        return
    mesh = _mesh(device, world_size, hybrid=False)
    module = nn.Module()
    module.register_parameter(
        "weight", nn.Parameter(torch.zeros(2 * world_size, 4, device=device))
    )
    shard_module(
        module, mesh, "fully_shard", shard_dim=0, param_dtype=None, reduce_dtype=None
    )
    shard = module._parameters["weight"]
    with dtensor_params(module):
        view = module.weight
        assert isinstance(view, DTensor), type(view)
        assert view.placements == (Shard(0),), view.placements
        assert tuple(view.shape) == (2 * world_size, 4), view.shape
        with torch.no_grad():
            view.to_local().fill_(3.0)
    assert module._parameters["weight"] is shard
    assert not isinstance(module.weight, DTensor)
    torch.testing.assert_close(shard.detach(), torch.full_like(shard, 3.0))


def init_matches_simple_fsdp(device, rank, world_size, *, stacked, flatten=False):
    """Seeded initialization gives the same weights under both implementations.

    Checked through ``state_dict`` as whole tensors, which is also the
    checkpoint format, so the two write interchangeable checkpoints.
    """
    del rank
    if flatten and (world_size < 4 or world_size % 2):
        logging.warning("skipped: 2 experts flatten only on an even mesh of >= 4")
        return
    import torchtitan.distributed.parallel_dims as parallel_dims  # pylint: disable=g-import-not-at-top

    parallel_dims.device_type = device.type
    config = test_layer_stack._moe_config(num_experts=2) if flatten else None  # pylint: disable=protected-access
    models = {}
    for fsdp in ("simple_fsdp", "explicit"):
        models[fsdp] = test_layer_stack._build(  # pylint: disable=protected-access
            stacked=stacked,
            flatten=flatten,
            config=config,
            device=device,
            world_size=world_size,
            fsdp=fsdp,
        )
    want = {k: v.full_tensor() for k, v in models["simple_fsdp"].state_dict().items()}
    got = {k: v.full_tensor() for k, v in models["explicit"].state_dict().items()}
    assert set(want) == set(got), set(want) ^ set(got)
    for key in want:
        torch.testing.assert_close(got[key], want[key], rtol=0, atol=0, msg=key)
    # The explicit model holds plain shards at rest, and none of them is a DTensor.
    explicit = models["explicit"]
    assert all(not isinstance(p, DTensor) for p in explicit.parameters())
    assert tpu_fsdp.uses_tpu_fsdp(explicit)


def forward_and_grads_match_simple_fsdp(device, rank, world_size, *, scan):
    """Same weights in, same logits and same shard gradients out."""
    del rank
    if world_size == 1:
        logging.warning("skipped: one rank, nothing is gathered")
        return
    import torchtitan.distributed.parallel_dims as parallel_dims  # pylint: disable=g-import-not-at-top

    parallel_dims.device_type = device.type
    models = {
        fsdp: test_layer_stack._build(  # pylint: disable=protected-access
            stacked=scan, scan=scan, device=device, world_size=world_size, fsdp=fsdp
        )
        for fsdp in ("simple_fsdp", "explicit")
    }
    tokens = torch.randint(
        0, models["explicit"].config.vocab_size, (test_layer_stack.TOKENS,), device=device
    )
    outs = {}
    for fsdp, model in models.items():
        out = model(tokens)
        outs[fsdp] = out.detach()
        out.float().sum().backward()
    torch.testing.assert_close(outs["explicit"], outs["simple_fsdp"], rtol=0, atol=0)
    want = {
        name: p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
        for name, p in models["simple_fsdp"].named_parameters()
    }
    for name, p in models["explicit"].named_parameters():
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, want[name], rtol=0, atol=0, msg=name)
    if scan:
        holder = getattr(models["explicit"], STACK_ATTR)
        assert getattr(holder, SPECS_ATTR), "the stacks were not sharded explicitly"


def grad_norm_matches_dtensor(device, rank, world_size, *, hybrid):
    """The explicit global norm equals the DTensor-derived one on the same grads."""
    if _skip_small(world_size, hybrid=hybrid):
        return
    mesh = _mesh(device, world_size, hybrid=hybrid)
    mode = "hybrid_shard" if hybrid else "fully_shard"
    module = nn.Module()
    module.register_parameter(
        "wide", nn.Parameter(_randn(4 * world_size, 3, device=device))
    )
    module.register_parameter("narrow", nn.Parameter(_randn(1, 5, device=device)))
    shard_module(module, mesh, mode, shard_dim=0, param_dtype=None, reduce_dtype=None)
    specs = getattr(module, SPECS_ATTR)
    grads, spec_list, dgrads = [], [], []
    for name in ("wide", "narrow"):
        shard = module._parameters[name]
        g = _randn(*shard.shape, device=device, seed=100 + rank)
        if specs[name].shard_dim is None:
            # A replicated gradient is identical within the shard group after
            # backward (and across the replicate axis under HSDP).
            g = _wait(funcol.all_reduce(g, "sum", group=mesh.get_group(mesh.ndim - 1)))
            if mesh.ndim == 2:
                g = _wait(funcol.all_reduce(g, "sum", group=mesh.get_group(0)))
            g = g.clone()
        grads.append(g)
        spec_list.append(specs[name])
        dgrads.append(specs[name].to_dtensor(g))
    got = _wait(tpu_fsdp.global_grad_norm(grads, spec_list))
    want = torch.nn.utils.get_total_norm(dgrads, 2.0, False, True).full_tensor()
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


def scan_row_gather_matches_dtensor(device, rank, world_size, *, shard_dim):
    """The scan body's per-row gather from a spec matches SimpleFSDP unrolled."""
    if world_size == 1:
        logging.warning("skipped: one rank, both collectives are the identity")
        return
    mesh = _mesh(device, world_size, hybrid=False)
    holder = nn.Module()
    holder.register_parameter(
        "stack",
        nn.Parameter(_randn(2, 4 * world_size, 2 * world_size, device=device)),
    )
    shard_module(
        holder,
        mesh,
        "fully_shard",
        shard_dim=shard_dim,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    spec = getattr(holder, SPECS_ATTR)["stack"]
    shard = holder._parameters["stack"]

    want_shard = shard.detach().clone().requires_grad_(True)
    want = (
        spec.to_dtensor(want_shard)
        .redistribute(
            placements=[Replicate()],
            forward_dtype=torch.bfloat16,
            backward_dtype=torch.float32,
        )
        .to_local(grad_placements=[Partial()])[0]
    )
    got_shard = shard.detach().clone().requires_grad_(True)
    got = _wait(gather_slice(got_shard[0], spec.layer_slice().scan_spec()))
    assert got.dtype == torch.bfloat16, got.dtype
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    for gathered in (want, got):
        (gathered.float().sum() * (rank + 1)).backward()
    torch.testing.assert_close(got_shard.grad, want_shard.grad, rtol=0, atol=0)


class TestTpuFSDP(base_distributed_device_test.BaseDistributedDeviceTest):

    def _on_devices(self, worker):
        distributed.run_distributed(
            self.num_devices, self.accelerator_device_type, worker
        )

    def test_fsdp_dim0_bf16_matches_dtensor(self):
        self._on_devices(
            functools.partial(
                shard_matches_dtensor,
                hybrid=False,
                shard_dim=0,
                rows_per_rank=2,
                param_dtype=torch.bfloat16,
            )
        )

    def test_fsdp_dim1_matches_dtensor(self):
        self._on_devices(
            functools.partial(
                shard_matches_dtensor,
                hybrid=False,
                shard_dim=1,
                rows_per_rank=2,
                param_dtype=None,
            )
        )

    def test_uneven_shard_matches_dtensor(self):
        # 1.5 rows per rank: the last rank gets fewer rows (or none at 2 ranks
        # it gets one), and the gather has to pad around the collective.
        self._on_devices(
            functools.partial(
                shard_matches_dtensor,
                hybrid=False,
                shard_dim=0,
                rows_per_rank=1.5,
                param_dtype=torch.bfloat16,
            )
        )

    def test_hsdp_matches_dtensor(self):
        self._on_devices(
            functools.partial(
                shard_matches_dtensor,
                hybrid=True,
                shard_dim=1,
                rows_per_rank=2,
                param_dtype=torch.bfloat16,
            )
        )

    def test_narrow_weight_fsdp(self):
        self._on_devices(
            functools.partial(narrow_weight_is_replicated_and_summed, hybrid=False)
        )

    def test_narrow_weight_hsdp(self):
        self._on_devices(
            functools.partial(narrow_weight_is_replicated_and_summed, hybrid=True)
        )

    def test_dtensor_window_shares_storage(self):
        self._on_devices(dtensor_window_shares_storage)

    def test_init_matches_simple_fsdp_unrolled(self):
        self._on_devices(functools.partial(init_matches_simple_fsdp, stacked=False))

    def test_init_matches_simple_fsdp_stacked(self):
        self._on_devices(functools.partial(init_matches_simple_fsdp, stacked=True))

    def test_init_matches_simple_fsdp_flattened_stacks(self):
        self._on_devices(
            functools.partial(init_matches_simple_fsdp, stacked=True, flatten=True)
        )

    def test_forward_and_grads_match_simple_fsdp(self):
        self._on_devices(
            functools.partial(forward_and_grads_match_simple_fsdp, scan=False)
        )

    def test_scan_forward_and_grads_match_simple_fsdp(self):
        self._on_devices(
            functools.partial(forward_and_grads_match_simple_fsdp, scan=True)
        )

    def test_grad_norm_fsdp(self):
        self._on_devices(functools.partial(grad_norm_matches_dtensor, hybrid=False))

    def test_grad_norm_hsdp(self):
        self._on_devices(functools.partial(grad_norm_matches_dtensor, hybrid=True))

    def test_scan_row_gather_dim1(self):
        self._on_devices(functools.partial(scan_row_gather_matches_dtensor, shard_dim=1))

    def test_scan_row_gather_dim2(self):
        self._on_devices(functools.partial(scan_row_gather_matches_dtensor, shard_dim=2))


if __name__ == "__main__":
    absltest.main()
