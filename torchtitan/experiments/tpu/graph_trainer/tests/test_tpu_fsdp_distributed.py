"""The TPU FSDP must compute what DTensor's collectives compute.

Every per-parameter check compares against DTensor's own ``redistribute``
(the gather / reduce-scatter pair, with its dtypes and padding), and every
model-level check against the same model with DTensor parameters (for the
seeded init) or holding the whole weights unsharded (for the forward and the
gradients), on ``--device`` across ``--num_devices`` ranks. On one rank both
collectives are the identity and every version passes, so these only mean
anything on a real mesh; each worker skips itself with a warning when the mesh
is too small.
"""

import functools
import os
import tempfile

from absl import logging
from absl.testing import absltest
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)

from torchtitan.components.optimizer.utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu import grad_norm
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


def _distribute_like(reference: nn.Module, sharded: nn.Module) -> None:
    """Give ``reference``'s parameters the DTensor placements ``sharded`` uses."""
    modules = dict(reference.named_modules())
    for name, module in sharded.named_modules():
        specs = getattr(module, SPECS_ATTR, None)
        if not specs:
            continue
        target = modules[name]
        for pname, spec in specs.items():
            param = target._parameters[pname]
            target._parameters[pname] = nn.Parameter(
                distribute_tensor(param.detach(), spec.mesh, spec.placements),
                requires_grad=param.requires_grad,
            )


def _whole(state_dict) -> dict[str, torch.Tensor]:
    return {
        k: v.full_tensor() if isinstance(v, DTensor) else v
        for k, v in state_dict.items()
    }


def init_matches_dtensor_init(device, rank, world_size):
    """Seeded init of plain shards equals the init of the same DTensor parameters.

    The init runs inside the DTensor window, so it is DTensor's sharded RNG in
    both cases; this pins that the window hands the initializer the same
    placements the sharded model would have had as DTensors, unrolled. Checked
    through ``state_dict`` as whole tensors, which is also the checkpoint
    format.
    """
    del rank
    if world_size == 1:
        logging.warning("skipped: one rank, nothing is sharded")
        return
    import torchtitan.distributed.parallel_dims as parallel_dims  # pylint: disable=g-import-not-at-top

    parallel_dims.device_type = device.type
    explicit = test_layer_stack._build(  # pylint: disable=protected-access
        stacked=False, device=device, world_size=world_size
    )
    reference = test_layer_stack._build(  # pylint: disable=protected-access
        stacked=False,
        device=device,
        world_size=world_size,
        shard=False,
        prepare=functools.partial(_distribute_like, sharded=explicit),
    )
    want = _whole(reference.state_dict())
    got = _whole(explicit.state_dict())
    assert set(want) == set(got), set(want) ^ set(got)
    for key in want:
        torch.testing.assert_close(got[key], want[key], rtol=0, atol=0, msg=key)
    assert all(not isinstance(p, DTensor) for p in explicit.parameters())
    assert tpu_fsdp.uses_tpu_fsdp(explicit)


def stacked_init_matches_unrolled(device, rank, world_size):
    """Sharded stacks initialize to what the unrolled sharded model does.

    The per-layer DTensor views the stack puts up for ``init_states`` have to
    describe the same rows the unrolled model initializes, or the seeded draws
    land in different places. Only where the two models give a weight the
    same placements: without DTensor's offset-based RNG tracker (CPU and TPU
    have none) a sharded draw depends on the placement, so a flattened stack
    (``Shard(1)`` of ``[L, E*X, ...]``) and its unrolled ``[E, X, ...]`` expert
    cannot be compared this way; ``test_layer_stack_distributed`` checks the
    flattened views by loading weights through them instead.
    """
    del rank
    if world_size == 1:
        logging.warning("skipped: one rank, nothing is sharded")
        return
    import torchtitan.distributed.parallel_dims as parallel_dims  # pylint: disable=g-import-not-at-top

    parallel_dims.device_type = device.type
    unrolled = test_layer_stack._build(  # pylint: disable=protected-access
        stacked=False, device=device, world_size=world_size
    )
    stacked = test_layer_stack._build(  # pylint: disable=protected-access
        stacked=True, device=device, world_size=world_size
    )
    want = _whole(unrolled.state_dict())
    got = _whole(stacked.state_dict())
    assert set(want) == set(got), set(want) ^ set(got)
    for key in want:
        torch.testing.assert_close(got[key], want[key], rtol=0, atol=0, msg=key)


def forward_and_grads_match_unsharded(device, rank, world_size, *, scan, moe=False):
    """Same weights in, same logits and the summed gradient out.

    The reference holds the whole weights on every rank (no FSDP) and is
    loaded from the sharded model's checkpoint. Every rank runs the same
    tokens, so the sharded model's reduce-scatter (or all-reduce, for a
    replicated weight) hands each shard ``world_size`` times the reference's
    gradient, on this rank's rows. ``moe`` uses the 2-layer qwen3 MoE (TPU
    grouped experts, router gate); the llama3 debug model covers the dense
    path. Unrolled only: the test MoE config keeps the aux-loss routing metrics
    on, which a scan body cannot carry. ``scan`` runs on CPU only: eager
    ``scan`` does not compute the layers on TPU (the model has to be traced),
    and the traced scan is graded against the loop end to end by
    ``test_tpu_graph_trainer_distributed.test_scan_matches_the_layer_loop``.
    """
    del rank
    if world_size == 1:
        logging.warning("skipped: one rank, nothing is gathered")
        return
    if scan and device.type != "cpu":
        logging.warning("skipped: eager scan is CPU-only; the traced scan is covered e2e")
        return
    import torchtitan.distributed.parallel_dims as parallel_dims  # pylint: disable=g-import-not-at-top

    parallel_dims.device_type = device.type
    config = test_layer_stack._moe_config(num_experts=2) if moe else None  # pylint: disable=protected-access
    sharded = test_layer_stack._build(  # pylint: disable=protected-access
        stacked=scan, scan=scan, config=config, device=device, world_size=world_size
    )
    reference = test_layer_stack._build(  # pylint: disable=protected-access
        stacked=False, config=config, device=device, world_size=world_size, shard=False
    )
    reference.load_state_dict(_whole(sharded.state_dict()))

    tokens = torch.randint(
        0, sharded.config.vocab_size, (test_layer_stack.TOKENS,),
        generator=torch.Generator().manual_seed(7),
    ).to(device)
    positions = (
        torch.arange(test_layer_stack.SEQ_LEN, device=device).repeat(
            test_layer_stack.TOKENS // test_layer_stack.SEQ_LEN
        )
        if moe
        else None
    )
    outs = {}
    for arm, model in (("sharded", sharded), ("reference", reference)):
        out = model(tokens, positions) if moe else model(tokens)
        outs[arm] = out.detach()
        out.float().sum().backward()
    torch.testing.assert_close(outs["sharded"], outs["reference"], rtol=0, atol=0)

    # The reference's gradient of every stack row, as a stack, or of the
    # parameter itself, keyed by the sharded model's parameter names.
    ref_params = dict(reference.named_parameters())
    n_layers = len(reference.layers)

    def reference_grad(name: str) -> torch.Tensor:
        if name.startswith(f"{STACK_ATTR}."):
            per_layer = name[len(STACK_ATTR) + 1 :].replace("__", ".")
            return torch.stack(
                [ref_params[f"layers.{i}.{per_layer}"].grad for i in range(n_layers)]
            )
        return ref_params[name].grad

    specs = tpu_fsdp.param_specs(sharded)
    mismatched = []
    for name, p in sharded.named_parameters():
        assert p.grad is not None, name
        spec = specs[id(p)]
        want = reference_grad(name) * world_size
        d = spec.shard_dim
        if d is not None:
            shape, offset = spec.local_shape_and_offset()
            want = want.narrow(d, offset[d], shape[d])
        try:
            torch.testing.assert_close(p.grad, want, rtol=1e-5, atol=5e-5)
        except AssertionError as err:
            mismatched.append(f"{name}: shape {tuple(p.grad.shape)}: {err}")
    assert not mismatched, "gradients differ from the unsharded model:\n" + "\n".join(
        mismatched
    )
    if scan:
        holder = getattr(sharded, STACK_ATTR)
        assert getattr(holder, SPECS_ATTR), "the stacks were not sharded explicitly"


def grad_norm_matches_dtensor(device, rank, world_size, *, hybrid):
    """The spec-aware global norm equals the DTensor one on the same grads."""
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
    got = _wait(grad_norm.global_grad_norm(grads, spec_list))
    want = torch.nn.utils.get_total_norm(dgrads, 2.0, False, True).full_tensor()
    # Same DTensor arithmetic on both sides, so the bits agree.
    torch.testing.assert_close(got, want, rtol=0, atol=0)


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



def optimizer_state_round_trips_through_dcp(device, rank, world_size, *, path):
    """Adam's moments are saved as shards and come back rank by rank.

    Without the state-dict hooks every rank hands DCP a plain tensor under
    the same key; DCP keeps one copy and every rank loads it.
    """
    if world_size == 1:
        logging.warning("skipped: one rank cannot tell a shard from a replica")
        return
    mesh = _mesh(device, world_size, hybrid=False)

    def build():
        module = nn.Module()
        module.register_parameter(
            "weight", nn.Parameter(_randn(4 * world_size, 8, device=device))
        )
        # Narrower than the mesh: replicated, so its moments are identical
        # across ranks and DCP keeping one copy is correct.
        module.register_parameter("scale", nn.Parameter(_randn(1, 3, device=device)))
        shard_module(
            module, mesh, "fully_shard", shard_dim=0, param_dtype=None, reduce_dtype=None
        )
        optim = torch.optim.AdamW(list(module.named_parameters()), lr=1e-3)
        init_optim_state(optim)
        tpu_fsdp.install_optimizer_state_hooks([optim], [module])
        return module, optim

    def wanted(name: str, spec) -> float:
        return float(rank + 1) if spec.sharded else 5.0

    module, optim = build()
    specs = getattr(module, SPECS_ATTR)
    for name, p in module.named_parameters():
        optim.state[p]["exp_avg"].fill_(wanted(name, specs[name]))
        optim.state[p]["exp_avg_sq"].fill_(10 * wanted(name, specs[name]))
        optim.state[p]["step"].fill_(7)
    flat = get_flat_optim_state_dict(optim)
    view = flat["state.weight.exp_avg"]
    assert isinstance(view, DTensor), type(view)
    assert view.placements == specs["weight"].placements, view.placements
    assert tuple(view.shape) == (4 * world_size, 8), view.shape
    assert not isinstance(flat["state.weight.step"], DTensor)
    # The live state stays plain: the view is for DCP only.
    assert not isinstance(optim.state[module._parameters["weight"]]["exp_avg"], DTensor)
    dcp.save(flat, checkpoint_id=os.path.join(path, "optim"))

    module2, optim2 = build()
    flat2 = get_flat_optim_state_dict(optim2)
    dcp.load(flat2, checkpoint_id=os.path.join(path, "optim"))
    load_flat_optim_state_dict(optim2, flat2)
    specs2 = getattr(module2, SPECS_ATTR)
    for name, p in module2.named_parameters():
        state = optim2.state[p]
        assert not isinstance(state["exp_avg"], DTensor), name
        torch.testing.assert_close(
            state["exp_avg"], torch.full_like(p, wanted(name, specs2[name])), rtol=0, atol=0
        )
        torch.testing.assert_close(
            state["exp_avg_sq"],
            torch.full_like(p, 10 * wanted(name, specs2[name])),
            rtol=0,
            atol=0,
        )
        assert float(state["step"]) == 7.0, state["step"]


def sharded_buffers_checkpoint_as_dtensors(device, rank, world_size):
    """A buffer registered like a sharded parameter gets that parameter's view."""
    if world_size == 1:
        logging.warning("skipped: one rank cannot tell a shard from a replica")
        return
    mesh = _mesh(device, world_size, hybrid=False)
    module = nn.Module()
    module.register_parameter(
        "weight", nn.Parameter(_randn(4 * world_size, 8, device=device))
    )
    shard_module(
        module, mesh, "fully_shard", shard_dim=0, param_dtype=None, reduce_dtype=None
    )
    spec = getattr(module, SPECS_ATTR)["weight"]
    shard = module._parameters["weight"]
    tpu_fsdp.register_sharded_buffer(
        module, "_optstate_weight_exp_avg", torch.full_like(shard, rank + 1), like="weight"
    )
    module.register_buffer("count", torch.zeros((), device=device))
    try:
        tpu_fsdp.register_sharded_buffer(
            module, "wrong", torch.zeros(3, device=device), like="weight"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("a buffer of another shape was registered like weight")

    with dtensor_params(module):
        sd = module.state_dict()
    view = sd["_optstate_weight_exp_avg"]
    assert isinstance(view, DTensor), type(view)
    assert view.placements == spec.placements, view.placements
    assert tuple(view.shape) == (4 * world_size, 8), view.shape
    assert not isinstance(sd["count"], DTensor)
    buffer = module._buffers["_optstate_weight_exp_avg"]
    assert not isinstance(buffer, DTensor), "the window did not close"
    assert view.to_local().data_ptr() == buffer.data_ptr(), "the view copied"
    torch.testing.assert_close(
        view.to_local(), torch.full_like(shard, rank + 1), rtol=0, atol=0
    )
    assert not isinstance(module.state_dict()["_optstate_weight_exp_avg"], DTensor)


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

    def test_init_matches_dtensor_init(self):
        self._on_devices(init_matches_dtensor_init)

    def test_stacked_init_matches_unrolled(self):
        self._on_devices(stacked_init_matches_unrolled)

    def test_forward_and_grads_match_unsharded(self):
        self._on_devices(
            functools.partial(forward_and_grads_match_unsharded, scan=False)
        )

    def test_scan_forward_and_grads_match_unsharded(self):
        self._on_devices(
            functools.partial(forward_and_grads_match_unsharded, scan=True)
        )

    def test_moe_forward_and_grads_match_unsharded(self):
        self._on_devices(
            functools.partial(forward_and_grads_match_unsharded, scan=False, moe=True)
        )

    def test_optimizer_state_round_trips_through_dcp(self):
        with tempfile.TemporaryDirectory() as path:
            self._on_devices(
                functools.partial(optimizer_state_round_trips_through_dcp, path=path)
            )

    def test_fused_optimizer_buffers_checkpoint_as_dtensors(self):
        self._on_devices(sharded_buffers_checkpoint_as_dtensors)

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
