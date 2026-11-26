"""Store sibling decoder layers' weights in one stacked tensor per parameter.

Each layer used to own its weights. Now there is one tensor per *kind* of
weight, with one row per layer:

    before:  layers.0.attention.wq.weight ... layers.n-1.attention.wq.weight
             n separate [d, d] tensors, one per layer
    after:   layer_stacks.attention__wq__weight
             one [n, d, d] tensor, row i belonging to layer i

A later PR runs the layers as a single ``scan``. That leading axis is the reason
to stack them at all.

What changes for a reader:

* The layer modules are still there, and still run one at a time in a Python
  ``for``. They just have no parameters of their own.
  ``layers.7.attention.wq.weight`` does not resolve, and ``named_parameters()``
  reports the stack instead.
* The per-layer names come back, as views onto their rows, in exactly three
  windows: ``init_weights``, ``state_dict()`` and ``load_state_dict()``. So
  weights initialize as they did unstacked, and checkpoints keep their
  ``layers.i....`` keys -- a checkpoint moves freely between a stacked and an
  unstacked run. Everywhere else the views are down, the forward and backward
  included: those read rows out of the stack and never restore the names.
* Anything keyed off *live* parameter names sees the difference: the optimizer
  state dict and DCP's ``get_model_state_dict`` both do. Neither is handled
  yet: optimizer state does not carry across a stacked/unstacked switch, and
  ``get_model_state_dict`` does not work on a stacked model.
  https://github.com/google-pytorch/torchtitan/issues/21
  https://github.com/google-pytorch/torchtitan/issues/23
* Stacks whose axis 1 does not divide by the FSDP mesh degree (e.g. expert
  stacks ``[n, E, X, ...]`` when the mesh degree exceeds ``E``) can be
  flattened before sharding via ``LayerStackHolder.flatten_for(degree)`` into
  ``[n, E*X, ...]``. This allows FSDP to shard along axis 1 instead of an inner
  axis, avoiding transpose copies during all-gather and reduce-scatter.
  Per-layer views in ``init_weights`` and ``state_dict()`` still see
  ``[E, X, ...]`` via DTensors on split meshes, while the scan body gathers
  contiguous rows and views them back to ``[E, X, ...]``. Live parameters (and
  thus optimizer state) reflect the flattened shape.

Stacking happens in the model's ``__init__`` (:func:`with_tpu_stacked_layers`),
which is always before FSDP shards anything.
"""

import contextlib
import functools
from collections.abc import Iterator

import torch
from torch import nn
from torch._higher_order_ops.scan import scan
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.func import functional_call

from torchtitan.config import derive
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import (
    gather_like,
    gather_slice,
    plan_stacked_gathers,
    ScanFSDPSpec,
)
from torchtitan.experiments.tpu.graph_trainer.scan_remat_policy import (
    scan_body_remat_window,
)
from torchtitan.experiments.tpu.components.layer_loop import (
    model_subclass,
    tpu_model_class,
    runs_layer_loop,
    TpuLayerLoop,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import MoE
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


__all__ = [
    "check_layers_are_stackable",
    "LayerStack",
    "StackedLayers",
    "STACK_ATTR",
    "with_tpu_stacked_layers",
]


# Where the stacks' own module hangs off the model -- see `LayerStackHolder`.
STACK_ATTR = "layer_stacks"

# `register_parameter` rejects names with a dot in them, so a per-layer FQN like
# `attention.wq.weight` can't be used as a stack's name as-is. The mangled name
# never leaves this module: the state-dict hooks below emit the real FQNs.
def _mangle(name: str) -> str:
    return name.replace(".", "__")


def _stack_key(position: int, name: str, period: int) -> str:
    """Stack name for ``name`` at ``position`` in a cycle of ``period`` layers.

    With ``period=1`` this is the plain mangled name, so existing checkpoints of
    optimizer state keep their keys. ``period=4``: ``p3__attn__wq__weight``.
    """
    return _mangle(name) if period == 1 else f"p{position}__{_mangle(name)}"


def check_layers_are_stackable(
    layers: nn.ModuleDict | dict[str, nn.Module],
) -> None:
    """Every layer must match the first, or they cannot share a stack.

    ``layers`` maps each layer's number (as a string) to the layer, so errors
    name the real layer; with ``period > 1`` it holds one position's layers.
    Below, "layer 0" means the first of them.

    Parameters become rows of one tensor, so they have to agree on name, shape,
    dtype and ``requires_grad``. Dtype matters because the stack is allocated
    from layer 0, so a layer with a different dtype would be silently cast into
    it.

    Buffers aren't stacked at all -- every layer runs with layer 0's -- so they
    have to agree on *value* too, not just shape.

    Raises:
        ValueError: if any layer differs from layer 0.
    """

    def spec(layer: nn.Module) -> dict[str, tuple]:
        named = [*layer.named_parameters(), *layer.named_buffers()]
        return {n: (tuple(t.shape), t.dtype, t.requires_grad) for n, t in named}

    (first, layer0), *_ = layers.items()
    template, zero_buffers = spec(layer0), dict(layer0.named_buffers())
    for i, layer in layers.items():
        # Load balancing keeps its state in buffers, and buffers aren't stacked,
        # so every layer would route off layer 0's. The value check above misses
        # it: the buffers start equal and only diverge during training. With no
        # `load_balance_coeff` nothing reads them, so sharing is harmless.
        if any(
            isinstance(m, MoE) and m.load_balance_coeff is not None
            for m in layer.modules()
        ):
            raise ValueError(
                f"layer {i} holds an MoE with load balancing on, and its "
                "per-layer buffers cannot be shared across layers. Stacking "
                "does not support MoE load balancing yet "
                "(https://github.com/google-pytorch/torchtitan/issues/22); "
                "drop --compile.use-scan-layers, or set "
                "load_balance_coeff to None."
            )
        got = spec(layer)
        if got != template:
            # key=str: torch.dtype is not orderable.
            differences = sorted(set(template.items()) ^ set(got.items()), key=str)
        else:
            # Skip meta: nothing to compare, and refusing there would reject
            # every real model.
            differences = [
                f"buffer {n} holds different values"
                for n, b in layer.named_buffers()
                if not (b.is_meta or zero_buffers[n].is_meta)
                and not torch.equal(zero_buffers[n], b)
            ]
        if differences:
            raise ValueError(
                f"layer {i} does not match layer {first}, so their weights cannot "
                f"share a stack: {differences}. Stacking needs identical "
                "layers "
                "(https://github.com/google-pytorch/torchtitan/issues/24)."
            )


def _shard_dim(param: torch.Tensor, degree: int) -> int:
    """Which axis of one stack `degree` ranks should split it on."""
    # Dim 0 is the scan axis; sharding it would split one layer across ranks.
    if param.ndim < 2:
        # A stack is [n_layers, *weight.shape], so this needs a 0-D weight.
        raise ValueError(
            f"cannot shard a {param.ndim}-D stack; dim 0 is the scan axis"
        )
    # Outermost first: it leaves the minor axis whole, so the body's all-gather
    # stays a contiguous copy, and it picks the same axis the unrolled path
    # would. Preferring the *widest* divisible axis instead would send w2_EFD to
    # dim 3 while w1_EDF goes to dim 2, which no other lane does.
    # For 3-D stacks (2-D per-layer weights `[n_layers, rows, cols]`), prefer an
    # axis whose per-rank shard is a multiple of 8 (the TPU MXU row tile) so
    # `scan_fsdp` does not need `_fold_rows` reshapes around collectives, unless
    # `dim 1` has one entry per rank (`num_experts == degree` on a router).
    if param.ndim == 3 and param.shape[1] != degree:
        for d in range(1, param.ndim):
            if param.shape[d] % (degree * 8) == 0:
                return d
    for d in range(1, param.ndim):
        if param.shape[d] % degree == 0:
            return d
    # Nothing divides: the widest axis. `data_parallel` replicates it if it is
    # narrower than the mesh, and `plan_stacked_gathers` refuses it if not.
    return max(range(1, param.ndim), key=lambda d: param.shape[d])


@contextlib.contextmanager
def _no_sparse_mesh_push():
    """Hides the registered EP sparse mesh while the scan body is captured.

    Under EP, the MoE code enters ``maybe_set_sparse_mesh()``, which appends to
    a thread-local mesh stack. Inside ``scan`` that is a mutation of state from
    outside the body, which Dynamo refuses (``UncapturedHigherOrderOpError``).
    The pushed mesh only feeds spmd_types type checking, which TPU GraphTrainer
    runs with off; with no sparse mesh registered the context is a no-op, as it
    is at EP = 1. The backward body is derived from the captured graph, so the
    window only has to cover the ``scan`` call.
    """
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.distributed import spmd_types as tt_spmd

    tls = tt_spmd._MESH_TLS  # pylint: disable=protected-access
    saved = getattr(tls, "sparse_mesh", None)
    if saved is None:
        yield
        return
    import spmd_types as spmd  # pylint: disable=g-import-not-at-top

    if spmd.is_type_checking():
        raise NotImplementedError(
            "scan with expert parallelism needs spmd type checking off"
        )
    tls.sparse_mesh = None
    try:
        yield
    finally:
        tls.sparse_mesh = saved


class LayerStackHolder(Module):
    """``model.layer_stacks``: the stacks, and nothing else.

    The stacks live in the model tree so they count as ordinary parameters:
    found by ``model.parameters()``, sharded by FSDP, seen by the optimizer,
    written to the checkpoint.

    A ``Module`` rather than a bare ``nn.Module``, because the model stacks
    itself in its own ``__init__`` and so the holder is already in the tree when
    ``verify_module_protocol`` walks it.

    Put nothing else in here: SimpleFSDP skips a module once it has sharded it,
    so anything else parked here would silently go unsharded.
    """

    # Dim 1 for every stack, until someone calls `plan_shard_dims` with the mesh
    # width and replaces this with an axis per stack.
    _fsdp_shard_dim: int | dict[str, int] = 1

    def __init__(self):
        super().__init__()
        # Original per-layer shape of each stack that `flatten_for` flattened,
        # keyed by parameter name. Stacks not listed here are stored unchanged.
        self.layer_shapes: dict[str, torch.Size] = {}
        # Meshes used to view one layer of a flattened stack in its original
        # shape, keyed by (FSDP mesh, size of the first per-layer axis).
        self.split_meshes: dict[tuple[DeviceMesh, int], DeviceMesh] = {}
        # Expert-parallel (EP) stacks: stack name -> the per-layer axis EP
        # shards. They skip dense SimpleFSDP and live on the expert mesh
        # instead; see `stash_expert_stacks` and `place_expert_stacks`.
        self.ep_dims: dict[str, int] = {}
        self._ep_pending: dict[str, nn.Parameter] = {}

    def stash_expert_stacks(self, ep_dims: dict[str, int]) -> None:
        """Takes the EP-sharded stacks out before dense SimpleFSDP sees them.

        Dense SimpleFSDP would shard them over every rank, and the scan body
        would then gather all experts on every rank. ``place_expert_stacks``
        puts them back on ``([dp_replicate,] efsdp, ep)`` once dense FSDP is
        done. Idempotent: it runs on each ``configure_simple_fsdp_shard_dims``.
        """
        for name, dim in ep_dims.items():
            if name in self.ep_dims:
                continue
            self._ep_pending[name] = self._parameters.pop(name)
            self.ep_dims[name] = dim

    def place_expert_stacks(self, parallel_dims) -> None:
        """Puts the stashed stacks back, laid out as the splitter lays out one layer.

        The splitter shards a layer's expert weight ``[E, X, Y]`` with
        ``Shard(0)`` on ``ep`` and then SimpleFSDP shards each EP rank's block
        again on ``efsdp`` (``_StridedShard(0, split_factor=ep)`` over the
        spanned ``(efsdp, ep)`` mesh). A stack row gets the same placement, one
        axis further in, so the rank at ``(efsdp=i, ep=j)`` holds piece ``i`` of
        EP block ``j``, and an all-gather over ``efsdp`` gives EP block ``j``
        whole: the experts this rank's EP dispatch sends tokens to.
        ``efsdp = 1`` replicates on that axis. If the expert axis does not
        split ``efsdp * ep`` ways, ``efsdp`` shards per-layer axis 1 instead,
        as ``apply_qwen3_5_simple_fsdp`` does.
        """
        if not self._ep_pending:
            return
        names = (
            ["dp_replicate", "efsdp", "ep"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp", "ep"]
        )
        mesh = parallel_dims.get_mesh(names)
        efsdp, ep = mesh["efsdp"].size(), mesh["ep"].size()
        coord = mesh.get_coordinate()
        for name, param in list(self._ep_pending.items()):
            d = self.ep_dims[name] + 1  # stack axis = per-layer axis + 1
            if param.shape[d] % ep:
                raise ValueError(
                    f"{name} has {param.shape[d]} experts on axis {d}, which "
                    f"ep={ep} cannot split evenly"
                )
            if efsdp == 1:
                efsdp_pl = Replicate()
            elif param.shape[d] % (efsdp * ep) == 0:
                efsdp_pl = _StridedShard(d, split_factor=ep)
            elif d != 2 and param.ndim > 2 and param.shape[2] % efsdp == 0:
                efsdp_pl = Shard(2)
            else:
                raise ValueError(
                    f"cannot shard {name} {tuple(param.shape)} over efsdp="
                    f"{efsdp} x ep={ep}"
                )
            placements = (
                (Replicate(),) if len(names) == 3 else ()
            ) + (efsdp_pl, Shard(d))
            local_shape = list(param.shape)
            local_shape[d] //= ep
            # `_StridedShard` is not a `Shard` subclass in this torch.
            if isinstance(efsdp_pl, (Shard, _StridedShard)):
                local_shape[efsdp_pl.dim] //= efsdp
            if param.is_meta or coord is None:
                local = torch.empty(local_shape, dtype=param.dtype, device=param.device)
            else:
                i, j = coord[-2], coord[-1]
                blk = param.shape[d] // ep
                local = param.detach().narrow(d, j * blk, blk)
                if isinstance(efsdp_pl, (Shard, _StridedShard)):
                    k = efsdp_pl.dim
                    local = local.narrow(k, i * local_shape[k], local_shape[k])
                local = local.contiguous()
            self._parameters[name] = nn.Parameter(
                DTensor.from_local(
                    local,
                    mesh,
                    placements,
                    run_check=False,
                    shape=param.shape,
                    stride=param.stride(),
                ),
                requires_grad=param.requires_grad,
            )
            del self._ep_pending[name]
        logger.info(
            "Placed %d expert stacks on %s (efsdp=%d, ep=%d)",
            len(self.ep_dims),
            names,
            efsdp,
            ep,
        )

    def layer_view(self, name: str, stack: torch.Tensor, row: int) -> torch.Tensor:
        """Row ``row`` of stack ``name`` as one layer's weight, sharing storage."""
        if name not in self.ep_dims:
            return self.as_layer_shape(name, stack[row])
        # `stack[row]` on a strided-sharded DTensor is not a view; build the row
        # from the local rows with the placements shifted one axis in.
        def shift(p):
            if isinstance(p, _StridedShard):
                return _StridedShard(p.dim - 1, split_factor=p.split_factor)
            return Shard(p.dim - 1) if isinstance(p, Shard) else p

        return DTensor.from_local(
            stack.to_local()[row],
            stack.device_mesh,
            tuple(shift(p) for p in stack.placements),
            run_check=False,
            shape=stack.shape[1:],
            stride=torch.empty(stack.shape[1:], device="meta").stride(),
        )

    def flatten_for(self, degree: int) -> dict[str, torch.Size]:
        """Flattens axes 1 and 2 of stacks that can't be sharded evenly on axis 1.

        FSDP shards a stack ``[L, A, B, ...]`` on axis 1 when ``degree`` divides
        ``A``, and the scan body then all-gathers each layer along its leading
        axis without a transpose. Otherwise ``_shard_dim`` picks an inner axis,
        and every gather and reduce-scatter in the body needs a transpose.
        Storing the stack as ``[L, A*B, ...]`` makes axis 1 divisible again at
        no cost, because the data is already contiguous.

        A stack is flattened only when ``degree`` is a multiple of ``A`` and each
        rank's shard stays within one index of ``A``, so that one layer can
        still be viewed as ``[A, B, ...]`` (see ``as_layer_shape``).

        This replaces the flattened parameters, so call it before sharding, while
        the stacks are plain tensors and nothing else references them. Calling
        it again with the same ``degree`` is a no-op.

        Args:
          degree: Number of ranks FSDP shards each stack across.

        Returns:
          A copy of ``layer_shapes``.
        """

        # Flatten `[L, dim1_len, dim2_len, ...]` only if each rank's shard of the
        # flattened axis stays inside one entry of axis 1. Otherwise a layer
        # can't be viewed in its original shape without a copy (see
        # `as_layer_shape`).
        def can_flatten(dim1_len: int, dim2_len: int) -> bool:
            if dim1_len % degree == 0:  # Axis 1 already shards evenly.
                return False
            ranks_per_entry, rem = divmod(degree, dim1_len)
            return rem == 0 and dim2_len % ranks_per_entry == 0

        for name, param in list(self._parameters.items()):
            # Flattened by an earlier call (`layer_shapes` is only set below).
            if name in self.layer_shapes:
                if not can_flatten(*self.layer_shapes[name][:2]):
                    raise ValueError(
                        f"{name} is already flattened to {list(param.shape)}, "
                        f"which doesn't fit {degree} ranks"
                    )
                continue
            # 2-D stacks (e.g. norm weights `[L, dim]`) have no axis 2 to merge.
            if (
                param is None
                or param.dim() < 3
                or not can_flatten(*param.shape[1:3])
            ):
                continue
            if isinstance(param, DTensor):
                # Already distributed by another parallelism; flattening would
                # need a redistribute, so leave it as is.
                continue
            logger.info(
                "Flattening %s from %s to %s to shard it over %d ranks",
                name,
                list(param.shape),
                [param.shape[0], param.shape[1] * param.shape[2], *param.shape[3:]],
                degree,
            )
            self.layer_shapes[name] = param.shape[1:]
            # `view`, not `flatten`: on a non-contiguous stack `flatten`
            # silently copies, and the new parameter would no longer share
            # storage with the old one. `view` raises instead.
            self._parameters[name] = nn.Parameter(
                param.detach().view(
                    param.shape[0], param.shape[1] * param.shape[2], *param.shape[3:]
                ),
                requires_grad=param.requires_grad,
            )
        return dict(self.layer_shapes)

    def plan_shard_dims(self, degree: int) -> dict[str, int]:
        """Returns the axis ``apply_simple_fsdp`` shards each stack on.

        Read-only. The result reflects the stacks as currently stored, so call
        ``flatten_for`` first. Covers every stack, so a missing key downstream
        is a bug, not a default.
        """
        return {
            name: 1 if name in self.layer_shapes else _shard_dim(param, degree)
            for name, param in self._parameters.items()
            if param is not None
        }

    def build_split_meshes(self) -> None:
        """Creates the meshes ``as_layer_shape`` uses.

        For each flattened stack, splits the shard axis of its FSDP mesh into
        ``(A, ranks // A)``. This creates process groups, so all ranks must call
        it at the same point; call it right after ``apply_simple_fsdp``.
        """
        for name, shape in self.layer_shapes.items():
            param = self._parameters.get(name)
            if not isinstance(param, DTensor):
                continue
            mesh, dim1_len = param.device_mesh, shape[0]
            if (mesh, dim1_len) in self.split_meshes:
                continue
            ranks_per_entry = mesh.size(mesh.ndim - 1) // dim1_len
            axis = (mesh.mesh_dim_names or ("dp",))[-1]
            base = f"{axis}_{dim1_len}x{ranks_per_entry}"
            # pylint: disable-next=protected-access
            self.split_meshes[(mesh, dim1_len)] = mesh._unflatten(
                mesh.ndim - 1,
                (dim1_len, ranks_per_entry),
                (f"{base}_outer", f"{base}_inner"),
            )

    def as_layer_shape(self, name: str, layer: torch.Tensor) -> torch.Tensor:
        """Returns ``layer``, one layer of stack ``name``, in its original shape.

        Returns ``layer`` unchanged if the stack wasn't flattened. Otherwise the
        result is a view, so writes to it update the stack. Used by weight
        initialization and the state-dict hooks; the scan body reshapes plain
        tensors instead (see ``LayerStack.run``).

        A sharded layer is an ``[A*B, ...]`` DTensor with ``Shard(0)``. When
        ``A`` is smaller than the mesh, no placement on a 1-D mesh describes it
        as ``[A, B, ...]``. Splitting the shard axis into ``(A, ranks // A)``
        describes the same local data as ``(Shard(0), Shard(1))``, without the
        copy that ``redistribute`` would make.
        """
        shape = self.layer_shapes.get(name)
        if shape is None:
            return layer
        if not isinstance(layer, DTensor):
            return layer.view(shape)
        # SimpleFSDP shards on the last mesh axis. Earlier axes, such as HSDP's
        # replicate axis, are kept as they are.
        *outer, last = layer.placements
        if last != Shard(0):
            raise ValueError(
                f"expected SimpleFSDP's Shard(0) on the last mesh axis, got "
                f"{layer.placements}"
            )
        mesh, (dim1_len, dim2_len) = layer.device_mesh, shape[:2]
        ranks = mesh.size(mesh.ndim - 1)
        ranks_per_entry = ranks // dim1_len
        if ranks % dim1_len or dim2_len % ranks_per_entry:
            raise ValueError(
                f"{ranks} ranks don't split {list(shape)} by whole entries"
            )
        if (mesh, dim1_len) not in self.split_meshes:
            # Not built lazily here: that creates process groups, and only the
            # ranks that reach this line would join them, so the job hangs.
            raise RuntimeError(
                f"no split mesh for {name}; call build_split_meshes() on every "
                "rank right after apply_simple_fsdp"
            )
        return DTensor.from_local(
            layer.to_local().view(1, dim2_len // ranks_per_entry, *shape[2:]),
            self.split_meshes[(mesh, dim1_len)],
            (*outer, Shard(0), Shard(1)),
            run_check=False,
            shape=shape,
            stride=torch.empty(shape, device="meta").stride(),
        )


class StackedLayers(TpuLayerLoop):
    """``model.layers``, running its layers out of the stacks.

    A ``TpuLayerLoop`` runs its own layers, so swapping the container is all it
    takes to change how they run: as ``n_layers`` ``functional_call``s, or as
    one ``scan`` over the stacks. Either way the layer modules inside are the
    same objects as before, so their parameters, FQNs and SimpleFSDP
    parametrizations are untouched.
    """

    def __init__(self, layers: nn.ModuleDict, stack: "LayerStack", *, use_scan: bool):
        super().__init__(layers)
        self.stack = stack
        self.use_scan = use_scan

    def forward(self, h: torch.Tensor, *args) -> torch.Tensor:
        return self.stack.run(h, *args, use_scan=self.use_scan)


class LayerStack:
    """Owns the stacks, and switches the model between the two layouts.

    ``unrolled`` is the only method called from outside, by ``init_states`` and
    the state-dict hooks. ``_apply_stacked_layers`` below drives the rest, in the
    one order that works. Each step asserts what has already run, so getting the
    order wrong fails loudly.
    """

    def __init__(
        self, layers: nn.ModuleDict, holder: LayerStackHolder, period: int = 1
    ):
        self.layers = layers
        # The `LayerStackHolder` the stacks are registered on.
        self.holder = holder
        self.n_layers = len(layers)
        # One scan step runs `period` consecutive layers, each position reading
        # its own stacks. 10 layers, period 4: layers 0-7 scan as 2 steps,
        # layers 8-9 run unrolled with their own parameters.
        if period < 1 or period > self.n_layers:
            raise ValueError(f"period {period} needs 1 <= period <= {self.n_layers}")
        self.period = period
        self.n_cycles = self.n_layers // period
        self.n_stacked = self.n_cycles * period
        # Per position in the cycle: (per-layer FQN, stack key), e.g.
        # ("attention.wq.weight", "attention__wq__weight").
        self.keys: list[list[tuple[str, str]]] = [[] for _ in range(period)]
        # What has already run, so a caller can't reorder the steps.
        self._unregistered = False
        self._views_attached = False
        # Where the stacks wait while the views stand in for them, inside
        # `unrolled()`, and the views themselves, to check nothing swapped them.
        self._hidden: dict[str, nn.Parameter] = {}
        self._views: dict[tuple[int, str], nn.Parameter] = {}
        # The DTensor attributes each stack is sharded with, saved because the
        # scan body only ever sees plain tensors. Filled on the first scan, not
        # here; see `scan_fsdp`.
        self.fsdp_plan: dict[str, ScanFSDPSpec] = {}

    @property
    def all_keys(self) -> list[str]:
        return [key for keys in self.keys for _, key in keys]

    def _slots(self) -> Iterator[tuple[int, int, str, str, nn.Module, str]]:
        """Every stacked per-layer parameter, as (layer, row, FQN, stack key,
        owning module, attribute).

        `_parameters` is keyed by the last part of the name alone, so adding or
        deleting an entry needs the owning module as well as the name. Attaching
        and dropping views are inverses of each other, so both walk this.
        Layers past the last whole cycle are not stacked and are skipped.
        """
        for i in range(self.n_stacked):
            layer = self.layers[str(i)]
            for name, key in self.keys[i % self.period]:
                # "attention.wq.weight" splits into path ["attention", "wq"] and
                # attr "weight". Walking that path from a layer lands on the `wq`
                # Linear, and its _parameters["weight"] is the slot to swap.
                *path, attr = name.split(".")
                mod = layer
                for part in path:
                    mod = getattr(mod, part)
                yield i, i // self.period, name, key, mod, attr

    def expert_parallel_dims(self) -> dict[str, int]:
        """Stack key -> the per-layer axis expert parallelism shards.

        Read off the owning module's declared ``state_shardings`` (an ``ep``
        entry of ``S(d)``), not off parameter names, so any model whose
        experts declare an EP placement is covered. A module that declares an
        EP placement must resolve one for every weight it owns: otherwise that
        stack would be sharded by dense FSDP, and the scan body would gather
        every expert onto every rank. That raises.
        """

        def ep_dim(layout) -> int | None:
            t = layout.local_type.get("ep")
            return getattr(t, "dim", None)

        dims: dict[str, int] = {}
        for i, _, name, key, mod, attr in self._slots():
            if i >= self.period:  # Positions 0..period-1 cover every key.
                break
            cfg = getattr(mod, "_sharding_config", None)
            if cfg is None:
                continue
            declared = {
                n: ep_dim(l) for n, l in cfg.state_shardings.items()
                if ep_dim(l) is not None
            }
            if not declared:
                continue
            if attr in declared:
                dims[key] = declared[attr]
            elif len(set(declared.values())) == 1:
                # TpuGroupedExperts renames its weights to the kernel-native
                # names only when it parallelizes, after this runs.
                dims[key] = next(iter(declared.values()))
            else:
                raise ValueError(
                    f"layers.{i}.{name}: its module declares EP placements "
                    f"{declared} but none for {attr!r}; cannot place its stack"
                )
        return dims

    # -- construction ------------------------------------------------------

    def unregister_and_allocate(self) -> None:
        """Replace each *kind* of per-layer weight with one stacked tensor.

        Every layer holds the same kinds of weight: ``n_layers`` separate
        ``attention.wq.weight``, ``n_layers`` separate ``feed_forward.w1.weight``
        and so on. Each of those groups becomes one ``[n_layers, *shape]``
        tensor, which is why the first loop below is over kinds and not over
        layers. The second loop deletes the originals, leaving the layer modules
        with no parameters of their own.

        The originals have to go. A stack and its rows are the same memory, so if
        FSDP found both it would shard both, and one weight would collect two
        sets of gradients.

        Nothing is really allocated here. The model is still on meta, so these
        are shape-and-dtype placeholders that ``to_empty`` will materialize.
        """
        assert not self._unregistered, "unregister_and_allocate() ran twice"
        for pos in range(self.period):
            group = {
                str(i): self.layers[str(i)]
                for i in range(pos, self.n_stacked, self.period)
            }
            check_layers_are_stackable(group)
            for name, p in group[str(pos)].named_parameters():
                key = _stack_key(pos, name, self.period)
                self.keys[pos].append((name, key))
                self.holder.register_parameter(
                    key,
                    nn.Parameter(
                        torch.empty(
                            self.n_cycles, *p.shape, dtype=p.dtype, device=p.device
                        ),
                        requires_grad=p.requires_grad,
                    ),
                )
        # Row i of the stack is now the only copy. `layers.i.attention.wq.weight`
        # no longer resolves to a parameter, and nothing walking the tree finds
        # it.
        for *_, mod, attr in self._slots():
            del mod._parameters[attr]
        self._unregistered = True

    # -- the two layouts ---------------------------------------------------

    @contextlib.contextmanager
    def unrolled(self):
        """Make the model look unstacked for the duration of a block.

        Inside the block the model is indistinguishable from the unrolled one:
        same parameters, same names, same ``state_dict``, and writes land in the
        stack. It's a scope rather than two calls so the views still come down
        if the block raises.
        """
        self._attach_views()
        try:
            yield
        finally:
            self._drop_views()

    def _attach_views(self) -> None:
        """Put the per-layer names back, each aliasing its row of a stack.

        Only ever after ``to_empty``, which reallocates storage without
        preserving aliasing: a view taken while the model is still on meta
        would point at nothing, and every write to it would be lost.
        """
        assert self._unregistered and not self._views_attached
        for i, row, name, key, mod, attr in self._slots():
            # Read out of `_parameters`, not with `getattr`: after SimpleFSDP
            # the attribute returns the *gathered* tensor, and a view of that
            # would alias a temporary, not this rank's shard.
            stack = self.holder._parameters[key]
            # `stack[i]` is basic indexing -- a plain integer, no mask, no index
            # tensor -- which in torch always returns a view, never a copy. So
            # this parameter shares storage with row i, and writing the "layer"
            # writes the stack. That is what makes `init_weights` land in the
            # stack rather than in a temporary; `test_init_differs_per_row`
            # checks it.
            #
            # Registered by hand, because `setattr` would go through SimpleFSDP's
            # parametrization instead of the raw slot.
            layer = self.holder.layer_view(key, stack, row)
            view = nn.Parameter(layer, requires_grad=stack.requires_grad)
            mod._parameters[attr] = view
            self._views[(i, name)] = view
        # Move the stacks aside while the views stand in for them. Both would
        # otherwise be registered parameters of the same model pointing at the
        # same memory, so `model.parameters()` would yield every weight twice and
        # the optimizer would step it twice. `_drop_views` puts them back.
        self._hidden = dict(self.holder._parameters)
        self.holder._parameters.clear()
        self._views_attached = True

    def _drop_views(self) -> None:
        """Exact inverse of ``_attach_views``."""
        assert self._views_attached
        for i, _, name, _, mod, attr in self._slots():
            # `load_state_dict(assign=True)` puts the checkpoint's tensor in the
            # slot instead of copying into the view. The stack would keep its old
            # values and the loaded weights would be deleted on the next line --
            # wrong weights, no error. Nothing in torchtitan passes `assign`, so
            # this only has to be loud.
            if mod._parameters[attr] is not self._views[(i, name)]:
                raise RuntimeError(
                    f"layers.{i}.{name} was replaced rather than written to, so "
                    "the stack never saw the new value. `assign=True` is not "
                    "supported on a stacked model."
                )
            del mod._parameters[attr]
        self.holder._parameters.update(self._hidden)
        self._hidden = {}
        self._views = {}
        self._views_attached = False

    # -- installation ------------------------------------------------------

    def install(self, model: nn.Module, *, use_scan: bool) -> None:
        """Attach the stack to the model, at the end of its ``__init__``.

        Three attachments, and the model isn't stacked until all three are on:
        initialization runs with the views up, checkpoints keep the unrolled
        keys, and ``model.layers`` runs its layers out of the stacks.

        The state-dict hooks call the two halves of ``unrolled`` instead of the
        context manager, because torch's API is a pre-hook and a post-hook with
        the caller's own ``state_dict`` call in between, so there is no block to
        wrap. They are plain closures and not bound methods, because registering
        a hook sets an attribute on whatever it is given and a bound method
        rejects that.
        """
        assert self._unregistered, "install() before unregister_and_allocate()"
        # `init_states` (torchtitan/protocols/module.py) is torchtitan's weight
        # initializer. The trainer builds the model on meta, calls `to_empty` to
        # give it real but uninitialized storage, then calls `init_states` once.
        # That walks the tree and has each module fill in its own parameters and
        # buffers. It is the only thing that writes initial weights, so it is
        # the only place the per-layer views need to be up. Wrap it, and every
        # write lands in the right row of the stack.
        inner = model.init_states

        def init_states(**kwargs):
            with self.unrolled():
                inner(**kwargs)

        def attach(*args):
            self._attach_views()

        def drop(*args):
            self._drop_views()

        model.init_states = init_states
        # Save and load in the unrolled format, so a run with the flag on can
        # load a checkpoint written with it off and vice versa. Stacking is an
        # execution detail, not a change to the model.
        #
        # This puts the views up and lets the model produce its own state dict,
        # rather than translating keys afterwards, which is what keeps every
        # other hook in the tree working. The fused-QKV split is the one that
        # matters: it emits wq/wk/wv from a single wqkv parameter, and anything
        # rewriting keys from outside would bypass it.
        model.register_state_dict_pre_hook(attach)
        model.register_state_dict_post_hook(drop)
        model.register_load_state_dict_pre_hook(attach)
        model.register_load_state_dict_post_hook(drop)
        # Only a rewritten model calls its layers container. A stock `Decoder`
        # loops over the layers directly, so it would ignore the stacks and
        # quietly train nothing.
        if not runs_layer_loop(model):
            raise TypeError(
                f"{type(model).__name__} does not run its layers through its "
                "layers container, so stacking them would have no effect; the "
                "model config has to go through `tpu_model_class` first."
            )
        if use_scan:
            # Logged because a recipe would otherwise believe its own setting:
            # `compile.memory_policy` tags joint-graph nodes, and the
            # body's are not in it. Only `LayerStack.run`'s budget reaches them,
            # and it must be 0.0 or the in-body gather is saved instead of
            # recomputed.
            logger.info(
                "Decoder layers run as a scan: the body is fully "
                "activation-checkpointed, whatever the memory policy asks for"
            )
        # The container holds the same layer modules, so `self.layers` keeps
        # working and is deliberately not repointed at it.
        model.layers = StackedLayers(model.layers, self, use_scan=use_scan)

    # -- running the layers ------------------------------------------------

    def _plan_prequantized_gathers(
        self, template: nn.Module, layer_shapes, position: int = 0
    ):
        """``(position, buffer prefix, stack keys, spec, quantize)`` per group.

        A submodule of the template opts in with ``scan_prequantize_groups()``
        (buffer prefix -> weight names) and ``scan_prequantize(shard,
        all_gather)``. A group is used only when every weight in it is sharded
        whole along the gathered axis 0 (its first per-layer axis) by one
        group and was not flattened; otherwise the module quantizes the
        gathered weight itself.
        """
        key_of = dict(self.keys[position])
        plan = []
        for path, mod in template.named_modules():
            groups_fn = getattr(mod, "scan_prequantize_groups", None)
            if groups_fn is None:
                continue
            for key, weights in groups_fn().items():
                names = [f"{path}.{w}" if path else w for w in weights]
                keys = [key_of.get(n) for n in names]
                specs = [self.fsdp_plan.get(k) for k in keys]
                ok = all(
                    k is not None
                    and k not in layer_shapes
                    and s is not None
                    and s.sharded
                    and s.gather_dim == 0
                    and s.group_name == specs[0].group_name
                    for k, s in zip(keys, specs)
                )
                if not ok:
                    logger.warning(
                        "Not pre-quantizing %s.%s in the scan: its weights "
                        "aren't all sharded on their first axis",
                        path,
                        key,
                    )
                    continue
                prefix = f"{path}.{key}" if path else key
                plan.append(
                    (position, prefix, keys, specs[0], mod.scan_prequantize)
                )
        if plan:
            logger.info(
                "Scan body quantizes and gathers FP8: %s",
                [p[0] for p in plan],
            )
        return plan

    def _prequantized_scan_body(self, body, keys, prequant):
        """``run``'s scan body, plus each layer's pre-quantized FP8 weights.

        Like the plain body in ``run`` -- make every row whole with
        ``gather_slice``, run the layers, return a scalar y -- but each group in
        ``prequant`` is also quantized from this rank's shard, gathered in FP8,
        and passed to its position's layer as ``<prefix>_q`` /
        ``<prefix>_scale``. The BF16 gathers stay for their gradient routing.
        """

        def scan_body(carry, slices):
            params = {k: gather_slice(slices[k], self.fsdp_plan.get(k)) for k in keys}
            extra = {}
            for pos, prefix, group_keys, spec, quantize in prequant:
                shard = torch.cat([slices[k].detach() for k in group_keys], dim=-1)
                q, scale = quantize(shard, functools.partial(gather_like, spec=spec))
                extra.setdefault(pos, {})[f"{prefix}_q"] = q
                extra.setdefault(pos, {})[f"{prefix}_scale"] = scale
            out = body(carry, params, extra)
            # A scalar y, for the reason given in the plain body in ``run``.
            return out, out.new_zeros(())

        return scan_body

    def run(self, h: torch.Tensor, *args, use_scan: bool):
        """Run the layers out of the stacks, one at a time or as one ``scan``.

        Both paths run the same ``body``. Layers ``0..period-1`` act as the
        templates, one per position in the cycle: ``functional_call`` runs a
        template's ``forward`` with another layer's weights. The layers' own
        buffers (the RoPE cache) are identical across the layers of a position,
        so the template's are correct on every iteration.

        One difference is who slices the stacks. The loop indexes them itself;
        ``scan`` takes them as its ``xs``, slicing along their leading layer
        axis with the hidden state as the carry. That collapses ``n_layers``
        copies of the block into a single ``stablehlo.while`` body, so compile
        time and program size stop growing with depth.

        The other is where the all-gather happens. The loop reads each stack
        with ``getattr``, which runs SimpleFSDP's parametrization and gathers
        the whole ``[n_layers, ...]`` tensor. Every layer's weights are then
        resident until the backward is done: about 137 GiB per chip at
        llama3-70b, which does not fit. ``scan`` instead passes this rank's
        *shard* of each stack and gathers one layer inside the body, so only
        one layer is ever whole. See ``scan_fsdp``.
        """
        assert self._unregistered and not self._views_attached
        # Layers 0..period-1 stand in for every cycle, one per position.
        templates = [self.layers[str(pos)] for pos in range(self.period)]
        keys = self.all_keys
        # Original per-layer shape of each flattened stack, by stack key. The
        # body views a gathered layer back to it. Stacks that weren't
        # flattened aren't listed and pass through as they are.
        layer_shapes = {
            k: self.holder.layer_shapes[k]
            for k in keys
            if k in self.holder.layer_shapes
        }
        # Ensure every flattened stack in the holder maps to a layer parameter.
        # Otherwise, it would enter `body` still flattened without being reshaped.
        unmatched = set(self.holder.layer_shapes) - set(layer_shapes)
        if unmatched:
            raise RuntimeError(
                f"flattened stacks with no layer name: {sorted(unmatched)}"
            )

        def body(carry, params, extra=None):
            # One cycle: each position's template runs with its row of its own
            # stacks. That is what makes the scan body a single traced graph
            # instead of n_layers of them. The templates have no parameters of
            # their own to collide with: they were unregistered at allocation.
            for pos, template in enumerate(templates):
                layer_params = {
                    name: params[k].view(layer_shapes[k])
                    if k in layer_shapes
                    else params[k]
                    for name, k in self.keys[pos]
                }
                # The FP8 weights `_prequantized_scan_body` adds, if any.
                layer_params.update((extra or {}).get(pos, {}))
                carry = functional_call(template, layer_params, (carry, *args))
            return carry

        # One [n_cycles, *shape] stack per parameter (per position in the
        # cycle), keyed by its stack key. For llama3-70B, unsharded, that is
        #
        #     {"attention__wq__weight":     [80, 8192, 8192],
        #      "attention__wk__weight":     [80, 1024, 8192],
        #      ...
        #      "feed_forward__w1__weight":  [80, 28672, 8192]}
        #
        # `scan` slices each one along its leading axis, so iteration i gets a
        # dict with the same keys holding row i: [8192, 8192], [1024, 8192], ...
        if use_scan:
            # On the first forward, not at install time: the stacks are sharded
            # after `__init__`, so a plan built there would see meta tensors,
            # find nothing sharded, and silently run every layer on this rank's
            # shard of it.
            if not self.fsdp_plan:
                self.fsdp_plan = plan_stacked_gathers(self.holder)
            if templates[0].training and not h.requires_grad:
                # When the init carry doesn't require grad, `scan`'s backward
                # returns `zeros_like(carry)`. That cuts the gradient chain
                # between layers while leaving the loss bit-exact, so nothing
                # looks wrong. Fail loudly instead.
                raise RuntimeError(
                    "the scan init carry does not require grad, so every layer's "
                    "weight gradient would reflect only itself -- with a correct "
                    "loss and no error"
                )
            # `xs` is the body's input, one cycle per step. `getattr` would
            # fire SimpleFSDP's parametrization and gather the whole stack
            # here, outside the loop. `_parameters` passes this rank's shard
            # instead, and the body gathers one layer.
            xs = {}
            for k in keys:
                param = self.holder._parameters[k]
                xs[k] = param.to_local() if isinstance(param, DTensor) else param

            def scan_body(carry, slices):
                # `scan` hands the body row i of every stack, under the same
                # keys, and `gather_slice` makes each row whole.
                out = body(
                    carry,
                    {k: gather_slice(slices[k], self.fsdp_plan.get(k)) for k in keys},
                )
                # A `scan` body returns (next_carry, y), and `scan` stacks the y
                # of every iteration into one `[n_layers, ...]` output. We only
                # want the last hidden state, which is the carry. Returning `out`
                # as y as well would stack all n_layers of them -- for llama3-70B
                # that is 80 x the full activation -- and then throw the whole
                # thing away. `scan` does not accept a body with no y, so return
                # a scalar and drop it.
                return out, out.new_zeros(())

            # Opt-in FP8 path: modules that define `scan_prequantize_groups`
            # get their weights quantized per shard and gathered in FP8. With
            # no such module, `prequant` is empty and the body above runs as is.
            prequant = [
                group
                for pos, template in enumerate(templates)
                for group in self._plan_prequantized_gathers(
                    template, layer_shapes, pos
                )
            ]
            if prequant:
                scan_body = self._prequantized_scan_body(body, keys, prequant)

            # Budget 0.0 forces the gathered weight to be recomputed in the
            # backward. Left alone, the partitioner saves it -- a collective
            # looks expensive to redo -- and the saved copy is carried out of
            # the loop, which is the whole memory win gone.
            #
            # It costs a full recompute of the body. Tagging just the gather
            # would be cheaper, but that tag lives in `node.meta["custom"]`,
            # which the body's re-trace drops before it reaches any node inside
            # the scan. Switch once the meta survives.
            #
            # The one exception is the MoE aux loss's expert counts: redoing
            # their all-reduce in backward is an exposed DCN round trip per
            # layer on multi-slice. The window tags the re-traced joint graph
            # to save just those, and is exactly budget 0.0 for a body without
            # them. See `scan_remat_policy`.
            #
            # The patch is process-global, so it has to be a window; it closes
            # when this call returns. `test_layer_stack.py` checks both halves.
            with scan_body_remat_window(), _no_sparse_mesh_push():
                h, _ = scan(scan_body, h, xs)
        else:
            if self.holder.ep_dims:
                # The loop reads whole stacks; an expert stack holds only this
                # rank's EP block, and nothing here gathers efsdp.
                raise NotImplementedError(
                    "expert-parallel stacks run only as a scan "
                    "(--compile.use-scan-layers)"
                )
            stacks = {k: getattr(self.holder, k) for k in keys}
            for c in range(self.n_cycles):
                h = body(h, {k: stacks[k][c] for k in keys})
        # Layers past the last whole cycle own their parameters and run as
        # ordinary modules (MaxText does the same with `n_layers % period`).
        for i in range(self.n_stacked, self.n_layers):
            h = self.layers[str(i)](h, *args)
        return h


def _apply_stacked_layers(
    model: nn.Module, *, use_scan: bool, period: int = 1
) -> None:
    """Stack the decoder layers' weights, and with ``use_scan`` run them as one.

    Called from the model's own ``__init__`` (see
    :func:`with_tpu_stacked_layers`), so the model is stacked from the moment it
    exists. Everything downstream, sharding included, just sees a model whose
    layer weights happen to have a leading axis. Mutates ``model``.

    What the model has to provide, i.e. the checklist for porting a new one:

    * It runs its layers through its layers container instead of looping over
      them itself, which means its config went through ``tpu_model_class``.
      Stacking works by swapping that container for one that runs the layers out
      of the stacks.
    * The layers are homogeneous -- with ``period > 1``, the layers at the
      same position in each cycle -- and no layer is an MoE with load
      balancing on. Both checked, by :func:`check_layers_are_stackable`.
    * The layer signature is ``(h, *args)``, where ``h`` is the hidden state and
      layer ``i`` takes layer ``i-1``'s output. ``h`` is the only argument that
      changes between layers; ``*args`` (RoPE cache, attention masks) has to be
      the same for all of them.
    * ``model.init_states`` exists and is where the weights are initialized. It
      gets wrapped so the per-layer views are up while it runs.

    Two more requirements apply with ``use_scan``, and they come from ``scan``
    rather than from the stack:

    * The layer must not mutate its inputs. ``scan`` is a higher-order op, and a
      body that writes to what it was handed raises
      ``UncapturedHigherOrderOpError`` at trace time.
    * The step has to be traced: ``--compile.mode=aot_fx_trace``.
      Eager ``scan`` doesn't materialize on TPU.

    What does and does not change:

    * ``state_dict()``, ``load_state_dict()`` and the on-disk model checkpoint
      keep the unrolled ``layers.i.<p>`` keys, so a checkpoint moves freely
      between a stacked and an unstacked run. ``assign=True`` is the exception,
      and is refused.
    * ``named_parameters()`` does *not*. It reports the stacks, under
      ``layer_stacks.<mangled>``, and the per-layer attributes no longer exist.
      Anything keyed off live parameter names sees the difference; see the
      module docstring.

    With ``use_scan=False`` the layers still run one at a time, just out of the
    stacks. That's the storage change without the loop change, which is what the
    reshard and checkpoint tests compare against.
    """
    if use_scan:
        # Here because every scanning model comes through here. Without the
        # patch the backward reverses by copy and OOMs some steps later, so
        # fail at build time instead.
        from torchtitan.experiments.tpu.components.scan_backward_patch import (  # pylint: disable=g-import-not-at-top
            apply_scan_reverse_backward_patch,
        )

        if not apply_scan_reverse_backward_patch():
            raise RuntimeError(
                "scanned decoder layers need the scan backward patch and it did"
                " not install; the warning above says which case this is."
            )

    holder = LayerStackHolder()
    setattr(model, STACK_ATTR, holder)
    stack = LayerStack(model.layers, holder, period)
    stack.unregister_and_allocate()
    stack.install(model, use_scan=use_scan)
    logger.info(
        "Stacked %d weights across %d layers (period %d, %d unrolled), run as a %s",
        len(stack.all_keys),
        stack.n_stacked,
        period,
        stack.n_layers - stack.n_stacked,
        "scan" if use_scan else "loop",
    )


@functools.cache
def _stacked_model_class(
    config_cls: type, use_scan: bool, period: int = 1
) -> type:
    """Build the ``Stacked<Model>`` subclass that stacks itself on construction.

    It subclasses whatever layer-loop class ``tpu_model_class`` builds, because
    the model has to run its layers through its layers container for stacking to
    do anything. Its ``__init__`` builds the model as usual and then stacks it.

    One class per (config class, ``use_scan``, ``period``), cached.
    """
    base = tpu_model_class(config_cls)

    def __init__(self, *args, **kwargs):
        base.__init__(self, *args, **kwargs)
        _apply_stacked_layers(self, use_scan=use_scan, period=period)

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
        """Count on the unstacked model this config describes.

        Stacking moves each layer's weights out of the layer's modules into one
        tensor per weight. Upstream's active-parameter count applies the top-k
        discount to the parameters an MoE module owns, so on a stacked model it
        would count every expert as active (7.6x the FLOPs on the Qwen3-235B
        scan proxy). FLOPs depend on the model, not on how its weights are
        stored, so count on an unstacked build of the same config. It is built
        on the meta device: no memory, no initialization.
        """
        del model
        unstacked_config = derive(self, base.Config)
        with torch.device("meta"):
            unstacked = unstacked_config.build()
        return unstacked_config.get_nparams_and_flops(unstacked, seq_len)

    name = f"{'Scanned' if use_scan else 'Stacked'}{base.__name__}"
    return model_subclass(
        base,
        base.Config,
        name,
        config_body={"get_nparams_and_flops": get_nparams_and_flops},
        __init__=__init__,
    )


def with_tpu_stacked_layers(
    model_config: Decoder.Config, *, use_scan: bool, period: int = 1
) -> Decoder.Config:
    """Put ``model_config`` on a model that arrives already stacked.

    Stacking is a property of the model, not of how it gets parallelized. The
    model is built stacked, and ``apply_simple_fsdp`` then shards the stacks it
    finds like any other parameter, on the axis ``_fsdp_shard_dim`` names above.
    Nothing in the parallelize path has to know that stacking exists.

    ``use_scan`` additionally runs the layers as one ``scan`` rather than one
    at a time; it is fixed at construction because it decides which container
    ``model.layers`` becomes. ``period=k`` scans cycles of k layers whose
    types repeat (Qwen3.5: 3 GatedDeltaNet, then 1 full attention), with one
    stack set per position; leftover layers run unrolled.

    Builds on :func:`tpu_model_class`, and has to be applied in the same place
    and for the same reasons; see its docstring.
    """
    return derive(
        model_config,
        _stacked_model_class(type(model_config), use_scan, period).Config,
    )
