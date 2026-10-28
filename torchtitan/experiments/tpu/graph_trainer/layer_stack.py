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
* A stack ``[n, E, X, ...]`` whose axis 1 the FSDP mesh outgrows (experts past
  ``E`` ranks) is sharded on axes 1 and 2 of a split mesh; see
  ``LayerStackHolder.shard_split_stacks``.

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
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.func import functional_call

from torchtitan.config import derive
from torchtitan.experiments.graph_trainer.simple_fsdp import ReplicateComputation
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import (
    gather_slice,
    plan_stacked_gathers,
    ScanFSDPSpec,
    split_stack_spec,
)
from torchtitan.experiments.tpu.graph_trainer.scan_remat_policy import (
    scan_body_remat_window,
)
from torchtitan.experiments.tpu.layer_loop import (
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


def check_layers_are_stackable(layers: nn.ModuleDict) -> None:
    """Every layer must match layer 0, or they cannot share a stack.

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

    template, zero_buffers = spec(layers["0"]), dict(layers["0"].named_buffers())
    for i, layer in enumerate(layers.values()):
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
                "drop --tpu-config.use-scan-layers, or set "
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
                f"layer {i} does not match layer 0, so their weights cannot "
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
        # Each split mesh `shard_split_stacks` built -> the mesh it split.
        self.split_meshes: dict[DeviceMesh, DeviceMesh] = {}

    def plan_shard_dims(self, degree: int) -> dict[str, int]:
        """The axis `apply_simple_fsdp` shards each stack on.

        One axis cannot serve them all: dim 1 of an expert stack is
        `num_experts` = 128, which caps FSDP at 128 ranks, while dim 1 of a
        dense stack divides fine.

        Covers every stack, so a missing key downstream is a bug, not a default.
        """
        return {
            name: _shard_dim(param, degree)
            for name, param in self._parameters.items()
            if param is not None
        }

    def shard_split_stacks(self) -> None:
        """Moves stacks whose axis 1 the FSDP mesh outgrows onto a split mesh.

        SimpleFSDP shards one axis. When a stack ``[L, A, B, ...]`` has fewer
        entries ``A`` than ranks, ``_shard_dim`` picks an inner axis, and on
        XLA:TPU every gather and reduce-scatter of it then pays a relayout.
        If ``A`` divides the rank count ``n`` and ``n // A`` divides ``B``, this
        re-places the stack on an ``(A, n // A)`` split of the FSDP axis as
        ``(Shard(1), Shard(2))``. Rank ``r`` lands at ``(r // (n // A), r % (n
        // A))``, so its shard is the ``r``-th contiguous block of ``[L, A*B,
        ...]``, and ``gather_slice`` moves it as one flat collective over the
        unsplit group. The parameter, its gradient and optimizer state keep
        the logical shape, so init, checkpoints and DCP resharding see an
        ordinary 2-D DTensor.

        Creates process groups, so call it on every rank, right after
        ``apply_simple_fsdp`` and before ``to_empty``: on meta, re-placing
        moves no data.
        """
        computations = getattr(self, "_simple_fsdp_parametrization_list", None)
        if not computations or computations[0].mode == "replicate":
            return
        built: dict[tuple[DeviceMesh, int], DeviceMesh] = {}
        for name, param in list(self._parameters.items()):
            # Include Replicate: when every single axis is narrower than the
            # mesh (e.g. 30B at 4k ranks, 235B at 8k), data_parallel downgrades
            # Shard to Replicate even though axes 1 and 2 together divide.
            if not (
                isinstance(param, DTensor)
                and param.ndim >= 3
                and isinstance(param.placements[-1], (Shard, Replicate))
                and param.device_mesh not in self.split_meshes
            ):
                continue
            mesh, (dim1_len, dim2_len) = param.device_mesh, param.shape[1:3]
            ranks = mesh.size(mesh.ndim - 1)
            ranks_per_entry, rem = divmod(ranks, dim1_len)
            if dim1_len % ranks == 0 or rem or dim2_len % ranks_per_entry:
                continue
            if not param.is_meta:
                raise ValueError(
                    f"{name} is already materialized; call shard_split_stacks() "
                    "before to_empty"
                )
            if (mesh, dim1_len) not in built:
                axis = (mesh.mesh_dim_names or ("dp",))[-1]
                base = f"{axis}_{dim1_len}x{ranks_per_entry}"
                # pylint: disable-next=protected-access
                built[(mesh, dim1_len)] = mesh._unflatten(
                    mesh.ndim - 1,
                    (dim1_len, ranks_per_entry),
                    (f"{base}_outer", f"{base}_inner"),
                )
                self.split_meshes[built[(mesh, dim1_len)]] = mesh
            self._parameters[name] = nn.Parameter(
                distribute_tensor(
                    torch.empty(param.shape, dtype=param.dtype, device="meta"),
                    built[(mesh, dim1_len)],
                    (*param.placements[:-1], Shard(1), Shard(2)),
                ),
                requires_grad=param.requires_grad,
            )
        if self.split_meshes:
            for i, comp in enumerate(computations):
                if not isinstance(comp, _SplitStackReplicateComputation):
                    computations[i] = _SplitStackReplicateComputation(
                        comp, self.split_meshes
                    )


class _SplitStackReplicateComputation(ReplicateComputation):
    """SimpleFSDP's parametrization, gathering split stacks flat.

    The unrolled loop reads the stacks through this. ``redistribute`` would
    gather a split stack in two collectives, one per mesh axis; this runs the
    one flat all-gather (and reduce-scatter back) over the unsplit group that
    the scan body runs per layer. Everything else goes to SimpleFSDP.
    """

    def __init__(
        self,
        base: ReplicateComputation,
        split_meshes: dict[DeviceMesh, DeviceMesh],
    ):
        self.__dict__.update(base.__dict__)
        self.split_meshes = split_meshes

    def replicate_compute(self, x: DTensor) -> torch.Tensor:
        unsplit = self.split_meshes.get(x.device_mesh)
        if unsplit is None:
            return super().replicate_compute(x)
        if getattr(self, "full_dtensor", False):
            raise NotImplementedError("full_dtensor on a split stack")
        spec = split_stack_spec(
            unsplit, self.param_dtype, self.reduce_dtype, 1, x.shape
        )
        return gather_slice(x.to_local(), spec)


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

    def __init__(self, layers: nn.ModuleDict, holder: LayerStackHolder):
        self.layers = layers
        # The `LayerStackHolder` the stacks are registered on.
        self.holder = holder
        self.n_layers = len(layers)
        # Per-layer parameter FQNs relative to a layer, e.g. "attention.wq.weight".
        self.names: list[str] = []
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

    def _slots(self) -> Iterator[tuple[int, str, nn.Module, str]]:
        """Every per-layer parameter, as (row, FQN, owning module, attribute).

        `_parameters` is keyed by the last part of the name alone, so adding or
        deleting an entry needs the owning module as well as the name. Attaching
        and dropping views are inverses of each other, so both walk this.
        """
        for i, layer in enumerate(self.layers.values()):
            for name in self.names:
                # "attention.wq.weight" splits into path ["attention", "wq"] and
                # attr "weight". Walking that path from a layer lands on the `wq`
                # Linear, and its _parameters["weight"] is the slot to swap.
                *path, attr = name.split(".")
                mod = layer
                for part in path:
                    mod = getattr(mod, part)
                yield i, name, mod, attr

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
        check_layers_are_stackable(self.layers)
        for name, p in self.layers["0"].named_parameters():
            self.names.append(name)
            self.holder.register_parameter(
                _mangle(name),
                nn.Parameter(
                    torch.empty(
                        self.n_layers, *p.shape, dtype=p.dtype, device=p.device
                    ),
                    requires_grad=p.requires_grad,
                ),
            )
        # Row i of the stack is now the only copy. `layers.i.attention.wq.weight`
        # no longer resolves to a parameter, and nothing walking the tree finds
        # it.
        for _, _, mod, attr in self._slots():
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
        for i, name, mod, attr in self._slots():
            # Read out of `_parameters`, not with `getattr`: after SimpleFSDP
            # the attribute returns the *gathered* tensor, and a view of that
            # would alias a temporary, not this rank's shard.
            stack = self.holder._parameters[_mangle(name)]
            # `stack[i]` is basic indexing -- a plain integer, no mask, no index
            # tensor -- which in torch always returns a view, never a copy. So
            # this parameter shares storage with row i, and writing the "layer"
            # writes the stack. That is what makes `init_weights` land in the
            # stack rather than in a temporary; `test_init_differs_per_row`
            # checks it.
            #
            # Registered by hand, because `setattr` would go through SimpleFSDP's
            # parametrization instead of the raw slot.
            view = nn.Parameter(stack[i], requires_grad=stack.requires_grad)
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
        for i, name, mod, attr in self._slots():
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
            # `graph_trainer_memory_policy` tags joint-graph nodes, and the
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

    def run(self, h: torch.Tensor, *args, use_scan: bool):
        """Run the layers out of the stacks, one at a time or as one ``scan``.

        Both paths run the same ``body``. Layer 0 acts as the template:
        ``functional_call`` runs its ``forward`` with another layer's weights.
        The layers' own buffers (the RoPE cache) are identical across layers, so
        the template's are correct on every iteration.

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
        template = self.layers["0"]

        def body(carry, params):
            # Layer 0's module stands in for every layer, one row at a time.
            # That is what makes the scan body a single traced graph instead of
            # n_layers of them. It has no parameters of its own to collide with,
            # because they were unregistered when the stacks were allocated.
            return functional_call(template, params, (carry, *args))

        # One [n_layers, *shape] stack per parameter, keyed by its mangled
        # name. For llama3-70B, unsharded, that is
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
            if template.training and not h.requires_grad:
                # When the init carry doesn't require grad, `scan`'s backward
                # returns `zeros_like(carry)`. That cuts the gradient chain
                # between layers while leaving the loss bit-exact, so nothing
                # looks wrong. Fail loudly instead.
                raise RuntimeError(
                    "the scan init carry does not require grad, so every layer's "
                    "weight gradient would reflect only itself -- with a correct "
                    "loss and no error"
                )
            # `xs` is the body's input, one layer per step. `getattr` would
            # fire SimpleFSDP's parametrization and gather the whole stack
            # here, outside the loop. `_parameters` passes this rank's shard
            # instead, and the body gathers one layer.
            xs = {}
            for name in self.names:
                param = self.holder._parameters[_mangle(name)]
                xs[_mangle(name)] = (
                    param.to_local() if isinstance(param, DTensor) else param
                )

            def scan_body(carry, slices):
                # `scan` hands the body row i of every stack, under the same
                # keys, and `gather_slice` makes each row whole.
                out = body(
                    carry,
                    {
                        name: gather_slice(
                            slices[_mangle(name)], self.fsdp_plan.get(_mangle(name))
                        )
                        for name in self.names
                    },
                )
                # A `scan` body returns (next_carry, y), and `scan` stacks the y
                # of every iteration into one `[n_layers, ...]` output. We only
                # want the last hidden state, which is the carry. Returning `out`
                # as y as well would stack all n_layers of them -- for llama3-70B
                # that is 80 x the full activation -- and then throw the whole
                # thing away. `scan` does not accept a body with no y, so return
                # a scalar and drop it.
                return out, out.new_zeros(())

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
            with scan_body_remat_window():
                carry, _ = scan(scan_body, h, xs)
            return carry

        stacks = {_mangle(n): getattr(self.holder, _mangle(n)) for n in self.names}
        for i in range(self.n_layers):
            h = body(h, {n: stacks[_mangle(n)][i] for n in self.names})
        return h


def _apply_stacked_layers(model: nn.Module, *, use_scan: bool) -> None:
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
    * The layers are homogeneous, and no layer is an MoE with load balancing
      on. Both checked, by :func:`check_layers_are_stackable`.
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
    * The step has to be traced: ``--tpu-config.compile-mode=aot_fx_trace``.
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
        from torchtitan.experiments.tpu.scan_backward_patch import (  # pylint: disable=g-import-not-at-top
            apply_scan_reverse_backward_patch,
        )

        if not apply_scan_reverse_backward_patch():
            raise RuntimeError(
                "scanned decoder layers need the scan backward patch and it did"
                " not install; the warning above says which case this is."
            )

    holder = LayerStackHolder()
    setattr(model, STACK_ATTR, holder)
    stack = LayerStack(model.layers, holder)
    stack.unregister_and_allocate()
    stack.install(model, use_scan=use_scan)
    logger.info(
        "Stacked %d weights across %d layers, run as a %s",
        len(stack.names),
        stack.n_layers,
        "scan" if use_scan else "loop",
    )


@functools.cache
def _stacked_model_class(config_cls: type, use_scan: bool) -> type:
    """Build the ``Stacked<Model>`` subclass that stacks itself on construction.

    It subclasses whatever layer-loop class ``tpu_model_class`` builds, because
    the model has to run its layers through its layers container for stacking to
    do anything. Its ``__init__`` builds the model as usual and then stacks it.

    One class per (config class, ``use_scan``), cached.
    """
    base = tpu_model_class(config_cls)

    def __init__(self, *args, **kwargs):
        base.__init__(self, *args, **kwargs)
        _apply_stacked_layers(self, use_scan=use_scan)

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
    model_config: Decoder.Config, *, use_scan: bool
) -> Decoder.Config:
    """Put ``model_config`` on a model that arrives already stacked.

    Stacking is a property of the model, not of how it gets parallelized. The
    model is built stacked, and ``apply_simple_fsdp`` then shards the stacks it
    finds like any other parameter, on the axis ``_fsdp_shard_dim`` names above.
    Nothing in the parallelize path has to know that stacking exists.

    ``use_scan`` additionally runs the layers as one ``scan`` rather than one
    at a time; it is fixed at construction because it decides which container
    ``model.layers`` becomes.

    Builds on :func:`tpu_model_class`, and has to be applied in the same place
    and for the same reasons; see its docstring.
    """
    return derive(
        model_config, _stacked_model_class(type(model_config), use_scan).Config
    )
