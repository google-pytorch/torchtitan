"""Explicit FSDP for the TPU graph trainer: sharded parameters as plain tensors.

This is the TPU lane's replacement for SimpleFSDP's DTensor parametrization.
Each parameter is stored as this rank's local shard, a plain ``torch.Tensor``,
described by a :class:`ShardSpec` (the data-parallel mesh, which tensor axis
is split, which dtypes compute and reduce in). Reading the parameter through
its module attribute runs one hand-written autograd function: all-gather in
the forward, reduce-scatter in the backward, plus an all-reduce over the
replicate axis under HSDP. That is the pair the scan body has always used
(``scan_fsdp``); here it covers every parameter, and the scan body reuses it
for its per-layer rows.

What this removes from the runtime path: DTensor, its dispatcher, the
spmd_types translation SimpleFSDP does to get a DTensor back, and the
``ReplicateComputation`` subclass that re-implemented the gather for
non-leading shards. Under ``make_fx`` the traced graph is the same funcol
nodes SimpleFSDP produced, so the graph trainer's FSDP passes keep working.

DTensor is kept in exactly two windows, as a view over the local shard that
shares its storage:

* weight initialization, so DTensor's sharded RNG hands every shard the slice
  of the single-device draw -- bit-identical to a SimpleFSDP run with the same
  seed; and
* ``state_dict()`` / ``load_state_dict()``, so DCP sees the same sharded
  format as before. Checkpoints reshard across world sizes and interchange
  with SimpleFSDP and FSDP2 runs. The optimizer's parameter-shaped state
  (Adam's moments) and the fused optimizer's ``_optstate_*`` buffers are
  shards too, and get the same views (:func:`install_optimizer_state_hooks`,
  :func:`register_sharded_buffer`): a plain per-rank tensor under one key
  would be written by one rank and read back by all of them.

Both windows are opened by :func:`dtensor_params`.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterable, Iterator
from itertools import count
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.tools.logging import logger


__all__ = [
    "apply_tpu_fsdp",
    "disable_gather",
    "dtensor_params",
    "gather_slice",
    "grads_as_dtensors",
    "install_optimizer_state_hooks",
    "param_specs",
    "param_specs_by_fqn",
    "param_specs_for",
    "register_sharded_buffer",
    "ScanFSDPSpec",
    "shard_module",
    "ShardSpec",
    "unshard",
    "unshard_local",
    "uses_tpu_fsdp",
]


SPECS_ATTR = "_tpu_fsdp_specs"
# Buffers sharded like one of the module's parameters: buffer name -> parameter
# name (`register_sharded_buffer`). They get the parameter's DTensor view in
# the checkpoint window.
BUFFER_SPECS_ATTR = "_tpu_fsdp_buffer_params"


@dataclasses.dataclass(frozen=True)
class ScanFSDPSpec:
    """How the scan body gathers one row of a stacked parameter.

    The minimal, mesh-free form of :class:`ShardSpec`: the scan body sees plain
    tensors and process-group names only. ``ShardSpec.scan_spec`` produces one.

    Attributes:
        group_name: resolved process-group name, in the form funcol collectives
            take.
        param_dtype: compute dtype for the gathered weight; ``None`` leaves it
            alone.
        reduce_dtype: dtype to reduce the gradient in; ``None`` means reduce in
            whatever dtype the forward gathered in.
        sharded: ``False`` for a parameter left replicated. Nothing to gather
            in the forward; the gradient still has to be summed.
        replicate_group_name: under HSDP, the process group of the *replicate*
            mesh axis; ``None`` for a 1-D dp mesh. The forward needs nothing
            from it, but the gradient is Partial on it and has to be summed.
        gather_dim: which axis of the per-layer slice to gather on -- the
            stack's shard axis minus the scan axis, so dim 1 becomes 0.
    """

    group_name: str
    param_dtype: torch.dtype | None
    reduce_dtype: torch.dtype | None
    sharded: bool
    replicate_group_name: str | None = None
    gather_dim: int = 0


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(torch.empty(shape, device="meta").stride())


@dataclasses.dataclass(frozen=True)
class ShardSpec:
    """How one parameter is split over the data-parallel mesh.

    ``placements`` follow DTensor's vocabulary, one per mesh axis, so a
    DTensor view can always be built from the spec. The shard axis is the
    mesh's last axis; under HSDP the first axis is the replicate axis.

    Attributes:
      mesh: the data-parallel mesh, 1-D (``fsdp``) or 2-D (``dp_replicate``,
        ``fsdp``). Only this axis order is produced here.
      placements: ``(Shard(d),)`` / ``(Replicate(),)`` on a 1-D mesh, or
        ``(Replicate(), Shard(d))`` / ``(Replicate(), Replicate())`` on 2-D.
      global_shape: the whole parameter's shape.
      param_dtype: compute dtype of the gathered weight; ``None`` leaves it.
      reduce_dtype: dtype to reduce the gradient in; ``None`` means the dtype
        the gradient arrives in.
    """

    mesh: DeviceMesh
    placements: tuple[Placement, ...]
    global_shape: tuple[int, ...]
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if len(self.placements) != self.mesh.ndim:
            raise ValueError(
                f"{len(self.placements)} placements for a {self.mesh.ndim}-D mesh"
            )
        if self.mesh.ndim not in (1, 2):
            raise ValueError(f"expected a 1-D or 2-D dp mesh, got {self.mesh.ndim}-D")
        if self.mesh.ndim == 2 and not isinstance(self.placements[0], Replicate):
            raise ValueError(
                "the first mesh axis is the HSDP replicate axis and must be "
                f"Replicate, got {self.placements}"
            )
        if not isinstance(self.placements[-1], (Shard, Replicate)):
            raise ValueError(f"unsupported placement {self.placements[-1]}")

    # -- the shard axis -------------------------------------------------------

    @property
    def shard_dim(self) -> int | None:
        """Tensor axis split over the shard mesh axis; ``None`` if replicated."""
        last = self.placements[-1]
        return last.dim if isinstance(last, Shard) else None

    @property
    def shard_degree(self) -> int:
        return self.mesh.size(self.mesh.ndim - 1)

    @property
    def shard_group(self) -> str:
        return self.mesh.get_group(self.mesh.ndim - 1).group_name

    @property
    def replicate_group(self) -> str | None:
        """The HSDP replicate axis's group, or ``None`` on a 1-D mesh."""
        return self.mesh.get_group(0).group_name if self.mesh.ndim == 2 else None

    @property
    def sharded(self) -> bool:
        return self.shard_dim is not None

    @property
    def even(self) -> bool:
        d = self.shard_dim
        return d is None or self.global_shape[d] % self.shard_degree == 0

    def local_shape_and_offset(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """This rank's shard shape and its offset in the whole tensor.

        DTensor's own arithmetic, so the shard matches what a DTensor view with
        these placements describes -- uneven last shards included. Computed
        outside any fake-tensor mode: the arithmetic goes through tensors, and
        a caller sharding a model under ``FakeTensorMode`` (a pass test tracing
        a toy) would otherwise get a fake offset it cannot turn into an int.
        """
        with unset_fake_temporarily():
            shape, offset = compute_local_shape_and_global_offset(
                torch.Size(self.global_shape), self.mesh, self.placements
            )
        return tuple(int(x) for x in shape), tuple(int(x) for x in offset)

    def chunk_size(self) -> int:
        """Rows every rank's shard is padded to when the split is uneven."""
        d = self.shard_dim
        assert d is not None
        return -(-self.global_shape[d] // self.shard_degree)

    # -- derived specs ----------------------------------------------------------

    def layer_slice(self) -> ShardSpec:
        """The spec of one row of a ``[n_layers, ...]`` stack.

        Dropping the leading layer axis moves the shard axis down by one. The
        scan body gathers one such row per iteration.
        """
        d = self.shard_dim
        if d == 0:
            raise ValueError("dim 0 is the layer axis; a stack is never sharded on it")
        placements = tuple(
            Shard(p.dim - 1) if isinstance(p, Shard) else p for p in self.placements
        )
        return dataclasses.replace(
            self, placements=placements, global_shape=tuple(self.global_shape[1:])
        )

    def scan_spec(self) -> ScanFSDPSpec:
        """This spec in the mesh-free form the scan body dispatches on.

        Call on ``layer_slice()``: the scan body gathers rows, not stacks. An
        uneven split is refused, as it always was in the scan body -- the
        gather there is a bare collective with no padding around it.
        """
        d = self.shard_dim
        if d is not None and not self.even:
            raise ValueError(
                f"dim {d} = {self.global_shape[d]} cannot be split evenly over "
                f"{self.shard_degree} ranks, and the scan body's all-gather "
                "needs equal shards"
            )
        return ScanFSDPSpec(
            group_name=self.shard_group,
            param_dtype=self.param_dtype,
            reduce_dtype=self.reduce_dtype,
            sharded=d is not None,
            replicate_group_name=self.replicate_group,
            gather_dim=d if d is not None else 0,
        )

    def to_dtensor(self, local: torch.Tensor) -> DTensor:
        """A DTensor view of ``local``, sharing its storage."""
        return DTensor.from_local(
            local,
            self.mesh,
            self.placements,
            run_check=False,
            shape=torch.Size(self.global_shape),
            stride=_contiguous_stride(self.global_shape),
        )


# ---------------------------------------------------------------------------
# Collectives
# ---------------------------------------------------------------------------


def _waited(t: torch.Tensor) -> torch.Tensor:
    """Wait on a functional collective's result, as DTensor's redistribute does.

    An unwaited ``AsyncCollectiveTensor`` is a tensor subclass; letting one
    flow out of an autograd function makes the joint graph's output and
    tangent counts disagree. Waiting here puts the ``wait_tensor`` node right
    after the collective, which is also the shape the FSDP passes match.
    """
    if isinstance(t, funcol.AsyncCollectiveTensor):
        return t.wait()
    return t


def _gather_on(x: torch.Tensor, dim: int, group_name: str) -> torch.Tensor:
    """All-gather ``x`` along ``dim``, always collecting on dim 0.

    Off dim 0, funcol falls back to ``cat(chunk(t, world_size))`` -- a 2048-rank
    gather would emit 2048 slices and a concat. A transpose either side is two
    ops instead. ``scan_fsdp`` may have patched funcol with TPU tile-alignment
    fixes; they apply here too, because the attribute is read at call time.
    """
    if dim == 0:
        return _waited(funcol.all_gather_tensor(x, gather_dim=0, group=group_name))
    out = _waited(
        funcol.all_gather_tensor(x.movedim(dim, 0), gather_dim=0, group=group_name)
    )
    return out.movedim(0, dim)


def _scatter_on(x: torch.Tensor, dim: int, group_name: str) -> torch.Tensor:
    """Reduce-scatter ``x`` along ``dim``, always scattering on dim 0."""
    if dim == 0:
        return _waited(
            funcol.reduce_scatter_tensor(
                x, reduceOp="sum", scatter_dim=0, group=group_name
            )
        )
    out = _waited(
        funcol.reduce_scatter_tensor(
            x.movedim(dim, 0), reduceOp="sum", scatter_dim=0, group=group_name
        )
    )
    return out.movedim(0, dim)


def _pad_dim(x: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """Zero-pad ``x`` along ``dim`` up to ``size`` rows."""
    missing = size - x.shape[dim]
    if missing <= 0:
        return x
    pad = list(x.shape)
    pad[dim] = missing
    return torch.cat([x, x.new_zeros(pad)], dim=dim)


class _AllGatherReduceScatter(torch.autograd.Function):
    """All-gather a shard in the forward, reduce-scatter its gradient back.

    Written out by hand so the two halves can use different dtypes -- bf16 to
    gather, to halve the comm volume, and fp32 to reduce, to keep the gradient
    accurate. A plain collective reduces in whatever dtype the forward used,
    and casting after the reduce is already too late.

    ``global_size`` is the whole tensor's extent on ``shard_dim``. When the
    shard degree does not divide it, every rank's shard is zero-padded to the
    same chunk before the gather and the gradient is padded before the
    reduce-scatter; both results are narrowed back. ``None`` asserts an even
    split.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        shard: torch.Tensor,
        group_name: str,
        shard_dim: int,
        param_dtype: torch.dtype | None,
        reduce_dtype: torch.dtype | None,
        replicate_group_name: str | None,
        global_size: int | None,
    ) -> torch.Tensor:
        ctx.group_name = group_name
        ctx.shard_dim = shard_dim
        ctx.reduce_dtype = reduce_dtype
        ctx.replicate_group_name = replicate_group_name
        ctx.shard_dtype = shard.dtype
        ctx.local_size = shard.shape[shard_dim]
        ctx.chunk_size = ctx.local_size
        ctx.global_size = global_size
        if param_dtype is not None:
            shard = shard.to(param_dtype)
        if global_size is not None:
            degree = funcol.c10d._get_group_size_by_name(group_name)
            ctx.chunk_size = -(-global_size // degree)
            shard = _pad_dim(shard, shard_dim, ctx.chunk_size)
        out = _gather_on(shard, shard_dim, group_name)
        if global_size is not None and out.shape[shard_dim] != global_size:
            out = out.narrow(shard_dim, 0, global_size)
        return out

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):  # pyrefly: ignore[bad-override]
        if ctx.reduce_dtype is not None:
            grad = grad.to(ctx.reduce_dtype)
        if ctx.global_size is not None:
            degree = funcol.c10d._get_group_size_by_name(ctx.group_name)
            grad = _pad_dim(grad, ctx.shard_dim, ctx.chunk_size * degree)
        out = _scatter_on(grad, ctx.shard_dim, ctx.group_name)
        if ctx.replicate_group_name is not None:
            # HSDP: the gradient is Partial on the replicate axis too. Reduce it
            # after the scatter, not before -- same sum, on a tensor that is
            # already `1/shard_degree` the size.
            out = _waited(funcol.all_reduce(out, "sum", group=ctx.replicate_group_name))
        if out.shape[ctx.shard_dim] != ctx.local_size:
            out = out.narrow(ctx.shard_dim, 0, ctx.local_size)
        return out.to(ctx.shard_dtype), None, None, None, None, None, None


class _ReduceGradOnly(torch.autograd.Function):
    """Sum the gradient of a weight that is replicated on the mesh.

    The forward only casts: the weight is already whole on every rank. The
    gradient is not -- each rank computed its own from its own microbatch -- so
    the backward all-reduces it over every mesh axis the weight is replicated
    on. Unrolled under SimpleFSDP that sum came from
    ``to_local(grad_placements=Partial)``.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        replica: torch.Tensor,
        group_names: tuple[str, ...],
        param_dtype: torch.dtype | None,
        reduce_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        ctx.group_names = group_names
        ctx.reduce_dtype = reduce_dtype
        ctx.replica_dtype = replica.dtype
        return replica if param_dtype is None else replica.to(param_dtype)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):  # pyrefly: ignore[bad-override]
        if ctx.reduce_dtype is not None:
            grad = grad.to(ctx.reduce_dtype)
        for group_name in ctx.group_names:
            grad = _waited(funcol.all_reduce(grad, "sum", group=group_name))
        return grad.to(ctx.replica_dtype), None, None, None


def unshard_local(
    shard: torch.Tensor,
    *,
    group_name: str,
    shard_dim: int | None,
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    replicate_group_name: str | None = None,
    global_size: int | None = None,
) -> torch.Tensor:
    """Turn a local shard into the whole weight, with the backward that undoes it.

    ``shard_dim=None`` means the weight is replicated on the shard axis: no
    gather, but the gradient still owes a sum on every axis. An empty
    ``group_name`` means no mesh at all (single process): only the cast.
    """
    if shard_dim is None:
        groups = tuple(g for g in (group_name, replicate_group_name) if g)
        if not groups:
            return shard if param_dtype is None else shard.to(param_dtype)
        return _ReduceGradOnly.apply(shard, groups, param_dtype, reduce_dtype)
    return _AllGatherReduceScatter.apply(
        shard,
        group_name,
        shard_dim,
        param_dtype,
        reduce_dtype,
        replicate_group_name,
        global_size,
    )


def unshard(shard: torch.Tensor, spec: ShardSpec) -> torch.Tensor:
    """The whole weight described by ``spec``, from this rank's ``shard``."""
    d = spec.shard_dim
    return unshard_local(
        shard,
        group_name=spec.shard_group,
        shard_dim=d,
        param_dtype=spec.param_dtype,
        reduce_dtype=spec.reduce_dtype,
        replicate_group_name=spec.replicate_group,
        global_size=None if d is None or spec.even else spec.global_shape[d],
    )


def gather_slice(shard: torch.Tensor, spec: ScanFSDPSpec | None) -> torch.Tensor:
    """Turns one layer's local shard into that layer's whole weight (scan body)."""
    if spec is None:
        return shard
    return unshard_local(
        shard,
        group_name=spec.group_name,
        shard_dim=spec.gather_dim if spec.sharded else None,
        param_dtype=spec.param_dtype,
        reduce_dtype=spec.reduce_dtype,
        replicate_group_name=spec.replicate_group_name,
    )


# ---------------------------------------------------------------------------
# Sharding a module
# ---------------------------------------------------------------------------

# Whether reading a sharded parameter through its module attribute gathers it.
# Off inside `dtensor_params` (init, state dict), where the raw slot is wanted.
_gather_enabled = True


@contextlib.contextmanager
def disable_gather() -> Iterator[None]:
    """Make parameter attributes return the raw slot instead of gathering."""
    global _gather_enabled
    previous = _gather_enabled
    _gather_enabled = False
    try:
        yield
    finally:
        _gather_enabled = previous


def _read_param(module: nn.Module, name: str) -> torch.Tensor:
    param = module._parameters[name]
    if not _gather_enabled or param is None or isinstance(param, DTensor):
        return param
    return unshard(param, getattr(module, SPECS_ATTR)[name])


_wrap_class_id = count()
_wrap_class_cache: dict[tuple[type, tuple[str, ...]], type] = {}


def _install_properties(module: nn.Module, names: list[str]) -> None:
    """Make ``module.<name>`` return the gathered weight.

    Same mechanism as SimpleFSDP's: the module's class is swapped for a
    subclass carrying one property per sharded parameter, shared across
    modules of the same class so per-layer compile guards on ``type(self)``
    hit. The property reads this module's spec, so the class holds no state.
    """
    key = (module.__class__, tuple(names))
    cls = _wrap_class_cache.get(key)
    if cls is None:
        attrs = {
            name: property(lambda self, name=name: _read_param(self, name))
            for name in names
        }
        cls = type(
            f"TpuFSDP{module.__class__.__name__}_{next(_wrap_class_id)}",
            (module.__class__,),
            attrs,
        )
        # A real, importable symbol, so pickle / GraphPickler can resolve it.
        import sys  # pylint: disable=g-import-not-at-top

        sys.modules[cls.__module__].__dict__[cls.__name__] = cls
        _wrap_class_cache[key] = cls
    module.__class__ = cls


def _placements(
    mode: str, shard_dim: int, shape: torch.Size, mesh: DeviceMesh
) -> tuple[Placement, ...]:
    """SimpleFSDP's placement rule: shard, unless the axis is narrower than the mesh."""
    if mode == "replicate":
        return (Replicate(),)
    if mode == "fully_shard":
        base: tuple[Placement, ...] = (Shard(shard_dim),)
    elif mode == "hybrid_shard":
        base = (Replicate(), Shard(shard_dim))
    else:
        raise ValueError(f"Unsupported mode {mode}")
    return tuple(
        Replicate()
        if isinstance(p, Shard) and shape[p.dim] < mesh.size(i)
        else p
        for i, p in enumerate(base)
    )


def shard_module(
    module: nn.Module,
    mesh: DeviceMesh,
    mode: str,
    *,
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    shard_dim: int | dict[str, int] = 0,
    ignored_params: set[nn.Parameter] | None = None,
) -> None:
    """Replace ``module``'s direct parameters with local shards, in place.

    Each parameter becomes an ``nn.Parameter`` holding this rank's shard, with
    its :class:`ShardSpec` recorded in ``module._tpu_fsdp_specs``, and the
    module's attribute for it gathers on read. A module already sharded here
    is left alone, so a submodule sharded on its own axis first is not
    re-sharded by a later whole-model pass.

    ``shard_dim`` names the tensor axis to split, or one axis per parameter.
    """
    if hasattr(module, SPECS_ATTR):
        return
    if mode == "hybrid_shard" and mesh.ndim != 2:
        raise ValueError("hybrid_shard needs a 2-D (dp_replicate, fsdp) mesh")
    if mode in ("fully_shard", "replicate") and mesh.ndim != 1:
        raise ValueError(f"{mode} needs a 1-D mesh, got {mesh.ndim}-D")

    specs: dict[str, ShardSpec] = {}
    for name, param in list(module._parameters.items()):
        if param is None or param.numel() == 0:
            continue
        if ignored_params is not None and param in ignored_params:
            continue
        if isinstance(param, DTensor):
            raise NotImplementedError(
                f"{name} is a DTensor (TP/EP sharded); explicit TPU FSDP shards "
                "plain tensors only"
            )
        dim = shard_dim[name] if isinstance(shard_dim, dict) else shard_dim
        if dim < 0 or dim >= param.ndim:
            raise ValueError(f"shard dim {dim} out of range for {name}{tuple(param.shape)}")
        spec = ShardSpec(
            mesh=mesh,
            placements=_placements(mode, dim, param.shape, mesh),
            global_shape=tuple(param.shape),
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        )
        local = param.detach()
        d = spec.shard_dim
        if d is not None:
            shape, offset = spec.local_shape_and_offset()
            # `narrow` on a meta tensor is free; on a materialized one the
            # `contiguous` copy is what lets the whole tensor be freed.
            local = local.narrow(d, offset[d], shape[d]).contiguous()
        module._parameters[name] = nn.Parameter(local, requires_grad=param.requires_grad)
        specs[name] = spec
    if not specs:
        return
    setattr(module, SPECS_ATTR, specs)
    _install_properties(module, list(specs))


def uses_tpu_fsdp(model: nn.Module) -> bool:
    return any(hasattr(m, SPECS_ATTR) for m in model.modules())


def param_specs(model: nn.Module) -> dict[int, ShardSpec]:
    """``id(parameter) -> spec`` for every explicitly sharded parameter."""
    out: dict[int, ShardSpec] = {}
    for module in model.modules():
        specs = getattr(module, SPECS_ATTR, None)
        if not specs:
            continue
        for name, spec in specs.items():
            param = module._parameters.get(name)
            if param is not None:
                out[id(param)] = spec
    return out


def param_specs_for(model_parts: list[nn.Module]) -> dict[int, ShardSpec]:
    """``id(parameter) -> spec`` over every model part, from the module tables.

    Resolved from the modules, not from a stamp on the tensor: the graph
    trainer re-registers parameters after sharding (its own unwrapping and
    static-input handling), which creates new Parameter objects, while the
    module's ``_tpu_fsdp_specs`` table keys them by name and stays valid.
    """
    out: dict[int, ShardSpec] = {}
    for model in model_parts:
        out.update(param_specs(model))
    return out


def param_specs_by_fqn(model_parts: list[nn.Module]) -> dict[str, ShardSpec]:
    """``parameter FQN -> spec`` over every model part, from the module tables.

    The optimizer names its parameters by FQN (``param_names``), which is what
    its state dict is keyed by; the module tables are read at call time, so a
    parameter re-registered after sharding is still found by its name.
    """
    out: dict[str, ShardSpec] = {}
    for model in model_parts:
        for module_fqn, module in model.named_modules():
            specs = getattr(module, SPECS_ATTR, None)
            if not specs:
                continue
            prefix = f"{module_fqn}." if module_fqn else ""
            for name, spec in specs.items():
                out[prefix + name] = spec
    return out


def register_sharded_buffer(
    module: nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    like: str,
    persistent: bool = True,
) -> None:
    """Register ``tensor`` as a buffer shaped like ``module``'s parameter ``like``.

    If that parameter is sharded here, the buffer is this rank's shard of a
    tensor the parameter's spec describes, and :func:`dtensor_params` presents
    it as that DTensor view: the fused optimizer's moments checkpoint and
    reshard exactly like the weights they belong to. On a module that is not
    sharded this is ``register_buffer``.
    """
    module.register_buffer(name, tensor, persistent=persistent)
    specs = getattr(module, SPECS_ATTR, None)
    if not specs or like not in specs:
        return
    if tuple(tensor.shape) != tuple(module._parameters[like].shape):
        raise ValueError(
            f"{name} has shape {tuple(tensor.shape)}, but the shard of {like} it "
            f"is registered like has {tuple(module._parameters[like].shape)}"
        )
    table = getattr(module, BUFFER_SPECS_ATTR, None)
    if table is None:
        table = {}
        setattr(module, BUFFER_SPECS_ATTR, table)
    table[name] = like


# ---------------------------------------------------------------------------
# The DTensor windows: initialization and checkpoints
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def dtensor_params(model: nn.Module) -> Iterator[None]:
    """Present every sharded parameter as a DTensor view for the block.

    The views share the shards' storage, so writes land in the shards. Reading
    a parameter attribute returns the view rather than gathering. Modules
    whose parameters are temporarily hidden (a layer stack with its per-layer
    views up) have nothing to swap and are skipped; the stack builds its own
    row views from the same specs.
    """
    swapped: list[tuple[nn.Module, str, nn.Parameter, nn.Parameter]] = []
    swapped_buffers: list[tuple[nn.Module, str, torch.Tensor]] = []
    with disable_gather():
        try:
            for module in model.modules():
                specs = getattr(module, SPECS_ATTR, None)
                if not specs:
                    continue
                for name, spec in specs.items():
                    param = module._parameters.get(name)
                    if param is None or isinstance(param, DTensor):
                        continue
                    view = nn.Parameter(
                        spec.to_dtensor(param.detach()),
                        requires_grad=param.requires_grad,
                    )
                    module._parameters[name] = view
                    swapped.append((module, name, param, view))
                for name, like in getattr(module, BUFFER_SPECS_ATTR, {}).items():
                    buffer = module._buffers.get(name)
                    if buffer is None or isinstance(buffer, DTensor):
                        continue
                    module._buffers[name] = specs[like].to_dtensor(buffer)
                    swapped_buffers.append((module, name, buffer))
            yield
        finally:
            for module, name, buffer in swapped_buffers:
                module._buffers[name] = buffer
            # `Decoder.init_states` re-ties `tok_embeddings.weight` to
            # `lm_head.weight` inside this window. A slot that now holds a
            # different module's view is re-tied to that module's shard, so
            # the tie survives the window the way it does under SimpleFSDP.
            plain_of_view = {id(view): param for _, _, param, view in swapped}
            for module, name, param, view in swapped:
                current = module._parameters.get(name)
                if current is view:
                    module._parameters[name] = param
                elif id(current) in plain_of_view:
                    module._parameters[name] = plain_of_view[id(current)]


def install_optimizer_state_hooks(
    optimizers: Iterable[torch.optim.Optimizer], model_parts: list[nn.Module]
) -> None:
    """Present parameter-shaped optimizer state as DTensor views in ``state_dict``.

    An optimizer over plain shards keeps plain per-rank moments. Handed to DCP
    under one key per parameter they look like one replicated tensor, so a
    single rank's copy is written and every rank reads it back. The post-hook
    wraps each tensor shaped like its parameter's shard in that parameter's
    view (``step`` and other scalars pass through); the load pre-hook unwraps
    what the checkpoint hands back, so the optimizer's own ``load_state_dict``
    sees plain tensors again. Specs are resolved by FQN at call time.
    """
    for optim in optimizers:
        if getattr(optim, "_tpu_fsdp_state_hooks", False):
            continue

        def index_to_spec(optim: torch.optim.Optimizer) -> dict[int, ShardSpec]:
            # `Optimizer.state_dict` numbers parameters across the groups in
            # order; `param_names` sits beside `params` in each group.
            specs = param_specs_by_fqn(model_parts)
            out: dict[int, ShardSpec] = {}
            index = 0
            for group in optim.param_groups:
                names = group.get("param_names")
                if names is None:
                    raise ValueError(
                        "the optimizer must be built with (name, param) tuples so "
                        "its state can be matched to the parameters' shard specs"
                    )
                for name in names:
                    spec = specs.get(name)
                    if spec is not None:
                        out[index] = spec
                    index += 1
            return out

        def as_views(optim: torch.optim.Optimizer, state_dict: dict[str, Any]):
            specs = index_to_spec(optim)
            state: dict[Any, Any] = {}
            for index, per_param in state_dict["state"].items():
                spec = specs.get(index)
                if spec is None:
                    state[index] = per_param
                    continue
                local_shape, _ = spec.local_shape_and_offset()
                state[index] = {
                    key: spec.to_dtensor(value)
                    if isinstance(value, torch.Tensor)
                    and not isinstance(value, DTensor)
                    and tuple(value.shape) == local_shape
                    else value
                    for key, value in per_param.items()
                }
            return {**state_dict, "state": state}

        def as_locals(optim: torch.optim.Optimizer, state_dict: dict[str, Any]):
            del optim
            state = {
                index: {
                    key: value.to_local() if isinstance(value, DTensor) else value
                    for key, value in per_param.items()
                }
                for index, per_param in state_dict["state"].items()
            }
            return {**state_dict, "state": state}

        optim.register_state_dict_post_hook(as_views)
        optim.register_load_state_dict_pre_hook(as_locals)
        optim._tpu_fsdp_state_hooks = True


def _install_root_hooks(model: nn.Module) -> None:
    """Open the DTensor window around init and around state-dict traffic."""
    if getattr(model, "_tpu_fsdp_root_hooks", False):
        return
    inner_init = model.init_states

    def init_states(**kwargs):
        with dtensor_params(model):
            inner_init(**kwargs)

    model.init_states = init_states

    # torch's API is a pre-hook and a post-hook with the caller's own call in
    # between, so the context is entered in one and left in the other. The two
    # paths (save, load) never nest.
    open_windows: list[Any] = []

    def enter(*args, **kwargs):
        del args, kwargs
        window = dtensor_params(model)
        window.__enter__()
        open_windows.append(window)

    def leave(*args, **kwargs):
        del args, kwargs
        open_windows.pop().__exit__(None, None, None)

    model.register_state_dict_pre_hook(enter)
    model.register_state_dict_post_hook(leave)
    model.register_load_state_dict_pre_hook(enter)
    model.register_load_state_dict_post_hook(leave)
    model._tpu_fsdp_root_hooks = True


# ---------------------------------------------------------------------------
# Applying to a model
# ---------------------------------------------------------------------------


def apply_tpu_fsdp(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
) -> nn.Module:
    """Shard ``model`` explicitly over the data-parallel mesh.

    The TPU counterpart of ``apply_simple_fsdp``: the same meshes, the same
    modes (``fully_shard`` / ``hybrid_shard`` / ``replicate``), the same
    mixed-precision policy from ``training``, the same per-submodule
    ``_fsdp_shard_dim`` hook, and the same separate wrap of routed experts on
    the expert-FSDP mesh under EP. Model-parallel axes other than EP are not
    handled: SimpleFSDP owns the TP-axis gradient reductions, and nothing here
    replaces that yet.
    """
    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "explicit TPU FSDP does not handle tensor parallelism yet "
            "(the TP-axis reductions SimpleFSDP inserts for replicated weights)"
        )
    use_spmd_types = get_spmd_backend() == "spmd_types"
    if use_spmd_types:
        fsdp_mesh = parallel_dims.get_optional_mesh(
            ["dp_shard", "cp"], include_singleton_axes=True
        )
        assert fsdp_mesh is not None
        fsdp_mesh = fsdp_mesh._flatten("fsdp")
    else:
        fsdp_mesh = parallel_dims.get_mesh("fsdp")

    if parallel_dims.dp_replicate_enabled:
        if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
            if use_spmd_types:
                dp_replicate_mesh = parallel_dims.get_optional_mesh(
                    "dp_replicate", include_singleton_axes=True
                )
                assert dp_replicate_mesh is not None
                dp_mesh = DeviceMesh._concatenate([dp_replicate_mesh, fsdp_mesh])
            else:
                dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
            dp_mode = "hybrid_shard"
        else:
            dp_mesh = parallel_dims.get_mesh("dp_replicate")
            dp_mode = "replicate"
    else:
        dp_mesh = fsdp_mesh
        dp_mode = "fully_shard"

    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
    dtypes = dict(param_dtype=param_dtype, reduce_dtype=reduce_dtype)

    if parallel_dims.ep_enabled and isinstance(model, Decoder):
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        assert edp_mesh is not None
        for _, transformer_block in model.layers.items():
            if not isinstance(transformer_block, TransformerBlock):
                continue
            moe = getattr(transformer_block, "moe", None)
            if moe is None:
                continue
            inner_experts = moe.routed_experts.inner_experts
            experts_shard_dim = 0
            if edp_mesh["efsdp"].size() * parallel_dims.ep > inner_experts.num_experts:
                experts_shard_dim = 1
            shard_module(
                inner_experts, edp_mesh, dp_mode, shard_dim=experts_shard_dim, **dtypes
            )

    # A submodule carrying `_fsdp_shard_dim` is sharded on that axis instead of
    # the default -- layer stacking sets it, because dim 0 of a stack is the
    # layer axis. Run first: `shard_module` skips a module already sharded.
    for sub in [m for m in model.modules() if hasattr(m, "_fsdp_shard_dim")]:
        shard_module(sub, dp_mesh, dp_mode, shard_dim=sub._fsdp_shard_dim, **dtypes)
    for module in model.modules():
        shard_module(module, dp_mesh, dp_mode, shard_dim=0, **dtypes)

    _install_root_hooks(model)
    logger.info(
        "Applied explicit TPU FSDP (dp mode=%s, %d sharded parameter(s)) to the model",
        dp_mode,
        len(param_specs(model)),
    )
    return model


# ---------------------------------------------------------------------------
# Gradient norm
# ---------------------------------------------------------------------------


def grads_as_dtensors(
    grads: list[torch.Tensor], specs: list[ShardSpec | None]
) -> list[torch.Tensor]:
    """Wrap each shard's gradient as the DTensor its spec describes.

    The gradient norm is then computed by the same DTensor code path SimpleFSDP
    used (``get_total_norm`` on DTensors, ``full_tensor`` for the reduction),
    which keeps its floating-point order and so the clipped step bit-identical.
    This is the third and last DTensor window; it exists only inside the norm.
    """
    out = []
    for grad, spec in zip(grads, specs, strict=True):
        if spec is None:
            raise ValueError(
                "a gradient without a shard spec cannot be mixed into the explicit "
                "FSDP norm; every parameter the graph trainer trains is sharded"
            )
        out.append(spec.to_dtensor(grad))
    return out
