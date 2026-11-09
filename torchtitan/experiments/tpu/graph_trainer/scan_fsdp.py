"""All-gather one decoder layer's weights, inside the scan body.

SimpleFSDP gathers on attribute access. Reading a stacked parameter therefore
pulls the whole ``[n_layers, ...]`` tensor, before the loop that consumes it, so
every layer's weights are resident at once. That stops fitting past a few tens
of billions of parameters.

Here the body takes this rank's shard of one layer and gathers that instead --
``n_layers`` times smaller. A functional collective traces into a scan body, so
the gather lowers into the loop and lands next to the compute that needs it.

``plan_stacked_gathers`` reads back what SimpleFSDP installed on the stacks;
``gather_slice`` is what the body calls on one row.

Recomputing this gather in the backward is safe, despite upstream's *Note
[Recomputing collectives in the partitioner]*: what that bans is a duplication
some ranks make and others don't, and every rank here compiles the same SPMD
graph.
"""

import dataclasses
import math
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard

from torchtitan.tools.logging import logger


__all__ = [
    "ScanFSDPSpec",
    "configure_simple_fsdp_shard_dims",
    "gather_slice",
    "plan_stacked_gathers",
]


@dataclasses.dataclass(frozen=True)
class ScanFSDPSpec:
    """How the scan body should gather one stacked parameter.

    Attributes:
        group_name: resolved process-group name, in the form funcol collectives
            take.
        param_dtype: compute dtype for the gathered weight; ``None`` leaves it
            alone.
        reduce_dtype: dtype to reduce the gradient in; ``None`` means reduce in
            whatever dtype the forward gathered in.
        sharded: ``False`` for a parameter SimpleFSDP left replicated. Nothing
            to gather in the forward; the gradient still has to be summed.
        replicate_group_name: under ``hybrid_shard``, the process group of the
            *replicate* mesh axis; ``None`` for a 1-D dp mesh. The forward needs
            nothing from it -- the weight is already replicated along it -- but
            the gradient is ``Partial`` on it and has to be summed.
        gather_dim: which axis of the per-layer slice to gather on -- the
            stack's shard axis minus the scan axis, so dim 1 becomes 0.
    """

    group_name: str
    param_dtype: torch.dtype | None
    reduce_dtype: torch.dtype | None
    sharded: bool
    replicate_group_name: str | None = None
    gather_dim: int = 0


_orig_all_gather_single = funcol.all_gather_single
_orig_reduce_scatter_single = funcol.reduce_scatter_single


def _fold_rows(t: torch.Tensor, k: int) -> torch.Tensor:
    """Views 2-D `t` as `[rows * k, cols // k]`.

    `.contiguous()` can't be used here: it is a no-op at trace time when the
    fake tensor looks row-major, but TorchTPU may replay the graph with a
    transposed input (e.g. the fused loss's `lm_head` gradient), and `view`
    would then read it in the wrong order.
    `clone(memory_format=torch.contiguous_format)` always records the copy.
    """
    return t.clone(memory_format=torch.contiguous_format).view(
        t.shape[0] * k, t.shape[1] // k
    )


def _movedim_all_gather_tensor(
    self: torch.Tensor,
    gather_dim: int,
    group: Any,
    tag: str = "",
) -> torch.Tensor:
    """TPU-friendly drop-in replacement for `funcol.all_gather_single` / `all_gather_tensor`.

    Addresses two TPU performance issues in PyTorch's default all-gather:

    1. Gathering on `gather_dim > 0`:
       - Problem: PyTorch's default `funcol` implementation splits the output
         into `world_size` chunks and concatenates them along `gather_dim`,
         creating thousands of slice/concat ops at large scale.
       - Solution: Transpose `gather_dim` to dim 0, run a single contiguous
         dim-0 all-gather, and transpose back.

    2. Unaligned 2-D weight shards (`rows % 8 != 0`):
       - Problem: TPU stores 2-D tensors in `(8, 128)` tiles. At high FSDP
         degrees, a rank's 2-D shard can have a row count not divisible by 8
         (e.g. 18 or 36 rows for fused `wqkv` at 512 ranks), forcing XLA to pad
         or relayout the collective.
       - Solution: Reshape `[rows, cols]` into `[rows * k, cols // k]` before
         the all-gather so the row count is a multiple of 8, then reshape back
         to `[-1, cols]` afterward.
    """
    if gather_dim < 0:
        gather_dim += self.ndim
    t = self.movedim(gather_dim, 0).contiguous() if gather_dim != 0 else self
    # Only 2-D shards fold: 1-D norms use 1-D tiles, and 3-D expert slices
    # `[E, D, F]` already tile on `(D, F)`.
    orig_cols = None
    if t.ndim == 2 and t.shape[0] % 8 != 0:
        k = 8 // math.gcd(t.shape[0], 8)
        if t.shape[1] % (k * 128) == 0:
            orig_cols = t.shape[1]
            t = _fold_rows(t, k)
    out = _orig_all_gather_single(t, gather_dim=0, group=group, tag=tag)
    if orig_cols is not None:
        out = out.reshape(-1, orig_cols)
    return out.movedim(0, gather_dim) if gather_dim != 0 else out


def _movedim_reduce_scatter_tensor(
    self: torch.Tensor,
    reduceOp: str,
    scatter_dim: int,
    group: Any,
    tag: str = "",
) -> torch.Tensor:
    """TPU-friendly drop-in replacement for `funcol.reduce_scatter_single` / `reduce_scatter_tensor`.

    The backward counterpart of `_movedim_all_gather_tensor`, applying the same
    two fixes:
    1. Transposes `scatter_dim` to dim 0 before reduce-scattering and back after.
    2. If a 2-D tensor's per-rank scattered output would have `shard_rows % 8 != 0`,
       folds `cols` into `rows` by `k = 8 // gcd(shard_rows, 8)` before the
       reduce-scatter and reshapes back after.
    """
    if scatter_dim < 0:
        scatter_dim += self.ndim
    t = self.movedim(scatter_dim, 0).contiguous() if scatter_dim != 0 else self
    orig_cols = None
    if t.ndim == 2:
        try:
            group_size = funcol.c10d._get_group_size_by_name(
                funcol._resolve_group(group, tag)
            )
        except Exception:  # pylint: disable=broad-except
            group_size = 0
        shard_rows = t.shape[0] // group_size if group_size > 0 else 0
        if (
            shard_rows > 0
            and t.shape[0] % group_size == 0
            and shard_rows % 8 != 0
        ):
            k = 8 // math.gcd(shard_rows, 8)
            if t.shape[1] % (k * 128) == 0:
                orig_cols = t.shape[1]
                t = _fold_rows(t, k)
    out = _orig_reduce_scatter_single(
        t, reduceOp=reduceOp, scatter_dim=0, group=group, tag=tag
    )
    if orig_cols is not None:
        out = out.reshape(-1, orig_cols)
    return out.movedim(0, scatter_dim) if scatter_dim != 0 else out


funcol.all_gather_single = _movedim_all_gather_tensor
funcol.all_gather_tensor = _movedim_all_gather_tensor
funcol.reduce_scatter_single = _movedim_reduce_scatter_tensor
funcol.reduce_scatter_tensor = _movedim_reduce_scatter_tensor


def _gather_on(x: torch.Tensor, dim: int, group_name: str) -> torch.Tensor:
    """All-gather `x` along `dim`, always collecting on dim 0.

    Off dim 0, `funcol` falls back to `cat(chunk(t, world_size))` unless every
    axis before `dim` has size 1. A layer slice starts with `num_experts`, so it
    never qualifies, and a 2048-rank gather would emit 2048 slices and a concat.
    A transpose either side is two ops instead.
    """
    if dim == 0:
        return funcol.all_gather_tensor(x, gather_dim=0, group=group_name)
    out = funcol.all_gather_tensor(x.movedim(dim, 0), gather_dim=0, group=group_name)
    return out.movedim(0, dim)


def _scatter_on(x: torch.Tensor, dim: int, group_name: str) -> torch.Tensor:
    """Reduce-scatter `x` along `dim`, always scattering on dim 0.

    `reduce_scatter` has no fast path off dim 0 at all, so this is the same
    trade as `_gather_on`.
    """
    if dim == 0:
        return funcol.reduce_scatter_tensor(
            x, reduceOp="sum", scatter_dim=0, group=group_name
        )
    out = funcol.reduce_scatter_tensor(
        x.movedim(dim, 0), reduceOp="sum", scatter_dim=0, group=group_name
    )
    return out.movedim(0, dim)


class _AllGatherReduceScatter(torch.autograd.Function):
    """Gathers one sharded layer's weight, and reduces its gradient back.

    This is the sharded case of ``gather_slice``: all-gather in the forward,
    reduce-scatter in the backward, the same pair SimpleFSDP runs unrolled.

    Written out by hand so the two halves can use different dtypes -- bf16 to
    gather, to halve the comm volume, and fp32 to reduce, to keep the gradient
    accurate. A plain collective reduces in whatever dtype the forward used,
    and casting after the reduce is already too late.
    """

    @staticmethod
    def forward(
        ctx: Any,
        shard: torch.Tensor,
        group_name: str,
        param_dtype: torch.dtype | None,
        reduce_dtype: torch.dtype | None,
        replicate_group_name: str | None,
        gather_dim: int,
    ) -> torch.Tensor:
        ctx.group_name = group_name
        ctx.replicate_group_name = replicate_group_name
        ctx.reduce_dtype = reduce_dtype
        ctx.shard_dtype = shard.dtype
        ctx.gather_dim = gather_dim
        if param_dtype is not None:
            shard = shard.to(param_dtype)
        return _gather_on(shard, gather_dim, group_name)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        if ctx.reduce_dtype is not None:
            grad = grad.to(ctx.reduce_dtype)
        out = _scatter_on(grad, ctx.gather_dim, ctx.group_name)
        if ctx.replicate_group_name is not None:
            # HSDP: the gradient is Partial on the replicate axis too. Reduce it
            # after the scatter, not before -- same sum, on a tensor that is
            # already `1/shard_degree` the size.
            out = funcol.all_reduce(out, "sum", group=ctx.replicate_group_name)
        return out.to(ctx.shard_dtype), None, None, None, None, None


class _ReduceGradOnly(torch.autograd.Function):
    """Sums the gradient of a weight SimpleFSDP left replicated.

    This is the unsharded case of ``gather_slice``: the weight is already whole
    on every rank, so the forward only casts. The gradient is not whole -- each
    rank computed its own from its own microbatch -- so the backward all-reduces.

    Unrolled, that sum comes from ``to_local(grad_placements=Partial)``. The
    body reads the local tensor straight off the ``DTensor``, which promises no
    such thing, so it has to ask for the sum itself.
    """

    @staticmethod
    def forward(
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
    def backward(ctx: Any, grad: torch.Tensor):
        if ctx.reduce_dtype is not None:
            grad = grad.to(ctx.reduce_dtype)
        for group_name in ctx.group_names:
            grad = funcol.all_reduce(grad, "sum", group=group_name)
        return grad.to(ctx.replica_dtype), None, None, None


def gather_like(x: torch.Tensor, spec: ScanFSDPSpec) -> torch.Tensor:
    """All-gathers ``x`` as ``gather_slice`` gathers a sharded layer, no grad.

    For tensors derived from a shard -- e.g. its FP8 bits or per-column
    amax -- that must be gathered across the same group along the same axis.
    """
    return _gather_on(x, spec.gather_dim, spec.group_name)


def gather_slice(shard: torch.Tensor, spec: ScanFSDPSpec | None) -> torch.Tensor:
    """Turns one layer's local shard into that layer's whole weight."""
    if spec is None:
        return shard
    if not spec.sharded:
        if not spec.group_name:
            # No mesh at all, so no other rank holds a gradient to add.
            return shard if spec.param_dtype is None else shard.to(spec.param_dtype)
        # Replicated on every axis it has, so the gradient is `Partial` on
        # every axis and each one needs its own sum.
        return _ReduceGradOnly.apply(
            shard,
            tuple(g for g in (spec.group_name, spec.replicate_group_name) if g),
            spec.param_dtype,
            spec.reduce_dtype,
        )
    return _AllGatherReduceScatter.apply(
        shard,
        spec.group_name,
        spec.param_dtype,
        spec.reduce_dtype,
        spec.replicate_group_name,
        spec.gather_dim,
    )


def plan_stacked_gathers(holder: nn.Module) -> dict[str, ScanFSDPSpec]:
    """Save what the body needs to know about each stack, before ``to_local``.

    The scan takes plain tensors, so the placements and dtypes that told
    SimpleFSDP how to gather are gone by the time the body runs. This reads
    them off the holder while they are still there and keeps them in a
    ``ScanFSDPSpec``, which is what ``gather_slice`` then dispatches on.

    The caller runs this on the first forward, not at construction: the model
    stacks itself in ``__init__``, which is before ``data_parallel`` and
    ``to_empty``, so there is nothing to read back yet.

    Raises:
        ValueError: on a mesh, placement or shard size the body cannot gather.
    """
    # Registered by `_register_parametrization` (see `simple_fsdp.py`). It is
    # missing only when the holder never went through `data_parallel`, and in
    # that case there is no mixed precision to mirror either.
    computation = getattr(holder, "_simple_fsdp_parametrization_list", [None])[0]
    param_dtype = computation.param_dtype if computation else None
    reduce_dtype = computation.reduce_dtype if computation else None

    plan: dict[str, ScanFSDPSpec] = {}
    n_sharded = 0
    for name, param in holder._parameters.items():
        if param is None:
            continue
        if not isinstance(param, DTensor):
            # Not under FSDP at all, so there is nothing to gather. Still
            # honour the compute dtype, so the scanned and unrolled paths agree.
            plan[name] = ScanFSDPSpec("", param_dtype, reduce_dtype, sharded=False)
            continue

        mesh = param.device_mesh
        # 1-D: `fully_shard`, one Shard placement. 2-D: `hybrid_shard`, which
        # SimpleFSDP builds as exactly (Replicate(), Shard(dim)) over
        # ("dp_replicate", "fsdp") -- so the shard axis is the last one, and the
        # first contributes only the backward all-reduce.
        if mesh.ndim == 1:
            replicate_group_name = None
        elif mesh.ndim == 2 and isinstance(param.placements[0], Replicate):
            replicate_group_name = mesh.get_group(0).group_name
        else:
            raise ValueError(
                "the scan body's gather understands a 1-D dp mesh or a 2-D "
                f"hybrid_shard mesh, got {mesh.ndim}-D with placements "
                f"{param.placements} for {name}"
            )
        placement = param.placements[-1]
        if isinstance(placement, Shard):
            if placement.dim == 0:
                raise ValueError(
                    f"{name} is sharded on dim 0, but dim 0 is the scan axis "
                    "and the body only ever sees one layer"
                )
            # Only the last axis shards the tensor, so only its degree matters.
            # `mesh.size()` is every axis multiplied together, which under
            # hybrid_shard asks for `replicate * shard` and rejects sizes that
            # shard fine.
            shard_degree = mesh.size(mesh.ndim - 1)
            if param.shape[placement.dim] % shard_degree:
                # Uneven shards. `redistribute` pads around its all-gather, so
                # the unrolled path handles them; a bare collective cannot.
                raise ValueError(
                    f"{name} has dim {placement.dim} = {param.shape[placement.dim]}, "
                    f"which {shard_degree} ranks cannot shard evenly, and the "
                    "scan body's all-gather needs equal shards"
                )
            gather_dim = placement.dim - 1
            sharded = True
            n_sharded += 1
        elif isinstance(placement, Replicate):
            # `data_parallel` downgrades Shard to Replicate when the shard axis
            # is smaller than the mesh. Mirror that: no gather, but keep the
            # group, because the backward still has to sum the gradient.
            sharded = False
            gather_dim = 0
        else:
            # `Partial` would reach the body as a tensor still owing a sum, and
            # skipping it here would read it as a finished weight.
            raise ValueError(
                f"{name} is {placement} on the shard axis, and the scan body's "
                "gather handles only Shard and Replicate"
            )
        plan[name] = ScanFSDPSpec(
            group_name=mesh.get_group(mesh.ndim - 1).group_name,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            sharded=sharded,
            replicate_group_name=replicate_group_name,
            gather_dim=gather_dim,
        )

    logger.info(
        "Gathering %d/%d stacked weights per layer inside the scan body",
        n_sharded,
        len(plan),
    )
    return plan


def configure_simple_fsdp_shard_dims(
    model: nn.Module,
    parallel_dims: Any,
) -> list[str]:
    """Picks SimpleFSDP shard axes that divide evenly, for any decoder model.

    Plans an axis per layer stack (calling ``flatten_for`` first if present) and
    moves ``tok_embeddings`` / ``lm_head`` to their hidden axis (dim 1) when the
    vocabulary does not divide the FSDP degree (e.g. Llama 3's 128,256 over 512+ ranks).

    Args:
        model: The model to configure, before ``apply_simple_fsdp``.
        parallel_dims: Parallel dimensions; FSDP degree is ``dp_shard * cp``.

    Returns:
        FQNs of the modules moved to dim 1.
    """
    if parallel_dims is None:
        return []
    fsdp_degree = getattr(parallel_dims, "dp_shard", 1) * getattr(
        parallel_dims, "cp", 1
    )
    if fsdp_degree <= 1:
        return []

    # Layer stacking moves the layer weights into a holder, which plans an axis
    # per stack -- dim 1 of a stack is one axis further in than dim 1 of a layer.
    for holder in model.modules():
        if hasattr(holder, "plan_shard_dims"):
            if hasattr(holder, "flatten_for"):
                holder.flatten_for(fsdp_degree)
            holder._fsdp_shard_dim = holder.plan_shard_dims(fsdp_degree)

    hidden_axis_modules: list[str] = []
    for module_fqn in ("tok_embeddings", "lm_head"):
        module = getattr(model, module_fqn, None)
        if module is None or not hasattr(module, "weight"):
            continue
        weight = module.weight
        if weight.shape[0] % fsdp_degree == 0:
            continue
        if weight.ndim != 2 or (
            weight.shape[1] % fsdp_degree != 0 and weight.shape[1] >= fsdp_degree
        ):
            raise ValueError(
                f"Cannot FSDP-shard {module_fqn}.weight with shape "
                f"{tuple(weight.shape)} over {fsdp_degree} ranks"
            )
        # This is apply_simple_fsdp's documented per-submodule shard-axis hook,
        # also used by LayerStackHolder for stacked layer weights. When
        # `weight.shape[1] < fsdp_degree` (e.g. Llama 3 8B at 8192 ranks),
        # `data_parallel` downgrades `Shard(1)` to `Replicate()`.
        if weight.shape[1] < fsdp_degree:
            replicated_mib = weight.numel() * weight.element_size() / (1024**2)
            logger.warning(
                "%s.weight with shape %s has hidden dim %d < fsdp_degree %d; "
                "SimpleFSDP will replicate it (%.1f MiB parameter) instead of "
                "sharding.",
                module_fqn,
                tuple(weight.shape),
                weight.shape[1],
                fsdp_degree,
                replicated_mib,
            )
        setattr(module, "_fsdp_shard_dim", 1)
        hidden_axis_modules.append(module_fqn)
    return hidden_axis_modules
