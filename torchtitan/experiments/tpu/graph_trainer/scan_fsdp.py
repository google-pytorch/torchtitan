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

import math
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard

# The collectives and the scan-body spec live with the explicit TPU FSDP; this
# module keeps the SimpleFSDP-side plan (read off DTensor placements) and the
# TPU funcol fixes. Re-exported so existing imports keep resolving.
from torchtitan.experiments.tpu.graph_trainer.tpu_fsdp import (  # noqa: F401  pylint: disable=unused-import
    _gather_on,
    _scatter_on,
    gather_slice,
    ScanFSDPSpec,
)
from torchtitan.tools.logging import logger


__all__ = [
    "ScanFSDPSpec",
    "configure_simple_fsdp_shard_dims",
    "gather_slice",
    "plan_stacked_gathers",
]


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
