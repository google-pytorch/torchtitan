"""Unified TPU SimpleFSDP shard-axis selection and 2-D split-mesh sharding.

Consolidates TPU FSDP sharding policy across both ``use_scan_layers=True``
(stacked ``LayerStackHolder``) and ``use_scan_layers=False`` (unrolled decoder
layers):

1. ``configure_tpu_fsdp_shard_dims`` (called before ``apply_simple_fsdp``):
   - Plans per-stack shard axes on ``LayerStackHolder``.
   - Moves ``tok_embeddings``, ``lm_head``, and unstacked decoder ``Linear``
     modules to ``Shard(1)`` when their leading dimension does not divide
     ``fsdp_degree`` (warning if ``shape[1] < fsdp_degree`` causes replication).
   - Moves unstacked ``TpuGroupedExperts`` to ``Shard(1)`` when ``num_experts``
     does not divide ``fsdp_degree`` (either because ``shape[1]`` divides
     ``fsdp_degree``, or to prepare for 2-D split-mesh sharding when
     ``fsdp_degree > shape[1]`` and ``(num_experts * shape[1]) % fsdp_degree == 0``).

2. ``shard_split_fsdp_modules`` (called right after ``apply_simple_fsdp``):
   - Moves any stacked ``[L, A, B, ...]`` parameter on ``LayerStackHolder``
     (``start_dim=1``) or unstacked 3-D ``[A, B, ...]`` parameter on
     ``TpuGroupedExperts`` (``start_dim=0``) whose leading per-layer axis ``A``
     is narrower than the FSDP mesh onto a 2-D ``(A, n // A)`` split mesh with
     placement ``(Shard(start_dim), Shard(start_dim + 1))``, while gathering it
     flat in a single collective over the unsplit FSDP group.
"""

from typing import Any

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard

from torchtitan.experiments.graph_trainer.simple_fsdp import ReplicateComputation
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import (
    gather_slice,
    split_stack_spec,
)
from torchtitan.tools.logging import logger


__all__ = [
    "SplitStackReplicateComputation",
    "configure_simple_fsdp_shard_dims",
    "configure_tpu_fsdp_shard_dims",
    "shard_module_split_axes",
    "shard_split_fsdp_modules",
]


class SplitStackReplicateComputation(ReplicateComputation):
    """SimpleFSDP's parametrization, gathering 2-D split parameters flat.

    ``redistribute`` would gather a 2-D split parameter in two collectives, one
    per mesh axis; this runs the one flat all-gather (and reduce-scatter back)
    over the unsplit group along ``gather_dim`` (``1`` for stacked ``[L, A, B,
    ...]``, ``0`` for unstacked ``[A, B, ...]``). Everything else delegates to
    SimpleFSDP.
    """

    def __init__(
        self,
        base: ReplicateComputation,
        split_meshes: dict[DeviceMesh, DeviceMesh],
        gather_dim: int = 1,
    ):
        self.__dict__.update(base.__dict__)
        self.split_meshes = split_meshes
        self.gather_dim = gather_dim

    def replicate_compute(self, x: DTensor) -> torch.Tensor:
        unsplit = self.split_meshes.get(x.device_mesh)
        if unsplit is None:
            return super().replicate_compute(x)
        if getattr(self, "full_dtensor", False):
            raise NotImplementedError("full_dtensor on a split stack")
        spec = split_stack_spec(
            unsplit,
            self.param_dtype,
            self.reduce_dtype,
            self.gather_dim,
            x.shape,
        )
        return gather_slice(x.to_local(), spec)


def shard_module_split_axes(
    module: nn.Module,
    *,
    start_dim: int = 1,
    built: dict[tuple[DeviceMesh, int], DeviceMesh] | None = None,
) -> dict[DeviceMesh, DeviceMesh]:
    """Moves parameters of ``module`` whose ``start_dim`` the FSDP mesh outgrows onto a 2-D split mesh.

    For ``LayerStackHolder`` (``start_dim=1``), splits ``[L, A, B, ...]`` onto
    ``(A, n // A)`` as ``(Shard(1), Shard(2))``.
    For unstacked 3-D expert modules (``start_dim=0``), splits ``[A, B, ...]``
    onto ``(A, n // A)`` as ``(Shard(0), Shard(1))``.
    """
    computations = getattr(module, "_simple_fsdp_parametrization_list", None)
    if not computations or computations[0].mode == "replicate":
        return getattr(module, "split_meshes", {})
    if not hasattr(module, "split_meshes"):
        module.split_meshes = {}
    if built is None:
        built = {}
    min_ndim = start_dim + 2
    for name, param in list(module._parameters.items()):
        # Include Replicate: when every single axis is narrower than the mesh
        # (e.g. 30B at 4k ranks, 235B at 8k), data_parallel downgrades Shard to
        # Replicate even though axes `start_dim` and `start_dim + 1` together
        # divide.
        if not (
            isinstance(param, DTensor)
            and param.ndim >= min_ndim
            and isinstance(param.placements[-1], (Shard, Replicate))
            and param.device_mesh not in module.split_meshes
        ):
            continue
        mesh = param.device_mesh
        dim_a, dim_b = param.shape[start_dim], param.shape[start_dim + 1]
        ranks = mesh.size(mesh.ndim - 1)
        ranks_per_entry, rem = divmod(ranks, dim_a)
        if dim_a % ranks == 0 or rem or dim_b % ranks_per_entry:
            continue
        if not param.is_meta:
            raise ValueError(
                f"{name} is already materialized; call shard_split_stacks() "
                "before to_empty"
            )
        if (mesh, dim_a) not in built:
            axis = (mesh.mesh_dim_names or ("dp",))[-1]
            base = f"{axis}_{dim_a}x{ranks_per_entry}"
            # pylint: disable-next=protected-access
            built[(mesh, dim_a)] = mesh._unflatten(
                mesh.ndim - 1,
                (dim_a, ranks_per_entry),
                (f"{base}_outer", f"{base}_inner"),
            )
        module.split_meshes[built[(mesh, dim_a)]] = mesh
        module._parameters[name] = nn.Parameter(
            distribute_tensor(
                torch.empty(param.shape, dtype=param.dtype, device="meta"),
                built[(mesh, dim_a)],
                (
                    *param.placements[:-1],
                    Shard(start_dim),
                    Shard(start_dim + 1),
                ),
            ),
            requires_grad=param.requires_grad,
        )
    if module.split_meshes:
        for i, comp in enumerate(computations):
            if not isinstance(comp, SplitStackReplicateComputation):
                computations[i] = SplitStackReplicateComputation(
                    comp, module.split_meshes, gather_dim=start_dim
                )
    return module.split_meshes


def shard_split_fsdp_modules(model: nn.Module) -> None:
    """Applies 2-D split-mesh sharding to all eligible stacked and unstacked modules."""
    built: dict[tuple[DeviceMesh, int], DeviceMesh] = {}
    for module in model.modules():
        if hasattr(module, "shard_split_stacks"):
            module.shard_split_stacks()
        elif isinstance(module, TpuGroupedExperts):
            shard_module_split_axes(module, start_dim=0, built=built)


def _configure_2d_module_shard_dim(
    module: nn.Module,
    module_fqn: str,
    fsdp_degree: int,
    hidden_axis_modules: list[str],
) -> None:
    """Moves a 2-D weight module (`tok_embeddings`, `lm_head`, or `Linear`) to `Shard(1)` if `dim 0` is uneven."""
    weight = getattr(module, "weight", None)
    if weight is None or isinstance(weight, DTensor):
        return
    if weight.shape[0] % fsdp_degree == 0:
        return
    if weight.ndim != 2 or (
        weight.shape[1] % fsdp_degree != 0 and weight.shape[1] >= fsdp_degree
    ):
        raise ValueError(
            f"Cannot FSDP-shard {module_fqn}.weight with shape "
            f"{tuple(weight.shape)} over {fsdp_degree} ranks"
        )
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


def configure_tpu_fsdp_shard_dims(
    model: nn.Module,
    parallel_dims: Any,
    *,
    include_unstacked_experts: bool = False,
) -> list[str]:
    """Picks SimpleFSDP shard axes that divide evenly, for any decoder model.

    Plans an axis per layer stack, moves ``tok_embeddings`` / ``lm_head`` (and
    any unstacked decoder 2-D weights whose leading dim does not divide
    ``fsdp_degree``) to ``Shard(1)``, and optionally configures unstacked
    ``TpuGroupedExperts`` modules when ``include_unstacked_experts=True``.
    """
    if parallel_dims is None:
        return []
    fsdp_degree = getattr(parallel_dims, "dp_shard", 1) * getattr(
        parallel_dims, "cp", 1
    )
    if fsdp_degree <= 1:
        return []

    stacks = False
    for holder in model.modules():
        if hasattr(holder, "plan_shard_dims"):
            stacks = True
            holder._fsdp_shard_dim = holder.plan_shard_dims(fsdp_degree)

    hidden_axis_modules: list[str] = []
    for module_fqn in ("tok_embeddings", "lm_head"):
        module = getattr(model, module_fqn, None)
        if module is None or not hasattr(module, "weight"):
            continue
        _configure_2d_module_shard_dim(
            module, module_fqn, fsdp_degree, hidden_axis_modules
        )

    # Also check unstacked 2-D decoder layer modules (when use_scan_layers=False)
    if not stacks and hasattr(model, "layers"):
        for module_fqn, module in model.named_modules():
            if module_fqn in ("tok_embeddings", "lm_head"):
                continue
            weight = getattr(module, "weight", None)
            bias = getattr(module, "bias", None)
            if (
                weight is not None
                and not isinstance(weight, DTensor)
                and weight.ndim == 2
                and bias is None
                and len(list(module.parameters(recurse=False))) == 1
                and weight.shape[0] % fsdp_degree != 0
            ):
                _configure_2d_module_shard_dim(
                    module, module_fqn, fsdp_degree, hidden_axis_modules
                )

    if include_unstacked_experts and not getattr(
        parallel_dims, "ep_enabled", False
    ):
        for module_fqn, module in model.named_modules():
            if not isinstance(module, TpuGroupedExperts):
                continue
            if module.num_experts % fsdp_degree == 0:
                continue
            parameters = list(module.named_parameters(recurse=False))
            if not parameters:
                if stacks:
                    continue
                raise ValueError(
                    f"{module_fqn} has no direct parameters to FSDP-shard"
                )
            incompatible = []
            for name, parameter in parameters:
                if (
                    parameter.ndim != 3
                    or parameter.shape[0] != module.num_experts
                ):
                    incompatible.append(f"{name}{tuple(parameter.shape)}")
                    continue
                dim1_divides = parameter.shape[1] % fsdp_degree == 0
                # Also allow 2-D split-mesh sharding when `fsdp_degree` outgrows
                # both `num_experts` and `shape[1]` (or `shape[1]` is a 3*2^k
                # expert dim like 768/1536 at 1024+ ranks) and `(num_experts,
                # fsdp_degree // num_experts)` divides `(shape[0], shape[1])`.
                can_2d_split = (
                    fsdp_degree > module.num_experts
                    and fsdp_degree % module.num_experts == 0
                    and parameter.shape[1] < fsdp_degree
                    and parameter.shape[1]
                    % (fsdp_degree // module.num_experts)
                    == 0
                )
                if not (dim1_divides or can_2d_split):
                    incompatible.append(f"{name}{tuple(parameter.shape)}")
            if incompatible:
                raise ValueError(
                    f"Cannot FSDP-shard {module_fqn} on Qwen's hidden dimension 1 "
                    f"over {fsdp_degree} ranks; incompatible expert parameters: "
                    f"{incompatible}"
                )
            setattr(module, "_fsdp_shard_dim", 1)
            hidden_axis_modules.append(module_fqn)

    return hidden_axis_modules


def configure_simple_fsdp_shard_dims(
    model: nn.Module,
    parallel_dims: Any,
) -> list[str]:
    """Backward-compatible entry point for configuring TPU SimpleFSDP shard dims."""
    return configure_tpu_fsdp_shard_dims(model, parallel_dims)
