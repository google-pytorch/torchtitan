"""TPU-local SimpleFSDP support for native non-leading collectives."""

from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard

from torchtitan.experiments.graph_trainer.simple_fsdp import ReplicateComputation
from torchtitan.tools.logging import logger


def _all_gather_on_dim(shard: torch.Tensor, dim: int, group_name: str) -> torch.Tensor:
  """Gather ``dim``, requiring the native dimensional operator on TPU."""
  if dim == 0:
    return funcol.all_gather_single(shard, gather_dim=0, group=group_name)
  if shard.device.type == "tpu":
    if not hasattr(torch.ops.tpu, "_all_gather_dim"):
      raise RuntimeError(
          "Non-leading TPU FSDP shards require TorchTPU's "
          "_all_gather_dim operator"
      )
    return torch.ops.tpu._all_gather_dim.default(
        shard,
        dim,
        c10d._get_group_size_by_name(group_name),
        group_name,
    )
  return funcol.all_gather_single(shard, gather_dim=dim, group=group_name)


def _reduce_scatter_on_dim(
    grad: torch.Tensor, dim: int, group_name: str
) -> torch.Tensor:
  """Reduce-scatter ``dim``, requiring its native TPU operator off dim 0."""
  if dim == 0:
    return funcol.reduce_scatter_single(
        grad, reduceOp="sum", scatter_dim=0, group=group_name
    )
  if grad.device.type == "tpu":
    if not hasattr(torch.ops.tpu, "_reduce_scatter_dim"):
      raise RuntimeError(
          "Non-leading TPU FSDP shards require TorchTPU's "
          "_reduce_scatter_dim operator"
      )
    return torch.ops.tpu._reduce_scatter_dim.default(
        grad,
        "sum",
        dim,
        c10d._get_group_size_by_name(group_name),
        group_name,
    )
  return funcol.reduce_scatter_single(
      grad,
      reduceOp="sum",
      scatter_dim=dim,
      group=group_name,
  )


class _ShardToReplicate(torch.autograd.Function):

  @staticmethod
  def forward(
      ctx: Any,
      shard: torch.Tensor,
      group_name: str,
      param_dtype: torch.dtype | None,
      reduce_dtype: torch.dtype | None,
      replicate_group_name: str | None,
      shard_dim: int,
  ) -> torch.Tensor:
    ctx.group_name = group_name
    ctx.replicate_group_name = replicate_group_name
    ctx.reduce_dtype = reduce_dtype
    ctx.shard_dtype = shard.dtype
    ctx.shard_dim = shard_dim
    if param_dtype is not None:
      shard = shard.to(param_dtype)
    return _all_gather_on_dim(shard, shard_dim, group_name)

  @staticmethod
  def backward(ctx: Any, grad: torch.Tensor):
    if ctx.reduce_dtype is not None:
      grad = grad.to(ctx.reduce_dtype)
    out = _reduce_scatter_on_dim(grad, ctx.shard_dim, ctx.group_name)
    if ctx.replicate_group_name is not None:
      out = funcol.all_reduce(out, "sum", group=ctx.replicate_group_name)
    return out.to(ctx.shard_dtype), None, None, None, None, None


def replicate_sharded_tensor(
    shard: torch.Tensor,
    *,
    group_name: str,
    shard_dim: int,
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    replicate_group_name: str | None = None,
) -> torch.Tensor:
  """Return a compute replica whose backward returns the local FSDP shard."""
  return _ShardToReplicate.apply(
      shard,
      group_name,
      param_dtype,
      reduce_dtype,
      replicate_group_name,
      shard_dim,
  )


class TpuReplicateComputation(ReplicateComputation):
  """Use native TPU collectives for even, non-leading SimpleFSDP shards."""

  def replicate_compute(self, x: DTensor) -> torch.Tensor:
    non_dp_mesh_dims = x._spec.mesh.ndim - self.device_mesh.ndim
    if non_dp_mesh_dims == 0:
      placements = x._spec.placements
      shard_mesh_dim: int | None = None
      replicate_group_name: str | None = None
      if len(placements) == 1 and isinstance(placements[0], Shard):
        shard_mesh_dim = 0
      elif (
          len(placements) == 2
          and isinstance(placements[0], Replicate)
          and isinstance(placements[1], Shard)
      ):
        replicate_group_name = x.device_mesh.get_group(0).group_name
        shard_mesh_dim = 1

      if shard_mesh_dim is not None:
        shard_placement = placements[shard_mesh_dim]
        assert isinstance(shard_placement, Shard)
        shard_dim = shard_placement.dim
        shard_degree = x.device_mesh.size(shard_mesh_dim)
        if shard_dim != 0 and x.shape[shard_dim] % shard_degree == 0:
          return replicate_sharded_tensor(
              x.to_local(),
              group_name=x.device_mesh.get_group(shard_mesh_dim).group_name,
              shard_dim=shard_dim,
              param_dtype=self.param_dtype,
              reduce_dtype=self.reduce_dtype,
              replicate_group_name=replicate_group_name,
          )
    return super().replicate_compute(x)


def _has_nonleading_shard(param: torch.Tensor) -> bool:
  return isinstance(param, DTensor) and any(
      isinstance(placement, Shard) and placement.dim != 0
      for placement in param.placements
  )


def install_tpu_replicate_computation(model: nn.Module) -> None:
  """Replace SimpleFSDP's parametrization only on the TPU Qwen model.

  Relies on this fork's SimpleFSDP keeping its per-parameter computations in
  ``_simple_fsdp_parametrization_list`` (google-pytorch/torchtitan 7e8acc32b).
  The computations are re-classed in place, so their constructor signature does
  not matter. Raises if a non-leading shard has no list to patch, so an upstream
  change cannot silently fall back to the copying path.
  """
  if any(hasattr(module, "plan_shard_dims") for module in model.modules()):
    # Scan flattens its stacks onto dim 0 so it scales past 512 ranks; keep
    # its collectives on the upstream path.
    logger.info("Layer stacks present; keeping SimpleFSDP's default replication")
    return
  installed = 0
  for module in model.modules():
    # SimpleFSDP's generated parameter properties resolve this private
    # per-parameter list on every access, so patching its elements after
    # wrapping changes the gather implementation without modifying the shared
    # SimpleFSDP module.
    parametrizations = getattr(module, "_simple_fsdp_parametrization_list", None)
    if parametrizations is None:
      unpatched = [
          name
          for name, param in module.named_parameters(recurse=False)
          if _has_nonleading_shard(param)
      ]
      if unpatched:
        raise RuntimeError(
            f"{type(module).__name__} has non-leading FSDP shards {unpatched} "
            "but no _simple_fsdp_parametrization_list to patch; SimpleFSDP's "
            "internals changed, so the TPU native collectives cannot be installed"
        )
      continue
    for computation in parametrizations:
      if isinstance(computation, TpuReplicateComputation):
        continue
      if type(computation) is not ReplicateComputation:
        raise RuntimeError(
            "Unexpected SimpleFSDP parametrization type: "
            f"{type(computation).__name__}"
        )
      computation.__class__ = TpuReplicateComputation
      installed += 1
  logger.info(
      "Installed TPU SimpleFSDP replication on %d parameter(s)", installed
  )


__all__ = [
    "TpuReplicateComputation",
    "install_tpu_replicate_computation",
    "replicate_sharded_tensor",
]
