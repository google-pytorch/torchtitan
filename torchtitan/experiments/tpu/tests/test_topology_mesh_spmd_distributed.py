"""spmd_types mesh axes on the TPU topology-aware device mesh.

``enable_tpu_topo_mesh`` builds the world mesh in the order that
``torch.tpu.topology_aware_mesh`` returns, which is not rank order: on a v6e-8
it is 1,3,2,0,6,5,7,4. spmd_types builds each mesh axis from a process group's
sorted global ranks, so on such a mesh the axes of a multi-axis mesh, such as
dp_replicate and dp_shard or efsdp and ep, come out non-orthogonal, or not a
grid at all, and entering that mesh raises on some ranks (b/587033846).

The test covers HSDP, TP and CP, and EP alone and combined with HSDP and with
TP, at EP=2 and EP=4. For each it enters the dense mesh the way forward and
backward do, and it resolves the process groups of the dense storage mesh the
way FSDP2 does. With EP it also enters the EP mesh the way expert dispatch
does, and it resolves the process groups of the expert storage mesh the way
FSDP2 does for expert weights. TPU only: other devices have no topology-aware
mesh. Each rank records its outcomes in a file and the test checks them all,
because raising on only some ranks would leave the others waiting.
"""

import json
import os

from absl import logging
from absl.testing import absltest
# pyformat: disable
# With torch_tpu, torch (here torch.multiprocessing) must be imported before
# spmd_types to avoid a circular import when torch autoloads torch_tpu ->
# torch.distributed.fsdp -> spmd_types.
import torch.multiprocessing as mp  # isort: skip
import spmd_types as spmd
# pyformat: enable
from torchtitan.distributed import fsdp
from torchtitan.distributed import parallel_dims as parallel_dims_lib
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.distributed import distributed
from torchtitan.experiments.tpu.testing import base_distributed_device_test


_NUM_RANKS = 8

# (name, ParallelDims degrees); dp_shard fills the rest of the 8 ranks.
_CASES = (
    ("hsdp", dict(dp_replicate=2)),
    ("tp2", dict(tp=2)),
    ("cp2", dict(cp=2)),
    ("hsdp_tp2", dict(dp_replicate=2, tp=2)),
    ("ep2", dict(ep=2)),
    ("ep4", dict(ep=4)),
    ("hsdp_ep2", dict(dp_replicate=2, ep=2)),
    ("hsdp_ep4", dict(dp_replicate=2, ep=4)),
    ("tp2_ep2", dict(tp=2, ep=2)),
    ("tp2_ep4", dict(tp=2, ep=4)),
)


def _enter_mesh(mesh):
  with spmd.set_current_mesh(mesh):
    pass


def _resolve_groups(mesh):
  # FSDP2 resolves the axes of its storage mesh from the process groups
  # directly (MeshAxis.of(group)), not through the DeviceMesh.
  with spmd.set_current_mesh([mesh.get_group(dim) for dim in range(mesh.ndim)]):
    pass


def _run_check(check, mesh):
  """Runs ``check`` on ``mesh``; returns its error and this rank's groups."""
  try:
    check(mesh)
    error = None
  except Exception as e:  # pylint: disable=broad-exception-caught
    error = f"{type(e).__name__}: {e}"
  return {
      "error": error,
      "groups": {
          name: mesh[name].mesh.flatten().tolist()
          for name in mesh.mesh_dim_names
      },
  }


def _check_case(world_size, degrees):
  """Builds the meshes for one case and checks both ways of reading axes."""
  dims = dict(dp_replicate=1, dp_shard=-1, cp=1, tp=1, pp=1, ep=1)
  dims.update(degrees)
  parallel_dims = parallel_dims_lib.ParallelDims(
      world_size=world_size, enable_tpu_topo_mesh=True, **dims
  )
  parallel_dims.build_mesh()
  # Forward/backward enters the dense mesh, and fully_shard() gets the dense
  # storage mesh for non-expert weights.
  dense_storage_mesh, _ = fsdp.resolve_fsdp_mesh(parallel_dims)
  checks = {
      "dense_mesh": _run_check(_enter_mesh, parallel_dims.spmd_dense_mesh()),
      "dense_groups": _run_check(_resolve_groups, dense_storage_mesh),
  }
  if parallel_dims.ep_enabled:
    # Expert dispatch enters the EP mesh, and fully_shard() gets the expert
    # storage mesh for expert weights.
    expert_storage_mesh, _ = fsdp.resolve_sparse_fsdp_mesh(parallel_dims)
    checks["sparse_mesh"] = _run_check(
        _enter_mesh, parallel_dims.spmd_sparse_mesh()
    )
    checks["sparse_groups"] = _run_check(_resolve_groups, expert_storage_mesh)
  return {
      "world_order": parallel_dims.world_mesh.mesh.flatten().tolist(),
      "checks": checks,
  }


def _check_cases(device, rank, world_size, out_dir):
  """Runs every case on this rank and writes the outcomes to a file."""
  del device
  outcomes = {
      name: _check_case(world_size, degrees) for name, degrees in _CASES
  }
  with open(os.path.join(out_dir, f"rank_{rank}.json"), "w") as f:
    json.dump(outcomes, f)


class TopologyMeshSpmdTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """spmd_types mesh axes on the TPU topology-aware mesh."""

  def test_mesh_axes_are_orthogonal(self):
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("Needs the TPU topology-aware mesh.")
    if self.num_devices != _NUM_RANKS:
      self.skipTest(f"Needs {_NUM_RANKS} chips, got {self.num_devices}.")
    out_dir = self.create_tempdir().full_path
    distributed.run_distributed(
        self.num_devices,
        self.accelerator_device_type,
        _check_cases,
        out_dir,
    )
    outcomes = []
    for rank in range(self.num_devices):
      with open(os.path.join(out_dir, f"rank_{rank}.json")) as f:
        outcomes.append(json.load(f))
    world_order = outcomes[0][_CASES[0][0]]["world_order"]
    logging.info("World mesh order: %s", world_order)
    if world_order == sorted(world_order):
      self.skipTest("The topology-aware mesh is in rank order on this slice.")
    for name, _ in _CASES:
      for check in outcomes[0][name]["checks"]:
        with self.subTest(case=name, check=check):
          results = [outcome[name]["checks"][check] for outcome in outcomes]
          failures = {
              rank: f"{result['error']} (groups {result['groups']})"
              for rank, result in enumerate(results)
              if result["error"]
          }
          self.assertEmpty(
              failures,
              f"{name}: {check} with world order {world_order} on"
              f" {len(failures)} of {len(outcomes)} ranks: {failures}",
          )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
