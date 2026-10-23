# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for configuring torch_tpu on Cloud."""

import functools
import os
import re

import frozendict
from torchtitan.experiments.tpu import local_tpu_device
from torchtitan.tools.logging import logger


# Trailing "-<job-index>-<pod-index>" of a JobSet pod name; see
# get_slice_id_from_worker_hostnames.
_POD_INDEX_RE = re.compile(r"-(\d+)-(\d+)$")

_TPU_CHIP_TO_HOST_BOUNDS = frozendict.frozendict({
    (local_tpu_device.TpuChip.V7X, 8): "2,2,1,2",
    (local_tpu_device.TpuChip.V7X, 4): "2,2,1,1",
    (local_tpu_device.TpuChip.V7X, 1): "1,1,1,1",
    (local_tpu_device.TpuChip.V6E, 8): "2,4,1",
    (local_tpu_device.TpuChip.V6E, 4): "2,2,1",
    (local_tpu_device.TpuChip.V6E, 1): "1,1,1",
    (local_tpu_device.TpuChip.V5P, 8): "2,4,1",
    (local_tpu_device.TpuChip.V5P, 4): "2,2,1",
    (local_tpu_device.TpuChip.V5P, 1): "1,1,1",
    (local_tpu_device.TpuChip.V5E, 8): "2,4,1",
    (local_tpu_device.TpuChip.V5E, 4): "2,2,1",
    (local_tpu_device.TpuChip.V5E, 1): "1,1,1",
})


def get_num_slices(environ: dict[str, str] | None = None) -> int:
  """Returns the number of TPU slices this job spans.

  Multi-slice jobs are identified by ``MEGASCALE_NUM_SLICES``, which the
  launcher sets (and which GKE propagates to every pod of a multi-slice
  JobSet). A single-slice job either omits it or sets it to 1.

  Args:
    environ: Environment to read. Defaults to ``os.environ``.

  Returns:
    The slice count, at least 1.

  Raises:
    ValueError: If ``MEGASCALE_NUM_SLICES`` is set but is not a positive
      integer.
  """
  if environ is None:
    environ = os.environ
  raw = environ.get("MEGASCALE_NUM_SLICES", "1")
  try:
    num_slices = int(raw)
  except ValueError as e:
    raise ValueError(
        f"MEGASCALE_NUM_SLICES must be a positive integer, got {raw!r}."
    ) from e
  if num_slices < 1:
    raise ValueError(
        f"MEGASCALE_NUM_SLICES must be a positive integer, got {raw!r}."
    )
  return num_slices


def get_nodes_per_slice(environ: dict[str, str] | None = None) -> int | None:
  """Returns the number of hosts in one slice, or None if GKE has not said.

  ``TPU_WORKER_HOSTNAMES`` is the slice's own host list, not the job's: the
  long-standing check in ``_maybe_init_distributed_on_gke`` rejects an
  environment where its length disagrees with the product of
  ``TPU_HOST_BOUNDS``, and those bounds describe one slice. So the length is
  the host count for this slice, independent of chips per host and of the
  topology string.

  **Assumes one pod per host**, which is what makes this the node count
  torchrun wants. That holds for Cluster Toolkit's TPU JobSet and for XPK
  without ``--use-parallel-containers``; a launcher that packs several
  containers onto a host has more nodes than hosts, and this value would be
  too small. ``TPU_WORKER_ID`` indexes the same thing -- the single-slice
  launch command uses it directly as ``--node_rank`` -- so the two agree or
  disagree together.

  Empty entries are ignored so a trailing comma does not inflate the count.
  """
  if environ is None:
    environ = os.environ
  hostnames = environ.get("TPU_WORKER_HOSTNAMES")
  if not hostnames:
    return None
  hosts = [h for h in hostnames.split(",") if h.strip()]
  return len(hosts) or None


def get_slice_id_from_worker_hostnames(
    environ: dict[str, str] | None = None,
) -> int | None:
  """Returns this slice's index, read out of ``TPU_WORKER_HOSTNAMES``, or None.

  GKE injects the slice's own pod DNS names, and a JobSet names a pod
  ``<jobset>-<replicated-job>-<job-index>-<pod-index>``. One replicated job is
  rendered per slice, so the job index *is* the slice index -- the same value
  the ``jobset.sigs.k8s.io/job-index`` label carries. Nothing else in the
  environment has it: GKE exposes no ``JOBSET_*`` variable, Kubernetes' own
  ``JOB_COMPLETION_INDEX`` is the pod's index *within* its job (the worker
  index, identical on every slice), and ``hostNetwork: true`` makes the
  hostname the node's name. Reading it here is what keeps the slice index out
  of the launch command.

  The two trailing numbers are parsed off every entry and then checked against
  what a per-slice host list has to look like: one job index shared by all of
  them, and pod indices that are exactly ``0..len-1``. A launcher whose pods
  are named some other way fails that check and gets None rather than a
  plausible wrong answer, which is the failure that matters -- every slice
  silently deciding it is slice 0 rendezvouses into a hang.

  Args:
    environ: Environment to read. Defaults to ``os.environ``.

  Returns:
    The slice index, or None if the host list is absent or not JobSet-shaped.
  """
  if environ is None:
    environ = os.environ
  hostnames = environ.get("TPU_WORKER_HOSTNAMES")
  if not hostnames:
    return None
  hosts = [h.strip() for h in hostnames.split(",") if h.strip()]
  if not hosts:
    return None

  parsed = []
  for host in hosts:
    match = _POD_INDEX_RE.search(host.split(".", 1)[0])
    if match is None:
      return None
    parsed.append((int(match.group(1)), int(match.group(2))))

  job_indices = {job_index for job_index, _ in parsed}
  pod_indices = sorted(pod_index for _, pod_index in parsed)
  if len(job_indices) != 1 or pod_indices != list(range(len(hosts))):
    return None

  slice_id = job_indices.pop()
  return slice_id if slice_id < get_num_slices(environ) else None


def get_global_node_rank(environ: dict[str, str] | None = None) -> int | None:
  """Returns this worker's node rank across the whole job, or None.

  ``slice_id * nodes_per_slice + within_slice_worker_id``. Global ranks have to
  be handed out in slice order, because that ordering is what lets the slice
  index be recovered from a rank later; ``get_slice_id`` cross-checks the two
  and raises on a mismatch.

  This runs *before* torchrun assigns ranks, so the slice index cannot be
  derived from one. It comes instead from ``TPU_WORKER_HOSTNAMES``, which GKE
  injects per slice -- see ``get_slice_id_from_worker_hostnames`` -- or from an
  explicit ``MEGASCALE_SLICE_ID``. The pod hostname is not an alternative: GKE
  TPU workloads run with ``hostNetwork: true``, so it is the node's name.

  Returns None when the environment does not carry what is needed, so a caller
  can fail rather than launch every slice at node rank 0.
  """
  if environ is None:
    environ = os.environ
  nodes_per_slice = get_nodes_per_slice(environ)
  worker_id = environ.get("TPU_WORKER_ID")
  if nodes_per_slice is None or worker_id is None:
    return None
  if (
      get_num_slices(environ) > 1
      and "MEGASCALE_SLICE_ID" not in environ
      and get_slice_id_from_worker_hostnames(environ) is None
  ):
    return None  # no usable slice index; see the docstring
  return get_slice_id(environ) * nodes_per_slice + int(worker_id)


def get_slice_id(environ: dict[str, str] | None = None) -> int:
  """Returns this process's slice index, 0 on a single-slice job.

  Derived from the rank when it is not set explicitly. Global ranks are handed
  out in slice order -- a worker's node rank is
  ``slice_id * vms_per_slice + job_completion_index`` -- so the first
  ``WORLD_SIZE / num_slices`` ranks belong to slice 0, the next to slice 1, and
  so on. That makes ``MEGASCALE_SLICE_ID`` optional in the launcher: the
  slice count and a rendezvous address are enough.

  When the variable *is* set, it is cross-checked against the rank and a
  mismatch raises. That catches a launcher computing slice ids one way and
  node ranks another, which otherwise shows up as a silent rendezvous hang.

  Args:
    environ: Environment to read. Defaults to ``os.environ``.

  Returns:
    The slice index.

  Raises:
    ValueError: If an explicit ``MEGASCALE_SLICE_ID`` disagrees with the rank.
  """
  if environ is None:
    environ = os.environ

  num_slices = get_num_slices(environ)
  derived = None
  if num_slices > 1 and "RANK" in environ and "WORLD_SIZE" in environ:
    world_size = int(environ["WORLD_SIZE"])
    if world_size % num_slices == 0:
      derived = int(environ["RANK"]) // (world_size // num_slices)

  from_hostnames = (
      get_slice_id_from_worker_hostnames(environ) if num_slices > 1 else None
  )

  if "MEGASCALE_SLICE_ID" not in environ:
    if from_hostnames is not None:
      return from_hostnames
    return derived if derived is not None else 0

  declared = int(environ["MEGASCALE_SLICE_ID"])
  if from_hostnames is not None and declared != from_hostnames:
    raise ValueError(
        f"MEGASCALE_SLICE_ID={declared} disagrees with the slice index in"
        f" TPU_WORKER_HOSTNAMES={environ['TPU_WORKER_HOSTNAMES']!r}, which is"
        f" {from_hostnames}. GKE injects that host list per slice, so it is the"
        " authority on which slice this pod belongs to; a disagreement means"
        " the variable is wired to the wrong label or carries a stale value."
    )
  if derived is not None and declared != derived:
    raise ValueError(
        f"MEGASCALE_SLICE_ID={declared} disagrees with the slice implied by"
        f" RANK={environ['RANK']} of WORLD_SIZE={environ['WORLD_SIZE']} across"
        f" {num_slices} slices, which is {derived}. Global ranks must be handed"
        " out in slice order: node_rank = slice_id * vms_per_slice +"
        " job_completion_index."
    )
  return declared


def is_multi_slice(environ: dict[str, str] | None = None) -> bool:
  """Whether this job spans more than one TPU slice."""
  return get_num_slices(environ) > 1


def _maybe_init_distributed_on_gke(
    slicebuilder_first_worker_port: int = 10000,
    environ: dict[str, str] | None = None,
) -> bool:
  """Initializes environment variables for distributed TPU training on GKE.

  This function configures environment variables required by SliceBuilder
  when running torch_tpu on GKE, based on the provided environment
  variables set by the Kubernetes setup. It adjusts bounds and addresses
  to match the expected configuration for distributed TPU workers.

  Args:
    slicebuilder_first_worker_port: The starting port number for SliceBuilder
      worker communication.
    environ: A dictionary of environment variables. Defaults to `os.environ`.

  Returns:
    True if the environment variables were successfully initialized for
    distributed TPU training on GKE, False otherwise.
  """
  if environ is None:
    environ = os.environ

  if (
      "TPU_WORKER_HOSTNAMES" not in environ
      or "TPU_CHIPS_PER_HOST_BOUNDS" not in environ
      or "TPU_HOST_BOUNDS" not in environ
      or "WORLD_SIZE" not in environ
      or "LOCAL_RANK" not in environ
      or "RANK" not in environ
  ):
    return False

  logger.info("Trying to initialize distributed TPU training on GKE.")
  world_size = int(environ["WORLD_SIZE"])
  local_rank = int(environ["LOCAL_RANK"])
  rank = int(environ["RANK"])

  # On a multi-slice job WORLD_SIZE spans every slice, while the TPU_*_BOUNDS
  # that GKE injects describe ONE slice. SliceBuilder is also per-slice: each
  # slice builds its own ICI mesh and MegaScale joins them over DCN. So every
  # comparison against the bounds below, and the task id handed to the slice's
  # PJRT client, must use the PER-SLICE world size and the WITHIN-SLICE rank.
  # Using the global values makes the bounds check fail and this function bail
  # out, leaving TORCH_TPU_SLICEBUILDER_ADDRESSES unset.
  num_slices = get_num_slices(environ)
  if world_size % num_slices:
    raise ValueError(
        f"WORLD_SIZE={world_size} is not divisible by "
        f"MEGASCALE_NUM_SLICES={num_slices}; every slice must contribute the "
        "same number of processes."
    )
  slice_world_size = world_size // num_slices
  slice_rank = rank % slice_world_size
  is_v7 = ("TPU_ACCELERATOR_TYPE" in environ and
           environ["TPU_ACCELERATOR_TYPE"].startswith("tpu7x"))

  tpu_chips_per_host_bounds = list(
      map(int, environ["TPU_CHIPS_PER_HOST_BOUNDS"].split(","))
  )
  tpu_chips_per_host_bounds_product = functools.reduce(
      lambda x, y: x * y, tpu_chips_per_host_bounds
  )

  if tpu_chips_per_host_bounds_product == 1:
    return False  # either single host or environment is set manually.

  tpu_host_bounds = list(map(int, environ["TPU_HOST_BOUNDS"].split(",")))
  tpu_host_bounds_product = functools.reduce(
      lambda x, y: x * y, tpu_host_bounds
  )

  tpu_worker_hostnames = environ["TPU_WORKER_HOSTNAMES"].split(",")

  if tpu_host_bounds_product != len(tpu_worker_hostnames):
    return False  # environment is likely to set incorrectly.

  if is_v7:
    tpu_chips_per_host_bounds_product *= 2

  if (
      tpu_chips_per_host_bounds_product * tpu_host_bounds_product
      != slice_world_size
  ):
    return False  # environment is likely to set incorrectly.

  environ["TPU_CHIPS_PER_HOST_BOUNDS"] = ",".join(
      ["1"] * (4 if is_v7 else 3)
  )
  for i in range(len(tpu_host_bounds)):
    tpu_host_bounds[i] = tpu_host_bounds[i] * tpu_chips_per_host_bounds[i]
  environ["TPU_HOST_BOUNDS"] = ",".join(map(str, tpu_host_bounds))
  if is_v7:
    environ["TPU_HOST_BOUNDS"] += ",2"

  environ["TPU_VISIBLE_CHIPS"] = str(local_rank)
  # Within-slice: the slice's PJRT client numbers its own tasks from zero.
  environ["CLOUD_TPU_TASK_ID"] = str(slice_rank)
  ports = list(
      range(
          slicebuilder_first_worker_port,
          slicebuilder_first_worker_port + tpu_chips_per_host_bounds_product,
      )
  )
  environ["TPU_PROCESS_PORT"] = str(ports[local_rank])
  tpu_worker_addresses = []
  for tpu_worker_hostname in tpu_worker_hostnames:
    for port in ports:
      tpu_worker_addresses.append(tpu_worker_hostname + ":" + str(port))
  tpu_worker_addresses_str = ",".join(tpu_worker_addresses)
  environ["TPU_PROCESS_ADDRESSES"] = tpu_worker_addresses_str
  environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = tpu_worker_addresses_str
  environ["TORCH_TPU_TOPOLOGY"] = environ["TPU_HOST_BOUNDS"]

  if num_slices > 1:
    # The MegaScale PJRT client is what gives devices a slice index and a
    # globally unique id; without it every slice numbers its devices from zero
    # and torch_tpu's topology_aware_mesh rejects the job with "duplicate pjrt
    # device ids across torch ranks".
    # MEGASCALE_SLICE_ID is optional in the launch command -- get_slice_id()
    # recovers it from the per-slice host list -- but libtpu reads it straight
    # from the environment, so it has to be exported before the first device is
    # used. The coordinator address is what makes libtpu create the MegaScale
    # PJRT client at all, and that one cannot be inferred.
    if "MEGASCALE_SLICE_ID" not in environ:
      slice_id = get_slice_id(environ)
      environ["MEGASCALE_SLICE_ID"] = str(slice_id)
      logger.info(
          "Exporting MEGASCALE_SLICE_ID=%d, recovered from"
          " TPU_WORKER_HOSTNAMES; libtpu reads this variable when it creates"
          " the MegaScale PJRT client.",
          slice_id,
      )
    missing = [
        v for v in ("MEGASCALE_COORDINATOR_ADDRESS",) if v not in environ
    ]
    if missing:
      raise ValueError(
          f"MEGASCALE_NUM_SLICES={num_slices} requires {missing} to be set so "
          "libtpu creates the MegaScale PJRT client. Without it each slice "
          "numbers its devices from zero and the mesh cannot be built."
      )
    logger.info(
        "Multi-slice TPU: %d slices, this is slice %d. Per-slice world size"
        " %d (global %d), within-slice task id %d, per-slice topology %s.",
        num_slices,
        get_slice_id(environ),
        slice_world_size,
        world_size,
        slice_rank,
        environ["TORCH_TPU_TOPOLOGY"],
    )
  return True


def _maybe_init_distributed_on_tpu_vm() -> bool:
  """Initializes environment variables for distributed TPU training on a TPU VM.

  This function configures environment variables required for distributed
  training when running on a single TPU VM. It sets up bounds, visible chips,
  and worker addresses based on the local TPU configuration and world size.

  Returns:
    True if the environment variables were successfully initialized for
    distributed TPU training on a TPU VM, False otherwise.

  Raises:
    ValueError: If the number of local chips does not match the WORLD_SIZE.
  """
  world_size = int(os.getenv("WORLD_SIZE", "1"))
  if world_size == 1:
    return False

  logger.info("Trying to initialize distributed TPU training on TPU VM.")
  local_chips = local_tpu_device.get_local_chips()
  if local_chips is None:
    return False
  tpu_chip, num_chips = local_chips
  if tpu_chip is None:
    return False
  if num_chips is None:
    return False

  if num_chips != world_size:
    raise ValueError(
        f"Assuming single-chip setup, got {num_chips} chips"
        f" and world size {world_size}."
    )

  local_rank = int(os.environ["LOCAL_RANK"])
  os.environ["CLOUD_TPU_TASK_ID"] = str(local_rank)
  os.environ["TPU_VISIBLE_CHIPS"] = str(local_rank)
  if tpu_chip == local_tpu_device.TpuChip.V7X:
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1,1"
  else:
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
  os.environ["TPU_HOST_BOUNDS"] = _TPU_CHIP_TO_HOST_BOUNDS[tpu_chip, num_chips]

  # only works for singlehost !!!
  worker0 = os.getenv("WORKERS_0_HOSTNAME", "localhost")
  ports = list(range(10000, 10000 + world_size))
  tpu_worker_addresses = []
  for port in ports:
    tpu_worker_addresses.append(worker0 + ":" + str(port))
  tpu_worker_addresses_str = ",".join(tpu_worker_addresses)
  os.environ["TPU_PROCESS_ADDRESSES"] = tpu_worker_addresses_str
  os.environ["TPU_PROCESS_PORT"] = str(10000 + local_rank)
  os.environ["TPU_ACCELERATOR_TYPE"] = f"{tpu_chip.name}-{num_chips}"
  os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = tpu_worker_addresses_str
  os.environ["TORCH_TPU_TOPOLOGY"] = os.environ["TPU_HOST_BOUNDS"]
  return True


def maybe_validate_multislice_parallelism(parallelism_config) -> None:
  """Rejects the one multi-slice layout torch_tpu cannot detect.

  ``torch.tpu.topology_aware_mesh`` puts the slice split on the outermost mesh
  axis whose extent is above 1. torchtitan passes it a bare shape ordered
  ``(pp, dp_replicate, dp_shard, cp, tp)``, so it has to skip extent-1 axes --
  ``pp`` is 1 on every non-pipelined run -- and without axis names it cannot
  tell an unused ``pp`` from an unset ``dp_replicate``. With
  ``pipeline_parallel_degree`` and ``data_parallel_replicate_degree`` both at
  their default of 1, the split therefore lands on ``dp_shard``: FSDP shard
  groups straddle the inter-slice link and every weight all-gather and gradient
  reduce-scatter crosses DCN. torch_tpu only logs that at INFO; the job trains
  correctly at a fraction of the throughput.

  Every other case is left to torch_tpu, which raises when the axis taking the
  split is not divisible by the slice count (a replicate degree of 3 over 2
  slices, say) and handles pipeline stages placed across slices.

  Checks rather than fixes. Quietly setting the degree would mean the config a
  recipe declares is not the config that ran, and a job that is merely slow is
  far more expensive to notice than one that refuses to start. The check runs
  before ``ParallelDims`` is built, so it costs seconds.

  "maybe" because it returns immediately on a single-slice job, so it is safe to
  call unconditionally.

  Args:
    parallelism_config: The run's ``ParallelismConfig``. Not modified.

  Raises:
    ValueError: On a multi-slice job with neither pipeline nor replicate
      parallelism, where the slice split would fall on ``dp_shard``.
  """
  num_slices = get_num_slices()
  if num_slices == 1:
    return

  pipeline = parallelism_config.pipeline_parallel_degree
  replicate = parallelism_config.data_parallel_replicate_degree
  if pipeline == 1 and replicate == 1:
    raise ValueError(
        f"A {num_slices}-slice job with data_parallel_replicate_degree=1 and"
        " pipeline_parallel_degree=1 has no axis for the slice split but"
        " dp_shard, so torch_tpu would place FSDP shard groups across the"
        " inter-slice link and every weight all-gather would cross DCN. Pass"
        " --parallelism.data_parallel_replicate_degree=$MEGASCALE_NUM_SLICES"
        " so the replicate axis takes the split."
    )
  logger.info(
      "Multi-slice: %d slices, pipeline_parallel_degree=%d,"
      " data_parallel_replicate_degree=%d; topology_aware_mesh splits the"
      " outermost of these above 1 across slices.",
      num_slices,
      pipeline,
      replicate,
  )


def maybe_init_distributed() -> bool:
  """Initializes environment variables for distributed TPU training."""
  return _maybe_init_distributed_on_gke() or _maybe_init_distributed_on_tpu_vm()
