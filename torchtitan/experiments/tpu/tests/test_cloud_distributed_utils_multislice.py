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

"""Multi-slice environment setup in cloud_distributed_utils.

These run on CPU: the functions under test only rewrite an environment dict.
"""

import unittest
from unittest import mock

from torchtitan.config.configs import ParallelismConfig
from torchtitan.experiments.tpu import cloud_distributed_utils as cdu


def _gke_env(
    *, num_slices: int = 2, slice_id: int = 0, vms: int = 2, world: int | None = None
) -> dict[str, str]:
  """A v7x multi-slice pod environment as GKE injects it.

  The TPU_*_BOUNDS describe ONE slice while WORLD_SIZE spans every slice; that
  asymmetry is what the code under test has to cope with.
  """
  procs_per_vm = 8  # v7x: 4 chips x 2 TensorCores
  return {
      "TPU_WORKER_HOSTNAMES": ",".join(f"h{slice_id}-{v}" for v in range(vms)),
      "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1,1",
      "TPU_HOST_BOUNDS": "1,1,2",
      "TPU_ACCELERATOR_TYPE": "tpu7x-8",
      "WORLD_SIZE": str(
          world if world is not None else num_slices * vms * procs_per_vm
      ),
      "RANK": str(slice_id * vms * procs_per_vm + 3),
      "LOCAL_RANK": "3",
      "MEGASCALE_NUM_SLICES": str(num_slices),
      "MEGASCALE_SLICE_ID": str(slice_id),
      "MEGASCALE_COORDINATOR_ADDRESS": "h0-0:8080",
  }


class SliceCountTest(unittest.TestCase):

  def test_defaults_to_one_slice(self):
    self.assertEqual(cdu.get_num_slices({}), 1)
    self.assertFalse(cdu.is_multi_slice({}))

  def test_reads_megascale_num_slices(self):
    self.assertEqual(cdu.get_num_slices({"MEGASCALE_NUM_SLICES": "4"}), 4)
    self.assertTrue(cdu.is_multi_slice({"MEGASCALE_NUM_SLICES": "2"}))

  def test_slice_id_defaults_to_zero(self):
    self.assertEqual(cdu.get_slice_id({}), 0)

  def test_pod_hostname_is_not_used(self):
    """GKE TPU pods run hostNetwork, so the hostname is the NODE's name.

    Deriving a slice index from it would silently give 0 for every slice.
    """
    self.assertEqual(
        cdu.get_slice_id({
            "MEGASCALE_NUM_SLICES": "2",
            "HOSTNAME": "gke-some-pool-9b74ef28-wx3l",
        }),
        0,
    )
    self.assertFalse(hasattr(cdu, "_slice_id_from_hostname"))

  def test_slice_id_is_derived_from_the_rank(self):
    """Ranks are handed out in slice order, so MEGASCALE_SLICE_ID is optional."""
    base = {"MEGASCALE_NUM_SLICES": "2", "WORLD_SIZE": "128"}
    for rank, expected in ((0, 0), (63, 0), (64, 1), (127, 1)):
      with self.subTest(rank=rank):
        self.assertEqual(cdu.get_slice_id({**base, "RANK": str(rank)}), expected)

  def test_slice_id_derivation_handles_four_slices(self):
    self.assertEqual(
        cdu.get_slice_id(
            {"MEGASCALE_NUM_SLICES": "4", "WORLD_SIZE": "128", "RANK": "90"}
        ),
        2,
    )

  def test_explicit_slice_id_is_cross_checked_against_the_rank(self):
    """A launcher computing slice ids and node ranks differently would hang."""
    env = {"MEGASCALE_NUM_SLICES": "2", "WORLD_SIZE": "128", "RANK": "64"}
    self.assertEqual(cdu.get_slice_id({**env, "MEGASCALE_SLICE_ID": "1"}), 1)
    with self.assertRaisesRegex(ValueError, "disagrees with the slice implied"):
      cdu.get_slice_id({**env, "MEGASCALE_SLICE_ID": "0"})

  def test_rejects_non_positive_and_garbage(self):
    for bad in ("0", "-1", "two", ""):
      with self.subTest(value=bad):
        with self.assertRaises(ValueError):
          cdu.get_num_slices({"MEGASCALE_NUM_SLICES": bad})


class GkeMultiSliceInitTest(unittest.TestCase):

  def test_initializes_on_multi_slice(self):
    """The per-slice bounds vs global WORLD_SIZE must not bail the setup out."""
    env = _gke_env(num_slices=2, slice_id=1, vms=2)
    self.assertTrue(cdu._maybe_init_distributed_on_gke(environ=env))
    self.assertIn("TORCH_TPU_SLICEBUILDER_ADDRESSES", env)

  def test_task_id_is_within_slice(self):
    """The slice's PJRT client numbers its own tasks from zero."""
    env = _gke_env(num_slices=2, slice_id=1, vms=2)
    rank, world = int(env["RANK"]), int(env["WORLD_SIZE"])
    cdu._maybe_init_distributed_on_gke(environ=env)
    self.assertEqual(int(env["CLOUD_TPU_TASK_ID"]), rank % (world // 2))

  def test_slicebuilder_addresses_cover_only_the_local_slice(self):
    """SliceBuilder is per-slice; MegaScale joins slices over DCN."""
    env = _gke_env(num_slices=2, slice_id=0, vms=2)
    cdu._maybe_init_distributed_on_gke(environ=env)
    addrs = env["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")
    self.assertEqual(len(addrs), 2 * 8)
    hosts = {a.rsplit(":", 1)[0] for a in addrs}
    self.assertEqual(hosts, {"h0-0", "h0-1"})

  def test_single_slice_task_id_is_the_global_rank(self):
    env = _gke_env(num_slices=1, slice_id=0, vms=2)
    for k in (
        "MEGASCALE_NUM_SLICES",
        "MEGASCALE_SLICE_ID",
        "MEGASCALE_COORDINATOR_ADDRESS",
    ):
      del env[k]
    rank = int(env["RANK"])
    self.assertTrue(cdu._maybe_init_distributed_on_gke(environ=env))
    self.assertEqual(int(env["CLOUD_TPU_TASK_ID"]), rank)

  def test_requires_the_megascale_client_on_multi_slice(self):
    """Without it each slice numbers devices from zero and the mesh is rejected."""
    env = _gke_env(num_slices=2)
    del env["MEGASCALE_COORDINATOR_ADDRESS"]
    with self.assertRaisesRegex(ValueError, "MEGASCALE_COORDINATOR_ADDRESS"):
      cdu._maybe_init_distributed_on_gke(environ=env)

  def test_slice_id_is_not_required_in_the_environment(self):
    """It is derived from the rank; only the coordinator address is mandatory."""
    env = _gke_env(num_slices=2, slice_id=1, vms=2)
    del env["MEGASCALE_SLICE_ID"]
    self.assertTrue(cdu._maybe_init_distributed_on_gke(environ=env))

  def test_rejects_world_size_not_divisible_by_slices(self):
    with self.assertRaisesRegex(ValueError, "not divisible"):
      cdu._maybe_init_distributed_on_gke(environ=_gke_env(num_slices=2, world=31))

  def test_bails_out_on_inconsistent_bounds(self):
    """Divisible but the per-slice size disagrees with the bounds -> no setup."""
    self.assertFalse(
        cdu._maybe_init_distributed_on_gke(environ=_gke_env(num_slices=2, world=30))
    )


class ReplicateAxisValidationTest(unittest.TestCase):
  """torch_tpu's topology_aware_mesh puts the slice split on the outermost mesh
  axis above extent 1 and, given a bare (pp, dp_replicate, dp_shard, cp, tp)
  shape, cannot tell an unused pp from an unset dp_replicate. So with both at 1
  it splits dp_shard across DCN, logging only at INFO. That is the one case
  checked here; divisibility is torch_tpu's to reject, and pipeline stages
  across slices are a valid layout it handles. Checked, never corrected.

  Each test is one row of torch_tpu 0.2.1's _infer_dcn_mesh_shape(shape, 2).
  """

  def _check(self, num_slices, **degrees):
    with mock.patch.dict("os.environ", {"MEGASCALE_NUM_SLICES": str(num_slices)}):
      config = ParallelismConfig(**degrees)
      cdu.maybe_validate_multislice_parallelism(config)
      return config

  def test_accepts_replicate_equal_to_slice_count(self):
    """(1, 2, 8, 1, 1): torch_tpu splits dp_replicate, as intended."""
    self._check(2, data_parallel_replicate_degree=2)

  def test_accepts_a_multiple_of_the_slice_count(self):
    self._check(2, data_parallel_replicate_degree=4)

  def test_rejects_the_default_on_multi_slice(self):
    """(1, 1, 16, 1, 1): torch_tpu would split dp_shard across DCN, silently."""
    self.assertEqual(ParallelismConfig().data_parallel_replicate_degree, 1)
    self.assertEqual(ParallelismConfig().pipeline_parallel_degree, 1)
    with self.assertRaisesRegex(ValueError, "no axis for the slice split"):
      self._check(2)

  def test_leaves_an_indivisible_replicate_degree_to_torch_tpu(self):
    """(1, 3, 8, 1, 1): torch_tpu raises itself, so this check need not."""
    self._check(2, data_parallel_replicate_degree=3)

  def test_allows_pipeline_stages_across_slices(self):
    """(2, 1, 8, 1, 1): torch_tpu splits pp, a valid layout. The earlier
    divisibility check rejected this."""
    self._check(2, pipeline_parallel_degree=2)

  def test_allows_pipeline_and_replicate_together(self):
    """(2, 2, 4, 1, 1): torch_tpu splits pp, the outermost axis."""
    self._check(2, pipeline_parallel_degree=2, data_parallel_replicate_degree=2)

  def test_error_names_the_flag_to_pass(self):
    with self.assertRaises(ValueError) as caught:
      self._check(2)
    self.assertIn("data_parallel_replicate_degree=$MEGASCALE_NUM_SLICES",
                  str(caught.exception))

  def test_never_mutates(self):
    config = self._check(4, data_parallel_replicate_degree=8)
    self.assertEqual(config.data_parallel_replicate_degree, 8)
    self.assertEqual(config.pipeline_parallel_degree, 1)

  def test_noop_on_single_slice(self):
    """Even the default, which is rejected on multi-slice."""
    config = self._check(1)
    self.assertEqual(config.data_parallel_replicate_degree, 1)


class PinnedHostMemoryTest(unittest.TestCase):
  """Multi-slice must opt out of page-locked batches.

  The MegaScale PJRT client implements no HostMemoryAllocator, and the pinned
  allocation aborts the process from inside the PJRT C API rather than raising,
  so the decision has to be made before the first batch is built. ``collators``
  is left untouched: ``TextCollator`` reads ``HAS_PIN_MEMORY`` as a module
  global on every call, so rebinding it is enough.
  """

  def test_disables_pinning_and_is_idempotent(self):
    from torchtitan.components.data import collators
    from torchtitan.experiments.tpu import workarounds

    original = collators.HAS_PIN_MEMORY
    try:
      collators.HAS_PIN_MEMORY = True
      workarounds.disable_pinned_host_memory()
      self.assertFalse(collators.HAS_PIN_MEMORY)
      workarounds.disable_pinned_host_memory()  # no-op the second time
      self.assertFalse(collators.HAS_PIN_MEMORY)
    finally:
      collators.HAS_PIN_MEMORY = original


class NodeRankTest(unittest.TestCase):
  """Nodes-per-slice is derived, never a function of chips per host."""

  def _env(self, *, vms: int, slices: int, slice_id: int, worker: int):
    return {
        "TPU_WORKER_HOSTNAMES": ",".join(f"h{i}" for i in range(vms)),
        "MEGASCALE_NUM_SLICES": str(slices),
        "TPU_WORKER_ID": str(worker),
        # from the JobSet job-index label via the downward API; the pod hostname
        # is unusable because TPU workloads run hostNetwork
        "MEGASCALE_SLICE_ID": str(slice_id),
    }

  def test_nodes_per_slice_is_the_host_list_length(self):
    for vms in (1, 4, 8, 16):
      with self.subTest(vms=vms):
        env = self._env(vms=vms, slices=2, slice_id=0, worker=0)
        self.assertEqual(cdu.get_nodes_per_slice(env), vms)

  def test_ignores_empty_entries(self):
    """A trailing comma must not inflate the host count."""
    self.assertEqual(
        cdu.get_nodes_per_slice({"TPU_WORKER_HOSTNAMES": "h0,h1,h2,"}), 3
    )
    self.assertEqual(cdu.get_nodes_per_slice({"TPU_WORKER_HOSTNAMES": "h0"}), 1)
    self.assertIsNone(cdu.get_nodes_per_slice({"TPU_WORKER_HOSTNAMES": ","}))

  def test_global_node_rank_is_slice_major(self):
    # 8 VMs per slice, 2 slices: slice 1 starts at node rank 8.
    for slice_id, worker, expected in ((0, 0, 0), (0, 7, 7), (1, 0, 8), (1, 7, 15)):
      with self.subTest(slice_id=slice_id, worker=worker):
        env = self._env(vms=8, slices=2, slice_id=slice_id, worker=worker)
        self.assertEqual(cdu.get_global_node_rank(env), expected)

  def test_independent_of_chips_per_host(self):
    """A generation with a different VM count per slice needs no code change."""
    env = self._env(vms=4, slices=2, slice_id=1, worker=2)
    self.assertEqual(cdu.get_global_node_rank(env), 6)

  def test_four_slices(self):
    env = self._env(vms=8, slices=4, slice_id=3, worker=5)
    self.assertEqual(cdu.get_global_node_rank(env), 29)

  def test_returns_none_rather_than_guessing(self):
    self.assertIsNone(cdu.get_nodes_per_slice({}))
    self.assertIsNone(cdu.get_global_node_rank({}))
    # host list present but no worker id
    self.assertIsNone(cdu.get_global_node_rank({"TPU_WORKER_HOSTNAMES": "h0,h1"}))

  def test_none_when_multi_slice_has_no_slice_index(self):
    """Better to fail than to put every slice at node rank 0."""
    self.assertIsNone(cdu.get_global_node_rank({
        "TPU_WORKER_HOSTNAMES": "h0,h1",
        "TPU_WORKER_ID": "1",
        "MEGASCALE_NUM_SLICES": "2",
    }))


if __name__ == "__main__":
  unittest.main()


class SliceIdFromWorkerHostnamesTest(unittest.TestCase):
  """The slice index recovered from the per-slice pod DNS list.

  GKE exposes no JOBSET_* variable and Kubernetes' JOB_COMPLETION_INDEX is the
  worker index, so this host list is the only place the slice index appears
  without wiring the job-index label through the downward API. Hostnames below
  are verbatim from a 2-slice v6e run.
  """

  def test_recovers_each_slice_index(self):
    for slice_id in (0, 1):
      environ = {
          "MEGASCALE_NUM_SLICES": "2",
          "TPU_WORKER_HOSTNAMES": (
              f"ms-v6e-fast-main-job-{slice_id}-0.ms-v6e-fast,"
              f"ms-v6e-fast-main-job-{slice_id}-1.ms-v6e-fast"
          ),
      }
      self.assertEqual(
          cdu.get_slice_id_from_worker_hostnames(environ),
          slice_id,
      )

  def test_single_slice_job_is_slice_zero(self):
    environ = {
        "MEGASCALE_NUM_SLICES": "1",
        "TPU_WORKER_HOSTNAMES": "job-main-job-0-0.svc",
    }
    self.assertEqual(
        cdu.get_slice_id_from_worker_hostnames(environ), 0
    )

  def test_bare_hostnames_without_subdomain_still_parse(self):
    environ = {
        "MEGASCALE_NUM_SLICES": "2",
        "TPU_WORKER_HOSTNAMES": "w-main-job-1-0,w-main-job-1-1",
    }
    self.assertEqual(
        cdu.get_slice_id_from_worker_hostnames(environ), 1
    )

  def test_node_names_are_rejected(self):
    """hostNetwork pods see node names; those carry no slice index."""
    environ = {
        "MEGASCALE_NUM_SLICES": "2",
        "TPU_WORKER_HOSTNAMES": "gke-tpu-0d0013c1-1r6v,gke-tpu-46dd3e54-gcfn",
    }
    self.assertIsNone(
        cdu.get_slice_id_from_worker_hostnames(environ)
    )

  def test_disagreeing_job_indices_are_rejected(self):
    environ = {
        "MEGASCALE_NUM_SLICES": "2",
        "TPU_WORKER_HOSTNAMES": "w-main-job-0-0.s,w-main-job-1-1.s",
    }
    self.assertIsNone(
        cdu.get_slice_id_from_worker_hostnames(environ)
    )

  def test_non_contiguous_pod_indices_are_rejected(self):
    environ = {
        "MEGASCALE_NUM_SLICES": "2",
        "TPU_WORKER_HOSTNAMES": "w-main-job-1-0.s,w-main-job-1-7.s",
    }
    self.assertIsNone(
        cdu.get_slice_id_from_worker_hostnames(environ)
    )

  def test_slice_index_beyond_the_slice_count_is_rejected(self):
    environ = {
        "MEGASCALE_NUM_SLICES": "2",
        "TPU_WORKER_HOSTNAMES": "w-main-job-5-0.s,w-main-job-5-1.s",
    }
    self.assertIsNone(
        cdu.get_slice_id_from_worker_hostnames(environ)
    )

  def test_missing_host_list_is_none(self):
    self.assertIsNone(
        cdu.get_slice_id_from_worker_hostnames(
            {"MEGASCALE_NUM_SLICES": "2"}
        )
    )

  def test_get_slice_id_uses_it_without_the_variable(self):
    """No MEGASCALE_SLICE_ID and no slice-ordered ranks: hostnames decide."""
    environ = {
        "MEGASCALE_NUM_SLICES": "2",
        "TPU_WORKER_HOSTNAMES": "w-main-job-1-0.s,w-main-job-1-1.s",
    }
    self.assertEqual(cdu.get_slice_id(environ), 1)

  def test_get_slice_id_prefers_hostnames_over_the_rank(self):
    """A rank that is not slice-major must not override the host list."""
    environ = {
        "MEGASCALE_NUM_SLICES": "2",
        "TPU_WORKER_HOSTNAMES": "w-main-job-1-0.s,w-main-job-1-1.s",
        "RANK": "0",
        "WORLD_SIZE": "16",
    }
    self.assertEqual(cdu.get_slice_id(environ), 1)

  def test_declared_variable_disagreeing_with_hostnames_raises(self):
    environ = {
        "MEGASCALE_NUM_SLICES": "2",
        "MEGASCALE_SLICE_ID": "0",
        "TPU_WORKER_HOSTNAMES": "w-main-job-1-0.s,w-main-job-1-1.s",
    }
    with self.assertRaisesRegex(ValueError, "TPU_WORKER_HOSTNAMES"):
      cdu.get_slice_id(environ)

  def test_global_node_rank_without_the_variable(self):
    environ = {
        "MEGASCALE_NUM_SLICES": "2",
        "TPU_WORKER_HOSTNAMES": "w-main-job-1-0.s,w-main-job-1-1.s",
        "TPU_WORKER_ID": "1",
    }
    self.assertEqual(cdu.get_global_node_rank(environ), 3)

  def test_global_node_rank_none_when_nothing_carries_the_slice(self):
    environ = {
        "MEGASCALE_NUM_SLICES": "2",
        "TPU_WORKER_HOSTNAMES": "gke-tpu-aaaa-bbbb,gke-tpu-cccc-dddd",
        "TPU_WORKER_ID": "1",
    }
    self.assertIsNone(
        cdu.get_global_node_rank(environ)
    )
