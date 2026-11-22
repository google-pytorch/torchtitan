"""Tests for the distributed test harness's worker bootstrap."""

import os
import types
from unittest import mock

from absl.testing import absltest

from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.distributed import distributed_utils
from torchtitan.experiments.tpu.testing import base_distributed_device_test

_RAGGED_A2A_SC = "xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all"


class DistributedInitWrapperTest(absltest.TestCase):

  def _libtpu_args_seen_by_entry_point(self, ep_degree: int) -> str | None:
    seen = {}

    def entry_point(config):
      seen["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS")
      return config

    config = types.SimpleNamespace(
        parallelism=types.SimpleNamespace(expert_parallel_degree=ep_degree)
    )
    wrapper = base_distributed_device_test._DistributedInitWrapper(entry_point)
    with mock.patch.dict(os.environ), mock.patch.object(
        distributed_utils, "maybe_init_distributed"
    ) as maybe_init_distributed, mock.patch.object(
        tpu_flags, "_installed_libtpu_version", return_value="0.0.50"
    ):
      os.environ.pop("LIBTPU_INIT_ARGS", None)

      self.assertIs(wrapper(config), config)

    maybe_init_distributed.assert_called_once_with()
    return seen["LIBTPU_INIT_ARGS"]

  def test_ep_worker_gets_the_tensorcore_flag_before_the_entry_point(self):
    self.assertEqual(
        tpu_flags.parse_flags(self._libtpu_args_seen_by_entry_point(2)),
        {_RAGGED_A2A_SC: "false"},
    )

  def test_worker_without_ep_gets_no_libtpu_flags(self):
    self.assertIsNone(self._libtpu_args_seen_by_entry_point(1))


if __name__ == "__main__":
  absltest.main()
