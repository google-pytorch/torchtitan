"""Tests for pallas_vmem."""

from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import torch
from torchtitan.experiments.tpu.kernels import pallas_vmem


class PallasVmemTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    pallas_vmem.tpu_generation.cache_clear()
    self.addCleanup(pallas_vmem.tpu_generation.cache_clear)

  def test_tpu_generation_device_name_parsing(self):
    cases = (
        ("TPU v5 lite", 5),
        ("TPU v6e", 6),
        ("tpu7x", 7),
        ("TPU-7", 7),
        ("TPU 7", 7),
        ("unknown accelerator", None),
    )
    for dev_name, expected in cases:
      pallas_vmem.tpu_generation.cache_clear()
      fake_tpu = SimpleNamespace(
          get_device_properties=lambda n=dev_name: SimpleNamespace(name=n)
      )
      with (
          self.subTest(dev_name=dev_name),
          mock.patch.object(torch, "tpu", fake_tpu, create=True),
      ):
        self.assertEqual(pallas_vmem.tpu_generation(), expected)

    pallas_vmem.tpu_generation.cache_clear()
    failing_tpu = SimpleNamespace(
        get_device_properties=mock.Mock(side_effect=RuntimeError("no device"))
    )
    with mock.patch.object(torch, "tpu", failing_tpu, create=True):
      self.assertIsNone(pallas_vmem.tpu_generation())

  def test_vmem_limit_bytes_per_generation(self):
    mib = 1024 * 1024
    cases = ((None, 16 * mib), (5, 16 * mib), (6, 32 * mib), (7, 64 * mib))
    for gen, expected in cases:
      with (
          self.subTest(gen=gen),
          mock.patch.object(pallas_vmem, "tpu_generation", return_value=gen),
      ):
        self.assertEqual(pallas_vmem.vmem_limit_bytes(), expected)


if __name__ == "__main__":
  absltest.main()
