# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.tools.utils import get_local_device


def _tpu_device_type_registered() -> bool:
    try:
        torch.device("tpu")
    except RuntimeError:
        return False
    return True


# TODO(jerryx99): We skip running on tpu as this pytest test runs on cpu runner.
# (This is the only test in tests/unit_tests that parameterizes over cpu and tpu).
# We cannot move to the single-chip TPU shard as it expects tests in absltest
# format. Either enable the TPU shards to dispatch both absltest and pytest tests
# and route this specific test there, or move all TPU specific testing under
# experiments/tpu for consistency
_TPU = pytest.param(
    "tpu",
    marks=pytest.mark.skipif(
        not _tpu_device_type_registered(),
        reason="torch has no tpu device type registered",
    ),
)


class _FakeDeviceModule:
    def __init__(self, num_devices: int):
        self.num_devices = num_devices

    def device_count(self) -> int:
        return self.num_devices


def _patch_device(
    monkeypatch: pytest.MonkeyPatch, device_type: str, num_devices: int
) -> None:
    monkeypatch.setattr("torchtitan.tools.utils.device_type", device_type)
    monkeypatch.setattr(
        "torchtitan.tools.utils.device_module", _FakeDeviceModule(num_devices)
    )


def test_get_local_device_uses_local_rank_when_multiple_devices_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")
    _patch_device(monkeypatch, "cuda", 8)

    assert get_local_device() == torch.device("cuda:3")


def test_get_local_device_uses_zero_when_one_device_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")
    _patch_device(monkeypatch, "cuda", 1)

    assert get_local_device() == torch.device("cuda:0")


@pytest.mark.parametrize("device_type", ["cuda", "cpu", _TPU])
def test_get_local_device_ignores_local_rank_for_a_single_device(
    monkeypatch: pytest.MonkeyPatch, device_type: str
) -> None:
    # When a process owns one device, LOCAL_RANK is not a device index on any
    # lane. Assert that invariant rather than an exact device object, so the
    # test holds whether the lane reports index 0 or no index at all.
    monkeypatch.setenv("LOCAL_RANK", "3")
    _patch_device(monkeypatch, device_type, 1)

    device = get_local_device()
    assert device.type == device_type
    assert device.index in (None, 0)


def test_get_local_device_rejects_out_of_range_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "8")
    _patch_device(monkeypatch, "cuda", 8)

    with pytest.raises(ValueError, match="outside the visible cuda device count"):
        get_local_device()


def test_get_local_device_requires_local_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    _patch_device(monkeypatch, "cuda", 8)

    with pytest.raises(ValueError, match="LOCAL_RANK must be set"):
        get_local_device()


@pytest.mark.parametrize("device_type", ["cpu", _TPU])
def test_get_local_device_is_index_free_for_single_device_lanes(
    monkeypatch: pytest.MonkeyPatch, device_type: str
) -> None:
    # A process owns exactly one cpu/tpu device, so LOCAL_RANK is not a device
    # index there. tpu additionally reports the host chip count from
    # device_count() until the PJRT runtime is initialized, so the
    # visible-device heuristic must not run on that lane.
    monkeypatch.setenv("LOCAL_RANK", "3")
    _patch_device(monkeypatch, device_type, 4)

    assert get_local_device() == torch.device(device_type)
