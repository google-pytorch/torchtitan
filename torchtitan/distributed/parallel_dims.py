# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

from torchtitan.distributed import parallelism_context as _parallelism_context
from torchtitan.distributed.parallelism_context import (
    DistributedTopology,
    MeshAxisName,
    ParallelismContext,
    init_device_mesh,
    unfold_dp_axes,
    unfold_dp_axis,
)

ParallelDims = ParallelismContext

__all__ = [
    "DistributedTopology",
    "MeshAxisName",
    "ParallelDims",
    "ParallelismContext",
    "init_device_mesh",
    "unfold_dp_axes",
    "unfold_dp_axis",
]
