# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MfuBenchmark:
    """Get a longer-time average without requiring a less frequent ``log_freq``
    for other metrics.
    """

    metrics_processor: Any
    start_step: int
    final_step: int
    _start_time: float | None = None
    _start_tokens: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.start_step < self.final_step:
            raise ValueError("MFU benchmark requires 0 < start_step < final_step")
        for step in (self.start_step, self.final_step):
            if not self.metrics_processor.should_log(step):
                raise ValueError(
                    "MFU benchmark boundaries must also be metric log steps; "
                    f"step {step} is not logged"
                )

    def record(self, step: int, ntokens_seen: int) -> None:
        """Record a boundary after the normal metrics path has drained the device."""
        if step == self.start_step:
            self._start_time = time.perf_counter()
            self._start_tokens = ntokens_seen
            return
        if step != self.final_step:
            return
        if self._start_time is None or self._start_tokens is None:
            raise RuntimeError("MFU benchmark final step reached before start step")

        elapsed = time.perf_counter() - self._start_time
        measured_steps = self.final_step - self.start_step
        tokens_per_rank = ntokens_seen - self._start_tokens
        non_data_parallel_size = (
            self.metrics_processor.parallel_dims.non_data_parallel_size
        )
        tps = tokens_per_rank / (elapsed * non_data_parallel_size)
        tflops = self.metrics_processor.num_flops_per_token * tps / 1e12
        mfu = 100 * tflops * 1e12 / self.metrics_processor.gpu_peak_flops
        logger.info(
            "[mfu_benchmark] start_step=%d final_step=%d "
            "measured_steps=%d wall=%.6fs mean_step_time=%.6fs "
            "tokens_per_rank=%d tflops=%.6f mfu=%.6f%%",
            self.start_step,
            self.final_step,
            measured_steps,
            elapsed,
            elapsed / measured_steps,
            tokens_per_rank,
            tflops,
            mfu,
        )
