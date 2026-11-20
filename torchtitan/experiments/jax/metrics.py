"""Metrics tracking for JAX-based experiments (jax/ and torchax/).

``JaxMetricsProcessor`` holds the shared TPS/TFLOPs/MFU computation and the
colorized log line. Torchax's subclass adds ``parallel_dims`` plus a richer
``log()`` signature for global-avg/max loss and grad_norm.
"""

import logging
import time
from typing import Any

from torchtitan.tools import utils

logger = logging.getLogger(__name__)


def get_peak_flops(accelerator: str, num_devices, tpu_megacore) -> int:
    """Return the system-wide peak theoretical BF16 FLOPS for MFU.

    Args:
        accelerator: Short tag ('v4', 'v5e', 'v5p', 'v6e', 'v7x', 'h100', 'a100', 'cpu').
        num_devices: Total addressable devices (``jax.device_count()``).
        tpu_megacore: For v4/v5p, multiply per-core FLOPS by 2 (chip-level).
    """
    tflops_per_core = {
        # TPUs. Chips with 2 cores quote per-chip peak; we store per-core here.
        # https://cloud.google.com/tpu/docs/v4
        'v4': 275 // 2,
        # https://cloud.google.com/tpu/docs/v5p
        'v5p': 459 // 2,
        # https://cloud.google.com/tpu/docs/v5e
        'v5e': 197,
        # https://cloud.google.com/tpu/docs/v6e
        'v6e': 918,
        # https://cloud.google.com/tpu/docs/tpu7x
        'v7x': 2307 // 2,
        # GPU SXM dense peaks.
        # https://www.nvidia.com/en-us/data-center/a100/
        'a100': 312,
        # https://www.nvidia.com/en-us/data-center/h100/
        'h100': 989,
    }

    if accelerator not in tflops_per_core:
        raise ValueError(f'Unsupported accelerator type: {accelerator}')

    flops_per_jax_device = tflops_per_core[accelerator]
    if tpu_megacore and accelerator in ('v4', 'v5p'):
        flops_per_jax_device *= 2

    return int(flops_per_jax_device * num_devices * 1_000_000_000_000)


class JaxMetricsProcessor:
    """Metrics processor for jax/ and torchax/ experiments.

    Tracks tokens/step, step time, TPS, TFLOPs, and MFU, and emits a
    colorized log line. ``log()``'s signature accepts both the minimal
    ``(step, loss)`` form used by the jax experiment and the torchtitan-style
    ``(step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=...)``
    form used by the torchax experiment — only the average loss drives the
    summary line; the extra arguments are accepted for interface parity.
    """

    def __init__(
        self,
        job_config,
        accelerator: str,
        num_global_devices: int,
        tpu_megacore: bool,
        log_freq: int = 10,
        parallel_dims=None,
        tag: str | None = None,
    ):
        self.job_config = job_config
        self.log_freq = log_freq
        self.color = (
            utils.NoColor()
            if job_config.metrics.disable_color_printing
            else utils.Color()
        )
        self.gpu_peak_flops = get_peak_flops(
            accelerator, num_global_devices, tpu_megacore
        )
        self.num_global_devices = num_global_devices
        self.ntokens_since_last_log = 0
        self.data_loading_times: list[float] = []
        self.time_last_log = time.perf_counter()
        self.num_flops_per_token: int = 0

        # Optional fields used by the torchax trainer (kept as None in the
        # pure-jax case).
        self.parallel_dims = parallel_dims
        self.tag = tag
        self.optimizers = None
        self.lr_schedulers = None
        self.model_parts = None

    def should_log(self, step: int) -> bool:
        return step == 1 or step % self.log_freq == 0

    def log(
        self,
        step: int,
        loss: float,
        max_loss: float | None = None,
        grad_norm: float | None = None,
        extra_metrics: dict[str, Any] | None = None,
    ):
        # Extra arguments are accepted for interface parity; the summary
        # log line itself only uses the average loss.
        self._log_summary(step, loss)

    def _log_summary(self, step: int, avg_loss: float):
        """Compute TPS/TFLOPs/MFU, emit the colorized log line, reset counters."""
        time_delta = time.perf_counter() - self.time_last_log
        tps = self.ntokens_since_last_log / time_delta
        per_device_tps = tps / self.num_global_devices
        mfu = (
            100 * self.num_flops_per_token * tps / self.gpu_peak_flops
            if self.num_flops_per_token > 0 else 0.0
        )
        tflops = self.num_flops_per_token * tps / 1e12

        actual_steps = max(1, len(self.data_loading_times))
        time_e2e = time_delta / actual_steps

        c = self.color
        logger.info(
            f'{c.red}step: {step:2}  '
            f'{c.green}loss: {avg_loss:7.4f}  '
            f'{c.orange}step_time: {time_e2e:7.4f}s  '
            f'{c.blue}tps: {round(tps):,}  '
            f'per_device_tps: {round(per_device_tps):,}  '
            f'{c.cyan}tflops: {tflops:,.2f}  '
            f'{c.magenta}mfu: {mfu:.2f}%{c.reset}'
        )

        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
