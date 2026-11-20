# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import os as os

import torch

from torchtitan.config import ConfigManager, ParallelismConfig
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.trainer import Trainer


logger = logging.getLogger(__name__)


def _maybe_enable_min_cost_redistribute(
    stack: contextlib.ExitStack, parallelism: ParallelismConfig
) -> None:
    """Switches DTensor to the min-cost redistribution planner, if requested.

    Must be entered before the model is built and compiled: the planner runs
    during tracing, so a plan chosen under the greedy planner is baked into the
    compiled graph and a later switch has no effect.

    Args:
      stack: Exit stack that must outlive both model construction and training.
      parallelism: Parallelism config carrying the opt-in flag.
    """
    if not parallelism.enable_min_cost_redistribute:
        return

    # Private API: no public entry point exists for planner selection.
    from torch.distributed.tensor._redistribute import (  # pylint: disable=g-import-not-at-top
        _gen_transform_infos,
        use_min_cost_redistribution_plan,
    )

    stack.enter_context(use_min_cost_redistribution_plan(True))
    # The context manager only flips a global; it does not invalidate the plan
    # cache, so plans already computed by the greedy planner would be reused.
    _gen_transform_infos.cache_clear()
    logger.info("Using DTensor min-cost (graph-based) redistribution planner.")


def main() -> None:
    """Main entry point for training."""
    init_logger()

    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    config_manager = ConfigManager()
    config = config_manager.parse_args()

    # NOTE: internal meta tooling relies on source="training".
    sl.init_structured_logger(
        source="training",
        # pyrefly: ignore [missing-attribute]
        output_dir=config.dump_folder,
        # pyrefly: ignore [missing-attribute]
        enable=config.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    trainer: Trainer | None = None

    try:
        # The planner selection has to span model construction, since
        # redistribution plans are resolved while tracing, not just at runtime.
        with contextlib.ExitStack() as stack:
            _maybe_enable_min_cost_redistribute(
                stack,
                config.parallelism,  # pyrefly: ignore [missing-attribute]
            )

            trainer = config.build()  # pyrefly: ignore [missing-attribute]

            if (
                config.checkpoint.create_seed_checkpoint  # pyrefly: ignore[missing-attribute]
            ):
                assert (
                    int(os.environ["WORLD_SIZE"]) == 1
                ), "Must create seed checkpoint using a single device, to disable sharding."
                assert (
                    config.checkpoint.enable  # pyrefly: ignore [missing-attribute]
                ), "Must enable checkpointing when creating a seed checkpoint."
                trainer.checkpointer.save(curr_step=0, last_step=True)
                logger.info("Created seed checkpoint")
            else:
                trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            with sl.log_trace_span("torch_distributed_teardown"):
                torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
