"""CPU tests for genuine two-rank Qwen3 joint graph capture and replay."""

import os
import tempfile
import unittest
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

from torchtitan.experiments.tpu.experimental_flex_shard.validation import (
    validate_worker,
)


def worker(rank, store, output):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"file://{store}"
    )
    try:
        for checkpointed in (False, True):
            validate_worker(device="cpu", checkpointed=checkpointed, output=output)
    finally:
        dist.destroy_process_group()


class TraceableQwenTest(unittest.TestCase):
    def test_real_rank_graph_and_numerics(self):
        from torchtitan.experiments.tpu.experimental_flex_shard.tests.common import (
            require_executor,
        )

        require_executor(self)
        with tempfile.TemporaryDirectory() as temporary:
            output = os.environ.get("TORCHTITAN_FLEX_SHARD_RESULTS", temporary)
            mp.start_processes(
                worker,
                args=(str(Path(temporary) / "store"), output),
                nprocs=2,
                start_method="fork",
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
