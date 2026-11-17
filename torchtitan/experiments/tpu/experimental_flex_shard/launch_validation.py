"""Run source-identical graph validation on each detected TPU device rank."""

import torch
from torch.distributed import launcher
from torch.tpu import distributed

from torchtitan.experiments.tpu.experimental_flex_shard.validation import (
    validate_worker,
)


def worker():
    import torch.distributed as dist

    dist.init_process_group("tpu_dist")
    try:
        for checkpointed in (False, True):
            validate_worker(
                device="tpu",
                checkpointed=checkpointed,
                output="/tmp/traceable-flex-results",
                steps=10,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    distributed.set_tpu_launch_env()
    count = torch.accelerator.device_count()
    config = launcher.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=count,
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
    )
    launcher.elastic_launch(config, worker)()
