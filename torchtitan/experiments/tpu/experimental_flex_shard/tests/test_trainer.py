"""Full-Trainer CPU parity with common initial weights and exact repeat runs."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor

from torchtitan.experiments.tpu.experimental_flex_shard.config_registry import (
    qwen3_traceable_flex_shard,
)
from torchtitan.experiments.tpu.experimental_flex_shard.validation import (
    build_initialized_model,
)


def trainer_worker(rank, store, output):
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE="2",
        GLOO_SOCKET_IFNAME="lo",
    )
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"file://{store}"
    )
    try:
        recorded = []
        for variant in ("simple_fsdp", "flex_shard", "flex_shard_repeat"):
            config = qwen3_traceable_flex_shard()
            config.use_traceable_flex_shard = variant != "simple_fsdp"
            config.compile.enable_passes = False  # CPU executes the captured FX graph.
            config.dump_folder = str(Path(output) / variant / f"rank{rank}")
            config.metrics.enable_tensorboard = True
            trainer = config.build()
            # Stock SimpleFSDP initializes already-sharded DTensors; the portable
            # executor shards full initialized weights. Normalize both to one
            # initial state before comparing algorithms, not RNG layouts.
            full_parameters = dict(build_initialized_model().named_parameters())
            with torch.no_grad():
                for name, parameter in trainer.model_parts[0].named_parameters():
                    expected = full_parameters[name].chunk(2, 0)[rank]
                    local = (
                        parameter.to_local()
                        if isinstance(parameter, DTensor)
                        else parameter
                    )
                    local.copy_(expected)
            rows = []
            original_log = trainer.metrics_processor.log

            def log(
                step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None
            ):
                rows.append(
                    {
                        "step": step,
                        "loss": float(global_avg_loss),
                        "grad_norm": float(grad_norm),
                    }
                )
                original_log(
                    step, global_avg_loss, global_max_loss, grad_norm, extra_metrics
                )

            trainer.metrics_processor.log = log
            try:
                trainer.train()
                assert trainer._traced_step is not None
                if config.use_traceable_flex_shard:
                    assert hasattr(trainer.model_parts[0], "_graph_flex_shard_buckets")
                path = Path(config.dump_folder)
                path.mkdir(parents=True, exist_ok=True)
                (path / "joint.fx.txt").write_text(
                    trainer._traced_step.gm.print_readable(print_output=False)
                )
                (path / "metrics.json").write_text(json.dumps(rows, indent=2))
                recorded.append(rows)
            finally:
                trainer.close()
        assert len(recorded[0]) == 10
        assert recorded[0] == recorded[1], "SimpleFSDP/FlexShard loss or norm differs"
        assert recorded[1] == recorded[2], (
            "FlexShard repeat is not bitwise deterministic"
        )
        print(f"FULL_TRAINER_PASS rank={rank} steps=10 bitwise_loss_and_norm=True")
    finally:
        dist.destroy_process_group()


class TrainerParityTest(unittest.TestCase):
    def test_bitwise_loss_norm_and_repeat(self):
        from torchtitan.experiments.tpu.experimental_flex_shard.tests.common import (
            require_executor,
        )

        require_executor(self)
        with tempfile.TemporaryDirectory() as temporary:
            output = os.environ.get("TORCHTITAN_FLEX_SHARD_RESULTS", temporary)
            mp.start_processes(
                trainer_worker,
                args=(str(Path(temporary) / "store"), output),
                nprocs=2,
                start_method="fork",
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
