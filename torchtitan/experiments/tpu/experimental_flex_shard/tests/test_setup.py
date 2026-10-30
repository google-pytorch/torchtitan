"""Setup and gradient-norm contracts for the opt-in storage executor."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.config import TrainingConfig
from torchtitan.experiments.tpu.experimental_flex_shard.config_registry import (
    qwen3_traceable_flex_shard,
)
from torchtitan.experiments.tpu.experimental_flex_shard.parallelize import (
    clip_grad_norm,
    defer_installation,
)
from torchtitan.experiments.tpu.experimental_flex_shard.trainer import validate_config
from torchtitan.experiments.tpu.experimental_flex_shard.validation import (
    build_initialized_model,
)


def setup_worker(rank, store):
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"file://{store}"
    )
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("fsdp",))
    dims = SimpleNamespace(get_mesh=lambda name: mesh)
    training = TrainingConfig(mixed_precision_param="float32")
    reference = build_initialized_model()
    with torch.device("meta"):
        model = reference.config.build()
    defer_installation(model, parallel_dims=dims, training=training)
    assert not hasattr(model, "_graph_flex_shard_buckets")
    model.to_empty(device="cpu")
    torch.manual_seed(42)
    model.init_weights()
    assert hasattr(model, "_graph_flex_shard_buckets")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    assert [id(p) for p in optimizer.param_groups[0]["params"]] == [
        id(p) for p in model.parameters()
    ]
    try:
        model.init_states()
    except ValueError:
        pass
    else:
        raise AssertionError("Reinitializing sharded storage must fail")
    # A size-one scalar is replicated; the matrix is sharded. Only shard
    # squared norms get reduced, or replicated values are counted twice.
    from torchtitan.experiments.tpu.experimental_flex_shard.dependency import (
        load_executor,
    )

    simple = nn.Linear(4, 4, bias=False)
    simple.scale = nn.Parameter(torch.tensor(1.0))
    executor = load_executor()
    executor.data_parallel(simple, mesh, mode="fully_shard")
    simple._parameters["weight"].grad = torch.full_like(
        simple._parameters["weight"], rank + 1.0
    )
    simple._parameters["scale"].grad = torch.tensor(3.0)
    expected = torch.tensor((8 * 1.0 + 8 * 4.0 + 9.0) ** 0.5)
    actual = clip_grad_norm(simple, 1.0)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        simple._parameters["scale"].grad, 3.0 / (expected + 1e-6)
    )
    for parameter in simple.parameters():
        parameter.grad = None
    torch.testing.assert_close(clip_grad_norm(simple, 1.0), torch.tensor(0.0))
    dist.destroy_process_group()


class SetupContractTest(unittest.TestCase):
    def test_config_rejections(self):
        config = qwen3_traceable_flex_shard()
        validate_config(config)
        config.parallelism.tensor_parallel_degree = 2
        with self.assertRaises(ValueError):
            validate_config(config)
        config = qwen3_traceable_flex_shard()
        config.checkpoint.enable = True
        with self.assertRaises(ValueError):
            validate_config(config)
        config = qwen3_traceable_flex_shard()
        config.training.num_tokens_per_train_step = 64
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_deferred_setup_and_global_norm(self):
        with tempfile.TemporaryDirectory() as temporary:
            mp.start_processes(
                setup_worker,
                args=(str(Path(temporary) / "store"),),
                nprocs=2,
                start_method="fork",
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
