"""A checkpoint written by a stacked model must reshard like any other.

``test_layer_stack.py`` pins the saved *format* -- same keys, same bytes as the
unrolled model -- but does it at world size 1, where nothing is actually
sharded. The claim this file tests is the one an owner of a 70B checkpoint
cares about: save on eight ranks, come back on four, get the same weights, in
either layout.

That needs two real process groups of different widths, so each case spawns its
own. These tests are therefore slow (seconds, not milliseconds) and live apart
from the single-process ones.
"""

import contextlib
import functools
import os
import socket
import tempfile
import unittest

from absl.testing import absltest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
import torch.multiprocessing as mp

from torchtitan.components.optimizer.utils import (
    get_flat_optim_state_dict,
    init_optim_state,
)
from torchtitan.config import derive
from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.graph_trainer.layer_stack import (
    with_tpu_stacked_layers,
)
from torchtitan.experiments.tpu.graph_trainer.tests import test_layer_stack
from torchtitan.experiments.tpu.layer_loop import tpu_model_class
from torchtitan.models.llama3 import model_registry


SEED = 1234


def _layer_loop(config):
    """The unstacked arm: the same model class, without the stacking."""
    return derive(config, tpu_model_class(type(config)).Config)


def _build(world_size: int, *, stacked: bool) -> torch.nn.Module:
    """The trainer's build order, at an arbitrary data-parallel width."""
    # ParallelDims reads the accelerator type at import; there is none here.
    import torchtitan.distributed.parallel_dims as parallel_dims_module

    parallel_dims_module.device_type = "cpu"

    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=world_size,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    # The rewrite `TpuGraphTrainer.Config.from_tpu_trainer_config` applies.
    # Both arms get one, so stacking is the only difference; a stock `Decoder`
    # loops over its layers directly and would ignore the stacks.
    convert = (
        functools.partial(with_tpu_stacked_layers, use_scan=False)
        if stacked
        else _layer_loop
    )
    with torch.device("meta"):
        model = convert(
            model_registry("debugmodel", attn_backend="sdpa").model
        ).build()
    model = apply_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=TrainingConfig(
            mixed_precision_param="float32", mixed_precision_reduce="float32"
        ),
    )
    model.to_empty(device="cpu")
    torch.manual_seed(SEED)
    model.init_weights()
    return model


def _gathered(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The state dict as whole tensors. Collective: every rank must call it."""
    return {k: v.full_tensor() for k, v in model.state_dict().items()}


@contextlib.contextmanager
def _process_group(rank: int, world_size: int, port: int):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        world_size=world_size,
        rank=rank,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def _save_worker(rank: int, world_size: int, port: int, path: str, stacked: bool):
    with _process_group(rank, world_size, port):
        model = _build(world_size, stacked=stacked)
        dcp.save(model.state_dict(), checkpoint_id=os.path.join(path, "ckpt"))
        # Written from the source of truth so the load side has something to be
        # wrong against; a load that quietly no-ops would otherwise pass.
        whole = _gathered(model)
        if rank == 0:
            torch.save(whole, os.path.join(path, "expected.pt"))


def _load_worker(rank: int, world_size: int, port: int, path: str, stacked: bool):
    with _process_group(rank, world_size, port):
        model = _build(world_size, stacked=stacked)
        # Perturb first: `init_weights` is seeded, so without this the model
        # already holds the saved values and any load at all would pass.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        state_dict = model.state_dict()
        dcp.load(state_dict, checkpoint_id=os.path.join(path, "ckpt"))
        model.load_state_dict(state_dict)

        got = _gathered(model)
        if rank != 0:
            return
        want = torch.load(os.path.join(path, "expected.pt"), weights_only=True)
        assert set(got) == set(want), (set(got) ^ set(want))
        for key, expected in want.items():
            torch.testing.assert_close(got[key], expected, rtol=0, atol=0, msg=key)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def _flattened_moe_optim_state(world_size: int) -> dict[str, torch.Tensor]:
    """A 4-expert stacked MoE's flat AdamW state, at ``world_size`` ranks.

    Above 4 ranks the expert and router stacks are stored flattened, and so
    are their moments; at 4 ranks nothing is flattened.
    """
    # ParallelDims reads the accelerator type at import; there is none here.
    import torchtitan.distributed.parallel_dims as parallel_dims_module

    parallel_dims_module.device_type = "cpu"
    model = test_layer_stack._build(  # pylint: disable=protected-access
        stacked=True,
        flatten=True,
        config=test_layer_stack._moe_config(num_experts=4),  # pylint: disable=protected-access
        world_size=world_size,
    )
    optim = torch.optim.AdamW(list(model.named_parameters()))
    init_optim_state(optim)
    return get_flat_optim_state_dict(optim)


def _save_flattened_optim_worker(rank: int, world_size: int, port: int, path: str):
    with _process_group(rank, world_size, port):
        state = _flattened_moe_optim_state(world_size)
        dcp.save(state, checkpoint_id=os.path.join(path, "optim"))


def _load_flattened_optim_worker(rank: int, world_size: int, port: int, path: str):
    with _process_group(rank, world_size, port):
        state = _flattened_moe_optim_state(world_size)
        try:
            dcp.load(state, checkpoint_id=os.path.join(path, "optim"))
        except CheckpointException as err:
            assert "Size mismatch" in str(err), err
        else:
            raise AssertionError(
                f"rank {rank} loaded flattened optimizer state into unflattened "
                "stacks"
            )


class ReshardTest(unittest.TestCase):
    """Eight ranks out, four ranks back, across both layouts."""

    def _round_trip(self, *, saved_stacked: bool, loaded_stacked: bool):
        with tempfile.TemporaryDirectory() as path:
            mp.spawn(
                _save_worker, args=(8, _free_port(), path, saved_stacked), nprocs=8
            )
            mp.spawn(
                _load_worker, args=(4, _free_port(), path, loaded_stacked), nprocs=4
            )

    def test_stacked_to_stacked(self):
        self._round_trip(saved_stacked=True, loaded_stacked=True)

    def test_stacked_to_unrolled(self):
        """Nothing traps a run that started with scan into keeping it."""
        self._round_trip(saved_stacked=True, loaded_stacked=False)

    def test_unrolled_to_stacked(self):
        """The migration this work exists to make safe: an old checkpoint in."""
        self._round_trip(saved_stacked=False, loaded_stacked=True)


class OptimizerStateTest(unittest.TestCase):
    """The weights above reshard across layouts. The optimizer state does not.

    TODO: make this pass. The model's state dict is written in the unrolled
    format because ``state_dict`` and ``load_state_dict`` run inside
    ``unrolled()``, which re-registers ``layers.i.<p>`` as views onto the rows.
    The optimizer state never goes through those hooks -- it is keyed off
    ``named_parameters()``, which stacking rewrites to
    ``layer_stacks.attention__wq__weight`` -- so the two layouts write disjoint
    key sets, which is what this test asserts against.

    The consequence is silent rather than loud:
    ``_unflatten_optim_state_dict`` treats an FQN that is absent from the
    checkpoint as "this parameter has no state" rather than as an error. So
    flipping ``--compile.use-scan-layers`` between a save and a resume
    restores every weight correctly and restarts Adam from zero for every
    layer parameter, with nothing in the logs to say so. Parameters outside
    the layers (embeddings, output, the final norm) keep their moments,
    because their names never changed.

    The fix is to re-key the flat optimizer state dict the way ``unrolled()``
    re-keys the model's: on save, split each ``[n_layers, *shape]`` moment into
    one entry per layer; on load, ``torch.stack`` them back. Both directions
    hold under sharding for the same reason the weights do -- row ``i`` of a
    ``Shard(1)`` stack is ``Shard(0)``. Watch for ``step``, which is a scalar
    rather than a stacked tensor, and for the ``param_groups.<fqn>.<key>``
    entries, which key off the group's first parameter.

    Nothing in the stack hits this today: same-layout resumes are unaffected,
    and an initial load of a foreign checkpoint is model-only by default.
    """

    @unittest.skip("TODO: known gap, see the class docstring. Remove to fix.")
    def test_the_optimizer_state_is_keyed_the_same_way_in_both_layouts(self):
        port = _free_port()
        keys = {}
        for stacked in (False, True):
            with _process_group(0, 1, port):
                model = _build(1, stacked=stacked)
                optim = torch.optim.AdamW(list(model.named_parameters()))
                init_optim_state(optim)
                keys[stacked] = set(get_flat_optim_state_dict(optim))

        self.assertEqual(keys[True], keys[False])


class FlattenedOptimizerStateTest(unittest.TestCase):
    """Optimizer state follows the live shape, and flattening changes it.

    ``flatten_for`` stores a stack whose axis 1 the mesh outgrows as
    ``[L, E*X, ...]``, and Adam's moments take that shape. Model checkpoints
    are unaffected -- ``state_dict`` views each layer back to ``[E, X, ...]`` --
    but the flat optimizer state is keyed off the live parameters, so a run
    that crosses the E-rank boundary between save and resume sees different
    shapes under the same keys. That has to fail at load, not resume on
    moments reshaped into the wrong rows.
    """

    def test_loading_across_the_flatten_boundary_raises(self):
        with tempfile.TemporaryDirectory() as path:
            # 8 ranks > 4 experts: flattened. 4 ranks: not.
            mp.spawn(
                _save_flattened_optim_worker,
                args=(8, _free_port(), path),
                nprocs=8,
            )
            # Every rank asserts it raised, so a rank that loaded fails here.
            mp.spawn(
                _load_flattened_optim_worker,
                args=(4, _free_port(), path),
                nprocs=4,
            )


if __name__ == "__main__":
    absltest.main()
