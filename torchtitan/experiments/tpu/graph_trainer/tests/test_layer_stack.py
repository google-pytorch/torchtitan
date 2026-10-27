"""Stacking a decoder's layer weights must change storage and nothing else.

Every check here is against an unstacked model built the same way, because the
failure modes are silent: a stack initialized in the wrong order trains to a
plausible loss, and a dropped view shows up only as a checkpoint that no longer
loads.
"""

import copy
import dataclasses
import functools
import itertools
import unittest
from unittest.mock import patch

from absl.testing import absltest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch._functorch import config as functorch_config, partitioners
from torch._higher_order_ops.scan import ScanAutogradImpl
from torch.distributed.tensor import DTensor, Shard

from torchtitan.components.optimizer.optimizer import (
    default_adamw,
    OptimizersContainer,
)
from torchtitan.config import apply_overrides, derive, OverrideConfig
from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.graph_trainer.layer_stack import (
    check_layers_are_stackable,
    LayerStackHolder,
    STACK_ATTR,
    with_tpu_stacked_layers,
)
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import gather_slice

# Imported here so it is not first imported under the mocked `call_backward`
# below: it captures that method at import time.
from torchtitan.experiments.tpu import scan_backward_patch  # noqa: F401  pylint: disable=unused-import,g-bad-import-order
from torchtitan.experiments.tpu.layer_loop import tpu_model_class
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
from torchtitan.models import qwen3 as qwen3_models
from torchtitan.models.common import CosSinRoPE
from torchtitan.models.common.config_utils import make_aux_loss_config
from torchtitan.models.llama3 import model_registry
from torchtitan.models.qwen3 import model_registry as qwen3_registry


SEQ_LEN = 32
TOKENS = 2 * SEQ_LEN
SEED = 1234

# `scan` re-traces its body through functionalization, which has no derivative
# for `view_as_complex_copy`. The llama3 TPU recipes request this override for
# the same reason; here there is no recipe to carry it. qwen3 uses `CosSinRoPE`
# and needs nothing. Drop it once `view_as_complex_copy` has a derivative.
# https://github.com/google-pytorch/torchtitan/issues/20
_REAL_ROPE_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.real_rope"
    ".tpu_real_arithmetic_complex_rope"
)
_DISCARDED_TOKEN_COUNTER_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.discarded_token_counter"
    ".tpu_discarded_token_counter"
)


def _layer_loop(config):
    """The unstacked arm: the same model class, without the stacking."""
    return derive(config, tpu_model_class(type(config)).Config)


def _config(registry=model_registry):
    """A debug model, shortened to ``TOKENS`` so the RoPE cache stays small.

    Stock and unconverted; ``_build`` applies the rewrite. Every arm gets one,
    so that stacking is the only difference between them: a stock ``Decoder``
    loops over its layers directly and would ignore the stacks entirely.
    """
    config = registry("debugmodel", attn_backend="sdpa").model
    for layer in config.layers:
        layer.attention.rope = dataclasses.replace(
            layer.attention.rope, max_context_length=TOKENS
        )
    if registry is model_registry:
        apply_overrides(OverrideConfig(imports=[_REAL_ROPE_OVERRIDE]), config)
    return config


def _moe_config(num_experts: int):
    """A 2-layer qwen3 MoE with ``num_experts`` experts, stackable.

    Few experts, so that a small mesh already exceeds them and ``flatten_for``
    has something to flatten. Load balancing is off because stacking refuses
    it, and weight tying is off for the reason ``TestQwen3._build`` gives.
    """
    config = dataclasses.replace(
        _config(qwen3_registry), enable_weight_tying=False
    )
    config.layers = qwen3_models._build_qwen3_moe_layers(  # pylint: disable=protected-access
        fuse_qkv=True,
        n_layers=2,
        dim=config.dim,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        moe_hidden_dim=768,
        num_experts=num_experts,
        top_k=2,
        load_balance_coeff=None,
        attn_backend="sdpa",
        moe_comm_backend="standard",
        rope=CosSinRoPE.Config(
            dim=128, max_context_length=SEQ_LEN, theta=1000000.0
        ),
    )
    return config


def _build(
    *,
    stacked: bool,
    scan: bool = False,
    param_dtype: str = "float32",
    config=None,
    device: str | torch.device = "cpu",
    world_size: int = 1,
    flatten: bool = False,
):
    """Build a model the way the trainer does: convert, build, parallelize, init.

    ``stacked``/``scan`` pick the converter, exactly as
    ``TpuGraphTrainer.Config.from_tpu_trainer_config`` does -- the model arrives
    from ``build()`` already stacked, and ``apply_simple_fsdp`` is given the same
    arguments in every arm.

    ``device`` and ``world_size`` exist for ``test_layer_stack_distributed``,
    which builds the same two models on a real mesh.

    ``flatten`` runs the stack half of ``configure_qwen_simple_fsdp_shard_dims``
    before sharding, and the ``build_split_meshes`` both Qwen3 parallelize
    paths run right after it.
    """
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=world_size,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    training = TrainingConfig(
        mixed_precision_param=param_dtype, mixed_precision_reduce="float32"
    )
    if stacked or scan:
        convert = functools.partial(with_tpu_stacked_layers, use_scan=scan)
    else:
        convert = _layer_loop
    with torch.device("meta"):
        model = convert(config or _config()).build()
    holders = [m for m in model.modules() if hasattr(m, "flatten_for")]
    if flatten:
        for holder in holders:
            holder.flatten_for(world_size)
            holder._fsdp_shard_dim = holder.plan_shard_dims(world_size)  # pylint: disable=protected-access
    model = apply_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=training,
    )
    if flatten:
        # Creates process groups, so every rank runs it, before anything
        # (here `init_weights`) views a flattened layer.
        for holder in holders:
            holder.build_split_meshes()
    # `to_empty` leaves garbage, and `init_weights` does not necessarily write
    # every parameter (qwen3's `tok_embeddings`, with tying off). Garbage is
    # sometimes NaN, and NaN fails `assert_close` at any tolerance, so zero
    # everything first and let `init_weights` overwrite what it owns.
    model.to_empty(device=device)
    with torch.no_grad():
        for t in itertools.chain(model.parameters(), model.buffers()):
            t.zero_()
    torch.manual_seed(SEED)
    model.init_weights()
    return model


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
class TestLayerStack(unittest.TestCase):
    def setUp(self):
        # Same reason as `TestQwen3.setUp`: a scan body of a different shape
        # traced earlier in the process leaves an unbacked symbol behind.
        torch._dynamo.reset()
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12377",
                world_size=1,
                rank=0,
            )

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_state_dict_is_identical(self):
        """Checkpoints stay in the unrolled format, down to the bits.

        This is the check that an existing 70B checkpoint still loads.
        """
        plain = _build(stacked=False).state_dict()
        stacked = _build(stacked=True).state_dict()

        self.assertEqual(set(stacked), set(plain))
        for key, want in plain.items():
            torch.testing.assert_close(
                stacked[key].to_local(), want.to_local(), rtol=0, atol=0, msg=key
            )

    def test_init_differs_per_row(self):
        """Every layer's row is different.

        llama3 scales each layer's init by its depth. Initializing the stack in
        the wrong order -- or initializing one row and broadcasting it -- gives a
        model that still trains, so compare the rows against each other.
        """
        model = _build(stacked=True)
        state = model.state_dict()
        rows = [
            state[f"layers.{i}.attention.wo.weight"] for i in range(len(model.layers))
        ]
        for i in range(1, len(rows)):
            self.assertFalse(
                torch.equal(rows[0].to_local(), rows[i].to_local()),
                f"layer {i} was initialized identically to layer 0",
            )

    def test_module_tree_is_unchanged(self):
        """All n layer modules stay registered, so tree walkers still find them.

        The splash-attention patcher matches on class name over
        ``model.modules()``, activation checkpointing parses layer numbers out
        of FQNs, and the device move visits every module. None of them are
        touched by this change, and this is why.
        """
        plain = dict(_build(stacked=False).named_modules())
        stacked = dict(_build(stacked=True).named_modules())

        self.assertEqual(set(stacked) - set(plain), {STACK_ATTR})
        self.assertEqual(set(plain) - set(stacked), set())

    def test_each_weight_is_a_parameter_exactly_once(self):
        """Otherwise the optimizer applies every layer gradient twice.

        The row views exist only while ``init_states`` runs. If they are left
        registered, ``model.parameters()`` yields each weight both as a row and
        inside its stack.
        """
        model = _build(stacked=True)
        params = list(model.parameters())
        self.assertEqual(len({id(p) for p in params}), len(params))

        # The layer modules keep no parameters of their own, and the holder holds
        # nothing but the stacks -- one per parameter of a single layer.
        self.assertEqual(list(model.layers.parameters()), [])
        holder = getattr(model, STACK_ATTR)
        plain_layer = dict(_build(stacked=False).layers["0"].named_parameters())
        self.assertEqual(
            len(dict(holder.named_parameters(recurse=False))), len(plain_layer)
        )
        self.assertEqual(
            len(list(holder.parameters(recurse=False))), len(list(holder.parameters()))
        )

    def test_stacks_are_sharded_off_the_layer_axis(self):
        """Sharding dim 0 would split one layer's weight across ranks.

        It would also silently fall back to ``Replicate`` whenever ``n_layers``
        is below the mesh size, which is the usual case -- so pin the dim.
        """
        holder = getattr(_build(stacked=True), STACK_ATTR)
        for name, stack in holder.named_parameters(recurse=False):
            self.assertEqual(stack.placements, (Shard(1),), name)

    def test_forward_matches_unrolled(self):
        model = _build(stacked=False)
        stacked = _build(stacked=True)
        tokens = torch.randint(0, model.config.vocab_size, (TOKENS,))

        torch.testing.assert_close(
            stacked(tokens), model(tokens), rtol=0, atol=0
        )

    def test_scan_forward_matches_the_loop(self):
        plain = _build(stacked=False)
        scanned = _build(stacked=True, scan=True)
        tokens = torch.randint(0, plain.config.vocab_size, (TOKENS,))

        # Bit-exact on x86, but on arm traced scan and eager loop pick
        # different kernels and drift in the lowest bits (~1e-6).
        torch.testing.assert_close(
            scanned(tokens), plain(tokens), rtol=1e-5, atol=1e-6
        )

    def test_building_a_scan_refuses_without_the_backward_patch(self):
        """Failing to install is a warning; building a scan without it is not --
        that run reverses by copy and OOMs some steps later."""

        def not_the_method(self, *grad_fw_outputs):
            raise AssertionError("unreachable")

        with patch.object(ScanAutogradImpl, "call_backward", not_the_method):
            with self.assertRaises(RuntimeError):
                _build(stacked=True, scan=True)
            # ... and a model that never asked for scan is unaffected.
            _build(stacked=True)

    def test_the_body_recomputes_its_gather_in_backward(self):
        """The in-body gather is worth nothing unless backward redoes it.

        At the default budget the partitioner saves the gathered per-layer
        weight across the cut: the gather runs inside the loop, but its result
        is carried out of it. ``run`` opens an ``activation_memory_budget=0.0``
        window to stop that. Two things are checked here -- that the window
        reaches the body's partition, and that it closes again afterwards,
        since the config is process-global.
        """
        captured = []
        partition = partitioners.min_cut_rematerialization_partition

        def capture(joint, inputs, **kwargs):
            forward, backward = partition(joint, inputs, **kwargs)
            captured.append((functorch_config.activation_memory_budget, backward))
            return forward, backward

        model = _build(stacked=True, scan=True)
        tokens = torch.randint(0, model.config.vocab_size, (TOKENS,))
        default = functorch_config.activation_memory_budget
        with patch.object(
            partitioners, "min_cut_rematerialization_partition", capture
        ):
            model(tokens).sum().backward()

        self.assertTrue(captured, "the scan body was never partitioned")
        budget, backward = captured[0]
        self.assertEqual(budget, 0.0)
        self.assertTrue(
            any("all_gather" in str(n.target) for n in backward.graph.nodes),
            "the body's backward reuses a saved gather instead of redoing it",
        )
        self.assertEqual(functorch_config.activation_memory_budget, default)

    def test_scan_gradients_match_the_loop(self):
        """Per layer, because a severed carry chain keeps the loss exact.

        If the scan's init carry does not carry a gradient, the inter-layer
        chain is cut: the loss is unchanged, no error is raised, and only the
        layer at one end of the stack is visibly wrong.
        """
        plain = _build(stacked=False)
        scanned = _build(stacked=True, scan=True)
        tokens = torch.randint(0, plain.config.vocab_size, (TOKENS,))

        for model in (plain, scanned):
            model(tokens).sum().backward()

        # `_parameters`, not `get_parameter`: after SimpleFSDP the attribute
        # returns the gathered tensor, which has no `.grad`.
        stacks = getattr(scanned, STACK_ATTR)._parameters
        for i, layer in enumerate(plain.layers.values()):
            for name, param in layer.named_parameters():
                stack = stacks[name.replace(".", "__")]
                torch.testing.assert_close(
                    stack.grad[i].to_local(),
                    param.grad.to_local(),
                    rtol=1e-5,
                    atol=5e-5,
                    msg=f"layers.{i}.{name}",
                )

    def test_the_body_gathers_in_the_compute_dtype(self):
        """The gather is also the cast, exactly as SimpleFSDP's is.

        The stacks are stored in fp32 and the layers run in bf16; unrolled,
        SimpleFSDP's parametrization casts as it gathers. If the in-body gather
        skipped that, the scanned model would run its layers in fp32 -- slower,
        more memory, and numerically *closer* to the reference, so a loss
        comparison would not catch it.
        """
        model = _build(stacked=True, scan=True, param_dtype="bfloat16")
        # The plan is read off the installed container, so it is the one the
        # body will use. The body itself cannot be instrumented: dynamo rejects
        # any side effect inside a HOP.
        stack = model.layers.stack
        # Nothing to inspect until a forward has run, since the plan is read
        # off the sharded stacks. Asserted rather than assumed: a plan built at
        # construction would read meta tensors, find nothing sharded, and run
        # each layer on this rank's slice.
        self.assertFalse(stack.fsdp_plan, "the plan was built before a forward")
        model(torch.randint(0, model.config.vocab_size, (TOKENS,)))
        self.assertTrue(stack.fsdp_plan, "no gather plan was installed")

        holder = getattr(model, STACK_ATTR)
        for name, spec in stack.fsdp_plan.items():
            param = holder._parameters[name]
            local = param.to_local() if isinstance(param, DTensor) else param
            gathered = funcol.wait_tensor(gather_slice(local[0], spec))
            self.assertEqual(gathered.dtype, torch.bfloat16, name)
            # ... and the stored weight is untouched, so the optimizer still
            # steps in fp32.
            self.assertEqual(param.dtype, torch.float32, name)

    def test_the_optimizer_sees_each_stack_once(self):
        """Param groups are built from ``named_parameters``, which the stacks change.

        Under the default catch-all pattern the assignment must still be exactly
        the stacks plus the non-layer weights, each once. Compared by identity:
        the row views and the stacks share storage, so a name- or value-based
        check would not notice the optimizer stepping a view.
        """
        model = _build(stacked=True)
        optimizers = OptimizersContainer(default_adamw(), model_parts=[model])

        assigned = [
            p
            for opt in optimizers
            for group in opt.param_groups
            for p in group["params"]
        ]
        self.assertEqual(len({id(p) for p in assigned}), len(assigned))
        self.assertEqual(
            {id(p) for p in assigned}, {id(p) for p in model.parameters()}
        )
        holder = getattr(model, STACK_ATTR)
        self.assertTrue(
            {id(p) for p in holder.parameters()} <= {id(p) for p in assigned}
        )

    def test_loads_an_unrolled_checkpoint(self):
        """A checkpoint written by the unrolled model loads into the stacked one."""
        plain = _build(stacked=False)
        stacked = _build(stacked=True)
        tokens = torch.randint(0, plain.config.vocab_size, (TOKENS,))

        # Perturb the source so a no-op load cannot pass.
        with torch.no_grad():
            for p in plain.parameters():
                p.add_(torch.randn_like(p))
        checkpoint = copy.deepcopy(plain.state_dict())

        stacked.load_state_dict(checkpoint)
        torch.testing.assert_close(stacked(tokens), plain(tokens), rtol=0, atol=0)

    def test_refuses_load_state_dict_with_assign(self):
        """``assign=True`` would load into a tensor the stack never sees.

        It replaces the per-layer view instead of writing through it, so the
        stack keeps its old values and the loaded ones are dropped with the
        views. Silently wrong weights, so it has to raise.
        """
        stacked = _build(stacked=True)
        checkpoint = copy.deepcopy(_build(stacked=False).state_dict())

        with self.assertRaisesRegex(RuntimeError, "assign=True"):
            stacked.load_state_dict(checkpoint, assign=True)

    def test_plan_shard_dims_prefers_8_row_tile_aligned_axis_for_2d_weights(self):
        """For 3-D stacks (2-D per-layer weights), prefer an axis whose per-rank
        shard size is a multiple of 8 (TPU MXU row tile) to avoid `_fold_rows`."""
        holder = LayerStackHolder()
        with torch.device("meta"):
            # Llama3-405B at 2048 ranks: w1 [126, 53248, 16384] has 26 rows/rank
            # on dim 1 (not % 8) vs 8 cols/rank on dim 2 (% 8 == 0) -> Shard(2).
            holder.register_parameter(
                "w1", nn.Parameter(torch.empty(2, 53248, 16384))
            )
            # wqkv [126, 18432, 16384] has 9 rows/rank on dim 1 vs 8 on dim 2 -> Shard(2).
            holder.register_parameter(
                "wqkv", nn.Parameter(torch.empty(2, 18432, 16384))
            )
            # w2 [126, 16384, 53248] has 8 rows/rank on dim 1 (% 8 == 0) -> Shard(1).
            holder.register_parameter(
                "w2", nn.Parameter(torch.empty(2, 16384, 53248))
            )
            # 4-D MoE stack [2, 128, 1536, 4096] at 128 ranks keeps dim 1 (num_experts).
            holder.register_parameter(
                "experts_w1", nn.Parameter(torch.empty(2, 128, 1536, 4096))
            )

        self.assertEqual(
            holder.plan_shard_dims(2048),
            {"w1": 2, "wqkv": 2, "w2": 1, "experts_w1": 3},
        )
        self.assertEqual(
            holder.plan_shard_dims(128)["experts_w1"],
            1,
        )


def _toy_layer(dim=4, extra=None, out=None, cache=None, frozen=False):
    layer = nn.Module()
    layer.wq = nn.Linear(dim, out if out is not None else dim, bias=False)
    layer.wq.weight.requires_grad_(not frozen)
    if extra is not None:
        layer.wextra = nn.Linear(dim, extra, bias=False)
    if cache is not None:
        layer.register_buffer("cache", torch.as_tensor(cache))
    return layer


def _toy_layers(*mods):
    return nn.ModuleDict({str(i): m for i, m in enumerate(mods)})


class TestStackable(unittest.TestCase):
    """Every rejection here would otherwise be a silently wrong run.

    A layer that does not match layer 0 still trains: its weights are cast into
    layer 0's stack, or it runs with layer 0's buffers. So each case checks the
    message names the layer and the thing that differs.
    """

    def test_identical_layers_are_stackable(self):
        check_layers_are_stackable(_toy_layers(*[_toy_layer() for _ in range(3)]))
        check_layers_are_stackable(
            _toy_layers(_toy_layer(cache=[1.0]), _toy_layer(cache=[1.0]))
        )

    def test_a_differing_parameter_set_names_the_layer_and_the_parameter(self):
        # Both directions -- extra and missing.
        for label, layers in (
            (
                "extra",
                _toy_layers(_toy_layer(), _toy_layer(extra=8), _toy_layer(extra=16)),
            ),
            ("missing", _toy_layers(_toy_layer(extra=8), _toy_layer(), _toy_layer())),
        ):
            with self.subTest(label):
                with self.assertRaises(ValueError) as caught:
                    check_layers_are_stackable(layers)
                self.assertIn("layer 1", str(caught.exception))
                self.assertIn("wextra.weight", str(caught.exception))

    def test_a_same_named_parameter_shows_what_differs(self):
        # Dtype and requires_grad are the quiet ones: the stack is allocated
        # from layer 0, so a layer that disagrees is folded into it in silence.
        odd_dtype = _toy_layer()
        odd_dtype.wq.to(torch.float64)
        for expected, layers in (
            (["(4, 4)", "(6, 4)"], _toy_layers(_toy_layer(), _toy_layer(out=6))),
            (["float64"], _toy_layers(_toy_layer(), odd_dtype)),
            (["False"], _toy_layers(_toy_layer(), _toy_layer(frozen=True))),
        ):
            with self.subTest(expected):
                with self.assertRaises(ValueError) as caught:
                    check_layers_are_stackable(layers)
                message = str(caught.exception)
                self.assertIn("wq.weight", message)
                for fragment in expected:
                    self.assertIn(fragment, message)

    def test_a_differing_buffer_is_rejected(self):
        # Same name, shape and dtype: nothing structural to notice, and every
        # layer would run with layer 0's value.
        for label, layers in (
            ("values", _toy_layers(_toy_layer(cache=[1.0]), _toy_layer(cache=[2.0]))),
            ("absent", _toy_layers(_toy_layer(cache=[1.0]), _toy_layer())),
        ):
            with self.subTest(label):
                with self.assertRaises(ValueError) as caught:
                    check_layers_are_stackable(layers)
                message = str(caught.exception)
                self.assertIn("layer 1", message)
                self.assertIn("cache", message)

    def test_meta_buffers_are_compared_by_shape_only(self):
        # No values to compare yet; refusing here would reject every model.
        with torch.device("meta"):
            check_layers_are_stackable(
                _toy_layers(_toy_layer(cache=[1.0]), _toy_layer(cache=[2.0]))
            )

    def _moe_model(self, load_balance_coeff: float | None):
        config = qwen3_registry("debugmodel_moe", attn_backend="sdpa").model
        for layer in config.layers:
            layer.moe.load_balance_coeff = load_balance_coeff
        with torch.device("meta"):
            return config.build()

    def test_moe_layers_are_refused(self):
        """An MoE's load-balancing buffers are per-layer, and buffers aren't
        stacked, so every layer would route off layer 0's."""
        with self.assertRaisesRegex(ValueError, "MoE"):
            check_layers_are_stackable(self._moe_model(1e-3).layers)

    def test_moe_layers_without_load_balancing_are_stackable(self):
        """With no coefficient there is no `expert_bias_E`, and nothing reads
        `tokens_per_expert_E`, so sharing layer 0's buffers changes nothing."""
        check_layers_are_stackable(self._moe_model(None).layers)


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
class TestQwen3(unittest.TestCase):
    """The same two properties on a second model, to pin that none of this is llama3.

    Stacking reads the layers' own parameters and never their names, so a model
    with a different attention shape, a different norm placement and its own
    RoPE should need no code. That is a claim about the implementation, and the
    only way to keep it true is to check it against something that is not
    llama3.
    """

    def setUp(self):
        # `scan` traces its body through dynamo. Tracing a *differently shaped*
        # body in the same process reuses the cached shape environment and
        # fails on an unbacked symbol. Both directions break, so both classes
        # reset.
        torch._dynamo.reset()
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12378",
                world_size=1,
                rank=0,
            )

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _build(self, **kwargs):
        # Weight tying off, and not because of stacking: `init_states` re-ties
        # by assigning `lm_head.weight` to `tok_embeddings.weight`, and under
        # SimpleFSDP that attribute is a parametrization returning a plain
        # gathered tensor, which `nn.Module.__setattr__` refuses. The unrolled
        # arm fails identically, so it is a gap in this lane, not in the stack.
        #
        # With tying off nothing initializes `tok_embeddings` -- `init_states`
        # was its only writer, and `skip_param_init` skips it -- so it holds
        # whatever `to_empty` left. Hence `_pair` below, and the restriction of
        # the state-dict comparison to the decoder.
        config = dataclasses.replace(
            _config(qwen3_registry), enable_weight_tying=False
        )
        return _build(config=config, **kwargs)

    def _pair(self, **kwargs):
        """A plain model and a stacked one holding exactly its weights."""
        plain = self._build(stacked=False)
        other = self._build(**kwargs)
        other.load_state_dict(copy.deepcopy(plain.state_dict()))
        return plain, other

    def test_layer_weights_are_identical(self):
        plain = self._build(stacked=False).state_dict()
        stacked = self._build(stacked=True).state_dict()

        self.assertEqual(set(stacked), set(plain))
        layer_keys = [k for k in plain if k.startswith("layers.")]
        self.assertTrue(layer_keys)
        for key in layer_keys:
            torch.testing.assert_close(
                stacked[key].to_local(),
                plain[key].to_local(),
                rtol=0,
                atol=0,
                msg=key,
            )

    def test_scan_forward_matches_the_loop(self):
        plain, scanned = self._pair(stacked=True, scan=True)
        tokens = torch.randint(0, plain.config.vocab_size, (TOKENS,))

        # Bit-exact on x86, but on arm traced scan and eager loop pick
        # different kernels and drift in the lowest bits (~1e-6).
        torch.testing.assert_close(
            scanned(tokens), plain(tokens), rtol=1e-5, atol=1e-6
        )

    def _moe_config_with_global_batch_aux_loss(self, coeff):
        config = qwen3_registry("debugmodel_moe", attn_backend="sdpa").model
        config = dataclasses.replace(config, enable_weight_tying=False)
        for layer in config.layers:
            layer.attention.rope = dataclasses.replace(
                layer.attention.rope, max_context_length=TOKENS
            )
            layer.moe.load_balance_coeff = None
            if coeff is not None:
                aux_loss = make_aux_loss_config(
                    type="batch_wise", coeff=coeff, top_k=layer.moe.router.top_k
                )
                aux_loss.global_batch_size = TOKENS // SEQ_LEN
                aux_loss.seq_len = SEQ_LEN
                layer.moe.router.aux_loss = aux_loss
        # As in the 235B scan recipe: no `tokens_per_expert_E` write either.
        apply_overrides(
            OverrideConfig(imports=[_DISCARDED_TOKEN_COUNTER_OVERRIDE]), config
        )
        return config

    def test_scan_gradients_match_the_loop_with_global_batch_aux_loss(self):
        """The fork's router aux loss (the 235B recipes') runs inside a scan.

        It must not write a buffer in the forward, which the scan body cannot
        do, and its injected gradient must reach every layer's router as it
        does unrolled. The routed experts run in bf16, so scan and loop agree
        only to a few ulps; the check is that the aux-loss gradient, which
        is far larger than that, is carried through the scan.
        """
        # Large enough that the aux loss dominates the gate gradient of the
        # `output.sum()` loss, so a dropped injection cannot hide in the noise.
        aux_coeff = 100.0
        plain = _build(
            stacked=False,
            config=self._moe_config_with_global_batch_aux_loss(aux_coeff),
        )
        scanned = _build(
            stacked=True,
            scan=True,
            config=self._moe_config_with_global_batch_aux_loss(aux_coeff),
        )
        no_aux = _build(
            stacked=False,
            config=self._moe_config_with_global_batch_aux_loss(None),
        )
        scanned.load_state_dict(copy.deepcopy(plain.state_dict()))
        no_aux.load_state_dict(copy.deepcopy(plain.state_dict()))
        tokens = torch.randint(0, plain.config.vocab_size, (TOKENS,))

        with set_unflatten_seq_len(SEQ_LEN):
            outputs = [model(tokens) for model in (plain, scanned, no_aux)]
            for output in outputs:
                output.sum().backward()
        # Relative Frobenius norm, not elementwise: the bf16 experts leave
        # scattered few-ulp differences (large relative error on near-zero
        # logits; up to ~1e-3 in norm depending on the host's CPU kernels),
        # while a broken scan is off by O(1).
        output_error = ((outputs[1] - outputs[0]).norm() / outputs[0].norm()).item()
        self.assertLess(output_error, 1e-2)

        stacks = getattr(scanned, STACK_ATTR)._parameters
        gate = "moe.router.gate.weight"
        scan_error, aux_effect = [], []
        for i, (layer, no_aux_layer) in enumerate(
            zip(plain.layers.values(), no_aux.layers.values(), strict=True)
        ):
            want = dict(layer.named_parameters())[gate].grad.to_local()
            got = stacks[gate.replace(".", "__")].grad[i].to_local()
            reference = dict(no_aux_layer.named_parameters())[gate]
            reference = reference.grad.to_local()
            scan_error.append(((got - want).norm() / want.norm()).item())
            aux_effect.append(((want - reference).norm() / want.norm()).item())
        # Per layer, relative Frobenius norms: scan-vs-loop error, and how much
        # the aux loss changes the gate gradient. Scan error is ~6e-4 on most
        # hosts and up to ~2e-2 on some (host-dependent CPU kernels); the aux
        # effect is >= 0.2, so a scan that dropped the injected gradient
        # (scan error == aux effect) fails the first bound.
        summary = f"scan error {scan_error}, aux effect {aux_effect}"
        self.assertLess(max(scan_error), 5e-2, summary)
        self.assertGreater(min(aux_effect), 3 * max(scan_error), summary)


if __name__ == "__main__":
    absltest.main()
