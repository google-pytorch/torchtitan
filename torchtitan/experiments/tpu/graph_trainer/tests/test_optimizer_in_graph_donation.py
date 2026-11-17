"""The in-graph optimizer must donate its parameter and moment inputs.

Without donation the step program takes the sharded fp32 parameters and both
AdamW moments as inputs and returns fresh copies as outputs, and XLA keeps both
sets resident for the whole step: 35.9 GiB in + 35.4 GiB out per TensorCore on
llama3-405B at 128-way FSDP, more than the whole activation stack the scan
offloads to host. With ``jax.buffer_donor`` on those inputs XLA aliases each to
its output (``input_output_alias``) and the second copy disappears.

Three things have to hold, each checked here on its own:

1. ``adamw_donated_input_indices`` picks exactly the optimizer-owned parameters and
   the ``_optstate_*_exp_avg`` / ``_exp_avg_sq`` buffers, in graph-input index
   space (a DTensor parameter widens to its local tensors).
2. The indices reach torch_tpu: the compiled StableHLO carries
   ``jax.buffer_donor = true`` on those arguments and no others.
3. The runtime contract the trainer now relies on: a donated input is consumed
   by the execution, so the module's parameter and moment entries must be
   replaced by the outputs (``rebind_model_and_optstate_buffers``) rather than
   written back with ``copy_``, and the next step runs on the rebound tensors.
   The optimizer's ``param_groups`` follow, since checkpointing steps them.
4. The trainer's own step function (``_make_fwd_bwd_adamw_fn``), three steps
   with the lr changed in between, is bit-identical with and without donation,
   including a tied parameter, which is passed in two argument slots and must
   therefore not be donated (PJRT rejects ``f(donate(a), a)``).
"""

import re
import unittest

from absl.testing import absltest
import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx

# Imported as a module so pytest does not collect the base class itself.
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    SubclassLayout,
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.components.checkpointer.base import ModelWrapper
from torchtitan.experiments.tpu.compiled_optimizers.functional import (
    adamw_donated_input_indices,
    aliased_state_fqns,
    rebind_model_and_optstate_buffers,
    update_model_and_optstate_buffers,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
    tpu_backend_compile_pass,
)


def _tpu_available() -> bool:
    # `torch.tpu.device_count` became TorchTPU-internal (raises when called from
    # user code), which made the previous check skip these tests everywhere.
    try:
        accelerator = torch.accelerator.current_accelerator()
        return accelerator is not None and accelerator.type == "tpu" and torch.accelerator.device_count() > 0
    except Exception:  # torch_tpu not loaded, or no chip visible
        return False


# --- 1. index selection -----------------------------------------------------


class DonatedInputIndicesTest(unittest.TestCase):
    """Mirrors the naming convention of ``_register_optimizer_buffers``."""

    def test_selects_owned_params_and_both_moments_only(self):
        state_fqns = [
            "tok_embeddings.weight",  # 0: owned (its moments are 1, 2)
            "tok_embeddings._optstate_weight_exp_avg",  # 1: moment
            "tok_embeddings._optstate_weight_exp_avg_sq",  # 2: moment
            "layers.w1.weight",  # 3: owned (its moments are 4, 5)
            "layers.w1._optstate_weight_exp_avg",  # 4: moment
            "layers.w1._optstate_weight_exp_avg_sq",  # 5: moment
            "layers.frozen.weight",  # 6: requires_grad=False -> no moments -> keep
            "norm.scale",  # 7: plain buffer -> keep
            "_optstate_step",  # 8: scalar state, copied back, never donated
            "_optstate_lr",  # 9: scalar state, refilled each step, never donated
        ]
        self.assertEqual(adamw_donated_input_indices(state_fqns, {}), [0, 1, 2, 3, 4, 5])

    def test_root_level_param_uses_root_level_moments(self):
        state_fqns = ["weight", "_optstate_weight_exp_avg", "_optstate_weight_exp_avg_sq", "_optstate_step"]
        self.assertEqual(adamw_donated_input_indices(state_fqns, {}), [0, 1, 2])

    def test_dtensor_state_widens_to_its_local_tensors(self):
        state_fqns = [
            "a.weight",  # 0 -> graph inputs 0,1 (a 2-tensor subclass)
            "a._optstate_weight_exp_avg",  # 1 -> graph input 2
            "a._optstate_weight_exp_avg_sq",  # 2 -> graph input 3
            "b.bias",  # 3 -> graph input 4, not owned
            "b._optstate_bias_exp_avg",  # 4 -> graph input 5
            "b._optstate_bias_exp_avg_sq",  # 5 -> graph inputs 6,7
        ]
        layouts = {0: SubclassLayout(2, None), 5: SubclassLayout(2, None)}
        # b.bias IS owned (its exp_avg sibling exists) -> 4 is donated too.
        self.assertEqual(
            adamw_donated_input_indices(state_fqns, layouts), [0, 1, 2, 3, 4, 5, 6, 7]
        )

    def test_nothing_to_donate_without_in_graph_optimizer_state(self):
        self.assertEqual(adamw_donated_input_indices(["layers.0.w", "norm.scale"], {}), [])

    def test_aliased_state_is_never_donated(self):
        # The tracer keeps one graph input per FQN, so a tied parameter is the
        # same buffer in slots 0 and 1. Names alone cannot tell; the trainer
        # passes the FQNs that aliased_state_fqns finds on the real module.
        state_fqns = [
            "emb.weight",  # 0: owns the moments
            "out.weight",  # 1: alias of 0, same buffer
            "emb._optstate_weight_exp_avg",  # 2
            "emb._optstate_weight_exp_avg_sq",  # 3
        ]
        self.assertEqual(adamw_donated_input_indices(state_fqns, {}), [0, 2, 3])
        self.assertEqual(
            adamw_donated_input_indices(
                state_fqns, {}, aliased_fqns={"emb.weight", "out.weight"}
            ),
            [2, 3],
        )


class AliasedStateFqnsTest(unittest.TestCase):
    def test_tied_parameter_aliases_but_its_moments_do_not(self):
        model = nn.Module()
        model.emb = _Leaf()
        model.out = nn.Module()
        model.out.weight = model.emb.weight
        self.assertEqual(aliased_state_fqns(model), {"emb.weight", "out.weight"})

    def test_shared_submodule_donates_nothing_of_its_state(self):
        # A submodule registered twice exposes its parameter AND its optimizer
        # buffers under both prefixes; each sits in two argument slots.
        model = nn.Module()
        model.a = _Leaf()
        model.b = model.a
        aliased = aliased_state_fqns(model)
        state_fqns = [name for name, _ in model.named_parameters(remove_duplicate=False)]
        state_fqns += [name for name, _ in model.named_buffers(remove_duplicate=False)]
        self.assertEqual(aliased, set(state_fqns))
        self.assertEqual(
            adamw_donated_input_indices(state_fqns, {}, aliased_fqns=aliased), []
        )

    def test_buffer_viewing_parameter_storage_is_aliased(self):
        # Distinct tensor objects over one storage still reach the step
        # program through two argument slots.
        model = nn.Module()
        model.leaf = _Leaf()
        model.register_buffer("mirror", model.leaf.weight.detach())
        self.assertEqual(aliased_state_fqns(model), {"leaf.weight", "mirror"})

    def test_wrapper_subclass_leaves_are_compared_recursively(self):
        # A wrapper (DTensor, possibly around another wrapper) has no storage
        # of its own; aliasing is decided on its inner tensors at any depth.
        from torch.testing._internal.two_tensor import TwoTensor

        model = nn.Module()
        model.leaf = _Leaf()
        inner = TwoTensor(model.leaf.weight.detach(), torch.zeros(4))
        model.register_buffer("nested", TwoTensor(inner, torch.zeros(4)))
        model.register_buffer("separate", TwoTensor(torch.zeros(4), torch.zeros(4)))
        self.assertEqual(aliased_state_fqns(model), {"leaf.weight", "nested"})

    def test_unshared_module_has_no_aliases(self):
        model = nn.Module()
        model.a = _Leaf()
        model.b = _Leaf()
        self.assertEqual(aliased_state_fqns(model), set())


# --- 3a. rebind semantics (device independent) -----------------------------


class _Leaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))
        self.register_buffer("_optstate_weight_exp_avg", torch.zeros(4))
        self.register_buffer("_optstate_weight_exp_avg_sq", torch.zeros(4))


class RebindTest(unittest.TestCase):
    def test_rebind_replaces_entries_with_the_outputs_without_copy(self):
        model = nn.Module()
        model.leaf = _Leaf()
        new_p, new_ea, new_eas = torch.ones(4), torch.full((4,), 2.0), torch.full((4,), 3.0)

        rebind_model_and_optstate_buffers(
            model, ["leaf.weight"], [new_p], [new_ea], [new_eas]
        )

        # The module now hands out a Parameter backed by the output's storage:
        # no copy, and the state extraction the next step performs sees it.
        self.assertIsInstance(model.leaf.weight, nn.Parameter)
        self.assertEqual(model.leaf.weight.data_ptr(), new_p.data_ptr())
        self.assertEqual(model.leaf._optstate_weight_exp_avg.data_ptr(), new_ea.data_ptr())
        self.assertEqual(model.leaf._optstate_weight_exp_avg_sq.data_ptr(), new_eas.data_ptr())
        self.assertEqual(dict(model.named_parameters())["leaf.weight"].data_ptr(), new_p.data_ptr())
        self.assertEqual(dict(model.named_buffers())["leaf._optstate_weight_exp_avg"].data_ptr(), new_ea.data_ptr())
        # Still a leaf that requires grad, as an optimizer-owned parameter must.
        self.assertTrue(model.leaf.weight.requires_grad)
        self.assertTrue(model.leaf.weight.is_leaf)

    def test_tied_parameters_stay_tied(self):
        # A tied parameter is not donated (it sits in two argument slots), so
        # it is updated in place and both FQNs keep sharing it.
        model = nn.Module()
        model.leaf = _Leaf()
        model.head = nn.Module()
        model.head.weight = model.leaf.weight  # tied: one Parameter, two FQNs
        before = model.leaf.weight

        rebind_model_and_optstate_buffers(
            model,
            ["leaf.weight"],
            [torch.ones(4)],
            [torch.zeros(4)],
            [torch.zeros(4)],
            aliased_fqns=aliased_state_fqns(model),
        )

        self.assertIs(model.leaf.weight, before)
        self.assertIs(model.head.weight, model.leaf.weight)
        torch.testing.assert_close(model.head.weight.detach(), torch.ones(4))

    def test_buffer_aliasing_a_parameter_keeps_seeing_its_values(self):
        # The same object registered as a parameter and as a buffer: not
        # donated, updated in place, so the buffer is never left stale.
        model = nn.Module()
        model.leaf = _Leaf()
        model.register_buffer("mirror", model.leaf.weight)

        rebind_model_and_optstate_buffers(
            model,
            ["leaf.weight"],
            [torch.ones(4)],
            [torch.zeros(4)],
            [torch.zeros(4)],
            aliased_fqns=aliased_state_fqns(model),
        )

        torch.testing.assert_close(model.mirror.detach(), torch.ones(4))

    def test_optimizer_param_groups_follow_the_rebind(self):
        # Checkpointing runs `init_optim_state`, a zero-lr step over the
        # optimizer's param_groups; after a donated step those must be the
        # live Parameters, not the consumed inputs.
        from torchtitan.components.optimizer.utils import init_optim_state

        model = nn.Module()
        model.leaf = _Leaf()
        opt = torch.optim.AdamW([("leaf.weight", model.leaf.weight)], lr=1e-3)
        old = model.leaf.weight

        rebind_model_and_optstate_buffers(
            model, ["leaf.weight"], [torch.ones(4)], [torch.zeros(4)], [torch.zeros(4)],
            optimizers=[opt],
        )

        (group,) = opt.param_groups
        self.assertIs(group["params"][0], model.leaf.weight)
        self.assertIsNot(group["params"][0], old)
        init_optim_state(opt)
        self.assertIn(model.leaf.weight, opt.state)
        self.assertNotIn(old, opt.state)

    def test_cleared_checkpoint_cache_follows_the_rebind(self):
        # ModelWrapper caches state_dict tensors at construction and copy_()s
        # into them on save. After a rebind the cached entries still view the
        # consumed inputs; the trainer clears the cache so the next save
        # rebuilds it from the live module instead of writing into them.
        model = nn.Module()
        model.leaf = _Leaf()
        wrapper = ModelWrapper(model)
        stale = wrapper.cached_state_dict["leaf.weight"]
        new_p = torch.ones(4)

        rebind_model_and_optstate_buffers(
            model, ["leaf.weight"], [new_p], [torch.zeros(4)], [torch.zeros(4)]
        )
        wrapper.cached_state_dict.clear()
        state = wrapper.state_dict()

        self.assertEqual(state["leaf.weight"].data_ptr(), new_p.data_ptr())
        self.assertNotEqual(state["leaf.weight"].data_ptr(), stale.data_ptr())
        torch.testing.assert_close(stale, torch.zeros(4))  # never written into


# --- 2 + 3b. the torch_tpu contract, on a chip ------------------------------


def _adamw_like(p, m, g):
    """A stand-in for one in-graph optimizer update: returns fresh p and m."""
    new_m = m * 0.9 + g * 0.1
    new_p = p - new_m * 0.5
    return new_p, new_m


# `BaseAcceleratorDeviceTest` is what `.ci/discover_tpu_tests.py` keys on to run
# a module in the single-TPU shard; a plain TestCase lands in the CPU shard,
# where these would always skip.
@unittest.skipUnless(_tpu_available(), "needs a TPU device")
class DonationOnDeviceTest(base_device_test.BaseAcceleratorDeviceTest):
    @classmethod
    def setUpClass(cls):
        # torch_tpu's compile passes read absl flags; the trainer runs under
        # absl.app, a pytest process does not, so mark them parsed here.
        from absl import flags

        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["test"])
        super().setUpClass()

    def _graph(self):
        cpu = [torch.randn(8, 16) for _ in range(3)]
        gm = make_fx(_adamw_like)(*cpu)
        return gm, cpu

    def test_donated_arguments_are_marked_in_stablehlo(self):
        from torch_tpu._internal.compile.compiler import StaticCompiler

        gm, cpu = self._graph()
        tpu = [t.to("tpu") for t in cpu]
        executable = StaticCompiler(debug=True)(gm, tpu, donated_inputs=[0, 1])

        donors = re.findall(r"%arg(\d+): tensor<[^>]*> \{[^}]*jax\.buffer_donor = true", executable.mlir_text)
        self.assertEqual(sorted(int(d) for d in donors), [0, 1])
        self.assertEqual(executable.mlir_text.count("jax.buffer_donor"), 2)

    def test_compile_pass_donates_and_rebound_step_stays_correct(self):
        gm, cpu = self._graph()
        p, m, g = (t.to("tpu") for t in cpu)
        p_in, m_in = p, m

        gm = tpu_backend_compile_pass(gm, (p, m, g), donated_input_positions=(0, 1))
        new_p, new_m = gm(p, m, g)

        exp_p, exp_m = _adamw_like(*cpu)
        torch.testing.assert_close(new_p.cpu(), exp_p)
        torch.testing.assert_close(new_m.cpu(), exp_m)
        # The donated inputs are gone; the non-donated one is untouched.
        with self.assertRaisesRegex(RuntimeError, "deleted or donated|donated or deleted"):
            p_in.cpu()
        with self.assertRaisesRegex(RuntimeError, "deleted or donated|donated or deleted"):
            m_in.cpu()
        torch.testing.assert_close(g.cpu(), cpu[2])

        # What the trainer does between steps: rebind, then run on the rebound tensors.
        model = nn.Module()
        model.leaf = _Leaf()
        model.leaf.weight = nn.Parameter(p_in, requires_grad=False)
        model.leaf._optstate_weight_exp_avg = m_in
        model.leaf._optstate_weight_exp_avg_sq = torch.zeros_like(g)
        rebind_model_and_optstate_buffers(
            model, ["leaf.weight"], [new_p], [new_m], [torch.zeros_like(new_m)]
        )
        self.assertEqual(model.leaf.weight.data_ptr(), new_p.data_ptr())

        new_p2, new_m2 = gm(model.leaf.weight, model.leaf._optstate_weight_exp_avg, g)
        exp_p2, exp_m2 = _adamw_like(exp_p, exp_m, cpu[2])
        torch.testing.assert_close(new_p2.cpu(), exp_p2)
        torch.testing.assert_close(new_m2.cpu(), exp_m2)


# --- 4. the trainer's own step function, three steps, with and without donation


def _register_adamw_buffers(model: nn.Module, lr: float):
    """What ``TpuGraphTrainer._register_optimizer_buffers`` does, on a plain module."""
    seen: dict[int, str] = {}
    fqns: list[str] = []
    fqn_to_canonical: dict[str, str] = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        if id(p) in seen:
            fqn_to_canonical[name] = seen[id(p)]
            continue
        seen[id(p)] = name
        fqn_to_canonical[name] = name
        fqns.append(name)
        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        parent.register_buffer(f"_optstate_{leaf}_exp_avg", torch.zeros_like(p), persistent=True)
        parent.register_buffer(f"_optstate_{leaf}_exp_avg_sq", torch.zeros_like(p), persistent=True)
    device = next(model.parameters()).device
    model.register_buffer("_optstate_step", torch.zeros((), dtype=torch.float32, device=device), persistent=True)
    model.register_buffer("_optstate_lr", torch.tensor(lr, dtype=torch.float32, device=device), persistent=False)
    return fqns, fqn_to_canonical


class _Tiny(nn.Module):
    def __init__(self, tie: bool = False):
        super().__init__()
        self.emb = nn.Embedding(16, 8)
        self.hidden = nn.Linear(8, 8)
        self.out = nn.Linear(8, 16, bias=False)
        if tie:
            self.out.weight = self.emb.weight

    def forward(self, tokens):
        return self.out(torch.relu(self.hidden(self.emb(tokens))))


def _mse(pred, labels, global_valid_tokens=None):
    return ((pred.float() - labels) ** 2).mean()


@unittest.skipUnless(_tpu_available(), "needs a TPU device")
class InGraphAdamWStepOnDeviceTest(base_device_test.BaseAcceleratorDeviceTest):
    """Runs ``_make_fwd_bwd_adamw_fn`` the way ``_fwd_bwd_adamw_step`` does."""

    @classmethod
    def setUpClass(cls):
        from absl import flags

        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["test"])
        super().setUpClass()

    def _train(self, *, donate: bool, tie: bool, lrs=(1e-2, 3e-2, 1e-2)):
        from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
            _make_fwd_bwd_adamw_fn,
        )

        torch.manual_seed(0)
        model = _Tiny(tie=tie).to("tpu")
        if tie:
            model.out.weight = model.emb.weight  # Module.to() re-creates Parameters and unties
        fqns, _ = _register_adamw_buffers(model, lrs[0])
        opt = torch.optim.AdamW(
            [(n, p) for n, p in model.named_parameters()], lr=lrs[0], weight_decay=0.1
        )
        tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="tpu")
        labels = torch.full((2, 4, 16), 0.5, device="tpu")
        valid = torch.tensor(8.0, device="tpu")

        step_fn = _make_fwd_bwd_adamw_fn(
            model, loss_fn=_mse, param_fqns=fqns,
            beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, max_norm=1.0,
        )
        traced = minimal_fx_tracer(step_fn, module=model)(tokens, labels, valid, {})
        donated = (
            adamw_donated_input_indices(
                traced.state_fqns,
                traced.input_subclass_layouts,
                aliased_fqns=aliased_state_fqns(model),
            )
            if donate else []
        )
        traced.gm = tpu_backend_compile_pass(
            traced.gm, tuple(traced.example_inputs), donated_input_positions=tuple(donated)
        )

        losses = []
        first_inputs = [dict(model.named_parameters())[f] for f in fqns]
        n = len(fqns)
        for lr in lrs:
            model._optstate_lr.fill_(lr)
            outs = run_traced(traced, module=model)(tokens, labels, valid, {})
            losses.append(float(outs[0].cpu()))
            new_p, new_ea, new_eas = outs[1 : 1 + n], outs[1 + n : 1 + 2 * n], outs[1 + 2 * n : 1 + 3 * n]
            if donate:
                rebind_model_and_optstate_buffers(
                    model, fqns, new_p, new_ea, new_eas,
                    aliased_fqns=aliased_state_fqns(model), optimizers=[opt],
                )
            else:
                update_model_and_optstate_buffers(model, fqns, new_p, new_ea, new_eas)
            model._optstate_step.copy_(outs[1 + 3 * n])

        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return losses, state, first_inputs, donated, opt, model

    def test_three_donated_steps_match_the_undonated_program(self):
        ref_losses, ref_state, _, _, _, _ = self._train(donate=False, tie=False)
        losses, state, first_inputs, donated, opt, model = self._train(donate=True, tie=False)

        self.assertTrue(donated, "nothing was donated")
        self.assertEqual(losses, ref_losses)
        self.assertEqual(set(state), set(ref_state))
        for key in ref_state:
            torch.testing.assert_close(state[key], ref_state[key], rtol=0, atol=0, msg=key)
        self.assertEqual(float(state["_optstate_step"]), 3.0)
        # The step-1 inputs were consumed; the optimizer follows the live Parameters.
        with self.assertRaisesRegex(RuntimeError, "deleted or donated|donated or deleted"):
            first_inputs[0].cpu()
        live = {id(p) for p in model.parameters()}
        self.assertTrue(all(id(p) in live for g in opt.param_groups for p in g["params"]))

    def test_tied_parameter_trains_with_donation(self):
        # Before the tied exclusion this died in PJRT: "attempt to use a buffer
        # that was previously donated in the same call to Execute()".
        ref_losses, ref_state, _, _, _, _ = self._train(donate=False, tie=True)
        losses, state, _, donated, _, model = self._train(donate=True, tie=True)

        self.assertTrue(donated)
        self.assertEqual(losses, ref_losses)
        for key in ref_state:
            torch.testing.assert_close(state[key], ref_state[key], rtol=0, atol=0, msg=key)
        self.assertIs(model.out.weight, model.emb.weight)


if __name__ == "__main__":
    absltest.main()
