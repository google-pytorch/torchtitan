# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TokenChoiceTopKRouter's sort-free group top-2 and dense gather backward
must be bitwise equal to the topk / scatter-add versions they replace."""

import unittest
from unittest import mock

import torch
import torch.nn as nn

from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.config_utils import make_router_config
from torchtitan.models.common.moe import (
    _GatherTopKScores,
    MicrobatchWiseLoadBalanceLoss,
    TokenChoiceTopKRouter,
)


# ---- Reference copies of the replaced code ---------------------------------


def _reference_gather(scores_TE: torch.Tensor, ids_TK: torch.Tensor) -> torch.Tensor:
    """The old gather; its backward is a scatter-add into ``(T, E)``."""
    return scores_TE.gather(dim=-1, index=ids_TK)


def _reference_get_node_limited_routing_scores(
    self, scores_for_choice_TE: torch.Tensor
) -> torch.Tensor:
    """Verbatim copy of the topk(2)-based group selection."""
    if self.num_limited_groups is None:
        raise ValueError("num_limited_groups must be set when num_expert_groups is set")
    assert self.num_expert_groups is not None
    experts_per_group = self.num_experts // self.num_expert_groups
    scores_grouped = scores_for_choice_TE.unflatten(
        -1, (self.num_expert_groups, experts_per_group)
    )
    top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
    group_scores = top2_scores_in_group.sum(dim=-1)
    _, group_idx = torch.topk(
        group_scores, k=self.num_limited_groups, dim=-1, sorted=False
    )
    group_mask = torch.ones_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(-1, group_idx, False)
    return scores_grouped.masked_fill(group_mask.unsqueeze(-1), float("-inf")).flatten(
        -2
    )


def _reference_top2_sum(scores_grouped: torch.Tensor) -> torch.Tensor:
    return scores_grouped.topk(2, dim=-1).values.sum(dim=-1)


def _new_top2_sum(scores_grouped: torch.Tensor) -> torch.Tensor:
    """Runs the router's new group-score code on ``scores_grouped``.

    The group scores only feed the group top-k, so they are read back by
    capturing the input of that top-k.
    """
    captured = {}
    real_topk = torch.topk

    def spy_topk(input, k, *args, **kwargs):
        captured.setdefault("group_scores", input)
        return real_topk(input, k, *args, **kwargs)

    num_groups, experts_per_group = scores_grouped.shape[-2:]
    router = _router(
        num_experts=num_groups * experts_per_group,
        num_expert_groups=num_groups,
        num_limited_groups=1,
        top_k=1,
        score_func="sigmoid",
    )
    with mock.patch.object(torch, "topk", spy_topk):
        router._get_node_limited_routing_scores(scores_grouped.flatten(-2))
    return captured["group_scores"]


# ---- Helpers ---------------------------------------------------------------


def _router(
    *,
    num_experts: int,
    top_k: int,
    score_func: str,
    dim: int = 64,
    num_expert_groups: int | None = None,
    num_limited_groups: int | None = None,
    route_norm: bool = False,
    route_scale: float = 1.0,
    aux_loss_coeff: float | None = None,
    debug_force_load_balance: bool = False,
) -> TokenChoiceTopKRouter:
    config = make_router_config(
        dim=dim,
        num_experts=num_experts,
        gate_param_init={"weight": nn.init.normal_},
        top_k=top_k,
        score_func=score_func,
        route_norm=route_norm,
        route_scale=route_scale,
        num_expert_groups=num_expert_groups,
        num_limited_groups=num_limited_groups,
    )
    if aux_loss_coeff is not None:
        config.aux_loss = MicrobatchWiseLoadBalanceLoss.Config(coeff=aux_loss_coeff)
    config._debug_force_load_balance = debug_force_load_balance
    return config.build()


# Router configs the TPU lane runs: DeepSeek-V3 671B (and the 31-layer proxy,
# same router), DeepSeek-V3 debugmodel (with its aux loss), DeepSeek-V3 236B
# (softmax + groups, 20 experts per group), and Qwen3 MoE (no groups, softmax).
_CONFIGS = {
    "dsv3_671b": dict(
        num_experts=256,
        top_k=8,
        score_func="sigmoid",
        num_expert_groups=8,
        num_limited_groups=4,
        route_norm=True,
        route_scale=2.5,
    ),
    "dsv3_debugmodel": dict(
        num_experts=8,
        top_k=3,
        score_func="sigmoid",
        route_norm=True,
        aux_loss_coeff=1e-3,
    ),
    "dsv3_236b": dict(
        num_experts=160,
        top_k=6,
        score_func="softmax",
        num_expert_groups=8,
        num_limited_groups=3,
        route_scale=16.0,
    ),
    "qwen3_moe": dict(
        num_experts=128,
        top_k=8,
        score_func="softmax",
        route_norm=True,
    ),
}


def _inputs(num_tokens: int, dim: int, num_experts: int, dtype, ties: bool, seed: int):
    g = torch.Generator().manual_seed(seed)
    if ties:
        # Small integer logits (a sparse input and a weight in {-1, 0, 1}) and
        # a bias in {-1/4, 0, 1/4}: sigmoid/softmax scores collide everywhere,
        # both within and across expert groups.
        sparse = torch.rand(num_tokens, dim, generator=g) < 4 / dim
        x = (torch.randint(-1, 2, (num_tokens, dim), generator=g) * sparse).to(dtype)
        w = torch.randint(-1, 2, (num_experts, dim), generator=g).to(dtype)
        bias = torch.randint(-1, 2, (num_experts,), generator=g).float() / 4
    else:
        x = torch.randn(num_tokens, dim, generator=g).to(dtype)
        w = (torch.randn(num_experts, dim, generator=g) * 0.1).to(dtype)
        bias = torch.randn(num_experts, generator=g) * 0.01
    return x, w, bias


def _run(router, x, w, bias, grad_seed: int, *, reference: bool):
    """One forward + backward; returns ids, scores and the two gradients."""
    with torch.no_grad():
        router.gate.weight.copy_(w)
    router.gate.weight.grad = None
    x = x.clone().requires_grad_(True)
    patches = []
    if reference:
        patches = [
            mock.patch.object(_GatherTopKScores, "apply", _reference_gather),
            mock.patch.object(
                TokenChoiceTopKRouter,
                "_get_node_limited_routing_scores",
                _reference_get_node_limited_routing_scores,
            ),
        ]
    for p in patches:
        p.start()
    try:
        scores_TK, ids_TK, _ = router(x, bias)
        cotangent = torch.randn(
            scores_TK.shape, generator=torch.Generator().manual_seed(grad_seed)
        ).to(scores_TK.dtype)
        (scores_TK * cotangent).sum().backward()
    finally:
        for p in reversed(patches):
            p.stop()
    return ids_TK, scores_TK.detach(), x.grad, router.gate.weight.grad.clone()


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal including the sign of zero and NaN payloads."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.is_floating_point():
        int_dtype = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.view(int_dtype), b.view(int_dtype))
    return torch.equal(a, b)


# ---- Tests -----------------------------------------------------------------


class TestRouterEndToEnd(unittest.TestCase):
    """New router vs the reference copies: ids, scores, input and gate grads."""

    def setUp(self):
        # The aux loss (debugmodel) scales by the step's valid-token count.
        self._saved_denominator = AuxLoss._step_denominator
        AuxLoss.set_step_denominator(torch.tensor(96.0))

    def tearDown(self):
        AuxLoss._step_denominator = self._saved_denominator

    def _check(self, name: str, *, debug_force_load_balance: bool = False):
        cfg = _CONFIGS[name]
        dim = 64
        for dtype in (torch.float32, torch.bfloat16):
            for ties in (False, True):
                for seed in range(3):
                    with self.subTest(name=name, dtype=dtype, ties=ties, seed=seed):
                        router = _router(
                            dim=dim,
                            debug_force_load_balance=debug_force_load_balance,
                            **cfg,
                        ).to(dtype)
                        router.train()
                        x, w, bias = _inputs(
                            num_tokens=96,
                            dim=dim,
                            num_experts=cfg["num_experts"],
                            dtype=dtype,
                            ties=ties,
                            seed=seed,
                        )
                        new = _run(router, x, w, bias, seed, reference=False)
                        ref = _run(router, x, w, bias, seed, reference=True)
                        for label, a, b in zip(
                            ("ids", "scores", "grad_x", "grad_gate_weight"), new, ref
                        ):
                            self.assertTrue(
                                torch.equal(a, b) and _bitwise_equal(a, b),
                                f"{label} differs",
                            )
                        # The gradient check is not vacuous.
                        self.assertGreater(new[3].abs().sum().item(), 0)

    def test_dsv3_671b(self):
        self._check("dsv3_671b")

    def test_dsv3_debugmodel(self):
        self._check("dsv3_debugmodel")

    def test_dsv3_236b(self):
        self._check("dsv3_236b")

    def test_qwen3_moe(self):
        self._check("qwen3_moe")

    def test_debug_force_load_balance(self):
        self._check("dsv3_671b", debug_force_load_balance=True)

    def test_ties_change_group_selection_inputs(self):
        # Sanity: the tie-heavy 671B inputs really tie the top 2 of most
        # groups, and the 4th/5th group scores (the group-selection boundary)
        # of many tokens.
        cfg = _CONFIGS["dsv3_671b"]
        x, w, bias = _inputs(96, 64, cfg["num_experts"], torch.float32, True, 0)
        scores = torch.sigmoid(x @ w.T) + bias
        top2 = scores.unflatten(-1, (8, 32)).topk(2, dim=-1).values
        self.assertGreater((top2[..., 0] == top2[..., 1]).float().mean().item(), 0.5)
        group_scores = top2.sum(-1).sort(dim=-1, descending=True).values
        self.assertGreater(
            (group_scores[:, 3] == group_scores[:, 4]).float().mean().item(), 0.25
        )


class TestGroupTop2Sum(unittest.TestCase):
    def test_bitwise_equal_to_topk_sum(self):
        g = torch.Generator().manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            for experts_per_group in (2, 3, 20, 32):
                cases = {
                    "random": torch.randn(64, 8, experts_per_group, generator=g),
                    "ties": torch.randint(0, 3, (64, 8, experts_per_group), generator=g)
                    .float()
                    .div(4),
                    "all_equal": torch.full((4, 8, experts_per_group), 0.375),
                    "neg_inf": torch.where(
                        torch.rand(64, 8, experts_per_group, generator=g) < 0.5,
                        float("-inf"),
                        torch.randn(64, 8, experts_per_group, generator=g),
                    ),
                }
                for case, scores in cases.items():
                    with self.subTest(dtype=dtype, epg=experts_per_group, case=case):
                        scores = scores.to(dtype)
                        new = _new_top2_sum(scores)
                        ref = _reference_top2_sum(scores)
                        self.assertTrue(_bitwise_equal(new, ref))

    def test_group_scores_carry_no_gradient_path(self):
        # The group scores only feed the group top-k's indices, and the masked
        # scores only feed the expert top-k's indices: no gradient flows
        # through the node-limited routing to the router input.
        router = _router(**_CONFIGS["dsv3_671b"])
        scores = torch.rand(16, 256, requires_grad=True)
        ids = router._select_experts(scores, torch.zeros(256))
        self.assertFalse(ids.requires_grad)
        self.assertIsNone(ids.grad_fn)


class TestDenseGatherBackward(unittest.TestCase):
    def _check(self, scores, ids, grad):
        a = scores.clone().requires_grad_(True)
        b = scores.clone().requires_grad_(True)
        out_new = _GatherTopKScores.apply(a, ids)
        out_ref = _reference_gather(b, ids)
        self.assertTrue(_bitwise_equal(out_new, out_ref))
        out_new.backward(grad)
        out_ref.backward(grad)
        self.assertEqual(a.grad.dtype, b.grad.dtype)
        self.assertTrue(torch.equal(a.grad, b.grad))
        self.assertTrue(_bitwise_equal(a.grad, b.grad))

    def test_bitwise_equal_to_scatter_add(self):
        g = torch.Generator().manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            for num_experts, top_k in ((256, 8), (160, 6), (8, 3), (128, 8), (4, 1)):
                for case in ("random", "ties"):
                    with self.subTest(dtype=dtype, E=num_experts, K=top_k, case=case):
                        if case == "random":
                            scores = torch.randn(128, num_experts, generator=g)
                        else:
                            scores = torch.randint(
                                0, 3, (128, num_experts), generator=g
                            ).float()
                        scores = scores.to(dtype)
                        ids = scores.topk(top_k, dim=-1).indices
                        grad = torch.randn(128, top_k, generator=g).to(dtype)
                        self._check(scores, ids, grad)

    def test_special_gradient_values(self):
        # Signed zeros, infinities and NaN in the incoming gradient.
        scores = torch.randn(6, 16)
        for top_k in (1, 4):
            ids = scores.topk(top_k, dim=-1).indices
            for value in (-0.0, 0.0, float("inf"), float("-inf"), float("nan")):
                with self.subTest(top_k=top_k, value=value):
                    grad = torch.full((6, top_k), value)
                    a = scores.clone().requires_grad_(True)
                    b = scores.clone().requires_grad_(True)
                    _GatherTopKScores.apply(a, ids).backward(grad)
                    _reference_gather(b, ids).backward(grad)
                    self.assertTrue(_bitwise_equal(a.grad, b.grad))

    def test_backward_has_no_scatter(self):
        from torch.fx.experimental.proxy_tensor import make_fx

        def fwd_bwd(scores, ids, grad):
            return torch.autograd.grad(
                _GatherTopKScores.apply(scores, ids), scores, grad
            )

        scores = torch.randn(8, 32, requires_grad=True)
        ids = scores.topk(4, dim=-1).indices
        gm = make_fx(fwd_bwd)(scores, ids, torch.randn(8, 4))
        targets = {str(n.target) for n in gm.graph.nodes if n.op == "call_function"}
        self.assertFalse(any("scatter" in t for t in targets), targets)


class TestDTensorInputs(unittest.TestCase):
    """Under TP the router sees DTensors: both changes mix in a plain iota."""

    @classmethod
    def setUpClass(cls):
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh

        cls._owns_pg = not dist.is_initialized()
        if cls._owns_pg:
            dist.init_process_group(
                "gloo", store=dist.HashStore(), world_size=1, rank=0
            )
        cls.mesh = init_device_mesh("cpu", (1,))

    @classmethod
    def tearDownClass(cls):
        import torch.distributed as dist

        if cls._owns_pg:
            dist.destroy_process_group()

    def test_gather_and_group_scores(self):
        from torch.distributed.tensor import distribute_tensor, Replicate, Shard

        g = torch.Generator().manual_seed(0)
        scores = torch.randint(0, 3, (32, 256), generator=g).float().div(4)
        ids = scores.topk(8, dim=-1).indices
        grad = torch.randn(32, 8, generator=g)
        router = _router(**_CONFIGS["dsv3_671b"])
        for placement in (Shard(0), Replicate()):
            with self.subTest(placement=placement):
                scores_dt = distribute_tensor(scores, self.mesh, [placement])
                scores_dt.requires_grad_(True)
                out = _GatherTopKScores.apply(
                    scores_dt, distribute_tensor(ids, self.mesh, [placement])
                )
                out.backward(distribute_tensor(grad, self.mesh, [placement]))
                ref = scores.clone().requires_grad_(True)
                _reference_gather(ref, ids).backward(grad)
                self.assertEqual(scores_dt.grad.placements, (placement,))
                self.assertTrue(_bitwise_equal(scores_dt.grad.full_tensor(), ref.grad))

                masked = router._get_node_limited_routing_scores(
                    distribute_tensor(scores, self.mesh, [placement])
                )
                self.assertTrue(
                    _bitwise_equal(
                        masked.full_tensor(),
                        _reference_get_node_limited_routing_scores(router, scores),
                    )
                )


if __name__ == "__main__":
    unittest.main()
