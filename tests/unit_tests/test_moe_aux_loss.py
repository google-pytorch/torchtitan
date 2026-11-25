# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from types import SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from torch.nn import functional as F

from torchtitan.components.metrics import TensorBoardLogger
from torchtitan.components.optimizer.optimizer import (
    collect_moe_routing_metric_tensors,
    configure_moe_routing_metrics,
    get_moe_load_balancing_metrics,
    set_moe_routing_metric_tensors,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step
from torchtitan.models.common import moe as moe_module
from torchtitan.models.common.aux_loss import _AuxLossInjection, AuxLoss
from torchtitan.models.common.config_utils import (
    make_aux_loss_config,
    update_moe_aux_loss_configs,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import (
    _batch_wise_load_balance_loss,
    _sequence_wise_load_balance_loss,
    GroupedExperts,
    MoE,
    MoELoadBalanceAuxLoss,
    RoutedExperts,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher


def _routing_map(topk_expert_ids, num_experts):
    """One-hot routing map ``[..., E]`` from top-k expert ids ``[..., K]``."""
    return F.one_hot(topk_expert_ids, num_classes=num_experts).sum(dim=-2).bool()


@pytest.mark.parametrize(
    "loss_fn",
    [_sequence_wise_load_balance_loss, _batch_wise_load_balance_loss],
)
def test_aux_loss_injection_matches_explicit_loss(loss_fn):
    """
    Verify aux-loss injection matches adding the aux loss to the main loss.

    _AuxLossInjection forwards top-k scores unchanged, then injects a unit
    gradient for the precomputed aux loss during backward. This should produce
    the same score gradients as the ordinary autograd expression
    (topk_scores.sum() + aux_loss).backward().
    """
    B, L, E, K = 3, 5, 7, 2
    coeff = 0.125
    torch.manual_seed(0)

    scores_BLE = torch.rand(B, L, E, dtype=torch.double, requires_grad=True)
    topk_expert_ids_BLK = torch.topk(scores_BLE.detach(), k=K, dim=-1).indices
    routing_map_BLE = _routing_map(topk_expert_ids_BLK, E)

    injected_scores_BLE = scores_BLE.clone().detach().requires_grad_(True)
    injected_topk_scores_BLK = injected_scores_BLE.gather(
        dim=-1, index=topk_expert_ids_BLK
    )
    injected_aux_loss = loss_fn(injected_scores_BLE, routing_map_BLE, K, coeff)
    injected_output = _AuxLossInjection.apply(
        injected_topk_scores_BLK, injected_aux_loss
    )
    injected_output.sum().backward()

    explicit_scores_BLE = scores_BLE.clone().detach().requires_grad_(True)
    explicit_topk_scores_BLK = explicit_scores_BLE.gather(
        dim=-1, index=topk_expert_ids_BLK
    )
    explicit_aux_loss = loss_fn(explicit_scores_BLE, routing_map_BLE, K, coeff)
    (explicit_topk_scores_BLK.sum() + explicit_aux_loss).backward()

    torch.testing.assert_close(
        injected_scores_BLE.grad,
        explicit_scores_BLE.grad,
        rtol=1e-12,
        atol=1e-12,
    )


def test_batch_wise_aux_loss_matches_qwen_definition():
    scores_BLE = torch.tensor(
        [
            [[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]],
            [[0.3, 0.2, 0.5], [0.6, 0.1, 0.3]],
        ]
    )
    topk_expert_ids_BLK = torch.tensor(
        [
            [[0, 1], [1, 2]],
            [[2, 0], [0, 1]],
        ]
    )
    coeff = 1e-3

    actual = _batch_wise_load_balance_loss(
        scores_BLE,
        _routing_map(topk_expert_ids_BLK, 3),
        top_k=2,
        aux_loss_coeff=coeff,
    )

    counts_E = torch.tensor([3.0, 3.0, 2.0])
    num_tokens = counts_E.sum() / 2
    token_fraction_E = counts_E * (3 / (2 * num_tokens))
    probability_fraction_E = scores_BLE.sum(dim=(0, 1)) / num_tokens
    expected = coeff * torch.sum(token_fraction_E * probability_fraction_E)
    torch.testing.assert_close(actual, expected)


def test_batch_wise_aux_loss_rejects_gradient_accumulation():
    aux_loss = MoELoadBalanceAuxLoss.Config(
        type="batch_wise",
        coeff=1e-3,
        top_k=2,
    )
    layer = SimpleNamespace(
        moe=SimpleNamespace(router=SimpleNamespace(aux_loss=aux_loss))
    )

    update_moe_aux_loss_configs(
        [layer],
        pp_enabled=False,
        local_global_batch_size=16,
        global_batch_size=16,
        seq_len=8,
    )
    assert aux_loss.global_batch_size == 16

    with pytest.raises(ValueError, match="gradient accumulation"):
        update_moe_aux_loss_configs(
            [layer],
            pp_enabled=False,
            local_global_batch_size=16,
            global_batch_size=32,
            seq_len=8,
        )


def test_batch_wise_aux_loss_rejects_pipeline_parallelism():
    aux_loss = MoELoadBalanceAuxLoss.Config(
        type="batch_wise",
        coeff=1e-3,
        top_k=2,
    )
    layer = SimpleNamespace(
        moe=SimpleNamespace(router=SimpleNamespace(aux_loss=aux_loss))
    )

    with pytest.raises(ValueError, match="pipeline parallelism"):
        update_moe_aux_loss_configs(
            [layer],
            pp_enabled=True,
            local_global_batch_size=16,
            global_batch_size=16,
            seq_len=8,
        )


@pytest.mark.parametrize("loss_type", ["batch_wise", "sequence_wise"])
def test_aux_loss_requires_global_batch_size(loss_type):
    """Without it the B / global_batch_size weight would silently become 1."""
    aux_loss = MoELoadBalanceAuxLoss.Config(
        type=loss_type, coeff=1e-3, top_k=2
    ).build()
    scores = torch.rand(2, 4, 5)
    expert_ids = scores.topk(k=2, dim=-1).indices
    with pytest.raises(ValueError, match="global_batch_size is not set"):
        aux_loss(
            scores,
            _routing_map(expert_ids, 5),
            carrier=scores.gather(-1, expert_ids),
        )


@pytest.mark.parametrize(
    "meshes, ep_enabled",
    [({"cp": object()}, False), ({"tp": object()}, True)],
    ids=["cp", "tp_with_ep"],
)
def test_aux_loss_reduction_rejects_token_sharding(meshes, ep_enabled):
    """CP, or TP with EP, would regroup rank-local tokens into the wrong B."""
    model = nn.ModuleDict(
        {"aux": _global_aux_loss_config().build()}
    )
    parallel_dims = SimpleNamespace(
        get_optional_mesh=meshes.get, ep_enabled=ep_enabled
    )
    with pytest.raises(NotImplementedError, match="context parallelism"):
        moe_module.configure_moe_aux_loss_reduction(model, parallel_dims)


def test_aux_loss_reduction_allows_tp_without_ep():
    """Without EP, TP replicates router tokens, so the grouping is correct."""
    model = nn.ModuleDict(
        {"aux": _global_aux_loss_config().build()}
    )
    parallel_dims = SimpleNamespace(
        get_optional_mesh={"tp": object()}.get, ep_enabled=False
    )
    moe_module.configure_moe_aux_loss_reduction(model, parallel_dims)
    assert model["aux"].tp_mesh is None


def test_sequence_wise_rejects_partial_sequences():
    scores_TE = torch.rand(_L + 1, _E)
    expert_ids_TK = scores_TE.topk(k=_K, dim=-1).indices
    aux_loss = MoELoadBalanceAuxLoss.Config(
        type="sequence_wise", coeff=1e-3, top_k=_K, global_batch_size=1, seq_len=_L
    ).build()
    with pytest.raises(ValueError, match="not a multiple of seq_len"):
        aux_loss(
            scores_TE,
            _routing_map(expert_ids_TK, _E),
            carrier=scores_TE.gather(-1, expert_ids_TK),
        )


def test_aux_loss_requires_seq_len_for_flat_tokens():
    scores_TE = torch.rand(_B * _L, _E)
    expert_ids_TK = scores_TE.topk(k=_K, dim=-1).indices
    aux_loss = MoELoadBalanceAuxLoss.Config(
        type="batch_wise", coeff=1e-3, top_k=_K, global_batch_size=_B
    ).build()
    with pytest.raises(ValueError, match="seq_len is not set"):
        aux_loss(
            scores_TE,
            _routing_map(expert_ids_TK, _E),
            carrier=scores_TE.gather(-1, expert_ids_TK),
        )


@pytest.mark.parametrize("loss_type", ["batch_wise", "sequence_wise"])
@pytest.mark.parametrize("use_routing_map", [True, False], ids=["map", "histc"])
def test_flat_tokens_match_batched_sequences(loss_type, use_routing_map):
    """The router's [T, E] outputs give the [B, L, E] loss and gradient.

    No unflatten-shim context is active, as on non-TPU trainers.
    """
    if loss_type == "sequence_wise" and not use_routing_map:
        pytest.skip("sequence_wise always consumes the routing map")
    torch.manual_seed(0)
    global_batch_size = 2 * _B
    scores_BLE = torch.rand(_B, _L, _E, dtype=torch.double)
    expert_ids_BLK = scores_BLE.topk(k=_K, dim=-1).indices
    routing_map_BLE = _routing_map(expert_ids_BLK, _E)
    grads = []
    for flat in (False, True):
        scores = scores_BLE.clone().requires_grad_(True)
        aux_loss = MoELoadBalanceAuxLoss.Config(
            type=loss_type,
            coeff=1e-2,
            top_k=_K,
            global_batch_size=global_batch_size,
            seq_len=_L,
        ).build()
        if flat:
            scores_in = scores.reshape(-1, _E)
            map_in = routing_map_BLE.reshape(-1, _E) if use_routing_map else None
            ids = expert_ids_BLK.reshape(-1, _K)
        else:
            scores_in, map_in, ids = scores, routing_map_BLE, expert_ids_BLK
        out = aux_loss(
            scores_in,
            map_in,
            carrier=scores_in.gather(-1, ids),
            expert_ids_TK=ids if flat else None,
        )
        out.sum().backward()
        grads.append(scores.grad)
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-12, atol=1e-12)


def test_aux_loss_metric_capture_does_not_change_output_or_gradient():
    torch.manual_seed(0)
    scores = torch.rand(2, 4, 5, dtype=torch.double)
    expert_ids = scores.topk(k=2, dim=-1).indices
    routing_map = _routing_map(expert_ids, 5)
    outputs = []
    gradients = []

    for enabled in (False, True):
        current_scores = scores.clone().requires_grad_(True)
        topk_scores = current_scores.gather(-1, expert_ids)
        aux_loss = MoELoadBalanceAuxLoss.Config(
            type="batch_wise",
            coeff=1e-3,
            top_k=2,
            global_batch_size=2,
        ).build()
        aux_loss.collect_routing_metrics = enabled
        output = aux_loss(current_scores, routing_map, carrier=topk_scores)
        output.sum().backward()
        outputs.append(output.detach())
        gradients.append(current_scores.grad)

    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(gradients[0], gradients[1])


def test_routing_metrics_ignore_model_parts_without_top_level_layers():
    class FakeModelWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(2, 2))

    model_parts = [FakeModelWrapper()]

    assert not configure_moe_routing_metrics(model_parts, enabled=True)
    assert collect_moe_routing_metric_tensors(model_parts) == []
    assert get_moe_load_balancing_metrics(model_parts) == {}


def test_aux_loss_metrics_include_route_histogram():
    scores_BLE = torch.tensor(
        [
            [[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]],
            [[0.3, 0.2, 0.5], [0.6, 0.1, 0.3]],
        ]
    )
    topk_expert_ids_BLK = torch.tensor(
        [
            [[0, 1], [1, 2]],
            [[2, 0], [0, 1]],
        ]
    )
    aux_loss = MoELoadBalanceAuxLoss.Config(
        type="batch_wise",
        coeff=1e-3,
        top_k=2,
        global_batch_size=2,
    ).build()
    class FakeRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.aux_loss = aux_loss

    class FakeMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = FakeRouter()

    class FakeBlock(nn.Module):
        moe_enabled = True

        def __init__(self):
            super().__init__()
            self.moe = FakeMoE()

    class FakeModelPart(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict({"0": FakeBlock()})

    model_parts = [FakeModelPart()]
    assert configure_moe_routing_metrics(model_parts, enabled=True)

    topk_scores_BLK = scores_BLE.gather(-1, topk_expert_ids_BLK)
    routing_map_BLE = _routing_map(topk_expert_ids_BLK, 3)
    aux_loss(scores_BLE, routing_map_BLE, carrier=topk_scores_BLK)
    torch.testing.assert_close(
        aux_loss._last_local_tokens_per_expert_E, torch.tensor([3.0, 3.0, 2.0])
    )
    normalized_aux_loss = aux_loss._last_local_normalized_aux_loss
    set_moe_routing_metric_tensors(
        model_parts,
        [torch.tensor([[3.0, 3.0, 2.0]]), normalized_aux_loss.unsqueeze(0)],
    )
    captured_counts, captured_aux_loss = collect_moe_routing_metric_tensors(
        model_parts
    )
    torch.testing.assert_close(captured_counts, torch.tensor([[3.0, 3.0, 2.0]]))
    torch.testing.assert_close(captured_aux_loss, normalized_aux_loss.unsqueeze(0))

    metrics = get_moe_load_balancing_metrics(model_parts)
    expected_normalized_aux_loss = _batch_wise_load_balance_loss(
        scores_BLE,
        routing_map_BLE,
        top_k=2,
        aux_loss_coeff=1.0,
    )

    assert json.loads(metrics["moe/route_counts_json"]) == [[3, 3, 2]]
    assert not any(tag.startswith("moe/route_count/layer_") for tag in metrics)
    torch.testing.assert_close(
        metrics["moe/route_assignments/layer_000"],
        torch.tensor(8.0),
    )
    torch.testing.assert_close(
        metrics["moe/global_aux_loss_normalized"], expected_normalized_aux_loss
    )
    assert "moe/router_logits_abs_mean" not in metrics
    assert "moe/dropped_routes_total" not in metrics


def test_graph_trainer_returns_routing_metrics_as_graph_outputs():
    class FakeRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.aux_loss = MoELoadBalanceAuxLoss.Config(
                type="batch_wise",
                coeff=1e-3,
                top_k=2,
                global_batch_size=2,
            ).build()

    class FakeMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = FakeRouter()

    class FakeBlock(nn.Module):
        moe_enabled = True

        def __init__(self):
            super().__init__()
            self.moe = FakeMoE()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3, 3))
            self.layers = nn.ModuleDict({"0": FakeBlock()})

        def forward(self, x, **_kwargs):
            scores = (x @ self.weight).softmax(dim=-1)
            expert_ids = scores.topk(k=2, dim=-1).indices
            topk_scores = scores.gather(-1, expert_ids)
            aux_loss = self.layers["0"].moe.router.aux_loss
            return aux_loss(
                scores, _routing_map(expert_ids, 3), carrier=topk_scores
            )

    model = FakeModel()
    assert configure_moe_routing_metrics([model], enabled=True)

    def loss_fn(pred, labels, *, global_valid_tokens):
        return ((pred - labels) ** 2).sum() / global_valid_tokens

    fwd_bwd_step = make_fwd_bwd_step(
        model,
        loss_fn,
        collect_moe_routing_metrics=True,
    )
    inputs = torch.randn(2, 4, 3)
    labels = torch.randn(2, 4, 2)
    eager_outputs = fwd_bwd_step(inputs, labels, 8.0, {})
    traced_step = minimal_fx_tracer(fwd_bwd_step, module=model)(
        inputs, labels, 8.0, {}
    )
    traced_outputs = run_traced(traced_step, module=model)(
        inputs, labels, 8.0, {}
    )

    assert len(traced_step.gm.graph.find_nodes(op="output")[0].args[0]) == 4
    assert len(traced_outputs.routing_metrics) == 2
    assert traced_outputs.routing_metrics[0].shape == (1, 3)
    assert traced_outputs.routing_metrics[1].shape == (1,)
    torch.testing.assert_close(eager_outputs, traced_outputs)


_E, _K, _D = 4, 2, 8
_B, _L = 2, 6


def _reference_global_batch_loss(scores_TE, routing_map_TE, top_k, coeff):
    """Qwen's global-batch load-balance loss, written out directly.

    ``coeff * sum_i f_i * P_i`` with ``f_i = E / (K T) * count_i`` and
    ``P_i`` the mean of the per-token normalized scores over all ``T`` tokens.
    """
    num_tokens, num_experts = scores_TE.shape
    counts_E = routing_map_TE.to(scores_TE.dtype).sum(dim=0)
    f_E = counts_E * num_experts / (top_k * num_tokens)
    p_E = (scores_TE / scores_TE.sum(dim=-1, keepdim=True)).mean(dim=0)
    return coeff * (f_E * p_E).sum()


def _router_config(aux_loss=None):
    return TokenChoiceTopKRouter.Config(
        num_experts=_E,
        top_k=_K,
        score_func="softmax",
        route_norm=True,
        gate=Linear.Config(in_features=_D, out_features=_E, bias=False),
        aux_loss=aux_loss,
    )


def _global_aux_loss_config(global_batch_size=_B):
    config = make_aux_loss_config(type="batch_wise", coeff=1e-2, top_k=_K)
    config.global_batch_size = global_batch_size
    config.seq_len = _L
    return config


def _moe_config(*, load_balance_coeff=None, aux_loss=None):
    return MoE.Config(
        num_experts=_E,
        load_balance_coeff=load_balance_coeff,
        router=_router_config(aux_loss),
        routed_experts=RoutedExperts.Config(
            inner_experts=GroupedExperts.Config(
                dim=_D, hidden_dim=_D, num_experts=_E
            ),
            token_dispatcher=LocalTokenDispatcher.Config(num_experts=_E, top_k=_K),
        ),
    )


def _init_normal(module):
    torch.manual_seed(0)
    for param in module.parameters():
        nn.init.normal_(param, std=0.5)
    return module


def test_global_batch_aux_loss_is_not_a_logged_aux_loss_group():
    """``collect_aux_loss_metrics`` would log a constant 0 for this loss."""
    aux_loss = _global_aux_loss_config().build()
    assert isinstance(aux_loss, AuxLoss)
    assert (aux_loss.reduce_mesh, aux_loss.metric_name) not in AuxLoss._group_counts


def test_router_calls_global_batch_aux_loss_only_in_training():
    router = _init_normal(_router_config(_global_aux_loss_config()).build())
    calls = []
    router.aux_loss.register_forward_hook(lambda *_: calls.append(1))
    x_TD = torch.randn(_B * _L, _D)

    router.train()
    router(x_TD)
    assert len(calls) == 1
    router.eval()
    router(x_TD)
    assert len(calls) == 1


def test_router_injects_global_batch_aux_loss_gradient():
    """The gate gradient is that of ``topk_scores.sum() + (B / GBS) * loss``."""
    global_batch_size = 2 * _B
    coeff = 1e-2
    x_TD = torch.randn(_B * _L, _D)

    router = _init_normal(
        _router_config(_global_aux_loss_config(global_batch_size)).build()
    )
    topk_scores_TK, topk_expert_ids_TK, _ = router(x_TD)
    topk_scores_TK.sum().backward()

    reference = _init_normal(_router_config().build())
    scores_TE = F.softmax(reference.gate(x_TD), dim=-1)
    ref_topk_scores_TK, ref_topk_expert_ids_TK, _ = reference(x_TD)
    assert torch.equal(ref_topk_expert_ids_TK, topk_expert_ids_TK)
    routing_map_TE = _routing_map(topk_expert_ids_TK, _E)
    no_aux_grad = torch.autograd.grad(
        ref_topk_scores_TK.sum(), reference.gate.weight, retain_graph=True
    )[0]
    aux_loss = _reference_global_batch_loss(
        scores_TE, routing_map_TE, _K, coeff * _B / global_batch_size
    )
    (ref_topk_scores_TK.sum() + aux_loss).backward()

    torch.testing.assert_close(router.gate.weight.grad, reference.gate.weight.grad)
    assert not torch.allclose(router.gate.weight.grad, no_aux_grad)


def test_global_batch_aux_loss_sums_to_qwen_loss_over_dp(monkeypatch):
    """Per-DP-rank losses sum to the global-batch loss of the whole batch.

    Two DP ranks are simulated: the all-reduce of the expert counts adds the
    other rank's counts, which is what ``dist.all_reduce`` over the batch mesh
    does.
    """
    torch.manual_seed(0)
    coeff, dp, local_batch = 1e-2, 2, 2
    scores_RBLE = torch.rand(dp, local_batch, _L, _E, dtype=torch.double)
    routing_map_RBLE = _routing_map(scores_RBLE.topk(k=_K, dim=-1).indices, _E)
    counts_RE = routing_map_RBLE.sum(dim=(1, 2)).double()
    fake_dp_mesh = SimpleNamespace(size=lambda: dp, get_group=lambda: None)

    per_rank_losses = []
    for rank in range(dp):
        other_counts_E = counts_RE[1 - rank]

        def fake_all_reduce(tensor, group=None, op=None, _other=other_counts_E):
            tensor.add_(_other)

        monkeypatch.setattr(moe_module.dist, "all_reduce", fake_all_reduce)
        aux_loss = make_aux_loss_config(type="batch_wise", coeff=coeff, top_k=_K)
        aux_loss.global_batch_size = dp * local_batch
        module = aux_loss.build()
        module.set_meshes(dp_mesh=fake_dp_mesh, cp_mesh=None, tp_mesh=None)
        scores_BLE = scores_RBLE[rank].clone().requires_grad_(True)
        carrier = scores_BLE.sum(dim=-1)
        module.collect_routing_metrics = True
        module(scores_BLE, routing_map_RBLE[rank], carrier=carrier)
        scaled_coeff = coeff * local_batch / (dp * local_batch)
        per_rank_losses.append(module._last_local_normalized_aux_loss * scaled_coeff)

    expected = _reference_global_batch_loss(
        scores_RBLE.reshape(-1, _E), routing_map_RBLE.reshape(-1, _E), _K, coeff
    )
    torch.testing.assert_close(
        sum(per_rank_losses).double(), expected, rtol=1e-6, atol=0
    )


def test_moe_forward_produces_routing_metrics():
    """With the routing metrics on, a real MoE forward feeds the logger."""

    class Block(nn.Module):
        moe_enabled = True

        def __init__(self):
            super().__init__()
            self.moe = _moe_config(aux_loss=_global_aux_loss_config(_B)).build()

    class ModelPart(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict({"0": Block()})

    model_part = _init_normal(ModelPart())
    moe = model_part.layers["0"].moe
    assert configure_moe_routing_metrics([model_part], enabled=True)
    x_TD = torch.randn(_B * _L, _D)
    moe(x_TD)

    metrics = get_moe_load_balancing_metrics([model_part])

    counts_E = moe.tokens_per_expert_E
    assert json.loads(metrics["moe/route_counts_json"]) == [
        counts_E.to(torch.int64).tolist()
    ]
    torch.testing.assert_close(
        metrics["moe/route_assignments/layer_000"],
        torch.tensor(float(_B * _L * _K)),
    )
    with torch.no_grad():
        scores_TE = F.softmax(moe.router.gate(x_TD), dim=-1)
    routing_map_TE = _routing_map(scores_TE.topk(k=_K, dim=-1).indices, _E)
    torch.testing.assert_close(
        metrics["moe/global_aux_loss_normalized"],
        _reference_global_batch_loss(scores_TE, routing_map_TE, _K, 1.0),
    )


def test_routing_metrics_skip_upstream_router_aux_losses():
    class Block(nn.Module):
        moe_enabled = True

        def __init__(self):
            super().__init__()
            self.moe = _moe_config(
                aux_loss=moe_module.MicrobatchWiseLoadBalanceLoss.Config(coeff=1e-3)
            ).build()

    class ModelPart(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict({"0": Block()})

    assert not configure_moe_routing_metrics([ModelPart()], enabled=True)


def test_load_balance_coeff_and_global_batch_aux_loss_are_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _moe_config(
            load_balance_coeff=1e-3, aux_loss=_global_aux_loss_config()
        ).build()


def test_route_histogram_round_trips_as_one_tensorboard_event(tmp_path):
    logger = TensorBoardLogger(str(tmp_path))
    logger.log({"moe/route_counts_json": "[[3,3,2]]"}, step=1)
    logger.close()

    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    tag = "moe/route_counts_json/text_summary"
    events = accumulator.Tensors(tag)

    assert len(events) == 1
    assert events[0].step == 1
    assert json.loads(events[0].tensor_proto.string_val[0].decode()) == [[3, 3, 2]]


def test_tensorboard_logs_scalar_and_text_metrics(tmp_path):
    logger = TensorBoardLogger(str(tmp_path))
    logger.log(
        {
            "loss_metrics/global_avg_loss": 1.25,
            "moe/global_aux_loss_normalized": torch.tensor(1.01),
            "moe/route_counts_json": "[[3,3,2]]",
        },
        step=7,
    )
    logger.close()

    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    assert accumulator.Scalars("loss_metrics/global_avg_loss")[0].value == 1.25
    assert accumulator.Scalars("moe/global_aux_loss_normalized")[
        0
    ].value == pytest.approx(1.01)
    assert json.loads(
        accumulator.Tensors("moe/route_counts_json/text_summary")[
            0
        ].tensor_proto.string_val[0].decode()
    ) == [[3, 3, 2]]
