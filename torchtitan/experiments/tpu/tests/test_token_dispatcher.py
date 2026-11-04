from unittest import mock

import pytest
import torch

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRoutedExperts,
    TpuTokenDispatcherConverter,
    scale_route_weights,
    stable_grouped_reduce,
)
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher


def test_scale_route_weights_matches_native_autograd():
    generator = torch.Generator().manual_seed(2468)
    routed = torch.randn(17, 11, dtype=torch.bfloat16, generator=generator)
    scores = torch.randn(17, dtype=torch.float32, generator=generator)
    grad_output = torch.randn(17, 11, dtype=torch.bfloat16, generator=generator)

    for routed_requires_grad, scores_require_grad in (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ):
        actual_routed = routed.detach().requires_grad_(routed_requires_grad)
        actual_scores = scores.detach().requires_grad_(scores_require_grad)
        actual = scale_route_weights(actual_routed, actual_scores)

        reference_routed = routed.detach().requires_grad_(routed_requires_grad)
        reference_scores = scores.detach().requires_grad_(scores_require_grad)
        reference = (
            reference_routed.float() * reference_scores.reshape(-1, 1)
        ).to(reference_routed.dtype)

        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        actual_inputs = tuple(
            value
            for value in (actual_routed, actual_scores)
            if value.requires_grad
        )
        if not actual_inputs:
            continue
        reference_inputs = tuple(
            value
            for value in (reference_routed, reference_scores)
            if value.requires_grad
        )
        actual_grads = torch.autograd.grad(actual, actual_inputs, grad_output)
        reference_grads = torch.autograd.grad(
            reference, reference_inputs, grad_output
        )
        for actual_grad, reference_grad in zip(
            actual_grads, reference_grads, strict=True
        ):
            torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)


def test_scale_route_weights_validates_shapes():
    with pytest.raises(ValueError, match=r"routed\[N, D\] and scores\[N\]"):
        scale_route_weights(torch.ones(3), torch.ones(3))
    with pytest.raises(ValueError, match="row count must match"):
        scale_route_weights(torch.ones(3, 2), torch.ones(4))


def test_scale_route_weights_retains_native_dtype_fallback():
    routed = torch.randn(5, 3, dtype=torch.float32, requires_grad=True)
    scores = torch.randn(5, dtype=torch.float32, requires_grad=True)
    grad_output = torch.randn_like(routed)

    actual = scale_route_weights(routed, scores)
    expected = routed * scores.reshape(-1, 1)
    actual_grads = torch.autograd.grad(actual, (routed, scores), grad_output)
    expected_grads = torch.autograd.grad(expected, (routed, scores), grad_output)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(
        actual_grads, expected_grads, strict=True
    ):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_scale_route_weights_traces_forward_and_backward():
    def step(routed, scores, grad_output):
        weighted = scale_route_weights(routed, scores)
        return torch.autograd.grad(
            weighted, (routed, scores), grad_output
        )

    generator = torch.Generator().manual_seed(9753)
    routed = torch.randn(
        17, 11, dtype=torch.bfloat16, generator=generator, requires_grad=True
    )
    scores = torch.randn(
        17, dtype=torch.float32, generator=generator, requires_grad=True
    )
    grad_output = torch.randn(17, 11, dtype=torch.bfloat16, generator=generator)
    expected = step(routed, scores, grad_output)
    traced = minimal_fx_tracer(step)(routed, scores, grad_output)
    actual = run_traced(traced)(routed, scores, grad_output)

    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_local_reorder_matches_reference_forward_and_backward():
    generator = torch.Generator().manual_seed(1234)
    dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=3, top_k=8
    ).build()
    x = torch.randn(5, 4, dtype=torch.bfloat16, generator=generator)
    x.requires_grad_()
    scores = torch.randn(5, 8, generator=generator, requires_grad=True)
    expert_ids = torch.randint(0, 3, (5, 8), generator=generator)
    grad_routed = torch.randn(40, 4, dtype=torch.bfloat16, generator=generator)
    grad_scores = torch.randn(40, generator=generator)

    actual_routed, actual_indices, actual_scores = dispatcher._local_reorder(
        x, scores, expert_ids
    )
    actual_grads = torch.autograd.grad(
        (actual_routed, actual_scores),
        (x, scores),
        (grad_routed, grad_scores),
    )

    reference_x = x.detach().requires_grad_()
    reference_scores = scores.detach().requires_grad_()
    permutation = torch.argsort(expert_ids.reshape(-1), stable=True)
    reference_indices = permutation // 8
    reference_routed = reference_x[reference_indices]
    reference_sorted_scores = reference_scores.reshape(-1)[permutation]
    reference_grads = torch.autograd.grad(
        (reference_routed, reference_sorted_scores),
        (reference_x, reference_scores),
        (grad_routed, grad_scores),
    )

    torch.testing.assert_close(actual_routed, reference_routed)
    torch.testing.assert_close(actual_indices, reference_indices)
    torch.testing.assert_close(actual_scores, reference_sorted_scores)
    torch.testing.assert_close(
        actual_grads[0], reference_grads[0], rtol=0.05, atol=0.125
    )
    torch.testing.assert_close(actual_grads[1], reference_grads[1])


def test_local_reorder_preserves_unused_gradients():
    dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=2, top_k=2
    ).build()
    expert_ids = torch.tensor([[1, 0], [0, 1]])
    grad_routed = torch.tensor([[1.0], [0.0], [3.0], [0.0]])
    grad_scores = torch.tensor([1.0, 0.0, 3.0, 0.0])

    x = torch.tensor([[2.0], [4.0]], requires_grad=True)
    scores = torch.arange(4, dtype=torch.float32).reshape(2, 2).requires_grad_()
    routed, _, sorted_scores = dispatcher._local_reorder(x, scores, expert_ids)
    grad_x, grad_score_input = torch.autograd.grad(
        (routed, sorted_scores),
        (x, scores),
        (grad_routed, grad_scores),
    )
    permutation = torch.argsort(expert_ids.reshape(-1), stable=True)
    expected_x = torch.zeros_like(x).index_add(0, permutation // 2, grad_routed)
    expected_scores = torch.zeros_like(scores).reshape(-1)
    expected_scores[permutation] = grad_scores
    torch.testing.assert_close(grad_x, expected_x)
    torch.testing.assert_close(grad_score_input, expected_scores.reshape_as(scores))

    x = x.detach().requires_grad_()
    scores = scores.detach().requires_grad_()
    routed, _, _ = dispatcher._local_reorder(x, scores, expert_ids)
    routed_only = torch.autograd.grad(
        routed, (x, scores), grad_routed, allow_unused=True
    )
    assert routed_only[1] is None

    x = x.detach().requires_grad_()
    scores = scores.detach().requires_grad_()
    _, _, sorted_scores = dispatcher._local_reorder(x, scores, expert_ids)
    scores_only = torch.autograd.grad(
        sorted_scores, (x, scores), grad_scores, allow_unused=True
    )
    assert scores_only[0] is None


def test_local_combine_matches_standard_dispatcher():
    generator = torch.Generator().manual_seed(5678)
    actual_dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=3,
        top_k=2,
        fuse_route_score_backward=True,
    ).build()
    reference_dispatcher = AllToAllTokenDispatcher.Config(
        num_experts=3, top_k=2
    ).build()
    x = torch.randn(5, 4, dtype=torch.bfloat16, generator=generator)
    expert_ids = torch.randint(0, 3, (5, 2), generator=generator)
    counts = torch.bincount(expert_ids.reshape(-1), minlength=3)

    actual_scores = torch.randn(5, 2, generator=generator, requires_grad=True)
    reference_scores = actual_scores.detach().requires_grad_()
    _, _, actual_metadata = actual_dispatcher.dispatch(
        x, actual_scores, expert_ids, counts
    )
    _, _, reference_metadata = reference_dispatcher.dispatch(
        x, reference_scores, expert_ids, counts
    )

    actual_routed = torch.randn(
        10, 4, dtype=torch.bfloat16, generator=generator, requires_grad=True
    )
    reference_routed = actual_routed.detach().requires_grad_()
    actual = actual_dispatcher.combine(actual_routed, actual_metadata, x)
    reference = reference_dispatcher.combine(
        reference_routed, reference_metadata, x
    )
    grad_output = torch.randn(5, 4, dtype=torch.bfloat16, generator=generator)
    actual_grads = torch.autograd.grad(
        actual, (actual_routed, actual_scores), grad_output
    )
    reference_grads = torch.autograd.grad(
        reference, (reference_routed, reference_scores), grad_output
    )

    torch.testing.assert_close(actual, reference, rtol=0.05, atol=0.125)
    torch.testing.assert_close(
        actual_grads[0], reference_grads[0], rtol=0.05, atol=0.125
    )
    torch.testing.assert_close(actual_grads[1], reference_grads[1])


def test_ep_combine_delegates_to_standard_dispatcher():
    dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=2, top_k=2
    ).build()
    dispatcher.ep_mesh = object()
    expected = torch.tensor([1.0])

    with mock.patch.object(
        AllToAllTokenDispatcher, "combine", return_value=expected
    ) as combine:
        actual = dispatcher.combine(
            torch.empty(0), mock.sentinel.metadata, torch.empty(0)
        )

    assert actual is expected
    combine.assert_called_once()


def test_grouped_reduce_saves_only_row_index():
    index = torch.tensor([2, 0, 2, 1, 0, 1])
    src = torch.arange(24, dtype=torch.float32).reshape(6, 4).requires_grad_()
    grad_output = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tensor.shape)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = stable_grouped_reduce(src, index, top_k=2)
        (grad_src,) = torch.autograd.grad(actual, (src,), grad_output)

    reference = torch.zeros(3, 4).scatter_add(
        0, index.reshape(-1, 1).expand_as(src), src.detach()
    )
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(grad_src, torch.index_select(grad_output, 0, index))
    assert saved_shapes == [index.shape]


def test_converter_preserves_fields_and_is_idempotent():
    class RootConfig:
        def __init__(self):
            self.dispatcher = AllToAllTokenDispatcher.Config(
                num_experts=8, top_k=2
            )

        def traverse(self, config_type):
            if isinstance(self.dispatcher, config_type):
                yield "dispatcher", self.dispatcher, self, "dispatcher"

    root = RootConfig()
    converter = TpuTokenDispatcherConverter.Config().build()
    converted = converter.convert(root)

    assert converted is root
    assert isinstance(root.dispatcher, TpuAllToAllTokenDispatcher.Config)
    assert root.dispatcher.num_experts == 8
    assert root.dispatcher.top_k == 2
    assert not root.dispatcher.fuse_route_score_backward
    # Drop-free unless a recipe opts into a capacity factor.
    assert root.dispatcher.capacity_factor is None

    dispatcher = root.dispatcher
    converter.convert(root)
    assert root.dispatcher is dispatcher


class _SwiGLUExperts(torch.nn.Module):
  """Per-row SwiGLU experts that accept ``row_scale_R`` like TpuGroupedExperts."""

  def __init__(self, w1_EDF, w3_EDF, w2_EFD):
    super().__init__()
    self.w1_EDF, self.w3_EDF, self.w2_EFD = w1_EDF, w3_EDF, w2_EFD

  def forward(self, x_RD, num_tokens_per_expert_E, row_scale_R=None):
    expert_R = torch.repeat_interleave(
        torch.arange(len(num_tokens_per_expert_E)), num_tokens_per_expert_E
    )
    gate_RF = torch.einsum("rd,rdf->rf", x_RD, self.w1_EDF[expert_R])
    up_RF = torch.einsum("rd,rdf->rf", x_RD, self.w3_EDF[expert_R])
    hidden_RF = torch.nn.functional.silu(gate_RF) * up_RF
    if row_scale_R is not None:
      hidden_RF = hidden_RF * row_scale_R.reshape(-1, 1)
    return torch.einsum("rf,rfd->rd", hidden_RF, self.w2_EFD[expert_R])


def test_scale_before_down_proj_matches_scaling_the_output():
  generator = torch.Generator().manual_seed(1357)
  num_experts, top_k, dim, hidden = 3, 2, 4, 6
  weights = [
      torch.randn(num_experts, *shape, generator=generator, requires_grad=True)
      for shape in ((dim, hidden), (dim, hidden), (hidden, dim))
  ]
  x = torch.randn(5, dim, generator=generator, requires_grad=True)
  scores = torch.rand(5, top_k, generator=generator, requires_grad=True)
  expert_ids = torch.randint(0, num_experts, (5, top_k), generator=generator)
  counts = torch.bincount(expert_ids.reshape(-1), minlength=num_experts)

  def run(scale_before_down_proj):
    routed = TpuRoutedExperts.Config(
        inner_experts=GroupedExperts.Config(
            dim=dim, hidden_dim=hidden, num_experts=num_experts
        ),
        token_dispatcher=TpuAllToAllTokenDispatcher.Config(
            num_experts=num_experts, top_k=top_k
        ),
        scale_before_down_proj=scale_before_down_proj,
    ).build()
    routed.inner_experts = _SwiGLUExperts(*weights)
    out = routed(x, scores, expert_ids, counts)
    return out, torch.autograd.grad(out.square().sum(), (x, scores, *weights))

  actual, actual_grads = run(True)
  expected, expected_grads = run(False)
  torch.testing.assert_close(actual, expected)
  for actual_grad, expected_grad in zip(actual_grads, expected_grads):
    torch.testing.assert_close(actual_grad, expected_grad)

  with pytest.raises(ValueError, match="enable only one"):
    TpuRoutedExperts.Config(
        inner_experts=GroupedExperts.Config(
            dim=dim, hidden_dim=hidden, num_experts=num_experts
        ),
        token_dispatcher=TpuAllToAllTokenDispatcher.Config(
            num_experts=num_experts, top_k=top_k, fuse_route_score_backward=True
        ),
        scale_before_down_proj=True,
    ).build()


def test_ep_dispatch_uses_ring_of_experts_by_default():
    dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=4, top_k=2, capacity_factor=1.5
    ).build()
    dispatcher.ep_mesh = mock.sentinel.ep_mesh
    args = (torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
    target = "torchtitan.experiments.tpu.token_dispatcher.ring_of_experts_dispatch"

    with mock.patch(target, return_value=mock.sentinel.out) as dispatch:
        assert dispatcher.dispatch(*args) is mock.sentinel.out

    dispatch.assert_called_once_with(
        *args[:3],
        num_experts=4,
        ep_mesh=mock.sentinel.ep_mesh,
        capacity_factor=1.5,
    )


def test_ep_dispatch_all_to_all_transport_delegates():
    dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=4, top_k=2, ep_transport="all_to_all"
    ).build()
    dispatcher.ep_mesh = mock.sentinel.ep_mesh
    args = (torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))

    with mock.patch.object(
        AllToAllTokenDispatcher, "dispatch", return_value=mock.sentinel.out
    ) as dispatch:
        assert dispatcher.dispatch(*args) is mock.sentinel.out
    dispatch.assert_called_once()


def test_rejects_unknown_ep_transport():
    with pytest.raises(ValueError, match="ep_transport"):
        TpuAllToAllTokenDispatcher.Config(
            num_experts=4, top_k=2, ep_transport="ragged"
        ).build()
