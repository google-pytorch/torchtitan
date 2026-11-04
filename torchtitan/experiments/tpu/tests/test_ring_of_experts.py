from unittest import mock

import pytest
import torch

from torchtitan.experiments.tpu import ring_of_experts
from torchtitan.experiments.tpu.ring_of_experts import (
    CAPACITY_TILE,
    ring_of_experts_capacity,
    ring_of_experts_combine,
    ring_of_experts_dispatch,
)
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher

_FUNCOL = "torchtitan.experiments.tpu.ring_of_experts.funcol"


def _fake_mesh(ep_size, rank):
    mesh = mock.MagicMock()
    mesh.size.return_value = ep_size
    mesh.get_local_rank.return_value = rank
    return mesh


class _FakeEp:
    """Simulates one EP rank's collectives given every rank's inputs.

    All-gathers return the concatenation over ranks; reduce-scatters record
    their input (so tests can sum them over ranks) and return this rank's
    slice of it.
    """

    def __init__(self, rank, ep_size, xs, scores, ids, counts, grad_outs):
        self.rank = rank
        self.ep_size = ep_size
        self.x_r = xs[rank]
        self.global_x = torch.cat(xs)
        self.global_scores = torch.cat([s.reshape(-1) for s in scores])
        self.global_ids = torch.cat([i.reshape(-1).to(torch.int32) for i in ids])
        self.global_routes = torch.cat([
            torch.cat([
                i.reshape(-1).to(torch.int32),
                sc.reshape(-1).float().view(torch.int32),
            ])
            for i, sc in zip(ids, scores)
        ])
        self.global_grad = (
            torch.cat(grad_outs) if grad_outs is not None else None
        )
        self.num_experts = int(counts[0].numel())
        self.reduce_scatter_inputs = []

    def all_gather(self, tensor, gather_dim=0, group=None):
        del gather_dim, group
        if tensor.dtype == torch.int32:
            return self.global_routes
        assert tensor.ndim == 3, tensor.shape
        x_3d = self.x_r.reshape(tensor.shape)
        source = (
            self.global_x
            if torch.equal(tensor.detach(), x_3d.detach())
            else self.global_grad
        )
        return source.reshape(-1, *tensor.shape[1:])

    def reduce_scatter(self, tensor, op, scatter_dim=0, group=None):
        del op, scatter_dim, group
        self.reduce_scatter_inputs.append(tensor.detach().clone())
        n = tensor.shape[0] // self.ep_size
        return tensor[self.rank * n : (self.rank + 1) * n]

    def patch(self):
        return mock.patch.multiple(
            _FUNCOL,
            all_gather_tensor=mock.Mock(side_effect=self.all_gather),
            reduce_scatter_tensor=mock.Mock(side_effect=self.reduce_scatter),
        )


def _expert_fn(routed_RD, group_sizes_e, scales_eD):
    """Scales each row by its expert; rows past the groups become NaN.

    The grouped matmuls leave rows past `sum(group_sizes_e)` unwritten, so
    poisoning them checks that no result depends on them.
    """
    num_rows = int(group_sizes_e.sum())
    e_idx = torch.repeat_interleave(
        torch.arange(group_sizes_e.numel()),
        group_sizes_e.to(torch.int64),
        output_size=num_rows,
    )
    scaled = routed_RD[:num_rows] * scales_eD[e_idx]
    tail = torch.full(
        (routed_RD.shape[0] - num_rows, routed_RD.shape[1]), float("nan")
    )
    return torch.cat([scaled, tail])


def _random_inputs(generator, ep_size, num_tokens, top_k, num_experts, dim):
    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    xs = [randn(num_tokens, dim) for _ in range(ep_size)]
    scores = [randn(num_tokens, top_k) for _ in range(ep_size)]
    ids = [
        torch.stack([
            torch.randperm(num_experts, generator=generator)[:top_k]
            for _ in range(num_tokens)
        ])
        for _ in range(ep_size)
    ]
    counts = [torch.bincount(i.reshape(-1), minlength=num_experts) for i in ids]
    return xs, scores, ids, counts


@pytest.mark.parametrize(
    "num_routed, ep_size, factor, expected",
    [
        (100, 4, None, CAPACITY_TILE * 2),
        (4096, 4, 1.125, 4608),
        (4096, 1, 1.125, 4096),
        (1000, 2, 1.0, 1024),
    ],
)
def test_capacity(num_routed, ep_size, factor, expected):
    assert ring_of_experts_capacity(num_routed, ep_size, factor) == expected


def test_dispatch_combine_matches_global_reference():
    ep_size, num_experts, top_k, num_tokens, dim = 2, 4, 2, 6, 256
    num_local_experts = num_experts // ep_size
    generator = torch.Generator().manual_seed(2027)
    xs, scores, ids, counts = _random_inputs(
        generator, ep_size, num_tokens, top_k, num_experts, dim
    )
    scales = torch.randn(num_experts, dim, generator=generator)
    grad_outs = [torch.randn(num_tokens, dim, generator=generator)
                 for _ in range(ep_size)]

    # Reference: the upstream dispatcher on the concatenated global batch.
    ref_x = torch.cat(xs).requires_grad_()
    ref_scores = torch.cat(scores).requires_grad_()
    ref_scales = scales.clone().requires_grad_()
    ref = AllToAllTokenDispatcher.Config(
        num_experts=num_experts, top_k=top_k
    ).build()
    routed, sizes, meta = ref.dispatch(
        ref_x, ref_scores, torch.cat(ids), torch.stack(counts).sum(0)
    )
    ref_out = ref.combine(_expert_fn(routed, sizes, ref_scales), meta, ref_x)
    ref_grads = torch.autograd.grad(
        ref_out, (ref_x, ref_scores, ref_scales), torch.cat(grad_outs)
    )

    outs, grad_xs, grad_scores = [], [], []
    for rank in range(ep_size):
        fake = _FakeEp(rank, ep_size, xs, scores, ids, counts, grad_outs)
        x_r = xs[rank].clone().requires_grad_()
        scores_r = scores[rank].clone().requires_grad_()
        local = slice(rank * num_local_experts, (rank + 1) * num_local_experts)
        scales_r = scales[local].clone().requires_grad_()
        with fake.patch():
            routed_r, sizes_r, meta_r = ring_of_experts_dispatch(
                x_r,
                scores_r,
                ids[rank],
                num_experts=num_experts,
                ep_mesh=_fake_mesh(ep_size, rank),
                capacity_factor=None,
            )
            assert int(sizes_r.sum()) == int(meta_r.valid_count_1)
            assert int(meta_r.valid_count_1) < routed_r.shape[0]
            assert routed_r.shape[0] % CAPACITY_TILE == 0
            out_r = ring_of_experts_combine(
                _expert_fn(routed_r, sizes_r, scales_r), meta_r
            )
            (_, _, grad_scales_r) = torch.autograd.grad(
                out_r, (x_r, scores_r, scales_r), grad_outs[rank]
            )
        torch.testing.assert_close(grad_scales_r, ref_grads[2][local])
        # [combine fwd, then the x and score reduce-scatters of backward]
        rs = fake.reduce_scatter_inputs
        assert len(rs) == 3
        outs.append(rs[0])
        grad_xs.append(next(t for t in rs[1:] if t.ndim == 3))
        grad_scores.append(next(t for t in rs[1:] if t.ndim == 1))

    torch.testing.assert_close(
        torch.stack(outs).sum(0).reshape(-1, dim), ref_out.detach()
    )
    torch.testing.assert_close(
        torch.stack(grad_xs).sum(0).reshape(-1, dim), ref_grads[0]
    )
    torch.testing.assert_close(
        torch.stack(grad_scores).sum(0).reshape(-1, top_k), ref_grads[1]
    )


def test_capacity_drops_late_routes_in_local_expert_order():
    ep_size, num_experts, top_k, num_tokens, dim = 2, 4, 2, 400, 128
    generator = torch.Generator().manual_seed(5)
    xs, scores, ids, _ = _random_inputs(
        generator, ep_size, num_tokens, top_k, num_experts, dim
    )
    # Route everything to rank 0's experts (0 and 1), overflowing its capacity.
    ids = [torch.stack([torch.tensor([0, 1])] * num_tokens) for _ in ids]
    ids[1][::2] = torch.tensor([1, 0])
    counts = [torch.bincount(i.reshape(-1), minlength=num_experts) for i in ids]
    max_R = ring_of_experts_capacity(num_tokens * top_k, ep_size, 1.0)
    assert max_R < ep_size * num_tokens * top_k

    fake = _FakeEp(0, ep_size, xs, scores, ids, counts, None)
    with fake.patch():
        routed, sizes, meta = ring_of_experts_dispatch(
            xs[0], scores[0], ids[0],
            num_experts=num_experts,
            ep_mesh=_fake_mesh(ep_size, 0),
            capacity_factor=1.0,
        )
        ring_of_experts_combine(routed, meta)
    assert routed.shape[0] == max_R
    assert int(meta.valid_count_1) == max_R
    assert int(sizes.sum()) == max_R

    # Kept routes: all of expert 0 (in global route order), then expert 1.
    flat_ids = fake.global_ids.long()
    order = torch.cat(
        [torch.nonzero(flat_ids == e).flatten() for e in range(2)]
    )[:max_R]
    assert torch.equal(sizes, torch.bincount(flat_ids[order], minlength=2))
    expected = torch.zeros(ep_size * num_tokens, dim)
    expected.index_add_(
        0,
        order // top_k,
        fake.global_x[order // top_k] * fake.global_scores[order, None],
    )
    torch.testing.assert_close(
        fake.reduce_scatter_inputs[0].reshape(-1, dim), expected
    )


def test_rejects_hidden_size_not_multiple_of_128():
    fake = _FakeEp(
        0, 1, [torch.zeros(4, 8)], [torch.zeros(4, 1)],
        [torch.zeros(4, 1, dtype=torch.long)],
        [torch.tensor([4, 0])], None,
    )
    with fake.patch(), pytest.raises(ValueError, match="multiple of 128"):
        ring_of_experts_dispatch(
            torch.zeros(4, 8),
            torch.zeros(4, 1),
            torch.zeros(4, 1, dtype=torch.long),
            num_experts=2,
            ep_mesh=_fake_mesh(1, 0),
            capacity_factor=None,
        )


def test_rejects_experts_not_divisible_by_ep():
    with pytest.raises(ValueError, match="divisible by the EP degree"):
        ring_of_experts_dispatch(
            torch.zeros(4, 128),
            torch.zeros(4, 1),
            torch.zeros(4, 1, dtype=torch.long),
            num_experts=6,
            ep_mesh=_fake_mesh(4, 0),
            capacity_factor=None,
        )


def test_module_exports():
    assert set(ring_of_experts.__all__) >= {
        "ring_of_experts_dispatch",
        "ring_of_experts_combine",
    }
