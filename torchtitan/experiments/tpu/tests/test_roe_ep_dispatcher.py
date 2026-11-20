"""Tests for Ring-of-Experts, ``ep_transport="auto"``, and EP ``scale_before_down_proj``.

Production Ring-of-Experts runs only on the Pallas TensorCore kernels, so the
multi-process CPU (gloo) tests use ``ReferenceRoETokenDispatcher``, which
swaps in the pure-PyTorch reference dispatch and combine and keeps the
production transport selection, capacity, dropped-route, row-scale, and
ragged all-to-all logic.
"""

from unittest import mock

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.experiments.tpu.components.expert_parallel import use_expert_parallel
from torchtitan.experiments.tpu.components.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRaggedDispatchMetadata,
    TpuRoEDispatchMetadata,
    TpuRoutedExperts,
    collect_ep_drop_stat_tensors,
    configure_ep_drop_stats,
    ep_capacity_bounded,
)
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.distributed import distributed
from torchtitan.experiments.tpu.testing.roe_reference import (
    ReferenceRoETokenDispatcher,
    RoEReferenceDispatchMetadata,
)
from torchtitan.models.common.moe import GroupedExperts, RoutedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalDispatchMetadata,
)


class _RankHolder(torch.nn.Module):
    """Stands in for the TpuRoutedExperts that owns the EP rank buffer."""

    def __init__(self, rank: int, device=None):
        super().__init__()
        self.register_buffer(
            "_ep_local_rank", torch.tensor([rank], dtype=torch.long, device=device)
        )


class _RaggedSwiGLUExperts(torch.nn.Module):
    """Per-row SwiGLU experts that ignore rows past ``sum(counts)`` like Pallas GMMs."""

    def __init__(self, w1_EDF: torch.Tensor, w3_EDF: torch.Tensor, w2_EFD: torch.Tensor):
        super().__init__()
        self.w1_EDF = torch.nn.Parameter(w1_EDF)
        self.w3_EDF = torch.nn.Parameter(w3_EDF)
        self.w2_EFD = torch.nn.Parameter(w2_EFD)

    def forward(self, x_RD, num_tokens_per_expert_E, row_scale_R=None):
        valid_rows = int(num_tokens_per_expert_E.sum().item())
        if valid_rows == 0:
            zero = (
                x_RD * 0.0
                + (self.w1_EDF.sum() + self.w3_EDF.sum() + self.w2_EFD.sum()) * 0.0
            )
            if row_scale_R is not None:
                zero = zero + row_scale_R.unsqueeze(-1) * 0.0
            return zero.to(x_RD.dtype)
        counts = num_tokens_per_expert_E.to(torch.long)
        expert_R = torch.repeat_interleave(
            torch.arange(len(counts), device=x_RD.device), counts
        )
        x_valid = x_RD[:valid_rows]
        gate = F.silu(torch.einsum("rd,rdf->rf", x_valid, self.w1_EDF[expert_R]))
        up = torch.einsum("rd,rdf->rf", x_valid, self.w3_EDF[expert_R])
        hidden = gate * up
        if row_scale_R is not None:
            hidden = (hidden.float() * row_scale_R[:valid_rows].reshape(-1, 1)).to(
                hidden.dtype
            )
        out_valid = torch.einsum("rf,rfd->rd", hidden, self.w2_EFD[expert_R])
        if valid_rows < x_RD.shape[0]:
            return F.pad(out_valid, (0, 0, 0, x_RD.shape[0] - valid_rows))
        return out_valid


def _run_ep_roe_parity(device, rank, world_size, mode, fuse_score_bwd):
    bsz, seq_len, num_experts, top_k, dim = 2, 5, 8, 2, 16
    total_tokens = bsz * seq_len * world_size
    torch.manual_seed(42)
    global_x = torch.randn(total_tokens, dim, requires_grad=True, device=device)
    global_scores = (
        0.1 + torch.rand(total_tokens, top_k, device=device)
    ).requires_grad_()
    if mode == "skewed":
        # Route 100% of global tokens to rank 0's local experts (count == cap on rank 0, 0 elsewhere).
        global_expert_ids = torch.randint(0, 2, (total_tokens, top_k), device=device)
    else:
        global_expert_ids = torch.randint(
            0, num_experts, (total_tokens, top_k), device=device
        )
        if mode == "empty":
            global_expert_ids[global_expert_ids == 1] = 2

    local_start = rank * (bsz * seq_len)
    local_end = local_start + bsz * seq_len
    local_x = global_x[local_start:local_end].detach().clone().requires_grad_()
    local_scores = (
        global_scores[local_start:local_end].detach().clone().requires_grad_()
    )
    local_expert_ids = global_expert_ids[local_start:local_end]

    dispatcher = ReferenceRoETokenDispatcher.Config(
        num_experts=num_experts,
        top_k=top_k,
        ep_transport="ring_of_experts",
        fuse_route_score_backward=fuse_score_bwd,
    ).build()
    dispatcher.ep_mesh = init_device_mesh(
        device.type, (world_size,), mesh_dim_names=("ep",)
    )
    holder = _RankHolder(rank, device)
    dispatcher.bind_ep_rank_holder(holder)

    local_counts = torch.bincount(local_expert_ids.reshape(-1), minlength=num_experts)
    routed_input, num_global_tokens_e, metadata = dispatcher.dispatch(
        local_x, local_scores, local_expert_ids, local_counts
    )
    assert isinstance(metadata, RoEReferenceDispatchMetadata)

    global_counts = torch.bincount(
        global_expert_ids.reshape(-1), minlength=num_experts
    )
    local_e = num_experts // world_size
    expected_counts = global_counts[rank * local_e : (rank + 1) * local_e].to(
        torch.int32
    )
    torch.testing.assert_close(num_global_tokens_e, expected_counts)
    total_valid = int(expected_counts.sum().item())
    assert int(metadata.valid_count.item()) == total_valid

    # Inject NaNs past valid_count to ensure combine masks non-local / tail rows.
    if total_valid < routed_input.shape[0]:
        nan_tail = torch.full(
            (routed_input.shape[0] - total_valid, dim),
            float("nan"),
            dtype=routed_input.dtype,
            device=device,
        )
        routed_for_combine = torch.cat(
            [routed_input[:total_valid] * 2.0, nan_tail], dim=0
        )
    else:
        routed_for_combine = routed_input * 2.0
    combined = dispatcher.combine(routed_for_combine, metadata, local_x)

    torch.manual_seed(100 + rank)
    local_grad_out = torch.randn_like(combined)
    combined.backward(local_grad_out)

    ref_x = global_x.clone().detach().requires_grad_()
    ref_scores = global_scores.clone().detach().requires_grad_()
    ref_dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=num_experts,
        top_k=top_k,
        fuse_route_score_backward=fuse_score_bwd,
    ).build()
    ref_routed, _, ref_metadata = ref_dispatcher.dispatch(
        ref_x, ref_scores, global_expert_ids, global_counts
    )
    ref_combined = ref_dispatcher.combine(ref_routed * 2.0, ref_metadata, ref_x)
    global_grad_out = torch.zeros_like(ref_combined)
    global_grad_out[local_start:local_end] = local_grad_out
    dist.all_reduce(global_grad_out)
    ref_combined.backward(global_grad_out)

    assert torch.isfinite(combined).all()
    assert torch.isfinite(local_x.grad).all()
    assert torch.isfinite(local_scores.grad).all()
    torch.testing.assert_close(combined, ref_combined[local_start:local_end])
    torch.testing.assert_close(local_x.grad, ref_x.grad[local_start:local_end])
    torch.testing.assert_close(
        local_scores.grad, ref_scores.grad[local_start:local_end]
    )


@pytest.mark.parametrize(
    "ep_size, mode, fuse_score_bwd",
    [
        (2, "random", False),
        (2, "skewed", True),
        (2, "empty", False),
        (4, "random", True),
        (4, "skewed", False),
    ],
)
def test_roe_ep_matches_ep1(ep_size, mode, fuse_score_bwd):
    distributed.run_distributed(
        ep_size,
        device_type.AcceleratorDeviceType.CPU,
        _run_ep_roe_parity,
        mode,
        fuse_score_bwd,
    )


def _run_auto_ep_transport(device, rank, world_size, top_k, expect_roe):
    num_experts, tokens_per_rank, dim = 8, 6, 8
    total_tokens = tokens_per_rank * world_size
    torch.manual_seed(77)
    global_x = torch.randn(total_tokens, dim, requires_grad=True, device=device)
    global_scores = torch.rand(
        total_tokens, top_k, requires_grad=True, device=device
    )
    global_ids = torch.randint(0, num_experts, (total_tokens, top_k), device=device)

    s = rank * tokens_per_rank
    e = s + tokens_per_rank
    local_x = global_x[s:e].detach().clone().requires_grad_()
    local_scores = global_scores[s:e].detach().clone().requires_grad_()
    local_ids = global_ids[s:e]
    local_counts = torch.bincount(local_ids.reshape(-1), minlength=num_experts)

    dispatcher = ReferenceRoETokenDispatcher.Config(
        num_experts=num_experts, top_k=top_k, ep_transport="auto"
    ).build()
    dispatcher.ep_mesh = init_device_mesh(
        device.type, (world_size,), mesh_dim_names=("ep",)
    )
    holder = _RankHolder(rank, device)
    dispatcher.bind_ep_rank_holder(holder)

    routed, _, metadata = dispatcher.dispatch(
        local_x, local_scores, local_ids, local_counts
    )
    expected_cls = (
        RoEReferenceDispatchMetadata if expect_roe else TpuRaggedDispatchMetadata
    )
    assert isinstance(metadata, expected_cls)
    combined = dispatcher.combine(routed * 1.5, metadata, local_x)

    torch.manual_seed(200 + rank)
    local_grad = torch.randn_like(combined)
    combined.backward(local_grad)

    ref_x = global_x.detach().clone().requires_grad_()
    ref_scores = global_scores.detach().clone().requires_grad_()
    ref_counts = torch.bincount(global_ids.reshape(-1), minlength=num_experts)
    ref_disp = TpuAllToAllTokenDispatcher.Config(
        num_experts=num_experts, top_k=top_k
    ).build()
    ref_routed, _, ref_meta = ref_disp.dispatch(
        ref_x, ref_scores, global_ids, ref_counts
    )
    ref_out = ref_disp.combine(ref_routed * 1.5, ref_meta, ref_x)
    global_grad = torch.zeros_like(ref_out)
    global_grad[s:e] = local_grad
    dist.all_reduce(global_grad)
    ref_out.backward(global_grad)

    torch.testing.assert_close(combined, ref_out[s:e])
    torch.testing.assert_close(local_x.grad, ref_x.grad[s:e])
    torch.testing.assert_close(local_scores.grad, ref_scores.grad[s:e])


@pytest.mark.parametrize(
    "ep_size, top_k, expect_roe",
    [
        (2, 2, True),
        (4, 2, False),
        (4, 4, True),
    ],
)
def test_auto_ep_transport_matches_ep1(ep_size, top_k, expect_roe):
    distributed.run_distributed(
        ep_size,
        device_type.AcceleratorDeviceType.CPU,
        _run_auto_ep_transport,
        top_k,
        expect_roe,
    )


def _run_ep_scale_before_down_proj(device, rank, world_size, ep_transport):
    num_experts, top_k, tokens_per_rank, dim, hidden = 8, 2, 6, 8, 12
    total_tokens = tokens_per_rank * world_size
    local_e = num_experts // world_size
    torch.manual_seed(123)
    w1_full = 0.2 * torch.randn(num_experts, dim, hidden, device=device)
    w3_full = 0.2 * torch.randn(num_experts, dim, hidden, device=device)
    w2_full = 0.2 * torch.randn(num_experts, hidden, dim, device=device)
    global_x = torch.randn(total_tokens, dim, device=device)
    global_scores = 0.1 + torch.rand(total_tokens, top_k, device=device)
    global_ids = torch.randint(0, num_experts, (total_tokens, top_k), device=device)
    # Include skewed routes on the first two tokens to exercise unbalanced ranks.
    global_ids[:2] = 0

    s = rank * tokens_per_rank
    e = s + tokens_per_rank
    local_x = global_x[s:e].clone().requires_grad_()
    local_scores = global_scores[s:e].clone().requires_grad_()
    local_ids = global_ids[s:e]
    local_counts = torch.bincount(local_ids.reshape(-1), minlength=num_experts)

    routed = TpuRoutedExperts.Config(
        inner_experts=GroupedExperts.Config(
            dim=dim, hidden_dim=hidden, num_experts=local_e
        ),
        token_dispatcher=ReferenceRoETokenDispatcher.Config(
            num_experts=num_experts, top_k=top_k, ep_transport=ep_transport
        ),
        scale_before_down_proj=True,
    ).build()
    routed.inner_experts = _RaggedSwiGLUExperts(
        w1_full[rank * local_e : (rank + 1) * local_e].clone(),
        w3_full[rank * local_e : (rank + 1) * local_e].clone(),
        w2_full[rank * local_e : (rank + 1) * local_e].clone(),
    )
    ep_mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("ep",))
    routed.token_dispatcher.ep_mesh = ep_mesh
    routed.register_buffer(
        "_ep_local_rank",
        torch.tensor([rank], dtype=torch.long, device=device),
        persistent=False,
    )
    routed.token_dispatcher.bind_ep_rank_holder(routed)

    out = routed(local_x, local_scores, local_ids, local_counts)
    torch.manual_seed(300 + rank)
    local_grad = torch.randn_like(out)
    out.backward(local_grad)

    ref_routed = TpuRoutedExperts.Config(
        inner_experts=GroupedExperts.Config(
            dim=dim, hidden_dim=hidden, num_experts=num_experts
        ),
        token_dispatcher=TpuAllToAllTokenDispatcher.Config(
            num_experts=num_experts, top_k=top_k
        ),
        scale_before_down_proj=True,
    ).build()
    ref_routed.inner_experts = _RaggedSwiGLUExperts(
        w1_full.clone(), w3_full.clone(), w2_full.clone()
    )
    ref_x = global_x.clone().requires_grad_()
    ref_scores = global_scores.clone().requires_grad_()
    ref_counts = torch.bincount(global_ids.reshape(-1), minlength=num_experts)
    ref_out = ref_routed(ref_x, ref_scores, global_ids, ref_counts)
    global_grad = torch.zeros_like(ref_out)
    global_grad[s:e] = local_grad
    dist.all_reduce(global_grad)
    ref_out.backward(global_grad)

    torch.testing.assert_close(out, ref_out[s:e], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(local_x.grad, ref_x.grad[s:e], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        local_scores.grad, ref_scores.grad[s:e], rtol=1e-5, atol=1e-5
    )
    es = rank * local_e
    ee = (rank + 1) * local_e
    torch.testing.assert_close(
        routed.inner_experts.w1_EDF.grad,
        ref_routed.inner_experts.w1_EDF.grad[es:ee],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        routed.inner_experts.w3_EDF.grad,
        ref_routed.inner_experts.w3_EDF.grad[es:ee],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        routed.inner_experts.w2_EFD.grad,
        ref_routed.inner_experts.w2_EFD.grad[es:ee],
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "ep_size, ep_transport",
    [
        (2, "ring_of_experts"),
        (4, "ring_of_experts"),
        (2, "ragged_all_to_all"),
        (4, "ragged_all_to_all"),
        (2, "auto"),
        (4, "auto"),
    ],
)
def test_ep_scale_before_down_proj_matches_ep1(ep_size, ep_transport):
    distributed.run_distributed(
        ep_size,
        device_type.AcceleratorDeviceType.CPU,
        _run_ep_scale_before_down_proj,
        ep_transport,
    )


def test_auto_transport_resolution():
    x = torch.zeros(2, 128)
    reference = ReferenceRoETokenDispatcher.Config(
        num_experts=16, top_k=4, ep_transport="auto"
    ).build()
    assert reference._resolve_ep_transport(2, x) == "ring_of_experts"
    assert reference._resolve_ep_transport(4, x) == "ring_of_experts"
    assert reference._resolve_ep_transport(8, x) == "ragged_all_to_all"

    dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=16, top_k=4, ep_transport="auto"
    ).build()
    # Off TPU the Pallas Ring-of-Experts cannot run, so auto picks ragged.
    assert dispatcher._resolve_ep_transport(2, x) == "ragged_all_to_all"
    with mock.patch.object(
        dispatcher, "ring_of_experts_unsupported_reason", return_value=None
    ):
        assert dispatcher._resolve_ep_transport(4, x) == "ring_of_experts"
        assert dispatcher._resolve_ep_transport(8, x) == "ragged_all_to_all"


def test_ring_of_experts_unsupported_reason():
    dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=16, top_k=4, ep_transport="ring_of_experts"
    ).build()
    assert dispatcher.ring_of_experts_unsupported_reason("tpu", 256) is None
    assert dispatcher.ring_of_experts_unsupported_reason("tpu") is None
    assert "needs a TPU" in dispatcher.ring_of_experts_unsupported_reason(
        "cpu", 256
    )
    assert "% 128" in dispatcher.ring_of_experts_unsupported_reason("tpu", 96)


def test_explicit_ring_of_experts_raises_when_pallas_cannot_run():
    dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=4, top_k=2, ep_transport="ring_of_experts"
    ).build()
    ep_mesh = mock.Mock()
    ep_mesh.size.return_value = 2
    dispatcher.ep_mesh = ep_mesh
    holder = _RankHolder(0)  # bind_ep_rank_holder keeps only a weak reference
    dispatcher.bind_ep_rank_holder(holder)
    with pytest.raises(ValueError, match="ring_of_experts.*needs a TPU"):
        dispatcher.dispatch(
            torch.randn(2, 128),
            torch.randn(2, 2),
            torch.zeros(2, 2, dtype=torch.long),
            torch.tensor([4, 0, 0, 0]),
        )
    with mock.patch.object(
        dispatcher,
        "ring_of_experts_unsupported_reason",
        side_effect=lambda device_type, hidden_dim=None: (
            TpuAllToAllTokenDispatcher.ring_of_experts_unsupported_reason(
                dispatcher, "tpu", hidden_dim
            )
        ),
    ):
        with pytest.raises(ValueError, match="hidden_dim % 128"):
            dispatcher.dispatch(
                torch.randn(2, 96),
                torch.randn(2, 2),
                torch.zeros(2, 2, dtype=torch.long),
                torch.tensor([4, 0, 0, 0]),
            )


@pytest.mark.parametrize(
    "factor, num_routed, ep_size, align, expected",
    [
        (None, 512, 4, 1024, 2048),  # dropless worst case, never rounded
        (2.0, 512, 4, 1024, 1024),  # already aligned
        (3.0, 512, 4, 1024, 2048),  # 1536 -> 2048, clamped to worst case
        (1.25, 2048, 2, 1024, 3072),  # 2560 -> 3072
        (0.01, 2048, 2, 1024, 1024),  # 20 -> 1024
        (1.5, 4096, 8, 1024, 6144),
        (0.75, 8, 4, 1, 6),  # no alignment (ragged all-to-all, reference)
        (3.0, 512, 4, 1, 1536),
        (2.0, 0, 4, 1024, 1),  # no routes
    ],
)
def test_ep_buffer_capacity_rounding(factor, num_routed, ep_size, align, expected):
    dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=8, top_k=2, ep_capacity_factor=factor
    ).build()
    cap = dispatcher._ep_buffer_capacity(num_routed, ep_size, align_rows=align)
    assert cap == expected
    assert cap <= ep_size * max(num_routed, 1)
    if factor is not None and num_routed > 0 and cap < ep_size * num_routed:
        assert cap % align == 0


def test_roe_metadata_is_not_local_metadata():
    assert not issubclass(TpuRoEDispatchMetadata, LocalDispatchMetadata)


@pytest.mark.parametrize("ep_transport", ["ring_of_experts", "auto"])
def test_roe_dispatch_without_rank_owner_raises(ep_transport):
    dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=4, top_k=2, ep_transport=ep_transport
    ).build()
    ep_mesh = mock.Mock()
    ep_mesh.size.return_value = 2
    dispatcher.ep_mesh = ep_mesh
    with pytest.raises(RuntimeError, match="bound to a"):
        dispatcher.dispatch(
            torch.randn(2, 4),
            torch.randn(2, 2),
            torch.zeros(2, 2, dtype=torch.long),
            torch.tensor([4, 0, 0, 0]),
        )


def test_roe_tracing_off_tpu_raises_instead_of_repeating_shards():
    from torchtitan.experiments.tpu.components.token_dispatcher import (
        _EpAllGather,
        _EpReduceScatter,
    )

    ep_mesh = mock.Mock()
    ep_mesh.size.return_value = 2
    ep_mesh.get_local_rank.return_value = 0
    ep_mesh.get_group.return_value = mock.Mock()

    with pytest.raises(NotImplementedError, match="Ring-of-Experts"):
        make_fx(lambda x: _EpAllGather.apply(x, ep_mesh, ep_mesh.get_group()))(
            torch.randn(2, 4)
        )
    with pytest.raises(NotImplementedError, match="Ring-of-Experts"):
        make_fx(
            lambda x: _EpReduceScatter.apply(x, ep_mesh, ep_mesh.get_group())
        )(torch.randn(4, 4))

    dispatcher = ReferenceRoETokenDispatcher.Config(
        num_experts=4, top_k=2, ep_transport="ring_of_experts"
    ).build()
    dispatcher.ep_mesh = ep_mesh
    holder = _RankHolder(0)
    dispatcher.bind_ep_rank_holder(holder)

    def step(x, scores, ids, counts):
        return dispatcher.dispatch(x, scores, ids, counts)

    with pytest.raises(NotImplementedError):
        make_fx(step)(
            torch.randn(2, 4),
            torch.randn(2, 2),
            torch.zeros(2, 2, dtype=torch.long),
            torch.tensor([4, 0, 0, 0]),
        )


@pytest.mark.parametrize(
    "ep_transport, use_pallas, expected_zero_pad",
    [
        ("ring_of_experts", False, True),
        ("ring_of_experts", True, False),
        ("auto", False, True),
        ("auto", True, False),
    ],
)
def test_parallelize_registers_rank_buffer_and_sets_tail_zero_pad(
    ep_transport, use_pallas, expected_zero_pad
):
    routed = TpuRoutedExperts.Config(
        inner_experts=TpuGroupedExperts.Config(
            dim=128,
            hidden_dim=128,
            num_experts=4,
            use_pallas_grouped_mm=use_pallas,
        ),
        token_dispatcher=TpuAllToAllTokenDispatcher.Config(
            num_experts=4, top_k=2, ep_transport=ep_transport
        ),
    ).build()
    mesh = mock.Mock()
    mesh.device_type = "tpu"
    mesh.get_local_rank.return_value = 1
    parallel_dims = mock.Mock()
    parallel_dims.get_optional_mesh.side_effect = (
        lambda name: mesh if name == "ep" else None
    )
    with mock.patch("torchtitan.models.common.moe.RoutedExperts.parallelize"):
        routed.parallelize(parallel_dims)
    assert "_ep_local_rank" in dict(routed.named_buffers())
    assert int(routed._ep_local_rank.item()) == 1
    assert routed.token_dispatcher.roe_zero_pad_tail is expected_zero_pad


@pytest.mark.parametrize(
    "ep_transport, raises",
    [("ring_of_experts", True), ("auto", False), ("ragged_all_to_all", False)],
)
def test_parallelize_rejects_explicit_ring_of_experts_off_tpu(
    ep_transport, raises
):
    routed = TpuRoutedExperts.Config(
        inner_experts=GroupedExperts.Config(dim=128, hidden_dim=128, num_experts=4),
        token_dispatcher=TpuAllToAllTokenDispatcher.Config(
            num_experts=4, top_k=2, ep_transport=ep_transport
        ),
    ).build()
    mesh = mock.Mock()
    mesh.device_type = "cpu"
    mesh.get_local_rank.return_value = 0
    parallel_dims = mock.Mock()
    parallel_dims.get_optional_mesh.side_effect = (
        lambda name: mesh if name == "ep" else None
    )
    with mock.patch(
        "torchtitan.models.common.moe.RoutedExperts.parallelize"
    ) as parent_parallelize:
        if raises:
            with pytest.raises(ValueError, match="ring_of_experts.*needs a TPU"):
                routed.parallelize(parallel_dims)
            parent_parallelize.assert_not_called()
        else:
            routed.parallelize(parallel_dims)
            parent_parallelize.assert_called_once()


@pytest.mark.parametrize("cap_factor", [None, 1.5])
def test_drop_stats_stay_off_until_configured(cap_factor):
    routed = TpuRoutedExperts.Config(
        inner_experts=GroupedExperts.Config(dim=16, hidden_dim=16, num_experts=4),
        token_dispatcher=TpuAllToAllTokenDispatcher.Config(
            num_experts=4,
            top_k=2,
            ep_transport="ring_of_experts",
            ep_capacity_factor=cap_factor,
        ),
    ).build()
    mesh = mock.Mock()
    mesh.device_type = "tpu"
    mesh.get_local_rank.return_value = 0
    parallel_dims = mock.Mock()
    parallel_dims.get_optional_mesh.side_effect = (
        lambda name: mesh if name == "ep" else None
    )
    with mock.patch("torchtitan.models.common.moe.RoutedExperts.parallelize"):
        routed.parallelize(parallel_dims)

    # Off after parallelize: only a trainer that returns the counts from its
    # step turns them on, and only capacity-bounded layers have any.
    assert routed.collect_ep_drop_stats is False
    capacity_bounded = cap_factor is not None
    assert ep_capacity_bounded([routed]) is capacity_bounded
    assert configure_ep_drop_stats([routed], enabled=True) is capacity_bounded
    assert routed.collect_ep_drop_stats is capacity_bounded
    if capacity_bounded:
        with pytest.raises(RuntimeError, match="did not record"):
            collect_ep_drop_stat_tensors([routed])
    else:
        assert collect_ep_drop_stat_tensors([routed]) == []
    assert configure_ep_drop_stats([routed], enabled=False) is False
    assert routed.collect_ep_drop_stats is False


def test_traced_dropped_count_is_kept_only_for_a_collecting_owner():
    from torch._subclasses.fake_tensor import FakeTensorMode

    dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=4, top_k=2, ep_transport="ring_of_experts", ep_capacity_factor=1.0
    ).build()
    holder = _RankHolder(0)
    holder.collect_ep_drop_stats = False
    dispatcher.bind_ep_rank_holder(holder)
    with FakeTensorMode():
        x = torch.empty(2, 4)
        dropped = torch.zeros((), dtype=torch.int32)

    # A traced count must not outlive its trace unless the step returns it.
    dispatcher._record_dropped_count(x, dropped)
    assert dispatcher.last_dropped_count is None
    holder.collect_ep_drop_stats = True
    dispatcher._record_dropped_count(x, dropped)
    assert dispatcher.last_dropped_count is dropped


def test_use_expert_parallel_defaults_to_auto_and_handles_scale_before_down_proj():
    class _ModelCfg:
        def __init__(self, routed):
            self.routed = routed

        def traverse(self, config_type):
            if isinstance(self.routed, config_type):
                yield "routed", self.routed, self, "routed"
            if isinstance(self.routed.inner_experts, config_type):
                yield (
                    "routed.inner_experts",
                    self.routed.inner_experts,
                    self.routed,
                    "inner_experts",
                )
            if isinstance(self.routed.token_dispatcher, config_type):
                yield (
                    "routed.token_dispatcher",
                    self.routed.token_dispatcher,
                    self.routed,
                    "token_dispatcher",
                )

    cfg = _ModelCfg(
        RoutedExperts.Config(
            inner_experts=TpuGroupedExperts.Config(
                dim=128,
                hidden_dim=128,
                num_experts=8,
                use_pallas_grouped_mm=True,
                use_pallas_paired_grouped_mm=True,
            ),
            token_dispatcher=TpuAllToAllTokenDispatcher.Config(
                num_experts=8, top_k=2, fuse_route_score_backward=True
            ),
        )
    )
    use_expert_parallel(
        cfg,
        ep_degree=2,
        scale_before_down_proj=True,
        ep_capacity_factor=1.5,
    )
    assert isinstance(cfg.routed, TpuRoutedExperts.Config)
    assert cfg.routed.scale_before_down_proj is True
    assert cfg.routed.token_dispatcher.ep_transport == "auto"
    assert cfg.routed.token_dispatcher.ep_capacity_factor == 1.5
    assert cfg.routed.token_dispatcher.fuse_route_score_backward is False
    # The paired W1+W3 kernel skips rows past the routed count, so it stays on.
    assert cfg.routed.inner_experts.use_pallas_paired_grouped_mm is True
    use_expert_parallel(cfg, ep_degree=2, paired_grouped_mm=False)
    assert cfg.routed.inner_experts.use_pallas_paired_grouped_mm is False

    # Explicit non-default transport and capacity factor on dispatcher are
    # preserved when overrides are None.
    cfg2 = _ModelCfg(
        TpuRoutedExperts.Config(
            inner_experts=GroupedExperts.Config(dim=16, hidden_dim=16, num_experts=8),
            token_dispatcher=TpuAllToAllTokenDispatcher.Config(
                num_experts=8,
                top_k=2,
                ep_transport="ragged_all_to_all",
                ep_capacity_factor=2.0,
            ),
        )
    )
    use_expert_parallel(cfg2, ep_degree=2)
    assert cfg2.routed.token_dispatcher.ep_transport == "ragged_all_to_all"
    assert cfg2.routed.token_dispatcher.ep_capacity_factor == 2.0


@pytest.mark.parametrize(
    "inner_experts",
    [
        # Stock experts: no TpuGroupedExperts.Config in the tree at all.
        GroupedExperts.Config(dim=16, hidden_dim=16, num_experts=8),
        TpuGroupedExperts.Config(
            dim=128, hidden_dim=128, num_experts=8, use_pallas_grouped_mm=False
        ),
    ],
)
def test_use_expert_parallel_scale_before_down_proj_needs_pallas_gmm(inner_experts):
    cfg = _routed_model_cfg(TpuAllToAllTokenDispatcher.Config(num_experts=8, top_k=2))
    cfg.routed.inner_experts = inner_experts
    with pytest.raises(ValueError, match="needs the Pallas grouped matmuls"):
        use_expert_parallel(cfg, ep_degree=2, scale_before_down_proj=True)


def _routed_model_cfg(token_dispatcher):
    class _ModelCfg:
        def __init__(self, routed):
            self.routed = routed

        def traverse(self, config_type):
            if isinstance(self.routed, config_type):
                yield "routed", self.routed, self, "routed"
            if isinstance(self.routed.token_dispatcher, config_type):
                yield (
                    "routed.token_dispatcher",
                    self.routed.token_dispatcher,
                    self.routed,
                    "token_dispatcher",
                )

    return _ModelCfg(
        RoutedExperts.Config(
            inner_experts=GroupedExperts.Config(dim=16, hidden_dim=16, num_experts=8),
            token_dispatcher=token_dispatcher,
        )
    )


@pytest.mark.parametrize(
    "dispatcher_transport, override, expected",
    [
        # Unset (None) on the TPU config, or no field on upstream's: "auto".
        (None, None, "auto"),
        ("upstream", None, "auto"),
        # Any explicit choice is kept, including upstream's "all_to_all".
        ("all_to_all", None, "all_to_all"),
        ("ring_of_experts", None, "ring_of_experts"),
        ("ragged_all_to_all", None, "ragged_all_to_all"),
        ("auto", None, "auto"),
        # An explicit override applies to every dispatcher.
        ("all_to_all", "ragged_all_to_all", "ragged_all_to_all"),
        (None, "all_to_all", "all_to_all"),
        ("upstream", "ring_of_experts", "ring_of_experts"),
    ],
)
def test_use_expert_parallel_keeps_explicit_transport(
    dispatcher_transport, override, expected
):
    if dispatcher_transport == "upstream":
        dispatcher = AllToAllTokenDispatcher.Config(num_experts=8, top_k=2)
    else:
        dispatcher = TpuAllToAllTokenDispatcher.Config(
            num_experts=8, top_k=2, ep_transport=dispatcher_transport
        )
    cfg = _routed_model_cfg(dispatcher)
    use_expert_parallel(cfg, ep_degree=2, ep_transport=override)
    assert isinstance(cfg.routed.token_dispatcher, TpuAllToAllTokenDispatcher.Config)
    assert cfg.routed.token_dispatcher.ep_transport == expected


def test_unset_ep_transport_runs_upstream_all_to_all():
    config = TpuAllToAllTokenDispatcher.Config(num_experts=8, top_k=2)
    assert config.ep_transport is None
    dispatcher = config.build()
    assert dispatcher.ep_transport == "all_to_all"
    with pytest.raises(ValueError, match="ep_transport must be"):
        TpuAllToAllTokenDispatcher.Config(
            num_experts=8, top_k=2, ep_transport="ring"  # type: ignore[arg-type]
        ).build()


@pytest.mark.parametrize(
    "bad_factor",
    [0.0, -1.0, float("nan"), float("inf"), -float("inf"), True, False],
)
def test_invalid_ep_capacity_factor_raises(bad_factor):
    with pytest.raises(ValueError, match="ep_capacity_factor"):
        TpuAllToAllTokenDispatcher.Config(
            num_experts=8,
            top_k=2,
            ep_transport="ring_of_experts",
            ep_capacity_factor=bad_factor,  # type: ignore[arg-type]
        ).build()


def _run_ep_capacity_no_overflow(
    device, rank, world_size, ep_transport, scale_before_w2
):
    num_experts, top_k, tokens_per_rank, dim, hidden = 8, 2, 6, 8, 12
    total_tokens = tokens_per_rank * world_size
    local_e = num_experts // world_size
    balanced_rows = tokens_per_rank * top_k  # 12
    capacity_factor = 2.0
    expected_cap = int(balanced_rows * capacity_factor)  # 24 (< worst-case 48)

    torch.manual_seed(321)
    w1_full = 0.2 * torch.randn(num_experts, dim, hidden, device=device)
    w3_full = 0.2 * torch.randn(num_experts, dim, hidden, device=device)
    w2_full = 0.2 * torch.randn(num_experts, hidden, dim, device=device)
    global_x = torch.randn(total_tokens, dim, device=device)
    global_scores = 0.1 + torch.rand(total_tokens, top_k, device=device)
    # Deterministic balanced-enough routing: each rank gets <= 18 <= 24 slots.
    base_ids = torch.arange(total_tokens * top_k, device=device) % num_experts
    global_ids = base_ids.reshape(total_tokens, top_k)

    s = rank * tokens_per_rank
    e = s + tokens_per_rank
    local_x = global_x[s:e].clone().requires_grad_()
    local_scores = global_scores[s:e].clone().requires_grad_()
    local_ids = global_ids[s:e]
    local_counts = torch.bincount(local_ids.reshape(-1), minlength=num_experts)

    routed = TpuRoutedExperts.Config(
        inner_experts=GroupedExperts.Config(
            dim=dim, hidden_dim=hidden, num_experts=local_e
        ),
        token_dispatcher=ReferenceRoETokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
            ep_transport=ep_transport,
            ep_capacity_factor=capacity_factor,
            fuse_route_score_backward=not scale_before_w2,
        ),
        scale_before_down_proj=scale_before_w2,
    ).build()
    routed.inner_experts = _RaggedSwiGLUExperts(
        w1_full[rank * local_e : (rank + 1) * local_e].clone(),
        w3_full[rank * local_e : (rank + 1) * local_e].clone(),
        w2_full[rank * local_e : (rank + 1) * local_e].clone(),
    )
    ep_mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("ep",))
    routed.token_dispatcher.ep_mesh = ep_mesh
    routed.register_buffer(
        "_ep_local_rank",
        torch.tensor([rank], dtype=torch.long, device=device),
        persistent=False,
    )
    routed.token_dispatcher.bind_ep_rank_holder(routed)
    # As TpuRoutedExperts.parallelize and TpuGraphTrainer do.
    routed._ep_mesh = ep_mesh
    assert configure_ep_drop_stats([routed], enabled=True)

    disp_in, _, disp_meta = routed.token_dispatcher.dispatch(
        local_x.detach(), local_scores.detach(), local_ids, local_counts
    )
    assert disp_in.shape == (expected_cap, dim)
    assert disp_meta.dropped_count is not None
    assert int(disp_meta.dropped_count.item()) == 0

    out = routed(local_x, local_scores, local_ids, local_counts)
    assert routed.token_dispatcher.last_dropped_count is not None
    assert int(routed.token_dispatcher.last_dropped_count.item()) == 0
    (stats,) = collect_ep_drop_stat_tensors([routed])
    assert stats.dtype == torch.int32
    assert stats.tolist() == [[0, tokens_per_rank * top_k]]  # [[0, 12]]
    torch.manual_seed(500 + rank)
    local_grad = torch.randn_like(out)
    out.backward(local_grad)

    ref_routed = TpuRoutedExperts.Config(
        inner_experts=GroupedExperts.Config(
            dim=dim, hidden_dim=hidden, num_experts=num_experts
        ),
        token_dispatcher=TpuAllToAllTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
            fuse_route_score_backward=not scale_before_w2,
        ),
        scale_before_down_proj=scale_before_w2,
    ).build()
    ref_routed.inner_experts = _RaggedSwiGLUExperts(
        w1_full.clone(), w3_full.clone(), w2_full.clone()
    )
    ref_x = global_x.clone().requires_grad_()
    ref_scores = global_scores.clone().requires_grad_()
    ref_counts = torch.bincount(global_ids.reshape(-1), minlength=num_experts)
    ref_out = ref_routed(ref_x, ref_scores, global_ids, ref_counts)
    global_grad = torch.zeros_like(ref_out)
    global_grad[s:e] = local_grad
    dist.all_reduce(global_grad)
    ref_out.backward(global_grad)

    torch.testing.assert_close(out, ref_out[s:e], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(local_x.grad, ref_x.grad[s:e], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        local_scores.grad, ref_scores.grad[s:e], rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        routed.inner_experts.w2_EFD.grad,
        ref_routed.inner_experts.w2_EFD.grad[rank * local_e : (rank + 1) * local_e],
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "ep_transport", ["ring_of_experts", "ragged_all_to_all"]
)
@pytest.mark.parametrize("scale_before_w2", [False, True])
def test_ep_capacity_factor_no_overflow_matches_ep1(
    ep_transport, scale_before_w2
):
    distributed.run_distributed(
        4,
        device_type.AcceleratorDeviceType.CPU,
        _run_ep_capacity_no_overflow,
        ep_transport,
        scale_before_w2,
    )


def _run_ep_capacity_forced_overflow(device, rank, world_size, scale_before_w2):
    # 4 ranks, 8 experts (2 per rank), top_k=2, 4 tokens/rank -> balanced=8.
    # ep_capacity_factor=0.75 -> cap=6 rows per rank (vs 32 worst-case).
    num_experts, top_k, tokens_per_rank, dim, hidden = 8, 2, 4, 8, 12
    total_tokens = tokens_per_rank * world_size  # 16
    local_e = num_experts // world_size  # 2
    capacity_factor = 0.75
    expected_cap = 6

    torch.manual_seed(909)
    w1_full = 0.2 * torch.randn(num_experts, dim, hidden, device=device)
    w3_full = 0.2 * torch.randn(num_experts, dim, hidden, device=device)
    w2_full = 0.2 * torch.randn(num_experts, hidden, dim, device=device)
    global_x = torch.randn(total_tokens, dim, device=device)
    global_scores = 0.1 + torch.rand(total_tokens, top_k, device=device)
    # Route all 32 global slots to rank 0's experts (0 and 1):
    # 4 slots -> expert 0 (1 per rank: token 0 slot 0 on each rank),
    # 28 slots -> expert 1.
    # Rank 0 keeps all 4 of expert 0 + first 2 of expert 1 (both from rank 0!),
    # dropping 26 slots (2 from rank 0, and 6 from each of ranks 1..3).
    global_ids = torch.ones((total_tokens, top_k), dtype=torch.long, device=device)
    for r in range(world_size):
        global_ids[r * tokens_per_rank, 0] = 0

    s = rank * tokens_per_rank
    e = s + tokens_per_rank
    local_ids = global_ids[s:e]
    local_counts = torch.bincount(local_ids.reshape(-1), minlength=num_experts)

    ep_mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("ep",))
    torch.manual_seed(700 + rank)
    local_grad = torch.randn(tokens_per_rank, dim, device=device)

    results = {}
    for transport in ("ring_of_experts", "ragged_all_to_all"):
        x_in = global_x[s:e].clone().requires_grad_()
        sc_in = global_scores[s:e].clone().requires_grad_()
        routed = TpuRoutedExperts.Config(
            inner_experts=GroupedExperts.Config(
                dim=dim, hidden_dim=hidden, num_experts=local_e
            ),
            token_dispatcher=ReferenceRoETokenDispatcher.Config(
                num_experts=num_experts,
                top_k=top_k,
                ep_transport=transport,
                ep_capacity_factor=capacity_factor,
                fuse_route_score_backward=not scale_before_w2,
            ),
            scale_before_down_proj=scale_before_w2,
        ).build()
        routed.inner_experts = _RaggedSwiGLUExperts(
            w1_full[rank * local_e : (rank + 1) * local_e].clone(),
            w3_full[rank * local_e : (rank + 1) * local_e].clone(),
            w2_full[rank * local_e : (rank + 1) * local_e].clone(),
        )
        routed.token_dispatcher.ep_mesh = ep_mesh
        routed.register_buffer(
            "_ep_local_rank",
            torch.tensor([rank], dtype=torch.long, device=device),
            persistent=False,
        )
        routed.token_dispatcher.bind_ep_rank_holder(routed)
        routed._ep_mesh = ep_mesh
        assert configure_ep_drop_stats([routed], enabled=True)

        disp_in, counts_e, meta = routed.token_dispatcher.dispatch(
            x_in.detach(), sc_in.detach(), local_ids, local_counts
        )
        assert disp_in.shape == (expected_cap, dim)
        assert meta.valid_count is not None
        assert meta.dropped_count is not None
        if rank == 0:
            assert counts_e.tolist() == [4, 2]
            assert int(meta.valid_count.item()) == 6
            assert int(meta.dropped_count.item()) == 26
        else:
            assert counts_e.tolist() == [0, 0]
            assert int(meta.valid_count.item()) == 0
            assert int(meta.dropped_count.item()) == 0

        out = routed(x_in, sc_in, local_ids, local_counts)
        assert routed.token_dispatcher.last_dropped_count is not None
        assert int(routed.token_dispatcher.last_dropped_count.item()) == (
            26 if rank == 0 else 0
        )
        # Dropped routes are counted on the receiving rank (all on rank 0) and
        # the total on the sending rank, so only the EP-group sum, 26/32, is a
        # rate. Neither transport rounds this cap (the reference RoE has no
        # Pallas block alignment), so both drop the same routes.
        assert collect_ep_drop_stat_tensors([routed])[0].tolist() == [
            [26 if rank == 0 else 0, tokens_per_rank * top_k]
        ]
        out.backward(local_grad)

        # On ranks 1..3, only token 0 slot 0 (expert 0) was kept; tokens 1..3
        # had both expert-1 slots dropped, so their output, dX, and dScores are 0.
        if rank > 0:
            assert torch.equal(out[1:], torch.zeros_like(out[1:]))
            assert torch.equal(x_in.grad[1:], torch.zeros_like(x_in.grad[1:]))
            assert torch.equal(sc_in.grad[:, 1], torch.zeros_like(sc_in.grad[:, 1]))
            assert torch.equal(sc_in.grad[1:, 0], torch.zeros_like(sc_in.grad[1:, 0]))
        else:
            # On rank 0, tokens 2 and 3 had both expert-1 slots dropped
            # (rank 0 sent 7 expert-1 slots; only the first 2 were kept).
            assert torch.equal(out[2:], torch.zeros_like(out[2:]))
            assert torch.equal(x_in.grad[2:], torch.zeros_like(x_in.grad[2:]))
            assert torch.equal(sc_in.grad[2:], torch.zeros_like(sc_in.grad[2:]))

        results[transport] = (
            out.detach(),
            x_in.grad.detach(),
            sc_in.grad.detach(),
            routed.inner_experts.w1_EDF.grad.detach(),
            routed.inner_experts.w2_EFD.grad.detach(),
        )

    roe_res = results["ring_of_experts"]
    ra2a_res = results["ragged_all_to_all"]
    for roe_t, ra2a_t in zip(roe_res, ra2a_res, strict=True):
        torch.testing.assert_close(roe_t, ra2a_t, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("scale_before_w2", [False, True])
def test_ep_capacity_factor_forced_overflow_roe_matches_ragged_a2a(
    scale_before_w2,
):
    distributed.run_distributed(
        4,
        device_type.AcceleratorDeviceType.CPU,
        _run_ep_capacity_forced_overflow,
        scale_before_w2,
    )

