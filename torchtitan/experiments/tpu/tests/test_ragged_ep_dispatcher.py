"""Tests for the ragged all-to-all EP transport of TpuAllToAllTokenDispatcher.

The multi-process tests run EP=2 and EP=4 on CPU (gloo), where the exchange
uses the slow eager reference built from all-gathers, and compare against the
EP=1 local path run on the whole batch.
"""

from unittest import mock

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRoutedExperts,
)
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.distributed import distributed
from torchtitan.models.common.moe import GroupedExperts


class _RankHolder(torch.nn.Module):
    """Stands in for the TpuRoutedExperts that owns the EP rank buffer."""

    def __init__(self, rank: int, device=None):
        super().__init__()
        self.register_buffer(
            "_ep_local_rank", torch.tensor([rank], dtype=torch.long, device=device)
        )


def _test_ep_ragged_all_to_all(device, rank, world_size, ep_size, mode):
    del ep_size  # == world_size
    bsz, seq_len, num_experts, top_k, dim = 2, 5, 8, 2, 16
    total_tokens = bsz * seq_len * world_size
    torch.manual_seed(42)
    global_x = torch.randn(total_tokens, dim, requires_grad=True, device=device)
    global_scores = torch.rand(
        total_tokens, top_k, requires_grad=True, device=device
    )
    if mode == "skewed":
        # Every route goes to the first two experts (both on rank 0).
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

    dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=num_experts, top_k=top_k, ep_transport="ragged_all_to_all"
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

    global_counts = torch.bincount(
        global_expert_ids.reshape(-1), minlength=num_experts
    )
    local_e = num_experts // world_size
    torch.testing.assert_close(
        num_global_tokens_e,
        global_counts[rank * local_e : (rank + 1) * local_e].to(torch.int32),
    )

    # Rows past the routed count are stale in a real receive buffer; make them
    # NaN so any read of them shows up in the outputs or gradients.
    total_valid = int(num_global_tokens_e.sum().item())
    routed_input_mod = routed_input.clone()
    if total_valid < routed_input_mod.shape[0]:
        routed_input_mod.data[total_valid:] = float("nan")
    routed_input_mod.requires_grad_()
    combined = dispatcher.combine(routed_input_mod * 2.0, metadata, local_x)

    torch.manual_seed(100 + rank)
    local_grad_out = torch.randn_like(combined)
    combined.backward(local_grad_out)

    # Reference: the EP=1 local path on the whole batch.
    ref_x = global_x.clone().detach().requires_grad_()
    ref_scores = global_scores.clone().detach().requires_grad_()
    ref_dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=num_experts, top_k=top_k
    ).build()
    ref_routed, _, ref_metadata = ref_dispatcher.dispatch(
        ref_x, ref_scores, global_expert_ids, global_counts
    )
    ref_combined = ref_dispatcher.combine(ref_routed * 2.0, ref_metadata, ref_x)
    global_grad_out = torch.zeros_like(ref_combined)
    global_grad_out[local_start:local_end] = local_grad_out
    dist.all_reduce(global_grad_out)
    ref_combined.backward(global_grad_out)

    torch.testing.assert_close(combined, ref_combined[local_start:local_end])
    torch.testing.assert_close(local_x.grad, ref_x.grad[local_start:local_end])
    torch.testing.assert_close(
        local_scores.grad, ref_scores.grad[local_start:local_end]
    )


@pytest.mark.parametrize(
    "ep_size, mode",
    [(2, "random"), (2, "skewed"), (2, "empty"), (4, "random"), (4, "skewed")],
)
def test_ragged_ep_matches_ep1(ep_size, mode):
    distributed.run_distributed(
        ep_size,
        device_type.AcceleratorDeviceType.CPU,
        _test_ep_ragged_all_to_all,
        ep_size,
        mode,
    )


def test_config_rejects_unknown_transport():
    with pytest.raises(ValueError, match="ep_transport must be"):
        TpuAllToAllTokenDispatcher.Config(
            num_experts=8, top_k=2, ep_transport="invalid"
        ).build()


def _ragged_dispatcher_with_fake_mesh():
    dispatcher = TpuAllToAllTokenDispatcher.Config(
        num_experts=2, top_k=1, ep_transport="ragged_all_to_all"
    ).build()
    ep_mesh = mock.Mock()
    ep_mesh.size.return_value = 2
    dispatcher.ep_mesh = ep_mesh
    return dispatcher


def _inputs():
    return (
        torch.randn(2, 4),
        torch.randn(2, 1),
        torch.zeros(2, 1, dtype=torch.long),
        torch.tensor([2, 0]),
    )


def test_tracing_off_tpu_raises_instead_of_returning_zeros():
    dispatcher = _ragged_dispatcher_with_fake_mesh()
    dispatcher._gather_expert_counts = mock.Mock(return_value=torch.zeros(2, 2))
    holder = _RankHolder(0)
    dispatcher.bind_ep_rank_holder(holder)

    def dispatch(x, scores, ids, counts):
        return dispatcher.dispatch(x, scores, ids, counts)

    with pytest.raises(NotImplementedError):
        make_fx(dispatch)(*_inputs())


def test_dispatch_without_a_rank_owner_raises():
    dispatcher = _ragged_dispatcher_with_fake_mesh()
    with pytest.raises(RuntimeError, match="bound to a"):
        dispatcher.dispatch(*_inputs())


@pytest.mark.parametrize(
    "has_ep, ep_transport, registers",
    [
        (True, "ragged_all_to_all", True),
        (True, "all_to_all", False),
        (False, "ragged_all_to_all", False),
    ],
)
def test_parallelize_registers_the_rank_buffer_only_for_ragged_ep(
    has_ep, ep_transport, registers
):
    routed = TpuRoutedExperts.Config(
        inner_experts=GroupedExperts.Config(dim=4, hidden_dim=4, num_experts=2),
        token_dispatcher=TpuAllToAllTokenDispatcher.Config(
            num_experts=2, top_k=1, ep_transport=ep_transport
        ),
    ).build()
    mesh = mock.Mock()
    mesh.device_type = "cpu"
    mesh.get_local_rank.return_value = 0
    parallel_dims = mock.Mock()
    parallel_dims.get_optional_mesh.return_value = mesh if has_ep else None
    routed.parallelize(parallel_dims)
    assert ("_ep_local_rank" in dict(routed.named_buffers())) == registers
