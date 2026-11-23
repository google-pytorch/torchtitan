# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for EP token chunking, FP32 cross-chunk dW, and the chunk barrier pass."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.device_mesh import init_device_mesh
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.experiments.tpu.components.expert_parallel import (
    default_num_token_chunks,
)
from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRoutedExperts,
    collect_ep_drop_stat_tensors,
    configure_ep_drop_stats,
)
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.distributed import distributed
from torchtitan.experiments.tpu.testing.roe_reference import (
    ReferenceRoETokenDispatcher,
)
from torchtitan.experiments.tpu.graph_trainer.passes.insert_ep_chunk_barriers import (
    apply_ep_chunk_barrier_pass,
)
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    _optimization_barrier,
)
from torchtitan.models.common.moe import _tokens_per_expert, GroupedExperts


class _CountingRaggedSwiGLUExperts(torch.nn.Module):
    """Per-row SwiGLU experts (rows past ``sum(counts)`` ignored, like Pallas GMMs).

    Counts weight unshards so tests can check that chunking unshards once.
    """

    def __init__(self, w1_EDF: torch.Tensor, w3_EDF: torch.Tensor, w2_EFD: torch.Tensor):
        super().__init__()
        self.w1_EDF = torch.nn.Parameter(w1_EDF)
        self.w3_EDF = torch.nn.Parameter(w3_EDF)
        self.w2_EFD = torch.nn.Parameter(w2_EFD)
        self.unshard_calls = 0

    def unshard_weights(self):
        self.unshard_calls += 1
        return self.w1_EDF, self.w2_EFD, self.w3_EDF

    def forward(
        self,
        x_RD,
        num_tokens_per_expert_E,
        row_scale_R=None,
        *,
        pre_unsharded_weights=None,
    ):
        w1_EDF, w2_EFD, w3_EDF = (
            pre_unsharded_weights
            if pre_unsharded_weights is not None
            else self.unshard_weights()
        )
        valid_rows = int(num_tokens_per_expert_E.sum().item())
        if valid_rows == 0:
            # Stay in the autograd graph (zero gradients, not none), as the
            # Pallas GMMs do, so every rank runs the same backward collectives.
            zero = (w1_EDF.sum() + w3_EDF.sum() + w2_EFD.sum()) * 0.0
            if row_scale_R is not None:
                zero = zero + row_scale_R.sum() * 0.0
            return x_RD * 0.0 + zero
        counts = num_tokens_per_expert_E.to(torch.long)
        expert_R = torch.repeat_interleave(
            torch.arange(len(counts), device=x_RD.device), counts
        )
        x_valid = x_RD[:valid_rows]
        gate = F.silu(torch.einsum("rd,rdf->rf", x_valid, w1_EDF[expert_R]))
        up = torch.einsum("rd,rdf->rf", x_valid, w3_EDF[expert_R])
        hidden = gate * up
        if row_scale_R is not None:
            hidden = (hidden.float() * row_scale_R[:valid_rows].reshape(-1, 1)).to(
                hidden.dtype
            )
        out_valid = torch.einsum("rf,rfd->rd", hidden, w2_EFD[expert_R])
        if valid_rows < x_RD.shape[0]:
            return F.pad(out_valid, (0, 0, 0, x_RD.shape[0] - valid_rows))
        return out_valid


def _run_chunked_ep_parity(
    device,
    rank,
    world_size,
    ep_transport,
    scale_before_w2,
    num_chunks,
    tokens_per_rank,
):
    num_experts, top_k, dim, hidden = 8, 2, 8, 12
    total_tokens = tokens_per_rank * world_size
    local_e = num_experts // world_size
    torch.manual_seed(321)
    w1_full = 0.2 * torch.randn(num_experts, dim, hidden, device=device)
    w3_full = 0.2 * torch.randn(num_experts, dim, hidden, device=device)
    w2_full = 0.2 * torch.randn(num_experts, hidden, dim, device=device)
    global_x = torch.randn(total_tokens, dim, device=device)
    global_scores = 0.1 + torch.rand(total_tokens, top_k, device=device)
    global_ids = torch.randint(0, num_experts, (total_tokens, top_k), device=device)
    # Skew the first tokens onto expert 0 and leave expert 5 empty.
    global_ids[:3] = 0
    global_ids[global_ids == 5] = 6

    s = rank * tokens_per_rank
    e = s + tokens_per_rank
    local_x = global_x[s:e].clone().requires_grad_()
    local_scores = global_scores[s:e].clone().requires_grad_()
    local_ids = global_ids[s:e]
    local_counts = _tokens_per_expert(local_ids, num_experts)

    routed = TpuRoutedExperts.Config(
        inner_experts=GroupedExperts.Config(
            dim=dim, hidden_dim=hidden, num_experts=local_e
        ),
        token_dispatcher=ReferenceRoETokenDispatcher.Config(
            num_experts=num_experts, top_k=top_k, ep_transport=ep_transport
        ),
        scale_before_down_proj=scale_before_w2,
        num_token_chunks=num_chunks,
        max_inflight_chunks=2,
    ).build()
    experts = _CountingRaggedSwiGLUExperts(
        w1_full[rank * local_e : (rank + 1) * local_e].clone(),
        w3_full[rank * local_e : (rank + 1) * local_e].clone(),
        w2_full[rank * local_e : (rank + 1) * local_e].clone(),
    )
    routed.inner_experts = experts
    ep_mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("ep",))
    routed.token_dispatcher.ep_mesh = ep_mesh
    routed.register_buffer(
        "_ep_local_rank",
        torch.tensor([rank], dtype=torch.long, device=device),
        persistent=False,
    )
    routed.token_dispatcher.bind_ep_rank_holder(routed)

    out = routed(local_x, local_scores, local_ids, local_counts)
    # Chunking unshards the expert weights once per layer, not once per chunk.
    assert experts.unshard_calls == 1

    torch.manual_seed(500 + rank)
    local_grad = torch.randn_like(out)
    out.backward(local_grad)

    ref_routed = TpuRoutedExperts.Config(
        inner_experts=GroupedExperts.Config(
            dim=dim, hidden_dim=hidden, num_experts=num_experts
        ),
        token_dispatcher=TpuAllToAllTokenDispatcher.Config(
            num_experts=num_experts, top_k=top_k
        ),
        scale_before_down_proj=scale_before_w2,
    ).build()
    ref_experts = _CountingRaggedSwiGLUExperts(
        w1_full.clone(), w3_full.clone(), w2_full.clone()
    )
    ref_routed.inner_experts = ref_experts
    ref_x = global_x.clone().requires_grad_()
    ref_scores = global_scores.clone().requires_grad_()
    ref_counts = _tokens_per_expert(global_ids, num_experts)
    ref_out = ref_routed(ref_x, ref_scores, global_ids, ref_counts)
    global_grad = torch.zeros_like(ref_out)
    global_grad[s:e] = local_grad
    dist.all_reduce(global_grad)
    ref_out.backward(global_grad)

    tol = dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out, ref_out[s:e], **tol)
    torch.testing.assert_close(local_x.grad, ref_x.grad[s:e], **tol)
    torch.testing.assert_close(local_scores.grad, ref_scores.grad[s:e], **tol)
    es, ee = rank * local_e, (rank + 1) * local_e
    for name in ("w1_EDF", "w3_EDF", "w2_EFD"):
        torch.testing.assert_close(
            getattr(experts, name).grad,
            getattr(ref_experts, name).grad[es:ee],
            **tol,
        )


class ChunkedEpParityTest(parameterized.TestCase):
    """Multi-process parity of chunked EP against an EP=1 reference."""

    @parameterized.parameters(
        (2, "ring_of_experts", False, 3, 12),
        (2, "ring_of_experts", True, 4, 12),
        (2, "ragged_all_to_all", False, 3, 12),
        (4, "ragged_all_to_all", True, 4, 8),
        (4, "auto", False, 4, 8),
        # 10 tokens are not divisible by 4 chunks: snaps to 2 chunks.
        (2, "ring_of_experts", False, 4, 10),
    )
    def test_chunked_ep_matches_ep1(
        self, ep_size, ep_transport, scale_before_w2, num_chunks, tokens_per_rank
    ):
        distributed.run_distributed(
            ep_size,
            device_type.AcceleratorDeviceType.CPU,
            _run_chunked_ep_parity,
            ep_transport,
            scale_before_w2,
            num_chunks,
            tokens_per_rank,
        )


class _LinearExperts(torch.nn.Module):
    """One shared linear 'expert' that records the weight dtype each chunk sees."""

    def __init__(self, dim: int, dtype: torch.dtype):
        super().__init__()
        self.w = torch.nn.Parameter((0.1 * torch.randn(dim, dim)).to(dtype))
        self.unshard_calls = 0
        self.weight_dtypes: list[torch.dtype] = []

    def unshard_weights(self):
        self.unshard_calls += 1
        return self.w, self.w, self.w

    def forward(
        self,
        x_RD,
        num_tokens_per_expert_E,
        row_scale_R=None,
        *,
        pre_unsharded_weights=None,
    ):
        weights = (
            pre_unsharded_weights
            if pre_unsharded_weights is not None
            else self.unshard_weights()
        )
        self.weight_dtypes.append(weights[0].dtype)
        return (x_RD.bfloat16() @ weights[0].bfloat16()).to(x_RD.dtype)


def _stub_chunked_routed(dim: int, num_chunks: int, accumulate_fp32: bool):
    """A chunked ``TpuRoutedExperts`` with identity dispatch/combine on one process."""
    routed = TpuRoutedExperts.Config(
        inner_experts=GroupedExperts.Config(dim=dim, hidden_dim=dim, num_experts=2),
        token_dispatcher=ReferenceRoETokenDispatcher.Config(
            num_experts=2, top_k=1, ep_transport="ring_of_experts"
        ),
        num_token_chunks=num_chunks,
        accumulate_chunk_weight_grads_fp32=accumulate_fp32,
    ).build()
    routed.inner_experts = _LinearExperts(dim, torch.bfloat16)
    dispatcher = routed.token_dispatcher
    ep_mesh = mock.Mock()
    ep_mesh.size.return_value = 2
    dispatcher.ep_mesh = ep_mesh
    chunk_rows: list[int] = []

    def fake_dispatch(x_c, scores_c, ids_c, counts_c):
        chunk_rows.append(x_c.shape[0])
        return x_c, counts_c, None

    def fake_combine(routed_output_c, metadata_c, x_c):
        return routed_output_c

    dispatcher.dispatch = fake_dispatch
    dispatcher.combine = fake_combine
    return routed, chunk_rows


def _stub_inputs(tokens: int):
    return (
        torch.ones(tokens, 1),
        torch.zeros(tokens, 1, dtype=torch.long),
        torch.tensor([tokens, 0]),
    )


class ChunkedRoutedExpertsTest(parameterized.TestCase):
    """Single-process checks of chunked TpuRoutedExperts."""

    @parameterized.parameters(
        (True, torch.float32), (False, torch.bfloat16)
    )
    def test_chunked_forward_unshards_once_and_widens_weights(
        self, accumulate_fp32, weight_dtype
    ):
        torch.manual_seed(0)
        dim, tokens = 8, 12
        routed, chunk_rows = _stub_chunked_routed(dim, 4, accumulate_fp32)
        x = torch.randn(tokens, dim)
        out = routed(x, *_stub_inputs(tokens))
        experts = routed.inner_experts
        assert experts.unshard_calls == 1
        assert chunk_rows == [3, 3, 3, 3]
        assert experts.weight_dtypes == [weight_dtype] * 4
        torch.testing.assert_close(
            out, (x.bfloat16() @ experts.w.bfloat16()).float(), rtol=1e-2, atol=1e-2
        )
        out.sum().backward()
        assert experts.w.grad is not None
        assert experts.w.grad.dtype == torch.bfloat16

    @parameterized.parameters(
        (True, torch.float32), (False, torch.bfloat16)
    )
    def test_chunk_weight_grads_accumulate_in_fp32(
        self, accumulate_fp32, add_dtype
    ):
        dim, tokens, num_chunks = 8, 16, 4
        routed, _ = _stub_chunked_routed(dim, num_chunks, accumulate_fp32)
        stub_inputs = _stub_inputs(tokens)

        def step(x, w):
            out = torch.func.functional_call(
                routed, {"inner_experts.w": w}, (x, *stub_inputs)
            )
            (dw,) = torch.autograd.grad(out.float().sum(), (w,))
            return dw

        with torch.fx.traceback.preserve_node_meta():
            gm = make_fx(step)(
                torch.randn(tokens, dim),
                routed.inner_experts.w.detach().clone().requires_grad_(True),
            )
        add_ops = (torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor)
        weight_grad_adds = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target in add_ops
            and isinstance(n.meta.get("val"), torch.Tensor)
            and tuple(n.meta["val"].shape) == (dim, dim)
        ]
        # Autograd sums the C per-chunk weight gradients with C - 1 adds.
        assert len(weight_grad_adds) == num_chunks - 1
        assert {n.meta["val"].dtype for n in weight_grad_adds} == {add_dtype}
        chunk_ids = {
            n.meta["custom"]["ep_chunk_idx"]
            for n in gm.graph.nodes
            if "ep_chunk_idx" in n.meta.get("custom", {})
        }
        assert chunk_ids == set(range(num_chunks))

    def test_unshard_weights_matches_forward_layout_and_module_fqn(self):
        from torchtitan.experiments.graph_trainer.common_utils import (
            annotate_module_fqns,
        )
        from torchtitan.experiments.tpu.components.grouped_experts import (
            TpuGroupedExperts,
        )
        from torchtitan.experiments.tpu.graph_trainer.qwen3_5 import (
            annotate_expert_unshard_fqns,
        )

        class _Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = TpuGroupedExperts.Config(
                    dim=128, hidden_dim=128, num_experts=2, use_pallas_grouped_mm=True
                ).build()

        model = torch.nn.Module()
        model.block = _Block()
        experts = model.block.experts
        with torch.no_grad():  # build() leaves the parameters uninitialized
            for param in experts.parameters():
                param.copy_(torch.randn_like(param))
        w1, w2, w3 = experts.unshard_weights()
        # The Pallas grouped matmuls take [E, K, N] weights.
        assert torch.equal(w1, experts.w1_EFD.transpose(-2, -1))
        assert torch.equal(w3, experts.w3_EFD.transpose(-2, -1))
        assert w2.shape == (2, 128, 128)

        annotate_module_fqns(model)
        annotate_expert_unshard_fqns(model)

        def step(x):
            u1, u2, u3 = experts.unshard_weights()
            return x + u1.sum() + u2.sum() + u3.sum()

        with torch.fx.traceback.preserve_node_meta():
            gm = make_fx(step)(torch.zeros(()))
        transposes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.transpose.int
        ]
        assert transposes
        assert {n.meta.get("custom", {}).get("module_fqn") for n in transposes} == {
            "block.experts"
        }

    def test_chunked_ep_rejects_fsdp2_experts(self):
        routed, _ = _stub_chunked_routed(8, 4, True)
        experts = routed.inner_experts
        # fully_shard swaps the module class the same way.
        experts.__class__ = type(
            f"FSDP{type(experts).__name__}", (FSDPModule, type(experts)), {}
        )
        with self.assertRaisesRegex(NotImplementedError, "SimpleFSDP"):
            routed(torch.randn(8, 8), *_stub_inputs(8))


def _tag(node, val, *, fqn=None, chunk=None, bwd=False):
    node.meta["val"] = val
    custom = {}
    if fqn is not None:
        custom["module_fqn"] = fqn
    if chunk is not None:
        custom["ep_chunk_idx"] = chunk
        custom["ep_max_inflight_chunks"] = 2
    if custom:
        node.meta["custom"] = custom
    if bwd:
        node.meta["autograd_backward"] = True
    return node


def _chunked_moe_graph(num_layers=2, num_chunks=4, rows=2, dim=4):
    """A joint graph shaped like chunked MoE layers with FP32 dW accumulation."""
    aten = torch.ops.aten
    graph = torch.fx.Graph()

    def bf16(*shape):
        return torch.empty(*shape, dtype=torch.bfloat16)

    def f32(*shape):
        return torch.empty(*shape, dtype=torch.float32)

    tokens = num_chunks * rows
    x = _tag(graph.placeholder("x"), bf16(tokens, dim))
    tangent = _tag(graph.placeholder("tangent"), bf16(tokens, dim))
    weights = [
        _tag(graph.placeholder(f"w{i}"), bf16(dim, dim)) for i in range(num_layers)
    ]

    h = x
    for layer in range(num_layers):
        fqn = f"layers.{layer}.moe.routed_experts"
        h = _tag(graph.call_function(aten.relu.default, (h,)), bf16(tokens, dim), fqn=fqn)
        w_fp32 = _tag(
            graph.call_function(
                aten._to_copy.default, (weights[layer],), {"dtype": torch.float32}
            ),
            f32(dim, dim),
            fqn=fqn,
        )
        ys = []
        for c in range(num_chunks):
            x_c = _tag(
                graph.call_function(aten.slice.Tensor, (h, 0, c * rows, (c + 1) * rows)),
                bf16(rows, dim),
                fqn=fqn,
                chunk=c,
            )
            w_c = _tag(
                graph.call_function(
                    aten._to_copy.default, (w_fp32,), {"dtype": torch.bfloat16}
                ),
                bf16(dim, dim),
                fqn=fqn,
                chunk=c,
            )
            ys.append(
                _tag(
                    graph.call_function(aten.mm.default, (x_c, w_c)),
                    bf16(rows, dim),
                    fqn=fqn,
                    chunk=c,
                )
            )
        h = _tag(graph.call_function(aten.cat.default, (ys, 0)), bf16(tokens, dim), fqn=fqn)

    grads = []
    g = tangent
    for layer in reversed(range(num_layers)):
        fqn = f"layers.{layer}.moe.routed_experts"
        g = _tag(
            graph.call_function(aten.relu.default, (g,)),
            bf16(tokens, dim),
            fqn=fqn,
            bwd=True,
        )
        acc = None
        for c in reversed(range(num_chunks)):
            g_c = _tag(
                graph.call_function(aten.slice.Tensor, (g, 0, c * rows, (c + 1) * rows)),
                bf16(rows, dim),
                fqn=fqn,
                chunk=c,
                bwd=True,
            )
            g_c_t = _tag(
                graph.call_function(aten.t.default, (g_c,)),
                bf16(dim, rows),
                fqn=fqn,
                chunk=c,
                bwd=True,
            )
            dw_c = _tag(
                graph.call_function(aten.mm.default, (g_c_t, g_c)),
                bf16(dim, dim),
                fqn=fqn,
                chunk=c,
                bwd=True,
            )
            dw_c_fp32 = _tag(
                graph.call_function(
                    aten._to_copy.default, (dw_c,), {"dtype": torch.float32}
                ),
                f32(dim, dim),
                fqn=fqn,
                chunk=c,
                bwd=True,
            )
            # Autograd's cross-chunk accumulation carries no module_fqn.
            acc = (
                dw_c_fp32
                if acc is None
                else _tag(
                    graph.call_function(aten.add.Tensor, (acc, dw_c_fp32)),
                    f32(dim, dim),
                )
            )
        grads.append(
            _tag(
                graph.call_function(
                    aten._to_copy.default, (acc,), {"dtype": torch.bfloat16}
                ),
                bf16(dim, dim),
                fqn=fqn,
                bwd=True,
            )
        )
    graph.output((h, *grads))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


class EpChunkBarrierPassTest(parameterized.TestCase):
    """Tests for the EP chunk optimization-barrier pass."""

    def test_ep_chunk_barrier_pass_bounds_inflight_chunks(self):
        num_layers, num_chunks, rows, dim = 2, 4, 2, 4
        torch.manual_seed(1)
        inputs = (
            torch.randn(num_chunks * rows, dim).bfloat16(),
            torch.randn(num_chunks * rows, dim).bfloat16(),
            *(torch.randn(dim, dim).bfloat16() for _ in range(num_layers)),
        )
        expected = _chunked_moe_graph(num_layers, num_chunks, rows, dim)(*inputs)

        gm = apply_ep_chunk_barrier_pass(
            _chunked_moe_graph(num_layers, num_chunks, rows, dim)
        )
        barriers = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _optimization_barrier
        ]
        fwd = [b for b in barriers if not b.meta.get("autograd_backward")]
        bwd = [b for b in barriers if b.meta.get("autograd_backward")]
        # layers * (C - W) barriers per phase, with W = 2.
        assert len(fwd) == num_layers * (num_chunks - 2)
        assert len(bwd) == num_layers * (num_chunks - 2)
        # The BF16 -> FP32 weight widening stays out of forward barriers, so XLA can
        # fold bf16(fp32(w)) back to w instead of materializing FP32 weights.
        for barrier in fwd:
            assert all(arg.meta["val"].dtype == torch.bfloat16 for arg in barrier.args)
        # Backward barriers hold the running FP32 weight-gradient accumulators.
        fp32_accumulators = [
            arg
            for barrier in bwd
            for arg in barrier.args
            if arg.target is torch.ops.aten.add.Tensor
            and arg.meta["val"].dtype == torch.float32
        ]
        assert len(fp32_accumulators) == num_layers
        # Barriers are identities on CPU: the outputs are unchanged.
        for got, want in zip(gm(*inputs), expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


class _AllGather(torch.autograd.Function):
    """SimpleFSDP-like all-gather on a 1-rank gloo group (funcol ops in the graph)."""

    @staticmethod
    def forward(ctx, shard):
        group_name = dist.group.WORLD.group_name
        ctx.group_name = group_name
        out = torch.ops._c10d_functional.all_gather_into_tensor(shard, 1, group_name)
        return torch.ops._c10d_functional.wait_tensor(out)

    @staticmethod
    def backward(ctx, grad):
        out = torch.ops._c10d_functional.reduce_scatter_tensor(
            grad.contiguous(), "sum", 1, ctx.group_name
        )
        return torch.ops._c10d_functional.wait_tensor(out)


class _GatheredSwiGLUExperts(torch.nn.Module):
    """Dense SwiGLU experts whose unshard all-gathers, with per-chunk BF16 casts."""

    def __init__(self, dim, hidden):
        super().__init__()
        generator = torch.Generator().manual_seed(0)

        def init(*shape):
            return torch.nn.Parameter(
                (0.1 * torch.randn(*shape, generator=generator)).bfloat16()
            )

        self.w1 = init(dim, hidden)
        self.w3 = init(dim, hidden)
        self.w2 = init(hidden, dim)

    def unshard_weights(self):
        return tuple(_AllGather.apply(w) for w in (self.w1, self.w2, self.w3))

    def forward(
        self, x_RD, num_tokens_per_expert_E, row_scale_R=None, *, pre_unsharded_weights
    ):
        w1, w2, w3 = pre_unsharded_weights
        x_bf16 = x_RD.bfloat16()
        h = F.silu(x_bf16 @ w1.bfloat16()) * (x_bf16 @ w3.bfloat16())
        return (h @ w2.bfloat16()).to(x_RD.dtype)


class Zero3EpChunkBarrierTest(parameterized.TestCase):
    """EP chunk barriers under ZeRO-3 (full recompute) on a 1-rank gloo group."""

    def test_zero3_backward_barriers_keep_fp32_weight_widen_out(self):
        """ZeRO-3 recomputes the FP32 weight widen in backward; barriers skip it."""
        from torchtitan.experiments.graph_trainer.common_utils import (
            _is_backward_node,
            _MODULE_FQN,
        )
        from torchtitan.experiments.graph_trainer.make_fx_tracer import (
            minimal_fx_tracer,
        )
        from torchtitan.experiments.graph_trainer.memory_policy import (
            _make_full_memory_policy,
            tag_sac_policy,
        )
        from torchtitan.experiments.graph_trainer.selective_activation_remat import (
            selective_activation_remat_pass,
        )

        dim, hidden, tokens, num_chunks = 24, 40, 32, 4
        weight_shapes = {(dim, hidden), (hidden, dim)}
        created_group = not dist.is_initialized()
        if created_group:
            dist.init_process_group(
                "gloo", rank=0, world_size=1, store=dist.HashStore()
            )
        try:
            routed = TpuRoutedExperts.Config(
                inner_experts=GroupedExperts.Config(
                    dim=dim, hidden_dim=hidden, num_experts=2
                ),
                token_dispatcher=ReferenceRoETokenDispatcher.Config(
                    num_experts=2, top_k=1, ep_transport="ring_of_experts"
                ),
                num_token_chunks=num_chunks,
                accumulate_chunk_weight_grads_fp32=True,
            ).build()
            routed.inner_experts = _GatheredSwiGLUExperts(dim, hidden)
            dispatcher = routed.token_dispatcher
            ep_mesh = mock.Mock()
            ep_mesh.size.return_value = 2
            dispatcher.ep_mesh = ep_mesh
            dispatcher.dispatch = lambda x_c, s_c, i_c, c_c: (x_c, c_c, None)
            dispatcher.combine = lambda out_c, meta_c, x_c: out_c

            def step(x, scores, ids, counts):
                with torch.fx.traceback.annotate({_MODULE_FQN: "layers.0.moe.experts"}):
                    out = routed(x, scores, ids, counts)
                params = [p for p in routed.parameters() if p.requires_grad]
                return torch.autograd.grad(out.float().sum(), [x, *params])

            gm = minimal_fx_tracer(step, module=routed)(
                torch.randn(tokens, dim, requires_grad=True), *_stub_inputs(tokens)
            ).gm
            # ZeRO-3: full recompute, including the SimpleFSDP all-gathers.
            tag_sac_policy(gm, policy_fn=_make_full_memory_policy())
            selective_activation_remat_pass(gm)
            apply_ep_chunk_barrier_pass(gm)
        finally:
            if created_group:
                dist.destroy_process_group()

        def is_weight(arg):
            val = arg.meta.get("val") if isinstance(arg, torch.fx.Node) else None
            return isinstance(val, torch.Tensor) and tuple(val.shape) in weight_shapes

        barriers = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _optimization_barrier
        ]
        fwd = [b for b in barriers if not _is_backward_node(b)]
        bwd = [b for b in barriers if _is_backward_node(b)]
        assert len(fwd) == len(bwd) == num_chunks - 2
        # The recomputed bf16 -> fp32 weight widen (`_to_copy*_recomputed`) is not a
        # barrier operand in either phase.
        weight_args = [a for b in barriers for a in b.args if is_weight(a)]
        recomputed_fp32 = [
            a.name
            for a in weight_args
            if "recomputed" in a.name and a.meta["val"].dtype == torch.float32
        ]
        assert not recomputed_fp32, recomputed_fp32
        assert not [a for b in fwd for a in b.args if is_weight(a)]
        # Backward barriers still order the FP32 dW accumulators.
        assert any(
            a.target is torch.ops.aten.add.Tensor and a.meta["val"].dtype == torch.float32
            for b in bwd
            for a in b.args
            if is_weight(a)
        )


class DefaultNumTokenChunksTest(parameterized.TestCase):
    """Tests for default_num_token_chunks."""

    def test_default_num_token_chunks(self):
        assert default_num_token_chunks(1) == 1
        assert default_num_token_chunks(2) == 4
        assert default_num_token_chunks(8) == 8
        assert default_num_token_chunks(32) == 16


def _run_chunked_ep_capacity_overflow(device, rank, world_size):
    # 2 ranks, 4 experts (2/rank), top_k=2, 8 tokens/rank split into 2 chunks of 4.
    # Per-chunk balanced = 4 * 2 = 8; ep_capacity_factor = 0.75 -> cap = 6/chunk.
    num_experts, top_k, tokens_per_rank, num_chunks, dim, hidden = 4, 2, 8, 2, 8, 12
    local_e = num_experts // world_size
    torch.manual_seed(404)
    w1_full = 0.2 * torch.randn(num_experts, dim, hidden, device=device)
    w3_full = 0.2 * torch.randn(num_experts, dim, hidden, device=device)
    w2_full = 0.2 * torch.randn(num_experts, hidden, dim, device=device)
    global_x = torch.randn(tokens_per_rank * world_size, dim, device=device)
    global_scores = 0.1 + torch.rand(
        tokens_per_rank * world_size, top_k, device=device
    )
    # Chunk 0 on both ranks (tokens 0..3 and 8..11 -> 16 slots): all -> expert 0
    # (rank 0 keeps 6, drops 10). Chunk 1 (tokens 4..7 and 12..15 -> 16 slots):
    # 10 -> expert 1 on rank 0 (keeps 6, drops 4), 6 -> expert 2 on rank 1 (0 drops).
    # Total drops across 2 chunks: rank 0 = 10 + 4 = 14; rank 1 = 0.
    global_ids = torch.zeros(
        (tokens_per_rank * world_size, top_k), dtype=torch.long, device=device
    )
    global_ids[4:8, 0] = 1
    global_ids[4:8, 1] = 2
    global_ids[12:16, 0] = 1
    global_ids[12:14, 1] = 1
    global_ids[14:16, 1] = 2

    s = rank * tokens_per_rank
    e = s + tokens_per_rank
    local_ids = global_ids[s:e]
    local_counts = _tokens_per_expert(local_ids, num_experts)
    ep_mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("ep",))
    torch.manual_seed(600 + rank)
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
                ep_capacity_factor=0.75,
            ),
            scale_before_down_proj=True,
            num_token_chunks=num_chunks,
            max_inflight_chunks=2,
        ).build()
        experts = _CountingRaggedSwiGLUExperts(
            w1_full[rank * local_e : (rank + 1) * local_e].clone(),
            w3_full[rank * local_e : (rank + 1) * local_e].clone(),
            w2_full[rank * local_e : (rank + 1) * local_e].clone(),
        )
        routed.inner_experts = experts
        routed.token_dispatcher.ep_mesh = ep_mesh
        routed.register_buffer(
            "_ep_local_rank",
            torch.tensor([rank], dtype=torch.long, device=device),
            persistent=False,
        )
        routed.token_dispatcher.bind_ep_rank_holder(routed)
        routed._ep_mesh = ep_mesh  # As TpuRoutedExperts.parallelize does.
        assert configure_ep_drop_stats([routed], enabled=True)

        out = routed(x_in, sc_in, local_ids, local_counts)
        assert experts.unshard_calls == 1
        # The drop stats carry the sum over chunks (received on rank 0) and
        # this rank's sent routes, 8 tokens * top_k 2.
        assert collect_ep_drop_stat_tensors([routed])[0].tolist() == [
            [14 if rank == 0 else 0, tokens_per_rank * top_k]
        ]
        assert routed.token_dispatcher.last_dropped_count is not None
        assert int(routed.token_dispatcher.last_dropped_count.item()) == (
            14 if rank == 0 else 0
        )
        out.backward(local_grad)
        results[transport] = (
            out.detach(),
            x_in.grad.detach(),
            sc_in.grad.detach(),
            experts.w1_EDF.grad.detach(),
            experts.w2_EFD.grad.detach(),
        )

    for roe_t, ra2a_t in zip(
        results["ring_of_experts"], results["ragged_all_to_all"], strict=True
    ):
        torch.testing.assert_close(roe_t, ra2a_t, rtol=1e-5, atol=1e-5)


class ChunkedEpCapacityFactorTest(parameterized.TestCase):
    """Multi-process chunked EP with a capacity factor."""

    def test_chunked_ep_with_capacity_factor(self):
        distributed.run_distributed(
            2,
            device_type.AcceleratorDeviceType.CPU,
            _run_chunked_ep_capacity_overflow,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
    absltest.main()
