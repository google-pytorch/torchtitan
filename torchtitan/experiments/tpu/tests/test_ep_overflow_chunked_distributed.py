"""Tests for ``ep_overflow_mode="chunked"`` (no-drop capacity chunking).

Multi-process CPU (gloo) tests run ``ReferenceRoETokenDispatcher`` with a
capacity small enough to overflow (including several overflow windows) and
check that the chunked overflow tail reproduces the dropless EP=1 result in
forward and backward. A single-process test checks the TPU tail's pure-JAX
``lax.while_loop`` implementation against the PyTorch reference.
"""

import inspect
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import torch.nn.functional as F

from torchtitan.experiments.tpu.components.ep_overflow import (
    EpOverflowRouting,
    local_routing,
    overflow_tail_reference,
)
from torchtitan.experiments.tpu.components.expert_parallel import (
    use_expert_parallel,
)
from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRoutedExperts,
)
from torchtitan.experiments.tpu.distributed import (
    accelerator_device_type as device_type,
)
from torchtitan.experiments.tpu.distributed import distributed
from torchtitan.experiments.tpu.testing.roe_reference import (
    ReferenceRoETokenDispatcher,
)
from torchtitan.models.common.moe import GroupedExperts, RoutedExperts


class _SwiGLUExperts(torch.nn.Module):
    """Per-row SwiGLU experts; rows past ``sum(counts)`` are ignored."""

    def __init__(self, w1_EDF, w3_EDF, w2_EFD):
        super().__init__()
        self.w1_EDF = torch.nn.Parameter(w1_EDF)
        self.w3_EDF = torch.nn.Parameter(w3_EDF)
        self.w2_EFD = torch.nn.Parameter(w2_EFD)

    def overflow_tail_weights(self, pre_unsharded_weights=None):
        del pre_unsharded_weights
        return self.w1_EDF, self.w3_EDF, self.w2_EFD

    def forward(
        self, x_RD, num_tokens_per_expert_E, row_scale_R=None, **kwargs
    ):
        del kwargs
        counts = num_tokens_per_expert_E.to(torch.long)
        valid_rows = int(counts.sum().item())
        expert_R = torch.repeat_interleave(
            torch.arange(len(counts), device=x_RD.device), counts
        )
        x_valid = x_RD[:valid_rows]
        gate = F.silu(
            torch.einsum("rd,rdf->rf", x_valid, self.w1_EDF[expert_R])
        )
        hidden = gate * torch.einsum(
            "rd,rdf->rf", x_valid, self.w3_EDF[expert_R]
        )
        if row_scale_R is not None:
            hidden = (
                hidden.float() * row_scale_R[:valid_rows].reshape(-1, 1)
            ).to(hidden.dtype)
        out_valid = torch.einsum("rf,rfd->rd", hidden, self.w2_EFD[expert_R])
        # Touch every weight so all of them get (possibly zero) gradients.
        zero = (self.w1_EDF.sum() + self.w3_EDF.sum() + self.w2_EFD.sum()) * 0.0
        return F.pad(out_valid, (0, 0, 0, x_RD.shape[0] - valid_rows)) + zero


def _run_chunked_overflow(
    device, rank, world_size, routing, scale_before_w2, num_token_chunks
):
    num_experts, top_k, tokens_per_rank, dim, hidden = 8, 2, 8, 8, 12
    total_tokens = tokens_per_rank * world_size
    local_e = num_experts // world_size
    capacity_factor = 0.5  # cap = 8 rows (4 per chunk with 2 token chunks).
    torch.manual_seed(1234)
    w1_full = 0.2 * torch.randn(num_experts, dim, hidden, device=device)
    w3_full = 0.2 * torch.randn(num_experts, dim, hidden, device=device)
    w2_full = 0.2 * torch.randn(num_experts, hidden, dim, device=device)
    global_x = torch.randn(total_tokens, dim, device=device)
    global_scores = 0.1 + torch.rand(total_tokens, top_k, device=device)
    if routing == "skewed":
        # Every route goes to rank 0's experts: ep_size * T * top_k rows there,
        # several overflow windows past the capacity.
        global_ids = torch.randint(
            0, local_e, (total_tokens, top_k), device=device
        )
    else:
        global_ids = torch.randint(
            0, num_experts, (total_tokens, top_k), device=device
        )

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
            ep_transport="ring_of_experts",
            ep_capacity_factor=capacity_factor,
            ep_overflow_mode="chunked",
        ),
        scale_before_down_proj=scale_before_w2,
        num_token_chunks=num_token_chunks,
    ).build()
    routed.inner_experts = _SwiGLUExperts(
        w1_full[rank * local_e : (rank + 1) * local_e].clone(),
        w3_full[rank * local_e : (rank + 1) * local_e].clone(),
        w2_full[rank * local_e : (rank + 1) * local_e].clone(),
    )
    ep_mesh = init_device_mesh(
        device.type, (world_size,), mesh_dim_names=("ep",)
    )
    routed.token_dispatcher.ep_mesh = ep_mesh
    routed.register_buffer(
        "_ep_local_rank",
        torch.tensor([rank], dtype=torch.long, device=device),
        persistent=False,
    )
    routed.token_dispatcher.bind_ep_rank_holder(routed)

    out = routed(local_x, local_scores, local_ids, local_counts)
    # Nothing was dropped although the capacity overflowed.
    assert int(routed.token_dispatcher.last_dropped_count.item()) == 0
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
        scale_before_down_proj=scale_before_w2,
    ).build()
    ref_routed.inner_experts = _SwiGLUExperts(
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

    tol = dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out, ref_out[s:e], **tol)
    torch.testing.assert_close(local_x.grad, ref_x.grad[s:e], **tol)
    torch.testing.assert_close(local_scores.grad, ref_scores.grad[s:e], **tol)
    es, ee = rank * local_e, (rank + 1) * local_e
    for name in ("w1_EDF", "w3_EDF", "w2_EFD"):
        torch.testing.assert_close(
            getattr(routed.inner_experts, name).grad,
            getattr(ref_routed.inner_experts, name).grad[es:ee],
            **tol,
        )


class ChunkedOverflowParityTest(parameterized.TestCase):

    @parameterized.parameters(
        (2, "skewed", True, 1),
        (2, "skewed", False, 1),
        (2, "random", True, 2),
        (4, "skewed", True, 2),
        (4, "random", False, 1),
    )
    def test_chunked_overflow_matches_dropless_ep1(
        self, ep_size, routing, scale_before_w2, num_token_chunks
    ):
        distributed.run_distributed(
            ep_size,
            device_type.AcceleratorDeviceType.CPU,
            _run_chunked_overflow,
            routing,
            scale_before_w2,
            num_token_chunks,
        )


def _random_overflow_case(seed, *, g=12, top_k=3, e=4, dim=8, hidden=6, cap=5):
    gen = torch.Generator().manual_seed(seed)
    m = g * top_k
    ids = torch.randint(0, 2 * e, (m,), generator=gen)
    slot_order, gs, count = local_routing(ids, torch.tensor([0]), e)
    x_all = torch.randn(g, dim, generator=gen)
    scores = torch.rand(m, generator=gen) + 0.1
    w1 = 0.3 * torch.randn(e, dim, hidden, generator=gen)
    w3 = 0.3 * torch.randn(e, dim, hidden, generator=gen)
    w2 = 0.3 * torch.randn(e, hidden, dim, generator=gen)
    partial = torch.randn(g, dim, generator=gen)
    return (
        partial,
        x_all,
        scores,
        (w1, w3, w2),
        slot_order,
        gs,
        count,
        cap,
        top_k,
    )


_SCHEMA_LIB = None


def _schema_checked_overflow_ops(jax, np, tc):
    """CPU torch ops with the overflow jax_op schemas that run the JAX code."""
    global _SCHEMA_LIB
    ns = "tt_moe_overflow_cputest"
    if _SCHEMA_LIB is None:
        lib = torch.library.Library(ns, "DEF")
        type_names = {jax.Array: "Tensor", int: "int", bool: "bool", str: "str"}
        for name, fn in (
            ("roe_overflow_fwd", tc._roe_overflow_fwd_jax),
            ("roe_overflow_bwd", tc._roe_overflow_bwd_jax),
        ):
            sig = inspect.signature(fn)
            args = ", ".join(
                f"{type_names[p.annotation]} {n}"
                for n, p in sig.parameters.items()
            )
            ret = sig.return_annotation
            rets = (
                "Tensor"
                if ret is jax.Array
                else "(" + ", ".join(["Tensor"] * len(ret.__args__)) + ")"
            )
            lib.define(f"{name}({args}) -> {rets}")

            def impl(*a, _fn=fn):
                j = [
                    jax.numpy.asarray(v.detach().numpy())
                    if isinstance(v, torch.Tensor)
                    else v
                    for v in a
                ]
                out = _fn(*j)
                to_t = lambda o: torch.from_numpy(np.array(o))
                return (
                    to_t(out)
                    if not isinstance(out, tuple)
                    else tuple(to_t(o) for o in out)
                )

            lib.impl(name, impl, "CPU")
        _SCHEMA_LIB = lib
    ops = getattr(torch.ops, ns)
    return ops.roe_overflow_fwd, ops.roe_overflow_bwd


class OverflowTailTest(parameterized.TestCase):

    def test_local_routing_orders_local_slots_first_by_expert(self):
        ids = torch.tensor([3, 0, 5, 1, 0, 2, 7, 1])
        slot_order, gs, count = local_routing(ids, torch.tensor([0]), 4)
        assert int(count) == 6
        assert gs.tolist() == [2, 2, 1, 1]
        assert slot_order[:6].tolist() == [1, 4, 3, 7, 5, 0]

    def test_reference_tail_adds_exactly_the_routes_past_cap(self):
        partial, x_all, scores, w, slot_order, gs, count, cap, top_k = (
            _random_overflow_case(0)
        )
        routing = EpOverflowRouting(
            x_all_G=x_all,
            scores_flat_M=scores,
            slot_order_M=slot_order,
            group_sizes_e=gs,
            count_total=count,
            cap=cap,
            top_k=top_k,
        )
        out = overflow_tail_reference(partial, routing, w)
        # Brute force: every local route past `cap` in sorted order.
        expected = partial.clone()
        ends = gs.cumsum(0)
        for s in range(cap, int(count)):
            slot = int(slot_order[s])
            j = int(torch.searchsorted(ends, torch.tensor(s), right=True))
            row = x_all[slot // top_k]
            h = F.silu(row @ w[0][j]) * (row @ w[1][j]) * scores[slot]
            expected[slot // top_k] += h @ w[2][j]
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        # No overflow: the tail is the identity.
        routing0 = EpOverflowRouting(
            x_all_G=x_all,
            scores_flat_M=scores,
            slot_order_M=slot_order,
            group_sizes_e=gs,
            count_total=count,
            cap=int(count),
            top_k=top_k,
        )
        assert overflow_tail_reference(partial, routing0, w) is partial

    @parameterized.parameters(0, 1, 2)
    def test_jax_while_loop_tail_matches_reference(self, seed):
        try:
            import jax
            import numpy as np

            from torchtitan.experiments.tpu.kernels import roe_overflow_tc as tc
        except (
            ImportError
        ) as exc:  # pragma: no cover - env without JAX/torch_tpu
            self.skipTest(f"needs JAX and torch_tpu: {exc}")
        jax.config.update("jax_enable_x64", False)
        partial, x_all, scores, w, slot_order, gs, count, cap, top_k = (
            _random_overflow_case(seed)
        )
        leaves = [x_all, scores, *w]
        leaves = [t.clone().requires_grad_() for t in leaves]
        routing = EpOverflowRouting(
            x_all_G=leaves[0],
            scores_flat_M=leaves[1],
            slot_order_M=slot_order,
            group_sizes_e=gs,
            count_total=count,
            cap=cap,
            top_k=top_k,
        )
        ref_out = overflow_tail_reference(partial, routing, tuple(leaves[2:]))
        grad_out = torch.randn_like(ref_out)
        ref_grads = torch.autograd.grad(ref_out, leaves, grad_out)

        j = lambda t: jax.numpy.asarray(t.detach().numpy())
        args = (
            j(x_all),
            j(scores),
            *[j(t) for t in w],
            j(slot_order),
            j(gs.to(torch.int32)),
            j(count.reshape(1).to(torch.int32)),
        )
        kw = dict(cap=cap, top_k=top_k, activation="silu")
        out = jax.jit(lambda p, *a: tc.overflow_fwd_impl(p, *a, **kw))(
            j(partial), *args
        )
        grads = jax.jit(lambda g, *a: tc.overflow_bwd_impl(g, *a, **kw))(
            j(grad_out), *args
        )
        np.testing.assert_allclose(
            np.asarray(out), ref_out.detach().numpy(), rtol=1e-4, atol=1e-4
        )
        for got, ref in zip(grads, ref_grads):
            np.testing.assert_allclose(
                np.asarray(got), ref.numpy(), rtol=1e-4, atol=1e-4
            )
        # No overflow: zero iterations, partial unchanged, zero gradients.
        args0 = args[:-1] + (j(torch.tensor([cap - 1], dtype=torch.int32)),)
        out0 = tc.overflow_fwd_impl(j(partial), *args0, **kw)
        np.testing.assert_array_equal(np.asarray(out0), partial.numpy())
        for got in tc.overflow_bwd_impl(j(grad_out), *args0, **kw):
            assert not np.asarray(got).any()

    @parameterized.parameters(
        ("silu", "while", "0"), ("gelu_tanh", "fori", "1")
    )
    def test_tpu_autograd_tail_through_schema_checked_ops(
        self, activation, impl, donate
    ):
        """Runs ``roe_overflow_tail`` end to end through real torch op schemas.

        The jax_op schemas are rebuilt from the JAX functions' annotations
        (``jax.Array`` -> ``Tensor``, ``int``, ``bool``) and registered as CPU
        torch ops running the JAX code, so a wrong argument type (e.g. a str
        for a bool) fails here exactly as on TPU, and the autograd wrapper's
        forward and backward are checked against the PyTorch reference.
        """
        try:
            import jax
            import numpy as np

            from torchtitan.experiments.tpu.kernels import roe_overflow_tc as tc
        except (
            ImportError
        ) as exc:  # pragma: no cover - env without JAX/torch_tpu
            self.skipTest(f"needs JAX and torch_tpu: {exc}")
        fwd_op, bwd_op = _schema_checked_overflow_ops(jax, np, tc)

        partial, x_all, scores, w, slot_order, gs, count, cap, top_k = (
            _random_overflow_case(7)
        )
        leaves = [t.clone().requires_grad_() for t in (x_all, scores, *w)]
        part = partial.clone().requires_grad_()

        def routing(xa, sc):
            return EpOverflowRouting(
                x_all_G=xa,
                scores_flat_M=sc,
                slot_order_M=slot_order,
                group_sizes_e=gs,
                count_total=count,
                cap=cap,
                top_k=top_k,
            )

        env = {
            "TT_EP_OVERFLOW_IMPL": impl,
            "TT_EP_OVERFLOW_DONATE": donate,
            "TT_EP_OVERFLOW_TILES": "128,0,256",
        }
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(tc, "_roe_overflow_fwd", fwd_op),
            mock.patch.object(tc, "_roe_overflow_fwd_donated", fwd_op),
            mock.patch.object(tc, "_roe_overflow_bwd", bwd_op),
        ):
            out = tc.roe_overflow_tail(
                part.clone(),
                routing(leaves[0], leaves[1]),
                tuple(leaves[2:]),
                activation,
            )
            grad_out = torch.randn_like(out)
            grads = torch.autograd.grad(out, [part, *leaves], grad_out)

        ref_leaves = [t.detach().clone().requires_grad_() for t in leaves]
        ref_part = partial.clone().requires_grad_()
        ref = overflow_tail_reference(
            ref_part,
            routing(ref_leaves[0], ref_leaves[1]),
            tuple(ref_leaves[2:]),
            activation,
        )
        ref_grads = torch.autograd.grad(ref, [ref_part, *ref_leaves], grad_out)
        tol = dict(rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out, ref, **tol)
        for got, want in zip(grads, ref_grads):
            torch.testing.assert_close(got, want, **tol)

        # A str where the schema says bool is rejected, as on TPU.
        with self.assertRaisesRegex(RuntimeError, "bool"):
            fwd_op(
                partial,
                x_all,
                scores,
                *w,
                slot_order,
                gs,
                count.reshape(1),
                cap,
                top_k,
                "silu",
                False,
                False,
                0,
                0,
                0,
                0,
            )

    def test_pallas_gmm_tail_matches_reference_in_interpret_mode(self):
        """The TPU (megablox Pallas gmm/tgmm) path, run in interpret mode."""
        try:
            import jax
            import numpy as np

            from torchtitan.experiments.tpu.kernels import roe_overflow_tc as tc
        except (
            ImportError
        ) as exc:  # pragma: no cover - env without JAX/torch_tpu
            self.skipTest(f"needs JAX and torch_tpu: {exc}")
        # 128-aligned shapes for the Pallas tiles; ~192 local routes for a
        # 128-slot capacity: one partial overflow window.
        partial, x_all, scores, w, slot_order, gs, count, cap, top_k = (
            _random_overflow_case(
                5, g=96, top_k=4, e=2, dim=128, hidden=128, cap=128
            )
        )
        assert cap < int(count) < 2 * cap
        j = lambda t: jax.numpy.asarray(t.detach().numpy())
        args = (
            j(x_all),
            j(scores),
            *[j(t) for t in w],
            j(slot_order),
            j(gs.to(torch.int32)),
            j(count.reshape(1).to(torch.int32)),
        )
        grad_out = torch.randn_like(partial)
        worst = tc.overflow_env_config(slot_order.shape[0], cap)["max_windows"]
        outs = {}
        # (use_pallas, fori, tiles): the while/fori loops and explicit tiles.
        for key in (
            (False, False, (0, 0, 0)),
            (True, False, (0, 0, 0)),
            (True, True, (128, 128, 128)),
            (False, True, (0, 0, 0)),
        ):
            use_pallas, fori, tiles = key
            kw = dict(
                cap=cap,
                top_k=top_k,
                activation="silu",
                use_pallas=use_pallas,
                interpret=use_pallas,
                fori=fori,
                max_windows=worst,
                tiles=tiles,
            )
            outs[key] = (
                tc.overflow_fwd_impl(j(partial), *args, **kw),
                tc.overflow_bwd_impl(j(grad_out), *args, **kw),
            )
        ref_out, ref_grads = outs[(False, False, (0, 0, 0))]
        for key, (out, grads) in outs.items():
            np.testing.assert_allclose(
                np.asarray(out), np.asarray(ref_out), rtol=1e-4, atol=1e-4
            )
            for got, want in zip(grads, ref_grads):
                assert np.isfinite(np.asarray(got)).all(), key
                np.testing.assert_allclose(
                    np.asarray(got), np.asarray(want), rtol=1e-4, atol=1e-4
                )

    def test_fori_fallback_matches_while_and_skips_empty_windows(self):
        try:
            import jax
            import numpy as np

            from torchtitan.experiments.tpu.kernels import roe_overflow_tc as tc
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"needs JAX and torch_tpu: {exc}")
        partial, x_all, scores, w, slot_order, gs, count, cap, top_k = (
            _random_overflow_case(3)
        )
        j = lambda t: jax.numpy.asarray(t.detach().numpy())
        grad_out = torch.randn_like(partial)
        worst = tc.overflow_env_config(slot_order.shape[0], cap)["max_windows"]
        assert worst == -(-(slot_order.shape[0] - cap) // cap)
        for cnt in (int(count), cap - 1, 0):  # overflow, none, empty
            args = (
                j(x_all),
                j(scores),
                *[j(t) for t in w],
                j(slot_order),
                j(gs.to(torch.int32)),
                j(torch.tensor([cnt], dtype=torch.int32)),
            )
            res = []
            for fori in (False, True):
                kw = dict(
                    cap=cap,
                    top_k=top_k,
                    activation="silu",
                    fori=fori,
                    max_windows=worst,
                )
                res.append(
                    (
                        tc.overflow_fwd_impl(j(partial), *args, **kw),
                        *tc.overflow_bwd_impl(j(grad_out), *args, **kw),
                    )
                )
            for a, b in zip(*res):
                np.testing.assert_allclose(
                    np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5
                )

    def test_overflow_env_config_and_tiling(self):
        try:
            from torchtitan.experiments.tpu.kernels import roe_overflow_tc as tc
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"needs JAX and torch_tpu: {exc}")
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in list(os.environ):
                if k.startswith("TT_EP_OVERFLOW_"):
                    del os.environ[k]
            cfg = tc.overflow_env_config(num_slots=10_000, cap=3_000)
            assert cfg == dict(
                fori=False, max_windows=3, tiles=(0, 0, 0), donate=False
            )
            os.environ.update(
                TT_EP_OVERFLOW_IMPL="FORI",
                TT_EP_OVERFLOW_MAX_WINDOWS="2",
                TT_EP_OVERFLOW_TILES="512,256,128",
                TT_EP_OVERFLOW_DONATE="1",
            )
            cfg = tc.overflow_env_config(num_slots=10_000, cap=3_000)
            assert cfg == dict(
                fori=True, max_windows=2, tiles=(512, 256, 128), donate=True
            )
            os.environ["TT_EP_OVERFLOW_MAX_WINDOWS"] = "99"
            assert tc.overflow_env_config(10_000, 3_000)["max_windows"] == 3
            os.environ["TT_EP_OVERFLOW_TILES"] = "500,0,0"
            with self.assertRaisesRegex(ValueError, "TILES"):
                tc.overflow_env_config(10_000, 3_000)
            os.environ["TT_EP_OVERFLOW_TILES"] = "0,0,0"
            os.environ["TT_EP_OVERFLOW_IMPL"] = "scan"
            with self.assertRaisesRegex(ValueError, "IMPL"):
                tc.overflow_env_config(10_000, 3_000)
        # Requested tiles are kept where they divide; otherwise auto.
        assert tc._tiling(8192, 4096, 1024, (0, 0, 0)) == (512, 512, 512)
        assert tc._tiling(8192, 4096, 1024, (1024, 256, 128)) == (
            1024,
            256,
            128,
        )
        assert tc._tiling(1024, 4096, 1024, (2048, 384, 0)) == (512, 512, 512)
        assert tc._tiling(1024, 1024, 4096, (256, 1024, 1024)) == (
            256,
            1024,
            1024,
        )

    def test_invalid_overflow_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "ep_overflow_mode"):
            TpuAllToAllTokenDispatcher.Config(
                num_experts=8,
                top_k=2,
                ep_overflow_mode="spill",  # type: ignore[arg-type]
            ).build()

    def test_chunked_ragged_all_to_all_raises(self):
        dispatcher = TpuAllToAllTokenDispatcher.Config(
            num_experts=4,
            top_k=2,
            ep_transport="ragged_all_to_all",
            ep_capacity_factor=0.5,
            ep_overflow_mode="chunked",
        ).build()
        ep_mesh = mock.Mock()
        ep_mesh.size.return_value = 2
        dispatcher.ep_mesh = ep_mesh

        class _Holder(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.register_buffer("_ep_local_rank", torch.tensor([0]))

        holder = _Holder()
        dispatcher.bind_ep_rank_holder(holder)
        with self.assertRaisesRegex(
            NotImplementedError, "ring_of_experts.* only"
        ):
            dispatcher.dispatch(
                torch.randn(4, 8),
                torch.rand(4, 2),
                torch.zeros(4, 2, dtype=torch.long),
                torch.tensor([8, 0, 0, 0]),
            )

    def test_use_expert_parallel_sets_overflow_mode(self):
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

        cfg = _ModelCfg(
            RoutedExperts.Config(
                inner_experts=GroupedExperts.Config(
                    dim=16, hidden_dim=16, num_experts=8
                ),
                token_dispatcher=TpuAllToAllTokenDispatcher.Config(
                    num_experts=8, top_k=2
                ),
            )
        )
        use_expert_parallel(cfg, ep_degree=2, ep_capacity_factor=1.5)
        assert cfg.routed.token_dispatcher.ep_overflow_mode == "drop"
        use_expert_parallel(cfg, ep_degree=2, ep_overflow_mode="chunked")
        assert cfg.routed.token_dispatcher.ep_overflow_mode == "chunked"
        assert cfg.routed.token_dispatcher.ep_capacity_factor == 1.5


if __name__ == "__main__":
    absltest.main()
