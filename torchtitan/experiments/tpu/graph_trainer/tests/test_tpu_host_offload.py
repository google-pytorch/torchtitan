"""What the scan offload does to a forward/backward scan pair.

`_pair()` builds the graph the joint-graph tracer produces for scanned layers:

    fwd body: (carry, x) -> (out, clone(carry), attn)
    layer_stack = getitem(fwd_scan, 1)   # [6, 2, 512, 256]
    attn_stack  = getitem(fwd_scan, 2)   # [6, 2, 1024, 256], larger
    bwd_scan(init=[grad], xs=[counter], additional_inputs=[layer_stack, attn_stack])

`RealScanTest` traces a real `scan` instead, to check the pass finds the
saved carry after the split. The real llama3 case is in test_tpu_host_offload_distributed.py.
"""

import operator
from unittest import mock

from absl.testing import absltest, parameterized
import torch
from torch._higher_order_ops.scan import scan

from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.tpu.components import scan_backward_patch
from torchtitan.experiments.tpu.graph_trainer.scan_remat_policy import (
    scan_with_remat_policy,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_host_offload import (
    offload_scan_activations,
)
from torchtitan.experiments.tpu.components.remat_regions import (
    annotate_region,
    ATTENTION_INNER_ATTENTION,
    REMAT_REGION,
)

_SCAN = torch.ops.higher_order.scan
_OFFLOAD = torch.ops.ao.offload.default
_RELOAD = torch.ops.ao.reload.default
_AO = (_OFFLOAD, _RELOAD, torch.ops.ao.wait_tensor.default)

_STACK = (6, 2, 512, 256)  # 6 layers, one [2, 512, 256] layer input each
_ATTN = (6, 2, 1024, 256)  # a larger saved value that must stay on device


def _ao_targets(graph):
    return [n.target for n in graph.nodes if n.target in _AO]


def _body_of(gm, scan_node):
    return getattr(gm, scan_node.args[0].target)


def _op(graph, target, src, shape, *, name=None, region=None, bwd=False):
    node = graph.call_function(target, args=(src, src))
    if name:
        node.name = name
    node.meta["val"] = torch.empty(shape)
    node.meta["autograd_backward"] = bwd
    if region:
        node.meta["custom"] = {REMAT_REGION: region}
    return node


def _mm(graph, src, shape, **kwargs):
    return _op(graph, torch.ops.aten.mm.default, src, shape, **kwargs)


def _fwd_body(saves_carry):
    g = torch.fx.Graph()
    for name in ("carry", "x"):
        g.placeholder(name).meta["val"] = torch.empty(4, 4)
    carry, x = g.nodes
    saved = g.call_function(
        torch.ops.aten.clone.default, args=(carry if saves_carry else x,)
    )
    saved.meta["val"] = torch.empty(_STACK[1:])
    attn = _mm(g, carry, _ATTN[1:], region=ATTENTION_INNER_ATTENTION)
    g.output((_mm(g, carry, (4, 4)), saved, attn))
    return g


def _bwd_body(reads):
    """``(grad, i, stack, attn) -> (grad_out,)``."""
    g = torch.fx.Graph()
    g.placeholder("grad").meta["val"] = torch.empty(4, 4)
    i = g.placeholder("i")
    i.meta["val"] = torch.empty(())
    stack = g.placeholder("stack")
    stack.meta["val"] = torch.empty(_STACK)
    attn = g.placeholder("attn")
    attn.meta["val"] = torch.empty(_ATTN)
    rows = []
    for src, shape in ((stack, _STACK), (attn, _ATTN)):
        if reads == "row" or src is attn:
            row = g.call_function(
                torch.ops.aten.index_select.default, args=(src, 0, i)
            )
            row.meta["val"] = torch.empty(shape[1:])
        else:
            row = _mm(g, src, shape)
        rows.append(row)
    g.output((_mm(g, rows[0], (4, 4)),))
    return g


class ScanOffloadTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.root = torch.nn.Module()
        self.graph = torch.fx.Graph()
        self.x = self.graph.placeholder("x")
        self.x.meta["val"] = torch.empty(8, 8)

    def _gm(self):
        return torch.fx.GraphModule(self.root, self.graph)

    def _attach(self, body):
        """Register a loop body the way ``trace_scan`` does, as a submodule."""
        name = f"scan_combine_graph_{len(list(self.root.children())) + 1}"
        self.root.add_module(name, torch.fx.GraphModule(torch.nn.Module(), body))
        return self.graph.get_attr(name)

    def _pair(self, reads="row", saves_carry=True):
        g = self.graph
        carry = _mm(g, self.x, (4, 4), name="carry_init")
        xs = _mm(g, self.x, (6, 4, 4), name="xs_in")
        fwd = g.call_function(
            _SCAN, args=(self._attach(_fwd_body(saves_carry)), [carry], [xs], ())
        )
        fwd.meta["autograd_backward"] = False
        stacks = []
        for index, (name, shape) in enumerate(
            (("layer_stack", _STACK), ("attn_stack", _ATTN)), start=1
        ):
            node = g.call_function(operator.getitem, args=(fwd, index))
            node.name = name
            node.meta["val"] = torch.empty(shape)
            node.meta["autograd_backward"] = False
            stacks.append(node)
        counter = _mm(g, self.x, (6,), name="arange")
        grad = _mm(g, self.x, (4, 4), name="grad_init")
        bwd = g.call_function(
            _SCAN,
            args=(self._attach(_bwd_body(reads)), [grad], [counter], tuple(stacks)),
        )
        bwd.meta["autograd_backward"] = True
        return fwd, stacks[0], bwd

    def test_only_the_saved_carry_goes_to_host(self):
        fwd, stack, bwd = self._pair()
        self.graph.output(bwd)

        gm = self._gm()
        offload_scan_activations(gm)

        fwd_body = _body_of(gm, fwd).graph
        bwd_body = _body_of(gm, bwd).graph
        self.assertEqual(_ao_targets(gm.graph), [])
        # One offload, on the carry's clone; the larger attention save stays.
        (offload,) = [n for n in fwd_body.nodes if n.target is _OFFLOAD]
        self.assertIs(offload.args[0].target, torch.ops.aten.clone.default)
        self.assertEqual(stack.meta["val"].device.type, "cpu")
        # One reload, on the indexed row of the layer input, not the attention.
        (reload,) = [n for n in bwd_body.nodes if n.target is _RELOAD]
        self.assertIs(reload.args[0].target, torch.ops.aten.index_select.default)
        self.assertEqual(reload.args[0].args[0].name, "stack")

    def test_a_dead_reader_of_the_stack_does_not_block_the_offload(self):
        # Tracing a scan leaves unused element-0 slices of its operands.
        fwd, stack, bwd = self._pair()
        dead = self.graph.call_function(
            torch.ops.aten.select_copy.int, args=(stack, 0, 0)
        )
        dead.meta["val"] = torch.empty(_STACK[1:])
        self.graph.output(bwd)

        gm = self._gm()
        offload_scan_activations(gm)
        self.assertIn(_OFFLOAD, _ao_targets(_body_of(gm, fwd).graph))
        self.assertIn(_RELOAD, _ao_targets(_body_of(gm, bwd).graph))

    @parameterized.named_parameters(
        # It would silently read host memory.
        ("a_live_second_reader", {}, lambda g, s: [_mm(g, s, _STACK)], "outside"),
        # The output node has no users, but it is not a dead reader.
        ("the_graph_output", {}, lambda g, s: [s], "outside the backward scan"),
        # Reloading the whole stack every iteration is worse than no offload.
        ("a_whole_stack_read", {"reads": "whole"}, lambda g, s: [], "whole"),
        ("no_saved_carry", {"saves_carry": False}, lambda g, s: [], "none saves"),
    )
    def test_raises_and_leaves_the_bodies_untouched(self, pair, outputs, error):
        fwd, stack, bwd = self._pair(**pair)
        self.graph.output((bwd, *outputs(self.graph, stack)))

        gm = self._gm()
        with self.assertRaisesRegex(RuntimeError, error):
            offload_scan_activations(gm)
        self.assertEqual(_ao_targets(_body_of(gm, fwd).graph), [])
        self.assertEqual(_ao_targets(_body_of(gm, bwd).graph), [])

    def test_a_graph_with_no_scan_raises(self):
        big = _mm(self.graph, self.x, (4096, 2048), name="log_softmax")
        self.graph.output(_mm(self.graph, big, (4096, 2048), bwd=True))

        with self.assertRaisesRegex(RuntimeError, "use-scan-layers"):
            offload_scan_activations(self._gm())


_LAYERS, _TOKENS, _DIM = 3, 16, 8


def _plain_layer(h, w):
    return torch.tanh(h @ w)


def _attention_layer(h, w):
    q = h @ w
    with annotate_region(ATTENTION_INNER_ATTENTION):
        a = torch.sigmoid(torch.cat([q, q], -1))
    return torch.tanh(h + a[:, :_DIM] * a[:, _DIM:])


class RealScanTest(parameterized.TestCase):
    """The pass finds the saved carry, ``clone(carry)``, after ``scan``'s split."""

    def _trace(self, layer, save_regions):
        def step(h, w):
            out, _ = scan_with_remat_policy(
                lambda c, ww: (layer(c, ww), c.new_zeros(())), h, w, save_regions
            )
            return torch.autograd.grad(out.square().sum(), (h, w))

        self.assertTrue(scan_backward_patch.apply_scan_reverse_backward_patch())
        torch._dynamo.reset()
        h = torch.randn(_TOKENS, _DIM, requires_grad=True)
        w = torch.randn(_LAYERS, _DIM, _DIM, requires_grad=True)
        # Trace the reverse-by-index backward that TPU runs.
        with mock.patch.object(scan_backward_patch, "_REVERSE_SCAN_DEVICE_TYPE", "cpu"):
            gm = minimal_fx_tracer(step)(h, w).gm
        (fwd,) = [
            n for n in gm.graph.find_nodes(op="call_function", target=_SCAN)
            if not n.meta.get("autograd_backward")
        ]
        return gm, fwd

    def _outputs(self, gm, fwd):
        return list(_body_of(gm, fwd).graph.find_nodes(op="output")[0].args[0])

    def _saved_carry(self, gm, fwd):
        carry = _body_of(gm, fwd).graph.find_nodes(op="placeholder")[0]
        (saved,) = [
            o
            for o in self._outputs(gm, fwd)
            if o.target is torch.ops.aten.clone.default and o.args[0] is carry
        ]
        return saved

    @parameterized.named_parameters(
        ("plain_layer", _plain_layer, frozenset()),
        ("saved_attention", _attention_layer, {ATTENTION_INNER_ATTENTION}),
    )
    def test_only_the_saved_carry_goes_to_host(self, layer, save_regions):
        gm, fwd = self._trace(layer, save_regions)
        saved = self._saved_carry(gm, fwd)
        regions = [
            (o.meta.get("custom") or {}).get(REMAT_REGION)
            for o in self._outputs(gm, fwd)
        ]
        self.assertEqual(ATTENTION_INNER_ATTENTION in regions, bool(save_regions))

        offload_scan_activations(gm)
        (offload,) = [
            n for n in _body_of(gm, fwd).graph.nodes if n.target is _OFFLOAD
        ]
        self.assertIs(offload.args[0], saved)


if __name__ == "__main__":
    absltest.main()
