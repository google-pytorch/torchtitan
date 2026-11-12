"""What the scan offload does to a forward/backward scan pair.

`_pair()` builds the graph the joint-graph tracer produces for scanned layers:

    fwd body: (carry, x) -> (out, clone(carry) [layer_input], attn)
    layer_stack = getitem(fwd_scan, 1)   # [6, 2, 512, 256]
    attn_stack  = getitem(fwd_scan, 2)   # [6, 2, 1024, 256], larger
    bwd_scan(init=[grad], xs=[counter], additional_inputs=[layer_stack, attn_stack])

`RealScanTest` traces a real `scan` instead, to check the tag survives the
split. The real llama3 case is in test_tpu_host_offload_distributed.py.
"""

import operator
from unittest import mock

from absl.testing import absltest
import torch
from torch._higher_order_ops.scan import scan

from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.tpu.components import scan_backward_patch
from torchtitan.experiments.tpu.graph_trainer.scan_remat_policy import (
    scan_body_remat_window,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_host_offload import (
    offload_scan_activations,
)
from torchtitan.experiments.tpu.components.remat_regions import (
    annotate_region,
    ATTENTION_INNER_ATTENTION,
    LAYER_INPUT,
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


def _fwd_body(tag):
    g = torch.fx.Graph()
    for name in ("carry", "x"):
        g.placeholder(name).meta["val"] = torch.empty(4, 4)
    carry = next(iter(g.nodes))
    saved = g.call_function(torch.ops.aten.clone.default, args=(carry,))
    saved.meta["val"] = torch.empty(_STACK[1:])
    if tag:
        saved.meta["custom"] = {REMAT_REGION: LAYER_INPUT}
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


class ScanOffloadTest(absltest.TestCase):

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

    def _pair(self, reads="row", tag=True):
        g = self.graph
        carry = _mm(g, self.x, (4, 4), name="carry_init")
        xs = _mm(g, self.x, (6, 4, 4), name="xs_in")
        fwd = g.call_function(
            _SCAN, args=(self._attach(_fwd_body(tag)), [carry], [xs], ())
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

    def _assert_untouched(self, gm, fwd, bwd):
        self.assertEqual(_ao_targets(_body_of(gm, fwd).graph), [])
        self.assertEqual(_ao_targets(_body_of(gm, bwd).graph), [])

    def test_only_the_tagged_layer_input_goes_to_host(self):
        fwd, stack, bwd = self._pair()
        self.graph.output(bwd)

        gm = self._gm()
        offload_scan_activations(gm)

        fwd_body = _body_of(gm, fwd).graph
        bwd_body = _body_of(gm, bwd).graph
        self.assertEqual(_ao_targets(gm.graph), [])
        # One offload, on the tagged clone; the larger attention save stays.
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

    def test_a_live_second_reader_raises(self):
        # It would silently read host memory.
        fwd, stack, bwd = self._pair()
        self.graph.output((bwd, _mm(self.graph, stack, _STACK)))

        gm = self._gm()
        with self.assertRaisesRegex(RuntimeError, "outside the backward scan"):
            offload_scan_activations(gm)
        self._assert_untouched(gm, fwd, bwd)

    def test_the_graph_output_reading_the_stack_raises(self):
        # The output node has no users, but it is not a dead reader.
        fwd, stack, bwd = self._pair()
        self.graph.output((bwd, stack))

        gm = self._gm()
        with self.assertRaisesRegex(RuntimeError, "outside the backward scan"):
            offload_scan_activations(gm)
        self._assert_untouched(gm, fwd, bwd)

    def test_a_body_that_reads_the_stack_whole_raises(self):
        # Reloading the whole stack every iteration is worse than no offload.
        fwd, _, bwd = self._pair(reads="whole")
        self.graph.output(bwd)

        gm = self._gm()
        with self.assertRaisesRegex(RuntimeError, "whole"):
            offload_scan_activations(gm)
        self._assert_untouched(gm, fwd, bwd)

    def test_an_untagged_save_raises_rather_than_guessing(self):
        fwd, _, bwd = self._pair(tag=False)
        self.graph.output(bwd)

        gm = self._gm()
        with self.assertRaisesRegex(
            RuntimeError, "1 forward scans, none saves a tagged layer_input"
        ):
            offload_scan_activations(gm)
        self._assert_untouched(gm, fwd, bwd)

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


class RealScanTest(absltest.TestCase):
    """The ``layer_input`` tag on the carry survives ``scan``'s split."""

    def _trace(self, layer, save_regions):
        def step(h, w):
            with scan_body_remat_window(save_regions):
                out, _ = scan(lambda c, ww: (layer(c, ww), c.new_zeros(())), h, w)
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

    def _regions(self, gm, fwd):
        outs = list(_body_of(gm, fwd).graph.find_nodes(op="output")[0].args[0])
        return {
            (o.meta.get("custom") or {}).get(REMAT_REGION): o for o in outs
        }

    def test_the_layer_input_tag_reaches_the_forward_body(self):
        gm, fwd = self._trace(_plain_layer, frozenset())
        saved = self._regions(gm, fwd)[LAYER_INPUT]
        self.assertIs(saved.target, torch.ops.aten.clone.default)

        offload_scan_activations(gm)
        (offload,) = [
            n for n in _body_of(gm, fwd).graph.nodes if n.target is _OFFLOAD
        ]
        self.assertIs(offload.args[0], saved)

    def test_a_saved_region_keeps_its_tag_next_to_the_layer_input(self):
        gm, fwd = self._trace(_attention_layer, {ATTENTION_INNER_ATTENTION})
        regions = self._regions(gm, fwd)
        self.assertIs(regions[LAYER_INPUT].target, torch.ops.aten.clone.default)
        self.assertIn(ATTENTION_INNER_ATTENTION, regions)

        offload_scan_activations(gm)
        # Only the layer input goes to host.
        (offload,) = [
            n for n in _body_of(gm, fwd).graph.nodes if n.target is _OFFLOAD
        ]
        self.assertIs(offload.args[0], regions[LAYER_INPUT])


if __name__ == "__main__":
    absltest.main()
