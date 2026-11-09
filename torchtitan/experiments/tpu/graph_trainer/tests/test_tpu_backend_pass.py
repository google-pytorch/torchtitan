import copy
import operator
from unittest import mock

from absl.testing import absltest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.split_module import split_module

from torchtitan.experiments.tpu.graph_trainer import tpu_backend_pass


class PartitionDonationTest(absltest.TestCase):

  def test_compiled_additive_buffer_updates_survive_replay(self):
    """An add_ on a non-grad input buffer must reach the caller's tensor.

    MoE.tokens_per_expert_E is such a buffer; the expert-bias balancer reads it
    in an optimizer pre-hook. The fake compiler below returns values only, as
    the StaticCompiler executable does, so the update survives only if the pass
    returns it and copies it back.
    """

    def fn(counts, assignments):
      counts.add_(assignments)
      return assignments.sum(), "tag"

    gm = make_fx(fn, tracing_mode="fake")(torch.zeros(3), torch.ones(3))

    class ValueOnlyCompiler:

      def __call__(self, graph, inputs, **kwargs):
        del inputs, kwargs
        forward = graph.forward
        return lambda *args: forward(*(arg.clone() for arg in args))

    counts = torch.zeros(3)
    with mock.patch.object(
        tpu_backend_pass,
        "_import_static_compiler",
        return_value=ValueOnlyCompiler,
    ):
      tpu_backend_pass.tpu_backend_compile_pass(gm, (counts, torch.ones(3)))
    for total in (1, 2):
      loss, tag = gm(counts, torch.ones(3))
      self.assertEqual(loss.item(), 3)
      self.assertEqual(tag, "tag")
      torch.testing.assert_close(counts, torch.full((3,), float(total)))

  def _split(self, fn, inputs, partition_names):
    gm = make_fx(fn, tracing_mode="fake")(*inputs)
    partition_map = {
        node: partition_names[node.name]
        for node in gm.graph.nodes
        if node.op not in ("placeholder", "output")
    }
    return split_module(gm, None, partition_map.__getitem__)

  def test_donates_internal_boundary_at_last_use(self):
    def fn(x):
      boundary = x + 1
      first = boundary * 2
      second = boundary.sin()
      return first + second

    gm = self._split(
        fn,
        (torch.randn(2, 3),),
        {"add": 0, "mul": 1, "sin": 2, "add_1": 2},
    )

    tpu_backend_pass._annotate_internal_donations(gm)

    children = list(gm.children())
    self.assertEqual(children[0].meta["tpu_donated_input_positions"], ())
    self.assertEqual(children[1].meta["tpu_donated_input_positions"], ())
    self.assertEqual(children[2].meta["tpu_donated_input_positions"], (0, 1))
    self.assertEqual(
        copy.deepcopy(children[2]).meta["tpu_donated_input_positions"],
        (0, 1),
    )

  def test_forced_donation_does_not_require_internal_inference(self):
    def fn(x):
      boundary = x + 1
      return boundary.sin()

    gm = self._split(
        fn,
        (torch.randn(2, 3),),
        {"add": 0, "sin": 1},
    )
    child = list(gm.children())[1]
    placeholder = next(
        node for node in child.graph.nodes if node.op == "placeholder"
    )
    placeholder.meta["tpu_force_donate"] = True
    # Forced host donors do not depend on whole-parent alias analysis. Keep
    # them enabled even if an unrelated boundary lacks fake-tensor metadata.
    first_child = list(gm.children())[0]
    first_child_output = next(
        node for node in first_child.graph.nodes if node.op == "output"
    )
    first_child_output.args[0].meta.pop("val", None)

    tpu_backend_pass._annotate_internal_donations(
        gm,
        donate_internal_buffers=False,
    )

    self.assertEqual(child.meta["tpu_donated_input_positions"], (0,))

  def test_does_not_donate_view_of_parent_input(self):
    def fn(x):
      boundary = x.view(2, 3)
      return boundary.sin()

    gm = self._split(
        fn,
        (torch.randn(6),),
        {"view": 0, "sin": 1},
    )

    tpu_backend_pass._annotate_internal_donations(gm)

    for child in gm.children():
      self.assertEqual(child.meta["tpu_donated_input_positions"], ())

  def test_does_not_donate_escaping_output(self):
    def fn(x):
      boundary = x + 1
      return boundary.sin(), boundary

    gm = self._split(
        fn,
        (torch.randn(2, 3),),
        {"add": 0, "sin": 1},
    )

    tpu_backend_pass._annotate_internal_donations(gm)

    children = list(gm.children())
    self.assertEqual(children[1].meta["tpu_donated_input_positions"], ())

  def test_does_not_donate_aliased_producer_outputs(self):
    def fn(x):
      boundary = x + 1
      alias = boundary.view(2, 3)
      return boundary.sin() + alias.cos()

    gm = self._split(
        fn,
        (torch.randn(2, 3),),
        {"add": 0, "view": 0, "sin": 1, "cos": 1, "add_1": 1},
    )

    tpu_backend_pass._annotate_internal_donations(gm)

    children = list(gm.children())
    self.assertEqual(children[1].meta["tpu_donated_input_positions"], ())

  def test_review_is_not_donated_before_sibling_alias_use(self):
    def fn(x):
      boundary = x + 1
      sibling = boundary.view(2, 3)
      review = boundary.view(2, 3)
      first = review.sin()
      second = sibling.cos()
      return first + second

    gm = self._split(
        fn,
        (torch.randn(2, 3),),
        {
            "add": 0,
            "view": 0,
            "view_1": 1,
            "sin": 2,
            "cos": 3,
            "add_1": 3,
        },
    )

    tpu_backend_pass._annotate_internal_donations(gm)

    children = list(gm.children())
    self.assertEqual(children[2].meta["tpu_donated_input_positions"], ())

  def test_review_is_not_donated_when_sibling_alias_escapes(self):
    def fn(x):
      boundary = x + 1
      sibling = boundary.view(2, 3)
      review = boundary.view(2, 3)
      return review.sin(), sibling

    gm = self._split(
        fn,
        (torch.randn(2, 3),),
        {"add": 0, "view": 0, "view_1": 1, "sin": 2},
    )

    tpu_backend_pass._annotate_internal_donations(gm)

    children = list(gm.children())
    self.assertEqual(children[2].meta["tpu_donated_input_positions"], ())

  def test_review_is_donated_after_last_sibling_alias_use(self):
    def fn(x):
      boundary = x + 1
      sibling = boundary.view(2, 3)
      first = sibling.cos()
      review = boundary.view(2, 3)
      return first + review.sin()

    gm = self._split(
        fn,
        (torch.randn(2, 3),),
        {
            "add": 0,
            "view": 0,
            "cos": 1,
            "view_1": 2,
            "sin": 3,
            "add_1": 3,
        },
    )

    tpu_backend_pass._annotate_internal_donations(gm)

    children = list(gm.children())
    self.assertIn(0, children[3].meta["tpu_donated_input_positions"])

  def test_duplicate_getitem_keeps_storage_live(self):
    def fn(x):
      boundary = x + 1
      extra = x + 2
      first = boundary.sin()
      second = boundary.cos()
      return first + second + extra

    gm = self._split(
        fn,
        (torch.randn(2, 3),),
        {
            "add": 0,
            "add_1": 0,
            "sin": 1,
            "cos": 2,
            "add_2": 2,
            "add_3": 2,
        },
    )
    calls = [node for node in gm.graph.nodes if node.op == "call_module"]
    first_source = calls[1].args[0]
    self.assertIsInstance(first_source, torch.fx.Node)
    self.assertIs(first_source.target, operator.getitem)
    with gm.graph.inserting_before(calls[2]):
      duplicate = gm.graph.call_function(operator.getitem, first_source.args)
    calls[2].replace_input_with(first_source, duplicate)
    gm.graph.lint()
    gm.recompile()

    tpu_backend_pass._annotate_internal_donations(gm)

    children = list(gm.children())
    self.assertNotIn(0, children[1].meta["tpu_donated_input_positions"])

  def test_missing_tensor_value_disables_donation(self):
    def fn(x):
      boundary = x + 1
      return boundary.sin()

    gm = self._split(
        fn,
        (torch.randn(2, 3),),
        {"add": 0, "sin": 1},
    )
    producer_output = tpu_backend_pass._submodule_outputs(
        list(gm.children())[0]
    )[0]
    self.assertIn("tensor_meta", producer_output.meta)
    producer_output.meta.pop("val")

    tpu_backend_pass._annotate_internal_donations(gm)

    for child in gm.children():
      self.assertNotIn("tpu_donated_input_positions", child.meta)

  def test_missing_alias_metadata_disables_donation(self):
    def fn(x):
      boundary = x + 1
      sibling = boundary.view(2, 3)
      first = boundary.sin()
      second = sibling.cos()
      return first + second

    gm = self._split(
        fn,
        (torch.randn(2, 3),),
        {
            "add": 0,
            "view": 0,
            "sin": 1,
            "cos": 2,
            "add_1": 2,
        },
    )
    producer_outputs = tpu_backend_pass._submodule_outputs(
        list(gm.children())[0]
    )
    self.assertEqual(
        tpu_backend_pass._tensor_storage_id(
            producer_outputs[0].meta["val"]
        ),
        tpu_backend_pass._tensor_storage_id(
            producer_outputs[1].meta["val"]
        ),
    )
    producer_outputs[1].meta.clear()

    tpu_backend_pass._annotate_internal_donations(gm)

    for child in gm.children():
      self.assertNotIn("tpu_donated_input_positions", child.meta)

  def test_maps_argument_positions_to_tensor_indices(self):
    gm = make_fx(lambda x, y: (x + y,))(
        torch.randn(2, 3), torch.randn(2, 3)
    )
    executable = mock.Mock(return_value=(torch.randn(2, 3),))
    compiler_instance = mock.Mock(return_value=executable)
    compiler = mock.Mock(return_value=compiler_instance)

    with mock.patch.object(
        tpu_backend_pass, "_import_static_compiler", return_value=compiler
    ):
      tpu_backend_pass.tpu_backend_compile_pass(
          gm,
          ("opaque", torch.randn(2, 3), torch.randn(2, 3)),
          donated_input_positions=(1, 2),
      )

    self.assertEqual(
        compiler_instance.call_args.kwargs["donated_inputs"], [0, 1]
    )

  def test_direct_compile_restores_pinned_host_reload_input(self):
    graph = torch.fx.Graph()
    host = graph.placeholder("host")
    host.meta["val"] = torch.empty_strided((2, 3), (3, 1))
    reload_node = graph.call_function(
        torch.ops.ao.reload.default,
        args=(host, torch.device("cpu"), [2, 3], [3, 1]),
    )
    reload_node.meta["val"] = torch.empty_strided((2, 3), (3, 1))
    graph.output((reload_node,))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    executable = mock.Mock(return_value=(torch.randn(2, 3),))
    compiler_instance = mock.Mock(return_value=executable)
    compiler = mock.Mock(return_value=compiler_instance)
    real_empty_strided = torch.empty_strided

    def make_test_tensor(*args, **kwargs):
      kwargs["device"] = "cpu"
      return real_empty_strided(*args, **kwargs)

    with (
        mock.patch.object(
            tpu_backend_pass, "_import_static_compiler", return_value=compiler
        ),
        mock.patch.object(
            tpu_backend_pass.torch._subclasses.fake_tensor,
            "FakeTensorMode",
            return_value=mock.MagicMock(),
        ),
        mock.patch.object(
            tpu_backend_pass.torch,
            "empty_strided",
            side_effect=make_test_tensor,
        ),
    ):
      tpu_backend_pass.tpu_backend_compile_pass(
          gm,
          (torch.randn(2, 3),),
          is_fwd=False,
      )

    compile_inputs = compiler_instance.call_args.args[1]
    self.assertEqual(len(compile_inputs), 1)
    self.assertTrue(compile_inputs[0]._is_pinned_host)

  def test_legacy_static_compiler_works_when_donation_is_disabled(self):
    gm = make_fx(lambda x: (x.sin(),))(torch.randn(2, 3))
    executable = mock.Mock(return_value=(torch.randn(2, 3),))

    class LegacyStaticCompiler:

      def __call__(self, gm, inputs, is_fwd=True):
        del gm, inputs, is_fwd
        return executable

    with mock.patch.object(
        tpu_backend_pass,
        "_import_static_compiler",
        return_value=LegacyStaticCompiler,
    ):
      tpu_backend_pass.tpu_backend_compile_pass(
          gm,
          (torch.randn(2, 3),),
      )

    self.assertIs(gm._tpu_executable, executable)

if __name__ == "__main__":
  absltest.main()
