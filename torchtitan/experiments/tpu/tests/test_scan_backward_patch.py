"""Unit tests for the reverse-by-index scan backward.

The patch is measured -- it is what turned an OOM into a run with room to spare
-- but that evidence lives in a cluster log, so these tests state the
properties a reader of the diff would otherwise have to take on faith: the
backward stops calling `aten.flip`, the gradients do not change when it does,
an upstream edit that leaves the reversal alone keeps working, and one that
does not fails loudly instead of quietly reverting to copies.

The torch-bump tests build their fake upstream by editing torch's own
`call_backward` source and exec'ing it back into `scan.py`'s namespace, which
is where a real bump's method would live and what the patch swaps under.

Everything runs under `torch.compile`, on `--device`. Never eagerly: an eager
scan never materialises its stacks, so an eager test passes without exercising
the thing the patch changes.
"""

import inspect
import sys
import textwrap
from unittest import mock

from absl.testing import absltest
import torch
from torch._higher_order_ops.scan import ScanAutogradImpl, scan
from torch.utils._python_dispatch import TorchDispatchMode
import torch.utils._pytree as pytree
from torch_tpu._internal.compile import scan as tpu_scan

from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.experiments.tpu.components import scan_backward_patch


_SCAN_MODULE = sys.modules["torch._higher_order_ops.scan"]


def _body(carry, x):
  nxt = torch.tanh(carry * 0.9 + x)
  # The clone is not decoration: a HOP body may not alias its carry to its y.
  return nxt, nxt.clone()


def _scan_of_tanh(init, xs):
  """A scan whose backward has to walk the saved stack in reverse."""
  return scan(_body, init, xs)


class _RecordOps(TorchDispatchMode):
  """The flips happen at runtime inside `call_backward`, not in any traced
  graph, so the only place to see them is the dispatcher."""

  supports_higher_order_operators = True

  def __init__(self):
    super().__init__()
    self.ops = []

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    self.ops.append(str(func))
    return func(*args, **(kwargs or {}))


class ScanBackwardPatchTest(base_device_test.BaseAcceleratorDeviceTest):

  def setUp(self):
    super().setUp()
    self._unpatched = ScanAutogradImpl.call_backward
    self.addCleanup(setattr, ScanAutogradImpl, "call_backward", self._unpatched)
    self.addCleanup(
        setattr,
        scan_backward_patch,
        "_UPSTREAM_CALL_BACKWARD",
        scan_backward_patch._UPSTREAM_CALL_BACKWARD,
    )
    self.addCleanup(torch._dynamo.reset)
    # The TPU backend is the one that matters -- `aot_eager` is the stand-in
    # when this file runs on the CPU lane.
    self._backend = (
        "tpu"
        if self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU
        else "aot_eager"
    )

  def _backward(self):
    """One compiled forward and backward. Returns (grads, ops seen)."""
    torch.manual_seed(0)
    device = self.accelerator_device
    init = torch.randn(3, device=device, requires_grad=True)
    xs = torch.randn(5, 3, device=device, requires_grad=True)
    torch._dynamo.reset()
    out = torch.compile(_scan_of_tanh, backend=self._backend)(init, xs)
    loss = sum(o.square().sum() for o in pytree.tree_leaves(out))
    recorder = _RecordOps()
    with recorder:
      grads = torch.autograd.grad(loss, (init, xs))
    return [g.cpu() for g in grads], [op for op in recorder.ops if "flip" in op]

  def _install_fake_bump(self, *replacements):
    """Installs an edited copy of torch's `call_backward` as if torch had.

    Exec'd into `scan.py`'s real globals, not a copy, because that is where a
    bump's method would resolve `torch` and `scan_op` from -- the two names
    the patch swaps.
    """
    src = textwrap.dedent(
        inspect.getsource(scan_backward_patch._UPSTREAM_CALL_BACKWARD)
    )
    for old, new in replacements:
      self.assertIn(old, src, "the fake bump did not match upstream's source")
      src = src.replace(old, new)
    exec(compile(src, "<fake torch bump>", "exec"), _SCAN_MODULE.__dict__)  # pylint: disable=exec-used
    bumped = _SCAN_MODULE.__dict__.pop("call_backward")
    ScanAutogradImpl.call_backward = bumped
    scan_backward_patch._UPSTREAM_CALL_BACKWARD = bumped

  def test_it_removes_the_flips_and_leaves_the_gradients_alone(self):
    """The gradient check also proves torch_tpu honoured the reverse request.

    A loop that ran forwards would return the `xs` gradient reversed.
    """
    before, flips_before = self._backward()
    # Asserted, not assumed: without this the next assertion passes for any
    # reason at all, including the test no longer reaching a scan.
    self.assertNotEqual(
        flips_before, [], "unpatched torch did not flip -- test is vacuous"
    )

    self.assertTrue(scan_backward_patch.apply_scan_reverse_backward_patch())
    after, flips_after = self._backward()

    self.assertEqual(flips_after, [], "the backward still reverses by copy")
    for name, b, a in zip(("init", "xs"), before, after):
      torch.testing.assert_close(a, b, msg=f"{name} gradient changed")

  def _record_rewritten_scans(self):
    """Calls through to the patch's own `scan_op`, keeping what it was given."""
    real_scan_op = scan_backward_patch.scan_op
    seen = []

    def recording(combine_gm, init, xs, additional_inputs=(), **kwargs):
      seen.append(combine_gm)
      return real_scan_op(combine_gm, init, xs, additional_inputs, **kwargs)

    self.enter_context(
        mock.patch.object(scan_backward_patch, "scan_op", recording)
    )
    return seen

  def test_it_asks_torch_tpu_for_a_reverse_loop_without_index_copy(self):
    """The structure the collective pipeliner needs, checked on the graph.

    The step body has to carry torch_tpu's reverse annotation -- checked with
    torch_tpu's own predicate, so the two cannot drift apart -- and must not
    write its gradient rows with `index_copy`, which lowers to a scatter that
    XLA's collective pipeliner cannot move collectives through.
    """
    seen = self._record_rewritten_scans()

    self.assertTrue(scan_backward_patch.apply_scan_reverse_backward_patch())
    self._backward()

    self.assertNotEmpty(seen, "the rewrite never ran -- test is vacuous")
    for combine_gm in seen:
      self.assertTrue(
          tpu_scan._body_asks_for_reverse(combine_gm),
          "the step body does not ask torch_tpu for a reverse loop",
      )
      targets = [str(node.target) for node in combine_gm.graph.nodes]
      self.assertFalse(
          [t for t in targets if "index_copy" in t or "scatter" in t],
          f"the step body still writes rows by scatter: {targets}",
      )

  def test_the_gradients_depend_on_torch_tpu_reversing_the_loop(self):
    """Why the gradient check above also covers the reverse request.

    With the annotation key changed to one torch_tpu does not know, the loop
    runs forwards and the gradients come out wrong. If they did not, the
    gradient check would pass whether or not torch_tpu reversed anything.
    """
    expected, _ = self._backward()
    self.enter_context(
        mock.patch.object(
            scan_backward_patch, "SCAN_REVERSE_ANNOTATION", "not_the_key"
        )
    )

    self.assertTrue(scan_backward_patch.apply_scan_reverse_backward_patch())
    grads, _ = self._backward()

    self.assertFalse(
        torch.allclose(grads[1], expected[1]),
        "the xs gradient is right without the reverse request -- the gradient"
        " check does not cover it",
    )

  def test_it_launches_its_scan_with_torch_s_own_names(self):
    """A scan of ours has to be traced as the real HOP, not as the stand-in.

    `scan.py` labels the node it emits with whatever `scan_op` names at that
    moment, and passes downstream find their scans by
    `node.target is torch.ops.higher_order.scan`. A node labelled with the
    stand-in still runs, so nothing fails -- activation offloading just
    offloads nothing and the run quietly spends the memory it meant to save.
    """
    torchs_own = (_SCAN_MODULE.torch, _SCAN_MODULE.scan_op)
    inner = scan_backward_patch._scan_over_descending_index
    seen = []

    def recording(*args, **kwargs):
      seen.append((_SCAN_MODULE.torch, _SCAN_MODULE.scan_op))
      return inner(*args, **kwargs)

    self.addCleanup(
        setattr, scan_backward_patch, "_scan_over_descending_index", inner
    )
    scan_backward_patch._scan_over_descending_index = recording

    self.assertTrue(scan_backward_patch.apply_scan_reverse_backward_patch())
    self._backward()

    self.assertNotEmpty(seen, "the rewrite never ran -- test is vacuous")
    for names in seen:
      self.assertEqual(
          names, torchs_own, "the scan was launched with the stand-ins in place"
      )

  def test_it_absorbs_an_upstream_change_that_is_not_the_reversal(self):
    """The reason this shims two names instead of forking the method.

    Upstream owns everything either side of the flip and the scan; a torch
    bump is free to edit any of it. Here it renames a local and adds a
    statement to the step body, and the patch carries on.
    """
    expected, _ = self._backward()
    self._install_fake_bump(
        (
            "n_carry = len(self.init)",
            "# an upstream comment\n    n_carry_renamed = len(self.init)",
        ),
        (
            "grad_fw_outputs[:n_carry], grad_fw_outputs[n_carry:]",
            "grad_fw_outputs[:n_carry_renamed], grad_fw_outputs[n_carry_renamed:]",
        ),
        (
            "        grad_fw_outputs = (*grad_carry, *grad_y)",
            "        grad_carry = [c + 0.0 for c in grad_carry]\n"
            "        grad_fw_outputs = (*grad_carry, *grad_y)",
        ),
    )

    self.assertTrue(scan_backward_patch.apply_scan_reverse_backward_patch())
    grads, flips = self._backward()

    self.assertEqual(flips, [], "the backward still reverses by copy")
    for name, want, got in zip(("init", "xs"), expected, grads):
      torch.testing.assert_close(got, want, msg=f"{name} gradient changed")

  def test_it_raises_rather_than_reverse_by_copy(self):
    """The canary, and the one case that cannot be absorbed.

    When upstream stops reversing by flip the rewrite no longer applies, and
    running on is an OOM with no obvious cause. Fail there instead.
    """
    self._install_fake_bump(
        (
            "[torch.flip(x, (0,)) for x in pytree.tree_flatten(bw_xs)[0]]",
            "pytree.tree_flatten(bw_xs)[0]",
        ),
        ("*[torch.flip(elem, (0,)) for elem in grad_xs],", "*grad_xs,"),
    )
    self.assertTrue(scan_backward_patch.apply_scan_reverse_backward_patch())

    with self.assertRaisesRegex(Exception, "no longer reverses"):
      self._backward()

  def test_it_refuses_to_install_when_the_source_says_the_fix_landed(self):
    """Same break, caught at startup instead of one compile later."""

    def no_longer_flips(self, *grad_fw_outputs):
      return scan_op(None, None, None)  # noqa: F821  # pylint: disable=undefined-variable

    ScanAutogradImpl.call_backward = no_longer_flips
    scan_backward_patch._UPSTREAM_CALL_BACKWARD = no_longer_flips
    self.assertFalse(scan_backward_patch.apply_scan_reverse_backward_patch())
    self.assertIs(ScanAutogradImpl.call_backward, no_longer_flips)

  def test_it_refuses_to_install_over_someone_else(self):
    def not_the_method(self, *grad_fw_outputs):
      raise AssertionError("unreachable")

    ScanAutogradImpl.call_backward = not_the_method
    self.assertFalse(scan_backward_patch.apply_scan_reverse_backward_patch())
    self.assertIs(ScanAutogradImpl.call_backward, not_the_method)

  def test_it_raises_when_a_flip_is_added_before_the_scan(self):
    """A flip this skips has to be one the scan below undoes.

    An unrelated pre-scan flip would otherwise be swallowed: nobody performs
    it, and the gradients are quietly wrong.
    """
    self._install_fake_bump(
        (
            "flat_grads = scan_op(",
            "_unrelated = torch.flip(torch.zeros(2, 2), (0,))\n"
            "    flat_grads = scan_op(",
        ),
    )
    self.assertTrue(scan_backward_patch.apply_scan_reverse_backward_patch())

    with self.assertRaisesRegex(Exception, "no longer reverses"):
      self._backward()

  def test_it_does_not_swallow_a_flip_added_after_the_scan(self):
    """Same gap on the output side.

    Here a bump also reverses the carry gradients, which our scan did not
    write, so that flip has to actually happen.
    """
    self._install_fake_bump(
        ("*grad_init,", "*[torch.flip(g, (0,)) for g in grad_init],"),
    )
    expected, _ = self._backward()

    self.assertTrue(scan_backward_patch.apply_scan_reverse_backward_patch())
    grads, flips = self._backward()

    self.assertNotEqual(flips, [], "the added flip was swallowed")
    for name, want, got in zip(("init", "xs"), expected, grads):
      torch.testing.assert_close(got, want, msg=f"{name} gradient changed")

  def test_it_falls_back_on_a_device_that_ignores_the_reverse_request(self):
    """Off TPU: a forward loop and one flip, same gradients, no annotation.

    CPU tests and meta-device traces install this patch too, so it must not
    raise there, and must not ask a backend that ignores the request for a
    reverse loop. Simulated by naming another device as the one that honours
    it, since this test only runs on TPU.
    """
    expected, _ = self._backward()
    self.enter_context(
        mock.patch.object(
            scan_backward_patch, "_REVERSE_SCAN_DEVICE_TYPE", "not_a_device"
        )
    )
    seen = self._record_rewritten_scans()

    self.assertTrue(scan_backward_patch.apply_scan_reverse_backward_patch())
    grads, _ = self._backward()

    self.assertNotEmpty(seen, "the rewrite never ran -- test is vacuous")
    for combine_gm in seen:
      self.assertFalse(
          tpu_scan._body_asks_for_reverse(combine_gm),
          "asked for a reverse loop on a device that ignores the request",
      )
    for name, want, got in zip(("init", "xs"), expected, grads):
      torch.testing.assert_close(got, want, msg=f"{name} gradient changed")


if __name__ == "__main__":
  absltest.main()
