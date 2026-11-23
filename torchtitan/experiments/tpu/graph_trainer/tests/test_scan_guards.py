"""Each guard must raise, and must say which knob to change.

The message is part of the contract: every condition here otherwise produces a
*successful* unrolled run under a scan-shaped command line, so a bare
`ValueError` would tell the reader nothing.
"""

import unittest

from torch import nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.tpu.graph_trainer.layer_stack import (
    LayerStack,
    STACK_ATTR,
    StackedLayers,
)
from torchtitan.experiments.tpu.graph_trainer.scan_guards import (
    check_scan_parallelism,
    check_scan_was_applied,
)


def _parallelism(tp=1, cp=1, pp=1):
    return ParallelismConfig(
        tensor_parallel_degree=tp,
        context_parallel_degree=cp,
        pipeline_parallel_degree=pp,
    )


def _layer(dim=4):
    layer = nn.Module()
    layer.wq = nn.Linear(dim, dim, bias=False)
    return layer


def _layers(*mods):
    return nn.ModuleDict({str(i): m for i, m in enumerate(mods)})


class ParallelismTest(unittest.TestCase):

    def test_plain_data_parallel_is_allowed(self):
        check_scan_parallelism(_parallelism())

    def test_each_degree_names_its_own_flag(self):
        for arg, kind, degree in (("tp", "tensor", 2), ("cp", "context", 4),
                                  ("pp", "pipeline", 2)):
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError) as caught:
                    check_scan_parallelism(_parallelism(**{arg: degree}))
                self.assertIn(
                    f"--parallelism.{kind}-parallel-degree={degree}",
                    str(caught.exception),
                )

    def test_every_conflicting_degree_is_listed_not_just_the_first(self):
        # Otherwise you fix them one launch at a time.
        with self.assertRaises(ValueError) as caught:
            check_scan_parallelism(_parallelism(tp=2, cp=2, pp=2))
        message = str(caught.exception)
        for flag in ("tensor-parallel", "context-parallel", "pipeline-parallel"):
            self.assertIn(flag, message)


class WasAppliedTest(unittest.TestCase):

    def _model(self, *, scan: bool | None):
        """``None`` for plain layers, otherwise stacked with that ``use_scan``."""
        model = nn.Module()
        layers = _layers(_layer(), _layer())
        setattr(model, STACK_ATTR, nn.Module())
        model.layers = (
            layers
            if scan is None
            else StackedLayers(
                layers, LayerStack(layers, getattr(model, STACK_ATTR)), use_scan=scan
            )
        )
        return model

    def test_a_run_that_did_not_ask_for_scan_is_not_checked(self):
        check_scan_was_applied(self._model(scan=None), use_scan_layers=False)

    def test_a_scan_run_that_got_a_scan_passes(self):
        check_scan_was_applied(self._model(scan=True), use_scan_layers=True)

    def test_a_scan_run_that_got_nothing_raises(self):
        # This model trains fine. It is wrong only because the number it
        # reports is not the number that was asked for.
        with self.assertRaises(ValueError) as caught:
            check_scan_was_applied(self._model(scan=None), use_scan_layers=True)
        self.assertIn("use-scan-layers", str(caught.exception))
        self.assertIn("one at a time", str(caught.exception))

    def test_stacked_but_not_scanned_still_raises(self):
        # The same container can be installed without scan, so stacks
        # existing is no evidence the scan flag was honoured.
        with self.assertRaises(ValueError):
            check_scan_was_applied(self._model(scan=False), use_scan_layers=True)

    def test_the_container_is_found_below_the_root(self):
        # Lanes may hang the decoder off a submodule; the guard must not care.
        model = nn.Module()
        model.inner = self._model(scan=True)
        check_scan_was_applied(model, use_scan_layers=True)


if __name__ == "__main__":
    unittest.main()
