"""Explicitly reject unsupported helper state and modified core source."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from torchtitan.experiments.tpu.experimental_flex_shard.dependency import (
    CORE_FILES,
    verify_core,
)
from torchtitan.experiments.tpu.experimental_flex_shard.tests.common import (
    require_executor,
)
from torchtitan.experiments.tpu.experimental_flex_shard.trace import capture_sgd_step


class HelperContractTest(unittest.TestCase):
    def test_frozen_parameter_rejected(self):
        model = nn.Linear(2, 2)
        model.bias.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "untied, trainable"):
            capture_sgd_step(model, None, None, None, None, None)

    def test_tied_parameter_rejected(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        model[1].weight = model[0].weight
        with self.assertRaisesRegex(ValueError, "untied, trainable"):
            capture_sgd_step(model, None, None, None, None, None)

    def test_modified_private_core_rejected(self):
        require_executor(self)
        identity = verify_core()
        source = Path(identity["path"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in CORE_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((source / relative).read_bytes())
            (root / "flex_shard/bucket_storage.py").write_text("# modified\n")
            with patch(
                "importlib.util.find_spec",
                return_value=SimpleNamespace(origin=str(root / "__init__.py")),
            ):
                with self.assertRaisesRegex(ValueError, "core differs"):
                    verify_core()


if __name__ == "__main__":
    unittest.main()
