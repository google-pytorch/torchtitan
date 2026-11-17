"""Load the reviewed portable executor instead of duplicating its source.

The dependency is the core-only graph executor from TorchTPU PR #5959, not
FlexShard's CUDA eager runtime. Both public source revisions are explicit.
"""

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

EXECUTOR_COMMIT = "3f095aa23fb835cc017e538ad313c17945879d6e"
EXECUTOR_SHA256 = "0e02bfc383ad60bdf2ce5bfe94f4ac88756362a592190d6ed4c08374edfcb373"
FLEX_SHARD_COMMIT = "233e90691a5523023f6cf98c20b4f4529bee9ea4"
CORE_SHA256 = "226a3ab2a7162f9b4966ec5ae5393d772fe754e4505e7727c0934db552dd19bb"
CORE_FILES = (
    "__init__.py",
    "flex_shard/__init__.py",
    "flex_shard/bucket_comm.py",
    "flex_shard/bucket_runtime.py",
    "flex_shard/bucket_storage.py",
    "flex_shard/checkpoint.py",
    "flex_shard/flex_shard.py",
    "flex_shard/param_init.py",
    "flex_shard/placement_contract.py",
    "flex_shard/sharded_param.py",
    "flex_shard/utils.py",
)


def verify_core():
    """Verify the imported private core against the reviewed source revision."""
    core_spec = importlib.util.find_spec("flex_shard")
    if core_spec is None or core_spec.origin is None:
        raise ValueError("Put the pinned public FlexShard src directory on PYTHONPATH")
    root = Path(core_spec.origin).resolve().parent
    fingerprint = "".join(
        f"src/flex_shard/{relative}:"
        + hashlib.sha256((root / relative).read_bytes()).hexdigest()
        + "\n"
        for relative in CORE_FILES
    )
    if hashlib.sha256(fingerprint.encode()).hexdigest() != CORE_SHA256:
        raise ValueError("FlexShard private core differs from the reviewed public pin")
    return {"commit": FLEX_SHARD_COMMIT, "core_sha256": CORE_SHA256, "path": str(root)}


def load_executor():
    """Load only the exact public graph executor named by the environment."""
    verify_core()
    name = "torchtitan_traceable_flex_executor"
    if name in sys.modules:
        return sys.modules[name]
    path_value = os.environ.get("TORCHTITAN_FLEX_SHARD_EXECUTOR")
    if not path_value:
        raise ValueError(
            "Set TORCHTITAN_FLEX_SHARD_EXECUTOR to graph_flex_shard.py from "
            f"torch_tpu commit {EXECUTOR_COMMIT}; "
            "put pinned FlexShard src on PYTHONPATH"
        )
    path = Path(path_value).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXECUTOR_SHA256:
        raise ValueError(
            "Traceable FlexShard executor differs from the tested public pin"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module
