"""Gate optional executor tests without disabling dependency-free contracts."""

import os


def require_executor(test_case):
    if not os.environ.get("TORCHTITAN_FLEX_SHARD_EXECUTOR"):
        test_case.skipTest(
            "Optional pinned FlexShard executor is not configured; see README.md"
        )
