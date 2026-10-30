"""Refuse to run when ``--compile.use-scan-layers`` cannot be honoured.

Every failure mode in here is otherwise silent. The run succeeds, either
unrolled or stacked from layer 0, and the log looks like a correct scan run. So
none of these guards fall back to anything: they raise, and they name the flag
to change.
"""

from torch import nn

from torchtitan.config.configs import ParallelismConfig
from torchtitan.experiments.tpu.graph_trainer.layer_stack import StackedLayers


__all__ = [
    "check_scan_parallelism",
    "check_scan_was_applied",
]


def check_scan_parallelism(parallelism: ParallelismConfig) -> None:
    """Scan is 1-D data parallel only.

    The in-body gather assumes each stacked weight has exactly one shard axis.
    TP and CP add a second one, and PP splits up the very layer range the scan
    is looping over.

    This reads the config rather than the resolved ``ParallelDims`` because the
    config is where scan is decided, and that happens before anything is built.
    Only ``data_parallel_shard_degree`` is allowed to be negative, so the three
    degrees checked here are already exact numbers.
    """
    conflicts = [
        f"--parallelism.{kind}-parallel-degree={degree}"
        for kind, degree in (
            ("tensor", parallelism.tensor_parallel_degree),
            ("context", parallelism.context_parallel_degree),
            ("pipeline", parallelism.pipeline_parallel_degree),
        )
        if degree > 1
    ]
    if conflicts:
        raise ValueError(
            "--compile.use-scan-layers supports 1-D data parallelism only, "
            f"but this run sets {', '.join(conflicts)}. Drop the flag, or set "
            "those degrees to 1."
        )


def check_scan_was_applied(model: nn.Module, *, use_scan_layers: bool) -> None:
    """After parallelization: did the flag actually reach anything?

    This asks the built model what it ended up with, rather than checking a
    hand-maintained list of supported lanes. A lane added later that forgets to
    wire the flag up then fails on its first run instead of going unnoticed.
    """
    if not use_scan_layers:
        return
    # Check `use_scan`, not just "is it stacked". The container can be
    # installed without scan, so a lane that did only that would otherwise
    # pass this check.
    if any(isinstance(m, StackedLayers) and m.use_scan for m in model.modules()):
        return
    raise ValueError(
        "--compile.use-scan-layers was set, but nothing in this run "
        "consulted it: the decoder layers ran one at a time. Scan is wired up "
        "only for recipes that return TpuGraphTrainer.Config."
    )
