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
    "check_scan_expert_parallel",
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


def check_scan_expert_parallel(config) -> None:
    """EP under scan: refuse what a scan body cannot trace or schedule.

    The in-body EP transports need static shapes, so the expert buffer is a
    fixed capacity: dropless worst case by default, or
    ``--tpu_config.ep_capacity_factor``. Three settings break that or are lost
    inside the body, and each would otherwise fail late or run silently worse:

    * ``ep_transport="all_to_all"``: upstream's transport reads split sizes on
      the host.
    * ``ep_transport="ragged_all_to_all"``, also what ``"auto"`` picks when EP >
      top_k: capturing the scan body fails with
      ``UnsupportedFakeTensorException`` (8 chips, EP = 4 and 8; the cause is
      swallowed by Dynamo and not yet found). Ring-of-Experts runs.
    * ``ep_capacity_overflow="grow"`` (tt#240): reads the expert counts on the
      host to size the buffer.
    * more EP token chunks than ``ep_max_inflight_chunks``:
      ``insert_ep_chunk_barriers`` bounds live chunks from node meta, which the
      scan body's re-trace drops, so every chunk's buffer could be live.
    """
    from torchtitan.experiments.tpu.components.token_dispatcher import (  # pylint: disable=g-import-not-at-top
        TpuAllToAllTokenDispatcher,
        TpuRoutedExperts,
    )

    if config.parallelism.expert_parallel_degree <= 1:
        return
    tpu = getattr(config, "tpu_config", None)
    model = config.model_spec.model
    dispatchers = [c for _, c, _, _ in model.traverse(TpuAllToAllTokenDispatcher.Config)]
    experts = [c for _, c, _, _ in model.traverse(TpuRoutedExperts.Config)]
    transport = getattr(tpu, "ep_transport", None)
    if transport == "all_to_all" or (
        transport is None and any(d.ep_transport == "all_to_all" for d in dispatchers)
    ):
        raise ValueError(
            "--compile.use-scan-layers with EP needs a static-shape transport; "
            "ep_transport='all_to_all' reads split sizes on the host. Use "
            "--tpu_config.ep_transport=auto (or ring_of_experts / ragged_all_to_all)."
        )
    from torchtitan.models.common.token_dispatcher import (  # pylint: disable=g-import-not-at-top
        LocalTokenDispatcher,
    )

    ep = config.parallelism.expert_parallel_degree
    for d in [None, *(c for _, c, _, _ in model.traverse(LocalTokenDispatcher.Config))]:
        chosen = transport or getattr(d, "ep_transport", None) or "auto"
        top_k = getattr(d, "top_k", ep)
        if chosen == "ragged_all_to_all" or (chosen == "auto" and ep > top_k):
            raise ValueError(
                f"--compile.use-scan-layers with EP={ep} would use the ragged "
                f"all-to-all transport (ep_transport={chosen!r}, top_k={top_k}), "
                "which does not trace inside a scan body yet. Use "
                "--tpu_config.ep_transport=ring_of_experts."
            )
    if getattr(tpu, "ep_capacity_overflow", None) == "grow" or any(
        getattr(d, "ep_capacity_overflow", "drop") == "grow" for d in dispatchers
    ):
        raise ValueError(
            "--compile.use-scan-layers cannot run ep_capacity_overflow='grow': it "
            "sizes the expert buffer from counts read on the host. Use a fixed "
            "capacity: the dropless default, or --tpu_config.ep_capacity_factor."
        )
    chunks = getattr(tpu, "ep_token_chunks", None)
    inflight = getattr(tpu, "ep_max_inflight_chunks", None)
    for e in experts or [None]:
        c = chunks if chunks is not None else getattr(e, "num_token_chunks", 1)
        w = inflight if inflight is not None else getattr(e, "max_inflight_chunks", 2)
        if c > w:
            raise ValueError(
                f"--compile.use-scan-layers with {c} EP token chunks and at most "
                f"{w} in flight: the chunk barriers do not reach inside the scan "
                "body, so all chunks could be live. Set --tpu_config.ep_token_chunks "
                f"<= {w}."
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
