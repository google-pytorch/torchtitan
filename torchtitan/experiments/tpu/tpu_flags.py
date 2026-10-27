"""XLA and libtpu init flags carried by TPU configs.

The ``XLA_FLAGS`` and ``LIBTPU_INIT_ARGS`` environment variables are part of
``BaseTPUTrainerConfig`` via the ``xla_flags`` and ``libtpu_init_flags`` fields,
so the flags a recipe needs are part of the training configuration instead of
being documented only at the launcher level. ``apply_init_flags`` renders them
before the first TPU device is used, for the ``libtpu.so`` that parses them out
of the environment as it loads.

Their override precedence is as follows:

  1. Config: the whole set a recipe declares. Setting ``--libtpu-init-flags`` or
     ``--xla-flags`` on the command line replaces that set.
  2. CLI: ``--extra-libtpu-init-flags`` and ``--extra-xla-flags`` add flags the
     config does not set and override the ones it does, leaving the rest alone;
     ``!flag`` drops one.
  3. Environment: an exported ``LIBTPU_INIT_ARGS`` or ``XLA_FLAGS`` has the last
     word. It names the process being launched as specifically as a config can,
     so it is not overwritten.

Values are whitespace-separated ``flag=value`` entries. Groups of flags that
only make sense together are the named bundles below, composed with
``merge_flags``.
"""

import os
from typing import Iterable, Mapping, MutableMapping

import tyro

from torchtitan.config.manager import custom_registry
from torchtitan.tools.logging import logger


FlagValue = bool | int | float | str

# Value marking a flag as removed rather than set: a later layer writes it over
# an earlier layer's flag and the merged result drops the entry, which is how a
# launch command takes a flag off a recipe without having to know the value
# libtpu would otherwise have used. Spelled "!flag"; the NUL keeps the sentinel
# from colliding with any value a real flag could carry.
REMOVE = "\x00remove"


class FlagMap(dict[str, str]):
  """A map of bare flag name -> value, parsed from one command-line string.

  A plain ``dict`` field would take its CLI options from the keys present in its
  default, so a config that already declares flags could only retune those --
  never add one. Giving the field its own type lets the parser rule below accept
  the same ``flag=value`` string the environment variables use, whatever the
  config declares.
  """


@custom_registry.primitive_rule
def _flag_map_rule(
    type_info: tyro.constructors.PrimitiveTypeInfo,
) -> tyro.constructors.PrimitiveConstructorSpec | None:
  """Parses a ``FlagMap`` field from one ``flag=value ...`` string."""
  if type_info.type is not FlagMap:
    return None
  return tyro.constructors.PrimitiveConstructorSpec(
      nargs=1,
      metavar='"FLAG=VALUE ..."',
      instance_from_str=lambda args: FlagMap(parse_flags(args[0])),
      is_instance=lambda instance: isinstance(instance, dict),
      str_from_instance=lambda instance: [format_flags(instance)],
  )


def merge_flags(
    *bundles: Mapping[str, FlagValue], **overrides: FlagValue
) -> FlagMap:
  """Merges bundles then keyword overrides, later arguments winning."""
  merged = FlagMap()
  for layer in (*bundles, overrides):
    merged.update({name: _flag_value_str(value) for name, value in layer.items()})
  return merged


def format_flags(flags: Mapping[str, str]) -> str:
  """Renders a flag map as a command-line string for libtpu or XLA."""
  return " ".join(_format_flag(name, value) for name, value in flags.items())


def parse_flags(flag_string: str) -> FlagMap:
  """Parses whitespace-separated ``name=value`` entries; ``!name`` removes."""
  flags = FlagMap()
  for entry in flag_string.split():
    name, _, value = entry.lstrip("-").partition("=")
    if name.startswith("!"):
      name, value = name[1:], REMOVE
    if name:
      flags[name] = value
  return flags


def resolve_flags(*layers: Mapping[str, str]) -> FlagMap:
  """Merges flag layers in increasing precedence, dropping removed flags."""
  merged = FlagMap()
  for layer in layers:
    merged.update(layer)
  return FlagMap(
      {name: value for name, value in merged.items() if value != REMOVE}
  )


def apply_init_flags(
    config,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> None:
  """Applies a config's flags, before any TPU device is used.

  Exports ``LIBTPU_INIT_ARGS`` and ``XLA_FLAGS`` for a ``libtpu.so`` to parse.
  The consumer latches their values while the first program is compiled, so
  flags applied after that point are silently ignored.

  Args:
    config: The trainer config carrying the four flag fields.
    environ: The environment to render into, ``os.environ`` by default. Passing
      a mapping of one's own renders the variables without reconfiguring this
      process, which is what the tests want and what a caller preparing a
      subprocess would.
  """
  if environ is None:
    environ = os.environ
  _apply_flags(
      "LIBTPU_INIT_ARGS",
      config.libtpu_init_flags,
      config.extra_libtpu_init_flags,
      environ,
  )
  _apply_flags(
      "XLA_FLAGS",
      config.xla_flags,
      config.extra_xla_flags,
      environ,
  )


# ---------------------------------------------------------------------------
# libtpu flag bundles
# ---------------------------------------------------------------------------

# Let the latency-hiding layer scheduler overlap FSDP collectives with compute.
# The scheduler requires the SparseCore collective aggregator, so the two flags
# move together.
SPARSECORE_LATENCY_HIDING_SCHEDULER: FlagMap = {
    "xla_tpu_enable_sparse_core_collective_aggregator": "true",
    "xla_tpu_enable_latency_hiding_layer_scheduler": "true",
}

# Let eligible reduce-scatter operations use both SparseCores. Optimal for a
# Zero-2 schedule, where backward has no parameter all-gathers competing with
# reduce-scatter, and for a dependency-ordered Zero-3 interleave that cannot
# overlap its all-gathers and reduce-scatters anyway. Re-evaluate for Zero-3
# prefetch or any schedule that can make those collectives concurrent.
DUAL_SPARSECORE_REDUCE_SCATTER: FlagMap = {
    "xla_tpu_use_single_sparse_core_for_reduce_scatter_offload": "false",
}

# Split eligible multi-dimensional reduce-scatter operations into hierarchical
# stages and run those stages across both SparseCores. This lowering is
# available in libtpu 0.0.47 and newer. Keep the three flags together: the ND
# offload path exposes the topology to the hierarchical lowering, while using
# both SparseCores provides the measured throughput gain.
HIERARCHICAL_REDUCE_SCATTER: FlagMap = {
    "xla_tpu_use_single_sparse_core_for_reduce_scatter_offload": "false",
    "xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter": "true",
    "xla_tpu_enable_sparse_core_hierarchical_reduce_scatter": "true",
}

# Pipeline small (<= 1024 KiB per shard) reduce-scatters across the scan
# backward while-loop back-edge while keeping multi-MiB weight reductions
# inside their own iteration to avoid double-buffering in VMEM/HBM.
SELECTIVE_ICI_RS_PIPELINING: FlagMap = {
    "xla_tpu_enable_ici_rs_pipelining": "true",
    "xla_tpu_ici_rs_pipelining_threshold_kib": "1024",
}

# Start each layer's FSDP weight all-gather one scan iteration early, so it
# overlaps the previous layer's compute instead of sitting on the critical path.
# Costs extra HBM for the prefetched next-layer weights.
ICI_AG_PIPELINING: FlagMap = {
    "xla_tpu_enable_ici_ag_pipelining": "true",
}

# Offload eligible FSDP all-gathers to SparseCore asynchronously and prefer
# async all-gather over all-reduce.
SPARSECORE_ASYNC_ALL_GATHER: FlagMap = {
    "xla_enable_async_all_gather": "true",
    "xla_tpu_prefer_async_allgather_to_allreduce": "true",
    "xla_tpu_enable_sparse_core_collective_offload_all_gather": "true",
}

# Unfuse elementwise ops from MXU matmuls when fusing degrades tile selection
# (e.g. restores the 416x8 FFN backward tile on Llama3-405B scan).
CONVOLUTION_UNFUSING: FlagMap = {
    "xla_tpu_enable_convolution_unfusing": "true",
}

# Offload FSDP all-gather and reduce-scatter onto the SparseCores, queued by the
# latency-hiding scheduler. Broader than
# SPARSECORE_LATENCY_HIDING_SCHEDULER: this moves the collectives themselves off
# the TensorCore rather than only rescheduling them.
SPARSECORE_COLLECTIVE_OFFLOAD: FlagMap = {
    "xla_tpu_enable_sparse_core_collective_aggregator": "true",
    "xla_tpu_enable_sparse_core_offload_queuing_in_lhs": "true",
    "xla_tpu_enable_sparse_core_collective_offload_2d_all_gather": "true",
    "xla_tpu_enable_sparse_core_collective_offload_3d_all_gather": "true",
    "xla_tpu_enable_sparse_core_collective_offload_reduce_scatter": "true",
}

# Request TPU7x performance state 7, the throughput-oriented operating point.
# This trades higher device power for faster compute; it does not change the
# graph.
TPU7X_THROUGHPUT_P_STATE: FlagMap = {
    "xla_tpu_dvfs_p_state": "7",
}

# Avoid the distributed hangs the enhanced launch barrier causes (b/477673365).
# torch_tpu already forces this off for distributed runs when the flag is
# absent; recipes benchmarked with it set explicitly keep it explicit.
DISABLE_ENHANCED_LAUNCH_BARRIER: FlagMap = {
    "xla_tpu_use_enhanced_launch_barrier": "false",
}

# MaxText's latency-hiding scheduler settings for its Ironwood MoE recipes:
# rerun the scheduler, track memory pressure, and prefer deeper async overlap
# over avoiding stalls. They hide more of Ring-of-Experts' token collectives
# but slow the FSDP-only Qwen3-235B recipe.
EXPERIMENTAL_LATENCY_HIDING_SCHEDULER: FlagMap = {
    "xla_tpu_enable_all_experimental_scheduler_features": "true",
    "xla_tpu_enable_scheduler_memory_pressure_tracking": "true",
    "xla_lhs_prioritize_async_depth_over_stall": "ENABLED",
    "xla_latency_hiding_scheduler_rerun": "2",
}


def _apply_flags(
    env_var: str,
    config_flags: Mapping[str, str],
    extra_flags: Mapping[str, str],
    environ: MutableMapping[str, str],
) -> None:
  """Merges the config and ``extra_`` layers under the existing env var."""
  env_flags = parse_flags(environ.get(env_var, ""))
  if not config_flags and not extra_flags:
    return
  for name in _removed_names(config_flags, extra_flags):
    logger.info(
        "Removing %s %s=%s at the request of the launch command.",
        env_var,
        name,
        config_flags[name],
    )
  for name in _conflicting_names(config_flags, extra_flags):
    if extra_flags[name] == REMOVE:
      continue
    logger.info(
        "Overriding %s %s=%s with %s from the launch command.",
        env_var,
        name,
        config_flags[name],
        extra_flags[name],
    )
  declared = {**config_flags, **extra_flags}
  for name in _conflicting_names(declared, env_flags):
    logger.warning(
        "%s already sets %s=%s in the environment; keeping it and ignoring the"
        " config value %s.",
        env_var,
        name,
        env_flags[name],
        declared[name],
    )
  resolved = resolve_flags(declared, env_flags)
  environ[env_var] = format_flags(resolved)
  logger.info("Applied %s=%s", env_var, environ[env_var])


def _removed_names(
    config_flags: Mapping[str, str], extra_flags: Mapping[str, str]
) -> Iterable[str]:
  """Returns the config's flags that ``extra_flags`` asks to remove."""
  return sorted(
      name
      for name, value in extra_flags.items()
      if value == REMOVE and name in config_flags
  )


def _conflicting_names(
    config_flags: Mapping[str, str], env_flags: Mapping[str, str]
) -> Iterable[str]:
  """Returns the flags both maps set, to a different value."""
  return sorted(
      name
      for name, value in env_flags.items()
      if name in config_flags and config_flags[name] != value
  )


def _flag_value_str(value: FlagValue) -> str:
  """Renders a flag value the way libtpu and XLA parse it."""
  # libtpu and XLA parse booleans as lowercase "true"/"false", which is not what
  # str(True) produces.
  if isinstance(value, bool):
    return "true" if value else "false"
  return str(value)


def _format_flag(name: str, value: str) -> str:
  """Renders one flag, valueless when the value is empty."""
  return f"--{name}={value}" if value else f"--{name}"
