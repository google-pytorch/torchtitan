"""JAX distributed utilities: sharding, mesh setup, parameter initialization.

This module owns the shared sharding-map matcher used by both the jax/ and
torchax/ experiments. The matcher supports two path styles:

- Exact paths with integer indices replaced by '*' (e.g.
  ``layers/*/attention/wq/kernel`` — jax style).
- Regex patterns (e.g. ``r'.*embedding\\.weight'`` — torchax style).

``_match_path`` tries, in order: exact match, exact match after replacing
integer path segments with ``*``, then a regex pattern fallback.
"""

import logging
import re

import jax
from flax import nnx

logger = logging.getLogger(__name__)

P = jax.sharding.PartitionSpec


def sharded_device_put(
    array,
    sharding,
    num_global_devices: int,
    num_local_devices: int,
):
    """Place an array (or tuple of arrays) on devices with the given sharding.

    Handles both single-host (direct ``jax.device_put``) and multi-host
    (assemble from addressable slices) setups. Accepts tuples so callers in
    torchax can pass grouped tensors.
    """
    if isinstance(array, tuple):
        return tuple(
            sharded_device_put(t, sharding, num_global_devices, num_local_devices)
            for t in array
        )

    if num_global_devices == num_local_devices:
        return jax.device_put(array, sharding)

    # Multi-host: each host only provides its addressable slices.
    shape = array.shape
    x_split = [
        jax.device_put(array[i], device)
        for device, i in sharding.addressable_devices_indices_map(shape).items()
    ]
    return jax.make_array_from_single_device_arrays(shape, sharding, x_split)


def _is_integer(token: str) -> bool:
    try:
        int(token)
        return True
    except (ValueError, TypeError):
        return False


def _process_sharding_name(name: str) -> str:
    """Replace integer path segments with '*'.

    Handles the common path separators used across jax/ and torchax/: ``.``,
    ``/``, and ``___`` (torchax scanned-layer flattened names). Integer
    segments are replaced so that ``layers/0/attention/wq/kernel`` and
    ``layers.0.attention.wq`` both map to their ``*``-bearing form.
    """
    for sep in ('.', '/', '___'):
        if sep in name:
            tokens = name.split(sep)
            tokens = ['*' if _is_integer(t) else t for t in tokens]
            name = sep.join(tokens)
    return name


def _match_path(path_str: str, sharding_map: dict) -> tuple | None:
    """Return the partition spec for ``path_str`` from ``sharding_map``.

    Tries in order:
      1. Exact match on ``path_str``.
      2. Exact match after replacing integer segments with ``*``.
      3. Regex match (``re.match``) of each map key against both forms.
    """
    if path_str in sharding_map:
        return sharding_map[path_str]

    processed = _process_sharding_name(path_str)
    if processed != path_str and processed in sharding_map:
        return sharding_map[processed]

    for pattern, spec in sharding_map.items():
        try:
            if re.match(pattern, path_str) or re.match(pattern, processed):
                return spec
        except re.error:
            pass
    return None


def apply_sharding_to_state(
    state: nnx.State,
    sharding_map: dict,
    mesh: jax.sharding.Mesh,
) -> nnx.State:
    """Apply JAX sharding to every leaf in an NNX state pytree.

    Args:
        state: NNX state (e.g. from nnx.split(model)).
        sharding_map: dict mapping path patterns to partition spec tuples.
        mesh: JAX device mesh.
    Returns:
        New state with all arrays placed on-device with proper sharding.
    """
    def shard_leaf(path, array):
        # Build a '/'-separated path string from the pytree key path.
        # nnx.Variable registers itself as a pytree with a trailing
        # GetAttrKey('value') leaf — strip that so paths match the sharding map.
        parts = []
        for k in path:
            if isinstance(k, jax.tree_util.DictKey):
                parts.append(str(k.key))
            elif isinstance(k, jax.tree_util.GetAttrKey):
                if k.name != 'value':
                    parts.append(k.name)
            elif isinstance(k, jax.tree_util.SequenceKey):
                parts.append(str(k.idx))
            else:
                parts.append(str(k))
        path_str = '/'.join(p for p in parts if p)

        spec = _match_path(path_str, sharding_map)
        if spec is None:
            logger.warning('No sharding spec for %s — replicating.', path_str)
            spec = ()
        sharding = jax.sharding.NamedSharding(mesh, P(*spec))
        logger.info(
            'Sharding %s  shape=%s  spec=%s',
            path_str, array.shape, spec,
        )
        # Use sharded_device_put so multi-host runs (num_global_devices >
        # num_local_devices) assemble per-host addressable slices into a
        # global jax.Array, instead of attempting an unsupported cross-host
        # device_put. The model is built deterministically on each host's
        # CPU with the same RNG seed, so every host already holds the full
        # replicated array and just needs to peel off its local shard.
        # Materialise as numpy first: slicing a CPU-bound jax.Array and
        # then device_put-ing the slice triggers "Cannot reshard an input
        # that is not fully addressable" because each sliced jax.Array is
        # still single-device-committed. numpy arrays have no sharding so
        # they assemble cleanly into a multi-host jax.Array.
        import numpy as _np
        array_np = _np.asarray(jax.device_get(array))
        return sharded_device_put(
            array_np, sharding,
            num_global_devices=len(jax.devices()),
            num_local_devices=jax.local_device_count(),
        )

    return jax.tree_util.tree_map_with_path(shard_leaf, state)


def shard_input(
    x,
    mesh: jax.sharding.Mesh,
    num_global_devices: int,
    num_local_devices: int,
):
    """Shard a batch input tensor across the FSDP axis (batch dimension)."""
    sharding = jax.sharding.NamedSharding(mesh, P('fsdp'))
    return sharded_device_put(x, sharding, num_global_devices, num_local_devices)
