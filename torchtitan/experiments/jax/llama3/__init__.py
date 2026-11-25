"""Llama3 model for JAX experiment."""

from .model import ModelArgs, Transformer
from .model import llama3_8b, llama3_70b, llama3_debug
from .sharding import sharding_map_original, sharding_map_scan

# Flavor keys match upstream ``torchtitan.models.llama3.llama3_configs`` so the
# cross-lane registries can look a flavor up by the same name.
# ``debug`` kept as a back-compat alias for tests that still launch with
# ``--model.flavor=debug``.
args = {
    'debugmodel': llama3_debug,
    'debug': llama3_debug,
    '8B': llama3_8b,
    '70B': llama3_70b,
}

__all__ = [
    'ModelArgs',
    'Transformer',
    'args',
    'sharding_map_original',
    'sharding_map_scan',
]
