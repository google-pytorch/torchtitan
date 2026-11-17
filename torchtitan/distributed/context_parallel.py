# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel partitioning and load-balancing APIs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import spmd_types as spmd
import torch
from spmd_types import SpmdType

# TODO(acisseJZhong): Move these implementations into TorchTitan to avoid
# depending on private PyTorch context-parallel APIs.
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _HeadTailLoadBalancer,
    _PTRRLoadBalancer,
)
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config.configurable import Configurable
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import (
    _per_axis_types,
    current_spmd_mesh,
    spmd_mesh_group,
)

if TYPE_CHECKING:
    from torchtitan.models.common.attention import (
        AttentionMasksType,
        FlexAttentionMetadata,
    )

__all__ = [
    "ContextParallelLoadBalancer",
    "HeadTailCPLoadBalancer",
    "PTRRFlexAttentionCPLoadBalancer",
    "_cp_shard_dims",
    "cp_shard",
    "get_cp_input_seq_len",
    "prepare_context_parallel_input",
    "shard_tensors",
]


class ContextParallelLoadBalancer(Configurable, ABC):
    """Generate a context-parallel token permutation for one batch."""

    @abstractmethod
    def generate_permutation(self) -> torch.Tensor:
        """Generate the global token permutation for this batch."""
        raise NotImplementedError


class HeadTailCPLoadBalancer(ContextParallelLoadBalancer):
    """Balance CP tokens using PyTorch's head-tail strategy."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configuration for head-tail context-parallel load balancing."""

    def __init__(
        self,
        config: Config,
        *,
        seq_len: int,
        attention_metadata: AttentionMasksType | None,
    ) -> None:
        del config, attention_metadata
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        spmd_mesh = current_spmd_mesh()
        if cp_group is None or spmd_mesh is None:
            raise RuntimeError(
                "CP load balancing requires an active multi-rank CP mesh axis."
            )
        self.seq_len = seq_len
        self.cp_size = cp_group.size()
        self.device = spmd_mesh.device_type
        if seq_len % (2 * self.cp_size) != 0:
            raise ValueError(
                f"Head-tail load balancing requires sequence length ({seq_len}) "
                f"to be divisible by 2 * CP size ({2 * self.cp_size})."
            )

    def generate_permutation(self) -> torch.Tensor:
        """Generate a head-tail token permutation."""
        permutation = _HeadTailLoadBalancer(
            self.seq_len, self.cp_size, self.device
        )._generate_indices()
        return permutation


# TODO(acisseJZhong): Maintain an explicit mapping from each CP load balancer
# to the CP attention backends it supports, and validate configured pairs.
class PTRRFlexAttentionCPLoadBalancer(ContextParallelLoadBalancer):
    """Balance FlexAttention tokens with PTRR.

    The current cost model assumes a K/V all-gather CP backend.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configuration for PTRR context-parallel load balancing."""

        mask_key: str | None = None
        """Mask used to derive the partition when context metadata is a mapping."""

    def __init__(
        self,
        config: Config,
        *,
        seq_len: int,
        attention_metadata: FlexAttentionMetadata,
    ) -> None:
        del seq_len
        mask_key = config.mask_key

        if attention_metadata is None:
            raise ValueError(
                "PTRR load balancing requires context metadata to be a BlockMask "
                "or Mapping[str, BlockMask], but got None."
            )
        if isinstance(attention_metadata, Mapping):
            if mask_key is None:
                raise ValueError(
                    "PTRR load balancing received a Mapping[str, BlockMask] but no "
                    "mask key was specified. Set "
                    "PTRRFlexAttentionCPLoadBalancer.Config(mask_key=...) "
                    "to one of: "
                    f"{sorted(attention_metadata.keys())}"
                )
            if mask_key not in attention_metadata:
                raise ValueError(
                    f"PTRR mask key '{mask_key}' is not a key in context metadata. "
                    f"Available keys: {sorted(attention_metadata.keys())}"
                )
            block_mask = attention_metadata[mask_key]
        else:
            block_mask = attention_metadata
        if not isinstance(block_mask, BlockMask):
            raise ValueError(
                "PTRR load balancing requires the selected metadata to be a "
                f"BlockMask, but got {type(block_mask).__name__}."
            )
        self.block_mask = block_mask
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "CP load balancing requires an active multi-rank CP mesh axis."
            )
        self.cp_size = cp_group.size()

    def generate_permutation(self) -> torch.Tensor:
        """Generate a PTRR token permutation from the selected BlockMask."""
        permutation = _PTRRLoadBalancer(
            self.block_mask, self.cp_size
        )._generate_indices()
        return permutation


def _cp_shard_dims(input_shardings: dict[str, SpmdType]) -> dict[str, int]:
    """Derive ``{name: seq_dim}`` for inputs whose CP mesh axis is a Shard.

    Inputs whose CP axis is Replicate/Partial (e.g. an image stream that is
    not sequence-sharded) are omitted and thus left untouched by CP.
    """
    dims: dict[str, int] = {}
    for name, layout in input_shardings.items():
        axis_type = _per_axis_types(layout).get(MeshAxisName.CP)
        if isinstance(axis_type, spmd.Shard):
            dims[name] = axis_type.dim
    return dims


def get_cp_input_seq_len(
    input_dict: Mapping[str, Any],
    *,
    input_shardings: dict[str, SpmdType],
) -> int:
    """Return the pre-sharding sequence length shared by the CP inputs."""
    seq_lens: dict[str, int] = {}
    for name, seq_dim in _cp_shard_dims(input_shardings).items():
        value = input_dict.get(name)
        if isinstance(value, torch.Tensor):
            seq_lens[name] = value.size(seq_dim)

    if not seq_lens:
        raise ValueError("CP requires at least one declared sequence tensor.")
    if len(set(seq_lens.values())) != 1:
        raise ValueError(
            "All CP-sharded tensors must have the same sequence length, "
            f"but got {seq_lens}."
        )
    return next(iter(seq_lens.values()))


def _permute_tensor(
    tensor: torch.Tensor,
    *,
    seq_dim: int,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Apply a CP token permutation using PyTorch CP indexing semantics."""
    if permutation.ndim != 2:
        raise ValueError(
            "CP permutation must have shape [1, seq_len] or [batch, seq_len], "
            f"but got {tuple(permutation.shape)}."
        )
    if permutation.shape[1] != tensor.shape[seq_dim]:
        raise ValueError(
            f"CP permutation length ({permutation.shape[1]}) must match tensor "
            f"sequence length ({tensor.shape[seq_dim]})."
        )
    permutation_batch_size = permutation.shape[0]
    tensor_batch_size = tensor.shape[0] if seq_dim > 0 else 1
    if permutation_batch_size not in (1, tensor_batch_size):
        raise ValueError(
            "CP permutation batch size must be 1 or match the tensor batch "
            f"size, but got {permutation_batch_size} and {tensor_batch_size}."
        )

    if seq_dim == 0:
        # tensor has shape [seq_len] or [seq_len, ...]
        # Just use the first (and only) batch of indices
        tensor = torch.index_select(tensor, dim=0, index=permutation[0])
    else:
        indices = permutation
        if permutation_batch_size == 1:
            indices = indices.expand(tensor_batch_size, -1)

        # permutation has shape [B, seq_len] where:
        #   - dim 0 corresponds to tensor dim 0 (batch)
        #   - dim 1 corresponds to tensor dim seq_dim
        # Need to insert dimensions for all dims between 0 and seq_dim,
        # and all dims after seq_dim.

        # Insert dimensions between batch (dim 0) and seq_dim
        for dim in range(1, seq_dim):
            indices = indices.unsqueeze(dim)

        # Insert dimensions after seq_dim
        for _ in range(seq_dim + 1, tensor.ndim):
            indices = indices.unsqueeze(-1)

        # Expand to match tensor's shape
        indices = indices.expand(tensor.shape)
        tensor = torch.gather(tensor, dim=seq_dim, index=indices)

    return tensor


def shard_tensors(
    input_dict: dict[str, Any],
    *,
    input_shardings: dict[str, SpmdType],
    permutation: torch.Tensor | None,
) -> dict[str, Any]:
    """Permute and shard declared tensor inputs over the active CP mesh axis."""
    shard_dims = _cp_shard_dims(input_shardings)
    shard_names = tuple(
        name for name in shard_dims if isinstance(input_dict.get(name), torch.Tensor)
    )
    if not shard_names:
        return input_dict

    tensors = cast(
        tuple[torch.Tensor, ...], tuple(input_dict[name] for name in shard_names)
    )
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors[1:]):
        raise ValueError("All CP-sharded inputs must be on the same device.")
    if permutation is not None and permutation.device != device:
        raise ValueError("The CP permutation and sharded inputs must share a device.")

    cp_group = spmd_mesh_group(MeshAxisName.CP)
    if cp_group is None:
        raise RuntimeError("CP sharding requires an active multi-rank CP mesh axis.")
    cp_size = cp_group.size()

    for name, tensor in zip(shard_names, tensors, strict=True):
        seq_dim = shard_dims[name]
        if tensor.shape[seq_dim] % cp_size != 0:
            raise ValueError(
                f"CP-sharded input '{name}' sequence length "
                f"({tensor.shape[seq_dim]}) must be divisible by CP size "
                f"({cp_size})."
            )
        if permutation is not None:
            tensor = _permute_tensor(
                tensor,
                seq_dim=seq_dim,
                permutation=permutation,
            )
        input_dict[name] = spmd.shard(
            tensor,
            cp_group,
            src=spmd.R,
            dst=spmd.S(seq_dim),
        )
    return input_dict


def prepare_context_parallel_input(
    input_dict: dict[str, Any],
    input_shardings: dict[str, SpmdType] | None,
    cp_mesh: DeviceMesh,
    load_balancer_type: str | None = "headtail",
    ptrr_mask_key: str | None = None,
) -> dict[str, Any]:
    """Shard named tensors and attention masks for Context Parallel."""
    if input_shardings is not None:
        shard_dims = _cp_shard_dims(input_shardings)
    else:
        shard_dims = {"input": 0, "labels": 0, "positions": 0}

    named: dict[str, torch.Tensor] = {
        k: v for k, v in input_dict.items() if isinstance(v, torch.Tensor)
    }

    shard_names = [n for n in shard_dims if n in named]
    if not shard_names:
        return input_dict
    buffers = tuple(named[n] for n in shard_names)
    seq_dims = tuple(shard_dims[n] for n in shard_names)

    attention_masks = input_dict.get("attention_masks", None)
    sharded_buffers, attention_masks = cp_shard(
        cp_mesh,
        buffers,
        attention_masks,
        load_balancer_type,
        input_seq_dims=seq_dims,
        ptrr_mask_key=ptrr_mask_key,
    )

    for n, buf in zip(shard_names, sharded_buffers):
        input_dict[n] = buf
    if attention_masks is not None:
        input_dict["attention_masks"] = attention_masks
    return input_dict


def cp_shard(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    attention_masks: AttentionMasksType | None,
    load_balancer_type: str | None = "headtail",
    input_seq_dims: int | tuple[int, ...] = 0,
    ptrr_mask_key: str | None = None,
) -> tuple[tuple[torch.Tensor, ...], AttentionMasksType | None]:
    if isinstance(input_seq_dims, tuple):
        assert len(input_seq_dims) == len(inputs)
        seq_dims = input_seq_dims
    else:
        seq_dims = tuple(input_seq_dims for _ in inputs)
    seq_len = inputs[0].size(seq_dims[0])
    cp_world_size = cp_mesh.size(0)

    load_balancer = None
    if load_balancer_type:
        match load_balancer_type:
            case "headtail":
                load_balancer = _HeadTailLoadBalancer(
                    seq_len, cp_world_size, cp_mesh.device_type
                )
            case "ptrr":
                if attention_masks is None:
                    raise ValueError(
                        "PTRRLoadBalancer requires attention_masks to be a "
                        "BlockMask or dict[str, BlockMask], but got None"
                    )
                if isinstance(attention_masks, dict):
                    if ptrr_mask_key is None:
                        raise ValueError(
                            "PTRRLoadBalancer received a dict[str, BlockMask] "
                            "but no mask key was specified. Set "
                            "--parallelism.context_parallel_ptrr_mask_key to "
                            f"one of: {sorted(attention_masks.keys())}"
                        )
                    if ptrr_mask_key not in attention_masks:
                        raise ValueError(
                            f"context_parallel_ptrr_mask_key '{ptrr_mask_key}' "
                            f"is not a key in attention_masks. Available keys: "
                            f"{sorted(attention_masks.keys())}"
                        )
                    ptrr_mask = attention_masks[ptrr_mask_key]
                else:
                    ptrr_mask = attention_masks
                if not isinstance(ptrr_mask, BlockMask):
                    raise ValueError(
                        f"PTRRLoadBalancer requires the mask to be a "
                        f"BlockMask, but got {type(ptrr_mask)}"
                    )
                load_balancer = _PTRRLoadBalancer(ptrr_mask, cp_world_size)
            case _:
                raise ValueError(
                    f"Invalid load_balancer_type '{load_balancer_type}'. "
                    f"Must be one of: 'headtail', 'ptrr', or None"
                )

    inputs = cast(
        tuple[torch.Tensor, ...],
        _context_parallel_shard(
            mesh=cp_mesh,
            buffers=inputs,
            seq_dims=seq_dims,
            load_balancer=load_balancer,
        ),
    )

    MASK_Q_SEQ_DIM = 2
    if attention_masks is not None:
        assert isinstance(attention_masks, (BlockMask, dict))
        masks: list[BlockMask] = []
        for mask in (
            [attention_masks]
            if isinstance(attention_masks, BlockMask)
            else attention_masks.values()
        ):
            if not isinstance(mask, BlockMask):
                raise ValueError(
                    "Context parallelism can only shard BlockMask attention "
                    f"masks, got {type(mask).__name__} in the mask dict."
                )
            masks.append(mask)
        sharded_masks = _context_parallel_shard(
            mesh=cp_mesh,
            buffers=masks,
            seq_dims=(MASK_Q_SEQ_DIM,) * len(masks),
            load_balancer=load_balancer,
        )
        attention_masks = cast(
            (BlockMask | dict[str, BlockMask]),
            (
                sharded_masks[0]
                if isinstance(attention_masks, BlockMask)
                else {k: v for k, v in zip(attention_masks.keys(), sharded_masks)}
            ),
        )

    return inputs, attention_masks
