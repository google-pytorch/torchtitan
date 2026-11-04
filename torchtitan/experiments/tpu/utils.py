# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.algorithms._checkpoint import checkpoint_wrapper


__all__ = [
    "ptd_checkpoint_wrapper_with_early_stop",
    "split_optimizers_by_mesh",
]


def ptd_checkpoint_wrapper_with_early_stop(
    module: torch.nn.Module,
    checkpoint_impl: checkpoint_wrapper.CheckpointImpl = checkpoint_wrapper.CheckpointImpl.NO_REENTRANT,
    checkpoint_fn=None,
    early_stop: bool = False,
    **checkpoint_fn_kwargs,
) -> torch.nn.Module:
    """Wraps a module with PyTorch Distributed's checkpoint_wrapper.

    This function is a wrapper around
    `torch.distributed.algorithms._checkpoint.checkpoint_wrapper`
    with an added `early_stop` parameter, although early stopping is not
    currently supported in g3.

    Args:
        module: The module to wrap.
        checkpoint_impl: The checkpoint implementation to use.
        checkpoint_fn: Optional function to use for checkpointing.
        early_stop: Whether to enable early stopping.
            Currently raises NotImplementedError if True.
        **checkpoint_fn_kwargs: Additional keyword arguments to pass to
            `checkpoint_fn`.

    Returns:
        The wrapped module.

    Raises:
        NotImplementedError: If `early_stop` is set to True.
    """
    if early_stop:
      raise NotImplementedError(
          "early_stop is not supported by ptd_checkpoint_wrapper."
      )

    checkpoint_fn_kwargs["early_stop"] = early_stop

    return checkpoint_wrapper.checkpoint_wrapper(
        module,
        checkpoint_impl=checkpoint_impl,
        checkpoint_fn=checkpoint_fn,
        **checkpoint_fn_kwargs,
    )


def split_optimizers_by_mesh(optimizers_container) -> bool:
    """Split each optimizer param group into one group per ``DeviceMesh``.

    Under Expert Parallelism, routed experts live on ``(efsdp, ep)`` while
    dense parameters live on ``(fsdp,)``; ``torch_tpu``'s fused and foreach
    AdamW require every param group to share a single mesh.

    Returns ``True`` if any param group was split across multiple meshes.
    """

    def _mesh_key(param: torch.Tensor) -> int | None:
        spec = getattr(param, "_spec", None)
        return None if spec is None else id(spec.mesh)

    did_split = False
    for opt in optimizers_container.optimizers:
        new_groups: list[dict] = []
        for group in opt.param_groups:
            params = list(group["params"])
            names = group.get("param_names")
            if names is not None and len(names) != len(params):
                raise ValueError(
                    f"param_names length ({len(names)}) != "
                    f"params length ({len(params)})"
                )
            options = {
                k: v
                for k, v in group.items()
                if k not in ("params", "param_names")
            }
            buckets: dict[int | None, dict[str, list]] = {}
            for i, param in enumerate(params):
                bucket = buckets.setdefault(
                    _mesh_key(param), {"params": [], "param_names": []}
                )
                bucket["params"].append(param)
                if names is not None:
                    bucket["param_names"].append(names[i])
            if len(buckets) > 1:
                did_split = True
            for bucket in buckets.values():
                if names is None:
                    del bucket["param_names"]
                new_groups.append({**bucket, **options})
        if len(new_groups) == len(opt.param_groups):
            continue
        opt.param_groups = []
        for group in new_groups:
            opt.add_param_group(group)
    return did_split
