# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Store expert dWeights as FP32 when an FP32 reduction widens them anyway.

The expert TGMM kernels accumulate dWeight in FP32 and round it to the BF16
weight dtype at the final store, as autograd requires. FP32 FSDP reduction
then widens that BF16 tensor in a separate full-size HBM pass:

    grouped_mm_bwd -> getitem (BF16) -> [views] -> _to_copy(FP32) -> RS

This pass swaps in the op variant that stores the same BF16-rounded values as
FP32, re-types the views, and drops the widening. Values are bit-identical;
only the HBM round trip disappears. A backward op is rewritten only when every
dWeight it returns reaches an FP32 ``_to_copy`` through single-user views.
"""

from __future__ import annotations

import operator

import torch
from torch._guards import detect_fake_mode
from torch.fx.passes.shape_prop import _extract_tensor_metadata

# Imported for the side effect: they register the tt_experts ops.
import torchtitan.experiments.tpu.kernels.expert_segment_matmul  # noqa: F401
import torchtitan.experiments.tpu.kernels.paired_segment_matmul  # noqa: F401
from torchtitan.tools.logging import logger


# Backward op -> (FP32-dWeight variant, getitem indices of its dWeights).
_FP32_DWEIGHT_VARIANTS = {
    torch.ops.tt_experts.grouped_mm_bwd.default: (
        torch.ops.tt_experts.grouped_mm_bwd_fp32_dweight.default,
        (1,),
    ),
    torch.ops.tt_experts.paired_grouped_mm_bwd.default: (
        torch.ops.tt_experts.paired_grouped_mm_bwd_fp32_dweight.default,
        (1, 2),
    ),
}


def _widening_chain(getitem: torch.fx.Node) -> list[torch.fx.Node] | None:
    """Return ``[getitem, *views, upcast]`` if the dWeight is only widened."""
    chain = [getitem]
    node = getitem
    while len(node.users) == 1:
        (user,) = node.users
        if user.op != "call_function" or user.args[0] is not node:
            return None
        if user.target is torch.ops.aten._to_copy.default:
            value = user.meta.get("val")
            if (
                set(user.kwargs) - {"dtype", "layout", "device"}
                or user.kwargs.get("dtype") is not torch.float32
                or not isinstance(value, torch.Tensor)
                or value.device != node.meta["val"].device
            ):
                return None
            return [*chain, user]
        if not getattr(user.target, "is_view", False):
            return None
        chain.append(user)
        node = user
    return None


def _set_value(node: torch.fx.Node, value) -> None:
    node.meta["val"] = value
    if "tensor_meta" in node.meta:
        node.meta["tensor_meta"] = _extract_tensor_metadata(value)


def apply_fp32_expert_dweight_pass(
    gm: torch.fx.GraphModule, example_inputs
) -> torch.fx.GraphModule:
    del example_inputs
    rewrites = 0
    for backward in list(gm.graph.nodes):
        if backward.op != "call_function":
            continue
        variant = _FP32_DWEIGHT_VARIANTS.get(backward.target)
        if variant is None:
            continue
        fp32_target, dweight_indices = variant
        getitems = {
            user.args[1]: user
            for user in backward.users
            if user.target is operator.getitem
        }
        chains = [
            _widening_chain(getitems[index]) if index in getitems else None
            for index in dweight_indices
        ]
        if any(chain is None for chain in chains):
            continue

        fake_mode = detect_fake_mode(list(backward.meta["val"]))
        with fake_mode:
            values = list(backward.meta["val"])
            for index in dweight_indices:
                values[index] = torch.empty_like(
                    values[index], dtype=torch.float32
                )
            backward.target = fp32_target
            _set_value(backward, tuple(values))
            for index, chain in zip(dweight_indices, chains, strict=True):
                getitem, *views, upcast = chain
                _set_value(getitem, values[index])
                for view in views:
                    args = [
                        arg.meta["val"] if isinstance(arg, torch.fx.Node) else arg
                        for arg in view.args
                    ]
                    _set_value(view, view.target(*args, **view.kwargs))
                upcast.replace_all_uses_with(upcast.args[0])
                gm.graph.erase_node(upcast)
                rewrites += 1
    if rewrites:
        gm.graph.lint()
        gm.recompile()
    logger.info(
        "[fp32_expert_dweight] stored %d expert dWeights as FP32", rewrites
    )
    return gm


__all__ = ["apply_fp32_expert_dweight_pass"]
