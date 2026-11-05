# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Give rematerialized expert token gathers their own chunk count.

``tt_experts::gather_paired_grouped_mm`` pipelines its token gather in
``chunks`` pieces. Its forward ignores the trailing ``recompute_chunks``
argument; this pass makes the backward's recomputed copies use it instead.
The chunk count changes only scheduling, never values.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_OP_NAME = "tt_experts::gather_paired_grouped_mm"
_CHUNKS_ARG = 9
_RECOMPUTE_CHUNKS_ARG = 10


def apply_recompute_gather_chunks(
    gm: torch.fx.GraphModule,
    example_inputs=None,
) -> torch.fx.GraphModule:
    del example_inputs
    rewritten = 0
    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or not isinstance(node.target, torch._ops.OpOverload)
            or node.target.name() != _OP_NAME
            # selective_activation_remat_pass names its copies this way.
            or not node.name.endswith("_recomputed")
        ):
            continue
        args = list(node.args)
        if args[_CHUNKS_ARG] != args[_RECOMPUTE_CHUNKS_ARG]:
            args[_CHUNKS_ARG] = args[_RECOMPUTE_CHUNKS_ARG]
            node.args = tuple(args)
            rewritten += 1
    if rewritten:
        gm.recompile()
        logger.info(
            "Set the chunk count of %d recomputed expert token gathers",
            rewritten,
        )
    return gm
