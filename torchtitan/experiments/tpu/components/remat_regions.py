# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Names of the regions that TPU selective memory policies can save.

The names use upstream ``RegionAC``'s naming (``<block-relative module
FQN>.<local name>``), so eager and graph AC can later share one vocabulary.
``compile.save_regions`` and ``compile.save_last_n_layers`` take
these names.

Model code marks values that op names alone cannot tell apart by computing
them inside ``with annotate_region(<region name>):``.

This module imports only torch at load time, so model code can use it without
importing the graph trainer.
"""

from contextlib import AbstractContextManager

from torch.fx.traceback import annotate, annotate_fn

# The FX node annotation key whose value names a region.
REMAT_REGION = "remat_region"

# The Splash output and LSE that the attention backward reads.
ATTENTION_INNER_ATTENTION = "attention.inner_attention"
# Each MoE layer's routing: the router's top-k expert ids (upstream's
# ``routing_decision`` region in ``TokenChoiceTopKRouter``) and the
# expert-sort permutation that ``token_dispatcher.local_reorder`` computes from
# them.
MOE_ROUTING = "moe.routing"
# Each layer's FFN down-projection output (the w2 matmul), which the post-FFN
# norm backward reads; saving it avoids recomputing that matmul.
FEED_FORWARD_W2 = "feed_forward.w2"
# Each layer's FSDP all-gather inside a scan body. The scan body always
# recomputes it, so the gathered weight is never kept for backward.
FSDP_GATHER = "fsdp.gather"
# The MoE aux loss's all-reduced expert counts. The scan body always saves
# them: redoing the all-reduce in backward is an exposed round trip per layer.
MOE_EXPERT_COUNTS = "moe.expert_counts"


def annotate_region(name: str) -> AbstractContextManager[None]:
  """Mark the FX nodes traced inside this ``with`` block as region ``name``.

  This only adds node metadata (``node.meta["custom"][REMAT_REGION]``); it does
  not change what runs.
  """
  return annotate({REMAT_REGION: name})


_MOE_EXPERT_COUNTS_ANNOTATED = False


def annotate_moe_expert_counts() -> None:
  """Mark the MoE aux loss's count all-reduce as region ``MOE_EXPERT_COUNTS``.

  Wraps ``moe._all_reduce_expert_counts`` once, at setup, so the shared model
  file stays untouched. The module global is replaced, not a reference to it:
  ``moe.py``'s loss functions look the name up in their own module each call.
  """
  global _MOE_EXPERT_COUNTS_ANNOTATED
  if _MOE_EXPERT_COUNTS_ANNOTATED:
    return
  from torchtitan.models.common import moe  # pylint: disable=g-import-not-at-top

  moe._all_reduce_expert_counts = annotate_fn(  # pylint: disable=protected-access
      {REMAT_REGION: MOE_EXPERT_COUNTS}
  )(moe._all_reduce_expert_counts)  # pylint: disable=protected-access
  _MOE_EXPERT_COUNTS_ANNOTATED = True
