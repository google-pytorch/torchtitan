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

This module imports only torch, so model code can use it without importing
the graph trainer.
"""

from contextlib import AbstractContextManager

from torch.fx.traceback import annotate

# The FX node annotation key whose value names a region.
REMAT_REGION = "remat_region"

# The Splash output and LSE that the attention backward reads.
ATTENTION_INNER_ATTENTION = "attention.inner_attention"
# Each MoE layer's routing: the router's top-k expert ids (upstream's
# ``routing_decision`` region in ``TokenChoiceTopKRouter``) and the
# expert-sort permutation that ``token_dispatcher.local_reorder`` computes from
# them.
MOE_ROUTING = "moe.routing"
# Ring-of-Experts routing tables (expert sort, window, gather-reduce block
# tables) computed from the all-gathered top-k ids; small integer tensors.
MOE_ROE_ROUTING = "moe.roe_routing"
# Expert-parallel activation all-gather outputs in Ring-of-Experts dispatch.
MOE_EP_DISPATCH = "moe.ep_dispatch"
# Each layer's FFN down-projection output (the w2 matmul), which the post-FFN
# norm backward reads; saving it avoids recomputing that matmul.
FEED_FORWARD_W2 = "feed_forward.w2"
# Every forward value of a transformer layer: a layer saved this way is not
# recomputed in backward. Meant for ``compile.save_last_n_layers`` when HBM
# allows keeping some layers whole.
LAYER_ALL = "layer.all"
# Every forward value of a transformer layer before its FFN/MoE sub-block (the
# attention or GatedDeltaNet mixer plus the post-mixer residual). Avoids saving
# the 16x gathered MoE token buffers while skipping mixer recompute in backward.
LAYER_MIXER = "layer.mixer"


def annotate_region(name: str) -> AbstractContextManager[None]:
  """Mark the FX nodes traced inside this ``with`` block as region ``name``.

  This only adds node metadata (``node.meta["custom"][REMAT_REGION]``); it does
  not change what runs.
  """
  return annotate({REMAT_REGION: name})
