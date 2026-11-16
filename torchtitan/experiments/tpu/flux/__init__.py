# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan.models.flux as native_flux_models
from torchtitan.experiments.tpu.flux.infra.parallelize import parallelize_flux

__all__ = [
    "flux_configs",
    "parallelize_flux",
]


def _upstream_flavor(flavor: str):
  """Builds an upstream flux flavor.

  Upstream d771abb0 changed `native_flux_models.flux_configs[flavor]` from a
  bare callable into a `(builder, max_context_length)` pair. Flux builders take
  no arguments, so only the unpack changes here.
  """
  build_config = native_flux_models.flux_configs[flavor]
  return build_config()


flux_configs = {
    "flux-dev": _upstream_flavor("flux-dev"),
    "flux-schnell": _upstream_flavor("flux-schnell"),
    "flux-debug": _upstream_flavor("flux-debug"),
}
