# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-TransformerBlock ``torch.compile`` for the TPU lane.

Upstream #4895 removed ``torchtitan/distributed/compile.py`` together with
block-level model compilation (``CompileConfig.components``); core models now
only compile named local regions (``LocalCompileConfig.regions``). The TPU
parallelize paths still compile each transformer block with the ``tpu``
backend under the fork's ``compile.enable`` / ``compile.components`` policy,
so the helpers they used live here.
"""

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def model_compile_enabled(compile_config: Any) -> bool:
  """Whether the fork compile policy asks for block-level model compilation."""
  return bool(
      compile_config is not None
      and getattr(compile_config, "enable", False)
      and "model" in getattr(compile_config, "components", ())
  )


def apply_compile(
    model: nn.Module,
    compile_config: Any = None,
    *,
    parallelism_context: Any = None,
    **kwargs: Any,
) -> None:
  """Applies ``torch.compile`` to each ``model.layers`` child.

  Compiling the repeated block instead of the whole model keeps compilation
  cheap. No-op unless ``compile_config`` enables ``"model"`` compilation.
  """
  del parallelism_context, kwargs
  if not model_compile_enabled(compile_config):
    return

  # Needed for torch.compile to handle data-dependent dynamic shapes in
  # token-choice MoE dispatch. Harmless for dense models.
  torch._dynamo.config.capture_scalar_outputs = True
  # Skip replaying forward side effects (e.g. RoPE cache updates) during the
  # AC recompute in backward; torch.compile cannot reapply python mutations in
  # the backward, so accept this eager/compile divergence.
  torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = (
      True  # pyrefly: ignore [bad-assignment]
  )

  backend = maybe_regional_inductor_backend(model, compile_config.backend)
  # pyrefly: ignore [missing-attribute]
  for _, transformer_block in model.layers.named_children():
    transformer_block.compile(backend=backend, fullgraph=True)
  logger.info("Compiling each TransformerBlock with torch.compile")


def maybe_regional_inductor_backend(
    model: nn.Module, backend: str
) -> str | Callable:
  """Wraps ``aot_eager`` so inductor-only flex regions are scooped out.

  ``regional_inductor`` lowers just the regions annotated with
  ``compile_with_inductor`` to inductor while the rest stays in aot_eager.
  Only applied for ``aot_eager`` on models that use FlexInnerAttention; other
  non-inductor backends raise rather than silently decomposing flex to eager
  aten ops.
  """
  from torchtitan.models.common.attention import FlexInnerAttention  # pylint: disable=g-import-not-at-top

  uses_flex = any(isinstance(m, FlexInnerAttention) for m in model.modules())
  if not uses_flex or backend == "inductor":
    return backend
  if backend != "aot_eager":
    raise ValueError(
        f"Model uses FlexInnerAttention but compile backend {backend!r} is "
        "neither 'inductor' nor 'aot_eager'; the flex region would decompose "
        "to eager aten ops (no Triton kernel). Use 'inductor' or 'aot_eager'."
    )

  from torch._dynamo.backends.common import aot_autograd  # pylint: disable=g-import-not-at-top
  from torch.fx.passes.regional_inductor import regional_inductor  # pylint: disable=g-import-not-at-top

  logger.info("regional_inductor is enabled")
  return aot_autograd(fw_compiler=regional_inductor, bw_compiler=regional_inductor)


__all__ = [
    "apply_compile",
    "maybe_regional_inductor_backend",
    "model_compile_enabled",
]
