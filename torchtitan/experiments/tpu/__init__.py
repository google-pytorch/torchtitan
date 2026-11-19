"""TPU lane for torchtitan experiments.

This package contains TPU-specific training infrastructure, model variants
(conformer, flux, qwen3, ...), and the gmain entry point.
"""

# pylint: disable=g-statement-before-imports,g-import-not-at-top

# ---------------------------------------------------------------------------
# typeguard 2.x <-> tyro 1.x compatibility shim
# ---------------------------------------------------------------------------
# tokamax pins typeguard==2.13.3 (jaxtyping uses it for shape-annotation
# parsing; tokamax's setup.py is explicit).  tyro 1.x references
# ``typeguard.TypeCheckError`` at import time, but typeguard 2.x does not
# define that symbol (it was added in typeguard 3.x+).  Without this shim,
# ``import tyro`` raises:
#
#     AttributeError: module 'typeguard' has no attribute 'TypeCheckError'.
#         Did you mean: 'TypeChecker'?
#
# Alias to ``TypeError`` so tyro's ``except typeguard.TypeCheckError`` clauses
# still catch the underlying ``TypeError`` raised by typeguard 2.x -- which is
# what tyro's TypeCheckError inherits from anyway, so behavior matches.
#
# Must live in this ``__init__.py`` (the TPU experiment package's entry point)
# rather than in ``gmain.py``: ``torchtitan.config`` -- which transitively
# imports tyro -- is imported BEFORE gmain.py's body runs, so by the time
# gmain.py:1 executes, ``from typeguard import TypeCheckError`` has already
# fired and the run has crashed.  Importing ``torchtitan.experiments.tpu``
# (via any TPU train_minimal or test) triggers this file first.
#
# Drop this block when either of these lands upstream:
#   - tyro stops dereferencing ``typeguard.TypeCheckError`` at import time
#     (use ``getattr`` with a TypeError fallback in tyro's compat layer), or
#   - tokamax supports typeguard >= 3.x (jaxtyping needs to follow).
try:
  import typeguard as _typeguard

  if not hasattr(_typeguard, "TypeCheckError"):
    _typeguard.TypeCheckError = TypeError

  del _typeguard
except ImportError:
  pass

from torchtitan.experiments.tpu import utils as _tpu_utils
from torchtitan.experiments.tpu.components.scatter_add import (
    install_tpu_scatter_add_row_gather as _install_tpu_scatter_add_row_gather,
)
from torchtitan.tools import utils as _utils

# Gate automatic registration on TPU availability to avoid global autograd side
# effects on CPU/CUDA imports, while installing at ``torchtitan.experiments.tpu``
# package import time so callers that construct TPU models without going through
# ``TPUTrainer`` (e.g. ``torchtitan.experiments.tpu.qwen3_5.model_registry(...)``
# or external RL harnesses) still receive the row-gather backward.
if _utils.device_type == "tpu":
  _install_tpu_scatter_add_row_gather()
