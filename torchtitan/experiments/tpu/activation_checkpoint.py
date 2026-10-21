"""Activation-checkpoint policies for the TPU lane."""

from dataclasses import dataclass

import torch
from torchtitan.distributed.activation_checkpoint import SelectiveAC

_PALLAS_ATTENTION_FWD_MARKER = "splash_attn_fwd"
_WAIT_TENSOR = torch.ops._c10d_functional.wait_tensor.default


def _op_name(op) -> str:
  """Qualified name of an OpOverload, however this torch build spells it."""
  name = getattr(op, "name", None)
  if callable(name):
    try:
      return str(name())
    except Exception:  # pylint: disable=broad-except
      pass
  return str(op)


def _is_functional_collective(op) -> bool:
  return (
      getattr(op, "namespace", None) == "_c10d_functional"
      and op is not _WAIT_TENSOR
  )


class _SaveOpsKeepingCollectiveWaits(set):
  """The default save set, plus the ``wait_tensor`` of each saved collective.

  Works around a torch-tpu 0.2.0 compiler gap. Core ``SelectiveAC`` saves the
  outputs of ``all_to_all_single`` / ``reduce_scatter_tensor`` but recomputes
  their ``wait_tensor``, so backward re-runs ``wait_tensor`` on the saved
  tensor. On GPU that is a no-op that returns the same, already-filled tensor:
  ``wait_tensor`` only waits on work still pending at runtime, and the
  forward's wait already finished it. torch-tpu's split compiler instead pairs
  each ``wait_tensor`` with its collective at compile time and rejects one
  whose input is a graph input ("Expected call_function for wait_tensor
  producer, got placeholder"), e.g. in compiled Ulysses CP under selective AC.

  Also saving the ``wait_tensor`` right after a saved collective (funcol waits
  right after issuing while tracing) leaves backward without such a wait. No
  extra memory: ``wait_tensor`` returns its input, the buffer saved anyway.
  Waits of recomputed collectives (e.g. FSDP all-gathers) stay recomputed.
  TODO: drop once torch-tpu treats such a wait as a no-op.

  Relies on ``SelectiveAC``'s policy asking ``func in save_ops`` once per op,
  in program order.
  """

  def __init__(self, ops=()):
    super().__init__(ops)
    self._last_was_saved_collective = False

  def __contains__(self, op) -> bool:
    follows_saved_collective = self._last_was_saved_collective
    saved = super().__contains__(op) or (
        op is _WAIT_TENSOR and follows_saved_collective
    )
    self._last_was_saved_collective = saved and _is_functional_collective(op)
    return saved


class _SaveOpsIncludingPallasAttention(_SaveOpsKeepingCollectiveWaits):
  """The default save set, plus any Pallas splash-attention forward op.

  ``SelectiveAC``'s policy asks ``func in save_ops``. The splash forward is a
  ``torch.library.custom_op`` whose name carries a hash of the kernel's
  configuration (``pallas::splash_attn_fwd_<sha16>``), so it cannot be named in
  a static set -- and it is registered lazily, on the first call, which is
  after the AC wrapper is built. Match it by name instead.
  """

  def __contains__(self, op) -> bool:
    if super().__contains__(op):
      return True
    return _PALLAS_ATTENTION_FWD_MARKER in _op_name(op)


class TPUSelectiveAC(SelectiveAC):
  """``SelectiveAC`` for torch-tpu: saved collectives keep their waits.

  See ``_SaveOpsKeepingCollectiveWaits``. Otherwise identical to core
  ``SelectiveAC``.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(SelectiveAC.Config):
    pass

  def get_save_ops(self) -> set:
    return _SaveOpsKeepingCollectiveWaits(super().get_save_ops())


class AttentionSavingSelectiveAC(TPUSelectiveAC):
  """``TPUSelectiveAC`` that also saves the attention output on TPU.

  Upstream's default save set already holds every attention implementation it
  knows about -- the three ``aten::_scaled_dot_product_*`` variants and the
  flex-attention HOP -- so on GPU the attention output is kept and only cheap
  ops are recomputed. The TPU lane's attention is a Pallas custom op, which is
  in none of those lists, so selective AC recomputes it: measured on
  Llama-3.2-1B / 4x v6e, the splash forward kernel runs 32 times per step (16
  layers, twice) and the duplicate costs 81 ms of an 840 ms step.

  Saving it costs one attention output plus its logsumexp per layer
  (``bf16[1, H, T, D]`` + ``f32[1, H, T]``: ~69 MB per layer at T=16384, H=32,
  D=64), which is the same trade upstream already makes for SDPA.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TPUSelectiveAC.Config):
    pass

  def get_save_ops(self) -> set:
    return _SaveOpsIncludingPallasAttention(super().get_save_ops())
