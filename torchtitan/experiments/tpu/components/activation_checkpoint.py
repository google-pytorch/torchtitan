"""Activation-checkpoint policies for the TPU lane."""

from dataclasses import dataclass, fields
from typing import Any

import torch
from torchtitan.distributed.activation_checkpoint import SelectiveAC


_PALLAS_ATTENTION_FWD_MARKER = "splash_attn_fwd"

# The collectives in core SelectiveAC's default save set.
_SAVED_COLLECTIVES = frozenset({
    torch.ops._c10d_functional.all_to_all_single.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
})


def _op_name(op) -> str:
  """Qualified name of an OpOverload, however this torch build spells it."""
  name = getattr(op, "name", None)
  if callable(name):
    try:
      return str(name())
    except Exception:  # pylint: disable=broad-except
      pass
  return str(op)


class TPUSelectiveAC(SelectiveAC):
  """``SelectiveAC`` that recomputes collectives instead of saving them.

  Core ``SelectiveAC`` saves the outputs of ``all_to_all_single`` and
  ``reduce_scatter_tensor`` (e.g. Ulysses CP's all-to-alls) and recomputes
  everything else, including the ``wait_tensor`` after each collective. On
  torch-tpu that fails both ways:

  - Compiled, backward runs that ``wait_tensor`` on the saved tensor, a graph
    input. torch-tpu's split compiler pairs every ``wait_tensor`` with its
    collective and rejects it ("Expected call_function for wait_tensor
    producer, got placeholder").
  - Eager, the saved output is an un-waited ``AsyncCollectiveTensor`` that is
    kept across forward and backward.

  Recomputing each collective together with its wait avoids both, at the cost
  of running those collectives again in backward (and saving fewer tensors).
  The gemma4 recipes use this policy; TPU context parallelism switches plain
  ``SelectiveAC`` to it (``tpu_selective_ac_config``).

  TODO: Save the collectives again once torch-tpu accepts a wait on a saved
  tensor.

  Like upstream's default save set, which holds every attention op it knows
  about (SDPA, flex attention, varlen attention), the save set also holds the
  Pallas splash-attention forward (output + logsumexp, matched by name), so
  backward runs only the splash backward kernel instead of re-running the
  forward kernel first. Measured on Llama-3.2-1B / 4x v6e, the recomputed
  splash forward ran 32 times per step (16 layers, twice) and the duplicate
  cost 81 ms of an 840 ms step. Saving it costs one attention output plus its
  logsumexp per layer (``bf16[1, H, T, D]`` + ``f32[1, H, T]``: ~69 MB per
  layer at T=16384, H=32, D=64), the same trade upstream makes for SDPA.
  ``save_pallas_attention=False`` recomputes it instead. A core
  ``SelectiveAC.Config`` passed to this policy directly saves it too.

  Saving or recomputing an op runs the same op on the same inputs. On CPU the
  values are bit-identical; on TPU a recomputed op is compiled into the
  backward program, where XLA may fuse it differently, so results can differ
  by rounding.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(SelectiveAC.Config):
    save_pallas_attention: bool = True
    """Save the Pallas splash-attention forward (output + logsumexp).

    On by default, as upstream saves SDPA, flex and varlen attention. False
    recomputes the splash forward in backward.
    """

  def get_save_ops(self) -> set:
    config = self.config
    save_ops = super().get_save_ops() - _SAVED_COLLECTIVES
    # A core SelectiveAC.Config has no field: it gets the default.
    if getattr(config, "save_pallas_attention", True):
      save_ops = _SaveOpsIncludingPallasAttention(save_ops)
    return save_ops


def tpu_selective_ac_config(ac_config: Any) -> Any:
  """Returns a plain ``SelectiveAC.Config`` as a ``TPUSelectiveAC.Config``.

  Other configs (full AC, ``TPUSelectiveAC`` or another subclass, None) are
  returned unchanged. Parallelize functions call this rather than recipes
  setting the config, so CLI parsing, which only knows the core AC configs,
  cannot undo it.
  """
  if type(ac_config) is not SelectiveAC.Config:
    return ac_config
  return TPUSelectiveAC.Config(**{
      f.name: getattr(ac_config, f.name) for f in fields(ac_config) if f.init
  })


class _SaveOpsIncludingPallasAttention(set):
  """The default save set, plus any Pallas splash-attention forward op.

  ``SelectiveAC``'s policy asks ``func in save_ops``. The splash forward is a
  ``torch.library.custom_op`` whose name carries a hash of the kernel's
  configuration (``pallas::splash_attn_fwd_<sha16>``), so it cannot be named in
  a static set -- and it is registered lazily, on the first call, which is
  after the AC wrapper is built. Match it by name instead.

  Unions, differences and copies keep the name match, so a subclass can return
  ``super().get_save_ops() | {op}`` or ``- {op}`` without losing it.
  """

  def __contains__(self, op) -> bool:
    if super().__contains__(op):
      return True
    return _PALLAS_ATTENTION_FWD_MARKER in _op_name(op)

  def _keep_match(self, result):
    return result if result is NotImplemented else type(self)(result)

  def __or__(self, other):
    return self._keep_match(super().__or__(other))

  __ror__ = __or__

  def __sub__(self, other):
    return self._keep_match(super().__sub__(other))

  def union(self, *others):
    return type(self)(super().union(*others))

  def difference(self, *others):
    return type(self)(super().difference(*others))

  def copy(self):
    return type(self)(self)
