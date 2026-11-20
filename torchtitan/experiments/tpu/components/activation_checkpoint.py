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

  Two flags tune the save set further; both off (the default) is the policy
  above, unchanged, and they compose:

  - ``recompute_bmm``: recompute ``aten.bmm`` outputs instead of saving them.
    A Python loop of small batched matmuls (the XLA GatedDeltaNet chunk
    recurrence: three per 64-token chunk) otherwise hands one saved tensor per
    loop step from the compiled forward to the compiled backward.
  - ``save_pallas_attention``: also save the Pallas splash-attention forward
    (output + logsumexp, matched by name as in ``AttentionSavingSelectiveAC``),
    so backward does not re-run the forward kernel.

  Saving or recomputing an op runs the same op on the same inputs, so the
  values do not change.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(SelectiveAC.Config):
    recompute_bmm: bool = False
    """Recompute ``aten.bmm`` outputs in backward instead of saving them."""

    save_pallas_attention: bool = False
    """Also save the Pallas splash-attention forward (output + logsumexp)."""

  def get_save_ops(self) -> set:
    config = self.config
    save_ops = super().get_save_ops() - _SAVED_COLLECTIVES
    if getattr(config, "recompute_bmm", False):
      save_ops.discard(torch.ops.aten.bmm.default)
    if getattr(config, "save_pallas_attention", False):
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
  """

  def __contains__(self, op) -> bool:
    if super().__contains__(op):
      return True
    return _PALLAS_ATTENTION_FWD_MARKER in _op_name(op)


class AttentionSavingSelectiveAC(SelectiveAC):
  """``SelectiveAC`` that also saves the attention output on TPU.

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
  class Config(SelectiveAC.Config):
    pass

  def get_save_ops(self) -> set:
    return _SaveOpsIncludingPallasAttention(super().get_save_ops())
