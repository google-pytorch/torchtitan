"""Unit tests for ``TPUSelectiveAC.Config.save_kda_residuals``.

Covers which ops the flag adds to the save set (the Tokamax KDA forward ops,
matched by qualified name, and nothing else), and, eagerly on CPU with a
stand-in op, that saving the forward op stops its re-run in backward without
changing a single bit of the output or the gradients.
"""

import collections
import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.tpu.components import activation_checkpoint as tpu_ac

# The ops `qwen3_5/tokamax_gated_delta.py` registers.
_KDA_FWD = (
    "tt_qwen35::gated_delta_fwd_res",
    "tt_qwen35::gated_delta_segments_fwd_res",
)
_KDA_BWD = (
    "tt_qwen35::gated_delta_bwd_res",
    "tt_qwen35::gated_delta_segments_bwd_res",
)
# Forward only, for calls without a gradient: there are no residuals to save.
_KDA_FWD_ONLY = (
    "tt_qwen35::gated_delta_fwd",
    "tt_qwen35::gated_delta_segments_fwd",
)
_UNKNOWN = (
    "tt_qwen35::gated_delta",
    "tt_qwen35::gated_delta_fwd_res_v2",
    "other::gated_delta_fwd_res",
    "aten::gated_delta_fwd_res",
)
_SPLASH_FWD = "pallas::splash_attn_fwd_0123456789abcdef"
_SPLASH_BWD = "pallas::splash_attn_bwd_0123456789abcdef"

# A stand-in for the KDA op pair, under a test namespace.
_STAND_IN_FWD = "tpu_ac_kda_test::gated_delta_fwd_res"
_CALLS = collections.Counter()


class _FakeOp:
  """Something with an ``OpOverload``-like ``name()``."""

  def __init__(self, name: str):
    self._name = name

  def name(self) -> str:
    return self._name


@dataclasses.dataclass(kw_only=True, slots=True)
class _SplashConfig(tpu_ac.TPUSelectiveAC.Config):
  """Adds ``save_pallas_attention``, which ``get_save_ops`` reads by name."""

  save_pallas_attention: bool = False


@torch.library.custom_op(_STAND_IN_FWD, mutates_args=())
def _stand_in_fwd_res(x: torch.Tensor, w: torch.Tensor) -> list[torch.Tensor]:
  """``[out, *residuals]``, as the KDA forward ops return: ``sin(x) * w``."""
  _CALLS["fwd"] += 1
  return [torch.sin(x) * w, torch.cos(x) * w, torch.sin(x)]


@_stand_in_fwd_res.register_fake
def _(x, w):
  return [torch.empty_like(x) for _ in range(3)]


@torch.library.custom_op(
    "tpu_ac_kda_test::gated_delta_bwd_res", mutates_args=()
)
def _stand_in_bwd_res(
    residuals: list[torch.Tensor], grad_out: torch.Tensor
) -> list[torch.Tensor]:
  """The backward from the residuals alone, as the KDA backward ops run."""
  _CALLS["bwd"] += 1
  dx_factor, sin_x = residuals
  return [grad_out * dx_factor, (grad_out * sin_x).sum(0)]


@_stand_in_bwd_res.register_fake
def _(residuals, grad_out):
  return [torch.empty_like(grad_out), torch.empty_like(grad_out[0])]


def _stand_in_setup_context(ctx, inputs, output):
  residuals = output[1:]
  ctx.mark_non_differentiable(*residuals)
  ctx.set_materialize_grads(False)
  ctx.save_for_backward(*residuals)


def _stand_in_backward(ctx, grad_outputs):
  grad_out = grad_outputs[0]
  if grad_out is None:
    return None, None
  dx, dw = _stand_in_bwd_res(list(ctx.saved_tensors), grad_out.contiguous())
  return dx, dw


_stand_in_fwd_res.register_autograd(
    _stand_in_backward, setup_context=_stand_in_setup_context
)


class _KdaLikeBlock(nn.Module):
  """A block whose only expensive op is the stand-in KDA forward."""

  def __init__(self, dim: int):
    super().__init__()
    self.scale = nn.Parameter(torch.randn(dim))
    self.weight = nn.Parameter(torch.randn(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = _stand_in_fwd_res(torch.tanh(x * self.scale), self.weight)[0]
    return out * 2.0


class _Model(nn.Module):
  """``policy.apply`` wraps every child of ``layers``."""

  def __init__(self, block: nn.Module):
    super().__init__()
    self.layers = nn.ModuleDict({"0": block})

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layers["0"](x)


def _save_ops(config=None, **flags) -> set:
  config = config or tpu_ac.TPUSelectiveAC.Config(**flags)
  return tpu_ac.TPUSelectiveAC(config).get_save_ops()


def _bits(t: torch.Tensor) -> torch.Tensor:
  return t.detach().contiguous().view(torch.uint8)


class SaveKdaResidualsTest(parameterized.TestCase):
  """``TPUSelectiveAC``'s ``save_kda_residuals``."""

  def test_flag_is_off_by_default(self):
    names = {f.name for f in dataclasses.fields(tpu_ac.TPUSelectiveAC.Config)}
    self.assertIn("save_kda_residuals", names)
    self.assertIs(tpu_ac.TPUSelectiveAC.Config().save_kda_residuals, False)
    converted = tpu_ac.tpu_selective_ac_config(
        SelectiveAC.Config(preserve_rng_state=False)
    )
    self.assertIs(type(converted), tpu_ac.TPUSelectiveAC.Config)
    self.assertIs(converted.save_kda_residuals, False)

  def test_flag_off_leaves_the_save_set_unchanged(self):
    default = _save_ops()
    off = _save_ops(save_kda_residuals=False)
    matcher = tpu_ac._SaveOpsIncludingKdaForward  # pylint: disable=protected-access
    self.assertIs(type(off), type(default))
    self.assertNotIsInstance(off, matcher)
    self.assertEqual(off, default)
    for name in _KDA_FWD + _KDA_BWD + _KDA_FWD_ONLY:
      self.assertNotIn(_FakeOp(name), off, msg=name)

  def test_flag_saves_exactly_the_kda_forward_ops(self):
    off = _save_ops()
    save_ops = _save_ops(save_kda_residuals=True)
    matcher = tpu_ac._SaveOpsIncludingKdaForward  # pylint: disable=protected-access
    self.assertIsInstance(save_ops, matcher)
    # The matcher removes nothing from the set and adds no op object to it.
    self.assertEqual(set(save_ops), off)
    for op in off:
      self.assertIn(op, save_ops)
    for name in _KDA_FWD:
      self.assertIn(_FakeOp(name), save_ops, msg=name)
    for name in _KDA_BWD + _KDA_FWD_ONLY + _UNKNOWN:
      self.assertNotIn(_FakeOp(name), save_ops, msg=name)
    # Without `save_pallas_attention`, the splash forward is not matched.
    self.assertNotIn(_FakeOp(_SPLASH_FWD), save_ops)

  @parameterized.parameters(False, True)
  def test_flag_keeps_the_splash_forward_rule(self, save_pallas_attention):
    save_ops = _save_ops(
        _SplashConfig(
            save_kda_residuals=True, save_pallas_attention=save_pallas_attention
        )
    )
    self.assertEqual(_FakeOp(_SPLASH_FWD) in save_ops, save_pallas_attention)
    self.assertNotIn(_FakeOp(_SPLASH_BWD), save_ops)
    for name in _KDA_FWD:
      self.assertIn(_FakeOp(name), save_ops, msg=name)
    for name in _KDA_BWD + _KDA_FWD_ONLY + _UNKNOWN:
      self.assertNotIn(_FakeOp(name), save_ops, msg=name)
    # The flag off: no by-name match at all from this policy.
    off = _save_ops(_SplashConfig(save_pallas_attention=save_pallas_attention))
    self.assertNotIn(_FakeOp(_KDA_FWD[0]), off)

  def test_flag_matches_op_overloads_by_qualified_name(self):
    stand_in_fwd = torch.ops.tpu_ac_kda_test.gated_delta_fwd_res.default
    stand_in_bwd = torch.ops.tpu_ac_kda_test.gated_delta_bwd_res.default
    with mock.patch.object(tpu_ac, "_KDA_FWD_OPS", frozenset({_STAND_IN_FWD})):
      save_ops = _save_ops(save_kda_residuals=True)
      self.assertIn(stand_in_fwd, save_ops)
      self.assertNotIn(stand_in_bwd, save_ops)
      self.assertNotIn(stand_in_fwd, _save_ops())

  def _run(self, x: torch.Tensor, **flags):
    """Forward + backward of a wrapped block; stand-in calls in backward."""
    torch.manual_seed(0)
    model = _Model(_KdaLikeBlock(x.shape[-1]))
    with mock.patch.object(tpu_ac, "_KDA_FWD_OPS", frozenset({_STAND_IN_FWD})):
      tpu_ac.TPUSelectiveAC(tpu_ac.TPUSelectiveAC.Config(**flags)).apply(model)
      leaf = x.clone().requires_grad_()
      out = model(leaf)
      _CALLS.clear()
      out.sum().backward()
    grads = [leaf.grad] + [p.grad for p in model.parameters()]
    return out, grads, _CALLS["fwd"], _CALLS["bwd"]

  def test_flag_skips_the_forward_rerun_bitwise(self):
    x = torch.randn(4, 8)
    out, grads, fwd_recomputed, bwd_off = self._run(x)
    out_s, grads_s, fwd_saved, bwd_on = self._run(x, save_kda_residuals=True)
    # Recomputed: backward replays the forward op once, then runs the backward
    # op. Saved: backward runs only the backward op.
    self.assertEqual((fwd_recomputed, bwd_off), (1, 1))
    self.assertEqual((fwd_saved, bwd_on), (0, 1))
    self.assertTrue(torch.equal(_bits(out), _bits(out_s)))
    self.assertLen(grads, 3)
    for g, g_s in zip(grads, grads_s, strict=True):
      self.assertIsNotNone(g)
      self.assertTrue(torch.equal(_bits(g), _bits(g_s)))


if __name__ == "__main__":
  absltest.main()
