"""Unit tests for the flag of the TPU selective activation checkpointing.

Covers ``TPUSelectiveAC``'s save set with ``save_pallas_attention`` on (the
default) and off, and, eagerly on CPU, that the flag changes which forward ops
backward re-runs but not a single bit of the gradients.
"""

import collections
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.tpu.components import activation_checkpoint as tpu_ac

_DEFAULTS = {"save_pallas_attention": True}
_FLAGS = tuple(_DEFAULTS)
_COLLECTIVES = {
    torch.ops._c10d_functional.all_to_all_single.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}
_CALLS = collections.Counter()


class _FakeOp:
  """Something with an ``OpOverload``-like ``name()``."""

  def __init__(self, name: str):
    self._name = name

  def name(self) -> str:
    return self._name


_SPLASH_FWD = _FakeOp("pallas::splash_attn_fwd_0123456789abcdef")
_SPLASH_BWD = _FakeOp("pallas::splash_attn_bwd_0123456789abcdef")


@torch.library.custom_op(
    "tpu_ac_test::splash_attn_fwd_0123456789abcdef", mutates_args=()
)
def _splash_like_fwd(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
  """Stands in for the splash forward: the policy matches it by name."""
  _CALLS["splash_fwd"] += 1
  return torch.sin(q * k)


@_splash_like_fwd.register_fake
def _(q, k):
  return torch.empty_like(q)


def _splash_like_setup_context(ctx, inputs, output):
  ctx.save_for_backward(*inputs)


def _splash_like_backward(ctx, grad):
  q, k = ctx.saved_tensors
  dout = grad * torch.cos(q * k)
  return dout * k, dout * q


_splash_like_fwd.register_autograd(
    _splash_like_backward, setup_context=_splash_like_setup_context
)


class _SplashLikeBlock(nn.Module):
  """A block whose only expensive op is the splash-like forward."""

  def __init__(self, dim: int):
    super().__init__()
    self.scale = nn.Parameter(torch.randn(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return _splash_like_fwd(torch.tanh(x * self.scale), x) * 2.0


class _Model(nn.Module):
  """``policy.apply`` wraps every child of ``layers``."""

  def __init__(self, block: nn.Module):
    super().__init__()
    self.layers = nn.ModuleDict({"0": block})

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layers["0"](x)


def _save_ops(**flags) -> set:
  config = tpu_ac.TPUSelectiveAC.Config(**flags)
  return tpu_ac.TPUSelectiveAC(config).get_save_ops()


def _bits(t: torch.Tensor) -> torch.Tensor:
  return t.detach().contiguous().view(torch.uint8)


class TPUSelectiveACFlagTest(parameterized.TestCase):
  """The flag of ``TPUSelectiveAC``."""

  def test_flag_defaults(self):
    names = {f.name for f in dataclasses.fields(tpu_ac.TPUSelectiveAC.Config)}
    core = {f.name for f in dataclasses.fields(SelectiveAC.Config)}
    self.assertEqual(names - core, set(_FLAGS))
    config = tpu_ac.TPUSelectiveAC.Config()
    for flag, default in _DEFAULTS.items():
      self.assertIs(getattr(config, flag), default, msg=flag)

  def test_flag_off_is_core_minus_the_collectives(self):
    """The flag off: core ``SelectiveAC``'s save set minus the collectives."""
    off = _save_ops(save_pallas_attention=False)
    core = SelectiveAC(SelectiveAC.Config()).get_save_ops()
    self.assertIs(type(off), set)
    self.assertEqual(off, core - _COLLECTIVES)
    self.assertIn(torch.ops.aten.bmm.default, off)
    self.assertNotIn(_SPLASH_FWD, off)

  def test_default_saves_the_splash_forward(self):
    """Default: the same ops, plus the splash forward, as upstream for SDPA."""
    default = _save_ops()
    self.assertEqual(set(default), _save_ops(save_pallas_attention=False))
    self.assertIn(_SPLASH_FWD, default)
    self.assertNotIn(_SPLASH_BWD, default)

  def test_core_config_given_to_the_policy_saves_the_splash_forward(self):
    """A core config built into ``TPUSelectiveAC`` directly gets the default."""
    save_ops = tpu_ac.TPUSelectiveAC(SelectiveAC.Config()).get_save_ops()
    self.assertEqual(set(save_ops), _save_ops(save_pallas_attention=False))
    self.assertIn(_SPLASH_FWD, save_ops)

  def test_save_pallas_attention_matches_the_splash_forward_by_name(self):
    off = _save_ops(save_pallas_attention=False)
    save_ops = _save_ops(save_pallas_attention=True)
    matcher = tpu_ac._SaveOpsIncludingPallasAttention  # pylint: disable=protected-access
    self.assertIsInstance(save_ops, matcher)
    self.assertEqual(set(save_ops), off)
    self.assertIn(_SPLASH_FWD, save_ops)
    splash_like = torch.ops.tpu_ac_test.splash_attn_fwd_0123456789abcdef
    self.assertIn(splash_like.default, save_ops)
    self.assertNotIn(_SPLASH_BWD, save_ops)

  def test_set_operations_keep_the_splash_match(self):
    """A subclass may return ``super().get_save_ops() | {op}`` or ``- {op}``."""
    save_ops = _save_ops()
    extra = {torch.ops.aten.cumsum.default}
    bmm = {torch.ops.aten.bmm.default}
    derived = {
        "a | b": save_ops | extra,
        "b | a": extra | save_ops,
        "a - b": save_ops - bmm,
        "union": save_ops.union(extra),
        "difference": save_ops.difference(bmm),
        "copy": save_ops.copy(),
    }
    for name, ops in derived.items():
      self.assertIn(_SPLASH_FWD, ops, msg=name)
      self.assertNotIn(_SPLASH_BWD, ops, msg=name)
    self.assertIn(torch.ops.aten.cumsum.default, derived["a | b"])
    self.assertIn(torch.ops.aten.cumsum.default, derived["b | a"])
    self.assertNotIn(torch.ops.aten.bmm.default, derived["a - b"])
    self.assertEqual(set(derived["copy"]), set(save_ops))

  def test_build_keeps_every_field(self):
    config = tpu_ac.TPUSelectiveAC.Config(
        save_pallas_attention=False,
        preserve_rng_state=False,
        debug=True,
    )
    policy = config.build(dump_folder="")
    self.assertIs(type(policy), tpu_ac.TPUSelectiveAC)
    self.assertFalse(policy.config.save_pallas_attention)
    self.assertFalse(policy.config.preserve_rng_state)
    self.assertTrue(policy.config.debug)

  def test_tpu_selective_ac_config_uses_the_flag_defaults(self):
    converted = tpu_ac.tpu_selective_ac_config(
        SelectiveAC.Config(preserve_rng_state=False)
    )
    self.assertIs(type(converted), tpu_ac.TPUSelectiveAC.Config)
    self.assertFalse(converted.preserve_rng_state)
    for flag, default in _DEFAULTS.items():
      self.assertIs(getattr(converted, flag), default, msg=flag)

  def _run(self, block_fn, x, **flags):
    """Forward + backward of a wrapped block; splash calls during backward."""
    torch.manual_seed(0)
    model = _Model(block_fn())
    tpu_ac.TPUSelectiveAC(tpu_ac.TPUSelectiveAC.Config(**flags)).apply(model)
    leaf = x.clone().requires_grad_()
    out = model(leaf)
    _CALLS.clear()
    out.sum().backward()
    grads = [leaf.grad] + [p.grad for p in model.parameters()]
    return out, grads, _CALLS["splash_fwd"]

  def test_save_pallas_attention_skips_the_forward_rerun_bitwise(self):
    dim = 8
    x = torch.randn(4, dim)

    def block():
      return _SplashLikeBlock(dim)

    out, grads, fwd_recomputed = self._run(
        block, x, save_pallas_attention=False
    )
    out_s, grads_s, fwd_saved = self._run(block, x)  # The default.
    self.assertEqual(fwd_recomputed, 1)
    self.assertEqual(fwd_saved, 0)
    self.assertTrue(torch.equal(_bits(out), _bits(out_s)))
    for g, g_s in zip(grads, grads_s, strict=True):
      self.assertTrue(torch.equal(_bits(g), _bits(g_s)))


if __name__ == "__main__":
  absltest.main()
