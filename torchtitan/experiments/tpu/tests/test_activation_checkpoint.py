"""Unit tests for the flags of the TPU selective activation checkpointing.

Covers ``TPUSelectiveAC``'s save set for every flag combination, and, eagerly on
CPU, that a flag changes which forward ops backward re-runs but not a single
bit of the gradients.
"""

import collections
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.tpu.components import activation_checkpoint as tpu_ac

_FLAGS = ("recompute_bmm", "save_pallas_attention")
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


class _BmmLoopBlock(nn.Module):
  """A Python loop of small batched matmuls, as in a chunked recurrence."""

  def __init__(self, dim: int, num_steps: int):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(num_steps, dim, dim) / dim**0.5)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    state = x
    for weight in self.weight.unbind(0):
      state = torch.tanh(torch.bmm(state, weight.expand(x.shape[0], -1, -1)))
    return state


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


class _CountOps(TorchDispatchMode):
  """Counts calls of the given aten ops."""

  def __init__(self, ops):
    super().__init__()
    self.ops = ops
    self.counts = collections.Counter()

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    if func in self.ops:
      self.counts[func] += 1
    return func(*args, **(kwargs or {}))


def _save_ops(**flags) -> set:
  config = tpu_ac.TPUSelectiveAC.Config(**flags)
  return tpu_ac.TPUSelectiveAC(config).get_save_ops()


def _bits(t: torch.Tensor) -> torch.Tensor:
  return t.detach().contiguous().view(torch.uint8)


class TPUSelectiveACFlagsTest(parameterized.TestCase):
  """The flags of ``TPUSelectiveAC``."""

  def test_flags_are_off_by_default(self):
    names = {f.name for f in dataclasses.fields(tpu_ac.TPUSelectiveAC.Config)}
    core = {f.name for f in dataclasses.fields(SelectiveAC.Config)}
    self.assertEqual(names - core, set(_FLAGS))
    config = tpu_ac.TPUSelectiveAC.Config()
    for flag in _FLAGS:
      self.assertIs(getattr(config, flag), False, msg=flag)

  def test_flags_off_is_the_unchanged_policy(self):
    """Both flags off: core ``SelectiveAC``'s save set minus the collectives."""
    off = _save_ops()
    core = SelectiveAC(SelectiveAC.Config()).get_save_ops()
    self.assertIs(type(off), set)
    self.assertEqual(off, core - _COLLECTIVES)
    self.assertIn(torch.ops.aten.bmm.default, off)
    self.assertNotIn(_SPLASH_FWD, off)

  def test_recompute_bmm_drops_only_bmm(self):
    off = _save_ops()
    self.assertEqual(
        _save_ops(recompute_bmm=True), off - {torch.ops.aten.bmm.default}
    )
    self.assertIn(torch.ops.aten.mm.default, _save_ops(recompute_bmm=True))

  def test_save_pallas_attention_matches_the_splash_forward_by_name(self):
    off = _save_ops()
    save_ops = _save_ops(save_pallas_attention=True)
    matcher = tpu_ac._SaveOpsIncludingPallasAttention  # pylint: disable=protected-access
    self.assertIsInstance(save_ops, matcher)
    self.assertEqual(set(save_ops), off)
    self.assertIn(_SPLASH_FWD, save_ops)
    splash_like = torch.ops.tpu_ac_test.splash_attn_fwd_0123456789abcdef
    self.assertIn(splash_like.default, save_ops)
    self.assertNotIn(_SPLASH_BWD, save_ops)

  @parameterized.product(
      recompute_bmm=(False, True), save_pallas_attention=(False, True)
  )
  def test_flags_compose(self, recompute_bmm, save_pallas_attention):
    off = _save_ops()
    save_ops = _save_ops(
        recompute_bmm=recompute_bmm, save_pallas_attention=save_pallas_attention
    )
    want = off - ({torch.ops.aten.bmm.default} if recompute_bmm else set())
    self.assertEqual(set(save_ops), want)
    self.assertEqual(_SPLASH_FWD in save_ops, save_pallas_attention)

  def test_build_keeps_every_field(self):
    config = tpu_ac.TPUSelectiveAC.Config(
        recompute_bmm=True,
        save_pallas_attention=True,
        preserve_rng_state=False,
        debug=True,
    )
    policy = config.build(dump_folder="")
    self.assertIs(type(policy), tpu_ac.TPUSelectiveAC)
    self.assertTrue(policy.config.recompute_bmm)
    self.assertTrue(policy.config.save_pallas_attention)
    self.assertFalse(policy.config.preserve_rng_state)
    self.assertTrue(policy.config.debug)

  def test_tpu_selective_ac_config_leaves_the_flags_off(self):
    converted = tpu_ac.tpu_selective_ac_config(
        SelectiveAC.Config(preserve_rng_state=False)
    )
    self.assertIs(type(converted), tpu_ac.TPUSelectiveAC.Config)
    self.assertFalse(converted.preserve_rng_state)
    for flag in _FLAGS:
      self.assertIs(getattr(converted, flag), False, msg=flag)

  def _run(self, block_fn, x, **flags):
    """Forward + backward of a wrapped block; ops counted during backward."""
    torch.manual_seed(0)
    model = _Model(block_fn())
    tpu_ac.TPUSelectiveAC(tpu_ac.TPUSelectiveAC.Config(**flags)).apply(model)
    leaf = x.clone().requires_grad_()
    out = model(leaf)
    _CALLS.clear()
    with _CountOps({torch.ops.aten.bmm.default}) as counter:
      out.sum().backward()
    grads = [leaf.grad] + [p.grad for p in model.parameters()]
    bmm_calls = counter.counts[torch.ops.aten.bmm.default]
    return out, grads, bmm_calls, _CALLS["splash_fwd"]

  def test_recompute_bmm_reruns_the_loop_bmm_in_backward_bitwise(self):
    num_steps, dim = 4, 8
    x = torch.randn(3, 5, dim)

    def block():
      return _BmmLoopBlock(dim, num_steps)

    out, grads, bmm_saved, _ = self._run(block, x)
    out_r, grads_r, bmm_recomputed, _ = self._run(block, x, recompute_bmm=True)
    # Saved: backward runs only the gradient bmm. Recomputed: plus the
    # forward's num_steps bmm, once each.
    self.assertEqual(bmm_recomputed - bmm_saved, num_steps)
    self.assertTrue(torch.equal(_bits(out), _bits(out_r)))
    for g, g_r in zip(grads, grads_r, strict=True):
      self.assertTrue(torch.equal(_bits(g), _bits(g_r)))

  def test_save_pallas_attention_skips_the_forward_rerun_bitwise(self):
    dim = 8
    x = torch.randn(4, dim)

    def block():
      return _SplashLikeBlock(dim)

    out, grads, _, fwd_recomputed = self._run(block, x)
    out_s, grads_s, _, fwd_saved = self._run(
        block, x, save_pallas_attention=True
    )
    self.assertEqual(fwd_recomputed, 1)
    self.assertEqual(fwd_saved, 0)
    self.assertTrue(torch.equal(_bits(out), _bits(out_s)))
    for g, g_s in zip(grads, grads_s, strict=True):
      self.assertTrue(torch.equal(_bits(g), _bits(g_s)))


if __name__ == "__main__":
  absltest.main()
