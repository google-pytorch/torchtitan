"""Tests for the PyTorch / TorchTPU `qk_norm_rope` bridge."""

import inspect
import sys
import types
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import qk_norm_rope

EPS = 1e-5


def _inputs(
    generator: torch.Generator,
    *,
    batch: int,
    seq: int,
    heads: int,
    head_dim: int,
    cos_batch: int,
):
  hidden_dim = heads * head_dim

  def rnd(*shape, scale=1.0, shift=0.0):
    return (
        torch.randn(*shape, generator=generator) * scale + shift
    ).to(torch.bfloat16)

  xq = rnd(batch, seq, hidden_dim, scale=2.0, shift=0.3)
  xk = rnd(batch, seq, hidden_dim, scale=1.5)
  xv = rnd(batch, seq, hidden_dim)
  wq = rnd(hidden_dim, scale=0.1, shift=1.0)
  wk = rnd(hidden_dim, scale=0.1, shift=1.0)
  pos = (torch.arange(seq)[None] % 300) + torch.arange(cos_batch)[:, None]
  inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
  freqs = pos[..., None].float() * inv
  emb = torch.cat([freqs, freqs], -1)
  grads = [rnd(batch, heads, seq, head_dim) for _ in range(3)]
  return (
      [xq, xk, xv, wq, wk],
      (emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)),
      grads,
  )


def _reference(
    xq, xk, xv, wq, wk, cos, sin, *, heads: int, head_dim: int, q_scale: float
):
  """HuggingFace `EsmcAttention` prologue (heads-first, `rotate_half` RoPE)."""
  batch, seq, hidden_dim = xq.shape

  def heads_first(t):
    return t.view(batch, seq, heads, head_dim).transpose(1, 2)

  def rotate_half(t):
    return torch.cat([-t[..., head_dim // 2 :], t[..., : head_dim // 2]], -1)

  q = heads_first(
      torch.nn.functional.layer_norm(xq, (hidden_dim,), wq, None, EPS).to(
          xq.dtype
      )
  )
  k = heads_first(
      torch.nn.functional.layer_norm(xk, (hidden_dim,), wk, None, EPS).to(
          xk.dtype
      )
  )
  c, s = cos.unsqueeze(1), sin.unsqueeze(1)
  q_out = q * c + rotate_half(q) * s
  if q_scale != 1.0:
    q_out = q_out * q_scale
  return q_out, k * c + rotate_half(k) * s, heads_first(xv)


class QkNormRopeTest(
    base_device_test.BaseAcceleratorDeviceTest, parameterized.TestCase
):

  def test_supported_contract_on_cpu(self):
    # Importing `qk_norm_rope` and calling pure-Python / tensor-contract helpers
    # must succeed on CPU without importing `torch_tpu` or `jax`.
    cpu_xq = torch.empty(2, 512, 18 * 64, device="cpu")
    self.assertIn("not tpu", qk_norm_rope.supported(cpu_xq, 18, 64) or "")
    self.assertEqual(qk_norm_rope.lane_group_width(64), 128)
    self.assertEqual(qk_norm_rope.lane_group_width(128), 128)
    self.assertEqual(qk_norm_rope.lane_group_width(256), 256)
    with self.assertRaisesRegex(ValueError, "head_dim"):
      qk_norm_rope.lane_group_width(32)
    self.assertEqual(qk_norm_rope.pick_seq_block(2048, 512), 512)
    self.assertEqual(qk_norm_rope.pick_seq_block(128, 512), 128)
    self.assertIsNone(qk_norm_rope.pick_seq_block(24, 12))
    self.assertIsNone(qk_norm_rope.pick_seq_block(1000, 512))
    self.assertEqual(qk_norm_rope.default_block_s(2048, 18, 64), 1024)
    self.assertEqual(qk_norm_rope.default_block_s(1536, 18, 64), 512)
    self.assertEqual(qk_norm_rope.default_block_s(2048, 40, 64), 256)
    self.assertEqual(qk_norm_rope.default_block_s(2048, 32, 128), 256)

  def test_lazy_pallas_registration_signature_eval_str(self):
    # `torch.tpu.pallas.jax_op` calls `inspect.signature(fn, eval_str=True)`.
    # Verify on CPU (by stubbing `pallas.jax_op`) that both lazy JAX op wrappers
    # expose resolvable type annotations without NameError. Pre-import `jax`
    # outside `mock.patch.dict` so C++ `jaxlib` extensions are never evicted
    # from `sys.modules` if TPU hardware tests run in the same process.
    import jax  # pylint: disable=g-import-not-at-top,unused-import
    from torchtitan.experiments.tpu.kernels import qk_norm_rope_kernel  # pylint: disable=g-import-not-at-top,unused-import

    seen_names: list[str] = []

    def fake_jax_op(name, fn):
      sig = inspect.signature(inspect.unwrap(fn), eval_str=True)
      self.assertGreaterEqual(len(sig.parameters), 11)
      seen_names.append(name)
      op = mock.MagicMock(name=name)
      return op

    fake_pallas = types.ModuleType("torch_tpu._internal.pallas")
    fake_pallas.jax_op = fake_jax_op
    fake_internal = types.ModuleType("torch_tpu._internal")
    fake_internal.pallas = fake_pallas
    fake_root = types.ModuleType("torch_tpu")
    fake_root._internal = fake_internal

    qk_norm_rope._get_pallas_ops.cache_clear()
    try:
      with mock.patch.dict(
          sys.modules,
          {
              "torch_tpu": fake_root,
              "torch_tpu._internal": fake_internal,
              "torch_tpu._internal.pallas": fake_pallas,
          },
      ):
        fwd_op, _ = qk_norm_rope._get_pallas_ops()
        fwd_op.register_autograd.assert_called_once()
    finally:
      qk_norm_rope._get_pallas_ops.cache_clear()
    self.assertEqual(
        seen_names, ["pallas::qk_norm_rope_fwd", "pallas::qk_norm_rope_bwd"]
    )

  def _run(self, fn, device, dtype, leaves, tables, grads):
    leaves = [
        t.to(device=device, dtype=dtype).requires_grad_(True) for t in leaves
    ]
    cos, sin = (t.to(device=device, dtype=dtype) for t in tables)
    outs = fn(*leaves, cos, sin)
    loss = sum(
        (o.float() * g.to(device).float()).sum() for o, g in zip(outs, grads)
    )
    loss.backward()
    return [o.detach().float().cpu() for o in outs] + [
        t.grad.float().cpu() for t in leaves
    ]

  @parameterized.named_parameters(
      dict(
          testcase_name="single_tile_d64_batched_qscale",
          seq=512,
          block_s=512,
          heads=18,
          head_dim=64,
          cos_batch=2,
          q_scale=0.125,
      ),
      dict(
          testcase_name="multi_tile_s2048_d64_unbatched_qscale",
          seq=2048,
          block_s=512,
          heads=18,
          head_dim=64,
          cos_batch=1,
          q_scale=0.125,
      ),
      dict(
          testcase_name="multi_tile_d128_unit_scale",
          seq=512,
          block_s=256,
          heads=8,
          head_dim=128,
          cos_batch=2,
          q_scale=1.0,
      ),
      dict(
          testcase_name="adaptive_default_s2048_esmc600m_bs1024",
          seq=2048,
          block_s=None,
          heads=18,
          head_dim=64,
          cos_batch=1,
          q_scale=0.125,
      ),
      dict(
          testcase_name="adaptive_default_s1536_esmc600m_bs512",
          seq=1536,
          block_s=None,
          heads=18,
          head_dim=64,
          cos_batch=1,
          q_scale=0.125,
      ),
      dict(
          testcase_name="adaptive_default_s2048_esmc6b_bs256",
          seq=2048,
          block_s=None,
          heads=40,
          head_dim=64,
          cos_batch=1,
          q_scale=0.125,
      ),
      dict(
          testcase_name="adaptive_default_s512_h32_d128_bs256",
          seq=512,
          block_s=None,
          heads=32,
          head_dim=128,
          cos_batch=1,
          q_scale=1.0,
      ),
  )
  def test_matches_reference_forward_and_backward(
      self,
      seq: int,
      block_s: int | None,
      heads: int,
      head_dim: int,
      cos_batch: int,
      q_scale: float,
  ):
    if self.accelerator_device.type != "tpu":
      raise unittest.SkipTest("Pallas TPU hardware execution test")

    leaves, tables, grads = _inputs(
        torch.Generator().manual_seed(0),
        batch=2,
        seq=seq,
        heads=heads,
        head_dim=head_dim,
        cos_batch=cos_batch,
    )

    def fused(*args):
      return qk_norm_rope.fused_qk_norm_rope(
          *args, heads=heads, eps=EPS, q_scale=q_scale, block_s=block_s
      )

    def ref(*args):
      return _reference(
          *args, heads=heads, head_dim=head_dim, q_scale=q_scale
      )

    got = self._run(
        fused, self.accelerator_device, torch.bfloat16, leaves, tables, grads
    )
    xla = self._run(
        ref, self.accelerator_device, torch.bfloat16, leaves, tables, grads
    )
    gold = self._run(ref, "cpu", torch.float32, leaves, tables, grads)
    names = ["q", "k", "v", "dxq", "dxk", "dxv", "dwq", "dwk"]
    for name, a, b, c in zip(names, got, xla, gold):
      err = ((a - c).norm() / c.norm().clamp_min(1e-30)).item()
      xla_err = ((b - c).norm() / c.norm().clamp_min(1e-30)).item()
      self.assertLessEqual(err, max(1.5 * xla_err, 1e-6) + 1e-3, name)


if __name__ == "__main__":
  absltest.main()
