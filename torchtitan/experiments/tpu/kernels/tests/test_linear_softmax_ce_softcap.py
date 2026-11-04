"""Parity tests for the soft-capped fused linear cross entropy.

Two oracles:

* ``soft_cap=None`` / ``output_multiplier=1.0``: the vendored kernel must be
  bit-identical to the installed Tokamax kernel (same JAX inputs, compared
  with ``jnp.array_equal``), and the torch wrapper must route the identity case
  unchanged.
* A configured cap: the whole torchtitan stack (``pallas_cross_entropy_loss``,
  which also handles ``IGNORE_INDEX`` and the ``log(V)`` correction) against a
  plain-torch fp32 reference, ``logits = x @ W.T`` then
  ``cap * tanh(logits * mul / cap)`` then ``cross_entropy(ignore_index=-100)``;
  forward loss and ``dx`` / ``dW``.

Run on TPU with ``--device=tpu``; the kernels need a real chip.
"""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torchtitan.components import loss as components_loss
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.experiments.tpu.components import loss as tpu_loss
from torchtitan.experiments.tpu.kernels import linear_softmax_ce_softcap_kernel as softcap_kernel
from torchtitan.experiments.tpu.kernels import linear_softmax_cross_entropy_loss as wrapper

_CAP = 20.0
_MUL_30B = 0.19611613513


def _reference_loss(x, weights, labels, soft_cap, output_multiplier):
  """fp32 torch reference of the SoftCappedLinear head + CE(ignore_index)."""
  logits = (x.float() @ weights.float().t()).float()
  if soft_cap is not None:
    logits = soft_cap * torch.tanh(logits * output_multiplier / soft_cap)
  else:
    logits = logits * output_multiplier
  return torch.nn.functional.cross_entropy(
      logits,
      labels,
      ignore_index=components_loss.IGNORE_INDEX,
      reduction="sum",
  )


class SoftCapKernelIdentityTest(base_device_test.BaseAcceleratorDeviceTest):
  """The vendored kernel with no transform is the Tokamax kernel, bit for bit."""

  @parameterized.parameters(
      (jnp.float32, 2048, 256, 2048),
      (jnp.bfloat16, 2048, 256, 2048),
      (jnp.bfloat16, 1024, 128, 8192 + 64),  # ragged V (202048 % 128 == 64)
  )
  def test_identity_matches_tokamax_bitwise(self, dtype, t, d, v):
    if self.accelerator_device.type != "tpu":
      self.skipTest("Pallas Mosaic kernels need a TPU")
    from tokamax._src.ops.linear_softmax_cross_entropy_loss import (  # pylint: disable=g-import-not-at-top
        pallas_mosaic_tpu_kernel as tokamax_kernel,
    )

    key = jax.random.PRNGKey(0)
    kx, kw, kl = jax.random.split(key, 3)
    x = jax.random.normal(kx, (t, d), dtype=jnp.float32).astype(dtype)
    w = (jax.random.normal(kw, (d, v), dtype=jnp.float32) * 0.05).astype(dtype)
    labels = jax.random.randint(kl, (t,), 0, v, dtype=jnp.int32)
    tiles = dict(b_block_size=1024, h_block_size=128, v_block_size=2048)

    loss_ref, lse_ref = (
        tokamax_kernel.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
            x, labels, w, reduction="sum", **tiles
        )
    )
    loss_new, lse_new = (
        softcap_kernel.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
            x, labels, w, reduction="sum", **tiles
        )
    )
    self.assertTrue(bool(jnp.array_equal(loss_ref, loss_new)))
    self.assertTrue(bool(jnp.array_equal(lse_ref, lse_new)))

    dout = jnp.asarray(1.0, dtype=jnp.float32)
    dx_ref, dw_ref = (
        tokamax_kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
            dout, lse_ref, x, labels, w, reduction="sum", **tiles
        )
    )
    dx_new, dw_new = (
        softcap_kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
            dout, lse_new, x, labels, w, reduction="sum", **tiles
        )
    )
    self.assertTrue(bool(jnp.array_equal(dx_ref, dx_new)))
    self.assertTrue(bool(jnp.array_equal(dw_ref, dw_new)))
    logging.info("identity bitwise parity OK for %s T=%d D=%d V=%d", dtype, t, d, v)

  def test_identity_uses_tokamax_op_names(self):
    # The identity case must keep its historical custom-op names so that the
    # llama3 / qwen3 recipes are byte-for-byte unaffected.
    # pylint: disable=protected-access
    names_default = wrapper._get_loss_op_names("sum", "mosaic_tpu", 1024, 512, 2048)
    names_explicit = wrapper._get_loss_op_names(
        "sum", "mosaic_tpu", 1024, 512, 2048, None, 1.0
    )
    names_capped = wrapper._get_loss_op_names(
        "sum", "mosaic_tpu", 1024, 512, 2048, _CAP, 1.0
    )
    self.assertEqual(names_default, names_explicit)
    self.assertNotEqual(names_default, names_capped)


class SoftCapParityTest(base_device_test.BaseAcceleratorDeviceTest):
  """Capped fused CE (full torchtitan stack) vs a plain-torch fp32 reference."""

  # Tolerances follow test_linear_softmax_cross_entropy_loss.py. fp32 operands
  # still go through the MXU at TPU default (bf16-pass) precision -- as in the
  # upstream identity kernel -- so fp32 is ~1e-5 relative on the loss and
  # ~2.5e-3 relative-Frobenius on the gradients (7e-3 for the uncapped case
  # whose logits reach |z| ~ 100), i.e. no better than bf16 operands, which
  # are exact products in fp32 (the reference upcasts the same rounded
  # operands; measured loss parity 1e-7) with gradients returned in bf16.
  # rel_fro is the relative Frobenius error bound on dx and dW.
  _FP32 = dict(loss_rtol=1e-4, grad_rtol=1e-2, grad_atol=1e-2, rel_fro=1e-2)
  _BF16 = dict(loss_rtol=2e-3, grad_rtol=5e-2, grad_atol=5e-2, rel_fro=2e-2)
  # Uncapped fp32 with these 4x-scaled inputs has |z| ~ 100 and a near-one-hot
  # softmax, where the bf16-pass GEMMs of the (upstream, unmodified) identity
  # path amplify to ~2.6% of max|dx| on a few hundred elements (rel-Fro 7e-3).
  # That is the regime the soft-cap exists to prevent; keep the case as a
  # regression check of the identity routing with a wider elementwise bound.
  _FP32_UNCAPPED = dict(loss_rtol=1e-4, grad_rtol=1e-2, grad_atol=5e-2, rel_fro=1e-2)

  @parameterized.named_parameters(
      # name, T, D, V, dtype, soft_cap, multiplier, tolerances
      ("t2048_v2048_fp32_nocap", 2048, 256, 2048, torch.float32, None, 1.0, _FP32_UNCAPPED),
      ("t2048_v2048_fp32_cap20_m0196", 2048, 256, 2048, torch.float32, _CAP, _MUL_30B, _FP32),
      ("t2048_v2048_fp32_cap20_m1", 2048, 256, 2048, torch.float32, _CAP, 1.0, _FP32),
      ("t4096_v2048_fp32_cap20_m0196", 4096, 256, 2048, torch.float32, _CAP, _MUL_30B, _FP32),
      ("t2048_v8192_fp32_cap20_m0196", 2048, 256, 8192, torch.float32, _CAP, _MUL_30B, _FP32),
      ("t2048_v8192_fp32_cap20_m1", 2048, 256, 8192, torch.float32, _CAP, 1.0, _FP32),
      ("t2048_v2048_fp32_nocap_m0196", 2048, 256, 2048, torch.float32, None, _MUL_30B, _FP32),
      ("t2048_v2048_bf16_cap20_m0196", 2048, 256, 2048, torch.bfloat16, _CAP, _MUL_30B, _BF16),
      ("t4096_v2048_bf16_cap20_m1", 4096, 256, 2048, torch.bfloat16, _CAP, 1.0, _BF16),
      ("t2048_v8192_bf16_cap20_m0196", 2048, 256, 8192, torch.bfloat16, _CAP, _MUL_30B, _BF16),
  )
  def test_parity(self, t, d, v, dtype, soft_cap, output_multiplier, tol):
    if self.accelerator_device.type != "tpu":
      self.skipTest("Pallas Mosaic kernels need a TPU")
    device = self.accelerator_device
    gen = torch.Generator().manual_seed(1234)
    # Scale so the logits actually reach the tanh's nonlinear range (|z| ~ 20).
    x = torch.randn(t, d, generator=gen) * 4.0
    weights = torch.randn(v, d, generator=gen) * 0.25  # [V, D] like lm_head.weight
    labels = torch.randint(0, v, (t,), generator=gen)
    labels[::7] = components_loss.IGNORE_INDEX  # ~14% ignored rows
    x = x.to(dtype)
    weights = weights.to(dtype)

    # Reference (fp32 math on the same bf16/fp32-rounded operands), on CPU.
    x_ref = x.clone().requires_grad_(True)
    w_ref = weights.clone().requires_grad_(True)
    loss_ref = _reference_loss(x_ref, w_ref, labels, soft_cap, output_multiplier)
    loss_ref.backward()

    # Fused path: (hidden, weight.T) exactly as FusedLossLinearWrapper hands it over.
    x_tpu = x.to(device).requires_grad_(True)
    w_tpu = weights.to(device).requires_grad_(True)
    loss_tpu = tpu_loss.pallas_cross_entropy_loss(
        (x_tpu, w_tpu.t()),
        labels.to(device),
        implementation="mosaic_tpu",
        b_block_size=1024,
        h_block_size=128,
        v_block_size=2048,
        soft_cap=soft_cap,
        output_multiplier=output_multiplier,
    )
    loss_tpu.backward()

    loss_val = loss_tpu.detach().cpu().float()
    dx = x_tpu.grad.cpu().float()
    dw = w_tpu.grad.cpu().float()
    dx_ref = x_ref.grad.float()
    dw_ref = w_ref.grad.float()
    rel_fro_dx = ((dx - dx_ref).norm() / dx_ref.norm()).item()
    rel_fro_dw = ((dw - dw_ref).norm() / dw_ref.norm()).item()
    logging.info(
        "T=%d D=%d V=%d %s cap=%s mul=%s: fused=%.6f ref=%.6f rel=%.3e"
        " |dx|max=%.3e |dW|max=%.3e relfro(dx)=%.3e relfro(dW)=%.3e",
        t, d, v, dtype, soft_cap, output_multiplier,
        loss_val.item(), loss_ref.item(),
        abs(loss_val.item() - loss_ref.item()) / abs(loss_ref.item()),
        (dx - dx_ref).abs().max().item(),
        (dw - dw_ref).abs().max().item(),
        rel_fro_dx, rel_fro_dw,
    )
    self.assertEqual(dx.shape, x.shape)
    self.assertEqual(dw.shape, weights.shape)
    self.assertEqual(x_tpu.grad.dtype, dtype)
    self.assertEqual(w_tpu.grad.dtype, dtype)
    torch.testing.assert_close(
        loss_val, loss_ref.detach().float(), rtol=tol["loss_rtol"], atol=1e-3
    )
    # Elementwise check with an absolute tolerance relative to the gradient's
    # own scale (dx entries reach ~10, dW entries ~100 at these shapes).
    torch.testing.assert_close(
        dx,
        dx_ref,
        rtol=tol["grad_rtol"],
        atol=tol["grad_atol"] * dx_ref.abs().max().item(),
    )
    torch.testing.assert_close(
        dw,
        dw_ref,
        rtol=tol["grad_rtol"],
        atol=tol["grad_atol"] * dw_ref.abs().max().item(),
    )
    self.assertLess(rel_fro_dx, tol["rel_fro"])
    self.assertLess(rel_fro_dw, tol["rel_fro"])
    # Ignored rows must get exactly zero hidden-state gradient.
    self.assertEqual(dx[::7].abs().max().item(), 0.0)

  def test_capped_matches_uncapped_only_when_cap_is_off(self):
    """Sanity: the cap is actually firing (capped loss != uncapped loss)."""
    if self.accelerator_device.type != "tpu":
      self.skipTest("Pallas Mosaic kernels need a TPU")
    device = self.accelerator_device
    gen = torch.Generator().manual_seed(7)
    x = (torch.randn(2048, 256, generator=gen) * 4.0).to(device)
    w = (torch.randn(2048, 256, generator=gen) * 0.25).to(device)
    labels = torch.randint(0, 2048, (2048,), generator=gen).to(device)
    kw = dict(b_block_size=1024, h_block_size=128, v_block_size=2048)
    plain = tpu_loss.pallas_cross_entropy_loss((x, w.t()), labels, **kw)
    capped = tpu_loss.pallas_cross_entropy_loss(
        (x, w.t()), labels, soft_cap=_CAP, output_multiplier=1.0, **kw
    )
    self.assertNotAlmostEqual(plain.item(), capped.item(), delta=1e-2)
    self.assertLess(
        capped.item(), 2048 * (np.log(2048) + 2 * _CAP) + 1.0
    )  # capped logits are in [-cap, cap] so the loss is bounded


if __name__ == "__main__":
  absltest.main()
