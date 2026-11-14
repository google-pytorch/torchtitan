"""Accelerator tests for the Pallas QK RMSNorm + CosSinRoPE kernels."""

import contextlib
import copy
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
# pyformat: disable
# pylint: disable=g-bad-import-order
# With torch_tpu, torch must be imported before spmd_types to avoid a circular
# import when torch autoloads torch_tpu -> torch.distributed.fsdp -> spmd_types.
import torch  # isort: skip
import spmd_types as spmd
from spmd_types.checker import typecheck as spmd_typecheck
# pylint: enable=g-bad-import-order
# pyformat: enable
from torch import nn
from torch.distributed.tensor import DTensor
import torch.nn.functional as F

from torchtitan.experiments.tpu.components import bld_layout
from torchtitan.experiments.tpu.components import fused_qk_norm_rope
from torchtitan.experiments.tpu.components import unflatten_shim
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.examples import gemma4
from torchtitan.experiments.tpu.examples.gemma4 import model as gemma4_model
from torchtitan.experiments.tpu.kernels import qk_norm_rope
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.models import common
from torchtitan.models import qwen3


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
  """PyTorch reference matching ``F.rms_norm`` followed by ``cossin_rotate``."""
  y = F.rms_norm(x, (x.shape[-1],), weight, eps)
  rotary_half = x.shape[-1] // 2
  cs_shape = (*x.shape[:-2], 1, rotary_half)
  return bld_layout.cossin_rotate(
      y, cos.float().reshape(cs_shape), sin.float().reshape(cs_shape)
  )


def _reference_fp64(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
  """FP64 oracle preserving the ``x.dtype`` intermediate rounding boundary."""
  xd = x.double()
  rstd = torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)
  y = (weight.double() * (xd * rstd)).to(x.dtype).double()
  rotary_half = x.shape[-1] // 2
  cs_shape = (*x.shape[:-2], 1, rotary_half)
  cd = cos.double().reshape(cs_shape)
  sd = sin.double().reshape(cs_shape)
  y1, y2 = y[..., :rotary_half], y[..., rotary_half:]
  o1 = (y1 * cd - y2 * sd).to(x.dtype).double()
  o2 = (y2 * cd + y1 * sd).to(x.dtype).double()
  return torch.cat((o1, o2), dim=-1)


def _relative_l2_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
  ad = actual.detach().double()
  ed = expected.detach().double()
  return float((ad - ed).norm() / ed.norm().clamp_min(1e-12))


class QkNormRopeTest(base_device_test.BaseAcceleratorDeviceTest):
  """Parity, compile, validation, SPMD, and module-wiring tests."""

  def _require_tpu(self) -> None:
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("Pallas qk_norm_rope requires TPU.")

  @parameterized.named_parameters(
      dict(
          testcase_name="_4d_qwen3_q_bf16",
          shape=(2, 2048, 16, 128),
          cs_4d=False,
          x_dtype=torch.bfloat16,
          w_dtype=torch.bfloat16,
          cs_dtype=torch.float32,
          tol=1e-2,
      ),
      dict(
          testcase_name="_4d_qwen3_k_fp32_weight",
          shape=(2, 2048, 4, 128),
          cs_4d=True,
          x_dtype=torch.bfloat16,
          w_dtype=torch.float32,
          cs_dtype=torch.float32,
          tol=1e-2,
      ),
      dict(
          testcase_name="_3d_gemma4_sliding_dim256",
          shape=(2048, 8, 256),
          cs_4d=True,
          x_dtype=torch.bfloat16,
          w_dtype=torch.float32,
          cs_dtype=torch.bfloat16,
          tol=1e-2,
      ),
      dict(
          testcase_name="_3d_gemma4_global_dim512_bf16",
          shape=(1024, 2, 512),
          cs_4d=True,
          x_dtype=torch.bfloat16,
          w_dtype=torch.bfloat16,
          cs_dtype=torch.float32,
          tol=1e-2,
      ),
      dict(
          testcase_name="_3d_gemma4_global_dim512_fp32",
          shape=(1024, 2, 512),
          cs_4d=False,
          x_dtype=torch.float32,
          w_dtype=torch.float32,
          cs_dtype=torch.float32,
          tol=1e-5,
      ),
      dict(
          testcase_name="_4d_single_kv_head_dim64",
          shape=(2, 512, 1, 64),
          cs_4d=False,
          x_dtype=torch.bfloat16,
          w_dtype=torch.bfloat16,
          cs_dtype=torch.float32,
          tol=1e-2,
      ),
      dict(
          testcase_name="_4d_non_pow2_dim80_h6_seq384",
          shape=(2, 384, 6, 80),
          cs_4d=False,
          x_dtype=torch.bfloat16,
          w_dtype=torch.bfloat16,
          cs_dtype=torch.float32,
          tol=1e-2,
      ),
      dict(
          testcase_name="_3d_dim192",
          shape=(512, 3, 192),
          cs_4d=True,
          x_dtype=torch.bfloat16,
          w_dtype=torch.bfloat16,
          cs_dtype=torch.float32,
          tol=1e-2,
      ),
      dict(
          testcase_name="_3d_max_dim1024",
          shape=(512, 2, 1024),
          cs_4d=False,
          x_dtype=torch.bfloat16,
          w_dtype=torch.bfloat16,
          cs_dtype=torch.float32,
          tol=1e-2,
      ),
  )
  def test_matches_reference_forward_and_backward(
      self,
      shape: tuple[int, ...],
      cs_4d: bool,
      x_dtype: torch.dtype,
      w_dtype: torch.dtype,
      cs_dtype: torch.dtype,
      tol: float,
  ):
    self._require_tpu()
    torch.manual_seed(0)
    head_dim = shape[-1]
    cs_shape = (
        (*shape[:-2], 1, head_dim // 2)
        if cs_4d
        else (*shape[:-2], head_dim // 2)
    )
    x = (torch.randn(*shape) * 2.5).to(x_dtype)
    weight = (1.0 + 0.15 * torch.randn(head_dim)).to(w_dtype)
    angle = torch.rand(*cs_shape) * 50.0
    cos = angle.cos().to(cs_dtype)
    sin = angle.sin().to(cs_dtype)
    grad = torch.randn(*shape, dtype=x_dtype)

    x_ref = x.clone().requires_grad_(True)
    w_ref = weight.clone().requires_grad_(True)
    out_ref = _reference(x_ref, w_ref, cos, sin, 1e-6)
    dx_ref, dw_ref = torch.autograd.grad(out_ref, (x_ref, w_ref), grad)

    dev = self.accelerator_device
    x_dev = x.to(dev).detach().requires_grad_(True)
    w_dev = weight.to(dev).detach().requires_grad_(True)
    cos_dev, sin_dev = cos.to(dev), sin.to(dev)
    self.assertTrue(qk_norm_rope.supports(x_dev, w_dev, cos_dev, sin_dev))
    out = qk_norm_rope.qk_norm_rope(x_dev, w_dev, cos_dev, sin_dev, 1e-6)
    dx, dw = torch.autograd.grad(out, (x_dev, w_dev), grad.to(dev))

    self.assertEqual(out.shape, shape)
    self.assertEqual(out.dtype, x_dtype)
    self.assertLess(_relative_l2_error(out.cpu(), out_ref), tol)
    for name, actual, expected in (
        ("dx", dx, dx_ref),
        ("dw", dw, dw_ref),
    ):
      self.assertEqual(actual.shape, expected.shape, name)
      self.assertEqual(actual.dtype, expected.dtype, name)
      self.assertLess(_relative_l2_error(actual.cpu(), expected), tol, name)

  def test_gemma4_partial_rotary_factor_matches_unfused(self):
    self._require_tpu()
    torch.manual_seed(3)
    dev = self.accelerator_device
    rope = (
        gemma4_model.Gemma4RoPE.Config(
            dim=512,
            max_context_length=1024,
            theta=1000000.0,
            partial_rotary_factor=0.25,
        )
        .build()
        .to(dev)
    )
    x = torch.randn(512, 2, 512, device=dev, dtype=torch.bfloat16)
    w = 1.0 + 0.1 * torch.randn(512, device=dev, dtype=torch.float32)
    positions = torch.cat([torch.arange(300), torch.arange(212)]).to(dev)
    grad = torch.randn_like(x)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    y_ref = F.rms_norm(x_ref, (512,), w_ref, 1e-6)
    out_ref = rope(y_ref, None, positions)
    dx_ref, dw_ref = torch.autograd.grad(out_ref, (x_ref, w_ref), grad)

    x_f = x.detach().clone().requires_grad_(True)
    w_f = w.detach().clone().requires_grad_(True)
    cache = rope._reshape_cache(  # pylint: disable=protected-access
        x_f, positions
    )
    cos_h = cache[..., :256]
    sin_h = cache[..., 512:768]
    out = qk_norm_rope.qk_norm_rope(x_f, w_f, cos_h, sin_h, 1e-6)
    dx, dw = torch.autograd.grad(out, (x_f, w_f), grad)

    self.assertLess(_relative_l2_error(out.cpu(), out_ref.cpu()), 1e-2)
    self.assertLess(_relative_l2_error(dx.cpu(), dx_ref.cpu()), 1e-2)
    self.assertLess(_relative_l2_error(dw.cpu(), dw_ref.cpu()), 1e-2)

  def test_large_row_biased_bf16_weight_gradient_accumulates_in_fp32(self):
    self._require_tpu()
    self.addCleanup(torch._dynamo.reset)  # pylint: disable=protected-access
    torch.manual_seed(7)
    shape = (2, 16384, 2, 128)
    x = (0.3 + torch.randn(*shape)).to(torch.bfloat16)
    weight_bf16 = (1.0 + 0.2 * torch.randn(128)).to(torch.bfloat16)
    angle = torch.rand(2, 16384, 64) * 10.0
    cos, sin = angle.cos(), angle.sin()
    grad = (0.2 + 0.5 * torch.randn(*shape)).to(torch.bfloat16)

    x_ref = x.clone().double().requires_grad_(True)
    w_ref = weight_bf16.clone().double().requires_grad_(True)
    out_ref = _reference_fp64(x_ref, w_ref, cos, sin, 1e-6)
    _, dw_ref = torch.autograd.grad(out_ref, (x_ref, w_ref), grad.double())

    dev = self.accelerator_device
    x_dev = x.to(dev).detach().requires_grad_(True)
    w_f32 = weight_bf16.float().to(dev).detach().requires_grad_(True)
    cos_dev, sin_dev = cos.to(dev), sin.to(dev)
    grad_dev = grad.to(dev)
    out = qk_norm_rope.qk_norm_rope(x_dev, w_f32, cos_dev, sin_dev, 1e-6)
    _, dw_f32 = torch.autograd.grad(out, (x_dev, w_f32), grad_dev)
    self.assertEqual(dw_f32.dtype, torch.float32)
    self.assertLess(_relative_l2_error(dw_f32.cpu(), dw_ref), 2e-3)

    # Verify torch.compile with BF16 weights preserves FP32 reduction before
    # the final BF16 cast (guards against XLA Bfloat16Propagation demotion).
    def compiled_bwd(x_in, w_in, c_in, s_in, dy_in):
      y = qk_norm_rope.qk_norm_rope(x_in, w_in, c_in, s_in, 1e-6)
      return torch.autograd.grad(y, (w_in,), dy_in)[0]

    compiled_fn = torch.compile(compiled_bwd, fullgraph=True, backend="tpu")
    w_bf16_dev = weight_bf16.to(dev).detach().requires_grad_(True)
    dw_eager_bf16 = compiled_bwd(x_dev, w_bf16_dev, cos_dev, sin_dev, grad_dev)
    dw_comp_bf16 = compiled_fn(x_dev, w_bf16_dev, cos_dev, sin_dev, grad_dev)
    self.assertEqual(dw_comp_bf16.dtype, torch.bfloat16)
    self.assertLess(_relative_l2_error(dw_comp_bf16.cpu(), dw_ref), 3e-3)
    torch.testing.assert_close(
        dw_comp_bf16.cpu(), dw_eager_bf16.cpu(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        dw_comp_bf16.cpu(), dw_f32.cpu().to(torch.bfloat16), rtol=0, atol=0
    )

  def test_non_contiguous_partial_grads_and_fullgraph_compile(self):
    self._require_tpu()
    self.addCleanup(torch._dynamo.reset)  # pylint: disable=protected-access
    torch.manual_seed(11)
    dev = self.accelerator_device
    full = torch.randn(2, 512, 4, 256, device=dev, dtype=torch.bfloat16)
    x_nc = full[..., :128].detach().requires_grad_(True)
    dy_nc = full[..., 128:].detach()
    w = torch.ones(128, device=dev, dtype=torch.bfloat16, requires_grad=False)
    cos = torch.ones(2, 512, 64, device=dev, dtype=torch.float32)
    sin = torch.zeros(2, 512, 64, device=dev, dtype=torch.float32)
    self.assertFalse(x_nc.is_contiguous())
    self.assertFalse(dy_nc.is_contiguous())

    # need_dx=True, need_dw=False with non-contiguous x and dy
    out = qk_norm_rope.qk_norm_rope(x_nc, w, cos, sin, 1e-6)
    (dx,) = torch.autograd.grad(out, (x_nc,), dy_nc)
    self.assertEqual(dx.shape, x_nc.shape)

    # need_dx=False, need_dw=True
    x_frozen = x_nc.detach().requires_grad_(False)
    w_only = w.detach().clone().requires_grad_(True)
    out_w = qk_norm_rope.qk_norm_rope(x_frozen, w_only, cos, sin, 1e-6)
    (dw_only,) = torch.autograd.grad(out_w, (w_only,), dy_nc)
    self.assertEqual(dw_only.shape, w_only.shape)

    # Inference (grad disabled)
    with torch.no_grad():
      out_ng = qk_norm_rope.qk_norm_rope(x_frozen, w, cos, sin, 1e-6)
    self.assertFalse(out_ng.requires_grad)

    def step(x_in, w_in, c_in, s_in, dy_in):
      y = qk_norm_rope.qk_norm_rope(x_in, w_in, c_in, s_in, 1e-6)
      return torch.autograd.grad(y, (x_in, w_in), dy_in)

    compiled = torch.compile(step, fullgraph=True, backend="tpu")
    x_c = x_nc.detach().contiguous().requires_grad_(True)
    w_c = w.detach().clone().requires_grad_(True)
    dy_c = dy_nc.contiguous()
    grads_c = compiled(x_c, w_c, cos, sin, dy_c)
    grads_e = step(x_c, w_c, cos, sin, dy_c)
    for g_comp, g_eager in zip(grads_c, grads_e, strict=True):
      torch.testing.assert_close(g_comp.cpu(), g_eager.cpu())

  def test_patched_qwen3_and_gemma4_modules_execute_fused_kernel(self):
    self._require_tpu()
    torch.manual_seed(17)
    dev = self.accelerator_device
    batch, seq_len, dim, head_dim = 2, 128, 128, 32

    q_cfg = qwen3._build_qwen3_layers(  # pylint: disable=protected-access
        n_layers=1,
        dim=dim,
        n_heads=4,
        n_kv_heads=2,
        head_dim=head_dim,
        hidden_dim=256,
        fuse_qkv=True,
        attn_backend="sdpa",
        rope=common.CosSinRoPE.Config(dim=head_dim, max_context_length=256),
    )[0]
    q_ref = q_cfg.build().to(device=dev, dtype=torch.bfloat16)
    q_bld = copy.deepcopy(q_ref)
    q_2d = copy.deepcopy(q_ref)
    self.assertEqual(bld_layout.apply_bld_layout(q_bld), 1)
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(q_bld), 1)
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(q_2d), 1)

    x_raw = torch.randn(batch * seq_len, dim, device=dev, dtype=torch.bfloat16)
    dy = torch.randn_like(x_raw)
    positions = torch.arange(seq_len, device=dev).repeat(batch)

    with unflatten_shim.set_unflatten_seq_len(seq_len):
      x_ref = x_raw.detach().clone().requires_grad_(True)
      out_ref = q_ref(x_ref, None, positions)
      out_ref.backward(dy)

      for name, patched_mod in (("qwen3_bld", q_bld), ("qwen3_2d", q_2d)):
        x_pat = x_raw.detach().clone().requires_grad_(True)
        with mock.patch.object(
            qk_norm_rope, "qk_norm_rope", wraps=qk_norm_rope.qk_norm_rope
        ) as spy:
          out_pat = patched_mod(x_pat, None, positions)
          out_pat.backward(dy)
        self.assertEqual(spy.call_count, 2, name)
        self.assertLess(_relative_l2_error(out_pat.cpu(), out_ref.cpu()), 1e-2)
        self.assertLess(
            _relative_l2_error(x_pat.grad.cpu(), x_ref.grad.cpu()), 2e-2
        )

    # Gemma 4 sliding (D=64) + global k_eq_v (D=128, partial_rotary_factor=0.25)
    g_cfgs = gemma4._build_gemma4_layers(  # pylint: disable=protected-access
        n_layers=2,
        dim=dim,
        hidden_dim=256,
        n_heads=4,
        n_kv_heads=2,
        n_global_kv_heads=2,
        head_dim=64,
        global_head_dim=128,
        norm_eps=1e-6,
        max_seq_len=256,
        sliding_window_size=64,
        global_attn_every_n_layers=2,
        use_k_eq_v_on_global=True,
        attn_backend="sdpa",
    )
    g_ref = nn.ModuleList([cfg.build() for cfg in g_cfgs]).to(
        device=dev, dtype=torch.bfloat16
    )
    g_pat = copy.deepcopy(g_ref)
    self.assertEqual(fused_qk_norm_rope.apply_fused_qk_norm_rope(g_pat), 2)

    xg_ref = x_raw.detach().clone().requires_grad_(True)
    hg_ref = xg_ref
    for layer in g_ref:
      hg_ref = layer(hg_ref, None, positions)
    hg_ref.backward(dy)

    xg_pat = x_raw.detach().clone().requires_grad_(True)
    with mock.patch.object(
        qk_norm_rope, "qk_norm_rope", wraps=qk_norm_rope.qk_norm_rope
    ) as spy:
      hg_pat = xg_pat
      for layer in g_pat:
        hg_pat = layer(hg_pat, None, positions)
      hg_pat.backward(dy)
    self.assertEqual(spy.call_count, 4)
    self.assertLess(_relative_l2_error(hg_pat.cpu(), hg_ref.cpu()), 1e-2)
    self.assertLess(
        _relative_l2_error(xg_pat.grad.cpu(), xg_ref.grad.cpu()), 2e-2
    )

  def test_supports_and_validation(self):
    x_cpu = torch.zeros(2, 256, 4, 128)
    w_cpu = torch.ones(128)
    c_cpu = torch.ones(2, 256, 64)
    self.assertFalse(qk_norm_rope.supports(x_cpu, w_cpu, c_cpu, c_cpu))
    with self.assertRaisesRegex(ValueError, "Unsupported inputs"):
      qk_norm_rope.qk_norm_rope(x_cpu, w_cpu, c_cpu, c_cpu, 1e-6)

    with mock.patch.object(
        qk_norm_rope.pltpu,
        "get_tpu_info",
        side_effect=ValueError("Unsupported TPU device kind: cpu"),
    ):
      self.assertIsNone(
          qk_norm_rope._vmem_limit_bytes()  # pylint: disable=protected-access
      )

    dev = self.accelerator_device
    x_ok = torch.zeros(2, 256, 4, 128, device=dev, dtype=torch.bfloat16)
    w_ok = torch.ones(128, device=dev, dtype=torch.bfloat16)
    c_ok = torch.ones(2, 256, 64, device=dev, dtype=torch.float32)
    c_grad = c_ok.detach().clone().requires_grad_(True)

    bad_cases = [
        # Wrong rank (2-D)
        (
            torch.zeros(256, 128, device=dev, dtype=torch.bfloat16),
            w_ok,
            c_ok,
            c_ok,
        ),
        # Unaligned 4-D seq_len
        (
            torch.zeros(2, 100, 4, 128, device=dev, dtype=torch.bfloat16),
            w_ok,
            torch.ones(2, 100, 64, device=dev),
            torch.ones(2, 100, 64, device=dev),
        ),
        # Unaligned 3-D token count
        (
            torch.zeros(100, 4, 128, device=dev, dtype=torch.bfloat16),
            w_ok,
            torch.ones(100, 64, device=dev),
            torch.ones(100, 64, device=dev),
        ),
        # Invalid 3-D cos shape
        (
            torch.zeros(256, 4, 128, device=dev, dtype=torch.bfloat16),
            w_ok,
            torch.ones(256, 32, device=dev),
            torch.ones(256, 32, device=dev),
        ),
        # Unaligned head_dim (not multiple of 16)
        (
            torch.zeros(2, 256, 4, 24, device=dev, dtype=torch.bfloat16),
            torch.ones(24, device=dev, dtype=torch.bfloat16),
            torch.ones(2, 256, 12, device=dev),
            torch.ones(2, 256, 12, device=dev),
        ),
        # head_dim below minimum (< 16)
        (
            torch.zeros(2, 256, 4, 8, device=dev, dtype=torch.bfloat16),
            torch.ones(8, device=dev, dtype=torch.bfloat16),
            torch.ones(2, 256, 4, device=dev),
            torch.ones(2, 256, 4, device=dev),
        ),
        # head_dim above maximum (> 1024)
        (
            torch.zeros(2, 256, 2, 1040, device=dev, dtype=torch.bfloat16),
            torch.ones(1040, device=dev, dtype=torch.bfloat16),
            torch.ones(2, 256, 520, device=dev),
            torch.ones(2, 256, 520, device=dev),
        ),
        # Mismatched weight shape
        (x_ok, torch.ones(64, device=dev, dtype=torch.bfloat16), c_ok, c_ok),
        # Mismatched cos shape vs sin shape
        (x_ok, w_ok, c_ok, torch.ones(2, 256, 32, device=dev)),
        # Unsupported dtype (float16)
        (
            torch.zeros(2, 256, 4, 128, device=dev, dtype=torch.float16),
            torch.ones(128, device=dev, dtype=torch.float16),
            c_ok,
            c_ok,
        ),
        # Differentiable cos or sin rejected when grad is enabled
        (x_ok, w_ok, c_grad, c_ok),
        (x_ok, w_ok, c_ok, c_grad),
    ]
    fake_tpu_dev = types.SimpleNamespace(type="tpu", index=0)
    if dev.type == "tpu":
      dev_ctx = contextlib.nullcontext()
      # Also verify cross-device mismatch when a real TPU device is present.
      self.assertFalse(qk_norm_rope.supports(x_ok, torch.ones(128), c_ok, c_ok))
    else:
      dev_ctx = mock.patch.object(
          torch.Tensor, "device", new=property(lambda _self: fake_tpu_dev)
      )
    with dev_ctx:
      self.assertTrue(qk_norm_rope.supports(x_ok, w_ok, c_ok, c_ok))
      with torch.no_grad():
        self.assertTrue(qk_norm_rope.supports(x_ok, w_ok, c_grad, c_grad))
      for x_bad, w_bad, c_bad, s_bad in bad_cases:
        self.assertFalse(qk_norm_rope.supports(x_bad, w_bad, c_bad, s_bad))
        with self.assertRaisesRegex(ValueError, "Unsupported inputs"):
          qk_norm_rope.qk_norm_rope(x_bad, w_bad, c_bad, s_bad, 1e-6)

      with self.assertRaisesRegex(ValueError, "eps must be non-negative"):
        qk_norm_rope.qk_norm_rope(x_ok, w_ok, c_ok, c_ok, -1e-6)

    fake_dtensor = mock.create_autospec(DTensor, instance=True)
    self.assertFalse(qk_norm_rope.supports(fake_dtensor, w_ok, c_ok, c_ok))
    self.assertFalse(qk_norm_rope.supports(x_ok, fake_dtensor, c_ok, c_ok))
    self.assertFalse(qk_norm_rope.supports(x_ok, w_ok, fake_dtensor, c_ok))
    self.assertFalse(qk_norm_rope.supports(x_ok, w_ok, c_ok, fake_dtensor))

    ok_jax = jnp.zeros((2, 256, 4, 128), dtype=jnp.bfloat16)
    w_jax = jnp.ones((128,), dtype=jnp.float32)
    cs_jax = jnp.ones((2, 256, 64), dtype=jnp.float32)
    jax_bad_cases = [
        (jnp.zeros((256, 4, 128), jnp.bfloat16), w_jax, cs_jax, "4-D"),
        (jnp.zeros((0, 256, 4, 128), jnp.bfloat16), w_jax, cs_jax, "positive"),
        (
            jnp.zeros((2, 256, 4, 24), jnp.bfloat16),
            jnp.ones((24,), jnp.float32),
            jnp.ones((2, 256, 12), jnp.float32),
            "divisible",
        ),
        (
            jnp.zeros((2, 100, 4, 128), jnp.bfloat16),
            w_jax,
            jnp.ones((2, 100, 64), jnp.float32),
            "positive multiple",
        ),
        (ok_jax, jnp.ones((64,), jnp.float32), cs_jax, "weight shape"),
        (ok_jax, w_jax, jnp.ones((2, 256, 32), jnp.float32), "cos/sin shapes"),
    ]
    for x_j, w_j, cs_j, pat in jax_bad_cases:
      with self.assertRaisesRegex(ValueError, pat):
        qk_norm_rope.qk_norm_rope_fwd_jax(x_j, w_j, cs_j, cs_j, 1e-6)
    with self.assertRaisesRegex(ValueError, "dy shape"):
      qk_norm_rope.qk_norm_rope_bwd_jax(
          ok_jax,
          jnp.zeros((2, 256, 2, 128), jnp.bfloat16),
          w_jax,
          cs_jax,
          cs_jax,
          1e-6,
      )

  def test_spmd_types_global_and_local_typecheck(self):
    dp = spmd.MeshAxis.of(2, 4)
    tp = spmd.MeshAxis.of(4, 1)

    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      with spmd.set_current_mesh({"dp": dp, "tp": tp}):
        x_cpu = torch.zeros(1024, 4, 128)
        x_ok_cpu = torch.zeros(1024, 4, 128)
        w_cpu = torch.ones(128)
        w_ok_cpu = torch.ones(128)
        c_cpu = torch.ones(1024, 1, 64)
        c_bad_cpu = torch.ones(1024, 1, 64)
        spmd.assert_type(x_cpu, {dp: spmd.S(0), tp: spmd.S(2)})
        spmd.assert_type(x_ok_cpu, {dp: spmd.S(0), tp: spmd.S(1)})
        spmd.assert_type(w_cpu, {dp: spmd.R, tp: spmd.S(0)})
        spmd.assert_type(w_ok_cpu, {dp: spmd.R, tp: spmd.R})
        spmd.assert_type(c_cpu, {dp: spmd.S(0), tp: spmd.R})
        spmd.assert_type(c_bad_cpu, {dp: spmd.S(0), tp: spmd.S(2)})
        with spmd_typecheck(local=False):
          self.assertFalse(qk_norm_rope.supports(x_cpu, w_ok_cpu, c_cpu, c_cpu))
          self.assertFalse(qk_norm_rope.supports(x_ok_cpu, w_cpu, c_cpu, c_cpu))
          self.assertFalse(
              qk_norm_rope.supports(x_ok_cpu, w_ok_cpu, c_bad_cpu, c_cpu)
          )
      return

    dev = self.accelerator_device
    torch.manual_seed(23)
    with spmd.set_current_mesh({"dp": dp, "tp": tp}):
      x = torch.randn(
          1024, 4, 128, device=dev, dtype=torch.bfloat16, requires_grad=True
      )
      w = torch.ones(128, device=dev, dtype=torch.float32, requires_grad=True)
      cos = torch.ones(1024, 1, 64, device=dev, dtype=torch.float32)
      sin = torch.zeros(1024, 1, 64, device=dev, dtype=torch.float32)
      dy = torch.randn(1024, 4, 128, device=dev, dtype=torch.bfloat16)
      x_bad = torch.randn(1024, 4, 128, device=dev, dtype=torch.bfloat16)
      w_bad = torch.ones(128, device=dev, dtype=torch.float32)

      spmd.assert_type(x, {dp: spmd.S(0), tp: spmd.S(1)})
      spmd.assert_type(w, {dp: spmd.R, tp: spmd.R})
      spmd.assert_type(cos, {dp: spmd.S(0), tp: spmd.R})
      spmd.assert_type(sin, {dp: spmd.S(0), tp: spmd.R})
      spmd.assert_type(dy, {dp: spmd.S(0), tp: spmd.S(1)})
      spmd.assert_type(x_bad, {dp: spmd.S(0), tp: spmd.S(2)})
      spmd.assert_type(w_bad, {dp: spmd.R, tp: spmd.S(0)})

      x_ref = x.detach().clone().requires_grad_(True)
      w_ref = w.detach().clone().requires_grad_(True)
      out_ref = _reference(x_ref, w_ref, cos, sin, 1e-6)
      out_ref.backward(dy)

      with spmd_typecheck(local=False):
        self.assertTrue(qk_norm_rope.supports(x, w, cos, sin))
        self.assertFalse(qk_norm_rope.supports(x_bad, w, cos, sin))
        self.assertFalse(qk_norm_rope.supports(x, w_bad, cos, sin))
        with self.assertRaises((spmd.SpmdTypeError, ValueError)):
          qk_norm_rope._QkNormRopeSpmdFn.apply(  # pylint: disable=protected-access
              x_bad, w, cos, sin, 1e-6
          )
        out = qk_norm_rope.qk_norm_rope(x, w, cos, sin, 1e-6)
        expected_ps = spmd.PartitionSpec(
            spmd.normalize_axis(dp), spmd.normalize_axis(tp), None
        )
        self.assertEqual(spmd.get_partition_spec(out), expected_ps)
        self.assertEqual(
            spmd.get_local_type(out),
            {spmd.normalize_axis(dp): spmd.V, spmd.normalize_axis(tp): spmd.V},
        )
        out.backward(dy)
      self.assertLess(_relative_l2_error(out, out_ref), 1e-2)
      self.assertLess(_relative_l2_error(x.grad, x_ref.grad), 2e-2)
      self.assertLess(_relative_l2_error(w.grad, w_ref.grad), 1e-2)

      x_loc = x.detach().clone().requires_grad_(True)
      w_loc = w.detach().clone().requires_grad_(True)
      c_loc = cos.detach().clone()
      s_loc = sin.detach().clone()
      dy_loc = dy.detach().clone()
      spmd.assert_type(x_loc, {dp: spmd.V, tp: spmd.V})
      spmd.assert_type(w_loc, {dp: spmd.R, tp: spmd.R})
      spmd.assert_type(c_loc, {dp: spmd.V, tp: spmd.R})
      spmd.assert_type(s_loc, {dp: spmd.V, tp: spmd.R})
      spmd.assert_type(dy_loc, {dp: spmd.V, tp: spmd.V})
      with spmd_typecheck(local=True):
        out_loc = qk_norm_rope.qk_norm_rope(x_loc, w_loc, c_loc, s_loc, 1e-6)
        out_loc.backward(dy_loc)
      torch.testing.assert_close(x_loc.grad, x.grad, rtol=0, atol=0)
      torch.testing.assert_close(w_loc.grad, w.grad, rtol=0, atol=0)


if __name__ == "__main__":
  absltest.main()
