"""Tests for `fp8_linear` and `fp8_linear_kernel`."""

import inspect
import sys
import types
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import fp8_linear


class Fp8LinearTest(
    base_device_test.BaseAcceleratorDeviceTest, parameterized.TestCase
):

  def test_supported_contract_on_cpu(self):
    x = torch.empty(2, 256, 1152, dtype=torch.bfloat16)
    weight = torch.empty(3072, 1152, dtype=torch.bfloat16)
    self.assertIn("not tpu", fp8_linear.supported(x, weight) or "")

    tpu_dev = types.SimpleNamespace(type="tpu")

    def fake_tensor(shape, dtype=torch.bfloat16):
      numel = 1
      for d in shape:
        numel *= d
      return types.SimpleNamespace(
          device=tpu_dev,
          ndim=len(shape),
          shape=shape,
          dtype=dtype,
          numel=lambda: numel,
      )

    self.assertIsNone(
        fp8_linear.supported(
            fake_tensor((2, 256, 1152)), fake_tensor((3072, 1152))
        )
    )
    self.assertIn(
        "2-D",
        fp8_linear.supported(fake_tensor((256, 128)), fake_tensor((128,)))
        or "",
    )
    self.assertIn(
        "activations",
        fp8_linear.supported(fake_tensor((128,)), fake_tensor((128, 128)))
        or "",
    )
    self.assertIn(
        "in_features",
        fp8_linear.supported(fake_tensor((256, 128)), fake_tensor((128, 256)))
        or "",
    )
    self.assertIn(
        "rows dimension",
        fp8_linear.supported(fake_tensor((100, 128)), fake_tensor((128, 128)))
        or "",
    )
    self.assertIn(
        "out dimension",
        fp8_linear.supported(fake_tensor((128, 128)), fake_tensor((192, 128)))
        or "",
    )
    self.assertIn(
        "bfloat16 or float32",
        fp8_linear.supported(
            fake_tensor((128, 128), dtype=torch.float16),
            fake_tensor((128, 128)),
        )
        or "",
    )

  def test_lazy_pallas_registration_signature_eval_str(self):
    import jax  # pylint: disable=g-import-not-at-top,unused-import
    from torchtitan.experiments.tpu.kernels import fp8_linear_kernel  # pylint: disable=g-import-not-at-top,unused-import

    seen_names: list[str] = []

    def fake_jax_op(name, fn):
      sig = inspect.signature(inspect.unwrap(fn), eval_str=True)
      self.assertGreaterEqual(len(sig.parameters), 2)
      seen_names.append(name)
      return mock.MagicMock(name=name)

    fake_pallas = types.ModuleType("torch_tpu._internal.pallas")
    fake_pallas.jax_op = fake_jax_op
    fake_internal = types.ModuleType("torch_tpu._internal")
    fake_internal.pallas = fake_pallas
    fake_root = types.ModuleType("torch_tpu")
    fake_root._internal = fake_internal  # pylint: disable=protected-access

    fp8_linear._get_pallas_ops.cache_clear()  # pylint: disable=protected-access
    try:
      with mock.patch.dict(
          sys.modules,
          {
              "torch_tpu": fake_root,
              "torch_tpu._internal": fake_internal,
              "torch_tpu._internal.pallas": fake_pallas,
          },
      ):
        fwd_op, _ = fp8_linear._get_pallas_ops()  # pylint: disable=protected-access
        fwd_op.register_autograd.assert_called_once()
    finally:
      fp8_linear._get_pallas_ops.cache_clear()  # pylint: disable=protected-access
    self.assertEqual(
        seen_names, ["pallas::fp8_linear_fwd", "pallas::fp8_linear_bwd"]
    )

  def test_tile_config_validation_and_single_k_shapes(self):
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    from torchtitan.experiments.tpu.kernels import fp8_linear_kernel as kernel  # pylint: disable=g-import-not-at-top

    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      kernel.TileConfig(block_m=64, block_k=128, block_n=128)
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      kernel.TileConfig(block_m=128, block_k=0, block_n=128)

    self.assertEqual(
        kernel.get_fwd_tiles(16384, 5632, 2048, lhs_dtype=jnp.bfloat16),
        kernel.TileConfig(block_m=512, block_k=5632, block_n=2048),
    )
    self.assertEqual(
        kernel.get_fwd_tiles(16384, 2048, 5632, lhs_dtype=jnp.bfloat16),
        kernel.TileConfig(block_m=512, block_k=2048, block_n=5632),
    )
    self.assertEqual(
        kernel.get_fwd_tiles(16384, 2048, 11264, lhs_dtype=jnp.bfloat16),
        kernel.TileConfig(block_m=256, block_k=2048, block_n=11264),
    )
    self.assertEqual(
        kernel.get_fwd_tiles(32768, 1152, 6144, lhs_dtype=jnp.bfloat16),
        kernel.TileConfig(block_m=512, block_k=1152, block_n=6144),
    )
    self.assertEqual(
        kernel.get_fwd_tiles(32768, 3072, 1152, lhs_dtype=jnp.bfloat16),
        kernel.TileConfig(block_m=1024, block_k=3072, block_n=1152),
    )

  @parameterized.parameters(
      (8192, 5632, 2048),
      (16384, 5632, 2048),
      (8192, 2048, 5632),
      (16384, 2048, 5632),
      (8192, 2048, 11264),
      (16384, 2048, 11264),
      (32768, 1152, 6144),
      (32768, 3072, 1152),
      (4096, 16384, 16384),
      (16384, 9600, 9600),
      (256, 128, 128),
  )
  def test_tiles_divide_and_bound_vmem(self, m, k, n):
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    from torchtitan.experiments.tpu.kernels import fp8_linear_kernel as kernel  # pylint: disable=g-import-not-at-top

    for dtype in (jnp.bfloat16, jnp.float32):
      itemsize = jnp.dtype(dtype).itemsize
      fwd = kernel.get_fwd_tiles(m, k, n, lhs_dtype=dtype)
      dlhs = kernel.get_dlhs_tiles(m, k, n, lhs_dtype=dtype)
      drhs = kernel.get_drhs_tiles(m, k, n, lhs_dtype=dtype)

      for cfg in (fwd, dlhs, drhs):
        self.assertEqual(m % cfg.block_m, 0)
        self.assertEqual(k % cfg.block_k, 0)
        self.assertEqual(n % cfg.block_n, 0)

      self.assertLessEqual(
          kernel.forward_vmem_bytes(
              fwd.block_m,
              fwd.block_k,
              fwd.block_n,
              k,
              n,
              lhs_itemsize=itemsize,
          ),
          kernel.VMEM_BUDGET_BYTES,
      )
      self.assertLessEqual(
          kernel.dlhs_vmem_bytes(
              dlhs.block_m,
              dlhs.block_k,
              dlhs.block_n,
              k,
              n,
              cotangent_itemsize=itemsize,
          ),
          kernel.DLHS_VMEM_BUDGET_BYTES,
      )
      self.assertLessEqual(
          drhs.block_m * drhs.block_k, kernel.MAX_DRHS_ACTIVATION_BLOCK_ELEMS
      )
      self.assertLessEqual(
          drhs.block_k * drhs.block_n, kernel.MAX_DRHS_OUT_TILE_ELEMS
      )
      self.assertLessEqual(
          drhs.block_m * drhs.block_n * itemsize // 2,
          kernel.MAX_DRHS_MEASURED_COTANGENT_ELEMS,
      )

  @parameterized.parameters(
      (256, 128, 384, "bfloat16", 1.0),
      (256, 128, 384, "bfloat16", 1e-6),
      (128, 384, 256, "float32", 1e-6),
  )
  def test_cpu_jax_quantizers_and_references_match_fp32_matmul(
      self, m, k, n, dtype_name, grad_scale
  ):
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    import numpy as np  # pylint: disable=g-import-not-at-top
    from torchtitan.experiments.tpu.kernels import fp8_linear_kernel as kernel  # pylint: disable=g-import-not-at-top

    dtype = getattr(jnp, dtype_name)
    with jax.default_device(jax.devices("cpu")[0]):
      k0, k1, k2 = jax.random.split(jax.random.PRNGKey(0), 3)
      lhs = jax.random.normal(k0, (m, k), dtype=dtype)
      rhs = (jax.random.normal(k1, (k, n), dtype=dtype) * 0.02).astype(dtype)
      dout = (jax.random.normal(k2, (m, n), dtype=dtype) * grad_scale).astype(
          dtype
      )

      zero_rows = lhs.at[0].set(0)
      xq, sx = kernel.quantize_rows_fp8(zero_rows)
      self.assertEqual(xq.dtype, kernel.FP8_DTYPE)
      self.assertEqual(sx.shape, (m, 1))
      self.assertEqual(sx.dtype, jnp.float32)
      self.assertTrue(bool(jnp.all(jnp.isfinite(sx))))

      zero_cols = rhs.at[:, 0].set(0)
      gq, sg = kernel.quantize_cols_fp8(zero_cols)
      self.assertEqual(gq.dtype, kernel.FP8_DTYPE)
      self.assertEqual(sg.shape, (n,))
      self.assertEqual(sg.dtype, jnp.float32)
      self.assertTrue(bool(jnp.all(jnp.isfinite(sg))))

      out, (lhs_q, lhs_scale, rhs_q, rhs_scale) = kernel.fp8_matmul_reference(
          lhs, rhs
      )
      dlhs = kernel.fp8_dlhs_reference(dout, rhs_q, rhs_scale)
      drhs = kernel.fp8_drhs_reference(lhs_q, lhs_scale, dout)

      self.assertEqual(out.dtype, dtype)
      self.assertEqual(dlhs.dtype, dtype)
      self.assertEqual(drhs.dtype, dtype)

      lhs_f = np.asarray(lhs, dtype=np.float32)
      rhs_f = np.asarray(rhs, dtype=np.float32)
      dout_f = np.asarray(dout, dtype=np.float32)
      for name, got, gold in (
          ("out", out, lhs_f @ rhs_f),
          ("dlhs", dlhs, dout_f @ rhs_f.T),
          ("drhs", drhs, lhs_f.T @ dout_f),
      ):
        got_f = np.asarray(got, dtype=np.float32)
        self.assertEqual(got_f.shape, gold.shape, name)
        rel_err = np.linalg.norm(got_f - gold) / np.linalg.norm(gold)
        self.assertLess(rel_err, 0.08, name)

  @parameterized.named_parameters(
      dict(testcase_name="square", rows=1024, k=1152, n=1152, grad_scale=1.0),
      dict(testcase_name="widening", rows=1024, k=1152, n=3072, grad_scale=1.0),
      dict(
          testcase_name="narrowing_3d_input",
          rows=1024,
          k=3072,
          n=1152,
          grad_scale=1.0,
      ),
      dict(
          testcase_name="g8_3b_down_proj_small_grad",
          rows=512,
          k=5632,
          n=2048,
          grad_scale=1e-6,
      ),
  )
  def test_matches_bf16_linear_forward_and_backward(
      self, rows, k, n, grad_scale
  ):
    if self.accelerator_device.type != "tpu":
      raise unittest.SkipTest("Pallas TPU hardware execution test")

    generator = torch.Generator().manual_seed(0)
    x = torch.randn(4, rows // 4, k, generator=generator)
    weight = torch.randn(n, k, generator=generator) / k**0.5
    grad = torch.randn(4, rows // 4, n, generator=generator) * grad_scale

    def run(fn, device, dtype):
      x_leaf = x.to(device=device, dtype=dtype).requires_grad_(True)
      w_leaf = weight.to(device=device, dtype=dtype).requires_grad_(True)
      out = fn(x_leaf, w_leaf)
      (out.float() * grad.to(device)).sum().backward()
      return [
          t.detach().float().cpu() for t in (out, x_leaf.grad, w_leaf.grad)
      ]

    got = run(fp8_linear.fp8_linear, self.accelerator_device, torch.bfloat16)
    gold = run(torch.nn.functional.linear, "cpu", torch.float32)
    for name, a, b in zip(("out", "dx", "dweight"), got, gold):
      self.assertEqual(a.shape, b.shape, name)
      rel_err = ((a - b).norm() / b.norm()).item()
      self.assertLess(rel_err, 0.08, name)


if __name__ == "__main__":
  absltest.main()
