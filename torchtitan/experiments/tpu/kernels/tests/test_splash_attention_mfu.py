# Copyright 2026 Google LLC

"""Full-attention MFU and parity regression tests for TPU Splash kernels.

Lives in its own module so ``setUpModule`` can raise
``xla_tpu_scoped_vmem_limit_kib`` to 64 MiB before the first TPU program
compiles on TPU >= v6 (unlocking 2048x4096 tiles at 32K context) without
changing the default scoped VMEM budget for the rest of ``kernels/tests/``.
"""

import re
import time
import types

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.nn import attention
import torch.nn.functional as F
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.kernels import splash_attention
from torchtitan.experiments.tpu.kernels import splash_bias_dropout
from torchtitan.experiments.tpu.kernels import tokamax_splash_attention
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.models.utils import quadratic_attention_flops_per_token
from torchtitan.tools import utils

_MIN_ATTENTION_MFU = 0.30


def _tpu_device_name() -> str | None:
  get_props = getattr(
      getattr(torch, "tpu", None), "get_device_properties", None
  )
  if get_props is None:
    return None
  try:
    return get_props().name
  except RuntimeError:
    return None


def _tpu_generation(device_name: str | None = None) -> int | None:
  name = (device_name if device_name is not None else _tpu_device_name()) or ""
  match = re.search(r"(?:v|tpu[- ]?)(\d+)", name.lower())
  return int(match.group(1)) if match else None


def setUpModule():  # pylint: disable=invalid-name
  # PCI sysfs lookup does not initialize PJRT, so libtpu / XLA flags applied
  # here take effect when the first TPU tensor or compiled kernel runs.
  gen = _tpu_generation()
  if gen is not None and gen >= 6:
    tpu_flags.apply_init_flags(
        types.SimpleNamespace(
            libtpu_init_flags={"xla_tpu_scoped_vmem_limit_kib": "65536"},
            extra_libtpu_init_flags={},
            xla_flags={},
            extra_xla_flags={},
        )
    )


def _relative_l2_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
  actual_f = actual.double().cpu()
  expected_f = expected.double().cpu()
  return float(
      (actual_f - expected_f).norm() / expected_f.norm().clamp_min(1e-12)
  )


def _sdpa_math_bhsd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
  with attention.sdpa_kernel([attention.SDPBackend.MATH]):
    return F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=False
    )


def _sampled_full_attention_bhsd(
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
  """Exact FP32 full attention on ``row_indices`` queries over all keys."""
  q_sub = q_bhsd[:, :, row_indices, :].float().cpu()
  k_cpu = k_bhsd.float().cpu()
  v_cpu = v_bhsd.float().cpu()
  scale = q_bhsd.shape[-1] ** -0.5
  scores = torch.matmul(q_sub, k_cpu.transpose(-2, -1)) * scale
  probs = torch.softmax(scores, dim=-1)
  return torch.matmul(probs, v_cpu)


class SplashAttentionMfuTest(base_device_test.BaseAcceleratorDeviceTest):
  """Verifies forward/backward parity and >30% attention MFU on TPU."""

  def _require_tpu(self) -> tuple[torch.device, str, int]:
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("Splash Attention MFU test requires a TPU device.")
    device_name = _tpu_device_name() or "TPU"
    gen = _tpu_generation(device_name) or 5
    return self.accelerator_device, device_name, gen

  def _save_and_update(self, cfg_cls, **updates) -> None:
    for key, value in updates.items():
      self.addCleanup(setattr, cfg_cls, key, getattr(cfg_cls, key))
      setattr(cfg_cls, key, value)

  def _benchmark_spec(
      self, gen: int
  ) -> tuple[tuple[int, int, int, int], dict[str, int]]:
    if gen >= 6:
      # FLUX 32K full-attention benchmark shape on TPU v6e / TPU v7x (64 MiB
      # scoped VMEM unlocked in setUpModule).
      bhsd = (1, 32, 32768, 128)
      blocks = dict(
          block_q=2048,
          block_kv=4096,
          block_kv_compute=512,
          block_q_dkv=2048,
          block_kv_dkv=2048,
          block_kv_dkv_compute=2048,
      )
    else:
      # TPU v5e has 16 MiB physical VMEM and 16 GiB HBM per core.
      bhsd = (1, 16, 16384, 128)
      blocks = dict(
          block_q=512,
          block_kv=1024,
          block_kv_compute=512,
          block_q_dkv=512,
          block_kv_dkv=1024,
          block_kv_dkv_compute=512,
      )
    return bhsd, blocks

  def _make_kernel_fn(
      self, kernel_kind: str, blocks: dict[str, int], gen: int
  ) -> tuple[callable, bool]:
    """Returns ``(attn_fn, heads_first)`` configured with ``blocks``."""
    if kernel_kind == "splash_sdpa":
      stock_kwargs = dict(blocks)
      if gen >= 6:
        stock_kwargs.update(
            block_kv_dkv=4096,
            block_kv_dkv_compute=512,
            k_layout="SEQ_MINOR",
        )

      def run_stock(q, k, v):
        return splash_attention.splash_sdpa(
            q,
            k,
            v,
            is_causal=False,
            enable_gqa=False,
            use_vmap_bwd=True,
            **stock_kwargs,
        )

      return run_stock, True

    if kernel_kind == "tokamax_splash_sdpa":
      tokamax_kwargs = dict(blocks)
      if gen >= 6:
        tokamax_kwargs.update(
            block_kv_dkv=4096,
            block_kv_dkv_compute=512,
            k_layout="SEQ_MINOR",
        )
      self._save_and_update(
          tokamax_splash_attention.TokamaxConfig,
          use_vmap_bwd=True,
          **tokamax_kwargs,
      )

      def run_tokamax(q, k, v):
        return tokamax_splash_attention.splash_sdpa(
            q, k, v, is_causal=False, enable_gqa=False
        )

      return run_tokamax, False

    if kernel_kind == "sbd_splash_attention":
      self._save_and_update(
          splash_bias_dropout.SbdConfig,
          use_base2_exp=True,
          dq_reduction_steps=3,
          use_experimental_scheduler=(gen >= 7),
          **blocks,
      )

      def run_sbd(q, k, v):
        return splash_bias_dropout.sbd_splash_attention(
            q, k, v, is_causal=False, heads_first=True
        )

      return run_sbd, True

    raise ValueError(f"Unknown kernel_kind: {kernel_kind}")

  @parameterized.named_parameters(
      dict(testcase_name="_splash_sdpa", kernel_kind="splash_sdpa"),
      dict(
          testcase_name="_tokamax_splash_sdpa",
          kernel_kind="tokamax_splash_sdpa",
      ),
      dict(
          testcase_name="_sbd_splash_attention",
          kernel_kind="sbd_splash_attention",
      ),
  )
  def test_full_attention_parity_and_mfu_above_30_percent(
      self, kernel_kind: str
  ):
    dev, device_name, gen = self._require_tpu()
    (batch_size, heads, seq_len, head_dim), blocks = self._benchmark_spec(gen)
    attn_fn, heads_first = self._make_kernel_fn(kernel_kind, blocks, gen)

    # 1. Full forward + backward numerical parity on a multi-tile shape.
    torch.manual_seed(0)
    parity_seq_len = 4096
    parity_bhsd = (1, 4, parity_seq_len, head_dim)
    q_cpu, k_cpu, v_cpu, do_cpu = (
        torch.randn(parity_bhsd, dtype=torch.bfloat16) for _ in range(4)
    )
    q_ref = q_cpu.float().requires_grad_(True)
    k_ref = k_cpu.float().requires_grad_(True)
    v_ref = v_cpu.float().requires_grad_(True)
    out_ref = _sdpa_math_bhsd(q_ref, k_ref, v_ref)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out_ref, (q_ref, k_ref, v_ref), do_cpu.float()
    )

    def to_layout(x_bhsd: torch.Tensor) -> torch.Tensor:
      return x_bhsd if heads_first else x_bhsd.transpose(1, 2).contiguous()

    def from_layout(x: torch.Tensor) -> torch.Tensor:
      return x if heads_first else x.transpose(1, 2)

    q_p, k_p, v_p = (
        to_layout(t).to(dev).requires_grad_(True) for t in (q_cpu, k_cpu, v_cpu)
    )
    out_p = attn_fn(q_p, k_p, v_p)
    dq_p, dk_p, dv_p = torch.autograd.grad(
        out_p, (q_p, k_p, v_p), to_layout(do_cpu).to(dev)
    )
    for name, got, want in (
        ("out", from_layout(out_p), out_ref),
        ("dq", from_layout(dq_p), dq_ref),
        ("dk", from_layout(dk_p), dk_ref),
        ("dv", from_layout(dv_p), dv_ref),
    ):
      self.assertLess(_relative_l2_error(got, want), 1.5e-2, msg=name)

    # 2. Compiled full-scale FLUX benchmark: verify sampled 32K accuracy + MFU.
    torch.manual_seed(42)
    bench_shape = (
        (batch_size, heads, seq_len, head_dim)
        if heads_first
        else (batch_size, seq_len, heads, head_dim)
    )
    q, k, v = (
        torch.randn(
            bench_shape, device=dev, dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    )
    grad_out = torch.randn_like(q)

    @torch.compile(backend="tpu", dynamic=False)
    def step(q_in, k_in, v_in, do_in):
      out = attn_fn(q_in, k_in, v_in)
      dq, dk, dv = torch.autograd.grad(out, (q_in, k_in, v_in), do_in)
      return out, dq, dk, dv

    torch.accelerator.synchronize()
    out, dq, dk, dv = step(q, k, v, grad_out)
    torch.accelerator.synchronize()

    for name, tensor in (("out", out), ("dq", dq), ("dk", dk), ("dv", dv)):
      self.assertEqual(tuple(tensor.shape), bench_shape, msg=name)
      self.assertEqual(tensor.dtype, torch.bfloat16, msg=name)
      sample = tensor[0, 0, :16] if heads_first else tensor[0, :16, 0]
      self.assertTrue(
          bool(torch.isfinite(sample.cpu()).all().item()),
          msg=f"{name} has non-finite values",
      )

    row_idx = torch.tensor(
        [0, seq_len // 2 - 1, seq_len // 2, seq_len - 1], dtype=torch.long
    )
    out_bhsd_rows = (
        out[:, :, row_idx.to(dev), :].cpu()
        if heads_first
        else out[:, row_idx.to(dev), :, :].transpose(1, 2).cpu()
    )
    expected_rows = _sampled_full_attention_bhsd(
        from_layout(q.detach()),
        from_layout(k.detach()),
        from_layout(v.detach()),
        row_idx,
    )
    self.assertLess(
        _relative_l2_error(out_bhsd_rows, expected_rows),
        1.5e-2,
        msg="full-scale sampled output parity",
    )

    warmup_iters = 3
    repeats = 3
    timed_iters = 10
    for _ in range(warmup_iters):
      out, dq, dk, dv = step(q, k, v, grad_out)
    torch.accelerator.synchronize()

    best_seconds_per_iter = float("inf")
    for _ in range(repeats):
      start = time.perf_counter()
      for _ in range(timed_iters):
        out, dq, dk, dv = step(q, k, v, grad_out)
      torch.accelerator.synchronize()
      seconds_per_iter = (time.perf_counter() - start) / timed_iters
      best_seconds_per_iter = min(best_seconds_per_iter, seconds_per_iter)

    flops_per_token = quadratic_attention_flops_per_token(
        num_heads=heads,
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        seq_len=seq_len,
    )
    total_flops = batch_size * seq_len * flops_per_token
    peak_flops = utils.get_peak_flops(device_name)
    achieved_flops = total_flops / best_seconds_per_iter
    mfu = achieved_flops / peak_flops

    logging.info(
        "[%s on %s] shape=%s fwd+bwd=%.2f ms, %.2f TFLOP/s / %.2f TFLOP/s"
        " peak -> attention MFU=%.2f%%",
        kernel_kind,
        device_name,
        (batch_size, heads, seq_len, head_dim),
        best_seconds_per_iter * 1e3,
        achieved_flops / 1e12,
        peak_flops / 1e12,
        mfu * 100.0,
    )
    self.assertGreater(
        mfu,
        _MIN_ATTENTION_MFU,
        msg=(
            f"{kernel_kind} achieved {mfu:.2%} attention MFU on {device_name}"
            f" ({best_seconds_per_iter * 1e3:.2f} ms/iter,"
            f" {achieved_flops / 1e12:.2f} TFLOP/s vs"
            f" {peak_flops / 1e12:.2f} TFLOP/s peak), expected"
            f" > {_MIN_ATTENTION_MFU:.0%}."
        ),
    )


if __name__ == "__main__":
  absltest.main()
