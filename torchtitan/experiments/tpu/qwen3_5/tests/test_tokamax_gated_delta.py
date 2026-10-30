"""Tests Tokamax's Mosaic KDA GatedDeltaNet kernels against the XLA path."""

import contextlib
import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
import torchtitan.experiments.tpu.qwen3_5 as qwen3_5_tpu
from torchtitan.experiments.tpu.qwen3_5 import gated_delta
from torchtitan.experiments.tpu.qwen3_5 import tokamax_gated_delta
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
from torchtitan.models.common.attention import segment_ids_from_positions


def _relative_l2_error(test: torch.Tensor, reference: torch.Tensor) -> float:
  test, reference = test.float().cpu(), reference.float().cpu()
  return float((test - reference).norm() / reference.norm().clamp_min(1e-12))


class TokamaxGatedDeltaTest(base_device_test.BaseAcceleratorDeviceTest):
  """``tokamax_gated_delta`` vs ``chunked_gated_delta`` on one TPU chip."""

  def _require_mosaic_kda(self) -> None:
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("The Mosaic KDA kernels need a TPU.")
    if not tokamax_gated_delta.supports_current_tpu():
      self.skipTest("The Mosaic KDA kernels need TPU v6 or newer.")
    if not tokamax_gated_delta.has_tokamax_kda():
      self.skipTest(
          tokamax_gated_delta.tokamax_kda_import_error()
          or "Tokamax KDA op is not installed."
      )

  @parameterized.named_parameters(
      dict(testcase_name="_small", batch=2, seq_len=256, heads=4),
      dict(
          testcase_name="_custom_min_log_decay",
          batch=2,
          seq_len=256,
          heads=4,
          min_log_decay=-3.5,
      ),
      # The 35B-A3B GatedDeltaNet shape (32 value heads x 128) and its
      # initialization's decays: A_log = log(U(0.01, 16)), dt_bias = 1 and
      # g = -exp(A_log) * softplus(a + dt_bias), down to about -70, so many
      # are clamped to `min_log_decay`.
      dict(
          testcase_name="_model_shape_init_gates",
          batch=4,
          seq_len=2048,
          heads=32,
          init_gates=True,
      ),
  )
  def test_tokamax_matches_xla(
      self,
      batch: int,
      seq_len: int,
      heads: int,
      init_gates: bool = False,
      min_log_decay: float = -5.0,
  ):
    self._require_mosaic_kda()

    torch.manual_seed(0)
    dev = self.accelerator_device
    shape = (batch, seq_len, heads)
    q = torch.randn(*shape, 128, dtype=torch.bfloat16, device=dev)
    k = torch.randn(*shape, 128, dtype=torch.bfloat16, device=dev)
    v = torch.randn(*shape, 128, dtype=torch.bfloat16, device=dev)
    if init_gates:
      a_log = torch.empty(heads, device=dev).uniform_(0.01, 16.0).log()
      g = -torch.exp(a_log) * F.softplus(torch.randn(*shape, device=dev) + 1.0)
    else:
      # Log decays between about -6 and 0.
      g = -F.softplus(2.0 * torch.randn(*shape, device=dev))
    beta = torch.sigmoid(torch.randn(*shape, device=dev)).to(torch.bfloat16)
    grad_out = torch.randn(*shape, 128, dtype=torch.bfloat16, device=dev)

    def run(impl, g_in):
      inputs = [
          x.detach().clone().requires_grad_() for x in (q, k, v, g_in, beta)
      ]
      out = impl(*inputs)
      out.backward(grad_out)
      return [out] + [x.grad for x in inputs]

    # The reference sees the same clamped decays.
    expected = run(gated_delta.chunked_gated_delta, g.clamp_min(min_log_decay))
    actual = run(
        lambda *x: tokamax_gated_delta.tokamax_gated_delta(
            *x, min_log_decay=min_log_decay
        ),
        g,
    )
    clamped = g < min_log_decay
    # The clamp passes no gradient to clamped decays.
    self.assertTrue(bool((actual[4][clamped] == 0).all().item()))
    for name, a_t, e_t in zip(
        ("out", "dq", "dk", "dv", "dg", "dbeta"), actual, expected, strict=True
    ):
      if name == "dg":
        # The reference's leaf is already clamped, so compare the others.
        a_t, e_t = a_t * ~clamped, e_t * ~clamped
      self.assertTrue(torch.isfinite(a_t).all().item(), msg=f"{name}")
      # dg sums bf16-rounded terms over the key channels and time (TPU7x:
      # 1.7e-2 to 3.1e-2); the other outputs are at 3.7e-3 to 5.0e-3.
      self.assertLess(
          _relative_l2_error(a_t, e_t),
          5e-2 if name == "dg" else 1e-2,
          msg=f"{name}",
      )
    # Control: halving the decays moves the output well beyond the bound.
    halved = run(
        gated_delta.chunked_gated_delta, 0.5 * g.clamp_min(min_log_decay)
    )
    self.assertGreater(_relative_l2_error(halved[0], expected[0]), 5e-2)
    if init_gates:
      # The clamp changes the function: compare with the unclamped one, which
      # the XLA path computes (many of the initialization's decays are clamped;
      # TPU7x: 7.2e-3).
      unclamped = run(gated_delta.chunked_gated_delta, g)
      deviation = _relative_l2_error(actual[0], unclamped[0])
      self.assertLess(deviation, 2e-2, msg=f"clamp deviation {deviation:.2e}")

  def test_override_arguments_and_validation(self):
    """Tests the override's arguments and the Config checks."""
    kernel_cfg = tokamax_gated_delta.tokamax_gated_delta_kernel(
        gated_delta.TpuGatedDeltaKernel.Config(chunk_size=32),
        min_log_decay=-4.0,
        max_num_segments=32,
    )
    self.assertIsInstance(
        kernel_cfg, tokamax_gated_delta.TokamaxGatedDeltaKernel.Config
    )
    self.assertEqual(
        (kernel_cfg.chunk_size, kernel_cfg.min_log_decay),
        (32, -4.0),
    )
    self.assertEqual(kernel_cfg.max_num_segments, 32)
    kernel = kernel_cfg.build()
    self.assertEqual(kernel._num_segments(8192), 32)  # pylint: disable=protected-access
    self.assertEqual(kernel._num_segments(16), 16)  # pylint: disable=protected-access
    default = tokamax_gated_delta.TokamaxGatedDeltaKernel.Config().build()
    self.assertEqual(default._num_segments(512), 4)  # pylint: disable=protected-access
    self.assertEqual(default._num_segments(8192), 16)  # pylint: disable=protected-access
    for bad in (dict(min_log_decay=-6.0), dict(min_log_decay=0.0)):
      with self.assertRaisesRegex(ValueError, "min_log_decay"):
        tokamax_gated_delta.TokamaxGatedDeltaKernel.Config(**bad).build()
    with self.assertRaisesRegex(ValueError, "max_num_segments"):
      tokamax_gated_delta.TokamaxGatedDeltaKernel.Config(
          max_num_segments=0
      ).build()

  def test_unsupported_inputs_fall_back_to_xla(self):
    """Tests that inputs the Mosaic kernels do not take run the XLA path."""
    kernel = tokamax_gated_delta.TokamaxGatedDeltaKernel.Config().build()
    cases = [("cpu", torch.device("cpu"), 100, None)]
    if self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
      # 100-token rows without document ids: not a multiple of 64.
      cases.append(("non_chunk_multiple", self.accelerator_device, 100, None))
      # TPU v5 generation (< v6 Mosaic requirement) on a 128-token row.
      cases.append(("tpu_v5_gen", self.accelerator_device, 128, 5))
    torch.manual_seed(0)
    for case_name, dev, seq_len, mock_gen in cases:
      with self.subTest(case=case_name):
        shape = (2, seq_len, 4)
        q, k = (torch.randn(*shape, 128, device=dev) for _ in range(2))
        v = torch.randn(*shape, 128, device=dev)
        g = -F.softplus(torch.randn(*shape, device=dev))
        beta = torch.sigmoid(torch.randn(*shape, device=dev))
        gen_ctx = (
            mock.patch.object(
                tokamax_gated_delta, "_tpu_generation", return_value=mock_gen
            )
            if mock_gen is not None
            else contextlib.nullcontext()
        )
        with gen_ctx, mock.patch.object(
            tokamax_gated_delta,
            "tokamax_gated_delta",
            side_effect=AssertionError("ran the Mosaic kernels"),
        ):
          out = kernel.recurrence(q, k, v, g, beta)
        # On TPU two evaluations of the same fp32 XLA computation differ
        # slightly (up to 6.7e-5 here).
        tolerances = dict(atol=2e-4, rtol=2e-3) if dev.type == "tpu" else {}
        torch.testing.assert_close(
            out, gated_delta.chunked_gated_delta(q, k, v, g, beta), **tolerances
        )

  @parameterized.named_parameters(
      dict(testcase_name="_all_documents", max_num_segments=8),
      # Raw 0-based `segment_ids_from_positions` passed directly into
      # `tokamax_gated_delta`: normalized per row instead of zeroing id 0.
      dict(
          testcase_name="_raw_zero_based_ids",
          max_num_segments=8,
          pass_raw_ids=True,
      ),
      # Default budget (`max_num_segments=None` on `TokamaxGatedDeltaKernel`):
      # keeps at least `MIN_NUM_SEGMENTS` (4) documents apart even on a
      # 512-token row (`512 // TOKENS_PER_SEGMENT == 1`).
      dict(testcase_name="_kernel_default", max_num_segments=None),
      # Fewer segments than documents: the last documents of row 0 share one
      # recurrence, as tokamax_segment_ids merges them.
      dict(testcase_name="_capped", max_num_segments=3),
      # 8192-token context at 35B head count (32 x 128) with the default 16
      # segment budget (`8192 // 512 == 16`), packing 20 short documents so the
      # last 5 merge into segment 16.
      dict(
          testcase_name="_ctx8k_35b_heads",
          max_num_segments=None,
          seq_len=8192,
          heads=32,
          doc_lens=((400,) * 19 + (592,),),
      ),
  )
  def test_tokamax_matches_xla_with_packed_documents(
      self,
      max_num_segments: int | None,
      pass_raw_ids: bool = False,
      seq_len: int = 512,
      heads: int = 4,
      doc_lens: tuple[tuple[int, ...], ...] = (
          (100, 28, 1, 200, 183),
          (512,),
      ),
  ):
    """Tests Tokamax's segment_ids path against the XLA segment masks."""
    self._require_mosaic_kda()

    min_log_decay = -5.0
    torch.manual_seed(0)
    dev = self.accelerator_device
    batch = len(doc_lens)
    positions = torch.stack(
        [torch.cat([torch.arange(n) for n in lens]) for lens in doc_lens]
    )
    segment_ids_BS = (
        segment_ids_from_positions(positions.reshape(-1))
        .reshape(batch, seq_len)
        .to(dev)
    )
    effective_segments = max_num_segments or max(
        tokamax_gated_delta.MIN_NUM_SEGMENTS,
        seq_len // tokamax_gated_delta.TOKENS_PER_SEGMENT,
    )
    segment_ids = tokamax_gated_delta.tokamax_segment_ids(
        segment_ids_BS, effective_segments
    )
    shape = (batch, seq_len, heads)
    q = torch.randn(*shape, 128, dtype=torch.bfloat16, device=dev)
    k = torch.randn(*shape, 128, dtype=torch.bfloat16, device=dev)
    v = torch.randn(*shape, 128, dtype=torch.bfloat16, device=dev)
    # Log decays between about -6 and 0.
    g = -F.softplus(2.0 * torch.randn(*shape, device=dev))
    beta = torch.sigmoid(torch.randn(*shape, device=dev)).to(torch.bfloat16)
    grad_out = torch.randn(*shape, 128, dtype=torch.bfloat16, device=dev)

    def run(impl, g_in):
      inputs = [
          x.detach().clone().requires_grad_() for x in (q, k, v, g_in, beta)
      ]
      out = impl(*inputs)
      out.backward(grad_out)
      return [out] + [x.grad for x in inputs]

    expected = run(
        lambda *xs: gated_delta.chunked_gated_delta(
            *xs, segment_ids=segment_ids
        ),
        g.clamp_min(min_log_decay),
    )
    if max_num_segments is None:
      kernel = tokamax_gated_delta.TokamaxGatedDeltaKernel(
          tokamax_gated_delta.TokamaxGatedDeltaKernel.Config(
              min_log_decay=min_log_decay
          )
      )
      actual = run(
          lambda *xs: kernel.recurrence(*xs, segment_ids_BS=segment_ids_BS),
          g,
      )
    else:
      ids_arg = segment_ids_BS if pass_raw_ids else segment_ids
      actual = run(
          lambda *xs: tokamax_gated_delta.tokamax_gated_delta(
              *xs,
              min_log_decay=min_log_decay,
              segment_ids=ids_arg,
              max_num_segments=max_num_segments,
          ),
          g,
      )
    for name, a_t, e_t in zip(
        ("out", "dq", "dk", "dv", "dg", "dbeta"), actual, expected, strict=True
    ):
      if name == "dg":
        mask = g > min_log_decay
        a_t, e_t = a_t * mask, e_t * mask
      self.assertTrue(torch.isfinite(a_t).all().item(), msg=f"{name}")
      self.assertLess(
          _relative_l2_error(a_t, e_t),
          5e-2 if name == "dg" else 1e-2,
          msg=f"{name}",
      )

    # Control: without segment ids the state leaks into the first tokens of
    # row 0's second document.
    first_boundary = doc_lens[0][0]
    leaky = run(
        lambda *xs: tokamax_gated_delta.tokamax_gated_delta(
            *xs, min_log_decay=min_log_decay
        ),
        g,
    )
    self.assertGreater(
        _relative_l2_error(
            leaky[0][0, first_boundary : first_boundary + 4],
            expected[0][0, first_boundary : first_boundary + 4],
        ),
        5e-2,
    )

  def test_gated_delta_net_module_matches_xla(self):
    """Tests full 35B TpuGatedDeltaNet (conv, GQA, gates, norm) vs XLA."""
    self._require_mosaic_kda()

    torch.manual_seed(0)
    dev = self.accelerator_device
    base_cfg = qwen3_5_tpu.model_registry("35B-A3B", seq_len=256).model
    xla_cfg = copy.deepcopy(base_cfg.layers[0].delta_net)
    tok_cfg = copy.deepcopy(xla_cfg)
    tok_cfg.inner_gated_delta_net.kernel = (
        tokamax_gated_delta.TokamaxGatedDeltaKernel.Config(min_log_decay=-5.0)
    )

    xla_gdn = xla_cfg.build()
    xla_gdn.init_states(buffer_device=torch.device("cpu"))
    # Keep decays above the -5 clamp (`exp(A_log) <= 0.5`) so XLA and Tokamax
    # evaluate the exact same mathematical function across all parameters.
    with torch.no_grad():
      xla_gdn.A_log.uniform_(-3.0, -0.7)
      xla_gdn.dt_bias.uniform_(-0.5, 0.5)
    tok_gdn = tok_cfg.build()
    tok_gdn.load_state_dict(copy.deepcopy(xla_gdn.state_dict()))
    xla_gdn = xla_gdn.to(device=dev, dtype=torch.bfloat16)
    tok_gdn = tok_gdn.to(device=dev, dtype=torch.bfloat16)
    # Keep A_log and dt_bias in fp32 as Qwen3.5 mixed precision / init expects.
    for mod in (xla_gdn, tok_gdn):
      mod.A_log.data = mod.A_log.data.float()
      mod.dt_bias.data = mod.dt_bias.data.float()

    batch, seq_len = 2, 256
    doc_lens = ((96, 64, 96), (160, 96))
    positions = torch.cat(
        [torch.cat([torch.arange(n) for n in lens]) for lens in doc_lens]
    ).to(dev)
    x_base = torch.randn(
        batch * seq_len,
        xla_cfg.in_proj_q.in_features,
        dtype=torch.bfloat16,
        device=dev,
    )
    grad_out = torch.randn_like(x_base)

    def run_module(mod):
      x_in = x_base.detach().clone().requires_grad_()
      with set_unflatten_seq_len(seq_len):
        out = mod(x_in, positions=positions)
      out.backward(grad_out)
      grads = {n: p.grad for n, p in mod.named_parameters()}
      return out, x_in.grad, grads

    expected_out, expected_dx, expected_grads = run_module(xla_gdn)
    actual_out, actual_dx, actual_grads = run_module(tok_gdn)

    self.assertLess(_relative_l2_error(actual_out, expected_out), 1.5e-2)
    self.assertLess(_relative_l2_error(actual_dx, expected_dx), 1.5e-2)
    for name, e_grad in expected_grads.items():
      a_grad = actual_grads[name]
      self.assertIsNotNone(a_grad, msg=name)
      self.assertTrue(torch.isfinite(a_grad).all().item(), msg=name)
      bound = 5e-2 if name in ("A_log", "dt_bias", "in_proj_a.weight") else 2e-2
      self.assertLess(
          _relative_l2_error(a_grad, e_grad), bound, msg=f"grad {name}"
      )


if __name__ == "__main__":
  absltest.main()
