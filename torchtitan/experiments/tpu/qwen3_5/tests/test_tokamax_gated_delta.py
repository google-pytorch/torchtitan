"""Tests Tokamax's Mosaic KDA GatedDeltaNet kernels against the XLA path."""

import contextlib
import copy
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.testing import base_device_test
import torchtitan.experiments.tpu.qwen3_5 as qwen3_5_tpu
from torchtitan.experiments.tpu.qwen3_5 import gated_delta
from torchtitan.experiments.tpu.qwen3_5 import tokamax_gated_delta
from torchtitan.experiments.tpu.components import activation_checkpoint
from torchtitan.experiments.tpu.components.unflatten_shim import set_unflatten_seq_len
from torchtitan.models.common.attention import segment_ids_from_positions


def _relative_l2_error(test: torch.Tensor, reference: torch.Tensor) -> float:
  test, reference = test.float().cpu(), reference.float().cpu()
  return float((test - reference).norm() / reference.norm().clamp_min(1e-12))


def _clamp_decays_ctx(
    kernel: gated_delta.TpuGatedDeltaKernel, min_log_decay: float
):
  """Patches an XLA kernel to clamp its decays as the Tokamax kernels do."""
  parent_recurrence = gated_delta.TpuGatedDeltaKernel.recurrence

  def recurrence(
      q_BSHK,
      k_BSHK,
      v_BSHV,
      g_BSH,
      beta_BSH,
      segment_ids_BS=None,
  ):
    return parent_recurrence(
        kernel,
        q_BSHK,
        k_BSHK,
        v_BSHV,
        g_BSH.clamp_min(min_log_decay),
        beta_BSH,
        segment_ids_BS,
    )

  return mock.patch.object(
      kernel, "recurrence", autospec=True, side_effect=recurrence
  )


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
      # fp32 q/k/v/beta, which `_unsupported` also routes to the kernels.
      dict(
          testcase_name="_fp32",
          batch=2,
          seq_len=256,
          heads=4,
          dtype=torch.float32,
      ),
  )
  def test_tokamax_matches_xla(
      self,
      batch: int,
      seq_len: int,
      heads: int,
      init_gates: bool = False,
      min_log_decay: float = -5.0,
      dtype: torch.dtype = torch.bfloat16,
  ):
    self._require_mosaic_kda()

    torch.manual_seed(0)
    dev = self.accelerator_device
    shape = (batch, seq_len, heads)
    q = torch.randn(*shape, 128, dtype=dtype, device=dev)
    k = torch.randn(*shape, 128, dtype=dtype, device=dev)
    v = torch.randn(*shape, 128, dtype=dtype, device=dev)
    if init_gates:
      a_log = torch.empty(heads, device=dev).uniform_(0.01, 16.0).log()
      g = -torch.exp(a_log) * F.softplus(torch.randn(*shape, device=dev) + 1.0)
    else:
      # Log decays between about -6 and 0.
      g = -F.softplus(2.0 * torch.randn(*shape, device=dev))
    beta = torch.sigmoid(torch.randn(*shape, device=dev)).to(dtype)
    grad_out = torch.randn(*shape, 128, dtype=dtype, device=dev)

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
    self.assertEqual(actual[0].dtype, dtype)
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
      with mock.patch.object(
          tokamax_gated_delta, "_require_tokamax_kda", return_value=mock.Mock()
      ):
        dummy = torch.zeros(1, 64, 1, 128)
        dummy_g = torch.zeros(1, 64, 1)
        with self.assertRaisesRegex(ValueError, "min_log_decay"):
          tokamax_gated_delta.tokamax_gated_delta(
              dummy, dummy, dummy, dummy_g, dummy_g, **bad
          )
    with self.assertRaisesRegex(ValueError, "max_num_segments"):
      tokamax_gated_delta.TokamaxGatedDeltaKernel.Config(
          max_num_segments=0
      ).build()

    # `_tpu_generation` parses `"tpu7x"`, `"TPU v6e"`, `"TPU v5 lite"`, `"tpu-4"`.
    try:
      for dev_name, expected_gen in (
          ("TPU7x", 7),
          ("TPU v6e", 6),
          ("TPU v5 lite", 5),
          ("tpu-4", 4),
          ("unknown", None),
      ):
        tokamax_gated_delta._tpu_generation.cache_clear()  # pylint: disable=protected-access
        fake_tpu = mock.Mock()
        fake_tpu.get_device_properties.return_value = mock.Mock(name=dev_name)
        fake_tpu.get_device_properties.return_value.name = dev_name
        with mock.patch.object(torch, "tpu", fake_tpu, create=True):
          self.assertEqual(
              tokamax_gated_delta._tpu_generation(),  # pylint: disable=protected-access
              expected_gen,
              msg=dev_name,
          )
    finally:
      tokamax_gated_delta._tpu_generation.cache_clear()  # pylint: disable=protected-access

  def test_unsupported_inputs_fall_back_to_xla(self):
    """Tests that unsupported inputs run XLA with the same min_log_decay clamp."""
    kernel = tokamax_gated_delta.TokamaxGatedDeltaKernel.Config().build()
    cases = [
        (
            "cpu",
            torch.device("cpu"),
            100,
            128,
            128,
            torch.float32,
            torch.float32,
            None,
        ),
        (
            "empty_seq",
            torch.device("cpu"),
            0,
            128,
            128,
            torch.float32,
            torch.float32,
            None,
        ),
    ]
    if self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
      dev_tpu = self.accelerator_device
      # 100-token rows without document ids: not a multiple of 64.
      cases.append((
          "non_chunk_multiple",
          dev_tpu,
          100,
          128,
          128,
          torch.float32,
          torch.float32,
          None,
      ))
      # TPU v5 generation (< v6 Mosaic requirement) on a 128-token row.
      cases.append((
          "tpu_v5_gen",
          dev_tpu,
          128,
          128,
          128,
          torch.float32,
          torch.float32,
          5,
      ))
      # Head dims not divisible by 128 (`d_k=64` or `d_v=64`) and mixed dtypes.
      cases.append((
          "unaligned_dk",
          dev_tpu,
          128,
          64,
          128,
          torch.bfloat16,
          torch.bfloat16,
          None,
      ))
      cases.append((
          "unaligned_dv",
          dev_tpu,
          128,
          128,
          64,
          torch.bfloat16,
          torch.bfloat16,
          None,
      ))
      cases.append((
          "mixed_dtypes",
          dev_tpu,
          128,
          128,
          128,
          torch.bfloat16,
          torch.float32,
          None,
      ))
      cases.append((
          "empty_seq_tpu",
          dev_tpu,
          0,
          128,
          128,
          torch.bfloat16,
          torch.bfloat16,
          None,
      ))
    torch.manual_seed(0)
    for case_name, dev, seq_len, d_k, d_v, q_dtype, v_dtype, mock_gen in cases:
      with self.subTest(case=case_name):
        shape = (2, seq_len, 4)
        q = torch.randn(*shape, d_k, dtype=q_dtype, device=dev)
        k = torch.randn(*shape, d_k, dtype=q_dtype, device=dev)
        v = torch.randn(*shape, d_v, dtype=v_dtype, device=dev)
        # Include decays below `min_log_decay` (-5.0) to verify fallback clamping.
        g = -10.0 * torch.rand(*shape, device=dev)
        beta = torch.sigmoid(torch.randn(*shape, device=dev)).to(q_dtype)
        gen_ctx = (
            mock.patch.object(
                tokamax_gated_delta,
                "_tpu_generation",
                autospec=True,
                return_value=mock_gen,
            )
            if mock_gen is not None
            else contextlib.nullcontext()
        )
        q_in = q.detach().clone().requires_grad_()
        k_in = k.detach().clone().requires_grad_()
        v_in = v.detach().clone().requires_grad_()
        g_in = g.detach().clone().requires_grad_()
        beta_in = beta.detach().clone().requires_grad_()
        with (
            gen_ctx,
            mock.patch.object(
                tokamax_gated_delta,
                "tokamax_gated_delta",
                autospec=True,
                side_effect=AssertionError("ran the Mosaic kernels"),
            ),
        ):
          out = kernel.recurrence(q_in, k_in, v_in, g_in, beta_in)
        self.assertEqual(out.shape, (2, seq_len, 4, d_v))
        self.assertEqual(out.dtype, q_dtype)
        if seq_len == 0:
          out.float().sum().backward()
          self.assertIsNotNone(g_in.grad)
          continue
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        clamped = g < kernel.min_log_decay
        self.assertTrue(bool((g_in.grad[clamped] == 0).all().item()))
        # On TPU two evaluations of the same fp32 XLA computation differ
        # slightly (up to 6.7e-5 here); bf16 uses looser tolerances.
        tolerances = (
            dict(atol=1e-2, rtol=1e-2)
            if q_dtype == torch.bfloat16
            else (dict(atol=2e-4, rtol=2e-3) if dev.type == "tpu" else {})
        )
        expected = gated_delta.chunked_gated_delta(
            q, k, v, g.clamp_min(kernel.min_log_decay), beta
        )
        torch.testing.assert_close(out.detach(), expected, **tolerances)
        unclamped = gated_delta.chunked_gated_delta(q, k, v, g, beta)
        self.assertGreater(_relative_l2_error(out.detach(), unclamped), 1e-4)

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
      # 200-token rows (not a multiple of the 64-token chunk) through
      # `TokamaxGatedDeltaKernel.recurrence`: with document ids `_unsupported`
      # routes them to the kernels, which pad every document to a chunk.
      dict(
          testcase_name="_non_chunk_multiple_len",
          max_num_segments=4,
          seq_len=200,
          doc_lens=((70, 50, 80), (130, 70)),
          via_kernel=True,
      ),
      # fp32 q/k/v/beta with packed documents, also through the kernel.
      dict(
          testcase_name="_fp32",
          max_num_segments=8,
          dtype=torch.float32,
          via_kernel=True,
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
      dtype: torch.dtype = torch.bfloat16,
      via_kernel: bool = False,
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
    q = torch.randn(*shape, 128, dtype=dtype, device=dev)
    k = torch.randn(*shape, 128, dtype=dtype, device=dev)
    v = torch.randn(*shape, 128, dtype=dtype, device=dev)
    # Log decays between about -6 and 0.
    g = -F.softplus(2.0 * torch.randn(*shape, device=dev))
    beta = torch.sigmoid(torch.randn(*shape, device=dev)).to(dtype)
    grad_out = torch.randn(*shape, 128, dtype=dtype, device=dev)

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
    if max_num_segments is None or via_kernel:
      kernel = tokamax_gated_delta.TokamaxGatedDeltaKernel(
          tokamax_gated_delta.TokamaxGatedDeltaKernel.Config(
              min_log_decay=min_log_decay, max_num_segments=max_num_segments
          )
      )
      # The kernels, not the parent's XLA fallback, must run these inputs.
      with mock.patch.object(
          gated_delta.TpuGatedDeltaKernel,
          "recurrence",
          side_effect=AssertionError("fell back to the XLA recurrence"),
      ):
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
    self.assertEqual(actual[0].dtype, dtype)
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
    # row 0's second document (the Mosaic kernels need 64-token multiples
    # without document ids, so other lengths use the XLA recurrence).
    first_boundary = doc_lens[0][0]
    leaky = run(
        lambda *xs: tokamax_gated_delta.tokamax_gated_delta(
            *xs, min_log_decay=min_log_decay
        )
        if seq_len % 64 == 0
        else gated_delta.chunked_gated_delta(*xs),
        g.clamp_min(min_log_decay),
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

  def test_init_fails_fast_without_kda_on_supported_tpu(self):
    """Tests the build-time check for a TPU >= v6 without Tokamax's KDA op."""
    missing = (None, ImportError("No module named 'tokamax.ops'"))
    with (
        mock.patch.object(
            tokamax_gated_delta, "supports_current_tpu", return_value=True
        ),
        mock.patch.object(
            tokamax_gated_delta, "_import_kda", return_value=missing
        ),
    ):
      with self.assertRaisesRegex(ImportError, r"tokamax>=0\.0\.17"):
        tokamax_gated_delta.TokamaxGatedDeltaKernel.Config().build()
    # Hosts that never run the Mosaic kernels (CPU, TPU < v6) build fine and
    # fall back to the XLA recurrence.
    with (
        mock.patch.object(
            tokamax_gated_delta, "supports_current_tpu", return_value=False
        ),
        mock.patch.object(
            tokamax_gated_delta, "_import_kda", return_value=missing
        ),
    ):
      tokamax_gated_delta.TokamaxGatedDeltaKernel.Config().build()

  @parameterized.named_parameters(
      dict(testcase_name="_dense", with_segments=False),
      dict(testcase_name="_segments", with_segments=True),
  )
  def test_no_grad_forward_skips_residuals(self, with_segments: bool):
    """Tests that no_grad runs the forward-only op, with the same output."""
    self._require_mosaic_kda()
    torch.manual_seed(0)
    dev = self.accelerator_device
    shape = (2, 256, 4)
    q, k, v = (
        torch.randn(*shape, 128, dtype=torch.bfloat16, device=dev)
        for _ in range(3)
    )
    g = -F.softplus(2.0 * torch.randn(*shape, device=dev))
    beta = torch.sigmoid(torch.randn(*shape, device=dev)).to(torch.bfloat16)
    kwargs = {}
    if with_segments:
      positions = torch.cat([torch.arange(n) for n in (100, 156)] * 2)
      kwargs = dict(
          segment_ids=segment_ids_from_positions(positions)
          .reshape(2, 256)
          .to(dev),
          max_num_segments=4,
      )
    inputs = [x.requires_grad_() for x in (q, k, v, g, beta)]
    with_grad = tokamax_gated_delta.tokamax_gated_delta(*inputs, **kwargs)
    res_op = (
        "_gated_delta_segments_fwd_res"
        if with_segments
        else "_gated_delta_fwd_res"
    )
    with (
        torch.no_grad(),
        mock.patch.object(
            tokamax_gated_delta,
            res_op,
            side_effect=AssertionError("materialized the backward's residuals"),
        ),
    ):
      without_grad = tokamax_gated_delta.tokamax_gated_delta(*inputs, **kwargs)
    self.assertFalse(without_grad.requires_grad)
    torch.testing.assert_close(
        without_grad.float().cpu(),
        with_grad.detach().float().cpu(),
        atol=1e-2,
        rtol=1e-2,
    )

  def test_kda_fwd_ops_match_the_selective_ac_set(self):
    """Tests that ``TPUSelectiveAC.save_kda_residuals`` saves these ops."""
    self.assertEqual(
        set(tokamax_gated_delta.KDA_FWD_OPS),
        activation_checkpoint._KDA_FWD_OPS,  # pylint: disable=protected-access
    )

  @parameterized.named_parameters(
      dict(testcase_name="_bf16", dtype=torch.bfloat16, min_log_decay=-5.0),
      dict(testcase_name="_fp32", dtype=torch.float32, min_log_decay=-5.0),
  )
  def test_decay_clamp_deviation_at_init(
      self, dtype: torch.dtype, min_log_decay: float
  ):
    """Measures the -5 decay clamp on a real-init 35B-A3B GatedDeltaNet layer.

    Three arms run the same initialized layer (``init_states``, so A_log =
    log(U(0.01, 16)) and the decays reach about -70) on the same inputs: the
    XLA recurrence (unclamped: the reference), the XLA recurrence on clamped
    decays (the clamp alone), and the Tokamax kernels (the clamp and the
    kernels' numerics). Logs each parameter's gradient deviation.
    """
    self._require_mosaic_kda()
    batch, seq_len = 2, 2048
    torch.manual_seed(0)
    dev = self.accelerator_device
    base_cfg = qwen3_5_tpu.model_registry("35B-A3B", seq_len=seq_len).model
    xla_cfg = copy.deepcopy(base_cfg.layers[0].delta_net)
    tok_cfg = copy.deepcopy(xla_cfg)
    tok_cfg.inner_gated_delta_net.kernel = (
        tokamax_gated_delta.TokamaxGatedDeltaKernel.Config(
            min_log_decay=min_log_decay
        )
    )
    reference = xla_cfg.build()
    reference.init_states(buffer_device=torch.device("cpu"))
    state = copy.deepcopy(reference.state_dict())
    modules = {"xla": reference, "xla_clamped": xla_cfg.build()}
    modules["tokamax"] = tok_cfg.build()
    for name, mod in modules.items():
      if name != "xla":
        mod.load_state_dict(state)
      mod.to(device=dev, dtype=dtype)
      # A_log and dt_bias stay fp32, as Qwen3.5's init keeps them.
      mod.A_log.data = mod.A_log.data.float()
      mod.dt_bias.data = mod.dt_bias.data.float()

    decays = []
    xla_kernel = modules["xla"].inner_gated_delta_net.kernel
    clamped_kernel = modules["xla_clamped"].inner_gated_delta_net.kernel
    parent_recurrence = gated_delta.TpuGatedDeltaKernel.recurrence

    def record(
        q_BSHK,
        k_BSHK,
        v_BSHV,
        g_BSH,
        beta_BSH,
        segment_ids_BS=None,
    ):
      decays.append(g_BSH.detach().float().cpu())
      return parent_recurrence(
          xla_kernel, q_BSHK, k_BSHK, v_BSHV, g_BSH, beta_BSH, segment_ids_BS
      )

    positions = torch.arange(seq_len).repeat(batch).to(dev)
    x_base = torch.randn(
        batch * seq_len,
        xla_cfg.in_proj_q.in_features,
        dtype=dtype,
        device=dev,
    )
    grad_out = torch.randn_like(x_base)

    def run_module(mod):
      x_in = x_base.detach().clone().requires_grad_()
      with set_unflatten_seq_len(seq_len):
        out = mod(x_in, positions=positions)
      out.backward(grad_out)
      grads = {n: p.grad for n, p in mod.named_parameters()}
      return {"out": out, "dx": x_in.grad, **grads}

    with (
        mock.patch.object(
            xla_kernel, "recurrence", autospec=True, side_effect=record
        ),
        _clamp_decays_ctx(clamped_kernel, min_log_decay),
    ):
      results = {name: run_module(mod) for name, mod in modules.items()}
    g = decays[0]
    clamped_fraction = float((g < min_log_decay).float().mean())
    logging.info(
        "35B GatedDeltaNet at init: %.1f%% of decays below %.1f (min %.1f,"
        " median %.2f)",
        100 * clamped_fraction,
        min_log_decay,
        float(g.min()),
        float(g.median()),
    )

    def cosine(a, b):
      return float(
          F.cosine_similarity(
              a.float().flatten().cpu(), b.float().flatten().cpu(), dim=0
          )
      )

    deviation = {}
    for name, ref in results["xla"].items():
      tok = results["tokamax"][name]
      clamped = results["xla_clamped"][name]
      self.assertIsNotNone(tok, msg=name)
      self.assertTrue(torch.isfinite(tok).all().item(), msg=name)
      deviation[name] = (
          _relative_l2_error(tok, ref),
          _relative_l2_error(clamped, ref),
          _relative_l2_error(tok, clamped),
          cosine(tok, ref),
      )
      logging.info(
          "[%s, min_log_decay %.1f] %-18s rel L2 vs unclamped XLA: tokamax"
          " %.2e, clamped XLA %.2e; tokamax vs clamped XLA %.2e;"
          " cosine(tokamax, XLA) %.5f",
          dtype,
          min_log_decay,
          name,
          *deviation[name],
      )

    self.assertGreater(clamped_fraction, 0.05)
    # The decay parameters' gradients sum per-token decay gradients of random
    # sign over all 4096 tokens, which mostly cancel, so a small systematic
    # error in the per-token decay gradient shows up large relative to them.
    # Measured on TPU7x at -5 (71.5% of the decays clamped): the clamp alone
    # moves A_log / dt_bias / in_proj_a by 4.5-6.3% (rel. L2) and everything
    # else by <= 1.6%. In fp32 the kernels match the clamped XLA recurrence to
    # <= 6.1e-3 everywhere; in bf16 they deviate from it by 65% / 55% / 12%
    # (cosine 0.85 / 0.89 / 0.99) on A_log / dt_bias / in_proj_a and by
    # <= 9.3e-3 elsewhere.
    # TODO: Add an fp32 clamped-XLA arm as the reference for the bf16 run, to
    # tell whether the bf16 A_log / dt_bias / in_proj_a gap comes from the
    # kernels or from the bf16 XLA recurrence.
    decay_params = ("A_log", "dt_bias", "in_proj_a.weight")
    for name, (vs_xla, clamp_only, vs_clamped, cos) in deviation.items():
      if name in decay_params:
        self.assertLess(clamp_only, 1e-1, msg=name)
        self.assertGreater(cos, 0.8, msg=name)
      else:
        self.assertLess(clamp_only, 2e-2, msg=name)
        self.assertLess(vs_xla, 3e-2, msg=name)
      if dtype == torch.float32 or name not in decay_params:
        self.assertLess(vs_clamped, 2e-2, msg=name)

  def test_debug_model_losses_match_xla(self):
    """Compares a few training steps of the debug model, Tokamax vs XLA.

    The Qwen3.5 debug model (6 GatedDeltaNet + 2 attention layers, K = V = 128)
    from its real initialization, bf16, AdamW (lr 1e-3) on one fixed batch of
    4 x 512 tokens; the arms differ only in the GatedDeltaNet recurrence.
    """
    self._require_mosaic_kda()
    min_log_decay = tokamax_gated_delta.DEFAULT_MIN_LOG_DECAY
    batch, seq_len, steps, vocab_size = 4, 512, 10, 1024
    dev = self.accelerator_device
    config = qwen3_5_tpu.model_registry("debugmodel", seq_len=seq_len).model
    config.vocab_size = vocab_size
    config.tok_embeddings.num_embeddings = vocab_size
    config.lm_head.out_features = vocab_size
    torch.manual_seed(0)
    reference = config.build()
    reference.init_states(buffer_device=torch.device("cpu"))
    state = copy.deepcopy(reference.state_dict())
    tok_config = copy.deepcopy(config)
    for layer in tok_config.layers:
      if layer.delta_net is not None:
        layer.delta_net.inner_gated_delta_net.kernel = (
            tokamax_gated_delta.TokamaxGatedDeltaKernel.Config(
                min_log_decay=min_log_decay
            )
        )

    gen = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, vocab_size, (batch, seq_len + 1), generator=gen)
    inputs = tokens[:, :-1].reshape(-1).to(dev)
    labels = tokens[:, 1:].reshape(-1).to(dev)
    positions = torch.arange(seq_len).repeat(batch).to(dev)

    def train(cfg, clamp_xla: bool = False) -> list[float]:
      model = cfg.build()
      model.load_state_dict(state)
      model.to(device=dev, dtype=torch.bfloat16)
      for name, param in model.named_parameters():
        if "A_log" in name or "dt_bias" in name:
          param.data = param.data.float()
      if cfg.enable_weight_tying:
        model.tok_embeddings.weight = model.lm_head.weight
      with contextlib.ExitStack() as stack:
        if clamp_xla:
          for module in model.modules():
            if isinstance(module, gated_delta.TpuGatedDeltaKernel):
              stack.enter_context(_clamp_decays_ctx(module, min_log_decay))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(steps):
          optimizer.zero_grad()
          with set_unflatten_seq_len(seq_len):
            logits = model(inputs, positions=positions)
          loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), labels)
          loss.backward()
          optimizer.step()
          losses.append(float(loss.detach().cpu()))
        return losses

    losses = {
        "xla": train(config),
        "xla_clamped": train(config, clamp_xla=True),
        "tokamax": train(tok_config),
    }
    for step, (ref, clamped, tok) in enumerate(
        zip(losses["xla"], losses["xla_clamped"], losses["tokamax"]), 1
    ):
      logging.info(
          "debug model step %d loss: XLA %.5f, clamped XLA %.5f (%.2e"
          " relative), Tokamax %.5f (%.2e relative)",
          step,
          ref,
          clamped,
          abs(clamped - ref) / ref,
          tok,
          abs(tok - ref) / ref,
      )
    self.assertLess(losses["xla"][-1], losses["xla"][0])
    for ref, tok in zip(losses["xla"], losses["tokamax"]):
      # Measured on TPU7x: <= 3.8e-4 relative over 10 steps (the clamp alone:
      # <= 1.0e-4).
      self.assertLess(abs(tok - ref), 2e-3 * ref)


if __name__ == "__main__":
  absltest.main()
