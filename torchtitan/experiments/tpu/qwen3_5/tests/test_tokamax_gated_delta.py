"""Tests Tokamax's Mosaic KDA GatedDeltaNet kernels against the XLA path."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.qwen3_5 import gated_delta
from torchtitan.models.common.attention import segment_ids_from_positions

try:
  from torchtitan.experiments.tpu.qwen3_5 import tokamax_gated_delta
except ImportError as e:
  # No released Tokamax wheel (up to 0.1.0) has the KDA op.
  tokamax_gated_delta = None
  _TOKAMAX_IMPORT_ERROR = str(e)


def _relative_l2_error(test: torch.Tensor, reference: torch.Tensor) -> float:
  test, reference = test.float().cpu(), reference.float().cpu()
  return float((test - reference).norm() / reference.norm().clamp_min(1e-12))


class TokamaxGatedDeltaTest(base_device_test.BaseAcceleratorDeviceTest):
  """``tokamax_gated_delta`` vs ``chunked_gated_delta`` on one TPU chip."""

  @parameterized.named_parameters(
      dict(testcase_name="_small", batch=2, seq_len=256, heads=4),
      # The 35B-A3B GatedDeltaNet shape (32 value heads x 128) and its
      # initialization's decays: A_log = log(U(0.01, 16)), dt_bias = 1 and
      # g = -exp(A_log) * softplus(a + dt_bias), down to about -34, so many
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
      self, batch: int, seq_len: int, heads: int, init_gates: bool = False
  ):
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("The Mosaic KDA kernels need a TPU.")
    if tokamax_gated_delta is None:
      self.skipTest(_TOKAMAX_IMPORT_ERROR)

    min_log_decay = -5.0
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
    if tokamax_gated_delta is None:
      self.skipTest(_TOKAMAX_IMPORT_ERROR)
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
    if tokamax_gated_delta is None:
      self.skipTest(_TOKAMAX_IMPORT_ERROR)
    kernel = tokamax_gated_delta.TokamaxGatedDeltaKernel.Config().build()
    devices = [torch.device("cpu")]
    if self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
      # 100-token rows without document ids: not a multiple of 64.
      devices.append(self.accelerator_device)
    torch.manual_seed(0)
    for dev in devices:
      with self.subTest(device=dev.type):
        shape = (2, 100, 4)
        q, k = (torch.randn(*shape, 128, device=dev) for _ in range(2))
        v = torch.randn(*shape, 128, device=dev)
        g = -F.softplus(torch.randn(*shape, device=dev))
        beta = torch.sigmoid(torch.randn(*shape, device=dev))
        with mock.patch.object(
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
      # Default budget (`max_num_segments=None` on `TokamaxGatedDeltaKernel`):
      # keeps at least `MIN_NUM_SEGMENTS` (4) documents apart even on a
      # 512-token row (`512 // TOKENS_PER_SEGMENT == 1`).
      dict(testcase_name="_kernel_default", max_num_segments=None),
      # Fewer segments than documents: the last documents of row 0 share one
      # recurrence, as tokamax_segment_ids merges them.
      dict(testcase_name="_capped", max_num_segments=3),
  )
  def test_tokamax_matches_xla_with_packed_documents(
      self, max_num_segments: int | None
  ):
    """Tests Tokamax's segment_ids path against the XLA segment masks."""
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("The Mosaic KDA kernels need a TPU.")
    if tokamax_gated_delta is None:
      self.skipTest(_TOKAMAX_IMPORT_ERROR)

    min_log_decay = -5.0
    torch.manual_seed(0)
    dev = self.accelerator_device
    # Row 0 packs documents starting mid-chunk and at a chunk start (128), a
    # 1-token document and documents spanning chunks; row 1 is one document.
    doc_lens = ((100, 28, 1, 200, 183), (512,))
    batch, seq_len, heads = len(doc_lens), 512, 4
    positions = torch.stack(
        [torch.cat([torch.arange(n) for n in lens]) for lens in doc_lens]
    )
    segment_ids_BS = segment_ids_from_positions(positions).to(dev)
    effective_segments = (
        max_num_segments or tokamax_gated_delta.MIN_NUM_SEGMENTS
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
      actual = run(
          lambda *xs: tokamax_gated_delta.tokamax_gated_delta(
              *xs,
              min_log_decay=min_log_decay,
              segment_ids=segment_ids,
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
    leaky = run(
        lambda *xs: tokamax_gated_delta.tokamax_gated_delta(
            *xs, min_log_decay=min_log_decay
        ),
        g,
    )
    self.assertGreater(
        _relative_l2_error(leaky[0][0, 100:104], expected[0][0, 100:104]),
        5e-2,
    )


if __name__ == "__main__":
  absltest.main()
