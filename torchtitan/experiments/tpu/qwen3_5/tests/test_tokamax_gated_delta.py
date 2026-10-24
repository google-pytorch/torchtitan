"""Tests Tokamax's Mosaic KDA GatedDeltaNet kernels against the XLA path."""

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
    for name, a_t, e_t in zip(
        ("out", "dq", "dk", "dv", "dg", "dbeta"), actual, expected, strict=True
    ):
      if name == "dg":
        # Clamped decays have no gradient through the clamp in either path.
        mask = g > min_log_decay
        a_t, e_t = a_t * mask, e_t * mask
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

  @parameterized.named_parameters(
      dict(testcase_name="_all_documents", max_num_segments=8),
      # Fewer segments than documents: the last documents of row 0 share one
      # recurrence, as tokamax_segment_ids merges them.
      dict(testcase_name="_capped", max_num_segments=3),
  )
  def test_tokamax_matches_xla_with_packed_documents(
      self, max_num_segments: int
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
    segment_ids = tokamax_gated_delta.tokamax_segment_ids(
        segment_ids_from_positions(positions), max_num_segments
    ).to(dev)
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
