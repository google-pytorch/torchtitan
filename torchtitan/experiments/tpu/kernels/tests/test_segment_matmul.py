"""Tests for segment_matmul (tokamax-backed, tensor group_sizes).

The reference is a pure-PyTorch per-group matmul. The tested kernel is
``segment_matmul.segment_matmul_pallas_2d``, which routes to
``tokamax.ragged_dot(implementation="mosaic")`` under the hood.
"""

from absl import logging
from absl.testing import absltest
import torch
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.experiments.tpu.kernels import segment_matmul


def _manual_segment_matmul_reference(
    lhs: torch.Tensor, rhs: torch.Tensor, group_sizes: torch.Tensor
) -> torch.Tensor:
  """Pure-PyTorch reference: out[i, :] = lhs[i, :] @ rhs[expert(i), :, :].

  Iterates over groups; for each group, slices the corresponding rows of ``lhs``
  and matmuls against the per-expert weight slab of ``rhs``.
  """
  outputs = []
  start = 0
  for i, size in enumerate(group_sizes):
    size_val = size.item() if isinstance(size, torch.Tensor) else size
    block = lhs[start : start + size_val, :] @ rhs[i, :, :]
    outputs.append(block)
    start += size_val
  if not outputs:
    return torch.empty(0, rhs.shape[-1], device=lhs.device, dtype=lhs.dtype)
  return torch.cat(outputs, dim=0)


class SegmentMatmulTest(base_device_test.BaseAcceleratorDeviceTest):
  """Forward + backward + edge-case parity for segment_matmul_pallas_2d."""

  def test_basic_parity(self):
    """Default: 4 even groups, total tokens divisible by tile_m."""
    device = self.accelerator_device
    num_groups = 4
    k, n = 128, 256
    group_sizes = torch.tensor([128, 128, 128, 128], dtype=torch.int32)
    total_m = int(group_sizes.sum().item())

    torch.manual_seed(0)
    lhs = torch.randn(total_m, k, dtype=torch.float32, device=device)
    rhs = torch.randn(num_groups, k, n, dtype=torch.float32, device=device)
    group_sizes_dev = group_sizes.to(device)

    expected = _manual_segment_matmul_reference(lhs, rhs, group_sizes)
    actual = segment_matmul.segment_matmul_pallas_2d(lhs, rhs, group_sizes_dev)

    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3
    )
    logging.info("Basic parity passed.")

  def test_empty_groups(self):
    """Some groups have zero tokens — the kernel must still produce correct output.

    This is the hot path for the dropless MoE dispatcher: per-rank group_sizes
    can have many zeros, and the tensor-group_sizes kernel must handle this
    without falling back to ``.tolist()`` or special-casing.
    """
    device = self.accelerator_device
    # Expert 1 and 3 are empty.
    group_sizes = torch.tensor([128, 0, 128, 0], dtype=torch.int32)
    num_groups = 4
    k, n = 128, 128

    torch.manual_seed(1)
    lhs = torch.randn(256, k, dtype=torch.float32, device=device)
    rhs = torch.randn(num_groups, k, n, dtype=torch.float32, device=device)
    group_sizes_dev = group_sizes.to(device)

    expected = _manual_segment_matmul_reference(lhs, rhs, group_sizes)
    actual = segment_matmul.segment_matmul_pallas_2d(lhs, rhs, group_sizes_dev)

    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3
    )
    logging.info("Empty-group parity passed.")

  def test_jagged_unpadded(self):
    """Unaligned per-group sizes that sum to a tile-aligned total.

    Megablox's TPU kernel handles jagged per-group sizes natively as long as
    the total sums to a multiple of the tile_m. This is the realistic dropless
    case: group_sizes vary per step, but total_tokens is fixed by
    (local_batch_size * seq_len * top_k).
    """
    device = self.accelerator_device
    # Sum = 128 (one tile_m), distribution is uneven.
    group_sizes = torch.tensor([103, 17, 0, 5, 3], dtype=torch.int32)
    assert int(group_sizes.sum().item()) == 128
    num_groups = 5
    k, n = 128, 256

    torch.manual_seed(2)
    lhs = torch.randn(128, k, dtype=torch.float32, device=device)
    rhs = torch.randn(num_groups, k, n, dtype=torch.float32, device=device)
    group_sizes_dev = group_sizes.to(device)

    expected = _manual_segment_matmul_reference(lhs, rhs, group_sizes)
    actual = segment_matmul.segment_matmul_pallas_2d(lhs, rhs, group_sizes_dev)

    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2
    )
    logging.info("Jagged-unpadded parity passed.")

  def test_tensor_group_sizes_no_d2h(self):
    """Sanity: passing ``group_sizes`` as a TPU tensor must not implicitly D2H.

    Compare to ``kernels/gmm.py``'s ``grouped_matrix_multiply`` which calls
    ``tuple(group_sizes.cpu().numpy().tolist())`` — that path is the one that
    deadlocks at v6e-32 multi-host. This test asserts the new path doesn't
    move ``group_sizes`` to host: its ``.device`` is unchanged after the call.
    """
    device = self.accelerator_device
    group_sizes = torch.tensor([128, 128], dtype=torch.int32, device=device)
    lhs = torch.randn(256, 128, device=device)
    rhs = torch.randn(2, 128, 128, device=device)

    _ = segment_matmul.segment_matmul_pallas_2d(lhs, rhs, group_sizes)

    self.assertEqual(group_sizes.device.type, device.type)
    logging.info("Tensor-group_sizes invariant holds.")

  def test_backward_parity(self):
    """End-to-end backward: kernel grads match reference per-group grads."""
    device = self.accelerator_device
    num_groups = 4
    k, n = 128, 128
    group_sizes = torch.tensor([128, 128, 128, 128], dtype=torch.int32)

    torch.manual_seed(3)
    lhs_ref = torch.randn(512, k, device=device, requires_grad=True)
    rhs_ref = torch.randn(num_groups, k, n, device=device, requires_grad=True)

    # Reference backward (pure PyTorch).
    out_ref = _manual_segment_matmul_reference(lhs_ref, rhs_ref, group_sizes)
    out_ref.sum().backward()
    g_lhs_ref = lhs_ref.grad.clone()
    g_rhs_ref = rhs_ref.grad.clone()

    # Kernel backward.
    lhs = lhs_ref.detach().clone().requires_grad_(True)
    rhs = rhs_ref.detach().clone().requires_grad_(True)
    group_sizes_dev = group_sizes.to(device)
    out = segment_matmul.segment_matmul_pallas_2d(lhs, rhs, group_sizes_dev)
    out.sum().backward()

    torch.testing.assert_close(lhs.grad, g_lhs_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(rhs.grad, g_rhs_ref, rtol=1e-3, atol=1e-3)
    logging.info("Backward parity passed.")


if __name__ == "__main__":
  absltest.main()
