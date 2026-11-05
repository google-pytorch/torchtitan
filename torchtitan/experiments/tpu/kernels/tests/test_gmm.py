"""Tests for grouped_matrix_multiply."""

from absl import logging
from absl.testing import absltest
import torch
import torch.nn.functional as F
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.experiments.tpu.kernels import gmm


def manual_gmm_reference(lhs, rhs, group_sizes):
  """PyTorch reference implementation for GMM."""
  outputs = []
  start = 0
  for i, size in enumerate(group_sizes):
    size_val = size.item() if isinstance(size, torch.Tensor) else size
    # We always slice, even if size is 0, to keep the 'start' pointer
    # moving correctly relative to the total_m logic.
    l = lhs[start : start + size_val, :]
    r = rhs[i, :, :]
    outputs.append(l @ r)
    start += size_val
  return (
      torch.cat(outputs, dim=0)
      if outputs
      else torch.empty(0, rhs.shape[-1], device=lhs.device, dtype=lhs.dtype)
  )


class GMMTest(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for grouped_matrix_multiply."""

  def test_manual_gmm_reference_parity(self):
    """Verifies that the TPU kernel output matches manual reference implementation."""
    device = self.accelerator_device
    num_groups = 8
    k, n = 128, 128
    tm = 128  # default tile size for M in JAX kernel

    # Generate random group sizes that sum to a multiple of tm=128
    group_sizes = torch.randint(0, 200, (num_groups,), dtype=torch.int32)
    total_m = group_sizes.sum().item()

    # Adjust to make total_m divisible by tm
    remainder = total_m % tm
    if remainder != 0:
      padding = tm - remainder
      group_sizes[0] += padding
      total_m += padding

    # Create tensors on CPU
    lhs = torch.randn(total_m, k, dtype=torch.float32)
    rhs = torch.randn(num_groups, k, n, dtype=torch.float32)

    # Move to TPU
    lhs_tpu = lhs.to(device)
    rhs_tpu = rhs.to(device)
    group_sizes_tpu = group_sizes.to(device)

    # Reference on TPU
    expected_out = manual_gmm_reference(lhs_tpu, rhs_tpu, group_sizes)
    # Kernel on TPU
    offs_tpu = torch.cumsum(group_sizes, dim=0).to(device=device)
    actual_out = gmm.grouped_matrix_multiply(lhs_tpu, rhs_tpu, offs_tpu)

    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=1e-5, atol=1e-5
    )
    logging.info("Parity test passed.")

  def test_torch_grouped_mm_parity(self):
    """Verifies the custom kernel output against F.grouped_mm."""

    device = self.accelerator_device
    num_groups = 4
    # Non-symmetric k and n so orientation errors crash with shape mismatch
    k, n = 64, 128

    group_sizes = torch.tensor([128, 0, 256, 128], dtype=torch.int32)
    total_m = group_sizes.sum().item()

    lhs_tpu = torch.randn(total_m, k, dtype=torch.float32, device=device)
    rhs_tpu = torch.randn(num_groups, k, n, dtype=torch.float32, device=device)

    offs = torch.cumsum(group_sizes, dim=0).to(dtype=torch.int32, device=device)
    actual_out = gmm.grouped_matrix_multiply(
        lhs_tpu, rhs_tpu, offs
    )

    # Pass rhs natively (no transpose).
    f_out = F.grouped_mm(lhs_tpu, rhs_tpu, offs=offs)

    # Check shapes
    self.assertEqual(actual_out.shape, f_out.shape)
    torch.testing.assert_close(actual_out, f_out, rtol=1e-3, atol=1e-3)
    logging.info("F.grouped_mm parity test passed.")

  def test_gradient(self):
    """Checks gradients numerically using torch.autograd.gradcheck."""
    device = self.accelerator_device
    # Keep it at the minimum aligned size (128)
    num_groups = 1
    k, n = 128, 128
    group_sizes = torch.tensor([128], dtype=torch.int32)
    total_m = 128

    lhs = torch.randn(
        total_m, k, dtype=torch.float32, device=device, requires_grad=True
    )
    rhs = torch.randn(
        num_groups, k, n, dtype=torch.float32, device=device, requires_grad=True
    )
    offs_tpu = torch.cumsum(group_sizes, dim=0).to(device)

    def func(l, r):
      return gmm.grouped_matrix_multiply(l, r, offs_tpu)

    # fast_mode=True is critical here; it uses the backward pass to check
    # instead of building the full massive numerical Jacobian matrix.
    torch.autograd.gradcheck(
        func, (lhs, rhs), eps=1e-3, atol=1e-2, rtol=1e-2, fast_mode=True
    )
    logging.info("Gradient test passed.")

  def test_gradient_manual(self):
    """Compares analytical gradients to those from manual reference implementation."""
    device = self.accelerator_device
    num_groups = 4
    k, n = 128, 128
    group_sizes = torch.tensor([128, 128, 128, 128], dtype=torch.int32)
    total_m = 512

    # Standard float32 path
    lhs = torch.randn(total_m, k, device=device, requires_grad=True)
    rhs = torch.randn(num_groups, k, n, device=device, requires_grad=True)
    group_sizes_tpu = group_sizes.to(device)

    # 1. Reference backward
    expected_out = manual_gmm_reference(lhs, rhs, group_sizes)
    # Use a dummy scalar loss to trigger backward
    loss_expected = expected_out.sum()
    loss_expected.backward()
    expected_grad_lhs = lhs.grad.clone()
    expected_grad_rhs = rhs.grad.clone()

    # 2. Kernel backward
    lhs.grad.zero_()
    rhs.grad.zero_()
    offs_tpu = torch.cumsum(group_sizes, dim=0).to(device)
    actual_out = gmm.grouped_matrix_multiply(lhs, rhs, offs_tpu)
    loss_actual = actual_out.sum()
    loss_actual.backward()

    # 3. Compare
    torch.testing.assert_close(
        lhs.grad, expected_grad_lhs, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        rhs.grad, expected_grad_rhs, rtol=1e-3, atol=1e-3
    )
    logging.info("Manual gradient comparison passed.")

  def test_empty_groups(self):
    """Ensures the kernel appropriately handles empty expert groups within a batch."""
    device = self.accelerator_device
    # Expert 1 has data, Expert 2 is empty, Expert 3 has data, Expert 4 is empty
    # Total M must still be multiple of 128 (256 here)
    group_sizes = torch.tensor([128, 0, 128, 0], dtype=torch.int32)
    num_groups = 4
    k, n = 128, 128

    lhs = torch.randn(256, k, device=device)
    rhs = torch.randn(num_groups, k, n, device=device)

    expected_out = manual_gmm_reference(lhs, rhs, group_sizes)
    offs_tpu = torch.cumsum(group_sizes, dim=0).to(device)
    actual_out = gmm.grouped_matrix_multiply(lhs, rhs, offs_tpu)

    torch.testing.assert_close(actual_out, expected_out)
    logging.info("Zero-size group test passed.")

  def test_jagged_unpadded_offsets(self):
    """Verifies that the Pallas kernel securely handles unaligned, jagged offsets natively.

    This test proves that the JAX Pallas Megablox engine mathematically bounds
    the jagged tensors perfectly without any injected dummy padding tokens
    between the experts. However, without dummy padding, kernel compilation time
    may blow up, therefore custom fill_indices kernel should be used in tandem
    with GMM.
    """
    device = self.accelerator_device
    logging.info(f"Running on device: {device}")

    # Construct completely jagged, unaligned tokens for 5 experts
    # Notice that none of these are aligned to 16, 32, etc. (which Triton would require)
    # The sum total must still be a multiple of the configured Pallas tile size (128).
    num_tokens_per_expert = torch.tensor(
        [103, 17, 0, 5, 3], dtype=torch.int32, device=device
    )
    offs = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    num_experts = 5
    total_tokens = int(num_tokens_per_expert.sum().item())
    assert total_tokens % 128 == 0, (
        f"Pallas requires total global sequence size {total_tokens} to be tiled"
        " natively!"
    )

    in_features = 128
    out_features = 256

    torch.manual_seed(42)
    x = torch.randn(
        total_tokens, in_features, dtype=torch.float32, device=device
    )
    w = torch.randn(
        num_experts,
        in_features,
        out_features,
        dtype=torch.float32,
        device=device,
    )

    # 1. Run custom Pallas TPU GMM kernel
    gmm_out = gmm.grouped_matrix_multiply(x, w, offs)

    # 2. Compute the exact theoretical output via PyTorch reference
    group_sizes = num_tokens_per_expert
    expected_out = manual_gmm_reference(x, w, group_sizes)

    # 3. Assert equality computationally
    torch.testing.assert_close(gmm_out, expected_out, atol=1e-2, rtol=1e-2)
    logging.info(
        "SUCCESS: Pallas JAX kernel successfully executed completely"
        " jagged/unpadded offsets!"
    )

if __name__ == "__main__":
  absltest.main()
