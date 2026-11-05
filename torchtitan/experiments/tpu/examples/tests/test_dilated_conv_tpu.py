from absl.testing import absltest
import torch
import torch.nn.functional as F
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.examples.dilated_conv import DilatedGroupCausalConv


def naive_dilated_causal_conv(
    x, d_model, group_size, kernel_size, conv_weights, conv_biases
):
  # Reference implementation without contiguous padding view optimisations
  B, T, D = x.shape
  h = x.transpose(1, 2)
  ng = d_model // group_size
  outs = []
  for i in range(ng):
    pad_len = (kernel_size - 1) * (2**i)
    # Pad specifically for this group, dynamically in the loop (slow on hardware)
    sliced = h[:, i * group_size : (i + 1) * group_size, :]
    padded_slice = F.pad(sliced, (pad_len, 0))
    # Apply the explicit functional conv so we can share weights directly
    out = F.conv1d(
        padded_slice,
        conv_weights[i],
        conv_biases[i],
        dilation=2**i,
        groups=group_size,
    )
    outs.append(out)
  return torch.cat(outs, dim=1).transpose(1, 2)


class TestTPUDilatedConv(base_device_test.BaseAcceleratorDeviceTest):

  def test_causal_dilated_conv_parity(self):
    B, T, D = 2, 128, 64
    GS = 4
    K = 4
    # 1. Initialize our Optimized Module on CPU
    model_cpu, model_device = self._setup_module_cpu_device(
        DilatedGroupCausalConv,
        lambda m: None,
        d_model=D,
        group_size=GS,
        kernel_size=K,
    )

    # 2. Extract weights to pass into our naive reference python function
    weights = [c.weight for c in model_cpu.convs]
    biases = [c.bias for c in model_cpu.convs]

    # 3. Create matched input tensors mapped appropriately
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (B, T, D), requires_grad=True
    )

    # 4. Generate Hardware (Optimized) Output
    out_device = model_device(x_device)

    # 5. Generate Reference (Naive) Output
    out_ref = naive_dilated_causal_conv(
        x_cpu,
        d_model=D,
        group_size=GS,
        kernel_size=K,
        conv_weights=weights,
        conv_biases=biases,
    )

    # 6. Verify outputs match closely!
    torch.testing.assert_close(out_device.cpu(), out_ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  absltest.main()
