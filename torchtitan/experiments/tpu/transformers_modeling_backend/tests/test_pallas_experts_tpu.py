"""Device tests for the HF backend's TPU MoE substitutions.

Checks that the default K/N-tiled kernels compile within scoped VMEM at real
expert widths, that batches whose routed-row count is not a tile_m multiple
are padded correctly, and that the swapped MoE gets the TPU token dispatcher.
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F

from torchtitan.experiments.tpu import base_device_test
from types import SimpleNamespace

from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.token_dispatcher import TpuAllToAllTokenDispatcher
from torchtitan.experiments.tpu.transformers_modeling_backend import model as hf_model
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher


def _reference(x, w1, w2, w3, group_sizes):
  """FP32 per-expert SwiGLU over native (E, K, N) weights."""
  outs, start = [], 0
  for e, size in enumerate(group_sizes.tolist()):
    xe = x[start:start + size]
    outs.append((F.silu(xe @ w1[e]) * (xe @ w3[e])) @ w2[e])
    start += size
  return torch.cat(outs)


class _ExpertsOnly:
  """Minimal stand-in for a converted MoE config holding one experts config."""

  def __init__(self, experts):
    self.experts = experts

  def traverse(self, cls):
    yield None, self.experts, None, None


def _rel_err(actual, expected):
  return ((actual.float() - expected).norm() / expected.norm()).item()


class PallasExpertsTest(base_device_test.BaseAcceleratorDeviceTest):

  @parameterized.named_parameters(
      # (dim, hidden, group sizes)
      ("qwen3_30b", 2048, 768, [0, 300, 100, 112]),
      ("mixtral_8x7b", 4096, 14336, [100, 156]),
      # 16 tokens x top-k 8 = 128 rows: below one 256-row tile.
      ("short_batch", 2048, 768, [16, 0, 40, 8, 24, 16, 8, 16]),
      ("small_dims", 256, 128, [100, 0, 28]),
  )
  def test_forward_backward(self, dim, hidden, group_sizes):
    device = self.accelerator_device
    cfg = TpuGroupedExperts.Config(dim=dim, hidden_dim=hidden, num_experts=len(group_sizes))
    hf_model._use_pallas_grouped_mm(_ExpertsOnly(cfg))
    experts = TpuGroupedExperts(cfg)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
      for name, k in (("w1_EDF", dim), ("w2_EFD", hidden), ("w3_EDF", dim)):
        param = getattr(experts, name)
        param.copy_(torch.randn(param.shape, generator=gen) / k**0.5)
    experts = experts.to(device=device, dtype=torch.bfloat16)

    rows = sum(group_sizes)
    x_cpu = torch.randn(rows, dim, generator=gen).bfloat16()
    dout_cpu = torch.randn(rows, dim, generator=gen).bfloat16()
    sizes = torch.tensor(group_sizes, dtype=torch.int32)

    x = x_cpu.to(device).requires_grad_()
    # Compiled: eager Pallas calls do not enforce the scoped VMEM limit.
    out = torch.compile(experts, backend="tpu")(x, sizes.to(device))
    params = (experts.w1_EDF, experts.w2_EFD, experts.w3_EDF)
    grads = torch.autograd.grad(out, (x, *params), dout_cpu.to(device))

    ref_inputs = [x_cpu.float().requires_grad_()] + [
        p.detach().cpu().float().requires_grad_() for p in params
    ]
    x_ref, w1, w2, w3 = ref_inputs
    ref = _reference(x_ref, w1, w2, w3, sizes)
    ref_grads = torch.autograd.grad(ref, ref_inputs, dout_cpu.float())

    self.assertEqual(tuple(out.shape), (rows, dim))
    self.assertLess(_rel_err(out.cpu(), ref.detach()), 1e-2)
    for name, g, rg in zip(("dx", "dw1", "dw2", "dw3"), grads, ref_grads):
      self.assertLess(_rel_err(g.cpu(), rg), 2e-2, name)


class TokenDispatcherTest(base_device_test.BaseAcceleratorDeviceTest):

  def test_swapped_moe_gets_tpu_dispatcher(self):
    # The standard dispatcher's combine backward is an element-wise gather of
    # rows x dim values, which XLA runs at a few percent of HBM bandwidth on
    # TPU; the TPU dispatcher's VJPs gather whole rows.
    cfg = make_routed_experts_config(
        dim=256, hidden_dim=128, num_experts=4, top_k=2, param_init={},
        comm_backend="standard",
    )
    layer = SimpleNamespace(_native_moe_config=cfg)
    model = SimpleNamespace(layers={"0": layer}, _tpu_moe_pallas=False)
    hf_model.substitute_tpu_experts(model)
    found = [type(c) for _, c, _, _ in
             layer._native_moe_config.traverse(AllToAllTokenDispatcher.Config)]
    self.assertEqual(found, [TpuAllToAllTokenDispatcher.Config])


if __name__ == "__main__":
  absltest.main()
