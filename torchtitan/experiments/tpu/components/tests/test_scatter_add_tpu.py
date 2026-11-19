# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-accelerator (TPU / CPU) tests for deterministic_scatter_add row-gather backward."""

from absl.testing import absltest
from absl.testing import parameterized
from functorch.compile import aot_function, make_boxed_func
import torch
from torch import nn

import torchtitan.experiments.tpu  # noqa: F401 - triggers package init
from torchtitan.experiments.tpu.components.scatter_add import (
    install_tpu_scatter_add_row_gather,
    uninstall_tpu_scatter_add_row_gather,
)
from torchtitan.experiments.tpu.testing.base_device_test import (
    BaseAcceleratorDeviceTest,
)
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
from torchtitan.ops.scatter_add import deterministic_scatter_add


class _TinyMoEBlock(nn.Module):
  """Minimal MoE layer exercising LocalTokenDispatcher.dispatch + combine."""

  def __init__(self, dim: int, num_experts: int, top_k: int):
    super().__init__()
    self.num_experts = num_experts
    self.top_k = top_k
    self.router = nn.Linear(dim, num_experts, bias=False)
    self.up = nn.Linear(dim, dim * 2, bias=False)
    self.down = nn.Linear(dim * 2, dim, bias=False)
    self.dispatcher = LocalTokenDispatcher.Config(
        num_experts=num_experts, top_k=top_k
    ).build()

  def forward(self, x_td: torch.Tensor) -> torch.Tensor:
    logits = self.router(x_td)
    scores, expert_ids = torch.topk(
        torch.softmax(logits.float(), dim=-1), k=self.top_k, dim=-1
    )
    counts = torch.bincount(
        expert_ids.view(-1), minlength=self.num_experts
    )
    routed_in, _, meta = self.dispatcher.dispatch(
        x_td, scores, expert_ids, counts
    )
    routed_out = self.down(torch.nn.functional.silu(self.up(routed_in)))
    return x_td + self.dispatcher.combine(routed_out, meta, x_td)


class ScatterAddRowGatherDeviceTest(BaseAcceleratorDeviceTest):

  def setUp(self):
    super().setUp()
    install_tpu_scatter_add_row_gather()

  def tearDown(self):
    install_tpu_scatter_add_row_gather()
    super().tearDown()

  def test_save_on_cpu_storage_bytes_on_accelerator(self):
    device = self.accelerator_device
    gen = torch.Generator().manual_seed(2718)
    num_tokens, num_routed, dim = 64, 256, 128
    out_dev = torch.zeros(num_tokens, dim, dtype=torch.float32, device=device)
    src_dev = torch.randn(
        num_routed, dim, dtype=torch.float32, generator=gen
    ).to(device)
    grad_dev = torch.randn(
        num_tokens, dim, dtype=torch.float32, generator=gen
    ).to(device)
    idx_1d = torch.randint(
        0, num_tokens, (num_routed,), dtype=torch.int64, generator=gen
    ).to(device)
    index_dev = idx_1d.reshape(-1, 1).expand(-1, dim)

    def _measure_cpu_offloaded_saved_bytes(*, installed: bool):
      if installed:
        install_tpu_scatter_add_row_gather()
      else:
        uninstall_tpu_scatter_add_row_gather()
      packed_meta: list[tuple[torch.Size, int]] = []

      # Mirror torch.autograd.graph.save_on_cpu()'s pack/unpack behavior
      # (`tensor.to("cpu", non_blocking=...)`) while inspecting storage bytes:
      # on accelerator devices, `.to("cpu")` materializes an expanded `(N, D)`
      # stride-(1, 0) view into a dense `N * D * 8`-byte CPU buffer when
      # uninstalled, vs. a compact `N * 8`-byte 1D buffer when installed.
      def _pack_to_cpu(t: torch.Tensor) -> tuple[torch.device, torch.Tensor]:
        cpu_t = t.to("cpu")
        packed_meta.append((cpu_t.shape, cpu_t.untyped_storage().nbytes()))
        return (t.device, cpu_t)

      def _unpack_from_cpu(packed: tuple[torch.device, torch.Tensor]) -> torch.Tensor:
        orig_dev, cpu_t = packed
        return cpu_t.to(orig_dev)

      src_in = src_dev.clone().requires_grad_(True)
      with torch.autograd.graph.saved_tensors_hooks(
          _pack_to_cpu, _unpack_from_cpu
      ):
        y = deterministic_scatter_add(out_dev.clone(), index_dev, src_in)
      y.backward(grad_dev)
      assert src_in.grad is not None
      return packed_meta, src_in.grad.detach().cpu()

    ref_meta, ref_grad = _measure_cpu_offloaded_saved_bytes(installed=False)
    opt_meta, opt_grad = _measure_cpu_offloaded_saved_bytes(installed=True)

    self.assertEqual(opt_meta, [(torch.Size([num_routed]), num_routed * 8)])
    if device.type != "cpu":
      self.assertEqual(
          ref_meta,
          [(torch.Size([num_routed, dim]), num_routed * dim * 8)],
      )
    self.assertTrue(torch.equal(opt_grad, ref_grad))

    # Also exercise stock torch.autograd.graph.save_on_cpu() end-to-end on device.
    src_stock = src_dev.clone().requires_grad_(True)
    with torch.autograd.graph.save_on_cpu():
      y_stock = deterministic_scatter_add(
          out_dev.clone(), index_dev, src_stock
      )
    y_stock.backward(grad_dev)
    assert src_stock.grad is not None
    self.assertTrue(torch.equal(src_stock.grad.detach().cpu(), ref_grad))

  @parameterized.parameters(
      (torch.float32, 256),
      (torch.bfloat16, 256),
      (torch.bfloat16, 1),
  )
  def test_device_bitwise_parity(self, dtype: torch.dtype, dim: int):
    device = self.accelerator_device
    gen = torch.Generator().manual_seed(314 + dim)
    num_tokens, num_routed = 128, 512
    out_cpu = torch.zeros(num_tokens, dim, dtype=dtype)
    src_cpu = torch.randn(num_routed, dim, dtype=dtype, generator=gen)
    grad_cpu = torch.randn(num_tokens, dim, dtype=dtype, generator=gen)
    idx_cpu = torch.randint(
        0, num_tokens, (num_routed,), dtype=torch.int64, generator=gen
    )

    out_dev = out_cpu.to(device)
    grad_dev = grad_cpu.to(device)
    index_dev = idx_cpu.to(device).reshape(-1, 1).expand(-1, dim)

    uninstall_tpu_scatter_add_row_gather()
    ref_src = src_cpu.to(device).requires_grad_(True)
    ref_y = deterministic_scatter_add(out_dev.clone(), index_dev, ref_src)
    ref_y.backward(grad_dev)

    install_tpu_scatter_add_row_gather()
    opt_src = src_cpu.to(device).requires_grad_(True)
    opt_y = deterministic_scatter_add(out_dev.clone(), index_dev, opt_src)
    opt_y.backward(grad_dev)

    self.assertTrue(torch.equal(opt_y.cpu(), ref_y.cpu()))
    assert opt_src.grad is not None and ref_src.grad is not None
    self.assertTrue(torch.equal(opt_src.grad.cpu(), ref_src.grad.cpu()))

  def test_compiled_fullgraph_row_gather_on_device(self):
    device = self.accelerator_device
    dtype = torch.bfloat16 if device.type == "tpu" else torch.float32

    bw_targets: list[str] = []

    def _fn(out, idx_1d, src):
      return deterministic_scatter_add(
          out, idx_1d.reshape(-1, 1).expand(-1, src.shape[1]), src
      )

    # First verify AOTAutograd backward FX graph emits aten.index_select.default.
    fx_checked = torch.compile(
        _fn,
        fullgraph=True,
        backend=lambda gm, _ex: aot_function(
            gm,
            fw_compiler=lambda fgm, _: make_boxed_func(fgm.forward),
            bw_compiler=lambda bgm, _: (
                bw_targets.extend(
                    str(n.target)
                    for n in bgm.graph.nodes
                    if n.op == "call_function"
                )
                or make_boxed_func(bgm.forward)
            ),
        ),
    )

    gen = torch.Generator().manual_seed(777)
    out = torch.zeros(64, 128, dtype=dtype, device=device)
    idx_1d = torch.randint(0, 64, (256,), dtype=torch.int64, generator=gen).to(
        device
    )
    src_base = torch.randn(256, 128, dtype=dtype, generator=gen).to(device)
    grad_out = torch.randn(64, 128, dtype=dtype, generator=gen).to(device)

    install_tpu_scatter_add_row_gather()
    src_fx = src_base.clone().requires_grad_(True)
    fx_checked(out.clone(), idx_1d, src_fx).backward(grad_out)
    self.assertIn("aten.index_select.default", bw_targets)
    self.assertNotIn("aten.gather.default", bw_targets)

    # Next compile with the native accelerator backend ("tpu" on TPU).
    backend = "tpu" if device.type == "tpu" else "aot_eager"
    compiled_native = torch.compile(_fn, fullgraph=True, backend=backend)
    src_comp = src_base.clone().requires_grad_(True)
    y_comp = compiled_native(out.clone(), idx_1d, src_comp)
    y_comp.backward(grad_out)

    uninstall_tpu_scatter_add_row_gather()
    src_ref = src_base.clone().requires_grad_(True)
    y_ref = _fn(out.clone(), idx_1d, src_ref)
    y_ref.backward(grad_out)
    install_tpu_scatter_add_row_gather()

    self.assertTrue(torch.equal(y_comp.cpu(), y_ref.cpu()))
    assert src_comp.grad is not None and src_ref.grad is not None
    self.assertTrue(torch.equal(src_comp.grad.cpu(), src_ref.grad.cpu()))

  def test_moe_12_step_training_convergence_and_step1_bitwise_parity(self):
    device = self.accelerator_device
    dtype = torch.bfloat16 if device.type == "tpu" else torch.float32
    dim, num_experts, top_k = 64, 8, 2

    torch.manual_seed(1234)
    init_model = _TinyMoEBlock(dim=dim, num_experts=num_experts, top_k=top_k).to(
        dtype=dtype
    )
    init_state = {k: v.clone() for k, v in init_model.state_dict().items()}

    x_fixed = torch.randn(32, dim, dtype=dtype).to(device)
    target_fixed = torch.randn(32, dim, dtype=dtype).to(device)

    # Check step-1 loss and grad_norm bit-identity against upstream gather backward.
    uninstall_tpu_scatter_add_row_gather()
    ref_model = _TinyMoEBlock(dim=dim, num_experts=num_experts, top_k=top_k).to(
        device=device, dtype=dtype
    )
    ref_model.load_state_dict(init_state)
    ref_loss = torch.nn.functional.mse_loss(ref_model(x_fixed), target_fixed)
    ref_loss.backward()
    ref_norm = torch.nn.utils.clip_grad_norm_(ref_model.parameters(), 1.0)

    install_tpu_scatter_add_row_gather()
    model = _TinyMoEBlock(dim=dim, num_experts=num_experts, top_k=top_k).to(
        device=device, dtype=dtype
    )
    model.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses: list[float] = []
    for step in range(12):
      opt.zero_grad()
      loss = torch.nn.functional.mse_loss(model(x_fixed), target_fixed)
      loss.backward()
      grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      if step == 0:
        # Step-0 forward loss equality is a sanity check on identical initial
        # state; grad_norm equality below is the actual VJP bit-identity check.
        self.assertTrue(torch.equal(loss.detach().cpu(), ref_loss.detach().cpu()))
        self.assertTrue(
            torch.equal(grad_norm.detach().cpu(), ref_norm.detach().cpu())
        )
      loss_val = float(loss.detach().cpu().item())
      norm_val = float(grad_norm.detach().cpu().item())
      self.assertTrue(
          torch.isfinite(torch.tensor([loss_val, norm_val])).all().item(),
          f"Non-finite loss/grad_norm at step {step}: {loss_val}, {norm_val}",
      )
      losses.append(loss_val)
      opt.step()

    # Verify loss decreases past warmup (steps 2..11).
    self.assertLess(losses[-1], losses[2])
    self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
  absltest.main()
