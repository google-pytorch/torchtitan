# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for the TPU row-gather backward of deterministic_scatter_add."""

from absl.testing import absltest
from absl.testing import parameterized
from functorch.compile import aot_function, make_boxed_func
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import torchtitan.experiments.tpu  # noqa: F401 - triggers package init
from torchtitan.experiments.tpu.components.scatter_add import (
    install_tpu_scatter_add_row_gather,
    is_tpu_scatter_add_row_gather_installed,
    uninstall_tpu_scatter_add_row_gather,
)
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
from torchtitan.ops.scatter_add import deterministic_scatter_add


class _OpRecorderMode(TorchDispatchMode):
  """Records dispatched ATen ops during backward execution."""

  def __init__(self) -> None:
    super().__init__()
    self.ops: list[str] = []

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    self.ops.append(str(func))
    return func(*args, **(kwargs or {}))


class ScatterAddRowGatherCpuTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    install_tpu_scatter_add_row_gather()

  def tearDown(self):
    install_tpu_scatter_add_row_gather()
    super().tearDown()

  def test_installed_on_tpu_package_import_and_idempotent(self):
    self.assertTrue(is_tpu_scatter_add_row_gather_installed())
    install_tpu_scatter_add_row_gather()
    self.assertTrue(is_tpu_scatter_add_row_gather_installed())
    uninstall_tpu_scatter_add_row_gather()
    self.assertFalse(is_tpu_scatter_add_row_gather_installed())
    uninstall_tpu_scatter_add_row_gather()
    self.assertFalse(is_tpu_scatter_add_row_gather_installed())
    install_tpu_scatter_add_row_gather()
    self.assertTrue(is_tpu_scatter_add_row_gather_installed())

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16, torch.float16],
      dim=[0, 1, 128],
      num_routed=[0, 1, 64],
      out_requires_grad=[False, True],
  )
  def test_row_broadcast_bitwise_parity_against_upstream(
      self,
      dtype: torch.dtype,
      dim: int,
      num_routed: int,
      out_requires_grad: bool,
  ):
    gen = torch.Generator().manual_seed(1000 + dim * 17 + num_routed)
    num_tokens = 16
    out_base = torch.randn(num_tokens, dim, dtype=dtype, generator=gen)
    src_base = torch.randn(num_routed, dim, dtype=dtype, generator=gen)
    grad_out = torch.randn(num_tokens, dim, dtype=dtype, generator=gen)
    if num_routed > 0:
      idx_1d = torch.randint(
          0, num_tokens, (num_routed,), dtype=torch.int64, generator=gen
      )
      # Force duplicate destination rows when possible.
      if num_routed >= 4:
        idx_1d[1] = idx_1d[0]
        idx_1d[3] = idx_1d[0]
    else:
      idx_1d = torch.empty((0,), dtype=torch.int64)
    index_2d = idx_1d.reshape(-1, 1).expand(-1, dim)

    uninstall_tpu_scatter_add_row_gather()
    ref_out_in = out_base.clone().requires_grad_(out_requires_grad)
    ref_src_in = src_base.clone().requires_grad_(True)
    ref_y = deterministic_scatter_add(ref_out_in, index_2d, ref_src_in)
    ref_y.backward(grad_out)

    install_tpu_scatter_add_row_gather()
    opt_out_in = out_base.clone().requires_grad_(out_requires_grad)
    opt_src_in = src_base.clone().requires_grad_(True)
    opt_y = deterministic_scatter_add(opt_out_in, index_2d, opt_src_in)
    with _OpRecorderMode() as recorder:
      opt_y.backward(grad_out)

    self.assertTrue(torch.equal(opt_y, ref_y))
    assert opt_src_in.grad is not None and ref_src_in.grad is not None
    self.assertTrue(torch.equal(opt_src_in.grad, ref_src_in.grad))
    if out_requires_grad:
      assert opt_out_in.grad is not None and ref_out_in.grad is not None
      self.assertTrue(torch.equal(opt_out_in.grad, ref_out_in.grad))
    if dim > 0:
      self.assertTrue(any("aten.index_select" in op for op in recorder.ops))
      self.assertFalse(any("aten.gather" in op for op in recorder.ops))

  def test_non_contiguous_grad_output_bitwise_parity(self):
    gen = torch.Generator().manual_seed(42)
    num_tokens, num_routed, dim = 12, 30, 64
    out_base = torch.zeros(num_tokens, dim, dtype=torch.bfloat16)
    src_base = torch.randn(
        num_routed, dim, dtype=torch.bfloat16, generator=gen
    )
    idx_1d = torch.randint(
        0, num_tokens, (num_routed,), dtype=torch.int64, generator=gen
    )
    index_2d = idx_1d.reshape(-1, 1).expand(-1, dim)
    # Construct a non-contiguous upstream gradient view (stride != (dim, 1)).
    grad_full = torch.randn(
        dim, num_tokens * 2, dtype=torch.bfloat16, generator=gen
    )
    grad_noncontig = grad_full[:, ::2].transpose(0, 1)
    self.assertFalse(grad_noncontig.is_contiguous())

    uninstall_tpu_scatter_add_row_gather()
    ref_src = src_base.clone().requires_grad_(True)
    deterministic_scatter_add(out_base.clone(), index_2d, ref_src).backward(
        grad_noncontig
    )

    install_tpu_scatter_add_row_gather()
    opt_src = src_base.clone().requires_grad_(True)
    deterministic_scatter_add(out_base.clone(), index_2d, opt_src).backward(
        grad_noncontig
    )

    assert opt_src.grad is not None and ref_src.grad is not None
    self.assertTrue(torch.equal(opt_src.grad, ref_src.grad))

  def test_save_on_cpu_saves_compact_1d_indices(self):
    gen = torch.Generator().manual_seed(7)
    num_tokens, num_routed, dim = 32, 128, 256
    out = torch.zeros(num_tokens, dim, dtype=torch.float32)
    src = torch.randn(num_routed, dim, dtype=torch.float32, generator=gen)
    grad_out = torch.randn(num_tokens, dim, dtype=torch.float32, generator=gen)
    idx_1d = torch.randint(
        0, num_tokens, (num_routed,), dtype=torch.int64, generator=gen
    )
    index_2d = idx_1d.reshape(-1, 1).expand(-1, dim)

    uninstall_tpu_scatter_add_row_gather()
    ref_src = src.clone().requires_grad_(True)
    deterministic_scatter_add(out.clone(), index_2d, ref_src).backward(grad_out)

    install_tpu_scatter_add_row_gather()
    saved_shapes_and_nbytes: list[tuple[torch.Size, int]] = []
    orig_pack = torch.autograd.graph.save_on_cpu

    with orig_pack():
      pass

    def _pack_hook(tensor: torch.Tensor) -> torch.Tensor:
      cpu_copy = tensor.detach().to("cpu", copy=True)
      saved_shapes_and_nbytes.append(
          (cpu_copy.shape, cpu_copy.untyped_storage().nbytes())
      )
      return cpu_copy

    opt_src = src.clone().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(_pack_hook, lambda x: x):
      y_hooks = deterministic_scatter_add(out.clone(), index_2d, opt_src)
    y_hooks.backward(grad_out)

    self.assertEqual(
        saved_shapes_and_nbytes,
        [(torch.Size([num_routed]), num_routed * 8)],
    )
    assert opt_src.grad is not None and ref_src.grad is not None
    self.assertTrue(torch.equal(opt_src.grad, ref_src.grad))

    opt_src_cpu_offload = src.clone().requires_grad_(True)
    with torch.autograd.graph.save_on_cpu():
      y_cpu = deterministic_scatter_add(
          out.clone(), index_2d, opt_src_cpu_offload
      )
    y_cpu.backward(grad_out)
    assert opt_src_cpu_offload.grad is not None
    self.assertTrue(torch.equal(opt_src_cpu_offload.grad, ref_src.grad))

  def test_non_row_broadcast_2d_index_falls_back_to_gather(self):
    gen = torch.Generator().manual_seed(99)
    num_tokens, num_routed, dim = 16, 24, 32
    out = torch.zeros(num_tokens, dim, dtype=torch.float32)
    src = torch.randn(num_routed, dim, dtype=torch.float32, generator=gen)
    grad_out = torch.randn(num_tokens, dim, dtype=torch.float32, generator=gen)
    index_non_broadcast = torch.randint(
        0, num_tokens, (num_routed, dim), dtype=torch.int64, generator=gen
    )
    self.assertNotEqual(index_non_broadcast.stride(1), 0)

    uninstall_tpu_scatter_add_row_gather()
    ref_src = src.clone().requires_grad_(True)
    ref_y = deterministic_scatter_add(
        out.clone(), index_non_broadcast, ref_src
    )
    ref_y.backward(grad_out)

    install_tpu_scatter_add_row_gather()
    opt_src = src.clone().requires_grad_(True)
    opt_y = deterministic_scatter_add(
        out.clone(), index_non_broadcast, opt_src
    )
    with _OpRecorderMode() as recorder:
      opt_y.backward(grad_out)

    self.assertTrue(torch.equal(opt_y, ref_y))
    assert opt_src.grad is not None and ref_src.grad is not None
    self.assertTrue(torch.equal(opt_src.grad, ref_src.grad))
    self.assertTrue(any("aten.gather" in op for op in recorder.ops))
    self.assertFalse(any("aten.index_select" in op for op in recorder.ops))

  def test_torch_compile_fullgraph_emits_index_select_in_backward_fx(self):
    bw_targets_broadcast: list[str] = []
    bw_targets_dense: list[str] = []

    def _broadcast_fn(out, idx_1d, src):
      return deterministic_scatter_add(
          out, idx_1d.reshape(-1, 1).expand(-1, src.shape[1]), src
      )

    def _dense_fn(out, idx_2d, src):
      return deterministic_scatter_add(out, idx_2d, src)

    compiled_broadcast = torch.compile(
        _broadcast_fn,
        fullgraph=True,
        backend=lambda gm, ex: aot_function(
            gm,
            fw_compiler=lambda fgm, _: make_boxed_func(fgm.forward),
            bw_compiler=lambda bgm, _: (
                bw_targets_broadcast.extend(
                    str(n.target)
                    for n in bgm.graph.nodes
                    if n.op == "call_function"
                )
                or make_boxed_func(bgm.forward)
            ),
        ),
    )
    compiled_dense = torch.compile(
        _dense_fn,
        fullgraph=True,
        backend=lambda gm, ex: aot_function(
            gm,
            fw_compiler=lambda fgm, _: make_boxed_func(fgm.forward),
            bw_compiler=lambda bgm, _: (
                bw_targets_dense.extend(
                    str(n.target)
                    for n in bgm.graph.nodes
                    if n.op == "call_function"
                )
                or make_boxed_func(bgm.forward)
            ),
        ),
    )

    gen = torch.Generator().manual_seed(123)
    out = torch.zeros(16, 64, dtype=torch.bfloat16)
    idx_1d = torch.randint(0, 16, (40,), dtype=torch.int64, generator=gen)
    idx_2d = torch.randint(0, 16, (40, 64), dtype=torch.int64, generator=gen)
    src = torch.randn(40, 64, dtype=torch.bfloat16, generator=gen)
    grad_out = torch.randn(16, 64, dtype=torch.bfloat16, generator=gen)

    src_compile = src.clone().requires_grad_(True)
    y_compile = compiled_broadcast(out.clone(), idx_1d, src_compile)
    y_compile.backward(grad_out)

    uninstall_tpu_scatter_add_row_gather()
    src_ref = src.clone().requires_grad_(True)
    y_ref = _broadcast_fn(out.clone(), idx_1d, src_ref)
    y_ref.backward(grad_out)
    install_tpu_scatter_add_row_gather()

    self.assertTrue(torch.equal(y_compile, y_ref))
    assert src_compile.grad is not None and src_ref.grad is not None
    self.assertTrue(torch.equal(src_compile.grad, src_ref.grad))
    self.assertIn("aten.index_select.default", bw_targets_broadcast)
    self.assertNotIn("aten.gather.default", bw_targets_broadcast)

    src_dense = src.clone().requires_grad_(True)
    y_dense = compiled_dense(out.clone(), idx_2d, src_dense)
    y_dense.backward(grad_out)
    self.assertIn("aten.gather.default", bw_targets_dense)
    self.assertNotIn("aten.index_select.default", bw_targets_dense)

  def test_local_token_dispatcher_moe_combine_bitwise_parity(self):
    gen = torch.Generator().manual_seed(2026)
    num_tokens, dim, num_experts, top_k = 24, 64, 8, 2
    dispatcher = LocalTokenDispatcher.Config(
        num_experts=num_experts, top_k=top_k
    ).build()

    x_base = torch.randn(
        num_tokens, dim, dtype=torch.bfloat16, generator=gen
    )
    scores_base = torch.rand(
        num_tokens, top_k, dtype=torch.float32, generator=gen
    )
    expert_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int64, generator=gen
    )
    counts = torch.bincount(expert_ids.view(-1), minlength=num_experts)
    w_proj = torch.randn(dim, dim, dtype=torch.bfloat16, generator=gen)
    grad_out = torch.randn(
        num_tokens, dim, dtype=torch.bfloat16, generator=gen
    )

    def _run_once():
      x = x_base.clone().requires_grad_(True)
      scores = scores_base.clone().requires_grad_(True)
      w = w_proj.clone().requires_grad_(True)
      routed_in, _, meta = dispatcher.dispatch(x, scores, expert_ids, counts)
      routed_out = routed_in @ w
      combined = dispatcher.combine(routed_out, meta, x)
      combined.backward(grad_out)
      return combined, x.grad, scores.grad, w.grad

    uninstall_tpu_scatter_add_row_gather()
    ref_out, ref_gx, ref_gs, ref_gw = _run_once()

    install_tpu_scatter_add_row_gather()
    opt_out, opt_gx, opt_gs, opt_gw = _run_once()

    self.assertTrue(torch.equal(opt_out, ref_out))
    self.assertTrue(torch.equal(opt_gx, ref_gx))
    self.assertTrue(torch.equal(opt_gs, ref_gs))
    self.assertTrue(torch.equal(opt_gw, ref_gw))


if __name__ == "__main__":
  absltest.main()
