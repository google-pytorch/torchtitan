# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-accelerator (TPU / CPU) tests for FSDP2 CPU offload optimizations."""

import os
from typing import Any

from absl.testing import absltest
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)

from torchtitan.experiments.tpu.distributed import distributed_utils
from torchtitan.experiments.tpu.distributed.fsdp_cpu_offload import (
    get_cpu_offload_stats,
    install_tpu_fsdp_cpu_offload,
    materialize_host_grads,
    reset_cpu_offload_stats,
    uninstall_tpu_fsdp_cpu_offload,
    wire_trainer_cpu_offload,
)
from torchtitan.experiments.tpu.testing.base_device_test import (
    BaseAcceleratorDeviceTest,
)


def _ensure_single_rank_mesh(device: torch.device) -> DeviceMesh:
  os.environ.setdefault("TEST_TARGET", "1")
  distributed_utils.init_single_rank_process_group()
  return DeviceMesh(device.type, [0])


class _TinyTransformerStack(nn.Module):

  def __init__(self, dim: int = 64, num_layers: int = 3):
    super().__init__()
    self.embed = nn.Linear(dim, dim, bias=False)
    self.blocks = nn.ModuleList(
        [
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2, bias=False),
                nn.SiLU(),
                nn.Linear(dim * 2, dim, bias=False),
            )
            for _ in range(num_layers)
        ]
    )
    self.head = nn.Linear(dim, dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.embed(x)
    for block in self.blocks:
      h = h + block(h)
    return self.head(h)


def _shard_on_device(
    mesh: DeviceMesh,
    init_state: dict[str, torch.Tensor],
    *,
    nested_blocks: bool = True,
) -> _TinyTransformerStack:
  model = _TinyTransformerStack()
  model.load_state_dict(init_state)
  mp_policy = MixedPrecisionPolicy(
      param_dtype=torch.bfloat16,
      reduce_dtype=torch.float32,
  )
  offload_policy = CPUOffloadPolicy(pin_memory=False)
  if nested_blocks:
    for block in model.blocks:
      fully_shard(
          block,
          mesh=mesh,
          mp_policy=mp_policy,
          offload_policy=offload_policy,
          reshard_after_forward=True,
      )
  fully_shard(
      model,
      mesh=mesh,
      mp_policy=mp_policy,
      offload_policy=offload_policy,
      reshard_after_forward=True,
  )
  return model


class FSDPCPUOffloadAcceleratorTest(BaseAcceleratorDeviceTest):

  def tearDown(self):
    uninstall_tpu_fsdp_cpu_offload()
    reset_cpu_offload_stats()
    super().tearDown()

  def test_single_group_synchronized_step_bitwise_parity(self):
    """Verifies exact IEEE-754 bit-for-bit parity when HLO step boundaries align."""
    device = self.accelerator_device
    mesh = _ensure_single_rank_mesh(device)

    torch.manual_seed(314)
    base_model = _TinyTransformerStack()
    init_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    grad_accum = 3
    microbatches = [
        (
            torch.randn(16, 64, dtype=torch.bfloat16).to(device),
            torch.randn(16, 64, dtype=torch.float32).to(device),
        )
        for _ in range(grad_accum)
    ]

    def _sync() -> None:
      if device.type != "cpu" and torch.accelerator.is_available():
        torch.accelerator.synchronize()

    def _run_one_step(*, use_optimized: bool) -> dict[str, Any]:
      if use_optimized:
        install_tpu_fsdp_cpu_offload(
            enable_param_cache=True,
            enable_device_grad_accum=True,
        )
      else:
        uninstall_tpu_fsdp_cpu_offload()
      reset_cpu_offload_stats()

      model = _shard_on_device(mesh, init_state, nested_blocks=False)
      opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
      if use_optimized:
        wire_trainer_cpu_offload(opt)

      opt.zero_grad(set_to_none=True)
      mb_losses: list[torch.Tensor] = []
      for x, target in microbatches:
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred.float(), target) / grad_accum
        mb_losses.append(loss.detach().cpu())
        _sync()
        loss.backward()
        _sync()

      if use_optimized:
        materialize_host_grads()
      grads = {
          k: p.grad.to_local().detach().cpu().clone()
          for k, p in model.named_parameters()
          if p.grad is not None
      }
      grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      norm_cpu = (
          grad_norm.to_local().detach().cpu().clone()
          if hasattr(grad_norm, "to_local")
          else grad_norm.detach().cpu().clone()
      )
      opt.step()
      params = {
          k: p.to_local().detach().cpu().clone()
          for k, p in model.named_parameters()
      }
      return {
          "mb_losses": mb_losses,
          "grads": grads,
          "grad_norm": norm_cpu,
          "params": params,
      }

    ref = _run_one_step(use_optimized=False)
    opt = _run_one_step(use_optimized=True)
    for mb in range(grad_accum):
      self.assertTrue(torch.equal(opt["mb_losses"][mb], ref["mb_losses"][mb]))
    self.assertTrue(torch.equal(opt["grad_norm"], ref["grad_norm"]))
    for name in ref["grads"]:
      self.assertTrue(torch.equal(opt["grads"][name], ref["grads"][name]), name)
    for name in ref["params"]:
      self.assertTrue(
          torch.equal(opt["params"][name], ref["params"][name]), name
      )

  def test_12_step_3_microbatch_device_parity_and_convergence(self):
    device = self.accelerator_device
    mesh = _ensure_single_rank_mesh(device)

    torch.manual_seed(404)
    base_model = _TinyTransformerStack()
    init_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    num_params = sum(1 for _ in base_model.parameters())

    num_steps = 12
    grad_accum = 3
    # Use a fixed set of 3 microbatches repeated across steps so training loss
    # decreases monotonically while still exercising multi-microbatch accumulation.
    fixed_microbatches = [
        (
            torch.randn(16, 64, dtype=torch.bfloat16).to(device),
            torch.randn(16, 64, dtype=torch.float32).to(device),
        )
        for _ in range(grad_accum)
    ]

    def _run(*, use_optimized: bool) -> tuple[torch.Tensor, list[dict[str, Any]]]:
      if use_optimized:
        install_tpu_fsdp_cpu_offload(
            enable_param_cache=True,
            enable_device_grad_accum=True,
        )
      else:
        uninstall_tpu_fsdp_cpu_offload()
      reset_cpu_offload_stats()

      model = _shard_on_device(mesh, init_state, nested_blocks=True)
      for p in model.parameters():
        self.assertEqual(p.to_local().device.type, "cpu")
      opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
      if use_optimized:
        wire_trainer_cpu_offload(opt)

      first_fwd_loss: torch.Tensor | None = None
      records: list[dict[str, Any]] = []
      for step_idx in range(num_steps):
        opt.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), dtype=torch.float32)
        mb_losses: list[torch.Tensor] = []
        for mb_idx, (x, target) in enumerate(fixed_microbatches):
          pred = model(x)
          loss = torch.nn.functional.mse_loss(pred.float(), target) / grad_accum
          if step_idx == 0 and mb_idx == 0:
            first_fwd_loss = loss.detach().cpu()
          loss.backward()
          loss_cpu = loss.detach().cpu()
          mb_losses.append(loss_cpu)
          step_loss = step_loss + loss_cpu

        if use_optimized:
          materialize_host_grads()
        grads = {
            k: p.grad.to_local().detach().cpu().clone()
            for k, p in model.named_parameters()
            if p.grad is not None
        }
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        norm_cpu = (
            grad_norm.to_local().detach().cpu().clone()
            if hasattr(grad_norm, "to_local")
            else grad_norm.detach().cpu().clone()
        )
        opt.step()
        params = {
            k: p.to_local().detach().cpu().clone()
            for k, p in model.named_parameters()
        }
        records.append(
            {
                "step_loss": step_loss,
                "mb_losses": mb_losses,
                "grad_norm": norm_cpu,
                "grads": grads,
                "params": params,
            }
        )
      assert first_fwd_loss is not None
      return first_fwd_loss, records

    ref_fwd0, ref_records = _run(use_optimized=False)
    opt_fwd0, opt_records = _run(use_optimized=True)
    stats = get_cpu_offload_stats()

    self.assertTrue(torch.equal(opt_fwd0, ref_fwd0))
    self.assertEqual(stats.param_cache_misses, num_params * num_steps)
    self.assertGreater(stats.param_cache_hits, 0)
    self.assertEqual(stats.d2h_sync_calls, num_steps)

    losses: list[float] = []
    for step in range(num_steps):
      ref_r = ref_records[step]
      opt_r = opt_records[step]
      if device.type == "cpu":
        for mb in range(grad_accum):
          self.assertTrue(
              torch.equal(opt_r["mb_losses"][mb], ref_r["mb_losses"][mb]),
              f"mb loss mismatch at step {step} mb {mb}",
          )
        self.assertTrue(
            torch.equal(opt_r["grad_norm"], ref_r["grad_norm"]),
            f"grad_norm mismatch at step {step}",
        )
        for name in ref_r["grads"]:
          self.assertTrue(
              torch.equal(opt_r["grads"][name], ref_r["grads"][name]),
              f"grad mismatch for {name} at step {step}",
          )
        for name in ref_r["params"]:
          self.assertTrue(
              torch.equal(opt_r["params"][name], ref_r["params"][name]),
              f"param mismatch for {name} at step {step}",
          )
      else:
        # On LazyTensor/XLA backends (TPU), eliminating stock FSDP2's per-block
        # `.to("cpu")` D2H syncs fuses backward across FSDP blocks into a single
        # XLA HLO module instead of `num_blocks + 1` separate graphs.
        for mb in range(grad_accum):
          torch.testing.assert_close(
              opt_r["mb_losses"][mb],
              ref_r["mb_losses"][mb],
              rtol=2e-3,
              atol=2e-3,
          )
        torch.testing.assert_close(
            opt_r["grad_norm"],
            ref_r["grad_norm"],
            rtol=5e-3,
            atol=5e-3,
        )
        for name in ref_r["grads"]:
          torch.testing.assert_close(
              opt_r["grads"][name],
              ref_r["grads"][name],
              rtol=5e-3,
              atol=5e-3,
          )
        for name in ref_r["params"]:
          torch.testing.assert_close(
              opt_r["params"][name],
              ref_r["params"][name],
              rtol=5e-3,
              atol=5e-3,
          )

      loss_val = float(opt_r["step_loss"].item())
      norm_val = float(opt_r["grad_norm"].item())
      self.assertTrue(
          torch.isfinite(torch.tensor([loss_val, norm_val])).all().item()
      )
      losses.append(loss_val)

    self.assertLess(losses[-1], losses[0])
    self.assertLess(losses[-1], losses[2])


if __name__ == "__main__":
  absltest.main()
