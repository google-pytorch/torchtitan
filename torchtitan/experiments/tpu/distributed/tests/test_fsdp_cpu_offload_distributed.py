# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-rank distributed tests for TPU FSDP2 CPU offload optimizations."""

import os
from typing import Any

from absl.testing import absltest
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
import torch.multiprocessing as mp

from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.tpu.distributed import distributed
from torchtitan.experiments.tpu.distributed.fsdp_cpu_offload import (
    get_cpu_offload_stats,
    install_tpu_fsdp_cpu_offload,
    materialize_host_grads,
    reset_cpu_offload_stats,
    uninstall_tpu_fsdp_cpu_offload,
    wire_trainer_cpu_offload,
)
from torchtitan.experiments.tpu.testing import base_distributed_device_test


class _DistributedOffloadNet(nn.Module):
  """Multi-block network including uneven dim-0 shapes across world_size."""

  def __init__(self, world_size: int, dim: int = 64):
    super().__init__()
    odd_hidden = world_size * 7 + 3
    self.in_proj = nn.Linear(dim, dim, bias=False)
    self.layers = nn.ModuleList(
        [
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, odd_hidden, bias=True),
                nn.GELU(),
                nn.Linear(odd_hidden, dim, bias=False),
            )
            for _ in range(3)
        ]
    )
    self.out_proj = nn.Linear(dim, dim, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.in_proj(x)
    for layer in self.layers:
      h = h + layer(h)
    return self.out_proj(h)


def _init_dual_backend_pg(
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
  if dist.is_initialized():
    return
  os.environ.setdefault("TEST_TARGET", "1")
  if device.type == "tpu":
    backend = "tpu:tpu_dist,cpu:gloo"
  elif device.type == "cuda":
    backend = "cuda:nccl,cpu:gloo"
  else:
    backend = "gloo"
  dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _run_distributed_fsdp_cpu_offload_parity(
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
  _init_dual_backend_pg(device, rank, world_size)
  mesh = DeviceMesh(device.type, list(range(world_size)))

  torch.manual_seed(9001)
  base_model = _DistributedOffloadNet(world_size=world_size)
  init_state = {k: v.clone() for k, v in base_model.state_dict().items()}
  num_params = sum(1 for _ in base_model.parameters())

  num_steps = 10
  grad_accum = 2
  # Per-rank deterministic microbatches repeated across steps to verify both
  # multi-rank reduce-scatter parity and monotone multi-step convergence.
  gen = torch.Generator().manual_seed(5000 + rank * 37)
  microbatches = [
      (
          torch.randn(8, 64, dtype=torch.bfloat16, generator=gen).to(device),
          torch.randn(8, 64, dtype=torch.float32, generator=gen).to(device),
      )
      for _ in range(grad_accum)
  ]

  def _sync() -> None:
    if device.type != "cpu" and torch.accelerator.is_available():
      torch.accelerator.synchronize()

  def _shard_model(*, nested_layers: bool = True) -> _DistributedOffloadNet:
    model = _DistributedOffloadNet(world_size=world_size)
    model.load_state_dict(init_state)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    offload_policy = CPUOffloadPolicy(pin_memory=False)
    if nested_layers:
      for layer in model.layers:
        fully_shard(
            layer,
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

  def _run_synced_single_group_step(*, use_optimized: bool) -> dict[str, Any]:
    if use_optimized:
      install_tpu_fsdp_cpu_offload(
          enable_param_cache=True,
          enable_device_grad_accum=True,
      )
    else:
      uninstall_tpu_fsdp_cpu_offload()
    reset_cpu_offload_stats()

    model = _shard_model(nested_layers=False)
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
    grad_norm = dist_utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
        foreach=True,
    )
    norm_cpu = grad_norm.detach().cpu().clone()
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

  def _run(
      *, use_optimized: bool
  ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if use_optimized:
      install_tpu_fsdp_cpu_offload(
          enable_param_cache=True,
          enable_device_grad_accum=True,
      )
    else:
      uninstall_tpu_fsdp_cpu_offload()
    reset_cpu_offload_stats()

    model = _shard_model(nested_layers=True)
    for p in model.parameters():
      assert p.to_local().device.type == "cpu"
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    if use_optimized:
      wire_trainer_cpu_offload(opt)

    first_fwd_loss: torch.Tensor | None = None
    records: list[dict[str, Any]] = []
    for step_idx in range(num_steps):
      opt.zero_grad(set_to_none=True)
      mb_losses: list[torch.Tensor] = []
      for mb_idx, (x, target) in enumerate(microbatches):
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred.float(), target) / grad_accum
        if step_idx == 0 and mb_idx == 0:
          first_fwd_loss = loss.detach().cpu()
        loss.backward()
        mb_losses.append(loss.detach().cpu())

      if use_optimized:
        materialize_host_grads()
      grads = {
          k: p.grad.to_local().detach().cpu().clone()
          for k, p in model.named_parameters()
          if p.grad is not None
      }
      grad_norm = dist_utils.clip_grad_norm_(
          model.parameters(),
          max_norm=1.0,
          foreach=True,
      )
      norm_cpu = grad_norm.detach().cpu().clone()
      opt.step()
      params = {
          k: p.to_local().detach().cpu().clone()
          for k, p in model.named_parameters()
      }
      records.append(
          {
              "mb_losses": mb_losses,
              "grad_norm": norm_cpu,
              "grads": grads,
              "params": params,
          }
      )
    assert first_fwd_loss is not None
    return first_fwd_loss, records

  try:
    # 1) Single-group synchronized step: bit-for-bit exact across all ranks
    # (including uneven dim-0 shards) on both CPU and TPU.
    sync_ref = _run_synced_single_group_step(use_optimized=False)
    dist.barrier()
    sync_opt = _run_synced_single_group_step(use_optimized=True)
    for mb in range(grad_accum):
      assert torch.equal(
          sync_opt["mb_losses"][mb], sync_ref["mb_losses"][mb]
      ), f"Rank {rank} synced mb {mb} loss mismatch"
    assert torch.equal(
        sync_opt["grad_norm"], sync_ref["grad_norm"]
    ), f"Rank {rank} synced grad_norm mismatch"
    for name in sync_ref["grads"]:
      assert torch.equal(
          sync_opt["grads"][name], sync_ref["grads"][name]
      ), f"Rank {rank} synced grad mismatch on {name}"
    for name in sync_ref["params"]:
      assert torch.equal(
          sync_opt["params"][name], sync_ref["params"][name]
      ), f"Rank {rank} synced param mismatch on {name}"
    dist.barrier()

    # 2) Full multi-step nested FSDP2 training loop with gradient accumulation.
    ref_fwd0, ref_records = _run(use_optimized=False)
    dist.barrier()
    opt_fwd0, opt_records = _run(use_optimized=True)
    stats = get_cpu_offload_stats()

    assert torch.equal(
        opt_fwd0, ref_fwd0
    ), f"Rank {rank} step 0 mb 0 pre-backward forward loss mismatch"
    assert stats.param_cache_misses == num_params * num_steps, (
        f"Rank {rank}: expected {num_params * num_steps} misses, got"
        f" {stats.param_cache_misses}"
    )
    assert stats.param_cache_hits > 0
    assert stats.d2h_sync_calls == num_steps

    step_losses: list[float] = []
    for step in range(num_steps):
      ref_r = ref_records[step]
      opt_r = opt_records[step]
      if device.type == "cpu":
        for mb in range(grad_accum):
          assert torch.equal(
              opt_r["mb_losses"][mb], ref_r["mb_losses"][mb]
          ), f"Rank {rank} step {step} mb {mb} loss mismatch"
        assert torch.equal(
            opt_r["grad_norm"], ref_r["grad_norm"]
        ), f"Rank {rank} step {step} grad_norm mismatch"
        for name in ref_r["grads"]:
          assert torch.equal(
              opt_r["grads"][name], ref_r["grads"][name]
          ), f"Rank {rank} step {step} grad mismatch on {name}"
        for name in ref_r["params"]:
          assert torch.equal(
              opt_r["params"][name], ref_r["params"][name]
          ), f"Rank {rank} step {step} param mismatch on {name}"
      else:
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

      step_loss = sum(float(t.item()) for t in opt_r["mb_losses"])
      norm_val = float(opt_r["grad_norm"].item())
      assert torch.isfinite(
          torch.tensor([step_loss, norm_val])
      ).all().item(), f"Rank {rank} non-finite loss/grad_norm at step {step}"
      step_losses.append(step_loss)

    assert (
        step_losses[-1] < step_losses[0]
    ), f"Rank {rank} loss did not decrease: {step_losses}"
    dist.barrier()
  finally:
    uninstall_tpu_fsdp_cpu_offload()
    reset_cpu_offload_stats()
    if dist.is_initialized():
      dist.destroy_process_group()


class FSDPCPUOffloadDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):

  def test_distributed_fsdp2_cpu_offload_bitwise_parity(self) -> None:
    num_devices = self.num_devices
    if self.accelerator_device_type.value == "cpu":
      num_devices = 4
    elif num_devices < 2:
      self.skipTest("Distributed FSDP2 CPU offload test needs >= 2 devices")
    distributed.run_distributed(
        num_devices,
        self.accelerator_device_type,
        _run_distributed_fsdp_cpu_offload_parity,
        run_init_process_group=False,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
