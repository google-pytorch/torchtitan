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
    verify_tpu_fsdp_cpu_offload_no_pin_memory,
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
  # Match dist_utils.init_distributed / init_single_rank_process_group setup;
  # disabling multithreaded autograd is required for FSDP2 hooks on torch_tpu.
  dist_utils.enable_fp32_matmul_emulation_with_bf16x9()
  torch.autograd.set_multithreading_enabled(False)
  if device.type == "tpu":
    backend = "tpu:tpu_dist,cpu:gloo"
  elif device.type == "cuda":
    backend = "cuda:nccl,cpu:gloo"
  else:
    backend = "gloo"
  dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _compute_cpu_math_reference(
    init_state: dict[str, torch.Tensor],
    world_size: int,
    num_steps: int,
    grad_accum: int,
    lr: float,
    eps: float,
) -> list[dict[str, Any]]:
  """Exact unsharded bf16-compute / fp32-master+reduce multi-rank reference."""
  rank_mbs: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
  for r in range(world_size):
    gen = torch.Generator().manual_seed(5000 + r * 37)
    rank_mbs.append(
        [
            (
                torch.randn(8, 64, dtype=torch.bfloat16, generator=gen),
                torch.randn(8, 64, dtype=torch.float32, generator=gen),
            )
            for _ in range(grad_accum)
        ]
    )

  master_model = _DistributedOffloadNet(world_size=world_size).float()
  master_model.load_state_dict(init_state)
  opt = torch.optim.AdamW(master_model.parameters(), lr=lr, eps=eps)
  bf16_model = _DistributedOffloadNet(world_size=world_size).bfloat16()

  records: list[dict[str, Any]] = []
  for _ in range(num_steps):
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
      for (_, mp_p), (_, bp_p) in zip(
          master_model.named_parameters(), bf16_model.named_parameters()
      ):
        bp_p.copy_(mp_p.to(torch.bfloat16))

    accum_grads = {
        k: torch.zeros_like(p, dtype=torch.float32)
        for k, p in master_model.named_parameters()
    }
    mb_losses_by_rank: list[list[torch.Tensor]] = [
        [] for _ in range(world_size)
    ]

    for mb_idx in range(grad_accum):
      for r in range(world_size):
        bf16_model.zero_grad(set_to_none=True)
        x_r, tgt_r = rank_mbs[r][mb_idx]
        pred_r = bf16_model(x_r)
        loss_r = (
            torch.nn.functional.mse_loss(pred_r.float(), tgt_r) / grad_accum
        )
        mb_losses_by_rank[r].append(loss_r.detach().clone())
        loss_r.backward()
        for k, bp_p in bf16_model.named_parameters():
          assert bp_p.grad is not None
          accum_grads[k].add_(bp_p.grad.float() / world_size)

    for k, mp_p in master_model.named_parameters():
      mp_p.grad = accum_grads[k].clone()

    full_grads = {k: v.clone() for k, v in accum_grads.items()}
    total_norm = torch.nn.utils.clip_grad_norm_(
        master_model.parameters(), max_norm=1.0, foreach=True
    )
    opt.step()
    full_params = {
        k: p.detach().clone() for k, p in master_model.named_parameters()
    }
    records.append(
        {
            "mb_losses_by_rank": mb_losses_by_rank,
            "grad_norm": total_norm.detach().clone(),
            "full_grads": full_grads,
            "full_params": full_params,
        }
    )
  return records


def _local_chunk_for_rank(
    full_tensor: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
  chunks = torch.chunk(full_tensor, world_size, dim=0)
  return chunks[rank] if rank < len(chunks) else full_tensor[:0]


def _run_distributed_fsdp_cpu_offload_parity(
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
  # Cap per-rank OpenMP threads so 8 spawned workers do not oversubscribe host CPUs.
  torch.set_num_threads(4)
  _init_dual_backend_pg(device, rank, world_size)
  mesh = DeviceMesh(device.type, list(range(world_size)))

  torch.manual_seed(9001)
  base_model = _DistributedOffloadNet(world_size=world_size)
  init_state = {k: v.clone() for k, v in base_model.state_dict().items()}
  num_params = sum(1 for _ in base_model.parameters())

  num_steps = 10
  grad_accum = 2
  # Use lr=5e-2 (25x larger than atol=2e-3) so any missed parameter update or
  # stale cached shard across steps diverges immediately by >> atol, while
  # eps=1e-2 avoids near-zero coordinate sign flips from 1-ulp bf16 fusion.
  lr = 5e-2
  eps = 1e-2
  gen = torch.Generator().manual_seed(5000 + rank * 37)
  microbatches = [
      (
          torch.randn(8, 64, dtype=torch.bfloat16, generator=gen).to(device),
          torch.randn(8, 64, dtype=torch.float32, generator=gen).to(device),
      )
      for _ in range(grad_accum)
  ]

  cpu_math_records = _compute_cpu_math_reference(
      init_state=init_state,
      world_size=world_size,
      num_steps=num_steps,
      grad_accum=grad_accum,
      lr=lr,
      eps=eps,
  )

  def _shard_model() -> _DistributedOffloadNet:
    model = _DistributedOffloadNet(world_size=world_size)
    model.load_state_dict(init_state)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    offload_policy = CPUOffloadPolicy(pin_memory=False)
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

  def _run(
      *,
      installed: bool,
      enable_param_cache: bool = False,
      enable_device_grad_accum: bool = False,
  ) -> list[dict[str, Any]]:
    if installed:
      install_tpu_fsdp_cpu_offload(
          enable_param_cache=enable_param_cache,
          enable_device_grad_accum=enable_device_grad_accum,
      )
    else:
      uninstall_tpu_fsdp_cpu_offload()
    reset_cpu_offload_stats()

    model = _shard_model()
    verify_tpu_fsdp_cpu_offload_no_pin_memory([model])
    for p in model.parameters():
      assert p.to_local().device.type == "cpu"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps)
    if installed:
      wire_trainer_cpu_offload(opt)

    records: list[dict[str, Any]] = []
    for _ in range(num_steps):
      opt.zero_grad(set_to_none=True)
      mb_losses: list[torch.Tensor] = []
      for x, target in microbatches:
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred.float(), target) / grad_accum
        loss.backward()
        mb_losses.append(loss.detach().cpu())

      if installed:
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
    if device.type != "cpu" and torch.accelerator.is_available():
      torch.accelerator.synchronize()
    return records

  try:
    # Stock unpatched FSDP2 uses `non_blocking=True` H2D on `all_gather_copy_in_stream`,
    # which races on multi-rank `torch_tpu`; run `installed=False` on CPU/Gloo to
    # assert bitwise identity against stock PyTorch, while on TPU compare both
    # default (`enable_param_cache=False, enable_device_grad_accum=False`) and
    # full opt-in (`True, True`) modes against `_compute_cpu_math_reference` and
    # against each other.
    ref_records: list[dict[str, Any]] | None = None
    if device.type == "cpu":
      ref_records = _run(installed=False)
      dist.barrier()

    # Default opt-in-off mode (TPUCPUOffloadPolicy + non_blocking=False H2D, cache & accum off).
    default_records = _run(
        installed=True,
        enable_param_cache=False,
        enable_device_grad_accum=False,
    )
    default_stats = get_cpu_offload_stats()
    assert default_stats.param_cache_misses == 0
    assert default_stats.param_cache_hits == 0
    assert default_stats.d2h_sync_calls == 0
    dist.barrier()

    # Full opt-in mode (both device param cache and device grad accum enabled).
    opt_records = _run(
        installed=True,
        enable_param_cache=True,
        enable_device_grad_accum=True,
    )
    stats = get_cpu_offload_stats()

    assert stats.param_cache_misses == num_params * num_steps, (
        f"Rank {rank}: expected {num_params * num_steps} misses, got"
        f" {stats.param_cache_misses}"
    )
    assert stats.param_cache_hits >= num_params * num_steps * (
        2 * grad_accum - 1
    )
    assert stats.d2h_sync_calls == num_steps

    init_local_chunks = {
        k: _local_chunk_for_rank(v, rank, world_size)
        for k, v in init_state.items()
    }
    step0_max_delta = max(
        float(
            (opt_records[0]["params"][k] - init_local_chunks[k])
            .abs()
            .max()
            .item()
        )
        for k in opt_records[0]["params"]
        if opt_records[0]["params"][k].numel() > 0
    )
    assert step0_max_delta > 1e-2, (
        f"Rank {rank}: step-0 param update {step0_max_delta} too small to"
        " separate from tolerance"
    )

    step_losses: list[float] = []
    for step in range(num_steps):
      def_r = default_records[step]
      opt_r = opt_records[step]
      cpu_r = cpu_math_records[step]
      if device.type == "cpu":
        assert ref_records is not None
        ref_r = ref_records[step]
        for mb in range(grad_accum):
          assert torch.equal(
              def_r["mb_losses"][mb], ref_r["mb_losses"][mb]
          ), f"Rank {rank} default step {step} mb {mb} loss mismatch"
          assert torch.equal(
              opt_r["mb_losses"][mb], ref_r["mb_losses"][mb]
          ), f"Rank {rank} step {step} mb {mb} loss mismatch"
          torch.testing.assert_close(
              opt_r["mb_losses"][mb],
              cpu_r["mb_losses_by_rank"][rank][mb],
              rtol=2e-3,
              atol=2e-3,
          )
        assert torch.equal(
            def_r["grad_norm"], ref_r["grad_norm"]
        ), f"Rank {rank} default step {step} grad_norm mismatch"
        assert torch.equal(
            opt_r["grad_norm"], ref_r["grad_norm"]
        ), f"Rank {rank} step {step} grad_norm mismatch"
        torch.testing.assert_close(
            opt_r["grad_norm"], cpu_r["grad_norm"], rtol=2e-3, atol=2e-3
        )
        for name in ref_r["grads"]:
          assert torch.equal(
              def_r["grads"][name], ref_r["grads"][name]
          ), f"Rank {rank} default step {step} grad mismatch on {name}"
          assert torch.equal(
              opt_r["grads"][name], ref_r["grads"][name]
          ), f"Rank {rank} step {step} grad mismatch on {name}"
        for name in ref_r["params"]:
          assert torch.equal(
              def_r["params"][name], ref_r["params"][name]
          ), f"Rank {rank} default step {step} param mismatch on {name}"
          assert torch.equal(
              opt_r["params"][name], ref_r["params"][name]
          ), f"Rank {rank} step {step} param mismatch on {name}"
      else:
        for candidate_r in (def_r, opt_r):
          for mb in range(grad_accum):
            torch.testing.assert_close(
                candidate_r["mb_losses"][mb],
                cpu_r["mb_losses_by_rank"][rank][mb],
                rtol=2e-3,
                atol=2e-3,
            )
          torch.testing.assert_close(
              candidate_r["grad_norm"],
              cpu_r["grad_norm"],
              rtol=2e-3,
              atol=2e-3,
          )
          for name in def_r["grads"]:
            torch.testing.assert_close(
                candidate_r["grads"][name],
                _local_chunk_for_rank(
                    cpu_r["full_grads"][name], rank, world_size
                ),
                rtol=2e-3,
                atol=2e-3,
            )
          for name in def_r["params"]:
            torch.testing.assert_close(
                candidate_r["params"][name],
                _local_chunk_for_rank(
                    cpu_r["full_params"][name], rank, world_size
                ),
                rtol=2e-3,
                atol=2e-3,
            )
        for mb in range(grad_accum):
          torch.testing.assert_close(
              opt_r["mb_losses"][mb],
              def_r["mb_losses"][mb],
              rtol=2e-3,
              atol=2e-3,
          )
        torch.testing.assert_close(
            opt_r["grad_norm"],
            def_r["grad_norm"],
            rtol=2e-3,
            atol=2e-3,
        )
        for name in def_r["grads"]:
          torch.testing.assert_close(
              opt_r["grads"][name],
              def_r["grads"][name],
              rtol=2e-3,
              atol=2e-3,
          )
        for name in def_r["params"]:
          torch.testing.assert_close(
              opt_r["params"][name],
              def_r["params"][name],
              rtol=2e-3,
              atol=2e-3,
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
