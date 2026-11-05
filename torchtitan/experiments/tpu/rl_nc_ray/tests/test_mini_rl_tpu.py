"""
A mini verl-style Multi-Host Reinforcement Learning POC on TPU.
How to run on Ray Cluster:
1. Get the ray head pod name:
   kubectl get pods -n default | grep ray.*head
   # Example output: ray-tpu-cluster-head-wmtdh
2. Export the head node name and local path:
   export HEAD_NODE=ray-tpu-cluster-head-wmtdh
   export RL_PATH="/your/local/path/of/torchtitan"
3. Copy this file to the head container:
   kubectl cp $RL_PATH/torchtitan/experiments/tpu/rl/mini_rl.py $HEAD_NODE:/home/ray/ray/torch_tpu/mini_rl.py -c ray-head
4. Submit the Ray Job:
   kubectl exec $HEAD_NODE -c ray-head -- bash -c "ray job submit --address='http://localhost:8265' --working-dir /home/ray/ray/torch_tpu --runtime-env-json='{\"env_vars\": {\"RAY_DEDUP_LOGS\": \"0\"}}' -- python mini_rl.py"
"""

import hashlib
import os
import sys
import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import SGD
import torch_tpu.api


# each ray worker is running on 1 TPU chip.
@ray.remote(resources={"TPU": 1})
class VerlWorker:

  def __init__(self, rank: int, world_size: int, config: dict):
    self.rank = rank
    self.world_size = world_size
    self.config = config

  def setup(self, master_addr: str, master_port: int, sb_addresses: str):
    torch_tpu.api.tpu_device()
    torch.utils.rename_privateuse1_backend("tpu")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(self.rank)  # number from 0 to 15
    os.environ["LOCAL_RANK"] = str(
        self.rank % 4
    )  # number from 0 to 3 on tpu chip on each host
    os.environ["WORLD_SIZE"] = str(self.world_size)
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = sb_addresses
    # torpology is 4 * 4 * 1 for v6e-16
    os.environ["TORCH_TPU_TOPOLOGY"] = "4,4,1"
    dist.init_process_group("tpu_dist")
    self.device = torch.device("tpu")
    torch.manual_seed(42)
    self.model = nn.Linear(10, 1).to(self.device)
    self.optimizer = SGD(self.model.parameters(), lr=0.1)
    print(f"Worker {self.rank} initialized.")

  def get_weights_hash(self):
    with torch.no_grad():
      w = self.model.weight.cpu().numpy().tobytes()
      return hashlib.md5(w).hexdigest()

  def rollout(self, batch_size: int):
    with torch.no_grad():
      # Set seed based on rank to ensure different data on each worker
      torch.manual_seed(100 + self.rank)
      return torch.randn(batch_size, 10).numpy()

  def update_policy(self, data: np.ndarray):
    self.optimizer.zero_grad()
    inputs = torch.from_numpy(data).to(self.device)
    targets = torch.zeros(inputs.size(0), 1).to(self.device)
    loss = nn.MSELoss()(self.model(inputs), targets)
    loss.backward()
    for param in self.model.parameters():
      if param.grad is not None:
        print(
            f"Rank {self.rank} - Grad mean BEFORE all_reduce:"
            f" {param.grad.mean().item():.6f}"
        )
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        print(
            f"Rank {self.rank} - Grad mean AFTER all_reduce:"
            f" {param.grad.mean().item():.6f}"
        )
    self.optimizer.step()
    return loss.item()

  def get_ip(self):
    return ray.util.get_node_ip_address()


class RayWorkerGroup:

  def __init__(self, world_size: int):
    self.world_size = world_size
    self.actors = [
        VerlWorker.remote(i, world_size, {}) for i in range(world_size)
    ]
    print(f"Ray Worker group initialized with {self.actors} {self.world_size}")

  def init_model(self):
    # Get IPs of all actors first
    ips = ray.get([a.get_ip.remote() for a in self.actors])

    master_addr = ips[0]  # Use the IP of the first worker as the master address
    master_port = (
        29500  # this port is used by torch.distributed.init_process_group
    )

    sb_addresses_list = []
    for i in range(self.world_size):
      host_ip = ips[i]
      port = 8471 + (i % 4)
      sb_addresses_list.append(f"{host_ip}:{port}")

    sb_addresses = ",".join(sb_addresses_list)
    print(f"Ray Worker group initializing on {master_addr} {sb_addresses}")
    ray.get([
        a.setup.remote(master_addr, master_port, sb_addresses)
        for a in self.actors
    ])

  def execute_rl_iteration(self):
    print(
        "Ray Worker group executing RL iteration on"
        f" {ray.get_runtime_context().gcs_client}"
    )
    rollouts = ray.get([a.rollout.remote(16) for a in self.actors])
    losses = ray.get(
        [a.update_policy.remote(rollouts[i]) for i, a in enumerate(self.actors)]
    )
    hashes = ray.get([a.get_weights_hash.remote() for a in self.actors])
    print(f"Weight hashes: {hashes}")
    if len(set(hashes)) == 1:
      print("VERIFIED: Weights are bit-identical across all 16 chips.")
    else:
      print("ERROR: Weights diverged!")


def main():
  ray.init()
  # world size is 16 because we have 16 TPU chips.
  wg = RayWorkerGroup(world_size=16)
  wg.init_model()
  for i in range(3):
    print(f"\n--- RL Iteration {i} ---")
    wg.execute_rl_iteration()
  print("\nSUCCESS: verl-style Synchronized RL works!")


if __name__ == "__main__":
  main()