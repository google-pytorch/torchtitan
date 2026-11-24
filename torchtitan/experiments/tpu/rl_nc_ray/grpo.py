# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Ray-based RL training orchestrator for TPU.

Implements `RayRLTrainer`, which inherits core GRPO logic from the base GPU `RLTrainer` 
but overrides `setup()` to spawn Ray actors instead of Monarch procs. Uses proxy classes 
(`RayTrainerProxy`, `RayGeneratorProxy`) to transparently translate Monarch `.call().get()` 
RPCs into Ray `.remote()` and `ray.get()` futures. Pins the trainer to CPU and the 
generator to TPU to bypass PJRT client deadlock limitations.
"""

import asyncio
import copy
import os
import math
import typing
from dataclasses import dataclass

import ray
import ray.util.scheduling_strategies

import torchtitan.experiments.tpu.rl_nc_ray  # Ensure mock runs first

from torchtitan.config.manager import ConfigManager
from torchtitan.rl.controller import Controller
from torchtitan.experiments.tpu.rl_nc_ray.trainer import RayTPUPolicyTrainer
from torchtitan.experiments.tpu.rl_nc_ray.generator import RayVLLMGenerator
from torchtitan.observability.logging import init_logger

from torchtitan.experiments.tpu.rl_nc_ray.tpu_env_utils import (
    discover_tpu_cluster_layout,
    build_trainer_env_vars,
    build_generator_env_vars,
)


class ValueMeshProxy:
    """Mocks the Monarch ValueMesh returned by awaiting a future"""
    def __init__(self, vals):
        self.vals = vals

    def get(self, rank):
        if rank < len(self.vals):
            return self.vals[rank]
        return None

class FutureProxy:
    """Mocks the Monarch Future returned by .call()"""
    def __init__(self, obj_refs):
        self.obj_refs = obj_refs if isinstance(obj_refs, list) else [obj_refs]

    async def _await_internal(self):
        vals = await asyncio.gather(*self.obj_refs)
        return ValueMeshProxy(vals)

    def __await__(self):
        return self._await_internal().__await__()

    def get(self):
        vals = ray.get(self.obj_refs)
        return vals[0] if vals else None


class RayTrainerProxy:
    """Mocks the Monarch Actor wrapper for the Trainer"""
    def __init__(self, ray_trainers):
        self.ray_trainers = ray_trainers if isinstance(ray_trainers, list) else [ray_trainers]

    def __getattr__(self, name):
        class MethodProxy:
            def __init__(self, proxy):
                self.proxy = proxy
            def call(self, *args, **kwargs):
                if name == "push_model_state_dict":
                    # We skip torchstore, so pushing model state dict just means
                    # returning an empty future (the actual state dict is pulled later)
                    return FutureProxy([ray.put(None)] * len(self.proxy.ray_trainers))
                
                futures = []
                for trainer in self.proxy.ray_trainers:
                    method = getattr(trainer, name)
                    futures.append(method.remote(*args, **kwargs))
                return FutureProxy(futures)
        return MethodProxy(self)

# TODO: currently we only allow num_generator to be 1, route and fanout are the same.
class RayInterGeneratorRouter:
    """Mocks the InterGeneratorRouter for Ray"""
    def __init__(self, ray_generator, trainer_proxy, hot_swap=True):
        self.ray_generator = ray_generator
        self.trainer_proxy = trainer_proxy
        self.hot_swap = hot_swap

    async def route(self, method, *args, routing_ctx, **kwargs):
        # Delegate to Ray actor's method (e.g. generate)
        return await getattr(self.ray_generator, method).remote(*args, **kwargs)

     
    async def fanout(self, method, *args, return_exceptions=False, **kwargs):
        # Call on all generators (we only have one).
        try:
            return [await getattr(self.ray_generator, method).remote(*args, **kwargs)]
        except Exception as e:
            if return_exceptions:
                return [e]
            raise

    async def pull_model_state_dict(self, *, policy_version: int) -> None:
        """Ray-native weight injection."""
        # Call on all trainers to participate in collective operation!
        future_proxy = self.trainer_proxy.get_model_state_dict.call()
        
        # [TPU HACK / OPTIMIZATION]
        # Why: Passing refs[0] directly to the remote actor auto-resolves/deserializes it on the generator driver.
        # This double-serialization on the generator driver CPU causes massive CPU/memory bottlenecks.
        # Solution: Wrap refs[0] in a list [refs[0]] to bypass auto-resolution, passing raw ObjectRef.
        ref = self.ray_generator.load_model_state_dict.remote(policy_version, [future_proxy.obj_refs[0]])
        await ref


class RayController(Controller):
    """Subclasses the GPU Controller, overriding ONLY setup() to inject Ray."""

    def __init__(self, config: Controller.Config, trainer_profiler_config=None, generator_profiler_config=None):
        super().__init__(config)
        self.trainer_profiler_config = trainer_profiler_config or config.trainer.profiler
        self.generator_profiler_config = generator_profiler_config or config.generator.profiler

    async def setup_async(self, **kwargs):
        config = self.config
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        layout = discover_tpu_cluster_layout(config)
        
        sampler_nodes_info = layout["sampler_nodes_info"]
        self.trainer_world_size = layout["trainer_world_size"]
        self.generator_world_size = layout["generator_world_size"]
        self.trainer_dp_degree = self.trainer_world_size
        self._multi_node = False
 
        trainer_device_type = layout["trainer_device_type"]

        # Compute absolute paths for the worker runtime environment
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        dummy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "dummy_modules"))
        worker_pythonpath = f"{dummy_dir}:{base_dir}"

        # 2. Launch Ray Trainer actors with worker-specific layout environments
        trainer_actors = []
        for rank in range(self.trainer_world_size):
            env_vars, host_ip = build_trainer_env_vars(rank, layout, worker_pythonpath)
                
            trainer_resources = {f"node:{host_ip}": 0.01}
            if trainer_device_type == "tpu":
                trainer_resources["TPU"] = 1
                
            trainer_config = copy.deepcopy(config.trainer)
            if not trainer_config.dump_folder:
                trainer_config.dump_folder = config.dump_folder

            trainer_actor = RayTPUPolicyTrainer.options(
                resources=trainer_resources,
                runtime_env={"env_vars": env_vars}
            ).remote(
                trainer_config,
                model_spec=config.model_spec,
                hf_assets_path=config.hf_assets_path,
                generator_dtype=config.generator.model_dtype,
                compile_config=config.compile,
                profiler_config=self.trainer_profiler_config,
                output_dir=config.dump_folder or "./outputs",
            )
            trainer_actors.append(trainer_actor)

        # 3. Launch Ray Generator actor
        generator_env_vars = build_generator_env_vars(layout, worker_pythonpath, dump_folder=config.dump_folder)
        
        if self.generator_profiler_config and self.generator_profiler_config.enable_profiling:
            profile_dir = os.path.abspath(os.path.join(
                config.dump_folder or "./outputs",
                self.generator_profiler_config.save_traces_folder or "profile_traces"
            ))
            generator_env_vars["VLLM_TORCH_PROFILER_DIR"] = profile_dir
            
        generator_resources = {}

        generator_compile_config = copy.deepcopy(config.compile)
        generator_compile_config.enable = False

        async_loop = config.async_loop
        max_active_rollout_groups = (
            async_loop.max_offpolicy_steps + 1
        ) * async_loop.num_groups_per_train_step
        rollout_concurrency = max(
            max_active_rollout_groups * async_loop.group_size,
            async_loop.validation.num_samples,
        )
        generator_dp_degree = max(config.generator.parallelism.data_parallel_degree, 1)
        # Note: In RayRLTrainer, we only launch one generator actor wrapper 
        # (generator_world_size specifies total TPUs for it, handled inside).
        # We can assume num_generator_dp_shards = 1 * generator_dp_degree 
        num_generator_dp_shards = 1 * generator_dp_degree
        max_num_seqs = min(
            math.ceil(rollout_concurrency / num_generator_dp_shards), 512
        )

        generator_actor = RayVLLMGenerator.options(
            resources=generator_resources,
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=sampler_nodes_info[0]["NodeID"],
                soft=False
            ) if sampler_nodes_info else "DEFAULT",
            runtime_env={
                "env_vars": generator_env_vars
            }
        ).remote(
            config.generator,
            model_spec=config.model_spec,
            model_path=config.hf_assets_path,
            compile_config=generator_compile_config,
            max_num_seqs=max_num_seqs,
            dump_folder=config.dump_folder,
        )

        # 4. Wrap actors in Monarch-compatible proxies / routers
        self.trainer = RayTrainerProxy(trainer_actors)
        self.generator_router = RayInterGeneratorRouter(generator_actor, self.trainer)
        
        self._proc_meshes = [] # Dummy out Monarch meshes for cleanup

        await self.trainer.push_model_state_dict.call()
        
        self.start_step = self._get_rank_0_value(
            await self.trainer.get_policy_version.call()
        )
        if self.start_step > 0:
            print(f"Resuming RL training from step {self.start_step}")

        await self.generator_router.fanout("start_engine_loop")

        await self.generator_router.pull_model_state_dict(policy_version=self.start_step)

def main():
    init_logger()
    config = typing.cast(Controller.Config, ConfigManager().parse_args())

    rl_trainer = RayController(config)
    try:
        asyncio.run(rl_trainer.setup_async())
        asyncio.run(rl_trainer.run())
    finally:
        asyncio.run(rl_trainer.close())
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main()
