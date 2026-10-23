"""
TPU-specific Ray Actor for vLLM Generation.

Uses composition rather than inheriting from `VLLMGenerator` to defer `vllm` imports until 
the Ray worker initializes, preventing the orchestrator from eagerly locking the TPU PJRT client.
Replaces Monarch/Torchstore weight sync with `load_model_state_dict()`, receiving weights 
via Ray's object store and manually copying them into the vLLM model's `DTensor` parameters.

TODO(jialei777): use the latest torchtpu-vllm, and confirm each of those TPU hacks is still needed. 
"""

import asyncio
import logging
import os
import pickle
import sys
import types

import ray
import torchtitan.experiments.tpu.rl_nc_ray  # Ensure mock runs first

from torchtitan.experiments.rl.actors.generator import VLLMGenerator
from torchtitan.experiments.rl.models.vllm_registry import register_to_vllm

# Modular Weight Transfer Helpers
from torchtitan.experiments.tpu.rl_nc_ray.weight_transfer_utils import (
    load_weights_on_driver,
)

logger = logging.getLogger(__name__)

# --- GLOBAL CONFIGURATION / CONSTANTS ---
# TPU Hack Constants for Teleporting Configs to Ray Workers
VLLM_ARGS_PATH_LOCAL = "/tmp/torchtitan_vllm_args.pkl"
VLLM_ARGS_PATH_REMOTE = "/data/assets/torchtitan_vllm_args.pkl"
VLLM_ARGS_PATH = VLLM_ARGS_PATH_LOCAL if os.environ.get("LOCAL_VM_RUN") == "1" else VLLM_ARGS_PATH_REMOTE


def patch_vllm_for_tpu(vllm_args_path: str) -> None:
    """
    Apply TPU-specific patches and workarounds to vLLM and torchtpu-vllm workers.
    """
    import vllm
    
    try:  
        # [TPU HACK 2]: Add missing 4-chip and 8-chip topologies to torchtpu-vllm.
        # The library hardcodes 16-chip multi-host topologies. We must inject our
        # smaller slice topologies for it to correctly dispatch PJRT clients.
        import vllm_torchtpu.platforms.tpu_platform as tpu_platform
        tpu_platform.TPU_MULTIHOST_TOPOLOGY_MAP[4] = "2,2,1"
        tpu_platform.TPU_MULTIHOST_TOPOLOGY_MAP[8] = "2,4,1"

        # [TPU HACK 3]: Prevent torchtpu-vllm from claiming the entire Ray cluster.
        # torchtpu-vllm assumes it is the only application running on the Ray cluster.
        # It queries `ray.nodes()` and overwrites `TORCH_TPU_TOPOLOGY` and slice builder 
        # addresses with ALL available TPUs (16 chips), stealing the Trainer's 8 chips.
        # We freeze the env vars to the 8 chips provided by grpo.py.
        original_driver_environ_setitem = os.environ.__class__.__setitem__
        def patched_driver_environ_setitem(self, key, value):
            if key in ("TORCH_TPU_SLICEBUILDER_ADDRESSES", "TPU_PROCESS_ADDRESSES", 
                       "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS", "TORCH_TPU_TOPOLOGY"):
                value = os.environ.get(key, value)
            elif key == "LIBTPU_INIT_ARGS":
                value = value.replace("--deepsea_chip_config_name=megachip_tccontrol", "")
            original_driver_environ_setitem(self, key, value)
        os.environ.__class__.__setitem__ = patched_driver_environ_setitem
        
        # [TPU HACK 4]: Hide the Trainer's TPU nodes from vLLM's RayExecutor.
        # vLLM blindly spawns Ray workers on all `ray.nodes()` that have "TPU" resources.
        # We patch `ray.nodes()` to only return the 8 IPs allocated to the Sampler.
        allowed_ips = set(addr.split(':')[0] for addr in os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES", "").split(',') if addr)
        if allowed_ips:
            original_ray_nodes = ray.nodes
            def patched_ray_nodes(*args, **kwargs):
                nodes = original_ray_nodes(*args, **kwargs)
                return [n for n in nodes if n.get("NodeManagerAddress") in allowed_ips or "TPU" not in n.get("Resources", {})]
            ray.nodes = patched_ray_nodes
            
            original_cluster_resources = ray.cluster_resources
            def patched_cluster_resources(*args, **kwargs):
                res = original_cluster_resources(*args, **kwargs)
                res["TPU"] = int(os.environ.get("WORLD_SIZE", "8"))
                return res
            ray.cluster_resources = patched_cluster_resources

        # [TPU HACK 5]: Restore TPU Pallas attention backend in vLLM registry overwritten by CUDA PyTorchVarlenAttentionBackend
        try:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
            from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend
            register_backend(AttentionBackendEnum.CUSTOM)(PallasAttentionBackend)
        except Exception as e:
            logger.debug(f"Could not re-register PallasAttentionBackend: {e}")

        # [TPU HACK] get worker rank
        def _get_worker_rank(worker_obj):
            try:
                from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
                return get_tensor_model_parallel_rank()
            except Exception:
                pass
            if hasattr(worker_obj, "worker") and hasattr(worker_obj.worker, "rank"):
                return worker_obj.worker.rank
            if hasattr(worker_obj, "local_rank"):
                return worker_obj.local_rank
            if hasattr(worker_obj, "rank"):
                return worker_obj.rank
            return int(os.environ.get("RANK", "0"))

        from vllm.v1.executor.ray_executor_v2 import RayWorkerProc
        original_initialize_worker = RayWorkerProc.initialize_worker
        
        def patched_initialize_worker(self, local_rank, env_vars=None, driver_env_vars=None, *args, **kwargs):
            import sys
            import torch
            from torchtitan.experiments.tpu.rl_nc_ray.tpu_env_utils import GENERATOR_BASE_PORT

            # Set dynamic slice builder port based on local_rank so each TPU process binds to its assigned port
            worker_sb_port = GENERATOR_BASE_PORT + local_rank
            libtpu_args = f"--xla_tpu_use_enhanced_launch_barrier=false --deepsea_slice_builder_worker_port={worker_sb_port}"
            if env_vars is not None:
                env_vars["LIBTPU_INIT_ARGS"] = libtpu_args
            os.environ["LIBTPU_INIT_ARGS"] = libtpu_args

            # Prevent RecursionError during TorchDynamo AOT tracing of deep transformer models
            sys.setrecursionlimit(50000)

            if "vllm.utils.import_utils" not in sys.modules:
                sys.modules["vllm.utils.import_utils"] = types.ModuleType("import_utils")
            sys.modules["vllm.utils.import_utils"].init_cached_hf_modules = lambda: None
            
            import vllm_torchtpu.runner.tpu_runner
            vllm_torchtpu.runner.tpu_runner._get_padded_num_reqs_with_upper_limit = lambda x, upper_limit: upper_limit
            import vllm.model_executor.model_loader.weight_utils
            vllm.model_executor.model_loader.weight_utils.initialize_dummy_weights = lambda *args, **kwargs: None

            def load_weights_from_state_dict_on_worker(self, state_dict):
                vllm_model = getattr(self, "model_runner", getattr(self, "worker", self)).model
                from torchtitan.experiments.tpu.rl_nc_ray.weight_transfer_utils import load_weights_on_worker
                rank = _get_worker_rank(self)
                return load_weights_on_worker(vllm_model, state_dict, rank)
                
            try:
                from vllm_torchtpu.worker.tpu_worker import TPUWorker
                TPUWorker.load_weights_from_state_dict_on_worker = load_weights_from_state_dict_on_worker
            except Exception as e:
                logger.warning(f"Could not patch TPUWorker: {e}")

            try:
                from vllm.v1.worker.worker_base import WorkerBase
                WorkerBase.load_weights_from_state_dict_on_worker = load_weights_from_state_dict_on_worker
            except Exception:
                pass

            self.__class__.load_weights_from_state_dict_on_worker = load_weights_from_state_dict_on_worker
            
            if os.path.exists(vllm_args_path):
                try:
                    with open(vllm_args_path, "rb") as f:
                        ms_spec, c_config, gen_config = pickle.load(f)
                    register_to_vllm(
                        ms_spec,
                        parallelism=gen_config.parallelism,
                        compile_config=c_config,
                        checkpoint_config=gen_config.checkpoint,
                        override=gen_config.override,
                    )
                except Exception as e:
                    logger.error(f"Failed to load TorchTitan args: {e}")
            
            # [TPU HACK]: Restore TPU Pallas attention backend in worker process after register_to_vllm imports attention.py
            try:
                from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
                from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend
                register_backend(AttentionBackendEnum.CUSTOM)(PallasAttentionBackend)
            except Exception as e:
                pass

            import torch._dynamo
            torch._dynamo.config._config["recompile_limit"].default = 100
            torch._dynamo.config._config["cache_size_limit"].default = 100
            torch._dynamo.config._config["accumulated_cache_size_limit"].default = 100
            
            res = original_initialize_worker(self, local_rank, env_vars, driver_env_vars, *args, **kwargs)
            if hasattr(self, "worker"):
                setattr(self.worker, "load_weights_from_state_dict_on_worker", types.MethodType(load_weights_from_state_dict_on_worker, self.worker))
            return res
            
        RayWorkerProc.initialize_worker = patched_initialize_worker
    except ImportError as e:
        logger.warning(f"Could not patch RayWorkerProc: {e}")


@ray.remote
class RayVLLMGenerator:
    def __init__(self, config, *, model_spec, model_path, compile_config, max_num_seqs, dump_folder=None, **kwargs):
        import sys
        # Prevent RecursionError during TorchDynamo AOT tracing on driver/worker
        sys.setrecursionlimit(50000)
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")  # different port from trainer just in case
        
        # [TPU HACK] Initialize single-rank CPU Gloo process group so upstream VLLMGenerator's
        # continuous-batching loop decisions (dist.broadcast_object_list) run as local no-ops
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                rank=0,
                world_size=1,
            )

        # Apply all TPU patches and hacks to vLLM & torchtpu-vllm
        patch_vllm_for_tpu(VLLM_ARGS_PATH)

        # Teleport configs to Ray workers using VLLM_ARGS_PATH file
        with open(VLLM_ARGS_PATH, "wb") as f:
            pickle.dump((model_spec, compile_config, config), f)

        # [TPU HACK] Monkey patch VLLMGenerator to support Ray weight transfer in engine loop
        import torchtitan.experiments.rl.actors.generator as gen_mod
        
        if not hasattr(gen_mod, "_OriginalModelStateDictPullRequest"):
            gen_mod._OriginalModelStateDictPullRequest = gen_mod.ModelStateDictPullRequest
            
            @gen_mod.dataclass(kw_only=True, slots=True)
            class CustomModelStateDictPullRequest:
                version: int
                refs: list | None = None
                
            gen_mod.ModelStateDictPullRequest = CustomModelStateDictPullRequest
            
        if not hasattr(gen_mod.VLLMGenerator, "_original_pull_model_state_dict"):
            gen_mod.VLLMGenerator._original_pull_model_state_dict = gen_mod.VLLMGenerator._pull_model_state_dict
            
            async def custom_pull(gen_self, version):
                request = gen_self._model_state_dict_pull_request
                if hasattr(request, "refs") and request.refs is not None:
                    state_dict_ref = request.refs[0]
                    
                    executor = gen_self._engine.model_executor
                    if hasattr(executor, "workers"):
                        refs = []
                        for worker in executor.workers:
                            refs.append(worker.execute_method.remote("load_weights_from_state_dict_on_worker", state_dict_ref))
                        ray.get(refs)
                    elif hasattr(executor, "collective_rpc"):
                        # [TPU HACK] Dispatch via collective_rpc on v1 MultiprocExecutor / RayDistributedExecutorV2
                        state_dict_data = ray.get(state_dict_ref)
                        executor.collective_rpc("load_weights_from_state_dict_on_worker", args=(state_dict_data,))
                    else:
                        state_dict_data = ray.get(state_dict_ref)
                        vllm_model = executor.driver_worker.get_model()
                        load_weights_on_driver(vllm_model, state_dict_data)
                                    
                                    
                    gen_self.policy_version = version
                    gen_self._engine.reset_prefix_cache()
                    
                    if gen_self._rank == 0 and gen_self._pull_model_state_dict_future is not None:
                        gen_self._pull_model_state_dict_future.set_result(version)
                        gen_self._pull_model_state_dict_future = None
                        gen_self._model_state_dict_pull_request = None
                else:
                    await gen_self._original_pull_model_state_dict(version)
                    
            gen_mod.VLLMGenerator._pull_model_state_dict = custom_pull

        self.profiler_config = getattr(config, "profiler", None)
        self._is_profiling = False

        # [TPU HACK]: Restore TPU Pallas attention backend before VLLMGenerator instantiation
        try:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
            from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend
            register_backend(AttentionBackendEnum.CUSTOM)(PallasAttentionBackend)
        except Exception as e:
            pass

        self.generator = VLLMGenerator(
            config,
            model_spec=model_spec,
            model_path=model_path,
            compile_config=compile_config,
            max_num_seqs=max_num_seqs,
            output_dir=dump_folder or "./outputs"
        )

    def _is_profiling_step(self, step: int) -> bool:
        if not self.profiler_config or not self.profiler_config.enable_profiling:
            return False

        cfg = self.profiler_config
        wait = cfg.profile_freq - (cfg.profiler_active + cfg.profiler_warmup)
        cycle_step = step % cfg.profile_freq

        start_active_bound = wait + cfg.profiler_warmup
        end_active_bound = wait + cfg.profiler_warmup + cfg.profiler_active
        return start_active_bound <= cycle_step < end_active_bound

    async def generate(self, *args, **kwargs):
        return await self.generator.generate(*args, **kwargs)

    async def start_engine_loop(self):
        return await self.generator.start_engine_loop()

    async def sync_log_step(self, step: int, relative_step: int | None = None):
        if self.profiler_config and self.profiler_config.enable_profiling:
            should_profile = self._is_profiling_step(step)
            if should_profile and not self._is_profiling:
                profile_prefix = f"step_{step}"
                logger.info(f"@@@ Generator: Starting profiler trace for step {step} with prefix {profile_prefix}")
                self.start_profile(profile_prefix)
                self._is_profiling = True
            elif not should_profile and self._is_profiling:
                logger.info(f"@@@ Generator: Stopping profiler trace at step {step}")
                self.stop_profile()
                self._is_profiling = False

        return await self.generator.sync_log_step(step, relative_step)

    async def close(self):
        if self._is_profiling:
            self.stop_profile()
            self._is_profiling = False
        return await self.generator.close()

    async def load_model_state_dict(self, version: int, state_dict_ref_list: list) -> None:
        """Queue Ray-native weight injection into engine loop."""
        self.generator._rank0_check_engine_loop_running("load_model_state_dict")
        
        pull_model_state_dict_future = asyncio.get_running_loop().create_future()
        
        import torchtitan.experiments.rl.actors.generator as gen_mod
        
        async with self.generator._engine_loop_condition:
            self.generator._model_state_dict_pull_request = gen_mod.ModelStateDictPullRequest(
                version=version,
                refs=state_dict_ref_list
            )
            self.generator._pull_model_state_dict_future = pull_model_state_dict_future
            self.generator._engine_loop_condition.notify()
            
        await pull_model_state_dict_future

    def start_profile(self, profile_prefix: str | None = None) -> None:
        if hasattr(self.generator, "_engine"):
            self.generator._engine.start_profile(profile_prefix)

    def stop_profile(self) -> None:
        if hasattr(self.generator, "_engine"):
            self.generator._engine.stop_profile()