"""
TPU-specific Ray Actor for RL Policy Training.

Inherits from `PolicyTrainer` but is wrapped with `@ray.remote` instead of Monarch's Actor framework. 
Runs exclusively on CPU to decouple from the TPU Generator, patching `VarlenAttention` to use CPU 
SDPA. Replaces `torchstore` weight pushing with `get_model_state_dict()`, extracting CPU tensors for 
Ray to send to the Generator. Implements DTensor workarounds for CPU Gloo checkpoint loading.
"""

import copy
import logging
import os
import torch
import ray
import torch.distributed.checkpoint as dcp
from torchtitan.experiments.tpu.rl_nc_ray.padding_packing_utils import (
    patch_varlen_attention,
    patch_create_varlen_metadata,
)

from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed.tensor import DTensor
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.observability.profiler import Profiler
from torchtitan.experiments.tpu import utils as tpu_utils
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools import utils
from torchtitan.components.checkpointer.utils import canonical_fqn

logger = logging.getLogger(__name__)

# --- GLOBAL CONFIGURATION / CONSTANTS ---
PAD_LEN = 4096
MAX_SEQS = 64


def patch_torch_and_dist_for_trainer(target_device: str) -> None:
    """
    Apply TPU/CPU-specific PyTorch and distributed backend monkey-patches to PolicyTrainer.
    """
    import torch._dynamo
    # PyTorch Dynamo specializes on the tensor type. In TP overlap, collectives 
    # return AsyncCollectiveTensor or a plain Tensor depending on timing. 
    # Each switch triggers a recompile. We must increase the limit to prevent crashes.
    torch._dynamo.config.recompile_limit = 100
    
    if target_device == "tpu":
        # =====================================================================
        # TPU vs GPU Kernel Implementation Gaps:
        # On GPU/CUDA, `torch.std` is backed by native, highly optimized CUDA kernels
        # supporting variance correction parameters natively.
        # TorchTPU lacks a direct implementation for `aten::std.correction`,
        # which throws a hardware runtime error when executing reward normalization.
        # We bypass this gap by monkey-patching `std()` with an explicit eager-form
        # mathematical calculation of variance, preserving numerical precision and safety.
        # =====================================================================
        orig_std = torch.Tensor.std
        def patched_std(self, *args, **kwargs):
            mean = self.mean()
            if kwargs.get("unbiased", True) and self.numel() > 1:
                var = ((self - mean) ** 2).sum() / (self.numel() - 1)
            else:
                var = ((self - mean) ** 2).mean()
            return torch.sqrt(var)
        torch.Tensor.std = patched_std
        
    # When running the trainer on CPU (e.g. single-VM development/testing),
    # force distributed backend to Gloo and disable TPU device probing so PyTorch
    # does not attempt to initialize libtpu/PJRT, avoiding hardware lock conflicts
    # with the co-located TPU vLLM generator actor.
    if target_device == "cpu":
        dist_utils._get_distributed_backend = lambda x: "gloo"
        if hasattr(torch, "tpu"):
            torch.tpu.is_available = lambda: False


def load_initial_hf_weights_on_cpu(model, torchtitan_state_dict: dict) -> None:
    """
    Manually copies initial HF state dict tensors into model's DTensors on CPU.
    """
    # state_dict() exposes the logical slices of fused parameters (wq/wk/wv,
    # w1/w3) as tensors built by a save hook, so writing into those would not
    # load the fused parameter. Merge the checkpoint into the fused layout and
    # copy into the parameters instead; state_dict() still supplies buffers.
    torchtitan_state_dict = tpu_utils.merge_into_fused_parameters(
        model, torchtitan_state_dict
    )
    model_sd = {
        **model.state_dict(),
        **{canonical_fqn(name): p for name, p in model.named_parameters()},
    }
    with torch.no_grad():
        for k, v in torchtitan_state_dict.items():
            if k in model_sd:
                orig_v = model_sd[k]
                
                def handle_dtensor_copy(target_dtensor, source_tensor):
                    target_local = target_dtensor.to_local() if hasattr(target_dtensor, "to_local") else target_dtensor
                    
                    if hasattr(source_tensor, "to_local"):
                        source_tensor = source_tensor.to_local()
                    elif isinstance(source_tensor, torch.nn.Parameter) and hasattr(source_tensor.data, "to_local"):
                        source_tensor = source_tensor.data.to_local()
                    elif isinstance(source_tensor, torch.nn.Parameter):
                        source_tensor = source_tensor.data
                        
                    if target_local.shape == source_tensor.shape:
                        target_local.copy_(source_tensor)
                    else:
                        for dim in range(len(target_dtensor.shape)):
                            if target_dtensor.shape[dim] != target_local.shape[dim]:
                                shard_size = target_local.shape[dim]
                                rank = int(os.environ.get("LOCAL_RANK", 0)) if os.environ.get("TORCHTITAN_DEVICE_TYPE", "cpu") == "cpu" else int(os.environ.get("RANK", 0))
                                rank_offset = shard_size * rank
                                indices = [slice(None)] * len(target_dtensor.shape)
                                indices[dim] = slice(rank_offset, rank_offset + shard_size)
                                target_local.copy_(source_tensor[tuple(indices)])
                                return
                        # Fallback if no dim diff found but shape mismatch
                        target_local.copy_(source_tensor)

                if isinstance(orig_v, DTensor):
                    handle_dtensor_copy(orig_v, v)
                elif isinstance(orig_v, torch.nn.Parameter) and isinstance(orig_v.data, DTensor):
                    handle_dtensor_copy(orig_v.data, v)
                else:
                    orig_v.copy_(v)


@ray.remote
class RayTPUPolicyTrainer(PolicyTrainer):

    def __init__(self, *args, **kwargs):
        profiler_config = kwargs.pop("profiler_config", None)
        # Ray actors don't have torchrun env vars by default.
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        
        target_device = os.environ.get("TORCHTITAN_DEVICE_TYPE", "cpu")
        tpu_utils.set_device_type(target_device)
        
        # Apply trainer-specific PyTorch and distributed backend patches
        patch_torch_and_dist_for_trainer(target_device)
        
        # Monkey patch torchtitan utils before super().__init__ calls it
        if not hasattr(utils, "device_module"):
            utils.device_module = utils.get_device_module()  
            utils.device_type = target_device  
            
        # Manually initialize distributed for CPU to prevent XLA from hijacking it
        if target_device == "cpu" and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
            
        # TPU does not support fused AdamW natively, so we fallback to the foreach implementation
        config = args[0] if len(args) > 0 else kwargs.get("config")
        if config is not None:
            config.optimizer.implementation = "foreach"
            
        super().__init__(*args, **kwargs)
        
        if profiler_config and profiler_config.enable_profiling:
            self.profiler = profiler_config.build(
                global_step=0,
                base_folder=self.config.dump_folder or "./outputs",
            )
            self.profiler.__enter__()
        else:
            self.profiler = None

    def _build_model(
        self,
        model_spec,
        config,
        device_type: str,
    ):
        from dataclasses import replace
        from torchtitan.config import apply_overrides, TORCH_DTYPE_MAP
        from torchtitan.tools import utils as titan_utils

        target_device = os.environ.get("TORCHTITAN_DEVICE_TYPE", "cpu")
        patch_varlen_attention(target_device)

        model_spec.model.update_from_config(config=config)
        max_seq_len = model_spec.model.max_seq_len
        # Upstream d771abb0 renamed training.seq_len ->
        # training.max_context_length.
        seq_len = config.training.max_context_length
        if seq_len > max_seq_len:
            raise ValueError(
                f"Training sequence length {seq_len} exceeds "
                f"attention RoPE maximum supported sequence length {max_seq_len}."
            )

        for layer_cfg in model_spec.model.layers:
            attention_cfg = getattr(layer_cfg, "attention", None)
            if attention_cfg is not None:
                attention_cfg.rope = replace(attention_cfg.rope, max_context_length=seq_len)

        if config.override.imports:
            apply_overrides(config.override, model_spec.model)

        with torch.device("meta"):
            with titan_utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
                model = model_spec.model.build()

        skip_dp_val = (target_device == "cpu" or not self.parallel_dims.dp_enabled)
        model = model_spec.parallelize_fn(
            model,
            parallel_dims=self.parallel_dims,
            training=config.training,
            parallelism=config.parallelism,
            compile_config=self.compile_config,
            ac_config=config.ac_config,
            dump_folder=config.dump_folder,
            skip_dp=skip_dp_val,
        )

        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        return model

    def _load_initial_hf_weights(self, model, checkpoint_path: str) -> None:
        if os.environ.get("TORCHTITAN_DEVICE_TYPE", "cpu") != "cpu":
            return super()._load_initial_hf_weights(model, checkpoint_path)
            
        if self.sd_adapter is None:
            return

        storage_reader = self.sd_adapter.get_hf_storage_reader(checkpoint_path)
        hf_state_dict = self.sd_adapter.to_hf(model.state_dict())
        
        dcp.load(hf_state_dict, storage_reader=storage_reader)
        torchtitan_state_dict = self.sd_adapter.from_hf(hf_state_dict)

        # Fix DTensor copy issue on CPU manually using helper function
        load_initial_hf_weights_on_cpu(model, torchtitan_state_dict)
        logger.info(f"Loaded initial weights from {checkpoint_path}")

    def get_model_state_dict(self) -> dict:
        """Ray-native weight extraction. Returns state dict on CPU."""
        print(f"@@@ Trainer Rank {os.environ.get('RANK', '0')}: Extracting model state dict...")

        target_device = os.environ.get("TORCHTITAN_DEVICE_TYPE", "cpu")
        if target_device == "tpu":
            import torch_tpu
            from torchtitan.experiments.tpu.rl_nc_ray.weight_transfer_utils import (
                prepare_trainer_state_dict,
            )
            print(f"@@@ Trainer Rank {os.environ.get('RANK', '0')}: Sync 1 done")
            torch_tpu._internal.sync.synchronize(wait=True)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            print(f"@@@ Trainer Rank {os.environ.get('RANK', '0')}: Barrier 1 done")
            
            # Use modularized, high-performance weight transfer helper
            cpu_sd = prepare_trainer_state_dict(self.model)
            
            torch_tpu._internal.sync.synchronize(wait=True)
        else:
            cpu_sd = {}
            raw_params = dict(self.model.named_parameters())
            if "lm_head.weight" not in raw_params and "tok_embeddings.weight" in raw_params:
                raw_params["lm_head.weight"] = raw_params["tok_embeddings.weight"]
            with torch.no_grad():
                for k, v in raw_params.items():
                    if hasattr(v, "to_local"):
                        cpu_sd[k] = v.to_local().detach().cpu().clone().contiguous()
                    else:
                        cpu_sd[k] = v.detach().cpu().clone().contiguous()
            print("@@@ Trainer: Finished extracting model state dict on CPU.")

        print(f"@@@ Trainer Rank {os.environ.get('RANK', '0')}: Returning cpu_sd!")            
        return cpu_sd



    async def optim_step(self) -> dict:
        res = await super().optim_step()
        if hasattr(self, "profiler") and self.profiler is not None:
            self.profiler.step()
        return res

    def finish(self) -> bool:
        if hasattr(self, "profiler") and self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
            print(f"@@@ Trainer Rank {os.environ.get('RANK', '0')}: Profiler cleaned up.")
        return True
