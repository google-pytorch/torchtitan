"""
TPU/Ray specific initialization and dependency mocking.

Dynamically mocks `monarch` and `torchstore` in `sys.modules` to allow inheriting from 
core torchtitan RL classes (`RLTrainer`, `PolicyTrainer`, `VLLMGenerator`) without 
pulling in Monarch's networking or RDMA dependencies, which conflict with Ray and TPU.
"""

import sys
import copyreg
import torch.utils._config_module
from unittest.mock import MagicMock
import logging

import importlib

# Register custom reducer on PyTorch ConfigModule base class so that cloudpickle / Ray 
# can serialize RayTPUPolicyTrainer without failing on dynamically modified PyTorch config modules.
torch.utils._config_module.ConfigModule.__reduce__ = lambda self: (
    importlib.import_module,
    (self.__module__,),
)

logger = logging.getLogger(__name__)

# =====================================================================
# Monarch / Torchstore Mocking for Ray
# =====================================================================
# The RL Trainer and VLLMGenerator from torchtitan.rl (torchtitan.rl.distributed.actors)
# inherit from monarch.actor.Actor and use torchstore. 
# Since we are using Ray and do not want Monarch dependencies or its 
# Actor metaclass/endpoints to interfere with Ray, we dynamically mock 
# them out before importing the RL actors.
#
# This allows us to cleanly inherit from the torchtitan classes without 
# requiring Monarch or torchstore to be installed, and it completely 
# disarms the monarch Actor base class.
# =====================================================================

class DummyActor:
    pass

if 'monarch' not in sys.modules:
    import types
    sys.modules['torchstore'] = MagicMock()
    
    import os
    real_file = os.path.abspath(__file__)
    # Create real module objects for monarch instead of MagicMocks so they are serializable by reference
    monarch = types.ModuleType("monarch")
    monarch.__file__ = real_file
    monarch_actor = types.ModuleType("monarch.actor")
    monarch_actor.__file__ = real_file
    monarch_spmd = types.ModuleType("monarch.spmd")
    monarch_spmd.__file__ = real_file
    monarch_rdma = types.ModuleType("monarch.rdma")
    monarch_rdma.__file__ = real_file
    
    sys.modules['monarch'] = monarch
    sys.modules['monarch.actor'] = monarch_actor
    sys.modules['monarch.spmd'] = monarch_spmd
    sys.modules['monarch.rdma'] = monarch_rdma

    def dummy_endpoint(func):
        return func

    async def dummy_setup_torch_elastic_env_async(*args, **kwargs):
        pass

    monarch_actor.Actor = DummyActor
    monarch_actor.endpoint = dummy_endpoint
    monarch_actor.this_host = MagicMock()
    
    class DummyChannel:
        @staticmethod
        def open():
            return MagicMock(), MagicMock()
            
    class RankObj:
        def __init__(self):
            import os
            self.rank = int(os.environ.get("RANK", 0))
            
    monarch_actor.Channel = DummyChannel
    monarch_actor.Port = MagicMock
    monarch_actor.PortReceiver = MagicMock
    monarch_actor.HostMesh = MagicMock
    monarch_actor.ProcMesh = MagicMock
    monarch_actor.current_rank = RankObj
    
    monarch_rdma.is_rdma_available = lambda: False
    monarch_spmd.setup_torch_elastic_env_async = dummy_setup_torch_elastic_env_async


# =====================================================================
# vLLM GPU Codegen Mocking for TPU/CPU
# =====================================================================
# When torchtitan's vllm_wrapper.py imports vllm.compilation.codegen, it will
# crash on TPU or CPU-only container environments because Triton or GPU-specific
# codegen tools are absent. This shim preemptively injects codegen as None into
# the vLLM package hierarchy if it fails to import.
# =====================================================================
import os
if os.environ.get("TORCHTITAN_DEVICE_TYPE") != "cpu":
    import torchtitan.experiments.tpu.rl_nc_ray.vllm_mocks

