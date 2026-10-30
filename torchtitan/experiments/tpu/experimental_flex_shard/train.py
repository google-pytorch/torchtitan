"""Launch the opt-in trainer through TorchTitan's existing configuration system."""

import torch.distributed as dist

from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
from torchtitan.tools.logging import init_logger


def start_trainer(config):
    init_logger()
    trainer = config.build()
    try:
        with set_unflatten_seq_len(config.training.max_context_length):
            trainer.train()
    finally:
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    gmain.handle_main(start_trainer)
