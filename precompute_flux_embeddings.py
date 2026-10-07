"""Precomputes and saves embeddings from Flux encoders (VAE, CLIP, T5).

This script processes a dataset specified via configuration, runs it through
the various encoders of a Flux model (Autoencoder, CLIP, T5), and saves
the precomputed embeddings to disk. Each batch is saved as a separate
'batch_{i}.pt' file in the specified output directory. This is useful for
speeding up training and reducing memory by avoiding repeated embedding
computation. This script is intended to be run on a TPU VM.

Example usage:
    python precompute_flux_embeddings.py \
        --module=flux \
        --config=flux_dev \
        --dataloader.dataset_path=tests/assets/cc12m \
        --dataloader.dataset=cc12m \
        --hf_assets_path=tests/assets/tokenizer \
        --training.steps=20 \
        --training.mixed_precision_param=bfloat16 \
        --training.mixed_precision_reduce=float32 \
        --dataloader.img_size=1024 \
        --training.local_batch_size=3 \
        --encoder.clip_encoder=openai/clip-vit-large-patch14 \
        --encoder.t5_encoder=tests/assets/flux_test_encoders/t5-v1_1-xxl \
        --output_dir=precomputed_embeddings
"""

import os
import sys
import argparse
import typing
import torch

from torchtitan.config import ConfigManager
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.models.flux.model.autoencoder import load_ae
from torchtitan.models.flux.model.hf_embedder import FluxEmbedder
from torchtitan.models.flux.utils import preprocess_data
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig


def main():
    init_logger()

    # Extract --output_dir before ConfigManager (tyro) parses sys.argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_dir", default="precomputed_embeddings")
    args, remaining_argv = parser.parse_known_args()
    output_dir = args.output_dir
    sys.argv = [sys.argv[0]] + remaining_argv

    # Parse remaining arguments using native ConfigManager
    config_manager = ConfigManager()
    config = typing.cast(
        TPUTrainerConfig,
        config_manager.parse_args(sys.argv[1:]),
    )

    # Initialize devices
    device = torch.device("tpu")
    device_module = utils.get_device_module()
    device_module.set_device(device)

    logger.info(f"Using device: {device}")
    dtype = TORCH_DTYPE_MAP[config.training.mixed_precision_param]

    # Build tools from config
    logger.info("Building tokenizers...")
    tokenizer = config.tokenizer.build()

    logger.info("Building dataset...")
    # Upstream d771abb0 deleted training.local_batch_size and retokenized the
    # dataloader interface. FluxCollator divides num_tokens_per_batch by
    # max_context_length to recover the image-row count, so pass the pair
    # straight through from the training config.
    dataloader = config.dataloader.build(
        dp_world_size=1,
        dp_rank=0,
        max_context_length=config.training.max_context_length,
        num_tokens_per_batch=(
            config.training.num_tokens_per_microbatch_per_dp_rank
        ),
        tokenizer=tokenizer,
    )

    model_args = config.model_spec.model

    logger.info("Loading Autoencoder...")
    autoencoder = load_ae(
        config.encoder.autoencoder_path,
        model_args.autoencoder_params,
        device="cpu",
        dtype=dtype,
        random_init=False,
    ).to(device=device, dtype=dtype)

    logger.info("Loading CLIP encoder...")
    clip_encoder = FluxEmbedder(
        version=config.encoder.clip_encoder,
        random_init=True,
    ).to(device=device, dtype=dtype)

    logger.info("Loading T5 encoder...")
    t5_encoder = FluxEmbedder(
        version=config.encoder.t5_encoder,
        random_init=True,
    ).to(device=device, dtype=dtype)

    data_iterator = iter(dataloader)
    max_steps = config.training.steps

    os.makedirs(output_dir, exist_ok=True)

    for i in range(max_steps):
        try:
            batch = next(data_iterator)
        except StopIteration:
            break

        input_dict, batch_labels = batch
        input_dict["image"] = batch_labels

        with torch.no_grad():
            processed_dict = preprocess_data(
                device=device,
                dtype=dtype,
                autoencoder=autoencoder,
                clip_encoder=clip_encoder,
                t5_encoder=t5_encoder,
                batch=input_dict,
            )

            # Sync TPU tensors
            torch.accelerator.synchronize()

            # Move to CPU for saving
            save_dict = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) 
                         for k, v in processed_dict.items()}

            output_path = os.path.join(output_dir, f"batch_{i}.pt")
            torch.save(save_dict, output_path)

        if (i + 1) % 10 == 0:
            logger.info(f"Precomputed {i+1}/{max_steps} batches")

    logger.info(f"Done! Precomputed embeddings saved to {output_dir}")


if __name__ == "__main__":
    main()
