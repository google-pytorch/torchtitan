# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field, replace
from typing import Annotated

import torch
import tyro

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.flux.configs import FluxEncoderConfig, Inference
from torchtitan.models.flux.model.autoencoder import load_ae
from torchtitan.models.flux.model.model import FluxModel
import os as os
from torchtitan.models.flux.parallelize import parallelize_encoders
from torchtitan.models.flux.precomputed_loader import PrecomputedDataloader
from torchtitan.models.flux.random_loader import RandomDataloader
from torchtitan.models.flux.tokenizer import FluxTokenizerContainer
from torchtitan.models.flux.validate import FluxValidator
from torchtitan.trainer import Trainer
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

# Tokenizer asset that FluxTrainer builds instead of the Hugging Face T5/CLIP
# tokenizers when random or precomputed embeddings make text tokenization
# unnecessary. The path is relative to the repository root.
_LOCAL_TOKENIZER_PATH = "tests/assets/tokenizer"

logger = logging.getLogger(__name__)


class FluxTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(TPUTrainerConfig):
        # Overwrite parent class tokenizer
        tokenizer: FluxTokenizerContainer.Config = (  # pyrefly: ignore [bad-override]
            field(default_factory=FluxTokenizerContainer.Config)
        )
        validator: Annotated[  # pyrefly: ignore [bad-override]
            FluxValidator.Config | None, tyro.conf.AvoidSubcommands
        ] = None
        encoder: FluxEncoderConfig = field(default_factory=FluxEncoderConfig)
        """Configuration for Flux encoders (T5 text encoder, CLIP text encoder, and autoencoder)."""
        inference: Inference = field(default_factory=Inference)

    def __init__(self, config: Config):
        assert config.model_spec is not None
        model_args = config.model_spec.model
        assert isinstance(model_args, FluxModel.Config)

        # If precomputed_dataset_path or random_dataset is provided in
        # experiment configs, we load precomputed/random embeddings directly
        # and skip running the autoencoder, text encoders (T5, CLIP), and the
        # Hugging Face dataset/tokenizer in Trainer.__init__.
        precompute_path = config.flux.precomputed_dataset_path
        random_dataset = getattr(config.flux, "random_dataset", False)
        self._use_precomputed_embeddings = (
            precompute_path is not None or random_dataset
        )

        if self._use_precomputed_embeddings:
            rank = int(os.environ.get("RANK", 0))
            if config.validator is None and not config.tokenizer.test_mode:
                # Trainer.__init__ always builds config.tokenizer even though
                # precomputed/random embeddings bypass text tokenization. Use
                # the local test tokenizer asset so distributed jobs do not hit
                # Hugging Face Hub rate limits across all ranks at startup.
                config.tokenizer = replace(
                    config.tokenizer,
                    test_mode=True,
                    t5_tokenizer_path=_LOCAL_TOKENIZER_PATH,
                    clip_tokenizer_path=_LOCAL_TOKENIZER_PATH,
                )
            if random_dataset:
                logger.info("Using RandomDataloader for embeddings")

                img_size = getattr(config.flux, "img_size", None)
                if img_size is None:
                    processor = getattr(
                        getattr(config.dataloader, "dataset", None),
                        "processor",
                        None,
                    )
                    img_size = getattr(processor, "img_size", None)
                if img_size is None:
                    raise ValueError(
                        "flux.random_dataset needs an image size, but "
                        f"{type(getattr(config.dataloader, 'dataset', None)).__name__} "
                        "exposes no processor.img_size. Configure the dataloader "
                        "with _flux_dataset(...) or set --flux.img_size."
                    )
                config.dataloader = RandomDataloader.Config(
                    bsz=(
                        config.training.num_tokens_per_microbatch_per_dp_rank
                        // config.training.max_context_length
                    ),
                    max_steps=config.training.steps,
                    img_size=img_size,
                    max_t5_len=config.tokenizer.max_t5_encoding_len,
                    t5_dim=model_args.context_in_dim,
                    clip_dim=model_args.vec_in_dim,
                )
            elif precompute_path is not None:
                logger.info(
                    f"Loading precomputed embeddings from {precompute_path}"
                )
                config.dataloader = PrecomputedDataloader.Config(
                    data_dir=precompute_path,
                    max_steps=config.training.steps,
                    rank=rank,
                )

        super().__init__(config)

        # Flux samples diffusion noise and timesteps during each model step, so
        # data-parallel ranks need distinct model RNG streams. Dataset
        # transformations such as prompt dropout use Grain's separate RNG.
        distinct_seed_mesh_dims = ["cp", "dp_shard", "dp_replicate"]
        dist_utils.set_determinism(
            self.engine.parallel_dims,
            self.engine.device,
            config.debug,
            distinct_seed_mesh_dims=distinct_seed_mesh_dims,
        )

        # NOTE: self._dtype is the data type used for encoders (image encoder, T5 text encoder, CLIP text encoder).
        # We cast the encoders and it's input/output to this dtype.  If FSDP with mixed precision training is not used,
        # the dtype for encoders is torch.float32 (default dtype for Flux Model).
        # Otherwise, we use the same dtype as mixed precision training process.
        self._dtype = (
            TORCH_DTYPE_MAP[config.training.mixed_precision_param]
            if self.engine.parallel_dims.dp_shard_enabled
            else torch.float32
        )

        if self._use_precomputed_embeddings:
            self.autoencoder = None
            self.clip_encoder = None
            self.t5_encoder = None
        else:
            # To support FSDP, we follow a four-step initialization pattern.
            # 1. Initialize on the 'meta' device to define the model.
            with torch.device("meta"):
                self.autoencoder = load_ae(
                    config.encoder.autoencoder_path,
                    model_args.autoencoder,
                    device="meta",
                    dtype=self._dtype,
                    random_init=config.encoder.random_init,
                )

                # Use the encoder configs from the model registry, overriding
                # version and random_init from the trainer encoder config if set.
                clip_encoder_config = model_args.clip_encoder
                t5_encoder_config = model_args.t5_encoder
                self.clip_encoder = replace(
                    clip_encoder_config,
                    version=config.encoder.clip_encoder or clip_encoder_config.version,
                    random_init=config.encoder.random_init,
                ).build()

                self.t5_encoder = replace(
                    t5_encoder_config,
                    version=config.encoder.t5_encoder or t5_encoder_config.version,
                    random_init=config.encoder.random_init,
                ).build()

            # 2. Apply FSDP to the T5 model / CLIP model.
            # pyrefly: ignore [bad-assignment]
            self.t5_encoder, self.clip_encoder = parallelize_encoders(
                t5_model=self.t5_encoder,
                clip_model=self.clip_encoder,
                parallel_dims=self.engine.parallel_dims,
                training=config.training,
                symm_mem_scope=config.parallelism.fsdp_symm_mem_scope,
                parallelism=config.parallelism,
            )

            # 3. Materialize on device.
            self.autoencoder.to_empty(device=self.engine.device)
            self.clip_encoder.to_empty(device=self.engine.device)
            self.t5_encoder.to_empty(device=self.engine.device)

            # 4. Initialize parameters.
            with torch.no_grad():
                self.autoencoder.init_weights()
                self.clip_encoder.init_weights()
                self.t5_encoder.init_weights()

        # The model's preprocess_inputs runs the encoders (or, with
        # precomputed/random embeddings, consumes the encodings as given).
        self.engine.preprocess_inputs_kwargs = {
            "autoencoder": self.autoencoder,
            "clip_encoder": self.clip_encoder,
            "t5_encoder": self.t5_encoder,
            "dtype": self._dtype,
        }

        if config.validator is not None:
            # pyrefly: ignore [missing-attribute]
            self.validator.flux_init(
                device=self.engine.device,
                _dtype=self._dtype,
                autoencoder=self.autoencoder,
                t5_encoder=self.t5_encoder,
                clip_encoder=self.clip_encoder,
                dump_folder=config.dump_folder,
            )
