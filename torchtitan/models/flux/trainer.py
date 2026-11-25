# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field, replace
from typing import Annotated, Any

import torch
import torch.nn as nn
import tyro
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    enable_fsdp_symm_mem,
    resolve_fsdp_mesh,
)
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from torchtitan.models.flux.configs import FluxEncoderConfig, Inference
from torchtitan.models.flux.model.autoencoder import load_ae
from torchtitan.models.flux.model.model import FluxModel
import os as os
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel as simple_fsdp_data_parallel,
    MixedPrecisionPolicy as SimpleFSDPMixedPrecisionPolicy,
)
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
        model_args = config.model
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
            self.t5_encoder, self.clip_encoder = self._parallelize_encoders(
                t5_model=self.t5_encoder,
                clip_model=self.clip_encoder,
                parallel_dims=self.engine.parallel_dims,
                training=config.training,
                parallelism=config.parallelism,
                symm_mem_scope=config.parallelism.fsdp_symm_mem_scope,
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

    def _parallelize_encoders(
        self,
        *,
        t5_model,
        clip_model,
        parallel_dims,
        training,
        parallelism=None,
        symm_mem_scope=None,
    ):
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        if getattr(parallelism, "use_simple_fsdp", False):
            dp_mode = (
                "hybrid_shard"
                if parallel_dims.dp_replicate_enabled
                else "fully_shard"
            )
            mp_policy = SimpleFSDPMixedPrecisionPolicy(
                param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
                reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            )
            t5_model = simple_fsdp_data_parallel(
                t5_model,
                dp_mesh,
                mode=dp_mode,
                mp_policy=mp_policy,
            )
            logger.info(f"Applied SimpleFSDP ({dp_mode}) to the T5 encoder model")
            return t5_model, clip_model

        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )
        fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
        if dp_mesh_dims is not None:
            fsdp_config["dp_mesh_dims"] = dp_mesh_dims
        if training.enable_cpu_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy()

        hf_module = t5_model.hf_module
        assert isinstance(hf_module, nn.Module)
        annotate_replicated_parameters(hf_module, parallel_dims)
        for block in hf_module.encoder.block:  # pyrefly: ignore [missing-attribute]
            fully_shard(block, **fsdp_config)
        fully_shard(hf_module, **fsdp_config)
        enable_fsdp_symm_mem(hf_module, symm_mem_scope)
        disable_fsdp_gradient_division(hf_module)
        return t5_model, clip_model
