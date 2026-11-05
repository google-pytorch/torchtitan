# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import spmd_types as spmd
import torch

from torchtitan.components.data.collators import TrainerBatch
from torchtitan.components.data.loader import DataloaderExhaustedError
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.flux.configs import FluxEncoderConfig, Inference
from torchtitan.models.flux.model.autoencoder import load_ae
from torchtitan.models.flux.model.model import FluxModel
import os as os
from torchtitan.models.flux.parallelize import parallelize_encoders
from torchtitan.models.flux.precomputed_loader import PrecomputedDataloader
from torchtitan.models.flux.random_loader import RandomDataloader
from torchtitan.models.flux.sharding import annotate_flux_forward_inputs
from torchtitan.models.flux.tokenizer import FluxTokenizerContainer
from torchtitan.models.flux.utils import (
    create_position_encoding_for_latents,
    pack_latents,
    preprocess_data,
)
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

# Tokenizer asset that FluxTrainer builds instead of the Hugging Face T5/CLIP
# tokenizers when random or precomputed embeddings make text tokenization
# unnecessary. The path is relative to the repository root.
_LOCAL_TOKENIZER_PATH = "tests/assets/tokenizer"


class FluxTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(TPUTrainerConfig):
        # Overwrite parent class tokenizer
        tokenizer: FluxTokenizerContainer.Config = (  # pyrefly: ignore [bad-override]
            field(default_factory=FluxTokenizerContainer.Config)
        )
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

        if precompute_path is not None or random_dataset:
            rank = int(os.environ.get("RANK", 0))
            if not config.validator.enable and not config.tokenizer.test_mode:
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

                # Upstream d771abb0 deleted training.local_batch_size in
                # favour of a token budget; flux batches images, so recover
                # the sample count.
                #
                # The same release deleted training.seq_len, which used to be
                # computed here from `config.dataloader.img_size`. The recipe
                # now owns that arithmetic (`_flux_seq_len` sets
                # `max_context_length`), so read `img_size` straight back off
                # `config.flux.img_size` or the processor that `_flux_dataset`
                # configured.
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
        distinct_seed_mesh_dims = (
            ["cp", "dp_shard", "dp_replicate"]
            if config.parallelism.spmd_backend == "spmd_types"
            else ["fsdp", "dp_replicate"]
        )
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=distinct_seed_mesh_dims,
        )

        # NOTE: self._dtype is the data type used for encoders (image encoder, T5 text encoder, CLIP text encoder).
        # We cast the encoders and it's input/output to this dtype.  If FSDP with mixed precision training is not used,
        # the dtype for encoders is torch.float32 (default dtype for Flux Model).
        # Otherwise, we use the same dtype as mixed precision training process.
        self._dtype = (
            TORCH_DTYPE_MAP[config.training.mixed_precision_param]
            if self.parallel_dims.dp_shard_enabled
            else torch.float32
        )

        if precompute_path is not None or random_dataset:
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

        # Apply FSDP to the T5 model / CLIP model
        if self.t5_encoder is not None or self.clip_encoder is not None:
            # pyrefly: ignore [bad-assignment]
            self.t5_encoder, self.clip_encoder = parallelize_encoders(
                t5_model=self.t5_encoder,
                clip_model=self.clip_encoder,
                parallel_dims=self.parallel_dims,
                training=config.training,
                enable_symm_mem=config.parallelism.enable_fsdp_symm_mem,
                parallelism=config.parallelism,
            )

            # 3. Materialize on device.
            self.autoencoder.to_empty(device=self.device)
            self.clip_encoder.to_empty(device=self.device)
            self.t5_encoder.to_empty(device=self.device)

            # 4. Initialize parameters.
            with torch.no_grad():
                self.autoencoder.init_weights()
                self.clip_encoder.init_weights()
                self.t5_encoder.init_weights()

        if config.validator.enable:
            # pyrefly: ignore [missing-attribute]
            self.validator.flux_init(
                device=self.device,
                _dtype=self._dtype,
                autoencoder=self.autoencoder,
                t5_encoder=self.t5_encoder,
                clip_encoder=self.clip_encoder,
                dump_folder=config.dump_folder,
            )

    def batch_generator(
        self, data_iterable: Iterable[TrainerBatch]
    ) -> Iterator[dict[str, Any]]:
        """Override to count transformer tokens (image patches + text tokens)
        instead of raw pixel count from labels.numel().
        """
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = time.perf_counter()
            try:
                input_dict = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            bsz = input_dict["labels"].shape[0]
            ntokens_batch = bsz * self.config.training.max_context_length
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )
            yield input_dict

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, Any] | list[dict[str, Any]],
        global_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform a single forward and backward pass through the model.

        Args:
            input_dict: Dictionary containing model inputs and labels.
            global_valid_tokens: Optional tensor tracking the total number of
                valid tokens across all processes.
                This field is a placeholder for now as we rescale the loss within forward_backward_step for FLUX.

        Returns:
            torch.Tensor: The computed loss value for this training step
        """

        assert (
            global_valid_tokens is None
        ), "FLUX model don't need to rescale loss by number of global valid tokens"
        assert isinstance(input_dict, dict)
        input_dict = dict(input_dict)
        labels = input_dict.pop("labels")

        # generate t5 and clip embeddings
        if self.autoencoder is not None:
            input_dict["image"] = labels
            input_dict = preprocess_data(
                device=self.device,
                dtype=self._dtype,
                autoencoder=self.autoencoder,
                clip_encoder=self.clip_encoder,
                t5_encoder=self.t5_encoder,
                batch=input_dict,
            )
            labels = input_dict["img_encodings"]
        else:
            labels = input_dict["img_encodings"].to(device=self.device, dtype=self._dtype)
            input_dict["clip_encodings"] = input_dict["clip_encodings"].to(device=self.device, dtype=self._dtype)
            input_dict["t5_encodings"] = input_dict["t5_encodings"].to(device=self.device, dtype=self._dtype)

        # rewrite the global_valid_tokens because the `labels` are reset after image encoder.
        local_valid_tokens = torch.tensor(
            labels.numel(), dtype=torch.float32, device=self.device
        )

        if self.parallel_dims.dp_enabled:
            batch_mesh = self.parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum_tensor(
                local_valid_tokens, batch_mesh
            )
        else:
            global_valid_tokens = local_valid_tokens

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        # explicitly convert flux model to be Bfloat16 no matter FSDP is applied or not
        model = self.model_parts[0]

        # image in latent space transformed by self.auto_encoder
        clip_encodings = input_dict["clip_encodings"]
        t5_encodings = input_dict["t5_encodings"]

        bsz = labels.shape[0]

        with torch.no_grad(), torch.device(self.device):
            noise = torch.randn_like(labels)
            timesteps = torch.rand((bsz,), dtype=self._dtype, device=self.device)
            sigmas = timesteps.view(-1, 1, 1, 1)
            latents = (1 - sigmas) * labels + sigmas * noise

        bsz, _, latent_height, latent_width = latents.shape

        POSITION_DIM = 3  # constant for Flux flow model
        with torch.no_grad(), torch.device(self.device):
            # Create positional encodings
            latent_pos_enc = create_position_encoding_for_latents(
                bsz, latent_height, latent_width, POSITION_DIM
            ).to(self._dtype)
            text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM, dtype=self._dtype, device=self.device)

            # Patchify: Convert latent into a sequence of patches
            latents = pack_latents(latents)
            target = pack_latents(noise - labels)

        # Apply CP sharding if enabled
        if self.parallel_dims.cp_enabled:
            from torchtitan.distributed.context_parallel import cp_shard

            (
                latents,
                latent_pos_enc,
                t5_encodings,
                text_pos_enc,
                target,
            ), _ = cp_shard(
                self.parallel_dims.get_mesh("cp"),
                (latents, latent_pos_enc, t5_encodings, text_pos_enc, target),
                None,  # No attention masks for Flux
                load_balancer_type=None,
                input_seq_dims=1,
            )

        # Accumulate after CP sharding so the count reflects the actual
        # unique tokens this rank processes (not the full pre-split sequence).
        self.ntokens_seen += (
            bsz * self.config.training.max_context_length // self.parallel_dims.cp
        )

        with self.train_context():
            annotate_flux_forward_inputs(
                latents=latents,
                latent_pos_enc=latent_pos_enc,
                t5_encodings=t5_encodings,
                text_pos_enc=text_pos_enc,
                target=target,
                clip_encodings=clip_encodings,
                timesteps=timesteps,
            )
            latent_noise_pred = model(
                img=latents,
                img_ids=latent_pos_enc,
                txt=t5_encodings,
                txt_ids=text_pos_enc,
                y=clip_encodings,
                timesteps=timesteps,
            )

            # Scale loss as we used SUM reduction for mse loss function
            loss, _ = self.loss_fn(latent_noise_pred, target, global_valid_tokens)
            # latent_noise_pred.shape=(bs, seq_len, vocab_size)
            # need to free to before bwd to avoid peaking memory
            # pyrefly: ignore[unsupported-delete]
            del (latent_noise_pred, noise, target)
            with spmd.no_typecheck():
                loss.backward()

        return loss

    def train_step(self, data_iterator: Iterator[dict[str, Any]]):
        self.optimizers.zero_grad()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        if self.gradient_accumulation_steps > 1:
            raise ValueError("FLUX doesn't support gradient accumulation for now.")

        input_dict = next(data_iterator)

        loss = self.forward_backward_step(input_dict=input_dict)

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            loss_mesh = parallel_dims.get_optional_mesh("loss")

            # NOTE: the loss returned by train
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_sum(loss, loss_mesh),
                dist_utils.dist_max(loss, loss_mesh),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                ),
            )
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())
            global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )
