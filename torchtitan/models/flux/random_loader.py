from dataclasses import dataclass
import logging
from typing import Any, Iterator

import torch

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.models.flux.utils import IMAGE_LATENT_SIZE_RATIO, LATENT_CHANNELS

logger = logging.getLogger(__name__)


class RandomDataloader(BaseDataLoader):
    """
    A dataloader that generates random encodings.
    Pre-generates a configurable pool of batches and loops them round-robin
    to avoid generation bottlenecks and memory bloat during training.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        bsz: int = 1
        max_steps: int = 1
        img_size: int = 256
        max_t5_len: int = 256
        t5_dim: int = 4096
        clip_dim: int = 768
        max_cached_batches: int = 100

    def __init__(
        self,
        bsz: int | Config = 1,
        max_steps: int = 1,
        img_size: int = 256,
        max_t5_len: int = 256,
        t5_dim: int = 4096,
        clip_dim: int = 768,
        max_cached_batches: int = 100,
        *,
        config: Config | None = None,
        **kwargs: Any,
    ):
        del kwargs
        if isinstance(bsz, RandomDataloader.Config):
            config = bsz
        if config is not None:
            bsz = config.bsz
            max_steps = config.max_steps
            img_size = config.img_size
            max_t5_len = config.max_t5_len
            t5_dim = config.t5_dim
            clip_dim = config.clip_dim
            max_cached_batches = config.max_cached_batches

        self.bsz = bsz
        self.max_steps = max_steps
        self.img_size = img_size
        self.max_t5_len = max_t5_len
        self.t5_dim = t5_dim
        self.clip_dim = clip_dim

        self.batches = []
        h_latent = self.img_size // IMAGE_LATENT_SIZE_RATIO
        w_latent = self.img_size // IMAGE_LATENT_SIZE_RATIO

        # Cap the pool size based on the parameter.
        num_samples = min(self.max_steps, max_cached_batches)

        for _ in range(num_samples):
            img_encodings = torch.randn((self.bsz, LATENT_CHANNELS, h_latent, w_latent))
            t5_encodings = torch.randn((self.bsz, self.max_t5_len, self.t5_dim))
            clip_encodings = torch.randn((self.bsz, self.clip_dim))

            data = {
                "img_encodings": img_encodings,
                "t5_encodings": t5_encodings,
                "clip_encodings": clip_encodings,
            }
            self.batches.append((data, img_encodings))

    def __iter__(self) -> Iterator:
        step = 0
        num_batches = len(self.batches)

        while True:
            if step > 0 and step % num_batches == 0:
                logger.warning("RandomDataset is being re-looped.")

            data, img_enc = self.batches[step % num_batches]

            # Yield a dictionary copy so inplace `.to(device)` calls in
            # trainer.py don't overwrite the cached CPU tensors in self.batches,
            # which grows memory usage and can cause TPU OOM.
            # FluxTrainer expects a single TrainerBatch dict keyed by "labels".
            data_copy = {
                "labels": img_enc,
                "img_encodings": data["img_encodings"],
                "t5_encodings": data["t5_encodings"],
                "clip_encodings": data["clip_encodings"],
            }
            yield data_copy
            step += 1

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict

    def close(self) -> None:
        """No-op; called by Trainer.close()."""

