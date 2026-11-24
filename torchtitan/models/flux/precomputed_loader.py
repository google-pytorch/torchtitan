from dataclasses import dataclass
import glob
import os
from typing import Any, Iterator

import torch

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import TrainingMicrobatch


@dataclass(kw_only=True, slots=True)
class FluxEncodedMicrobatch(TrainingMicrobatch):
    """One DP-local Flux microbatch whose encodings are already computed.

    The precomputed and random dataloaders yield this instead of raw images
    and prompts; ``FluxModel.preprocess_inputs`` consumes the encodings as is
    when the trainer has no encoders. ``labels`` aliases ``img_encodings``.
    """

    labels: torch.Tensor
    img_encodings: torch.Tensor
    t5_encodings: torch.Tensor
    clip_encodings: torch.Tensor
    num_valid_tokens: int

    @classmethod
    def from_encodings(
        cls,
        *,
        img_encodings: torch.Tensor,
        t5_encodings: torch.Tensor,
        clip_encodings: torch.Tensor,
    ) -> "FluxEncodedMicrobatch":
        return cls(
            labels=img_encodings,
            img_encodings=img_encodings,
            t5_encodings=t5_encodings,
            clip_encodings=clip_encodings,
            # As FluxCollator: one "token" per latent element.
            num_valid_tokens=img_encodings.numel(),
        )

    def as_input_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels,
            "img_encodings": self.img_encodings,
            "t5_encodings": self.t5_encodings,
            "clip_encodings": self.clip_encodings,
        }

    def to_input_dict(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> dict[str, Any]:
        # `labels` and `img_encodings` are one tensor: copy it to the device
        # once and keep the alias.
        inputs = super(FluxEncodedMicrobatch, self).to_input_dict(
            device, non_blocking=non_blocking
        )
        if self.labels is self.img_encodings:
            inputs["labels"] = inputs["img_encodings"]
        return inputs


class PrecomputedDataloader(BaseDataLoader):
    """
    A dataloader that loads precomputed batches from a directory.

    This dataloader looks for files named `batch_*_rank_{rank}.pt` or `batch_*.pt`
    in the given data directory. It loads them in numerical order of batch index
    and yields each batch and its image encodings up to `max_steps`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        data_dir: str = ""
        max_steps: int = 1
        rank: int = 0

    def __init__(
        self,
        data_dir: str | Config = "",
        max_steps: int = 1,
        rank: int = 0,
        *,
        config: Config | None = None,
        **kwargs: Any,
    ):
        del kwargs
        if isinstance(data_dir, PrecomputedDataloader.Config):
            config = data_dir
        if config is not None:
            data_dir = config.data_dir
            max_steps = config.max_steps
            rank = config.rank

        self.data_dir = data_dir
        self.max_steps = max_steps
        self.rank = rank

        # Look for rank-specific files first
        self.files = glob.glob(os.path.join(data_dir, f"batch_*_rank_{rank}.pt"))
        if len(self.files) == 0:
            # Fall back to standalone files if rank specific don't exist
            self.files = glob.glob(os.path.join(data_dir, "batch_*.pt"))
            self.files = [f for f in self.files if "rank" not in f]

        if len(self.files) == 0:
            raise ValueError(f"No precomputed batches found in {data_dir} for rank {rank}")

        # Sort numerically by batch index
        self.files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

    def __iter__(self) -> Iterator:
        step = 0
        while True:
            for file_path in self.files:
                if step >= self.max_steps:
                    break
                data = torch.load(file_path, weights_only=False)
                # Upstream ec953b36: the training loop consumes
                # TrainingMicrobatch objects.
                yield FluxEncodedMicrobatch.from_encodings(
                    img_encodings=data["img_encodings"],
                    t5_encodings=data["t5_encodings"],
                    clip_encodings=data["clip_encodings"],
                )
                step += 1
            if step >= self.max_steps:
                raise RuntimeError(f"Reached max_steps={self.max_steps} while iterating through precomputed batches. Please ensure that the number of precomputed batches is at least max_steps.")

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict

    def close(self) -> None:
        """No-op; called by Trainer.close()."""

