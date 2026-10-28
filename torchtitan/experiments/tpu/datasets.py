"""Grain dataset/dataloader configs shared by the TPU recipes.

Upstream d771abb0 migrated the data stack to Grain and deleted
`HuggingFaceTextDataLoader`, which every TPU `config_registry.py` used as
`HuggingFaceTextDataLoader.Config(dataset="c4_test")`. `TPUGrainDataLoader`
keeps that spelling (and the `--dataloader.dataset` /
`--dataloader.dataset_path` flags) on top of upstream's `GrainDataLoader`.
"""

import dataclasses
import os
from typing import Any

# Imported from the defining submodules rather than the
# `torchtitan.components.data` package: strict-deps checking cannot tell a
# package attribute from an unlisted submodule, so the package-level form
# is reported as an ambiguous import.
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
from torchtitan.hf_datasets.text_datasets import ChatProcessor
from torchtitan.hf_datasets.text_datasets import DATASETS

__all__ = [
    "CHAT_DATASETS",
    "FieldsToChat",
    "TPUGrainDataLoader",
    "c4_test_dataloader",
]


def _text_dataset(name: str, dataset_path: str | None) -> SingleDatasetConfig:
  """Resolves a registered text dataset, applying `dataset_path` if given.

  For `c4_test` (a local json file), `dataset_path` selects the file; a
  directory or non-`.json` path resolves to `<path>/data.json`. For every other
  entry, `dataset_path` replaces `source.path` (e.g. a local mirror of
  `allenai/c4`).

  Args:
    name: Key in upstream `DATASETS`.
    dataset_path: Optional `--dataloader.dataset_path` override.

  Returns:
    The resolved dataset config.
  """
  base_ds = DATASETS[name]
  if not dataset_path:
    return base_ds
  if name != "c4_test":
    return dataclasses.replace(
        base_ds, source=dataclasses.replace(base_ds.source, path=dataset_path)
    )
  if os.path.isdir(dataset_path) or not dataset_path.endswith(".json"):
    dataset_path = os.path.join(dataset_path, "data.json")
  return dataclasses.replace(
      base_ds,
      source=dataclasses.replace(
          base_ds.source,
          load_dataset_kwargs={
              **base_ds.source.load_dataset_kwargs,
              "data_files": dataset_path,
          },
      ),
  )



@dataclasses.dataclass(frozen=True)
class FieldsToChat:
  """Turns a record into a single-turn chat from two of its fields.

  A `ChatProcessor` `messages_fn` that is set from field names, so single-turn
  datasets need no Python code.
  """

  prompt_field: str
  response_field: str

  def __call__(self, sample: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": sample[self.prompt_field]},
        {"role": "assistant", "content": sample[self.response_field]},
    ]


@dataclasses.dataclass(frozen=True)
class _ChatDataset:
  """A single-turn chat dataset: a Hugging Face source and its two fields."""

  path: str
  name: str | None = None
  prompt_field: str = "prompt"
  response_field: str = "response"


CHAT_DATASETS: dict[str, _ChatDataset] = {
    # Grade-school math word problems with worked solutions; train/test splits.
    "gsm8k": _ChatDataset(
        path="openai/gsm8k",
        name="main",
        prompt_field="question",
        response_field="answer",
    ),
    # A local JSON or JSON Lines file, given by `dataset_path`.
    "chat_json": _ChatDataset(path="json"),
}
"""Single-turn chat datasets for fine-tuning, by `--dataloader.dataset` name."""


def _is_not_none(sample: Any) -> bool:
  return sample is not None


def _chat_dataset(
    name: str,
    *,
    dataset_path: str | None,
    split: str | None,
    prompt_field: str | None,
    response_field: str | None,
) -> SingleDatasetConfig:
  """Resolves a registered chat dataset, applying the loader's overrides.

  Args:
    name: Key in `CHAT_DATASETS`.
    dataset_path: For `chat_json`, the JSON/JSON Lines file (required); for
      other entries, replaces the Hugging Face path (e.g. a local mirror).
    split: Dataset split; defaults to "train". Ignored for local JSON files
      (`chat_json`): the file is the split, so point `dataset_path` at a
      separate file for evaluation.
    prompt_field: Overrides the entry's prompt field.
    response_field: Overrides the entry's response field.

  Returns:
    The dataset config, tokenized with `ChatProcessor` so that the loss covers
    the response only.
  """
  entry = CHAT_DATASETS[name]
  path, load_dataset_kwargs = entry.path, {}
  if entry.path == "json":
    if not dataset_path:
      raise ValueError(
          f"--dataloader.dataset={name} needs --dataloader.dataset_path"
      )
    load_dataset_kwargs = {"data_files": dataset_path}
  elif dataset_path:
    path = dataset_path
  return SingleDatasetConfig(
      source=HuggingFaceRandomAccessSource.Config(
          path=path,
          name=entry.name,
          # Hugging Face's json loader names a single file's rows "train".
          split="train" if entry.path == "json" else (split or "train"),
          load_dataset_kwargs=load_dataset_kwargs,
      ),
      processor=ChatProcessor.Config(
          messages_fn=FieldsToChat(
              prompt_field=prompt_field or entry.prompt_field,
              response_field=response_field or entry.response_field,
          )
      ),
      post_filters=(_is_not_none,),
  )

class TPUGrainDataLoader(GrainDataLoader):
  """GrainDataLoader wrapper preserving TPU CLI flags (`dataset`, `dataset_path`)."""

  @dataclasses.dataclass(kw_only=True, slots=True)
  class Config(GrainDataLoader.Config):
    dataset: str = "c4_test"  # pytype: disable=signature-mismatch
    dataset_path: str | None = None

  def __init__(self, config: Config, **kwargs: Any) -> None:
    if isinstance(config.dataset, str):
      if config.dataset in DATASETS:
        ds_cfg = ConcatThenSplitPackingConfig(
            dataset=_text_dataset(config.dataset, config.dataset_path)
        )
      else:
        from torchtitan.models.flux.flux_datasets import (  # pylint: disable=g-import-not-at-top
            DATASETS as FLUX_DATASETS,
            FluxSampleProcessor,
        )

        base_ds = FLUX_DATASETS[config.dataset]
        src = base_ds.source
        if config.dataset_path:
          src = dataclasses.replace(src, path=config.dataset_path)
        proc = base_ds.processor
        if isinstance(proc, FluxSampleProcessor.Config) and proc.img_size is None:
          proc = dataclasses.replace(proc, img_size=256)
        ds_cfg = dataclasses.replace(base_ds, source=src, processor=proc)
      config = dataclasses.replace(config, dataset=ds_cfg)  # pytype: disable=wrong-arg-types
    super().__init__(config, **kwargs)


def c4_test_dataloader(**kwargs: Any) -> TPUGrainDataLoader.Config:
  """Builds the packed `c4_test` dataloader used by the TPU test recipes.

  Args:
    **kwargs: Forwarded to `TPUGrainDataLoader.Config`; use this to override
      defaults such as `shuffle` or `seed`.

  Returns:
    A dataloader config that concatenates then splits `c4_test` into
    fixed-length sequences, while keeping `--dataloader.dataset` and
    `--dataloader.dataset_path` CLI-configurable for the TPU test harnesses.
  """
  kwargs.setdefault("shuffle", False)
  return TPUGrainDataLoader.Config(**kwargs)

