"""Grain dataset/dataloader configs shared by the TPU recipes.

Upstream d771abb0 migrated the data stack to Grain and deleted
`HuggingFaceTextDataLoader`, which every TPU `config_registry.py` used as
`HuggingFaceTextDataLoader.Config(dataset="c4_test")`. `TPUGrainDataLoader`
keeps that spelling (and the `--dataloader.dataset` /
`--dataloader.dataset_path` flags) on top of upstream's `GrainDataLoader`.

The google3 location of the `c4_test` asset is handled by copybara, which
rewrites the literal in `hf_datasets/text_datasets.py`, so every dataset here
is upstream's `DATASETS` entry unchanged.
"""

import dataclasses
import os
from typing import Any

import grain.python as grain
import numpy as np

# Imported from the defining submodules rather than the
# `torchtitan.components.data` package: go/py-strict-deps cannot tell a
# package attribute from an unlisted submodule, so the package-level form
# is reported as an ambiguous import.
from torchtitan.components.data.dataset import (
    DatasetConfig,
    SampleProcessor,
    SingleDatasetConfig,
    TextSequence,
)
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.sources import HuggingFaceStreamingSource
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.hf_datasets.text_datasets import DATASETS

__all__ = [
    "MLMProcessor",
    "RowAlignedPackingConfig",
    "TPUGrainDataLoader",
    "c4_test_dataloader",
]


# The ESM / ESM-C amino-acid alphabet (20 standard + X, B, U, Z, O), in
# vocabulary order. These are the only tokens masked or used as replacements.
_ESM_AMINO_ACIDS = "LAGVSERTIDPKQNFYMHWCXBUZO"


class MLMProcessor(SampleProcessor):
  """BERT/ESM-style masked-language-modeling rows for bidirectional encoders.

  ``TextProcessor`` emits next-token pairs, which a bidirectional model (ESM,
  ESM-C, BERT) solves by reading the answer from the token to its right. This
  processor instead masks a fraction of the amino-acid tokens and asks only for
  those: ``input_ids`` carries the corruption, ``labels`` the original token at
  the masked positions and ``IGNORE_INDEX`` elsewhere, with no shift.

  Defaults follow ESM-2 / ESM-C pretraining: ``r_mask = 0.15``; of the chosen
  positions 80% become ``<mask>``, 10% a random amino acid, 10% stay. BOS/EOS
  are never masked. Grain's per-sample ``rng`` makes the corruption
  deterministic under the dataloader seed.
  """

  @dataclasses.dataclass(kw_only=True, slots=True)
  class Config(SampleProcessor.Config):
    text_key: str = "text"
    mask_rate: float = 0.15
    mask_token_frac: float = 0.8
    random_token_frac: float = 0.1

  def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
    tokenizer = context.tokenizer
    hf_tokenizer = getattr(tokenizer, "tokenizer", None)
    mask_id = (
        hf_tokenizer.token_to_id("<mask>") if hf_tokenizer is not None else None
    )
    if mask_id is None or tokenizer.bos_id is None or tokenizer.eos_id is None:
      raise ValueError("MLMProcessor needs a tokenizer with <mask>, BOS and EOS")
    self._config = config
    self._tokenizer = tokenizer
    self._mask_id = int(mask_id)
    self._amino_acid_ids = np.array([
        i for i in map(hf_tokenizer.token_to_id, _ESM_AMINO_ACIDS) if i is not None
    ])
    self._max_len = context.max_context_length

  def __call__(
      self, sample: dict[str, Any], rng: np.random.Generator
  ) -> TextSequence | None:
    text = sample[self._config.text_key]
    if not text:
      return None
    ids = np.asarray(
        self._tokenizer.encode(text, add_bos=True, add_eos=True), dtype=np.int64
    )
    ids = self._crop(ids, rng)
    maskable = np.isin(ids, self._amino_acid_ids)
    if not maskable.any():
      return None
    inputs, labels = self._corrupt(ids, maskable, rng)
    return TextSequence(input_ids=inputs, labels=labels)

  def _crop(self, ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random window of the body, keeping BOS/EOS, when over the context."""
    body_len = self._max_len - 2
    if len(ids) - 2 <= body_len:
      return ids
    start = 1 + int(rng.integers(0, len(ids) - 2 - body_len + 1))
    return np.concatenate([ids[:1], ids[start : start + body_len], ids[-1:]])

  def _corrupt(
      self, ids: np.ndarray, maskable: np.ndarray, rng: np.random.Generator
  ) -> tuple[np.ndarray, np.ndarray]:
    """80/10/10 corruption with at least one target per row."""
    cfg = self._config
    chosen = (rng.random(len(ids)) < cfg.mask_rate) & maskable
    if not chosen.any():
      chosen[rng.choice(np.flatnonzero(maskable))] = True
    labels = np.where(chosen, ids, IGNORE_INDEX)
    u = rng.random(len(ids))
    inputs = np.where(chosen & (u < cfg.mask_token_frac), self._mask_id, ids)
    random_hi = cfg.mask_token_frac + cfg.random_token_frac
    randomized = chosen & (u >= cfg.mask_token_frac) & (u < random_hi)
    inputs[randomized] = rng.choice(self._amino_acid_ids, size=int(randomized.sum()))
    return inputs, labels


def _is_not_none(sample: Any) -> bool:
  return sample is not None


# UniRef50 (HF mirror `bloyal/uniref50`: parquet with a `text` column holding
# the sequence), the redistributable part of the ESM-C pretraining mix, as
# masked-LM rows. Packed like every other dataset here, so document segment ids
# keep each protein's attention to itself.
DATASETS.setdefault(
    "uniref50",
    SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(path="bloyal/uniref50", split="train"),
        processor=MLMProcessor.Config(),
        post_filters=(_is_not_none,),
    ),
)



@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class RowAlignedPackingConfig:
  """Packs whole documents into ``max_context_length`` rows, padding the rest.

  ``ConcatThenSplitPackingConfig`` lays documents back to back across the whole
  ``num_tokens_per_batch`` stream, so a document regularly crosses a multiple
  of ``max_context_length``. A model that runs the stream as ``[B, S]`` rows
  (``TPUHFTransformerModel.forward`` does, for bidirectional encoders) then
  sees that document as two halves that cannot attend to each other. This
  packer never crosses a row boundary: each row holds whole documents (chunks
  of ``max_context_length`` for longer ones), filled first-fit across the
  ``num_tokens_per_batch // max_context_length`` rows of a batch, and the
  unfilled tail of every row is padding (``IGNORE_INDEX`` labels, positions
  restarting at 0 so it forms its own segment). Costs a few percent of tokens
  as padding; with ~300-token proteins in 2048-token rows about 2%.
  """

  dataset: DatasetConfig

  def build(
      self,
      *,
      context: DatasetBuildContext,
      dataset_iteration_policy: DatasetIterationPolicy,
  ) -> grain.IterDataset:
    dataset = self.dataset.build(
        context=context, dataset_iteration_policy=dataset_iteration_policy
    )
    if isinstance(dataset, grain.MapDataset):
      dataset = dataset.to_iter_dataset(read_options=context.read_options)
    return _RowAlignedPackingIterDataset(
        dataset,
        max_context_length=context.max_context_length,
        num_tokens_per_batch=context.num_tokens_per_batch,
    )


class _RowAlignedPackingIterDataset(grain.IterDataset):

  def __init__(
      self, parent: grain.IterDataset, *, max_context_length: int, num_tokens_per_batch: int
  ) -> None:
    super().__init__(parent)
    if num_tokens_per_batch % max_context_length:
      raise ValueError(
          f"num_tokens_per_batch={num_tokens_per_batch} must be a multiple of "
          f"max_context_length={max_context_length} to pack whole rows"
      )
    self._max_context_length = max_context_length
    self._num_rows = num_tokens_per_batch // max_context_length

  def __iter__(self) -> grain.DatasetIterator:
    return _RowAlignedPackingIterator(
        iter(self._parent),
        max_context_length=self._max_context_length,
        num_rows=self._num_rows,
    )


class _RowAlignedPackingIterator(grain.DatasetIterator):
  """First-fit of document chunks into ``num_rows`` rows per emitted batch.

  A document the parent yields is consumed in chunks of at most
  ``max_context_length`` tokens; a chunk goes to the first row with room, and
  a batch is closed when a chunk fits in no row (it opens the next batch) or
  the parent is exhausted. Checkpoint state is the parent's state before the
  document in flight plus the offset into it, as in
  ``torchtitan.components.data.packing``.
  """

  def __init__(
      self, parent: grain.DatasetIterator, *, max_context_length: int, num_rows: int
  ) -> None:
    super().__init__(parent)
    self._max_context_length = max_context_length
    self._num_rows = num_rows
    self._pending: TextSequence | None = None
    self._pending_parent_state: dict[str, Any] | None = None
    self._pending_offset = 0
    self._finished = False

  def _next_pending(self) -> bool:
    """Fetches the next non-empty document into ``_pending``; False at end."""
    while self._pending is None:
      parent_state = self._parent.get_state()
      try:
        sequence = next(self._parent)
      except StopIteration:
        self._finished = True
        return False
      if len(sequence.input_ids):
        self._pending = sequence
        self._pending_parent_state = parent_state
        self._pending_offset = 0
    return True

  def __next__(self) -> TextSequence:
    self._assert_not_closed()
    if self._finished:
      raise StopIteration
    ctx = self._max_context_length
    rows: list[list[TextSequence]] = [[] for _ in range(self._num_rows)]
    used = [0] * self._num_rows
    while self._next_pending():
      doc = self._pending
      assert doc is not None
      chunk_len = min(len(doc.input_ids) - self._pending_offset, ctx)
      row = next((r for r in range(self._num_rows) if used[r] + chunk_len <= ctx), None)
      if row is None:
        break
      chunk = slice(self._pending_offset, self._pending_offset + chunk_len)
      rows[row].append(
          TextSequence(input_ids=np.asarray(doc.input_ids[chunk]),
                       labels=np.asarray(doc.labels[chunk]))
      )
      used[row] += chunk_len
      self._pending_offset += chunk_len
      if self._pending_offset == len(doc.input_ids):
        self._pending = None
        self._pending_parent_state = None
        self._pending_offset = 0
    if not any(used):
      raise StopIteration
    return self._assemble(rows, used)

  def _assemble(self, rows: list[list[TextSequence]], used: list[int]) -> TextSequence:
    ctx = self._max_context_length
    input_ids, labels, positions, padding = [], [], [], []
    for row, n_used in zip(rows, used):
      for chunk in row:
        input_ids.append(chunk.input_ids)
        labels.append(chunk.labels)
        positions.append(np.arange(len(chunk.input_ids), dtype=np.int64))
        padding.append(np.zeros(len(chunk.input_ids), dtype=np.bool_))
      pad = ctx - n_used
      input_ids.append(np.zeros(pad, dtype=np.int64))
      labels.append(np.full(pad, IGNORE_INDEX, dtype=np.int64))
      positions.append(np.arange(pad, dtype=np.int64))
      padding.append(np.ones(pad, dtype=np.bool_))
    return TextSequence(
        input_ids=np.concatenate(input_ids),
        labels=np.concatenate(labels),
        positions=np.concatenate(positions),
        padding_mask=np.concatenate(padding),
    )

  def get_state(self) -> dict[str, Any]:
    parent_state = (
        self._parent.get_state() if self._pending is None else self._pending_parent_state
    )
    return {
        "parent": parent_state,
        "has_pending": self._pending is not None,
        "pending_offset": self._pending_offset,
        "finished": self._finished,
    }

  def set_state(self, state: dict[str, Any]) -> None:
    self._parent.set_state(state["parent"])
    self._pending = None
    self._pending_parent_state = None
    self._pending_offset = 0
    self._finished = state["finished"]
    if state["has_pending"]:
      self._pending_parent_state = state["parent"]
      self._pending = next(self._parent)
      self._pending_offset = state["pending_offset"]


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


class TPUGrainDataLoader(GrainDataLoader):
  """GrainDataLoader wrapper preserving TPU CLI flags (`dataset`, `dataset_path`)."""

  @dataclasses.dataclass(kw_only=True, slots=True)
  class Config(GrainDataLoader.Config):
    dataset: str = "c4_test"  # pytype: disable=signature-mismatch
    dataset_path: str | None = None
    # Keep every document inside one ``max_context_length`` row
    # (``RowAlignedPackingConfig``) instead of concatenating across rows.
    pack_rows_to_context: bool = False

  def __init__(self, config: Config, **kwargs: Any) -> None:
    if isinstance(config.dataset, str):
      if config.dataset in DATASETS:
        packing = (
            RowAlignedPackingConfig if config.pack_rows_to_context
            else ConcatThenSplitPackingConfig
        )
        ds_cfg = packing(dataset=_text_dataset(config.dataset, config.dataset_path))
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

