"""Tests for TPU dataset selection in `experiments/tpu/datasets.py`."""

import os
import tempfile

from absl.testing import absltest
import grain.python as grain
import numpy as np

from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.experiments.tpu import datasets
from torchtitan.hf_datasets.text_datasets import DATASETS


class TextDatasetTest(absltest.TestCase):

  def test_c4_test_default_uses_repo_asset(self) -> None:
    ds = datasets._text_dataset("c4_test", None)
    self.assertIs(ds, DATASETS["c4_test"])
    data_files = ds.source.load_dataset_kwargs["data_files"]
    self.assertEqual(
        data_files, "tests/assets/c4_test/data.json"
    )

  def test_c4_keeps_upstream_source(self) -> None:
    ds = datasets._text_dataset("c4", None)
    self.assertEqual(ds, DATASETS["c4"])
    self.assertNotIn("data_files", ds.source.load_dataset_kwargs)

  def test_c4_dataset_path_replaces_source_path(self) -> None:
    ds = datasets._text_dataset("c4", "/data/c4_mirror")
    upstream = DATASETS["c4"]
    self.assertEqual(ds.source.path, "/data/c4_mirror")
    self.assertEqual(ds.source.name, upstream.source.name)
    self.assertEqual(
        ds.source.load_dataset_kwargs, upstream.source.load_dataset_kwargs
    )

  def test_c4_test_dataset_path_merges_kwargs(self) -> None:
    with tempfile.TemporaryDirectory() as d:
      ds = datasets._text_dataset("c4_test", d)
      self.assertEqual(
          ds.source.load_dataset_kwargs["data_files"],
          os.path.join(d, "data.json"),
      )
    ds = datasets._text_dataset("c4_test", "/tmp/custom.json")
    self.assertEqual(
        ds.source.load_dataset_kwargs["data_files"], "/tmp/custom.json"
    )
    self.assertEqual(ds.source.path, DATASETS["c4_test"].source.path)


class _FakeHfTokenizer:
  """The ESM-C vocabulary layout: specials, 25 amino acids, <mask> last."""

  vocab = {"<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3, ".": 29, "<mask>": 32}
  vocab.update({a: 4 + i for i, a in enumerate(datasets._ESM_AMINO_ACIDS)})

  def token_to_id(self, token: str) -> int | None:
    return self.vocab.get(token)


class _FakeTokenizer:
  bos_id, eos_id = 0, 2
  tokenizer = _FakeHfTokenizer()

  def encode(self, text: str, add_bos: bool, add_eos: bool) -> list[int]:
    ids = [self.tokenizer.token_to_id(c) for c in text]
    return [self.bos_id] * add_bos + ids + [self.eos_id] * add_eos


class MLMProcessorTest(absltest.TestCase):

  def _processor(self, max_len: int = 16) -> datasets.MLMProcessor:
    context = DatasetBuildContext(
        tokenizer=_FakeTokenizer(),
        max_context_length=max_len,
        num_tokens_per_batch=max_len,
        read_options=grain.ReadOptions(),
    )
    return datasets.MLMProcessor.Config().build(context=context)

  def test_masks_only_amino_acids_and_keeps_bos_eos(self) -> None:
    proc = self._processor(max_len=64)
    text = "LAGVSERTIDPKQNFYMHWC"
    row = proc({"text": text}, np.random.default_rng(0))
    original = np.asarray(_FakeTokenizer().encode(text, True, True))
    self.assertLen(row.input_ids, len(original))
    targets = row.labels != IGNORE_INDEX
    self.assertTrue(targets.any())
    self.assertFalse(targets[0] or targets[-1])
    np.testing.assert_array_equal(row.labels[targets], original[targets])
    np.testing.assert_array_equal(row.input_ids[~targets], original[~targets])

  def test_long_sequence_is_cropped_to_context(self) -> None:
    row = self._processor(max_len=16)({"text": "LAGV" * 20}, np.random.default_rng(0))
    self.assertLen(row.input_ids, 16)
    self.assertEqual((row.input_ids[0], row.input_ids[-1]), (0, 2))

  def test_empty_and_unmaskable_rows_are_dropped(self) -> None:
    proc = self._processor()
    self.assertIsNone(proc({"text": ""}, np.random.default_rng(0)))
    self.assertIsNone(proc({"text": "."}, np.random.default_rng(0)))


class _ListIterDataset(grain.IterDataset):
  """A checkpointable parent over an in-memory list of ``TextSequence``."""

  def __init__(self, rows):
    super().__init__()
    self._rows = rows

  def __iter__(self):
    return _ListIterator(self._rows)


class _ListIterator(grain.DatasetIterator):

  def __init__(self, rows):
    super().__init__()
    self._rows, self._i = rows, 0

  def __next__(self):
    if self._i == len(self._rows):
      raise StopIteration
    self._i += 1
    return self._rows[self._i - 1]

  def get_state(self):
    return {"i": self._i}

  def set_state(self, state):
    self._i = state["i"]


def _docs(lengths):
  return [
      TextSequence(input_ids=np.full(n, i + 1, dtype=np.int64), labels=np.full(n, i + 1, dtype=np.int64))
      for i, n in enumerate(lengths)
  ]


class RowAlignedPackingTest(absltest.TestCase):

  CTX = 8

  def _batches(self, lengths, num_rows=2):
    ds = datasets._RowAlignedPackingIterDataset(
        _ListIterDataset(_docs(lengths)),
        max_context_length=self.CTX,
        num_tokens_per_batch=self.CTX * num_rows,
    )
    return list(ds)

  def test_documents_never_cross_a_row_boundary(self) -> None:
    lengths = [5, 4, 3, 6, 2, 7, 1, 8, 5]
    for batch in self._batches(lengths):
      self.assertLen(batch.input_ids, 2 * self.CTX)
      for row in range(2):
        sl = slice(row * self.CTX, (row + 1) * self.CTX)
        ids, pos, pad = batch.input_ids[sl], batch.positions[sl], batch.padding_mask[sl]
        # Every row starts a document and padding sits at its end only.
        self.assertEqual(pos[0], 0)
        self.assertTrue(np.all(np.diff(pad.astype(int)) >= 0))
        # A document id never continues past a position reset.
        for t in range(1, self.CTX):
          if pos[t] == 0:
            self.assertNotEqual(ids[t], ids[t - 1]) if not pad[t] else None
    # Every token of every document is emitted exactly once, in order.
    emitted = np.concatenate([b.input_ids[~b.padding_mask] for b in self._batches(lengths)])
    expected = np.concatenate([d.input_ids for d in _docs(lengths)])
    self.assertEqual(sorted(emitted.tolist()), sorted(expected.tolist()))

  def test_long_document_is_chunked_to_the_context(self) -> None:
    (batch,) = self._batches([20], num_rows=3)
    self.assertEqual(batch.input_ids.tolist(), [1] * 20 + [0] * 4)
    self.assertEqual(batch.positions[:16].tolist(), list(range(8)) * 2)

  def test_padding_is_ignored_by_the_loss(self) -> None:
    (batch,) = self._batches([3], num_rows=1)
    self.assertEqual(batch.labels.tolist(), [1, 1, 1] + [IGNORE_INDEX] * 5)
    self.assertEqual(batch.padding_mask.tolist(), [False] * 3 + [True] * 5)

  def test_state_round_trip_resumes_mid_document(self) -> None:
    ds = datasets._RowAlignedPackingIterDataset(
        _ListIterDataset(_docs([6, 20, 4])), max_context_length=self.CTX, num_tokens_per_batch=self.CTX
    )
    it = iter(ds)
    first = next(it)
    state = it.get_state()  # the 20-token document is in flight
    rest = [b.input_ids.tolist() for b in it]
    it2 = iter(ds)
    it2.set_state(state)
    self.assertEqual([b.input_ids.tolist() for b in it2], rest)
    self.assertEqual(first.input_ids.tolist(), [1] * 6 + [0, 0])


if __name__ == "__main__":
  absltest.main()
