"""Tests for TPU dataset selection in `experiments/tpu/datasets.py`."""

import os
import pickle
import tempfile

from absl.testing import absltest

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


class FieldsToChatTest(absltest.TestCase):

  def test_builds_single_turn_chat(self) -> None:
    to_chat = datasets.FieldsToChat(prompt_field="q", response_field="a")
    self.assertEqual(
        to_chat({"q": "What is 2 + 3?", "a": "5", "other": "ignored"}),
        [
            {"role": "user", "content": "What is 2 + 3?"},
            {"role": "assistant", "content": "5"},
        ],
    )

  def test_survives_pickling(self) -> None:
    to_chat = datasets.FieldsToChat(prompt_field="q", response_field="a")
    self.assertEqual(pickle.loads(pickle.dumps(to_chat)), to_chat)


class ChatDatasetTest(absltest.TestCase):

  def _resolve(self, name, **overrides):
    kwargs = dict(
        dataset_path=None, split=None, prompt_field=None, response_field=None
    )
    kwargs.update(overrides)
    return datasets._chat_dataset(name, **kwargs)

  def test_gsm8k_defaults(self) -> None:
    ds = self._resolve("gsm8k")
    self.assertEqual(ds.source.path, "openai/gsm8k")
    self.assertEqual(ds.source.name, "main")
    self.assertEqual(ds.source.split, "train")
    self.assertEqual(
        ds.processor.messages_fn,
        datasets.FieldsToChat(prompt_field="question", response_field="answer"),
    )

  def test_overrides(self) -> None:
    ds = self._resolve(
        "gsm8k",
        dataset_path="/mirror/gsm8k",
        split="test",
        prompt_field="prompt",
        response_field="completion",
    )
    self.assertEqual(ds.source.path, "/mirror/gsm8k")
    self.assertEqual(ds.source.split, "test")
    self.assertEqual(
        ds.processor.messages_fn,
        datasets.FieldsToChat(prompt_field="prompt", response_field="completion"),
    )

  def test_chat_json_reads_the_given_file(self) -> None:
    ds = self._resolve(
        "chat_json",
        dataset_path="/data/train.jsonl",
        prompt_field="question",
        response_field="answer",
    )
    self.assertEqual(ds.source.path, "json")
    self.assertEqual(
        ds.source.load_dataset_kwargs, {"data_files": "/data/train.jsonl"}
    )
    self.assertEqual(ds.source.split, "train")

  def test_chat_json_reads_the_whole_file_whatever_the_split(self) -> None:
    ds = self._resolve("chat_json", dataset_path="/data/eval.jsonl", split="test")
    self.assertEqual(ds.source.split, "train")

  def test_chat_json_requires_a_path(self) -> None:
    with self.assertRaisesRegex(ValueError, "dataset_path"):
      self._resolve("chat_json")


if __name__ == "__main__":
  absltest.main()
