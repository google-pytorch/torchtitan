"""Tests for TPU dataset selection in `experiments/tpu/datasets.py`."""

import json
import os
import pickle
import tempfile

from absl.testing import absltest

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.tpu import datasets
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.tools.logging import logger


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



class TPUGrainDataLoaderChatTest(absltest.TestCase):

  _TOKENIZER_PATH = "tests/assets/tokenizer"
  _SFT_DATA_PATH = "tests/assets/sft_test/data.json"

  def _build(self, **config):
    return datasets.TPUGrainDataLoader.Config(**config).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=HuggingFaceTokenizer(tokenizer_path=self._TOKENIZER_PATH),
        max_context_length=128,
        num_tokens_per_batch=128,
    )

  def test_chat_json_trains_on_the_response_only(self) -> None:
    loader = self._build(
        dataset="chat_json",
        dataset_path=self._SFT_DATA_PATH,
        prompt_field="question",
        response_field="answer",
        shuffle=False,
    )
    labels = next(iter(loader))["labels"].flatten().tolist()
    with open(self._SFT_DATA_PATH) as f:
      first_answer = json.load(f)[0]["answer"]
    tokenizer = HuggingFaceTokenizer(tokenizer_path=self._TOKENIZER_PATH)
    answer = tokenizer.encode(first_answer, add_bos=False, add_eos=False)
    # The first document's prompt is masked, so its answer is the first
    # supervised span.
    self.assertEqual(labels[0], IGNORE_INDEX)
    trained = [t for t in labels if t != IGNORE_INDEX]
    self.assertEqual(trained[: len(answer)], answer)

  def test_split_warns_and_is_ignored_for_chat_json(self) -> None:
    with self.assertLogs(logger, level="WARNING") as logs:
      loader = self._build(
          dataset="chat_json",
          dataset_path=self._SFT_DATA_PATH,
          dataset_split="test",
          prompt_field="question",
          response_field="answer",
      )
    self.assertIn("is ignored", "\n".join(logs.output))
    self.assertIn("labels", next(iter(loader)))

  def test_chat_options_warn_for_text_datasets(self) -> None:
    with self.assertLogs(logger, level="WARNING") as logs:
      self._build(dataset="c4_test", prompt_field="question")
    self.assertIn("only apply to chat datasets", "\n".join(logs.output))

if __name__ == "__main__":
  absltest.main()
