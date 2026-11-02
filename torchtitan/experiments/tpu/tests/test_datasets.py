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
        ds.source.load_dataset_kwargs,
        {"data_files": {"train": "/data/train.jsonl"}},
    )
    self.assertEqual(ds.source.split, "train")

  def test_chat_json_file_gets_the_requested_split(self) -> None:
    ds = self._resolve("chat_json", dataset_path="/data/eval.jsonl", split="test")
    self.assertEqual(
        ds.source.load_dataset_kwargs, {"data_files": {"test": "/data/eval.jsonl"}}
    )
    self.assertEqual(ds.source.split, "test")

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
    with open(self._SFT_DATA_PATH) as f:
      first_answer = json.load(f)[0]["answer"]
    tokenizer = HuggingFaceTokenizer(tokenizer_path=self._TOKENIZER_PATH)
    answer = tokenizer.encode(first_answer, add_bos=False, add_eos=False)
    # A local file is read the same way whichever split it is loaded as,
    # e.g. "test" for a validator, without warnings.
    for split in (None, "test"):
      with self.subTest(split=split), self.assertNoLogs(logger, "WARNING"):
        loader = self._build(
            dataset="chat_json",
            dataset_path=self._SFT_DATA_PATH,
            dataset_split=split,
            prompt_field="question",
            response_field="answer",
            shuffle=False,
        )
        labels = next(iter(loader))["labels"].flatten().tolist()
        # The first document's prompt is masked, so its answer is the first
        # supervised span.
        self.assertEqual(labels[0], IGNORE_INDEX)
        trained = [t for t in labels if t != IGNORE_INDEX]
        self.assertEqual(trained[: len(answer)], answer)

  def test_chat_options_warn_for_text_datasets(self) -> None:
    with self.assertLogs(logger, level="WARNING") as logs:
      self._build(dataset="c4_test", prompt_field="question")
    self.assertIn("only apply to chat datasets", "\n".join(logs.output))


class FluxRandomDataloaderTest(absltest.TestCase):

  def test_random_dataloader_config_builds_stateful_loader(self) -> None:
    from torchtitan.models.flux.random_loader import RandomDataloader

    cfg = RandomDataloader.Config(
        bsz=2,
        max_steps=4,
        img_size=256,
        max_t5_len=64,
        t5_dim=128,
        clip_dim=32,
    )
    loader = cfg.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=None,
        max_context_length=320,
        num_tokens_per_batch=640,
    )
    self.assertIsInstance(loader, RandomDataloader)
    self.assertEqual(loader.state_dict(), {})
    loader.load_state_dict({})
    batch = next(iter(loader))
    self.assertEqual(batch["labels"].shape, (2, 16, 32, 32))
    self.assertEqual(batch["t5_encodings"].shape, (2, 64, 128))
    self.assertEqual(batch["clip_encodings"].shape, (2, 32))

  def test_flux_trainer_random_dataset_avoids_hf_hub(self) -> None:
    from unittest import mock
    import datasets as hf_datasets
    from torchtitan.experiments.tpu import train as tpu_train
    from torchtitan.models.flux.config_registry import flux_dev, model_registry
    from torchtitan.models.flux.random_loader import RandomDataloader
    from torchtitan.models.flux.tokenizer import FluxTokenizer
    from torchtitan.models.flux.trainer import FluxTrainer
    from torchtitan.trainer import Trainer

    config = flux_dev()
    config.model_spec = model_registry("flux-test")
    config.flux.random_dataset = True
    config.flux.img_size = 1024
    config.training.num_tokens_per_microbatch_per_dp_rank = 9216
    config.training.steps = 2
    tpu_train._apply_flux_img_size(config)

    def _fake_super_init(trainer_self, cfg):
      trainer_self.tokenizer = cfg.tokenizer.build(
          tokenizer_path=cfg.hf_assets_path
      )
      trainer_self.dataloader = cfg.dataloader.build(
          dp_world_size=1,
          dp_rank=0,
          tokenizer=trainer_self.tokenizer,
          max_context_length=cfg.training.max_context_length,
          num_tokens_per_batch=(
              cfg.training.num_tokens_per_microbatch_per_dp_rank
          ),
      )
      trainer_self.parallel_dims = mock.MagicMock(dp_shard_enabled=False)
      trainer_self.device = "cpu"

    with (
        mock.patch.object(Trainer, "__init__", _fake_super_init),
        mock.patch.object(
            hf_datasets,
            "load_dataset",
            side_effect=AssertionError("Should not call datasets.load_dataset"),
        ),
        mock.patch.object(
            FluxTokenizer,
            "__init__",
            side_effect=AssertionError("Should not call FluxTokenizer.__init__"),
        ),
        mock.patch("torchtitan.distributed.utils.set_determinism"),
    ):
      trainer = FluxTrainer(config)

    self.assertTrue(config.tokenizer.test_mode)
    self.assertIsInstance(trainer.dataloader, RandomDataloader)
    self.assertEqual(trainer.dataloader.bsz, 2)
    self.assertEqual(trainer.dataloader.img_size, 1024)
    self.assertEqual(trainer.dataloader.max_t5_len, 512)
    self.assertIsNone(trainer.autoencoder)
    self.assertIsNone(trainer.clip_encoder)
    self.assertIsNone(trainer.t5_encoder)


if __name__ == "__main__":
  absltest.main()

