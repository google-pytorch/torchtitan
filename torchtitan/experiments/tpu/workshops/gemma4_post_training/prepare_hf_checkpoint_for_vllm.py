#!/usr/bin/env python3
"""Prepare a TorchTitan-TorchTPU checkpoint directory for vLLM-TorchTPU serving.

When TorchTitan saves a checkpoint with ``--checkpoint.last_save_in_hf`` (or
``last_save_in_hf=True`` in the config), ``HuggingFaceStorageWriter`` writes the
consolidated model weights (``model.safetensors``) into:
    ``<dump_folder>/checkpoint/step-<N>/``

To load that checkpoint directly into ``vllm-torchtpu`` (or
``AutoModelForCausalLM``) without network calls, ``vllm`` also expects the
Hugging Face metadata and tokenizer assets (``config.json``,
``generation_config.json``, ``tokenizer.json``, ``tokenizer_config.json``,
``special_tokens_map.json``, ``tokenizer.model``) to reside in the same
directory.

This script:
1. Verifies that ``model.safetensors`` (or sharded ``*.safetensors``) exists in
   the TorchTitan ``step-<N>`` checkpoint folder.
2. Copies all tokenizer and config files from ``--hf_assets_path`` into the
   checkpoint directory (or into ``--output_dir`` if specified).
3. Validates that ``transformers.AutoConfig`` and ``AutoTokenizer`` can load the
   resulting directory offline.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

Path = pathlib.Path

METADATA_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
    "vocab.txt",
)


def prepare_checkpoint(
    checkpoint_dir: Path,
    hf_assets_path: Path,
    output_dir: Path | None = None,
) -> Path:
  """Bundle HF config and tokenizer files alongside saved safetensors."""
  if not checkpoint_dir.is_dir():
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
  if not hf_assets_path.is_dir():
    raise FileNotFoundError(f"HF assets directory not found: {hf_assets_path}")

  safetensors_files = list(checkpoint_dir.glob("*.safetensors"))
  if not safetensors_files:
    raise FileNotFoundError(
        f"No .safetensors files found in {checkpoint_dir}. "
        "Ensure TorchTitan ran with --checkpoint.enable "
        "--checkpoint.last_save_model_only --checkpoint.last_save_in_hf."
    )

  target_dir = output_dir if output_dir is not None else checkpoint_dir
  target_dir.mkdir(parents=True, exist_ok=True)

  # If exporting to a separate output_dir, link or copy the safetensors files
  if target_dir.resolve() != checkpoint_dir.resolve():
    for sf in safetensors_files:
      dest = target_dir / sf.name
      if dest.exists() or dest.is_symlink():
        dest.unlink()
      dest.symlink_to(sf.resolve())
    index_json = checkpoint_dir / "model.safetensors.index.json"
    if index_json.is_file():
      shutil.copy2(index_json, target_dir / index_json.name)

  copied = []
  for pattern in METADATA_PATTERNS:
    src = hf_assets_path / pattern
    if src.is_file():
      shutil.copy2(src, target_dir / pattern)
      copied.append(pattern)

  # Ensure Gemma 4 chat turn terminator `<turn|>` (token ID 106) is included in
  # generation_config.json when chat_template.jinja is present (base gemma-4-E2B
  # only sets eos_token_id=1).
  gen_cfg_path = target_dir / "generation_config.json"
  if gen_cfg_path.is_file() and (target_dir / "chat_template.jinja").is_file():
    gen_cfg = json.loads(gen_cfg_path.read_text())
    eos = gen_cfg.get("eos_token_id")
    if eos == 1:
      gen_cfg["eos_token_id"] = [1, 106]
      gen_cfg_path.write_text(json.dumps(gen_cfg, indent=2) + "\n")

  print(f"Prepared vLLM-ready HF checkpoint at: {target_dir}")
  print(f"  Safetensors shards : {[f.name for f in safetensors_files]}")
  print(f"  Copied HF metadata : {copied}")
  return target_dir


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Prepare a TorchTitan HF checkpoint directory for vLLM-TorchTPU."
      )
  )
  parser.add_argument(
      "--checkpoint_dir",
      type=Path,
      required=True,
      help=(
          "Path to TorchTitan step-<N> checkpoint "
          "(e.g., ./outputs/gemma4_e2b_sft/checkpoint/step-200)"
      ),
  )
  parser.add_argument(
      "--hf_assets_path",
      type=Path,
      required=True,
      help="Path to original HF assets directory with config & tokenizer files",
  )
  parser.add_argument(
      "--output_dir",
      type=Path,
      default=None,
      help=(
          "Optional clean output directory to populate "
          "(defaults to in-place in --checkpoint_dir)"
      ),
  )
  args = parser.parse_args()
  prepare_checkpoint(args.checkpoint_dir, args.hf_assets_path, args.output_dir)


if __name__ == "__main__":
  main()
