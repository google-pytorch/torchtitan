#!/usr/bin/env python3
r"""Lab 4: Offline Inference & Evaluation with vLLM-TorchTPU on Cloud TPU v6e.

This script loads the post-trained Gemma 4 E2B Hugging Face checkpoint exported
by TorchTitan-TorchTPU in Lab 3 (and optionally compares it against the base
pretrained checkpoint) using ``vllm-torchtpu``.

Example usage (Single-chip or multi-chip TPU v6e):
--------------------------------------------------
  export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
  export TORCHINDUCTOR_AUTOGRAD_CACHE=0

  python3 lab4_vllm_torchtpu_inference.py \
      --model ./outputs/gemma4_e2b_sft/checkpoint/step-200 \
      --tensor_parallel_size 1 \
      --max_model_len 1024 \
      --max_tokens 256
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

# Recommended TorchTPU runtime settings for vllm-torchtpu
os.environ.setdefault(
    "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false"
)
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

DEFAULT_CODING_PROMPTS = [
    (
        "Write a Python function "
        "`merge_k_sorted_lists(lists: list[list[int]]) -> list[int]` "
        "using a min-heap (`heapq`) that runs in O(N log k) time. "
        "Include docstrings and a short test."
    ),
    (
        "Write an asynchronous Python retry decorator "
        "`async_retry(max_retries=3, base_delay=0.5)` "
        "with exponential backoff for `asyncio` coroutines."
    ),
    (
        "Implement a PyTorch custom `nn.Module` for Root Mean Square "
        "Layer Normalization (RMSNorm) in bfloat16 with an epsilon parameter."
    ),
    (
        "Write a Python function "
        "`lru_cache_with_ttl(maxsize=128, ttl_seconds=60)` using "
        "`collections.OrderedDict` and `time.monotonic()`."
    ),
]


def run_vllm_inference(
    model_path: str,
    prompts: list[str],
    *,
    tensor_parallel_size: int = 1,
    max_model_len: int = 1024,
    max_num_batched_tokens: int = 1024,
    max_tokens: int = 256,
    temperature: float = 0.0,
    enforce_eager: bool = False,
    use_chat_template: bool = True,
    profile_dir: str | None = None,
) -> list[str]:
  """Run batch inference using vLLM on TorchTPU and return generated text."""
  from vllm import LLM, SamplingParams  # pylint: disable=g-import-not-at-top,g-multiple-import

  llm_kwargs: dict[str, Any] = {
      "model": model_path,
      "tensor_parallel_size": tensor_parallel_size,
      "max_model_len": max_model_len,
      "max_num_batched_tokens": max_num_batched_tokens,
      "enforce_eager": enforce_eager,
  }
  if profile_dir:
    llm_kwargs["profiler_config"] = {
        "profiler": "torch",
        "torch_profiler_dir": profile_dir,
    }

  print(f"\n[vLLM-TorchTPU] Initializing engine for model: {model_path}")
  t_init = time.perf_counter()
  llm = LLM(**llm_kwargs)
  dt_init = time.perf_counter() - t_init
  print(f"[vLLM-TorchTPU] Engine initialized in {dt_init:.2f}s")

  sampling_params = SamplingParams(
      temperature=temperature,
      max_tokens=max_tokens,
  )

  if profile_dir:
    llm.start_profile()

  t_gen = time.perf_counter()
  if use_chat_template:
    conversations = [[{"role": "user", "content": p}] for p in prompts]
    outputs = llm.chat(messages=conversations, sampling_params=sampling_params)
  else:
    outputs = llm.generate(prompts, sampling_params=sampling_params)
  elapsed = time.perf_counter() - t_gen

  if profile_dir:
    llm.stop_profile()
    print(f"[vLLM-TorchTPU] Saved XProf trace to: {profile_dir}")

  total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
  print(
      f"[vLLM-TorchTPU] Generated {total_out_tokens} tokens across "
      f"{len(prompts)} prompts in {elapsed:.2f}s "
      f"({total_out_tokens / max(elapsed, 1e-6):.1f} output tok/s)"
  )
  return [o.outputs[0].text for o in outputs]


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Lab 4: vLLM-TorchTPU Inference on Post-Trained Gemma 4 E2B"
  )
  parser.add_argument(
      "--model",
      type=str,
      default="./outputs/gemma4_e2b_sft/checkpoint/step-200",
      help="Path to the post-trained HF checkpoint directory (or HF model ID)",
  )
  parser.add_argument(
      "--prompt",
      type=str,
      nargs="+",
      default=None,
      help="Custom prompt(s) to evaluate (defaults to 4 Python coding prompts)",
  )
  parser.add_argument("--tensor_parallel_size", type=int, default=1)
  parser.add_argument("--max_model_len", type=int, default=1024)
  parser.add_argument("--max_num_batched_tokens", type=int, default=1024)
  parser.add_argument("--max_tokens", type=int, default=256)
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument(
      "--enforce_eager",
      action="store_true",
      help="Disable torch.compile in vLLM for quick eager smoke-testing",
  )
  parser.add_argument(
      "--no_chat_template",
      action="store_true",
      help="Use raw llm.generate() instead of llm.chat() with chat template",
  )
  parser.add_argument(
      "--profile_dir",
      type=str,
      default=None,
      help="Optional directory to capture XProf / Torch profiler trace",
  )
  args = parser.parse_args()

  prompts = args.prompt or DEFAULT_CODING_PROMPTS
  responses = run_vllm_inference(
      args.model,
      prompts,
      tensor_parallel_size=args.tensor_parallel_size,
      max_model_len=args.max_model_len,
      max_num_batched_tokens=args.max_num_batched_tokens,
      max_tokens=args.max_tokens,
      temperature=args.temperature,
      enforce_eager=args.enforce_eager,
      use_chat_template=not args.no_chat_template,
      profile_dir=args.profile_dir,
  )

  print("\n" + "=" * 80)
  for idx, (prompt, resp) in enumerate(zip(prompts, responses), start=1):
    print(f"[Prompt #{idx}]\n{prompt}\n")
    print(f"[Generated Response #{idx}]\n{resp.strip()}")
    print("=" * 80)


if __name__ == "__main__":
  main()
