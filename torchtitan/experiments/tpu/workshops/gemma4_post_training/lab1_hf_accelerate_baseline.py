#!/usr/bin/env python3
"""Lab 1: Baseline Hugging Face Accelerate Post-Training Script.

This script represents the standard single-host / multi-GPU Hugging Face
``accelerate`` post-training loop that ML engineers typically start with before
migrating to TorchTitan on Cloud TPU v6e (TorchTPU).

It supports two modes matching Labs 2 & 3 of the workshop:
  - ``--mode cpt``: Continuous Pre-Training on
    ``codeparrot/codeparrot-clean-valid``
  - ``--mode sft``: Supervised Fine-Tuning on
    ``iamtarun/python_code_instructions_18k_alpaca``

Why migrate from raw HF Accelerate to TorchTitan-TorchTPU?
----------------------------------------------------------
1. **Parallelism**: Raw ``accelerator.prepare(model)`` wraps in DDP (or FSDP1),
   whereas TorchTitan's ``parallelize_hf_transformers_tpu`` applies
   compiler-aware ``SimpleFSDP`` over ``spmd_types`` meshes where the AllGather
   sits cleanly outside each decoder layer's ``torch.compile(backend="tpu")``
   region.
2. **Attention & Packing**: Standard HF DataLoaders pad to ``max_length`` and
   run dense SDPA. TorchTitan's Grain pipeline packs sequences with
   ``ConcatThenSplitPackingConfig`` (CPT) or ``FirstFitPackingConfig`` (SFT) and
   routes document-masked attention through the Pallas ``splash_tpu`` kernel
   (``attn_mask_type="block_causal"``) so packed samples never attend across
   document boundaries.
3. **Vocabulary Bottleneck (256k Vocab)**: Gemma models have a ~256k-token
   vocabulary. Materializing ``[batch * seq_len, vocab_size]`` logits in fp32
   consumes gigabytes of HBM per step. TorchTitan-TorchTPU replaces the output
   projection + CrossEntropy with ``PallasCrossEntropyLoss``, fusing
   ``hidden @ weight.T + soft_cap + log_softmax + NLL`` in TPU VMEM without ever
   materializing the full logits tensor.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from accelerate import Accelerator
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import (  # pylint: disable=g-multiple-import
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

IGNORE_INDEX = -100


def format_alpaca_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
  """Format an Alpaca sample into standard user/assistant chat messages."""
  instruction = (sample.get("instruction") or "").strip()
  inp = (sample.get("input") or "").strip()
  output = (sample.get("output") or "").strip()
  user_content = (
      f"{instruction}\n\n### Input:\n{inp}"
      if inp and inp.lower() != "not applicable"
      else instruction
  )
  return [
      {"role": "user", "content": user_content},
      {"role": "assistant", "content": output},
  ]


def build_cpt_dataloader(
    tokenizer: AutoTokenizer, seq_len: int, batch_size: int
) -> DataLoader:
  """Baseline padded/truncated CPT DataLoader (contrasted with Grain)."""
  raw_ds = load_dataset(
      "codeparrot/codeparrot-clean-valid", split="train[:2000]"
  )

  def tokenize_fn(batch: dict[str, list[Any]]) -> dict[str, Any]:
    enc = tokenizer(
        batch["content"],
        truncation=True,
        max_length=seq_len + 1,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"][:, :-1]
    labels = enc["input_ids"][:, 1:].clone()
    labels[enc["attention_mask"][:, 1:] == 0] = IGNORE_INDEX
    return {"input_ids": input_ids, "labels": labels}

  tokenized = raw_ds.map(
      tokenize_fn, batched=True, remove_columns=raw_ds.column_names
  )
  tokenized.set_format(type="torch", columns=["input_ids", "labels"])
  # On TPU, DataLoader must use num_workers=0 to avoid PJRT fork deadlocks.
  return DataLoader(
      tokenized,
      batch_size=batch_size,
      shuffle=True,
      drop_last=True,
      num_workers=0,
  )


def build_sft_dataloader(
    tokenizer: AutoTokenizer, seq_len: int, batch_size: int
) -> DataLoader:
  """Baseline SFT DataLoader with prompt label masking (IGNORE_INDEX = -100)."""
  raw_ds = load_dataset(
      "iamtarun/python_code_instructions_18k_alpaca", split="train[:2000]"
  )

  def tokenize_sft(sample: dict[str, Any]) -> dict[str, Any]:
    messages = format_alpaca_messages(sample)
    prompt_enc = tokenizer.apply_chat_template(
        messages[:1], tokenize=True, add_generation_prompt=True
    )
    prompt_ids = (
        prompt_enc["input_ids"]
        if isinstance(prompt_enc, dict) or hasattr(prompt_enc, "input_ids")
        else list(prompt_enc)
    )
    full_enc = tokenizer.apply_chat_template(messages, tokenize=True)
    full_ids = (
        full_enc["input_ids"]
        if isinstance(full_enc, dict) or hasattr(full_enc, "input_ids")
        else list(full_enc)
    )
    full_ids = list(full_ids)[: seq_len + 1]

    pad_len = (seq_len + 1) - len(full_ids)
    pad_id = tokenizer.pad_token_id or 0
    padded = full_ids + [pad_id] * pad_len

    input_ids = torch.tensor(padded[:-1], dtype=torch.long)
    labels = torch.tensor(padded[1:], dtype=torch.long)
    # Mask out prompt tokens and right-padding tokens so loss is on response.
    prompt_cutoff = max(len(prompt_ids) - 1, 0)
    labels[:prompt_cutoff] = IGNORE_INDEX
    if pad_len > 0:
      labels[len(full_ids) - 1 :] = IGNORE_INDEX
    return {"input_ids": input_ids, "labels": labels}

  tokenized = raw_ds.map(tokenize_sft, remove_columns=raw_ds.column_names)
  tokenized.set_format(type="torch", columns=["input_ids", "labels"])
  return DataLoader(
      tokenized,
      batch_size=batch_size,
      shuffle=True,
      drop_last=True,
      num_workers=0,
  )


def _run_xla_fsdp_worker(index: int, args: argparse.Namespace) -> None:
  """4-Chip HF Accelerate + torch_xla (XlaFSDP) worker spawned via xmp.spawn.

  Demonstrates the 8 non-obvious torch_xla workarounds required to run Gemma 4
  E2B across 4 TPU v6e chips (32 GiB HBM/chip) without crashing or OOMing:
    1. xmp.spawn(..., nprocs=None) (nprocs=4 raises ValueError on PJRT).
    2. Primary parameters in float32 (XlaFSDP raises TypeError on bf16 params).
    3. Custom auto_wrap_policy wrapping Gemma4's 2.35B embed_tokens_per_layer
       in addition to Gemma4TextDecoderLayer (while keeping embed_tokens in root
       for weight tying with lm_head).
    4. torch_xla.distributed.fsdp.checkpoint_module instead of standard PyTorch
       model.gradient_checkpointing_enable() (which raises AttributeError:
       module 'torch' has no attribute 'xla' in torch/utils/checkpoint.py).
    5. shard_param_on_dim_0=True (avoids an 8.75 GiB 1D-flatten reshape copy of
       embed_tokens_per_layer's gradient on TPU).
    6. disable_reshard_on_root=False and fp32_reduce_scatter=False.
    7. Explicit xm.mark_step() after XlaFSDP init and between backward/step.
    8. Adafactor optimizer (since fp32 AdamW states consume 8.62 GiB/chip on top
       of 4.31 GiB fp32 weights + 4.31 GiB fp32 grads + 15.23 GiB program).

  Args:
    index: Process index passed by xmp.spawn.
    args: Parsed CLI arguments.
  """
  del index
  import torch_xla.core.xla_model as xm  # pylint: disable=g-import-not-at-top
  from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XlaFSDP  # pylint: disable=g-import-not-at-top
  from torch_xla.distributed.fsdp import checkpoint_module  # pylint: disable=g-import-not-at-top
  from transformers import Adafactor, AutoConfig, Gemma4ForCausalLM  # pylint: disable=g-import-not-at-top,g-multiple-import
  from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer  # pylint: disable=g-import-not-at-top

  os.environ["PJRT_DEVICE"] = "TPU"
  accelerator = Accelerator()
  rank = accelerator.process_index
  device = accelerator.device
  if rank == 0:
    print(
        f"[Lab 1 Baseline: xla_fsdp] world_size={accelerator.num_processes} "
        f"distributed_type={accelerator.distributed_type} device={device} "
        f"batch_size={args.batch_size} seq_len={args.seq_len}",
        flush=True,
    )

  cfg = AutoConfig.from_pretrained(args.model_id).text_config
  model = Gemma4ForCausalLM(cfg).to(dtype=torch.float32)
  model.config.use_cache = False
  per_layer_emb_ref = getattr(model.model, "embed_tokens_per_layer", None)
  per_layer_proj_ref = getattr(model.model, "per_layer_model_projection", None)

  def _gemma4_auto_wrap_policy(
      module: torch.nn.Module,
      recurse: bool,
      unwrapped_params: int = 0,
      **kwargs: Any,
  ) -> bool:
    del unwrapped_params, kwargs
    if recurse:
      return True
    if isinstance(module, Gemma4TextDecoderLayer):
      return True
    if module is per_layer_emb_ref or module is per_layer_proj_ref:
      return True
    return False

  def _auto_wrapper_callable(m: torch.nn.Module, *a: Any, **kw: Any) -> XlaFSDP:
    if isinstance(m, Gemma4TextDecoderLayer) or m is per_layer_emb_ref:
      m = checkpoint_module(m)
    return XlaFSDP(m, *a, **kw)

  model = XlaFSDP(
      model,
      auto_wrap_policy=_gemma4_auto_wrap_policy,
      auto_wrapper_callable=_auto_wrapper_callable,
      compute_dtype=torch.bfloat16,
      buffer_dtype=torch.bfloat16,
      fp32_reduce_scatter=False,
      shard_param_on_dim_0=True,
      disable_reshard_on_root=False,
  )
  xm.mark_step()
  xm.wait_device_ops()

  optimizer = Adafactor(
      model.parameters(),
      lr=args.lr,
      scale_parameter=False,
      relative_step=False,
      warmup_init=False,
  )
  input_ids = torch.randint(
      0, 1000, (args.batch_size, args.seq_len), device=device
  )
  labels = input_ids.clone()

  model.train()
  for step in range(1, args.steps + 1):
    t0 = time.perf_counter()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss_detached = loss.detach()
    del outputs
    accelerator.backward(loss)
    del loss
    xm.mark_step()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    xm.mark_step()
    xm.wait_device_ops()
    dt = time.perf_counter() - t0
    loss_val = loss_detached.cpu().item()
    del loss_detached
    tps_chip = (args.batch_size * args.seq_len) / max(dt, 1e-6)
    if rank == 0:
      print(
          f"step: {step:3d}/{args.steps} | loss: {loss_val:.4f} | "
          f"tps/chip: {tps_chip:,.0f} | "
          f"global_tps (4 chips): {4 * tps_chip:,.0f} | "
          f"step_time: {dt:.2f}s",
          flush=True,
      )


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Lab 1: Baseline HF Accelerate Post-Training Script"
  )
  parser.add_argument(
      "--model_id",
      type=str,
      default="google/gemma-4-E2B",
      help="Hugging Face model ID or local path",
  )
  parser.add_argument(
      "--mode",
      choices=["cpt", "sft"],
      default="sft",
      help="Post-training stage: 'cpt' (CodeParrot) or 'sft' (Python Alpaca)",
  )
  parser.add_argument(
      "--strategy",
      choices=["accelerate", "xla_fsdp", "ddp_tpu", "fsdp2_tpu"],
      default="accelerate",
      help=(
          "Baseline strategy to benchmark against TorchTitan-TorchTPU: "
          "'accelerate' (1-chip default HF Accelerator), "
          "'xla_fsdp' (4-chip HF Accelerate + torch_xla XlaFSDP via "
          "xmp.spawn), 'ddp_tpu' (4-chip PyTorch DDP on torch_tpu), "
          "or 'fsdp2_tpu' (4-chip PyTorch FSDP2 fully_shard on torch_tpu)."
      ),
  )
  parser.add_argument("--seq_len", type=int, default=2048)
  parser.add_argument("--batch_size", type=int, default=2)
  parser.add_argument("--steps", type=int, default=50)
  parser.add_argument("--lr", type=float, default=2e-5)
  parser.add_argument(
      "--output_dir", type=str, default="./outputs/lab1_accelerate"
  )
  parser.add_argument(
      "--no_save",
      action="store_true",
      help="Skip saving checkpoint at end of run",
  )
  args = parser.parse_args()

  if args.strategy == "xla_fsdp":
    import torch_xla.distributed.xla_multiprocessing as xmp  # pylint: disable=g-import-not-at-top

    xmp.spawn(_run_xla_fsdp_worker, args=(args,), nprocs=None)
    return

  if args.strategy in ("ddp_tpu", "fsdp2_tpu"):
    import torch.distributed as dist  # pylint: disable=g-import-not-at-top
    from torchtitan.experiments.tpu import cloud_distributed_utils  # pylint: disable=g-import-not-at-top

    cloud_distributed_utils.maybe_init_distributed()
    dist.init_process_group(backend="tpu")
    rank = dist.get_rank()
    device = torch.device("tpu:0")
    if rank == 0:
      print(
          f"[Lab 1 Baseline: {args.strategy}] Initialized "
          f"world_size={dist.get_world_size()} on {device}",
          flush=True,
      )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.gradient_checkpointing_enable()
    if args.strategy == "ddp_tpu":
      from torch.nn.parallel import DistributedDataParallel as DDP  # pylint: disable=g-import-not-at-top

      model = DDP(model, gradient_as_bucket_view=True)
    else:
      from torch.distributed.fsdp import fully_shard  # pylint: disable=g-import-not-at-top

      base_model = getattr(model, "model", model)
      lang_model = getattr(base_model, "language_model", base_model)
      for layer in lang_model.layers:
        fully_shard(layer)
      fully_shard(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, foreach=False
    )
    input_ids = torch.randint(
        0, 1000, (args.batch_size, args.seq_len), device=device
    )
    labels = input_ids.clone()
    for step in range(1, args.steps + 1):
      t0 = time.perf_counter()
      optimizer.zero_grad()
      out = model(input_ids=input_ids, labels=labels)
      out.loss.backward()
      optimizer.step()
      torch.accelerator.synchronize()
      dt = time.perf_counter() - t0
      if rank == 0:
        tps_rank = (args.batch_size * args.seq_len) / dt
        print(
            f"step: {step:3d}/{args.steps} | loss: {out.loss.item():.4f} | "
            f"tps/rank: {tps_rank:,.0f} | step_time: {dt:.2f}s",
            flush=True,
        )
    dist.destroy_process_group()
    return

  accelerator = Accelerator(mixed_precision="bf16")
  if accelerator.is_main_process:
    print(
        f"[Lab 1 Baseline] Launching HF Accelerate ({args.mode.upper()}) "
        f"on {args.model_id}"
    )

  tokenizer = AutoTokenizer.from_pretrained(args.model_id)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  # Migration Note #1: In TorchTitan-TorchTPU, model is initialized on 'meta'
  # device and sharded via SimpleFSDP + compiled per decoder block.
  model = AutoModelForCausalLM.from_pretrained(
      args.model_id,
      torch_dtype=torch.bfloat16,
      attn_implementation="sdpa",
  )
  model.gradient_checkpointing_enable()

  dataloader = (
      build_cpt_dataloader(tokenizer, args.seq_len, args.batch_size)
      if args.mode == "cpt"
      else build_sft_dataloader(tokenizer, args.seq_len, args.batch_size)
  )

  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=True)
  lr_scheduler = get_cosine_schedule_with_warmup(
      optimizer, num_warmup_steps=10, num_training_steps=args.steps
  )

  model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
      model, optimizer, dataloader, lr_scheduler
  )

  model.train()
  data_iter = iter(dataloader)
  for step in range(1, args.steps + 1):
    t0 = time.perf_counter()
    try:
      batch = next(data_iter)
    except StopIteration:
      data_iter = iter(dataloader)
      batch = next(data_iter)

    optimizer.zero_grad()
    outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    lr_scheduler.step()

    step_time = time.perf_counter() - t0
    tokens_per_sec = (args.batch_size * args.seq_len) / max(step_time, 1e-6)
    should_log = args.steps <= 10 or step == 1 or step % 5 == 0
    if accelerator.is_main_process and should_log:
      print(
          f"step: {step:3d}/{args.steps} | loss: {loss.item():.4f} | "
          f"tps/rank: {tokens_per_sec:,.0f} | step_time: {step_time:.2f}s",
          flush=True,
      )

  accelerator.wait_for_everyone()
  if not args.no_save:
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
      unwrapped.save_pretrained(args.output_dir, safe_serialization=True)
      tokenizer.save_pretrained(args.output_dir)
      print(f"[Lab 1 Baseline] Saved HF checkpoint to {args.output_dir}")


if __name__ == "__main__":
  main()
