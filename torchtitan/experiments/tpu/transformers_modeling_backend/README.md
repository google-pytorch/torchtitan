# HF transformers backend on TPU

Train decoder-only text models from the Hugging Face Hub on Cloud TPU with TorchTPU, without porting
the model. This is the TPU lane of the
[HF transformers modeling backend](../../transformers_modeling_backend/README.md): the model is built
from the checkpoint's `config.json` (weights are random-init), sharded with SimpleFSDP, and runs with
TPU kernels swapped in underneath HF's modeling code.

> **Status: experimental.** Like the upstream HF backend, this is backend coverage, not a set of
> recommended production training recipes. It is meant for bring-up, coverage and performance work on
> TPU: there is no expert parallelism yet, MoE models are bound by expert-weight communication, and
> some architectures fall back to slower paths. Check loss parity before relying on a new architecture.

## Coverage

Tested on 500+ of the most-downloaded text-generation checkpoints on the Hub, across multiple
architectures: Llama, Qwen, Mistral/Mixtral, Gemma, Phi, OLMo/OLMoE, Granite, DeepSeek, Falcon-H1, LFM2,
Mamba and more. Models up to 10B parameters were tested at full size on v6e-8, 10-35B at full size on
v6e-32, and larger ones at a reduced debug size (the architecture, not the full model).

### Model sizes: full vs debug

Config names say which size is built; `--hf_model` always picks the architecture (attention type,
MLP/MoE, norms, ...):

- `hf_full_*` and `hf_moe_auto*` build the checkpoint at its own size: the shapes in its `config.json`
  are used as-is.
- `hf_debugmodel` overrides the size with a two-layer debug model (hidden size 256, 16 heads);
  `hf_debugmodel_max` and the other `hf_debugmodel_kernels_*` configs use a two-layer model shaped for
  the TPU kernels (hidden size 512, 4 query / 2 KV heads, head_dim 128). Use these for smoke tests,
  parity tests, and for architectures too large to build at full size.

## Optimizations enabled out of the box

With `hf_full_max` (dense) or `hf_moe_auto` (MoE), no flags needed:

- **Splash attention** on every HF architecture that uses the standard attention path, through HF's
  attention interface: document-aware block skipping for packed sequences, grouped/multi-query KV
  without expansion, sliding windows, attention-logit soft-cap, and in-kernel dropout. Anything the
  kernel cannot express falls back to SDPA and is logged once per reason.
- **Fused Pallas cross entropy** that never materializes the full logits, including the head's logit
  transform (Gemma soft-cap, Cohere scale, Granite divisor).
- **Per-layer `torch.compile`**, plus **selective activation checkpointing** that keeps the attention
  output instead of recomputing it.
- **SimpleFSDP** over all chips.
- **MoE**: the HF MoE block is swapped for TorchTitan's grouped experts, with **Pallas grouped
  matmuls** and a **TPU token dispatcher** whose backward gathers whole rows (3.6x to 65x faster than
  the standard dispatcher at equal batch on DeepSeek-V2-Lite, Qwen3-30B-A3B and OLMoE).

## Quick start

From the repo root on a single host (e.g. v6e-4):

```bash
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.transformers_modeling_backend \
    --config=hf_full_max \
    --hf_model=Qwen/Qwen3-0.6B \
    --training.max_context_length=2048 \
    --training.num_tokens_per_microbatch_per_dp_rank=4096 \
    --training.steps=20
```

`--hf_model` takes a Hub id or a local directory with a `config.json` (only the config is read, so a
gated repo's config can be copied locally). Use `hf_moe_auto` for MoE checkpoints.

To train on GKE (multi-host slices such as v6e-8 and larger), follow
[Multi-host model training on GKE](../README.md#multi-host-model-training-on-gke) in the main TPU
README: build and push the image, then pass the same `--module`, `--config` and `--hf_model` flags in
the workload command.

## Common flags

Append these to the quick-start command. The configs' defaults are a starting point: for best
performance, batch size and tile sizes have to be tuned manually for each model and slice.

### Batch size, sequence length and steps

There is no separate batch-size flag: batches are token budgets. Per-chip batch size (in sequences) is
`num_tokens_per_microbatch_per_dp_rank / max_context_length`. Raise it until throughput stops improving
or HBM runs out.

```bash
--training.max_context_length=4096                      # sequence length
--training.num_tokens_per_microbatch_per_dp_rank=16384  # 4 sequences of 4096 per chip
--training.num_tokens_per_train_step=1048576            # global batch; above microbatch x chips
                                                        # means gradient accumulation
--training.steps=10000
```

### Tile sizes (splash attention, fused cross entropy)

Splash block sizes default to 1024 (measured best on packed 2048-token sequences, not on every model
or sequence length); each is stepped down to the largest size that divides the sequence length. Other
`sa_*` fields are ignored by this kernel and reported once. The fused-CE token tile must divide the
per-rank token count.

```bash
--splash_attention_kernel.sa_block_q=512
--splash_attention_kernel.sa_block_kv=512
--splash_attention_kernel.sa_block_kv_compute=512
--splash_attention_kernel.sa_block_q_dkv=512
--splash_attention_kernel.sa_block_kv_dkv=512
--splash_attention_kernel.sa_block_kv_dkv_compute=512
--splash_attention_kernel.no-sa_use_document_aware_block_skipping  # mask, don't skip, documents
--loss.b_block_size=256                                 # fused-CE token tile
```

### Turning features on and off

```bash
--compile.no-enable                                     # eager (shows attention fallbacks)
--config=hf_full_max_plain_ce                           # model's own head, plain CE
--config=hf_moe_auto_compile                            # MoE without the Pallas grouped matmuls
--hf_config_overrides='{"num_hidden_layers": 4}'        # JSON edits to config.json
activation-checkpoint:full                              # or :selective (default), :none; put last
```


## Summary

The TPU HF backend is the fastest way to get a Hugging Face architecture training on TPU: point
`--hf_model` at a checkpoint, and splash attention, the fused cross entropy, per-layer compile and
(for MoE) the Pallas grouped matmuls are already on. Use it to bring up new architectures, check
coverage, compare models, and find where TPU performance work would pay off. For a production run on
a specific model, start from a native recipe under [`recipes/`](../recipes/) instead (e.g.
`recipes/trillium/qwen3-8b`), which is tuned per model and slice.
