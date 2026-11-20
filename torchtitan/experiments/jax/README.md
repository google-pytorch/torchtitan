# jax experiment

LLM training on TPU using pure [JAX](https://github.com/google/jax) with
[Flax NNX](https://flax.readthedocs.io/en/latest/nnx/) — no PyTorch dependency
in the training path.

Parallel to `torchtitan/experiments/torchax/` but uses Flax NNX + optax directly
instead of the torchax PyTorch-on-JAX wrapper.

## Overview

- **Single-process, multi-device**: JAX auto-discovers all TPU chips; no `torchrun` needed.
- **Flax NNX models**: PyTorch-like Module API with explicit state split/merge.
- **Scan-based layers**: `jax.lax.scan` over stacked transformer blocks — one compact XLA
  graph regardless of depth, dramatically reducing compile time.
- **FSDP**: Parameters sharded via JAX named mesh (`fsdp`, `tp` axes) using
  `jax.sharding.NamedSharding`.
- **Splash attention**: Pallas-based fused attention kernel on TPU.
- **optax optimizer**: adamw/adam/sgd via `nnx.Optimizer`.
- **Models**: Llama3 (8B, 70B).

## File layout

```
jax/
├── train_minimal.py        # Entry point: parses flags, builds mesh, runs trainer
├── trainer.py              # JaxTrainer: model/optimizer/dataloader setup + train loop
├── jax_job_config.py       # JaxConfig + JaxJobConfig dataclasses (CLI --jax_config.* flags)
├── distributed.py          # JAX sharding utilities: apply_sharding_to_state, shard_input
├── splash_attn.py          # Builds splash attention callable from Pallas kernel
├── metrics.py              # JaxMetricsProcessor: TPS, TFLOPs, MFU logging
├── data_utils.py           # fake_dataloader, torch_loader_to_jax converter
└── llama3/
    ├── model.py            # Flax NNX Llama3: RMSNorm, Attention (GQA+RoPE), FFN, Transformer
    ├── sharding.py         # Parameter sharding maps for scan and non-scan variants
    └── __init__.py         # Re-exports ModelArgs, Transformer, args dict, sharding maps
```

## Quick start

```bash
python -m torchtitan.experiments.jax.train_minimal \
    --model.name=llama3 \
    --model.flavor=8B \
    --training.dataset_path=tests/assets/c4_test \
    --model.hf_assets_path=tests/assets/tokenizer \
    --training.max_context_length=2048 \
    --training.num_tokens_per_train_step=8192 \
    --training.steps=10 \
    --jax_config.use_scan \
    --activation_checkpoint.mode=full
```

Use `--training.dataset=fake` to skip the real dataset and use synthetic data.
Use `--jax_config.model_layer_override=2` to debug with a small 2-layer model.

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--jax_config.use_scan` | True | Enable scan-based layers (smaller XLA graph) |
| `--jax_config.tpu_megacore` | True | Enable megacore (v4/v5p) |
| `--jax_config.model_layer_override` | None | Override n_layers for debugging |
| `--parallelism.tensor_parallel_degree` | 1 | TP degree (mesh tp axis size) |
| `--activation_checkpoint.mode` | full | `full`, `selective`, or `nothing` |

## Parallelism

The mesh is shaped `(fsdp, tp)` where:
- `fsdp = num_devices // tp`
- `tp` is set by `--parallelism.tensor_parallel_degree`

For v6e-4 with tp=1: `fsdp=4, tp=1`.

Multi-slice runs use `create_hybrid_device_mesh` with the `--jax_config.tpu_num_slices` flag.

## Architecture notes

### Scan-based layers (`use_scan=True`)

`ScannedTransformerBlocks` stacks all layer parameters along a leading axis and
runs them with `jax.lax.scan`. Each parameter gets a named `nnx.Param` attribute
(e.g. `attention_wq_kernel`, `feed_forward_w2_kernel`) derived from the block's
NNX state paths. This gives:
- Constant XLA graph size regardless of number of layers
- Faster compilation than unrolled layers
- Easy sharding via named attributes in the sharding map

### Sharding

The embedding table is replicated (`()`) to avoid gather ambiguity when tokens
are batch-sharded. All attention and FFN weights are FSDP-sharded on their
leading weight dimension. The `freqs_cis` RoPE table is replicated.

### NNX API notes (Flax 0.11+)

- `nnx.Optimizer(model, tx, wrt=nnx.Param)` — explicit `wrt` required
- `optimizer.update(model, grads)` — model passed explicitly (no `optimizer.model`)
- `nnx.value_and_grad(fn, argnums=nnx.DiffState(0, nnx.Param))` — filter grads to Param only
- `jax.tree_util.tree_flatten(nnx.State)` — leaves are raw `jax.Array` values

## Performance (v6e-4, 4 chips)

| Model | seq | batch | MFU (after warmup) |
|-------|-----|-------|---------------------|
| Llama3 8B (scan, full AC) | 2048 | 4 | ~20% |
