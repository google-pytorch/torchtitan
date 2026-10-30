# Traceable FlexShard in TorchTitan (experimental)

This is an opt-in replacement for SimpleFSDP in a **dense Qwen3 tracing
recipe**, not a replacement for FlexShard's public CUDA execution runtime.
Existing SimpleFSDP code and existing TPU recipes are unchanged.

## Dependencies and model

- TorchTitan fork base: `7caedbfb54d8155e8dc9a461215c2a28d2073f68`.
- Public FlexShard core: `233e90691a5523023f6cf98c20b4f4529bee9ea4`.
- Portable graph executor from TorchTPU PR #5959:
  `3f095aa23fb835cc017e538ad313c17945879d6e`,
  `examples/fsdp/experimental_flex_shard/graph_flex_shard.py`.
- Executor SHA256:
  `0e02bfc383ad60bdf2ce5bfe94f4ac88756362a592190d6ed4c08374edfcb373`.

The loader verifies both the executor and the imported private FlexShard core.
It does not duplicate executor code into TorchTitan.

The model is TorchTitan's actual Qwen3 `testmodel`: three layers, model dim
128, eight attention/KV heads, head dim 16, FFN hidden dim 256, vocabulary
2048, QK norms, CosSinRoPE, attention, and LM loss. The recipe disables fused
QKV and weight tying and supplies the embedding initializer normally omitted
for tied embeddings. It has 36 trainable Parameters (1,016,800 elements).
SDPA receives one single document per rank; this recipe does not support
packed-document attention.

## Why initialization and clipping differ

Trainer builds on meta, calls `parallelize_fn`, then calls `to_empty` and
`init_weights`, and only afterward constructs the optimizer. Public portable
storage requires initialized ordinary tensors, unlike SimpleFSDP's DTensor
meta path. A setup-only root initialization adapter installs FlexShard after
initialization and before optimizer construction. There is no first-forward
installation or execution-time parameter-swapping hook.

Parameters remain local views into public `ShardedBucketStorage`. Pure
parameter properties emit functional all-gather/reduce-scatter operations.
There are no CUDA streams, bucket storage resizes, or native FSDP copy ops in
the captured model graph. One canonical parameter per bucket is intentional.

Ordinary shard Parameters do not automatically carry DTensor global-norm
semantics. The Trainer's clipping override uses **norm-only** DTensor views
to reuse the exact stock `get_total_norm` / `_NormPartial` reduction order,
including replicated fallback parameters. Parameters and graph-produced
gradients remain ordinary local tensors. The norm and eager AdamW step are
outside the Trainer's forward/backward graph.

For SimpleFSDP/FlexShard parity tests, both paths are loaded with the same full
initial weights. Initializing sharded DTensors and initializing full weights
before sharding have different RNG layouts even with the same seed.

## What is captured

`trace.py` reuses TorchTitan's unchanged `make_fwd_bwd_step`,
`minimal_fx_tracer`, and `run_traced`. It captures forward, globally normalized
loss, explicit `autograd.grad`, communication, and functional SGD math in one
FX graph. Parameters, buffers, input tokens, labels, valid-token denominator,
and LR are explicit graph inputs. The same graph is replayed with new inputs
and updated live parameter state.

SGD returns updated local shard tensors. Copies into live byte-backed storage
occur outside capture and preserve Parameter identities. This demonstration
requires untied, trainable parameters. The full Trainer separately uses eager
AdamW; it does **not** claim that AdamW or clipping is in the joint graph.

The standalone TPU validation compiles the joint graph with TorchTPU's
`StaticCompiler`. The Trainer uses its existing TPU graph-pass pipeline and
terminal compiler. `compile.memory_policy` owns Trainer rematerialization;
eager `activation_checkpoint` settings are not applied and produce a warning.
The initial recipe uses `memory_policy="none"`. The standalone checkpoint
case explicitly applies non-reentrant checkpointing to the real layers.

## Observed validation

Matched runtime: torch `2.16.0.dev20261007+cpu`, torch-tpu
`0.2.1+fcb215628399`, libtpu `0.0.51`, spmd_types `0.2.5`.
Tyro `1.0.13` needs typeguard 4.x for the current fork's config parser; the
validation runtime used `4.6.0` without optional Tokamax/Pallas kernels.
The public FlexShard private core was exercised; the complete public CUDA
package/runtime was not validated in this CPU-wheel environment.

- **Two real CPU/Gloo ranks:** ten-step joint-graph parity against an
  unsharded global-batch reference, with and without checkpointing. Inputs and
  weights change between replays. Every gradient, SGD delta, and used weight
  is checked. The captured graphs contain 37 all-gathers / 36 reduce-scatters
  without checkpointing, and 70 / 36 with checkpointing; neither contains
  native FSDP copy ops or opaque module calls.
- **Full CPU Trainer:** ten steps of SimpleFSDP, FlexShard, and a FlexShard
  repeat, using the same initial weights, FP32, foreach AdamW, clipping, seed
  42, and deterministic mode. Loss and global gradient norm are bitwise equal
  between all three runs, including TensorBoard evidence.
- **One TPU7x host, eight actual device ranks:** ten joint-graph SGD steps with
  checkpointing off and on; all 16 rank/case summaries PASS. Maximum absolute
  gradient discrepancy from CPU is `8.568167686462402e-8`; maximum SGD-delta
  discrepancy is `5.9585545386653394e-8`. Actual FX and compiled StableHLO are
  retained. No tolerance was relaxed after execution.
- **Full TPU Trainer:** the opt-in recipe completed ten steps with compiled
  forward/loss/backward and eager AdamW. TensorBoard global loss was
  `7.656745433807373` at step 1 and `5.026253700256348` at step 10. This is
  synthetic-data execution evidence, not convergence or performance evidence.

Fixed reference tolerances: loss `rtol=1e-5, atol=1e-6`; gradients
`rtol=1e-4, atol=1e-6`; SGD deltas `rtol=1e-4, atol=1e-7`; resulting weights
`rtol=1e-5, atol=2e-6`. The CPU reference evaluates documents separately so
causal attention cannot cross document/rank boundaries. The loss divides by
global valid tokens once; gradient communication uses SUM, not another DP
average.

## Running

From the TorchTitan repo root in a compatible runtime:

```bash
# Check out public FlexShard at the exact core revision above.
# Download the exact public graph_flex_shard.py revision above.
export PYTHONPATH="$PWD:/absolute/path/to/flex_shard/src"
export TORCHTITAN_FLEX_SHARD_EXECUTOR=/absolute/path/to/graph_flex_shard.py

# CPU-only tests; unset accelerator registration before Python imports torch.
TORCH_DEVICE_BACKEND_AUTOLOAD=0 CUDA_VISIBLE_DEVICES= \
  python -m unittest discover \
  -s torchtitan/experiments/tpu/experimental_flex_shard/tests -p 'test_*.py'

# Single-host TPU joint forward/loss/backward/SGD graph validation.
# The launcher discovers actual runtime device count, not google.com/tpu count.
python -m torchtitan.experiments.tpu.experimental_flex_shard.launch_validation

# Full Trainer on one host; NPROC must be actual runtime device count.
NPROC=$(python -c 'import torch, torch_tpu; print(torch.accelerator.device_count())')
torchrun --standalone --nproc_per_node="$NPROC" \
  -m torchtitan.experiments.tpu.experimental_flex_shard.train \
  --module=torchtitan.experiments.tpu.experimental_flex_shard.config_registry \
  --config=qwen3_traceable_flex_shard --metrics.enable-tensorboard
```

`--no-use-traceable-flex-shard` selects the unchanged SimpleFSDP control in
this isolated recipe. Use `--compile.no-enable-passes` for CPU FX replay;
it intentionally does not prove TPU compilation.

## Remaining scope

This is not generic trainer/checkpoint compatibility or a performance port.
HSDP, TP/EP/CP/PP, MoE, quantization/converters, offload, scan, graph splitting,
precompiled artifacts, distributed checkpoint integration, accumulated
microbatches, coalesced buckets, and explicit reshard-policy requests are not
supported. Tied/frozen parameters are rejected by the joint-SGD helper.
BF16, multi-host execution, model-size scaling, ZeRO-3 peak-memory lifetimes,
and performance have not been validated. No claim is made that the stock
public `flex_shard()` CUDA runtime is now portable or fully traceable.
