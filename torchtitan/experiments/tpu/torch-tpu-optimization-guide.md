# Guide to optimizing LLMs on TorchTPU

> ⚠️ **Work in progress.** TorchTPU and its tooling are evolving quickly — some recommendations here
> will change, and defaults/APIs may shift under you. Treat this as current best-practice guidance,
> not a stable contract, and always measure on your own setup.

A practical guide to getting good MFU (model FLOPs utilization) out of a dense decoder-only LLM on
**TorchTPU** — PyTorch running on Cloud TPU. It can execute op-by-op (convenient for development)
or, under `torch.compile`, fuse ops into compiled XLA programs (the performant path). Getting good
MFU is largely about staying in the compiled path and feeding it well — the playbook below.

**Worked example.** For a demonstration of high-performance dense LLM training on TorchTPU, see the
**llama3 sample** — `llama3` (configs in
[`llama3/config_registry.py`](llama3/config_registry.py), e.g. `llama3_8b()`) — which runs the
llama3 model ([`../../models/llama3/model.py`](../../models/llama3/model.py)) with the compile +
fused-kernel + FSDP recipe described here. Use it as the reference for how the pieces fit together.

It references two companion libraries you will lean on for kernels and parallelism:

- **[TorchTitan](https://github.com/pytorch/torchtitan)** — its `experiments/tpu` kernels
  (splash attention, fused cross-entropy), `apply_compile` helper, SimpleFSDP, and the
  `num_flops_per_token` MFU formula.
- **[tokamax](https://github.com/openxla/tokamax)** — mosaic-TPU Pallas kernels (e.g. the fused
  linear + softmax-cross-entropy kernel) that TorchTitan's TPU kernels build on.

---

## Scope — what applies to what

Most of this guide is **general TorchTPU** optimization (true of any model); a few sections are
**dense-specific**. MoE-specific guidance is in §10.

| Section | Applies to |
|---------|-----------|
| §1 Mental model | General |
| §2 Profile first | General |
| §3 Arithmetic intensity (dim, batch) | Dense-specific |
| §4 Reduce dispatch (compile, fused kernels) | General |
| §5 Memory & parallelism (SimpleFSDP + mixed precision) | General |
| §6 Precision, layout | General |
| §7 The dispatch ceiling | General |
| §8 Whole-step capture with GraphTrainer | General |
| §9 Large models: scan, graph splitter, CPU offload | General |
| §10 Mixture-of-Experts | MoE |

---

## 1. The mental model: compile, or be dispatch-bound  *(general)*

Every XLA program TorchTPU runs has to be dispatched by the host, and the TPU can sit idle between
launches. **Run uncompiled and each op becomes its own program** — so a naive model spends much of
every step idle between many tiny programs (the eager, dispatch-bound regime). `torch.compile` compiles
multiple ops into one program per compiled region, collapsing those gaps.

The practical consequence: for a small or naively-configured model you will often find the TPU is
**idle a large fraction of every step**, not compute-limited. When the matmuls actually run, the
MXU is typically fine; the time is lost *between* programs.

So the entire optimization strategy reduces to two ideas:

1. **Raise the work done per launched program** — bigger matmuls (model dim), bigger batch. This
   raises the compute-to-dispatch ratio.
2. **Reduce the number of host dispatches** — compile transformer blocks; use fused kernels.

Everything below is an instance of one of these two.

> **Eager mode is for development and debugging only.** Run uncompiled while you are building the
> model, checking numerics, or stepping through a bug — the eager per-op dispatch makes errors easy
> to localize. But an uncompiled model is **dispatch-bound and leaves most of the TPU idle**; it is
> *not* a performance configuration. **For any performance-sensitive run the model must be compiled
> with `torch.compile` (see §4).** Treat "compile it" as step zero of performance, not an optional
> lever — everything else in this guide assumes a compiled model.

---

## 2. Always profile first — classify the bottleneck  *(general)*

Do not optimize blind. Capture an **xprof** profile over a few steady-state steps and read the
**idle vs compute** split from the op-profile:

- **High idle (dispatch-bound)** — the common case for small models / small batch. The levers in
  §3–§4 apply directly.
- **Matmuls at a high fraction of peak (compute-bound)** — you are near the eager ceiling for that
  config; further gains need bigger problems (§3), and the residual dispatch glue is the ceiling (§7).
- **High HBM traffic / OOM (memory-bound)** — reach for the fused CE kernel and FSDP (§4, §5).

The single most useful number is the idle percentage. Optimize it down, or accept it as the
ceiling and move on.

---

## 3. Raise arithmetic intensity (the biggest lever)  *(dense-specific)*

### Batch size

A TPU is easily starved at small batch. Increasing the per-device batch is often the **single
largest MFU lever at small scale** — it directly raises the compute per launched program. Push
batch as high as HBM allows; the fused CE kernel (§4) frees room for a bigger one.

### Size the model to the hardware

TPU MXUs are large systolic arrays (typically 128×128 or 256×256 depending on generation). FLOPs
per matmul scale with `dim²` while the per-program dispatch cost is roughly constant, so a larger
hidden dim does much more useful work per launch.

- Use a **hidden dim that is a multiple of the MXU tile** (256 is a safe multiple across
  generations) — see *dimension alignment*.
- In practice, moving from a small debug dim to a production dim (e.g. 1024 → 2048) can **roughly
  double MFU** purely from better MXU utilization, before any other change.
- Keep scaling depth/dim until you hit the single-chip HBM ceiling, then sync memory relief (§4,
  §5) before scaling further.

---

## 4. Reduce dispatch and materialization  *(general)*

### torch.compile — mandatory for performance

**Compilation is essential; for models running on TPU, it is the critical step that transforms a
slow debug configuration into an optimized, performant execution.**
An uncompiled model is dispatch-bound (see §1); `torch.compile(backend="tpu")` fuses the ops
inside its captured region into one XLA program, collapsing many eager dispatches into few. This is
the single highest-impact change and typically yields a solid double-digit-percent speedup on its
own. Do this first.

**Compile at transformer-block granularity** — compile each block module (TorchTitan's
`apply_compile` does exactly this). This is the robust, low-risk granularity: it fuses the ops
*within* a block while keeping activation checkpointing *between* blocks, so memory stays bounded.

**Always compile with `dynamic=False`.** LLM training runs at a fixed batch × sequence length, so
the shapes are static; `dynamic=False` tells Dynamo to specialize to those static shapes and skip
dynamic-shape guards:

```python
compiled_block = torch.compile(block, backend="tpu", dynamic=False)
```

On a TPU/XLA backend, dynamic shapes are actively harmful — they force padding and recompilation
and defeat fusion. **Do not leave `dynamic` at its default (`None`)**: Dynamo can then silently
switch a region to dynamic shapes after a recompile, quietly tanking performance. If your shapes
genuinely are fixed (they should be for training), `dynamic=False` is always correct.

**Don't use `torch.compile` on the whole model.** Wrapping the entire model in one
`torch.compile(model, fullgraph=True)` (`compile.mode="fullgraph"`) is appealing but has real
failure modes on TorchTPU, and no TorchTitan TPU recipe uses it:

- Without activation checkpointing inside the graph, all layers' activations are live at once →
  **compile-time / runtime OOM** at any real depth. (Dynamo can capture non-reentrant
  `checkpoint()`, but on TorchTPU the RNG-state fork and restore that `preserve_rng_state=True`
  (the default) performs does not trace cleanly; the conformer recipe sets `preserve_rng_state=False`
  for that reason, and no TPU recipe puts `checkpoint()` inside a whole-model compile.)
- The token-embedding backward can bind incorrectly when compiled — leave the **embedding eager**.
- Lazily-created buffers (e.g. RoPE cos/sin) must be **materialized before tracing**, or the
  graph's input set shifts and recompilation/errors follow.
- And critically: even when whole-model compile *works*, it covers only the model's forward and
  backward. The loss and the optimizer step remain separate programs: eager by default, or
  separately compiled with `compile.optimizer_compile="separate_graph"` /
  `compile.compile_grad_norm=True` (and `loss` in `compile.components`). Either way it does **not** remove the whole-step dispatch idle (see §7).

`torch.compile` is a tool for **regions**: per-block compile is the right way to use it. If you
want the whole step as one program, use **GraphTrainer** (§8). It traces forward, loss, backward
and optionally the optimizer step into a single graph, and manages activation memory with a
graph-level policy instead of `checkpoint()`.

### Deferred-and-fused eager mode (a lighter-weight fallback)

TorchTPU also offers a **deferred-and-fused** execution mode that sits between plain op-by-op
execution and full `torch.compile`. Instead of dispatching each op immediately, it **defers** ops
and lets XLA **fuse across op boundaries**, cutting some of the per-op dispatch overhead without
requiring a captured graph.

**How to enable it** — there are 2 ways of doing this:

1. Set the env var **before `torch_tpu` is imported** (it's read at libtpu init):

   ```bash
   export TORCH_TPU_DEFER_AND_FUSE=1
   ```

   ```python
   # or, in Python, before the first `import torch_tpu`:
   import os
   os.environ["TORCH_TPU_DEFER_AND_FUSE"] = "1"
   ```

2. Wrap the region you want deferred-and-fused (e.g. the whole training loop) in the execution-mode
   context:

   ```python
   from torch_tpu._internal import execution_mode as em

   with em.set_eager_mode(em.EagerMode.DEFER_AND_FUSE):
       train()
   ```

**It can be combined with `torch.compile`.** The two are not mutually exclusive: keep your
transformer blocks under `torch.compile` and run the training loop inside the deferred-and-fused
context — the compiled regions still execute as compiled programs, while the mode defers and fuses
the *un-compiled* glue around them (the eager ops between compiled regions, and anything Dynamo
broke out of the graph). Use it as a fallback where you can't compile, and as a complement to shave
dispatch off the eager remainder where you can.

When it helps:

- Layered on top of `torch.compile` to reduce dispatch on the eager ops the compiled graph doesn't
  cover.
- Code that **can't be `torch.compile`d cleanly** — heavy graph breaks, data-dependent control flow,
  or Python that Dynamo won't trace — but where you still want dispatch overhead reduced.
- Quick experiments where you want some fusion without wiring up compilation.

Caveats:

- It is **not a substitute for `torch.compile`** for peak performance — compilation fuses more
  aggressively into larger programs. Compile your blocks first; use defer-and-fuse as a fallback
  where you can't compile, or layered on top to cover the eager remainder.
- Some op patterns don't lower cleanly under fusion (e.g. certain unaligned dynamic-update-slice
  fusions can fail at the libtpu level). If you hit a fusion crash, disable it for that model and
  fall back to `torch.compile` or plain execution.
- **Always measure.** On a model already covered by per-block `torch.compile`, defer-and-fuse
  typically adds little, because the compiled path already fuses the compute.

### Fused kernels (TorchTitan + tokamax)

Pull in the TPU Pallas kernels rather than hand-rolling — they are bridged into TorchTPU as
`custom_call`s and both fuse compute and cut dispatches, and both compose with compilation and with
FSDP. TorchTitan's **llama3** TPU config enables both — see
[`llama3/config_registry.py`](llama3/config_registry.py), where the `llama3_8b()` / `llama3_70b()`
configs set `use_splash_attention_kernel` and `use_loss_kernel`.

Both are Pallas kernels whose performance depends on their block/tile sizes: **the defaults are a
starting point, not an optimum — tune them per model shape and TPU generation, gated by the VMEM
budget (below). Sweep and measure.**

#### Fused linear + cross-entropy

tokamax mosaic kernel
([`kernels/linear_softmax_cross_entropy_loss.py`](kernels/linear_softmax_cross_entropy_loss.py)),
wired into training by [`loss.py`](loss.py) (`pallas_cross_entropy_loss` /
`build_cross_entropy_loss`). Fuses the `lm_head` matmul with softmax-cross-entropy so the full
`[batch, seq, vocab]` logits tensor is **never materialized**. Its biggest value is **memory** — at
large vocab/batch it removes a huge activation and unblocks a larger batch (feeding §3); the
throughput effect grows with batch and is neutral when the logits are already small.

**Tile sizes** — three block sizes in
[`LossKernelConfig` (`tpu_job_config.py`)](tpu_job_config.py): `loss_b_block_size` (default
**1024**), `loss_h_block_size` (default **512**), `loss_v_block_size` (default **2048**).
**Constraint:** the flattened token count (`batch × seq`) must be a multiple of `loss_b_block_size`,
or the kernel silently falls back to XLA — keep it divisible by 1024. Tune `loss_v_block_size`
against your vocab size.

#### Splash attention

A flash-style TPU attention kernel
([`kernels/splash_attention.py`](kernels/splash_attention.py)), patched into a model's attention by
[`workarounds.py`](workarounds.py) (`use_splash_attention_patch`). It matters most at long sequence
length and under memory pressure; at short sequence it is often a wash against TorchTPU's built-in
flash SDPA, so measure before adopting.

**Tile sizes** — nine block-size knobs in
[`SplashAttentionKernelConfig` (`tpu_job_config.py`)](tpu_job_config.py), each defaulting to
**512**: `sa_block_q`, `sa_block_kv`, `sa_block_dkv`, `sa_block_kv_compute`, `sa_block_q_dkv`,
`sa_block_kv_dkv`, `sa_block_kv_dkv_compute`, `sa_block_q_dq`, `sa_block_kv_dq`. **Meaningful
default: 512 across the board.** Larger blocks (1024/2048) can help at long sequence *if they fit
VMEM*; smaller blocks (256) cut VMEM but usually underperform and can trip kernel constraints.
Retune `sa_block_q` / `sa_block_kv` first.

#### VMEM budget — set it for v6e

Pallas tile sizes are bounded by the per-core **scoped VMEM limit**, and the default is
conservative, so the kernels above can't use their best tiles until you raise it. On
**v6e (Trillium)** set it via the libtpu init arg:

```bash
LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536"
```

**Meaningful v6e default: `65536` KiB (64 MiB)** — the value TorchTitan's Trillium (v6e) recipes
converged on (older baselines used `131072` / 128 MiB; 64 MiB is the current recommendation). Raise
it if a kernel reports insufficient VMEM for a larger tile; lower it if you hit VMEM-limit
rejections at startup. This is **generation-specific** — other TPU generations cap at different
values, so verify the value is accepted at init rather than assuming it transfers.

---

## 5. Memory and parallelism  *(general)*

### SimpleFSDP for sharding

For models (or batch/sequence lengths) that don't fit one chip, shard with **SimpleFSDP**
(TorchTitan's DTensor-based FSDP: `data_parallel(model, mesh, mode=...)`):

- `fully_shard` (ZeRO-3-like) shards params **and** optimizer state across chips — this is what
  makes large models trainable that plain DDP OOMs on.
- `hybrid_shard` (HSDP, a 2-D mesh: replicate across groups, shard within) is not a multi-host
  requirement. Full sharding spans hosts fine over ICI, and every shipped multi-host recipe on
  v7x (`qwen3-8b`, `qwen3-32b`, `qwen3-30b-a3b`, `muse-glimmer-30b`, `flux1` at 4x4x4 and 4x8x8)
  is plain FSDP with `data_parallel_replicate_degree=1`. HSDP buys cheaper gradient reduction
  within a replica group at the price of replicating parameters and optimizer state across
  groups, so per-chip memory goes up by the replicate degree; pick it when the FSDP group would
  otherwise be wider than the collectives need (very large chip counts) or when the replicate axis
  maps onto a slow link, as the multi-slice recipes do across DCN (replicate over slices, shard
  within). Choose the mesh from the interconnect, the collective cost and the memory budget, not
  from the host count.
- SimpleFSDP composes with `torch.compile`, which lets the compiler overlap the DTensor
  collectives with compute — combining FSDP with per-block compile is meaningfully faster than
  FSDP alone.

**Why SimpleFSDP over FSDP2 here?** SimpleFSDP expresses sharding purely as **DTensor
parametrizations riding DTensor autograd** — the all-gather is a `redistribute` in forward and the
reduce-scatter is implicit in backward, with **no module hooks and no explicit collective calls**.
That makes it **compiler-native**: the collectives live inside the traced graph, so `torch.compile`
can see, schedule, and overlap them with compute. FSDP2 (`fully_shard` module wrapper) drives its
collectives from pre/post-forward **hooks**, which are harder to trace cleanly and cause graph
breaks — the compiler can't overlap them as well. On the compile-first TorchTPU path, SimpleFSDP's
traceable design is the reason FSDP + per-block compile actually overlaps (and is much simpler:
a thin parametrization, not a wrapper with hook/state machinery).

### Use a mixed-precision policy

Run FSDP with a mixed-precision policy — the standard, correct setup is:

```python
MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
```

- `param_dtype=bfloat16` — parameters are all-gathered and used for compute in **bf16** (the
  matmuls run in bf16, which is what your MFU denominator assumes).
- `reduce_dtype=float32` — the gradient reduce-scatter accumulates in **fp32**, avoiding bf16
  cross-shard accumulation error.

This keeps effective master params/grads fp32-accurate while compute is bf16. **Do not** drop
`reduce_dtype` to bf16 to save bandwidth — that is a model-quality regression, not a perf lever.

---

## 6. Precision, layout, and the small stuff  *(general)*

- **Mixed precision — keep norms in fp32.** The bf16-compute / fp32-accurate-params setup itself is
  the FSDP `MixedPrecisionPolicy` (§5); without FSDP, use `torch.autocast`. The §6-specific point is
  a code-level gotcha: if you cast to bf16 *manually* (not via autocast), a bf16 input against an
  fp32 norm weight can raise a dtype-mismatch — compute **layer/RMS norm in fp32** and cast the
  result back. (Autocast keeps norms in fp32 already.)
- **Align dims to the MXU tile** so matmuls tile cleanly and don't pad. The MXU is 128×128 or
  256×256 depending on generation; **multiples of 256** are a safe choice across generations.
- **Avoid host syncs in the step loop.** In eager mode, anything that reads a device tensor on the
  host — `.item()`, `.cpu()`, `print(tensor)`, Python `if tensor > x`, logging the loss every step —
  forces a device→host sync that stalls dispatch and can dominate a small step. Log/inspect every N
  steps, keep metrics on-device, and detach reductions from the hot path.
- **Strip per-op Python overhead in benchmark runs.** If your code runs per-op runtime checks
  (e.g. shape/type validators), disable them for perf runs — they add host overhead in eager mode.
  Compilation bypasses them inside compiled regions anyway.

---

## 7. The dispatch ceiling — maximize compiled coverage  *(general)*

Even after per-block compile and fused kernels, TorchTPU has a residual dispatch floor: the
**optimizer step, the loss, the embedding, and layout ops remain separate programs**, and the
host-side gaps between whatever is *not* inside a compiled region set a structural MFU ceiling. You
can shrink that surface, but you cannot compile it away by only capturing the transformer.

Concretely: collapsing the transformer into a single compiled graph does **not** reduce the idle
*fraction* — the idle isn't located in the transformer, it's in everything *around* it.

**So the way to raise the ceiling is to maximize compiled coverage and minimize the eager
surface:**

- Compile every transformer block (§4) — that's the bulk of the FLOPs.
- Use the fused CE kernel (§4) so the loss is a single `custom_call` instead of a chain of eager
  ops.
- Prefer a fused kernel over an eager op sequence wherever one exists (attention, etc.).
- Compile the glue that has its own switch: `compile.optimizer_compile="separate_graph"` (the
  AdamW step as one graph), `compile.compile_grad_norm=True` (the global gradient norm; the
  in-place clip stays eager) and
  `loss` in `compile.components` (the loss function, when `compile.enable` is set). Each turns a
  chain of eager ops into one program; the dispatch between programs remains.
- Keep batch/dim large (§3) so the compiled regions dominate the step and the fixed eager glue is a
  smaller fraction of wall-clock.

Whatever training-step glue cannot be folded into a compiled region is your model's eager ceiling.
On a well-sized dense model with per-block compile + fused CE you will typically land in the
low-to-mid tens of percent MFU; pushing further means shrinking that eager surface.

To remove that surface entirely rather than shrink it, capture the whole training step with
GraphTrainer (§8).

---

## 8. Whole-step capture with GraphTrainer  *(general)*

Per-block `torch.compile` (§4) leaves everything between blocks eager, and the loss and the
optimizer step as separate programs, eager unless compiled on their own (§7). **GraphTrainer**
removes that surface.
`TpuGraphTrainer` ([`graph_trainer/tpu_graph_trainer.py`](graph_trainer/tpu_graph_trainer.py)),
the TPU subclass of TorchTitan's [GraphTrainer](../graph_trainer/README.md), traces the whole
training step (forward, loss, backward and optionally the optimizer step) with `make_fx` into a
single FX graph. It then runs graph passes over it and compiles the result for TPU, so the
forward, loss and backward run as one XLA program (the optimizer step too, with
`compile.optimizer_compile="in_graph"`; otherwise the optimizer step and gradient norm run outside
the step graph, eagerly or as their own graphs with `compile.optimizer_compile="separate_graph"` /
`compile.compile_grad_norm=True`).

How it differs from whole-model `torch.compile` (§4):

| | `torch.compile` on the model | GraphTrainer (`aot_fx_trace`) |
|---|---|---|
| Tracer | Dynamo: guards, graph breaks | `make_fx`: the whole step is traced ahead of time |
| Covers | model forward + backward | forward + loss + backward, plus the optimizer with `optimizer_compile="in_graph"` |
| Activation memory | `checkpoint()` inside the compiled region, if it traces at all on TorchTPU (§4: `preserve_rng_state` is the known problem; no TPU recipe does it) | a memory-policy pass on the graph decides per tensor: save, recompute or offload to host |
| FSDP | SimpleFSDP | SimpleFSDP collectives in the same graph, with passes that schedule all-gather prefetch |

**How to enable it:** a recipe selects GraphTrainer by returning a `TpuGraphTrainer.Config`;
`train.py` dispatches on the config type. Convert a regular TPU recipe and set its GraphTrainer
compile policy:

```python
config = TpuGraphTrainer.Config.from_tpu_trainer_config(
    base_config, graph_trainer_model_registry  # e.g. graph_trainer.qwen3.model_registry
)
config.compile = TpuGraphTrainerCompileConfig(
    enable=True,
    components=["model"],
    memory_policy="default",       # replaces the eager activation_checkpoint config
    optimizer_compile="in_graph",  # optional: fold AdamW into the step graph
)
```

`compile.mode` defaults to `"aot_fx_trace"`, the joint-graph mode. In it GraphTrainer does not wrap
blocks in `torch.compile`, so you do not compile twice. Reference recipes are `qwen3_8b()` in
[`qwen3/config_registry.py`](qwen3/config_registry.py), which includes the in-graph optimizer, and
the `llama3_*_scan()` recipes in [`llama3/config_registry.py`](llama3/config_registry.py).

**Memory policy.** GraphTrainer ignores the eager `activation_checkpoint` config. Activation memory
is set by `compile.memory_policy`:

- `default`: selective recompute. Saves the outputs of the ops on GraphTrainer's save list and
  recomputes the rest. On TPU the list covers `aten.linear` (torch_tpu traces `F.linear` as one
  op, so the projections are saved), the SDPA op, `topk` and the reduce-scatter / all-to-all
  collectives. It does not cover the Splash kernels or the grouped expert matmuls, which are
  Pallas custom ops: with Splash on, attention is recomputed under `default`, and only
  `tpu_selective` can save it.
- `full`: full recompute; only the layer boundaries are saved.
- `tpu_selective`: full recompute plus the regions named in `compile.save_regions` (every
  layer) or `compile.save_last_n_layers` (the last N layers, which backward reaches first).
  Two regions exist: `attention.inner_attention` (the Splash output and LSE, so backward skips
  re-running the kernel) and `moe.routing` (the top-k ids and the expert-sort permutation). It
  does not combine with scan. This is the TPU way to say "full recompute but keep X".
- `tpu_selective_and_offload`: `tpu_selective`, plus offload of everything else that is saved (the
  layer boundaries) to host memory as in `full_and_offload` (§9). The save-region values stay in
  HBM. The Qwen3-235B graph-splitter recipe uses it to keep the Splash output of its last 36
  layers on device.
- `eager`: alternates matmuls between save and recompute, matching TorchTitan's eager selective AC.
- `sac_and_offload`, `full_and_offload`, `full_and_scan_offload`: `default` or `full`, plus offload
  of saved activations to host memory (§9).

Exceptions that matter for the memory budget: under every policy, FSDP all-gather outputs are
force-saved only under ZeRO-2 (`fsdp_reshard_after_forward="never"`), because that is what "do not
reshard" means; under ZeRO-3 the policy may recompute them. RNG ops (dropout) are saved by `full`
and the policies built on it (`full_and_offload`, `full_and_scan_offload`, `tpu_selective`,
`tpu_selective_and_offload`),
because the graph path cannot replay their random state; under `default`, `eager` and
`sac_and_offload` an RNG op inside a recompute region is a runtime error (no TPU recipe has
dropout). Under scan, the MoE expert-count all-reduces are saved as well
(`scan_remat_policy.py`), so backward reuses the forward's counts instead of recomputing a
collective.

**Limitations:**

- **The whole step must trace.** There are no graph breaks to fall back on, and data-dependent
  Python control flow is not supported.
- **Keep `compile.mode="aot_fx_trace"`.** The TPU joint-graph options (scan, the graph splitter,
  remat barriers, the in-graph optimizer) are rejected in the other modes.
- **One unrolled step graph stops compiling or fitting at large sizes.** Use scan or the graph splitter
  (§9).
- **`optimizer_compile="in_graph"` costs HBM.** Step inputs are not donated today, so the graph
  holds both the input and the updated copies of parameters and optimizer state. Budget for it, or
  keep the optimizer as a separate graph (`optimizer_compile="separate_graph"`).
  `compile_grad_norm` cannot be combined with the in-graph optimizer.
- **libtpu 0.0.48+:** set `xla_enable_hlo_sharding_v3=False`, as the recipes do. Shardy HLO
  sharding v3 is on by default there and segfaults on GraphTrainer.

**When it pays.** The gain is largest where the step is dispatch-bound: small models, small batch.
On a well-sized model where per-block compile already leaves little device idle, it can roughly
tie. At the largest sizes, combined with scan or the graph splitter (§9), it is the path that compiles at
all. Measure on your own setup.

---

## 9. Large models: scan, graph splitter and CPU offload  *(general)*

GraphTrainer (§8) compiles the step as one program with every layer unrolled. That stops working
as the model grows, for three reasons:

- **Compile time and compile-time memory** grow with depth, because the compiler sees N copies of
  the layer.
- **Parameters.** With ZeRO-3 (`fsdp_reshard_after_forward="always"`) on one unrolled graph, XLA
  can merge the forward all-gathers with the recomputed ones and keep full parameters alive through
  backward. The config rejects this combination unless you pick one of the structures below.
- **Saved activations.** Even with full recompute, each layer saves its input, and all N of them
  are live at the end of the forward pass.

Scan and the graph splitter are two ways to give the step a structure that fixes the first two. CPU
offload fixes the third and works with either. The large recipes combine ZeRO-3, full recompute,
offload, and one of scan or the graph splitter.

| | Scan (`compile.use_scan_layers`) | Graph splitter (`compile.graph_split.enabled`) |
|---|---|---|
| Step runs as | one program, the layers as one loop | many programs, one per layer group |
| Compiler sees | one layer body | each child separately |
| Parameters | stacked `[n_layers, ...]`; per-layer names are gone | unchanged, per-layer |
| Parallelism | data-parallel meshes: FSDP and HSDP (2-D shard x replicate) are tested; TP, CP, PP, EP rejected | not restricted by the config; the recipes are FSDP and HSDP |
| In-graph optimizer | not rejected, but no recipe uses it | rejected |
| Offload policy | `full_and_scan_offload` | `full_and_offload` |
| Reference recipes | `llama3_*_scan()`, `qwen3_moe_235b_scan()` | `qwen3_moe_235b_fsdp_activation_offload_*()` |

No recipe combines the two; pick one.

For mid-size models there is a lighter third option that keeps one unrolled graph:
`compile.fsdp_barrier_policy="prefetch"` or `"interleave"` with
`compile.enable_remat_barriers=True`, as in `qwen3_8b_zero3()`.

### Scan over layers

`compile.use_scan_layers=True` runs the decoder layers as a single `scan`
([`graph_trainer/layer_stack.py`](graph_trainer/layer_stack.py)):

```python
config.compile.use_scan_layers = True
config.compile.memory_policy = "full_and_scan_offload"
```

What it changes:

- **One loop body.** Each kind of weight is stored as one tensor with a row per layer, and the
  compiler sees one layer instead of N. The layer body is compiled once, so the part of compile
  time and compile-time memory that scaled with depth no longer does. `llama3_405b_scan()` is the
  405B recipe because the per-layer path does not compile at that size.
- **FSDP gathers one layer at a time.** The all-gather runs inside the loop body on one layer's
  shard, next to the compute that needs it, so the full stack of weights is never resident.
- **The body is fully recomputed in backward**, including its all-gather. The only per-layer save
  is the layer input. See *CPU offload* below for why that one tensor matters.

Constraints:

- It is a GraphTrainer compile option, traced with `compile.mode="aot_fx_trace"`: an eager
  `scan` does not run on TPU.
- Not every parallelism is supported. Data-parallel meshes work, FSDP and HSDP (the 2-D shard x
  replicate mesh) are tested; tensor, context, pipeline and expert parallelism are rejected.
- The layer body must not mutate its inputs or module state; `scan` raises at trace time. MoE
  models hit this through load-balancing counters and routing metrics. `qwen3_moe_235b_scan()`
  shows the workarounds (an override for the counter, `metrics.enable_moe_routing = False`).
- Some ops do not trace inside a `scan` body. Llama3's complex RoPE is one; the recipes swap in a
  bit-identical real-arithmetic override
  ([`overrides/real_rope.py`](overrides/real_rope.py)).
- `named_parameters()` reports the stacks, not per-layer names, so anything keyed off per-layer
  parameter names stops matching. Model checkpoints keep their per-layer keys and move freely
  between scan and non-scan runs. Optimizer checkpoints do not, so a run cannot resume across a
  flip of the flag.
- The generic `*_and_offload` memory policies are rejected. Use `full_and_scan_offload`.
- Use the recipe's libtpu flag bundle. In the 235B recipe each layer's expert-gradient
  reduce-scatters land at the end of the loop body with little compute left to hide them, and the
  scan step runs measurably slower without the recipe's flags.

### Graph splitter

`compile.graph_split.enabled=True` keeps the layers unrolled but cuts the traced step into
separately compiled programs ([`graph_trainer/graph_split.py`](graph_trainer/graph_split.py)). It
first splits the step into a forward and a backward graph, then optionally subdivides each by
layer:

```python
config.compile.graph_split.enabled = True
config.compile.graph_split.forward_layers_per_executable = 1
config.compile.graph_split.forward_isolate_prefix = True
config.compile.graph_split.backward_partition_mode = "by_layer"
config.compile.graph_split.backward_layers_per_executable = 1
```

The knobs, in [`GraphSplitterConfig` (`graph_trainer/config.py`)](graph_trainer/config.py):

- `forward_layers_per_executable`: complete layers per forward program. The default `-1` keeps the
  forward as one program.
- `forward_isolate_prefix`: put the work before the first layer (embedding) in its own program.
- `backward_partition_mode`:
  - `none`: the backward stays one program.
  - `by_layer`: groups adjacent layers' backward, each with its own FSDP collectives.
  - `fsdp_pipeline`: moves each layer's all-gather earlier and its reduce-scatter later, to form
    stages where communication overlaps the neighbouring layer's compute.
- `backward_layers_per_executable`: complete layers per backward program.

A collective is never divided across programs: its launch and its wait stay in the same program.
Whether a layer's collectives stay with the layer's compute depends on the mode: `by_layer` keeps
them together; `fsdp_pipeline` deliberately puts a layer's all-gather in the preceding program and
its reduce-scatter in the following one, so the chain spans three programs by design. Each layer
keeps its own saved input and is recomputed just before its own backward, whatever the grouping.

The tradeoff is program size against scheduling scope. Small programs compile fast and bound
compile-time memory, but the compiler can only overlap communication with compute inside one
program, and every boundary is a host dispatch. Raising the `*_layers_per_executable` values
widens the scheduling window at the cost of larger programs. Start at 1 and measure.

Constraints:

- It requires `compile.fsdp_barrier_policy="none"`. Remat barriers do compose with it.
- It does not support `optimizer_compile="in_graph"`. Keep the optimizer as a separate program.
- It does not support serialized precompiled FX artifacts. TorchTPU's normal compilation cache
  still works.
- `donate_internal_buffers` (default `True`) lets XLA reuse dead buffers at program boundaries.
  Disable it only for diagnostics.

> **Not the regular trainer's forward/backward boundary.** Without GraphTrainer,
> `compile.materialize_forward_before_backward=True` only inserts a sync between the eager forward
> and backward. That splits one large XLA graph into two and cuts first-step compile time. None of
> the knobs above apply.

### CPU offload

Under full recompute the largest remaining activation cost is the saved layer inputs: one
`[batch, seq, dim]` tensor per layer, all live from the forward pass until that layer's backward.
CPU offload copies each one to host memory as the forward pass produces it and brings it back just
before the backward that needs it. It trades HBM for host RAM and transfer time. It is a memory
lever, not a speed lever: use it when the model is HBM-bound after recompute, and expect to work
to hide the transfers.

Select it through the memory policy:

| Policy | Use with | What is offloaded |
|---|---|---|
| `sac_and_offload` | unrolled graph or graph splitter | activations that selective recompute still saves |
| `full_and_offload` | unrolled graph or graph splitter | saved layer boundaries under full recompute |
| `tpu_selective_and_offload` | unrolled graph or graph splitter | saved layer boundaries; the `tpu_selective` save regions stay in HBM |
| `full_and_scan_offload` | scan | the one stacked layer-input tensor |

`compile.cpu_offload_budget_gb` (default 100) caps the host memory used per rank. Tensors
are picked largest-first until the budget is spent. It does not apply to the scan policy, which
always offloads its one tensor. Make sure the host has that much free RAM per rank.

**With scan, offload is close to mandatory.** A scan has one loop, so the per-layer saves come out
as one stacked `[n_layers, batch, seq, dim]` tensor. One tensor cannot be freed a row at a time, so
all of it stays live until the backward loop ends. On Llama3-405B at long sequence that stack
alone is why the shape does not fit. `full_and_scan_offload`
([`graph_trainer/tpu_host_offload.py`](graph_trainer/tpu_host_offload.py)) puts one offload in the
forward body and one reload in the backward body, so the device holds one layer's slice at a time.
The value carried to the next layer stays on device, so no iteration waits on a host round trip.
The Llama3 scan recipes above 8B need this policy to run at all.

**With the graph splitter, prefetch the reload.** By default each reload sits in the same program as the
backward that consumes it, so the transfer is exposed.
`compile.graph_split.cpu_offload_prefetch_executables = 1` issues it one backward program earlier, where
the previous layer's compute hides it. The cost is carrying one completed activation across one
program boundary. The 235B recipe uses it: the hidden transfer time is worth far more than the HBM
it costs. It requires a split backward (`backward_partition_mode` other than `none`).

**libtpu flags the offload recipes set:**

- `xla_tpu_enable_offloading_copy_to_sparsecore=False` keeps the host copies off SparseCore, where
  they would contend with the FSDP collectives. The measured effect on its own is small.
- `xla_tpu_copy_insertion_use_region_analysis=True` (with
  `xla_tpu_copy_insertion_use_region_analysis_limit=1000`) lets XLA prove the copy ordering instead
  of taking a defensive snapshot of each activation. On the 235B graph-splitter recipe it removed the
  device-to-host waits and gave a clear MFU gain.

**Check it in the profile.** After enabling offload, look for exposed host-to-device and
device-to-host waits. If the transfers are not hidden behind compute, offload costs you step time
for the memory it frees.

---

## 10. Mixture-of-Experts (MoE)  *(MoE-specific)*

Everything above still holds for MoE: compile the step, use the fused CE loss, SimpleFSDP and
mixed precision. What changes is the FFN. Instead of one dense MLP per layer, a router sends each
token to its top-k of N experts, so the expert FFN is a *grouped* matmul (a different number of
rows per expert) wrapped in a dispatch that sorts tokens by expert and a combine that puts them
back. Left on stock PyTorch ops, both run poorly on TPU. TorchTitan ships TPU kernels for each
part.

Reference recipes, all in [`qwen3/config_registry.py`](qwen3/config_registry.py):
`qwen3_moe_30b_fsdp_only_full_ac_benchmark()` (30B), `qwen3_moe_235b_fsdp_activation_offload_*()`
(235B, graph splitter) and `qwen3_moe_235b_scan()` (235B, scan).

### Which kernels to enable

| Kernel | What it replaces | Switch |
|---|---|---|
| Grouped expert matmul ([`kernels/expert_segment_matmul.py`](kernels/expert_segment_matmul.py)) | `torch._grouped_mm` for each expert projection, forward and backward | `use_pallas_grouped_mm` |
| Paired gate/up matmul ([`kernels/paired_segment_matmul.py`](kernels/paired_segment_matmul.py)) | two separate grouped matmuls for the gate (W1) and up (W3) projections, which read the same routed rows twice | `use_pallas_paired_grouped_mm` |
| Kernel-native weight layout | a transpose of every expert weight on each call | `use_kernel_native_weight_layout` |
| Dispatch and combine ([`token_dispatcher.py`](token_dispatcher.py)) | scatter-based autograd for the sort-by-expert reorder and the top-k combine | `TpuTokenDispatcherConverter` |
| Route-score gradient ([`kernels/route_score.py`](kernels/route_score.py)) | a saved fp32 copy of the expert output and a separate reduction in backward | `fuse_route_score_backward` |

Enable all five. The grouped matmul is the largest lever. The others remove copies and repeated
work around it.

### How to enable them

Two converters swap in the TPU classes when the model is built. The kernels are then switched on
per module:

```python
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuTokenDispatcherConverter,
)

cfg.model_spec = model_registry(
    "30B-A3B-global-aux",
    attn_backend="sdpa",
    converters=[
        TpuTokenDispatcherConverter.Config(),
        TpuGroupedExpertsConverter.Config(),
    ],
)

model_config = cfg.model_spec.model
for _, experts, _, _ in model_config.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    experts.use_pallas_paired_grouped_mm = True
    experts.use_kernel_native_weight_layout = True
    experts.pallas_grouped_mm_tile_m = 256
    experts.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)

for _, dispatcher, _, _ in model_config.traverse(TpuAllToAllTokenDispatcher.Config):
    dispatcher.fuse_route_score_backward = True

cfg.parallelism.expert_parallel_degree = 1
```

Notes on the switches:

- **Use the Pallas grouped matmul, not `torch._grouped_mm`.** The native op has produced incorrect
  weight gradients on TPU (torch_tpu issue 2668). The recipes use the Pallas path for that reason.
- **The paired kernel and the kernel-native layout both require `use_pallas_grouped_mm`.** Model
  construction fails otherwise.
- **Model checkpoints are unaffected by the kernel-native layout; optimizer state is not.** Weights
  are stored in the order the kernel reads, and the module's state-dict hooks translate to and from
  the standard keys (`w1_EFD` etc.) on save and load. The optimizer state dict has no such hook: it
  is keyed by the live parameter names (`state.w1_EDF.exp_avg` vs `state.w1_EFD.exp_avg`) with the
  live shapes, so resuming optimizer state across a layout change needs a conversion; a plain load
  fails on the missing key.
- **Expert compute runs in bf16** whatever the surrounding dtype. The weight-gradient kernels
  accumulate in fp32 and return bf16.
- **Route-score fusion keeps the fp32 multiply and reduction** of native autograd. Only the
  reduction order changes.

### Tile sizes

Like the other Pallas kernels (§4), the expert kernels are bounded by the scoped VMEM limit and
need tuning per model shape and TPU generation. The values in the snippet are the 30B recipe's.

- `pallas_grouped_mm_tile_m`: row tile for the forward matmul. With the paired kernel the routed
  row count (`tokens x top_k`) must be a multiple of it, or the kernel raises.
- `pallas_grouped_mm_tgmm_tiling = (m, k, n)`: tiles for the weight-gradient kernel. A `0` means
  the full axis. The paired kernel requires `n = 0`.
- `pallas_grouped_mm_w2_tgmm_tiling`: a separate tiling for the down projection (W2). The `k` tile
  runs along the weight's input dimension, which is the model dim for W1/W3 and the expert hidden
  dim for W2, so one value often cannot suit both. A `k` tile larger than the axis is **not**
  clamped: it allocates the full window and can exhaust VMEM before the first step.

At large expert shapes the weight-gradient kernel does not fit VMEM untiled, so the `k` tile is
required and not just a tuning knob. The 235B recipes show a worked case.

### Expert parallelism (EP)

**Expert parallelism is not fully implemented in TorchTitan TPU yet. Support is coming later. For now,
run with `expert_parallel_degree = 1`,** as every tuned recipe does, and let FSDP shard the expert
weights along with the rest of the model.

Setting EP above 1 is accepted, but the optimized path is not available:

- The paired gate/up kernel rejects it.
- The TPU combine and the route-score fusion are skipped, and the standard all-to-all path runs
  instead.
- Scan over layers (§9) rejects it.
- `compile_grad_norm` rejects it.

### Other MoE points

- **Scan and MoE.** A scan body cannot mutate module state, which MoE layers do through
  load-balancing counters and routing metrics. See the constraints in §9 and the workarounds in
  `qwen3_moe_235b_scan()`.
- **Profile dispatch and combine separately.** After the kernels are on, check that the reorder
  and combine around the expert matmul are no longer a visible bucket.
- **Router precision and load balance are quality-sensitive.** Keep routing in fp32. Capacity
  factor and token dropping affect throughput **and model quality**: treat any change that drops
  tokens as a correctness question, not a free perf lever.
- **MFU accounting.** Use **active** params (shared params plus the top-k experts a token actually
  touches), *not* total params, in `num_flops_per_token`. Total-param MFU overstates utilization
  for a sparse model.

---

## Quick-start checklist  *(dense)*

Eager (uncompiled) is only for steps 1's profiling and for debugging — every performance number
below assumes a compiled model.

1. Profile a few steady-state steps; read the idle fraction.
2. **Compile — mandatory.** `torch.compile` each transformer block with
   `backend="tpu", dynamic=False` (TorchTitan `apply_compile`). Do this before tuning anything
   else; an uncompiled model is a debug config, not a performance one.
3. Size the model: hidden dim ≥ a large MXU-aligned value; push batch until HBM-bound.
4. Add the fused kernels (tokamax / TorchTitan): the fused CE kernel (frees memory → bigger batch)
   and splash attention (long sequence). **Tune their tile sizes** (§4) — the defaults aren't
   optimal.
5. If it doesn't fit one chip, SimpleFSDP `fully_shard` with
   `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`, across hosts too; `hybrid_shard`
   only when the replicate axis buys something (multi-slice over DCN, very wide FSDP groups) and
   the memory for the replicated state is there (§5).
6. bf16 compute / fp32 master; norm in fp32; dims multiple of 256; shape-checks off for benchmarks.
7. Re-profile. If matmuls are near peak and the only idle left is the training-step glue, you're at
   the torch_tpu eager ceiling (§7) — raise it by shrinking the eager surface, not by more tuning,
   or remove that surface by capturing the whole step with GraphTrainer (§8).
8. If the step no longer compiles or fits as one unrolled graph, run the layers as a scan or let the graph splitter cut
   the graph into programs, and offload saved activations to host memory (§9).

---

### Operational notes

- **Build TorchTPU wheels with optimizations on** (`-c opt` for a Bazel build). A fastbuild wheel
  can have *dramatically* slower eager dispatch and will silently ruin every measurement here.
- **After any hard crash / killed process, clear the stale libtpu lock** (e.g.
  `rm -f /tmp/libtpu_lockfile`) or the next run may fail to initialize the device.
