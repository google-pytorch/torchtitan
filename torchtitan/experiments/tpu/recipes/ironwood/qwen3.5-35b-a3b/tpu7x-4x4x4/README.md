# Qwen3.5-35B-A3B on Ironwood (TPU7x)

Qwen3.5-35B-A3B (text only): 40 layers in a 3:1 cycle of `GatedDeltaNet`
(linear attention) and full-attention layers, 256 routed experts (top-8) plus a
gated shared expert in every layer; 34.66B parameters, 2.95B active per token.
`qwen35_moe_35b_split_tokamax_benchmark` trains it on any TPU7x slice from one
`tpu7x-2x2x1` host (4 chips, 8 TensorCore ranks) to `tpu7x-4x4x4` (64 chips,
128 ranks).

## Launch

One `tpu7x-2x2x1` host:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m torchtitan.experiments.tpu.train \
  --module=torchtitan.experiments.tpu.qwen3_5 \
  --config=qwen35_moe_35b_split_tokamax_benchmark \
  --hf_assets_path=tests/assets/tokenizer \
  --dataloader.dataset=c4_test \
  --dataloader.dataset_path=tests/assets/c4_test
```

`tpu7x-4x4x4` (16 hosts; see the TPU [README](../../../../README.md) for
cluster setup), with 64K tokens per rank:

```bash
torchrun --nnodes=16 --nproc_per_node=8 \
  --rdzv_backend=static --rdzv_endpoint=$WORKERS_0_HOSTNAME:29501 \
  --node_rank=$TPU_WORKER_ID \
  -m torchtitan.experiments.tpu.train \
  --module=torchtitan.experiments.tpu.qwen3_5 \
  --config=qwen35_moe_35b_split_tokamax_benchmark \
  --training.num_tokens_per_microbatch_per_dp_rank=65536 \
  --hf_assets_path=tests/assets/tokenizer \
  --dataloader.dataset=c4_test \
  --dataloader.dataset_path=tests/assets/c4_test
```

The image needs a Tokamax build with the Kimi Delta Attention op
(`tokamax._src.ops.kda`, [openxla/tokamax](https://github.com/openxla/tokamax)
HEAD): the tokamax wheels up to 0.1.0 (and the `tokamax==0.0.14` pinned in
`requirements.txt`) do not have it, so `tests/test_tokamax_gated_delta.py`
skips without it. Without Tokamax, keep only the fused-loss override
(`--override.imports
torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head`); the
GatedDeltaNet then runs the XLA `chunked_gated_delta`.

## Recipe

* **Precision and sharding**: bf16 parameters, gradients and AdamW moments (8
  bytes per parameter, 258 GiB in total), ZeRO-3 SimpleFSDP
  (`fsdp_reshard_after_forward=always`) with `mixed_precision_reduce="float32"`.
* **Selective activation checkpointing (`tpu_selective`)**: saves
  `moe.routing` (`routing_scores` and `selected_experts_indices`) and
  `attention.inner_attention` (Splash Attention output) across forward and
  backward executables instead of recomputing MoE top-k `sort`/`topk`/`softmax`
  and Splash Attention during backward. Cuts step-1 compile time by 35–36%
  (`303 s -> 198 s` at 64K, `355 s -> 228 s` at 96K) and improves steady-state
  step time (`6.94 s -> 6.86 s`, `18.21% MFU` at 64K `split=2`; `10.50 s ->
  10.01 s`, `18.73% MFU` at 96K `split=1` with
  `--tpu_config.graph_trainer_save_last_n_layers attention.inner_attention 2`).
* **GraphTrainer with graph split**: as one step graph, the 40-layer model
  neither compiles within an hour on 8 host CPUs nor fits in HBM on one host
  (XLA keeps most layers' unsharded expert gradients alive until late in the
  step). Per-layer executables bound both, and repeated layers hit the
  compilation cache (first step after 3-5 minutes on 224-CPU GKE hosts). Two
  layers per forward and backward executable (`split=2`, each pairing a
  full-attention layer with a `GatedDeltaNet` layer) with
  `backward_partition_mode=fsdp_pipeline`: 3.51 s/step vs 4.65 s/step with one
  layer per executable (`split=1`) at 32K tokens per rank on one host, while
  `split=1` halves peak per-executable unsharded parameter/gradient buffers at
  96K–128K tokens per rank (`10.01 s/step`, `18.73% MFU` at 96K on `2x2x1`;
  `13.56 s/step`, `18.43% MFU` at 128K on `4x4x4`).
* **AdamW**: torch_tpu's fused AdamW outside the step graph
  (`optimizer_in_graph=False`, `compile_grad_norm=True`).
* **`GatedDeltaNet`**: Tokamax's Mosaic TPU KDA kernels
  (`TokamaxGatedDeltaKernel`, through the `tokamax_gated_delta_kernel`
  override), with the forward returning the backward kernels' residuals; the
  causal convolution and its input gradient are computed from shifted slices.
* **MoE**: Pallas paired grouped matmuls with kernel-native expert weights
  and the fused route-score backward.
* **Attention**: Splash attention with segment ids from `positions`; 1024
  blocks keep `head_dim=256` within the scoped VMEM limit.
* **Loss**: Pallas fused linear cross-entropy (no `[tokens, 248320]` logits in
  HBM).
* **Logging** every 5 steps: a per-step `loss.item()` keeps the host from
  dispatching the next step while the current one runs.
* **libtpu flags**: `xla_tpu_dvfs_p_state=7` (throughput operating point);
  the SparseCore collective aggregator with the latency-hiding layer scheduler
  (overlaps the FSDP collectives with compute); SparseCore-offloaded async
  all-gathers; reduce-scatters on both SparseCores; and
  `xla_tpu_scoped_vmem_limit_kib=65472` for the Pallas kernels.

## Benchmark Results

Full 40-layer model (`qwen35_moe_35b_split_tokamax_benchmark`,
`mixed_precision_reduce="float32"`, `enable_tpu_topo_mesh=True`), context length
8192 (`--training.max_context_length=8192`), 20 steps:

| Slice (ranks) | Environment | Tokens/rank | Remat / Split | Step time (step 15 / 20) | Tokens/s/rank | TFLOPS/rank | MFU (step 15 / 20) | HBM (XLA reservation) | Loss @ steps 1/10/20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2x2x1` (8) | Forge (8-vCPU) | 65,536 (`8 × 8192`) | `full`, `split=2` | **7.03 s** / 7.15 s | **9,316** / 9,168 | **204.85** / 201.59 | **17.76%** / 17.48% | 39.39 GiB | 12.75 / 3.90 / 2.58 |
| `2x2x1` (8) | Cloud GKE (224-vCPU) | 65,536 (`8 × 8192`) | `full`, `split=2` | **6.94 s** / 6.95 s | **9,445** / 9,433 | **207.67** / 207.42 | **18.00%** / **17.98%** | 40.16 GiB | 12.75 / 3.90 / 2.58 |
| `2x2x1` (8) | Cloud GKE (224-vCPU) | 65,536 (`8 × 8192`) | `tpu_selective`, `split=2` | **6.86 s** / **6.88 s** | **9,554** / **9,531** | **210.07** / **209.56** | **18.21%** / **18.17%** | 40.12 GiB | 12.75 / 3.87 / 2.92 |
| `2x2x1` (8) | Cloud GKE (224-vCPU) | 98,304 (`12 × 8192`) | `full`, `split=1` | **10.50 s** / 10.51 s | **9,362** / 9,357 | **205.85** / 205.74 | **17.85%** / **17.84%** | 40.07 GiB | 12.75 / 3.57 / 2.58 |
| `2x2x1` (8) | Cloud GKE (224-vCPU) | 98,304 (`12 × 8192`) | `tpu_selective`, `split=1` | **10.02 s** / **10.01 s** | **9,809** / **9,823** | **215.68** / **216.00** | **18.70%** / **18.73%** | 39.99 GiB | 12.75 / 4.74 / 3.45 |
| `2x2x2` (16) | Cloud GKE (224-vCPU) | 65,536 (`8 × 8192`) | `full`, `split=2` | **7.06 s** / **7.05 s** | **9,281** / **9,296** | **204.08** / **204.41** | **17.69%** / **17.72%** | 40.08 GiB | 12.76 / 3.80 / 2.56 |
| `2x2x2` (16) | Cloud GKE (224-vCPU) | 98,304 (`12 × 8192`) | `full`, `split=1` | **10.24 s** / 10.25 s | **9,602** / 9,592 | **211.13** / 210.91 | **18.30%** / **18.28%** | 39.88 GiB | 12.76 / 3.85 / 2.59 |
| `2x2x2` (16) | Cloud GKE (224-vCPU) | 131,072 (`16 × 8192`) | `full`, `split=1` | **13.63 s** / 13.66 s | **9,619** / 9,596 | **211.50** / 210.99 | **18.34%** / **18.29%** | 50.87 GiB | 12.76 / 3.78 / 2.58 |
| `4x4x4` (128) | Cloud GKE (224-vCPU) | 65,536 (`8 × 8192`) | `full`, `split=2` | **7.28 s** / 7.28 s | **9,007** / 8,999 | **198.05** / 197.88 | **17.17%** / **17.15%** | 40.08 GiB | 12.76 / 3.89 / 2.59 |
| `4x4x4` (128) | Cloud GKE (224-vCPU) | 98,304 (`12 × 8192`) | `full`, `split=2` | **11.75 s** / 11.79 s | **8,369** / 8,337 | **184.03** / 183.32 | **15.95%** / **15.89%** | 65.75 GiB | 12.76 / 3.89 / 2.59 |
| `4x4x4` (128) | Cloud GKE (224-vCPU) | 131,072 (`16 × 8192`) | `full`, `split=1` | **13.56 s** / 13.56 s | **9,666** / 9,666 | **212.54** / 212.54 | **18.43%** / **18.43%** | 50.74 GiB | 12.76 / 3.91 / 2.59 |

With `enable_tpu_topo_mesh=True` preserving the physical 3D torus `(x, y, z)`
rank order across all 16 hosts, `4x4x4` (128 ranks) reaches `17.17%` MFU at 64K
tokens/rank (`95.4%` weak-scaling efficiency across 16x more chips). Because
128-way ZeRO-3 shards parameters, gradients, and AdamW states 16x finer than
8-way ZeRO-3 (`2.17 GiB/rank` vs `34.66 GiB/rank`), `4x4x4` fits 128K
tokens/rank (`16 × 8192`) with 1 layer per executable (`split=1`), reaching
**`18.43% MFU`** (`9,666 tokens/s/rank`, `1.24M tokens/s` cluster-wide,
`50.74 GiB` XLA reservation).

## Expert Parallelism

`--parallelism.expert_parallel_degree=<EP>` (or
`qwen35_moe_35b_split_tokamax_ep_benchmark`, EP=4) shards the routed experts
over `EP` ranks and FSDP-shards them over the remaining `efsdp` ranks; all
other parameters stay sharded over `fsdp`:

* `TpuQwen35TokenDispatcher` dispatches and combines tokens with static-shape
  all-to-alls (no host syncs), which `TpuGraphTrainer` can trace. Each rank
  sends at most `ceil(1.25 * T * top_k / EP)` tokens to each other rank;
  tokens beyond that are dropped (zero expert output), unlike the dropless
  `EP=1` path.
* The experts use the unpaired Pallas grouped matmuls (`ExpertParallel` shards
  the expert weights on dim 0).
* The optimizer gets one param group per device mesh, and the gradient norm
  is computed eagerly over both meshes (`compile_grad_norm=False`).
* Graph split uses one layer per executable with `by_layer` backward
  executables (3 distinct backward executables to compile instead of 6).

Full 40-layer model, `4x4x4` (128 ranks), 32K tokens per rank:

| EP | Step time | Tokens/s/rank | MFU | Memory (XLA reservation) |
| --- | --- | --- | --- | --- |
| 1 | 4.25 s | 7,707 | 12.7% | 20.8 GiB |
| 4 | 9.61 s | 3,410 | 5.6% | 17.3 GiB |
| 8 | 8.61 s | 3,808 | 6.3% | 17.2 GiB |
| 16 | 10.46 s | 3,134 | 5.2% | 17.1 GiB |

At this model size EP is about 2x slower than `EP=1`: the all-to-alls, the
unpaired grouped matmuls on smaller per-expert groups and the one-layer
executables cost more than the smaller expert all-gathers save. It is meant
for scaling beyond what FSDP alone fits.

## Notes

* `GatedDeltaNet` runs one recurrence per `[B, S]` row: documents packed into
  a row share the causal convolution and the delta-rule state (there are no
  varlen `cu_seqlens` on TPU); attention masks documents via `positions`.
* The Tokamax kernels clamp the per-token log decay at -5
  (`TokamaxGatedDeltaKernel.Config.min_log_decay`): the kernel's sub-chunk
  factoring overflows below it, which produced NaN gradients with the model's
  initialization (log decays down to about -34).
* The auxiliary-loss-free expert-bias balancer is off
  (`load_balance_coeff=None`): HF checkpoints carry no expert biases.
