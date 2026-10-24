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
  `--tpu_config.graph_trainer_save_last_n_layers attention.inner_attention 2`;
  measured without GatedDeltaNet document resets).
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
  `13.56 s/step`, `18.43% MFU` at 128K on `4x4x4`; measured without
  GatedDeltaNet document resets).
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
`mixed_precision_reduce="float32"`, `enable_tpu_topo_mesh=True`), one
`tpu7x-2x2x1` host (4 chips, 8 ranks), Cloud GKE, step 20 of 20:

| Context | Tokens/rank | Step time | Tokens/s/rank | TFLOPS/rank | MFU | HBM |
| --- | --- | --- | --- | --- | --- | --- |
| 2048 | 32,768 (`16 × 2048`) | 4.78 s | 6,849 | 129.91 | 11.26% | 23.96 GiB |
| 8192 | 65,536 (`8 × 8192`) | 9.78 s | 6,703 | 147.38 | 12.78% | 45.11 GiB |

The GatedDeltaNet restarts at every packed document here, and Tokamax lays
every document out from a 64-token chunk boundary for that (see Notes).

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

* Packed documents: the attention masks documents via `positions`, and the
  `GatedDeltaNet` restarts its causal convolution and delta-rule state at every
  document (`positions == 0`), through static-shape `[B, S]` segment ids
  instead of upstream's varlen `cu_seqlens`. The XLA kernel masks within
  chunks; Tokamax's kernels take `segment_ids` and lay every document out from
  a 64-token chunk boundary, keeping up to `max_num_segments` documents per row
  apart (default: one per 512 tokens, at most 12% more kernel tokens; later
  documents of a row share one recurrence).
* The Tokamax kernels clamp the per-token log decay at -5
  (`TokamaxGatedDeltaKernel.Config.min_log_decay`): the kernel's sub-chunk
  factoring overflows below it, which produced NaN gradients with the model's
  initialization (log decays down to about -34).
* The auxiliary-loss-free expert-bias balancer is off
  (`load_balance_coeff=None`): HF checkpoints carry no expert biases.
