# Qwen3.5-35B-A3B on Ironwood (TPU7x)

Qwen3.5-35B-A3B (text only): 40 layers in a 3:1 cycle of `GatedDeltaNet`
(linear attention) and full-attention layers, 256 routed experts (top-8) plus a
gated shared expert in every layer; 34.66B parameters, 2.95B active per token.
`qwen35_moe_35b_split_tokamax_benchmark` trains it on any TPU7x slice from one
`tpu7x-2x2x1` host (4 chips, 8 TensorCore ranks) up to 256 FSDP ranks (such as
`tpu7x-4x4x4`: 64 chips, 128 ranks; `TpuGroupedExperts` shards across the 256
experts on dim 0).

## Launch

One `tpu7x-2x2x1` host, at the recipe's context 2048 and 32K tokens per rank:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m torchtitan.experiments.tpu.train \
  --module=torchtitan.experiments.tpu.qwen3_5 \
  --config=qwen35_moe_35b_split_tokamax_benchmark \
  --training.steps=20 \
  --hf_assets_path=tests/assets/tokenizer \
  --dataloader.dataset=c4_test \
  --dataloader.dataset_path=tests/assets/c4_test
```

Add `--training.max_context_length=8192
--training.num_tokens_per_microbatch_per_dp_rank=65536` for context 8192 and
64K tokens per rank, as in the results below.

Multi-host launch on `tpu7x-4x4x4` (16 hosts, 128 ranks; see the TPU
[README](../../../../README.md) for cluster setup), at context 8192 and 64K
tokens per rank:

```bash
torchrun --nnodes=16 --nproc_per_node=8 \
  --rdzv_backend=static --rdzv_endpoint=$WORKERS_0_HOSTNAME:29501 \
  --node_rank=$TPU_WORKER_ID \
  -m torchtitan.experiments.tpu.train \
  --module=torchtitan.experiments.tpu.qwen3_5 \
  --config=qwen35_moe_35b_split_tokamax_benchmark \
  --training.steps=20 \
  --training.max_context_length=8192 \
  --training.num_tokens_per_microbatch_per_dp_rank=65536 \
  --hf_assets_path=tests/assets/tokenizer \
  --dataloader.dataset=c4_test \
  --dataloader.dataset_path=tests/assets/c4_test
```

The commands above use the repository's small `c4_test` fixture and test
tokenizer as self-contained throughput inputs (losses are not meaningful for
language modeling; point `--hf_assets_path` and `--dataloader.*` at real Qwen3.5
tokenizer assets and training data for real runs).

The image needs a Tokamax build with the Kimi Delta Attention op
(`tokamax._src.ops.kda`, [openxla/tokamax](https://github.com/openxla/tokamax)
HEAD): the tokamax wheels up to 0.1.0 (and the `tokamax==0.0.14` pinned in
`requirements.txt`) do not have it, so
`qwen3_5/tests/test_tokamax_gated_delta.py` skips its Mosaic KDA execution tests
without it. Without Tokamax, keep only the fused-loss override
(`--override.imports
torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head`); the
GatedDeltaNet then runs the XLA `chunked_gated_delta`.

## Recipe

* **Precision and sharding**: bf16 parameters, gradients and AdamW moments (8
  bytes per parameter, 258 GiB in total), ZeRO-3 SimpleFSDP
  (`fsdp_reshard_after_forward=always`) with `mixed_precision_reduce="float32"`.
  The regular-trainer recipe `qwen35_moe_35b` keeps FP32 master weights, as
  the Qwen3 recipes do; at 16 bytes per parameter they run out of HBM on one
  `tpu7x-2x2x1` host, so the benchmark sets `training.dtype="bfloat16"`.
* **Selective activation checkpointing (`tpu_selective`)**: saves
  `moe.routing` (each router's top-k expert ids and the route sort
  permutation) and `attention.inner_attention` (the Splash Attention output
  and its log-sum-exp) across forward and backward executables, so backward
  does not re-run the MoE top-k and sort or the Splash Attention forward.
  Without document resets (at context 8192), this cut step-1 compile time by
  35–36% (`303 s -> 198 s` at 64K tokens per rank, `355 s -> 228 s` at 96K) and
  step time from 6.94 to 6.86 s (`18.21% MFU`) at 64K; at 96K with one layer per
  executable it took 10.50 to 10.01 s (`18.73% MFU`) with Splash saved in the
  last two full-attention layers only
  (`--compile.graph_split.forward_layers_per_executable=1
  --compile.graph_split.backward_layers_per_executable=1
  --compile.save_regions moe.routing
  --compile.save_last_n_layers attention.inner_attention 2`).
* **GraphTrainer with graph split**: as one step graph, the 40-layer model
  neither compiles within an hour on 8 host CPUs nor fits in HBM on one host
  (XLA keeps most layers' unsharded `3 x 1 GiB` fp32 expert gradients alive
  until late in the step). Two-layer executables (`split=2`: consecutive pairs,
  half of which hold a full-attention layer) bound both, and repeated pairs hit
  the compilation cache (first step after 3-5 minutes on 224-CPU GKE hosts).
  With `backward_partition_mode=fsdp_pipeline`, `split=2` cuts the step from 84
  to 44 child executables; in the A/B that chose it (an earlier build of this
  recipe at 32K tokens per rank on one host), `split=2` had 24% shorter step
  time (+31% throughput) than one layer per executable (`split=1`). `split=1`
  halves the peak per-executable unsharded parameter and gradient buffers for
  96K–128K tokens per rank (without document resets, at context 8192:
  `10.01 s/step`, `18.73% MFU` at 96K on `2x2x1`; `13.56 s/step`, `18.43% MFU`
  at 128K on `4x4x4`).
* **AdamW**: torch_tpu's fused AdamW outside the step graph
  (`optimizer_compile="eager"`, `compile_grad_norm=True`).
* **`GatedDeltaNet`**: Tokamax's Mosaic TPU KDA kernels
  (`TokamaxGatedDeltaKernel`, through the `tokamax_gated_delta_kernel`
  override), with the forward returning the backward kernels' residuals.
  Pretraining runs it over whole `[B, S]` rows (`reset_at_documents=False`):
  restarting at packed documents makes the kernels lay every document out
  from a 64-token chunk boundary (6.74 vs 9.78 s/step at 64K tokens per rank
  on `2x2x1`). The causal convolution + SiLU and the gated RMSNorm run as
  Pallas kernels (`kernels/causal_conv1d.py`, `kernels/rms_norm_gated.py`).
* **MoE**: Pallas paired grouped matmuls with kernel-native expert weights
  and the fused route-score backward.
* **Attention**: Splash attention with segment ids from `positions`; 1024
  blocks keep `head_dim=256` within the scoped VMEM limit.
* **Loss**: Pallas fused linear cross-entropy (no `[tokens, 248320]` logits in
  HBM) with one hidden block (`(b, h, v) = (1024, 2048, 640)`).
* **MoE tiles**: 512-row grouped-matmul and dWeight tiles (~2K routed rows per
  expert at 64K tokens per rank).
* **Logging** every 5 steps: a per-step `loss.item()` keeps the host from
  dispatching the next step while the current one runs.
* **XLA and libtpu flags**: `xla_enable_hlo_sharding_v3=False` (avoids Shardy
  partitioner regressions on GraphTrainer child executables);
  `xla_tpu_dvfs_p_state=7` (throughput operating point); the SparseCore
  collective aggregator with the latency-hiding layer scheduler (overlaps the
  FSDP collectives with compute); SparseCore-offloaded async all-gathers;
  reduce-scatters on both SparseCores; and
  `xla_tpu_scoped_vmem_limit_kib=65472` (`63.94 MiB`, the maximum scoped VMEM
  cap accepted on TPU7x, carried from the Qwen3 recipes for Pallas tile tuning).

## Benchmark Results

Full 40-layer model (`qwen35_moe_35b_split_tokamax_benchmark`: no GatedDeltaNet
packed-document resets, `mixed_precision_reduce="float32"`,
`enable_tpu_topo_mesh=True`), context length 8192
(`--training.max_context_length=8192`), 65,536 tokens per rank (`8 × 8192`),
2 layers per executable, Cloud GKE (224-vCPU hosts), step 20 of 20:

| Slice (chips, ranks) | Step time | Tokens/s/rank | TFLOPS/rank | MFU | Trainer memory |
| --- | --- | --- | --- | --- | --- |
| `2x2x1` (4, 8) | 6.12 s | 10,708 | 235.44 | 20.41% | 36.58 GiB |
| `2x2x2` (8, 16) | 6.07 s | 10,797 | 237.40 | 20.58% | 36.58 GiB |
| `2x2x4` (16, 32) | 6.14 s | 10,681 | 234.86 | 20.36% | 36.58 GiB |
| `2x4x4` (32, 64) | 6.31 s | 10,389 | 228.44 | 19.80% | 36.58 GiB |

The recipe is tuned on `2x2x1` and run unchanged on the larger slices. Above
64K tokens per rank, `2x2x1` runs out of HBM with 2 layers per executable;
with 1 layer per executable, 96K tokens per rank reaches 20.10% MFU (comparable
to 20.41% at 64K).

`Trainer memory` is the trainer's `reserved_bytes.all.peak` metric (peak XLA
program reservation), excluding persistent ZeRO-3 sharded bf16 parameters,
gradients, and AdamW moments (`258 GiB / ranks`, or ~32.3 GiB per rank across
8 ranks).

## Notes

* With `reset_at_documents=False` (this recipe), `GatedDeltaNet` runs one
  recurrence per `[B, S]` row: documents packed into a row share the causal
  convolution and the delta-rule state. The default (`True`, for RL; or via
  `torchtitan.experiments.tpu.qwen3_5.gated_delta.gated_delta_document_resets`
  in `--override.imports`) restarts both at every document (`positions == 0`).
  Attention always masks documents via `positions`.
* The Tokamax kernels clamp the per-token log decay at -5
  (`TokamaxGatedDeltaKernel.Config.min_log_decay`): the backward kernel's
  16-token sub-block re-centering grows like `exp(-15 * g)` and overflows fp32
  below `-5.9`, which produced NaN gradients with the model's initialization
  (log decays down to about -70).
* The auxiliary-loss-free expert-bias balancer is off
  (`load_balance_coeff=None`): HF checkpoints carry no expert biases.
