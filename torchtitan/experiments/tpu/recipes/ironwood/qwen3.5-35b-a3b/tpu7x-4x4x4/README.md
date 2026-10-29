# Qwen3.5-35B-A3B on Ironwood (TPU7x)

Qwen3.5-35B-A3B (text only): 40 layers in a 3:1 cycle of `GatedDeltaNet`
(linear attention) and full-attention layers, 256 routed experts (top-8) plus a
gated shared expert in every layer; 34.66B parameters, 2.95B active per token.
`qwen35_moe_35b_split_tokamax_benchmark` trains it on any TPU7x slice from one
`tpu7x-2x2x1` host (4 chips, 8 TensorCore ranks) up to 256 FSDP ranks (such as
`tpu7x-4x4x4`: 64 chips, 128 ranks; `TpuGroupedExperts` shards across the 256
experts on dim 0).

## Launch

One `tpu7x-2x2x1` host, at the recipe's context 2048 and 32K tokens per rank
(the first row of the measured results below):

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
64K tokens per rank (the second row).

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
* **GraphTrainer with graph split**: as one step graph, the 40-layer model
  neither compiles within an hour on 8 host CPUs nor fits in HBM on one host
  (XLA keeps most layers' unsharded `3 x 1 GiB` fp32 expert gradients alive
  until late in the step). Two-layer executables (`split=2`: consecutive pairs,
  half of which hold a full-attention layer) bound both, and repeated pairs hit
  the compilation cache (first step after 3-5 minutes on 224-CPU GKE hosts).
  With `backward_partition_mode=fsdp_pipeline`, `split=2` cuts the step from 84
  to 44 child executables; in the A/B that chose it (an earlier build of this
  recipe at 32K tokens per rank on one host), `split=2` had 24% shorter step
  time (+31% throughput) than one layer per executable (`split=1`).
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
* **XLA and libtpu flags**: `xla_enable_hlo_sharding_v3=False` (avoids Shardy
  partitioner regressions on GraphTrainer child executables);
  `xla_tpu_dvfs_p_state=7` (throughput operating point); the SparseCore
  collective aggregator with the latency-hiding layer scheduler (overlaps the
  FSDP collectives with compute); SparseCore-offloaded async all-gathers;
  reduce-scatters on both SparseCores; and
  `xla_tpu_scoped_vmem_limit_kib=65472` (`63.94 MiB`, the maximum scoped VMEM
  cap accepted on TPU7x, carried from the Qwen3 recipes for Pallas tile tuning).

## Benchmark Results

Measured on one `tpu7x-2x2x1` host (4 chips, 8 ranks) on Cloud GKE with the
full 40-layer model (`qwen35_moe_35b_split_tokamax_benchmark`,
`mixed_precision_reduce="float32"`, `enable_tpu_topo_mesh=True`), step 20 of 20:

| Context | Tokens/rank | Step time | Tokens/s/rank | TFLOPS/rank | MFU | Trainer memory |
| --- | --- | --- | --- | --- | --- | --- |
| 2048 | 32,768 (`16 × 2048`) | 4.78 s | 6,849 | 129.91 | 11.26% | 23.96 GiB |
| 8192 | 65,536 (`8 × 8192`) | 9.78 s | 6,703 | 147.38 | 12.78% | 45.11 GiB |

`Trainer memory` is the trainer's `reserved_bytes.all.peak` metric (peak XLA
program reservation), excluding persistent ZeRO-3 sharded bf16 parameters,
gradients, and AdamW moments (~32.3 GiB per rank across 8 ranks).

The GatedDeltaNet restarts at every packed document here, and Tokamax lays
every document out from a 64-token chunk boundary for that (see Notes).

## Notes

* Packed documents: the attention masks documents via `positions`, and the
  `GatedDeltaNet` restarts its causal convolution and delta-rule state at every
  document (`positions == 0`), through static-shape `[B, S]` segment ids
  instead of upstream's varlen `cu_seqlens`. The XLA kernel masks within
  chunks; Tokamax's kernels take `segment_ids` and lay every document out from
  a 64-token chunk boundary, keeping up to `max_num_segments` documents per row
  apart (default: one per 512 tokens and at least 4, so 12% more kernel tokens
  from a 2048-token context up). A row's documents beyond that run as one
  sequence, in the causal convolution too.
* The Tokamax kernels clamp the per-token log decay at -5
  (`TokamaxGatedDeltaKernel.Config.min_log_decay`): the backward kernel's
  16-token sub-block re-centering grows like `exp(-15 * g)` and overflows fp32
  below `-5.9`, which produced NaN gradients with the model's initialization
  (log decays down to about -70).
* The auxiliary-loss-free expert-bias balancer is off
  (`load_balance_coeff=None`): HF checkpoints carry no expert biases.
