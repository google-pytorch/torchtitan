# Qwen3.5-35B-A3B on Ironwood (TPU7x)

Qwen3.5-35B-A3B (text only): 40 layers in a 3:1 cycle of `GatedDeltaNet`
(linear attention) and full-attention layers, 256 routed experts (top-8) plus a
gated shared expert in every layer; 34.66B parameters, 2.95B active per token.
`qwen35_moe_35b_split_tokamax_benchmark` trains it on any TPU7x slice from one
`tpu7x-2x2x1` host (4 chips, 8 TensorCore ranks) to `tpu7x-4x4x4` (64 chips,
128 ranks).

## Launch

One `tpu7x-2x2x1` host, at the recipe's context 2048 and 32K tokens per rank
(the first row of the results below):

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

`tpu7x-4x4x4` (16 hosts; see the TPU [README](../../../../README.md) for
cluster setup), at context 8192 and 64K tokens per rank:

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

The image needs a Tokamax build with the Kimi Delta Attention op
(`tokamax._src.ops.kda`, [openxla/tokamax](https://github.com/openxla/tokamax)
HEAD): the tokamax wheels up to 0.1.0 (and the `tokamax==0.0.14` pinned in
`requirements.txt`) do not have it, so
`qwen3_5/tests/test_tokamax_gated_delta.py` skips without it. Without Tokamax,
keep only the fused-loss override (`--override.imports
torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head`); the
GatedDeltaNet then runs the XLA `chunked_gated_delta`.

## Recipe

* **Precision and sharding**: bf16 parameters, gradients and AdamW moments (8
  bytes per parameter, 258 GiB in total), ZeRO-3 SimpleFSDP
  (`fsdp_reshard_after_forward=always`) with `mixed_precision_reduce="float32"`.
* **Selective activation checkpointing (`tpu_selective`)**: saves
  `moe.routing` (each router's top-k expert ids and the route sort
  permutation) and `attention.inner_attention` (the Splash Attention output
  and its log-sum-exp) across forward and backward executables, so backward
  does not re-run the MoE top-k and sort or the Splash Attention forward.
* **GraphTrainer with graph split**: as one step graph, the 40-layer model
  neither compiles within an hour on 8 host CPUs nor fits in HBM on one host
  (XLA keeps most layers' unsharded expert gradients alive until late in the
  step). Per-layer executables bound both, and repeated layers hit the
  compilation cache (first step after 3-5 minutes on 224-CPU GKE hosts). Two
  layers per forward and backward executable (`split=2`: consecutive pairs,
  half of which hold a full-attention layer) with
  `backward_partition_mode=fsdp_pipeline` cut the step from 84 to 44 child
  executables; in the A/B that chose it (an earlier build of this recipe at
  32K tokens per rank on one host), `split=2` was 24% faster than one layer
  per executable (`split=1`).
* **AdamW**: torch_tpu's fused AdamW outside the step graph
  (`optimizer_in_graph=False`, `compile_grad_norm=True`).
* **`GatedDeltaNet`**: Tokamax's Mosaic TPU KDA kernels
  (`TokamaxGatedDeltaKernel`, through the `tokamax_gated_delta_kernel`
  override), with the forward returning the backward kernels' residuals.
  Pretraining runs it over whole `[B, S]` rows (`reset_at_documents=False`):
  restarting at packed documents makes the kernels lay every document out
  from a 64-token chunk boundary (6.74 vs 9.78 s/step at 64K tokens per rank
  on `2x2x1`). The causal convolution and its input gradient are
  computed from shifted slices.
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

Full 40-layer model (`qwen35_moe_35b_split_tokamax_benchmark`: no GatedDeltaNet
packed-document resets, `mixed_precision_reduce="float32"`,
`enable_tpu_topo_mesh=True`), context length 8192
(`--training.max_context_length=8192`), 65,536 tokens per rank (`8 × 8192`),
2 layers per executable, Cloud GKE (224-vCPU hosts), step 20 of 20:

| Slice (chips, ranks) | Step time | Tokens/s/rank | TFLOPS/rank | MFU | HBM |
| --- | --- | --- | --- | --- | --- |
| `2x2x1` (4, 8) | 6.74 s | 9,727 | 213.89 | 18.54% | 40.12 GiB |

## Notes

* With `reset_at_documents=False` (this recipe), `GatedDeltaNet` runs one
  recurrence per `[B, S]` row: documents packed into a row share the causal
  convolution and the delta-rule state. The default (`True`, for RL) restarts
  both at every document (`positions == 0`). Attention always masks documents
  via `positions`.
* The Tokamax kernels clamp the per-token log decay at -5
  (`TokamaxGatedDeltaKernel.Config.min_log_decay`): the kernel's sub-chunk
  factoring overflows below it, which produced NaN gradients with the model's
  initialization (log decays down to about -34).
* The auxiliary-loss-free expert-bias balancer is off
  (`load_balance_coeff=None`): HF checkpoints carry no expert biases.
