# Gemma 4 31B on TPU7x 4x4x4

This recipe runs the `gemma4_31b` SimpleFSDP benchmark on one `tpu7x-4x4x4`
slice: 64 chips, 128 TensorCores, one process per core (16 hosts x 8 processes).
See the TPU [README](../../../../README.md) for Cluster Toolkit (`gcluster`)
installation, cluster setup and the Docker image build.

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE
```

## Notes

- **Batch shape**: 8 sequences of 4096 tokens per data-parallel rank
  (`--training.num_tokens_per_microbatch_per_dp_rank=32768`). The
  `gemma4_31b` config defaults to 2 sequences, which is what fits a
  `tpu7x-2x2x1` slice; with the weights and optimizer state sharded 128 ways,
  a 4x4x4 slice has room for more. Measured on this slice with the flags below
  and the config's default 512 Splash blocks (the 2-sequence row is without the
  scheduling flags):

  | Sequences per rank | Step time | TPS per rank | MFU | Peak HBM |
  |---|---|---|---|---|
  | 2 | 3.598 s | 2,277 | 37.7% | 35.2 GiB (36.7%) |
  | 4 | 6.681 s | 2,453 | 40.6% | 43.4 GiB (45.2%) |
  | 8 | 13.107 s | 2,500 | 41.3% | 64.7 GiB (67.4%) |

  Use `--training.num_tokens_per_microbatch_per_dp_rank=16384` for 4
  sequences: it gives up 1.8% throughput for 21 GiB less peak HBM per device.
- **Attention and loss kernels**: Splash attention and the fused soft-capped
  cross entropy are enabled by the `gemma4_31b` config.
- **Splash block sizes**: the command raises the Splash blocks from the
  config's 512 to 1024, keeping the compute blocks at 512
  (`--splash_attention_kernel.sa_block_*`). Measured with the flags below at 8
  sequences per rank on this slice:

  | Splash blocks (compute blocks) | Step time | TPS per rank | MFU |
  |---|---|---|---|
  | 512 (512) | 13.06 s | 2,510 | 41.5% |
  | 1024 (512) | 12.51 s | 2,619 | 43.3% |

  Peak HBM is the same. Like the VMEM flag, a block-size change reorders
  floating-point sums: loss matches at step 1 and the grad_norm agrees to
  0.05%, then trajectories drift apart from random initialization. On a
  `tpu7x-2x2x2` slice, 1024 compute blocks, 2048 blocks and a 2048 or 512
  backward Q block were all slower than this setting.

## libtpu init flags

The flags ride on the command line through `--libtpu-init-flags`, which
exports them before PJRT initialization.

| Flag | Why |
|---|---|
| `xla_tpu_dvfs_p_state=7` | TPU7x's throughput operating point. Trades device power for clock speed; does not change the graph (loss and grad_norm are unchanged at every step). |
| `xla_tpu_scoped_vmem_limit_kib=65536` | Raises the scoped VMEM budget from the 32 MiB default to the 64 MiB a TPU7x core has, so the compiler keeps more intermediates in VMEM. Changes the compiler's tiling, so results differ at round-off level. Together with the P-state: +15.6% throughput at 2 sequences per rank. |
| `xla_tpu_enable_sparse_core_collective_aggregator=true` `xla_tpu_enable_latency_hiding_layer_scheduler=true` | Lets the latency-hiding layer scheduler overlap the FSDP collectives with compute. The scheduler requires the SparseCore collective aggregator, so keep the two enabled or disabled together. |
| `xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false` `xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true` `xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true` | Splits multi-dimensional reduce-scatters into hierarchical stages across both SparseCores. Together with the scheduler pair: +1.4% throughput. |

## Training Command

```bash
LIBTPU_INIT_FLAGS="xla_tpu_dvfs_p_state=7 \
xla_tpu_scoped_vmem_limit_kib=65536 \
xla_tpu_enable_sparse_core_collective_aggregator=true \
xla_tpu_enable_latency_hiding_layer_scheduler=true \
xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false \
xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true"

gcluster job submit \
  --name "$WORKLOAD_NAME" \
  --cluster "$CLUSTER_NAME" \
  --location "$ZONE" \
  --project "$PROJECT_ID" \
  --compute-type tpu7x \
  --topology 4x4x4 \
  --num-slices 1 \
  --image "$WORKLOAD_IMAGE" \
  --gke-disable-parallel-containers \
  --env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-main-job-0-0.$WORKLOAD_NAME" \
  --command "\
    torchrun \
    --nnodes=16 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.examples.gemma4 \
    --config=gemma4_31b \
    --hf_assets_path=tests/assets/tokenizer \
    --dataloader.dataset_path=tests/assets/c4_test \
    --training.num_tokens_per_microbatch_per_dp_rank=32768 \
    --training.steps=30 \
    --splash_attention_kernel.sa_block_q=1024 \
    --splash_attention_kernel.sa_block_kv=1024 \
    --splash_attention_kernel.sa_block_kv_compute=512 \
    --splash_attention_kernel.sa_block_dkv=1024 \
    --splash_attention_kernel.sa_block_q_dkv=1024 \
    --splash_attention_kernel.sa_block_kv_dkv=1024 \
    --splash_attention_kernel.sa_block_kv_dkv_compute=512 \
    --splash_attention_kernel.sa_block_q_dq=1024 \
    --splash_attention_kernel.sa_block_kv_dq=1024 \
    --libtpu-init-flags='$LIBTPU_INIT_FLAGS'"
```

The command trains from random initialization on the bundled `c4_test` data,
which is what the numbers below measure. Exporting `LIBTPU_INIT_ARGS` from the
launcher still works and wins over `--libtpu-init-flags` for any flag it names.

## Benchmark Results

Run Date: 11/25/2026 (mean of steps 10-30, two runs of the command above)

* **MFU:** 43.3%
* **TPS (Tokens Per Second):** 2,619 per rank (about 335,000 across the slice)
* **TFLOPS:** 499.5 per core
* **Memory Usage:** 64.72 GiB (67.42%)
* **Step Time (end-to-end):** 12.51s

Measured at torchtitan `d0f0316e` with torch_tpu 0.3.0 and libtpu 0.0.50. The
two runs (12.507s and 12.518s per step) have identical loss and grad_norm at
all 30 steps. The same config with no libtpu flags, 512 Splash blocks and 2
sequences per rank reaches 1,969 TPS per rank and 32.6% MFU.

The batch-size table in the notes was measured with
`xla_tpu_scoped_vmem_limit_kib=65472`; a run with `65536` (64 MiB) gave
identical loss and grad_norm at all 30 steps and the same throughput.
