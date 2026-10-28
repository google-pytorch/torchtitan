# Recipe for training Qwen3-32B model on a 128 device TPU ironwood cluster (tpu7x-4x4x4)

These are instructions for deploying Qwen3-32B training to a 128 device (64 chip) tpu7x cluster using Cluster Toolkit (`gcluster`).

For Cluster Toolkit (`gcluster`) installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE
```

## Notes

- **Devices vs chips**: an ironwood chip exposes two TensorCores, and torchtitan runs one process per core. `tpu7x-4x4x4` is therefore 128 devices on 64 chips, run as 16 hosts x 8 processes.
- **Based on the Qwen3-8B Zero-3 recipe**: `qwen3_32b` reuses `qwen3_8b_zero3`
  (SimpleFSDP with parameters resharded after forward, full graph
  rematerialization, tuned Splash tiles) and the 8B launch flags. Zero-2 would
  keep all ~61 GiB of gathered BF16 weights resident.
- **Packed QKV**: Q/K/V projections run as one packed matmul
  (`TpuPackedQKVConverter`). Model checkpoint keys stay `wq`/`wk`/`wv`;
  optimizer state is layout-specific.
- **Prefetch FSDP barriers**: `compile.fsdp_barrier_policy="prefetch"`.
- **Config-carried flags**: libtpu init flags (p-state 7, scoped VMEM, SparseCore
  collective offload including hierarchical reduce-scatter) ride on the config
  and are exported before PJRT initialization, so the command below does not
  repeat them.
- **Tokenizer**: like the 8B recipe, the default `hf_assets_path` is the test
  tokenizer in `./tests/assets/tokenizer`. Pass a Qwen3 tokenizer via
  `--hf_assets_path` for meaningful loss values. Step time depends slightly
  on the data, so compare runs only with the same data and tokenizer.

## Training Command

```bash
export WORKLOAD_NAME="qwen3-32b-tpu7x-4x4x4-$(date +%H%M%S)"

gcluster job submit \
  --name "$WORKLOAD_NAME" \
  --cluster "$CLUSTER_NAME" \
  --compute-type tpu7x \
  --topology 4x4x4 \
  --num-slices 1 \
  --location "$ZONE" \
  --project "$PROJECT_ID" \
  --image "$WORKLOAD_IMAGE" \
  --gke-disable-parallel-containers \
  --env="PYTHONUNBUFFERED=1" \
  --env="WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-main-job-0-0.${WORKLOAD_NAME}" \
  --env="WORKLOAD_NAME=$WORKLOAD_NAME" \
  --command "torchrun \
    --nnodes=16 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.qwen3 \
    --config=qwen3_32b \
    --training.steps=30"
```

## Expected Metrics

With this configuration (local batch 8, global batch 1024, sequence 4096), you
should observe the following per-device metrics averaged over steps 10-30 (as
of 10/4/2026, measured with the command above, i.e. on the default `c4_test`
dataset and test tokenizer):

- **TPS (Tokens per second)**: ~2,618
- **TFLOPS**: ~536.1
- **MFU**: ~46.5%
- **Step Time (time_end_to_end)**: ~12.52s
- **Physical peak HBM**: 35.66GiB (37.14%)

On 9/25/2026, measured on streamed C4 (`--dataloader.dataset=c4`), the same
configuration gave ~2,742 TPS, ~561.5 TFLOPS, ~48.7% MFU, ~11.95s per step and
36.46GiB peak HBM.

`grad_norm` is logged as 0.0 because the optimizer runs inside the compiled
graph; gradient clipping is still applied there.

## Alternative configurations

| Configuration | Purpose | Main differences from `qwen3_32b` |
|---|---|---|
| `qwen3_32b_maxtext_parity` | MaxText policy comparison | BF16 FSDP reductions and BF16 RoPE |

`qwen3_32b_maxtext_parity` measured ~11.83s per step, ~49.2% MFU and
35.10GiB peak HBM in the 9/25/2026 streamed-C4 measurement.

Keep canonical FP32-reduction results separate from MaxText-policy results.
The latter deliberately changes reduction precision and RoPE arithmetic.
