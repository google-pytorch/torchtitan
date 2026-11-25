# Recipe for training Qwen3-32B model on a 512 device TPU ironwood cluster (tpu7x-4x8x8)

These are instructions for deploying Qwen3-32B training to a 512 device (256 chip) tpu7x cluster using Cluster Toolkit (`gcluster`).

For Cluster Toolkit (`gcluster`) installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE
```

## Notes

- **Devices vs chips**: `tpu7x-4x8x8` is 512 devices on 256 chips, run as 64 hosts x 8 processes.
- **Based on the 4x4x4 recipe**: `qwen3_32b_4x8x8` is `qwen3_32b` (see
  [../tpu7x-4x4x4/README.md](../tpu7x-4x4x4/README.md)) with the same local
  batch, launch flags and pure FSDP over all 512 devices, plus two changes.
- **All-reduce plus slice for the w1/w3 gradient**: w1 and w3 are stored as one
  packed parameter per layer (`feed_forward.w13_packed.weight`, 51200 x 5120).
  At world 512 its gradient shard is 51200 / 512 = 100 rows. That is not a
  multiple of 8, so its reduce-scatter runs on the TensorCore instead of
  SparseCore (~2.5 s per step). The recipe reduces that gradient with an
  all-reduce plus a local slice (`fsdp_allreduce_slice_parameter_patterns`),
  which stays on SparseCore. This requires FP32 reductions, so there is no
  MaxText-parity (BF16 reduction) variant at this scale yet.
- **Interleave FSDP schedule**: before the w1/w3 change, interleave beat
  prefetch at this scale by 2.3% step time and 56 GiB peak HBM.
- **Tokenizer**: as in the 4x4x4 recipe, pass a Qwen3 tokenizer via
  `--hf_assets_path` for meaningful loss values.

## Training Command

```bash
export WORKLOAD_NAME="qwen3-32b-tpu7x-4x8x8-$(date +%H%M%S)"

gcluster job submit \
  --name "$WORKLOAD_NAME" \
  --cluster "$CLUSTER_NAME" \
  --compute-type tpu7x \
  --topology 4x8x8 \
  --num-slices 1 \
  --location "$ZONE" \
  --project "$PROJECT_ID" \
  --image "$WORKLOAD_IMAGE" \
  --gke-disable-parallel-containers \
  --env="PYTHONUNBUFFERED=1" \
  --env="WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-main-job-0-0.${WORKLOAD_NAME}" \
  --env="WORKLOAD_NAME=$WORKLOAD_NAME" \
  --command "torchrun \
    --nnodes=64 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.qwen3 \
    --config=qwen3_32b_4x8x8 \
    --training.steps=30"
```

## Expected Metrics

With this configuration (local batch 8, global batch 4096, sequence 4096), you
should observe the following per-device metrics averaged over steps 10-30 (as
of 9/25/2026, measured with the original `xpk` launch on streamed C4 via
`--dataloader.dataset=c4`; not yet re-measured with the `gcluster` command
above):

- **TPS (Tokens per second)**: ~2,656
- **TFLOPS**: ~544.0
- **MFU**: ~47.2%
- **Step Time (time_end_to_end)**: ~12.34s
- **Physical peak HBM**: 83.07GiB

`grad_norm` is logged as 0.0 because the optimizer runs inside the compiled
graph; gradient clipping is still applied there.
