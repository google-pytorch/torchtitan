# Recipe for training Llama3-70B model on a 64 device TPU ironwood cluster (tpu7x-64)

These are instructions for deploying Llama3-70B training to a 64 device (32 chip) tpu7x cluster using Cluster Toolkit (`gcluster`).

For Cluster Toolkit (`gcluster`) installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE
```

## Notes

- **Devices vs chips**: an ironwood chip exposes two TensorCores, and torchtitan runs one
  process per core. `tpu7x-64` is therefore 64 devices on 32 chips, run as 8 hosts x 8
  processes. Throughput and MFU below are per device (per core); MFU is measured against
  the per-core bf16 peak of 1,153.5 TFLOP/s (half of the 2,307 TFLOP/s chip peak).
- **Master weights are float32** (config_registry default). Compute runs in bf16 via `mixed_precision_param=bfloat16`.
- VMEM tuning: `--libtpu-init-flags=xla_tpu_scoped_vmem_limit_kib=131072`, a
  libtpu init flag the config exports before PJRT initialization.
- Dataset: `c4_test` (synthetic/test dataset).
- Tokenizer: `tests/assets/tokenizer`.
- Model configuration used is `llama3_70b` from [`llama3/config_registry.py`](../../../../llama3/config_registry.py).

## Training Configuration Details

Defaults from [`llama3_70b`](../../../../llama3/config_registry.py):
- **Sequence Length**: 8192
- **Torch Compile**: Layerwise, with the `tpu` backend. `compile.components=["model"]`
  together with the default `compile.mode="layer"` compiles each
  TransformerBlock separately (`fullgraph=True`) rather than the model as one graph.
- **Simple FSDP**: Enabled.
- **Splash Attention**: Enabled with block size 2048.
- **Loss Kernel**: Enabled.
- **Activation Checkpointing**: Full.

Overridden on the command line:
- **Batch**: 4 sequences of 8192 tokens per device, passed as
  `--training.num_tokens_per_microbatch_per_dp_rank=32768` (Global batch size = 4 * 64 = 256)
- **Topology-Aware Mesh**: Enabled via `--parallelism.enable_tpu_topo_mesh`.

## Training Command

```bash
export WORKLOAD_NAME="llama3-70b-tpu7x64-$(date +%H%M%S)"

gcluster job submit \
  --name "$WORKLOAD_NAME" \
  --cluster "$CLUSTER_NAME" \
  --compute-type tpu7x \
  --topology 2x4x4 \
  --num-slices 1 \
  --project "$PROJECT_ID" \
  --location "$ZONE" \
  --gke-disable-parallel-containers \
  --image "$WORKLOAD_IMAGE" \
  --env="WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-main-job-0-0.${WORKLOAD_NAME}" \
  --env="WORKLOAD_NAME=$WORKLOAD_NAME" \
  --command "torchrun \
    --nnodes=8 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.llama3 \
    --config=llama3_70b \
    --hf_assets_path=tests/assets/tokenizer \
    --dataloader.dataset_path=tests/assets/c4_test \
    --training.steps=20 \
    --training.num_tokens_per_microbatch_per_dp_rank=32768 \
    --parallelism.enable_tpu_topo_mesh \
    --libtpu-init-flags=xla_tpu_scoped_vmem_limit_kib=131072"
```

## Expected Metrics

With this configuration you should observe the following metrics (as of 9/28/26):

- TPS/device: **1,157**
- MFU: **48.29%**
- Total TPS (64 devices / 32 chips): 74,049
