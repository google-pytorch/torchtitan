# Recipe for training Qwen3-8B model on a 16 device TPU ironwood cluster (tpu7x-2x2x2)

These are instructions for deploying Qwen3-8B training to a 16 device (8 chip) tpu7x cluster using Cluster Toolkit (`gcluster`).

For Cluster Toolkit (`gcluster`) installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE
```

## Notes

- **Devices vs chips**: an ironwood chip exposes two TensorCores, and torchtitan runs one process per core. `tpu7x-16` is therefore 16 devices on 8 chips, run as 2 hosts x 8 processes. 
- **SparseCore Offloading**: This configuration achieves high MFU by utilizing
  SparseCore collective offloading for FSDP, requested through the config's
  `--libtpu-init-flags` and exported before PJRT initialization.
- **Graph Capture**: The optimizer and backward pass are fully captured into a single XLA graph via `--tpu_config.compile_optimizer` and `--tpu_config.optimizer_in_graph`.
- **Splash Attention**: Q/K/V layouts and backward tiles are configured in the `qwen3_8b` model config.
- **Batch shape**: 6 sequences of 2048 tokens per data-parallel rank. Batch size
  is expressed in tokens, so the command passes
  `--training.num_tokens_per_microbatch_per_dp_rank=12288` (6 sequences * 2048
  tokens) together with `--training.max_context_length=2048`.

## Training Command

```bash
export WORKLOAD_NAME="qwen3-8b-tpu7x16-$(date +%H%M%S)"

gcluster job submit \
  --name "$WORKLOAD_NAME" \
  --cluster "$CLUSTER_NAME" \
  --compute-type tpu7x \
  --topology 2x2x2 \
  --num-slices 1 \
  --location "$ZONE" \
  --project "$PROJECT_ID" \
  --image "$WORKLOAD_IMAGE" \
  --gke-disable-parallel-containers \
  --env="PYTHONUNBUFFERED=1" \
  --env="WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-main-job-0-0.${WORKLOAD_NAME}" \
  --env="WORKLOAD_NAME=$WORKLOAD_NAME" \
  --command "torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.qwen3 \
    --config=qwen3_8b \
    --training.num_tokens_per_microbatch_per_dp_rank=12288 \
    --training.max_context_length=2048 \
    --training.steps=30 \
    --training.mixed_precision_reduce=float32 \
    --parallelism.use_simple_fsdp \
    --parallelism.enable_tpu_topo_mesh \
    --tpu_config.use_graph_trainer \
    --tpu_config.compile_mode=aot_fx_trace \
    --tpu_config.compile_optimizer \
    --tpu_config.optimizer_in_graph \
    --libtpu-init-flags='xla_tpu_dvfs_p_state=7 xla_tpu_scoped_vmem_limit_kib=65472 xla_tpu_enable_sparse_core_collective_aggregator=true xla_tpu_enable_latency_hiding_layer_scheduler=true xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false'"
```

## Expected Metrics

With this configuration, you should observe the following metrics around step 30 (as of 10/21/2026):

- **TPS (Tokens per second)**: ~12,550
- **TFLOPS**: ~616
- **MFU**: ~53.45%
- **Step Time (time_end_to_end)**: ~0.98s
- **Memory**: 79.61GiB (82.92%)
