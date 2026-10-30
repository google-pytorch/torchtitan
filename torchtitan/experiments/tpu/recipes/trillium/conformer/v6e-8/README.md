# Recipe for training Conformer model on 8 TPU trillium cluster (v6e-8)

These are instructions for deploying Conformer training to an 8 chip v6e cluster using Cluster Toolkit (`gcluster`).

For Cluster Toolkit (`gcluster`) installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE
```

## Notes as of 5/21/26
- `--tpu_config.use_simple_fsdp` is enabled.
- `--compile.enable` is enabled with `--compile.mode=layer`.

## DDP with torch.compile

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

gcluster job submit \
  --name "$WORKLOAD_NAME" \
  --cluster "$CLUSTER_NAME" \
  --compute-type ct6e-standard-4t \
  --topology 2x4 \
  --num-slices 1 \
  --location "$ZONE" \
  --project "$PROJECT_ID" \
  --image "$WORKLOAD_IMAGE" \
  --env=WORKERS_0_HOSTNAME="$WORKLOAD_NAME-main-job-0-0.$WORKLOAD_NAME" \
  --command "
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.conformer.train_minimal \
    --module=torchtitan.experiments.tpu.conformer \
    --config=conformer_test \
    --parallelism.data_parallel_shard_degree=1 \
    --parallelism.data_parallel_replicate_degree=-1 \
    --training.num_tokens_per_microbatch_per_dp_rank=262144 \
    --training.dtype=float32 \
    --training.mixed_precision_param=bfloat16 \
    --tpu_config.use_simple_fsdp \
    --compile.mode=layer \
    --splash_attention_kernel.use_splash_attention_kernel \
    --metrics.log_freq=1 \
    --conformer.use-ctc-loss \
    --compile.enable \
    --dataloader.dataset=random \
    --training.steps=20 \
    activation-checkpoint:full
"
```

**9/30/26: With this configuration you should observe approximately the following metrics**

- Average TPS/chip: **87,181**
- Average MFU: **6.37%**
- Total TPS (8 chips): **697,448**



## DDP eager mode

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

gcluster job submit \
  --name "$WORKLOAD_NAME" \
  --cluster "$CLUSTER_NAME" \
  --compute-type ct6e-standard-4t \
  --topology 2x4 \
  --num-slices 1 \
  --location "$ZONE" \
  --project "$PROJECT_ID" \
  --image "$WORKLOAD_IMAGE" \
  --env=WORKERS_0_HOSTNAME="$WORKLOAD_NAME-main-job-0-0.$WORKLOAD_NAME" \
  --command "
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.conformer.train_minimal \
    --module=torchtitan.experiments.tpu.conformer \
    --config=conformer_test \
    --parallelism.data_parallel_shard_degree=1 \
    --parallelism.data_parallel_replicate_degree=-1 \
    --training.num_tokens_per_microbatch_per_dp_rank=262144 \
    --training.dtype=float32 \
    --training.mixed_precision_param=bfloat16 \
    --tpu_config.use_simple_fsdp \
    --tpu_config.eager_mode=DEFER_AND_FUSE \
    --splash_attention_kernel.use_splash_attention_kernel \
    --metrics.log_freq=1 \
    --conformer.use-ctc-loss \
    --dataloader.dataset=random \
    --training.steps=20 \
    activation-checkpoint:full
"
```

**9/30/26: With this configuration you should observe approximately the following metrics**

- Average TPS/chip: **88,664**
- Average MFU: **6.48%**
- Total TPS (8 chips): **709,312**

