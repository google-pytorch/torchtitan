# Recipe for training FLUX.1-dev model on 8 TPU trillium cluster (v6e-8)

These are instructions for deploying FLUX.1-dev training (all 12B
parameters trained) to an 8 chip v6e cluster using Cluster Toolkit (`gcluster`).

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
For Cluster Toolkit (`gcluster`) installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE
```

> [!NOTE]
> As of upstream `d771abb0` (9/11/26), `--dataloader.img_size` and
> `--training.local_batch_size` no longer exist. The image size now lives on the
> dataset processor, which is not exposed on the CLI (`flux_dev` defaults to
> `img_size=256`), so use the TPU-side `--flux.img_size=1024` instead. It sets
> the processor image size and `max_context_length` (64 x 64 image patches +
> 512 T5 tokens = 4608). The batch size is expressed as a token budget:
> `num_tokens_per_microbatch_per_dp_rank` = samples x 4608.

## SimpleFSDP + SplashAttention + torch.compile

### float32 + MixedPrecision bfloat16

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
  --command "\
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=flux \
    --config=flux_dev \
    --metrics.log_freq=1 \
    --training.steps=40 \
    --training.dtype=float32 \
    --training.num_tokens_per_microbatch_per_dp_rank=9216 \
    --flux.img_size=1024 \
    --flux.random_dataset \
    --parallelism.use_simple_fsdp \
    --splash-attention-kernel.use-splash-attention-kernel \
    --compile.enable \
    --compile.backend=tpu \
    --libtpu-init-flags=xla_tpu_scoped_vmem_limit_kib=65536 \
    activation-checkpoint:full \
    --activation-checkpoint.no-preserve-rng-state"
```

**9/29/26: With this configuration you should observe the following metrics**

- Average Step time: **2.82s**
- Average Step time / PDB: **1.41s**

### bfloat16

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
  --command "\
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=flux \
    --config=flux_dev \
    --metrics.log_freq=1 \
    --training.steps=40 \
    --training.dtype=bfloat16 \
    --training.num_tokens_per_microbatch_per_dp_rank=13824 \
    --flux.img_size=1024 \
    --flux.random_dataset \
    --parallelism.use_simple_fsdp \
    --splash-attention-kernel.use-splash-attention-kernel \
    --compile.enable \
    --compile.backend=tpu \
    --libtpu-init-flags=xla_tpu_scoped_vmem_limit_kib=65536 \
    activation-checkpoint:full \
    --activation-checkpoint.no-preserve-rng-state"
```

**9/29/26: With this configuration you should observe the following metrics**

- Average Step time: **3.59s**
- Average Step time / PDB: **1.20s**
