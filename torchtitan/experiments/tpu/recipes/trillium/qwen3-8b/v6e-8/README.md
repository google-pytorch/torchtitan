# Recipe for training Qwen3-8B model on 8 TPU trillium cluster (v6e-8)

> **Note (8/30/2026):** This recipe uses the Regular Trainer configuration for Qwen3-8B. The Graph Trainer currently still encounters Out Of Memory (OOM) errors on v6e.

These are instructions for deploying Qwen3-8B training using the regular trainer to an 8 chip v6e cluster using Cluster Toolkit (`gcluster`).

For Cluster Toolkit (`gcluster`) installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE
```

## Notes

- **Model Configuration**: `reg_trainer_qwen3_8b` from `config_registry.py`.
- **Memory/Barrier Tuning**: `--libtpu-init-flags='xla_tpu_use_enhanced_launch_barrier=false xla_tpu_scoped_vmem_limit_kib=65536'`. These are libtpu init flags,
  carried by the torchtitan config and exported before PJRT initialization
  rather than set by the launcher.
- **Dataset**: `c4_test` (synthetic/test dataset)

## Training Configuration Details
We are running with the following key optimizations built into the configuration:
- **Sequence Length**: 2048 (passed as `--training.max_context_length=2048`)
- **Local Batch Size**: 6 sequences per data-parallel rank (Global batch size =
  6 * 8 = 48 sequences). Batch size is expressed in tokens on the command line:
  `--training.num_tokens_per_microbatch_per_dp_rank=12288` (6 sequences * 2048
  tokens).
- **Parallelism**: Simple FSDP with Topology Aware Mesh (`enable_tpu_topo_mesh=True`) mapped to the 2x4 ICI physical layout.
- **Activation Checkpointing**: FullAC (`SelectiveAC` causes OOM on v6e due to XLA activation limits).
- **Compile Mode**: Layer-by-layer compilation (`compile.mode="layer"`).
- **Precision**: `float32` reduction precision.
- **Loss Kernel & Splash Attention**: Both enabled (SA block sizes updated to 1024 for all blocks).

## Training Command
```bash
export WORKLOAD_NAME="qwen3-8b-v6e8-$(date +%H%M%S)"

gcluster job submit \
  --name "$WORKLOAD_NAME" \
  --cluster "$CLUSTER_NAME" \
  --compute-type ct6e-standard-4t \
  --topology 2x4 \
  --num-slices 1 \
  --location "$ZONE" \
  --project "$PROJECT_ID" \
  --image "$WORKLOAD_IMAGE" \
  --env="PYTHONUNBUFFERED=1" \
  --env="WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-main-job-0-0.${WORKLOAD_NAME}" \
  --env="WORKLOAD_NAME=$WORKLOAD_NAME" \
  --command "export PYTHONPATH=/torchtitan:\$PYTHONPATH && cd /torchtitan && torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.qwen3 \
    --config=reg_trainer_qwen3_8b \
    --training.num_tokens_per_microbatch_per_dp_rank=12288 \
    --training.max_context_length=2048 \
    --training.mixed_precision_reduce=float32 \
    --compile.enable \
    --parallelism.use_simple_fsdp \
    --parallelism.enable_tpu_topo_mesh \
    --splash_attention_kernel.use_splash_attention_kernel \
    --splash_attention_kernel.sa_block_q=1024 \
    --splash_attention_kernel.sa_block_kv=1024 \
    --splash_attention_kernel.sa_block_dkv=1024 \
    --splash_attention_kernel.sa_block_kv_compute=1024 \
    --splash_attention_kernel.sa_block_q_dkv=1024 \
    --splash_attention_kernel.sa_block_kv_dkv=1024 \
    --splash_attention_kernel.sa_block_kv_dkv_compute=1024 \
    --splash_attention_kernel.sa_block_q_dq=1024 \
    --splash_attention_kernel.sa_block_kv_dq=1024 \
    --loss_kernel.use_loss_kernel \
    --tpu_config.eager_mode=DEFER_AND_FUSE \
    --compile.materialize_forward_before_backward \
    --compile.mode=layer \
    --libtpu-init-flags='xla_tpu_use_enhanced_launch_barrier=false xla_tpu_scoped_vmem_limit_kib=65536'"
```

## Expected Metrics

With this configuration, we observe the following (run from 10/4/2026;
throughput metrics averaged over steps 2-15, loss/grad norm/memory at step 15):

- **TPS (Tokens per second)**: 5,753
- **TFLOPS**: 271.65
- **MFU**: 29.59%
- **Step Time (time_end_to_end)**: 2.14s
- **Loss**: 7.69447
- **Grad Norm**: 36.6285
- **Memory**: 3.67GiB (11.45%)
