# Qwen3.5-397B-A17B with expert parallelism on TPU7x 2x4x4

This recipe runs a Qwen3.5-397B-A17B (60 layers, 512 routed experts, top-10)
8K-context pretraining benchmark with expert parallelism (EP) on 64 TPU cores
(32 chips, 8 hosts): `qwen35_moe_397b_split_tokamax_ep_benchmark` in
`torchtitan/experiments/tpu/qwen3_5/config_registry.py`. See the TPU
[README](../../../../README.md) for Cluster Toolkit (`gcluster`) installation
and cluster setup, and the
[Qwen3.5-35B-A3B EP recipe](../../qwen3.5-35b-a3b/tpu7x-2x2x2/README.md) for
what EP changes in a recipe (the expert mesh, the transports, token chunking,
the paired expert matmuls, the graph split, and the topology-aware mesh).

## Settings

- **BF16 parameters, gradients and AdamW state** (8 bytes per parameter, about
  46 GiB per rank on 64 ranks; in FP32 the parameters, gradients and optimizer
  state alone would take about 92 GiB per rank) and a BF16 gradient
  reduce-scatter.
- **24,576 tokens per rank:** 3 sequences of 8,192 tokens.
- **EP = 8 with Ring-of-Experts** (EP is at most top-k = 10) in **12 token
  chunks** of 2,048 tokens per MoE layer. Each rank all-gathers 64 experts'
  weights per layer instead of all 512.
- **Expert matmul tiles** for D = 4096, F = 1024: the expert dWeight matmuls
  tile K at 1024 to fit VMEM, and the transposed W2 tiles D at 2048.
- **The default fused cross-entropy tiles** for the 4,096-wide hidden state.

## Training command

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE
export WORKLOAD_NAME="qwen35-397b-ep-$(date +%H%M%S)"

gcluster job submit \
  --name "$WORKLOAD_NAME" \
  --cluster "$CLUSTER_NAME" \
  --compute-type tpu7x \
  --topology 2x4x4 \
  --num-slices 1 \
  --location "$ZONE" \
  --project "$PROJECT_ID" \
  --image "$WORKLOAD_IMAGE" \
  --gke-disable-parallel-containers \
  --env="PYTHONUNBUFFERED=1" \
  --env="WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-main-job-0-0.${WORKLOAD_NAME}" \
  --command "torchrun \
    --nnodes=8 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.qwen3_5 \
    --config=qwen35_moe_397b_split_tokamax_ep_benchmark \
    --training.steps=20"
```

## Other EP degrees

- **EP = 16** (the ragged all-to-all, since EP is above top-k):
  `--parallelism.expert_parallel_degree=16 --tpu_config.ep_token_chunks=16`.
- **Without EP:** `--parallelism.expert_parallel_degree=1` keeps the same
  graph split, rematerialization and batch. At EP = 1 the compiled gradient
  norm works again (`--compile.compile_grad_norm`). Every rank then
  all-gathers all 512 experts' weights for each layer, which needs much more
  HBM than EP = 8.
- **Without chunking:** `--tpu_config.ep_token_chunks=1`. Each layer's
  dropless expert buffer then holds `8 * 24,576 * 10` rows at EP = 8.
