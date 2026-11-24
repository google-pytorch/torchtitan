# Qwen3.5-35B-A3B with expert parallelism on TPU7x 2x2x2

This recipe runs the Qwen3.5-35B-A3B pretraining benchmark with expert
parallelism (EP) on 16 TPU cores (8 chips, 2 hosts):
`qwen35_moe_35b_split_tokamax_ep_benchmark`, which is
`qwen35_moe_35b_split_tokamax_benchmark` plus `_use_expert_parallel` (both in
`torchtitan/experiments/tpu/qwen3_5/config_registry.py`). See the TPU
[README](../../../../README.md) for Cluster Toolkit (`gcluster`) installation
and cluster setup.

## What EP changes in the recipe

- **Where the experts live.** The 256 routed experts are split across the `ep`
  ranks of an `(efsdp, ep)` mesh, and each expert's weights are FSDP-sharded
  across the `efsdp = 16 / ep` ranks of its group. Dense layers stay
  FSDP-sharded across all 16 ranks. Parameter and optimizer memory per rank
  doesn't change with the EP degree; what changes is that each rank
  all-gathers only its own experts and exchanges tokens instead.
- **How tokens move.** `ep_transport="auto"` picks Ring-of-Experts (all-gather
  the tokens and routing, compute the local experts, reduce-scatter the
  partial outputs) while EP is at most top-k (8 here), and the dropless ragged
  all-to-all above that. Neither reads token counts back to the host, so
  `TpuGraphTrainer` compiles the whole step.
- **Token chunking.** Each MoE layer runs in `ep_token_chunks` sequence chunks
  (4 at EP = 4) with at most two in flight, so one chunk's token exchange
  overlaps another chunk's expert matmuls and each chunk's worst-case expert
  buffer is `1/C` of the unchunked one. The expert weights are still
  all-gathered once per layer, and their per-chunk gradients are summed in
  FP32 inside the dWeight kernels: each chunk's kernel adds onto a running sum
  instead of storing its own (`--tpu_config.ep_chunk_dw_fp32=False` sums in
  BF16; `--tpu_config.ep_chunk_dw_in_kernel=False` sums in autograd instead).
- **Routing weights before the down projection**
  (`ep_scale_before_down_proj`), so combine is a plain sum.
- **Paired W1 + W3 expert matmuls stay on.** The paired Pallas kernel skips the
  unused tail of EP's static-size receive buffer like the unpaired kernels do;
  `--tpu_config.ep_paired_grouped_mm=False` switches to the unpaired kernels.
- **Graph split.** One layer per forward and backward executable with
  `backward_partition_mode=by_layer`, and the eager gradient norm (the compiled
  one needs a single mesh).
- **Topology-aware mesh stays on.** `ParallelDims.build_mesh` registers each
  mesh axis's logical layout with `spmd_types`, so the `efsdp` and `ep` axes
  pass its orthogonality check when `enable_tpu_topo_mesh` reorders the
  devices. `--parallelism.no-enable_tpu_topo_mesh` turns it off.
- **SparseCore flags.** Collective offload for the EP and expert-FSDP
  collectives, with gathers and copies kept on the TensorCore
  (`tpu_flags.py`). They ride on the config, so the command below doesn't pass
  `--libtpu-init-flags`.

## Training command

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE
export WORKLOAD_NAME="qwen35-35b-ep-$(date +%H%M%S)"

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
  --command "torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.qwen3_5 \
    --config=qwen35_moe_35b_split_tokamax_ep_benchmark \
    --training.steps=30"
```

## Other EP degrees

Pass the EP degree and the chunk count together; the chunk count doesn't follow
`--parallelism.expert_parallel_degree` on its own:

- EP = 2: `--parallelism.expert_parallel_degree=2 --tpu_config.ep_token_chunks=4`
- EP = 8: `--parallelism.expert_parallel_degree=8 --tpu_config.ep_token_chunks=8`
- Without chunking: `--tpu_config.ep_token_chunks=1`

`expert_parallel.default_num_token_chunks(ep_degree)` gives the recipe's
default: the EP degree, clamped to 4..16. The token count per rank must be
divisible by the chunk count; otherwise the layer uses the largest divisor
below it and logs a warning.

The EP = 1 recipe, `qwen35_moe_35b_split_tokamax_benchmark`, is the baseline
to compare against.
