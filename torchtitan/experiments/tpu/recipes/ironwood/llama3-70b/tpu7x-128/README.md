# Recipe for training Llama3-70B with float8 on a 128 device TPU ironwood cluster (tpu7x-128)

These are instructions for deploying Llama3-70B training to a 128 device (64 chip, 4x4x4) tpu7x cluster using Cluster Toolkit (`gcluster`), in bf16 and with torchao float8 linears.

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
  process per core. `tpu7x-128` is therefore 128 devices on 64 chips, run as 16 hosts x 8
  processes. Throughput below is per device (per core); MFU is measured against the
  per-core bf16 peak of 1,153.5 TFLOP/s (half of the 2,307 TFLOP/s chip peak).
- **Master weights are float32** (config_registry default). Compute runs in bf16 via `mixed_precision_param=bfloat16`.
- **Float8**: `llama3_70b_float8` swaps the model's linears for torchao `Float8Linear`
  layers using the `rowwise` recipe, via `TpuFloat8LinearConverter`
  ([`float8_converter.py`](../../../../float8_converter.py)). The output projection
  (`output`/`lm_head`) stays in bf16. `use_fast_accum` is a no-op on TPU; TPU
  matmuls accumulate in float32.
- **torchao**: the float8 config needs torchao in the image. The TPU
  [Dockerfile](../../../../Dockerfile) installs it.
- Dataset: `c4_test` (synthetic/test dataset).
- Tokenizer: `tests/assets/tokenizer`.
- Model configurations used are `llama3_70b` and `llama3_70b_float8` from [`llama3/config_registry.py`](../../../../llama3/config_registry.py).

## Training Configuration Details

Defaults from [`llama3_70b`](../../../../llama3/config_registry.py):
- **Sequence Length**: 8192
- **Torch Compile**: Layerwise, with the `tpu` backend. `compile.components=["model"]`
  together with the default `tpu_config.compile_mode="layer"` compiles each
  TransformerBlock separately (`fullgraph=True`) rather than the model as one graph.
- **Simple FSDP**: Enabled.
- **Splash Attention**: Enabled with block size 2048.
- **Loss Kernel**: Enabled.
- **Activation Checkpointing**: Full.

Overridden on the command line:
- **Batch**: 2 sequences of 8192 tokens per device, passed as
  `--training.num_tokens_per_microbatch_per_dp_rank=16384` (Global batch size = 2 * 128 = 256)
- **Topology-Aware Mesh**: Enabled via `--parallelism.enable_tpu_topo_mesh`.
- **libtpu flags**: passed via `--extra-libtpu-init-flags` (see the command below).

## Training Command

Set `CONFIG=llama3_70b` for bf16 or `CONFIG=llama3_70b_float8` for float8.

```bash
export CONFIG=llama3_70b_float8
export WORKLOAD_NAME="llama3-70b-tpu7x128-$(date +%H%M%S)"

gcluster job submit \
  --name "$WORKLOAD_NAME" \
  --cluster "$CLUSTER_NAME" \
  --compute-type tpu7x \
  --topology 4x4x4 \
  --num-slices 1 \
  --project "$PROJECT_ID" \
  --location "$ZONE" \
  --gke-disable-parallel-containers \
  --image "$WORKLOAD_IMAGE" \
  --env="WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-main-job-0-0.${WORKLOAD_NAME}" \
  --env="WORKLOAD_NAME=$WORKLOAD_NAME" \
  --command "torchrun \
    --nnodes=16 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.llama3 \
    --config=$CONFIG \
    --hf_assets_path=tests/assets/tokenizer \
    --dataloader.dataset_path=tests/assets/c4_test \
    --training.steps=20 \
    --training.num_tokens_per_microbatch_per_dp_rank=16384 \
    --parallelism.enable_tpu_topo_mesh \
    --extra-libtpu-init-flags='xla_tpu_dvfs_p_state=7 xla_tpu_scoped_vmem_limit_kib=131072 xla_tpu_enable_offloading_copy_to_sparsecore=false xla_enable_async_all_gather=true xla_tpu_prefer_async_allgather_to_allreduce=true xla_tpu_enable_sparse_core_collective_offload_all_gather=true xla_tpu_copy_insertion_use_region_analysis=true xla_tpu_copy_insertion_use_region_analysis_limit=1000'"
```

## Expected Metrics

With this configuration you should observe the following steady-state metrics (as of 10/5/26):

| | bf16 (`llama3_70b`) | float8 (`llama3_70b_float8`) |
|---|---|---|
| Step time | 13.27 s | 10.02 s |
| TPS/device | 1,235 | 1,636 |
| TFLOP/s/device | 594 | 787 |
| MFU | 51.5% | N/A\* |
| Peak memory/device | 7.37 GiB | 8.67 GiB |
| Total TPS (128 devices / 64 chips) | 158,080 | 209,408 |

\* torchtitan does not report MFU when float8 is enabled (it logs `mfu: N/A`): with a mix of
float8 and bf16 matmuls there is no single peak to divide by.
Compare step time and TFLOP/s/device instead.