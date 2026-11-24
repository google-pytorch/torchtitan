# Qwen3.8-2.4T-A95B proxies on TPU7x 4x4x4

> **Status:** Job A and the bf16 part of Job B ran (see "Benchmark Results"):
> 4 and 8 layers at 1K and 2K tokens per rank. Not yet run: FP32 state, 4K
> tokens, TP. These launch blocks follow the `gcluster` setup of the TPU
> README; the runs used an equivalent JobSet on a GKE TPU7x cluster with
> Kueue.

The full model does not fit on one slice. These recipes run real-per-rank-shape
proxies on 64 chips (128 ranks): full width (dim 8192, 64/4 heads of 256,
GatedDeltaNet 16/128 heads, expert hidden size 2048, vocab 248320), all 512
experts at EP=64 (8 per rank, as in the full model), and 4 or 8 of the 92
layers. They answer questions that need TPU7x and can't be answered on a v6e-8
host:

* **Job A (smoke):** does the EP=64 proxy compile and run on TPU7x?
* **Job B (memory and step-time sweep):** peak HBM and step time per layer
  (4 vs 8 layers), per token count and per state dtype. These numbers decide
  how many chips the full model needs.

Kernel tile sweeps (grouped matmul at dim 8192, Splash at head_dim 256) and
parity at depth are not covered here yet.

See the TPU [README](../../../../README.md) for Cluster Toolkit (`gcluster`)
installation, cluster setup with the Cloud Storage FUSE CSI driver, and the
Docker image build.

## Before you launch

* **libtpu.** At dim 8192, eager XLA:TPU (optimization level O1) miscompiles
  `torch._grouped_mm` on libtpu 0.0.53.x: when a rank's first experts get no
  tokens, the first non-empty expert's rows come back wrong. The 0.0.54
  nightlies fix it (the pure-JAX reproducer passes with
  `0.0.54.dev20261124`). Build the image with such a libtpu, or with a
  torch_tpu that compiles ragged dots at O2 in eager mode, which is what the
  runs below used.
* **Tokens per rank.** The EP transport is the dropless `ragged_all_to_all`
  with a bounded receive buffer (`ep_capacity_factor=2.0`,
  `ep_capacity_overflow="grow"`): twice this rank's routed rows, doubled
  whenever a step routes more, instead of the worst case of 64 * T * 10 rows
  of dim 8192 (10.7 GB bf16 per buffer at T = 1024). The buffer, and with it
  memory and step time, therefore depends on how evenly the router spreads
  tokens (see "Benchmark Results"). The recipes default to 1K tokens per rank
  and size RoPE for up to 8K, so Job B's token overrides work.
* **Expert matmul.** The TPU7x proxies run the routed experts with the Pallas
  grouped matmul in 512 x 2048 x 1024 tiles. XLA's `torch._grouped_mm` for
  these shapes needs 64.9 MB of scoped VMEM at the 63.9 MB limit the recipes
  set, and the Pallas defaults need 108-120 MB. XLA's path does compile with
  `--extra-libtpu-init-flags=xla_tpu_scoped_vmem_limit_kib=32768`, and agrees
  with the Pallas path on step 1 (loss 12.97524 vs 12.97523, same grad_norm).
* **TP.** TP > 1 failed for every model on TPU v6e-8 (XLA `RET_CHECK` on the
  strided FSDP replica groups). These recipes use TP=1.
* **Gate.** Before every dispatch: the image carries the commits you expect,
  and `test_launch_configs_parse` passes for the arguments you launch with.

## Environment

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE
export BUCKET_NAME=gs://YOUR_BUCKET_NAME   # mounted at /data
```

Outputs go to the bucket, one folder per run:

```
gs://$BUCKET_NAME/qwen38/<job>/<run>/
  tb/                 TensorBoard events (loss, memory, step time, MFU)
  profile_trace/      xprof traces, when profiling is on
  log_worker_<N>.txt  stdout and stderr of each host
gs://$BUCKET_NAME/qwen38/compile_cache/   torch_tpu tier-3 compilation cache
```

The compilation cache persists across jobs, so a later job with the same shapes
skips most of the compile time.

## Job A: smoke, then 10 steps

One allocation. The first run stops after 1 step, so a failure costs one
compile. The second runs 10 steps and captures an xprof trace at step 8.

```bash
export WORKLOAD_NAME=qwen38-job-a

gcluster job submit \
  --name "$WORKLOAD_NAME" \
  --cluster "$CLUSTER_NAME" \
  --location "$ZONE" \
  --project "$PROJECT_ID" \
  --compute-type tpu7x \
  --topology 4x4x4 \
  --num-slices 1 \
  --image "$WORKLOAD_IMAGE" \
  --gke-disable-parallel-containers \
  --service-account workload-identity-k8s-sa \
  --mount "$BUCKET_NAME;/data;rw" \
  --env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-main-job-0-0.$WORKLOAD_NAME" \
  --env TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/data/qwen38/compile_cache \
  --command '
    set -o pipefail
    run() {
      out=/data/qwen38/job-a/$1; shift
      mkdir -p $out
      torchrun --nnodes=16 --nproc_per_node=8 --rdzv_backend=static \
        --rdzv_endpoint=$WORKERS_0_HOSTNAME:29501 --node_rank=$TPU_WORKER_ID \
        -m torchtitan.experiments.tpu.train \
        --module=torchtitan.experiments.tpu.qwen3_8 \
        --dump_folder=$out "$@" 2>&1 | tee $out/log_worker_$TPU_WORKER_ID.txt
    }
    run smoke --config=qwen38_2_4t_a95b_proxy_tpu7x \
      --training.steps=1 --metrics.log_freq=1 &&
    run main --config=qwen38_2_4t_a95b_proxy_tpu7x \
      --training.steps=10 --metrics.log_freq=1 --metrics.enable_tensorboard \
      --profiler.enable_profiling --profiler.profile_freq=8 \
      --profiler.profiler_warmup=3 --profiler.profiler_active=1'
```

## Job B: memory and step-time sweep

One allocation runs all points in sequence. A failing point (for example out of
memory) is recorded and the sweep moves on, so one allocation answers every
question it can.

```bash
export WORKLOAD_NAME=qwen38-job-b

gcluster job submit \
  --name "$WORKLOAD_NAME" \
  --cluster "$CLUSTER_NAME" \
  --location "$ZONE" \
  --project "$PROJECT_ID" \
  --compute-type tpu7x \
  --topology 4x4x4 \
  --num-slices 1 \
  --image "$WORKLOAD_IMAGE" \
  --gke-disable-parallel-containers \
  --service-account workload-identity-k8s-sa \
  --mount "$BUCKET_NAME;/data;rw" \
  --env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-main-job-0-0.$WORKLOAD_NAME" \
  --env TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/data/qwen38/compile_cache \
  --command '
    set -o pipefail
    for layers in 4l 8l; do for t in 1024 2048; do for dtype in bfloat16 float32; do
      config=qwen38_2_4t_a95b_proxy_tpu7x; [ $layers = 8l ] && config=${config}_8l
      out=/data/qwen38/job-b/$layers-t$t-$dtype
      mkdir -p $out
      torchrun --nnodes=16 --nproc_per_node=8 --rdzv_backend=static \
        --rdzv_endpoint=$WORKERS_0_HOSTNAME:29501 --node_rank=$TPU_WORKER_ID \
        -m torchtitan.experiments.tpu.train \
        --module=torchtitan.experiments.tpu.qwen3_8 --config=$config \
        --dump_folder=$out --training.steps=10 \
        --metrics.log_freq=1 --metrics.enable_tensorboard \
        --training.max_context_length=$t \
        --training.num_tokens_per_microbatch_per_dp_rank=$t \
        --training.dtype=$dtype 2>&1 | tee $out/log_worker_$TPU_WORKER_ID.txt
      echo "rc=$?" > $out/status_worker_$TPU_WORKER_ID.txt
    done; done; done'
```

`float32` stores parameters, gradients and AdamW state in FP32 (16 bytes per
parameter, against 8 for `bfloat16`) with bf16 compute.

## Reading the results

Each run's `tb/` folder holds the per-step scalars. Peak HBM per rank is
`memory/max_reserved(GiB)`, step time is `time_metrics/end_to_end(s)`; also
look at `throughput(tps)`, `mfu(%)`, `loss_metrics/global_avg_loss` and
`grad_norm`. Judge memory and step time from steps after the first two, which
include compilation.

```bash
gsutil -m cp -r "$BUCKET_NAME/qwen38/job-b" .
python - <<'EOF'
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for run in sorted(glob.glob("job-b/*/")):
  events = glob.glob(f"{run}tb/**/events*", recursive=True)
  if not events:
    print(f"{run}: no events (see status_worker_*.txt and log_worker_*.txt)")
    continue
  ea = EventAccumulator(events[0], size_guidance={"scalars": 0}); ea.Reload()
  last = lambda tag: ea.Scalars(tag)[-1].value
  print(f"{run}: peak {last('memory/max_reserved(GiB)'):.1f} GiB, "
        f"step {last('time_metrics/end_to_end(s)'):.2f} s, "
        f"loss {last('loss_metrics/global_avg_loss'):.3f}")
EOF
```

Per-layer memory and step time are the 8-layer value minus the 4-layer value,
divided by 4. Multiply by 92 and add the 4-layer intercept to estimate the full
model at the same tokens per rank and EP degree.

To stop a job early:

```bash
gcluster job cancel "$WORKLOAD_NAME" --cluster "$CLUSTER_NAME" \
  --location "$ZONE" --project "$PROJECT_ID"
```

## Benchmark Results

All runs: TPU7x 4x4x4 (128 ranks), all 512 experts at EP=64 (8 per rank,
expert state FSDP-sharded over the 2 ranks that hold the same experts), bf16
state, correctness path (eager, plain SDPA, reference GatedDeltaNet, full
activation checkpointing), libtpu 0.0.53.1 with torch_tpu compiling ragged
dots at O2. Medians of steps 2-10 of 10. Run date 11/24/2026. These runs
used a global-batch auxiliary load-balancing loss (coefficient 1e-3) from an
earlier revision of the recipes; the recipes now use upstream's expert-bias
balancer, which these numbers do not reflect yet.

Recipes as they are (dropless, growing receive buffer):

| Layers | Tokens per rank | Peak HBM per rank | Step time | MFU |
|---:|---:|---:|---:|---:|
| 4 | 1024 | 39.12 GiB (41%) | 3.94 s | 0.82% |
| 8 | 1024 | 50.07 GiB (52%) | 10.51 s | 0.51% |
| 4 | 2048 | 48.40 GiB (50%) | 6.39 s | 1.01% |
| 8 | 2048 | 74.20 GiB (77%) | 19.56 s | 0.55% |

From random initialization the proxies' loss is unstable (13.1 -> 7.0 -> 12.5
over the first five steps at 8 layers), and routing with it: at 8 layers some
rank received more than 8x its mean share of routed rows, so the receive
buffers grew to 163840 rows (six of them, 15 GiB, live at the peak) and step
time doubled. To measure memory and step time without that, the same points
ran with a fixed receive buffer, `ep_capacity_overflow="drop"` at
`ep_capacity_factor=2.0` on every `TpuAllToAllTokenDispatcher.Config`. That
drops the routes past the cap, so it is a measurement setting, not a
training recipe:

| Layers | Tokens per rank | Peak HBM per rank | Step time | MFU |
|---:|---:|---:|---:|---:|
| 4 | 1024 | 39.10 GiB (41%) | 2.72 s | 1.19% |
| 8 | 1024 | 43.78 GiB (46%) | 5.16 s | 1.04% |
| 4 | 2048 | 42.48 GiB (44%) | 4.39 s | 1.48% |
| 8 | 2048 | 47.22 GiB (49%) | 8.73 s | 1.24% |

In every fixed-buffer run the peak is at the LM head and loss: 8 GiB of
stack, 22.7 GiB of vocabulary-sized matrices (the embedding and LM head,
248320 x 8192, gathered unsharded in bf16 and in FP32), and the logits (2.8
GiB at 1K tokens, 5.7 GiB at 2K). In the profiled steps, collectives take most
of the device time: at 4 layers and 1K tokens, reduce-scatter 47-48%,
all-gather 14-15%, offloaded collectives 13%, ragged all-to-all 6-7%, and
the kernels (custom calls) 11-12%.

## Sizing the full model (estimate)

From the fixed-buffer points, at EP=64 with bf16 state:

| Tokens per rank | Per layer per rank | Fixed (stack, vocabulary, logits) |
|---:|---:|---:|
| 1024 | 1.17 GiB | 34.4 GiB |
| 2048 | 1.19 GiB | 37.7 GiB |

The per-layer term is the layer's sharded state: 26.26B parameters / 128
ranks x 6 bytes (bf16 parameters and both AdamW moments; at the peak, which
precedes the backward pass, the gradients are not yet allocated) = 1.15 GiB.
Per-layer activations are negligible under full activation checkpointing, so
the per-layer term shrinks with the rank count. The fixed term does not.

For 92 layers at 1K tokens per rank (94.7 GiB usable per rank, 2 ranks per
chip):

| State | 64 chips | 128 chips | 256 chips | 512 chips |
|---|---:|---:|---:|---:|
| bf16 (6 B/param at the peak) | 140 GiB | 87 GiB | 61 GiB | 48 GiB |
| FP32 (12 B/param at the peak) | 245 GiB | 140 GiB | 87 GiB | 61 GiB |

So about 128 chips with bf16 state (tight) or 256 with FP32 state (tight), and
twice that for headroom. Caveats: this is a two-point linear fit; it ignores
the gradients late in the backward pass (2 or 4 more bytes per parameter,
partly offset by the vocabulary buffers being freed by then); it uses the
fixed buffer, so routing imbalance adds to it; and 4K tokens adds about 8
GiB of logits. The largest term that does not shrink with chips is the 22.7
GiB of unsharded vocabulary matrices, the first thing to cut.
