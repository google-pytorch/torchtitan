# Qwen3.8-2.4T-A95B proxies on TPU7x 4x4x4

> **Status: not yet run.** These launch blocks are the harness for the first
> TPU7x jobs on Qwen3.8-2.4T-A95B. Their configs parse
> (`test_launch_configs_parse` in `qwen3_8/tests/test_qwen3_8_tpu.py`), and
> the same code path ran on one TPU v6e-8 host (`qwen38_2_4t_a95b_proxy`, and
> `qwen38_moe_debugmodel_ep8` with this recipe's libtpu flags). Record
> results under "Benchmark Results" after the first run.

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

* **Image.** Build the image from this torchtitan branch with a torch_tpu
  wheel that includes the eager O2 compilation of ragged dots (torch_tpu
  commit `9e40d7782`, "Compile ragged dot ops at O2 in eager mode"). Without
  it, `torch._grouped_mm` returns wrong rows whenever a rank's first experts
  get no tokens at dim 8192 (eager XLA:TPU at O1, libtpu >= 0.0.53). Put the
  wheel in the build context; the Dockerfile installs `torch_tpu-*.whl`.
* **Tokens per rank.** The EP transport is the dropless `ragged_all_to_all`,
  whose receive buffer is sized for the worst case: 64 * T * 10 rows of dim
  8192, so 10.7 GB bf16 at T = 1024 and 21.5 GB at 2048. The recipes default
  to 1K tokens per rank; Job B tries 2K. 4K and up need the bounded-capacity
  transport.
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

Not yet run.
