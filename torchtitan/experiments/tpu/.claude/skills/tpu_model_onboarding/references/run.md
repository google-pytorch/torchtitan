# Running the model: where, what closes where, how to read a run

Launch commands and environment setup are in `torchtitan/experiments/tpu/README.md`;
this file is the part the README does not cover.

## Execution target (Step 0, question 1)

```
Target                        | Commands run                 | Chips                      | Artifacts come back as      | You need from the user
------------------------------|------------------------------|----------------------------|-----------------------------|----------------------------------
A. this machine is the TPU VM | directly                     | torch.accelerator          | files in --dump_folder      | nothing; verify the chips
                              |                              | .device_count()            |                             |
B. a TPU VM over ssh          | gcloud compute tpus tpu-vm   | the VM's, detected over    | scp of dump_folder, or the  | TPU name, zone, project, checkout
                              | ssh <name> --zone=<z>        | ssh with the same call     | remote log tail             | path on the VM
                              | --command='...'              |                            |                             |
C. GKE via Cluster Toolkit    | gcluster job submit, per the | the slice in the blueprint | GCS paths from the launch   | project, cluster, blueprint,
   (gcluster)                 | recipe README                |                            | block; job logs via kubectl | image registry, may you build
                              |                              |                            |                             | images, dispatch budget
```

- Rung 1 (correctness, 10 deterministic steps on `debugmodel`) and the
  kernel-parity rungs close on A or B with 1-4 chips; only Step 3 needs C.
- C only: run every CPU-closable check locally first, treat each dispatch as
  expensive (the gate in SKILL.md Step 3), stop at the agreed budget. The
  image is the artifact: do not copy edited source into a running pod and
  hash-verify it afterwards (every result then describes an overlay no image
  carries). Rebuild instead; a layered `FROM <previous image>` + `COPY` build
  of a pure-Python change takes seconds.
- No silent fallback: A reports no chip, B's ssh fails, C's image is missing
  -> stop and say so; never switch to CPU and mark run-based rows done.
- On a shared host check that nothing else holds `/dev/vfio/*` before a run.
- B: sync code before each run (push + checkout on the VM, or rsync the
  worktree) and record the VM-side commit in the run notes.
- xpk instead of gcluster: same procedure; traps in `gotchas.md`.

## What closes without a TPU

```
Check                                           | CPU | 1 chip | 4 chips | slice
------------------------------------------------|-----|--------|---------|------
reference-parity test (in-process or golden)    | yes | yes    |         |
state-dict round trip, flavor / config checks   | yes |        |         |
mask-semantics test                             | yes |        |         |
test_train_distributed cases (3 steps)          | yes*| yes    | yes     |      * --device=cpu on tiny flavors
10-step run, determinism repeat, step-1 oracle  |     | yes    | yes     |
kernel parity (splash, fused CE, MoE)           |     | yes    |         |
FSDP / SimpleFSDP end to end                    |     |        | yes     |
HLO custom_call evidence                        |     | yes    |         |
target recipe, benchmark table, trajectory      |     |        |         | yes
```

CPU runs: `TORCH_DEVICE_BACKEND_AUTOLOAD=0 JAX_PLATFORMS=cpu`.

## Reading a run

Everything the bar in SKILL.md asks for is in `--dump_folder`; bring it
back to where you work (A: it is there; B: scp; C: `gsutil -m cp -r`) so all
three targets end in the same procedure.

```
Evidence                      | Where                                          | Note
------------------------------|------------------------------------------------|----------------------------------------------
per-step loss, grad_norm      | <dump_folder>/tb/*/events.out.tfevents.*       | only with --metrics.enable_tensorboard
                              |                                                | --metrics.log_freq=1 (defaults: off, 10);
                              |                                                | full precision. stdout prints 5 digits,
                              |                                                | never compare runs from stdout
peak HBM, step time, TPS, MFU | metrics log lines; --tpu_config.enable_mfu_    | "memory:" is the XLA temp region, not the
                              | benchmark past warmup                          | heap; heap peak from buffer-assignment.txt
HLO                           | --extra-xla-flags=xla_dump_to=<dir> (an XLA    | grep custom_call names for enabled kernels,
                              | flag: libtpu init flags reject it); after-     | input_output_alias for donation; an exported
                              | optimizations .txt per program                 | XLA_FLAGS / LIBTPU_INIT_ARGS overrides recipe
                              |                                                | flags per key
structure                     | startup log                                    | patch / replacement counts == layer count;
                              |                                                | heads / kv_heads / vocab == checkpoint
```

Comparing two runs step by step at full precision (both launched with
`--metrics.enable_tensorboard --metrics.log_freq=1 --training.steps=N`):

```python
from tensorboard.backend.event_processing import event_accumulator as ea
def series(folder, tag="loss_metrics/global_avg_loss"):
    acc = ea.EventAccumulator(folder); acc.Reload()
    return {e.step: e.value for e in acc.Scalars(tag)}
a, b = series("<run1>/tb/<subdir>"), series("<run2>/tb/<subdir>")
N = <training.steps>
assert set(a) == set(b) == set(range(1, N + 1)), (sorted(a), sorted(b))
for s in range(1, N + 1):
    print(s, a[s], b[s], a[s] - b[s])
```

The step-set assertion is part of the check: a run that logged every 10th
step, or stopped early, compares equal on the steps it has and proves nothing.
Determinism means every step equal, not close; trainer-mode agreement is
~1e-4 at step 1 on `debugmodel` (later steps drift, SKILL.md "Modes"); "no
spikes" means no step-to-step increase larger than
the largest decrease seen so far after warmup. Verdicts come from steps past
warmup only.

## When it fails

Reproduce at the cheapest tier that shows the symptom before changing anything
on the expensive one; run the cheap control first (the native model on the
same chips, the CPU reproduction, the non-MoE flavor). `gotchas.md` has the
known symptom -> cause pairs; `/numerics_debugging` finds the first diverging
op when none of them fits.
