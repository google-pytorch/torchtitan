---
name: tpu_model_onboarding
description: Onboard a model to TorchTitan's TPU lane (torchtitan/experiments/tpu): place the code, prove parity with its reference, make it train, then climb the optimization ladder one flag at a time and ship a target-size recipe. Use when a model must train on TPU, whether it exists upstream in torchtitan/models or is ported from HF transformers, Megatron, MaxText or custom code. Not for debugging an existing model's numerics (/numerics_debugging) or for GPU-only work.
---

# Model onboarding (TPU lane)

Onboarding is a ladder. Rung 1 makes the model correct on the plain SDPA path
and trains it end to end; it is a PR of its own, because everything later is
measured against it. Each later rung is one feature behind one flag, with one
parity test and one end-to-end case, as one commit. Muse Glimmer is the worked
example: commit 8be9902531 (rung 1), then 6fdce58073, a11f9a487d, 6bfbe321da
and 1741e8231d.

Where changes go: under `torchtitan/experiments/tpu/`, for the model, its
recipes, parallelize code, converters, overrides and tests alike. A change
outside it (`torchtitan/models/`, `torchtitan/components/`, `torchtitan/
distributed/`, `torchtitan/experiments/graph_trainer/`) is the exception, not
the default: it needs a one-line reason in the PR saying why the TPU lane's
extension points (converters, `override.imports`, `workarounds.py`, a TPU
subclass via `derive`) could not carry it, it is kept to the minimum that
unblocks the lane, and it is a candidate for its own PR so the core reviewers
see it on its own. `.claude/rules/experiments.md` is the repo-wide version of
this rule.

Read before starting: `torchtitan/experiments/tpu/README.md` (environment,
VM and cluster setup, launch commands), `torch-tpu-optimization-guide.md`
(the levers, in order), `.claude/rules/models.md`, `.claude/rules/experiments.md`.

Read the existing TPU models first and reuse what they do; a second
implementation of something the lane has needs a reason in the PR.
`tpu/llama3/` (dense; its parallelize is what a dense port reuses),
`tpu/qwen3/` (MoE, balancers, GraphTrainer, scan), `tpu/muse_glimmer/`
(wrapped upstream model, commit 8be9902531 and its rung commits),
`tpu/examples/mamba*/` (ports); shared: `kernels/`, `workarounds.py`,
`overrides/`, `graph_trainer/`, `tests/test_train_distributed.py`, the recipe
READMEs (`recipes/ironwood/qwen3-30b-a3b/tpu7x-4x4x4` for Step 3). Unmerged,
on branches: the Gemma 4 port (`gemma4-tpu` at 66d411eed5, parity-test
template), the OLMo 2 port (`alekseyv_olmo2_onboarding` at 7a19203e1f), the
GPT-OSS wrap (`codex/gpt-oss-tpu-onboarding` at e63d8b6d42).

References: [model.md](references/model.md) (port layout, the weight
mapping and how it is checked, registration), [tests.md](references/tests.md)
(which tests to write, the parity-test contract, CI rules),
[run.md](references/run.md) (execution targets, what closes on which tier,
reading a run),
[pr_template.md](references/pr_template.md) (the onboarding record),
[gotchas.md](references/gotchas.md).

## Step 0: three questions before any code

1. **Where does it run?** (`run.md` has the full table.)
   - A. this machine is the TPU VM: run directly; check the chips with
     `torch.accelerator.device_count()` (not `torch.tpu.*`, which raises) first.
   - B. a TPU VM over ssh: `gcloud compute tpus tpu-vm ssh <name> --zone=<z>
     --command='...'`; you need name, zone, project and the checkout path, and
     you sync code and fetch `--dump_folder` yourself.
   - C. GKE via Cluster Toolkit (`gcluster`): submit per the recipe README;
     you need project, cluster, image registry, whether you may build images,
     and a dispatch budget agreed before the first job.

   Rung 1 and the kernel rungs close on A or B with 1-4 chips; only Step 3
   needs C. Under C, every check that can run on CPU or one chip runs there
   first, and each dispatch passes the gate in Step 3. Never fall back
   silently: no chip, failed ssh or missing image means stop and say so.
2. **What is the reference, and how do the weights arrive?** HF transformers
   (safetensors, reference runs in-process on CPU) is the common case; upstream
   torchtitan GPU checkpoints run in-process too. Megatron/NeMo, MaxText and
   custom code usually cannot run in the test process: a script next to the
   test runs the reference on a pinned tiny config and dumps weights, tokens,
   logits, loss and every gradient with provenance, and the test consumes the
   dump. Either way the weight mapping is a key-and-layout translation under
   `BaseStateDictAdapter` (HF-named; the names are legacy) and the test checks
   keys both ways. If an HF checkpoint already trains through
   `experiments/tpu/transformers_modeling_backend`, the onboarding may be a
   recipe, not a port.
3. **Wrap or port?** Upstream has the model: add only `experiments/tpu/<name>/`
   (registry wrapper, `config_registry.py`, `infra/parallelize.py`, `tests/`).
   Otherwise: `experiments/tpu/examples/<name>/`, built on
   `torchtitan/models/common/` (subclass, do not copy; a copied block needs a
   one-line reason in the PR). `examples/` is a review requirement.

Then read the architecture off the reference and write it into the PR
description (`references/pr_template.md`): layer pattern, norms, RoPE, QK-norm,
softcaps, tying, MoE shapes, and anything the layer forward mutates (it blocks
scan). Every constant gets a source line.

## Step 1 (PR 1): correct on SDPA, trains end to end

- Model code, plain SDPA, no Pallas kernels, no GraphTrainer. `debugmodel`
  flavor plus the real sizes; a `<name>_debugmodel` recipe. Register in
  `experiments/tpu/train.py` and `train_minimal.py` (`model.md`). "Plain
  SDPA" means the document-masked one: the stock
  `ScaledDotProductInnerAttention` takes no `positions`, so packed rows attend
  across documents and rung 3 (Splash with segment ids) has nothing to compare
  against. Use `SlidingWindowScaledDotProductInnerAttention(window_size=None)`
  from `tpu/muse_glimmer/attention.py` (the OLMo 2 port).
- For a port, the weight mapping: a `StateDictAdapter` subclass with
  `from_hf` / `to_hf` (`model.md`), which is what `--checkpoint.initial_load_in_hf`
  and the parity test both go through. Write the key table into the PR before
  the code; the parity test proves it (no missing or unexpected keys, then the
  numbers), a round-trip test keeps it honest, and the KL check on the real
  checkpoint closes it before Step 3.
- Decide the large-size path now and write the layers for it. Homogeneous
  layers: keep the stock `Decoder.forward` loop and make
  `check_layers_are_stackable` pass, so scan (rung 7) stays open.
  Heterogeneous layers (sliding / global with different head dims): compile
  per block with `dynamic=False`, and the large-size path is graph split, not
  scan; do not bend the architecture to stack. Either way, no module-attribute
  or in-place writes in the layer forward (MoE counters and routing metrics
  behind config) and no complex ops (real-arithmetic RoPE); both also matter
  for compile. Upstream's expert-bias balancer (`load_balance_coeff`) is such
  a write: it accumulates `tokens_per_expert_E` in place. The lane's qwen3
  235B recipes switch to the batch-wise aux loss instead
  (`_use_global_batch_aux_loss`, `qwen3/config_registry.py`; routing metrics
  come back as graph outputs via `collect_moe_routing_metric_tensors`) and
  send the dead counter to a throwaway tensor
  (`overrides/discarded_token_counter.py`). Keeping the bias balancer means
  the in-place update must survive the compiled graph: assert in a test that
  `expert_bias_E` moves after an optimizer step on the compiled path.
  Switching balancer is the semantic class: a separate labelled recipe,
  measured against a reference that uses the same one.
- The tests in `tests.md`: the reference-parity test (the one that matters:
  tiny config covering every layer type, perturbed scales, weights through the
  adapter with keys checked both ways, logits + loss + every gradient
  compared, on `BaseAcceleratorDeviceTest`), state-dict round trip, flavor
  checks, a mask-semantics test if the architecture has one, and the
  end-to-end cases in `experiments/tpu/tests/test_train_distributed.py`. Do
  not pad the PR with per-component ladders or sharding tests.
- A run on the Step 0 target that meets the bar below.

## The bar for "it trains" (every rung; recorded in the PR with its command)

```
Steps          >= 10; judge from steps past warmup. 2-4 steps is not a verdict.
Loss           decreasing, no NaN/inf, no spikes
Determinism    two runs with --debug.seed=42 --debug.deterministic
               --metrics.enable_tensorboard --metrics.log_freq=1 give equal loss
               and grad_norm at every step, read from the TensorBoard event
               file (tag loss_metrics/global_avg_loss; TensorBoard is off and
               log_freq is 10 by default); both series must contain steps 1..N.
               stdout prints 5 digits and is not evidence
Oracle         record the deterministic per-step loss and grad_norm; step 1 is
               the smoke check every later rung compares against first
Structure      per-layer patch counts == layer count; heads / kv_heads / vocab ==
               the checkpoint; the HLO dump (--extra-xla-flags=xla_dump_to=
               <dir>) contains the custom_calls of enabled kernels.
               Eager log lines do not fire under compile; trust the HLO.
Modes          eager / per-block compile / GraphTrainer: step-1 loss within
               ~1e-4 on debugmodel. Step 1 has no optimizer state, so a larger
               gap there is a bug. Later steps are not gated by this bar: AdamW
               amplifies round-off, and the gap grows with the step count
               (OLMo 2 eager vs compile 1.6e-4 through step 10, 6.1e-3 by step
               12; GPT-OSS 2.5e-3 by step 10; the merged Muse debugmodel
               1.45e-2 under the same flags). Whether a trajectory is
               acceptable is the numerics class below, decided on post-warmup
               medians, never a per-step 1e-4 bound over N steps
```

Every change after PR 1 gets a numerics class: **bit-identical** (loss and
grad_norm equal to the oracle at every logged step of a >= 10-step run; two
scalars at step 1 do not pin the gradient tensors, a differing gradient shows
in step 2 after the optimizer applies it; for a frontier claim also compare
the final state dict); **rounding-order** (fp32 round-off from tiles, fusion or
reduction order; fine for speed, but a frontier recipe needs the 200-step
reference trajectory, compared on post-warmup medians); **semantic** (bf16
gradient reduce, dropped tokens, another balancer: a separate labelled recipe,
not a lever).

## Step 2: the ladder, one rung per commit

Order, with the guide section in parentheses:

```
1  per-block compile + SimpleFSDP + mp policy (4, 5)   PR 1
2  GraphTrainer wrapper (8)                            3-step loss vs per-block compile
3  splash / tokamax attention (4)                      patch replaces every layer; vs SDPA
                                                       only for model-specific masking
4  fused cross entropy (4)                             vs fp32 F.cross_entropy
5  MoE kernels, if MoE (10)                            grouped mm vs torch._grouped_mm,
                                                       empty groups included
6  batch / context to saturation (3)                   recipe change; re-measure
7  scan + offload, or graph split (9)                  scan vs unrolled 4-layer proxy,
                                                       5 steps; only once it no longer fits
8  libtpu flag bundle (9)                              bit-identical step-1 oracle
9  tile sweeps (4, 10)                                 rounding-order; per generation
```

Each rung: the flag stays off in the recipe until Step 3; a parity test
against the path it replaces plus the control that proves the test can fail
(drop the document mask, turn the cap off, watch the metric move), or, where
`kernels/tests` already cover the kernel's numerics and the model adds nothing
they cannot see, a test that the kernel is patched in (`tests.md`); one
end-to-end case with the flag on; a commit message with the numerics class,
before/after step time and MFU on the same recipe and chip count, and the HLO
evidence (custom_call present, segment ids among its operands). Measure on
the smallest real-size flavor that fits the target (the 1B on 4 chips), not
on `debugmodel`: `debugmodel` is for correctness, and at its shape levers
tie that move the real size (OLMo 2 port: GraphTrainer tie vs +10.6%, fused
CE tie vs +3.2% / -2.8%). Rung k passes
on the target before rung k+1 starts, and the ladder table in the PR is
updated after every rung. One branch; whether it ships as one PR or stacked
PRs is the human's call at the end.

The ladder is done when the profile shows no exposed gap that a listed lever
addresses, every enabled feature has its numerics class, the trajectory check
passed and the README table is filled. "Out of ideas" is not a stop condition;
an unmeasured lever on the list is not either.

## Step 3: target size

Production recipe plus `recipes/<generation>/<model>/<topology>/README.md`:
launch block, every libtpu flag with a one-line reason, `## Benchmark Results`
from `--tpu_config.enable_mfu_benchmark` over >= 10 post-warmup steps. Only
then flip the kernels on by default. The 200-step reference run is an input
you ask for (pinned optimizer, batch, reduce dtype, seed, data, balancer), not
something the TPU side produces.

Gate before any cluster dispatch, every time:

- the exact config parses (`ConfigManager().parse_args([...])` with the
  launch's `--module/--config` and overrides);
- a reduced proxy with the same flags runs 3 steps: the tiny flavor on CPU if
  the recipe has one, else the smallest mesh that fits; say in the PR which
  proxy it was. The exact config of a large model cannot run on one chip, so
  this is a proxy by design, not a substitute for the multi-rank smoke that
  the first dispatch is;
- the image carries the commit you think it does;
- the optimizer path is exactly one of eager / `compile_optimizer` /
  `optimizer_in_graph`;
- the token budget matches the trainer's formula: `num_tokens_per_train_step`
  is `-1` (one accumulation step) or a multiple of
  `num_tokens_per_microbatch_per_dp_rank x num_pp_microbatches x dp_degree`,
  where `dp_degree` is the batch mesh size, not the rank count (TP, CP and PP
  ranks do not multiply it). More than one accumulation step is fine unless an
  enabled feature rejects it (`optimizer_in_graph`, batch-wise aux loss);
- command-line flags are in the recipe (left on the command line they get
  forgotten).

When the budget is spent, write the ladder table and stop.

## Rules

- Run touched test modules the way CI does before pushing: `python -m <module>
  --device=tpu` on a chip, `--device=cpu` for the CPU shard. CI classifies a
  module by its base class (`tests.md`); take the device from the base class,
  never detect it yourself.
- Changes live under `experiments/tpu/`; anything outside it is an exception
  with its reason in the PR (see the top of this file). No Meta copyright
  headers on new files.
- Reuse, do not copy, and do not import across model packages: when the code
  you need lives under another model's directory (an attention class, a
  recipe helper, a converter), importing it drags that model in. The PR that
  makes it shared by a second model moves it to a `tpu/` module both import.
- No performance claim from fewer than 10 steps, from a single run, or before
  the numerics class is known.
- The PR description is the onboarding record and the only state that
  survives between sessions; keep it current on every push. No handover files
  in the repo.
- `references/gotchas.md` before debugging anything that "trains fine, but";
  `/numerics_debugging` for a first-diverging-op search.
