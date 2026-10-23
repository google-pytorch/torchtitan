# Tests: what to write, what not to, and the one that matters

`tpu/` means `torchtitan/experiments/tpu/`.

## Mandatory in PR 1

```
Test                                   | File                                         | Template
---------------------------------------|----------------------------------------------|------------------------------------------------
Reference parity (CPU + single chip)   | tpu/<path>/tests/test_<name>.py              | Gemma 4 port TestGemma4HFReferenceParity
State-dict round trip                  | same file                                    | Gemma 4 port TestGemma4StateDictAdapter
Flavor / config checks                 | same file                                    | Gemma 4 port TestGemma4Flavors; muse test_muse_glimmer_config
Mask semantics, if the arch has one    | same file                                    | Gemma 4 port TestGemma4SdpaSlidingWindow
End-to-end training cases              | tpu/tests/test_train_distributed.py          | muse_glimmer_fsdp / muse_glimmer_simplefsdp
                                       | tpu/tests/test_train_minimal_distributed.py  | qwen3_fsdp / qwen3_simplefsdp
```

The Gemma 4, OLMo 2 and GPT-OSS ports cited in this file are unmerged; see
`SKILL.md` for their branches and head commits.

Do not add per-component CPU-vs-device ladders (embedding, attention, norm,
FFN one by one), `test_dtensor_parallelize_distributed.py` or
`test_simple_fsdp_sharding.py` clones. They were written when DTensor itself
was unproven and are no longer required (Gemma 4 port review). A component
test is justified only when it guards something the parity test cannot see.

## The reference-parity test

Contract, for any reference: identical weights and inputs through the
reference and the TorchTitan model, one forward and one backward, logits,
loss and every parameter gradient compared with stated tolerances, and the
weight mapping checked both ways (no missing, no unexpected keys). One class
on `BaseAcceleratorDeviceTest` (`tpu/base_device_test.py`): CI runs it on one
chip; it also runs on CPU with `--device=cpu`, but only when you invoke it
that way (CI has no CPU copy of single-TPU modules). What changes with the
reference is where its half runs.

In-process (HF transformers, upstream torchtitan), the shape from the Gemma 4
port:

1. Reference built from a tiny config in fp32, `.eval()`: hidden 128, 2 layers
   covering every layer type (sliding + global; dense + MoE), 8 heads, head
   dim 16-32, vocab 2048, seq 16, batch 2. `self.skipTest` if the installed
   `transformers` lacks the class.
2. Perturb unit-initialized parameters (norm weights, learned scales,
   `layer_scalar`-style multipliers) with `1.0 + 0.1 * randn`; at their
   defaults a dropped multiply is invisible.
3. Geometry that exercises the architecture: window shorter than the
   sequence, more experts than top-k, tied embeddings if the real model ties.
4. Weights through your adapter (`adapter.from_hf(hf_model.state_dict())`),
   `load_state_dict(..., strict=False)`, assert `missing` and `unexpected`
   are both empty. This tests adapter and model together.
5. One forward and backward on the same tokens, model on
   `self.accelerator_device`, reference on CPU, in the trainer's `[T]` token
   layout (`set_unflatten_seq_len`, explicit `positions`).
6. Compare logits, loss and every parameter gradient; map reference gradients
   through the adapter too, drop the duplicate tied-head gradient, assert
   equal key sets before comparing values. Logits and loss go through
   `test_utils.check_equivalence` (elementwise `allclose`). Gradients do not:
   they span orders of magnitude, and an absolute tolerance that fits the
   large ones accepts a zeroed or sign-flipped small one. Bound each gradient
   tensor by relative L2 error, `||g - g_ref|| / ||g_ref||`, with a floor only
   for tensors whose reference norm is itself ~0.
7. The negative control, in the same test: feed the comparison the reference
   gradients zeroed, and again sign-flipped, and assert it fails both times.
   A bound that cannot reject those is not a bound.
8. Parameterize over flavors and converters (dense; MoE; MoE with
   `TpuGroupedExpertsConverter`, Pallas off so it runs on CPU).

Golden-tensor shape (Megatron/NeMo, MaxText, custom code; the reference cannot
run in the test process): a `tests/dump_<name>_reference.py` next to the test
is the only thing that runs on the reference side. It builds the same tiny
config, perturbs with a fixed seed, runs one forward and backward on fixed
tokens, and writes one file: config values, seed, tokens and positions, the
reference state dict in its own key names and layouts, logits, loss, every
gradient keyed by reference name, and provenance (repo, commit, framework
version, device, dtype policy). Sharded sources are gathered to one unsharded
state dict on the reference side; the adapter handles layout, never sharding.
The test loads the dump and runs steps 4-8; nothing in it imports the
reference framework. Keep the fixture under a few MB so it lives in the repo,
else pin it in GCS with the provenance in a sidecar and skip with a reason when
unreachable. Regenerate when the reference moves and say so in the commit.

Matmul precision decides the bounds. torch_tpu's default fp32 matmul is one
bf16 pass, and at that precision the comparison measures ~1e-2 (OLMo 2:
logits 1.6e-2 max abs, gradient relative L2 1.85e-2 against a 2e-2 bound,
i.e. no margin for a real bug). Run the test under
`torch.set_float32_matmul_precision("highest")` (restore it in `tearDown`),
which turns the same comparison into ~1e-6 and lets the bounds be tight:

```
                    "highest" (dense; OLMo 2)         default precision (Gemma 4, v6e + v7x)
                                                      dense                      MoE
logits              atol 1e-4 rtol 1e-4               atol 5e-2 rtol 1e-2        atol 1e-1 rtol 2e-2
loss                atol 1e-5 rtol 1e-5               atol 5e-3 rtol 1e-3        atol 5e-3 rtol 2e-3
gradients           relative L2 <= 1e-4 per tensor    relative L2 <= 2e-2 per tensor
                    absolute floor 1e-6 only where ||g_ref|| < 1e-4 (all columns)
```

The exception is the bf16 grouped matmul: libtpu 0.0.48 rejects HIGHEST for
the bf16 ragged dot behind `torch._grouped_mm`, so an MoE test scopes default
precision to the expert path (the GPT-OSS port does it in a context manager)
and keeps "highest" for the dense ops; the MoE column above is for a test that
cannot. (The Gemma 4 port used `atol 8e-2 rtol 2e-2` elementwise for gradients;
that bound passes a zeroed gradient whose entries are all below 0.08, so do not
copy it.) Record the measured error next to the bound in the PR.

An upstream torchtitan reference that uses FlexAttention has no CPU backward;
swap its inner attention for the math SDPA inside the test, with the same
mask (as the GPT-OSS port does).

## Feature parity tests (one per rung)

Each rung adds a test against the path it replaces, on the same packed batch,
comparing outputs and gradients, unless noted below:

- splash / tokamax attention: the kernels' numerics, segment ids included,
  are `kernels/tests`' job. The model test checks the patch takes: on a
  meta-device model, every layer's inner attention is replaced (Splash) or has
  `forward` rebound (tokamax); fails with the patch stubbed (OLMo 2
  `TestOlmo2SplashAttentionPatch`). Add a vs-SDPA parity test only when the
  model brings masking the kernel tests cannot see: a sliding window (Muse W
  -> Splash W - 1), a soft-cap, its own attention module or `positions`
  plumbing (commit a11f9a487d: context 4x the window, three documents packed
  in one row with boundaries inside it, both layer types; relative L2 <= 1e-2
  forward, <= 2.5e-2 gradients).
- fused cross entropy vs fp32 `F.cross_entropy` (commit 6bfbe321da):
  bit-identical to the stock kernel when the model-specific option (soft-cap)
  is off.
- MoE kernels vs `torch._grouped_mm`, including empty expert groups.
- scan vs unrolled on a 4-layer proxy, 5 steps, loss and grad_norm within bf16
  noise.
- GraphTrainer vs per-block compile: 3-step loss agreement on `debugmodel` at
  a short context (OLMo 2 `test_olmo2_graph_trainer_matches_per_block_compile`:
  relative 1e-4, measured 0 / 2.9e-5 / 1.4e-5 at fp32; a bf16-parameter MoE
  measured 8.2e-4 over steps 1-3, so set the bound from step 1 and the
  measured steps 2-3, not from the fp32 number). The control is a schedule
  change (warmup 3 instead of 2 moved OLMo 2 by 1.4e-2 / 2.7e-2). Scaling
  `max_norm` or the gradients is not one: AdamW's update is invariant to
  gradient scale, so it stays inside any bound.

The test must be able to fail. Add the control: remove the document mask and
assert the SDPA output moves by more than 10%; turn the cap off and assert
capped and uncapped differ. A kernel that silently ignores the feature then
fails the test instead of passing it.

## End-to-end cases

`tpu/tests/test_train_distributed.py` runs `--module/--config` in a subprocess
for 3 steps on `c4_test`. One case per parallelism the recipe supports (FSDP2
via `--parallelism.no-use_simple_fsdp`, SimpleFSDP, TP if the model allows),
later one per feature flag. Set `max_context_length` so the architecture does
its work (muse: 256 > the 128-token window; 512 with splash so one block spans
the window) and keep `num_tokens_per_microbatch_per_dp_rank` consistent with
it. `skip_devices` for TPU-only kernels. Mirror the FSDP cases in
`test_train_minimal_distributed.py`.

A multi-step CPU-vs-device test, if kept, bounds step 0 tightly and the
post-optimizer steps loosely (after two Adam steps the muse debugmodel gap
ranged 0.004 to 0.37 across seeds, commit 1741e8231d: amplified bf16 error, not
a bug), with each check independent so a failed loss check does not skip the
gradients.
The minimal trainer's CPU-vs-device gradient gap at default matmul precision
is the one-bf16-pass matmul, not the model (OLMo 2 3.7e-3 max abs, within
2.7e-6 relative at "highest"; GPT-OSS 9.5e-3): put the measured number next
to a `grad_atol` you widen, in the commit. `*_distributed.py` modules run on
the multi-chip shard only; their `--device=cpu` failures are not a gate
(`test_train_minimal_distributed` fails at step 0 for every model on CPU in
the current environment, qwen3 included).

## CI (`.ci/discover_tpu_tests.py`)

- A module is classified by the base classes of its TestCases (an AST match on
  the class name): `BaseDistributedDeviceTest` -> multi-chip shard, and the
  file name must end in `_distributed.py`; `BaseAcceleratorDeviceTest` ->
  single-TPU shard; anything else -> the CPU shard, which runs on a host with
  no chip. A plain `TestCase` with `skipUnless(<detect tpu>)` lands there and
  skips on every run, green. Take the device from the base class
  (`self.accelerator_device`, resolved from `--device`) and `self.skipTest` with
  a reason when it cannot run the case; never detect it yourself.
- Modules with an `absltest.main()` block run as `python -m <module>`
  (`--device=tpu` on the TPU shards, 900 s timeout); files without one run
  under pytest, CPU only. Under bare pytest the absl flags are unparsed and a
  device test's `setUp` raises: run them the way CI does.
- CPU runs set `TORCH_DEVICE_BACKEND_AUTOLOAD=0`; a test that imports
  `torch_tpu` unconditionally breaks the CPU shard. `torch.tpu.device_count()`
  raises from user code; use `torch.accelerator`.
- `@parameterized.named_parameters`; `testcase_name` is what CI reports.
- There is no lint job and no pre-commit config; "CI green" means the shards
  in `.github/workflows/tpu-test.yml`. Run the touched modules the same way
  before pushing.

## Outside CI

- `scripts/checkpoint_conversion/numerical_tests_*.py` compares logits by KL
  against an HF model on real weights; run it on the full checkpoint before
  Step 3 and quote the KL in the PR. For a non-HF reference the equivalent is
  the golden dump taken on real weights for a handful of sequences.
- `/numerics_debugging` when a parity test fails and the first diverging op
  is not obvious.
