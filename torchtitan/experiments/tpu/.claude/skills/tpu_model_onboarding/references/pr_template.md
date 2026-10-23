# The PR description is the onboarding record

It is the only state that survives between sessions: whoever resumes (a later
agent session, a reviewer, you next week) reads the ladder table first. Update
it on every push, not at the end. The merged Muse Glimmer commits (8be9902531,
6fdce58073, a11f9a487d, 6bfbe321da) are the standard: what changed and why,
which tests prove it, the measured bounds next to the asserted ones, what is
deliberately left for later.

## Ladder table (from Step 1 on; execution target and dispatch budget above it)

```
Rung | Feature / flag                  | Commit | Parity test (bounds, measured) | Numerics class | Run (target, steps, result)   | Status
-----|---------------------------------|--------|--------------------------------|----------------|-------------------------------|---------------------
1    | SDPA + per-block compile        | <sha>  | Test<Model>ReferenceParity ... | oracle         | A, v6e-4, 10 steps, det OK    | merged
2    | GraphTrainer (use_graph_trainer)| <sha>  | 3-step loss vs compile ...     | bit-identical  | A, v6e-4, 10 steps            | done
3    | splash (use_splash_attention)   | <sha>  | vs SDPA packed ...             | rounding-order | A, v6e-4                      | in progress: <open item>
```

## PR 1

```
<Model> on the TorchTitan TPU lane (<wraps torchtitan/models/<name> | ported from
<reference> under experiments/tpu/examples/<name>>). Execution target: <A | B <tpu, zone> | C <cluster>>.

Architecture (source: <file:line per constant>): layers <N>, pattern <...>;
attention <GQA heads/kv, head_dim, QK-norm, window, softcap>; norms <...>;
RoPE <theta, partial factor, real/complex>; FFN/MoE <act, experts, top-k, router,
balancer>; embeddings <tied, scaled>; forward side effects <none | X behind flags>.

Code: <file: one line each>; deviations from models/common and why; registration.

Tests: Test<Model>ReferenceParity (<in-process HF | golden dump from <ref, commit>>,
tiny config <dims>, <variants>; logits / loss / all grads; bounds <...>, measured <...>);
state-dict round trip; flavor checks; <mask test>; test_train_distributed <cases>.

Runs (<VM type>, torch_tpu <sha>, libtpu <ver>): <name>_debugmodel 10 steps, loss
<l0> -> <l10>; determinism bit-identical; step-1 oracle loss <x> grad_norm <y>;
eager / compile / (GraphTrainer) agree to <...>.

Not in this PR: GraphTrainer wrapper, splash/tokamax, fused CE, <MoE kernels>, <size> recipe.
```

## Rung commit (Step 2); the same text, condensed, is the table row

```
<Feature> for <Model> on TPU.

What: <what it replaces, how it is enabled, default off>. Why: <profile evidence>.
Numerics class: <bit-identical | rounding-order | semantic (why a separate recipe)>;
step-1 oracle <same | delta>.
Tests: <parity test, inputs, bounds, measured>; control: <what was disabled, how much
the metric moved>; test_train_distributed <case>.
Measurement (<recipe>, <chips>, >= 10 steps past warmup): before <...>, after <...>.
HLO: <custom_call present N times; segment ids in operands>.
```
