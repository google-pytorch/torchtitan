# Gotchas

Each cost a previous onboarding a day or more. Symptom first, then the check.

## Trains fine, wrong model

- **Packed batch, no segment ids.** Loss normal, throughput half of expected:
  the attention wrapper did not pass `positions` to the inner attention, so
  splash saw one causal problem per row. Check the custom_call operands in HLO.
  The SDPA side has the same hole: the stock `ScaledDotProductInnerAttention`
  has no `positions` argument at all, so a rung-1 model built on it attends
  across documents and the Splash parity has nothing to compare against
  (SKILL.md Step 1 names the document-masked one).
- **Sliding window off by one under splash** (window W is splash W - 1), or
  not applied on the SDPA path. Test with the reference's definition.
- **Config override builds the wrong shape.** `--config=<x> --hf-model=<y>` can
  build a hardcoded architecture for any checkpoint and every log line looks
  right. Assert per-layer replacement counts and heads / kv_heads / vocab.
- **A config patcher that matched nothing** reads as "the knob does not help".
  Assert the match count.
- **Unit-initialized scales hide a dropped multiply.** Perturb them in the
  parity test.
- **An isolated kernel test is not bit-identity.** Whole-program layout changes
  reduction order; label it rounding-order.
- **Empty expert groups** on libtpu < 0.0.48 left `torch._grouped_mm`'s weight
  gradient uninitialized. One clean run proves nothing for this class.

## Loud, with a misleading message

- `init_device_mesh` empty process-group name right after `import torch_tpu`
  worked: the wheel's torch glue does not match the image's torch nightly.
- Dynamo marks a dim symbolic on the second layer type: heterogeneous layers
  share a block class; compile per block with `dynamic=False`.
- `recompile_limit` at layer 9+: per-layer SimpleFSDP classes count as
  distinct; raise the limit, test a deeper flavor.
- `UncapturedHigherOrderOpError` at scan trace: the layer body writes a module
  attribute or captured buffer (MoE counters, routing metrics).
- Config ValueError that reads like a scan error:
  `num_tokens_per_train_step != ranks x num_tokens_per_microbatch_per_dp_rank`.
- Shardy segfault in `buildTensorFactorShardings` on GraphTrainer with libtpu
  0.0.48+: `xla_enable_hlo_sharding_v3=False`.
- VMEM OOM naming a kernel that is not the one under test: the scoped VMEM
  default is 16 MB on v5e, 64 MB on v6e / v7x; read both numbers first.
- `torch._grouped_mm` on bf16 rejects HIGHEST matmul precision (libtpu
  0.0.48): scope the parity test's "highest" to the dense path (`tests.md`).
- CPU FlexAttention has no backward: an upstream reference built on it needs
  its inner attention swapped for the math SDPA inside the test.

## Tests that pass for the wrong reason

- A plain `TestCase` with `skipUnless(<detect tpu>)` lands in the CPU shard
  and skips forever. Take the device from `BaseAcceleratorDeviceTest`.
- `Module.to(device)` unties tied weights; re-tie after the move and assert `is`.
- Device-test modules read `--device` with absl flags; under bare pytest
  `setUp` raises. Run them as CI does: `python -m <module> --device=...`.

## Environment

- `TORCHTITAN_DEVICE_TYPE=tpu`; `LD_LIBRARY_PATH=<env>/lib`;
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for CPU tests. Leave `JAX_PLATFORMS` unset
  on TPU runs, as CI does: torch_tpu keeps the chips, and under
  `JAX_PLATFORMS=cpu` Mosaic kernels that read the core count (tokamax fused
  CE) fail with `AttributeError` in `_get_default_num_cores`. `cpu` is for
  CPU runs only.
- `rm -f /tmp/libtpu_lockfile` after a libtpu crash or the chip stays wedged.
- xpk: explicit `--script-dir`, single-line `--command`, workload name under 40
  characters, one `--env` per flag, nodepool pinned for multi-host.
- "Got further than before" is not working; wait 6+ steps past warmup.
