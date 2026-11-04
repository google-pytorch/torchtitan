"""Standalone bench / correctness check for splash_tbwd at the Qwen3-235B per-layer shape.

B=16, S=4096, 64 q heads / 4 kv heads, D=128, bf16; q/k/v/dO = jax.random.normal(split(key(0), 4)) and packed
segment ids from a [B, S] .npy (C4, mean document ~481 tokens). o / lse come from a dense f32 forward (causal +
same-document); lse is converted to the log2 domain the kernel takes.

Kernel ms = summed device time of `splash_tbwd*` ops on TPU:0 per iteration (xprof). "full" also includes the
schedule build, di and the layout transposes. Note: this script feeds row-major [B, S, H, D] arrays, so those
transposes are real copies here; in the model graph q / k / v are sequence-minor and most of them are bitcasts.

Usage (TPU host, with this directory's parent on PYTHONPATH or the module installed):
  python3 splash_tbwd_bench.py OUT_DIR --segment-ids segment_ids.npy [--check] [--trace] [--kind-sweep]
"""
import argparse
import glob
import gzip
import json
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from torchtitan.experiments.tpu.kernels import splash_tbwd as tb

B, S, HQ, HKV, D = 16, 4096, 64, 4, 128
G = HQ // HKV
SCALE = 1.0 / math.sqrt(D)
PEAK_FLOPS = 1155e12
HIGHEST = jax.lax.Precision.HIGHEST


def _mask(seg_S):
  pos = jnp.arange(seg_S.shape[0])
  return (pos[:, None] >= pos[None, :]) & (seg_S[:, None] == seg_S[None, :])


@jax.jit
def dense_fwd(q_SHD, k_SND, v_SND, seg_S):
  """One example: o [S, H, D] bf16 and natural-log lse [H, S] f32."""
  mask = _mask(seg_S)

  def group(h):
    q = jax.lax.dynamic_slice_in_dim(q_SHD, h * G, G, 1).astype(jnp.float32)
    s = jnp.einsum("sgd,td->gst", q * SCALE, k_SND[:, h].astype(jnp.float32), precision=HIGHEST)
    s = jnp.where(mask[None], s, -jnp.inf)
    lse = jax.nn.logsumexp(s, axis=-1)
    p = jnp.exp(s - lse[..., None])
    return jnp.einsum("gst,td->sgd", p, v_SND[:, h].astype(jnp.float32), precision=HIGHEST), lse

  o, lse = jax.lax.map(group, jnp.arange(HKV))  # [N, S, G, D], [N, G, S]
  return jnp.transpose(o, (1, 0, 2, 3)).reshape(S, HQ, D).astype(jnp.bfloat16), lse.reshape(HQ, S)


@jax.jit
def dense_bwd(q_SHD, k_SND, v_SND, do_SHD, seg_S):
  """One example: f32 dq [S, H, D], dk / dv [S, N, D]."""
  mask = _mask(seg_S)

  def attn(q, k, v):
    s = jnp.einsum("sgd,td->gst", q * SCALE, k, precision=HIGHEST)
    s = jnp.where(mask[None], s, -jnp.inf)
    return jnp.einsum("gst,td->sgd", jax.nn.softmax(s, axis=-1), v, precision=HIGHEST)

  dqs, dks, dvs = [], [], []
  for h in range(HKV):
    hs = slice(h * G, (h + 1) * G)
    _, vjp = jax.vjp(attn, q_SHD[:, hs].astype(jnp.float32), k_SND[:, h].astype(jnp.float32),
                     v_SND[:, h].astype(jnp.float32))
    dq, dk, dv = vjp(do_SHD[:, hs].astype(jnp.float32))
    dqs.append(dq)
    dks.append(dk)
    dvs.append(dv)
  return jnp.concatenate(dqs, 1), jnp.stack(dks, 1), jnp.stack(dvs, 1)


def rel(x, ref):
  x = np.asarray(x, np.float64)
  ref = np.asarray(ref, np.float64)
  return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


def device_ms(trace_dir, iters, prefix="splash_tbwd"):
  f = glob.glob(f"{trace_dir}/plugins/profile/*/*.trace.json.gz")
  ev = json.load(gzip.open(f[0]))["traceEvents"]
  pn = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
  tn = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
  tpu0 = {p for p, n in pn.items() if n == "/device:TPU:0"}
  ops = [e for e in ev if e.get("ph") == "X" and e.get("pid") in tpu0 and tn.get((e["pid"], e["tid"])) == "XLA Ops"]
  kern = sum(e["dur"] for e in ops if e["name"].startswith(prefix)) / 1e3 / iters
  return kern, sum(e["dur"] for e in ops) / 1e3 / iters


def timed(fn, iters):
  jax.block_until_ready(fn())
  t0 = time.perf_counter()
  for _ in range(iters):
    out = fn()
  jax.block_until_ready(out)
  return (time.perf_counter() - t0) / iters * 1e3


def main():
  p = argparse.ArgumentParser()
  p.add_argument("out")
  p.add_argument("--segment-ids", required=True)
  p.add_argument("--iters", type=int, default=5)
  p.add_argument("--check", action="store_true", help="dense f32 reference on examples 0..1")
  p.add_argument("--trace", action="store_true")
  p.add_argument("--kind-sweep", action="store_true", help="time with every active tile forced to one kind")
  p.add_argument("--fwd-cache", default="/tmp/splash_tbwd_fwd_cache.npz")
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  kq, kk, kv, kg = jax.random.split(jax.random.key(0), 4)
  q = jax.random.normal(kq, (B, S, HQ, D), jnp.bfloat16)
  k = jax.random.normal(kk, (B, S, HKV, D), jnp.bfloat16)
  v = jax.random.normal(kv, (B, S, HKV, D), jnp.bfloat16)
  do = jax.random.normal(kg, (B, S, HQ, D), jnp.bfloat16)
  seg = jnp.asarray(np.load(a.segment_ids).astype(np.int32))
  if os.path.exists(a.fwd_cache):
    z = np.load(a.fwd_cache)
    o, lse = jnp.asarray(z["o"]).astype(jnp.bfloat16), jnp.asarray(z["lse"])
  else:
    outs = [dense_fwd(q[b], k[b], v[b], seg[b]) for b in range(B)]
    o, lse = jnp.stack([x[0] for x in outs]), jnp.stack([x[1] for x in outs])
    np.savez(a.fwd_cache, o=np.asarray(o.astype(jnp.float32)), lse=np.asarray(lse))
  lse2 = lse * math.log2(math.e)

  full = jax.jit(lambda *xs: tb.transposed_attention_bwd(*xs, scale=SCALE, log2e=math.log2(math.e)))
  args = (q, k, v, o, lse2, do, seg)
  t0 = time.perf_counter()
  dq, dk, dv = jax.block_until_ready(full(*args))
  res = dict(compile_s=time.perf_counter() - t0)

  sched = jax.jit(tb.build_schedule)(seg)
  kinds = np.asarray(sched["kind"])
  res["kinds"] = {int(x): int((kinds == x).sum()) for x in np.unique(kinds)}
  di = jnp.einsum("bshd,bshd->bhs", o.astype(jnp.float32), do.astype(jnp.float32))
  ins = (jnp.transpose(q, (0, 2, 3, 1)), jnp.transpose(do, (0, 2, 3, 1)), jnp.transpose(k, (0, 2, 3, 1)),
         jnp.swapaxes(k, 1, 2), jnp.transpose(v, (0, 2, 3, 1)), lse2, di,
         jnp.broadcast_to(seg[:, :, None], (B, S, tb.NUM_LANES)), jnp.broadcast_to(seg[:, None, :], (B, 8, S)))
  ins = jax.block_until_ready(ins)
  kern = jax.jit(lambda sc, *xs: tb.splash_tbwd_call(sc, *xs, scale=SCALE, log2e=math.log2(math.e)))

  if a.check:
    errs = {"dq": [], "dk": [], "dv": []}
    for b in range(2):
      refs = dense_bwd(q[b], k[b], v[b], do[b], seg[b])
      for name, x, r in zip(("dq", "dk", "dv"), (dq, dk, dv), refs):
        errs[name].append((np.asarray(x[b], np.float32), np.asarray(r)))
    res["rel_err_vs_dense_f32"] = {n: rel(np.stack([x for x, _ in e]), np.stack([r for _, r in e]))
                                   for n, e in errs.items()}
    print(json.dumps(res["rel_err_vs_dense_f32"]), flush=True)

  res["kernel_wall_ms"] = timed(lambda: kern(sched, *ins), a.iters)
  res["full_wall_ms"] = timed(lambda: full(*args), a.iters)
  if a.kind_sweep:
    sweep = {}
    for kind in sorted(res["kinds"]):
      forced = dict(sched, kind=jnp.where(sched["kind"] == 0, 0, kind).astype(jnp.int32))
      sweep[kind] = timed(lambda forced=forced: kern(forced, *ins), a.iters)
    res["kind_sweep_ms"] = sweep
  if a.trace:
    for tag, fn in (("kernel", lambda: kern(sched, *ins)), ("full", lambda: full(*args))):
      td = os.path.join(a.out, f"trace_{tag}")
      jax.profiler.start_trace(td)
      for _ in range(3):
        jax.block_until_ready(fn())
      jax.profiler.stop_trace()
      res[f"{tag}_kernel_ms"], res[f"{tag}_busy_ms"] = device_ms(td, 3)
    # Executed FLOPs of the 512-tile baseline (Tokamax): 5 matmuls per active 512^2 tile and q head.
    active = sum(v for kd, v in res["kinds"].items() if kd != 0) // 1
    res["pct_peak_vs_512_tiles"] = 100 * 5 * 2 * tb.BLOCK ** 2 * D * HQ * active / (res["kernel_kernel_ms"] / 1e3) \
        / PEAK_FLOPS
  print(json.dumps({k_: v_ for k_, v_ in res.items() if k_ != "rel_err_vs_dense_f32"}), flush=True)
  json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)


if __name__ == "__main__":
  main()
