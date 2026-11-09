"""AOT cross-compile check of the DeepSeek-V3 TPU lane's Pallas kernels.

Compiles each kernel x shape the 671B recipes run (expert grouped matmuls
forward / dInput / dWeight for W1/W3, W2 and the transposed-RHS W2, the paired
W1/W3 kernel, Splash and Tokamax Splash forward / backward at the MLA shapes,
the fused linear + cross-entropy loss, the route-score gradient) for a TPU
target topology on a CPU-only host, with no TPU devices, and prints OK / FAIL
per case with the scoped-VMEM numbers from the compiler's error text. A
CompileTimeScopedVmemOom shows up here exactly as on the slice, so tiles can be
checked for a generation (e.g. TPU7x) without its hardware.

Run it inside the training image, so jax / jaxlib / libtpu match the run, from
the repository root:

  docker run --rm --network none -e JAX_PLATFORMS=cpu \
    -e TPU_WORKER_HOSTNAMES=localhost -e TPU_SKIP_MDS_QUERY=1 \
    -v "$PWD":/src:ro -w /src <image> python \
    torchtitan/experiments/tpu/deepseek_v3/tests/aot_kernel_vmem_check.py \
    --target v7x --recipe deepseek_v3_671b_31l_4x4x4

(``TPU_SKIP_MDS_QUERY`` avoids libtpu's metadata-server lookups, which hang
without a network.) ``--case`` filters by regex, ``--probe-need`` and
``--margin-limits`` re-compile passing cases at smaller scoped-VMEM limits.
The lane's kernel modules are imported with torch_tpu's ``jax_op`` /
``custom_jax_kernel`` replaced by pass-through shims, so the JAX functions the
lane registers are what gets compiled; Pallas trace-time queries
(``pltpu.get_tpu_info``) see the target chip through ``jax.set_mesh``.
"""

import argparse
import functools
import json
import os
import pathlib
import re
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.topologies import get_topology_desc

# This checkout (repository root), unless TT_SRC names another.
sys.path.insert(
    0, os.environ.get("TT_SRC") or str(pathlib.Path(__file__).resolve().parents[5])
)

TARGETS = {
    # MaxText accelerator_to_spec_map: v6e-4 / tpu7x-8 (one host, 2x2 chips).
    "v6e": dict(topology_name="v6e:2x2", chips_per_host_bounds=(2, 2, 1)),
    "v7x": dict(topology_name="tpu7x:2x2x1", chips_per_host_bounds=(2, 2, 1)),
}


# ----------------------------------------------------------------------------
# torch_tpu shims: keep the raw JAX function the lane decorates.
# ----------------------------------------------------------------------------
class _RawOp:

  def __init__(self, fn):
    self.fn = fn
    functools.update_wrapper(self, fn)

  def __call__(self, *a, **k):
    return self.fn(*a, **k)

  def register_autograd(self, *a, **k):
    pass

  def register_fake(self, *a, **k):
    pass


def _install_shims():
  from torch_tpu._internal import pallas  # pylint: disable=g-import-not-at-top

  def jax_op(name, fn=None, /, **kw):
    del name, kw
    if fn is None:
      return _RawOp
    return _RawOp(fn)

  def custom_jax_kernel(fn, name=None, **kw):
    del name, kw
    return fn

  pallas.jax_op = jax_op
  pallas.custom_jax_kernel = custom_jax_kernel


# ----------------------------------------------------------------------------
# Error parsing
# ----------------------------------------------------------------------------
_SCOPED_RE = re.compile(
    r"Scoped allocation with size ([\d.]+[KMG]?) and limit ([\d.]+[KMG]?)"
    r" exceeded scoped vmem limit by ([\d.]+[KMG]?)"
)
_OP_RE = re.compile(r"allocating on stack for (%[\w.\-]+) = ([^\s]+)")


def _parse_error(msg: str) -> dict:
  out = {}
  m = _SCOPED_RE.search(msg)
  if m:
    out.update(size=m.group(1), limit=m.group(2), over_by=m.group(3))
  m = _OP_RE.search(msg)
  if m:
    out.update(op=m.group(1), op_shape=m.group(2))
  for kind in (
      "CompileTimeScopedVmemOom",
      "Ran out of memory in memory space vmem",
      "RESOURCE_EXHAUSTED",
      "INVALID_ARGUMENT",
      "UNIMPLEMENTED",
      "Mosaic",
  ):
    if kind in msg:
      out.setdefault("kinds", []).append(kind)
  return out


def _sds(shape, dtype, sharding):
  return jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)


# ----------------------------------------------------------------------------
# Case builders. Each returns (fn, [ShapeDtypeStruct...]).
# ----------------------------------------------------------------------------
BF16 = jnp.bfloat16
D, F, VOCAB, TOKENS = 7168, 2048, 129280, 4096
HEADS, QK_HEAD, V_HEAD = 128, 192, 128

# The lane's tiles (deepseek_v3.use_tpu_moe_kernels), overridable per case.
LANE_TILES = dict(
    tile_m=512,
    tile_k=1024,
    tile_n=1024,
    tgmm=(512, 1024, 0),
    w2_tgmm=(512, 256, 0),
)


def gmm_cases(E, M, tiles, tag):
  """Lane expert_segment_matmul fwd / dgrad / wgrad for W1 (=W3) and W2."""
  from torchtitan.experiments.tpu.kernels import expert_segment_matmul as esm  # pylint: disable=g-import-not-at-top

  tm, tk, tn = tiles["tile_m"], tiles["tile_k"], tiles["tile_n"]
  cases = []

  def add(name, fn, specs):
    cases.append((f"{tag}/{name}", fn, specs))

  # W1 / W3: x (M, D) @ w (E, D, F); W2: h (M, F) @ w (E, F, D);
  # W2 transposed RHS (FSDP-sharded on its trailing axis): w stored (E, D, F).
  projections = [
      ("w13", D, F, (E, D, F), tiles["tgmm"], False),
      ("w2", F, D, (E, F, D), tiles["w2_tgmm"], False),
      ("w2T", F, D, (E, D, F), tiles.get("w2T_tgmm", tiles["w2_tgmm"]), True),
  ]
  for pname, k_dim, n_dim, wshape, tg, transpose in projections:
    statics = (tm, tk, tn, *tg, transpose)

    def fwd(lhs, rhs, gs, statics=statics):
      return esm._grouped_mm_fwd(lhs, rhs, gs, *statics)

    def bwd_dgrad(dout, lhs, rhs, gs, statics=statics):
      return esm._grouped_mm_bwd(dout, lhs, rhs, gs, *statics)[0]

    def bwd_wgrad(dout, lhs, rhs, gs, statics=statics):
      return esm._grouped_mm_bwd(dout, lhs, rhs, gs, *statics)[1]

    def mk(sh, k_dim=k_dim, wshape=wshape):
      return [
          _sds((M, k_dim), BF16, sh),
          _sds(wshape, BF16, sh),
          _sds((E,), jnp.int32, sh),
      ]

    def mkb(sh, k_dim=k_dim, n_dim=n_dim, wshape=wshape):
      return [
          _sds((M, n_dim), BF16, sh),
          _sds((M, k_dim), BF16, sh),
          _sds(wshape, BF16, sh),
          _sds((E,), jnp.int32, sh),
      ]

    add(f"{pname}_fwd", fwd, mk)
    add(f"{pname}_dgrad", bwd_dgrad, mkb)
    add(f"{pname}_wgrad", bwd_wgrad, mkb)
  return cases


def paired_cases(E, M, tiles, tag):
  from torchtitan.experiments.tpu.kernels import paired_segment_matmul as psm  # pylint: disable=g-import-not-at-top

  statics = (tiles["tile_m"], tiles["tgmm"][0], tiles["tgmm"][1])

  def fwd(lhs, r1, r3, gs):
    return psm._paired_grouped_mm_fwd(lhs, r1, r3, gs, *statics)

  def bwd(d1, d3, lhs, r1, r3, gs):
    return psm._paired_grouped_mm_bwd(d1, d3, lhs, r1, r3, gs, *statics)

  def mk(sh):
    return [
        _sds((M, D), BF16, sh),
        _sds((E, D, F), BF16, sh),
        _sds((E, D, F), BF16, sh),
        _sds((E,), jnp.int32, sh),
    ]

  def mkb(sh):
    return [
        _sds((M, F), BF16, sh),
        _sds((M, F), BF16, sh),
        _sds((M, D), BF16, sh),
        _sds((E, D, F), BF16, sh),
        _sds((E, D, F), BF16, sh),
        _sds((E,), jnp.int32, sh),
    ]

  return [(f"{tag}/paired_fwd", fwd, mk), (f"{tag}/paired_bwd", bwd, mkb)]


def splash_cases(tag, tokamax=False, blocks=512, batch=1):
  if tokamax:
    from torchtitan.experiments.tpu.kernels import tokamax_splash_attention as sa  # pylint: disable=g-import-not-at-top
    extra = {}
  else:
    from torchtitan.experiments.tpu.kernels import splash_attention as sa  # pylint: disable=g-import-not-at-top
    extra = dict(use_document_aware_block_skipping=True)
  b = blocks
  fwd, bwd = sa._make_splash_attention_fn(
      seq_len=TOKENS,
      n_heads=HEADS,
      n_kv_heads=HEADS,
      is_causal=True,
      block_q=b,
      block_kv=b,
      block_kv_compute=b,
      block_q_dkv=b,
      block_kv_dkv=b,
      block_kv_dkv_compute=b,
      block_q_dq=b,
      block_kv_dq=b,
      use_fused_bwd_kernel=True,
      use_vmap_bwd=False,
      **extra,
  )
  B = batch

  def mk(sh):
    return [
        _sds((B, HEADS, TOKENS, QK_HEAD), BF16, sh),
        _sds((B, HEADS, TOKENS, QK_HEAD), BF16, sh),
        _sds((B, HEADS, TOKENS, V_HEAD), BF16, sh),
        _sds((B, TOKENS), jnp.int32, sh),
    ]

  def mkb(sh):
    return [
        _sds((B, HEADS, TOKENS, QK_HEAD), BF16, sh),
        _sds((B, HEADS, TOKENS, QK_HEAD), BF16, sh),
        _sds((B, HEADS, TOKENS, V_HEAD), BF16, sh),
        _sds((B, HEADS, TOKENS, V_HEAD), BF16, sh),
        _sds((B, HEADS, TOKENS), jnp.float32, sh),
        _sds((B, HEADS, TOKENS, V_HEAD), BF16, sh),
        _sds((B, TOKENS), jnp.int32, sh),
    ]

  f = lambda q, k, v, s: fwd(q, k, v, s)
  g = lambda q, k, v, o, l, d, s: bwd(q, k, v, o, l, d, s)
  name = "tokamax_splash" if tokamax else "splash"
  return [(f"{tag}/{name}_fwd", f, mk), (f"{tag}/{name}_bwd", g, mkb)]


def lce_cases(tag, bt=1024, ht=512, vt=2048):
  from torchtitan.experiments.tpu.kernels import linear_softmax_cross_entropy_loss as lce  # pylint: disable=g-import-not-at-top

  statics = ("sum", "mosaic_tpu", bt, ht, vt, None, 1.0)

  def fwd(x, labels, w):
    return lce._my_fwd(x, labels, w, *statics)

  def bwd(dout, lse, x, labels, w):
    return lce._my_bwd(dout, lse, x, labels, w, *statics)

  def mk(sh):
    return [
        _sds((TOKENS, D), BF16, sh),
        _sds((TOKENS,), jnp.int32, sh),
        _sds((D, VOCAB), BF16, sh),
    ]

  def mkb(sh):
    return [
        _sds((1,), jnp.float32, sh),
        _sds((TOKENS,), jnp.float32, sh),
        _sds((TOKENS, D), BF16, sh),
        _sds((TOKENS,), jnp.int32, sh),
        _sds((D, VOCAB), BF16, sh),
    ]

  return [(f"{tag}/lce_fwd", fwd, mk), (f"{tag}/lce_bwd", bwd, mkb)]


def route_score_cases(tag, N):
  from torchtitan.experiments.tpu.kernels import route_score as rs  # pylint: disable=g-import-not-at-top

  def fn(g, r):
    return rs._route_score_grad_tpu(g, r)

  def mk(sh):
    return [_sds((N, D), BF16, sh), _sds((N, D), BF16, sh)]

  return [(f"{tag}/route_score_grad", fn, mk)]


def recipe_tiles(recipe):
  """The expert tiles a lane recipe actually configures (first MoE layer)."""
  import torchtitan.config  # pylint: disable=g-import-not-at-top
  from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts  # pylint: disable=g-import-not-at-top

  job = torchtitan.config.ConfigManager().parse_args([
      "--module=torchtitan.experiments.tpu.deepseek_v3",
      f"--config={recipe}",
  ])
  e = next(
      c for _, c, _, _ in job.model_spec.model.traverse(TpuGroupedExperts.Config)
  )
  if not e.use_pallas_grouped_mm:
    raise ValueError(f"{recipe} does not use the Pallas grouped matmul")
  w2 = e.pallas_grouped_mm_w2_tgmm_tiling or e.pallas_grouped_mm_tgmm_tiling
  return dict(
      tile_m=e.pallas_grouped_mm_tile_m,
      tile_k=e.pallas_grouped_mm_tile_k,
      tile_n=e.pallas_grouped_mm_tile_n,
      tgmm=tuple(e.pallas_grouped_mm_tgmm_tiling),
      w2_tgmm=tuple(w2),
      w2T_tgmm=tuple(e.pallas_grouped_mm_w2_transposed_tgmm_tiling or w2),
  )


def all_cases(args):
  tiles = dict(LANE_TILES)
  if args.recipe:
    tiles.update(recipe_tiles(args.recipe))
  print(json.dumps(dict(tiles=tiles)), flush=True)
  if args.tiles:
    tiles.update(json.loads(args.tiles))
  cases = []
  # 16 local experts: EP 16 receive buffers (target) and EP 1 (2L proxy).
  for M in (32768, 65536, 131072, 524288):
    cases += gmm_cases(16, M, tiles, f"gmm_E16_M{M}")
  # 2 local experts (EP 8 on the 16-expert 2L proxy).
  for M in (32768, 65536):
    cases += gmm_cases(2, M, tiles, f"gmm_E2_M{M}")
  cases += paired_cases(16, 32768, tiles, "paired_E16_M32768")
  cases += splash_cases("attn_S4096_H128")
  cases += splash_cases("attn_S4096_H128", tokamax=True)
  cases += lce_cases("lce_T4096_V129280")
  cases += route_score_cases("route_N32768", TOKENS * 8)
  return cases


def compile_case(fn, specs_fn, dev, options):
  sh = jax.sharding.SingleDeviceSharding(dev)
  specs = specs_fn(sh)
  lowered = jax.jit(fn).lower(*specs)
  return lowered.compile(compiler_options=options or None)


def mosaic_kernel_count(compiled) -> int:
  """Pallas/Mosaic custom calls in the compiled module (silent-fallback guard)."""
  return compiled.as_text().count('custom_call_target="tpu_custom_call"')


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--target", choices=sorted(TARGETS), required=True)
  p.add_argument("--case", default=".*", help="regex over case names")
  p.add_argument("--tiles", default="", help="JSON override of LANE_TILES")
  p.add_argument(
      "--vmem-limit-kib",
      type=int,
      default=0,
      help="pass xla_tpu_scoped_vmem_limit_kib as a compiler option",
  )
  p.add_argument(
      "--probe-need",
      action="store_true",
      help=(
          "also compile with an 8 MiB scoped limit; the size in the error is a"
          " lower bound on the kernel's scoped VMEM (kernels that set their"
          " own vmem_limit_bytes are not constrained by it)"
      ),
  )
  p.add_argument(
      "--margin-limits",
      default="",
      help="comma-separated KiB limits to re-compile OK cases at",
  )
  p.add_argument("--recipe", default="", help="take expert tiles from recipe")
  p.add_argument("--out", default="")
  args = p.parse_args()

  _install_shims()
  topo = get_topology_desc(
      platform="tpu",
      chip_config_name="default",
      num_slices=1,
      wrap=(False, False, False),
      **TARGETS[args.target],
  )
  dev = topo.devices[0]
  mesh = jax.sharding.Mesh(np.array([dev]), ("x",))

  from jax.experimental.pallas import tpu as pltpu  # pylint: disable=g-import-not-at-top

  results = []
  # Build outside the mesh: splash materializes its static mask info as host
  # (CPU) constants at construction time.
  rx = re.compile(args.case)
  cases = [c for c in all_cases(args) if rx.search(c[0])]
  with jax.set_mesh(mesh):
    info = pltpu.get_tpu_info()
    header = dict(
        target=args.target,
        device_kind=dev.device_kind,
        num_devices=len(topo.devices),
        tpu_info_vmem_capacity_bytes=info.vmem_capacity_bytes,
        tpu_info_num_cores=info.num_cores,
        jax=jax.__version__,
    )
    print(json.dumps(header), flush=True)
    options = {}
    if args.vmem_limit_kib:
      options["xla_tpu_scoped_vmem_limit_kib"] = str(args.vmem_limit_kib)
    for name, fn, specs_fn in cases:
      rec = dict(case=name, target=args.target, options=options)
      try:
        compiled = compile_case(fn, specs_fn, dev, options)
        rec["status"] = "OK"
        rec["mosaic_kernels"] = mosaic_kernel_count(compiled)
      except Exception as e:  # pylint: disable=broad-exception-caught
        msg = f"{type(e).__name__}: {e}"
        rec["status"] = "FAIL"
        rec.update(_parse_error(msg))
        rec["error"] = msg[:1500]
        if "kinds" not in rec:
          rec["traceback"] = traceback.format_exc()[-1500:]
      if args.probe_need and rec["status"] == "OK":
        try:
          compile_case(
              fn, specs_fn, dev, {"xla_tpu_scoped_vmem_limit_kib": "8192"}
          )
          rec["need"] = "<=8M"
        except Exception as e:  # pylint: disable=broad-exception-caught
          parsed = _parse_error(f"{e}")
          rec["need"] = parsed.get("size", "?")
          rec["need_op"] = parsed.get("op", "?")
      if args.margin_limits and rec["status"] == "OK":
        rec["at_limit"] = {}
        for lim in args.margin_limits.split(","):
          try:
            compile_case(
                fn, specs_fn, dev, {"xla_tpu_scoped_vmem_limit_kib": lim}
            )
            rec["at_limit"][lim] = "OK"
          except Exception as e:  # pylint: disable=broad-exception-caught
            rec["at_limit"][lim] = "FAIL " + str(_parse_error(f"{e}").get("size"))
      results.append(rec)
      brief = {k: rec.get(k) for k in ("case", "status", "size", "limit",
                                       "over_by", "op", "op_shape", "need",
                                       "mosaic_kernels", "at_limit")}
      if rec["status"] == "FAIL" and "size" not in rec:
        brief["error"] = rec["error"][:600]
      print(json.dumps(brief), flush=True)
  if args.out:
    with open(args.out, "w") as f:
      json.dump(dict(header=header, results=results), f, indent=1)


if __name__ == "__main__":
  main()
