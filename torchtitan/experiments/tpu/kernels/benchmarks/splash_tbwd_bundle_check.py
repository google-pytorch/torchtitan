"""Compile splash_tbwd at the Qwen3-235B layer shape and check its VLIW bundle count against a safe limit.

The kernel is fully unrolled over query heads, so its code size is large. Above the TPU7x instruction-memory
capacity it runs from overlays and gets ~3.5x slower (an 81k-bundle variant: 43 ms vs 12 ms). This script dumps the
final LLO schedule (libtpu `--xla_jf_dump_to`), reads "total scheduled bundles" of the splash_tbwd custom call and
exits non-zero above `--max-bundles`. Run it after changing the head unroll or the number of tile kinds.

Usage: python3 -m torchtitan.experiments.tpu.kernels.benchmarks.splash_tbwd_bundle_check [--max-bundles N]
"""

import argparse
import glob
import math
import os
import re
import sys
import tempfile

# Safe ceiling, with margin under the cliff (53k bundles ran fine, 81k fell off it; see splash_tbwd.py "Limits").
MAX_BUNDLES = 45_000


def count_bundles(dump_dir):
  counts = []
  for path in glob.glob(os.path.join(dump_dir, "*splash_tbwd*schedule-analysis_final_bundles.txt")):
    m = re.search(r"total scheduled bundles:\s+(\d+)", open(path).read())
    if m:
      counts.append(int(m.group(1)))
  return max(counts) if counts else None


def enable_llo_dump(dump_dir=None):
  """Adds libtpu flags that dump final LLO schedules; returns the dump directory.

  libtpu reads its flags once, at backend start, so this must run before anything initializes the TPU.
  """
  dump_dir = dump_dir or tempfile.mkdtemp(prefix="splash_tbwd_llo_")
  os.environ["LIBTPU_INIT_ARGS"] = (os.environ.get("LIBTPU_INIT_ARGS", "")
                                    + f" --xla_jf_dump_to={dump_dir} --xla_jf_dump_llo_pass_label_regex=final")
  return dump_dir


def compile_q235_kernel():
  """Compiles transposed_attention_bwd at one Qwen3-235B example (S 4096, 16 q heads per kv head)."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from torchtitan.experiments.tpu.kernels import splash_tbwd as tb  # pylint: disable=g-import-not-at-top

  b, s, num_h, num_kv, d = 1, 4096, 16, 1, tb.HEAD_DIM
  bf16 = jnp.bfloat16
  shapes = (
      jax.ShapeDtypeStruct((b, s, num_h, d), bf16), jax.ShapeDtypeStruct((b, s, num_kv, d), bf16),
      jax.ShapeDtypeStruct((b, s, num_kv, d), bf16), jax.ShapeDtypeStruct((b, s, num_h, d), bf16),
      jax.ShapeDtypeStruct((b, num_h, s), jnp.float32), jax.ShapeDtypeStruct((b, s, num_h, d), bf16),
      jax.ShapeDtypeStruct((b, s), jnp.int32),
  )
  jax.jit(lambda *x: tb.transposed_attention_bwd(*x, scale=d ** -0.5, log2e=math.log2(math.e))).lower(*shapes).compile()


def main(argv=None):
  p = argparse.ArgumentParser()
  p.add_argument("--max-bundles", type=int, default=MAX_BUNDLES)
  p.add_argument("--dump-dir", default=None)
  a = p.parse_args(argv)
  dump_dir = enable_llo_dump(a.dump_dir)
  compile_q235_kernel()
  n = count_bundles(dump_dir)
  if n is None:
    print(f"no splash_tbwd bundle dump found under {dump_dir}", file=sys.stderr)
    return 2
  print(f"splash_tbwd: {n} bundles (limit {a.max_bundles}); dump {dump_dir}")
  return 0 if n <= a.max_bundles else 1


if __name__ == "__main__":
  sys.exit(main())
