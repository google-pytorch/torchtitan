#!/usr/bin/env bash
# Runs each TPU test module given as an argument. A module that fails because
# the TPU runtime could not initialize is retried, up to MAX_ATTEMPTS in total.
# Exits non-zero if any module still fails.
#
# Usage: .ci/run_tpu_test_modules.sh <module> [<module> ...]

set -uo pipefail
ulimit -n 65536

MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

# Retry only infra-side TPU init failures on the 4-chip runners
# ("START_SESSION failed", "vbar.sock" DEADLINE_EXCEEDED). Both log this
# message and usually pass on rerun; real test failures are not retried.
transient_error_pattern='TPU initialization failed'

log=$(mktemp)
trap 'rm -f "$log"' EXIT

failed=()
retried=()
for module in "$@"; do
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "=================================================="
    echo "Running test module: $module (attempt $attempt/$MAX_ATTEMPTS)"
    echo "=================================================="
    if timeout 900 python -m "$module" --device=tpu 2>&1 | tee "$log"; then
      [ "$attempt" -gt 1 ] && retried+=("$module")
      continue 2
    fi
    if ! grep -qi "$transient_error_pattern" "$log"; then
      break
    fi
    echo "::warning::$module hit '$transient_error_pattern' on attempt $attempt/$MAX_ATTEMPTS"
  done
  failed+=("$module")
done

# Keep retried modules visible so the infra flake rate stays tracked.
for target in "${retried[@]}"; do
  echo "::warning::$target passed only after retrying a TPU initialization failure"
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "=================================================="
  echo "Failed test target(s):"
  for target in "${failed[@]}"; do
    echo "::error::$target failed"
  done
  exit 1
fi
