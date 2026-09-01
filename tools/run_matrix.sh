#!/usr/bin/env bash

# Sweep the configurations that plausibly move tail latency and write one CSV
# row per (config, round).
#
# Configurations are interleaved in short rounds rather than run to completion
# one after another: a sequential A/B drifts by a few percent as the CPU heats
# up, which is the same order as the differences being measured.  Compare
# medians across rounds, not single runs.
#
# The machine must be otherwise idle.  A concurrent build saturating the cores
# does not add a little noise to these numbers, it invalidates them.
#
# Usage:
#   tools/run_matrix.sh [fixture] [iterations] [rounds] [csv]

set -euo pipefail

FIXTURE="${1:-mpc_solver}"
ITERATIONS="${2:-300}"
ROUNDS="${3:-3}"
CSV="${4:-artifacts/matrix_${FIXTURE}.csv}"
CASE="${CASE:-0}"
BENCH="${BENCH:-./artifacts/bench}"

if [[ ! -x "$BENCH" ]]; then
  echo "no bench binary at $BENCH (run: make artifacts/bench)" >&2
  exit 1
fi

load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)
echo "load average before starting: $load"

# label:flags -- the axes are the API, whether execution is inline, and how
# many threads XLA is allowed to use.
CONFIGS=(
  "legacy_default:--api legacy --threads 0 --devices 4"
  "rt_sync_t1:--api rt --threads 1 --devices 1"
  "rt_sync_t2:--api rt --threads 2 --devices 1"
  "rt_sync_t4:--api rt --threads 4 --devices 1"
  "rt_sync_tdefault:--api rt --threads 0 --devices 1"
  "rt_async_t1:--api rt --async --threads 1 --devices 1"
  "rt_async_t4:--api rt --async --threads 4 --devices 1"
  "rt_async_tdefault:--api rt --async --threads 0 --devices 4"
)

echo "fixture=$FIXTURE case=$CASE iterations=$ITERATIONS rounds=$ROUNDS"
echo "writing $CSV"

for round in $(seq 1 "$ROUNDS"); do
  for entry in "${CONFIGS[@]}"; do
    label="${entry%%:*}"
    flags="${entry#*:}"
    echo "--- round $round: $label"
    # shellcheck disable=SC2086
    "$BENCH" \
      --fixture "$FIXTURE" \
      --case "$CASE" \
      --iterations "$ITERATIONS" \
      --warmup 30 \
      --label "${label}_r${round}" \
      --csv "$CSV" \
      $flags 2>&1 | grep -E "p50|max/p50|max rel err|FAIL" || true
  done
done

echo
echo "=== medians across rounds ==="
python3 - "$CSV" <<'PY'
import csv, statistics, sys, re
rows = list(csv.DictReader(open(sys.argv[1])))
by = {}
for r in rows:
    label = re.sub(r"_r\d+$", "", r["label"])
    by.setdefault(label, []).append(r)
print(f"{'config':22s} {'p50':>9s} {'p99':>9s} {'p99.9':>9s} {'max':>9s} "
      f"{'max/p50':>8s}")
for label, rs in by.items():
    med = lambda k: statistics.median(float(x[k]) for x in rs)
    print(f"{label:22s} {med('p50_us'):9.1f} {med('p99_us'):9.1f} "
          f"{med('p999_us'):9.1f} {med('max_us'):9.1f} "
          f"{med('max_over_p50'):8.3f}")
PY
