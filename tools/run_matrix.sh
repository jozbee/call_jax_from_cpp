#!/usr/bin/env bash
#
# Sweep the configurations that plausibly move tail latency and write one CSV
# row per (config, round).
#
# Configurations are interleaved in short rounds rather than run to completion
# one after another: a sequential A/B drifts with CPU temperature by the same
# order as the differences being measured. Compare medians across rounds.
#
# The machine must be otherwise idle: a concurrent build does not add noise to
# these numbers, it invalidates them. /proc/loadavg is printed before and after.
#
# Usage:
#   tools/run_matrix.sh [fixture] [iterations] [rounds] [csv]
#
#   fixture     exported function to call (default: trajopt)
#   iterations  timed calls per (config, round) (default: 300)
#   rounds      interleaved repeats of the whole config list (default: 3)
#   csv         output, appended (default: artifacts/matrix_<fixture>.csv)
#
# Environment:
#   BENCH  benchmark binary (default: build/bin/bench)
#   CASE   reference case index passed to --case (default: 0)

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  # The header comment is the help text; it ends at the first non-comment line.
  awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"
  exit 0
fi

FIXTURE="${1:-trajopt}"
ITERATIONS="${2:-300}"
ROUNDS="${3:-3}"
CSV="${4:-artifacts/matrix_${FIXTURE}.csv}"
CASE="${CASE:-0}"
BENCH="${BENCH:-build/bin/bench}"

if [[ ! -x "$BENCH" ]]; then
  echo "no bench binary at $BENCH (run: make bench, or set BENCH=)" >&2
  exit 1
fi

load_before=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)
echo "load average before starting: $(cat /proc/loadavg 2>/dev/null || echo unknown)"
if awk -v l="$load_before" 'BEGIN { exit !(l > 0.5) }'; then
  echo
  echo "  *** this machine is busy (1-minute load $load_before). These numbers"
  echo "  *** will not be valid: a concurrent build has been measured to raise"
  echo "  *** p50 by 2.4x and max/p50 from 1.1 to 4.4. Stop everything else."
  echo
fi

# label:flags. The axes are inline vs async execution and the XLA thread count;
# `tdefault` leaves it unset, which is what a caller gets by accident. The last
# row adds the real-time hardening on top of the best-behaved configuration.
# docs: begin matrix-configs
CONFIGS=(
  "sync_t1:--threads 1"
  "sync_t2:--threads 2"
  "sync_t4:--threads 4"
  "sync_tdefault:--threads 0"
  "async_t1:--async --threads 1"
  "async_t4:--async --threads 4"
  "async_tdefault:--async --threads 0"
  "sync_t1_rt:--threads 1 --rt --cpu auto"
)
# docs: end matrix-configs

echo "fixture=$FIXTURE case=$CASE iterations=$ITERATIONS rounds=$ROUNDS"
echo "writing $CSV"
mkdir -p "$(dirname "$CSV")"
if [[ -e "$CSV" ]]; then
  echo "note: $CSV exists; rows are appended and the summary below covers all"
  echo "      of them, including any from an earlier sweep"
fi

# Rounds are the outer loop: that is what interleaves the configurations.
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
echo "load average after finishing: $(cat /proc/loadavg 2>/dev/null || echo unknown)"
echo
echo "=== medians across rounds ==="
python3 - "$CSV" <<'PY'
import csv, statistics, sys, re

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("no rows in", sys.argv[1])
    raise SystemExit(0)

# p99.99 needs ~10k samples to mean anything, so it is reported only when the
# bench binary wrote the column.
cols = [("p50_us", "p50"), ("p99_us", "p99"), ("p999_us", "p99.9")]
if "p9999_us" in rows[0]:
    cols.append(("p9999_us", "p99.99"))
cols += [("max_us", "max"), ("max_over_p50", "max/p50")]

by = {}
for r in rows:
    label = re.sub(r"_r\d+$", "", r["label"])
    by.setdefault(label, []).append(r)

print(f"{'config':22s}" + "".join(f"{t:>9s}" for _, t in cols) + f"{'rounds':>8s}")
for label, rs in by.items():
    def med(key):
        return statistics.median(float(row[key]) for row in rs)

    cells = "".join(
        f"{med(k):9.3f}" if k == "max_over_p50" else f"{med(k):9.1f}"
        for k, _ in cols
    )
    print(f"{label:22s}{cells}{len(rs):8d}")
PY
