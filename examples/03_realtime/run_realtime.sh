#!/usr/bin/env bash
#
# Launch the real-time example with whatever this host is willing to give it,
# and say plainly what it could not.
#
# The audit comes FIRST, before a single number is produced: a latency figure
# whose provenance was not written down cannot be defended three weeks later,
# and a figure from a busy machine is not noisy, it is wrong (the same
# configuration measured during a build reported p50 2.4x high and max/p50 4.4
# instead of 1.1).  So this prints tools/rt_check.sh and /proc/loadavg, then
# runs, then hands you both together.
#
# Usage:
#   examples/03_realtime/run_realtime.sh [--iterations N] [--period-us N] ...
#
#   Every argument is passed through to the binary; --help lists its flags.
#
# Environment:
#   CPUSET    taskset -c argument, e.g. "3" or "2-3".  Unset: no taskset.
#   CHRT      chrt -f priority, e.g. 80.  Unset: no chrt.  See the note below.
#   NO_GUARD  set to anything to skip preloading the allocation counter.
#   BIN       binary to run (default: build/bin/example_03_realtime)
#   PYTHON    interpreter for the export (default: "uv run python", because uv
#             comes from mise and is not on PATH in a non-interactive shell)
#
# CHRT vs --rt-priority: `chrt -f 80 ./example_03_realtime` starts the WHOLE
# process under SCHED_FIFO, and every thread XLA creates afterwards inherits
# that scheduling class -- including its pool threads, which then compete with
# the control loop at real-time priority instead of yielding to it, and which
# run the artifact load and the warm-up (unbounded work) at priority 80.  The
# in-process `set_realtime_priority`, which the binary applies through
# --rt-priority, runs after the Runtime exists and promotes only the calling
# thread.  Prefer it.  CHRT is here for the case where RLIMIT_RTPRIO is granted
# to the launcher and not to the process, and for comparing the two.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && [[ $# -eq 1 ]]; then
  # The header comment is the help text; stopping at the first non-comment line
  # means there is no line range to keep in step with edits.
  awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BIN="${BIN:-build/bin/example_03_realtime}"
PYTHON="${PYTHON:-uv run python}"
GUARD="build/lib/malloc_guard.so"
ARTIFACT="artifacts/trajopt"

have() { command -v "$1" >/dev/null 2>&1; }
note() { echo "  [note] $*"; }

# ------------------------------------------------------------------- audit
#
# rt_check.sh exits non-zero when anything is worth fixing, which is its whole
# point; `|| true` keeps that from ending this script under `set -e`.
if [[ -x tools/rt_check.sh ]]; then
  tools/rt_check.sh || true
else
  echo "=== no tools/rt_check.sh: host audit skipped ==="
fi

echo
echo "load average before starting: $(cat /proc/loadavg 2>/dev/null || echo unknown)"
load1="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)"
if awk -v l="$load1" 'BEGIN { exit !(l > 1.0) }'; then
  echo
  echo "  *** loadavg1 is $load1. Numbers from a busy machine are wrong, not"
  echo "  *** noisy. Stop everything else before quoting anything from this run."
fi
echo

# ------------------------------------------------------------------ export
#
# Only when the artifact this example defaults to is the one being loaded: a
# caller who passed --artifact somewhere else is managing their own.
uses_default_artifact=1
for arg in "$@"; do
  case "$arg" in
    --artifact|--artifact=*) uses_default_artifact=0 ;;
  esac
done

if [[ "$uses_default_artifact" == 1 && ! -f "$ARTIFACT.json" ]]; then
  echo "=== exporting $ARTIFACT (examples/02_trajopt/export.py) ==="
  mkdir -p artifacts
  # PYTHONPATH rather than an install, so a checkout runs without one.
  PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
    $PYTHON examples/02_trajopt/export.py --out artifacts --cases 4
  echo
fi

if [[ ! -x "$BIN" ]]; then
  echo "no binary at $BIN (run: make, or set BIN=)" >&2
  exit 1
fi

# ----------------------------------------------------------------- wrapping

CMD=()

if [[ -n "${CPUSET:-}" ]]; then
  if have taskset; then
    CMD+=(taskset -c "$CPUSET")
  else
    note "CPUSET=$CPUSET ignored: no taskset on this host"
  fi
fi

if [[ -n "${CHRT:-}" ]]; then
  if have chrt; then
    CMD+=(chrt -f "$CHRT")
    note "CHRT=$CHRT: XLA's pool threads will inherit SCHED_FIFO too; see the"
    note "       header of this script for why --rt-priority is preferred"
  else
    note "CHRT=$CHRT ignored: no chrt on this host"
  fi
fi

case "$(uname -s)" in
  Darwin) PRELOAD_VAR=DYLD_INSERT_LIBRARIES ;;
  *)      PRELOAD_VAR=LD_PRELOAD ;;
esac

if [[ -z "${NO_GUARD:-}" ]]; then
  if [[ -f "$GUARD" ]]; then
    # Absolute: the loader resolves a bare name against the library path, not
    # against the working directory.
    export "$PRELOAD_VAR"="$ROOT/$GUARD"
    note "allocation census on ($PRELOAD_VAR=$ROOT/$GUARD)"
  else
    note "no $GUARD (run: make guard); the allocation census will be absent"
  fi
else
  note "NO_GUARD set: no allocation census"
fi

CMD+=("$BIN" "$@")

echo
echo "=== ${CMD[*]} ==="
echo

# exec, so signals reach the loop directly: this script must not sit between
# a Ctrl-C and the handler that stops the loop and prints its report.
exec "${CMD[@]}"
