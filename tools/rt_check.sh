#!/usr/bin/env bash
#
# Audit the host settings that decide whether a bounded computation finishes
# on time. Read-only. Run it before trusting a latency number from a machine,
# and keep the output next to the number: a p99.9 means little without knowing
# what the governor was at the time.
#
# Usage:
#   tools/rt_check.sh [--quiet]
#
#   --quiet  print only the summary line (for use in a report header).
#
# Exits non-zero when anything is worth fixing, so it can gate a run. Sections:
# cpu, kernel, isolation, memory, process limits, container.

set -euo pipefail

QUIET=0

# The header comment is the help text; it ends at the first non-comment line.
usage() { awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -q|--quiet) QUIET=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# The detail goes to stdout and the one-line verdict to fd 3, so --quiet is one
# redirect instead of a conditional around every echo.
exec 3>&1
if [[ "$QUIET" == 1 ]]; then
  exec 1>/dev/null
fi

pass=0
warn=0

ok()   { echo "  [ ok ] $1";                 pass=$((pass + 1)); }
bad()  { echo "  [warn] $1";                 warn=$((warn + 1)); }
info() { echo "  [info] $1"; }

read_or() {  # path, fallback
  if [[ -r "$1" ]]; then cat "$1"; else echo "$2"; fi
}

echo "=== real-time host audit ==="
echo

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "  This host is $(uname -s). Real-time hardening is Linux-only;"
  echo "  use this machine for correctness, not for latency numbers."
  echo "=== not Linux: no audit ===" >&3
  exit 0
fi

echo "cpu"
info "cores: $(nproc)"
# No cpufreq at all (a VM) is a finding, not a reason to abort.
governors=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null \
            | sort -u | tr '\n' ' ' || true)
if [[ -z "$governors" ]]; then
  info "scaling governor: unavailable (virtualized?)"
elif [[ "$governors" == "performance " ]]; then
  ok "scaling governor: performance"
else
  bad "scaling governor: $governors (want: performance)"
fi

if [[ -r /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
  if [[ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)" == "1" ]]; then
    ok "turbo disabled (consistent clocks)"
  else
    info "turbo enabled: peak speed, less consistent clocks"
  fi
fi

smt=$(read_or /sys/devices/system/cpu/smt/control unavailable)
if [[ "$smt" == "off" || "$smt" == "notsupported" ]]; then
  ok "SMT: $smt"
else
  bad "SMT: $smt (a sibling thread steals from the control loop)"
fi

echo
echo "kernel"
kernel_version=$(uname -v)
if [[ "$(read_or /sys/kernel/realtime 0)" == "1" || "$kernel_version" == *PREEMPT_RT* ]]; then
  ok "PREEMPT_RT: $kernel_version"
else
  bad "not a PREEMPT_RT kernel: $kernel_version"
  if [[ "$kernel_version" == *PREEMPT_DYNAMIC* ]]; then
    if grep -q 'preempt=full' /proc/cmdline 2>/dev/null; then
      info "       PREEMPT_DYNAMIC booted preempt=full: preemptible, but the"
      info "       priority-inheritance and threaded-IRQ guarantees are still"
      info "       missing"
    else
      info "       PREEMPT_DYNAMIC: boot with preempt=full to get closer"
    fi
  fi
fi

rt_runtime=$(read_or /proc/sys/kernel/sched_rt_runtime_us unavailable)
rt_period=$(read_or /proc/sys/kernel/sched_rt_period_us unavailable)
if [[ "$rt_runtime" == "-1" ]]; then
  ok "sched_rt_runtime_us: -1 (real-time throttling disabled)"
else
  info "sched_rt_runtime_us: $rt_runtime of sched_rt_period_us $rt_period"
  info "       the default 950000/1000000 caps every SCHED_FIFO/RR task at 95%"
  info "       of each period; a loop that sleeps or blocks between calls never"
  info "       reaches the cap, a busy-wait one does and is then descheduled"
  info "       for the rest of the period"
fi

echo
echo "isolation"
isolated=$(read_or /sys/devices/system/cpu/isolated "")
if [[ -n "$isolated" ]]; then
  ok "isolcpus: $isolated"
else
  bad "no isolated cpus (add isolcpus=... to the kernel command line)"
fi

nohz=$(read_or /sys/devices/system/cpu/nohz_full "")
if [[ -n "$nohz" && "$nohz" != "(null)" ]]; then
  ok "nohz_full: $nohz"
else
  bad "nohz_full not set (the timer tick will interrupt the loop)"
fi

if grep -q "rcu_nocbs" /proc/cmdline 2>/dev/null; then
  ok "rcu_nocbs present on the kernel command line"
else
  bad "rcu_nocbs not set (RCU callbacks run on the loop's core)"
fi

echo
echo "memory"
thp=$(read_or /sys/kernel/mm/transparent_hugepage/enabled unavailable)
if [[ "$thp" == *"[never]"* || "$thp" == *"[madvise]"* ]]; then
  ok "transparent hugepages: $thp"
else
  bad "transparent hugepages: $thp (khugepaged stalls faulting threads)"
fi
info "swap: $(free -h 2>/dev/null | awk '/Swap/ {print $2 " total, " $3 " used"}')"

echo
echo "process limits"
rtprio=$(ulimit -r 2>/dev/null || echo 0)
if [[ "$rtprio" != "0" ]]; then
  ok "RLIMIT_RTPRIO: $rtprio (SCHED_FIFO available)"
else
  bad "RLIMIT_RTPRIO is 0 (run with --cap-add=SYS_NICE --ulimit rtprio=99)"
fi

memlock=$(ulimit -l 2>/dev/null || echo 0)
if [[ "$memlock" == "unlimited" ]]; then
  ok "RLIMIT_MEMLOCK: unlimited (mlockall will succeed)"
else
  bad "RLIMIT_MEMLOCK: $memlock KB (add --ulimit memlock=-1)"
fi

# Deep C-states are the classic source of a single multi-millisecond outlier.
if [[ -w /dev/cpu_dma_latency ]]; then
  ok "/dev/cpu_dma_latency writable (can pin latency to 0)"
else
  info "/dev/cpu_dma_latency not writable (needs root; holding it open at 0"
  info "       keeps cores out of deep C-states)"
fi

echo
echo "container"
if [[ -f /.dockerenv || -n "${CJFC_IN_CONTAINER:-}" ]]; then
  info "running inside a container"
  info "       the sections above read the HOST's kernel: governor, isolcpus,"
  info "       nohz_full, THP and C-states are the host's to set, and this"
  info "       audit cannot change them from in here"
  info "       what the container itself needs is"
  info "         --cap-add=SYS_NICE --ulimit rtprio=99 --ulimit memlock=-1"
  info "       SYS_NICE for sched_setscheduler(SCHED_FIFO), rtprio for the"
  info "       priority itself, memlock for mlockall; without them the"
  info "       real-time example falls back to plain scheduling"
else
  info "not in a container"
fi

echo
summary="=== $pass ok, $warn worth fixing ==="
echo "$summary"
if [[ "$QUIET" == 1 ]]; then
  echo "$summary" >&3
fi
[[ "$warn" -eq 0 ]]
