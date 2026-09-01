#!/usr/bin/env bash

# Audit the host settings that decide whether a bounded computation actually
# finishes on time.  Read-only: it reports, it does not change anything.
#
# Run this before trusting any latency number from a machine, and keep the
# output next to the numbers -- "p99.9 was 4.8 ms" means little without knowing
# whether the governor was on `powersave` at the time.

set -uo pipefail

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
  exit 0
fi

echo "cpu"
info "cores: $(nproc)"
governors=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null \
            | sort -u | tr '\n' ' ')
if [[ -z "$governors" ]]; then
  info "scaling governor: unavailable (virtualized?)"
elif [[ "$governors" == "performance " ]]; then
  ok "scaling governor: performance"
else
  bad "scaling governor: $governors (want: performance)"
fi

# Frequency scaling and deep idle states both trade wake-up latency for power.
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
echo "=== $pass ok, $warn worth fixing ==="
[[ "$warn" -eq 0 ]]
