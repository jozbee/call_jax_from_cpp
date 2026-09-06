# 03 · Real-time loop

`examples/03_realtime` runs the function example 02 exports on a fixed period
with the hardening applied, records every call and every wake-up, and prints
a report when it stops. It is the shape a control process takes: hardening
once at startup, then a loop that writes inputs, calls, reads outputs, and
sleeps until the next deadline.

Two limitations are stated up front. **An overrun is late data, not a
cancelled call**: PJRT cannot cancel a running CPU computation, so the loop
never skips a period — a deadline already in the past returns at once and the
loop catches up. And **a run without privileges is still a correct run**; it
is just not one to quote tail numbers from. Every hardening step reports
`[ok  ]` or `[skip]`, and the run continues.

## Prerequisites

`make export` (this example loads `artifacts/trajopt`), a plugin, and `make`.
Optionally: `CAP_SYS_NICE` or a raised `RLIMIT_RTPRIO` for `SCHED_FIFO`,
`RLIMIT_MEMLOCK` for `mlockall`, and `build/lib/malloc_guard.so` preloaded for
the allocation census.

## Hardening

```{literalinclude} ../../examples/03_realtime/realtime.cpp
:language: cpp
:start-after: docs: begin rt-harden
:end-before: docs: end rt-harden
```

Six steps in the one safe order: allocator, memory lock, pin, corral XLA's
threads (after the `Runtime` exists), C-states, priority last.
{doc}`../guides/realtime` says what each one buys.

## The loop

```{literalinclude} ../../examples/03_realtime/realtime.cpp
:language: cpp
:start-after: docs: begin rt-loop
:end-before: docs: end rt-loop
```

`sleep_until` is an absolute `clock_nanosleep` from
`examples/common/periodic.hpp`. Four recorders take one sample per cycle:
compute, cycle, wake-up latency, and signed period jitter.

## The measured window

```{literalinclude} ../../examples/03_realtime/realtime.cpp
:language: cpp
:start-after: docs: begin alloc-guard
:end-before: docs: end alloc-guard
```

The census and the rusage snapshots cover exactly the cycles the recorders do.
Warm-up is outside, because its faults are what warm-up exists to pay.

## The report

```{literalinclude} ../../examples/03_realtime/realtime.cpp
:language: cpp
:start-after: docs: begin rt-report
:end-before: docs: end rt-report
```

Correctness outranks the allocation gate in the exit code. Flags, the host
audit and the JSON writer live in `examples/03_realtime/support.hpp`.

## Build and run

```console
$ make plugin && make && make export
$ ./build/bin/example_03_realtime --iterations 1000
$ LD_PRELOAD=$PWD/build/lib/malloc_guard.so \
    ./build/bin/example_03_realtime --iterations 1000 --json artifacts/reports/realtime.json
```

The flags that matter: `--period-us` (default 10000), `--iterations` (`0`
runs until SIGINT), `--cpu auto|none|N`, `--rt-priority N` (default 80; `0`
opts out of `SCHED_FIFO`), `--dma-latency auto|off`, `--json`, `--samples`,
`--alloc-gate off|self|all`, `--require-guard`. `--help` is authoritative, and
an unknown flag is refused rather than ignored.

## Expected output

The hardening block, from this project's development host with no special
privileges and generous rlimits — the statuses depend on the host, and that
is the point of printing them:

```text
=== hardening ===
  [ok  ] harden_malloc: M_TRIM_THRESHOLD=-1 M_MMAP_MAX=0 M_ARENA_MAX=1
  [ok  ] lock_memory: mlockall(MCL_CURRENT|MCL_FUTURE)
  [skip] pin_current_thread: no isolated cpus available to this thread; running unpinned. Boot with isolcpus=/nohz_full=, or pass --cpu N to pin anyway
  [skip] corral_xla_threads: no XLA worker threads found (client not created?)
  [skip] cpu_dma_latency: not requested
  [ok  ] set_realtime_priority: SCHED_FIFO priority 80
```

A `[skip]` on `lock_memory` or `set_realtime_priority` is the usual result on
a stock login shell; `docker/compose.yml` grants both. How to read the four
distributions, the deadline line, the faults, the context switches and the
allocation census is on {doc}`../guides/realtime`.
