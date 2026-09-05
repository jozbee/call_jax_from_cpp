# 03 · Real-time loop

`examples/03_realtime` runs a function on a fixed period with the `pjrt::rt`
helpers applied, records every call and every wake-up, and prints a report when
it stops. It is the shape a control process actually takes: hardening once at
startup, then a loop that writes inputs, calls, reads outputs, and sleeps until
the next deadline.

It has no export of its own — it runs the artifact example 02 writes.

## Prerequisites

1. **`make export`**, which runs `examples/02_trajopt/export.py`. This example
   loads `artifacts/trajopt`. Re-run it after a JAX bump or on a new machine;
   `.binpb` artifacts are locked to the host that produced them.
2. **A plugin**: `make plugin`, or `$PJRT_CPU_PLUGIN` pointing at one.
3. **`make`**, which builds `build/bin/example_03_realtime`.
4. **Privileges, optionally.** `SCHED_FIFO` needs `CAP_SYS_NICE` or a raised
   `RLIMIT_RTPRIO`; `mlockall` needs `RLIMIT_MEMLOCK`. Without them the example
   runs and reports that it is not hardened, which is the expected result on an
   ordinary login shell.
5. **The allocation guard, optionally.** Preloading
   `build/lib/malloc_guard.so` turns on the per-call allocation census;
   without it the example runs and reports that the guard is absent.

## The loop

```{literalinclude} ../../examples/03_realtime/realtime.cpp
:language: cpp
:caption: examples/03_realtime/realtime.cpp
```

## Build and run

```console
$ make plugin && make && make export
$ ./build/bin/example_03_realtime --iterations 1000
```

With the allocation census, which is how `make run-examples` invokes it:

```console
$ LD_PRELOAD=$PWD/build/lib/malloc_guard.so \
    ./build/bin/example_03_realtime --iterations 1000 --json artifacts/reports/realtime.json
```

The flags are `--artifact` (the base path to load), `--iterations`, `--cpu`
(`auto`, `none`, or a CPU number to pin to), `--rt` (ask for `SCHED_FIFO`), and
`--json` (write the report as well as print it). `--help` prints the
authoritative list, and an unrecognized flag is refused rather than ignored —
a mistyped `--iterationss` silently falling back to a default is exactly how a
run ends up measuring a configuration nobody chose.

## What the hardening does, in order

Each step is independent, each returns a `pjrt::rt::Status`, and **none of them
fails the program**. A process that could not raise its own priority should run
anyway and say so in its startup log.

| Step | What it does | What it needs |
|---|---|---|
| `harden_malloc()` | stops the allocator returning memory to the kernel, so the next allocation does not have to fault it back in | nothing |
| `lock_memory()` | `mlockall`, plus one pass that grows and touches heap and stack so the first calls do not pay for lazily reserved pages | `RLIMIT_MEMLOCK` |
| `pin_current_thread(cpu)` | stops the loop thread migrating between caches | nothing beyond its own affinity |
| `corral_xla_threads({...})` | moves XLA's pools off that core, found by name in `/proc/self/task` | the `Runtime` to already exist |
| `set_realtime_priority(80)` | `SCHED_FIFO` at that priority | `CAP_SYS_NICE` or `RLIMIT_RTPRIO` |

Priority comes last so the setup work itself does not run on a real-time
thread, and `corral_xla_threads` comes after the `Runtime` because XLA names
those threads when the client is created.

## Expected output

Unprivileged, on an ordinary desktop, the hardening block reports what did not
take effect and the run continues:

```{code-block} text
:caption: Illustrative. The statuses depend on privileges; every number varies with the machine, and none of it is a latency result unless the machine was idle.

harden_malloc          ok
lock_memory            FAILED: mlockall: Cannot allocate memory (RLIMIT_MEMLOCK is 8 MiB)
pin_current_thread     ok (cpu 2)
corral_xla_threads     ok (2 threads moved to {3})
set_realtime_priority  FAILED: sched_setscheduler: Operation not permitted (need CAP_SYS_NICE)
```

That is the **expected** result outside a container that grants them. The
devcontainer does grant both (`cap_add: SYS_NICE`, and `rtprio` / `memlock`
ulimits), and `tools/rt_check.sh` audits the host settings underneath them —
governor, `isolcpus`, `nohz_full`, transparent hugepages, `RLIMIT_RTPRIO`.

## How to read the report

Two clocks are being watched, and confusing them is the usual mistake. **Call
latency** is how long `call()` took. **Period jitter** is how far the cycle
started from where it was scheduled to start; it is a signed quantity, because
a cycle that wakes early is as much a scheduling defect as one that wakes late,
and `LatencyRecorder` holds signed nanoseconds precisely so that a negative
sample is representable rather than clamped or lost.

| Line | What it means | What a bad value implies |
|---|---|---|
| call latency `p50` / `p99.9` / `max` | the computation itself | a `max/p50` above 2 means the tail is real, not sampling |
| period jitter (signed) | scheduled wake-up minus actual | a wide spread means the scheduler, not XLA; check the governor and `isolcpus` |
| wake-up latency | how late the sleep returned | tracks C-state exit and timer resolution |
| deadline misses | cycles whose work did not finish before the next period | one is a bug in the period; many are a bug in the workload |
| page faults | minor and major faults during the timed window | anything nonzero means `lock_memory` did not take effect, or the warm-up was too short |
| context switches | voluntary and involuntary | involuntary switches on a pinned `SCHED_FIFO` thread mean something else wants that core |
| allocations | the armed census, per call | `self` **must be zero**; `plugin` will be thousands per call (9,750 for this workload), which is XLA's thunk runtime and not reachable from here |

One honest limitation belongs on the same page as the deadline column: **PJRT
cannot cancel a running CPU computation.** A watchdog can return a stale result
and discard the late one, but an overrun means late data, not cancellation.
Design for a bounded computation rather than for an interrupt.

## What to change to make it yours

The period, the artifact, and the core assignment are the three knobs that
matter, and the last two interact: pin the loop to an isolated core and corral
XLA's pools onto different ones, or the two will contend for exactly the
resource the pinning was meant to reserve.

Beyond that, the structure is the lesson. Do the hardening once, before the
loop. Do the name-to-index resolution once, before the loop. Keep the loop body
to `memcpy` in, `call()`, read out — nothing in it should allocate, lock, log or
flush, and `make test-alloc` is the gate that proves the wrapper's half of that.
Logging in particular belongs to a separate thread or to the end of the run:
the recorder exists so that a run can be summarized after it finishes rather
than narrated while it happens.
