# 03 · Minimal real-time loop

*Assumes the {doc}`Quickstart </getting-started/quickstart>` only: this is
its artifact, called on a period. Nothing from examples 01 or 02.*

`examples/03_minimal/minimal.cpp` is the least a hardened periodic loop
needs, in one file you can copy: the `pjrt::rt` calls in the one safe order,
an absolute-time sleep, and two recorders. It loads `artifacts/basic` — the
four-by-four solve the Quickstart exported — and calls it on a fixed period,
writing a fresh right-hand side each cycle and checking the residual the
function returns.

:::{note}
Two namespaces appear: `pjrt::` is the library ({doc}`/api/index`) and
`pjrt::rt::` is its hardening layer ({doc}`/api/cpp/rt`). The file includes
nothing from `examples/common/`. The same loop with the measurement attached
is {doc}`04-realtime`.
:::

## Setup, in order

```{literalinclude} ../../examples/03_minimal/minimal.cpp
:language: cpp
:start-after: docs: begin minimal-setup
:end-before: docs: end minimal-setup
```

| Call | What it removes | Needs |
|---|---|---|
| {cpp:func}`~pjrt::rt::harden_malloc` | the allocator handing memory back to the kernel, to be faulted in again by a later call | nothing; first, before the `Runtime` allocates in bulk |
| {cpp:func}`~pjrt::rt::lock_memory` | a {term}`page fault` inside a call | `RLIMIT_MEMLOCK` unlimited |
| {cpp:func}`~pjrt::rt::pin_current_thread` | migration between cores; cold caches | a CPU number, and an {term}`isolated CPU` for it to be worth much |
| {cpp:func}`~pjrt::rt::corral_xla_threads` | XLA's {term}`thread pool` waking on the loop's core | the `Runtime` to exist, and a core to move them to |
| {cpp:func}`~pjrt::rt::set_realtime_priority` | anything at normal priority preempting the loop | `CAP_SYS_NICE` or `RLIMIT_RTPRIO`; **last**, and a loop that blocks |

Each returns a {cpp:struct}`~pjrt::rt::Status`, and `print` — the file's
own four-line helper — prints it. A `[skip]` is the normal result on a stock
login shell, and the run is still correct. Without a CPU argument the pin and
the corral are not attempted at all: `cpus_except(cpu)`, the file's other
helper, is every online CPU but the pinned one, which is where XLA's pools go. Deep {term}`C-states <C-state>` are left alone here because holding
`/dev/cpu_dma_latency` needs root; {cpp:class}`~cjfc::DmaLatencyHold` in the
example layer does it. {doc}`/guides/realtime` has what each call buys and
what the host has to provide.

## The loop

```{literalinclude} ../../examples/03_minimal/minimal.cpp
:language: cpp
:start-after: docs: begin minimal-loop
:end-before: docs: end minimal-loop
```

`now_ns` and `sleep_until` are the file's remaining two helpers:
`clock_gettime` and `clock_nanosleep` on `CLOCK_MONOTONIC`, the latter with
`TIMER_ABSTIME` — an {term}`absolute sleep`, so one wake-up's lateness does not
feed into the next period. Never skip a period: a deadline already in the past
returns at once and the loop catches up. Nothing in the body allocates or
logs — the two {cpp:class}`LatencyRecorder <pjrt::LatencyRecorder>`s reserved
their capacity before the loop. An overrun is late data, not a cancelled call.

## Build and run

```console
$ make plugin && make && make export
$ ./build/bin/example_03_minimal                               # artifacts/basic, 1000 us, 2000 cycles
$ ./build/bin/example_03_minimal artifacts/basic 1000 2000 3   # the same, pinned to cpu 3
```

The arguments are positional: artifact base path, period in microseconds,
cycle count, and the CPU to pin to. Omit the last to stay unpinned.

## Reading the output

Unpinned and unprivileged on this project's development host, the first
lines are:

```text
  [ok  ] harden_malloc: M_TRIM_THRESHOLD=-1 M_MMAP_MAX=0 M_ARENA_MAX=1
  [ok  ] lock_memory: mlockall(MCL_CURRENT|MCL_FUTURE)
  [ok  ] set_realtime_priority: SCHED_FIFO priority 80
```

One `[ok  ]`/`[skip]` line per helper that was attempted; then two summaries — wake-up latency,
the scheduler's and the idle state's share, and call latency, the
computation's — each with percentiles, the two {term}`tail` ratios and a
histogram; then {cpp:func}`~pjrt::rt::describe_environment` for the log, and
`max_residual`, the worst residual the function reported. Exit code 2 means
the residual was not small: the loop's inputs did not reach the function.
{doc}`/background/latency` defines the quantities; {doc}`/guides/measuring`
says what a number needs before it is quoted.

## Making it yours

Replace the artifact and the two input writes; keep the order. When you want
to know *why* a run was late — the host audit, deadline accounting, page-fault
and context-switch counts, the allocation census, a JSON report — that is
{doc}`04-realtime`, the same loop instrumented.
