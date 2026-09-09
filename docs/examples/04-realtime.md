# 04 · Real-time loop, instrumented

*Assumes the {doc}`Quickstart </getting-started/quickstart>` only — not
examples 01, 02 or 03. Everything borrowed from them is named in the note.*

`examples/04_realtime` is {doc}`example 03 <03-minimal>` with the measurement
attached: it runs the function example 02 exports on a fixed period with the
hardening applied, records every call and every wake-up, and prints a report
when it stops. An overrun is late data, not a cancelled call — PJRT cannot
cancel a running computation, so the loop never skips a period — and a run
without privileges is still a correct run: every step reports `[ok  ]` or
`[skip]` and the run continues.

:::{note}
**Before the code.** Four namespaces appear. `pjrt::` is the library
({doc}`/api/index`): `Runtime`, `Function`, `LatencyRecorder`, `AllocGuard`.
`cjfc::` — call_jax_from_cpp — is the layer the examples share, under
`examples/common/`, documented on {doc}`/api/cpp/examples` and written to be
copied. `cjfc::workload::` is example 02's exported function seen from C++:
`write_reference` fills an input arena and `feedback` copies one cycle's
outputs into the next cycle's inputs; its inputs and outputs are on
{ref}`the workload reference <cjfc-workload>`. `rt::`, unqualified, is this
example's own flags, printing and `Results` in `support.hpp` — not the
library's `pjrt::rt`.
:::

## What 04 adds to 03

| Added | What for | Optional? |
|---|---|---|
| {cpp:func}`~cjfc::detect_host_env`, {cpp:func}`~cjfc::choose_cpu` | read the host's real-time settings; pick an isolated core and say why | yes — pin by number without it |
| {cpp:func}`~cjfc::apply_hardening` | the helpers in the one safe order, one {cpp:struct}`~cjfc::Step` each, printed | yes — example 03 calls them directly |
| four recorders and deadline counters | call latency, cycle time, wake-up latency, signed period jitter; misses and the worst overrun | measurement |
| warm-up outside the measured window | the faults and lazy initialization a steady-state number must not contain | measurement |
| {cpp:struct}`~cjfc::Rusage` before and after | page faults and context switches in the window — proof that `mlockall` and pinning held | measurement |
| the {term}`allocation census` | `self` must be zero; XLA's own count is reported | measurement; needs the preload |
| the JSON report and `--samples` | every number, with the host and the plugin it came from | reporting |
| the flags | mirror {cpp:struct}`~cjfc::HardeningOptions`: `--cpu`, `--rt-priority`, `--dma-latency`, `--no-mlock`, `--no-malloc-tune`, `--no-corral` | configuration |

None of it changes the loop.

## Startup, in order

`main` runs in this order: flags and the signal handler; the host audit;
the `Runtime`, whose creation starts XLA's pools; the `Function` and the
workload's first inputs; hardening; recorders, guard and loop state; the cold
call and the warm-up; the measured window; the report.

```{literalinclude} ../../examples/04_realtime/realtime.cpp
:language: cpp
:start-after: docs: begin rt-harden
:end-before: docs: end rt-harden
```

| In the output | What it does |
|---|---|
| `harden_malloc` | {cpp:func}`~pjrt::rt::harden_malloc`: `mallopt` — never trim the heap, never serve a block with `mmap`, no new arenas |
| `lock_memory` | {cpp:func}`~pjrt::rt::lock_memory`: `mlockall` plus a prefault of heap and stack; needs `RLIMIT_MEMLOCK` |
| `pin_current_thread` | {cpp:func}`~pjrt::rt::pin_current_thread` to the core {cpp:func}`~cjfc::choose_cpu` picked, or unpinned with the reason; `--cpu N` overrides |
| `corral_xla_threads` | {cpp:func}`~pjrt::rt::corral_xla_threads`: XLA's pool threads onto every other CPU in the mask — hence after the `Runtime` |
| `cpu_dma_latency` | {cpp:class}`~cjfc::DmaLatencyHold`: `/dev/cpu_dma_latency` held at zero for the run; needs root; off by default |
| `set_realtime_priority` | {cpp:func}`~pjrt::rt::set_realtime_priority`: `SCHED_FIFO`, applied last; `CAP_SYS_NICE` or `RLIMIT_RTPRIO`; `--rt-priority 0` opts out |

`env` is the {cpp:struct}`~cjfc::HostEnv` read at startup,
`options.hardening` the {cpp:struct}`~cjfc::HardeningOptions` switches, and
`rt::print_steps` prints one line per step.

## The loop

```{literalinclude} ../../examples/04_realtime/realtime.cpp
:language: cpp
:start-after: docs: begin rt-loop
:end-before: docs: end rt-loop
```

| Name | What it is |
|---|---|
| `s` | `LoopState`, declared just above this region: the `Function`, the workload's {cpp:struct}`~cjfc::workload::Dims`, the `x_ref` input pointer, the four recorders, the deadline counters and the schedule (`target_ns`, `prev_wake_ns`), all resolved before the loop starts |
| {cpp:func}`~cjfc::stopping` | the flag `SIGINT`/`SIGTERM` sets through {cpp:func}`~cjfc::install_stop_handlers`; the report is printed on the way out, not from the handler |
| {cpp:func}`~cjfc::sleep_until`, `EINTR` | an {term}`absolute sleep` on `CLOCK_MONOTONIC`; `EINTR` means a signal arrived, so the loop checks the flag and sleeps again |
| {cpp:func}`~cjfc::now_ns` | `CLOCK_MONOTONIC` in nanoseconds — not the wall clock, which NTP moves |
| `s.rec->….record(…)` | four {cpp:class}`~pjrt::LatencyRecorder`s: a store into capacity reserved before the loop; a full one drops and counts rather than growing |
| {cpp:func}`~cjfc::workload::write_reference` | cycle `k`'s reference trajectory, written straight into the `x_ref` arena |
| `s.function->call()` | execute, one await, one `memcpy` per output: {cpp:func}`~pjrt::Function::call` |
| {cpp:func}`~cjfc::workload::feedback` | this cycle's `x_pred`, `u_opt` and `step_next` outputs into the next cycle's inputs, in place; returns whether the executable's step counter agrees with the loop's |
| `s.counters.observe(…)` | `rt::Deadlines`: a miss when the work ended after the *next* release, and the worst overrun |

## The window and the verdict

```{literalinclude} ../../examples/04_realtime/realtime.cpp
:language: cpp
:start-after: docs: begin alloc-guard
:end-before: docs: end alloc-guard
```

```{literalinclude} ../../examples/04_realtime/realtime.cpp
:language: cpp
:start-after: docs: begin rt-report
:end-before: docs: end rt-report
```

| Name | What it is |
|---|---|
| {cpp:func}`~cjfc::Rusage::now`, `after - before` | `getrusage(RUSAGE_THREAD)`: minor and major faults, voluntary and involuntary context switches, for this thread, over exactly the recorded cycles |
| `guard.arm()` / `disarm()` | {cpp:class}`~pjrt::AllocGuard`: the census a preloaded `malloc_guard.so` keeps; without the preload both are no-ops and the report says so |
| `rt::Results` | the four summaries, the counters and the rusage delta, computed once the loop is over — the first code allowed to allocate; `step_counter_ok()` is whether every cycle's `feedback` check passed |
| {cpp:var}`~cjfc::kExitCorrectness`, {cpp:func}`~cjfc::alloc_gate_exit_code` | the shared exit codes: correctness (2) outranks the allocation gate (3, or 4 when the guard was required but absent) |

The printing and the JSON writer follow, in `support.hpp`.

## Build and run

Needs `make export` (this example loads `artifacts/trajopt`), a plugin and
`make`; for the hardening to take effect, the {term}`rlimits`; for the
census, `build/lib/malloc_guard.so` preloaded.

```console
$ make plugin && make && make export
$ ./build/bin/example_04_realtime --iterations 1000
$ LD_PRELOAD=$PWD/build/lib/malloc_guard.so \
    ./build/bin/example_04_realtime --iterations 1000 --json artifacts/reports/realtime.json
```

`--help` is authoritative, and an unknown flag is refused rather than
ignored. Beyond the hardening flags: `--period-us`, `--iterations` (`0` runs
until `SIGINT`), `--json`, `--samples`, `--alloc-gate off|self|all`,
`--require-guard`.

## Expected output

The hardening block from this project's development host, unprivileged; the
statuses depend on the host, which is the point of printing them:

```text
=== hardening ===
  [ok  ] harden_malloc: M_TRIM_THRESHOLD=-1 M_MMAP_MAX=0 M_ARENA_MAX=1
  [ok  ] lock_memory: mlockall(MCL_CURRENT|MCL_FUTURE)
  [skip] pin_current_thread: no isolated cpus available to this thread; running unpinned. Boot with isolcpus=/nohz_full=, or pass --cpu N to pin anyway
  [ok  ] corral_xla_threads: moved 1 of 1 XLA threads
  [skip] cpu_dma_latency: not requested
  [ok  ] set_realtime_priority: SCHED_FIFO priority 80
```

A `[skip]` on `lock_memory` or `set_realtime_priority` is the usual result on
a stock login shell; `docker/compose.yml` grants both. Reading the rest of the
report is on {doc}`/guides/realtime`.

## Making it yours

Keep `periodic.hpp` and `rt_env.hpp`; replace `workload.hpp` with your own
signature — `check_signature`, `init_inputs`, `write_reference` and
`feedback` are the four things a workload has to say about itself. The same
loop without any of the instrumentation is {doc}`03-minimal`.
