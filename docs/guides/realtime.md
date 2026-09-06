# Real-time hardening

The helpers in `pjrt::rt` change how reliably the operating system lets a
computation finish on time, not what it computes. Each is independent, returns
a `Status{ok, detail}` instead of throwing, and no-ops where the platform
lacks it — so one binary runs unprivileged on a laptop and hardened on the
target, and says which it got. The host's half of the bargain is audited by
`tools/rt_check.sh`; the process's half is this menu.

## The menu

| Helper | Does | Buys | Needs | If skipped |
|---|---|---|---|---|
| `rt::harden_malloc()` | `mallopt`: never trim the heap, no mmap, one arena | the heap is never handed back and faulted in again mid-call | glibc | a routine call becomes an outlier after a trim |
| `rt::lock_memory()` | `mlockall`, plus a prefault of heap and stack | no page fault or swap-in inside a call | `RLIMIT_MEMLOCK` unlimited | a fault lands inside `call()` |
| `rt::pin_current_thread(cpu)` | thread affinity | no migration; warm caches | the CPU in this thread's mask | migrations show up as spread |
| `rt::corral_xla_threads({cpus})` | moves XLA's named pool threads (`tf_XLAEigen…`) off the loop's core | XLA's pool wake-ups land elsewhere | the `Runtime` to exist already | pools wake on the loop's core |
| `rt::set_realtime_priority(p)` | `SCHED_FIFO` | nothing at normal priority preempts the loop | `CAP_SYS_NICE` or `RLIMIT_RTPRIO`; **last**; a loop that blocks | anything can preempt the loop |
| `rt::describe_environment()` | one paragraph for the log | a number that can be interpreted later | — | hardened and unhardened runs look alike |
| `cjfc::detect_host_env()` | reads isolcpus, nohz_full, governor, THP, SMT, rlimits, throttling, C-state access, loadavg | the audit inside the report | Linux | a number with no provenance |
| `cjfc::choose_cpu()` | picks an isolated, `nohz_full` core and says why | pinning to the right core, or to none | a `HostEnv` | pinned onto a shared core |
| `cjfc::DmaLatencyHold` | holds `/dev/cpu_dma_latency` at 0 while alive | no deep C-state exit between calls | root | the median rises with idle time |
| `cjfc::Rusage` | faults and context switches, before and after | proof that `mlockall` and pinning held | `RUSAGE_THREAD` | a spike cannot be attributed |
| `cjfc::HardeningOptions` | the switches; priority and C-states off by default | explicit opt-in | — | — |
| `cjfc::apply_hardening()` | the six steps above, in the one safe order, one `Step` each | the order rule, enforced | `HostEnv`, `DmaLatencyHold` | priority raised before loading |
| `tools/rt_check.sh` | audits the host, exits non-zero when something is worth fixing | a gate for a measurement run | a shell | numbers without a host audit |

`pjrt::rt` is the library ({doc}`../api/cpp/rt`); `cjfc::` is the example
layer in `examples/common/rt_env.hpp`, written to be copied.

## Applying them

The call site from example 03. Nothing here is fatal: an unprivileged run
reports what it did not get and continues.

```{literalinclude} ../../examples/03_realtime/realtime.cpp
:language: cpp
:start-after: docs: begin rt-harden
:end-before: docs: end rt-harden
```

Three rules hide in the order. The allocator first, so the heap it configures
is the heap the rest of startup grows. `corral_xla_threads` after the
`Runtime`, because XLA's pools are created with the client. Priority last, and
only if the loop blocks: `SCHED_FIFO` does not time-slice against lower
priorities, so a real-time thread that spins starves the machine, and running
unbounded work — loading, compiling a `.mlirbc`, warm-up — at priority 80 is
how a machine stops responding.

## The loop

Example 03's cycle, as it runs:

```{literalinclude} ../../examples/03_realtime/realtime.cpp
:language: cpp
:start-after: docs: begin rt-loop
:end-before: docs: end rt-loop
```

Four rules. Sleep until an **absolute** deadline (`clock_nanosleep` with
`TIMER_ABSTIME`), so wake-up latency does not accumulate into the phase. Never
skip a period: a deadline already in the past returns at once, and the loop
catches up. Nothing in the body allocates, locks or logs — summarize after the
run. An overrun is late data, not a cancelled call; PJRT cannot cancel a
running computation.

## The host

What the helpers cannot do for you. Each row is a setting the host is booted
or configured with; `tools/rt_check.sh` reads them all and says which are
worth fixing.

| Setting | What it prevents |
|---|---|
| `isolcpus`, `nohz_full`, `rcu_nocbs` on the loop's core | other tasks, the timer tick and RCU callbacks landing there |
| `irqaffinity` away from it | device interrupts landing there |
| governor `performance` | the clock dropping between calls |
| deep C-states off — `DmaLatencyHold`, or `/dev/cpu_dma_latency` held at 0 | a slow wake from idle |
| turbo and SMT off | a clock that varies with temperature; a sibling thread stealing the core |
| transparent hugepages `never` | `khugepaged` stalling a faulting thread |
| no swap, or `lock_memory` | a page evicted and faulted back mid-call |
| `RLIMIT_RTPRIO`, `RLIMIT_MEMLOCK` — in a container, `cap_add: [SYS_NICE]` and `ulimits: {rtprio: 99, memlock: -1}` | `SCHED_FIFO` and `mlockall` being refused |

```console
$ tools/rt_check.sh            # the audit
$ tools/rt_check.sh --quiet    # just the verdict line, for a report header
```

Inside a container every row is the host's to set; `docker/compose.yml`
grants what a container can be given, and container timings are relative
signals.

## Reading the report

Example 03 records four distributions and prints them when it stops. Each
answers a different question.

| Line | Question | A bad value means |
|---|---|---|
| call latency | how long `call()` took | `max/p50` above 2 on an idle, tuned host: something preempted the loop |
| cycle time | wake to end of feedback | the loop body, not only the call |
| wake-up latency | how late the sleep returned | C-state exit, timer resolution, or a busy core |
| period jitter (signed) | wake-to-wake minus the period | the scheduler, not XLA — early is as much a defect as late |
| deadline misses | cycles that ended after the next release | one is a bug in the period; many are a bug in the workload |
| page faults | minor and major, in the timed window | `lock_memory` did not take effect, or warm-up was too short |
| context switches | voluntary and involuntary | involuntary, on a pinned `SCHED_FIFO` thread: something else wants that core |
| allocations | the armed census, per call | `self` must be zero; `plugin` is XLA's thunk runtime, reported |

The sign-off targets, and the numbers this project has measured, are on
{doc}`../benchmarks`.

## Deeper

{doc}`../developer/realtime-notes` — every host setting with how to check and
set it, the idle-period experiment, PREEMPT_RT, cyclictest, and
`apply_hardening` in full. {doc}`../developer/measurement` — before quoting a
number. {doc}`../api/cpp/rt` — the library helpers.
