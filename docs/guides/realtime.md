# Real-time hardening

*Assumes the {doc}`Quickstart </getting-started/quickstart>` and a loop that
calls ({doc}`calling`). Linux terms — C-states, `mlockall`, isolation — are
one paragraph each in {doc}`/background/realtime-linux`; `cjfc::` is
{doc}`the example layer </api/cpp/examples>`.*

The helpers in `pjrt::rt` change *when* a computation finishes, not *what* it
computes. Each is independent, returns a {cpp:struct}`~pjrt::rt::Status`
instead of throwing, and is a no-op that says so where the platform lacks it,
so one binary runs unprivileged on a laptop and hardened on the target. An
unprivileged run is a correct run; it is just not one to quote tail numbers
from. Three names appear below: `pjrt::rt::` is the library, `cjfc::` is the
example layer, written to be copied, and the host's half of the bargain is
`tools/rt_check.sh`.

## What goes wrong

Each row is one way a bounded computation is late, and what removes it.

| What goes wrong | Why | What removes it |
|---|---|---|
| A page is faulted in mid-call | memory is mapped lazily and can be swapped out; the first touch since warm-up, or a page evicted since, is a {term}`page fault` inside `call()` | {cpp:func}`~pjrt::rt::lock_memory`: `mlockall` plus a prefault; needs `RLIMIT_MEMLOCK` |
| A routine call becomes an outlier after a quiet stretch | glibc trims the heap and serves large blocks with `mmap`; the next allocation faults it all back in | {cpp:func}`~pjrt::rt::harden_malloc`: `mallopt` — no trim, no mmap, no new arenas |
| Spread with no pattern; involuntary {term}`context switches <context switch>` | the scheduler migrates the thread and shares its core; caches go cold | {cpp:func}`~pjrt::rt::pin_current_thread` onto an {term}`isolated CPU`; {cpp:func}`~cjfc::choose_cpu` picks one and says why, or declines |
| Wake-ups on the loop's core from inside the process | creating the `Runtime` starts XLA's {term}`thread pool` | {cpp:func}`~pjrt::rt::corral_xla_threads`, after the `Runtime` exists |
| The median rises with idle time | an idle core drops into a deep {term}`C-state` and is clocked down; it wakes slow and cold | {cpp:class}`~cjfc::DmaLatencyHold` (needs root) and {term}`governor` `performance` on the host |
| Anything preempts the loop | normal priority is time-sliced against everything else | {cpp:func}`~pjrt::rt::set_realtime_priority`: {term}`SCHED_FIFO`, last, and only for a loop that blocks |
| The phase drifts by the jitter being measured | a relative sleep adds each wake-up's lateness to the next period | an {term}`absolute sleep`: `sleep_until` in example 03, {cpp:func}`~cjfc::sleep_until` in the example layer |
| A spike cannot be attributed | nothing was recorded about the host or the thread | {cpp:func}`~pjrt::rt::describe_environment` and {cpp:func}`~cjfc::detect_host_env` in the log; {cpp:struct}`~cjfc::Rusage` around the window |

## The minimum, in your own code

Example 03 is a hardened loop with nothing else in it. Its setup, in the one
order that is safe:

```{literalinclude} ../../examples/03_minimal/minimal.cpp
:language: cpp
:start-after: docs: begin minimal-setup
:end-before: docs: end minimal-setup
```

Three rules hide in that order. `harden_malloc` first, before the `Runtime`:
from then on, nothing the process frees is handed back to the kernel.
`corral_xla_threads` after the `Runtime`, because XLA's pools are created with
the client. Priority last, and only if the loop blocks: `SCHED_FIFO` does not
time-slice against lower priorities, so unbounded work — loading, compiling a
`.mlirbc`, warm-up — at priority 80 is how a machine stops responding.
{cpp:func}`~pjrt::rt::set_realtime_priority` promotes only the calling
thread, after the `Runtime` exists; `chrt -f` on the launcher puts the whole
process under `SCHED_FIFO` before anything is loaded, so every thread XLA
starts inherits it and the load and warm-up run at real-time priority. That
is why the examples prefer the in-process call.

## The loop

```{literalinclude} ../../examples/03_minimal/minimal.cpp
:language: cpp
:start-after: docs: begin minimal-loop
:end-before: docs: end minimal-loop
```

Four rules. Sleep until an absolute deadline, so wake-up latency does not
accumulate into the phase. Never skip a period: a deadline already in the past
returns at once, and the loop catches up. Nothing in the body allocates, locks
or logs — summarize after the run. An overrun is late data, not a cancelled
call: PJRT cannot cancel a running computation.

## What example 04 adds

The same loop, instrumented: a host audit and {cpp:func}`~cjfc::choose_cpu`,
{cpp:func}`~cjfc::apply_hardening` returning one {cpp:struct}`~cjfc::Step`
per helper, four recorders and deadline counters, warm-up outside the measured
window, {cpp:struct}`~cjfc::Rusage` deltas, the {term}`allocation census`,
and a JSON report. None of it changes the loop; {doc}`/examples/04-realtime`
says what each piece is for.

## The host

What the helpers cannot do for you. `tools/rt_check.sh` reads every row and
says which are worth fixing; {doc}`/developer/realtime-notes` has how to check
and set each one.

| Setting | What it prevents |
|---|---|
| `isolcpus`, `nohz_full`, `rcu_nocbs` on the loop's core; `irqaffinity` away from it | other tasks, the timer tick, RCU callbacks and device interrupts landing there |
| governor `performance`; deep C-states off (`DmaLatencyHold`, or `/dev/cpu_dma_latency` held at zero); turbo off | the clock dropping between calls; a slow wake from idle; a clock that varies with temperature |
| SMT off | a sibling thread stealing the core |
| transparent hugepages `never`; no swap, or `lock_memory` | `khugepaged` stalling a faulting thread; a page evicted and faulted back mid-call |
| `RLIMIT_RTPRIO`, `RLIMIT_MEMLOCK` — in a container, `cap_add: [SYS_NICE]` and `ulimits: {rtprio: 99, memlock: -1}` | `SCHED_FIFO` and `mlockall` being refused |

```console
$ tools/rt_check.sh            # the audit
$ tools/rt_check.sh --quiet    # just the verdict line, for a report header
```

Inside a container every row is the host's to set; `docker/compose.yml`
grants what a container can be given.

## Reading the report

Example 04 records four distributions and prints them when it stops. The
quantities are defined in {doc}`/background/latency`.

| Line | Question | A bad value means |
|---|---|---|
| call latency | how long `call()` took | `max/p50` well above one on an idle, tuned host: something preempted the loop |
| cycle time | wake to end of feedback | the loop body, not only the call |
| wake-up latency | how late the sleep returned | C-state exit, timer resolution, or a busy core |
| period jitter (signed) | wake-to-wake minus the period | the scheduler, not XLA — early is as much a defect as late |
| deadline misses | cycles that ended after the next release | one is a bug in the period; many are a bug in the workload |
| page faults | minor and major, in the timed window | `lock_memory` did not take effect, or warm-up was too short |
| context switches | voluntary and involuntary | involuntary, on a pinned `SCHED_FIFO` thread: something else wants that core |
| allocations | the armed census, per call | `self` must be zero; `plugin` is XLA's thunk runtime, reported |

The sign-off targets, and the numbers this project has measured, are on
{doc}`/benchmarks`.

## Deeper

{doc}`/background/realtime-linux` — the mechanisms.
{doc}`/developer/realtime-notes` — every host setting with how to check and
set it, the idle-period experiment, PREEMPT_RT, cyclictest, and
`apply_hardening` in full. {doc}`/developer/measurement` — before quoting a
number. {doc}`/api/cpp/rt` and {doc}`/api/cpp/examples` — the helpers.
