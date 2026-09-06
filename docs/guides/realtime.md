# Real-time hardening

The helpers in `pjrt::rt` do not change what is computed, only how reliably the
operating system lets the computation finish on time. Each is independent,
reports whether it took effect, and no-ops on platforms that do not provide it.

## Why: the outliers are the operating system's

The finding that shaped this project is that **XLA:CPU was never the source of
the jitter**. In-process blocking PJRT from Python, on the exact same module,
showed a max/mean ratio of only 1.14–1.25x. The C++ wrapper was the problem —
it rebuilt device buffers on every call — and once that was fixed the worst
call in 112,000 was 1.72x its median.

What is left is environmental, and it is large. Two measurements make the point
better than any argument:

- The same configuration measured while a bazel build saturated the machine
  reported **p50 2.4x high and max/p50 of 4.4 instead of 1.1**. That is one
  competing workload, on an untuned host.
- Before any of this work, a 4000-call run produced p50 2996 µs and a max of
  **17,279 µs — 5.7x the median**. A single call, once, at nearly six times the
  typical cost. No amount of arithmetic optimization answers that; scheduling
  does.

So the host has to cooperate. The two checklists below are the host's side and
the process's side, in that order, because the second one cannot compensate for
the first.

## The environment checklist

`tools/rt_check.sh` audits all of it, read-only, and exits non-zero when
anything is worth fixing — so it can gate a measurement run. Run it before
trusting any number from a machine, and keep its output next to the numbers.

```console
$ tools/rt_check.sh
$ tools/rt_check.sh --quiet     # just the verdict line, for a report header
```

| Setting | What it prevents | How to check | How to set | Reference |
|---|---|---|---|---|
| `isolcpus=2-5` | Any other runnable task being scheduled onto the loop's core | `cat /sys/devices/system/cpu/isolated` | Kernel command line | [kernel parameters](https://docs.kernel.org/admin-guide/kernel-parameters.html) |
| `nohz_full=2-5` | The periodic timer tick interrupting a core that has one runnable task | `cat /sys/devices/system/cpu/nohz_full` | Kernel command line | [NO_HZ](https://docs.kernel.org/timers/no_hz.html) |
| `rcu_nocbs=2-5` | RCU callbacks running on the loop's core | `grep rcu_nocbs /proc/cmdline` | Kernel command line | [NO_HZ](https://docs.kernel.org/timers/no_hz.html) |
| `irqaffinity=0-1` | Device interrupts landing on the isolated cores | `cat /proc/interrupts` | Kernel command line, plus `/proc/irq/*/smp_affinity_list` for anything that arrives later | [kernel parameters](https://docs.kernel.org/admin-guide/kernel-parameters.html) |
| Governor `performance` | The clock dropping between calls and taking microseconds to ramp back | `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` | `cpupower frequency-set -g performance` | — |
| Deep C-states off | A core in a deep idle state taking tens of microseconds to wake | `cat /sys/devices/system/cpu/cpu*/cpuidle/state*/disable` | Hold `/dev/cpu_dma_latency` **open** with a 32-bit `0` written to it; the constraint lasts only while the descriptor is held | — |
| Turbo off | The clock varying with package temperature, which makes runs incomparable | `cat /sys/devices/system/cpu/intel_pstate/no_turbo` | Write `1`: peak speed traded for consistency | — |
| SMT off | A sibling hyperthread stealing execution resources from the loop | `cat /sys/devices/system/cpu/smt/control` | `echo off > /sys/devices/system/cpu/smt/control` | — |
| THP `never` or `madvise` | `khugepaged` stalling a faulting thread while it compacts | `cat /sys/kernel/mm/transparent_hugepage/enabled` | Write `never`, or boot `transparent_hugepage=never` | [kernel parameters](https://docs.kernel.org/admin-guide/kernel-parameters.html) |
| No swap, or `mlockall` | A page of the working set being evicted and faulted back in mid-call | `free -h`; `grep VmSwap /proc/self/status` | `swapoff -a`, and `pjrt::rt::lock_memory()` | [mlockall(2)](https://man7.org/linux/man-pages/man2/mlockall.2.html) |
| `RLIMIT_RTPRIO` > 0 | `SCHED_FIFO` being refused | `ulimit -r` | `/etc/security/limits.conf`, or `--ulimit rtprio=99` | [sched(7)](https://man7.org/linux/man-pages/man7/sched.7.html) |
| `RLIMIT_MEMLOCK` unlimited | `mlockall` being refused | `ulimit -l` | `/etc/security/limits.conf`, or `--ulimit memlock=-1` | [mlockall(2)](https://man7.org/linux/man-pages/man2/mlockall.2.html) |
| `sched_rt_runtime_us` | A busy-waiting real-time task being throttled off the CPU for the rest of the period | `cat /proc/sys/kernel/sched_rt_runtime_us` | `-1` disables throttling; see the note below | [sched(7)](https://man7.org/linux/man-pages/man7/sched.7.html) |

:::{note}
**Real-time throttling has an edge, not a slope.** The default 950000 out of a
1000000 µs period caps every `SCHED_FIFO` task at 95% of each period. A loop
that sleeps or blocks between calls never approaches it. A loop that busy-waits
does, and is then descheduled for the remaining 50 ms — which arrives as one
spectacular outlier rather than as gradual degradation.
:::

### Idling between calls is not a small effect

Of everything in the table above, the settings that govern what a core does
while it is *idle* are the ones most likely to surprise you, because they do
not look like jitter. They move the *median*.

Measured on an idle Intel i9-14900HX, running the trajectory-optimisation
example against the same artifact, pinned to the same core, with real-time
priority and memory locking in effect, `scaling_governor` at `powersave`
throughout and `/dev/cpu_dma_latency` not held. The only variable is how much
of each period the loop spends computing:

| Period | Duty cycle | min | p50 |
|---|---|---|---|
| 3 ms | ~65% | 1866 µs | **2027 µs** |
| 10 ms | ~20% | 1854 µs | **5196 µs** |

The same work, on the same core, takes 2.6 times longer at 100 Hz than at
333 Hz. Nothing is contended and nothing is preempted. The core simply idles
for 8 ms of every 10, and arrives at the next period in a worse state to do
the work.

**Which idle-state setting is responsible was not isolated here**, and it is
worth being precise about that. Two mechanisms scale with idle time in exactly
the same way: the frequency governor clocking the core down, and the latency of
leaving a deep C-state. This experiment varied only the period, with both
settings fixed, so it measures their combined cost and attributes it to
neither. To separate them, run the loop at both periods with
`scaling_governor` at `performance`, then again holding `/dev/cpu_dma_latency`
open at 0, and see which one closes the gap. Both are in the checklist above
for the same reason.

Two things follow. The first is that a control loop is close to the worst case
for power management of any kind: it is periodic, it is mostly idle, and it
needs the core at full speed exactly when it wakes. The second is a measurement trap — a
back-to-back benchmark keeps the core busy and therefore boosted, so it
reports the *compute* cost, while the loop at its real period reports what the
application actually experiences. Both numbers above are honest; they answer
different questions. Quote the one that matches how the code will run, and say
which it is.

The `min` column is the tell. When `min` is far below `p50` and the loop is
periodic on an otherwise idle machine, suspect the clock before you suspect
contention.

### PREEMPT_RT

A `PREEMPT_RT` kernel is the difference between "usually fast" and "bounded".
It makes almost all kernel sections preemptible, threads the interrupt
handlers, and gives priority inheritance on kernel locks. Without it, an
un-preemptible kernel section can hold the loop off for as long as it takes,
and nothing in user space can do anything about that.

`rt_check.sh` reports the kernel it finds. `PREEMPT_DYNAMIC` booted with
`preempt=full` is closer than the default, and is still not the same thing: the
priority-inheritance and threaded-IRQ guarantees are the part that is missing.

This project's numbers were **not** taken on a `PREEMPT_RT` kernel, which is
one of the reasons they are described as relative signals.

## The process checklist

In the order the real-time example applies them:

```{literalinclude} ../../examples/common/rt_env.hpp
:language: cpp
:start-after: docs: begin rt-harden-impl
:end-before: docs: end rt-harden-impl
```

| Step | Syscall | Needs | Failure mode |
|---|---|---|---|
| `harden_malloc()` | `mallopt(M_TRIM_THRESHOLD, -1)`, `M_MMAP_MAX=0`, `M_ARENA_MAX=1` | nothing | Returns `"mallopt had no effect (non-glibc allocator?)"`. The heap keeps being trimmed, so the next allocation faults it back in — a routine call becomes an outlier. |
| `lock_memory()` | `mlockall(MCL_CURRENT\|MCL_FUTURE)`, then a prefault of 64 MB | `RLIMIT_MEMLOCK` | `mlockall: Cannot allocate memory` or `Operation not permitted`. Pages can be swapped, and the fault lands inside a call. |
| `pin_current_thread(cpu)` | `pthread_setaffinity_np` | nothing (the cpu must exist) | `pthread_setaffinity_np: Invalid argument`. The thread migrates between cores and loses its caches. |
| `corral_xla_threads({...})` | walks `/proc/self/task`, `pthread_setaffinity_np` per thread | `/proc` readable | Reports what it could not move. XLA's pools (`XLAEigen*`, `XLAPjRtCpuClient*`) may wake on the loop's core. Call it **after** the `Runtime` exists — that is when the pools are created. |
| `set_realtime_priority(80)` | `pthread_setschedparam(SCHED_FIFO)` | `CAP_SYS_NICE` or `RLIMIT_RTPRIO` | `pthread_setschedparam(SCHED_FIFO) -- needs CAP_SYS_NICE: Operation not permitted`. The loop runs at normal priority and any other task can preempt it. |

Each returns a `pjrt::rt::Status{ok, detail}` rather than throwing, so a program
can report what it got and continue. That matters: the same binary has to run
unprivileged on a developer's laptop and hardened on the target.

:::{warning}
**`set_realtime_priority` goes last.** Everything before it — loading the
artifact, compiling a `.mlirbc`, faulting in arenas, warming up — is unbounded
work, and running unbounded work at `SCHED_FIFO` priority 80 is how a machine
stops responding.

For the same reason: **a real-time thread that spins forever starves the
machine.** `SCHED_FIFO` does not time-slice against lower priorities. The loop
must block — on a timer, on a queue — and the computation it runs must be
bounded. `PJRT cannot cancel a running CPU computation`, so an overrun means
late data, not a cancelled call.
:::

## What example 03 reports

`examples/03_realtime` applies each step, prints whether it took effect, runs
the function on a fixed period, and prints a latency summary at the end.

```{literalinclude} ../../examples/03_realtime/realtime.cpp
:language: cpp
:start-after: docs: begin rt-report
:end-before: docs: end rt-report
```

Run unprivileged, `mlockall` and `SCHED_FIFO` report `Operation not permitted`
and the run continues at normal priority. That is the expected result outside a
container that grants `CAP_SYS_NICE` and an unlimited `memlock`, and it is why
the helpers report rather than throw.

Two distributions come out of it, and they are different questions:

- **Call latency** — how long `call()` took. This is what the tail targets are
  about.
- **Period jitter** — scheduled time minus actual time, which is *negative*
  whenever a cycle runs early. The recorder holds signed nanoseconds precisely
  so that is representable.

How to read the summary:

{.results}
| Statistic | Target | What it means |
|---|---|---|
| `p50` | at or below the baseline | The typical call. Not the objective, but a regression here is still a regression. |
| `p99.9 / p50` | ≤ 1.3 | One call in a thousand relative to a typical one. This is the number the design is for. |
| `max / p50` | ≤ 2.0 | The worst call in the run. Above 2 on an idle, tuned host means something preempted the loop. |
| absolute max | < 6 ms | The deadline the workload was sized against. |
| wrapper allocations | 0 | `AllocGuard::allocs_self()`, over the steady-state window only. |

Those are the sign-off targets on a native x86-64 host. The measured numbers so
far are from an aarch64 container with no `SCHED_FIFO` and no core pinning; see
{doc}`../benchmarks` for what that does and does not license you to conclude.

## Containers

Two of the five steps need privileges a default container does not have.
`docker/compose.yml` grants them:

```yaml
cap_add:
  - SYS_NICE          # sched_setscheduler(SCHED_FIFO)
ulimits:
  rtprio: 99          # the priority itself
  memlock: -1         # mlockall
```

Without those, `rt_check.sh` says so and the real-time example falls back to
plain scheduling.

:::{caution}
Everything in the environment checklist is the **host's** to set. Inside a
container, `rt_check.sh` is reading the host's governor, `isolcpus`, `nohz_full`
and C-states, and can change none of them. A container can be given cores with
`cpuset`, but it cannot be given a tuned kernel.

**Container timings are relative signals.** They are good enough to compare two
configurations against each other in the same container, and not good enough to
quote as absolute latency.
:::

## The cyclictest cross-check

Before blaming the library, measure what the machine can do at all:

```console
$ cyclictest --mlockall --priority=80 --interval=1000 --distance=0 -t1 -a2
```

That reports the wake-up latency of a thread that asks to be woken on a fixed
period, and nothing more. **If `cyclictest` max on the isolated core is 200 µs,
no library setting gets you below that** — it is the floor the scheduler
imposes, and every call this project makes sits on top of it.

Run it before the tuning and after, on the same core the loop will use. The
difference is what the environment checklist bought, and the remaining number
is the budget everything else has to fit inside.
[cyclictest](https://wiki.linuxfoundation.org/realtime/documentation/howto/tools/cyclictest/start)
is documented by the Linux Foundation's real-time wiki.
