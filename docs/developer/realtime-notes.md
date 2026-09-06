# Real-time notes

The host settings, the experiments and the kernel details behind
{doc}`../guides/realtime`. That page is the menu; this one is the reasoning.

## The host, setting by setting

`tools/rt_check.sh` audits all of it, read-only, and exits non-zero when
anything is worth fixing — so it can gate a measurement run.

| Setting | What it prevents | How to check | How to set | Reference |
|---|---|---|---|---|
| `isolcpus=2-5` | Any other runnable task being scheduled onto the loop's core | `cat /sys/devices/system/cpu/isolated` | Kernel command line | [kernel parameters](https://docs.kernel.org/admin-guide/kernel-parameters.html) |
| `nohz_full=2-5` | The periodic timer tick interrupting a core that has one runnable task | `cat /sys/devices/system/cpu/nohz_full` | Kernel command line | [NO_HZ](https://docs.kernel.org/timers/no_hz.html) |
| `rcu_nocbs=2-5` | RCU callbacks running on the loop's core | `grep rcu_nocbs /proc/cmdline` | Kernel command line | [NO_HZ](https://docs.kernel.org/timers/no_hz.html) |
| `irqaffinity=0-1` | Device interrupts landing on the isolated cores | `cat /proc/interrupts` | Kernel command line, plus `/proc/irq/*/smp_affinity_list` for anything that arrives later | [kernel parameters](https://docs.kernel.org/admin-guide/kernel-parameters.html) |
| Governor `performance` | The clock dropping between calls and taking time to ramp back | `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` | `cpupower frequency-set -g performance` | — |
| Deep C-states off | A core in a deep idle state taking tens of microseconds to wake | `cat /sys/devices/system/cpu/cpu*/cpuidle/state*/disable` | Hold `/dev/cpu_dma_latency` **open** with a 32-bit `0` written to it; the constraint lasts only while the descriptor is held | — |
| Turbo off | The clock varying with package temperature, which makes runs incomparable | `cat /sys/devices/system/cpu/intel_pstate/no_turbo` | Write `1`: peak speed traded for consistency | — |
| SMT off | A sibling hyperthread stealing execution resources from the loop | `cat /sys/devices/system/cpu/smt/control` | `echo off > /sys/devices/system/cpu/smt/control` | — |
| THP `never` or `madvise` | `khugepaged` stalling a faulting thread while it compacts | `cat /sys/kernel/mm/transparent_hugepage/enabled` | Write `never`, or boot `transparent_hugepage=never` | [kernel parameters](https://docs.kernel.org/admin-guide/kernel-parameters.html) |
| No swap, or `mlockall` | A page of the working set being evicted and faulted back in mid-call | `free -h`; `grep VmSwap /proc/self/status` | `swapoff -a`, and `pjrt::rt::lock_memory()` | [mlockall(2)](https://man7.org/linux/man-pages/man2/mlockall.2.html) |
| `RLIMIT_RTPRIO` > 0 | `SCHED_FIFO` being refused | `ulimit -r` | `/etc/security/limits.conf`, or `--ulimit rtprio=99` | [sched(7)](https://man7.org/linux/man-pages/man7/sched.7.html) |
| `RLIMIT_MEMLOCK` unlimited | `mlockall` being refused | `ulimit -l` | `/etc/security/limits.conf`, or `--ulimit memlock=-1` | [mlockall(2)](https://man7.org/linux/man-pages/man2/mlockall.2.html) |
| `sched_rt_runtime_us` | A busy-waiting real-time task being throttled off the CPU for the rest of the period | `cat /proc/sys/kernel/sched_rt_runtime_us` | `-1` disables throttling; see below | [sched(7)](https://man7.org/linux/man-pages/man7/sched.7.html) |

The script checks twelve of these; `irqaffinity` is the one it does not, and
it also reports `rcu_nocbs`, turbo and swap in more detail than the table.

:::{note}
**Real-time throttling has an edge, not a slope.** The default `950000` out of
a `1000000` µs period caps every `SCHED_FIFO` task at 95% of each period. A
loop that sleeps or blocks between calls never approaches it. A loop that
busy-waits does, and is then descheduled for the remaining 5% of the period —
which arrives as one spectacular outlier rather than as gradual degradation.
:::

Inside a container, everything in the table is the **host's** to set.
`rt_check.sh` reads the host's governor, `isolcpus`, `nohz_full` and C-states
and can change none of them; `docker/compose.yml` grants what a container
*can* be given — `cap_add: [SYS_NICE]` and the `rtprio` / `memlock` ulimits —
and a `cpuset`. Container timings are relative signals.

## Idling between calls is not a small effect

Of everything in the table, the settings that govern what a core does while it
is *idle* are the ones most likely to surprise, because they do not look like
jitter: they move the **median**.

The experiment: the trajectory-optimisation example against one artifact, on
one pinned core, with real-time priority and memory locking in effect, the
governor on `powersave` throughout and `/dev/cpu_dma_latency` not held. The
only variable is the period — how much of each period the loop spends
computing. At a short period the core stays busy and the median is the
compute cost; at a long period the core idles for most of each cycle and
arrives at the next release in a worse state to do the same work, and the
median is a multiple of the compute cost. The table, with the host named, is
on {doc}`../benchmarks`.

**Which idle-state mechanism is responsible was not isolated.** Two scale with
idle time in the same way: the frequency governor clocking the core down, and
the latency of leaving a deep C-state. The experiment varied only the period,
with both fixed, so it measures their combined cost and attributes it to
neither. To separate them: run both periods with the governor on
`performance`, then again holding `/dev/cpu_dma_latency` open at 0, and see
which one closes the gap.

Two things follow. A control loop is close to the worst case for power
management of any kind: periodic, mostly idle, and needing the core at full
speed exactly when it wakes. And a back-to-back benchmark keeps the core busy
and therefore boosted, so it reports the *compute* cost, while the loop at its
real period reports what the application experiences. Both numbers are honest;
they answer different questions. Quote the one that matches how the code will
run, and say which it is. The `min` column is the tell: when `min` is far
below `p50` on an otherwise idle machine, suspect the clock before you suspect
contention.

## PREEMPT_RT

A `PREEMPT_RT` kernel is the difference between "usually fast" and "bounded".
It makes almost all kernel sections preemptible, threads the interrupt
handlers, and gives priority inheritance on kernel locks. Without it, an
un-preemptible kernel section can hold the loop off for as long as it takes,
and nothing in user space can do anything about that.

`rt_check.sh` reports the kernel it finds. `PREEMPT_DYNAMIC` booted with
`preempt=full` is closer than the default and still not the same thing: the
priority-inheritance and threaded-IRQ guarantees are the part that is missing.
None of this project's numbers were taken on a `PREEMPT_RT` kernel, which is
one reason they are described as relative signals.

## The cyclictest floor

Before blaming the library, measure what the machine can do at all:

```console
$ cyclictest --mlockall --priority=80 --interval=1000 --distance=0 -t1 -a2
```

That reports the wake-up latency of a thread that asks to be woken on a fixed
period, and nothing more. Whatever its maximum is on the isolated core, no
library setting gets below it: it is the floor the scheduler imposes, and every
call this project makes sits on top of it. Run it before the tuning and after,
on the same core the loop will use; the difference is what the host checklist
bought, and the remaining number is the budget everything else has to fit
inside. [cyclictest](https://wiki.linuxfoundation.org/realtime/documentation/howto/tools/cyclictest/start)
is documented by the Linux Foundation's real-time wiki.

## The sleep

Example 03 sleeps until an absolute `CLOCK_MONOTONIC` time, not for a
relative interval, so wake-up latency does not accumulate into the phase and
a deadline already in the past returns at once:

```{literalinclude} ../../examples/common/periodic.hpp
:language: cpp
:start-after: docs: begin sleep-until
:end-before: docs: end sleep-until
```

## `apply_hardening`, in full

The example layer applies the six steps in the one order that is safe, and
reports each rather than failing:

```{literalinclude} ../../examples/common/rt_env.hpp
:language: cpp
:start-after: docs: begin rt-harden-impl
:end-before: docs: end rt-harden-impl
```

The order is not arbitrary. `harden_malloc` first, so the heap it configures
is the heap the rest of startup grows. `lock_memory` next, which prefaults and
locks what exists by then. `pin_current_thread`, so the loop owns one core.
`corral_xla_threads` after the `Runtime` exists, because XLA's pools are
created with the client. `cpu_dma_latency`, held for the process lifetime.
`set_realtime_priority` **last**, so that loading, warm-up and every
allocation that comes with them never run at real-time priority, where a long
operation would starve the rest of the machine.

For the same reason, a real-time thread that spins forever starves the
machine: `SCHED_FIFO` does not time-slice against lower priorities. The loop
must block — on a timer, on a queue — and the computation it runs must be
bounded. PJRT cannot cancel a running CPU computation, so an overrun means late
data, not a cancelled call.
