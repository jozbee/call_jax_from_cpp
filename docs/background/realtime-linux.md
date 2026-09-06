# Real-time on Linux

*Assumes you know what `SCHED_FIFO` is and why a real-time kernel exists. This
page is the rest: the mechanisms that make a bounded computation late, one
paragraph each, with a link to the authoritative page. What this project does
about each is on {doc}`/guides/realtime`.*

## What "real-time" means here

Not fast: bounded. The figure of merit is the worst call relative to a typical
one — the {term}`tail` ratios `max/p50` and `p99.9/p50` — over a run long
enough to contain the rare event. A loop that is late once a minute has a
scheduling problem, and no average will show it.
[Real-time computing](https://en.wikipedia.org/wiki/Real-time_computing).

## Scheduling

{term}`SCHED_FIFO` and `SCHED_RR` run a thread ahead of every normal-class
thread until it blocks, at a priority from one to ninety-nine. Asking for one
needs `CAP_SYS_NICE` or a non-zero `RLIMIT_RTPRIO`, and by default the kernel
throttles the real-time class (`sched_rt_runtime_us`) so a runaway thread
cannot take the machine — which is exactly what a spinning real-time thread
does on a host where throttling is off.
[sched(7)](https://man7.org/linux/man-pages/man7/sched.7.html),
[real-time group scheduling](https://docs.kernel.org/scheduler/sched-rt-group.html),
[capabilities(7)](https://man7.org/linux/man-pages/man7/capabilities.7.html).

## Memory

Pages are mapped lazily. The first touch of a page is a *minor*
{term}`page fault` — the kernel finds a frame and maps it — and a page that
was swapped out returns through a *major* fault, which is a disk read. Either
one inside a call is an outlier. `mlockall(MCL_CURRENT | MCL_FUTURE)` pins
every page of the process, and every page mapped later, in RAM; it does not
touch pages that are reserved but not yet mapped, which is why a prefault
(grow the heap and the stack, write to them) goes with it. `RLIMIT_MEMLOCK`
bounds how much may be locked.
[mlockall(2)](https://man7.org/linux/man-pages/man2/mlockall.2.html);
[getrusage(2)](https://man7.org/linux/man-pages/man2/getrusage.2.html) for the
fault counters.

The allocator has habits of its own. glibc's `malloc` hands the top of the
heap back to the kernel when enough of it is free, serves large requests with
a fresh `mmap` and unmaps them on `free`, and grows per-thread arenas. Each of
those returns memory that the next call has to fault in again; `mallopt` turns
all three off.
[mallopt(3)](https://man7.org/linux/man-pages/man3/mallopt.3.html),
[glibc malloc tunables](https://sourceware.org/glibc/manual/latest/html_node/Malloc-Tunable-Parameters.html).
Transparent hugepages add a kernel thread, `khugepaged`, that can stall a
faulting thread while it compacts memory.
[Transparent hugepages](https://docs.kernel.org/admin-guide/mm/transhuge.html).

## The core

Affinity binds a thread to a set of CPUs. Without it the scheduler migrates
threads to balance load, and a migrated thread starts on cold caches.
`isolcpus=` removes CPUs from load balancing altogether, so nothing lands on
an {term}`isolated CPU` unless pinned there; {term}`nohz_full` stops the
periodic timer tick on a core with a single runnable task; `rcu_nocbs=` moves
RCU callbacks off it. Device interrupts go where `irqaffinity=` and
`/proc/irq/*/smp_affinity_list` send them. A hyperthread sibling shares the
core's execution units, so with SMT on, another thread can slow the loop from
inside the core.
[kernel parameters](https://docs.kernel.org/admin-guide/kernel-parameters.html),
[NO_HZ](https://docs.kernel.org/timers/no_hz.html),
[IRQ affinity](https://docs.kernel.org/core-api/irq/irq-affinity.html),
[sched_setaffinity(2)](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html).

## Idle and frequency

An idle core enters a {term}`C-state`; the deeper the state, the longer the
exit, so a sleep between two calls can cost the next call tens of
microseconds before its first instruction runs. Any process may hold
{term}`cpu_dma_latency` open with a zero written to it, and the constraint
keeps every core out of deep states for exactly as long as the descriptor is
open. The frequency {term}`governor` decides the clock: `powersave` lets it
fall between calls and ramp back during one, `performance` holds it up. Turbo
makes the clock a function of package temperature, which makes two runs
incomparable.
[CPU idle](https://docs.kernel.org/admin-guide/pm/cpuidle.html),
[PM QoS](https://docs.kernel.org/power/pm_qos_interface.html),
[cpufreq](https://docs.kernel.org/admin-guide/pm/cpufreq.html).

## Timers

A periodic loop sleeps until an *absolute* time on `CLOCK_MONOTONIC`
(`clock_nanosleep` with `TIMER_ABSTIME`). A relative sleep of one period adds
each wake-up's lateness to the next deadline, so the phase drifts by the very
jitter being measured; an {term}`absolute sleep` does not, and a deadline
already in the past returns at once, which is how a loop catches up instead of
skipping a period. The wall clock is the wrong clock: NTP moves it.
[clock_nanosleep(2)](https://man7.org/linux/man-pages/man2/clock_nanosleep.2.html),
[time(7)](https://man7.org/linux/man-pages/man7/time.7.html).

## PREEMPT_RT

{term}`PREEMPT_RT`, in the mainline kernel since 6.12, makes almost all kernel
code preemptible, so a real-time thread waits on a kernel path for a bounded
time rather than for however long a lock holder takes. It bounds the kernel's
contribution and does nothing about the allocator, the pools or the C-states
above.
[The Linux Foundation real-time wiki](https://wiki.linuxfoundation.org/realtime/start).

## Where these meet this project

Each mechanism above has a helper or a host setting that removes it:
{doc}`/guides/realtime` maps them one to one, and
{doc}`/developer/realtime-notes` lists every host setting with how to check
and set it.
