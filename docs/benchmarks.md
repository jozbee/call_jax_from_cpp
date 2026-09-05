# Benchmarks

The figure of merit is the tail: p99.9 relative to p50, and the worst call in a
long campaign. An average latency is not the objective and is not reported as
one.

:::{important}
This campaign predates the examples that ship in this tree and is **not
reproducible from it**. The fixture it used has been removed along with all the
MPC-specific code. For numbers you can reproduce today, run
{doc}`examples/02_trajopt <examples/02-trajopt>` on your own machine. Do not
put results from the two in the same table.
:::

## The comparison

One step of a nonlinear MPC controller: 16 inputs and 14 outputs, all rank-1
float64, about 10.5k StableHLO ops. A synthetic fixed-cost twin of the same
signature — no data-dependent control flow — served as the control, so that a
change could be attributed to the call path rather than to the workload's own
behaviour.

aarch64 devcontainer, machine otherwise idle, 28 runs × 4000 calls per API =
**112,000 calls each**, interleaved.

{.results}
| | legacy (per-call buffers) | `Runtime`/`Function` |
|---|---|---|
| median p50 | 4841.8 µs | **3979.4 µs** (−18%) |
| median p99.9 | 5416.9 µs | **4416.5 µs** |
| worst single call | 20168.7 µs | **6832.0 µs** |
| worst max/p50 | 4.198 | **1.719** |
| runs with a >2x outlier | 2 of 28 | **0 of 28** |
| allocations/call (ours) | ~520 | **0** |
| allocations/call (inside XLA) | ~15,400 | ~15,400 |

For context, before any of this work: 4000 MPC steps gave mean 3016 µs, p50
2996, p99 3516, p99.9 4665, and a **max of 17279 µs — 5.7x the median**. That
shape, a good median with a rare five-fold spike, is the problem the project
was started to fix.

## What moved the numbers

**Persistent zero-copy input buffers.** Roughly 18% off the median on the MPC
workload and roughly 16% on the synthetic twin. The agreement between the two
is what makes the attribution credible: a win that appears only on the workload
with data-dependent control flow would more likely be a change in that
workload's behaviour than in the call path. Removing the per-call
host-to-device copies also removed the per-call allocation and event churn that
produced the large spikes.

Zero copy needs the caller's memory aligned to at least `xla::cpu::MinAlign()`.
Below that XLA falls back to copying and says nothing, so the whole benefit
disappears without a diagnostic — which is why the runtime owns 64-byte-aligned
arenas instead of accepting an arbitrary caller pointer.

**Inline synchronous execution.** `asynchronous=false`, reachable only as a
create option, removes the hand-off to the dispatch thread pool.
`PJRT_ExecuteOptions` has no execution-mode field, so there is no other route.

## What did not move them

**Plugin build mode.** `-c opt` against bazel's default `fastbuild` moved p50
by 0.2%: 4718.7 µs versus 4728.4 µs. The compute kernels are LLVM-compiled at
*export* time and embedded in the artifact; the plugin only orchestrates. Do not
repeat this experiment — a 30-to-60-minute bazel build is off the critical path.

**The remaining allocations.** Roughly 15,400 allocations per call still happen
inside XLA's thunk runtime, about one per StableHLO op. Only ~520 were ever the
wrapper's own, and those are gone. Reaching the rest means a pooling allocator
behind `CpuClientOptions::allocator`, which the PJRT C API does not expose — a
third fork patch, justified only if real hardware still shows a tail.

## How much of this to believe

This section is not optional, and it is the part to quote alongside the table.

- **The magnitude difference is solid.** The largest call the new path produced
  in 112,000 was 1.7x its median; the old path reached 4.2x.
- **The frequency claim is the soft part.** Two outlier runs against zero, at
  n=28, is not on its own a significant difference and **must not be quoted as
  one**. It is consistent with the magnitude result; it does not independently
  establish anything.
- **Everything here is a container on aarch64**, with no `SCHED_FIFO` and no
  core pinning — the container lacked `CAP_SYS_NICE` when the measurements were
  taken. Treat all of it as a relative signal. Absolute numbers belong on the
  target hardware.

## Sign-off targets on the native machine

For the native x86-64 box, on the real workload, with at least 4000 steps and a
tuned real-time environment:

| Target | Threshold |
|---|---|
| p99.9 / p50 | ≤ 1.3 |
| max / p50 | ≤ 2.0 |
| p50 | ≤ baseline p50 |
| absolute max | < 6 ms |
| wrapper allocations per call | 0 |

The `max/p50` bound has a floor worth knowing: in-process Python PJRT on the
same workload sits at 1.14–1.25, so 2.0 is a bound on the wrapper's
contribution, not on the ratio itself.

## A campaign you can reproduce

The comparison above predates the shipped examples and cannot be re-run from
this tree. This one can:

```console
$ tools/run_matrix.sh trajopt 2000 3 artifacts/reports/matrix.csv
```

`examples/02_trajopt`, 2000 calls per run, three interleaved rounds per
configuration, on an idle x86-64 i9-14900HX. The host is **not** real-time
tuned: `powersave` governor, transparent hugepages on, no `isolcpus`, no
`nohz_full`. Medians across the three rounds:

```{table}
:class: results

| Configuration | p50 (µs) | p99 (µs) | p99.9 (µs) | max/p50 |
|---|---|---|---|---|
| `sync_t1` inline, 1 thread | 2240 | 2670 | 3355 | 2.00 |
| `sync_t2` inline, 2 threads | 2129 | 3690 | 4523 | 2.20 |
| `sync_t4` inline, 4 threads | 2070 | 4403 | 5350 | 2.88 |
| `sync_tdefault` inline, XLA's pool | 2141 | 6167 | 7462 | 3.96 |
| `async_t1` dispatched, 1 thread | 2263 | 3203 | 4569 | 2.26 |
| `async_t4` dispatched, 4 threads | 2196 | 5064 | 5866 | 3.23 |
| `async_tdefault` dispatched, XLA's pool | 2391 | 7314 | 8521 | 3.99 |
| **`sync_t1_rt`** inline, 1 thread, hardened | 2239 | **2378** | **2825** | **1.51** |
```

Read the first column and then the last, because they disagree.

**The median barely moves.** Every configuration lands between 2070 and 2391 µs
— a 15% spread with no clear ordering. Giving this workload four threads, or
XLA's whole default pool, does not make it faster. A benchmark reporting means
would conclude that none of these settings matters.

**The tail moves by a factor of three.** p99.9 runs from 2825 µs to 8521 µs, and
`max/p50` from 1.51 to 3.99, ordered almost perfectly by how much concurrency
the runtime was allowed: one thread beats two, two beat four, and four beat
letting XLA size the pool itself. Inline execution beats dispatch at every
thread count. The best configuration is the most restrictive one — a single
thread, executing inline, with the memory locked and the allocator pinned down.

That is the whole argument for this project in one table. The work is the same
in every row; what changes is how many ways the runtime can be interrupted
while doing it.

### What this does and does not establish

The **ordering is solid**. It holds in every individual round, not just in the
medians: `sync_t1_rt` measured 1.37, 1.51 and 1.76, while the two `tdefault`
rows never came in under 3.39. Three rounds of 2000 calls, interleaved so
neither thermal drift nor a background process can favour one configuration.

The **spike frequencies are not established here**, and no claim above depends
on them. A >2x outlier appears roughly once per 20,000 calls, and 6000 calls per
configuration is too few to say anything about how often. These are tail
*ratios* within a run, which is a different and much cheaper question.

The absolute numbers belong to this host and this workload. On a machine with
`isolcpus` and a `performance` governor, expect the hardened row to improve and
the gaps to widen; {doc}`guides/realtime` has the checklist, and the measured
2.6x the governor alone is worth on a 100 Hz loop.

## How these were measured

Every trap below produced numbers that looked plausible and were meaningless.
{doc}`guides/measuring` has them at length; the short form is:

- **Check `/proc/loadavg` before every run.** A busy machine invalidates a
  result rather than adding noise to it. The same configuration measured during
  a bazel build reported p50 2.4x high and max/p50 4.4 instead of 1.1. Discard;
  never correct.
- **Long campaigns, not long runs.** A >2x max/p50 outlier appears roughly once
  per 20,000+ calls, so runs of 300 or 4000 miss it. Hence 28 × 4000.
- **Warm up blocking on every output**, or async dispatch backlog from the
  warm-up lands on the first timed call.
- **Interleave configurations in short rounds.** A sequential A/B drifts with
  CPU temperature by the same order as the effect being measured.
- **Keep the host audit with the numbers.** `tools/rt_check.sh` reports the
  governor, `isolcpus`, `nohz_full`, transparent hugepages and `RLIMIT_RTPRIO`,
  and a latency number without it is hard to interpret later.
