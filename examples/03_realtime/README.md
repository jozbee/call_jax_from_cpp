# 03 · A real-time control loop

Runs the `02_trajopt` artifact on a fixed period: harden once at startup, then
sleep to the next deadline, write the inputs, `call()`, feed the outputs back.
It reports the four distributions a control engineer actually needs, plus page
faults, context switches and the per-call allocation census.

There is **no export script here.** This example loads what
`examples/02_trajopt/export.py` writes:

```console
$ make plugin && make && make export
$ ./build/bin/example_03_realtime --iterations 3000
$ examples/03_realtime/run_realtime.sh --iterations 3000   # audit + launch
```

`run_realtime.sh` prints the `tools/rt_check.sh` audit and the load average
first, exports the artifact if it is missing, applies `taskset`/`chrt` when
`$CPUSET`/`$CHRT` are set, preloads the allocation counter unless `$NO_GUARD`,
and `exec`s the binary with your arguments.

## Flags

| Flag | Default | What it does |
|---|---|---|
| `--artifact PATH` | `artifacts/trajopt` | Artifact base path, no extension. |
| `--period-us N` | `10000` | Control period. 10 ms is 100 Hz. |
| `--iterations N` | `3000` | Timed cycles. **`0` runs until SIGINT.** |
| `--warmup N` | `200` | Untimed cycles through the identical loop body. |
| `--cpu auto\|none\|N` | `auto` | `auto` takes an isolated, `nohz_full` core, or stays unpinned and says why. |
| `--rt-priority N` | `80` | `SCHED_FIFO` priority; `0` disables the request. |
| `--no-malloc-tune` | off | Skip `harden_malloc()`. |
| `--no-mlock` | off | Skip `lock_memory()`. |
| `--no-corral` | off | Leave XLA's pool threads where they are. |
| `--dma-latency auto\|off` | `off` | Hold `/dev/cpu_dma_latency` at 0 µs for the run (needs root). |
| `--threads N` | `1` | XLA worker threads (`PJRT_NPROC`); `0` is XLA's default. |
| `--json PATH` | — | Write the full report, host audit included. |
| `--samples PATH` | — | Every raw sample; one file per series (`_cycle`, `_wake`, `_jitter`). |
| `--alloc-gate off\|self\|all` | `self` | Which allocations fail the run. |
| `--require-guard` | off | Fail when `malloc_guard.so` was not preloaded. |
| `--no-check` | off | Do not fail on a step-counter mismatch. |
| `--quiet` | off | One summary line instead of the full report. |

Exit codes: `0` ok, `1` error, `2` wrong answer, `3` the loop allocated, `4` an
allocation gate was required and nothing was measured. `3` and `4` are distinct
on purpose — "the path allocated" and "nobody measured whether the path
allocated" must not look the same in a green CI log.

## How to read the report

Four distributions, because they answer four different questions. Confusing
the first with the last is the usual mistake: a perfect `call()` inside a loop
that wakes 400 µs late every third cycle is not a working control loop, and
only the jitter says so.

| Line | Question | A good value |
|---|---|---|
| **call latency** | how long `call()` took | `p99.9/p50 ≤ 1.3`, `max/p50 ≤ 2.0`; above 2 on an idle, tuned host means something preempted the loop |
| **cycle time** | wake-up to end of work | comfortably below the period — see utilization |
| **wake-up latency** | how late the sleep returned | tracks C-state exit and timer resolution; `cyclictest` on the same core is the floor, and no library setting gets below it |
| **period jitter** | this wake-up minus one period after the last, **signed** | centred on 0 and narrow; a wide spread is the scheduler, not XLA — check the governor and `isolcpus` |
| **deadlines** | cycles whose work ran past the next wake-up | 0. One is a bug in the period; many are a bug in the workload. Utilization p50 is the honest headroom figure |
| **major faults** | pages read from disk mid-cycle | **0**. Anything else means `lock_memory()` did not take effect |
| **involuntary switches** | the scheduler taking the core away | ~0 on an isolated core; nonzero means something else wants it |
| **allocations: self** | this project's own allocations in the timed window | **0**, always. This is the gate |
| **allocations: plugin** | XLA's | large and structural — roughly one per StableHLO op inside the thunk runtime, per call. Not reachable through the PJRT C API and not a defect |

Two things the numbers cannot tell you:

- **A missed deadline is late data, not a cancelled call.** PJRT cannot cancel a
  running CPU computation. Design for a bounded computation.
- **The loop never skips a period.** After an overrun the next target is already
  in the past, so the sleep returns immediately and the loop catches up — the
  overrun shows up as an inflated wake latency on the *following* cycle.

## Before quoting anything

`tools/rt_check.sh` audits all of this and exits non-zero when something is
worth fixing.

- **Host:** `isolcpus=`, `nohz_full=` and `rcu_nocbs=` on the kernel command
  line; `performance` governor; deep C-states disabled (hold
  `/dev/cpu_dma_latency` open at 0 — writing and closing achieves nothing);
  transparent hugepages `never` or `madvise`; no swap, or `mlockall`.
- **Process:** `RLIMIT_RTPRIO > 0` for `SCHED_FIFO`, `RLIMIT_MEMLOCK`
  unlimited for `mlockall`.
- **Container:** `--cap-add SYS_NICE --ulimit rtprio=99 --ulimit memlock=-1`.
  Everything in the host list is still the host's to set, and a container
  cannot be given a tuned kernel.
- **Load average.** A concurrent build does not add noise to a tail
  measurement, it invalidates it: the same configuration measured during one
  reported p50 2.4x high and max/p50 4.4 instead of 1.1.

Two limits on what a run licenses you to say. A **spike claim needs at least
20,000 iterations** — 200 s at 100 Hz — because p99.9 over 3,000 samples is
three samples. And **container numbers are relative signals only**: good enough
to compare two configurations inside the same container, not good enough to
quote as absolute latency.

## A measured example of why the environment matters

On an idle i9-14900HX with the `powersave` governor, pinned to one core, with
`mlockall` and `SCHED_FIFO` in effect, the only variable being the period:

| Period | Duty cycle | min | p50 | max/p50 |
|---|---|---|---|---|
| 3 ms | ~65% | 1866 µs | 2027 µs | — |
| 10 ms | ~20% | 1854 µs | 5196 µs | — |
| 10 ms, unpinned | ~20% | 1932 µs | 5201 µs | 2.14 |
| 10 ms, pinned | ~20% | 1878 µs | 5228 µs | 1.40 |

Two separate effects, and it is worth keeping them apart. Pinning does not
move the median at all; it cuts the *tail*, because the loop stops migrating
between cores. The governor does not move the tail much; it moves the
*median*, because a core that is idle 80% of the time is clocked down when the
next period arrives. Neither shows up in a back-to-back benchmark, which keeps
the core busy and therefore boosted.

`min` barely changes across all four rows: that is the speed this workload runs
at when the clock is up, and it is what `make bench` reports.
