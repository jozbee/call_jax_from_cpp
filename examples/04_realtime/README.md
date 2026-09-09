# 04 · A real-time control loop

Runs the `02_trajopt` artifact on a fixed period: harden once at startup, then
sleep to the next deadline, write the inputs, `call()`, feed the outputs back.
It records call latency, cycle time, wake-up latency and signed period jitter,
alongside page faults, context switches and the per-call allocation census.

The example is documented on
[04 · Real-time loop, instrumented](https://jozbee.github.io/call_jax_from_cpp/examples/04-realtime.html):
what each part of `realtime.cpp` does, and what it adds to example 03. How to
read the report and what the host still has to provide are in the
[real-time guide](https://jozbee.github.io/call_jax_from_cpp/guides/realtime.html);
how long a run has to be before it licenses a claim is in the
[measuring guide](https://jozbee.github.io/call_jax_from_cpp/guides/measuring.html);
the numbers this project has measured are on the
[benchmarks page](https://jozbee.github.io/call_jax_from_cpp/benchmarks.html).
What is here is what the site does not carry: the flags, and the launcher.

There is **no export script here.** This example loads what
`examples/02_trajopt/export.py` writes:

```console
$ make plugin && make && make export
$ ./build/bin/example_04_realtime --iterations 3000
$ examples/04_realtime/run_realtime.sh --iterations 3000   # audit + launch
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
