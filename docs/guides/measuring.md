# Measuring

A latency number from this project means something only if the machine was
idle, the campaign was long enough to contain a rare spike, and the
configurations being compared were interleaved rather than run in sequence.
The traps that produce plausible, meaningless numbers are catalogued in
{doc}`../developer/measurement`; the five headlines:

1. **A busy machine invalidates a result; it does not add noise to it.** Check
   `/proc/loadavg` first. Discard a contaminated run; never correct it.
2. **Rare spikes need long campaigns, not long runs.** How often a >2x outlier
   appears depends on the configuration, and on a good one it hides in a short
   run.
3. **Warm-up must block on every output**, or the backlog lands on the first
   timed call.
4. **A sequential A/B drifts with temperature.** Interleave in short rounds.
5. **A probe must be sensitive to what it perturbs**, or a negative means
   nothing.

## The benchmark

```console
$ make bench                                        # the default configuration
$ make bench BENCH_ARGS="--fixture trajopt --iterations 4000"
$ build/bin/bench --help                            # every flag
```

`make bench` builds and runs, and prints the load average first.

| Flag | Effect |
|---|---|
| `--fixture <name>` | Which exported artifact to call. |
| `--iterations N` | Timed calls. |
| `--warmup N` | Discarded calls first, all of them blocking. |
| `--case i` | Which reference case to feed in, and check against. |
| `--threads N` | `RuntimeOptions::worker_threads`; `0` leaves XLA's default. |
| `--async` | Ask for asynchronous dispatch instead of inline execution. |
| `--rt` | Apply the `pjrt::rt` hardening before the timed run. |
| `--cpu N`, `--cpu auto` | Pin to a core; `auto` picks an isolated one when the host has any. |
| `--label <name>` | Names the run in the CSV; without it a row is unattributable. |
| `--csv <path>` | Append one summary row. |
| `--samples <path>` | Write every raw sample as `index,ns`. |
| `--json <path>` | Write the machine-readable report, as `make run-examples` does. |
| `--alloc-gate self`, `--require-guard` | Fail the run when the wrapper allocates, and when the interposer is not loaded. |

A run prints the recorder's summary — count, mean, min, p50, p90, p99, p99.9,
p99.99, max, then `max/p50` and `p99.9/p50`, then a log-spaced histogram. Read
the ratios; the mean is printed because it is cheap, not because it is the
objective. Every iteration is checked against the frozen reference outputs,
and `--all-cases` sweeps them forwards and backwards.

## Recording latencies in your own program

```{literalinclude} ../../examples/02_trajopt/trajopt.cpp
:language: cpp
:start-after: docs: begin latency-recorder
:end-before: docs: end latency-recorder
```

Reserve the recorder's capacity once, before the loop: it drops rather than
grows. Write the raw samples out as well as the summary — a spike on call 3
and a spike on call 30,000 have the same p99.9 and different causes. The
field meanings and the CSV columns are on {doc}`../api/cpp/latency`.

## Sweeping configurations

```console
$ tools/run_matrix.sh trajopt 300 3
```

Fixture, iterations per run, rounds. Rounds are the outer loop: each
configuration runs for a short burst, then the next, then round two, so a
drift in temperature applies to all of them roughly equally. It writes one CSV
row per (config, round) and prints medians across rounds.

```{literalinclude} ../../tools/run_matrix.sh
:language: bash
:start-after: docs: begin matrix-configs
:end-before: docs: end matrix-configs
```

## Campaign shapes

| Question | Shape |
|---|---|
| Did this change break anything? | 20–200 calls, any machine; never quote the latencies |
| Did the median move? | 300–2000 calls × 3+ interleaved rounds |
| Did the tail move? | ≥ 20,000 calls per configuration |
| Did spike *frequency* change? | Many runs × thousands of calls, interleaved — and say which part is soft |

Whatever the shape, record `tools/rt_check.sh` output and `/proc/loadavg`
beside every number. A number whose provenance was not written down cannot be
defended later.

## The allocation census

```console
$ make test-alloc
```

Grepping the source for `malloc` proves nothing about what the linked binary
does at run time, so the census interposes the allocator: it preloads
`build/lib/malloc_guard.so` and runs the benchmark with the gate armed.
Nothing links against the guard; `pjrt::AllocGuard` resolves its markers with
`dlsym` and degrades to no-ops when they are absent.

```{literalinclude} ../../examples/03_realtime/realtime.cpp
:language: cpp
:start-after: docs: begin alloc-guard
:end-before: docs: end alloc-guard
```

Arm it around exactly the region being claimed clean — the steady-state calls,
after warm-up. Every armed allocation is attributed to the module that made
it:

| Class | Covers | Gate |
|---|---|---|
| `allocs_self()` | The main executable and `libpjrt_exec` | **must be zero** |
| `allocs_plugin()` | Inside `libpjrt_c_api_cpu_plugin` | reported: XLA's thunk runtime, about one per StableHLO op |
| `allocs_runtime()` | libc, libstdc++, LAPACK, the thread pool | reported |

So the honest claim is not "this allocates nothing". It is "the wrapper
allocates nothing in the steady state, and thousands of allocations per call
remain inside XLA". Why they are classified this way, and what would reach the
rest, is on {doc}`../developer/runtime-internals`.

## Deeper

{doc}`../developer/measurement` — the traps, and what each one cost.
{doc}`../benchmarks` — every figure, with its host. {doc}`../api/cpp/latency`
— the recorder.
