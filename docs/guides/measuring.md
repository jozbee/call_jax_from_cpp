# Measuring

A latency number from this project is only meaningful if the machine was idle,
the campaign was long enough to contain a rare spike, and the configurations
being compared were interleaved rather than run in sequence.

Every trap below was hit during this work. Each one produces numbers that look
plausible and are meaningless — which is worse than no numbers, because they
send the next several hours of work in the wrong direction.

## The five traps

### 1. A busy machine invalidates a result; it does not add noise to it

The new runtime first measured *worse* than the old one — p50 9168 µs, max/p50
4.36 — because a bazel XLA build was saturating all 16 cores (load average
16.69). The same configuration on an idle machine: p50 about 4000 µs, max/p50
about 1.1. That is **p50 2.4x high and max/p50 4.4 instead of 1.1**, from one
competing workload.

Check `/proc/loadavg` before every run. Do not start a build "in the background
while the benchmark runs". **Discard anything measured under load; never
attempt to correct for it.** `make bench` and `tools/run_matrix.sh` both print
the load average with the numbers, and the sweep says so loudly above 0.5.

### 2. Rare spikes need long campaigns, not long runs

A >2x max/p50 outlier appears roughly **once per 20,000+ calls**. Runs of 300,
or even 4000, miss it entirely — which is exactly why the original bug report
read "most of the time there is little jitter, but on some runs there are large
spikes".

The campaign behind the headline numbers was **28 runs × 4000 calls per API**,
112,000 calls each, interleaved. Any claim about spike *frequency* needs that
shape of evidence, and even then, see the honesty note on
{doc}`../benchmarks`.

### 3. Warm-up must block on every output

XLA dispatch is asynchronous unless the plugin honours the inline option. A
warm-up that does not wait on *every* output leaves a backlog that lands on the
first timed call and turns a 1.2x max/p50 into a 27x one. Every call the
benchmark makes, warm-up included, waits on all outputs.

`Function::call()` already does this, and `FunctionOptions::warmup_calls` runs
blocking calls at load. The trap is real for anything hand-rolled around the
PJRT C API.

### 4. A sequential A/B drifts with temperature

A few percent of drift across a long sequential comparison is the same order as
the effects being measured. Interleave configurations in short rounds and
compare medians across rounds. `tools/run_matrix.sh` exists for this; do the
same for any ad-hoc comparison.

### 5. A probe must be sensitive to what it perturbs

The zero-copy probe first reported "mutation observed: NO". The perturbed input
simply did not affect the output being checked. Re-running with all 1200
elements perturbed and a checksum over *every* output, against a
fresh-copying-buffer control, confirmed that reuse works. **When a probe
reports a negative, verify that it can detect a positive.**

Relatedly, and expensively: **XLA prunes parameters the computation never
reads**, changing the executable's arity. A synthetic kernel exported with 16
inputs came back expecting 4 — `Execution supplied 16 buffers but compiled
program expected 4`. Any synthetic benchmark kernel must make every output
depend on every input. The exporter now refuses this case at export time; see
{doc}`exporting`.

## The benchmark

```console
$ make bench                                        # the default configuration
$ make bench BENCH_ARGS="--fixture trajopt --iterations 4000"
$ build/bin/bench --help                            # every flag
```

`make bench` builds and runs, because a benchmark that was built but not run
tells you nothing. It prints the load average first.

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

A run prints the summary the recorder produces — count, mean and standard
deviation, min, p50, p90, p99, p99.9, p99.99, max, then the two ratios
`max/p50` and `p99.9/p50`, then a log-spaced ASCII histogram of the samples
between the first and last non-empty bucket. Read the ratios; the mean is
printed because it is cheap, not because it is the objective.

The correctness gates run alongside: every iteration is checked against the
frozen reference outputs, and `--all-cases` sweeps the reference cases forwards
and backwards. That sweep is what verifies that reusing input buffers across
calls with changing data is bit-exact.

### The CSV columns

`LatencyRecorder::write_csv_row` appends one row per run and writes the header
only when the file did not already exist, which is what makes an interleaved
sweep possible: each short run adds a row to the same file, and the comparison
happens across rows.

```text
label,config,n,dropped,mean_us,stddev_us,min_us,p50_us,p90_us,p99_us,
p999_us,p9999_us,max_us,max_over_p50,p999_over_p50
```

`config` is a free-form description of what produced the row — the API, the
thread count, the fixture, `$XLA_FLAGS` verbatim. It is quoted when it contains
a comma, because a flag list is entitled to contain one and an unquoted comma
shifts every later column and silently misattributes the numbers.

`dropped` is not decoration. The recorder has fixed capacity and **drops rather
than grows**, because growing would allocate in the middle of the run being
measured. A non-zero `dropped` means the summary describes a prefix of the run.

### Recording latencies in your own program

```{literalinclude} ../../examples/02_trajopt/trajopt.cpp
:language: cpp
:start-after: docs: begin latency-recorder
:end-before: docs: end latency-recorder
```

Three properties of `LatencyRecorder` are deliberate, and each came from a
measurement that went wrong: samples are **nanoseconds and signed** (whole
microseconds discard the jitter being hunted, and period jitter is negative
whenever a cycle runs early); `record()` **never allocates and never touches a
file**; and percentiles **interpolate between neighbouring ranks**, the way
NumPy's `percentile` does, so the analysis scripts and the C++ agree.

It times with `steady_clock` rather than `high_resolution_clock`, which is an
alias for the wall clock on some standard libraries — an NTP step mid-run would
otherwise show up as a spectacular outlier that never happened.

`write_samples()` is worth using even when the summary looks fine. Summaries
hide *when* an outlier happened, and a spike on call 3 (a page warm-up never
touched) and a spike on call 30,000 (something periodic) have the same p99.9
and completely different causes.

## Sweeping configurations

```console
$ tools/run_matrix.sh trajopt 300 3
```

Fixture, iterations per run, rounds. It writes one CSV row per (config, round)
and prints medians across rounds at the end.

**Rounds are the outer loop.** That is the whole point: each configuration is
run for a short burst, then the next, then round two, so a drift in machine
temperature applies to all of them roughly equally instead of to whichever one
ran last.

```{literalinclude} ../../tools/run_matrix.sh
:language: bash
:start-after: docs: begin matrix-configs
:end-before: docs: end matrix-configs
```

The axes are whether execution is inline and how many threads XLA may use, plus
one row that adds the real-time hardening on top of the best-behaved
configuration. `tdefault` leaves `PJRT_NPROC` unset — which is what a caller
gets by accident, and worth having in the table for that reason.

## Campaign shapes

| Question | Shape | Why |
|---|---|---|
| Did this change break anything? | 20–200 calls, any machine | Correctness only. Never quote the latencies. |
| Did the median move? | 300–2000 calls × 3+ interleaved rounds | Enough for p50 and p99; drift is controlled by interleaving. |
| Did the tail move? | ≥ 20,000 calls per configuration | The 2x outlier rate is about one per 20,000. Fewer calls cannot see it. |
| Did spike *frequency* change? | Many runs × thousands of calls, interleaved | And even 28 × 4000 was not enough to call two-versus-zero significant. |

Whatever the shape: record `tools/rt_check.sh` output and `/proc/loadavg`
beside every number. "p99.9 was 4.8 ms" means little without knowing whether
the governor was on `powersave` at the time, and a number whose provenance was
not written down cannot be defended three weeks later.

## The allocation census

Grepping the source for `malloc` proves nothing about what the linked binary
does at run time: OpenBLAS, libm and the C++ runtime all allocate behind the
caller's back, and an inlined `std::vector` growth is invisible to any static
check. The only trustworthy answer comes from interposing the allocator in the
real process.

```console
$ make test-alloc
```

That preloads `tests/support/malloc_guard.c` — built to `build/lib/malloc_guard.so`,
`LD_PRELOAD` on Linux and `DYLD_INSERT_LIBRARIES` on macOS — and runs the
benchmark with the gate armed. **Nothing links against the guard.**
`pjrt::AllocGuard` resolves its markers with `dlsym(RTLD_DEFAULT, ...)` and
degrades to no-ops returning 0 when they are absent, so one binary runs both
with and without the preload.

```{literalinclude} ../../examples/03_realtime/realtime.cpp
:language: cpp
:start-after: docs: begin alloc-guard
:end-before: docs: end alloc-guard
```

Arm it around exactly the region being claimed clean — the steady-state calls,
after warm-up. Including warm-up would fold in the page faults and lazy
initialization that warm-up exists to pay for, and turn a clean path into a few
thousand allocations.

### Why allocations are classified by calling module

A whole-process "zero allocations" gate is not achievable here. **XLA's thunk
runtime allocates roughly 15,400 times per call**, about one per StableHLO op,
inside the plugin, through an allocator the PJRT C API does not expose. What
*is* checkable is that this project adds none of its own, so every allocation
made while armed is attributed to the module containing its return address:

| Class | Covers | Gate |
|---|---|---|
| `allocs_self()` | The main executable and `libpjrt_exec` | **must be zero** |
| `allocs_plugin()` | Inside `libpjrt_c_api_cpu_plugin` | reported, ~15,400/call |
| `allocs_runtime()` | libc, libstdc++, LAPACK, the thread pool | reported |

The C++ `operator new` family is interposed too, under its Itanium-mangled
names — without that, an inlined `std::vector` growth would be charged to
libstdc++ rather than to the module that grew the vector, which is exactly the
attribution the gate depends on.

`total()` counts allocations for the whole process whether armed or not, and it
is what makes a zero armed count believable: thousands in total with zero while
armed means the path is clean, while zero in total means the preload never took
effect. That is what `--require-guard` checks.

Classification is ELF-only. On macOS the totals are still correct and
`classified()` reports false.

So the honest claim is not "this allocates nothing". It is **"the wrapper
allocates nothing in the steady state, and roughly 15,400 allocations per call
remain inside XLA, which is the next place to look"**. See
{doc}`../developer/open-threads`.
