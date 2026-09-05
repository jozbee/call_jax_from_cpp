# 02 — Measuring without producing garbage

Every trap below was hit during this work. Each one produces numbers that look
plausible and are meaningless. This is the most important file in the
directory: a wrong measurement here does not merely mislead, it sends the next
several hours of work in the wrong direction.

## Trap 1: a busy machine invalidates results, it does not add noise

The new runtime first measured *worse* than the old one — p50 9168 µs,
max/p50 4.36 — because a bazel XLA build was saturating all 16 cores
(loadavg 16.69). The same configuration on an idle machine: p50 ~4000 µs,
max/p50 ~1.1.

**Check `/proc/loadavg` before every run.** Do not start a build "in the
background while the benchmark runs." Discard anything measured under load;
do not attempt to correct for it.

## Trap 2: rare spikes need long campaigns, not long runs

A >2x max/p50 outlier appears roughly once per 20,000+ calls. Runs of 300 or
even 4000 calls miss it entirely. This is exactly why the original bug report
was "most of the time there is little jitter, but on some runs there are large
spikes."

The campaign that produced the headline numbers: **28 runs × 4000 calls per
API** (112,000 calls each), interleaved. Anything claiming a change in spike
*frequency* needs that shape of evidence. Note the honest limit even then: two
outlier runs against zero, at n=28, is not on its own a significant frequency
difference. The difference in spike *magnitude* is the solid part.

## Trap 3: warm-up must block on every output

Otherwise async dispatch backlog from warm-up lands on the first timed call.
`run_measured` in `src/bench/bench_main.cpp` times the first call separately,
then warms up, then checks correctness, then times the steady state with the
allocation guard armed.

## Trap 4: sequential A/B drifts with temperature

A few percent of drift across a long sequential comparison is the same order as
the effects being measured. `tools/run_matrix.sh` interleaves configurations in
short rounds and compares medians across rounds for this reason. Do the same
for any ad-hoc comparison.

## Trap 5: the fixture must be sensitive to what you perturb

The zero-copy probe first reported "mutation observed: NO". The perturbed input
(`acc_ref[0]`) simply did not affect the output being checked. Re-running with
all 1200 elements of `last_control` perturbed and a checksum over *every*
output, against a fresh-copying-buffer control, gave identical results and
confirmed reuse works. When a probe reports a negative, verify the probe can
detect a positive.

Relatedly: **XLA prunes parameters the computation never reads**, changing the
executable's arity. A synthetic kernel exported with 16 inputs came back
expecting 4 ("Execution supplied 16 buffers but compiled program expected 4").
Any synthetic benchmark kernel must touch every input.

## The measurement tools

```
make bench BENCH_ARGS="--api rt --fixture mpc_solver --iterations 2000"
tools/run_matrix.sh mpc_solver 300 3    # interleaved sweep -> CSV
tools/rt_check.sh                       # host audit: governor, isolcpus,
                                        # nohz_full, THP, RLIMIT_RTPRIO
make test_alloc                         # allocation census
```

`bench` flags: `--api rt|legacy`, `--fixture`, `--case`, `--all-cases`,
`--iterations`, `--warmup`, `--csv`, `--samples`, `--label`, `--async`,
`--threads`, `--devices`, `--rt`, `--cpu`, `--no-check`.

Keep `rt_check.sh` output next to any numbers you record. "p99.9 was 4.8 ms"
means little without knowing whether the governor was on `powersave`.

## Two fixtures, and why

- `mpc_solver` — the real acceptance workload. Its L-BFGS `while_loop`s exit
  early and a `cond` switches on after ~50 calls, so **part of its spread is
  algorithmic, not system jitter**.
- `synth_solver` — identical 16/14 float64 signature and comparable cost, but a
  fixed-trip-count `scan` with no data-dependent control flow. Any spread it
  shows is system jitter. It is the control.

When a change helps `mpc_solver`, check it also helps `synth_solver` before
concluding the win is in the call path rather than in the MPC's own behaviour.
The persistent-buffer change showed ~18% on MPC and ~16% on the twin, which is
what made the attribution credible.

## Correctness gates

Every benchmark iteration checks `out_13 == iter + 1` exactly (an integral
counter output) plus relative error against reference `.npz`-derived `.bin`
fixtures in `tests/assets/mpc/`. `--all-cases` sweeps the reference cases
forwards and backwards, which is what verifies that reusing input buffers
across calls with changing data is bit-exact.
