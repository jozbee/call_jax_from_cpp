# Measurement

This is the most important page in this directory. A wrong latency number does
not merely mislead: it sends the next several hours of work in the wrong
direction, because every decision after it is made against a fiction. Each trap
below was hit during this work, each one produces numbers that look entirely
plausible, and none of them announces itself.

Two facts frame everything here.

- **The objective is the tail.** p99.9/p50, max/p50, and the worst call in a
  long campaign. A mean-latency improvement is not the objective and should not
  be offered as one.
- **A contaminated run is discarded, never corrected.** There is no correction
  factor for a busy machine, and applying one produces a number that is wrong
  in a way nobody can audit later.

## Trap 1 — a busy machine invalidates a result, it does not add noise

The new runtime first measured *worse* than the path it replaced: p50 9168 µs
and max/p50 4.36. A bazel XLA build was saturating all 16 cores at the time
(load average 16.69). The same configuration on an idle machine gave p50 ~4000
µs and max/p50 ~1.1 — the p50 was 2.4x high and the tail ratio was four times
too large.

Check `/proc/loadavg` before every run. `make bench` prints it above the
numbers for exactly this reason, and `tools/run_matrix.sh` prints it before it
starts. Do not start a build "in the background while the benchmark runs": the
plugin build in particular takes half an hour of every core, and it will land
in the middle of a campaign.

## Trap 2 — rare spikes need long campaigns, not long runs

A greater-than-2x max/p50 outlier appears roughly once per 20,000+ calls. A run
of 300 calls misses it entirely, and so does a run of 4000 most of the time.
This is precisely why the original bug report read "most of the time there is
little jitter, but on some runs there are large spikes" rather than "the
latency is bad".

The campaign behind the headline numbers was **28 runs × 4000 calls per API**,
112,000 calls each, interleaved. Anything claiming a change in spike
*frequency* needs evidence of that shape — and note the honest limit even then:
two outlier runs against zero, at n=28, is not on its own a significant
difference. The change in spike *magnitude* is the solid part.

## Trap 3 — warm-up must block on every output

XLA dispatch is asynchronous unless inline execution was negotiated. A warm-up
that does not wait on *every* output leaves a backlog that lands on the first
timed call, which turns a 1.2x max/p50 into a wildly larger one and puts the
whole distribution's worst sample in the first bucket.

Every call the benchmark makes, warm-up included, waits on all outputs. It
times the first call separately — the first call after a load is always the
slowest, and burying it in the steady-state distribution hides both facts —
then warms up, then checks correctness, then times the steady state with the
allocation guard armed.

## Trap 4 — a sequential A/B drifts with temperature

A few percent of drift across a long sequential comparison is the same order as
the effects being measured, so "A then B" cannot distinguish a real difference
from a warm CPU. `tools/run_matrix.sh` interleaves the configurations in short
rounds and compares medians across rounds. Do the same for any ad-hoc
comparison, including the quick one you were not going to write down.

## Trap 5 — a probe must be sensitive to what it perturbs

The zero-copy probe first reported "mutation observed: NO". The perturbed input
simply did not affect the output being checked. Re-run with every element of a
large input perturbed and a checksum over *every* output, against a
fresh-copying-buffer control, it gave identical results and confirmed that
reusing input buffers works. **When a probe reports a negative, first verify
that the probe can detect a positive.**

Relatedly, and for the same reason: **XLA prunes parameters the computation
never reads**, which changes the executable's arity. A synthetic kernel
exported with 16 inputs came back expecting 4:

```
Execution supplied 16 buffers but compiled program expected 4
```

Any synthetic benchmark kernel must make every input feed an output, or the
thing you built to measure a 16-argument signature is measuring a 4-argument
one.

## The tools

```
make bench                        # builds, prints the load average, runs
make bench BENCH_ARGS="--fixture trajopt --iterations 4000"
tools/run_matrix.sh trajopt 300 3 # interleaved sweep -> CSV + medians
tools/rt_check.sh                 # host audit: governor, isolcpus, nohz_full,
                                  # THP, RLIMIT_RTPRIO
make test-alloc                   # the zero-allocation gate
make test-rt                      # the real-time statistics gates
```

`build/bin/bench` measures **exactly one configuration** and records what it
was; anything comparative is built out of several runs of it. Its flags select
the fixture and the campaign shape, the runtime configuration being measured,
where the output goes, and which gates are armed. The flag table lives in the
{doc}`measuring guide <../guides/measuring>` rather than here, and
`build/bin/bench --help` is authoritative — the flags move with the binary and
a second copy of them would only drift.

`tools/run_matrix.sh` exists to make trap 4 hard to fall into. It runs each
configuration for a short burst, rotates through them, repeats for the
requested number of rounds, and then reports the **median across rounds** for
each configuration rather than any single run. The axes are the ones that
plausibly move tail latency: inline versus asynchronous execution, the XLA
thread-pool size, and the real-time hardening on top of the best-behaved
configuration. It prints the load average before the first run and after the
last, and says so loudly when the machine is not idle.

Keep `tools/rt_check.sh` output next to any number you record. "p99.9 was
4.8 ms" means very little without knowing whether the governor was on
`powersave`, whether the cores were isolated, and whether `RLIMIT_RTPRIO`
allowed the priority the process asked for. The script is read-only: it reports,
it changes nothing.

## Campaign shapes

Pick the shortest shape that can answer the question being asked.

| Question | Shape |
|---|---|
| Does it still compute the right answer? | ~20 iterations, every reference case, forwards and backwards |
| Did this change move the median? | ~2000 iterations, both configurations interleaved in rounds |
| Did this change move the tail? | 28 rounds × 4000 calls per configuration, interleaved |
| Does the wrapper still allocate nothing? | 200 iterations under the preloaded counter |
| Is this host fit to be measured at all? | `tools/rt_check.sh`, before any of the above |

A campaign that answers the tail question takes hours, and the machine must be
idle for all of them. That is the cost of the only evidence that counts for a
frequency claim.

## Correctness gates come first

A latency number from a run that computed the wrong answer is not a
conservative estimate, it is noise. Every benchmark iteration checks its
outputs against reference cases written by `jax2exec.reference`, and the case
sweep runs them **forwards and then backwards**. The backwards pass is the
point: it is what verifies that reusing the same input arenas across calls with
changing data is bit-exact, rather than accidentally correct because the cases
happened to be visited in the order they were recorded.

Two more gates worth running before believing anything:

- `make test-alloc` counts allocations in the steady-state call path with a
  preloaded interposer. The number that must be zero is the wrapper's own; XLA's
  thunk runtime allocates thousands of times per call inside the plugin, and
  that is not reachable from here. See
  [runtime internals](runtime-internals.md).
- `python -m jax2exec check <base>` describes an artifact set and says whether
  it will run on this host, which is the cheap way to catch an artifact
  exported somewhere else before it fails as an illegal instruction.

## What a reportable number looks like

Everything below travels with the number, or the number does not travel:

- the machine, its architecture, and whether it was bare metal or a container;
- the load average at the start of the run;
- `tools/rt_check.sh` output, or a note that it was not run;
- the fixture, the iteration count, and how many runs the campaign contained;
- p50, p99, p99.9, max, and both ratios — not a mean;
- which parts of the conclusion are solid and which are soft.

That last line is not decoration. The honest limits of a result are what let
the next person build on it instead of re-deriving it.
