# 02 · Trajectory optimization

`examples/02_trajopt` is the workload to time: a trajectory-optimisation step
in float64, heavy enough that a call takes milliseconds, with reference cases
frozen from JAX so a fast answer is also checked to be the right one. It is
also the artifact example 03 runs and the fixture `make bench` defaults to, so
its export script is the one to re-run after a JAX bump or a machine change.

The model is a chain of masses with cubic springs, damping, a sine "gravity"
and dense coupling, integrated with RK4 over a fixed horizon; the optimizer is
a fixed number of warm-started gradient-descent iterations, each with a
fixed-size line search. Every trip count is static, so the spread in call
latency is the system's, not the data's. Model-predictive control is the
motivation; no controller ships here.

## The presets

```{literalinclude} ../../examples/02_trajopt/export.py
:language: python
:start-after: docs: begin trajopt-presets
:end-before: docs: end trajopt-presets
```

`default` is what the tests, the benchmark and example 03 use. `small` is a
fast export for a smoke test, not a workload to time.

## The model

```{literalinclude} ../../examples/02_trajopt/export.py
:language: python
:start-after: docs: begin trajopt-model
:end-before: docs: end trajopt-model
```

## The solver

```{literalinclude} ../../examples/02_trajopt/export.py
:language: python
:start-after: docs: begin trajopt-solve
:end-before: docs: end trajopt-solve
```

Alongside the three artifact files, the export writes reference cases —
`trajopt_cases.json` plus one `.bin` per case holding every input and output
in call order — which the C++ side replays forwards and then backwards. The
backwards pass verifies that reusing input buffers across calls with changing
data is bit-exact, which is the property the zero-copy design rests on.

## The caller

The cold call is timed alone, then warm-up, then the timed loop with the
allocation guard armed:

```{literalinclude} ../../examples/02_trajopt/trajopt.cpp
:language: cpp
:start-after: docs: begin trajopt-run
:end-before: docs: end trajopt-run
```

Between calls, the feedback step copies this cycle's outputs into the next
cycle's inputs. Flags, the finite-output audit, fault injection and the JSON
report live in `examples/02_trajopt/support.hpp`.

## Build and run

```console
$ uv sync && make plugin && make && make export
$ ./build/bin/example_02_trajopt --iterations 500 --json artifacts/reports/trajopt.json
$ uv run python examples/02_trajopt/export.py --out artifacts --preset small   # the quick variant
```

`--help` lists every flag. Exit codes: 0; 1 an error; 2 a correctness gate
failed; 3 the allocation gate failed; 4 the guard was required but not
preloaded.

## Reading the summary

Read the bottom two lines first.

| Number | What it says |
|---|---|
| `p50` | what a typical call costs — capacity, not a deadline |
| `p99.9` | the one-in-a-thousand call; on a 1 kHz loop, one cycle per second |
| `max` | the worst call in this run; a run too short to contain a rare spike flatters it |
| `max/p50`, `p99.9/p50` | the worst and the one-in-a-thousand call relative to a typical one — the ratios this project optimizes |
| `mean`, `stddev` | reported for completeness; an improvement in the mean that raises `max/p50` is a worse result |

A nonzero `dropped` means the recorder ran out of capacity and the percentiles
describe a truncated prefix. Before quoting any of it: an idle machine,
`tools/rt_check.sh` output beside the numbers, and a campaign long enough for
the question — {doc}`../guides/measuring`.

## Making it yours

Swap `Model` for your own function and keep the rest: the reference cases, the
checked replay and the summary are signature-agnostic. Every input must feed
an output, or XLA prunes it. And a workload with data-dependent control flow
has spread of its own — when attributing a change to the call path, run a
fixed-cost twin of the same signature and check that both move together.
