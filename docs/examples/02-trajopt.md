# 02 · Trajectory optimization

`examples/02_trajopt` is the example to quote numbers from. It is a workload
heavy enough to time — a trajectory-optimisation problem in float64, with
reference cases frozen from JAX so that a fast answer is also checked to be the
right one — and it builds and runs from this tree, so a reader can reproduce
what it reports on their own machine.

It is also the workload behind two other things. **The benchmark
(`make bench`) defaults to this fixture, and example 03 runs this artifact**;
neither has an export of its own. That makes `examples/02_trajopt/export.py`
the one export to re-run after a JAX bump, after a machine change, or before
recording any number at all — a serialized executable embeds machine code for
the host that produced it, so an artifact carried from elsewhere is either a
load failure or a silent in-process recompile.

:::{note}
The headline campaign on the {doc}`benchmarks page <../benchmarks>` predates
these examples and is **not** reproducible from the tree. This one is. Numbers
from here and numbers from there are not comparable, and should not be put in
the same table.
:::

## The workload

The export script below is the authoritative description: it defines the
function, the argument shapes it is traced with, and the number of reference
cases written. The signature that results — every input and output by name,
dtype, shape, element count and byte count — is recorded in
`artifacts/trajopt.json` and printed by:

```console
$ python -m jax2exec check artifacts/trajopt
```

which needs no JAX and is the fastest way to see what you are about to call.
Everything is float64, which means `jax_enable_x64` has to be set before
tracing; the sidecar's `export.x64_enabled` records whether it was.

```{literalinclude} ../../examples/02_trajopt/export.py
:language: python
:caption: examples/02_trajopt/export.py
```

Alongside the three artifact files, this writes reference cases:
`trajopt_cases.json` plus one `trajopt_case<i>.bin` per case, each holding
every input followed by every output in call order, little-endian, no header
and no padding. The C++ side replays them and compares. That is what makes the
example a correctness check and not only a stopwatch — and sweeping the cases
forwards and backwards is specifically what verifies that **reusing input
buffers across calls with changing data is bit-exact**, which is the property
the whole zero-copy design rests on.

## The caller

```{literalinclude} ../../examples/02_trajopt/trajopt.cpp
:language: cpp
:caption: examples/02_trajopt/trajopt.cpp
```

## Build and run

```console
$ uv sync
$ make plugin
$ make
$ make export
$ ./build/bin/example_02_trajopt --iterations 500 --json artifacts/reports/trajopt.json
```

`--json` writes the same summary as a machine-readable report, which is what
`make run-examples` does. Run it with `--help` for the full flag list.

## Expected output

```{code-block} text
:caption: Illustrative. Every number here varies with the machine, and is meaningless if the machine was not idle.

=== steady state (n=500, microseconds) ===
  mean      1180.4     stddev       31.2
  min       1142.7     p50        1173.9
  p90       1210.5     p99        1288.1
  p99.9     1402.6     p99.99     1402.6
  max       1402.6
  --- tail ratios ---
  max/p50     1.195      p99.9/p50   1.195
  --- histogram (us) ---
  [   1024.0,  2048.0)        500 ########################################
```

## Reading it

Read the bottom two lines first.

| Number | What it says |
|---|---|
| `p50` | what a typical call costs. Useful for capacity, not for a deadline. |
| `p99.9` | the one-in-a-thousand call. On a 1 kHz loop, that is one cycle per second. |
| `max` | the worst call in this run. A run too short to contain a rare spike will flatter it. |
| `max/p50` | the worst call relative to a typical one — the headline. |
| `p99.9/p50` | the same for the one-in-a-thousand call. This is the ratio this project optimizes. |
| `mean`, `stddev` | reported for completeness. Neither is the objective, and an improvement in the mean that raises `max/p50` is a worse result. |

A nonzero `dropped` line means the recorder ran out of capacity and you are
looking at the tail of a truncated prefix; raise the capacity rather than
trusting the percentiles.

## Before you believe a number from it

Each of these was learned by producing numbers that looked plausible and were
meaningless. {doc}`../guides/measuring` has them in full.

- **Check `/proc/loadavg` first.** A busy machine does not add noise, it
  invalidates the result — the same configuration measured during a bazel build
  reported p50 2.4x high and max/p50 4.4 instead of 1.1. Discard a contaminated
  run; never correct it.
- **Rare spikes need long campaigns, not long runs.** A >2x `max/p50` outlier
  appears roughly once per 20,000+ calls, so a 500- or 4000-call run can miss it
  entirely.
- **Interleave configurations in short rounds.** A sequential A/B drifts with
  CPU temperature by the same order as the effect being measured.
- **Keep `tools/rt_check.sh` output beside the numbers.** "p99.9 was 1.4 ms"
  means little without knowing whether the governor was on `powersave`.

## What to change to make it yours

Swap the function in `export.py` for your own and keep the rest: the reference
cases, the checked replay, and the summary are all signature-agnostic. Two
things to watch when you do.

**Every input must feed an output.** XLA prunes parameters the computation never
reads, and the executable then has a different arity than the sidecar says.
This bites synthetic workloads hardest, where an unused input is easy to write
by accident.

**A workload with data-dependent control flow has algorithmic spread of its
own.** An early-exiting loop or a branch that switches on after some number of
calls contributes to `max/p50` without any system jitter being involved. When
you are attributing a change to the call path rather than to the workload, run
it against a fixed-cost twin of the same signature and check that both move
together — that agreement is what made the attribution on the
{doc}`benchmarks page <../benchmarks>` credible.
