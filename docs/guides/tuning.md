# Tuning XLA flags

*Assumes the {doc}`Quickstart </getting-started/quickstart>` and
{doc}`measuring`, which define the A/A band and the tail rule. Nothing from
the examples: the function under test is your own.*

XLA's CPU backend has a few dozen code-generation switches, and which ones
make a given program faster depends on the program and on the host. Reading
the flag documentation does not answer it; measuring does.
{py:func}`~jax2exec.tune_flags` takes a JAX function and representative
inputs, runs it under each candidate flag set in a fresh process, and returns
the timing statistics per set for you to choose from. It installs nothing.

## One call

The library's own test is the smallest real run:

```{literalinclude} ../../tests/python/test_tune_flags_subprocess.py
:language: python
:start-after: docs: begin tune-call
:end-before: docs: end tune-call
```

| Name | What it is |
|---|---|
| `"jax.numpy:matmul"` | the function, by importable name; a callable is accepted and resolved to the same form |
| `(a, b)` | the representative inputs, pickled as NumPy leaves for the children |
| {py:class}`~jax2exec.Candidate` | one flag set; the default list is {py:data}`~jax2exec.tune.CATALOG` |
| `role="control"` | an {term}`arm` that must read slower, or the run is invalid |

The test's `rounds`, `reps`, `smoke` and `burn_in` are its own: they make it
finish in seconds. A run of your own leaves them at their defaults, which the
signature states, and passes `cpu=` so every child is pinned to one core.
`result.table()` renders one row per arm; `result.save(path)` keeps the whole
record, samples included.

The function has to be importable by name in a new interpreter: a lambda, a
closure or a function defined in `__main__` is refused. When the inputs are
built by code rather than data, pass the name of a factory and no `args`; the
child calls it and expects `(fn, args)` back.

## What the verdict means

Each round runs every arm once, in declared order on odd rounds and reversed
on even ones, so a position effect changes sign instead of favouring one arm.
Within a round an arm's ratio is its median over the baseline's median; ratios
are never paired across rounds. The {term}`A/A arm` is the baseline run twice,
and the width of its ratios about one is the band inside which nothing can be
told apart.

An arm is *faster* only when every contributing round's ratio falls below the
band, and it is a *winner* only when its median gain also clears `min_gain`.
The two gates are independent: a small, consistent gain fails the second; a
large, inconsistent one fails the first. The `control` arm is a flag known to
slow this kind of program down; a run in which it did not read slower is
reported as invalid rather than as a set of winners.

The median decides. p95, p99 and the worst call are in every row and a row
whose p99 moved against its median is marked, because a flag that improves
the middle and worsens the tail is the result this project refuses to call a
win by accident. p99 over two hundred calls is the third-worst sample, not a
stable statistic, which is why it reports and does not decide.

## What it cannot tell you

A flag the installed XLA does not know is rejected at start-up, and the smoke
probe drops it before any round. A `--xla_backend_extra_options` inner key is
different: the backend accepts any spelling and ignores the ones it does not
recognise, so a mistyped key reads as *within A/A*, exactly like a flag that
does nothing. The table marks those rows; there is no probe for them.

A busy host invalidates a round rather than adding noise to it. The driver
records the load average, the sibling hyperthread's occupancy and the child's
affinity for every arm, and flags the rows it could not trust. Re-run a
flagged session; do not correct it.

## Where the flags go

XLA reads `XLA_FLAGS` once, when the backend initialises, and a `.binpb` is
relinked at load and never recompiled. A tuned set therefore reaches a C++
caller only by being in the environment of the **export** run, never through
{cpp:class}`~pjrt::Runtime`, whose create options the plugin validates and
would reject it from. What the tuner measures is the in-process compiled
program, which is what {py:func}`~jax2exec.export` writes.

## Deeper

{doc}`../developer/measurement` — the traps, including the one above.
{doc}`../api/python` — every name, and {py:func}`~jax2exec.run_arms` for a
comparison that is not over flags, such as one interpreter against another.
