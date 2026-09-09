# Quickstart

Export a small JAX function, then load and call it from C++. This is the
shortest path that ends in a number computed by XLA inside a C++ process, and
it is `examples/01_basic` end to end. The Python side needs JAX
{{ jax_version }}; `uv sync` pins it.

## Five commands

```console
$ uv sync                       # the Python environment: JAX and the exporter
$ make plugin                   # the PJRT CPU plugin, sha256-verified
$ make                          # libpjrt_exec.a and the four C++ examples
$ make export                   # run the exporters; writes artifacts/
$ build/bin/example_01_basic    # load the artifact and call it
```

If `uv sync` fails with `uv: command not found`, see the mise note in
{doc}`installation`. If `make plugin` has no asset for your platform, the same
page has the two other routes to a plugin.

## The export

The function is `fun(A, b) -> (x, r)`: it solves a 4x4 dense linear system and
returns the solution and the residual norm JAX computed for it. `A` is rank 2
on purpose — a matrix argument is the first thing that exercises the row-major
layout path on both sides of the boundary, and a layout disagreement is
invisible in a rank-1 example.

```{literalinclude} ../../examples/01_basic/export.py
:language: python
:start-after: docs: begin export
:end-before: docs: end export
```

:::{warning}
`jax.config.update("jax_enable_x64", True)` is the first line for a reason.
Without it JAX traces a float64 argument as float32 and says nothing about it.
The export then succeeds, the sidecar honestly records `float32`, and a C++
caller writing `double`s into a four-byte-per-element arena walks off the end
of it. The exporter catches this specific case and refuses, but only because
it was worth writing a check for. See {ref}`the x64 trap <x64-trap>`.
:::

## The call

Load once, write the input arenas, call, read the output arenas. The full file
around this fragment also prints the signature the sidecar declares and checks
the solution against a residual it recomputes itself.

```{literalinclude} ../../examples/01_basic/basic.cpp
:language: cpp
:start-after: docs: begin call
:end-before: docs: end call
```

## What the export wrote

Three files, sharing a base path. The C++ side is handed that base path and
reads all three from it.

| File | What it is | Who reads it |
|---|---|---|
| `<name>.binpb` | The serialized PJRT executable, with the target machine code already in it. | `PJRT_Executable_DeserializeAndLoad`, which relinks it and never recompiles. |
| `<name>.mlirbc` | StableHLO bytecode for the same function. | `PJRT_Client_Compile`, when the `.binpb` will not run on this host. |
| `<name>.json` | The sidecar: every input and output by name, dtype, shape, element count and byte count, plus the versions and the host that produced it. | The loader, on every startup. |

With `--out artifacts`, those are `artifacts/basic.binpb`,
`artifacts/basic.mlirbc` and `artifacts/basic.json`. Look at any of them
without JAX installed:

```console
$ python -m jax2exec check artifacts/basic
```

That prints the declared signature, verifies each artifact against the sha256
in the sidecar, and compares the exporting host's instruction-set level with
this one's. It exits non-zero when the set is inconsistent or will not run
here.

## What just happened

The Python side traced the function once, lowered it to StableHLO, and let XLA
compile it. The compute kernels were LLVM-compiled at that moment and embedded
in the `.binpb`, which is why loading it later is a relink rather than a
compile — and why the file is locked to the architecture that produced it.

The C++ side opened the plugin, created one client, read the sidecar, and
allocated a 64-byte-aligned arena for every input and every output. Each input
arena was wrapped once in a zero-copy PJRT buffer, so XLA reads the arena where
it lies and a call transfers nothing. A call is then: execute, one await, and
one `memcpy` per output straight out of device memory.

None of that is visible in the six lines above, which is the intent. The
mechanism is in {doc}`../guides/how-it-works`.

## Where next

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`cpu;1em` Calling from C++
:link: ../guides/calling
:link-type: doc

The loop-shaped version: introspection, typed accessors, the two rules the API
cannot enforce, and the anti-patterns.
:::

:::{grid-item-card} {octicon}`package;1em` Exporting
:link: ../guides/exporting
:link-type: doc

What the exporter accepts and refuses, the dtype table, and the architecture
lock.
:::

:::{grid-item-card} {octicon}`clock;1em` Real-time hardening
:link: ../guides/realtime
:link-type: doc

Read this before anything is deployed. The operating system, not XLA, produces
the multi-millisecond outliers.
:::

:::{grid-item-card} {octicon}`graph;1em` Measuring
:link: ../guides/measuring
:link-type: doc

How to produce a latency number that means something, and the five ways to
produce one that does not.
:::

:::{grid-item-card} {octicon}`plug;1em` Integrate
:link: ../guides/integration
:link-type: doc

A ROS 2 package, a CMake submodule, FetchContent, or plain Make — one recipe
each.
:::

:::{grid-item-card} {octicon}`stopwatch;1em` 03 · Minimal real-time loop
:link: ../examples/03-minimal
:link-type: doc

The artifact you just exported, called on a period with the hardening
applied, in one file you can copy. Skip examples 01 and 02 if the loop is
your use case.
:::

::::

Terms you have not met yet — C-states, `mlockall`, StableHLO, the sidecar —
are two lines each in the {doc}`glossary <../background/glossary>`.
