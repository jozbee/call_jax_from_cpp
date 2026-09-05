# call_jax_from_cpp

{.hero-tagline}
Write the numerics in JAX. Export them ahead of time. Call them from a C++
control loop whose worst call you can put a deadline on.

:::{div} hero-buttons

```{button-ref} getting-started/quickstart
:ref-type: doc
:color: primary
:class: sd-px-4
Quickstart
```

```{button-link} https://github.com/jozbee/call_jax_from_cpp
:color: secondary
:class: sd-px-4
GitHub
```

:::

A JAX function is exported ahead of time to a serialized PJRT executable, a
JSON sidecar describing its inputs and outputs, and StableHLO bytecode to fall
back on. A C++ program loads that artifact through the PJRT C API on CPU and
calls it in a loop. Nothing on the steady-state path allocates, locks, logs or
flushes.

The motivating application is nonlinear model-predictive control alongside
[ros2_control](https://github.com/ros-controls/ros2_control), where a late
answer is a wrong answer. No control code ships here; the shape of the problem
is what set the priorities.

## What it is for

::::{grid} 1 1 3 3
:gutter: 3
:class-container: feature-grid

:::{grid-item-card} {octicon}`package;1.1em` Load once, call many
Every input and output gets a 64-byte-aligned arena the runtime owns. Input
arenas are wrapped once in zero-copy PJRT buffers, so a call transfers
nothing: XLA reads the arena where it lies, and outputs are read straight out
of device memory.
:::

:::{grid-item-card} {octicon}`pulse;1.1em` Built for the tail
The figure of merit is the worst call, not the average one. Across 112,000
calls per API, the worst call the new path produced was 1.72x its median; the
older per-call-buffer path reached 4.20x. Runs containing a >2x outlier: 0 of
28, against 2 of 28.
:::

:::{grid-item-card} {octicon}`clock;1.1em` Real-time ready
Optional, independently-failing helpers lock memory, stop the allocator
trimming the heap, pin the calling thread, move XLA's pools off that core, and
raise scheduling priority. Each reports whether it took effect.
:::

::::

## Both halves of the loop

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`code;1em` Python: export

```{literalinclude} ../examples/01_basic/export.py
:language: python
:start-after: docs: begin export
:end-before: docs: end export
```

:::

:::{grid-item-card} {octicon}`cpu;1em` C++: call

```{literalinclude} ../examples/01_basic/basic.cpp
:language: cpp
:start-after: docs: begin call
:end-before: docs: end call
```

:::

::::

The export writes `<name>.binpb`, `<name>.mlirbc` and `<name>.json` into a
directory; the C++ side is given the base path and reads all three. The sidecar
is not a convenience: the PJRT C API cannot be asked what parameters an
executable takes, so the sidecar is the only description of the signature that
exists.

## Where next

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`download;1em` Installation
:link: getting-started/installation
:link-type: doc

The Python environment, the prebuilt PJRT CPU plugin, and the C++ build.
:::

:::{grid-item-card} {octicon}`rocket;1em` Quickstart
:link: getting-started/quickstart
:link-type: doc

Export a function and call it from C++, end to end.
:::

:::{grid-item-card} {octicon}`book;1em` Guides
:link: guides/how-it-works
:link-type: doc

Exporting, calling, real-time hardening, measuring, debugging, and how the
pieces fit.
:::

:::{grid-item-card} {octicon}`list-unordered;1em` API reference
:link: api/index
:link-type: doc

The C++ classes, the Python exporter, and the artifact format.
:::

:::{grid-item-card} {octicon}`graph;1em` Benchmarks
:link: benchmarks
:link-type: doc

The measured numbers, how they were taken, and which parts are solid.
:::

:::{grid-item-card} {octicon}`tools;1em` Developer guide
:link: developer/index
:link-type: doc

Measurement method, runtime internals, the XLA fork, and open threads.
:::

::::

```{toctree}
:hidden:
:caption: Getting started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:hidden:
:caption: Guides

guides/exporting
guides/calling
guides/realtime
guides/measuring
guides/debugging
guides/how-it-works
```

```{toctree}
:hidden:
:caption: Examples

examples/index
examples/01-basic
examples/02-trajopt
examples/03-realtime
```

```{toctree}
:hidden:
:caption: Reference

api/index
api/cpp/runtime
api/cpp/function
api/cpp/rt
api/cpp/latency
api/cpp/alloc-guard
api/cpp/error
api/cpp/dtype
api/python
api/artifact-format
```

```{toctree}
:hidden:
:caption: Project

benchmarks
changelog
contributing
```

```{toctree}
:hidden:
:caption: Developer guide

developer/index
developer/measurement
developer/runtime-internals
developer/xla-fork
developer/bumping-jax
developer/integration-recipes
developer/release-process
developer/open-threads
```
