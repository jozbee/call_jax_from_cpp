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

A JAX function is exported once to a serialized PJRT executable, StableHLO
bytecode to fall back on, and a JSON sidecar describing its signature. A C++
program loads all three through the PJRT C API on CPU and calls the function
in a loop. Nothing on the steady-state path allocates, locks, logs or flushes.

## What it is for

::::{grid} 1 1 3 3
:gutter: 3
:class-container: feature-grid

:::{grid-item-card} {octicon}`package;1.1em` Load once, call many
Every input and output gets a 64-byte-aligned arena the runtime owns. Inputs
are wrapped once in zero-copy PJRT buffers and outputs are read straight out
of device memory, so a call transfers nothing.
:::

:::{grid-item-card} {octicon}`pulse;1.1em` Built for the tail
The figure of merit is the worst call, not the average one: p99.9 relative to
p50, and the worst call in a long campaign. The numbers, and the machines they
came from, are on the {doc}`benchmarks page <benchmarks>`.
:::

:::{grid-item-card} {octicon}`clock;1.1em` Real-time ready
Optional helpers lock memory, pin the thread, move XLA's pools off its core
and raise scheduling priority. Each reports whether it took effect, and none
of them fails the program.
:::

::::

## Both halves of the loop

### Python: define and export

```{literalinclude} ../examples/01_basic/export.py
:language: python
:pyobject: fun
```

```{literalinclude} ../examples/01_basic/export.py
:language: python
:start-after: docs: begin export
:end-before: docs: end export
```

### C++: load and call

```{literalinclude} ../examples/01_basic/basic.cpp
:language: cpp
:start-after: docs: begin load
:end-before: docs: end load
```

```{literalinclude} ../examples/01_basic/basic.cpp
:language: cpp
:start-after: docs: begin call
:end-before: docs: end call
```

The export writes `<name>.binpb`, `<name>.mlirbc` and `<name>.json`; the C++
side is given the base path and reads all three. The sidecar is the only
description of the inputs that exists — the PJRT C API cannot be asked what
parameters an executable takes.

## Where next

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1em` Quickstart
:link: getting-started/quickstart
:link-type: doc

Export a function and call it from C++, end to end.
:::

:::{grid-item-card} {octicon}`download;1em` Installation
:link: getting-started/installation
:link-type: doc

The Python environment, the PJRT CPU plugin, and the C++ build.
:::

:::{grid-item-card} {octicon}`book;1em` Guides
:link: guides/how-it-works
:link-type: doc

How it works, exporting, calling, real-time hardening, measuring, debugging.
:::

:::{grid-item-card} {octicon}`plug;1em` Integrate
:link: guides/integration
:link-type: doc

A ROS 2 `ament_cmake` package, a CMake submodule, FetchContent, or plain Make.
:::

:::{grid-item-card} {octicon}`list-unordered;1em` API reference
:link: api/index
:link-type: doc

The C++ classes, the Python exporter, and the artifact format.
:::

:::{grid-item-card} {octicon}`tools;1em` Developer guide
:link: developer/index
:link-type: doc

Measurement, runtime internals, the benchmarks, the XLA fork.
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

guides/how-it-works
guides/exporting
guides/calling
guides/integration
guides/realtime
guides/measuring
guides/debugging
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
:caption: Developer guide

developer/index
developer/measurement
developer/runtime-internals
developer/exporter-internals
developer/realtime-notes
benchmarks
developer/xla-fork
developer/bumping-jax
developer/release-process
developer/open-threads
```

```{toctree}
:hidden:
:caption: Project

changelog
contributing
```
