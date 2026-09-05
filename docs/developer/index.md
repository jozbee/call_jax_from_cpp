# Developer guide

Notes for working on the tree rather than with it. Most of what is surprising
about this project was established by measurement rather than by reading
documentation, and these pages record the finding together with the mistake
that produced it. The rest of the site describes the library; this section
describes how it got that way and what a change to it has to respect.

## The one-paragraph version

A JAX function is exported ahead of time to a serialized PJRT executable, a
JSON sidecar describing its signature, and StableHLO bytecode to fall back on.
A C++ program loads that artifact through the PJRT **C** API on CPU and calls
it in a loop. The motivating application is a robotics control loop, so the
figure of merit is **the worst call in a million, not the average** — p99.9/p50
and the rare spike. XLA:CPU was never the source of the jitter that motivated
the work: an earlier wrapper was, by rebuilding device buffers on every call.
`pjrt::Runtime` / `pjrt::Function` is the fix — load once, call many, allocate
nothing in the steady state.

## Read in this order

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 1 · Measurement
:link: measurement
:link-type: doc

Read before producing any latency number. The traps invalidate results
silently.
:::

:::{grid-item-card} 2 · Runtime internals
:link: runtime-internals
:link-type: doc

Verified PJRT and XLA:CPU behaviour, what follows from it, and the approaches
that were rejected.
:::

:::{grid-item-card} 3 · The XLA fork
:link: xla-fork
:link-type: doc

Two patches, what breaks without them, and what a rebase has to preserve.
:::

:::{grid-item-card} 4 · Bumping JAX
:link: bumping-jax
:link-type: doc

The checklist, step by step, with a verify line for each.
:::

:::{grid-item-card} 5 · Integration recipes
:link: integration-recipes
:link-type: doc

Dropping this into somebody else's build, in minutes.
:::

:::{grid-item-card} 6 · Release process
:link: release-process
:link-type: doc

Cutting a release, and publishing a plugin to go with it.
:::

:::{grid-item-card} 7 · Open threads
:link: open-threads
:link-type: doc

Unfinished work, and the caveats that will bite someone who assumes otherwise.
:::

::::

## The data flow

Three artifacts leave Python and one process reads all three. The sidecar is
not a convenience: the PJRT C API cannot be asked what parameters an
executable takes, so it is the only description of the signature that exists.

```{mermaid}
flowchart LR
  subgraph py["Python, once"]
    fn["JAX function"] --> jit["jax.jit(...).lower(...).compile()"]
    jit --> binpb[".binpb<br/>serialized executable"]
    jit --> mlirbc[".mlirbc<br/>StableHLO bytecode"]
    jit --> json[".json<br/>sidecar: names,<br/>dtypes, shapes, ISA"]
  end

  subgraph cpp["C++, load time"]
    plugin["PJRT CPU plugin<br/>dlopen, GetPjrtApi"] --> runtime["pjrt::Runtime<br/>one client"]
    runtime --> func["pjrt::Function<br/>arenas + zero-copy buffers"]
  end

  binpb -->|"deserialize (relink)"| func
  mlirbc -.->|"compile, if the binary<br/>will not run here"| func
  json --> func
  func --> loop["call() in a loop<br/>execute, one await,<br/>one memcpy per output"]
```

## Repo map

```
include/pjrt_exec/   Public headers. runtime.hpp is the one to read first.
src/pjrt_exec/       Implementation: runtime.cpp, rt.cpp, isa.cpp.
python/jax2exec/     The exporter, the reference-case writer, the check command.
examples/            01_basic, 02_trajopt, 03_realtime, and common/cli.hpp.
bench/               The measurement spine: one configuration, measured once.
tests/               python/ (pytest), cpp/ (test binaries), support/ (the
                     preloadable allocation counter).
tools/               get_plugin.sh, build_plugin.sh, rt_check.sh, run_matrix.sh,
                     plugin_probe.cpp.
cmake/               The CMake package, and the plugin download at configure time.
docker/              dev, ci and plugin-builder images.
third_party/         Vendored PJRT headers and JSON, the fork patches, and the
                     xla submodule.
docs/                This site. conf.py reads every version out of versions.env.
.github/workflows/   ci, docs and plugin.
artifacts/           Exported artifacts and reports. Gitignored: they are
                     machine-specific and do not travel.
build/               bin/, lib/, plugin/. Gitignored.
```

## Rules of thumb

These hold across the codebase, and each one was learned by breaking it.

1. **Never report a latency number measured on a busy machine.** It is not
   noisy, it is wrong. See [measurement](measurement.md).
2. **Prove plugin behaviour with a probe; do not infer it from the headers.**
   Several plausible-sounding PJRT facts turned out false here, and several
   implausible ones true. The pattern that works is to write a probe and run
   it — `tools/plugin_probe.cpp` is where those live.
3. **Create options are validated now.** At this XLA version the CPU plugin
   rejects an option name it does not know with
   `InvalidArgument("Unexpected option name passed to PJRT_Client_Create")`,
   where older plugins ignored unknown options silently. So ask
   `PJRT_Plugin_Attributes` what is supported *before* creating a client, and
   send only what the plugin admits to understanding.
4. **`.binpb` artifacts are locked to the exporting machine.** Loading relinks
   embedded machine code; it never recompiles. The `.mlirbc` is the answer when
   the binary will not run here.
5. **The steady-state call path must not allocate, lock, log or flush.** All
   four were present in the wrapper this project replaced.
6. **Run with `FunctionOptions::debug` on in development.** With it off, the
   same mistake is silent memory corruption rather than an exception, and the
   checks cost nothing when they are off.
7. **Never hand-type a version.** `versions.env` is the single source of truth,
   and it is available in any page here as a MyST substitution.

## Building these pages

The build runs with `-W`, so a warning is a failure: an orphan page, a broken
cross-reference or an unknown theme option stops the site rather than quietly
degrading it.

```
uv sync --extra docs     # uv comes from mise, and is not on PATH in a
                         # non-interactive shell
make docs                # one build; needs doxygen on PATH
make docs-live           # rebuild and reload in a browser as you edit
make docs-linkcheck      # verify every external link
```

`docs/conf.py` runs Doxygen at import time, so the XML that Breathe reads
always matches the headers on disk, and `make docs-live` regenerates it as a
header changes. If `doxygen` is not installed, build in the container instead:
`docker compose -f docker/compose.yml run --rm dev make docs`.
