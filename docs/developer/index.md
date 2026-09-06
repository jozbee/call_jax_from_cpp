# Developer guide

Notes for working on the tree rather than with it. Most of what is surprising
about this project was established by measurement rather than by reading
documentation, and these pages record the finding together with the mistake
that produced it. The rest of the site describes the library; this section
describes how it got that way and what a change to it has to respect.

## Read in this order

1. {doc}`measurement` — read before producing any latency number. The traps
   invalidate results silently.
2. {doc}`runtime-internals` — the PJRT C API in one page, the verified
   XLA:CPU behaviour, and why the hot path is shaped the way it is.
3. {doc}`exporter-internals` — what `export` checks before it writes, and the
   traps it catches.
4. {doc}`realtime-notes` — the host settings, the idle-period experiment,
   PREEMPT_RT, cyclictest, and `apply_hardening` in full.
5. {doc}`../benchmarks` — every measured figure, with the machine it came from.
6. {doc}`xla-fork` — the two patches, what breaks without them, and what a
   rebase must preserve.
7. {doc}`bumping-jax` — the checklist, with a verify line per step.
8. {doc}`release-process` — cutting a release, and publishing a plugin.
9. {doc}`open-threads` — unfinished work, and the caveats that will bite.

## Repo map

```
include/pjrt_exec/   Public headers. runtime.hpp is the one to read first.
src/pjrt_exec/       Implementation: runtime.cpp, rt.cpp, isa.cpp.
python/jax2exec/     The exporter, the reference-case writer, the check command.
examples/            01_basic, 02_trajopt, 04_realtime, and common/ (shared
                     helpers: the CLI parser, host audit, periodic loop, report).
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
   noisy, it is wrong. See {doc}`measurement`.
2. **Prove plugin behaviour with a probe; do not infer it from the headers.**
   Several plausible-sounding PJRT facts turned out false here, and several
   implausible ones true. `tools/plugin_probe.cpp` is where the probes live.
3. **Create options are validated now.** At this XLA version the CPU plugin
   rejects an option name it does not know, where older plugins ignored
   unknown options silently. Ask `PJRT_Plugin_Attributes` what is supported
   *before* creating a client, and send only what the plugin admits to
   understanding.
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
8. **Measured figures live only on this section's pages and the benchmarks
   page**, and every one names the machine it came from. A guide states the
   shape of a result and links here.

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
