# AGENTS.md

Instructions for an agent working in this repository. Read this before
changing anything.

## Read first

| | |
|---|---|
| `docs/developer/index.md` | The engineering primer: what was established by measurement rather than by reading documentation, and the mistakes made getting there. |
| `docs/developer/measurement.md` | **Read before producing any latency number.** Every trap in it produces numbers that look plausible and are meaningless. |
| `docs/developer/runtime-internals.md` | Verified PJRT/XLA:CPU behaviour, and why the hot path is shaped the way it is. |
| `docs/developer/xla-fork.md` | The two patches the fork carries, and what a rebase must preserve. |
| `docs/developer/open-threads.md` | Unfinished work and environment caveats. |
| `versions.env` | Every pinned version. Never hand-type one anywhere else. |

## The two rules that override normal instincts

**1. Never report a latency number measured on a busy machine.** A concurrent
build does not add noise to the result, it invalidates it: the same
configuration measured during a bazel build reported a p50 **2.4x high** and a
max/p50 of **4.4 instead of 1.1**. Check `/proc/loadavg` first, and discard a
contaminated run rather than correcting it. When you are asked for latency
numbers, run `tools/rt_check.sh`, read `/proc/loadavg`, and include both in the
answer.

**2. The goal is the tail, not the mean.** p99.9/p50, max/p50, and the worst
call in a long campaign. Do not substitute an average-latency objective, and do
not offer a mean improvement as a result. Rare spikes need long *campaigns*,
not long runs: a >2x max/p50 outlier appears roughly once per **20,000+ calls**,
so a claim about spike *frequency* needs at least that many calls, interleaved
between the configurations being compared — a sequential A/B drifts with CPU
temperature by the same order as the effect being measured.

## Working rules

**The public API is `pjrt::Runtime` / `pjrt::Function`, plus `pjrt::rt`.**
`include/pjrt_exec/runtime.hpp` is the call path; the real-time helpers are
optional and each reports whether it took effect. Anything in a call path uses
it: load once, call many.

**Nothing in `call()` may allocate, lock, log or flush.** All four were present
in the original wrapper and all four are gone. `make test-alloc` enforces it
with a preloaded allocator interposer, and the number that must stay at zero is
the wrapper's own allocations — not the process's. XLA's thunk runtime still
allocates ~15,400 times per call inside the plugin, roughly one per StableHLO
op, and that is not reachable through the PJRT C API.

**Artifacts are architecture-locked.** A `.binpb` embeds machine code;
`LoadSerializedExecutable` relinks it and never recompiles. Answer a load
failure with the `.mlirbc` fallback or a re-export on the target machine.
**Never by disabling `check_metadata`** — that check is what keeps a stale
sidecar from becoming a heap overrun discovered as corrupted output.

**A fork patch must exist in three places before anything depends on it**: a
pushed branch referenced by hash, a regenerated `third_party/patches/*.patch`
from `git format-patch <XLA_COMMIT>..HEAD`, and the hash written into
`versions.env`. A rebase is not finished until all three agree. The reason is
concrete: a previous patch was committed locally and referenced by the
submodule pointer but **never pushed**, so the commit could not be fetched by
anyone, including its author.

**Prove PJRT behaviour with a probe; do not infer it from headers.** Several
plausible-sounding facts about this plugin turned out to be false, and several
implausible ones true. Write a probe and run it — and make sure the probe is
sensitive to what it perturbs, because a probe that cannot detect a positive
reports a confident negative.

**Docs build with `-W`, so a warning is a failure**, and code in the
documentation comes from marked regions (`docs: begin <name>` /
`docs: end <name>`) via `literalinclude`, never pasted. A pasted snippet is a
copy that rots silently.

## Where things are

| Path | What is in it |
|---|---|
| `include/pjrt_exec/` | The public headers: `runtime.hpp` (Runtime, Function, Error, LoadError), `dtype.hpp`, `rt.hpp`, `latency.hpp`, `alloc_guard.hpp`. |
| `src/pjrt_exec/` | The implementation: `runtime.cpp`, `rt.cpp`, `isa.cpp`. |
| `python/jax2exec/` | The exporter, the sidecar writer, the dtype table, ISA detection, and the `check` CLI. |
| `examples/` | `01_basic`, `02_trajopt` (the workload for the benchmark **and** for example 03), `03_realtime`, and `common/cli.hpp`. |
| `bench/` | The measurement spine. Every runtime change is judged by its output. |
| `tests/` | `cpp/`, `python/`, and `support/malloc_guard.c` — preloaded, never linked. |
| `tools/` | `get_plugin.sh`, `build_plugin.sh`, `rt_check.sh`, `run_matrix.sh`, `plugin_probe.cpp`. |
| `docs/` | The Sphinx site. `developer/` is the engineering primer; the rest is user-facing. |
| `third_party/xla` | The fork submodule. See `versions.env` for the branch and commit. |
| `versions.env` | The single source of truth for every pinned version. |
| `artifacts/` | Export output and run reports. Gitignored: `.binpb` files belong to the machine that made them. |
| `build/` | `bin/`, `lib/`, `plugin/`. `make clean` keeps the plugin; `make distclean` does not. |

## Commands

```console
uv sync                  # Python environment (uv comes from mise, and is not
                         # on PATH in a non-interactive shell)
make plugin              # download the prebuilt PJRT CPU plugin, sha256-verified
make plugin-source       # build it from the XLA fork with bazel (30-60 min)

make                     # the library and the three examples
make examples            # just the examples
make bench               # build and run the benchmark
make export              # run the export scripts with JAX
make run-examples        # export, then run all three examples

make test                # the fast suite
make test-slow           # including the long campaigns
make test-rt             # real-time gates; tuned, idle host only
make test-alloc          # the zero-wrapper-allocations gate

make docs                # build the site (warnings are errors; needs doxygen)
make docs-live           # rebuild and reload while editing
make docs-linkcheck      # verify every external link

tools/rt_check.sh        # audit this host's real-time settings, read-only
make help                # every target
make print-config        # the resolved build configuration
```

Everything also runs in the container, which already has doxygen and bazel:

```console
docker compose -f docker/compose.yml run --rm dev   # or: ci, plugin-builder
```

## House voice

Precise, quietly explanatory, no marketing. Comments and prose explain *why*.
One idea per sentence. State facts and their provenance — "measured", "verified
against the plugin" — rather than adjectives. Never oversell: the honesty notes
about what is solid and what is soft are part of the value, not a hedge to be
edited out.
