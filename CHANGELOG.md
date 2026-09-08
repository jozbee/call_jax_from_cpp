# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- A dtype-generic, rank-general C++ API: `float64`, `float32`, every integer
  width and `bool`, at any rank. Typed accessors `input<T>(i)` / `output<T>(i)`
  sit alongside `input_raw(i)` for generic code.
- An opt-in debug mode (`FunctionOptions::debug`) that bounds-checks indices,
  rejects a typed accessor whose `T` disagrees with the declared dtype, and
  catches a re-entrant `call()`. With `check_values` it also scans float arenas
  for non-finite values and bool arenas for bytes other than 0 and 1. All of it
  costs nothing when debug is off.
- Artifact format v2: the sidecar records per-array names, dtypes, shapes and
  byte counts, the JAX and jaxlib versions, the exporting host and its ISA
  level, and artifact checksums. Version 1 sidecars still load.
- `.mlirbc` StableHLO bytecode alongside the serialized executable, so a
  `.binpb` that will not load on this machine (the architecture lock) can be
  compiled in-process instead. `Function::load_kind()` reports which route ran.
- The PJRT CPU plugin is now `dlopen`-ed at run time rather than linked, and is
  located by `RuntimeOptions::plugin_path`, `$PJRT_CPU_PLUGIN`, or a build-time
  default.
- Machinery for prebuilt plugin binaries as GitHub Release assets per JAX
  version: `make plugin` downloads and checksums one, `make plugin-source`
  builds it from the XLA fork. The first release, `plugin-jax-v0.11.1`,
  published a `linux-x86_64` asset. Nothing is published yet for the pinned
  `plugin-jax-v0.11.0`, so until it is, `make plugin` prints the remedies and
  exits non-zero on every platform.
- `pjrt::LatencyRecorder` and `pjrt::AllocGuard` promoted to public headers, so
  a caller can measure their own loop with the same tools this project uses.
- Four examples: a basic one, a trajectory-optimisation workload heavy enough
  to time, a minimal hardened periodic loop in one self-contained file, and the
  same loop instrumented to report jitter, wake-up latency, deadline misses,
  page faults, context switches and allocations.
- A Background section — real-time on Linux, XLA and PJRT, latency and tails,
  a glossary — one paragraph per mechanism, linking out; every guide and
  example page opens with what it assumes and links every foreign name.
- A reference page for the examples' shared layer (`cjfc`), and an extension
  that links C++ names inside code blocks to their reference entries.
- CMake support (`pjrt_exec::pjrt_exec`) beside the Makefile, plus a
  `pjrt_exec.mk` fragment for Make-based consumers.
- A documentation site built with Sphinx and published to GitHub Pages, with
  a logo: a lambda, an arrow, and angle brackets.

### Changed
- The trajopt workload's I/O helpers moved from `cjfc::` into
  `cjfc::workload::` (`examples/common/workload.hpp`), so the code says which
  helpers are the workload's contract and which are generic.
- The instrumented real-time loop is example 04; example 03 is now the minimal
  loop. The docs build is `nitpicky`: a dead cross-reference fails it.
- The documentation was reorganised for brevity. The guides keep the main
  ideas and end with a "Deeper" line; the reasoning moved under the developer
  guide, which gained `exporter-internals` and `realtime-notes`. The
  integration recipes moved under Guides with ROS 2 first, the real-time
  guide opens with a menu of every helper, and the example pages show marked
  regions of the code rather than whole files.
- Measured figures now live only under the developer guide and on the
  benchmarks page, and every one names the machine it came from or says the
  host was not recorded. `tests/test_docs_sync.py` checks the placement.
- JAX pin moved from 0.9.0.1 to 0.11.0 (jaxlib must match exactly), which moves
  the XLA base to `131bf41a`. The fork's two patches were re-applied on top.
  The pin stops at 0.11.0 rather than the newer 0.11.1 deliberately:
  [jax-ml/jax#40101](https://github.com/jax-ml/jax/issues/40101) is an XLA:CPU
  codegen regression, from 0.11.1 onward, that makes a `dynamic-update-slice`
  inside a loop body cost time proportional to the whole destination buffer
  rather than to the slice written. jaxlib compiles the machine code an
  artifact embeds, so the defect is baked in at export and no plugin-side
  patch removes it; pinning jaxlib is the fix. Measured on the development host,
  the change makes **no difference to the shipped workload**: this host
  reproduces the regression at three to four orders of magnitude on the
  reporter's own case, and `examples/02_trajopt` moved by 2 µs across 12,000
  interleaved calls, because its loop trip counts are far too short to pay for
  it. The pin is insurance for future fixtures, not a speed-up. The developer
  guide's open threads page carries both measurements.
- `docker compose run --rm plugin-builder tools/build_plugin.sh` works again.
  The `bazel-cache` volume mounts at `/home/dev/.cache/bazel`, and docker
  creates a mount point's missing parents as root, so bazelisk could not write
  `~/.cache/bazelisk` and the build died before it started; `dev` could not
  sync its venv for the same reason. The image now creates both directories as
  `dev`. The Dockerfile already guarded against the identical hazard for its
  build-time cache mount; the runtime volume was not covered.
- `PLUGIN_INFO.txt` from a container build now names the fork commit and
  branch. A submodule's `.git` is a file pointing into the superproject, which
  is not mounted at `/xla`, so git found nothing and the fields read `unknown`
  — in exactly the artifact whose whole purpose is to be auditable later.
  `tools/build_plugin.sh` falls back to `versions.env` and says when it does.
- Building the plugin from source on a distribution other than Debian or
  Ubuntu now needs `--arch-flags "--linkopt=-L/usr/lib"`, and the fork records
  why a general library directory must not be added to the patch instead: it is
  searched ahead of the hermetic sysroot for every implicit library too, and
  was measured to move the plugin's requirement from `GLIBC_2.27` to
  `GLIBC_2.44`.
- The plugin now negotiates create options: at this XLA version the CPU plugin
  validates option names and rejects unknown ones, so the runtime asks
  `PJRT_Plugin_Attributes` what is supported before creating a client, and
  retries without an option the plugin refuses. `Runtime::synchronous_mode()`
  reports what actually took effect.
- `input_size(i)` became `input_numel(i)`, and a scalar is now 1 element rather
  than the 0 that version 1 sidecars encode.
- `examples/02_trajopt/export.py` takes `--preset default|small` in place of
  the `--time` and `--stats` tuning aids. `default` is the workload the
  benchmark, example 04 and the tests all describe and is unchanged; `small`
  exports the same model at a quarter of the work, for a smoke test rather
  than for a number. The refusal that keeps `cost_history` non-increasing
  stayed; the advice about which constant to halve did not.
- Each example is now a call path and a `support.hpp` beside it. The `.cpp`
  holds the sequence a caller performs and nothing else; the flags, the host
  audit, the summaries and the reports moved into the header.
  `examples/common/` gained `periodic.hpp` (the absolute sleep and the stop
  flag a periodic loop needs) and `names.hpp` (the `SyncMode` and `LoadKind`
  spellings, so a program can name what it loaded without pulling in JSON).

### Fixed
- `pjrt::rt::corral_xla_threads()` finds XLA's pool threads again. It matched
  thread names by the prefix `XLAEigen` / `XLAPjRtCpuClient`, and TSL prefixes
  the name it sets, so at the pinned XLA version the threads are called
  `tf_XLAEigen` and the search matched nothing on any host. The step reported
  "no XLA worker threads found (client not created?)", which reads as nothing
  to do rather than as a helper that had stopped working, so example 04
  printed `[skip]` and no test objected. The match is now a substring test for
  `XLA`, which also covers the un-prefixed names older versions used, and the
  calling thread is excluded by thread id so the corral can never undo its own
  pinning. `tests/test_examples_realtime.py` now asserts that a two-worker run
  moves two threads.
- `make plugin-source` and `tools/release_plugin.sh` can build the plugin
  again. BuildKit creates a cache mount's missing parent directories as root,
  so mounting the bazel cache at `/home/dev/.cache/bazel` left `/home/dev/.cache`
  owned by root, and bazelisk -- which writes `~/.cache/bazelisk` beside it --
  died with "permission denied" before compiling anything. The release path
  had never been exercised, because no plugin release existed to exercise it.
- The library compiles again under the clang the CI container ships.
  `host_isa_level()` asked `__builtin_cpu_supports` for `"lzcnt"` and
  `"movbe"`, and the set of names that builtin accepts has grown over compiler
  releases: clang 18 rejects both at compile time rather than answering false,
  so every x86_64 CI job failed while the newer clang on the developer machine
  built the same file cleanly. Both bits are read from CPUID directly now, so
  the x86-64-v3 test still checks the full psABI feature set.

### Removed
- The legacy `Client` / `Buffer` / `AOTComputation` wrappers, which rebuilt
  device buffers on every call and were the original source of the jitter this
  project exists to remove.
- The vendored lc0 PJRT wrapper and the raw C API example.
- All MPC-specific code, fixtures and tooling. Nonlinear MPC remains the
  motivation described in the documentation; none of it ships as code.
