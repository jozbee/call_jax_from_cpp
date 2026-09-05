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
- Prebuilt plugin binaries published as GitHub Release assets per JAX version,
  with `make plugin` to download and verify one and `make plugin-source` to
  build it from the XLA fork.
- `pjrt::LatencyRecorder` and `pjrt::AllocGuard` promoted to public headers, so
  a caller can measure their own loop with the same tools this project uses.
- Three examples: a minimal one, a trajectory-optimisation workload heavy
  enough to time, and a periodic real-time control loop that reports jitter,
  wake-up latency, deadline misses, page faults, context switches and
  allocations.
- CMake support (`pjrt_exec::pjrt_exec`) beside the Makefile, plus a
  `pjrt_exec.mk` fragment for Make-based consumers.
- A documentation site built with Sphinx and published to GitHub Pages.

### Changed
- JAX pin moved from 0.9.0.1 to 0.11.1 (jaxlib must match exactly), which moves
  the XLA base to `dcf304bc`. The fork's two patches were re-applied on top.
- The plugin now negotiates create options: at this XLA version the CPU plugin
  validates option names and rejects unknown ones, so the runtime asks
  `PJRT_Plugin_Attributes` what is supported before creating a client, and
  retries without an option the plugin refuses. `Runtime::synchronous_mode()`
  reports what actually took effect.
- `input_size(i)` became `input_numel(i)`, and a scalar is now 1 element rather
  than the 0 that version 1 sidecars encode.

### Removed
- The legacy `Client` / `Buffer` / `AOTComputation` wrappers, which rebuilt
  device buffers on every call and were the original source of the jitter this
  project exists to remove.
- The vendored lc0 PJRT wrapper and the raw C API example.
- All MPC-specific code, fixtures and tooling. Nonlinear MPC remains the
  motivation described in the documentation; none of it ships as code.
