# Installation

Three things have to be in place before anything runs: a Python environment
holding JAX {{ jax_version }} and the exporter, a PJRT CPU plugin built for
that same JAX version, and a C++ toolchain for the caller. Only the plugin is
unusual, and it has a section of its own below.

## Supported platforms

Linux is the deployment target. macOS builds and runs and is useful for
development, but provides none of the real-time hardening.

| Platform | Status | Plugin | `pjrt::rt` helpers |
|---|---|---|---|
| Linux x86-64 | Supported target | prebuilt or from source | all |
| Linux aarch64 | Supported target | prebuilt or from source | all |
| macOS (arm64, x86-64) | Development and correctness only | from source only | none: each returns `Status{false, "not supported on this platform"}` |
| Windows | Unsupported | — | — |

## Requirements

| | |
|---|---|
| C++ compiler | C++17. `clang++` is the default; `make CXX=g++` works. |
| Build | GNU make. CMake ≥ 3.20 is an alternative, not a requirement. |
| Python | ≥ 3.12, with `uv` to manage the environment. |
| JAX | {{ jax_version }} with jaxlib {{ jaxlib_version }} — the versions must match **exactly**. |
| For `make plugin-source` | bazel or bazelisk, `liblapack-dev`, `libblas-dev`, and 30–60 minutes. |
| For `make docs` | doxygen and graphviz; the C++ pages are generated from the headers. |

Nothing links against the plugin, so a plain `make` needs neither bazel nor
the network; only the *run* targets care whether it is there.

## Getting the code

```console
$ git clone https://github.com/jozbee/call_jax_from_cpp.git
$ cd call_jax_from_cpp
```

The XLA fork is a submodule and is needed only to build the plugin from
source:

```console
$ git submodule update --init --depth 1 third_party/xla
```

## The Python environment

```console
$ uv sync
```

That resolves `pyproject.toml` into `.venv`, pinning jax {{ jax_version }} and
jaxlib {{ jaxlib_version }}. Add `--extra docs` if you also want to build this
site.

:::{note}
`uv` comes from [mise](https://mise.jdx.dev/), which puts it on `PATH` only in
an interactive shell. In a script, a CI step or an editor task, `uv: command
not found` is the usual first failure. Either run through `mise exec -- ...`,
or override the interpreter the Makefile uses:

```console
$ make PYTHON=python3 PYTEST=pytest test
```
:::

The exporter needs JAX; the C++ caller does not, and neither does
`python -m jax2exec check`, which is deliberately importable on a deployment
machine that has no JAX at all.

## The PJRT CPU plugin

**There is no official prebuilt CPU PJRT C-API plugin, from anyone.** jaxlib
links its CPU client statically and never exports `GetPjrtApi`, so the shared
object this project `dlopen`s does not exist in any JAX release. That is why
the project builds and publishes its own, from an XLA fork that adds two things
a stock plugin does not have: jaxlib's LAPACK FFI kernels, and the create
option that makes execution inline. See {doc}`../developer/xla-fork`.

Pick one of three routes.

### A. The prebuilt plugin

```console
$ make plugin
```

`tools/get_plugin.sh` looks up the asset for this platform and release
{{ plugin_release }} in `tools/plugin_versions.txt`, downloads it, checks it
against the recorded sha256, unpacks it into `build/plugin/`, and confirms the
result exports `GetPjrtApi`. A checksum mismatch is a hard failure, never a
warning.

When no asset has been published for your platform at that release, the script
says so and prints the remedies rather than guessing; that is the point at
which route B or C applies.

Verify:

```console
$ tools/get_plugin.sh --check
get_plugin: build/plugin/libpjrt_c_api_cpu_plugin.so looks like a PJRT plugin
```

### B. From source

```console
$ make plugin-source
```

`tools/build_plugin.sh` builds `//xla/pjrt/c:pjrt_c_api_cpu_plugin` out of
`third_party/xla` with bazel and writes the result to `build/plugin/`. The
first build takes 30–60 minutes and a few gigabytes of bazel cache; later ones
are incremental. It needs `liblapack-dev` and `libblas-dev`, which the fork's
first patch links against.

**On a distribution that is not Debian or Ubuntu**, the link step fails with
`cannot find -llapack`, because the patch adds only Debian's multiarch
directories to the link path and adding a general one would relink the C
runtime against the host's glibc. Give the path for that build alone:

```console
$ tools/build_plugin.sh --arch-flags "--linkopt=-L/usr/lib"
```

The flag is recorded in `PLUGIN_INFO.txt`, and a plugin built that way runs
here but is not portable enough to publish. {doc}`../developer/xla-fork` has the
measurement behind that.

If bazel is not on this machine, the container has it:

```console
$ docker compose -f docker/compose.yml run --rm plugin-builder tools/build_plugin.sh
```

Build mode does not change latency — the compute kernels are compiled at
export time and the plugin only orchestrates — so `opt` is the default only
because it is what gets published. {doc}`../developer/xla-fork` has the
measurement.

### C. A plugin you already have

```console
$ export PJRT_CPU_PLUGIN=/path/to/libpjrt_c_api_cpu_plugin.so
```

`RuntimeOptions::plugin_path` wins over this, and this wins over the path
compiled in at build time. A stock plugin — one built from unpatched XLA — does
work, with two losses:

- **Inline execution is unavailable.** The `asynchronous` create option is
  either rejected or unadvertised, so the dispatch hand-off stays in the call
  path. This costs latency, never correctness.
- **`jnp.linalg.*` will not load.** A stock plugin registers no FFI handlers,
  so an executable that lowers to `lapack_dgetrf_ffi` fails at load with *"No
  FFI handler registered for lapack_dgetrf_ffi on a platform Host"*.

The tell is `Runtime::synchronous_supported()`, and the probe prints it:

```console
$ make tools && build/bin/plugin_probe
plugin_path=/path/to/libpjrt_c_api_cpu_plugin.so
api_version=<major>.<minor>
platform_name=cpu
sync_mode=inline
synchronous_supported=1
advertises_synchronous_execution=1
```

`sync_mode=rejected` or `advertises_synchronous_execution=0` means route C,
whatever the file is called. `api_version` must agree with `PJRT_API_MAJOR` and
`PJRT_API_MINOR` in the matrix at the bottom of this page.

## Building

```console
$ make            # the library and the four C++ examples; 05 builds with colcon
$ make examples   # just the examples
$ make bench      # build and run the benchmark
$ make test       # build everything, then run the suite
$ make help       # every target, with one line each
```

Outputs land in `build/lib/libpjrt_exec.a`, `build/bin/`, `build/plugin/`, and
exported artifacts in `artifacts/` with reports under `artifacts/reports/`.
`make print-config` prints the resolved compiler, paths and versions when a
build does something unexpected.

CMake is supported for the same tree:

```console
$ cmake -B build -DCMAKE_BUILD_TYPE=Release
$ cmake --build build -j
```

## Docker

The container exists because two of the real-time helpers need privileges a
default container does not have: {term}`SCHED_FIFO` needs `CAP_SYS_NICE`
plus an `rtprio` limit, and {term}`mlockall <memory locking>` needs an
unlimited `memlock` — the {term}`rlimits`. `docker/compose.yml`
grants both to `dev` and `ci`.

```console
$ docker compose -f docker/compose.yml run --rm dev            # interactive
$ docker compose -f docker/compose.yml run --rm ci make test   # the CI path
```

The same file backs `.devcontainer/`, so an editor session gets the same
privileges. Container timings are relative signals only — see
{doc}`../guides/realtime`.

## Vendoring into your own project

The library is vendored per project — a ROS 2 package, a CMake submodule,
`FetchContent`, or copied sources with a Make fragment — and never installed
system-wide. {doc}`../guides/integration` has one recipe per route, ROS 2
first.

## Verify the install

Export the first example's artifact and run it:

```console
$ make export
$ build/bin/example_01_basic
load_kind=deserialized
synchronous_supported=1
sync_mode=inline
num_inputs=2 num_outputs=2
input[0]: dtype=float64 shape=[4,4] numel=16 nbytes=128
input[1]: dtype=float64 shape=[4] numel=4 nbytes=32
output[0]: dtype=float64 shape=[4] numel=4 nbytes=32
output[1]: dtype=float64 shape=[] numel=1 nbytes=8
x=[...]
residual_inf_norm=...
residual_from_jax=...
```

Example 01 solves a 4x4 dense linear system and checks the answer twice: once
against a residual it recomputes from its own arenas, once against the
residual XLA computed inside the same executable. Four lines are worth
reading:

| Line | Should say | If it does not |
|---|---|---|
| `load_kind` | `deserialized` | `compiled` means the ISA guard sent the loader to the `.mlirbc`; see {doc}`../guides/debugging`. |
| `sync_mode` | `inline` | `rejected` or `accepted` means you are on a stock plugin (route C). |
| `input[0]` | `dtype=float64` | `float32` means the export ran without `jax_enable_x64`. |
| the two residuals | both near zero | A correctness failure, not a setup one — worth a bug report. |

The example is also a deliberate test of the plugin: `jnp.linalg.inv` lowers
to a LAPACK custom call, so against a stock plugin it fails at `pjrt::Function`
construction with *"No FFI handler registered for lapack_dgetrf_ffi"* — here,
rather than in the field.

Add `--debug` to see the debug-mode checks fire on purpose:

```console
$ build/bin/example_01_basic --debug
...
debug_check[out_of_range]: input index 99 is out of range: function 'basic' has 2 inputs
debug_check[dtype_mismatch]: input 0 ('A') has dtype float64 but was accessed as float32
debug_check[non_finite]: input 0 ('A') element 0 is nan
```

If the run fails instead, {doc}`../guides/debugging` has one row per message
this project can produce.

## Compatibility matrix

Every pinned version lives in one file. The Makefile, CMake, CI and this page
all read it rather than repeating a number that would then rot.

```{literalinclude} ../../versions.env
:language: ini
:caption: versions.env
```

A serialized executable is not portable across a JAX bump, so a change to
`JAX_VERSION` means re-exporting artifacts and re-fetching the plugin. The
order of operations is in {doc}`../developer/bumping-jax`.
