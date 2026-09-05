# call_jax_from_cpp

[![ci](https://github.com/jozbee/call_jax_from_cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/jozbee/call_jax_from_cpp/actions/workflows/ci.yml) [![docs](https://github.com/jozbee/call_jax_from_cpp/actions/workflows/docs.yml/badge.svg)](https://jozbee.github.io/call_jax_from_cpp/) [![license: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](LICENSE)

Write the numerics in JAX, export them ahead of time, and call them from a C++
loop whose worst call you can put a deadline on. A JAX function becomes a
serialized PJRT executable, StableHLO bytecode to fall back on, and a JSON
sidecar describing its signature; a C++ program loads all three through the PJRT
C API on CPU. The distinguishing part is the call path: load once, call many,
into 64-byte-aligned arenas the runtime owns, inputs wrapped once in zero-copy
PJRT buffers, outputs read straight out of device memory, and nothing in the
steady state that allocates, locks, logs or flushes. The figure of merit is
**tail latency** — p99.9/p50 and the worst call in a million — never the mean.

## Quickstart

```console
uv sync              # Python environment (uv comes from mise)
make plugin          # the prebuilt PJRT CPU plugin, sha256-verified
make                 # the library and the three examples
make export          # run the export scripts with JAX
make run-examples    # call and measure
```

```python
import jax, jax.numpy as jnp, jax2exec

jax.config.update("jax_enable_x64", True)


def step(x0, u0):
    return u0 - 0.1 * (u0 - x0[None, :2])


jax2exec.export(
    step,
    [
        jax.ShapeDtypeStruct((6,), jnp.float64),
        jax.ShapeDtypeStruct((40, 2), jnp.float64),
    ],
    "artifacts",
    "trajopt",
)  # -> trajopt.binpb/.mlirbc/.json
```

```cpp
pjrt::Runtime rt;                              // one per process
pjrt::Function f(rt, "artifacts/trajopt");     // load once
const std::size_t x0 = *f.find_input("x0");    // resolve names at startup
double* state = f.input<double>(x0);

for (;;) {
  std::memcpy(state, measured, f.input_nbytes(x0));
  f.call();
  const double* u = f.output<double>(0);       // valid until the next call
}
```

## Measured

One step of a nonlinear MPC controller (16 in / 14 out, all rank-1 float64) on
an idle aarch64 devcontainer; 28 interleaved runs × 4000 calls per API.

| | legacy, per-call buffers | `Runtime`/`Function` |
|---|---|---|
| median p50 | 4842 µs | **3979 µs** |
| median p99.9 | 5417 µs | **4417 µs** |
| worst single call | 20169 µs | **6832 µs** |
| worst max/p50 | 4.198 | **1.719** |
| runs with a >2x outlier | 2 of 28 | **0 of 28** |
| wrapper allocations per call | ~520 | **0** |

The **magnitude** difference is solid; the **frequency** claim is soft — two
outlier runs against zero at n=28 is not a significant difference and must not
be quoted as one. This is a container on aarch64, so it is a relative signal,
and it predates the shipped examples: [`examples/02_trajopt`](examples/02_trajopt)
is what you can reproduce today, and the
[benchmarks page](https://jozbee.github.io/call_jax_from_cpp/benchmarks.html)
carries the method, the caveats and the sign-off targets.

## What you get

- **Zero-copy, 64-byte-aligned arenas** owned by the runtime. Below
  `xla::cpu::MinAlign()` XLA silently falls back to copying, which is why the
  arena is not the caller's pointer.
- **An allocation-free call path, enforced by a test.** `make test-alloc`
  preloads an allocator interposer and gates on zero *wrapper* allocations per
  call. XLA's thunk runtime still allocates ~15,400 times per call inside the
  plugin, unreachable through the PJRT C API; that number is reported, not hidden.
- **Dtype-generic and rank-general**: `bool`, every integer width, `float32` and
  `float64`, at any rank, through `input<T>(i)` / `output<T>(i)`, plus an opt-in
  debug mode catching a bad index, a dtype mismatch or a re-entrant call — and
  costing nothing when it is off.
- **A `.mlirbc` fallback for the architecture lock.** A `.binpb` embeds machine
  code and is locked to the exporting machine; the bytecode compiles in-process
  instead, and an ISA guard notices before an illegal instruction does.
- **Sidecar v2**: per-array names, dtypes, shapes and byte counts, the JAX and
  jaxlib versions, the exporting host and its ISA level, artifact digests.
  `python -m jax2exec check <base>` reads it on a machine with no JAX.
- **Optional real-time helpers** — `lock_memory`, `harden_malloc`,
  `pin_current_thread`, `corral_xla_threads`, `set_realtime_priority` — each
  reporting whether it took effect, plus `tools/rt_check.sh` to audit the host.
- **Prebuilt plugins.** There is no official prebuilt CPU PJRT C-API plugin, so
  this project publishes its own per JAX version; `make plugin` downloads and
  verifies one, `make plugin-source` builds it from the fork.
- **Two patches in the XLA fork.** *LAPACK FFI kernels* linked into the plugin;
  without them anything lowering to `jnp.linalg.inv` fails at load with `No FFI
  handler registered for lapack_dgetrf_ffi on a platform Host`, because a bare
  plugin never registers what jaxlib registers on import. And *CPU plugin create
  options* plus an attribute advertising synchronous execution; without them
  execution stays asynchronous — latency, never correctness. At this XLA version
  the plugin *validates* option names and rejects unknown ones, so the runtime
  asks `PJRT_Plugin_Attributes` first.

## Documentation

<https://jozbee.github.io/call_jax_from_cpp/> — quickstart, guides, the API
reference, the benchmarks and the developer guide (measurement method, runtime
internals, the XLA fork, open threads). Contributors start at
[CONTRIBUTING.md](CONTRIBUTING.md), agents at [AGENTS.md](AGENTS.md).

## Motivation

Numerics are easier to develop and debug in Python, but NumPy is slow and has no
automatic differentiation, which optimization problems want; JAX fixes both. The
other half is robotics control: for real-time performance and integration with
[ros2_control](https://github.com/ros-controls/ros2_control) the algorithms have
to live in a C++ program, and rewriting a JAX program in boutique C++ is exactly
the work this avoids. Nonlinear model-predictive control is the motivating
application, and the reason a late answer counts as a wrong one — but **no MPC
code ships here**.

## References

- [jax-ml/jax discussion 22184](https://github.com/jax-ml/jax/discussions/22184)
- [joaospinto/call_jax_from_cpp](https://github.com/joaospinto/call_jax_from_cpp/tree/main)
- [lc0's PJRT wrapper](https://github.com/LeelaChessZero/lc0/tree/97817028ae513cddd779abf606675c0808c353b2/src/neural/backends/xla)
- [gomlx/gopjrt](https://github.com/gomlx/gopjrt/tree/main)
- [OpenXLA: PJRT](https://openxla.org/xla/pjrt)

The name of this repository was stolen from `joaospinto`.
