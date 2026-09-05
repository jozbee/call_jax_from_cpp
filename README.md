# Calling JAX from C++

TLDR; we compile a [JAX](https://github.com/jax-ml/jax) function to a serialized executable, and we call this via the [PJRT](https://openxla.org/xla/pjrt) C API.

See [`docs/`](docs/README.md) for the engineering primer: measurement
methodology, verified PJRT/XLA:CPU behaviour, the fork patches, and open
threads.

## Motivation

I feel like it is easier to develop and debug numerics in Python and NumPy.
The problem is that Python is slow, and NumPy does not support automatic differentiation, which is helpful for optimization problems.
JAX helps solve these problems.

I am also interested in robotics control, and for real-time performance and integration with [ros2_control](https://github.com/ros-controls/ros2_control), I need my algorithms to be naturally embedded into a C++ program.
For hacking a project quickly, I don't want to rewrite the JAX programs in boutique C++ for extra performance.
JAX also solves this problem, but because this use case is niche, the documentation is poor.
This project provides some guiding examples for Float64 CPU integration.

## Calling it from a control loop

For a real-time caller, use `pjrt::Runtime` and `pjrt::Function` from
`src/pjrt_exec/runtime.hpp`. Load once, then call in a loop:

```cpp
pjrt::Runtime runtime;                                 // one per process
pjrt::Function mpc(runtime, "./artifacts/mpc_solver"); // load once

for (;;) {
  std::memcpy(mpc.input(3), acc_ref, 3 * sizeof(double));
  // ... write the rest of the inputs ...
  mpc.call();
  const double* u = mpc.output(0);
}
```

`input(i)` and `output(i)` point at 64-byte-aligned storage the `Function`
owns. The input arenas are wrapped once in zero-copy PJRT buffers, so a call
transfers nothing: XLA reads the arena in place, and the outputs are copied
straight out of device memory. Write the inputs *between* calls, never during
one — a `Function` is single-threaded by design.

`RuntimeOptions` controls the client: `synchronous` (run inline on the calling
thread instead of dispatching to a thread pool), `cpu_device_count`, and
`worker_threads` (XLA's pool size, applied through `PJRT_NPROC`).

The older `Client` / `Buffer` / `AOTComputation` wrappers still exist and still
work; they create device buffers on every call, which is fine for a script and
not what you want in a loop.

### Real-time hardening

`src/pjrt_exec/rt.hpp` has optional, independently-failing helpers for the
calling thread: `harden_malloc()`, `lock_memory()`, `pin_current_thread()`,
`corral_xla_threads()` (moves XLA's pools off your core) and
`set_realtime_priority()`. They are Linux-only and no-op elsewhere.
`tools/rt_check.sh` audits the host settings they depend on — governor,
`isolcpus`, `nohz_full`, transparent hugepages, `RLIMIT_RTPRIO`.

`SCHED_FIFO` and `mlockall` need privileges; the devcontainer grants them
(`cap_add: SYS_NICE`, `ulimits: rtprio/memlock`).

## Measuring

`make bench` builds and runs the benchmark; `make test_correct` checks both
APIs against reference fixtures; `make test_alloc` counts allocations in the
steady-state call path via a preloaded interposer.

```
make fixtures                      # export mpc_solver + synth_solver here
make bench BENCH_ARGS="--api rt --fixture mpc_solver --iterations 2000"
tools/run_matrix.sh mpc_solver 300 3   # sweep api x sync x threads
```

Two fixtures are exported. `mpc_solver` is the real acceptance workload
(16 in / 14 out, all rank-1 f64). `synth_solver` has the identical signature
and a comparable cost but no data-dependent control flow, so it separates
system jitter from the MPC's own bounded algorithmic variance.

Serialized executables embed target machine code — `LoadSerializedExecutable`
relinks it rather than recompiling — so `.binpb` files are locked to the
architecture that exported them. Run `make fixtures` on the machine that will
execute them.

**Measure on an idle machine.** A concurrent build does not add noise to these
numbers, it invalidates them: the same configuration measured during a bazel
build reported a p50 2.4x higher and a max/p50 of 4.4 instead of 1.1.

### What actually moved the numbers

MPC fixture on the aarch64 devcontainer, 28 runs of 4000 calls per API
(112,000 calls each), machine otherwise idle. Container numbers are relative
signals; absolute ones belong on the target hardware.

| | legacy, per-call buffers | `Runtime`/`Function` |
|---|---|---|
| median p50 | 4842 µs | **3979 µs** |
| median p99.9 | 5417 µs | **4417 µs** |
| worst single call | 20169 µs | **6832 µs** |
| worst max/p50 | 4.20 | **1.72** |
| runs with a >2x outlier | 2 of 28 | **0 of 28** |

- **Persistent zero-copy buffers are the win: ~18% off the median**, and the
  same ~16% shows up on the synthetic twin, so it is the call path rather than
  anything specific to the MPC. Zero copy needs the caller's memory aligned to
  at least `xla::cpu::MinAlign()`; below that XLA silently falls back to
  copying, which is why the runtime owns its arenas instead of accepting any
  pointer.
- **The spikes are real but rare** — roughly one run in fourteen contains one,
  which is why short benchmarks miss them entirely and why the original report
  was of occasional bad runs rather than consistently bad numbers. The largest
  call the new path produced in 112,000 was 1.7x its median; the old path
  reached 4.2x. Two outlier runs against zero is not on its own a significant
  frequency difference; the difference in magnitude is the solid part.
- **Building the plugin `-c opt` instead of the bazel default changed nothing
  measurable** (4719 µs vs 4728 µs). The compute kernels are LLVM-compiled at
  export time and embedded in the artifact; the plugin only orchestrates.
- **~15,400 allocations per call happen inside XLA's thunk runtime**, roughly
  one per StableHLO op, and the rework only removed the ~500 that were ours.
  That is the remaining structural jitter surface, and reaching it means a
  pooling allocator behind `CpuClientOptions::allocator` — another fork patch,
  worth doing only if the numbers on real hardware still show a tail.

## Patches carried in the XLA fork

`third_party/xla` is a fork, and rebasing it onto a new JAX pin means carrying:

1. **LAPACK FFI kernels** linked into the CPU plugin, so `jnp.linalg.*` works
   without jaxlib's Python extension registering the handlers.
2. **Create options in `pjrt_c_api_cpu_internal.cc`**: `asynchronous` and
   `max_inflight_computations`, plus a `supports_synchronous_execution`
   plugin attribute. `asynchronous=false` makes XLA run computations inline on
   the calling thread instead of handing them to its dispatch pool. It is not
   reachable any other way — the C API's `PJRT_ExecuteOptions` has no
   execution-mode field.

Unknown create options are ignored silently by every PJRT plugin, so dropping
patch 2 costs performance, never correctness. The runtime reports which it got
via `Runtime::synchronous_supported()`.

## Examples

This project provides 3 examples:

1. [`lc0`](https://github.com/LeelaChessZero/lc0) implementation: we demonstrate how to perform JIT computation of a JAX program using some C++ [PJRT wrappers](https://github.com/LeelaChessZero/lc0/tree/97817028ae513cddd779abf606675c0808c353b2/src/neural/backends/xla).

2. [`pjrt_c_api`](https://github.com/openxla/xla/blob/f47564c12397631f240de1ca44279fdf20b66d88/xla/pjrt/c/pjrt_c_api.h): we load and execute a program compiled AOT, only using the `pjrt_c_api` provided by [`xla`](https://github.com/openxla/xla).
Note that our example does not clean up after itself.
It simply shows that we can execute an AOT compiled program in C++.

3. A simple PJRT C++ wrapper: we implement some simple wrappers around the `pjrt_c_api` that performs cleanup and makes code very easy to write.
Note that we severely underexpose the flexibility of the PJRT API.
My applications find this to be mostly sufficient.
The example itself now uses the `Runtime`/`Function` path described above.

## Usage

The compiled examples only _mildly_ depend on XLA.
(Mildly after initial setup...)
They depend on a shared library and a couple of header files: `pjrt_c_api_cpu_plugin.so` (runtime), `pjrt_c_api.h` (API), and `pjrt_c_api_cpu.h` (API struct getter).
This requires the [`bazel`](https://github.com/bazelbuild/bazel) build system.
See the script `compile_xla_runtime.sh` for hints and the correct `bazel` target.

> **NOTE.**
> Make sure that you compile the PJRT runtime to be compatible with JAX.
> This note is probably immaterial if you are running an up-to-date version of JAX, but on the safe side, I've pinned an XLA commit that is compatible with JAX v0.9.0.1.
> Cf. [`jax/third_party/xla/revision.bzl`](https://github.com/jax-ml/jax/blob/jax-v0.9.0.1/third_party/xla/revision.bzl).

For compiling the examples, see the included `Makefile`.
These scripts depend on some byproducts produced from JAX.
The corresponding Python scripts are found in `src/examples`.
You should probably `pip install -e .` in base directory to install a couple of `jax2exec` modules to get this scripts to run.

## References

Note that I stole the name of this repository from `joaospinto`...

- https://github.com/jax-ml/jax/discussions/22184
- https://github.com/joaospinto/call_jax_from_cpp/tree/main
- https://github.com/LeelaChessZero/lc0/tree/97817028ae513cddd779abf606675c0808c353b2/src/neural/backends/xla
- https://github.com/gomlx/gopjrt/tree/main
