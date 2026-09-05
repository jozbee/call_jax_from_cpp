# 01 — Orientation

## What this is for

The owner develops numerics in JAX (autodiff, fast iteration) but has to run
them inside a C++ control loop — nonlinear MPC for robotics, integrating with
`ros2_control`. The pipeline is:

```
Python: jax.jit(f).lower(*args).compile() -> cpu.serialize_executable(...)
        => artifacts/<name>.binpb  (serialized PJRT executable)
        +  artifacts/<name>.json   (sidecar: input/output sizes, dtype)

C++:    dlopen'd PJRT CPU plugin -> PJRT_Client -> load executable -> call
```

The acceptance workload is a Stewart-platform MPC step: **16 inputs / 14
outputs, all rank-1 float64**, ~10.5 KB each way, ~10.5k StableHLO ops,
L-BFGS single-shooting over a horizon of 200 with 1200 decision variables. It
runs around 100 Hz. The consumer lives at `/Users/jozbee/work/eng/comp`.

The success criterion is explicitly **tail minimization** — drive p99.9/p50
toward 1.0 and eliminate rare multi-millisecond spikes. There is no fixed Hz
gate. Do not substitute a mean-latency goal.

## Repository layout

```
src/pjrt_exec/runtime.{hpp,cpp}   THE hot path: Runtime + Function. Start here.
src/pjrt_exec/pjrt_exec.{hpp,cpp} Legacy Client/Buffer/AOTComputation wrappers.
                                  Still present, still correct, allocates per
                                  call. Kept as the benchmark baseline.
src/pjrt_exec/rt.{hpp,cpp}        Optional Linux RT hardening (mlockall,
                                  affinity, SCHED_FIFO, malloc tuning).
                                  No-op stubs elsewhere.
src/bench/                        bench_main.cpp + stats/fixture/guard headers,
                                  and probe_zerocopy.cpp. The measurement spine.
src/jax2exec/jax2exec.py          The exporter. Asserts rank<=1 and float64.
src/examples/                     aot_jax2exec_example.cpp uses the new API;
                                  aot_example.cpp is raw C API; lc0_example.cpp
                                  is a third-party wrapper demo.
src/lc0/, src/nlohmann/, src/xla/ Vendored: lc0's PJRT wrapper, JSON, C headers.
tools/                            export_fixture.py, npz_to_bin.py,
                                  run_matrix.sh, rt_check.sh
tests/assets/mpc/                 Frozen reference cases (.bin + .npz + manifest)
tests/support/malloc_guard.c      LD_PRELOAD allocation interposer
third_party/xla                   Submodule: fork jozbee/xla, branch
                                  call_jax_from_cpp-lapack
build_scripts/compile_xla_runtime.sh  Builds the plugin via bazel
```

## The two APIs, and which to use

**Use `pjrt::Runtime` / `pjrt::Function`.** The legacy
`Client`/`Buffer`/`AOTComputation` classes create device buffers, events and
`shared_ptr`s on every call. They were the jitter source. They remain in the
tree only so the benchmark can measure against them (`--api legacy`) and until
`eng/comp` migrates.

```cpp
pjrt::Runtime rt;                              // one per process
pjrt::Function f(rt, "artifacts/mpc_solver");  // load once
std::memcpy(f.input(3), acc_ref, 3 * sizeof(double));
f.call();
const double* u = f.output(0);
```

`input(i)`/`output(i)` point at 64-byte-aligned arenas the `Function` owns.
Inputs are wrapped once in zero-copy PJRT buffers, so a call transfers nothing.
Write inputs *between* calls, never during one. A `Function` is single-threaded
by design (it owns fixed storage); use one per thread, sharing a `Runtime`.

MPC state recirculation (11 of 14 outputs feed back as inputs) is deliberately
caller-side `memcpy`, not API surface.

## Build and run

```
make                      # aot_jax2exec_example (default target)
make bench                # builds+runs artifacts/bench
make fixtures             # exports mpc_solver + synth_solver on THIS machine
make test_correct         # both APIs vs reference fixtures, all cases
make test_alloc           # allocation census under the preloaded interposer
make pjrt_runtime         # rebuilds the PJRT plugin via bazel (slow: ~12 min)
```

Development happens in the aarch64 devcontainer (`.devcontainer/compose.yml`,
which grants `SYS_NICE` and `rtprio`/`memlock` ulimits so the RT helpers can be
exercised). The **production target is native Intel amd64 Linux**; container
numbers are relative signals only.
