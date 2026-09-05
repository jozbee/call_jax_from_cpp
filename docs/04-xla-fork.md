# 04 — The XLA fork

`third_party/xla` is a submodule pointing at **github.com/jozbee/xla**, branch
**`call_jax_from_cpp-lapack`**. Rebasing it onto a new JAX pin means carrying
two patches. Neither is large; both are easy to lose silently.

## Patch 1 — LAPACK FFI kernels (pre-existing)

Links jaxlib's LAPACK FFI kernels into the CPU plugin so `jnp.linalg.*` works
without jaxlib's Python extension registering the handlers.

Commit `ce2f59bc51`, "Link jaxlib's LAPACK FFI kernels into the CPU PJRT
plugin".

Losing this shows up as link failures for `dgesvd_`, `sgeev_`, and friends. It
affects the `jnp.linalg.inv` example, not the MPC workload.

## Patch 2 — CPU plugin create options (added by this work)

Commit `300ace192a`, "Add CPU plugin create options for synchronous execution".
47 insertions in one file: `xla/pjrt/c/pjrt_c_api_cpu_internal.cc`.

It does three things:

1. parses the `asynchronous` create option (bool) into `CpuClientOptions`;
2. parses `max_inflight_computations` (int64) likewise;
3. adds `PJRT_Plugin_Attributes_Cpu`, which returns
   `GetXlaPluginCAttributes()` plus a `supports_synchronous_execution` int64
   marker, and wires it into `CreatePjrtApi`.

**Why it is necessary.** `CpuClientOptions::asynchronous` defaults to true;
`false` is documented as "always run computations inline", confirmed in
`cpu_client.cc` where `execute_inline = ... || !client->asynchronous_` and,
with empty input deps, thunks then run on the calling thread. The PJRT **C**
API's `PJRT_ExecuteOptions` has no execution-mode field, so a create option is
the only route. Before the patch the plugin parsed only `cpu_device_count`
(default 4).

**Why it is safe to lose.** Every PJRT plugin silently ignores unknown create
options. Dropping the patch costs performance, never correctness. The runtime
detects the situation via the plugin attribute and reports it through
`Runtime::synchronous_supported()`.

## Building the plugin

```
make pjrt_runtime          # or: build_scripts/compile_xla_runtime.sh
```

`BUILD_MODE` defaults to `opt`; `EXTRA_BAZEL_ARGS` passes flags through;
`--config=avx_linux` is added only on x86_64 Linux; the output is stripped, and
non-opt builds get a suffixed filename so both can coexist for A/B (there are
already `libpjrt_c_api_cpu_plugin_{opt,fastbuild}.so` in `artifacts/`).

Budget ~12 minutes of bazel wall clock. **Do not run it concurrently with a
benchmark** — see [02-measurement.md](02-measurement.md).

## Build mode does not matter — do not repeat this experiment

The plugin was originally shipping from a bazel `fastbuild` (`-O0`) build, and
rebuilding it `-c opt` was expected to be a large win. It moved p50 by 0.2%
(4718.7 vs 4728.4 µs). The compute kernels are LLVM-compiled at *export* time
and embedded in the `.binpb`; the plugin only orchestrates. That 12-minute
build is off the critical path.

## A candidate third patch, not yet justified

A pooling allocator installed behind `CpuClientOptions::allocator` (also not
reachable through the C API) would address the ~15,400 per-call allocations
inside the thunk runtime. Gate it on evidence from the real amd64 hardware
showing that plugin-internal allocation drives a residual tail.
