# 03 — PJRT/XLA:CPU facts and the shape of the hot path

Everything here was established by direct measurement against the pinned XLA
fork (jax 0.9.0.1), not from documentation. Several items contradict what the
headers suggest.

## Verified behaviour

**Zero-copy input buffers work on CPU.** `PJRT_Client_BufferFromHostBuffer`
with `kImmutableZeroCopy` or `kMutableZeroCopy` genuinely aliases the caller's
pointer, and writes made *between* executions are seen by the next one. Both
host-buffer semantics documents forbid mutating while the buffer is alive;
doing it between calls is a deliberate, verified bend — nothing is in flight.

**Alignment is a silent gate.** Below `xla::cpu::MinAlign()`
(= `EIGEN_MAX_ALIGN_BYTES`) XLA **falls back to copying without telling you**.
This is precisely why `Function` owns 64-byte-aligned arenas
(`posix_memalign`, 64 = `xla::cpu::Align()`) instead of accepting an arbitrary
caller pointer. If you ever change the API to take caller memory, you must
enforce alignment or you silently lose the entire win.

**Output pointers are directly readable.** `PJRT_Buffer_UnsafePointer` and
`PJRT_Buffer_OpaqueDeviceMemoryDataPointer` are both implemented for CPU, so
outputs need no `ToHostBuffer` round trip.
`PJRT_Client_CreateViewOfDeviceBuffer` is **not** implemented for CPU — which
is why persistent inputs go through `BufferFromHostBuffer`, not a view.

**Serialized executables embed target machine code.**
`LoadSerializedExecutable` → `CpuCompiler::LoadAotCompilationResult` relinks;
it never recompiles. `.binpb` artifacts are architecture- and ISA-locked to the
machine that exported them. Run `make fixtures` on the machine that will
execute them.

**~15,400 allocations happen per call inside XLA's thunk runtime**, roughly one
per StableHLO op. Only ~520 were ever attributable to the C++ wrapper, and
those are gone. This is the remaining structural jitter surface. Reaching it
means a pooling allocator behind `CpuClientOptions::allocator`, which is *not*
exposed through the PJRT C API — it would be a second fork patch. Do that only
if real hardware still shows a tail.

**Thread pool sizing needs no fork change.** `DefaultThreadPoolSize()` reads
the env var `PJRT_NPROC` (then `NPROC`) first. Both pools — named `XLAEigen*`
and `XLAPjRtCpuClient*` — are sized `max(DefaultThreadPoolSize(),
cpu_device_count)`. `RuntimeOptions::worker_threads` is applied by `setenv`
before client creation.

**`asynchronous=false` is only reachable through a create option.** The C API's
`PJRT_ExecuteOptions` has no execution-mode field. See
[04-xla-fork.md](04-xla-fork.md).

## Why the hot path looks the way it does

Load time does everything that can possibly be done at load time:

- read and cross-check the JSON sidecar against `PJRT_Executable_NumOutputs`
  (the sidecar going stale used to be a heap-overflow class of bug);
- `posix_memalign(64)` one arena per input and per output;
- wrap every input arena once in a zero-copy `PJRT_Buffer`;
- pre-size every per-call array as a member;
- run `warmup_calls` blocking calls to fault in pages and warm the runtime.

A call is then: execute → **one** await → per output
`PJRT_Buffer_OpaqueDeviceMemoryDataPointer` + `memcpy` into the output arena →
`PJRT_Buffer_Destroy`. No `get_dims`, no `shared_ptr`, no `vector` growth, no
iostream.

## Defects in the legacy path (the actual jitter source)

Kept here because they are the failure modes to avoid reintroducing:

- a host-to-device **copy per input per call**
  (`kImmutableUntilTransferCompletes`) — 16 copies + 16 events per call;
- `std::flush(std::cout)` inside `execute_blocking` — a syscall in the hot path;
- `non_donatable_input_indices` hardcoded to `{0}`;
- output count taken from the JSON sidecar instead of
  `PJRT_Executable_NumOutputs`;
- per-call `get_dims()` vector allocations;
- `shared_ptr` and `Event` churn throughout;
- a namespace-scope `static const PJRT_Api* const api_` **in the header**, so
  one copy and one `get_pjrt_api_()` per translation unit. Now a Meyers
  singleton `const PJRT_Api* api()` defined in the `.cpp`.

There was also a `sleep(1ms)` hack in the example that "fixed" a segfault. It
was root-caused before removal (60/60 runs with it, 200/200 without,
ASan/UBSan clean) rather than deleted on a hunch. Do the same with any other
superstition you find.

## Approaches that were considered and rejected

Recorded so nobody re-derives them:

- **Rewrite `jaxpile` with per-architecture SIMD** (from `eng/lib_emitc`). That
  project's own benchmarks show its scalar-C++ backend is 2.2–4.3x slower than
  XLA:CPU on this workload, and its loop-fusion experiment moved 0.6%. The gap
  is vectorization + tiling — a months-scale compiler project. Its hoped-for
  jitter advantage never materialized either (max/avg 1.22–1.24x vs PJRT's
  1.14–1.25x).
- **Fork and fix IREE.** IREE 3.11 cannot compile the MPC module at all
  (`tensor.expand_shape` failure on JAX's batched scatter). Forking means
  owning an upstream compiler bug plus a very large dependency.
- **A C++ shim wrapping `PjRtCpuClient` directly** instead of the C API. It
  would expose `CpuClientOptions` without a fork patch, but couples the repo to
  XLA's unstable C++ API across routine jax bumps. Explicit fallback trigger
  only; the ~47-line C-API patch is cheaper to carry.
- **tfcompile-style AOT-to-object.** The `.binpb` already loads AOT-compiled
  object code; this adds risk without new benefit.

The decisive evidence for the chosen path: in-process **blocking** PJRT from
Python, on the exact MPC module, shows max/avg of only 1.14–1.25x. XLA:CPU was
never the jitter source.
