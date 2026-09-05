# Runtime internals

Everything on this page was established against the plugin rather than inferred
from the headers. Several plausible-sounding PJRT facts turned out to be false
here, and several implausible ones turned out to be true, so the pattern that
works is to write a probe and run it — `tools/plugin_probe.cpp` is where those
live, and `build/bin/plugin_probe` prints what the loaded plugin actually
supports.

## Verified behaviour

### Zero-copy input buffers genuinely alias the caller's pointer

`PJRT_Client_BufferFromHostBuffer` with `kImmutableZeroCopy` or
`kMutableZeroCopy` aliases the memory it is given on CPU, and **writes made
between executions are seen by the next one**. That is the property the whole
design rests on: an input arena is wrapped once at load time and never
re-wrapped, so a call transfers nothing.

Both host-buffer semantics documents forbid mutating the memory while the
buffer is alive. Doing it *between* calls is a deliberate, verified bend of
that contract — nothing is in flight, and the documents speak about mutation
while a transfer is outstanding. Mutating *during* a call is a data race on the
computation's own operands, and that is why writing inputs from another thread
or a signal handler is out of bounds.

### Alignment is a silent gate

Below `xla::cpu::MinAlign()` (= `EIGEN_MAX_ALIGN_BYTES`), XLA **falls back to
copying the buffer and tells you nothing**. There is no error, no warning, and
no attribute to query; the only symptom is that the win you thought you had
does not appear.

This is why `Function` owns 64-byte-aligned arenas (`posix_memalign`, 64 =
`xla::cpu::Align()`) instead of accepting an arbitrary caller pointer. If the
API is ever changed to take caller memory, it must enforce the alignment, or
the entire benefit disappears without a diagnostic.

### Outputs are read straight out of device memory

`PJRT_Buffer_OpaqueDeviceMemoryDataPointer` and `PJRT_Buffer_UnsafePointer` are
both implemented for CPU, so an output needs no `PJRT_Buffer_ToHostBuffer`
round trip and no event to wait on. Device memory on CPU is ordinary memory,
and one `memcpy` per output is the whole of it.

`PJRT_Client_CreateViewOfDeviceBuffer` is **not** implemented for CPU. That is
why persistent inputs go through `BufferFromHostBuffer` rather than a view —
the obvious-looking API for "wrap this memory" is the one that is missing.

### Serialized executables embed target machine code

`PJRT_Executable_DeserializeAndLoad` reaches
`CpuCompiler::LoadAotCompilationResult`, which **relinks**; it never
recompiles. A `.binpb` is therefore architecture- and ISA-locked to the machine
that exported it, and loading one built for a wider instruction set fails as
SIGILL somewhere inside the executable, with a backtrace that says nothing
about artifacts.

Two things follow. The exporter records the host's ISA level in the sidecar and
the loader compares it before opening the binary (`FunctionOptions::isa_guard`).
And the `.mlirbc` exists as the portable route: `LoadPolicy::Auto` compiles it
in-process when the binary is absent or would not run here, at a cost of
seconds at load rather than milliseconds. `Function::load_kind()` reports which
route ran, and a deployment that cannot afford a surprise multi-second load
should ask for `LoadPolicy::BinaryOnly` and get a `LoadError` instead.

### thousands of allocations per call happen inside XLA

That is about one per StableHLO op, in XLA's thunk runtime, entirely inside the
plugin. Only ~520 allocations per call were ever attributable to this wrapper,
and those are gone: the steady-state path is now zero.

This is the remaining structural jitter surface, and it is not reachable from
here. Reaching it means installing a pooling allocator behind
`CpuClientOptions::allocator`, which the PJRT C API does not expose — a third
fork patch, justified only if real hardware still shows a tail. See
[the XLA fork](xla-fork.md) and [open threads](open-threads.md).

### Thread-pool sizing needs no fork patch

`DefaultThreadPoolSize()` reads the environment variable `PJRT_NPROC` (then
`NPROC`) before falling back to one thread per core. Both pools — named
`XLAEigen*` and `XLAPjRtCpuClient*` — are sized
`max(DefaultThreadPoolSize(), cpu_device_count)`.
`RuntimeOptions::worker_threads` is therefore applied with `setenv` before the
client is created, and no patch is involved. It is a process environment
variable, so it is visible to any client created afterwards.

### Create options are validated at this XLA version

This is new, and it invalidates a rule that used to be true. The CPU plugin now
**validates create option names and rejects unknown ones**:

```
InvalidArgument: Unexpected option name passed to PJRT_Client_Create
```

Older plugins ignored options they did not understand, which made sending a
speculative option free. It no longer is: an unrecognised name fails client
creation outright, so a caller cannot discover the surface by trying it.

`Runtime` therefore queries `PJRT_Plugin_Attributes` first — attributes are
readable before any client exists, which is exactly what makes them usable for
this — and sends only the options the plugin advertises. When creation still
fails naming an option, and `allow_async_fallback` permits it, the option is
dropped and creation is retried. `Runtime::synchronous_mode()` reports what
actually took effect: `Inline` when the plugin advertises
`supports_synchronous_execution` and accepted the option, `Accepted` when it was
accepted by a plugin that does not advertise the marker, `Rejected` when the
plugin refused it, and `Async` when it was never asked for.

### PJRT cannot cancel a running CPU computation

There is no cancellation in the C API and none underneath it. A watchdog can
only return a stale result and discard the late one; an overrun means late
data, not a cancelled call. Anything designed around a deadline has to be
honest about that — see [open threads](open-threads.md).

## Why the hot path is shaped this way

The shape follows from one decision: **everything that can happen before the
loop starts happens at load time.**

Load time does all of this, once:

- reads the JSON sidecar and cross-checks it against
  `PJRT_Executable_NumOutputs`, `..._OutputElementTypes` and
  `..._OutputDimensions` — a sidecar that has gone stale relative to its
  executable is otherwise a buffer-overrun class of bug, found much later as
  corrupted output;
- compares the sidecar's recorded ISA level against this host's, and chooses
  between the `.binpb` and the `.mlirbc`;
- `posix_memalign(64)`s one arena per input and one per output;
- wraps every input arena once in a zero-copy `PJRT_Buffer`;
- sizes every per-call array as a member, so nothing grows during a call;
- runs `warmup_calls` blocking calls to fault in the pages and warm the
  runtime's lazily-initialised state.

A call is then exactly this: execute → **one** await → for each output, take the
device pointer and `memcpy` it into the output arena → destroy the output
buffer. No `get_dims`, no `shared_ptr`, no vector growth, no iostream, no
lock, no syscall. What remains is inside XLA.

The debug checks (bounds, typed-accessor dtype, `call()` re-entrancy, and with
`check_values` the arena scans) are all gated on `FunctionOptions::debug`,
which costs one predictable branch on a member when it is on and nothing when
it is off. State the trade plainly whenever it comes up: with debug off, the
same mistake that would have thrown is silent memory corruption. Run debug on
in development.

## Failure modes to avoid reintroducing

The path this replaced was correct and slow, and it is worth knowing exactly
how it was slow, because each of these is easy to write again:

- a host-to-device **copy per input per call**
  (`kImmutableUntilTransferCompletes`), plus one event per copy;
- `std::flush(std::cout)` inside the execute wrapper — a syscall in the hot
  path;
- `non_donatable_input_indices` hardcoded to `{0}`, which is a guess about the
  caller's donation that happens to be wrong for everyone else;
- the output count taken from the JSON sidecar rather than from
  `PJRT_Executable_NumOutputs`, so a stale sidecar wrote past an arena;
- per-call `get_dims()` vector allocations, for dimensions that cannot change;
- `shared_ptr` and event churn throughout;
- a namespace-scope `static const PJRT_Api* const` **in a header**, giving one
  copy and one `GetPjrtApi()` call per translation unit.

There was also a `sleep(1ms)` in an example that "fixed" a segfault. It was
root-caused before removal — 60/60 runs with it, 200/200 without, ASan and
UBSan clean — rather than deleted on a hunch. Do the same with any other
superstition found in this tree: reproduce it, explain it, then remove it.

## Approaches considered and rejected

Recorded so nobody re-derives them. Each was rejected on evidence, and the
evidence is what would have to change to reopen the question.

**A hand-written scalar-C++ backend.** Emitting per-architecture SIMD from a
custom compiler was considered as a replacement for XLA:CPU. That project's own
benchmarks put its scalar-C++ backend **2.2–4.3x slower than XLA:CPU** on this
workload, and a loop-fusion experiment on top of it moved 0.6%. The gap is
vectorisation and tiling — a months-scale compiler project. Its hoped-for
jitter advantage never materialised either: max/avg of 1.22–1.24x, against
PJRT's 1.14–1.25x. It would have been slower and no steadier.

**Forking IREE.** IREE 3.11 cannot compile the module at all — a
`tensor.expand_shape` failure on JAX's batched scatter. Forking would mean
owning an upstream compiler bug plus a very large dependency, to reach a
runtime that has not been shown to be steadier than the one already in hand.

**A C++ shim wrapping `PjRtCpuClient` directly** instead of going through the C
API. It would expose `CpuClientOptions` — including the allocator — with no
fork patch at all. It also couples this repository to XLA's unstable C++ API
across every routine JAX bump, which is the maintenance cost the C API exists to
avoid. The small C-API patch is cheaper to carry. This is an explicit fallback
trigger, not a plan: if the allocator turns out to be the only remaining lever
and the patch cannot be carried, this is the next thing to try.

**tfcompile-style AOT-to-object.** The `.binpb` already loads AOT-compiled
object code, so this adds a second toolchain and risk without new benefit.

## The evidence that chose the current path

In-process **blocking** PJRT called from Python, on the same module, shows
max/avg of only **1.14–1.25x**. The runtime underneath was never producing
multi-millisecond spikes; the wrapper around it was. That single measurement is
why the work went into the call path rather than into replacing XLA:CPU, and it
is also where the sign-off target of max/p50 ≤ 2.0 comes from: the intrinsic
bound is around 1.2, and anything much above it is the caller's own doing.
