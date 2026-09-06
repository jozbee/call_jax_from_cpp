# Runtime internals

Everything on this page was established against the plugin rather than inferred
from the headers. Several plausible-sounding PJRT facts turned out to be false
here, and several implausible ones turned out to be true, so the pattern that
works is to write a probe and run it — `tools/plugin_probe.cpp` is where those
live, and `build/bin/plugin_probe` prints what the loaded plugin actually
supports.

## The PJRT C API in one page

PJRT's C API is a single struct of function pointers, obtained from a shared
object. Four conventions matter here.

**`struct_size` is the versioning mechanism.** Every call takes exactly one
`*_Args` struct, and every one of those begins with `struct_size` and
`extension_start`. The caller sets `struct_size` to the size of the struct it
compiled against; the plugin reads only as far as it understands. The same
applies to `PJRT_Api` itself: a plugin compiled against an older header
publishes a shorter table, and a field past its `struct_size` is memory that
belongs to somebody else. This project therefore treats a function pointer
beyond the end as absent rather than as whatever byte pattern happens to be
there, and refuses at load with the name of the first missing function.

**Errors are objects, and they must be destroyed.** Every function returns a
`PJRT_Error*`; null means success. The message and the status code are read
through two further calls, and the error is then freed with
`PJRT_Error_Destroy`. `pjrt::Error` consumes an error — copying out the
message and code, then destroying it — so a `PJRT_Error*` never needs freeing
at the call site. Where a failure is a fallback rather than a fault, an
internal holder frees it instead.

**The entry point is one symbol.** `dlopen` the plugin, resolve `GetPjrtApi`,
call it for the `const PJRT_Api*`, then call `PJRT_Plugin_Initialize`. The
library is opened `RTLD_NOW | RTLD_LOCAL` — `NOW` so an unresolved symbol is a
startup failure instead of a crash on the first call that reaches it, `LOCAL`
so the plugin's own copy of LLVM does not join the process's global symbol
namespace. It is never closed: XLA leaves statics behind that boundary which
live as long as the process, and unmapping the code they point into ends the
process in an `atexit` handler rather than anywhere diagnosable.

**Configuration is a list of named values.** `PJRT_Client_Create` takes an
array of `PJRT_NamedValue`, and that is the only way to reach settings the C
API has no field for — inline execution among them, since
`PJRT_ExecuteOptions` has no execution-mode field. `PJRT_Plugin_Attributes`
returns the plugin's own self-description and needs no client, which is
exactly what makes it usable for deciding which options are safe to send.
That ordering became load-bearing at this XLA version, where the CPU plugin
started rejecting unknown option names instead of ignoring them.

## Verified behaviour

### Zero-copy input buffers genuinely alias the caller's pointer

`PJRT_Client_BufferFromHostBuffer` with `kImmutableZeroCopy` or
`kMutableZeroCopy` aliases the memory it is given on CPU, and **writes made
between executions are seen by the next one**. That is the property the whole
design rests on: an input arena is wrapped once at load time and never
re-wrapped, so a call transfers nothing. Zero copy still produces a "done
with host buffer" event, which fires when the buffer is destroyed; nothing
waits on it, so it is released immediately.

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
`xla::cpu::Align()`) instead of accepting an arbitrary caller pointer. The
arenas are also rounded up to a whole multiple of 64, so the tail of the last
cache line belongs to us: XLA's vectorized epilogues read whole vectors, and a
read past the end of an exactly-sized allocation is a valgrind report at best.
If the API is ever changed to take caller memory, it must enforce the
alignment, or the entire benefit disappears without a diagnostic.

### Outputs are read straight out of device memory

`PJRT_Buffer_OpaqueDeviceMemoryDataPointer` and `PJRT_Buffer_UnsafePointer` are
both implemented for CPU, so an output needs no `PJRT_Buffer_ToHostBuffer`
round trip and no event to wait on. Device memory on CPU is ordinary memory,
and one `memcpy` per output is the whole of it.

`PJRT_Client_CreateViewOfDeviceBuffer` **is** implemented for CPU at this XLA
revision, and it works: it aliases the caller's pointer and sees writes made
after the view exists. An earlier note here said it was missing, and that was
either wrong or has stopped being true; `tools/plugin_probe --view` now checks
it on every plugin so the answer cannot rot again unnoticed.

Persistent inputs still go through `BufferFromHostBuffer`, for a reason that
survives the correction: a view is explicitly *non-owned*, so its lifetime
contract is the caller's problem, while `kImmutableZeroCopy` gives the same
aliasing with ownership semantics the runtime already handles. Two things
measured while checking this are worth carrying:

- The header calls `on_delete_callback` optional and nullable. The CPU
  implementation throws `std::bad_function_call` when it is null, which
  surfaces as a crash rather than a `PJRT_Error`. Pass a callback, even an
  empty one.
- The alias is genuine in both directions: a write through the caller's pointer
  after the view is created is visible through the buffer.

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

### Thousands of allocations per call happen inside XLA

About one and a half per StableHLO op, in XLA's thunk runtime, entirely inside
the plugin: 9,750 per call for the `02_trajopt` example this tree ships. Only
~520 per call were ever attributable to this wrapper, and those are gone; the
steady-state path is now zero.

That is why the allocation census classifies rather than counts. Grepping the
source for `malloc` proves nothing about what the linked binary does at run
time — OpenBLAS, libm and the C++ runtime all allocate behind the caller's
back, and an inlined `std::vector` growth is invisible to any static check —
so `tests/support/malloc_guard.c` interposes the allocator in the real
process and attributes every armed allocation to the module containing its
return address:

| Class | Covers | Gate |
|---|---|---|
| `allocs_self()` | The main executable and `libpjrt_exec` | **must be zero** |
| `allocs_plugin()` | Inside `libpjrt_c_api_cpu_plugin` | reported: 9,750/call for `02_trajopt` |
| `allocs_runtime()` | libc, libstdc++, LAPACK, the thread pool | reported |

The C++ `operator new` family is interposed too, under its Itanium-mangled
names — without that, an inlined `std::vector` growth would be charged to
libstdc++ rather than to the module that grew the vector, which is exactly the
attribution the gate depends on. `total()` counts allocations for the whole
process whether armed or not, and is what makes a zero armed count
believable: thousands in total with zero while armed means the path is clean,
while zero in total means the preload never took effect, which is what
`--require-guard` checks. Classification is ELF-only; on macOS the totals are
still correct and `classified()` reports false.

The plugin's share is the remaining structural jitter surface, and it is not
reachable from here. Reaching it means installing a pooling allocator behind
`CpuClientOptions::allocator`, which the PJRT C API does not expose — a third
fork patch, justified only if real hardware still shows a tail. See
[the XLA fork](xla-fork.md) and [open threads](open-threads.md).

### Thread-pool sizing needs no fork patch

`DefaultThreadPoolSize()` reads the environment variable `PJRT_NPROC` (then
`NPROC`) before falling back to one thread per core. Both pools — whose
threads carry TSL's `tf_XLAEigen…` names at the pinned version — are sized
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
workload (host not recorded), and a loop-fusion experiment on top of it moved
0.6%. The gap is
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
max/avg of only **1.14–1.25x** (host not recorded; an early measurement, before
the campaign on the benchmarks page). The runtime underneath was never producing
multi-millisecond spikes; the wrapper around it was. That single measurement is
why the work went into the call path rather than into replacing XLA:CPU, and it
is also where the sign-off target of max/p50 ≤ 2.0 comes from: the intrinsic
bound is around 1.2, and anything much above it is the caller's own doing.
