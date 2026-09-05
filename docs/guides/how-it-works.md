# How it works

Export lowers a JAX function to StableHLO, compiles it, and serializes the
result; the compute kernels are LLVM-compiled at that point and embedded in the
artifact. Loading relinks that machine code — it never recompiles — and the call
path is then execute, one await, and one `memcpy` per output.

## The pipeline

```{mermaid}
flowchart TB
  subgraph export["Export, once, in Python"]
    F["JAX function"] --> T["trace and lower"]
    T --> S["StableHLO"]
    S --> C["XLA compiles;<br/>LLVM emits machine code"]
    C --> B[".binpb<br/>serialized executable"]
    S --> M[".mlirbc<br/>StableHLO bytecode"]
    C --> J[".json<br/>sidecar"]
  end

  subgraph load["Load, once, in C++"]
    B --> D["DeserializeAndLoad<br/>relink"]
    M --> K["Client_Compile<br/>seconds"]
    J --> V["read and cross-check"]
    D --> E["LoadedExecutable"]
    K --> E
    V --> A["64-byte arenas +<br/>zero-copy input buffers"]
  end

  subgraph loop["Call, many times"]
    A --> W["write input arenas"]
    W --> X["Execute"]
    E --> X
    X --> Y["one Await"]
    Y --> Z["memcpy each output<br/>out of device memory"]
    Z --> W
  end
```

The sidecar is the thin arrow holding the two halves together. The PJRT C API
can be asked
how many outputs an executable has, what their element types are and what their
dimensions are — and **nothing at all about its parameters**. The sidecar is
the only description of the inputs that exists, which is why it is
cross-checked, versioned and checksummed rather than treated as a convenience.

## The PJRT C API in one page

PJRT's C API is a single struct of function pointers, obtained from a shared
object. Four conventions matter here.

**`struct_size` is the versioning mechanism.** Every call takes exactly one
`*_Args` struct, and every one of those begins with `struct_size` and
`extension_start`. The caller sets `struct_size` to the size of the struct it
compiled against; the plugin reads only as far as it understands. The same
applies to `PJRT_Api` itself: a plugin compiled against an older header
publishes a shorter table, and **a field past its `struct_size` is memory that
belongs to somebody else**. This project therefore treats a function pointer
beyond the end as absent rather than as whatever byte pattern happens to be
there, and refuses at load with the name of the first missing function.

**Errors are objects, and they must be destroyed.** Every function returns a
`PJRT_Error*`; null means success. The message and the status code are read
through two further calls, and the error is then freed with
`PJRT_Error_Destroy`. `pjrt::Error` consumes an error — copying out the message
and code, then destroying it — so a `PJRT_Error*` never needs freeing at the
call site. Where a failure is a fallback rather than a fault, an internal
holder frees it instead.

**The entry point is one symbol.** `dlopen` the plugin, resolve `GetPjrtApi`,
call it for the `const PJRT_Api*`, then call `PJRT_Plugin_Initialize`. The
library is opened `RTLD_NOW | RTLD_LOCAL` — `NOW` so an unresolved symbol is a
startup failure instead of a crash on the first call that reaches it, `LOCAL` so
the plugin's own copy of LLVM does not join the process's global symbol
namespace. It is never closed: XLA leaves statics behind that boundary which
live as long as the process, and unmapping the code they point into ends the
process in an `atexit` handler rather than anywhere diagnosable.

**Configuration is a list of named values.** `PJRT_Client_Create` takes an array
of `PJRT_NamedValue`, and that is the only way to reach settings the C API has
no field for — inline execution among them, since `PJRT_ExecuteOptions` has no
execution-mode field. `PJRT_Plugin_Attributes` returns the plugin's own
self-description and **needs no client**, which is exactly what makes it usable
for deciding which options are safe to send. That ordering became load-bearing
at this XLA version, where the CPU plugin started rejecting unknown option
names instead of ignoring them.

## Loading

```{mermaid}
sequenceDiagram
  participant App as Caller
  participant F as Function
  participant P as PJRT plugin
  App->>F: Function(runtime, "artifacts/trajopt")
  F->>F: read + validate trajopt.json
  F->>F: compare sidecar ISA level with this host
  F->>P: PJRT_Executable_DeserializeAndLoad(.binpb)
  P-->>F: LoadedExecutable
  F->>P: NumOutputs / OutputElementTypes / OutputDimensions
  F->>F: cross-check against the sidecar
  F->>F: posix_memalign(64) one arena per array
  F->>P: BufferFromHostBuffer(kImmutableZeroCopy) per input
  F->>P: Execute x warmup_calls
  F-->>App: ready
```

| Step | Why it is at load time |
|---|---|
| Read and validate the sidecar | A stale sidecar is otherwise a buffer-overrun class of bug, found later as corrupted output. |
| Compare ISA levels | A `.binpb` built for a wider instruction set is a `SIGILL` with no artifact anywhere in the backtrace. |
| Deserialize (or compile) | Deserializing relinks embedded machine code — milliseconds. Compiling the `.mlirbc` is seconds, and is the portable fallback. |
| Cross-check the outputs | The only half of the signature the executable can be asked about. |
| Allocate arenas | 64-byte aligned, zeroed, one per input and per output, owned for the life of the `Function`. |
| Wrap inputs once | The buffer aliases the arena for its whole lifetime, so a call transfers nothing. |
| Warm up | Faults in every page and resolves the runtime's lazy state. The first call after a load is always the slowest; this is where that cost is spent. |

**What a naive implementation does instead:** compiles at startup (seconds,
every time), and then discovers shapes per call because the executable is the
only thing it asked.

## Calling

```{mermaid}
sequenceDiagram
  participant App as Caller
  participant F as Function
  participant P as PJRT plugin
  App->>F: memcpy into input arenas
  App->>F: call()
  F->>P: PJRT_LoadedExecutable_Execute
  P-->>F: output buffers + one completion event
  F->>P: PJRT_Event_Await (ready already, when inline)
  loop each output
    F->>P: PJRT_Buffer_OpaqueDeviceMemoryDataPointer
    F->>F: memcpy into the output arena
    F->>P: PJRT_Buffer_Destroy
  end
  F-->>App: outputs in the arenas
```

```{literalinclude} ../../src/pjrt_exec/runtime.cpp
:language: cpp
:start-after: docs: begin call_path
:end-before: docs: end call_path
```

| This path | The naive one |
|---|---|
| Input buffers created once at load, aliasing the arenas. | `BufferFromHostBuffer` per input **per call**, with `kImmutableUntilTransferCompletes`: a host-to-device copy and a completion event each — sixteen of both, on the measured fixture. |
| One `Execute`, one `Await`. | The same, plus the events above to wait on. |
| One `OpaqueDeviceMemoryDataPointer` and one `memcpy` per output. | `PJRT_Buffer_ToHostBuffer` per output: the same copy, plus an event, plus an await, plus a second allocation. |
| Every per-call array sized at load. | `get_dims()` and friends allocating a vector per output per call. |
| Nothing logs. | `std::flush(std::cout)` inside the execute wrapper — a syscall in the hot path. |
| `non_donatable_input_indices` computed once from the sidecar. | Hardcoded to `{0}`. |

That right-hand column is not a straw man. It is the wrapper this project
started from, and it is where the tail came from: roughly 520 allocations per
call attributable to the wrapper, all of them now gone.

## Three decisions worth the space

**Why the arenas are 64-byte aligned.** Alignment is a silent gate. Below
`xla::cpu::MinAlign()` XLA **falls back to copying the host buffer and tells you
nothing** — no error, no log line, only a latency that is quietly worse. 64 is
`xla::cpu::Align()`. The arenas are also rounded up to a whole multiple of it,
so the tail of the last cache line belongs to us: XLA's vectorized epilogues
read whole vectors, and a read past the end of an exactly-sized allocation is a
valgrind report at best.

This is also why the API owns the memory instead of accepting a caller's
pointer. An arbitrary pointer would have to be checked, and an unchecked one
would silently lose the entire benefit.

**Why persistent inputs go through `BufferFromHostBuffer`.**
`PJRT_Client_CreateViewOfDeviceBuffer` is the call that sounds right, and it is
**not implemented for CPU**. `BufferFromHostBuffer` with
`kImmutableZeroCopy` genuinely aliases the caller's pointer on CPU — verified,
not assumed — and writes made between executions are seen by the next one. Zero
copy still produces a "done with host buffer" event, which fires when the buffer
is destroyed; nothing waits on it, so it is released immediately.

Both host-buffer semantics documents forbid mutating a buffer while it is alive.
Writing between calls is a deliberate, measured bend of that contract, because
nothing is in flight. It is the property the whole design rests on, and the
reference-case sweep is what keeps it honest.

**Why outputs are read directly.** On CPU, device memory is ordinary memory.
`PJRT_Buffer_OpaqueDeviceMemoryDataPointer` and `PJRT_Buffer_UnsafePointer` are
both implemented, so an output is one pointer query and one `memcpy` — the same
copy `ToHostBuffer` would perform, without the event, the await and the second
allocation it charges for it.

## Where the remaining cost is

Two facts bound what is left.

**The plugin is not where the time goes.** Building it `-c opt` versus bazel's
default `fastbuild` moved p50 by 0.2% — 4718.7 µs against 4728.4 µs. The compute
kernels were LLVM-compiled at export time and embedded in the artifact; the
plugin only orchestrates. Do not spend effort on plugin build flags.

**Roughly thousands of allocations happen per call inside XLA's thunk runtime**, about
one per StableHLO op. Only ~520 were ever the wrapper's own, and those are gone.
Reaching the rest means a pooling allocator behind `CpuClientOptions::allocator`
— which the PJRT C API does not expose, so it would be a third patch on the XLA
fork. That is evidence-gated: it happens if, and only if, real hardware still
shows a tail after everything cheaper has been done.

The other half of the remaining variance is not in this process at all. It is
the scheduler, the timer tick, the C-states and the governor; see
{doc}`realtime`.

For the verified PJRT and XLA:CPU behaviour behind all of this — including the
things that turned out to be false — see {doc}`../developer/runtime-internals`,
and {doc}`../developer/open-threads` for what is deliberately not done yet.
