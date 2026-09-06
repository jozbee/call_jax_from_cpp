# How it works

*Assumes the {doc}`Quickstart </getting-started/quickstart>`. The names —
StableHLO, XLA, PJRT — are one paragraph each in
{doc}`/background/xla-and-pjrt`.*

Export lowers a JAX function to {term}`StableHLO`, compiles it, and serializes
the result; the compute kernels are LLVM-compiled at that point and embedded in the
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

The {term}`sidecar` is the thin arrow holding the two halves together. The PJRT C API
can be asked
how many outputs an executable has, what their element types are and what their
dimensions are — and **nothing at all about its parameters**. The sidecar is
the only description of the inputs that exists, which is why it is
cross-checked, versioned and checksummed rather than treated as a convenience.

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
| Read and validate the sidecar | A stale sidecar is otherwise a buffer overrun, found later as corrupted output. |
| Compare ISA levels | A `.binpb` built for a wider instruction set is a `SIGILL` with no artifact in the backtrace. |
| Deserialize, or compile | Deserializing relinks embedded machine code: milliseconds. Compiling the `.mlirbc` is seconds, and portable. |
| Cross-check the outputs | The only half of the signature the executable can be asked about. |
| Allocate arenas | 64-byte aligned, zeroed, one per array, owned for the life of the `Function`. |
| Wrap inputs once | The buffer aliases the arena for its whole lifetime, so a call transfers nothing. |
| Warm up | Faults in every page and resolves the runtime's lazy state; the first call after a load is always the slowest. |

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

Everything a naive implementation does per call — a host-to-device copy per
input, a `ToHostBuffer` round trip per output, a vector allocation per shape
query, a flush — happens here once at load, or not at all. That naive path is
the wrapper this project started from, and it is where the tail came from.

## Where the remaining cost is

The plugin is not where the time goes: the compute kernels were compiled at
export time and the plugin only orchestrates, so its build flags do not move
the numbers. What remains inside the process is XLA's thunk runtime, which
allocates about once per StableHLO op inside the plugin, out of reach of the
PJRT C API. The other half of the variance is not in the process at all — the
scheduler, the timer tick, the {term}`C-states <C-state>` and the
{term}`governor` — and that is {doc}`realtime`.

## Deeper

{doc}`../developer/runtime-internals` — the PJRT C API conventions this
depends on, the verified XLA:CPU behaviour, and the three design decisions.
{doc}`../developer/open-threads` — what is deliberately not done yet.
{doc}`/background/xla-and-pjrt` — the four names, for a reader who has never
opened openxla.org.
