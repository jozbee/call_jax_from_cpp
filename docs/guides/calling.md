# Calling from C++

`pjrt::Runtime` is created once per process and `pjrt::Function` is loaded once
per artifact. After that a call writes the input arenas, calls `call()`, and
reads the output arenas. The steady-state path allocates nothing.

## The shape of a call

One cycle, with the loop taken away: write the input arenas, `call()`, read the
output arenas. Everything a control loop adds sits around this, not inside it.

```{literalinclude} ../../examples/01_basic/basic.cpp
:language: cpp
:start-after: docs: begin call
:end-before: docs: end call
```

Everything that can be done before the loop is done before the loop: names
resolved to indices, pointers taken, sizes read. What is left is a write, a
`call()`, and a read. That is the whole design — and the loop-shaped version,
where one cycle's outputs become the next cycle's inputs, is
[further down](#write-the-inputs-between-calls-never-during-one).

## Runtime

One per process. Creating a client starts XLA's thread pools and its
lazily-initialized statics, so **never destroy and recreate one mid-run** —
that is a guaranteed latency spike — and never let it go out of scope while a
`Function` still holds an executable, which is undefined. In practice the
`Runtime` is a long-lived object owned above everything that uses it. The
default `RuntimeOptions` are the control-loop defaults: inline execution, one
device, one worker thread.

### RuntimeOptions

| Field | Default | Meaning |
|---|---|---|
| `plugin_path` | `""` | The shared object to load. Empty means `$PJRT_CPU_PLUGIN`, then the path compiled in at build time. Opened `RTLD_NOW \| RTLD_LOCAL` and never closed: XLA leaves statics behind it that live as long as the process. |
| `synchronous` | `true` | Ask for computations to run inline on the calling thread instead of being handed to the dispatch pool. Removing that hand-off is the single biggest structural cut to tail latency. |
| `cpu_device_count` | `1` | Logical CPU devices. One is all a control loop needs, and it also bounds the size of the runtime's thread pools. |
| `worker_threads` | `1` | Threads for XLA's intra-op and dispatch pools. `0` leaves XLA's default of one per core, which oversubscribes a machine that is doing anything else. |
| `max_inflight_computations` | `0` | Concurrent executions the client admits; `0` leaves the plugin's default. Sent only when the plugin advertises `supports_max_inflight_computations`. |
| `allow_async_fallback` | `true` | On a create failure naming an option the plugin does not know, drop that option and retry. Turn it off to find out, loudly, that you are not running the plugin you think you are. |

:::{warning}
**`worker_threads` is applied with `setenv("PJRT_NPROC", ...)`**, because that
is where XLA's `DefaultThreadPoolSize()` reads it. It is a process environment
variable, not a per-client setting: it is visible to every client created
afterwards, and to anything else in the process that reads `PJRT_NPROC`. In a
plugin host — a ROS node, say — that is a process-wide side effect worth
knowing about before it surprises someone.
:::

### Did inline execution actually happen?

`synchronous = true` is a request. Whether the plugin honoured it is
`synchronous_mode()`:

| `SyncMode` | Meaning |
|---|---|
| `Inline` | The plugin advertises `supports_synchronous_execution` and the option was accepted. Computations run on the calling thread. |
| `Accepted` | The option was sent and client creation succeeded, but the plugin does not advertise the marker attribute. This is the best that can be said. |
| `Rejected` | The plugin refused the option and the client was created without it. Execution is asynchronous despite the request. |
| `Async` | `synchronous = false` was asked for; the option was never sent. |

`synchronous_supported()` collapses the first two to `true`. A rejected option
costs latency and never correctness, which is exactly why the fallback is
allowed to be silent by default — and exactly why `describe()` should be logged
once at startup. It names the plugin, the platform, the API version, the
execution mode and the thread configuration, and it is the first thing to ask
for when a latency number looks wrong.

:::{note}
As of the pinned XLA commit the CPU plugin **validates create option names** and
fails creation with `InvalidArgument("Unexpected option name passed to
PJRT_Client_Create")` on one it does not recognize. It used to ignore them
silently. That is why `Runtime` queries `PJRT_Plugin_Attributes` — which needs
no client — before deciding which options to send, and retries without a
rejected one.
:::

## Function

```{literalinclude} ../../examples/01_basic/basic.cpp
:language: cpp
:start-after: docs: begin load
:end-before: docs: end load
```

The base path is given without an extension: `artifacts/trajopt` reads
`artifacts/trajopt.json` and then `trajopt.binpb` or `trajopt.mlirbc` from the
same directory. Writing the `.json` out in full works too, because a shell
completes file names and not stems.

### What load time does

Everything that can possibly happen before the loop starts:

1. Reads the sidecar and cross-checks it against `PJRT_Executable_NumOutputs`,
   `..._OutputElementTypes` and `..._OutputDimensions`. A sidecar that has gone
   stale relative to its executable is otherwise a buffer-overrun class of bug,
   discovered later as corrupted output. Inputs cannot be checked this way —
   the PJRT C API has no parameter query — so an input mismatch surfaces as a
   warm-up failure instead.
2. Chooses an artifact under `load_policy` and the ISA guard, then deserializes
   or compiles it.
3. `posix_memalign(64, ...)` one arena per input and per output. Alignment is
   not cosmetic: below `xla::cpu::MinAlign()` XLA falls back to copying the
   buffer and says nothing, and the entire benefit disappears without a
   diagnostic.
4. Wraps each input arena once in a zero-copy `PJRT_Buffer`, for the life of
   the `Function`.
5. Sizes every per-call array, so `call()` never grows a vector.
6. Runs `warmup_calls` blocking calls on zeroed arenas, to fault in the pages
   and warm the runtime's lazy state. The first call after a load is always the
   slowest; this is where it gets spent.

### Introspection

Indices are the sidecar's, which are JAX's argument and result order. Resolve
names once, at startup:

```cpp
const std::size_t x0 = *f.find_input("x0");
```

`find_input` / `find_output` return `std::optional<std::size_t>` and are a
linear scan over the names — startup code, not loop code.

| Accessor | Returns |
|---|---|
| `name()` | The function's name from the sidecar, used in every error message. |
| `num_inputs()`, `num_outputs()` | Counts, flattened. |
| `input_spec(i)`, `output_spec(i)` | The whole `ArraySpec`: name, dtype, shape, numel, nbytes, donated. Bounds-checked in **every** build. |
| `input_dtype(i)`, `output_dtype(i)` | The `DType` the arena holds. |
| `input_shape(i)`, `output_shape(i)` | Row-major dimensions; empty for a scalar. |
| `input_rank/numel/nbytes(i)` and the output twins | The three numbers a `memcpy` needs. |
| `load_kind()`, `load_detail()` | Deserialized or compiled, and which file and why. |
| `fingerprint()` | `PJRT_Executable_Fingerprint`, empty when unimplemented. Two processes reporting the same fingerprint are running the same compiled program — the cheap way to confirm a benchmark and a deployment agree. |

### Reading and writing the arenas

`input<T>(i)` returns a writable `T*` into storage the `Function` owns and XLA
actually reads. `output<T>(i)` returns a `const T*` valid until the next call.
`T` must be one of the eleven types `dtype_of` names; anything else is a
compile error naming the offending type, not a run-time surprise.

:::{important}
**The dtype and bounds checks in the typed accessors run only under
`FunctionOptions::debug`; with debug off they are a pointer load.** The branch
is on a member that is always in cache and always predicted, so leaving debug
on outside the loop costs nothing measurable — and turning it off inside the
loop is what makes the accessor free.
:::

Generic code — a node that loads whatever artifact it is configured with —
goes through the untyped accessors and switches on the dtype:

```cpp
switch (f.input_dtype(i)) {                       // sketch, not from the tree
  case pjrt::DType::Float64:
    std::memcpy(f.input_raw(i), src, f.input_nbytes(i));
    break;
  case pjrt::DType::Int32:
    /* ... */
    break;
  default:
    throw std::runtime_error(std::string("unhandled dtype ") +
                             pjrt::dtype_name(f.input_dtype(i)));
}
```

`input_raw(i)` and `output_raw(i)` hand back `void*` and `const void*` and are
bounds-checked on the same terms as the typed accessors. The dtype names in
`dtype_name()` are exactly the strings the sidecar carries, so an error message
built from them matches what `python -m jax2exec check` prints.

:::{tip}
**Grab the pointers once, outside the loop.** `f.input<double>(x0)` is cheap,
but it is not free, and there is no reason to pay for it fifty thousand times:
the arenas do not move for the life of the `Function`.
:::

:::{tip}
**Run debug mode in development.** The natural spelling is to tie it to the
build:

```cpp
pjrt::FunctionOptions options;
#ifndef NDEBUG
options.debug = true;         // bounds, dtype and re-entrancy checks
options.check_values = true;  // and a scan for nan/inf and bad bools
#endif
```

With debug off, the same mistake is silent memory corruption rather than an
exception. That trade is the whole reason the checks are optional; it is not a
reason to develop without them.
:::

### FunctionOptions

| Field | Default | Meaning |
|---|---|---|
| `warmup_calls` | `3` | Calls made at load and discarded, to fault in the arenas and warm the runtime. |
| `check_metadata` | `true` | Cross-check the sidecar against the executable's outputs at load. |
| `debug` | `false` | Per-call bounds and dtype checks, plus the re-entrancy check on `call()`. |
| `check_values` | `false` | Audit every element of every arena before and after each call: no nan or inf in a float arena, nothing but 0 or 1 in a bool arena. A development and acceptance-test tool, not something to run in a control loop. |
| `load_policy` | `Auto` | `Auto` prefers the `.binpb` and falls back to the `.mlirbc`; `BinaryOnly` deserializes or fails; `CompileOnly` always compiles. |
| `isa_guard` | `true` | Under `Auto`, skip a `.binpb` whose sidecar records an `isa_level` this host does not implement. Without it the failure is an illegal instruction inside the executable, with no hint that the artifact came from a newer machine. |
| `compile_options` | `""` | Serialized `CompileOptionsProto` for the `.mlirbc` path. Ignored when the `.binpb` is used, which never recompiles. |

A deployment usually wants `load_policy = BinaryOnly`: a fallback that quietly
compiles for several seconds is not a fallback in a control loop.

## The two rules the API cannot enforce

### Write the inputs between calls, never during one

The PJRT buffers alias the arenas for the life of the `Function`, and XLA is
reading them while `call()` runs. Writing from another thread, or from a signal
handler, mid-call is a data race on the computation's own operands.

```{literalinclude} ../../examples/common/trajopt_signature.hpp
:language: cpp
:start-after: docs: begin recirculate
:end-before: docs: end recirculate
```

Between calls it is safe, and that is the property this whole design rests on.
It was verified rather than assumed: zero-copy input buffers genuinely alias
the caller's pointer on CPU, and **writes made between executions are seen by
the next one**. Both host-buffer semantics documents forbid mutating a buffer
while it is alive; doing it between calls is a deliberate, measured bend of
that contract, because nothing is in flight. The reference-case sweep — every
case forwards and backwards through one reused set of buffers, compared against
what JAX returned — is what keeps it honest.

### One Function per thread

A `Function` owns fixed storage, so sharing one between threads means sharing
those arenas and the executable's per-call state. `Runtime` is the object to
share: create it once, then give each thread its own `Function` loaded from the
same artifact. Each pays its own load cost and its own arenas, and neither can
corrupt the other.

## Exceptions

| Exception | Raised for | When |
|---|---|---|
| `pjrt::Error` | A PJRT C API failure: client creation, deserialize, compile, execute. Carries the plugin's `code()`. | Load and call. |
| `pjrt::LoadError` | An artifact or plugin problem the caller can act on: no plugin, an unreadable or stale sidecar, an element type this project does not support, a `.binpb` built for a wider instruction set than this host has. | Load only. |
| `std::out_of_range` | An index past the end. Always checked in `input_spec`/`output_spec`; in the typed and raw accessors only under `debug`. | |
| `std::invalid_argument` | `input<T>()` or `output<T>()` where `T` disagrees with the dtype the sidecar declares. | `debug` only. |
| `std::logic_error` | `call()` re-entered — from a signal handler, or from a callback the computation itself triggered. | `debug` only. |
| `std::domain_error` | A nan, an inf, or a bool byte other than 0 or 1, in a named arena at a named element. | `check_values` only. |

Every `LoadError` happens at startup, none during a call. That is the division
the taxonomy exists to make: a control loop that has entered its loop has
already survived every artifact problem there is.

{doc}`debugging` has one row per message.

## What `call()` costs

Execute, one await, and one `memcpy` per output read straight out of device
memory — outputs are ordinary memory on CPU, so nothing round-trips through
`PJRT_Buffer_ToHostBuffer` and its event.

What it does **not** do: allocate, take a lock, log, flush, copy an input to
the device, query a shape, or grow a vector. All of those were in the older
per-call-buffer path, and removing them is where the tail improvement came
from.

What it cannot do anything about: roughly **thousands of allocations per call inside
XLA's thunk runtime**, about one per StableHLO op. Those happen behind the
plugin's C API boundary and are not reachable from here. The number that must
stay at zero is the wrapper's own, and that is what the allocation census
measures ({doc}`measuring`).

`call()` blocks until the outputs are in their arenas. There is no cancellation
to be had: **PJRT cannot cancel a running CPU computation.** A watchdog can
only return a stale result and discard the late one — an overrun means late
data, not a cancelled call. Design the caller accordingly.

## Anti-patterns

| Don't | Why |
|---|---|
| Construct a `Function` per call | Load time reads files, allocates arenas, creates buffers and runs warm-up calls. It is startup work, and it is the thing this API exists to hoist out of the loop. |
| Share one `Function` across threads | Two threads writing one set of arenas is a race on the computation's operands, not a lock contention problem. Share the `Runtime`. |
| Hold an output pointer across calls | Output arenas are overwritten by the next `call()`. Copy what has to outlive it. |
| Write an input from another thread | Safe only between calls, and "between calls" is a property of the calling thread's own control flow. Anything else is the data race in the first rule. |
| Destroy and recreate the `Runtime` mid-run | Client creation starts XLA's thread pools and lazy statics. It is a guaranteed spike, and destroying one while a `Function` lives is undefined. |
| Call `find_input` in the loop | It is a linear scan over strings. Resolve to indices at startup. |
