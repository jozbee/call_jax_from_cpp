# Calling from C++

`pjrt::Runtime` is created once per process and `pjrt::Function` is loaded
once per artifact. After that a call writes the input arenas, calls `call()`,
and reads the output arenas. The steady-state path allocates nothing.

## The shape of a call

One cycle, with the loop taken away:

```{literalinclude} ../../examples/01_basic/basic.cpp
:language: cpp
:start-after: docs: begin call
:end-before: docs: end call
```

Everything that can be done before the loop is done before the loop: names
resolved to indices, pointers taken, sizes read. What is left is a write, a
`call()`, and a read. The loop-shaped version, where one cycle's outputs
become the next cycle's inputs, is
[further down](#write-the-inputs-between-calls-never-during-one).

## Runtime

One per process. Creating a client starts XLA's thread pools and its
lazily-initialized statics, so never destroy and recreate one mid-run, and
never let it go out of scope while a `Function` still holds an executable. The
default `RuntimeOptions` are the control-loop defaults: inline execution, one
device, one worker thread. The fields are on {doc}`../api/cpp/runtime`.

Two things to know before relying on them. `worker_threads` is applied with
`setenv("PJRT_NPROC", ...)`, a process-wide variable every later client
inherits — decide it once. And `synchronous = true` is a request: the plugin
may not honour it, and a rejected option costs latency, never correctness.
`synchronous_mode()` says what actually happened; log `describe()` once at
startup, because it is the first thing to ask for when a number looks wrong.

## Function

```{literalinclude} ../../examples/01_basic/basic.cpp
:language: cpp
:start-after: docs: begin load
:end-before: docs: end load
```

The base path is given without an extension: `artifacts/trajopt` reads
`artifacts/trajopt.json` and then `trajopt.binpb` or `trajopt.mlirbc` from
the same directory. Load time does everything that can possibly happen before
the loop — reads and cross-checks the sidecar, chooses between the `.binpb`
and the `.mlirbc`, allocates the arenas, wraps each input once, and runs the
warm-up calls. {doc}`how-it-works` has the sequence; the options are on
{doc}`../api/cpp/function`.

## Introspection

Indices are the sidecar's, which are JAX's argument and result order. Resolve
names once, at startup:

```cpp
const std::size_t x0 = *f.find_input("x0");
```

`find_input` / `find_output` return `std::optional<std::size_t>` and are a
linear scan over the names — startup code, not loop code. From an index,
`input_dtype`, `input_shape`, `input_numel` and `input_nbytes` (and the
`output_` twins) give everything a `memcpy` needs; `input_spec(i)` gives all
of it at once.

## Reading and writing the arenas

`input<T>(i)` returns a writable `T*` into storage the `Function` owns and XLA
actually reads. `output<T>(i)` returns a `const T*` valid until the next call.
`T` must be one of the eleven types `dtype_of` names; anything else is a
compile error naming the offending type. Generic code that loads whatever
artifact it is configured with uses `input_raw(i)` / `output_raw(i)` and
switches on `input_dtype(i)`; {doc}`../api/cpp/dtype` has the sketch.

The dtype and bounds checks in the typed accessors run only under
`FunctionOptions::debug`; with debug off they are a pointer load.

:::{tip}
**Grab the pointers once, outside the loop.** The arenas do not move for the
life of the `Function`, so there is no reason to look them up per call.
:::

:::{tip}
**Run debug mode in development.** The natural spelling ties it to the build:

```cpp
pjrt::FunctionOptions options;
#ifndef NDEBUG
options.debug = true;         // bounds, dtype and re-entrancy checks
options.check_values = true;  // and a scan for nan/inf and bad bools
#endif
```

With debug off, the same mistake is silent memory corruption rather than an
exception.
:::

## The two rules the API cannot enforce

### Write the inputs between calls, never during one

The PJRT buffers alias the arenas for the life of the `Function`, and XLA is
reading them while `call()` runs. Writing from another thread, or from a
signal handler, mid-call is a data race on the computation's own operands.

```{literalinclude} ../../examples/common/workload.hpp
:language: cpp
:start-after: docs: begin recirculate
:end-before: docs: end recirculate
```

Between calls it is safe, and that is the property this whole design rests
on. It was verified against the plugin rather than assumed, and the
reference-case sweep keeps it honest; {doc}`../developer/runtime-internals`
has the evidence.

### One Function per thread

A `Function` owns fixed storage, so sharing one between threads means sharing
those arenas and the executable's per-call state. `Runtime` is the object to
share: create it once, then give each thread its own `Function` loaded from
the same artifact.

## Exceptions

Every `pjrt::LoadError` — no plugin, a stale sidecar, an unsupported element
type, a `.binpb` built for a wider instruction set — happens at startup, none
during a call. A `pjrt::Error` is a PJRT failure and can also come from
`call()`. The standard exceptions (`std::out_of_range`,
`std::invalid_argument`, `std::logic_error`, `std::domain_error`) are thrown
only under `debug` or `check_values`. {doc}`../api/cpp/error` has the table
and {doc}`debugging` has one row per message.

## What `call()` costs

Execute, one await, and one `memcpy` per output straight out of device
memory. It does not allocate, lock, log, flush, copy an input, query a shape,
or grow a vector. It blocks until the outputs are in their arenas, and there
is no cancellation to be had: **PJRT cannot cancel a running CPU
computation**, so an overrun means late data, not a cancelled call.

## Anti-patterns

| Don't | Why |
|---|---|
| Construct a `Function` per call | Load time reads files, allocates arenas, creates buffers and runs warm-up calls. It is the thing this API exists to hoist out of the loop. |
| Share one `Function` across threads | Two threads writing one set of arenas is a race on the computation's operands. Share the `Runtime`. |
| Hold an output pointer across calls | Output arenas are overwritten by the next `call()`. Copy what has to outlive it. |
| Write an input from another thread | Safe only between calls, and "between calls" is a property of the calling thread's own control flow. |
| Destroy and recreate the `Runtime` mid-run | Client creation starts XLA's thread pools and lazy statics: a guaranteed spike, and undefined while a `Function` lives. |
| Call `find_input` in the loop | It is a linear scan over strings. Resolve to indices at startup. |

## Deeper

{doc}`../developer/runtime-internals` — what was verified against the plugin,
what `call()` does line by line, and why. {doc}`../api/cpp/function`,
{doc}`../api/cpp/runtime` — every option and accessor.
