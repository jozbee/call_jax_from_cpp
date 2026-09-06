# XLA, StableHLO and PJRT

*Assumes nothing about the JAX stack. Four names, one paragraph each, then
what they mean for a program that loads an exported function. How this
project uses them is {doc}`/guides/how-it-works`; what was verified against
the plugin is {doc}`/developer/runtime-internals`.*

## Four names

**JAX** is the Python library. It traces a function once, with abstract inputs
of fixed shape and dtype, into an intermediate representation, and compiles
that; `jax.jit` is the tracing. Anything the trace cannot freeze — a loop
whose trip count depends on the data, a shape that changes — has to be
rewritten so that it can.
[Key concepts](https://docs.jax.dev/en/latest/key-concepts.html),
[jit](https://docs.jax.dev/en/latest/jit-compilation.html),
[control flow](https://docs.jax.dev/en/latest/control-flow.html).

**{term}`StableHLO`** is the portable operation set the trace is lowered to: a
dialect of MLIR with a versioned serialization, the `.mlirbc` bytecode. It is
the contract between a front end and any compiler, and it is not machine code.
[StableHLO](https://openxla.org/stablehlo), [the spec](https://openxla.org/stablehlo/spec).

**{term}`XLA`** is the compiler. Its CPU backend fuses operations, lowers the
result to LLVM IR and emits machine code for the host it runs on — which is
why a compiled executable is locked to an architecture. At run time XLA's CPU
"thunk" runtime walks the compiled program operation by operation, and that
walk allocates, inside the plugin, out of reach of anything above it. Its
worker {term}`thread pool` is created with the client and its threads carry
XLA's name, which is what lets a program find them and move them.
[XLA](https://openxla.org/xla), [architecture](https://openxla.org/xla/architecture),
[the CPU runtime](https://github.com/openxla/xla/tree/main/xla/backends/cpu/runtime).

**{term}`PJRT`** is the runtime API: a C struct of function pointers that a
{term}`PJRT plugin` — a shared object with one entry symbol — implements, and
that a client calls to create devices and buffers, compile or deserialize
executables, and run them. JAX drives XLA through the same API. The struct is
versioned by its size, so a plugin and a caller built against different
headers can still agree on what they share.
[PJRT](https://openxla.org/xla/pjrt),
[pjrt_c_api.h](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h).

## Ahead of time

Export runs the trace, the compile and the serialization once, in Python, and
writes an {term}`artifact` of three files: the compiled executable, the
StableHLO bytecode, and a {term}`sidecar` describing the signature — the
executable can be asked about its outputs and nothing about its inputs.
Everything is frozen at that point: shapes, dtypes, constants, every trip
count. That is what makes a call's cost a property of the system rather than
of the data, and it is why the sidecar is cross-checked at load rather than
trusted.
[jax.export](https://docs.jax.dev/en/latest/export/export.html),
[ahead-of-time compilation](https://docs.jax.dev/en/latest/aot.html).

## The plugin in the process

Loading the plugin `dlopen`s it and creates a client, and creating the client
starts XLA's pools. Loading an executable relinks its machine code — no
compiler runs — and every call from then on is: execute, wait for one event,
copy each output out. Whether the computation runs on the calling thread or is
dispatched to a pool is a plugin option ({term}`inline execution`), and it is
the difference between a call that *is* the computation and a call that is the
computation plus a wake-up. Inside the plugin, the thunk runtime allocates as
it walks the program either way.

## Two things that bite

**x64.** JAX computes in 32-bit floats and integers unless `jax_enable_x64`
is set, so an export made without it is a `float32` function however the
Python was written. The sidecar records what was exported, and debug mode
refuses a `double` accessor on it; {ref}`the x64 rule <x64-trap>` says how to
make the export say what you meant.

**Custom calls.** Some operations — `jnp.linalg.inv` among them — lower to a
call into a library the plugin has to carry. A stock CPU plugin does not carry
LAPACK; this project's fork does. {doc}`/developer/xla-fork`.

## Vocabulary

{term}`artifact`, {term}`sidecar`, {term}`PJRT plugin`,
{term}`inline execution`, {term}`zero-copy buffer` — two lines each in the
{doc}`glossary`.
