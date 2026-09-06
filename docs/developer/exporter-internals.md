# Exporter internals

What `jax2exec.export` does between being called and writing a file, and the
three traps it exists to catch. {doc}`../guides/exporting` is the user-facing
version; this page holds the reasons.

## Nothing is written before the checks

The exporter this replaced wrote the executable first and asserted afterwards,
so a rejected function left a stale `.binpb` beside a sidecar describing
something else — which the C++ side then loaded, and which failed a long way
from the cause. `export` now runs every validation, compiles, serializes, and
only then writes, atomically, in the order `.binpb`, `.mlirbc`, `.json`: a
reader never sees a sidecar whose artifacts are missing.

## The x64 trap, in full

Without `jax_enable_x64`, JAX traces a float64 argument as float32 and says
nothing. Everything downstream is then honest about the wrong thing: the
sidecar records `float32`, the loader allocates four bytes per element, and a
C++ caller that still says `double` writes twice as far as the arena goes.

The exporter compares the dtype you asked for against the dtype JAX traced,
and refuses when they differ:

```text
input 0 ('x0') was requested as float64 but JAX traced it as float32:
call jax.config.update('jax_enable_x64', True) before exporting
```

The check fires only when the requested dtype is visible — an argument
declared float32 and traced float32 is a float32 function, not a trap. The
sidecar records `export.x64_enabled`, and `python -m jax2exec check` prints it
as `x64 on` or `x64 OFF`, so the answer is in the file. With debug mode on,
the C++ side catches the other half of the same mistake: `input<double>(0)` on
a float32 input throws `std::invalid_argument` naming both dtypes.

## The pruned-parameter trap

XLA drops a parameter the computation never reads, and the executable then
takes fewer arguments than the sidecar declares. The C++ loader cannot see
this — the PJRT C API has no parameter query — so it arrives as an opaque
warm-up failure:

```text
Execution supplied 16 buffers but compiled program expected 4
```

That message cost real time the first time: a synthetic benchmark kernel with
16 inputs came back expecting 4, because only four of them reached an output.
The exporter now checks `kept_var_idx` and refuses:

```text
inputs 5 ('w5'), 6 ('w6') do not reach any output, so XLA dropped them from
the executable: it takes 14 of the 16 inputs the sidecar declares. Make an
output depend on every input (even through a multiply by zero), or stop
passing it.
```

Both remedies are legitimate. A synthetic fixture wants the multiply by zero;
a real function usually wants the argument removed. The same fact bites a
probe: a perturbed input that does not reach the output being checked reports
a confident negative — see trap 5 in {doc}`measurement`.

## Unsupported element types

`float16`, `bfloat16`, `complex64`, `complex128`, the float8 families, the
sub-byte integers and JAX's extended dtypes (PRNG keys, `float0`) are rejected
by name, before anything is written:

```text
input 2 ('theta') has dtype bfloat16, which jax2exec does not support
(supported: bool, int8/16/32/64, uint8/16/32/64, float32, float64);
cast inside the function
```

They are absent because none of them has a C++ storage type this API could
hand back. Casting inside the traced function is the way out: compute in
bfloat16 if that is what the numerics want, and return float32. The exporter,
the sidecar and the C++ loader all have to agree about the spellings, so the
table is written once:

```{literalinclude} ../../python/jax2exec/_dtypes.py
:language: python
:start-after: docs: begin dtype-table
:end-before: docs: end dtype-table
```

## Donation is recorded, not exploited

`donate_argnums` is passed to `jax.jit` and recorded in the sidecar. The C++
runtime wraps each input arena once and reuses it for the life of the
`Function`, and a donated buffer is consumed by the execution — the two are
mutually exclusive, so every input is listed in `non_donatable_input_indices`
on every call, precisely so a may-alias in the compiled program cannot destroy
a buffer the next call still needs. See {doc}`open-threads`.

## Freezing what JAX returns

A C++ call path that runs is not the same as one that is right.
`jax2exec.reference.write_reference_cases(fun, arg_tuples, directory, name)`
evaluates the function on a list of argument tuples and writes every input and
output as raw bytes in call order, plus a manifest saying what those bytes are
and how close a match has to be — `1e-6` relative for float64, `1e-4` for
float32, because XLA is free to fuse and reassociate and the two paths need
not reassociate the same way.

The C++ tests read those files and sweep the cases forwards and then
backwards. The backwards pass is the point: it verifies that reusing one input
buffer across calls with changing data is bit-exact, rather than accidentally
correct because the cases were visited in the order they were recorded.

## The IFRT envelope

jaxlib wraps the serialized executable in an IFRT envelope that the PJRT C API
cannot deserialize; the exporter unwraps it before writing the `.binpb`, and
without that step every load silently falls back to compiling the `.mlirbc`.
The wire format and the check that catches it are in
{ref}`Bumping JAX, step 12 <ifrt-envelope>`.
