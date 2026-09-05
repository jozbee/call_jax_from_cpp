# Exporting a JAX function

`jax2exec.export` compiles a function ahead of time and writes three files:
the serialized PJRT executable, StableHLO bytecode to compile in-process
instead when that executable will not run here, and a JSON sidecar describing
every input and output. Nothing reaches the disk until every check has passed.

That ordering is not incidental. The exporter this replaces wrote the
executable first and asserted afterwards, so a rejected function left a stale
`.binpb` beside a sidecar describing something else — which the C++ side then
loaded, and which failed a long way from the cause.

## The three files

| File | Contents | Portable? |
|---|---|---|
| `<name>.binpb` | The serialized PJRT executable, with target machine code embedded. | No: architecture- and ISA-locked to the exporting machine. |
| `<name>.mlirbc` | StableHLO bytecode from `jax.export`. | Yes: compiled in-process, seconds at load. |
| `<name>.json` | The sidecar. Every input and output by name, dtype, shape, `numel` and `nbytes`; the jax and jaxlib versions; the exporting host and its ISA level; a sha256 per artifact. | Yes, and it is the only description of the inputs that exists. |

The sidecar is load-bearing rather than convenient: the PJRT C API has **no
query for parameter shapes**. The loader can cross-check the outputs against
`PJRT_Executable_NumOutputs`, `..._OutputElementTypes` and
`..._OutputDimensions`, and for the inputs it has the sidecar and nothing else.
Field-by-field detail is in {doc}`../api/artifact-format`.

## What a function has to look like

```python
result = export(fun, args, directory, name)
```

| | |
|---|---|
| `fun` | Anything `jax.jit` accepts. It is traced once, for CPU. |
| `args` | Example arguments, **positional only**. `jax.ShapeDtypeStruct(shape, dtype)` avoids materializing data that is only wanted for its shape. |
| `directory` | Where the artifacts go; created if missing. |
| `name` | Base name matching `[A-Za-z0-9_.-]+`. It becomes a file name and a C++ load path. |

Five rules follow from the executable being a positional, flat, single-shot
thing:

- **Trace with `ShapeDtypeStruct`.** Passing real arrays works and costs
  memory for nothing; passing a Python scalar does not work, because there is
  no dtype to record.
- **One array per positional argument.** A pytree argument flattens, and the
  flattened order is what the executable takes.
- **No keyword arguments.** `export` refuses them: a positional C++ call has
  nowhere to put them.
- **Outputs are the flattened pytree leaves, in order.** A dict or a
  namedtuple names its own leaves and those names reach the sidecar; a plain
  tuple carries only position, and gets `out_0`, `out_1`, and so on.
- **Closures and constants are baked in.** Anything the traced function closes
  over becomes part of the executable. Changing it means re-exporting.

Names are worth setting deliberately: `input_names=("x0", "u")` — or leaving
them to the parameter names of `fun`, which is what happens when they line up
one-to-one with the flattened inputs. The C++ side resolves them once at
startup with `find_input` / `find_output` and uses indices thereafter.

The remaining keyword arguments: `donate_argnums` is passed to `jax.jit` and
recorded, `write_mlir=False` skips the portable fallback, `verify=False` skips
the in-process deserialize check, and `overwrite=False` refuses rather than
replacing existing artifacts.

:::{note}
Donation is recorded faithfully and not yet exploited. The C++ runtime wraps
each input arena once and reuses it for the life of the `Function`, and a
donated buffer is consumed by the execution — the two are mutually exclusive.
Every input the sidecar does not mark donated is listed in
`non_donatable_input_indices` on every call, precisely so a may-alias in the
compiled program cannot destroy a buffer the next call still needs.
:::

## Supported element types

Eleven types, chosen because each maps onto a C++ scalar the caller can spell.

| JAX / NumPy | Sidecar | C++ | PJRT | Needs `jax_enable_x64` |
|---|---|---|---|:--:|
| `bool_` | `bool` | `bool` | `PRED` | |
| `int8` | `int8` | `std::int8_t` | `S8` | |
| `int16` | `int16` | `std::int16_t` | `S16` | |
| `int32` | `int32` | `std::int32_t` | `S32` | |
| `int64` | `int64` | `std::int64_t` | `S64` | ✓ |
| `uint8` | `uint8` | `std::uint8_t` | `U8` | |
| `uint16` | `uint16` | `std::uint16_t` | `U16` | |
| `uint32` | `uint32` | `std::uint32_t` | `U32` | |
| `uint64` | `uint64` | `std::uint64_t` | `U64` | ✓ |
| `float32` | `float32` | `float` | `F32` | |
| `float64` | `float64` | `double` | `F64` | ✓ |

`float16`, `bfloat16`, `complex64`, `complex128`, the float8 families, the
sub-byte integers and JAX's extended dtypes (PRNG keys, `float0`) are rejected
by name, with a message that lists the supported set:

```text
input 2 ('theta') has dtype bfloat16, which jax2exec does not support
(supported: bool, int8/16/32/64, uint8/16/32/64, float32, float64);
cast inside the function
```

They are absent because none of them has a C++ storage type this API could
hand back. Casting inside the traced function is the way out: compute in
bfloat16 if that is what the numerics want, and return float32.

:::{dropdown} The table itself, from the source
The exporter, the sidecar and the C++ loader all have to agree about these
four spellings, so they are written once:

```{literalinclude} ../../python/jax2exec/_dtypes.py
:language: python
:start-after: docs: begin dtype-table
:end-before: docs: end dtype-table
```
:::

(x64-trap)=

## The x64 trap

Without `jax_enable_x64`, **JAX traces a float64 argument as float32 and says
nothing.** Everything downstream is then honest about the wrong thing: the
sidecar records `float32`, the loader allocates four bytes per element, and a
C++ caller that still says `double` writes twice as far as the arena goes.

Set the flag before tracing, once, at the top of the export script:

```python
jax.config.update("jax_enable_x64", True)
```

The exporter compares what you asked for against what JAX traced and refuses
when they differ:

```text
input 0 ('x0') was requested as float64 but JAX traced it as float32:
call jax.config.update('jax_enable_x64', True) before exporting
```

That check only fires when the requested dtype is visible — an argument
declared float32 and traced float32 is not a trap, it is a float32 function.
The rule to remember: **if your sidecar says `float32` where you expected
`float64`, x64 was off.** The sidecar records `export.x64_enabled` so the
answer is in the file itself, and `python -m jax2exec check` prints it as
`x64 on` or `x64 OFF`.

With debug mode on, the C++ side catches the other half of the same mistake:
`input<double>(0)` on a float32 input throws `std::invalid_argument` naming
both dtypes. With debug off it is silent memory corruption. See
{doc}`calling`.

## Shapes

Any rank. Row-major, dense, no strides. A scalar has shape `[]` and `numel`
1 — it still gets an arena, and the C++ accessor still returns a pointer to one
element.

`nbytes` is `numel * itemsize`, and that is exactly what the loader allocates
and what a per-call `memcpy` should be given: `f.input_nbytes(i)`, not a
recomputed product. Zero-element arrays are refused, because nothing
establishes what `posix_memalign(64, 0)` followed by `BufferFromHostBuffer`
does.

## What export refuses, and why

Every one of these is raised before anything is written.

| Refusal | Cause |
|---|---|
| `keyword arguments are not supported` | The executable takes positional parameters. |
| unsupported dtype | No C++ storage type; see the table above. |
| requested float64, traced float32 | `jax_enable_x64` was off. |
| zero-element array | Untested on the zero-copy arena path. |
| `function has effects (io_callback/debug_print)` | Those compile into a call back into a Python process the C++ runtime does not have. |
| `function lowers for N devices` | The runtime creates a single-device CPU client. |
| an input XLA pruned | See below. |
| `name ... must match [A-Za-z0-9_.-]+` | It becomes a file name and a load path. |

### The pruned-parameter trap

**XLA drops a parameter the computation never reads**, and the executable then
takes fewer arguments than the sidecar declares. The C++ loader cannot see
this — there is no parameter query — so it arrives as an opaque warm-up
failure:

```text
Execution supplied 16 buffers but compiled program expected 4
```

That message cost real time to diagnose the first time: a synthetic benchmark
kernel with 16 inputs came back expecting 4, because only four of them reached
an output. The exporter now checks `kept_var_idx` and refuses:

```text
inputs 5 ('w5'), 6 ('w6') do not reach any output, so XLA dropped them from
the executable: it takes 14 of the 16 inputs the sidecar declares. Make an
output depend on every input (even through a multiply by zero), or stop
passing it.
```

Both remedies are legitimate. A synthetic fixture wants the multiply by zero;
a real function usually wants the argument removed.

## The architecture lock

`PJRT_Executable_DeserializeAndLoad` **relinks; it never recompiles**. A
`.binpb` therefore carries machine code for the machine that exported it, and
running it on a host without those instructions is a SIGILL somewhere deep
inside the executable, with a backtrace that says nothing about artifacts.

Two things follow:

1. **Export on the machine that will execute**, or on one that matches it.
   `artifacts/` is gitignored for this reason: the files do not travel.
2. **The `.mlirbc` is the answer when they must travel.** The sidecar records
   the exporting host's ISA level, the loader compares it against this host's
   before it opens the `.binpb`, and compiles the bytecode instead when the
   comparison does not come out in its favour.

Which happened is `Function::load_kind()` — `LoadKind::Deserialized` or
`LoadKind::Compiled` — and `load_detail()` says which file and why. Compiling
costs seconds at load and nothing per call. A deployment that cannot afford a
surprise several-second startup should set
`FunctionOptions::load_policy = LoadPolicy::BinaryOnly` and fail loudly
instead.

## Custom calls and `jnp.linalg`

`jnp.linalg.*` does not lower to pure HLO. `jnp.linalg.inv`, for instance,
lowers to the custom call `lapack_dgetrf_ffi`, and **only a plugin that
registers that handler can load the result**. jaxlib registers its LAPACK FFI
kernels when it is imported; a bare PJRT plugin never imports jaxlib, so a
stock plugin fails at load:

```text
No FFI handler registered for lapack_dgetrf_ffi on a platform Host
```

The plugin this project ships carries those kernels — that is the first of the
two patches in the XLA fork ({doc}`../developer/xla-fork`). If you are on a
stock plugin, either avoid `jnp.linalg` in the exported function or get the
patched one ({doc}`../getting-started/installation`).

## A worked export

```{literalinclude} ../../examples/01_basic/export.py
:language: python
:start-after: docs: begin export
:end-before: docs: end export
```

`export` returns an `ExportResult`: the three paths, the sidecar as a dict, and
the `jax.stages.Compiled` it came from — the last of which lets a test run the
same function in-process and compare.

## Freezing what JAX returns

A C++ call path that runs is not the same as a C++ call path that is right.
`jax2exec.reference.write_reference_cases(fun, arg_tuples, directory, name)`
evaluates the function on a list of argument tuples and writes every input and
output as raw bytes in call order, plus a manifest saying what those bytes are
and how close a match has to be (`1e-6` relative for float64, `1e-4` for
float32, because XLA is free to fuse and reassociate and the two paths need
not reassociate the same way).

The C++ tests read those files and sweep the cases forwards and backwards.
That sweep is what verifies the property this whole design rests on: that
reusing one input buffer across calls with changing data is bit-exact.

## Inspecting an artifact

```console
$ python -m jax2exec check artifacts/basic
basic  (artifacts/basic.json)
  schema 2, written by jax2exec 0.2.0
  jax <jax_version> / jaxlib <jaxlib_version>, platform cpu
  exported 2026-09-05T09:41:07Z on linux/x86_64 x86-64-v3, python 3.12.14, x64 on

inputs (2)
  idx  name  dtype    shape   numel  nbytes
    0  A     float64  [4, 4]     16     128
    1  b     float64  [4]         4      32

outputs (2)
  idx  name   dtype    shape  numel  nbytes
    0  out_0  float64  [4]        4      32
    1  out_1  float64  []         1       8

artifacts
  basic.binpb: 42104 bytes, sha256 matches
  basic.mlirbc: 3216 bytes, sha256 matches

instruction set
  host is x86-64-v3, exported on x86-64-v3: the .binpb runs
```

The inputs took their names from the parameters of `fun`; the outputs are a
plain tuple, which carries nothing but position, so they are `out_0` and
`out_1`. `export` takes `input_names` but derives the output names from the
result pytree, so the way to name an output is to return a dict or a
namedtuple.

It needs no JAX, which is the point: the machine running the C++ caller usually
has none. It exits 0 when the set is consistent and will run here, 1 when it is
not — a sha256 mismatch, a sidecar whose `numel` disagrees with its own shape,
or an executable built for a wider instruction set than this host implements —
and 2 when the sidecar could not be read at all.
