# Exporting a JAX function

`jax2exec.export` compiles a function ahead of time and writes three files:
the serialized PJRT executable, StableHLO bytecode to compile in-process when
that executable will not run here, and a JSON sidecar describing every input
and output. Nothing reaches the disk until every check has passed.

## The three files

| File | Contents | Portable? |
|---|---|---|
| `<name>.binpb` | The serialized PJRT executable, with target machine code embedded. | No: architecture- and ISA-locked to the exporting machine. |
| `<name>.mlirbc` | StableHLO bytecode from `jax.export`. | Yes: compiled in-process, seconds at load. |
| `<name>.json` | The sidecar. Every input and output by name, dtype, shape, `numel` and `nbytes`; the jax and jaxlib versions; the exporting host and its ISA level; a sha256 per artifact. | Yes, and it is the only description of the inputs that exists. |

The sidecar is load-bearing rather than convenient: the PJRT C API has no
query for parameter shapes, so for the inputs the loader has the sidecar and
nothing else. Field-by-field detail is in {doc}`../api/artifact-format`.

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

- **Trace with `ShapeDtypeStruct`.** A real array works and costs memory for
  nothing; a Python scalar does not work, because there is no dtype to record.
- **One array per positional argument.** A pytree argument flattens, and the
  flattened order is what the executable takes.
- **No keyword arguments.** A positional C++ call has nowhere to put them.
- **Outputs are the flattened pytree leaves, in order.** A dict or a
  namedtuple names its leaves and those names reach the sidecar; a plain tuple
  gets `out_0`, `out_1`, and so on.
- **Closures and constants are baked in.** Changing one means re-exporting.

Names are worth setting deliberately: `input_names=("x0", "u")`, or leave
them to the parameter names of `fun`. The C++ side resolves them once at
startup and uses indices thereafter. The remaining keyword arguments —
`donate_argnums`, `write_mlir`, `verify`, `overwrite` — are on
{doc}`../api/python`.

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

`float16`, `bfloat16`, the complex types and the rest are rejected by name
before anything is written; cast inside the function and return one of the
eleven. Any rank is accepted: row-major, dense, a scalar has shape `[]` and
`numel` 1, and `nbytes` is exactly what the loader allocates.

(x64-trap)=

## The x64 rule

Without `jax_enable_x64`, JAX traces a float64 argument as float32 and says
nothing. Set the flag before tracing, once, at the top of the export script:

```python
jax.config.update("jax_enable_x64", True)
```

The exporter refuses when the dtype you asked for differs from the one JAX
traced, and the sidecar records `export.x64_enabled` so the answer is in the
file. If a sidecar says `float32` where you expected `float64`, x64 was off.

## What export refuses

Every one of these is raised before anything is written.

| Refusal | Cause |
|---|---|
| `keyword arguments are not supported` | The executable takes positional parameters. |
| unsupported dtype | No C++ storage type; see the table above. |
| requested float64, traced float32 | `jax_enable_x64` was off. |
| zero-element array | Untested on the zero-copy arena path. |
| `function has effects (io_callback/debug_print)` | Those call back into a Python process the C++ runtime does not have. |
| `function lowers for N devices` | The runtime creates a single-device CPU client. |
| an input XLA pruned | An input that reaches no output is dropped from the executable, so the sidecar would lie about the arity. Make every input reach an output. |
| `name ... must match [A-Za-z0-9_.-]+` | It becomes a file name and a load path. |

## The architecture lock

A `.binpb` carries machine code for the machine that exported it; loading
relinks it and never recompiles. Two things follow:

1. **Export on the machine that will execute**, or on one that matches it.
   `artifacts/` is gitignored for this reason.
2. **The `.mlirbc` is the answer when artifacts must travel.** The sidecar
   records the exporting host's ISA level, the loader compares it with this
   host's before opening the `.binpb`, and compiles the bytecode instead when
   they disagree. `Function::load_kind()` reports which happened.

Compiling costs seconds at load and nothing per call. A deployment that
cannot afford a surprise several-second startup sets
`FunctionOptions::load_policy = LoadPolicy::BinaryOnly` and fails loudly
instead.

## Custom calls and `jnp.linalg`

`jnp.linalg.inv` lowers to the custom call `lapack_dgetrf_ffi`, and only a
plugin that registers that handler can load the result. The plugin this
project ships carries jaxlib's LAPACK kernels; a stock plugin fails at load
with `No FFI handler registered for lapack_dgetrf_ffi on a platform Host`.
Either use the patched plugin ({doc}`../getting-started/installation`) or
keep `jnp.linalg` out of the exported function.

## A worked export

```{literalinclude} ../../examples/01_basic/export.py
:language: python
:start-after: docs: begin export
:end-before: docs: end export
```

`export` returns an `ExportResult`: the three paths, the sidecar as a dict,
and the `jax.stages.Compiled` it came from, which lets a test run the same
function in-process and compare.

## Inspecting an artifact

```console
$ python -m jax2exec check artifacts/basic
basic  (artifacts/basic.json)
  schema 2, written by jax2exec 0.2.0
  jax <jax_version> / jaxlib <jaxlib_version>, platform cpu
  exported <time> on linux/x86_64 x86-64-v3, python 3.12, x64 on

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

It needs no JAX, which is the point: the machine running the C++ caller
usually has none. It exits 0 when the set is consistent and will run here, 1
when it is not, and 2 when the sidecar could not be read at all.

## Deeper

{doc}`../developer/exporter-internals` — why nothing is written before the
checks, the x64 and pruned-parameter traps in full, freezing what JAX returns.
{doc}`../api/python` — every argument. {doc}`../api/artifact-format` — the
sidecar, field by field.
