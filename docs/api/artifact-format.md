# Artifact format

An export produces three files that share a base path. The C++ side is handed
that base path — `artifacts/trajopt` — and reads all three from it.

| File | What it is | Who reads it |
|---|---|---|
| `<name>.binpb` | the serialized PJRT executable | `PJRT_Executable_DeserializeAndLoad`, which relinks it |
| `<name>.mlirbc` | StableHLO bytecode for the same function | `PJRT_Client_Compile`, when the `.binpb` will not run here |
| `<name>.json` | the sidecar describing every input and output | the loader, and `python -m jax2exec check` |

The sidecar is load-bearing rather than a convenience. A serialized executable
will answer `PJRT_Executable_NumOutputs`, `..._OutputElementTypes` and
`..._OutputDimensions`, but **the PJRT C API has no query for parameter
shapes**. Everything about the inputs — how many, how wide, what to call them —
exists only here. That asymmetry decides where errors appear, and it is worth
holding on to while reading the rest of this page.

## Sidecar, schema 2

```{code-block} json
:caption: artifacts/trajopt.json -- digests, sizes, times and host details vary.

{
  "schema": 2,
  "name": "trajopt",
  "generator": { "tool": "jax2exec", "version": "0.2.0" },
  "jax_version": "0.11.1",
  "jaxlib_version": "0.11.1",
  "platform": "cpu",
  "export": {
    "time_utc": "2026-09-05T14:02:11Z",
    "x64_enabled": true,
    "host": {
      "os": "linux",
      "arch": "x86_64",
      "isa_level": "x86-64-v3",
      "python": "3.12.7",
      "cpu_model": "13th Gen Intel(R) Core(TM) i7-13700H"
    },
    "xla_flags": ""
  },
  "artifacts": {
    "executable": "trajopt.binpb",
    "executable_sha256": "9f2c1d0b8a7e6f5d4c3b2a1908f7e6d5c4b3a2918077665544332211aabbccdd",
    "mlir": "trajopt.mlirbc",
    "mlir_sha256": "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f809",
    "stablehlo_calling_convention_version": 9
  },
  "inputs": [
    { "index": 0, "name": "x0", "dtype": "float64", "shape": [6],
      "numel": 6, "nbytes": 48, "donated": false },
    { "index": 1, "name": "u0", "dtype": "float64", "shape": [40, 2],
      "numel": 80, "nbytes": 640, "donated": false }
  ],
  "outputs": [
    { "index": 0, "name": "u", "dtype": "float64", "shape": [40, 2],
      "numel": 80, "nbytes": 640 }
  ],
  "donation": { "donate_argnums": [] }
}
```

### Fields

| Field | Type | What it is for |
|---|---|---|
| `schema` | int | Sidecar layout. The loader accepts 1 and 2 and refuses anything newer. |
| `name` | string | The artifact base name, matching the file names. Appears in every error message. |
| `generator.tool`, `generator.version` | string | Which exporter wrote this, so an artifact traces back to the code that made it. |
| `jax_version`, `jaxlib_version` | string | The versions that produced the executable. A `.binpb` is not portable across a JAX bump; this is how you find out that is what happened. |
| `platform` | string | The export platform. Only `"cpu"` is exercised. |
| `export.time_utc` | string | When, in `YYYY-MM-DDThh:mm:ssZ`. |
| `export.x64_enabled` | bool | Whether `jax_enable_x64` was set while tracing. A sidecar full of `float32` where the caller expected `float64` is explained by this line and nothing else. |
| `export.host.os`, `.arch` | string | `platform.system().lower()` and `platform.machine()` of the exporting machine. |
| `export.host.isa_level` | string | `x86-64-v1`..`v4`, `aarch64`, `aarch64+sve`, or `unknown`. What the ISA guard compares against. |
| `export.host.python`, `.cpu_model` | string | Diagnostic context for a load that fails somewhere else. |
| `export.xla_flags` | string | `$XLA_FLAGS` as it was during the export, verbatim. |
| `artifacts.executable`, `.executable_sha256` | string | File name and digest of the `.binpb`. |
| `artifacts.mlir`, `.mlir_sha256` | string | The same for the `.mlirbc`. **Both keys are omitted entirely** rather than written as `null` when no bytecode was written, so presence alone tests for the fallback. |
| `artifacts.stablehlo_calling_convention_version` | int | `jax.export`'s calling convention version for that bytecode. |
| `inputs`, `outputs` | array | One entry per array, in executable order. |
| `donation.donate_argnums` | array of int | As passed to `jax.jit`. |

Each entry in `inputs` and `outputs`:

| Field | Type | What it is for |
|---|---|---|
| `index` | int | Position in the flattened list, which is the order the executable takes and returns them in. |
| `name` | string | What `find_input` / `find_output` look up. |
| `dtype` | string | A NumPy dtype name from the {doc}`supported eleven <cpp/dtype>`. |
| `shape` | array of int | The exact JAX shape, row-major. Empty for a scalar. |
| `numel` | int | Product of `shape`; 1 for a scalar. |
| `nbytes` | int | `numel * itemsize` — the size of the arena the loader allocates and the length of the per-call `memcpy`. |
| `donated` | bool | Inputs only; outputs omit the key. |

`numel` and `nbytes` are redundant with `shape` and `dtype`, and are stored
anyway because the loader allocates `nbytes` and reads `numel` elements out of
it. A hand-edited sidecar that disagrees with itself is a buffer overrun
waiting for a caller, which is why `python -m jax2exec check` recomputes both
and reports the disagreement.

## What the loader validates, and when

At load, with `FunctionOptions::check_metadata` on (the default), the sidecar's
**outputs** are cross-checked against the executable: the count against
`PJRT_Executable_NumOutputs`, the element types against
`..._OutputElementTypes`, and the dimensions against `..._OutputDimensions`. A
disagreement is a `LoadError` at startup, which is the whole point — a sidecar
that has gone stale relative to its executable would otherwise be discovered as
corrupted output on call ten thousand.

**Inputs cannot be checked this way.** There is no parameter-shape query in the
PJRT C API, so an input that disagrees with the executable survives the load and
surfaces on the warm-up call instead, usually as a buffer-count or shape error
from the plugin. That is why the warm-up calls happen inside the constructor:
the error still arrives before the caller's first real call.

Never answer a metadata failure by turning `check_metadata` off. The two honest
answers are the `.mlirbc` fallback and a re-export.

## The architecture lock

A `.binpb` embeds machine code generated by the exporting machine.
`PJRT_Executable_DeserializeAndLoad` relinks it; it never recompiles. So a
`.binpb` is locked to the architecture *and the instruction set* of the machine
that produced it, and moving one between machines is not a supported operation
even when both are Linux on x86-64.

Three mechanisms handle that.

**The ISA guard** (`FunctionOptions::isa_guard`, on by default under
`LoadPolicy::Auto`) compares `export.host.isa_level` against this host and
passes over a `.binpb` built for a wider instruction set. Without it the
failure is an illegal instruction somewhere inside the executable, with no hint
that the artifact came from a newer machine. Note what the guard cannot do: a
level it does not recognize, or a different architecture family, is not
"probably fine" — an aarch64 host cannot run an x86-64 `.binpb` at all.

**The `.mlirbc` fallback** compiles the StableHLO in-process instead. It is
portable and costs seconds rather than milliseconds at load, and
`Function::load_kind()` reports `LoadKind::Compiled` when it ran, with
`load_detail()` saying why the `.binpb` was passed over. A deployment usually
sets `LoadPolicy::BinaryOnly` so that a fallback cannot happen silently: a route
that quietly compiles for seconds is not a fallback in a control loop.

**Re-exporting on the target machine** is the answer when neither will do, and
is why the export step is a `make` target rather than a committed file.

## Version 1 compatibility

A schema 1 sidecar recorded only sizes and a single `float64` element type,
with a size of 0 meaning a scalar, and carried no names. Both the loader and
`check` widen it to the same view: names become `arg<i>` and `out<i>`, a size of
0 becomes an empty shape with `numel` 1, and everything else follows from the
dtype. Version 1 artifacts still load, and nothing else about them is inferred.
