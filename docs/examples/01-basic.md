# 01 · Basic

`examples/01_basic` exports a small JAX function and calls it from C++. It
exists to show the whole path — export, three artifact files, load, call — with
nothing else in the way: no timing, no hardening, no reference cases.

It also exercises the one plugin feature that is easy to be missing.
`jnp.linalg.inv` lowers to a LAPACK custom call, and a bare PJRT CPU plugin
registers no LAPACK FFI handlers, because jaxlib registers those from Python on
import and nothing here imports jaxlib. If the load fails with `No FFI handler
registered for lapack_dgetrf_ffi on a platform Host`, the plugin is not the one
`make plugin` provides. See {doc}`the fork page <../developer/xla-fork>`.

## The export

```{literalinclude} ../../examples/01_basic/export.py
:language: python
:caption: examples/01_basic/export.py
```

`export` writes three files into `artifacts/`, all sharing the base name:

| File | What it is |
|---|---|
| `basic.binpb` | the serialized PJRT executable, with machine code for this host embedded in it |
| `basic.mlirbc` | StableHLO bytecode for the same function, portable, compiled in-process when the `.binpb` will not run here |
| `basic.json` | the sidecar: every input and output by name, dtype and shape, plus the versions, the host, and the digests |

The sidecar is not optional. The PJRT C API cannot be asked what parameters an
executable takes, so it is the only description of the signature that exists.
{doc}`../api/artifact-format` has it field by field.

## The call

```{literalinclude} ../../examples/01_basic/basic.cpp
:language: cpp
:caption: examples/01_basic/basic.cpp
```

## Build and run

```console
$ uv sync
$ make plugin
$ make
$ make export
$ ./build/bin/example_01_basic
```

`make export` runs both example export scripts; to run only this one:

```console
$ uv run python examples/01_basic/export.py --out artifacts
```

Add `--debug` to turn on `FunctionOptions::debug`, which bounds-checks every
index, rejects a typed accessor whose `T` disagrees with the declared dtype,
and catches a re-entrant `call()`:

```console
$ ./build/bin/example_01_basic --debug
```

`make run-examples` runs it both ways.

## Expected output

```{code-block} text
:caption: Illustrative. Every path, version, digest and number below varies with the machine and the artifact; the structure does not.

plugin   build/plugin/libpjrt_c_api_cpu_plugin.so       <- varies
platform cpu (XLA <build>), PJRT C API 0.114            <- varies
sync     Inline
loaded   artifacts/basic.binpb (deserialized)           <- Compiled when the .mlirbc ran instead
inputs   2   outputs 1

result matches JAX to 0.0e+00 relative error            <- varies
```

If it does not appear, the failure is almost always one of four things, and
each names itself:

- **`cannot find a PJRT plugin`** — `make plugin` has not run, and
  `$PJRT_CPU_PLUGIN` is unset.
- **`No FFI handler registered for lapack_dgetrf_ffi`** — the plugin is not
  built from the fork.
- **a `LoadError` naming the sidecar** — `artifacts/` is stale relative to the
  exporter. Re-run `make export`. Do not turn `check_metadata` off.
- **an illegal instruction, or a load failure naming the ISA** — the `.binpb`
  came from another machine. Re-export here, or let the ISA guard fall back to
  the `.mlirbc`.

`python -m jax2exec check artifacts/basic` answers most of these without
building anything, and needs no JAX.

## What to change to make it yours

Replace the function in `export.py` and the arguments it is traced with. Every
constraint the exporter enforces is worth knowing before you do:

- **Arguments are positional.** Keyword arguments have no place in a C++ call
  site, and `export` refuses them.
- **Every input must feed an output.** XLA prunes parameters the computation
  never reads, which changes the executable's arity and produces "Execution
  supplied N buffers but compiled program expected M" much later.
- **Dtypes come from the supported eleven** — `bool`, the integer widths,
  `float32`, `float64`. Set `jax_enable_x64` before tracing if you want the
  64-bit ones; without it JAX narrows them and the sidecar honestly records
  what was actually exported.
- **Any rank works**, including scalars, and shapes are row-major.

On the C++ side, resolve names to indices once with `find_input` and
`find_output` — they are linear scans over strings — then write
`input<double>(i)`, call, and read `output<double>(j)`. If you change the
function's signature, re-export before rebuilding: the sidecar and the
executable travel together.
