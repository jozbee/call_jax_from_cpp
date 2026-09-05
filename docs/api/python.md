# jax2exec

The Python side is one function most of the time. `export` compiles a JAX
function ahead of time and writes the three files a C++ caller needs;
`write_reference_cases` freezes what that function returns, so the C++ side can
be checked against JAX rather than merely observed to run; and
`python -m jax2exec check <base>` describes an artifact set and says whether it
will run on the host inspecting it.

Two design decisions are worth knowing before reading the signatures. **The
imports that pull JAX in are deferred**: `import jax2exec` costs nothing, and
`jax` is only imported when something asks for `export` or
`write_reference_cases`. That is because the machine running the C++ caller
usually has no JAX at all, and `check` has to work there. And **nothing is
written until every check has passed** — a failed export leaves no artifacts
behind, and the three files are written `.binpb`, `.mlirbc`, `.json` in that
order so a reader never sees a sidecar promising artifacts that are not there
yet.

```{eval-rst}
.. currentmodule:: jax2exec

.. autosummary::
   :nosignatures:

   export
   ExportResult
   ExportError
   write_reference_cases
   SUPPORTED_JAX
   SCHEMA_VERSION
```

## Exporting

```{eval-rst}
.. autofunction:: jax2exec.export
```

`args` are positional only, and `jax.ShapeDtypeStruct` is the usual way to pass
them: nothing is executed during tracing, so materializing data that is used
only for its shape and dtype is wasted work. Keyword arguments are refused
outright, because a C++ call site is positional.

The checks that raise `ExportError` are worth reading as a list of the ways an
export goes wrong: a name that is not a legal file name, a dtype outside the
supported eleven, a zero-element array, a function with effects, more than one
device, and **an argument XLA pruned away**. That last one is not hypothetical.
XLA removes parameters the computation never reads, which changes the
executable's arity, and the failure surfaces much later as "Execution supplied
16 buffers but compiled program expected 4". Every input has to feed an output.

```{eval-rst}
.. autoclass:: jax2exec.ExportResult
```

```{eval-rst}
.. autoclass:: jax2exec.ExportError
```

## Reference cases

```{eval-rst}
.. autofunction:: jax2exec.write_reference_cases
```

Also reachable as `jax2exec.reference.write_reference_cases`. Each `.bin` holds
every input followed by every output, in call order, C order, native width,
with no header and no padding; a scalar occupies exactly one element. The
layout is the dumbest thing that works on purpose, because the reader is a C++
test that already knows every shape and dtype from the manifest, and anything
cleverer is one more thing that can disagree between the two languages.

Values are frozen as JAX actually traced them. A float64 argument passed
without `jax_enable_x64` is frozen as the float32 the executable will really be
handed, which is the behaviour that makes the fixture agree with the artifact
rather than with the caller's intent.

## Constants

`SUPPORTED_JAX` is the JAX release this exporter was written against and is
tested on — {{ jax_version }}. A different version is a warning rather than a
refusal: artifacts are validated by the C++ loader on the way in, and a bump
usually just works, but a serialized executable is not portable across one, so
re-export rather than reusing artifacts.

`SCHEMA_VERSION` is the sidecar layout this package writes ({doc}`v2
<artifact-format>`). The C++ loader accepts 1 and 2 and refuses anything newer,
so this number moves only when the loader moves with it.

`SUPPORTED_DTYPES` maps each NumPy dtype name to how that element type is
spelled by every consumer of an artifact: the `PJRT_Buffer_Type` enumerator,
the C++ storage type, its width in bytes, and whether it needs
`jax_enable_x64`. The eleven entries are tabulated on the {doc}`DType
<cpp/dtype>` page. They live in one table so that a new dtype cannot be
half-added, and so the sidecar stays readable by a loader that has no NumPy.

`__version__` is the exporter version, and it is recorded in every sidecar's
`generator` block so an artifact can be traced back to the exact code that
produced it.

## Deprecated

```{eval-rst}
.. autofunction:: jax2exec.jax2exec
```

## The `check` command

`python -m jax2exec check <base>` answers the two questions that get asked when
a load fails on a machine that is not the one that exported: *what does this
sidecar actually declare*, and *will this executable run here*. It needs no
JAX, and it accepts the base path with or without one of its extensions.

```console
$ python -m jax2exec check artifacts/trajopt
```

```{code-block} text
:caption: Every number and identifier below varies with the artifact and the host.

trajopt  (artifacts/trajopt.json)
  schema 2, written by jax2exec 0.2.0
  jax 0.11.1 / jaxlib 0.11.1, platform cpu
  exported 2026-09-05T14:02:11Z on linux/x86_64 x86-64-v3, python 3.12.7, x64 on
  cpu 13th Gen Intel(R) Core(TM) i7-13700H

inputs (2)
  idx  name  dtype    shape  numel  nbytes
    0  x0    float64  [6]        6      48
    1  u0    float64  [40, 2]   80     640

outputs (1)
  idx  name  dtype    shape    numel  nbytes
    0  u     float64  [40, 2]     80     640

artifacts
  trajopt.binpb: 2118344 bytes, sha256 matches
  trajopt.mlirbc: 41120 bytes, sha256 matches

instruction set
  host is x86-64-v3, exported on x86-64-v3: the .binpb runs
```

It exits **0** when the artifact set is consistent and will run on this host,
**1** when it is not, and **2** when the sidecar could not be read at all. The
problems it reports are printed after the description, one `error:` line each:
a dtype the loader does not support, a `numel` or `nbytes` that disagrees with
the shape and dtype recorded beside it, a named artifact that is missing, a
digest that does not match, and an ISA level this host cannot execute. A
missing `.mlirbc` is not an error — it only means this artifact set cannot fall
back to compiling in-process.
