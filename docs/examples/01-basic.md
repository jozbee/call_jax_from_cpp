# 01 · Basic

*Assumes the {doc}`Quickstart </getting-started/quickstart>`, which is this
example end to end; this page shows the parts the Quickstart skipped.*

`examples/01_basic` exports one small function and calls it once from C++:
export, three artifact files, load, call, check. Nothing else is in the way.
The function is `fun(A, b) -> (x, r)`: `x` solves `A x = b`, and `r` is the
residual JAX computed for it, so the C++ side can check itself twice — once
against a residual it recomputes from its own arenas, once against `r`.

:::{note}
Two namespaces appear. `pjrt::` is the library ({doc}`/api/index`);
`basic::` is this example's own flags, printing and checks in `support.hpp`.
:::

## The export

```{literalinclude} ../../examples/01_basic/export.py
:language: python
:start-after: docs: begin export
:end-before: docs: end export
```

{py:func}`~jax2exec.export` writes `basic.binpb`, `basic.mlirbc` and
`basic.json` — an {term}`artifact` — into `artifacts/`. `jnp.linalg.inv` is there on purpose: it lowers to a LAPACK
custom call that only the fork's plugin can load, so a wrong plugin fails
here, at the first example, rather than in the field.

## Load

```{literalinclude} ../../examples/01_basic/basic.cpp
:language: cpp
:start-after: docs: begin load
:end-before: docs: end load
```

One {cpp:class}`~pjrt::Runtime` per process, one {cpp:class}`~pjrt::Function`
per artifact; {doc}`/guides/calling` has the rules that follow.

## Read the signature

Names resolve to indices once, at startup, with
{cpp:func}`~pjrt::Function::find_input`, and the order of the system comes
from the artifact rather than from a constant.

```{literalinclude} ../../examples/01_basic/basic.cpp
:language: cpp
:start-after: docs: begin signature
:end-before: docs: end signature
```

## Call

```{literalinclude} ../../examples/01_basic/basic.cpp
:language: cpp
:start-after: docs: begin call
:end-before: docs: end call
```

{cpp:func}`~pjrt::Function::input` and {cpp:func}`~pjrt::Function::output`
are pointers into the {term}`arenas <arena>` the `Function` owns;
{cpp:func}`~pjrt::Function::call` runs the executable over them. The residual
is then recomputed from the same arenas: a layout mistake on the C++ side
produces a believable `x` and a residual that is not small. Flags,
the `key=value` printing and the `--debug` demonstration live in
`examples/01_basic/support.hpp`.

## Build and run

```console
$ uv sync && make plugin && make && make export
$ ./build/bin/example_01_basic
$ ./build/bin/example_01_basic --debug     # the debug checks, made to fire
```

## Expected output

```text
load_kind=deserialized
synchronous_supported=1
sync_mode=inline
num_inputs=2 num_outputs=2
input[0]: dtype=float64 shape=[4,4] numel=16 nbytes=128
input[1]: dtype=float64 shape=[4] numel=4 nbytes=32
output[0]: dtype=float64 shape=[4] numel=4 nbytes=32
output[1]: dtype=float64 shape=[] numel=1 nbytes=8
x=[...]
residual_inf_norm=...
residual_from_jax=...
debug=0 (checks disabled; see --debug)
```

`load_kind=compiled` means the ISA guard sent the loader to the `.mlirbc`;
`sync_mode=rejected` or `accepted` means a stock plugin; `float32` in
`input[0]` means the export ran without `jax_enable_x64`. With `--debug`,
three more lines show each check catching a mistake made on purpose:

```text
debug_check[out_of_range]: input index 99 is out of range: function 'basic' has 2 inputs
debug_check[dtype_mismatch]: input 0 ('A') has dtype float64 but was accessed as float32
debug_check[non_finite]: input 0 ('A') element 0 is nan
```

If the run fails instead, {doc}`../guides/debugging` has one row per message.

## Making it yours

Replace `fun` and the shapes it is traced with; {doc}`../guides/exporting`
has the rules. On the C++ side, resolve names once, write `input<T>(i)`,
call, read `output<T>(j)`, and re-export whenever the signature changes — the
sidecar and the executable travel together.
