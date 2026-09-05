# Errors

Two exception types, and the line between them is about who can act. A
`pjrt::Error` is a failure the plugin reported through the PJRT C API and
carries the abseil-style status code it came with. A `pjrt::LoadError` is a
problem with the artifact or its surroundings: a missing plugin, a sidecar that
disagrees with its executable, an element type this project does not support, a
`.binpb` built for a wider instruction set than this host implements. Those are
the failures a caller can do something about, and every one of them happens at
load.

That timing is the point of the whole design. Loading is where the sidecar is
read, cross-checked, and turned into arenas, so a stale artifact is an
exception at startup rather than a silent overwrite past the end of an arena on
call ten thousand. In the steady state, `call()` can throw `Error` if the
execution itself fails, and — only when `FunctionOptions::debug` or
`check_values` is on — one of the standard exceptions below.

The mistake to avoid is freeing a `PJRT_Error*`. Constructing an `Error`
**consumes** it: the constructor copies out the message and the status code and
then calls `PJRT_Error_Destroy`. Hand every `PJRT_Error*` to `check_error` and
forget it; that function is the only place one should be inspected, and it
allocates nothing when the pointer is null.

## PJRT failures

```{doxygenclass} pjrt::Error
:members: Error, code
```

```{doxygenfunction} pjrt::check_error
```

## Artifact and plugin failures

```{doxygenclass} pjrt::LoadError
```

`LoadError` inherits `std::runtime_error`'s constructors and adds nothing: the
message is the whole payload, and it names the file, the index and the two
things that disagreed. A `LoadError` is not something to retry — see
{doc}`../artifact-format` for what the loader checks and when, and the
{doc}`function` page for the `.mlirbc` fallback that answers the architecture
lock.

## Standard exceptions from debug mode

These come from `<stdexcept>` and are thrown only when the corresponding option
is on. With the option off, the same mistake is silent memory corruption
instead, which is the trade to weigh when deciding where to run with `debug`.

| Exception | Thrown by | When |
|---|---|---|
| `std::out_of_range` | `input_spec`, `output_spec` | always, in every build — these are startup calls |
| `std::out_of_range` | `input`, `output`, `input_raw`, `output_raw` | `debug`: index past the end |
| `std::invalid_argument` | `input<T>`, `output<T>` | `debug`: `T` disagrees with the dtype the sidecar declares |
| `std::logic_error` | `call` | `debug`: a re-entrant call |
| `std::domain_error` | `call` | `check_values`: a nan or inf in a float arena, or a byte other than 0 or 1 in a bool arena |

Asking for a `T` that is not one of the eleven supported element types is not a
run-time error at all: `dtype_of<T>` is declared and never defined, so the
compiler rejects it and names the offending type. See {doc}`dtype`.
