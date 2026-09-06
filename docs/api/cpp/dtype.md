# DType

`pjrt::DType` is a closed set of eleven element types: `bool`, the four signed
and four unsigned integer widths, `float32` and `float64`. XLA has many more,
and the ones left out — F16, BF16, the complex types, the FP8 families, the
sub-byte integers — are absent because none of them has a C++ storage type this
API could hand back. The names are spelled the way NumPy spells them, so the
JSON sidecar the exporter writes can be read back without a translation table.

The header is free of runtime state and everything in it is `constexpr` or
`inline`, so this is a compile-time vocabulary rather than a runtime one. It is
also the same vocabulary on both sides of the artifact: `python/jax2exec`
carries the identical table keyed by NumPy name, which is what keeps a dtype
from being half-added.

Conversion runs both ways, and asymmetrically on purpose. `to_pjrt` is total.
`from_pjrt` is **partial** and returns `std::nullopt` for everything outside the
eleven, which is how the loader notices an executable it cannot represent
instead of handing the caller a pointer to bytes it would misread;
`pjrt_type_name` then names the offending type in the `LoadError`. The mistake
to avoid is on the C++ side of the same idea: write `std::int8_t` and
`std::uint8_t` in calling code, never plain `char`, whose signedness follows the
platform — signed on x86-64 Linux, unsigned on aarch64 Linux — so an artifact's
dtype would end up following the compiler rather than the artifact.

## The eleven

| `DType` | Sidecar name | `PJRT_Buffer_Type` | C++ storage | Bytes |
|---|---|---|---|---|
| `Bool` | `bool` | `PRED` | `bool` | 1 |
| `Int8` | `int8` | `S8` | `std::int8_t` | 1 |
| `Int16` | `int16` | `S16` | `std::int16_t` | 2 |
| `Int32` | `int32` | `S32` | `std::int32_t` | 4 |
| `Int64` | `int64` | `S64` | `std::int64_t` | 8 |
| `UInt8` | `uint8` | `U8` | `std::uint8_t` | 1 |
| `UInt16` | `uint16` | `U16` | `std::uint16_t` | 2 |
| `UInt32` | `uint32` | `U32` | `std::uint32_t` | 4 |
| `UInt64` | `uint64` | `U64` | `std::uint64_t` | 8 |
| `Float32` | `float32` | `F32` | `float` | 4 |
| `Float64` | `float64` | `F64` | `double` | 8 |

The four 64-bit types need `jax_enable_x64` set before tracing, or JAX narrows
them to their 32-bit counterparts and the sidecar honestly records what was
actually exported.

```{doxygenenum} pjrt::DType
```

`Bool` deserves its own sentence. It is one byte holding 0 or 1 — PJRT's
`PRED`, the same storage as NumPy's `bool_` and as C++ `bool` on every platform
this builds for. Any other byte value is *undefined* for XLA rather than merely
truthy: XLA does not normalize the byte, so a stray 2 can make a predicate read
as both true and false within one computation. That is why
`FunctionOptions::check_values` audits bool arenas rather than trusting the
caller to have written a clean 0 or 1.

## Sizes and names

```{doxygenfunction} pjrt::itemsize
```

```{doxygenfunction} pjrt::dtype_name
```

```{doxygenfunction} pjrt::parse_dtype
```

```{doxygenfunction} pjrt::is_floating
```

`parse_dtype` is deliberately strict: no aliases, no `"float"` or `"double"`,
no byte-order prefixes. A sidecar written by this project's exporter always
uses the canonical name, so anything else is a sidecar worth rejecting rather
than guessing at.

## PJRT conversion

```{doxygenfunction} pjrt::to_pjrt
```

```{doxygenfunction} pjrt::from_pjrt
```

```{doxygenfunction} pjrt::pjrt_type_name
```

The eleven enumerators involved hold the same numeric values in PJRT C API 0.90
and {{ pjrt_api_major }}.{{ pjrt_api_minor }}, so an artifact described against
one header loads against the other. `pjrt_type_name` covers the whole PJRT enum
rather than the supported subset, and that is the point: when a re-vendored
header adds an element type, its exhaustive `switch` is where `-Wswitch` says
so and forces a decision, while `from_pjrt`'s `default` keeps returning the
conservative answer in the meantime.

## Generic code

A program that loads whatever artifact it is configured with cannot name `T`
at compile time. It goes through the untyped accessors and switches on the
dtype the sidecar declared:

```cpp
switch (f.input_dtype(i)) {                       // sketch, not from the tree
  case pjrt::DType::Float64:
    std::memcpy(f.input_raw(i), src, f.input_nbytes(i));
    break;
  case pjrt::DType::Int32:
    /* ... */
    break;
  default:
    throw std::runtime_error(std::string("unhandled dtype ") +
                             pjrt::dtype_name(f.input_dtype(i)));
}
```

`input_raw(i)` and `output_raw(i)` hand back `void*` and `const void*` and
are bounds-checked on the same terms as the typed accessors. `dtype_name()`
returns exactly the strings the sidecar carries, so an error message built
from them matches what `python -m jax2exec check` prints.

## Mapping a C++ type back

`pjrt::dtype_of<T>` is the trait the typed accessors use, and
`pjrt::dtype_of_v<T>` is its shorthand:

```cpp
template <class T, class = void>
struct dtype_of;                       // declared, never defined

template <class T>
inline constexpr DType dtype_of_v = dtype_of<T>::value;
```

The primary template is declared and never defined, so `Function::input<T>()`
with an unsupported `T` fails to *compile*, with the offending type named in
the diagnostic — "implicit instantiation of undefined template
`pjrt::dtype_of<std::string>`" — rather than failing at run time or, worse, not
at all.

Specializations exist for `bool`, `float`, `double`, and every integer type
except `bool`. The integer specialization keys on width and signedness rather
than on spelling, so `long`, `long long` and `std::int64_t` all land on
`DType::Int64` whatever the platform calls them, and an integer of a width PJRT
has no element type for is a `static_assert` naming that width.
