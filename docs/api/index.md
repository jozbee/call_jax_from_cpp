# API reference

The C++ API is generated from the Doxygen comments in `include/pjrt_exec`; the
Python API from the docstrings in `python/jax2exec`. Members are listed
explicitly on each page, and an undocumented public member does not appear in
the Doxygen XML at all, so a directive naming one fails the build. That is
deliberate: it makes an undocumented public entity a build error rather than an
empty box on a reference page.

## C++

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Runtime
:link: cpp/runtime
:link-type: doc

The plugin, the PJRT client and the device, plus the options that shape all
three and the negotiation that decides whether execution is really inline.
:::

:::{grid-item-card} Function
:link: cpp/function
:link-type: doc

One loaded executable, its aligned arenas, the accessors, and `call()`.
:::

:::{grid-item-card} rt
:link: cpp/rt
:link-type: doc

Optional real-time hardening for the calling thread. Each step reports
whether it took effect.
:::

:::{grid-item-card} Latency
:link: cpp/latency
:link-type: doc

The allocation-free recorder, its summary, and the two tail ratios worth
reading.
:::

:::{grid-item-card} AllocGuard
:link: cpp/alloc-guard
:link-type: doc

The hook into the preloaded allocation counter, and what "zero allocations"
can and cannot mean here.
:::

:::{grid-item-card} Errors
:link: cpp/error
:link-type: doc

`Error` and `LoadError`, what distinguishes them, and the standard exceptions
debug mode raises.
:::

:::{grid-item-card} DType
:link: cpp/dtype
:link-type: doc

The eleven element types, their PJRT and NumPy spellings, and the C++ trait
that maps a scalar type back.
:::

:::{grid-item-card} Example helpers
:link: cpp/examples
:link-type: doc

`cjfc`: the layer the examples share, under `examples/common/`. Not part of
the library's API or ABI.
:::

::::

## Python and artifacts

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} jax2exec
:link: python
:link-type: doc

The exporter, the reference-case writer, and the `check` command that needs no
JAX.
:::

:::{grid-item-card} Artifact format
:link: artifact-format
:link-type: doc

The three files, the sidecar schema field by field, and what the loader can
and cannot validate.
:::

::::

## Headers and include paths

The public headers live in `include/pjrt_exec/`, and the include path is
`include/`, so every include is prefixed with the directory:

```cpp
#include "pjrt_exec/runtime.hpp"      // Runtime, Function, Error, LoadError
#include "pjrt_exec/dtype.hpp"        // DType and its conversions
#include "pjrt_exec/rt.hpp"           // pjrt::rt, the hardening helpers
#include "pjrt_exec/latency.hpp"      // LatencyRecorder, ScopedLatency
#include "pjrt_exec/alloc_guard.hpp"  // AllocGuard, AllocGuardScope
```

`runtime.hpp` is the only one a caller always needs; it includes `dtype.hpp`
itself. `latency.hpp` and `alloc_guard.hpp` are header-only and independent of
the rest — you can measure or audit a loop that has nothing to do with this
project. `rt.hpp` is the only header whose behaviour is platform-dependent, and
it degrades to no-ops that say so rather than to `#ifdef`s at the call site.

Two more paths matter at build time. `third_party/` is on the include path
because `runtime.hpp` includes the vendored PJRT C API header as
`"pjrt/pjrt_c_api.h"`; the vendored version is
{{ pjrt_api_major }}.{{ pjrt_api_minor }}. And **nothing links against the PJRT
plugin** — it is `dlopen`-ed at run time — so a consumer links
`build/lib/libpjrt_exec.a` with `-ldl -lpthread` and nothing else, and a plain
build never touches bazel or the network.

```makefile
CPPFLAGS += -Iinclude -Ithird_party
LDLIBS   += -ldl -lpthread
```

A CMake build gets the same thing from the imported target
`pjrt_exec::pjrt_exec`, and a Make-based consumer can include the
`pjrt_exec.mk` fragment instead of repeating the two lines above.
